"""
tracker.py — Multi-object tracker for consistent vehicle identity across frames.

Implements a lightweight SORT-style (Simple Online and Realtime Tracking) algorithm:
  • One Kalman Filter per track — smooth position/velocity estimation
  • Hungarian algorithm assignment (scipy.optimize.linear_sum_assignment)
  • IoU-based affinity matrix — no learned re-ID, no GPU required
  • Automatic track birth (tentative) → confirmation → death lifecycle
  • Returns a stable set of Track objects every frame

Why not just ByteTrack/DeepSORT?
  Those are great for production — this implementation gives you full control over
  every parameter and removes heavy dependencies (torchvision, faiss, etc.).
  Swap in ByteTrack via ultralytics' built-in tracker if you need state-of-the-art MOTA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Kalman Filter — constant-velocity model
# ---------------------------------------------------------------------------

class KalmanBoxTracker:
    """
    Represents a tracked bounding box using a Kalman Filter.

    State vector: [cx, cy, s, r, ċx, ċy, ṡ]
        cx, cy : centre coordinates
        s      : area (scale)
        r      : aspect ratio (w/h, kept constant)
        ċx, ċy : velocities
        ṡ      : area velocity

    Observation vector: [cx, cy, s, r]
    """

    count: int = 0   # global track-ID counter

    def __init__(self, bbox: np.ndarray) -> None:
        """
        Parameters
        ----------
        bbox : (x1, y1, x2, y2) array
        """
        KalmanBoxTracker.count += 1
        self.id            = KalmanBoxTracker.count
        self.hits          = 1       # frames with successful detection match
        self.hit_streak    = 1       # consecutive frames matched
        self.age           = 1       # total frames since birth
        self.time_since_update = 0   # frames since last matched detection

        # --- Kalman matrices (7-state, 4-observation) ----------------------
        dt = 1.0
        F = np.eye(7)
        F[0, 4] = F[1, 5] = F[2, 6] = dt  # position += velocity

        H = np.zeros((4, 7))
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0

        # Process noise
        Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001])
        # Measurement noise
        R = np.diag([1.0, 1.0, 10.0, 10.0])
        # Initial covariance
        P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])

        self._F = F
        self._H = H
        self._Q = Q
        self._R = R
        self._P = P
        self._x = np.zeros((7, 1))

        z = _xyxy_to_z(bbox)
        self._x[:4] = z

    # ---- Kalman predict step ----------------------------------------------
    def predict(self) -> np.ndarray:
        """Advance state estimate one time step. Return predicted bounding box."""
        if self._x[6] + self._x[2] <= 0:
            self._x[6] = 0.0

        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return _z_to_xyxy(self._x)

    # ---- Kalman update step -----------------------------------------------
    def update(self, bbox: np.ndarray) -> None:
        """Correct state with a new matched detection."""
        z = _xyxy_to_z(bbox)
        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (np.eye(7) - K @ self._H) @ self._P

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def get_state(self) -> np.ndarray:
        """Return current (x1, y1, x2, y2) estimate."""
        return _z_to_xyxy(self._x)


# ---------------------------------------------------------------------------
# Coordinate conversion helpers
# ---------------------------------------------------------------------------

def _xyxy_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert (x1,y1,x2,y2) → (cx,cy,s,r) column vector."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2
    cy = bbox[1] + h / 2
    s  = w * h
    r  = w / (h + 1e-6)
    return np.array([[cx], [cy], [s], [r]], dtype=float)


def _z_to_xyxy(x: np.ndarray) -> np.ndarray:
    """Convert state vector (cx,cy,s,r,…) → (x1,y1,x2,y2)."""
    cx, cy, s, r = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    if s <= 0:
        s = 1.0
    w = np.sqrt(s * abs(r))
    h = s / (w + 1e-6)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def _iou_matrix(detections: np.ndarray, trackers: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between *detections* (D×4) and *trackers* (T×4).
    Returns an (D, T) matrix.
    """
    D, T = len(detections), len(trackers)
    iou = np.zeros((D, T), dtype=float)
    for d in range(D):
        for t in range(T):
            iou[d, t] = _box_iou(detections[d], trackers[t])
    return iou


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter   = inter_w * inter_h

    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union  = area_a + area_b - inter

    return inter / (union + 1e-6)


# ---------------------------------------------------------------------------
# Track lifecycle data class
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """Public representation of a tracked vehicle."""
    track_id:   int
    bbox:       np.ndarray          # (x1, y1, x2, y2) in image coordinates
    class_name: str
    confidence: float
    hit_streak: int                 # consecutive frames matched (≥ min_hits → confirmed)
    age:        int                 # total frames since birth
    history:    list[np.ndarray] = field(default_factory=list)  # past centre points

    @property
    def confirmed(self) -> bool:
        return self.hit_streak >= 3

    @property
    def centre(self) -> tuple[float, float]:
        return (
            float((self.bbox[0] + self.bbox[2]) / 2),
            float((self.bbox[1] + self.bbox[3]) / 2),
        )


# ---------------------------------------------------------------------------
# SORT Tracker
# ---------------------------------------------------------------------------

class SORTTracker:
    """
    Simple Online and Realtime Tracker.

    Parameters
    ----------
    max_age       : Frames a track survives without a detection match.
    min_hits      : Frames a track must be matched before it is reported as confirmed.
    iou_threshold : Minimum IoU to associate a detection with a track.
    """

    def __init__(
        self,
        max_age: int = 5,
        min_hits: int = 3,
        iou_threshold: float = 0.30,
    ) -> None:
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self._trackers: list[KalmanBoxTracker] = []
        self._meta: dict[int, dict] = {}  # track_id → {class_name, conf}
        self.frame_count = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all tracks (call between independent video streams)."""
        self._trackers.clear()
        self._meta.clear()
        KalmanBoxTracker.count = 0
        self.frame_count = 0

    # ------------------------------------------------------------------
    def update(
        self,
        detections: Sequence[dict],
    ) -> list[Track]:
        """
        Update tracker with detections from the current frame.

        Parameters
        ----------
        detections : list of dicts, each with keys:
            bbox        : (x1, y1, x2, y2) as list or np.ndarray
            class_name  : str
            confidence  : float

        Returns
        -------
        List of active Track objects (confirmed + recently-active unconfirmed).
        """
        self.frame_count += 1

        # --- Step 1: Predict all existing tracks --------------------------
        predicted_boxes: list[np.ndarray] = []
        for trk in self._trackers:
            pb = trk.predict()
            predicted_boxes.append(pb)

        det_boxes      = np.array([np.array(d["bbox"], dtype=float) for d in detections]) if detections else np.empty((0, 4))
        pred_boxes     = np.array(predicted_boxes) if predicted_boxes else np.empty((0, 4))

        # --- Step 2: Hungarian assignment on IoU --------------------------
        matched, unmatched_dets, unmatched_trks = self._assign(det_boxes, pred_boxes)

        # --- Step 3: Update matched tracks --------------------------------
        for d_idx, t_idx in matched:
            det = detections[d_idx]
            trk = self._trackers[t_idx]
            trk.update(np.array(det["bbox"], dtype=float))
            self._meta[trk.id] = {
                "class_name": det["class_name"],
                "confidence": det["confidence"],
            }

        # --- Step 4: Spawn new tracks for unmatched detections ------------
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_trk = KalmanBoxTracker(np.array(det["bbox"], dtype=float))
            self._trackers.append(new_trk)
            self._meta[new_trk.id] = {
                "class_name": det["class_name"],
                "confidence": det["confidence"],
            }

        # --- Step 5: Remove dead tracks -----------------------------------
        self._trackers = [
            t for t in self._trackers
            if t.time_since_update <= self.max_age
        ]

        # --- Step 6: Build output Track objects ---------------------------
        tracks: list[Track] = []
        for trk in self._trackers:
            if trk.time_since_update > 1:
                continue  # only report tracks active this or last frame
            if trk.hit_streak < self.min_hits and self.frame_count <= self.min_hits:
                continue  # suppressed until confirmed (except warm-up frames)

            meta = self._meta.get(trk.id, {})
            bbox = trk.get_state()
            track = Track(
                track_id   = trk.id,
                bbox       = bbox,
                class_name = meta.get("class_name", "unknown"),
                confidence = meta.get("confidence", 0.0),
                hit_streak = trk.hit_streak,
                age        = trk.age,
            )
            tracks.append(track)

        logger.debug("Frame %d: %d tracks active", self.frame_count, len(tracks))
        return tracks

    # ------------------------------------------------------------------
    def _assign(
        self,
        det_boxes: np.ndarray,
        pred_boxes: np.ndarray,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        Hungarian assignment.

        Returns
        -------
        matched           : list of (det_idx, trk_idx) pairs
        unmatched_dets    : list of unmatched detection indices
        unmatched_trks    : list of unmatched tracker indices
        """
        if len(pred_boxes) == 0:
            return [], list(range(len(det_boxes))), []
        if len(det_boxes) == 0:
            return [], [], list(range(len(pred_boxes)))

        iou_mat = _iou_matrix(det_boxes, pred_boxes)
        row_ind, col_ind = linear_sum_assignment(-iou_mat)

        matched: list[tuple[int, int]] = []
        unmatched_dets = list(range(len(det_boxes)))
        unmatched_trks = list(range(len(pred_boxes)))

        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= self.iou_threshold:
                matched.append((int(r), int(c)))
                unmatched_dets.remove(r)
                unmatched_trks.remove(c)

        return matched, unmatched_dets, unmatched_trks