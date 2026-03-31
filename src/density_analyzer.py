"""
density_analyzer.py — Advanced traffic density analysis with lane-awareness.

Capabilities:
  • Frame-level vehicle counting per lane (configurable lane polygons)
  • Temporal smoothing via exponential moving average (EMA)
  • Vehicle flow rate estimation (vehicles / minute)
  • Occupancy ratio: fraction of detection-zone area covered by vehicles
  • Congestion score (0–100) combining count, occupancy, and flow rate
  • Rolling history for trend detection (rising / stable / falling)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from src.tracker import Track
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lane polygon helpers
# ---------------------------------------------------------------------------

@dataclass
class Lane:
    """
    A polygonal region of interest representing a traffic lane.

    Parameters
    ----------
    name    : Human-readable identifier, e.g. "Lane 1".
    polygon : (N, 2) array of (x, y) vertices in image coordinates.
              Counter-clockwise or clockwise — cv2.pointPolygonTest handles both.
    """
    name:    str
    polygon: np.ndarray   # shape (N, 2), dtype float32

    def __post_init__(self) -> None:
        self.polygon = np.asarray(self.polygon, dtype=np.float32)

    def contains_centre(self, cx: float, cy: float) -> bool:
        """True if point (cx, cy) lies inside or on the boundary of this lane."""
        result = cv2.pointPolygonTest(self.polygon, (cx, cy), measureDist=False)
        return result >= 0

    def area(self) -> float:
        """Polygon area via Shoelace formula."""
        pts = self.polygon
        n   = len(pts)
        if n < 3:
            return 0.0
        x, y = pts[:, 0], pts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def make_full_frame_lane(width: int, height: int, name: str = "Full Frame") -> Lane:
    """Convenience: create a Lane covering the entire frame."""
    return Lane(
        name=name,
        polygon=np.array(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
        ),
    )


# ---------------------------------------------------------------------------
# Per-frame density snapshot
# ---------------------------------------------------------------------------

@dataclass
class FrameDensity:
    """Density metrics for a single frame."""
    frame_idx:         int
    timestamp_ms:      float                           # wall-clock ms
    counts_per_lane:   dict[str, int]                  # lane_name → count
    total_count:       int
    density_label:     str                             # Low / Medium / High
    occupancy_ratio:   float                           # 0–1
    congestion_score:  float                           # 0–100
    ema_count:         float                           # smoothed total
    class_breakdown:   dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Density analyser
# ---------------------------------------------------------------------------

class DensityAnalyzer:
    """
    Analyses vehicle density from a sequence of tracked detections.

    Parameters
    ----------
    lanes           : Ordered list of Lane objects.
    ema_alpha       : EMA smoothing factor (0 < α ≤ 1).
    flow_window_s   : Duration in seconds used to estimate flow rate.
    low_threshold   : Total count below which density is "Low".
    high_threshold  : Total count above which density is "High".
    fps             : Expected video frame rate (used for flow estimation).
    """

    def __init__(
        self,
        lanes:           list[Lane],
        ema_alpha:       float = 0.20,
        flow_window_s:   float = 60.0,
        low_threshold:   int   = 10,
        high_threshold:  int   = 25,
        fps:             float = 30.0,
    ) -> None:
        if not 0 < ema_alpha <= 1:
            raise ValueError(f"ema_alpha must be in (0, 1]. Got {ema_alpha}.")

        self.lanes          = lanes
        self.alpha          = ema_alpha
        self.flow_window_s  = flow_window_s
        self.low_threshold  = low_threshold
        self.high_threshold = high_threshold
        self.fps            = fps

        # State
        self._ema: float = 0.0
        self._frame_idx: int = 0
        self._t0_ms: float = 0.0
        self._history: deque[FrameDensity] = deque(maxlen=500)

        # Flow estimation: store (timestamp_ms, count) per frame
        self._flow_buffer: deque[tuple[float, int]] = deque()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._ema       = 0.0
        self._frame_idx = 0
        self._t0_ms     = 0.0
        self._history.clear()
        self._flow_buffer.clear()
        logger.info("DensityAnalyzer reset.")

    def update(
        self,
        tracks:       Sequence[Track],
        timestamp_ms: float | None = None,
    ) -> FrameDensity:
        """
        Process a frame's confirmed tracks and return density metrics.

        Parameters
        ----------
        tracks       : Sequence of Track objects from the current frame.
        timestamp_ms : Wall-clock time in ms. Defaults to frame_idx / fps * 1000.
        """
        if timestamp_ms is None:
            timestamp_ms = self._frame_idx / self.fps * 1_000.0

        if self._frame_idx == 0:
            self._t0_ms = timestamp_ms

        # --- Per-lane and per-class counting ---------------------------------
        lane_counts: dict[str, int] = {lane.name: 0 for lane in self.lanes}
        class_counts: dict[str, int] = {}

        for track in tracks:
            cx, cy = track.centre
            for lane in self.lanes:
                if lane.contains_centre(cx, cy):
                    lane_counts[lane.name] += 1

            cls = track.class_name
            class_counts[cls] = class_counts.get(cls, 0) + 1

        total = sum(lane_counts.values()) if len(self.lanes) > 1 else sum(class_counts.values())

        # --- EMA smoothing ---------------------------------------------------
        if self._frame_idx == 0:
            self._ema = float(total)
        else:
            self._ema = self.alpha * total + (1 - self.alpha) * self._ema

        # --- Occupancy ratio -------------------------------------------------
        occupancy = self._compute_occupancy(tracks)

        # --- Flow rate (vehicles / min) --------------------------------------
        self._flow_buffer.append((timestamp_ms, total))
        self._prune_flow_buffer(timestamp_ms)

        # --- Congestion score (0–100) ----------------------------------------
        congestion = self._compute_congestion(total, occupancy)

        # --- Density label ---------------------------------------------------
        label = self._classify(int(round(self._ema)))

        fd = FrameDensity(
            frame_idx        = self._frame_idx,
            timestamp_ms     = timestamp_ms,
            counts_per_lane  = lane_counts,
            total_count      = total,
            density_label    = label,
            occupancy_ratio  = round(occupancy, 4),
            congestion_score = round(congestion, 2),
            ema_count        = round(self._ema, 2),
            class_breakdown  = class_counts,
        )

        self._history.append(fd)
        self._frame_idx += 1

        logger.debug(
            "Frame %d | count=%d | ema=%.1f | density=%s | congestion=%.1f",
            fd.frame_idx, total, self._ema, label, congestion,
        )

        return fd

    def flow_rate_per_minute(self) -> float:
        """Estimate vehicle throughput (vehicles per minute) over the flow window."""
        buf = list(self._flow_buffer)
        if len(buf) < 2:
            return 0.0
        total_vehicles = sum(c for _, c in buf)
        duration_ms    = buf[-1][0] - buf[0][0]
        if duration_ms <= 0:
            return 0.0
        return total_vehicles / (duration_ms / 60_000.0)

    def trend(self, window: int = 10) -> str:
        """
        Return traffic trend over the last *window* frames.
        One of: "rising", "stable", "falling".
        """
        hist = list(self._history)[-window:]
        if len(hist) < 3:
            return "stable"

        counts = [fd.ema_count for fd in hist]
        slope  = np.polyfit(range(len(counts)), counts, 1)[0]
        if slope > 0.5:
            return "rising"
        if slope < -0.5:
            return "falling"
        return "stable"

    def rolling_average(self, window: int = 30) -> float:
        """Mean total vehicle count over the last *window* frames."""
        hist = list(self._history)[-window:]
        if not hist:
            return 0.0
        return float(np.mean([fd.total_count for fd in hist]))

    def history(self) -> list[FrameDensity]:
        """Return full history as a list (newest last)."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify(self, count: int) -> str:
        if count < self.low_threshold:
            return "Low"
        if count <= self.high_threshold:
            return "Medium"
        return "High"

    def _compute_occupancy(self, tracks: Sequence[Track]) -> float:
        """
        Fraction of total lane area covered by vehicle bounding boxes.
        Clipped to [0, 1].
        """
        total_lane_area = sum(lane.area() for lane in self.lanes)
        if total_lane_area <= 0:
            return 0.0

        vehicle_area = 0.0
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            vehicle_area += max(0.0, x2 - x1) * max(0.0, y2 - y1)

        return min(vehicle_area / total_lane_area, 1.0)

    def _compute_congestion(self, count: int, occupancy: float) -> float:
        """
        Composite congestion score in [0, 100].

        Formula: 0.6 * count_norm + 0.4 * occupancy
          count_norm : count / high_threshold, clipped to 1.
        """
        count_norm = min(count / max(self.high_threshold, 1), 1.0)
        raw        = 0.60 * count_norm + 0.40 * occupancy
        return raw * 100.0

    def _prune_flow_buffer(self, now_ms: float) -> None:
        window_ms = self.flow_window_s * 1_000.0
        while self._flow_buffer and (now_ms - self._flow_buffer[0][0]) > window_ms:
            self._flow_buffer.popleft()


# ---------------------------------------------------------------------------
# Overlay helper: draw lane boundaries on a frame
# ---------------------------------------------------------------------------

def draw_lanes(
    frame: np.ndarray,
    lanes: list[Lane],
    density_labels: dict[str, str] | None = None,
) -> np.ndarray:
    """
    Draw semi-transparent lane polygons on *frame*.

    Parameters
    ----------
    density_labels : Optional {lane_name: "Low"|"Medium"|"High"} mapping;
                     used to colour each lane overlay.
    """
    density_colours = {
        "Low":    (0, 200, 100),
        "Medium": (0, 170, 255),
        "High":   (30,  30, 220),
        None:     (180, 180, 180),
    }

    out = frame.copy()
    overlay = frame.copy()

    for lane in lanes:
        pts = lane.polygon.astype(np.int32)
        label = (density_labels or {}).get(lane.name)
        colour = density_colours.get(label, density_colours[None])

        cv2.fillPoly(overlay, [pts], colour)
        cv2.polylines(out, [pts], isClosed=True, color=colour, thickness=2)

        # Lane name label at centroid
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        cv2.putText(
            out, lane.name,
            (cx - 30, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            colour, 2, cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)
    return out