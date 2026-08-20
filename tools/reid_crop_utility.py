#!/usr/bin/env python3
"""
ATOS v3.5 Vehicle Crop Extraction & Keyframe Feature Aggregation Utility
Provides safe bounding-box cropping, out-of-bounds clipping, quality & confidence gating,
keyframe sampling, and multi-observation embedding aggregation for Cross-Camera Vehicle Re-ID.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional

def extract_vehicle_crops(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    min_confidence: float = 0.5,
    min_dim: int = 32,
    valid_classes: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Extracts, clips, and validates vehicle crop patches from a video frame tensor based on detections.

    Input detections format:
      [{"track_id": int, "class": str, "confidence": float, "box": [x, y, w, h] or [x1, y1, x2, y2]}]

    Returns list of validated crop objects:
      [{
         "track_id": int,
         "class": str,
         "confidence": float,
         "box": [clip_x1, clip_y1, crop_w, crop_h],
         "crop": np.ndarray (BGR image patch)
      }]
    """
    if frame is None or frame.size == 0 or not detections:
        return []

    if valid_classes is None:
        valid_classes = ["car", "bus", "truck", "motorcycle", "vehicle"]

    h_img, w_img = frame.shape[:2]
    extracted = []

    for det in detections:
        conf = float(det.get("confidence", 0.0))
        if conf < min_confidence:
            continue

        cls_name = str(det.get("class", "car")).lower()
        if valid_classes and cls_name not in valid_classes:
            continue

        box = det.get("box", [])
        if len(box) < 4:
            continue

        raw1, raw2, raw3, raw4 = box[:4]
        box_format = str(det.get("box_format", "xywh")).lower()

        if box_format == "xyxy":
            x1, y1, x2, y2 = raw1, raw2, raw3, raw4
            bw, bh = x2 - x1, y2 - y1
        else:
            # Default ByteTrack format: [x, y, w, h]
            x1, y1 = raw1, raw2
            bw, bh = raw3, raw4
            x2, y2 = x1 + bw, y1 + bh

        # Safely clip coordinates to image boundaries
        clip_x1 = max(0, min(int(round(x1)), w_img - 1))
        clip_y1 = max(0, min(int(round(y1)), h_img - 1))
        clip_x2 = max(0, min(int(round(x2)), w_img))
        clip_y2 = max(0, min(int(round(y2)), h_img))

        crop_w = clip_x2 - clip_x1
        crop_h = clip_y2 - clip_y1

        # Reject invalid, empty, or small crops (< min_dim px)
        if crop_w < min_dim or crop_h < min_dim:
            continue

        crop_bgr = frame[clip_y1:clip_y2, clip_x1:clip_x2].copy()
        if crop_bgr.size == 0 or crop_bgr.shape[0] < min_dim or crop_bgr.shape[1] < min_dim:
            continue

        extracted.append({
            "track_id": int(det.get("track_id", -1)),
            "class": cls_name,
            "confidence": conf,
            "box": [clip_x1, clip_y1, crop_w, crop_h],
            "crop": crop_bgr
        })

    return extracted


class VehicleKeyframeAggregator:
    """
    Manages keyframe sampling and multi-observation embedding aggregation per local vehicle track.
    Buffers 3–5 keyframe feature embeddings per track_id, computes L2-normalized mean embedding vector,
    and formats aggregated data payloads for CrossCameraReIDManager.process_feature().
    """
    def __init__(
        self,
        sample_interval_frames: int = 5,
        target_keyframes_per_track: int = 3,
        max_buffered_keyframes: int = 5,
        min_confidence: float = 0.5,
        min_crop_dim: int = 32
    ):
        self.sample_interval = max(1, sample_interval_frames)
        self.target_keyframes = target_keyframes_per_track
        self.max_buffered_keyframes = max_buffered_keyframes
        self.min_confidence = min_confidence
        self.min_crop_dim = min_crop_dim

        # Per-track buffers: track_id -> dict
        self.track_buffers: Dict[int, Dict[str, Any]] = {}

    def should_sample(self, track_id: int, frame_idx: int) -> bool:
        """Determines if a frame observation should be sampled for keyframe feature extraction."""
        if track_id not in self.track_buffers:
            return True
        buf = self.track_buffers[track_id]
        if len(buf["embeddings"]) >= self.max_buffered_keyframes:
            return False
        return (frame_idx - buf["last_sample_frame"]) >= self.sample_interval

    def add_observation(
        self,
        camera_id: str,
        track_id: int,
        embedding: List[float],
        timestamp: float,
        bbox: List[int],
        frame_idx: int,
        cls_name: str = "car"
    ) -> Optional[Dict[str, Any]]:
        """
        Adds a keyframe embedding observation for a local track.
        If target_keyframes (3-5) are collected or buffer reaches capacity, computes L2-normalized
        aggregated feature vector and returns ready-to-process payload for CrossCameraReIDManager.
        """
        if not embedding or len(embedding) == 0:
            return None

        if track_id not in self.track_buffers:
            self.track_buffers[track_id] = {
                "embeddings": [],
                "bboxes": [],
                "timestamps": [],
                "camera_id": camera_id,
                "class": cls_name,
                "last_sample_frame": frame_idx
            }

        buf = self.track_buffers[track_id]
        buf["embeddings"].append(embedding)
        buf["bboxes"].append(bbox)
        buf["timestamps"].append(timestamp)
        buf["last_sample_frame"] = frame_idx

        # If target keyframe count (e.g. 3-5) reached
        if len(buf["embeddings"]) >= self.target_keyframes:
            agg_embedding = self.aggregate_embeddings(buf["embeddings"])
            aggregated_payload = {
                "camera_id": camera_id,
                "local_track_id": track_id,
                "embedding": agg_embedding,
                "timestamp": buf["timestamps"][-1],
                "bbox": buf["bboxes"][-1],
                "num_keyframes": len(buf["embeddings"]),
                "class": cls_name
            }
            return aggregated_payload

        return None

    @staticmethod
    def aggregate_embeddings(embeddings: List[List[float]]) -> List[float]:
        """
        Computes element-wise mean across keyframe embeddings and applies L2 normalization.
        """
        if not embeddings:
            return []
        arr = np.array(embeddings, dtype=np.float32)
        mean_vec = np.mean(arr, axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 0:
            mean_vec = mean_vec / norm
        return mean_vec.tolist()

    def get_track_aggregated_embedding(self, track_id: int) -> Optional[List[float]]:
        """Returns aggregated embedding for a track_id if buffer has embeddings."""
        if track_id in self.track_buffers and self.track_buffers[track_id]["embeddings"]:
            return self.aggregate_embeddings(self.track_buffers[track_id]["embeddings"])
        return None

    def clear_track(self, track_id: int):
        """Clears buffer when track terminates."""
        if track_id in self.track_buffers:
            del self.track_buffers[track_id]
