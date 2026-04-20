"""
speed_analyzer.py — Vehicle velocity and trajectory estimation engine.

Computes real-time speed and heading vectors for tracked objects using 
temporal displacement analysis. Includes multi-stage filtering 
(EMA + Median) and outlier rejection for high-accuracy telemetry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.tracker import Track
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VehicleSpeedInfo:
    """Speed and direction for one tracked vehicle."""
    track_id:      int
    speed_kmh:     float
    direction_deg: float    # 0=up/North, 90=right/East, 180=down/South, 270=left/West
    speed_class:   str      # "stopped", "slow", "normal", "fast", "speeding"
    is_violation:  bool


@dataclass
class SpeedConfig:
    """Configuration for speed estimation."""
    pixels_per_meter:   float = 8.0       # calibration factor
    ema_alpha:          float = 0.25      # smoothing for speed updates
    speed_limit_kmh:    float = 80.0      # speed limit for violation detection
    max_physical_speed: float = 220.0     # Ignore speeds > this (outlier rejection)
    min_speed_frames:   int   = 8         # Wait for N frames before reporting speed
    median_window:      int   = 10        # Window size for median filtering
    # Speed class thresholds (km/h)
    stopped_max:        float = 5.0
    slow_max:           float = 30.0
    normal_max:         float = 60.0
    fast_max:           float = 80.0


# ---------------------------------------------------------------------------
# Speed Analyzer
# ---------------------------------------------------------------------------

class SpeedAnalyzer:
    """
    Real-time telemetry engine for vehicle speed and heading estimation.

    Tracks frame-to-frame displacement vectors and applies pixel-to-meter 
    calibration, statistical outlier rejection, and multi-stage 
    smoothing to provide stable speed metrics.

    Parameters
    ----------
    fps : float
        Video stream frame rate.
    config : SpeedConfig | None
        Configuration parameters for telemetry and calibration.
    """

    def __init__(
        self,
        fps: float = 30.0,
        config: SpeedConfig | None = None,
    ) -> None:
        self.fps = fps
        self.cfg = config or SpeedConfig()

        # Per-track state: {track_id: telemetry_buffer}
        self._track_state: dict[int, dict[str, Any]] = {}

        self._total_measurements = 0
        self._total_violations = 0

        logger.info(
            "SpeedAnalyzer initialised: fps=%.1f, px/m=%.1f, limit=%.0f km/h",
            fps, self.cfg.pixels_per_meter, self.cfg.speed_limit_kmh,
        )

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, tracks: Sequence[Track]) -> list[VehicleSpeedInfo]:
        """
        Analyse vehicle tracks and compute current telemetry.

        Parameters
        ----------
        tracks : Sequence[Track]
            List of confirmed vehicle tracks from the tracking engine.

        Returns
        -------
        list[VehicleSpeedInfo]
            List of valid telemetry records for the current frame.
        """
        results: list[VehicleSpeedInfo] = []
        active_ids: set[int] = set()

        for track in tracks:
            tid = track.track_id
            cx, cy = float(track.centre[0]), float(track.centre[1])
            active_ids.add(tid)

            if tid in self._track_state:
                state = self._track_state[tid]
                state["hits"] = state.get("hits", 0) + 1
                prev_cx, prev_cy = state["prev_center"]

                # Pixel displacement
                dx = cx - prev_cx
                dy = cy - prev_cy
                dist_px = math.sqrt(dx * dx + dy * dy)

                # Convert to real-world speed
                dist_m = dist_px / self.cfg.pixels_per_meter
                speed_ms = dist_m * self.fps
                speed_kmh = speed_ms * 3.6

                # Outlier rejection: Physical impossibility check
                if speed_kmh > self.cfg.max_physical_speed:
                    logger.debug("Speed outlier rejected (physically impossible) for #%d: %.1f km/h", tid, speed_kmh)
                    state["prev_center"] = (cx, cy)
                    continue

                # Outlier rejection: Acceleration check (sudden jumps)
                prev_speed = state.get("speed_ema", 0.0)
                if state["hits"] > self.cfg.min_speed_frames and abs(speed_kmh - prev_speed) > 40:
                    logger.debug("Speed outlier rejected (acceleration spike) for #%d: %.1f km/h", tid, speed_kmh)
                    state["prev_center"] = (cx, cy)
                    continue

                # Minimum Track Age: Ensure stability before reporting
                if state["hits"] < self.cfg.min_speed_frames:
                    state["prev_center"] = (cx, cy)
                    state["speed_ema"] = speed_kmh  # seed the EMA
                    continue

                # Median Filtering: History-based outlier rejection
                if "speed_history" not in state:
                    state["speed_history"] = []
                state["speed_history"].append(speed_kmh)
                if len(state["speed_history"]) > self.cfg.median_window:
                    state["speed_history"].pop(0)

                median_speed = float(np.median(state["speed_history"]))

                # Multi-stage smoothing: EMA on top of median filter
                prev_ema = state.get("speed_ema", median_speed)
                smoothed = (self.cfg.ema_alpha * median_speed
                            + (1 - self.cfg.ema_alpha) * prev_ema)

                # Direction (in degrees, 0=up)
                direction = math.degrees(math.atan2(dx, -dy)) % 360

                # Classification
                speed_class = self._classify_speed(smoothed)
                is_violation = smoothed > self.cfg.speed_limit_kmh

                # Update state
                state["prev_center"] = (cx, cy)
                state["speed_ema"] = smoothed
                state["direction"] = direction

                info = VehicleSpeedInfo(
                    track_id=tid,
                    speed_kmh=round(smoothed, 1),
                    direction_deg=round(direction, 1),
                    speed_class=speed_class,
                    is_violation=is_violation,
                )
                results.append(info)

                self._total_measurements += 1
                if is_violation:
                    self._total_violations += 1
            else:
                # First sighting — initialise state, no speed yet
                self._track_state[tid] = {
                    "prev_center": (cx, cy),
                    "speed_ema": 0.0,
                    "direction": 0.0,
                    "hits": 1,
                }

        # Prune stale tracks
        stale_ids = set(self._track_state.keys()) - active_ids
        for sid in stale_ids:
            del self._track_state[sid]

        return results

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify_speed(self, speed_kmh: float) -> str:
        """
        Map a numeric speed value to a descriptive traffic category.

        Parameters
        ----------
        speed_kmh : float
            Smoothing velocity in km/h.

        Returns
        -------
        str
            Category string (e.g., 'speeding').
        """
        if speed_kmh <= self.cfg.stopped_max:
            return "stopped"
        elif speed_kmh <= self.cfg.slow_max:
            return "slow"
        elif speed_kmh <= self.cfg.normal_max:
            return "normal"
        elif speed_kmh <= self.cfg.fast_max:
            return "fast"
        else:
            return "speeding"

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------

    def get_summary(self, results: list[VehicleSpeedInfo]) -> dict[str, Any]:
        """
        Compute high-level telemetry statistics for the current frame.

        Parameters
        ----------
        results : list[VehicleSpeedInfo]
            Current telemetry records.

        Returns
        -------
        dict[str, Any]
            Dictionary containing 'avg_speed_kmh', 'max_speed_kmh', 
            'violations', and 'speed_distribution'.
        """
        if not results:
            return {
                "avg_speed_kmh": 0.0,
                "max_speed_kmh": 0.0,
                "violations": 0,
                "speed_distribution": {},
            }

        speeds = [r.speed_kmh for r in results]
        classes = [r.speed_class for r in results]

        distribution: dict[str, int] = {}
        for cls in classes:
            distribution[cls] = distribution.get(cls, 0) + 1

        return {
            "avg_speed_kmh": round(float(np.mean(speeds)), 1),
            "max_speed_kmh": round(float(max(speeds)), 1),
            "min_speed_kmh": round(float(min(speeds)), 1),
            "violations": sum(1 for r in results if r.is_violation),
            "speed_distribution": distribution,
            "total_tracked": len(results),
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all track state."""
        self._track_state.clear()
        self._total_measurements = 0
        self._total_violations   = 0

    @property
    def total_measurements(self) -> int:
        return self._total_measurements

    @property
    def total_violations(self) -> int:
        return self._total_violations
