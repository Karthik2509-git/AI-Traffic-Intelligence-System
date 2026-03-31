"""
speed_analyzer.py - Vehicle speed and direction estimation from tracker data.

Uses Kalman filter track displacement data to compute real-time speed
estimates for each tracked vehicle.

Method
------
  1. Track displacement: distance_px = ||center_t - center_{t-1}|| per frame
  2. Convert to m/s using configurable pixels_per_meter calibration
  3. Apply EMA smoothing to reduce single-frame noise
  4. Classify: stopped, slow, normal, fast, speeding
  5. Detect speed violations and report to anomaly detector

Calibration
-----------
  The pixels_per_meter factor depends on camera angle and height.
  Default: 8.0 px/m (typical for a 640px-wide view of a 80m road).
  To calibrate: measure a known real-world distance in the frame.
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
    pixels_per_meter: float = 8.0       # calibration factor
    ema_alpha:        float = 0.3       # smoothing for speed updates
    speed_limit_kmh:  float = 80.0      # speed limit for violation detection
    # Speed class thresholds (km/h)
    stopped_max:      float = 5.0
    slow_max:         float = 30.0
    normal_max:       float = 60.0
    fast_max:         float = 80.0


# ---------------------------------------------------------------------------
# Speed Analyzer
# ---------------------------------------------------------------------------

class SpeedAnalyzer:
    """
    Real-time vehicle speed and direction estimator.

    Maintains per-track state to compute speed from frame-to-frame
    displacement, with EMA smoothing.

    Usage
    -----
    analyzer = SpeedAnalyzer(fps=30.0)
    speeds = analyzer.update(tracks)

    for info in speeds:
        print(f"Track {info.track_id}: {info.speed_kmh:.1f} km/h ({info.speed_class})")
    """

    def __init__(
        self,
        fps: float = 30.0,
        config: SpeedConfig | None = None,
    ) -> None:
        self.fps = fps
        self.cfg = config or SpeedConfig()

        # Per-track state: {track_id: {"prev_center": (x, y), "speed_ema": float}}
        self._track_state: dict[int, dict[str, Any]] = {}

        # Overall statistics
        self._total_measurements = 0
        self._total_violations   = 0

        logger.info(
            "SpeedAnalyzer initialised: fps=%.1f, px/m=%.1f, limit=%.0f km/h",
            fps, self.cfg.pixels_per_meter, self.cfg.speed_limit_kmh,
        )

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, tracks: list[Track]) -> list[VehicleSpeedInfo]:
        """
        Compute speed for all currently tracked vehicles.

        Parameters
        ----------
        tracks : List of confirmed Track objects from the SORT tracker.

        Returns
        -------
        List of VehicleSpeedInfo, one per track with valid speed data.
        """
        results: list[VehicleSpeedInfo] = []
        active_ids: set[int] = set()

        for track in tracks:
            tid = track.track_id
            cx, cy = float(track.centre[0]), float(track.centre[1])
            active_ids.add(tid)

            if tid in self._track_state:
                state = self._track_state[tid]
                prev_cx, prev_cy = state["prev_center"]

                # Pixel displacement
                dx = cx - prev_cx
                dy = cy - prev_cy
                dist_px = math.sqrt(dx * dx + dy * dy)

                # Convert to real-world speed
                dist_m = dist_px / self.cfg.pixels_per_meter
                speed_ms = dist_m * self.fps
                speed_kmh = speed_ms * 3.6

                # EMA smoothing
                prev_speed = state.get("speed_ema", speed_kmh)
                smoothed = (self.cfg.ema_alpha * speed_kmh
                            + (1 - self.cfg.ema_alpha) * prev_speed)

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
        """Compute aggregate speed stats from the latest update."""
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
