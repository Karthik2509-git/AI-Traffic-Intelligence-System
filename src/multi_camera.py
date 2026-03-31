"""
multi_camera.py — Multi-camera traffic simulation and comparative analysis.

Enables running multiple TrafficPipeline instances in parallel (one per camera)
and aggregating / comparing their metrics for cross-junction decision making.

Features:
  • Named camera sources with independent pipeline configs
  • Per-camera metric snapshots (density, congestion, signal recommendations)
  • Cross-camera comparative analysis (busiest junction, risk ranking)
  • Aggregated system-wide statistics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.density_analyzer import DensityAnalyzer, FrameDensity, Lane, make_full_frame_lane
from src.pipeline import PipelineConfig, TrafficPipeline, FrameResult
from src.signal_optimizer import PhaseSchedule
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-camera snapshot
# ---------------------------------------------------------------------------

@dataclass
class CameraSnapshot:
    """Latest metrics for a single camera."""
    camera_name:       str
    frame_idx:         int              = 0
    fps:               float            = 0.0
    total_vehicles:    int              = 0
    density_label:     str              = "Low"
    congestion_score:  float            = 0.0
    ema_count:         float            = 0.0
    occupancy:         float            = 0.0
    trend:             str              = "stable"
    flow_per_min:      float            = 0.0
    schedule:          PhaseSchedule | None = None
    annotated_frame:   np.ndarray | None   = None


@dataclass
class SystemSummary:
    """Aggregated metrics across all cameras."""
    total_cameras:      int
    total_vehicles:     int
    avg_congestion:     float
    busiest_camera:     str
    risk_ranking:       list[str]         # cameras sorted by congestion (highest first)
    per_camera:         list[CameraSnapshot]
    system_status:      str               # "Normal", "Elevated", "Critical"


# ---------------------------------------------------------------------------
# Camera definition
# ---------------------------------------------------------------------------

@dataclass
class CameraSource:
    """
    Definition of a single camera feed.

    Parameters
    ----------
    name    : Human-readable camera identifier (e.g., "Junction A - North").
    source  : Video file path, RTSP URL, or integer webcam index.
    lanes   : Optional list of Lane objects for this camera.
    config  : Optional PipelineConfig override for this camera.
    """
    name:   str
    source: str | int | Path
    lanes:  list[Lane] | None       = None
    config: PipelineConfig | None   = None


# ---------------------------------------------------------------------------
# Multi-camera manager
# ---------------------------------------------------------------------------

class MultiCameraManager:
    """
    Manages multiple TrafficPipeline instances for comparative analysis.

    Usage
    -----
    cameras = [
        CameraSource("Junction A", "video_a.mp4"),
        CameraSource("Junction B", "video_b.mp4"),
    ]
    manager = MultiCameraManager(cameras)

    # Process one frame from each camera
    snapshots = manager.process_all_next()
    summary   = manager.system_summary()
    """

    def __init__(
        self,
        cameras: Sequence[CameraSource],
        default_config: PipelineConfig | None = None,
    ) -> None:
        if not cameras:
            raise ValueError("At least one CameraSource is required.")

        self._cameras = list(cameras)
        self._default_config = default_config or PipelineConfig()

        # Build pipelines
        self._pipelines: dict[str, TrafficPipeline] = {}
        self._snapshots: dict[str, CameraSnapshot]  = {}
        self._caps: dict[str, Any] = {}

        for cam in self._cameras:
            cfg = cam.config or self._default_config
            pipeline = TrafficPipeline(
                source = cam.source,
                lanes  = cam.lanes,
                config = cfg,
            )
            self._pipelines[cam.name]  = pipeline
            self._snapshots[cam.name]  = CameraSnapshot(camera_name=cam.name)

        logger.info("MultiCameraManager initialised with %d cameras.", len(cameras))

    @property
    def camera_names(self) -> list[str]:
        return [c.name for c in self._cameras]

    def get_pipeline(self, camera_name: str) -> TrafficPipeline:
        """Get the TrafficPipeline for a specific camera."""
        if camera_name not in self._pipelines:
            raise KeyError(f"Unknown camera: '{camera_name}'")
        return self._pipelines[camera_name]

    def get_snapshot(self, camera_name: str) -> CameraSnapshot:
        """Get the latest snapshot for a specific camera."""
        return self._snapshots.get(camera_name, CameraSnapshot(camera_name=camera_name))

    # ------------------------------------------------------------------
    # Process a single frame for one camera
    # ------------------------------------------------------------------

    def process_frame(
        self,
        camera_name: str,
        frame: np.ndarray,
        timestamp_ms: float | None = None,
    ) -> CameraSnapshot:
        """
        Process a single frame for one camera and update its snapshot.

        Parameters
        ----------
        camera_name  : Name of the camera to process.
        frame        : BGR image array.
        timestamp_ms : Optional wall-clock timestamp in ms.

        Returns
        -------
        Updated CameraSnapshot for this camera.
        """
        pipeline = self.get_pipeline(camera_name)
        result   = pipeline.process_frame(frame, timestamp_ms)

        snap = self._result_to_snapshot(camera_name, result)
        self._snapshots[camera_name] = snap
        return snap

    # ------------------------------------------------------------------
    # Comparative analysis
    # ------------------------------------------------------------------

    def system_summary(self) -> SystemSummary:
        """
        Aggregate metrics across all cameras and produce a system-wide summary.

        Returns
        -------
        SystemSummary with per-camera snapshots + aggregated stats.
        """
        snaps = list(self._snapshots.values())

        total_vehicles = sum(s.total_vehicles for s in snaps)
        avg_congestion = (
            float(np.mean([s.congestion_score for s in snaps])) if snaps else 0.0
        )

        # Sort by congestion score descending
        ranked = sorted(snaps, key=lambda s: s.congestion_score, reverse=True)
        risk_ranking = [s.camera_name for s in ranked]
        busiest = risk_ranking[0] if risk_ranking else "—"

        # System status
        if avg_congestion > 70:
            status = "Critical"
        elif avg_congestion > 40:
            status = "Elevated"
        else:
            status = "Normal"

        return SystemSummary(
            total_cameras  = len(snaps),
            total_vehicles = total_vehicles,
            avg_congestion = round(avg_congestion, 2),
            busiest_camera = busiest,
            risk_ranking   = risk_ranking,
            per_camera     = snaps,
            system_status  = status,
        )

    def comparative_table(self) -> list[dict[str, Any]]:
        """
        Return a list of dicts suitable for pd.DataFrame display.

        Each dict contains key metrics for one camera.
        """
        rows: list[dict[str, Any]] = []
        for snap in self._snapshots.values():
            row = {
                "Camera":           snap.camera_name,
                "Vehicles":         snap.total_vehicles,
                "Density":          snap.density_label,
                "Congestion":       round(snap.congestion_score, 1),
                "EMA Count":        round(snap.ema_count, 1),
                "Occupancy":        round(snap.occupancy, 3),
                "Trend":            snap.trend,
                "Flow (veh/min)":   round(snap.flow_per_min, 1),
                "FPS":              round(snap.fps, 1),
            }
            if snap.schedule:
                for lout in snap.schedule.lanes:
                    row[f"Green ({lout.lane_name})"] = f"{lout.green_time_s}s"
            rows.append(row)
        return rows

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, camera_name: str | None = None) -> None:
        """Reset one or all camera pipelines."""
        if camera_name:
            if camera_name in self._pipelines:
                self._snapshots[camera_name] = CameraSnapshot(camera_name=camera_name)
                logger.info("Reset camera '%s'.", camera_name)
        else:
            for name in self._pipelines:
                self._snapshots[name] = CameraSnapshot(camera_name=name)
            logger.info("Reset all %d cameras.", len(self._pipelines))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _result_to_snapshot(camera_name: str, result: FrameResult) -> CameraSnapshot:
        metrics = result.metrics
        return CameraSnapshot(
            camera_name      = camera_name,
            frame_idx        = result.frame_idx,
            fps              = result.fps,
            total_vehicles   = metrics.get("total_vehicles", 0),
            density_label    = metrics.get("density_label", "Low"),
            congestion_score = metrics.get("congestion_score", 0.0),
            ema_count        = metrics.get("ema_count", 0.0),
            occupancy        = metrics.get("occupancy", 0.0),
            trend            = metrics.get("trend", "stable"),
            flow_per_min     = metrics.get("flow_per_min", 0.0),
            schedule         = result.schedule,
            annotated_frame  = result.annotated_frame,
        )
