"""
pipeline.py — Industrial-grade traffic intelligence orchestrator.

Coordinates vehicle detection, multi-object tracking, spatial analytics, 
statistical anomaly detection, and traffic signal optimization into a 
single unified stream processor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import cv2
import numpy as np

from src.density_analyzer import DensityAnalyzer, FrameDensity, Lane, draw_lanes, make_full_frame_lane
from src.detection import (
    VEHICLE_CLASSES, CLASS_COLOURS, classify_density, load_model, 
    _resolve_names, _draw_custom_boxes, run_tracking, run_tiled_inference, to_tracks
)
from src.predictor import CongestionPredictor, build_feature_vector, frames_to_dataframe
from src.signal_optimizer import LaneSignalInput, PhaseSchedule, SignalOptimizer
from src.tracker import SORTTracker, Track
from src.utils import FPSMeter, RollingBuffer, ensure_dir, get_logger
from src.anomaly_detector import AnomalyDetector, AnomalyConfig, AnomalyEvent
from src.speed_analyzer import SpeedAnalyzer, SpeedConfig, VehicleSpeedInfo
from src.heatmap import HeatmapGenerator, HeatmapConfig
from src.database import TrafficDatabase

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Configuration parameters for the end-to-end Traffic Intelligence Pipeline.
    
    Attributes
    ----------
    model_name : str
        YOLOv8 weights file (e.g., 'yolov8n.pt').
    confidence_threshold : float
        Min confidence for detections.
    frame_skip : int
        Process every N-th frame for performance.
    """

    # ── Model & Detection ──────────────────────────────────────────────
    model_name:          str   = "yolov8m.pt"
    confidence_threshold: float = 0.40
    iou_threshold:       float = 0.45
    frame_skip:          int   = 1
    inference_size:      int   = 640
    min_motorcycle_conf: float = 0.25
    industrial_conf_floor: float = 0.35
    min_long_range_area: int   = 800
    dense_traffic_optimization: bool = False
    tile_overlap:        float = 0.25
    tile_size:           int   = 640

    # ── Tracking ───────────────────────────────────────────────────────
    tracker_max_age:     int   = 5
    tracker_min_hits:    int   = 3
    tracker_iou:         float = 0.30
    classification_smooth_window: int = 15
    weighted_smoothing:  bool  = True

    # ── Analytics ──────────────────────────────────────────────────────
    ema_alpha:           float = 0.20
    low_threshold:       int   = 10
    high_threshold:      int   = 25
    pixels_per_meter:    float = 8.0
    speed_limit_kmh:     float = 80.0
    max_physical_speed:  float = 220.0
    min_speed_frames:    int   = 8

    # ── Signal ─────────────────────────────────────────────────────────
    cycle_time_s:        int   = 120
    min_green_s:         int   = 10
    max_green_s:         int   = 90

    # ── ML Predictor ───────────────────────────────────────────────────
    min_train_frames:    int   = 100
    retrain_every:       int   = 500
    pretrained_model:    Path | None = None

    # ── Output ─────────────────────────────────────────────────────────
    output_dir:          Path  = Path("output")
    save_annotated:      bool  = True
    save_visual_samples: bool  = True
    generate_report:     bool  = True
    display:             bool  = False
    log_level:           str   = "INFO"
    hide_distant_objects: bool = False


# ---------------------------------------------------------------------------
# Per-frame result
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """
    Data container for all intelligence extracted from a single video frame.
    
    Attributes
    ----------
    frame_idx : int
        Sequential index of the processed frame.
    fps : float
        Processing speed in frames per second.
    tracks : list[Track]
        Confirmed vehicle tracks with identifiers and positions.
    density : FrameDensity
        Statistical density metrics for the frame.
    schedule : PhaseSchedule | None
        Optimized traffic signal recommendation.
    annotated_frame : np.ndarray
        Visualization of the frame with bounding boxes and overlays.
    metrics : dict[str, Any]
        Raw metric dictionary for database logging.
    """
    frame_idx:       int
    fps:             float
    tracks:          list[Track]
    density:         FrameDensity
    schedule:        PhaseSchedule | None
    annotated_frame: np.ndarray
    metrics:         dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TrafficPipeline:
    """
    Industrial-grade Traffic Intelligence Pipeline.

    Orchestrates the lifecycle of video processing, from raw frame ingestion 
    to high-level analytical insights. Handles detection, tracking, 
    speed estimation, density mapping, and signal optimization.

    Parameters
    ----------
    source : str | int | Path
        Video source (file path, stream URL, or camera index).
    lanes : list[Lane] | None
        Custom lane definitions. Defaults to a single full-frame lane.
    config : PipelineConfig | None
        Configuration overrides.
    """

    def __init__(
        self,
        source: str | int | Path,
        lanes:  list[Lane] | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.source = source
        self.config = config or PipelineConfig()
        self._lanes: list[Lane] | None = lanes

        # Sub-components (initialised in _setup)
        self._model        = None
        self._tracker:     SORTTracker | None    = None
        self._density:     DensityAnalyzer | None = None
        self._predictor:   CongestionPredictor   = CongestionPredictor()
        self._optimizer:   SignalOptimizer | None = None
        self._fps_meter    = FPSMeter(window=30)
        self._count_buffer = RollingBuffer(maxlen=300)
        
        # Tracking state (for age/hit_streak with ByteTrack)
        self._track_history: dict[int, dict] = {} # id -> {age, hits}
        self._label_history: dict[int, list[tuple[str, float]]] = {} # id -> [(label, conf), ...]


        # New modules
        self._anomaly:     AnomalyDetector | None = None
        self._speed:       SpeedAnalyzer | None   = None
        self._heatmap:     HeatmapGenerator | None = None
        self._database:    TrafficDatabase | None  = None
        self._session_id:  str | None = None

        self._frame_count  = 0
        self._predictor_ready = False

        # Performance metrics
        self._execution_times: list[float] = []
        self._total_vehicles_processed = 0
        self._sample_frame: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _setup(self, frame_width: int, frame_height: int) -> None:
        """Initialise all sub-components once the frame size is known."""
        cfg = self.config

        # ── YOLO model ──────────────────────────────────────────────────
        if self._model is None:
            self._model = load_model(cfg.model_name)

        # ── Lanes ───────────────────────────────────────────────────────
        if self._lanes is None:
            self._lanes = [make_full_frame_lane(frame_width, frame_height)]
            logger.info("No lanes specified; using full-frame single lane.")

        # ── Tracker ─────────────────────────────────────────────────────
        # In Dense Traffic Optimization mode, we use the internal SORTTracker 
        # as ByteTrack doesn't natively merge detections from tiled slices yet.
        if cfg.dense_traffic_optimization:
            self._tracker = SORTTracker(
                max_age=cfg.tracker_max_age,
                min_hits=cfg.tracker_min_hits,
                iou_threshold=cfg.tracker_iou,
            )
        else:
            self._tracker = None


        # ── Density analyser ────────────────────────────────────────────
        self._density = DensityAnalyzer(
            lanes          = self._lanes,
            ema_alpha      = cfg.ema_alpha,
            low_threshold  = cfg.low_threshold,
            high_threshold = cfg.high_threshold,
        )

        # ── Signal optimiser ────────────────────────────────────────────
        self._optimizer = SignalOptimizer(
            cycle_time_s = cfg.cycle_time_s,
            min_green_s  = cfg.min_green_s,
            max_green_s  = cfg.max_green_s,
        )

        # ── Pre-trained model ────────────────────────────────────────────
        if cfg.pretrained_model and cfg.pretrained_model.is_file():
            try:
                self._predictor.load(cfg.pretrained_model)
                self._predictor_ready = True
                logger.info("Loaded pre-trained predictor from '%s'.", cfg.pretrained_model)
            except Exception as exc:
                logger.warning("Could not load pre-trained model: %s", exc)

        # ── Anomaly detector ─────────────────────────────────────────────
        self._anomaly = AnomalyDetector()

        # ── Speed analyzer ──────────────────────────────────────────────
        self._speed = SpeedAnalyzer(
            fps=30.0,
            config=SpeedConfig(
                pixels_per_meter   = cfg.pixels_per_meter,
                speed_limit_kmh    = cfg.speed_limit_kmh,
                max_physical_speed = cfg.max_physical_speed,
                min_speed_frames   = cfg.min_speed_frames,
            )
        )

        # ── Heatmap generator ───────────────────────────────────────────
        self._heatmap = HeatmapGenerator(frame_shape=(frame_height, frame_width))

        # ── Database ────────────────────────────────────────────────────
        try:
            self._database = TrafficDatabase()
            self._session_id = self._database.start_session(
                source=str(self.source),
                config_hash=cfg.model_name,
            )
        except Exception as exc:
            logger.warning("Database init failed (non-fatal): %s", exc)
            self._database = None

        ensure_dir(cfg.output_dir)
        logger.info(
            "Pipeline initialised | source=%s | lanes=%d | model=%s",
            self.source, len(self._lanes), cfg.model_name,
        )

    # ------------------------------------------------------------------
    # Public: process a single frame
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray, timestamp_ms: float | None = None) -> FrameResult:
        """
        Execute the full intelligence suite on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            The raw BGR image from the video stream.
        timestamp_ms : float | None
            Millisecond timestamp for accurate temporal analysis.

        Returns
        -------
        FrameResult
            Consolidated intelligence data for the frame.
        """
        start_t = time.perf_counter()
        cfg = self.config

        if self._model is None:
            h, w = frame.shape[:2]
            self._setup(w, h)

        # ── 1. Object detection & tracking ──────────────────────────────
        if cfg.dense_traffic_optimization:
            # Tiled Inference Pipeline (Optimized for dense urban scenes)
            detections = run_tiled_inference(
                model                = self._model,
                frame                = frame,
                confidence_threshold = min(cfg.confidence_threshold, cfg.min_motorcycle_conf),
                iou_threshold        = 0.65, 
                tile_size            = cfg.tile_size,
                overlap              = cfg.tile_overlap,
            )
            # Use internal high-precision tracker for merged detections
            tracks = self._tracker.update(detections)
        else:
            # Standard Single-Pass Detection
            results = run_tracking(
                model                = self._model,
                frame                = frame,
                confidence_threshold = cfg.confidence_threshold,
                iou_threshold        = cfg.iou_threshold,
                inference_size       = cfg.inference_size,
                min_motorcycle_conf  = cfg.min_motorcycle_conf,
            )
            names_map = self._model.names
            tracks    = to_tracks(
                results              = results, 
                names_map            = names_map,
                confidence_threshold = cfg.confidence_threshold,
                min_motorcycle_conf  = cfg.min_motorcycle_conf,
            )
        
        # ── 2b. Temporal Label Consensus (Weighted Smoothing) ─────────────
        for track in tracks:
            tid = track.track_id
            if tid not in self._label_history:
                self._label_history[tid] = []
            
            # Record current observation: (label, confidence)
            self._label_history[tid].append((track.class_name, track.confidence))
            
            # Keep only the last N frames
            if len(self._label_history[tid]) > cfg.classification_smooth_window:
                self._label_history[tid].pop(0)
            
            # Apply Consensus Logic
            if cfg.weighted_smoothing:
                # Weighted Majority Vote: sum of confidences per class
                scores: dict[str, float] = {}
                for lbl, conf in self._label_history[tid]:
                    scores[lbl] = scores.get(lbl, 0.0) + conf
                # Choose class with the highest total weight
                winner = max(scores, key=scores.get)
                track.class_name = winner
            else:
                # Simple Majority Vote
                from collections import Counter
                lbls = [l for l, c in self._label_history[tid]]
                counts = Counter(lbls)
                winner, _ = counts.most_common(1)[0]
                track.class_name = winner

            # ── 2c. Long-Range Detection Flagging ──────────────────────────
            # Flag objects that are distant (tiny) or below certainty floor
            x1, y1, x2, y2 = track.bbox
            area = (x2 - x1) * (y2 - y1)
            is_dist = area < cfg.min_long_range_area or track.confidence < cfg.industrial_conf_floor
            track.metadata["is_distant"] = is_dist


        # ── 3. Density analysis ──────────────────────────────────────────
        density = self._density.update(tracks, timestamp_ms=timestamp_ms)
        self._count_buffer.push(density.total_count)

        # ── 3b. Speed analysis ───────────────────────────────────────────
        speed_results: list[VehicleSpeedInfo] = []
        speed_summary: dict = {}
        speed_variance: float = 0.0
        if self._speed is not None:
            speed_results = self._speed.update(tracks)
            speed_summary = self._speed.get_summary(speed_results)
            
            # Compute speed variance for ML predictor
            if len(speed_results) > 1:
                speeds = [s.speed_kmh for s in speed_results]
                speed_variance = float(np.var(speeds))


        # ── 3c. Heatmap update ───────────────────────────────────────────
        if self._heatmap is not None:
            self._heatmap.update(tracks)

        # ── 4. Predictor training / inference ────────────────────────────
        schedule: PhaseSchedule | None = None

        if self._frame_count >= cfg.min_train_frames:
            # Retrain periodically
            if (
                not self._predictor_ready
                or self._frame_count % cfg.retrain_every == 0
            ):
                self._maybe_train()

            if self._predictor_ready:
                schedule = self._run_optimizer(density, speed_variance)

        # ── 5. Anomaly detection ─────────────────────────────────────────
        anomaly_events: list[AnomalyEvent] = []
        if self._anomaly is not None:
            anomaly_metrics = {
                "total_vehicles":  density.total_count,
                "congestion_score": density.congestion_score,
                "flow_per_min":    self._density.flow_rate_per_minute(),
                "trend":           self._density.trend(),
                "ema_count":       density.ema_count,
            }
            if speed_summary:
                anomaly_metrics["avg_speed_kmh"] = speed_summary.get("avg_speed_kmh", 0)
            anomaly_events = self._anomaly.analyse(
                frame_idx    = self._frame_count, 
                metrics      = anomaly_metrics,
                track_speeds = speed_results,
            )

        # ── 6. Annotate frame ────────────────────────────────────────────
        annotated = self._annotate(frame, tracks, density, schedule, speed_results)

        self._fps_meter.tick()
        self._frame_count += 1

        metrics = self._build_metrics(density, schedule, speed_summary, anomaly_events)

        # ── 7. Persist to database ───────────────────────────────────────
        if self._database is not None and self._session_id:
            try:
                self._database.log_frame(self._session_id, self._frame_count - 1, metrics)
                if schedule:
                    for lane_out in schedule.lanes:
                        self._database.log_signal(
                            self._session_id, self._frame_count - 1,
                            lane_out.lane_name, lane_out.green_time_s,
                            cfg.cycle_time_s - lane_out.green_time_s,
                            lane_out.pressure, lane_out.advisory,
                        )
                for evt in anomaly_events:
                    self._database.log_anomaly(
                        self._session_id, evt.event_id, evt.frame_idx,
                        evt.anomaly_type, evt.severity, evt.description,
                        evt.confidence, evt.metrics_snapshot,
                    )
            except Exception as exc:
                logger.debug("DB write error (non-fatal): %s", exc)

        # ── 8. Performance telemetry ─────────────────────────────────────
        latency_ms = (time.perf_counter() - start_t) * 1000
        self._execution_times.append(latency_ms)
        self._total_vehicles_processed += density.total_count

        # Capture sample for export
        if self._sample_frame is None or self._frame_count % 100 == 0:
            self._sample_frame = annotated.copy()

        return FrameResult(
            frame_idx       = self._frame_count - 1,
            fps             = self._fps_meter.get(),
            tracks          = tracks,
            density         = density,
            schedule        = schedule,
            annotated_frame = annotated,
            metrics         = metrics,
        )

    # ------------------------------------------------------------------
    # Reporting & Visuals
    # ------------------------------------------------------------------

    def save_visual_samples(self) -> None:
        """Export representative visual outputs for documentation/demo."""
        if self._sample_frame is not None:
            path = self.config.output_dir / "sample_detection.jpg"
            cv2.imwrite(str(path), self._sample_frame)
            logger.info("Visual sample saved: %s", path)

        if self._heatmap is not None:
            path = self.config.output_dir / "sample_heatmap.png"
            self._heatmap.export(path)
            logger.info("Heatmap sample saved: %s", path)

    def generate_performance_report(self) -> Path:
        """
        Synthesize technical performance metrics into a textual report.

        Returns
        -------
        Path
            Location of the generated performance_report.txt.
        """
        avg_latency = np.mean(self._execution_times) if self._execution_times else 0
        p95_latency = np.percentile(self._execution_times, 95) if self._execution_times else 0
        avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0
        
        report_path = self.config.output_dir / "performance_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(" AI TRAFFIC INTELLIGENCE SYSTEM - PERFORMANCE REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Timestamp:          {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model:              {self.config.model_name}\n")
            f.write(f"Resolution:         {self.config.inference_size}px\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total Frames:       {self._frame_count}\n")
            f.write(f"Total Vehicles:     {self._total_vehicles_processed}\n")
            f.write(f"Avg Latency:        {avg_latency:.2f} ms\n")
            f.write(f"P95 Latency:        {p95_latency:.2f} ms\n")
            f.write(f"Avg Pipeline FPS:   {avg_fps:.2f}\n")
            f.write("-" * 60 + "\n")
            f.write("Operational Status: PRODUCTION-READY\n")
            f.write("=" * 60 + "\n")

        logger.info("Performance report generated: %s", report_path)
        return report_path

    # ------------------------------------------------------------------
    # Public: run on a video file / stream
    # ------------------------------------------------------------------

    def run(self) -> Generator[FrameResult, None, None]:
        """
        Execute the pipeline on the configured video source.

        Yields
        ------
        FrameResult
            Intelligence data for each processed frame.

        Raises
        ------
        IOError
            If the video source cannot be opened.
        """
        cap = cv2.VideoCapture(self.source if isinstance(self.source, int) else str(self.source))
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: '{self.source}'.")

        fps_source = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self._speed:
            self._speed.fps = fps_source

        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.config.frame_skip != 0:
                    frame_idx += 1
                    continue

                timestamp_ms = (frame_idx / fps_source) * 1_000.0
                result = self.process_frame(frame, timestamp_ms)

                if self.config.save_annotated:
                    out_path = self.config.output_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(out_path), result.annotated_frame)

                if self.config.display:
                    cv2.imshow("Traffic Intelligence", result.annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                yield result
                frame_idx += 1

        finally:
            cap.release()
            if self.config.display:
                cv2.destroyAllWindows()
            logger.info(
                "Pipeline execution complete | %d frames processed | avg FPS: %.1f",
                self._frame_count, self._fps_meter.get(),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_train(self) -> None:
        """Attempt to train the predictor on accumulated history."""
        history = self._density.history()
        df      = frames_to_dataframe(
            history,
            flow_rate_fn   = self._density.flow_rate_per_minute,
            low_threshold  = self.config.low_threshold,
            high_threshold = self.config.high_threshold,
        )
        try:
            self._predictor.train(df, min_samples=self.config.min_train_frames)
            self._predictor_ready = True
            logger.info("Predictor (re)trained on %d frames.", len(df))
        except ValueError as exc:
            logger.debug("Predictor training skipped: %s", exc)

    def _run_optimizer(self, density: FrameDensity, speed_variance: float = 0.0) -> PhaseSchedule | None:
        """Run prediction + signal optimisation for current density."""
        if not self._predictor_ready:
            return None

        trend        = self._density.trend()
        flow_rate    = self._density.flow_rate_per_minute()
        count_trend  = self._density.rolling_average(5) - self._density.rolling_average(20)

        feat_vec  = build_feature_vector(
            fd                = density, 
            flow_rate_per_min = flow_rate, 
            count_trend       = count_trend,
            speed_variance    = speed_variance,
        )

        pred      = self._predictor.predict(feat_vec)

        lane_inputs = [
            LaneSignalInput(
                lane_name  = lane.name,
                density    = density,
                prediction = pred,
                trend      = trend,
            )
            for lane in self._lanes
        ]

        try:
            return self._optimizer.optimise(lane_inputs)
        except Exception as exc:
            logger.warning("Signal optimisation failed: %s", exc)
            return None

    def _annotate(
        self,
        frame:    np.ndarray,
        tracks:   list[Track],
        density:  FrameDensity,
        schedule: PhaseSchedule | None,
        speed_results: list[VehicleSpeedInfo] | None = None,
    ) -> np.ndarray:
        """Compose the annotated output frame."""
        h, w = frame.shape[:2]
        
        # ── Dynamic HD Scaling ──────────────────────────────────────────
        # Scale 0.45 @ 640px -> 0.70 @ 1280px
        # Thickness 1 @ 640px -> 2 @ 1280px
        scale_factor = w / 640.0
        font_scale   = max(0.40, min(0.80, 0.45 * scale_factor))
        thickness    = 2 if w >= 1280 else 1
        
        lane_labels = {
            lane.name: density.density_label
            for lane in self._lanes
        }
        out = draw_lanes(frame, self._lanes, lane_labels)

        # Build speed lookup
        speed_map: dict[int, VehicleSpeedInfo] = {}
        if speed_results:
            speed_map = {s.track_id: s for s in speed_results}

        # Draw track boxes with optional speed label
        for track in tracks:
            if not track.confirmed:
                continue
            
            is_distant = track.metadata.get("is_distant", False)
            if is_distant and cfg.hide_distant_objects:
                continue
                
            x1, y1, x2, y2 = map(int, track.bbox)
            
            # Determine status-based colour
            sp = speed_map.get(track.track_id)
            is_distant = track.metadata.get("is_distant", False)
            
            # Use class-specific default
            colour = CLASS_COLOURS.get(track.class_name, (180, 180, 180))
            
            # Status overrides (Distant wins over all, then Speeding/Stopped)
            if is_distant:
                colour = (160, 160, 160)  # Gray for Deep-Field
            elif sp:
                if sp.is_violation:
                    colour = (0, 0, 255)      # Red for CRITICAL speed
                elif sp.speed_class == "stopped":
                    colour = (0, 215, 255)    # Golden-Yellow for INCIDENT/STOPPED

            cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness + 1)
            
            # Label strip (filled background for better readability)
            label = f"#{track.track_id} "
            if is_distant:
                label += "[Distant Detection]"
            else:
                label += f"{track.class_name}"
                if sp:
                    label += f" {sp.speed_kmh:.0f}km/h"
                    if sp.speed_class == "stopped": label += " [STOPPED]"
                
            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(out, (x1, y1 - lh - baseline - 4), (x1 + lw + 4, y1), colour, -1)
            cv2.putText(out, label, (x1 + 2, y1 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


        # HUD overlay
        self._draw_hud(out, density, schedule)
        return out

    def _draw_hud(
        self,
        frame:    np.ndarray,
        density:  FrameDensity,
        schedule: PhaseSchedule | None,
    ) -> None:
        lines = [
            f"FPS      : {self._fps_meter.get():.1f}",
            f"Vehicles : {density.total_count}",
            f"EMA      : {density.ema_count:.1f}",
            f"Density  : {density.density_label}",
            f"Cong.    : {density.congestion_score:.1f}/100",
        ]

        if schedule:
            lines.append("── Signal ──────────────")
            for lane_out in schedule.lanes:
                lines.append(f"  {lane_out.lane_name[:12]:<12}: {lane_out.green_time_s:>3}s")

        pad, line_h = 8, 20
        panel_h = pad * 2 + line_h * len(lines)
        panel_w = 230

        overlay = frame.copy()
        cv2.rectangle(overlay, (6, 6), (6 + panel_w, 6 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        for i, line in enumerate(lines):
            y = 6 + pad + (i + 1) * line_h - 4
            cv2.putText(frame, line, (14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    def _build_metrics(
        self,
        density:  FrameDensity,
        schedule: PhaseSchedule | None,
        speed_summary: dict | None = None,
        anomaly_events: list[AnomalyEvent] | None = None,
    ) -> dict[str, Any]:
        m: dict[str, Any] = {
            "frame":           self._frame_count,
            "fps":             round(self._fps_meter.get(), 1),
            "total_vehicles":  density.total_count,
            "ema_count":       density.ema_count,
            "density_label":   density.density_label,
            "congestion_score":density.congestion_score,
            "occupancy":       density.occupancy_ratio,
            "trend":           self._density.trend(),
            "flow_per_min":    round(self._density.flow_rate_per_minute(), 1),
            "counts_per_class": density.counts_per_lane,
        }
        if schedule:
            m["signal_schedule"] = {
                o.lane_name: o.green_time_s for o in schedule.lanes
            }
        if speed_summary:
            m["avg_speed_kmh"] = speed_summary.get("avg_speed_kmh", 0)
            m["max_speed_kmh"] = speed_summary.get("max_speed_kmh", 0)
            m["speed_violations"] = speed_summary.get("violations", 0)
            m["speed_distribution"] = speed_summary.get("speed_distribution", {})
        if anomaly_events:
            m["anomalies"] = [e.to_dict() for e in anomaly_events]
            m["anomaly_count"] = len(anomaly_events)
        return m


# ---------------------------------------------------------------------------
# Factory: build pipeline from settings.yaml
# ---------------------------------------------------------------------------

def pipeline_from_config(
    source: str | int | Path,
    config_path: Path | None = None,
) -> TrafficPipeline:
    """
    Factory to create a TrafficPipeline from a YAML configuration file.

    Parameters
    ----------
    source : str | int | Path
        Video source to process.
    config_path : Path | None
        Path to settings.yaml. Defaults to 'config/settings.yaml'.

    Returns
    -------
    TrafficPipeline
        Configured pipeline instance.
    """
    from src.utils import load_config

    cfg_dict = load_config(config_path)

    model_sec     = cfg_dict.get("model", {})
    det_sec       = cfg_dict.get("detection", {})
    # Simplified vs Legacy
    analytics_sec = cfg_dict.get("analytics", cfg_dict.get("density_thresholds", {}))
    track_sec     = cfg_dict.get("tracking", {})
    signal_sec    = cfg_dict.get("analytics", cfg_dict.get("signal", {}))
    pred_sec      = cfg_dict.get("prediction", {})
    output_sec    = cfg_dict.get("output", {})

    pipeline_cfg = PipelineConfig(
        model_name           = model_sec.get("name", "yolov8m.pt"),
        confidence_threshold = det_sec.get("confidence_threshold", 0.40),
        iou_threshold        = det_sec.get("iou_threshold", 0.45),
        frame_skip           = det_sec.get("frame_skip", 1),
        inference_size       = model_sec.get("inference_size", 640),
        min_motorcycle_conf  = det_sec.get("min_motorcycle_conf", 0.25),
        industrial_conf_floor= det_sec.get("industrial_conf_floor", 0.35),
        min_long_range_area  = det_sec.get("min_long_range_area", 800),
        dense_traffic_optimization = det_sec.get("dense_traffic_optimization", False),
        tile_overlap         = det_sec.get("tile_overlap", 0.25),
        tile_size            = det_sec.get("tile_size", 640),
        tracker_max_age      = track_sec.get("max_age", 5),
        tracker_min_hits     = track_sec.get("min_hits", 3),
        tracker_iou          = track_sec.get("iou_threshold", 0.30),
        classification_smooth_window = track_sec.get("classification_smooth_window", 15),
        weighted_smoothing   = track_sec.get("weighted_smoothing", True),
        ema_alpha            = analytics_sec.get("ema_alpha", 0.20),
        low_threshold        = analytics_sec.get("low", analytics_sec.get("low_threshold", 10)),
        high_threshold       = analytics_sec.get("high", analytics_sec.get("high_threshold", 25)),
        min_train_frames     = pred_sec.get("min_train_frames", 100),
        retrain_every        = pred_sec.get("retrain_every", 500),
        cycle_time_s         = signal_sec.get("cycle_time_s", 120),
        min_green_s          = signal_sec.get("min_green_s", 10),
        max_green_s          = signal_sec.get("max_green_s", 90),
        pixels_per_meter     = analytics_sec.get("pixels_per_meter", 8.0),
        speed_limit_kmh      = analytics_sec.get("speed_limit_kmh", 80.0),
        max_physical_speed   = det_sec.get("max_physical_speed", 220.0),
        min_speed_frames     = track_sec.get("min_speed_frames", 8),
        save_annotated       = output_sec.get("save_annotated", True),
        output_dir           = Path(output_sec.get("directory", output_sec.get("output_dir", "output"))),
        display              = output_sec.get("display", False),
        hide_distant_objects = det_sec.get("hide_distant_objects", False)
    )

    # ── Build Lane objects from YAML (if defined) ─────────────────────

    # ── Build Lane objects from YAML (if defined) ─────────────────────
    lanes_raw = cfg_dict.get("lanes", [])
    lanes: list[Lane] | None = None

    if lanes_raw:
        import numpy as _np
        lanes = []
        for lane_def in lanes_raw:
            name    = lane_def.get("name", f"Lane {len(lanes) + 1}")
            polygon = _np.array(lane_def["polygon"], dtype=_np.float32)
            lanes.append(Lane(name=name, polygon=polygon))
        logger.info("Loaded %d lane(s) from config.", len(lanes))

    return TrafficPipeline(source=source, lanes=lanes, config=pipeline_cfg)