"""
pipeline.py — End-to-end video-processing orchestrator.

Ties together:
  Detection → Tracking → Density analysis → ML prediction → Signal optimisation

Designed for:
  • Single-camera video file or live RTSP stream
  • Optional multi-camera comparison mode (run one Pipeline per camera)
  • Edge-AI mode: YOLOv8-nano + reduced resolution for Raspberry Pi / Jetson

Architecture
------------
  VideoCapture
    └─► Frame sampler (configurable skip)
          └─► YOLOv8 detector (detection.py)
                └─► SORT multi-object tracker (tracker.py)
                      └─► DensityAnalyzer (density_analyzer.py)
                            └─► CongestionPredictor (predictor.py)
                                  └─► SignalOptimizer (signal_optimizer.py)
                                        └─► Annotated output + metrics dict

Every component is independently replaceable — swap YOLO for any other
detector that returns {bbox, class_name, confidence} dicts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import cv2
import numpy as np

from src.density_analyzer import DensityAnalyzer, FrameDensity, Lane, draw_lanes, make_full_frame_lane
from src.detection import VEHICLE_CLASSES, classify_density, load_model, _resolve_names, _draw_custom_boxes
from src.predictor import CongestionPredictor, build_feature_vector, frames_to_dataframe
from src.signal_optimizer import LaneSignalInput, PhaseSchedule, SignalOptimizer
from src.tracker import SORTTracker, Track
from src.utils import FPSMeter, RollingBuffer, ensure_dir, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """All tunable parameters for the end-to-end pipeline."""

    # ── Detection ──────────────────────────────────────────────────────
    model_name:          str   = "yolov8n.pt"
    confidence_threshold: float = 0.40
    iou_threshold:       float = 0.45
    frame_skip:          int   = 1          # process every N-th frame (1 = all)
    inference_size:      int   = 640        # YOLO input resolution

    # ── Tracking ───────────────────────────────────────────────────────
    tracker_max_age:     int   = 5
    tracker_min_hits:    int   = 3
    tracker_iou:         float = 0.30

    # ── Density ────────────────────────────────────────────────────────
    ema_alpha:           float = 0.20
    low_threshold:       int   = 10
    high_threshold:      int   = 25

    # ── Prediction ─────────────────────────────────────────────────────
    min_train_frames:    int   = 100        # frames before first training
    retrain_every:       int   = 500        # retrain every N frames
    pretrained_model:    Path | None = None

    # ── Signal ─────────────────────────────────────────────────────────
    cycle_time_s:        int   = 120
    min_green_s:         int   = 10
    max_green_s:         int   = 90

    # ── Output ─────────────────────────────────────────────────────────
    save_annotated:      bool  = True
    output_dir:          Path  = Path("output")
    display:             bool  = False      # cv2.imshow


# ---------------------------------------------------------------------------
# Per-frame result
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """Everything the pipeline knows about one processed frame."""
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
    Single-camera traffic intelligence pipeline.

    Parameters
    ----------
    source : Video file path, RTSP URL, or integer webcam index.
    lanes  : Optional list of Lane objects. Defaults to full-frame single lane.
    config : PipelineConfig instance.
    """

    def __init__(
        self,
        source: str | int | Path,
        lanes:  list[Lane] | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.source = source
        self.config = config or PipelineConfig()
        self._lanes: list[Lane] | None = lanes  # deferred until first frame

        # Sub-components (initialised in _setup)
        self._model        = None
        self._tracker:     SORTTracker | None    = None
        self._density:     DensityAnalyzer | None = None
        self._predictor:   CongestionPredictor   = CongestionPredictor()
        self._optimizer:   SignalOptimizer | None = None
        self._fps_meter    = FPSMeter(window=30)
        self._count_buffer = RollingBuffer(maxlen=300)

        self._frame_count  = 0
        self._predictor_ready = False

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
        self._tracker = SORTTracker(
            max_age       = cfg.tracker_max_age,
            min_hits      = cfg.tracker_min_hits,
            iou_threshold = cfg.tracker_iou,
        )

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
        Run the full pipeline on a single BGR frame.

        Returns a FrameResult containing tracks, density, signal schedule,
        and an annotated copy of the frame.
        """
        cfg = self.config

        if self._model is None:
            h, w = frame.shape[:2]
            self._setup(w, h)

        # ── 1. YOLO detection ────────────────────────────────────────────
        results = self._model(
            frame,
            conf    = cfg.confidence_threshold,
            iou     = cfg.iou_threshold,
            imgsz   = cfg.inference_size,
            verbose = False,
        )
        prediction = results[0]
        names_map  = _resolve_names(prediction, self._model)

        detections: list[dict] = []
        if prediction.boxes is not None and prediction.boxes.cls is not None:
            cls_ids = prediction.boxes.cls.cpu().numpy().astype(int)
            confs   = prediction.boxes.conf.cpu().numpy().astype(float)
            xyxys   = prediction.boxes.xyxy.cpu().numpy()

            for cls_id, conf, xyxy in zip(cls_ids, confs, xyxys):
                name = names_map.get(int(cls_id), "")
                if name in VEHICLE_CLASSES and conf >= cfg.confidence_threshold:
                    detections.append({
                        "bbox":       xyxy.tolist(),
                        "class_name": name,
                        "confidence": float(conf),
                    })

        # ── 2. Multi-object tracking ─────────────────────────────────────
        tracks = self._tracker.update(detections)

        # ── 3. Density analysis ──────────────────────────────────────────
        density = self._density.update(tracks, timestamp_ms=timestamp_ms)
        self._count_buffer.push(density.total_count)

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
                schedule = self._run_optimizer(density)

        # ── 5. Annotate frame ────────────────────────────────────────────
        annotated = self._annotate(frame, tracks, density, schedule)

        self._fps_meter.tick()
        self._frame_count += 1

        return FrameResult(
            frame_idx       = self._frame_count - 1,
            fps             = self._fps_meter.get(),
            tracks          = tracks,
            density         = density,
            schedule        = schedule,
            annotated_frame = annotated,
            metrics         = self._build_metrics(density, schedule),
        )

    # ------------------------------------------------------------------
    # Public: run on a video file / stream
    # ------------------------------------------------------------------

    def run(self) -> Generator[FrameResult, None, None]:
        """
        Process all frames from self.source and yield FrameResult per frame.

        Usage
        -----
        for result in pipeline.run():
            process(result)
        """
        cap = cv2.VideoCapture(self.source if isinstance(self.source, int) else str(self.source))
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: '{self.source}'.")

        fps_source = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx  = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.config.frame_skip != 0:
                    frame_idx += 1
                    continue

                timestamp_ms = (frame_idx / fps_source) * 1_000.0
                result       = self.process_frame(frame, timestamp_ms)

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
                "Pipeline finished | %d frames processed | avg FPS: %.1f",
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

    def _run_optimizer(self, density: FrameDensity) -> PhaseSchedule | None:
        """Run prediction + signal optimisation for current density."""
        if not self._predictor_ready:
            return None

        trend        = self._density.trend()
        flow_rate    = self._density.flow_rate_per_minute()
        count_trend  = self._density.rolling_average(5) - self._density.rolling_average(20)

        feat_vec  = build_feature_vector(density, flow_rate, count_trend)
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
    ) -> np.ndarray:
        """Compose the annotated output frame."""
        lane_labels = {
            lane.name: density.density_label
            for lane in self._lanes
        }
        out = draw_lanes(frame, self._lanes, lane_labels)

        # Draw track boxes
        density_colours = {"Low": (0, 200, 100), "Medium": (0, 170, 255), "High": (0, 0, 220)}
        for track in tracks:
            if not track.confirmed:
                continue
            x1, y1, x2, y2 = map(int, track.bbox)
            colour = density_colours.get(density.density_label, (180, 180, 180))
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            label = f"#{track.track_id} {track.class_name}"
            cv2.putText(out, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

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
        }
        if schedule:
            m["signal_schedule"] = {
                o.lane_name: o.green_time_s for o in schedule.lanes
            }
        return m