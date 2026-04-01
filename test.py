"""
test_detection.py — Unit tests for the AI Traffic Intelligence System.

Run:
    pytest test.py -v --tb=short
    pytest test.py -v --cov=src --cov-report=term-missing
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# detection.py
# ─────────────────────────────────────────────────────────────────────────────

class TestDetection:
    def test_classify_density_low(self):
        from src.detection import classify_density
        assert classify_density(0)  == "Low"
        assert classify_density(9)  == "Low"

    def test_classify_density_medium(self):
        from src.detection import classify_density
        assert classify_density(10) == "Medium"
        assert classify_density(25) == "Medium"

    def test_classify_density_high(self):
        from src.detection import classify_density
        assert classify_density(26) == "High"
        assert classify_density(100) == "High"

    def test_resolve_names_dict(self):
        from src.detection import _resolve_names
        pred = MagicMock()
        pred.names = {0: "car", 1: "truck"}
        names = _resolve_names(pred, MagicMock())
        assert names[0] == "car"
        assert names[1] == "truck"

    def test_resolve_names_list(self):
        from src.detection import _resolve_names
        pred = MagicMock()
        pred.names = ["car", "bus"]
        names = _resolve_names(pred, MagicMock())
        assert names[0] == "car"
        assert names[1] == "bus"

    def test_run_detection_file_not_found(self, tmp_path):
        from src.detection import run_detection
        with pytest.raises(FileNotFoundError):
            run_detection(
                model=MagicMock(),
                input_path=tmp_path / "nonexistent.jpg",
                output_path=tmp_path / "out.jpg",
            )

    def test_run_detection_returns_required_keys(self, tmp_path):
        """run_detection returns a dict with expected keys on a blank image."""
        import cv2
        from src.detection import run_detection, VEHICLE_CLASSES

        # Create a tiny blank test image
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        img_path = tmp_path / "test.jpg"
        cv2.imwrite(str(img_path), img)
        out_path = tmp_path / "out.jpg"

        mock_model = _make_mock_model(num_detections=0)

        result = run_detection(
            model=mock_model,
            input_path=img_path,
            output_path=out_path,
        )

        required = {"timestamp", "total_vehicles", "density",
                    "counts_per_class", "mean_confidence", "processing_time_ms"}
        assert required.issubset(result.keys())
        assert result["total_vehicles"] == 0
        assert result["density"] == "Low"
        assert result["mean_confidence"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# tracker.py
# ─────────────────────────────────────────────────────────────────────────────

class TestKalmanBoxTracker:
    def test_predict_returns_array(self):
        from src.tracker import KalmanBoxTracker
        KalmanBoxTracker.count = 0
        trk = KalmanBoxTracker(np.array([0.0, 0.0, 100.0, 100.0]))
        pred = trk.predict()
        assert pred.shape == (4,)

    def test_update_resets_time_since_update(self):
        from src.tracker import KalmanBoxTracker
        KalmanBoxTracker.count = 0
        trk = KalmanBoxTracker(np.array([0.0, 0.0, 100.0, 100.0]))
        trk.predict()
        assert trk.time_since_update == 1
        trk.update(np.array([5.0, 5.0, 105.0, 105.0]))
        assert trk.time_since_update == 0


class TestSORTTracker:
    def setup_method(self):
        from src.tracker import SORTTracker, KalmanBoxTracker
        KalmanBoxTracker.count = 0
        self.tracker = SORTTracker(max_age=3, min_hits=1, iou_threshold=0.30)

    def test_empty_detections(self):
        tracks = self.tracker.update([])
        assert tracks == []

    def test_new_track_spawned(self):
        dets = [{"bbox": [10, 20, 110, 120], "class_name": "car", "confidence": 0.85}]
        tracks = self.tracker.update(dets)
        assert len(tracks) == 1
        assert tracks[0].class_name == "car"

    def test_track_persists_across_frames(self):
        dets = [{"bbox": [10, 20, 110, 120], "class_name": "car", "confidence": 0.85}]
        for _ in range(4):
            tracks = self.tracker.update(dets)
        assert any(t.class_name == "car" for t in tracks)

    def test_track_dies_after_max_age(self):
        dets = [{"bbox": [10, 20, 110, 120], "class_name": "car", "confidence": 0.85}]
        self.tracker.update(dets)
        # Stop providing detection
        for _ in range(10):
            tracks = self.tracker.update([])
        assert len(tracks) == 0

    def test_reset_clears_tracks(self):
        dets = [{"bbox": [0, 0, 50, 50], "class_name": "truck", "confidence": 0.9}]
        self.tracker.update(dets)
        self.tracker.reset()
        assert self.tracker._trackers == []


class TestIoU:
    def test_identical_boxes(self):
        from src.tracker import _box_iou
        box = np.array([0.0, 0.0, 100.0, 100.0])
        assert abs(_box_iou(box, box) - 1.0) < 1e-6

    def test_non_overlapping(self):
        from src.tracker import _box_iou
        a = np.array([0.0, 0.0, 10.0, 10.0])
        b = np.array([20.0, 20.0, 30.0, 30.0])
        assert _box_iou(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_half_overlap(self):
        from src.tracker import _box_iou
        a = np.array([0.0, 0.0, 20.0, 10.0])
        b = np.array([10.0, 0.0, 30.0, 10.0])
        # Intersection: 10x10=100, Union: 300, IoU = 1/3
        assert _box_iou(a, b) == pytest.approx(1 / 3, rel=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# density_analyzer.py
# ─────────────────────────────────────────────────────────────────────────────

class TestLane:
    def test_contains_centre_inside(self):
        from src.density_analyzer import Lane
        lane = Lane("L1", np.array([[0, 0], [100, 0], [100, 100], [0, 100]]))
        assert lane.contains_centre(50, 50)

    def test_contains_centre_outside(self):
        from src.density_analyzer import Lane
        lane = Lane("L1", np.array([[0, 0], [100, 0], [100, 100], [0, 100]]))
        assert not lane.contains_centre(200, 200)

    def test_area_square(self):
        from src.density_analyzer import Lane
        lane = Lane("L1", np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float))
        assert lane.area() == pytest.approx(100.0, rel=0.01)


class TestDensityAnalyzer:
    def _make_track(self, cx=50.0, cy=50.0):
        from src.tracker import Track
        return Track(
            track_id=1, bbox=np.array([cx-10, cy-10, cx+10, cy+10]),
            class_name="car", confidence=0.9, hit_streak=3, age=5,
        )

    def test_update_returns_frame_density(self):
        from src.density_analyzer import DensityAnalyzer, make_full_frame_lane
        analyzer = DensityAnalyzer([make_full_frame_lane(640, 480)])
        track    = self._make_track()
        fd = analyzer.update([track], timestamp_ms=0.0)
        assert fd.total_count == 1
        assert fd.density_label in {"Low", "Medium", "High"}

    def test_ema_smoothing(self):
        from src.density_analyzer import DensityAnalyzer, make_full_frame_lane
        analyzer = DensityAnalyzer([make_full_frame_lane(640, 480)], ema_alpha=0.5)
        for _ in range(5):
            analyzer.update([], timestamp_ms=0.0)
        fd = analyzer.update([], timestamp_ms=0.0)
        assert fd.ema_count == pytest.approx(0.0, abs=0.01)

    def test_congestion_score_zero_with_no_vehicles(self):
        from src.density_analyzer import DensityAnalyzer, make_full_frame_lane
        analyzer = DensityAnalyzer([make_full_frame_lane(640, 480)])
        fd = analyzer.update([], timestamp_ms=0.0)
        assert fd.congestion_score == pytest.approx(0.0, abs=1e-6)

    def test_trend_returns_valid_string(self):
        from src.density_analyzer import DensityAnalyzer, make_full_frame_lane
        analyzer = DensityAnalyzer([make_full_frame_lane(640, 480)])
        for _ in range(12):
            analyzer.update([], timestamp_ms=0.0)
        assert analyzer.trend() in {"rising", "stable", "falling"}


# ─────────────────────────────────────────────────────────────────────────────
# predictor.py
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictor:
    def _make_df(self, n: int = 120):
        import pandas as pd
        rng = np.random.default_rng(42)
        counts = rng.integers(0, 35, size=n)
        labels = np.where(counts < 10, 0, np.where(counts <= 25, 1, 2))
        return pd.DataFrame({
            "ema_count":         counts.astype(float),
            "occupancy_ratio":   rng.uniform(0, 0.5, n),
            "congestion_score":  counts.astype(float) * 2.5,
            "flow_rate_per_min": rng.uniform(0, 30, n),
            "count_trend":       rng.uniform(-5, 5, n),
            "speed_variance":    rng.uniform(0, 100, n),
            "time_of_day_sin":   np.sin(rng.uniform(0, 2 * math.pi, n)),
            "time_of_day_cos":   np.cos(rng.uniform(0, 2 * math.pi, n)),
            "label":             labels,
        })

    def test_train_and_predict(self):
        from src.predictor import CongestionPredictor
        pred = CongestionPredictor(n_estimators=50)
        df   = self._make_df(120)
        result = pred.train(df, min_samples=50)
        assert "cv_accuracy" in result
        assert result["cv_accuracy"] > 0.0

        fv = df[["ema_count", "occupancy_ratio", "congestion_score",
                  "flow_rate_per_min", "count_trend", "speed_variance",
                  "time_of_day_sin", "time_of_day_cos"]].values[[0]]
        pr = pred.predict(fv)
        assert pr.label in {"Low", "Medium", "High"}
        assert 0.0 <= pr.confidence <= 1.0

    def test_predict_without_training_raises(self):
        from src.predictor import CongestionPredictor
        from sklearn.exceptions import NotFittedError
        pred = CongestionPredictor()
        with pytest.raises(NotFittedError):
            pred.predict(np.zeros((1, 8)))

    def test_insufficient_data_raises(self):
        from src.predictor import CongestionPredictor
        import pandas as pd
        pred = CongestionPredictor()
        tiny_df = pd.DataFrame({
            col: [0.0] * 5
            for col in ["ema_count", "occupancy_ratio", "congestion_score",
                        "flow_rate_per_min", "count_trend", "speed_variance",
                        "time_of_day_sin", "time_of_day_cos", "label"]
        })
        with pytest.raises(ValueError, match="Insufficient"):
            pred.train(tiny_df, min_samples=50)

    def test_save_and_load(self, tmp_path):
        from src.predictor import CongestionPredictor
        pred = CongestionPredictor(n_estimators=10)
        pred.train(self._make_df(100), min_samples=50)
        save_path = tmp_path / "model.joblib"
        pred.save(save_path)
        assert save_path.is_file()

        loaded = CongestionPredictor()
        loaded.load(save_path)
        assert loaded._is_trained


# ─────────────────────────────────────────────────────────────────────────────
# signal_optimizer.py
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalOptimizer:
    def _make_input(self, density_label="Medium", trend="stable"):
        from src.density_analyzer import FrameDensity
        from src.predictor import PredictionResult
        from src.signal_optimizer import LaneSignalInput

        fd = FrameDensity(
            frame_idx=0, timestamp_ms=0,
            counts_per_lane={"Lane 1": 15},
            total_count=15, density_label=density_label,
            occupancy_ratio=0.3, congestion_score=50.0, ema_count=15.0,
        )
        pred = PredictionResult(
            label=density_label, label_index=1,
            probabilities={"Low": 0.1, "Medium": 0.7, "High": 0.2},
            confidence=0.7,
        )
        return LaneSignalInput(
            lane_name="Lane 1", density=fd, prediction=pred, trend=trend,
        )

    def test_single_lane_sums_to_cycle(self):
        from src.signal_optimizer import SignalOptimizer
        opt = SignalOptimizer(cycle_time_s=60, min_green_s=10, max_green_s=50)
        schedule = opt.optimise([self._make_input()])
        assert schedule.cycle_time_s == 60

    def test_two_lanes_sums_to_cycle(self):
        from src.signal_optimizer import SignalOptimizer
        opt = SignalOptimizer(cycle_time_s=120, min_green_s=10, max_green_s=90)
        inputs = [
            self._make_input("High",  "rising"),
            self._make_input("Low",   "falling"),
        ]
        inputs[1].lane_name = "Lane 2"
        schedule = opt.optimise(inputs)
        total = sum(o.green_time_s for o in schedule.lanes)
        assert total == 120

    def test_min_green_respected(self):
        from src.signal_optimizer import SignalOptimizer
        opt = SignalOptimizer(cycle_time_s=60, min_green_s=20, max_green_s=40)
        schedule = opt.optimise([self._make_input("Low")])
        for o in schedule.lanes:
            assert o.green_time_s >= 20

    def test_empty_inputs_raises(self):
        from src.signal_optimizer import SignalOptimizer
        opt = SignalOptimizer()
        with pytest.raises(ValueError):
            opt.optimise([])

    def test_infeasible_cycle_raises(self):
        from src.signal_optimizer import SignalOptimizer
        opt = SignalOptimizer(cycle_time_s=10, min_green_s=8, max_green_s=9)
        # 2 lanes × 8s = 16s > 10s cycle
        with pytest.raises(ValueError, match="cycle_time_s"):
            opt.optimise([self._make_input(), self._make_input()])


# ─────────────────────────────────────────────────────────────────────────────
# utils.py
# ─────────────────────────────────────────────────────────────────────────────

class TestUtils:
    def test_rolling_buffer(self):
        from src.utils import RollingBuffer
        buf = RollingBuffer(maxlen=5)
        for i in range(10):
            buf.push(float(i))
        # Only last 5 values retained
        assert buf.mean() == pytest.approx((5 + 6 + 7 + 8 + 9) / 5, rel=0.01)

    def test_fps_meter(self):
        from src.utils import FPSMeter
        import time
        meter = FPSMeter(window=10)
        for _ in range(5):
            meter.tick()
            time.sleep(0.01)
        fps = meter.get()
        assert fps > 0

    def test_get_project_root(self):
        from src.utils import get_project_root
        root = get_project_root()
        assert root.is_dir()

    def test_ensure_dir(self, tmp_path):
        from src.utils import ensure_dir
        new_dir = tmp_path / "a" / "b" / "c"
        result  = ensure_dir(new_dir)
        assert result.is_dir()



# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_model(num_detections: int = 0):
    """Create a YOLO-like mock that returns *num_detections* fake car boxes."""
    import cv2

    mock_model = MagicMock()
    mock_model.names = {0: "car", 1: "bus", 2: "truck", 3: "motorcycle"}

    prediction = MagicMock()
    prediction.names = mock_model.names

    if num_detections == 0:
        prediction.boxes = None
        prediction.plot.return_value = np.zeros((128, 128, 3), dtype=np.uint8)
    else:
        boxes = MagicMock()
        boxes.cls  = MagicMock()
        boxes.cls.cpu.return_value.numpy.return_value = np.zeros(num_detections, dtype=int)
        boxes.conf = MagicMock()
        boxes.conf.cpu.return_value.numpy.return_value = np.full(num_detections, 0.85)
        boxes.xyxy = MagicMock()
        boxes.xyxy.cpu.return_value.numpy.return_value = np.array(
            [[10, 10, 50, 50]] * num_detections, dtype=float
        )
        prediction.boxes = boxes
        prediction.plot.return_value = np.zeros((128, 128, 3), dtype=np.uint8)

    mock_model.return_value = [prediction]
    return mock_model


# ─────────────────────────────────────────────────────────────────────────────
# Config + pipeline_from_config
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigLoading:
    def test_load_config_returns_dict(self):
        from src.utils import load_config
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert "model" in cfg
        assert "detection" in cfg
        assert "density_thresholds" in cfg
        assert "signal" in cfg
        assert "prediction" in cfg

    def test_load_config_model_name(self):
        from src.utils import load_config
        cfg = load_config()
        assert cfg["model"]["name"] == "yolov8m.pt"

    def test_load_config_missing_file(self, tmp_path):
        from src.utils import load_config
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_pipeline_from_config_creates_pipeline(self):
        from src.pipeline import pipeline_from_config
        pipeline = pipeline_from_config(source=0)
        assert pipeline is not None
        assert pipeline.config is not None
        assert pipeline.config.model_name == "yolov8m.pt"
        assert pipeline.config.confidence_threshold == 0.40
        assert pipeline.config.cycle_time_s == 120

    def test_pipeline_from_config_reads_all_sections(self):
        from src.pipeline import pipeline_from_config
        pipeline = pipeline_from_config(source="dummy.mp4")
        cfg = pipeline.config
        # Detection
        assert cfg.confidence_threshold == 0.40
        assert cfg.iou_threshold == 0.45
        assert cfg.frame_skip == 1
        assert cfg.inference_size == 640
        # Tracking
        assert cfg.tracker_max_age == 5
        assert cfg.tracker_min_hits == 3
        assert cfg.tracker_iou == 0.30
        # Density
        assert cfg.ema_alpha == 0.20
        assert cfg.low_threshold == 10
        assert cfg.high_threshold == 25
        # Signal
        assert cfg.min_green_s == 10
        assert cfg.max_green_s == 90


# ─────────────────────────────────────────────────────────────────────────────
# Multi-camera
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiCamera:
    def test_manager_creation(self):
        from src.multi_camera import MultiCameraManager, CameraSource
        cameras = [
            CameraSource("Cam A", "video_a.mp4"),
            CameraSource("Cam B", "video_b.mp4"),
        ]
        mgr = MultiCameraManager(cameras)
        assert mgr.camera_names == ["Cam A", "Cam B"]

    def test_empty_cameras_raises(self):
        from src.multi_camera import MultiCameraManager
        with pytest.raises(ValueError):
            MultiCameraManager([])

    def test_system_summary_initial(self):
        from src.multi_camera import MultiCameraManager, CameraSource
        cameras = [CameraSource("Cam A", "v.mp4")]
        mgr = MultiCameraManager(cameras)
        summary = mgr.system_summary()
        assert summary.total_cameras == 1
        assert summary.total_vehicles == 0
        assert summary.system_status == "Normal"

    def test_get_snapshot_returns_default(self):
        from src.multi_camera import MultiCameraManager, CameraSource
        cameras = [CameraSource("Cam A", "v.mp4")]
        mgr = MultiCameraManager(cameras)
        snap = mgr.get_snapshot("Cam A")
        assert snap.camera_name == "Cam A"
        assert snap.total_vehicles == 0

    def test_comparative_table_returns_list(self):
        from src.multi_camera import MultiCameraManager, CameraSource
        cameras = [
            CameraSource("Cam A", "a.mp4"),
            CameraSource("Cam B", "b.mp4"),
        ]
        mgr = MultiCameraManager(cameras)
        table = mgr.comparative_table()
        assert len(table) == 2
        assert table[0]["Camera"] == "Cam A"

    def test_reset_clears_snapshot(self):
        from src.multi_camera import MultiCameraManager, CameraSource
        cameras = [CameraSource("Cam A", "v.mp4")]
        mgr = MultiCameraManager(cameras)
        mgr.reset("Cam A")
        snap = mgr.get_snapshot("Cam A")
        assert snap.total_vehicles == 0

    def test_unknown_camera_raises(self):
        from src.multi_camera import MultiCameraManager, CameraSource
        mgr = MultiCameraManager([CameraSource("Cam A", "v.mp4")])
        with pytest.raises(KeyError):
            mgr.get_pipeline("Unknown")


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic video generator
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticVideo:
    def test_generate_video_creates_file(self, tmp_path):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
        from generate_synthetic_video import generate_video

        out = tmp_path / "test_video.mp4"
        result = generate_video(
            output_path=str(out),
            width=320, height=240,
            fps=10, duration_s=1,
            num_lanes=2, spawn_rate=0.1,
        )
        assert out.is_file()
        assert out.stat().st_size > 0

    def test_generate_video_correct_frames(self, tmp_path):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
        from generate_synthetic_video import generate_video
        import cv2

        out = tmp_path / "test_count.mp4"
        generate_video(
            output_path=str(out),
            width=320, height=240,
            fps=10, duration_s=2,
            num_lanes=2,
        )
        cap = cv2.VideoCapture(str(out))
        count = 0
        while cap.read()[0]:
            count += 1
        cap.release()
        assert count == 20  # 10fps * 2s


# ─────────────────────────────────────────────────────────────────────────────
# Integration: full pipeline smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineIntegration:
    def test_pipeline_process_frame_with_mock(self):
        """Smoke test: process a blank frame through the pipeline with a mock model."""
        from src.pipeline import TrafficPipeline, PipelineConfig

        cfg = PipelineConfig(
            model_name="yolov8m.pt",
            save_annotated=False,
            display=False,
        )

        pipeline = TrafficPipeline(source=0, config=cfg)

        # Manually inject a mock model to avoid downloading YOLO weights
        mock_model = _make_mock_model(num_detections=3)
        pipeline._model = mock_model

        # Manually trigger setup with known frame size
        from src.density_analyzer import make_full_frame_lane
        from src.tracker import SORTTracker, KalmanBoxTracker
        from src.density_analyzer import DensityAnalyzer
        from src.signal_optimizer import SignalOptimizer

        KalmanBoxTracker.count = 0
        pipeline._lanes = [make_full_frame_lane(640, 480)]
        pipeline._tracker = SORTTracker()
        pipeline._density = DensityAnalyzer(pipeline._lanes)
        pipeline._optimizer = SignalOptimizer()

        # Process a blank frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame)

        assert result is not None
        assert result.frame_idx == 0
        assert isinstance(result.tracks, list)
        assert result.density is not None
        assert result.annotated_frame is not None
        assert "total_vehicles" in result.metrics


# ─────────────────────────────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────────────────────────────

class TestDatabase:
    def test_create_database(self, tmp_path):
        from src.database import TrafficDatabase
        db = TrafficDatabase(tmp_path / "test.db")
        assert (tmp_path / "test.db").is_file()

    def test_start_session(self, tmp_path):
        from src.database import TrafficDatabase
        db = TrafficDatabase(tmp_path / "test.db")
        sid = db.start_session(source="video.mp4")
        assert sid is not None
        assert len(sid) > 10

    def test_log_and_get_frame(self, tmp_path):
        from src.database import TrafficDatabase
        db = TrafficDatabase(tmp_path / "test.db")
        sid = db.start_session()
        db.log_frame(sid, frame_idx=0, metrics={
            "total_vehicles": 5,
            "density_label": "Low",
            "congestion_score": 15.0,
            "ema_count": 4.8,
            "counts_per_class": {"car": 3, "truck": 2},
        })
        frames = db.get_frame_metrics(sid)
        assert len(frames) == 1
        assert frames[0]["total_vehicles"] == 5
        assert frames[0]["cars"] == 3
        assert frames[0]["trucks"] == 2

    def test_batch_insert(self, tmp_path):
        from src.database import TrafficDatabase
        db = TrafficDatabase(tmp_path / "test.db")
        sid = db.start_session()
        batch = [
            (i, {"total_vehicles": i * 2, "density_label": "Low", "counts_per_class": {}})
            for i in range(50)
        ]
        db.log_frames_batch(sid, batch)
        frames = db.get_frame_metrics(sid)
        assert len(frames) == 50

    def test_session_summary(self, tmp_path):
        from src.database import TrafficDatabase
        db = TrafficDatabase(tmp_path / "test.db")
        sid = db.start_session()
        for i in range(10):
            db.log_frame(sid, i, {
                "total_vehicles": i + 1,
                "congestion_score": (i + 1) * 5.0,
                "counts_per_class": {},
            })
        summary = db.get_session_summary(sid)
        assert summary["total_frames"] == 10
        assert summary["peak_vehicles"] == 10
        assert summary["avg_vehicles"] == 5.5

    def test_log_signal(self, tmp_path):
        from src.database import TrafficDatabase
        db = TrafficDatabase(tmp_path / "test.db")
        sid = db.start_session()
        db.log_signal(sid, 0, "Lane 1", 60, 60, 0.5, "Normal")
        logs = db.get_signal_logs(sid)
        assert len(logs) == 1
        assert logs[0]["green_time_s"] == 60

    def test_log_anomaly(self, tmp_path):
        from src.database import TrafficDatabase
        db = TrafficDatabase(tmp_path / "test.db")
        sid = db.start_session()
        db.log_anomaly(sid, "evt-001", 42, "spike", "warning", "Test spike", 0.8)
        anomalies = db.get_anomalies(sid)
        assert len(anomalies) == 1
        assert anomalies[0]["severity"] == "warning"

    def test_export_csv(self, tmp_path):
        from src.database import TrafficDatabase
        db = TrafficDatabase(tmp_path / "test.db")
        sid = db.start_session()
        db.log_frame(sid, 0, {"total_vehicles": 5, "counts_per_class": {}})
        csv_path = db.export_csv(sid, tmp_path / "export.csv")
        assert csv_path.is_file()
        content = csv_path.read_text()
        assert "total_vehicles" in content

    def test_export_json(self, tmp_path):
        import json
        from src.database import TrafficDatabase
        db = TrafficDatabase(tmp_path / "test.db")
        sid = db.start_session()
        db.log_frame(sid, 0, {"total_vehicles": 3, "counts_per_class": {}})
        json_path = db.export_json(sid, tmp_path / "export.json")
        assert json_path.is_file()
        data = json.loads(json_path.read_text())
        assert data["session_id"] == sid
        assert len(data["frame_metrics"]) == 1

    def test_list_sessions(self, tmp_path):
        from src.database import TrafficDatabase
        db = TrafficDatabase(tmp_path / "test.db")
        db.start_session(source="a.mp4")
        db.start_session(source="b.mp4")
        sessions = db.list_sessions()
        assert len(sessions) == 2

    def test_delete_session(self, tmp_path):
        from src.database import TrafficDatabase
        db = TrafficDatabase(tmp_path / "test.db")
        sid = db.start_session()
        db.log_frame(sid, 0, {"total_vehicles": 1, "counts_per_class": {}})
        db.delete_session(sid)
        assert len(db.get_frame_metrics(sid)) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap
# ─────────────────────────────────────────────────────────────────────────────

class TestHeatmap:
    def _make_track(self, track_id=1, x1=80, y1=80, x2=120, y2=120):
        from src.tracker import Track
        return Track(track_id=track_id, bbox=np.array([x1, y1, x2, y2], dtype=float),
                     class_name="car", confidence=0.9, hit_streak=3, age=5)

    def test_init(self):
        from src.heatmap import HeatmapGenerator
        hm = HeatmapGenerator(frame_shape=(480, 640))
        assert hm.width == 640
        assert hm.height == 480
        assert hm.frame_count == 0

    def test_update_increments_count(self):
        from src.heatmap import HeatmapGenerator
        hm = HeatmapGenerator(frame_shape=(480, 640))
        tracks = [self._make_track(1, 300, 220, 340, 260)]
        hm.update(tracks)
        assert hm.frame_count == 1
        assert hm.total_points == 1

    def test_render_shape(self):
        from src.heatmap import HeatmapGenerator
        hm = HeatmapGenerator(frame_shape=(480, 640))
        rendered = hm.render()
        assert rendered.shape == (480, 640, 3)

    def test_overlay_shape(self):
        from src.heatmap import HeatmapGenerator
        hm = HeatmapGenerator(frame_shape=(480, 640))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracks = [self._make_track(1, 80, 80, 120, 120)]
        hm.update(tracks)
        overlay = hm.overlay(frame)
        assert overlay.shape == frame.shape

    def test_export(self, tmp_path):
        from src.heatmap import HeatmapGenerator
        hm = HeatmapGenerator(frame_shape=(240, 320))
        tracks = [self._make_track(1, 140, 100, 180, 140)]
        hm.update(tracks)
        out = hm.export(tmp_path / "hm.jpg")
        assert out.is_file()

    def test_reset(self):
        from src.heatmap import HeatmapGenerator
        hm = HeatmapGenerator(frame_shape=(240, 320))
        hm.update([self._make_track(1, 140, 100, 180, 140)])
        assert hm.frame_count == 1
        hm.reset()
        assert hm.frame_count == 0
        assert hm.total_points == 0

    def test_temporal_decay(self):
        from src.heatmap import HeatmapGenerator, HeatmapConfig
        cfg = HeatmapConfig(decay_factor=0.5)
        hm = HeatmapGenerator(frame_shape=(240, 320), config=cfg)
        hm.update([self._make_track(1, 140, 100, 180, 140)])
        peak_after_1 = hm.peak_intensity
        # Update with no tracks — should decay
        hm.update([])
        peak_after_2 = hm.peak_intensity
        assert peak_after_2 < peak_after_1


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestAnomalyDetector:
    def test_init(self):
        from src.anomaly_detector import AnomalyDetector
        det = AnomalyDetector()
        assert det.total_events == 0

    def test_no_anomaly_normal_traffic(self):
        from src.anomaly_detector import AnomalyDetector
        det = AnomalyDetector()
        for i in range(50):
            events = det.analyse(i, {
                "total_vehicles": 10,
                "congestion_score": 30.0,
                "flow_per_min": 20.0,
                "trend": "stable",
            })
        assert det.total_events == 0

    def test_spike_detected(self):
        from src.anomaly_detector import AnomalyDetector, AnomalyConfig
        cfg = AnomalyConfig(zscore_threshold=2.0, zscore_window=20, cooldown_frames=5)
        det = AnomalyDetector(config=cfg)
        # Build stable baseline
        for i in range(30):
            det.analyse(i, {"total_vehicles": 10, "congestion_score": 20.0,
                            "flow_per_min": 15.0, "trend": "stable"})
        # Inject spike
        events = det.analyse(30, {"total_vehicles": 50, "congestion_score": 90.0,
                                   "flow_per_min": 15.0, "trend": "rising"})
        assert len(events) > 0
        assert any(e.anomaly_type == "spike" for e in events)

    def test_sudden_drop_detected(self):
        from src.anomaly_detector import AnomalyDetector, AnomalyConfig
        cfg = AnomalyConfig(drop_pct_threshold=0.40, drop_window=5, cooldown_frames=5)
        det = AnomalyDetector(config=cfg)
        # Build baseline with high flow
        for i in range(20):
            det.analyse(i, {"total_vehicles": 20, "flow_per_min": 30.0,
                            "congestion_score": 40.0, "trend": "stable"})
        # Sudden drop in flow
        for i in range(20, 30):
            events = det.analyse(i, {"total_vehicles": 20, "flow_per_min": 5.0,
                                      "congestion_score": 40.0, "trend": "stable"})
        # Should have detected a drop
        assert det.total_events > 0

    def test_congestion_surge(self):
        from src.anomaly_detector import AnomalyDetector, AnomalyConfig
        cfg = AnomalyConfig(congestion_critical=70.0, congestion_persist=3, cooldown_frames=5)
        det = AnomalyDetector(config=cfg)
        for i in range(10):
            det.analyse(i, {"total_vehicles": 30, "congestion_score": 85.0,
                            "flow_per_min": 10.0, "trend": "rising"})
        assert det.total_events > 0

    def test_cooldown_prevents_spam(self):
        from src.anomaly_detector import AnomalyDetector, AnomalyConfig
        cfg = AnomalyConfig(zscore_threshold=2.0, zscore_window=20, cooldown_frames=50)
        det = AnomalyDetector(config=cfg)
        for i in range(30):
            det.analyse(i, {"total_vehicles": 10, "congestion_score": 20.0,
                            "flow_per_min": 15.0, "trend": "stable"})
        # First spike
        det.analyse(30, {"total_vehicles": 50, "congestion_score": 90.0,
                         "flow_per_min": 15.0, "trend": "rising"})
        count_after_first = det.total_events
        # Second spike immediately — should be blocked by cooldown
        events = det.analyse(31, {"total_vehicles": 50, "congestion_score": 90.0,
                                   "flow_per_min": 15.0, "trend": "rising"})
        spike_events = [e for e in events if e.anomaly_type == "spike"]
        assert len(spike_events) == 0

    def test_reset(self):
        from src.anomaly_detector import AnomalyDetector
        det = AnomalyDetector()
        for i in range(10):
            det.analyse(i, {"total_vehicles": 10, "congestion_score": 20.0,
                            "flow_per_min": 15.0, "trend": "stable"})
        det.reset()
        assert det.total_events == 0

    def test_event_to_dict(self):
        from src.anomaly_detector import AnomalyEvent
        from datetime import datetime
        event = AnomalyEvent(
            event_id="test-id", timestamp=datetime.now(), frame_idx=42,
            anomaly_type="spike", severity="warning",
            description="Test", confidence=0.8, metrics_snapshot={},
        )
        d = event.to_dict()
        assert d["event_id"] == "test-id"
        assert d["anomaly_type"] == "spike"


# ─────────────────────────────────────────────────────────────────────────────
# Speed Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class TestSpeedAnalyzer:
    def _make_track(self, track_id=1, x1=80, y1=80, x2=120, y2=120):
        from src.tracker import Track
        return Track(track_id=track_id, bbox=np.array([x1, y1, x2, y2], dtype=float),
                     class_name="car", confidence=0.9, hit_streak=3, age=5)

    def test_init(self):
        from src.speed_analyzer import SpeedAnalyzer
        sa = SpeedAnalyzer(fps=30.0)
        assert sa.total_measurements == 0

    def test_first_frame_no_speed(self):
        from src.speed_analyzer import SpeedAnalyzer
        sa = SpeedAnalyzer(fps=30.0)
        tracks = [self._make_track(1, 80, 80, 120, 120)]
        results = sa.update(tracks)
        assert len(results) == 0  # no speed on first sighting

    def test_speed_computed_on_nth_frame(self):
        """Should skip first 8 frames and report on the 9th frame (min_speed_frames=8)."""
        from src.speed_analyzer import SpeedAnalyzer
        sa = SpeedAnalyzer(fps=30.0)
        sa.update([self._make_track(1, 80, 80, 120, 120)])
        
        # Move car 10 pixels right per frame for 7 more frames (total 8 hits)
        for i in range(1, 7):
            sa.update([self._make_track(1, 80 + i*10, 80, 120 + i*10, 120)])
            
        # 8th hit (7th move) — still no speed because state["hits"] < min_speed_frames (8)
        res = sa.update([self._make_track(1, 160, 80, 200, 120)])
        assert len(res) == 0
        
        # 9th hit (8th move) — speed should finally be reported
        res = sa.update([self._make_track(1, 170, 80, 210, 120)])
        assert len(res) == 1
        assert res[0].speed_kmh > 0

    def test_speed_outlier_rejected(self):
        """Should ignore physically impossible speeds (e.g. 1000 km/h)."""
        from src.speed_analyzer import SpeedAnalyzer
        sa = SpeedAnalyzer(fps=30.0)
        sa.update([self._make_track(1, 80, 80, 120, 120)])
        
        # Massive jump (1000px displacement)
        res = sa.update([self._make_track(1, 1080, 80, 1120, 120)])
        assert len(res) == 0

    def test_stopped_vehicle(self):
        """Should report ~0 km/h for static track after stabilization."""
        from src.speed_analyzer import SpeedAnalyzer
        sa = SpeedAnalyzer(fps=30.0)
        sa.update([self._make_track(1, 80, 80, 120, 120)])
        
        # Stay still for 10 frames
        results = []
        for _ in range(10):
            results = sa.update([self._make_track(1, 80, 80, 120, 120)])
            
        assert len(results) == 1
        assert results[0].speed_kmh < 2.0
        assert results[0].speed_class == "stopped"

    def test_summary(self):
        from src.speed_analyzer import SpeedAnalyzer, VehicleSpeedInfo
        sa = SpeedAnalyzer(fps=30.0)
        info_list = [
            VehicleSpeedInfo(track_id=1, speed_kmh=40.0, direction_deg=90,
                             speed_class="normal", is_violation=False),
            VehicleSpeedInfo(track_id=2, speed_kmh=90.0, direction_deg=180,
                             speed_class="speeding", is_violation=True),
        ]
        summary = sa.get_summary(info_list)
        assert summary["avg_speed_kmh"] == 65.0
        assert summary["max_speed_kmh"] == 90.0
        assert summary["violations"] == 1

    def test_stale_tracks_pruned(self):
        from src.speed_analyzer import SpeedAnalyzer
        sa = SpeedAnalyzer(fps=30.0)
        sa.update([self._make_track(1, 80, 80, 120, 120)])
        # Track 1 disappears
        sa.update([self._make_track(2, 180, 180, 220, 220)])
        assert 1 not in sa._track_state

    def test_reset(self):
        from src.speed_analyzer import SpeedAnalyzer
        sa = SpeedAnalyzer(fps=30.0)
        sa.update([self._make_track(1, 80, 80, 120, 120)])
        sa.reset()
        assert sa.total_measurements == 0
        assert len(sa._track_state) == 0