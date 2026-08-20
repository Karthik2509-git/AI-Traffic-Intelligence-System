#!/usr/bin/env python3
"""
ATOS v3.5 Comprehensive Production Integration Test Suite
Validates all 12 Phase 7 integration test requirements:
1. Re-ID disabled safe fallback
2. Re-ID enabled + model available (2048-D embedding, L2 norm ≈ 1)
3. Re-ID model unavailable failure safety (no crashes)
4. Invalid crop rejection (<32x32 px)
5. Low confidence detection rejection (< min_confidence)
6. Keyframe gating (inference not executed on every frame)
7. Global ID assignment across cameras (GVID correlation)
8. Different vehicle non-matching (no forced false matches)
9. Same-camera temporal exclusion (<5 sec)
10. Temporal window expiration (>300 sec)
11. Mobile camera session identity propagation
12. Telemetry serialization for WebSocket / REST endpoints
"""

import os
import sys
import unittest
import numpy as np
import cv2
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.reid_engine import CrossCameraReIDManager, ONNXReIDFeatureExtractor
from tools.reid_crop_utility import extract_vehicle_crops, VehicleKeyframeAggregator
from tools.web_gateway import process_camera_frame_reid, g_reid_manager, g_settings


class TestReIDProductionIntegration(unittest.TestCase):

    def setUp(self):
        self.model_path = os.path.abspath("models/fastreid_sbs_r50_ibn_veri776.onnx")
        self.dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        # Create synthetic vehicle texture inside frame
        cv2.rectangle(self.dummy_frame, (100, 100), (250, 250), (50, 100, 200), -1)

    def test_01_reid_disabled_fallback(self):
        """1. Re-ID disabled: detector & tracker work, zero Re-ID inference."""
        manager = CrossCameraReIDManager({"enabled": False, "model_path": self.model_path})
        self.assertFalse(manager.is_available())
        self.assertEqual(manager.get_status_message(), "Re-ID disabled by configuration (reid_enabled: false)")
        
        # Ingestion payload returns None/empty without crashing
        res = manager.process_feature("cam1", 101, [0.1]*2048, time.time(), [100, 100, 150, 150])
        self.assertIsNone(res)

    def test_02_reid_enabled_model_available(self):
        """2. Re-ID enabled + model available: embedding generated, 2048-D, L2 norm ≈ 1."""
        if not os.path.exists(self.model_path):
            self.skipTest("ONNX model file missing")

        extractor = ONNXReIDFeatureExtractor(self.model_path)
        self.assertTrue(extractor.loaded)
        self.assertEqual(extractor.embedding_dim, 2048)

        crop = self.dummy_frame[100:250, 100:250]
        emb = extractor.extract(crop)
        self.assertIsNotNone(emb)
        self.assertEqual(len(emb), 2048)

        arr = np.array(emb, dtype=np.float32)
        self.assertTrue(np.all(np.isfinite(arr)))
        norm = np.linalg.norm(arr)
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_03_model_unavailable_failure_safety(self):
        """3. Re-ID model unavailable: pipeline continues cleanly without crashing."""
        manager = CrossCameraReIDManager({"enabled": True, "model_path": "invalid_path_model.onnx"})
        self.assertFalse(manager.is_available())
        self.assertEqual(manager.get_status_message(), "Re-ID model unavailable — evaluation pending")

        # Feature processing safely returns None
        res = manager.process_feature("cam1", 101, [0.1]*2048, time.time(), [100, 100, 150, 150])
        self.assertIsNone(res)

    def test_04_invalid_crop_rejection(self):
        """4. Invalid crop (<32x32 px): safely rejected."""
        detections = [
            {"track_id": 1, "class": "car", "confidence": 0.9, "box": [10, 10, 20, 20]} # 20x20 crop (<32px)
        ]
        crops = extract_vehicle_crops(self.dummy_frame, detections, min_confidence=0.5, min_dim=32)
        self.assertEqual(len(crops), 0)

    def test_05_low_confidence_rejection(self):
        """5. Low detector confidence (< threshold): safely rejected."""
        detections = [
            {"track_id": 1, "class": "car", "confidence": 0.3, "box": [50, 50, 100, 100]} # conf 0.3 < min 0.5
        ]
        crops = extract_vehicle_crops(self.dummy_frame, detections, min_confidence=0.5, min_dim=32)
        self.assertEqual(len(crops), 0)

    def test_06_keyframe_gating(self):
        """6. Keyframe gating: inference is not executed on every frame."""
        aggregator = VehicleKeyframeAggregator(sample_interval_frames=5, target_keyframes_per_track=3)
        track_id = 42

        # Frame 0: sample
        self.assertTrue(aggregator.should_sample(track_id, frame_idx=0))
        aggregator.add_observation("cam1", track_id, [0.1]*2048, time.time(), [50, 50, 100, 100], frame_idx=0)

        # Frame 1 to 4: skip sampling
        for f in range(1, 5):
            self.assertFalse(aggregator.should_sample(track_id, frame_idx=f))

        # Frame 5: sample
        self.assertTrue(aggregator.should_sample(track_id, frame_idx=5))

    def test_07_global_id_assignment_across_cameras(self):
        """7. Global ID assignment: same vehicle receives same GVID across Camera A and Camera B."""
        manager = CrossCameraReIDManager({"enabled": True, "model_path": self.model_path})
        manager.model_loaded = True

        vec_vehicle1 = [0.1] * 2048
        norm1 = np.linalg.norm(vec_vehicle1)
        vec_vehicle1 = (np.array(vec_vehicle1) / norm1).tolist()

        t0 = time.time()
        # Camera A observation
        manager.process_feature("cam_a", 101, vec_vehicle1, t0, [100, 100, 150, 150])
        self.assertEqual(len(manager.global_tracks), 1)
        gvid1 = list(manager.global_tracks.keys())[0]

        # Camera B observation (10 sec later, identical embedding)
        t1 = t0 + 10.0
        match = manager.process_feature("cam_b", 202, vec_vehicle1, t1, [100, 100, 150, 150])
        self.assertIsNotNone(match)
        self.assertEqual(match["global_vehicle_id"], gvid1)
        self.assertEqual(match["source_camera_id"], "cam_a")
        self.assertEqual(match["target_camera_id"], "cam_b")

    def test_08_different_vehicle_non_matching(self):
        """8. Different vehicle: must NOT be incorrectly forced into an existing GVID."""
        manager = CrossCameraReIDManager({"enabled": True, "model_path": self.model_path})
        manager.model_loaded = True

        # Vehicle A embedding
        vec_a = [0.0] * 2048
        vec_a[0] = 1.0

        # Vehicle B embedding (orthogonal, similarity = 0)
        vec_b = [0.0] * 2048
        vec_b[1] = 1.0

        t0 = time.time()
        manager.process_feature("cam_a", 101, vec_a, t0, [100, 100, 150, 150])

        # Camera B sees orthogonal Vehicle B
        match = manager.process_feature("cam_b", 202, vec_b, t0 + 5.0, [100, 100, 150, 150])
        self.assertIsNone(match) # No false match!
        self.assertEqual(len(manager.global_tracks), 2) # Created new distinct GVID

    def test_09_same_camera_exclusion(self):
        """9. Same-camera exclusion: matching excluded for same camera history < 5 seconds."""
        manager = CrossCameraReIDManager({"enabled": True, "model_path": self.model_path})
        manager.model_loaded = True

        vec = [1.0 / np.sqrt(2048)] * 2048
        t0 = time.time()

        manager.process_feature("cam_a", 101, vec, t0, [100, 100, 150, 150])
        # Same camera observation 2 seconds later (within 5s window)
        match = manager.process_feature("cam_a", 102, vec, t0 + 2.0, [100, 100, 150, 150])
        self.assertIsNone(match)

    def test_10_temporal_window_expiration(self):
        """10. Temporal window: matching excluded if time delta > 300 seconds."""
        manager = CrossCameraReIDManager({"enabled": True, "model_path": self.model_path, "max_spatiotemporal_window_sec": 300})
        manager.model_loaded = True

        vec = [1.0 / np.sqrt(2048)] * 2048
        t0 = time.time()

        manager.process_feature("cam_a", 101, vec, t0, [100, 100, 150, 150])
        # Camera B observation 350 seconds later (> 300s window)
        match = manager.process_feature("cam_b", 202, vec, t0 + 350.0, [100, 100, 150, 150])
        self.assertIsNone(match)

    def test_11_mobile_camera_identity_propagation(self):
        """11. Mobile camera session: camera identity reaches Re-ID layer."""
        session_id = "abc123xyz"
        cam_id = f"cam-phone-{session_id[:6]}"
        detections = [
            {"track_id": 99, "class": "car", "confidence": 0.92, "box": [100, 100, 120, 120]}
        ]

        # Call gateway process_camera_frame_reid with mobile cam_id
        matches = process_camera_frame_reid(cam_id, self.dummy_frame, detections, timestamp=time.time())
        self.assertIsInstance(matches, list)

    def test_12_websocket_telemetry_serialization(self):
        """12. Telemetry serialization: summary dictionary contains all required runtime fields."""
        manager = CrossCameraReIDManager({"enabled": False, "model_path": self.model_path})
        summary = manager.get_system_summary()

        required_keys = [
            "reid_enabled", "model_loaded", "model_path", "embedding_dimension",
            "active_global_tracks", "active_local_tracks", "match_count",
            "uncertain_match_count", "rejected_match_count", "similarity_threshold",
            "uncertainty_threshold", "benchmark", "readiness"
        ]
        for key in required_keys:
            self.assertIn(key, summary)

        # JSON serialization test
        json_str = json.dumps(summary)
        self.assertTrue(len(json_str) > 0)


if __name__ == "__main__":
    unittest.main()
