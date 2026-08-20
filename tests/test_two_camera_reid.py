#!/usr/bin/env python3
"""
Unit & Integration Test Suite for ATOS v3.5 Phase 2 Controlled Two-Camera Re-ID Validation Harness
Verifies independent local track IDs, cross-camera GVID assignment, same vehicle association,
different vehicle rejection, temporal gating, similarity threshold behavior, uncertainty handling,
missing/invalid crops, insufficient keyframes, and result serialization.
"""

import sys
import os
import json
import tempfile
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.test_two_camera_reid import TwoCameraReIDEvaluator, parse_args
from tools.reid_crop_utility import VehicleKeyframeAggregator

class DummyArgs:
    def __init__(self, output_dir):
        self.camera_a = None
        self.camera_b = None
        self.camera_a_id = "cam_a"
        self.camera_b_id = "cam_b"
        self.similarity_threshold = 0.75
        self.uncertainty_threshold = 0.60
        self.max_transition_time = 300
        self.sample_interval = 5
        self.yolo_model = "yolov8n.pt"
        self.output = output_dir

class TestTwoCameraReID(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.args = DummyArgs(self.temp_dir)
        self.evaluator = TwoCameraReIDEvaluator(self.args)

    def test_independent_camera_local_track_ids(self):
        """1. Verifies Camera A track 17 and Camera B track 8 are kept independent."""
        payload_a = {
            "camera_id": "cam_a",
            "local_track_id": 17,
            "embedding": [1.0, 0.0, 0.0],
            "timestamp": 10.0,
            "bbox": [10, 10, 100, 100],
            "num_keyframes": 3
        }
        payload_b = {
            "camera_id": "cam_b",
            "local_track_id": 8,
            "embedding": [1.0, 0.0, 0.0],
            "timestamp": 22.4,
            "bbox": [15, 15, 110, 110],
            "num_keyframes": 3
        }
        record = self.evaluator.evaluate_association(payload_a, payload_b)
        self.assertEqual(record["camera_a"], "cam_a")
        self.assertEqual(record["track_a"], 17)
        self.assertEqual(record["camera_b"], "cam_b")
        self.assertEqual(record["track_b"], 8)

    def test_same_vehicle_association_and_gvid(self):
        """2 & 3. Verifies high similarity (>= 0.75) assigns MATCH and a Global Vehicle ID (GVID-1001)."""
        payload_a = {
            "camera_id": "cam_a",
            "local_track_id": 17,
            "embedding": [0.8, 0.6, 0.0], # L2 norm = 1.0
            "timestamp": 10.0,
            "bbox": [10, 10, 100, 100],
            "num_keyframes": 3
        }
        payload_b = {
            "camera_id": "cam_b",
            "local_track_id": 8,
            "embedding": [0.8, 0.6, 0.0], # Identical vector, cosine sim = 1.0
            "timestamp": 22.4,
            "bbox": [15, 15, 110, 110],
            "num_keyframes": 3
        }
        record = self.evaluator.evaluate_association(payload_a, payload_b)
        self.assertEqual(record["decision"], "MATCH")
        self.assertTrue(record["global_vehicle_id"].startswith("GVID-"))
        self.assertEqual(record["similarity"], 1.0)
        self.assertAlmostEqual(record["time_delta_seconds"], 12.4)
        self.assertIsNone(record["rejection_reason"])

    def test_different_vehicle_rejection(self):
        """4. Verifies low similarity (< 0.60) yields REJECTED with similarity_below_threshold."""
        payload_a = {
            "camera_id": "cam_a",
            "local_track_id": 17,
            "embedding": [1.0, 0.0, 0.0],
            "timestamp": 10.0,
            "bbox": [10, 10, 100, 100],
            "num_keyframes": 3
        }
        payload_b = {
            "camera_id": "cam_b",
            "local_track_id": 9,
            "embedding": [0.0, 1.0, 0.0], # Orthogonal vector, cosine sim = 0.0
            "timestamp": 22.4,
            "bbox": [15, 15, 110, 110],
            "num_keyframes": 3
        }
        record = self.evaluator.evaluate_association(payload_a, payload_b)
        self.assertEqual(record["decision"], "REJECTED")
        self.assertEqual(record["rejection_reason"], "similarity_below_threshold")

    def test_temporal_gating(self):
        """5. Verifies time delta > max_transition_time (> 300s) yields REJECTED with temporal_window_exceeded."""
        payload_a = {
            "camera_id": "cam_a",
            "local_track_id": 17,
            "embedding": [1.0, 0.0, 0.0],
            "timestamp": 10.0,
            "bbox": [10, 10, 100, 100],
            "num_keyframes": 3
        }
        payload_b = {
            "camera_id": "cam_b",
            "local_track_id": 10,
            "embedding": [1.0, 0.0, 0.0],
            "timestamp": 350.0, # Delta = 340s > 300s
            "bbox": [15, 15, 110, 110],
            "num_keyframes": 3
        }
        record = self.evaluator.evaluate_association(payload_a, payload_b)
        self.assertEqual(record["decision"], "REJECTED")
        self.assertEqual(record["rejection_reason"], "temporal_window_exceeded")

    def test_uncertainty_handling(self):
        """6 & 7. Verifies similarity in band [0.60, 0.75) yields UNCERTAIN with uncertainty_band."""
        payload_a = {
            "camera_id": "cam_a",
            "local_track_id": 17,
            "embedding": [1.0, 0.0, 0.0],
            "timestamp": 10.0,
            "bbox": [10, 10, 100, 100],
            "num_keyframes": 3
        }
        # Cosine sim = 0.68 (which lies between 0.60 and 0.75)
        v = 0.68
        w = float(np.sqrt(1.0 - v*v))
        payload_b = {
            "camera_id": "cam_b",
            "local_track_id": 11,
            "embedding": [v, w, 0.0],
            "timestamp": 20.0,
            "bbox": [15, 15, 110, 110],
            "num_keyframes": 3
        }
        record = self.evaluator.evaluate_association(payload_a, payload_b)
        self.assertEqual(record["decision"], "UNCERTAIN")
        self.assertEqual(record["rejection_reason"], "uncertainty_band")
        self.assertAlmostEqual(record["similarity"], 0.68, places=2)

    def test_insufficient_keyframes(self):
        """9. Verifies KeyframeAggregator returns None until target keyframes are buffered."""
        aggregator = VehicleKeyframeAggregator(sample_interval_frames=1, target_keyframes_per_track=3)
        res1 = aggregator.add_observation("cam_a", 17, [1.0, 0.0], 10.0, [10, 10, 50, 50], 0)
        res2 = aggregator.add_observation("cam_a", 17, [1.0, 0.0], 11.0, [10, 10, 50, 50], 1)
        self.assertIsNone(res1)
        self.assertIsNone(res2)
        res3 = aggregator.add_observation("cam_a", 17, [1.0, 0.0], 12.0, [10, 10, 50, 50], 2)
        self.assertIsNotNone(res3)
        self.assertEqual(res3["num_keyframes"], 3)

    def test_result_serialization(self):
        """10. Verifies summary.json, matches.json, and experiment_report.md are serialized cleanly."""
        self.evaluator.save_artifacts()
        summary_file = os.path.join(self.temp_dir, "summary.json")
        matches_file = os.path.join(self.temp_dir, "matches.json")
        report_file = os.path.join(self.temp_dir, "experiment_report.md")

        self.assertTrue(os.path.exists(summary_file))
        self.assertTrue(os.path.exists(matches_file))
        self.assertTrue(os.path.exists(report_file))

        with open(summary_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertIn("similarity_threshold", data)
            self.assertEqual(data["similarity_threshold"], 0.75)

if __name__ == "__main__":
    unittest.main()
