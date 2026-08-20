#!/usr/bin/env python3
"""
Unit Test Suite for ATOS v3.5 Vehicle Crop Extraction & Keyframe Feature Aggregation
Verifies valid crop extraction, bounding-box clipping, invalid/empty crop rejection,
minimum crop-size rejection, confidence gating, keyframe sampling, multi-keyframe embedding aggregation, and L2 normalization.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.reid_crop_utility import extract_vehicle_crops, VehicleKeyframeAggregator

class TestReIDCropUtility(unittest.TestCase):

    def setUp(self):
        # Create a synthetic 720p BGR image (1280x720) with color blocks
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.frame[100:300, 100:400] = [255, 100, 50] # Blue-ish vehicle crop region

    def test_valid_crop_extraction(self):
        """1. Verifies valid crop extraction for in-bounds vehicle detection."""
        detections = [{
            "track_id": 101,
            "class": "car",
            "confidence": 0.92,
            "box": [100, 100, 300, 200]
        }]
        crops = extract_vehicle_crops(self.frame, detections, min_confidence=0.5, min_dim=32)
        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0]["track_id"], 101)
        self.assertEqual(crops[0]["class"], "car")
        self.assertAlmostEqual(crops[0]["confidence"], 0.92)
        self.assertEqual(crops[0]["box"], [100, 100, 300, 200])
        self.assertEqual(crops[0]["crop"].shape, (200, 300, 3))

    def test_bounding_box_clipping(self):
        """2. Verifies out-of-bounds bounding boxes are safely clipped to image boundaries."""
        detections = [{
            "track_id": 102,
            "class": "truck",
            "confidence": 0.88,
            "box": [-50, -50, 400, 400] # Partially negative coordinates
        }]
        crops = extract_vehicle_crops(self.frame, detections, min_confidence=0.5, min_dim=32)
        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0]["box"][0], 0)
        self.assertEqual(crops[0]["box"][1], 0)
        self.assertEqual(crops[0]["box"][2], 350)
        self.assertEqual(crops[0]["box"][3], 350)
        self.assertEqual(crops[0]["crop"].shape, (350, 350, 3))

    def test_invalid_empty_crop_rejection(self):
        """3. Verifies None, empty frame, or empty detections are safely rejected."""
        self.assertEqual(extract_vehicle_crops(None, [{"track_id": 1, "box": [10, 10, 50, 50]}]), [])
        empty_frame = np.empty((0, 0, 3), dtype=np.uint8)
        self.assertEqual(extract_vehicle_crops(empty_frame, [{"track_id": 1, "box": [10, 10, 50, 50]}]), [])
        self.assertEqual(extract_vehicle_crops(self.frame, []), [])

    def test_minimum_crop_size_rejection(self):
        """4. Verifies crops smaller than min_dim (e.g. 20x20 < 32x32) are rejected."""
        detections = [{
            "track_id": 103,
            "class": "car",
            "confidence": 0.95,
            "box": [100, 100, 20, 20] # 20x20 is < min_dim 32
        }]
        crops = extract_vehicle_crops(self.frame, detections, min_confidence=0.5, min_dim=32)
        self.assertEqual(len(crops), 0)

    def test_confidence_gating(self):
        """5. Verifies detections below min_confidence threshold are rejected."""
        detections = [{
            "track_id": 104,
            "class": "car",
            "confidence": 0.35, # Below 0.5 cutoff
            "box": [100, 100, 100, 100]
        }]
        crops = extract_vehicle_crops(self.frame, detections, min_confidence=0.5, min_dim=32)
        self.assertEqual(len(crops), 0)

    def test_keyframe_sampling(self):
        """6. Verifies keyframe sampling interval gating per local track."""
        aggregator = VehicleKeyframeAggregator(sample_interval_frames=5, target_keyframes_per_track=3)

        # Track 101 at frame 0 should sample
        self.assertTrue(aggregator.should_sample(track_id=101, frame_idx=0))

        # Add observation at frame 0
        aggregator.add_observation(
            camera_id="cam_01", track_id=101, embedding=[1.0, 0.0],
            timestamp=100.0, bbox=[10, 10, 50, 50], frame_idx=0
        )

        # Frames 1-4 should be skipped
        self.assertFalse(aggregator.should_sample(track_id=101, frame_idx=1))
        self.assertFalse(aggregator.should_sample(track_id=101, frame_idx=2))
        self.assertFalse(aggregator.should_sample(track_id=101, frame_idx=4))

        # Frame 5 should sample
        self.assertTrue(aggregator.should_sample(track_id=101, frame_idx=5))

    def test_multi_keyframe_embedding_aggregation(self):
        """7. Verifies multi-observation embedding aggregation across 3 keyframes."""
        aggregator = VehicleKeyframeAggregator(sample_interval_frames=1, target_keyframes_per_track=3)

        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]
        emb3 = [0.0, 0.0, 1.0]

        res1 = aggregator.add_observation("cam_01", 201, emb1, 10.0, [10, 10, 50, 50], frame_idx=0)
        self.assertIsNone(res1) # Needs 3 keyframes

        res2 = aggregator.add_observation("cam_01", 201, emb2, 11.0, [10, 10, 50, 50], frame_idx=1)
        self.assertIsNone(res2) # Needs 3 keyframes

        res3 = aggregator.add_observation("cam_01", 201, emb3, 12.0, [10, 10, 50, 50], frame_idx=2)
        self.assertIsNotNone(res3)

        self.assertEqual(res3["num_keyframes"], 3)
        self.assertEqual(res3["local_track_id"], 201)
        self.assertEqual(res3["camera_id"], "cam_01")
        self.assertEqual(len(res3["embedding"]), 3)

    def test_l2_normalization(self):
        """8. Verifies aggregated vector is strictly L2-normalized (norm = 1.0)."""
        aggregator = VehicleKeyframeAggregator(sample_interval_frames=1, target_keyframes_per_track=2)

        emb1 = [3.0, 0.0, 4.0]
        emb2 = [1.0, 2.0, 2.0]

        aggregator.add_observation("cam_01", 301, emb1, 10.0, [10, 10, 50, 50], frame_idx=0)
        res = aggregator.add_observation("cam_01", 301, emb2, 11.0, [10, 10, 50, 50], frame_idx=1)

        aggregated_emb = res["embedding"]
        norm = np.linalg.norm(np.array(aggregated_emb, dtype=np.float32))
        self.assertAlmostEqual(float(norm), 1.0, places=5)

if __name__ == "__main__":
    unittest.main()
