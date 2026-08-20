#!/usr/bin/env python3
"""
Phase 2 C++ -> Python Vehicle Crop Transport & Re-ID Integration Unit Tests
Validates all 16 Phase 2 requirements:
1. Valid crop extraction from authoritative bbox
2. Bbox clipping
3. Invalid bbox rejection
4. <32x32 crop rejection
5. Low-confidence rejection
6. JPEG encode/decode
7. Metadata preservation
8. Track ID preservation
9. Corrupted image payload handling
10. Missing metadata handling
11. Transport failure safety
12. Re-ID failure does not crash detector/tracker
13. No synthetic track ID generation
14. Keyframe interval preservation
15. 2048-D embedding validation
16. L2 normalization validation (||v||₂ ≈ 1.0)
"""

import os
import sys
import json
import struct
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.web_gateway import validate_and_decode_crop_package, process_production_cpp_crop, g_reid_manager
from tools.reid_crop_utility import extract_vehicle_crops, VehicleKeyframeAggregator
from tools.reid_engine import ONNXReIDFeatureExtractor, CrossCameraReIDManager


class TestCropTransportPhase2(unittest.TestCase):

    def setUp(self):
        self.model_path = os.path.abspath("models/fastreid_sbs_r50_ibn_veri776.onnx")
        self.dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.rectangle(self.dummy_frame, (100, 100), (250, 250), (50, 100, 200), -1)

        # Encode synthetic crop as JPEG
        _, self.dummy_jpeg = cv2.imencode(".jpg", self.dummy_frame[100:250, 100:250])
        self.dummy_jpeg_bytes = self.dummy_jpeg.tobytes()

    def test_01_valid_crop_extraction(self):
        """1. Valid crop extraction from authoritative bbox."""
        detections = [{"track_id": 1001, "class": "car", "confidence": 0.94, "box": [100, 100, 150, 150]}]
        crops = extract_vehicle_crops(self.dummy_frame, detections, min_confidence=0.5, min_dim=32)
        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0]["track_id"], 1001)

    def test_02_bbox_clipping(self):
        """2. Bbox clipping to image dimensions."""
        detections = [{"track_id": 1001, "class": "car", "confidence": 0.9, "box": [-20, -20, 200, 200]}]
        crops = extract_vehicle_crops(self.dummy_frame, detections, min_confidence=0.5, min_dim=32)
        self.assertEqual(len(crops), 1)
        h, w = crops[0]["crop"].shape[:2]
        self.assertTrue(h <= 480 and w <= 640)

    def test_03_invalid_bbox_rejection(self):
        """3. Invalid/out-of-bounds bbox rejection."""
        detections = [{"track_id": 1001, "class": "car", "confidence": 0.9, "box": [700, 700, 100, 100]}] # Outside 640x480
        crops = extract_vehicle_crops(self.dummy_frame, detections, min_confidence=0.5, min_dim=32)
        self.assertEqual(len(crops), 0)

    def test_04_small_crop_rejection(self):
        """4. Rejection of crops smaller than 32x32 px."""
        meta = {
            "type": "vehicle_crop", "camera_id": "cam-1", "frame_index": 1, "timestamp": 10.0,
            "track_id": 1001, "class_id": 2, "confidence": 0.9, "bbox": [10, 10, 20, 20]
        }
        _, small_jpg = cv2.imencode(".jpg", np.ones((20, 20, 3), dtype=np.uint8))
        res = validate_and_decode_crop_package(json.dumps(meta).encode("utf-8"), small_jpg.tobytes())
        self.assertIsNone(res)

    def test_05_low_confidence_rejection(self):
        """5. Rejection of low confidence detections (<0.50)."""
        meta = {
            "type": "vehicle_crop", "camera_id": "cam-1", "frame_index": 1, "timestamp": 10.0,
            "track_id": 1001, "class_id": 2, "confidence": 0.35, "bbox": [100, 100, 150, 150]
        }
        res = validate_and_decode_crop_package(json.dumps(meta).encode("utf-8"), self.dummy_jpeg_bytes)
        self.assertIsNone(res)

    def test_06_jpeg_encode_decode(self):
        """6. JPEG binary encoding and decoding verification."""
        meta = {
            "type": "vehicle_crop", "camera_id": "cam-1", "frame_index": 100, "timestamp": 200.0,
            "track_id": 1001, "class_id": 2, "confidence": 0.92, "bbox": [100, 100, 150, 150]
        }
        res = validate_and_decode_crop_package(json.dumps(meta).encode("utf-8"), self.dummy_jpeg_bytes)
        self.assertIsNotNone(res)
        self.assertIsInstance(res["crop"], np.ndarray)
        self.assertEqual(res["crop"].shape, (150, 150, 3))

    def test_07_metadata_preservation(self):
        """7. Metadata field preservation."""
        meta = {
            "type": "vehicle_crop", "camera_id": "cam-beta", "frame_index": 888, "timestamp": 999.5,
            "track_id": 1005, "class_id": 5, "confidence": 0.88, "bbox": [50, 50, 100, 100]
        }
        res = validate_and_decode_crop_package(json.dumps(meta).encode("utf-8"), self.dummy_jpeg_bytes)
        self.assertIsNotNone(res)
        self.assertEqual(res["camera_id"], "cam-beta")
        self.assertEqual(res["frame_index"], 888)
        self.assertEqual(res["timestamp"], 999.5)
        self.assertEqual(res["class_id"], 5)
        self.assertEqual(res["confidence"], 0.88)

    def test_08_track_id_preservation(self):
        """8. Authoritative C++ track_id preservation."""
        meta = {
            "type": "vehicle_crop", "camera_id": "cam-1", "frame_index": 1, "timestamp": 10.0,
            "track_id": 4321, "class_id": 2, "confidence": 0.95, "bbox": [100, 100, 150, 150]
        }
        res = validate_and_decode_crop_package(json.dumps(meta).encode("utf-8"), self.dummy_jpeg_bytes)
        self.assertIsNotNone(res)
        self.assertEqual(res["track_id"], 4321)

    def test_09_corrupted_image_payload(self):
        """9. Corrupted JPEG image payload handling."""
        meta = {
            "type": "vehicle_crop", "camera_id": "cam-1", "frame_index": 1, "timestamp": 10.0,
            "track_id": 1001, "class_id": 2, "confidence": 0.9, "bbox": [100, 100, 150, 150]
        }
        corrupted_jpeg = b"INVALID_CORRUPT_JPEG_HEADER_BYTES"
        res = validate_and_decode_crop_package(json.dumps(meta).encode("utf-8"), corrupted_jpeg)
        self.assertIsNone(res)

    def test_10_missing_metadata_handling(self):
        """10. Missing metadata payload handling."""
        res = validate_and_decode_crop_package(b"", self.dummy_jpeg_bytes)
        self.assertIsNone(res)

    def test_11_transport_failure_safety(self):
        """11. Transport failure safety (returns None gracefully)."""
        invalid_meta_json = b"INVALID_JSON_OBJECT"
        res = validate_and_decode_crop_package(invalid_meta_json, self.dummy_jpeg_bytes)
        self.assertIsNone(res)

    def test_12_reid_failure_does_not_crash(self):
        """12. Re-ID failure does not crash pipeline."""
        crop_obj = {
            "camera_id": "cam-1", "frame_index": 1, "timestamp": 10.0,
            "track_id": 1001, "class_id": 2, "confidence": 0.9, "bbox": [100, 100, 150, 150],
            "crop": None # None crop causes internal exception handled safely
        }
        res = process_production_cpp_crop(crop_obj)
        self.assertIsNone(res)

    def test_13_no_synthetic_track_ids(self):
        """13. Guarantee zero synthetic fallback track IDs (101, 102)."""
        meta = {
            "type": "vehicle_crop", "camera_id": "cam-1", "frame_index": 1, "timestamp": 10.0,
            "track_id": 1001, "class_id": 2, "confidence": 0.95, "bbox": [100, 100, 150, 150]
        }
        res = validate_and_decode_crop_package(json.dumps(meta).encode("utf-8"), self.dummy_jpeg_bytes)
        self.assertIsNotNone(res)
        self.assertNotIn(res["track_id"], [101, 102])
        self.assertEqual(res["track_id"], 1001)

    def test_14_keyframe_interval_preservation(self):
        """14. Keyframe interval gating (sampling 1 frame every 5 frames)."""
        aggregator = VehicleKeyframeAggregator(sample_interval_frames=5, target_keyframes_per_track=3)
        tid = 1001
        self.assertTrue(aggregator.should_sample(tid, frame_idx=0))
        aggregator.add_observation("cam-1", tid, [0.1]*2048, 10.0, [10, 10, 50, 50], frame_idx=0)
        self.assertFalse(aggregator.should_sample(tid, frame_idx=1))
        self.assertFalse(aggregator.should_sample(tid, frame_idx=2))
        self.assertFalse(aggregator.should_sample(tid, frame_idx=3))
        self.assertFalse(aggregator.should_sample(tid, frame_idx=4))
        self.assertTrue(aggregator.should_sample(tid, frame_idx=5))

    def test_15_embedding_dimension_2048(self):
        """15. Re-ID embedding dimension validation (2048-D)."""
        if not os.path.exists(self.model_path):
            self.skipTest("ONNX model file missing")
        extractor = ONNXReIDFeatureExtractor(self.model_path)
        crop = self.dummy_frame[100:250, 100:250]
        emb = extractor.extract(crop)
        self.assertIsNotNone(emb)
        self.assertEqual(len(emb), 2048)

    def test_16_l2_normalization_validation(self):
        """16. L2 normalization validation (||v||₂ ≈ 1.0)."""
        if not os.path.exists(self.model_path):
            self.skipTest("ONNX model file missing")
        extractor = ONNXReIDFeatureExtractor(self.model_path)
        crop = self.dummy_frame[100:250, 100:250]
        emb = extractor.extract(crop)
        self.assertIsNotNone(emb)
        arr = np.array(emb, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        self.assertAlmostEqual(norm, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
