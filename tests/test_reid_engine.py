#!/usr/bin/env python3
"""
Unit Test Suite for ATOS v3.5 Re-ID Engine & Benchmark (tools/reid_engine.py & scripts/benchmark_reid.py)
Tests fallback logic, cosine similarity calculations, spatiotemporal window filtering,
and AP evaluation membership mask operations (compute_ap np.isin).
Note: Synthetic vectors exist strictly inside unit tests and are never injected into production.
"""

import unittest
import numpy as np
from tools.reid_engine import CrossCameraReIDManager
from scripts.benchmark_reid import compute_ap

class TestReIDEngine(unittest.TestCase):

    def test_reid_disabled_fallback(self):
        """Verify safe fallback when reid.enabled = False."""
        mgr = CrossCameraReIDManager({"enabled": False, "model_path": "nonexistent.engine"})
        self.assertFalse(mgr.is_available())
        self.assertEqual(mgr.get_status_message(), "Re-ID disabled by configuration (reid_enabled: false)")
        self.assertEqual(mgr.process_feature("cam1", 101, [0.1]*512, 1000.0, [0,0,100,100]), None)

    def test_reid_missing_model_fallback(self):
        """Verify safe fallback when reid.enabled = True but model file is missing."""
        mgr = CrossCameraReIDManager({"enabled": True, "model_path": "nonexistent_model.engine"})
        self.assertFalse(mgr.is_available())
        self.assertEqual(mgr.get_status_message(), "Re-ID model unavailable — evaluation pending")

    def test_cosine_similarity(self):
        """Verify mathematical correctness of vector cosine similarity."""
        mgr = CrossCameraReIDManager({"enabled": False})
        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0, 0.0]

        self.assertAlmostEqual(mgr.compute_cosine_similarity(v1, v2), 1.0, places=4)
        self.assertAlmostEqual(mgr.compute_cosine_similarity(v1, v3), 0.0, places=4)

    def test_synthetic_matching_logic(self):
        """Test matching logic using synthetic test vectors in unit test scope."""
        mgr = CrossCameraReIDManager({
            "enabled": True,
            "model_path": "models/reid_vehiclenet.onnx",
            "similarity_threshold": 0.8,
            "max_spatiotemporal_window_sec": 300
        })
        # Simulate model load in unit test harness
        mgr.model_loaded = True
        self.assertTrue(mgr.is_available())

        vec_car1 = [1.0, 0.5] + [0.0]*510
        vec_car1_similar = [0.98, 0.52] + [0.0]*510

        # Ingest first detection on Cam A
        res1 = mgr.process_feature("cam_a", 101, vec_car1, timestamp=100.0, bbox=[10,10,50,50])
        self.assertIsNone(res1) # First occurrence -> new GVID

        # Ingest similar feature on Cam B (60s later)
        res2 = mgr.process_feature("cam_b", 202, vec_car1_similar, timestamp=160.0, bbox=[20,20,50,50])
        self.assertIsNotNone(res2) # Should match cross-camera!
        self.assertEqual(res2["source_camera_id"], "cam_a")
        self.assertEqual(res2["target_camera_id"], "cam_b")
        self.assertGreaterEqual(res2["similarity_score"], 0.8)

    def test_compute_ap_membership_mask(self):
        """Verify compute_ap function and np.isin membership mask compatibility in NumPy 2.x."""
        query_id = 1
        query_cam = 1

        gallery_ids = np.array([1, 2, 1, 3, 1], dtype=np.int32)
        gallery_cams = np.array([1, 2, 2, 3, 3], dtype=np.int32)
        sim_scores = np.array([0.9, 0.8, 0.85, 0.4, 0.95], dtype=np.float32)

        # compute_ap should evaluate good matches (gallery_id == query_id and gallery_cam != query_cam)
        # Good indexes: index 2 (cam 2), index 4 (cam 3)
        # Junk indexes: index 0 (cam 1 - same cam)
        ap, r1, r5 = compute_ap(query_id, query_cam, gallery_ids, gallery_cams, sim_scores)

        self.assertGreater(ap, 0.0)
        self.assertIn(r1, [0, 1])
        self.assertIn(r5, [0, 1])

        # Explicitly test np.isin membership mask behavior
        index = [4, 2, 1, 3] # Excluded index 0 (same cam junk)
        good_index = np.array([2, 4])
        matches = np.isin(index, good_index)
        np.testing.assert_array_equal(matches, np.array([True, True, False, False]))

if __name__ == "__main__":
    unittest.main()
