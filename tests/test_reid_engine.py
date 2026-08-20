#!/usr/bin/env python3
"""
Unit Test Suite for ATOS v3.5 Re-ID Engine (tools/reid_engine.py)
Tests fallback logic, cosine similarity calculations, and spatiotemporal window filtering.
Note: Synthetic vectors exist strictly inside unit tests and are never injected into production.
"""

import unittest
import numpy as np
from tools.reid_engine import CrossCameraReIDManager

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
            "model_path": "config/settings.yaml", # Uses an existing file to simulate model presence
            "similarity_threshold": 0.8,
            "max_spatiotemporal_window_sec": 300
        })
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

if __name__ == "__main__":
    unittest.main()
