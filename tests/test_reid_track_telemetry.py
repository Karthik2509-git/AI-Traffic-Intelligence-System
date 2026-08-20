#!/usr/bin/env python3
"""
Phase 1 C++ -> Python Track Telemetry Bridge Unit Tests
Validates all 10 required test requirements:
1. Valid track_telemetry packet
2. Multiple tracks handling
3. Vehicle class filtering (COCO: 2=car, 3=motorcycle, 5=bus, 7=truck)
4. Invalid/missing track_id handling
5. Invalid bbox format/dimensions
6. Invalid confidence values
7. Malformed JSON handling
8. Unknown packet type rejection
9. Guarantee zero synthetic fallback track IDs
10. Telemetry end-to-end JSON serialization/deserialization
"""

import os
import sys
import json
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.web_gateway import validate_and_parse_track_telemetry


class TestTrackTelemetryBridge(unittest.TestCase):

    def test_01_valid_track_telemetry_packet(self):
        """1. Valid track_telemetry packet parsing."""
        packet = {
            "type": "track_telemetry",
            "camera_id": "cam-1",
            "frame_index": 1234,
            "timestamp": 1724174000.123,
            "tracks": [
                {
                    "track_id": 1001,
                    "class_id": 2, # car
                    "confidence": 0.94,
                    "bbox": [120, 180, 240, 160]
                }
            ]
        }
        res = validate_and_parse_track_telemetry(packet)
        self.assertIsNotNone(res)
        self.assertEqual(res["camera_id"], "cam-1")
        self.assertEqual(res["frame_index"], 1234)
        self.assertEqual(res["timestamp"], 1724174000.123)
        self.assertEqual(len(res["tracks"]), 1)
        self.assertEqual(res["tracks"][0]["track_id"], 1001)

    def test_02_multiple_tracks(self):
        """2. Multiple valid tracks in a single packet."""
        packet = {
            "type": "track_telemetry",
            "camera_id": "cam-2",
            "frame_index": 500,
            "timestamp": 1724174100.0,
            "tracks": [
                {"track_id": 1001, "class_id": 2, "confidence": 0.92, "bbox": [10, 10, 50, 50]},
                {"track_id": 1002, "class_id": 5, "confidence": 0.88, "bbox": [100, 100, 200, 150]},
                {"track_id": 1003, "class_id": 7, "confidence": 0.95, "bbox": [300, 300, 100, 100]}
            ]
        }
        res = validate_and_parse_track_telemetry(packet)
        self.assertIsNotNone(res)
        self.assertEqual(len(res["tracks"]), 3)
        tids = [t["track_id"] for t in res["tracks"]]
        self.assertEqual(tids, [1001, 1002, 1003])

    def test_03_vehicle_class_filtering(self):
        """3. Vehicle class filtering: only COCO 2, 3, 5, 7 accepted; non-vehicles filtered out."""
        packet = {
            "type": "track_telemetry",
            "camera_id": "cam-1",
            "frame_index": 10,
            "timestamp": 100.0,
            "tracks": [
                {"track_id": 1001, "class_id": 0, "confidence": 0.9, "bbox": [10, 10, 20, 20]}, # person -> filter
                {"track_id": 1002, "class_id": 2, "confidence": 0.9, "bbox": [10, 10, 20, 20]}, # car -> accept
                {"track_id": 1003, "class_id": 1, "confidence": 0.9, "bbox": [10, 10, 20, 20]}, # bicycle -> filter
                {"track_id": 1004, "class_id": 3, "confidence": 0.9, "bbox": [10, 10, 20, 20]}  # motorcycle -> accept
            ]
        }
        res = validate_and_parse_track_telemetry(packet)
        self.assertIsNotNone(res)
        self.assertEqual(len(res["tracks"]), 2)
        cids = [t["class_id"] for t in res["tracks"]]
        self.assertEqual(cids, [2, 3])

    def test_04_invalid_missing_track_id(self):
        """4. Invalid or missing track_id filtered out safely."""
        packet = {
            "type": "track_telemetry",
            "camera_id": "cam-1",
            "frame_index": 10,
            "timestamp": 100.0,
            "tracks": [
                {"class_id": 2, "confidence": 0.9, "bbox": [10, 10, 20, 20]},           # missing track_id
                {"track_id": -1, "class_id": 2, "confidence": 0.9, "bbox": [10, 10, 20, 20]}, # negative track_id
                {"track_id": "1001", "class_id": 2, "confidence": 0.9, "bbox": [10, 10, 20, 20]}, # string track_id
                {"track_id": 1002, "class_id": 2, "confidence": 0.9, "bbox": [10, 10, 20, 20]} # valid
            ]
        }
        res = validate_and_parse_track_telemetry(packet)
        self.assertIsNotNone(res)
        self.assertEqual(len(res["tracks"]), 1)
        self.assertEqual(res["tracks"][0]["track_id"], 1002)

    def test_05_invalid_bbox(self):
        """5. Invalid bbox dimensions or types filtered out safely."""
        packet = {
            "type": "track_telemetry",
            "camera_id": "cam-1",
            "frame_index": 10,
            "timestamp": 100.0,
            "tracks": [
                {"track_id": 1001, "class_id": 2, "confidence": 0.9, "bbox": [10, 10, 20]},        # only 3 elements
                {"track_id": 1002, "class_id": 2, "confidence": 0.9, "bbox": [-5, 10, 20, 20]},     # negative coordinate
                {"track_id": 1003, "class_id": 2, "confidence": 0.9, "bbox": "invalid_bbox"},       # string bbox
                {"track_id": 1004, "class_id": 2, "confidence": 0.9, "bbox": [10, 10, 20, 20]}       # valid
            ]
        }
        res = validate_and_parse_track_telemetry(packet)
        self.assertIsNotNone(res)
        self.assertEqual(len(res["tracks"]), 1)
        self.assertEqual(res["tracks"][0]["track_id"], 1004)

    def test_06_invalid_confidence(self):
        """6. Invalid confidence values (<0 or >1 or wrong type) filtered out."""
        packet = {
            "type": "track_telemetry",
            "camera_id": "cam-1",
            "frame_index": 10,
            "timestamp": 100.0,
            "tracks": [
                {"track_id": 1001, "class_id": 2, "confidence": 1.5, "bbox": [10, 10, 20, 20]},   # conf > 1.0
                {"track_id": 1002, "class_id": 2, "confidence": -0.1, "bbox": [10, 10, 20, 20]},  # conf < 0.0
                {"track_id": 1003, "class_id": 2, "confidence": "high", "bbox": [10, 10, 20, 20]},# non-numeric
                {"track_id": 1004, "class_id": 2, "confidence": 0.85, "bbox": [10, 10, 20, 20]}  # valid
            ]
        }
        res = validate_and_parse_track_telemetry(packet)
        self.assertIsNotNone(res)
        self.assertEqual(len(res["tracks"]), 1)
        self.assertEqual(res["tracks"][0]["track_id"], 1004)

    def test_07_malformed_json_schema(self):
        """7. Malformed top-level payload structure returns None."""
        invalid_packets = [
            None,
            "string_payload",
            123,
            {"type": "track_telemetry"}, # missing camera_id, frame_index, timestamp, tracks
            {"type": "track_telemetry", "camera_id": "cam1", "frame_index": "abc", "timestamp": 10.0, "tracks": []}, # non-int frame_index
            {"type": "track_telemetry", "camera_id": "cam1", "frame_index": 1, "timestamp": 10.0, "tracks": "not_a_list"} # tracks not list
        ]
        for pkt in invalid_packets:
            self.assertIsNone(validate_and_parse_track_telemetry(pkt))

    def test_08_unknown_packet_type(self):
        """8. Unknown packet type rejected cleanly."""
        packet = {"type": "unknown_custom_packet", "data": 123}
        self.assertIsNone(validate_and_parse_track_telemetry(packet))

    def test_09_no_synthetic_track_ids(self):
        """9. Guarantee zero synthetic fallback track IDs (101, 102) are injected by bridge parser."""
        packet = {
            "type": "track_telemetry",
            "camera_id": "cam-1",
            "frame_index": 100,
            "timestamp": 200.0,
            "tracks": [] # Empty track list from C++
        }
        res = validate_and_parse_track_telemetry(packet)
        self.assertIsNotNone(res)
        self.assertEqual(len(res["tracks"]), 0) # Stays empty, NO 101/102 synthetic injection!

    def test_10_telemetry_serialization(self):
        """10. End-to-end JSON serialization and deserialization matching C++ output format."""
        # Simulated C++ JSON output string
        cpp_json_str = '{"type":"track_telemetry", "camera_id":"cam-1", "frame_index":55, "timestamp":1724174000.123, "tracks":[{"track_id":1001, "class_id":2, "confidence":0.94, "bbox":[120,180,240,160]}]}'
        payload = json.loads(cpp_json_str)
        res = validate_and_parse_track_telemetry(payload)

        self.assertIsNotNone(res)
        self.assertEqual(res["camera_id"], "cam-1")
        self.assertEqual(res["frame_index"], 55)
        self.assertEqual(res["tracks"][0]["track_id"], 1001)
        self.assertEqual(res["tracks"][0]["bbox"], [120, 180, 240, 160])


if __name__ == "__main__":
    unittest.main()
