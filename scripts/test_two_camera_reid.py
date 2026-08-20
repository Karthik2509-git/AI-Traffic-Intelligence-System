#!/usr/bin/env python3
"""
ATOS v3.5 — Phase 2: Controlled Two-Camera Vehicle Re-ID Validation Harness
Processes prerecorded video feeds for Camera A and Camera B through the real ATOS pipeline:
Camera Frame → YOLOv8 → ByteTrack → extract_vehicle_crops() → Keyframe Aggregator → Re-ID ONNX → CrossCameraReIDManager
Generates audit records and experiment reports in runs/two_camera_reid/.
"""

import os
import sys
import time
import json
import yaml
import argparse
import numpy as np
import cv2
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.reid_engine import CrossCameraReIDManager, ONNXReIDFeatureExtractor
from tools.reid_crop_utility import extract_vehicle_crops, VehicleKeyframeAggregator

DEFAULT_CONFIG_PATH = os.path.abspath("config/settings.yaml")
DEFAULT_OUTPUT_DIR = os.path.abspath("runs/two_camera_reid")

def load_default_config():
    if os.path.exists(DEFAULT_CONFIG_PATH):
        try:
            with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                return cfg.get("reid", {})
        except Exception as e:
            print(f"[Warning] Failed to read {DEFAULT_CONFIG_PATH}: {e}")
    return {}

def parse_args():
    reid_cfg = load_default_config()

    parser = argparse.ArgumentParser(description="ATOS Controlled Two-Camera Re-ID Validation Harness")
    parser.add_argument("--camera-a", type=str, default=None, help="Path to video file for Camera A")
    parser.add_argument("--camera-b", type=str, default=None, help="Path to video file for Camera B")
    parser.add_argument("--camera-a-id", type=str, default="cam_a", help="Camera A Identifier")
    parser.add_argument("--camera-b-id", type=str, default="cam_b", help="Camera B Identifier")
    parser.add_argument("--similarity-threshold", type=float,
                        default=reid_cfg.get("similarity_threshold", 0.75),
                        help="Cosine similarity threshold for MATCH")
    parser.add_argument("--uncertainty-threshold", type=float,
                        default=reid_cfg.get("uncertainty_threshold", 0.60),
                        help="Uncertainty band lower cutoff")
    parser.add_argument("--max-transition-time", type=int,
                        default=reid_cfg.get("max_spatiotemporal_window_sec", 300),
                        help="Maximum spatiotemporal window (seconds)")
    parser.add_argument("--sample-interval", type=int, default=5,
                        help="Keyframe sampling frame interval")
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt",
                        help="YOLOv8 weights path")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for reports")
    return parser.parse_args()

class TwoCameraReIDEvaluator:
    """
    Evaluates cross-camera identity matching between Camera A and Camera B video streams.
    """
    def __init__(self, args):
        self.args = args
        self.output_dir = os.path.abspath(args.output)
        os.makedirs(self.output_dir, exist_ok=True)

        model_path = "models/fastreid_sbs_r50_ibn_veri776.onnx"
        if not os.path.exists(model_path):
            model_path = "models/fastreid_resnet50_veri776.onnx"

        self.reid_manager = CrossCameraReIDManager({
            "enabled": True, # Active for test harness run only (isolated from production settings)
            "model_path": model_path,
            "similarity_threshold": args.similarity_threshold,
            "uncertainty_threshold": args.uncertainty_threshold,
            "max_spatiotemporal_window_sec": args.max_transition_time
        })

        self.aggregator_a = VehicleKeyframeAggregator(sample_interval_frames=args.sample_interval)
        self.aggregator_b = VehicleKeyframeAggregator(sample_interval_frames=args.sample_interval)

        self.comparison_records: List[Dict[str, Any]] = []
        self.stats = {
            "camera_a_frames": 0,
            "camera_b_frames": 0,
            "total_detections": 0,
            "local_tracks_a": set(),
            "local_tracks_b": set(),
            "valid_crops_extracted": 0,
            "embeddings_generated": 0,
            "keyframes_accepted": 0,
            "candidate_comparisons": 0,
            "accepted_matches": 0,
            "uncertain_matches": 0,
            "rejected_matches": 0,
            "rejection_reasons": {
                "similarity_below_threshold": 0,
                "uncertainty_band": 0,
                "temporal_window_exceeded": 0,
                "invalid_crop": 0,
                "insufficient_keyframes": 0,
                "same_camera_exclusion": 0
            },
            "total_processing_time_sec": 0.0,
            "reid_inference_latency_ms": 0.0,
            "global_vehicle_ids_created": 0
        }

    def process_camera_stream(self, video_path: str, camera_id: str, aggregator: VehicleKeyframeAggregator, yolo_model=None):
        if not video_path or not os.path.exists(video_path):
            print(f"[NOTICE] Video file not provided or missing for {camera_id}: {video_path}")
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open video stream at {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        frame_idx = 0
        extracted_payloads = []

        print(f"\nProcessing video stream for {camera_id}: {video_path}...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            if camera_id == self.args.camera_a_id:
                self.stats["camera_a_frames"] += 1
            else:
                self.stats["camera_b_frames"] += 1

            timestamp = frame_idx / fps

            # Real YOLOv8 + ByteTrack Tracking
            if yolo_model is not None:
                try:
                    results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
                    boxes_data = []
                    if results and len(results) > 0 and results[0].boxes is not None:
                        boxes = results[0].boxes
                        for box in boxes:
                            if box.id is None:
                                continue
                            track_id = int(box.id.item())
                            cls_id = int(box.cls.item())
                            conf = float(box.conf.item())
                            xywh = box.xywh[0].cpu().numpy().tolist() # [cx, cy, w, h]
                            # Convert [cx, cy, w, h] to [x, y, w, h]
                            x = int(xywh[0] - xywh[2] / 2)
                            y = int(xywh[1] - xywh[3] / 2)
                            w = int(xywh[2])
                            h = int(xywh[3])

                            cls_name = yolo_model.names.get(cls_id, "car")
                            boxes_data.append({
                                "track_id": track_id,
                                "class": cls_name,
                                "confidence": conf,
                                "box": [x, y, w, h]
                            })

                    self.stats["total_detections"] += len(boxes_data)
                    for b in boxes_data:
                        if camera_id == self.args.camera_a_id:
                            self.stats["local_tracks_a"].add(b["track_id"])
                        else:
                            self.stats["local_tracks_b"].add(b["track_id"])

                    # Phase 1 Crop Utility Execution
                    crops = extract_vehicle_crops(frame, boxes_data, min_confidence=0.5, min_dim=32)
                    self.stats["valid_crops_extracted"] += len(crops)

                    for crop_obj in crops:
                        tid = crop_obj["track_id"]
                        if aggregator.should_sample(tid, frame_idx):
                            t0 = time.time()
                            emb = self.reid_manager.extractor.extract(crop_obj["crop"])
                            t_inf = (time.time() - t0) * 1000.0
                            if t_inf > 0:
                                self.stats["reid_inference_latency_ms"] = t_inf

                            if emb is not None:
                                self.stats["embeddings_generated"] += 1
                                self.stats["keyframes_accepted"] += 1
                                payload = aggregator.add_observation(
                                    camera_id=camera_id,
                                    track_id=tid,
                                    embedding=emb,
                                    timestamp=timestamp,
                                    bbox=crop_obj["box"],
                                    frame_idx=frame_idx,
                                    cls_name=crop_obj["class"]
                                )
                                if payload is not None:
                                    extracted_payloads.append(payload)

                except Exception as e:
                    print(f"[Warning] Tracking error on frame {frame_idx}: {e}")

        cap.release()
        return extracted_payloads

    def evaluate_association(self, payload_a: Dict[str, Any], payload_b: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates cross-camera association decision between Camera A track and Camera B track."""
        self.stats["candidate_comparisons"] += 1

        cam_a = payload_a["camera_id"]
        track_a = payload_a["local_track_id"]
        cam_b = payload_b["camera_id"]
        track_b = payload_b["local_track_id"]
        time_a = payload_a["timestamp"]
        time_b = payload_b["timestamp"]

        emb_a = payload_a["embedding"]
        emb_b = payload_b["embedding"]

        time_delta = abs(time_b - time_a)
        sim_score = float(self.reid_manager.compute_cosine_similarity(emb_a, emb_b))

        decision = "REJECTED"
        rejection_reason = None

        if cam_a == cam_b and time_delta < 5.0:
            rejection_reason = "same_camera_exclusion"
            self.stats["rejection_reasons"]["same_camera_exclusion"] += 1
            self.stats["rejected_matches"] += 1
        elif time_delta > self.args.max_transition_time:
            rejection_reason = "temporal_window_exceeded"
            self.stats["rejection_reasons"]["temporal_window_exceeded"] += 1
            self.stats["rejected_matches"] += 1
        elif sim_score >= self.args.similarity_threshold:
            decision = "MATCH"
            self.stats["accepted_matches"] += 1
        elif sim_score >= self.args.uncertainty_threshold:
            decision = "UNCERTAIN"
            rejection_reason = "uncertainty_band"
            self.stats["rejection_reasons"]["uncertainty_band"] += 1
            self.stats["uncertain_matches"] += 1
        else:
            decision = "REJECTED"
            rejection_reason = "similarity_below_threshold"
            self.stats["rejection_reasons"]["similarity_below_threshold"] += 1
            self.stats["rejected_matches"] += 1

        gvid = f"GVID-{1000 + len(self.comparison_records) + 1}" if decision == "MATCH" else "UNASSIGNED"

        record = {
            "camera_a": cam_a,
            "track_a": track_a,
            "camera_b": cam_b,
            "track_b": track_b,
            "global_vehicle_id": gvid,
            "similarity": round(sim_score, 4),
            "time_delta_seconds": round(time_delta, 2),
            "decision": decision,
            "rejection_reason": rejection_reason,
            "num_keyframes_a": payload_a.get("num_keyframes", 1),
            "num_keyframes_b": payload_b.get("num_keyframes", 1)
        }
        self.comparison_records.append(record)
        return record

    def run(self):
        start_time = time.time()
        yolo_model = None

        if self.args.camera_a or self.args.camera_b:
            try:
                from ultralytics import YOLO
                print(f"Loading YOLOv8 detector from {self.args.yolo_model}...")
                yolo_model = YOLO(self.args.yolo_model)
            except Exception as e:
                print(f"[ERROR] Failed to load YOLOv8 detector: {e}")

        payloads_a = self.process_camera_stream(self.args.camera_a, self.args.camera_a_id, self.aggregator_a, yolo_model)
        payloads_b = self.process_camera_stream(self.args.camera_b, self.args.camera_b_id, self.aggregator_b, yolo_model)

        print("\nEvaluating Cross-Camera Vehicle Re-ID Associations...")
        for pa in payloads_a:
            for pb in payloads_b:
                self.evaluate_association(pa, pb)

        self.stats["total_processing_time_sec"] = round(time.time() - start_time, 2)
        self.stats["global_vehicle_ids_created"] = len([r for r in self.comparison_records if r["decision"] == "MATCH"])

        self.save_artifacts()

    def save_artifacts(self):
        summary_path = os.path.join(self.output_dir, "summary.json")
        matches_path = os.path.join(self.output_dir, "matches.json")
        report_path = os.path.join(self.output_dir, "experiment_report.md")

        summary_data = {
            "status": "completed" if (self.args.camera_a or self.args.camera_b) else "pending_video_files",
            "camera_a_video": self.args.camera_a,
            "camera_b_video": self.args.camera_b,
            "similarity_threshold": self.args.similarity_threshold,
            "uncertainty_threshold": self.args.uncertainty_threshold,
            "max_transition_time_sec": self.args.max_transition_time,
            "camera_a_frames": self.stats["camera_a_frames"],
            "camera_b_frames": self.stats["camera_b_frames"],
            "total_detections": self.stats["total_detections"],
            "local_tracks_a_count": len(self.stats["local_tracks_a"]),
            "local_tracks_b_count": len(self.stats["local_tracks_b"]),
            "valid_crops_extracted": self.stats["valid_crops_extracted"],
            "embeddings_generated": self.stats["embeddings_generated"],
            "keyframes_accepted": self.stats["keyframes_accepted"],
            "candidate_comparisons": self.stats["candidate_comparisons"],
            "accepted_matches": self.stats["accepted_matches"],
            "uncertain_matches": self.stats["uncertain_matches"],
            "rejected_matches": self.stats["rejected_matches"],
            "rejection_reasons": self.stats["rejection_reasons"],
            "reid_inference_latency_ms": round(self.stats["reid_inference_latency_ms"], 2),
            "total_processing_time_sec": self.stats["total_processing_time_sec"],
            "global_vehicle_ids_created": self.stats["global_vehicle_ids_created"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        with open(matches_path, "w", encoding="utf-8") as f:
            json.dump(self.comparison_records, f, indent=2)

        md_content = f"""# 🔬 ATOS v3.5 Controlled Two-Camera Vehicle Re-ID Experiment Report

**Status:** `{summary_data['status']}`  
**Timestamp:** `{summary_data['timestamp']}`  
**Camera A Video:** `{self.args.camera_a or 'NOT SUPPLIED (PENDING VIDEO FILES)'}`  
**Camera B Video:** `{self.args.camera_b or 'NOT SUPPLIED (PENDING VIDEO FILES)'}`  

---

## 📊 Empirically Measured Experiment Metrics

| Metric | Measured Value | Unit / Status |
| :--- | :---: | :--- |
| **Camera A Frames Processed** | `{self.stats['camera_a_frames']}` | frames |
| **Camera B Frames Processed** | `{self.stats['camera_b_frames']}` | frames |
| **Total Vehicle Detections** | `{self.stats['total_detections']}` | detections |
| **Local Tracks (Camera A)** | `{len(self.stats['local_tracks_a'])}` | track IDs |
| **Local Tracks (Camera B)** | `{len(self.stats['local_tracks_b'])}` | track IDs |
| **Valid Vehicle Crops Extracted** | `{self.stats['valid_crops_extracted']}` | crops (>= 32x32 px) |
| **Re-ID Embeddings Generated** | `{self.stats['embeddings_generated']}` | 2048-dim vectors |
| **Keyframes Accepted** | `{self.stats['keyframes_accepted']}` | keyframes |
| **Candidate Comparisons** | `{self.stats['candidate_comparisons']}` | comparisons |
| **Accepted Cross-Camera Matches** | `{self.stats['accepted_matches']}` | matches (>= 75%) |
| **Uncertain Matches (Band)** | `{self.stats['uncertain_matches']}` | matches (60% - 75%) |
| **Rejected Comparisons** | `{self.stats['rejected_matches']}` | rejections |
| **Re-ID Inference Latency** | `{summary_data['reid_inference_latency_ms']} ms` | per crop ONNX |
| **Total Processing Duration** | `{self.stats['total_processing_time_sec']} s` | execution time |
| **Global Vehicle IDs Created** | `{self.stats['global_vehicle_ids_created']}` | GVID assignments |

---

## 🛑 Rejection Reasons Breakdown

- **Similarity Below Cutoff (< 0.60)**: {self.stats['rejection_reasons']['similarity_below_threshold']}
- **Uncertainty Band (0.60 - 0.75)**: {self.stats['rejection_reasons']['uncertainty_band']}
- **Temporal Window Exceeded (> 300s)**: {self.stats['rejection_reasons']['temporal_window_exceeded']}
- **Invalid / Small Crop (< 32px)**: {self.stats['rejection_reasons']['invalid_crop']}
- **Insufficient Keyframes (< 3)**: {self.stats['rejection_reasons']['insufficient_keyframes']}
- **Same Camera Exclusion (< 5s)**: {self.stats['rejection_reasons']['same_camera_exclusion']}
"""

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"\n[SUCCESS] Experiment artifacts written to {self.output_dir}:")
        print(f"  - {summary_path}")
        print(f"  - {matches_path}")
        print(f"  - {report_path}")

def main():
    args = parse_args()
    evaluator = TwoCameraReIDEvaluator(args)
    evaluator.run()

if __name__ == "__main__":
    main()
