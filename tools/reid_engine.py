#!/usr/bin/env python3
"""
ATOS v3.5 Cross-Camera Vehicle Re-Identification (Re-ID) Engine
Handles feature matching, spatiotemporal window gating, and multi-camera trajectory graph.

Adheres strictly to zero-fake-telemetry guidelines. If no trained model file is present,
reports 'Re-ID model unavailable — evaluation pending' and operates as a safe no-op.
"""

import os
import json
import time
import math
import numpy as np
from typing import List, Dict, Any, Optional

BENCHMARK_RESULTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "runs", "reid_benchmark_results.json")
)
READINESS_RESULTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "runs", "reid_readiness_status.json")
)

class ONNXReIDFeatureExtractor:
    """
    Dynamic ONNX Feature Extractor adapter.
    Inspects model input/output shapes dynamically from model weights.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = [1, 3, 256, 256]
        self.embedding_dim = 512
        self.loaded = False

        if os.path.exists(model_path):
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                shape = self.session.get_inputs()[0].shape
                self.input_shape = [s if isinstance(s, int) and s > 0 else 1 for s in shape]
                
                out_shape = self.session.get_outputs()[0].shape
                if len(out_shape) >= 2 and isinstance(out_shape[1], int):
                    self.embedding_dim = out_shape[1]

                self.loaded = True
            except Exception as e:
                print(f"[ATOS Re-ID Adapter] Model load error for {model_path}: {e}")

    def extract(self, crop_bgr: np.ndarray) -> Optional[List[float]]:
        """
        Runs ImageNet normalization and forward pass to generate L2 normalized vector.
        """
        if not self.loaded or self.session is None:
            return None

        try:
            import cv2
            target_h = self.input_shape[2] if len(self.input_shape) >= 4 else 256
            target_w = self.input_shape[3] if len(self.input_shape) >= 4 else 256

            # Resize & BGR to RGB
            resized = cv2.resize(crop_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # ImageNet Normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (rgb - mean) / std

            # NCHW Format
            tensor_in = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]

            # Forward Inference
            outputs = self.session.run([self.output_name], {self.input_name: tensor_in})
            vec = outputs[0][0]

            # L2 Normalization
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.tolist()
        except Exception as e:
            print(f"[ATOS Re-ID Adapter] Extraction error: {e}")
            return None

    def extract_batch(self, crops_bgr: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Runs ImageNet normalization and batch inference for a list of BGR crop images.
        Returns [B, embedding_dim] float32 numpy array.
        """
        if not self.loaded or self.session is None or not crops_bgr:
            return None

        try:
            import cv2
            target_h = self.input_shape[2] if len(self.input_shape) >= 4 else 256
            target_w = self.input_shape[3] if len(self.input_shape) >= 4 else 256

            batch_tensors = []
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

            for crop in crops_bgr:
                resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                norm = (rgb - mean) / std
                tensor_chw = np.transpose(norm, (2, 0, 1))
                batch_tensors.append(tensor_chw)

            batch_in = np.array(batch_tensors, dtype=np.float32)
            outputs = self.session.run([self.output_name], {self.input_name: batch_in})
            vecs = outputs[0]

            # L2 Normalization
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs_norm = vecs / norms
            return vecs_norm
        except Exception as e:
            print(f"[ATOS Re-ID Adapter] Batch extraction error: {e}")
            return None


class CrossCameraReIDManager:
    """
    Manages vehicle embeddings, cosine similarity matching, and cross-camera vehicle tracking graphs.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        
        self.enabled = config.get("enabled", False)
        self.model_path = config.get("model_path", "models/reid_vehiclenet.engine")
        self.similarity_threshold = config.get("similarity_threshold", 0.75)
        self.uncertainty_threshold = config.get("uncertainty_threshold", 0.60)
        self.embedding_dim = config.get("embedding_dim", 512)
        self.max_window_sec = config.get("max_spatiotemporal_window_sec", 300)
        self.top_k = config.get("top_k_matches", 5)

        # In-memory storage for active global vehicle tracks
        self.global_tracks: Dict[str, Dict[str, Any]] = {}
        self.matches_history: List[Dict[str, Any]] = []

        # Feature extractor adapter
        self.extractor = ONNXReIDFeatureExtractor(self.model_path)
        self.model_loaded = self.extractor.loaded

    def is_available(self) -> bool:
        """Returns True only if reid is enabled in config AND model file exists on disk."""
        return self.enabled and self.model_loaded

    def get_status_message(self) -> str:
        """Returns human-readable diagnostic status string."""
        if not self.enabled:
            return "Re-ID disabled by configuration (reid_enabled: false)"
        if not self.model_loaded:
            return "Re-ID model unavailable — evaluation pending"
        return "Re-ID Engine Active"

    def get_benchmark_results(self) -> Dict[str, Any]:
        """
        Loads measured empirical benchmark results from runs/reid_benchmark_results.json.
        Returns 'pending' state if benchmark script has not been executed on real data.
        """
        if os.path.exists(BENCHMARK_RESULTS_PATH):
            try:
                with open(BENCHMARK_RESULTS_PATH, "r") as f:
                    return json.load(f)
            except Exception:
                pass

        return {
            "status": "pending",
            "evaluated": False,
            "rank1": None,
            "rank5": None,
            "mAP": None,
            "false_match_rate": None,
            "false_non_match_rate": None,
            "inference_ms": None,
            "matching_ms": None,
            "vram_used_mb": None,
            "dataset_name": None,
            "hardware": None,
            "timestamp": None
        }

    def get_system_summary(self) -> Dict[str, Any]:
        """Summary object returned to REST API and WebSocket controllers."""
        readiness_data = {}
        if os.path.exists(READINESS_RESULTS_PATH):
            try:
                with open(READINESS_RESULTS_PATH, "r") as f:
                    readiness_data = json.load(f)
            except Exception:
                pass

        return {
            "reid_enabled": self.enabled,
            "model_loaded": self.model_loaded,
            "model_path": self.model_path,
            "status": self.get_status_message(),
            "active_global_tracks": len(self.global_tracks),
            "total_matches_found": len(self.matches_history),
            "similarity_threshold": self.similarity_threshold,
            "uncertainty_threshold": self.uncertainty_threshold,
            "benchmark": self.get_benchmark_results(),
            "readiness": readiness_data
        }

    def compute_cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Calculates cosine similarity between two feature vectors."""
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def process_feature(
        self,
        camera_id: str,
        local_track_id: int,
        embedding: List[float],
        timestamp: float,
        bbox: List[int]
    ) -> Optional[Dict[str, Any]]:
        """
        Ingests a real vehicle feature embedding and evaluates cross-camera matching.
        Enforces uncertainty threshold cutoff (similarity >= similarity_threshold). Never forces uncertain matches.
        """
        if not self.is_available():
            return None

        # Ignore small or invalid crop bounding boxes (<32px)
        if len(bbox) >= 4 and (bbox[2] < 32 or bbox[3] < 32):
            return None

        best_match_id = None
        best_score = 0.0

        for gvid, track in self.global_tracks.items():
            # Spatiotemporal constraint check
            time_delta = abs(timestamp - track["last_seen_timestamp"])
            if time_delta > self.max_window_sec:
                continue

            # Don't match against same camera track history within recent 5 seconds
            if track["last_camera_id"] == camera_id and time_delta < 5.0:
                continue

            sim_score = self.compute_cosine_similarity(embedding, track["embedding"])
            # Never force uncertain match below uncertainty threshold (0.60) or target similarity threshold (0.75)
            if sim_score >= self.similarity_threshold and sim_score > best_score:
                best_score = sim_score
                best_match_id = gvid

        if best_match_id:
            # Match found: update existing global track
            prev_cam = self.global_tracks[best_match_id]["last_camera_id"]
            self.global_tracks[best_match_id]["last_camera_id"] = camera_id
            self.global_tracks[best_match_id]["last_track_id"] = local_track_id
            self.global_tracks[best_match_id]["last_seen_timestamp"] = timestamp
            self.global_tracks[best_match_id]["embedding"] = embedding
            
            match_record = {
                "global_vehicle_id": best_match_id,
                "source_camera_id": prev_cam,
                "target_camera_id": camera_id,
                "target_local_id": local_track_id,
                "similarity_score": round(best_score, 4),
                "timestamp": time.strftime("%H:%M:%S", time.localtime(timestamp))
            }
            self.matches_history.append(match_record)
            return match_record
        else:
            # New vehicle identity
            new_gvid = f"GVID-{1000 + len(self.global_tracks) + 1}"
            self.global_tracks[new_gvid] = {
                "global_vehicle_id": new_gvid,
                "first_camera_id": camera_id,
                "last_camera_id": camera_id,
                "last_track_id": local_track_id,
                "first_seen_timestamp": timestamp,
                "last_seen_timestamp": timestamp,
                "embedding": embedding,
                "bbox": bbox
            }
            return None

    def get_matches(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns recent cross-camera identity matches."""
        return self.matches_history[-limit:]

    def get_transition_graph(self) -> Dict[str, Any]:
        """Builds real transition network topology based on recorded matches."""
        nodes = set()
        edges_map = {}

        for m in self.matches_history:
            src = m["source_camera_id"]
            dst = m["target_camera_id"]
            nodes.add(src)
            nodes.add(dst)
            key = f"{src}->{dst}"
            edges_map[key] = edges_map.get(key, 0) + 1

        edges = []
        for key, weight in edges_map.items():
            src, dst = key.split("->")
            edges.append({"source": src, "target": dst, "weight": weight})

        return {
            "nodes": [{"id": n} for n in nodes],
            "edges": edges
        }
