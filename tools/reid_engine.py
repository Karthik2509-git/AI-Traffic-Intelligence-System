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
        self.embedding_dim = config.get("embedding_dim", 512)
        self.max_window_sec = config.get("max_spatiotemporal_window_sec", 300)
        self.top_k = config.get("top_k_matches", 5)

        # In-memory storage for active global vehicle tracks
        self.global_tracks: Dict[str, Dict[str, Any]] = {}
        self.matches_history: List[Dict[str, Any]] = []

        # Check model file availability on disk
        self.model_loaded = False
        if self.enabled and os.path.exists(self.model_path):
            self.model_loaded = True

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
        return {
            "reid_enabled": self.enabled,
            "model_loaded": self.model_loaded,
            "model_path": self.model_path,
            "status": self.get_status_message(),
            "active_global_tracks": len(self.global_tracks),
            "total_matches_found": len(self.matches_history),
            "similarity_threshold": self.similarity_threshold,
            "benchmark": self.get_benchmark_results()
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
        """
        if not self.is_available():
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
            if sim_score > self.similarity_threshold and sim_score > best_score:
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
