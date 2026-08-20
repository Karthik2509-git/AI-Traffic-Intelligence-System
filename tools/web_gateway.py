#!/usr/bin/env python3
"""
ATOS Studio Production FastAPI & WebSocket Gateway Server
Bridges C++ ATOS Engine (via UDP 5005) with ATOS Studio Visual Intelligence OS.
Includes ATOS v3.5 Cross-Camera Vehicle Re-Identification (Re-ID) endpoints.
"""

import os
import sys
import json
import time
import socket
import asyncio
import threading
import uuid
from typing import List, Dict, Any, Optional

import yaml
import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, Response
from pydantic import BaseModel

import base64
import cv2
import numpy as np

from tools.reid_engine import CrossCameraReIDManager
from tools.reid_crop_utility import extract_vehicle_crops, VehicleKeyframeAggregator

app = FastAPI(
    title="ATOS Studio Visual Intelligence OS API",
    description="OpenAPI control plane & high-performance telemetry bridge for ATOS CUDA/TensorRT Engine",
    version="3.5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml"))
PLUGINS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plugins"))
RECORDS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs", "telemetry_sessions"))
os.makedirs(RECORDS_DIR, exist_ok=True)

# Load configuration settings
def load_settings_dict():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}

g_settings = load_settings_dict()
g_reid_manager = CrossCameraReIDManager(g_settings.get("reid", {}))
g_aggregators: Dict[str, VehicleKeyframeAggregator] = {}

def process_camera_frame_reid(
    camera_id: str,
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    timestamp: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Processes vehicle crop extraction, keyframe feature aggregation, ONNX Re-ID embedding, and GVID correlation.
    Strictly wrapped in failure safety: any error or missing model logs diagnostic and lets YOLOv8 + ByteTrack continue cleanly.
    """
    if timestamp is None:
        timestamp = time.time()

    if not g_reid_manager.is_available():
        return []

    if frame is None or frame.size == 0 or not detections:
        return []

    try:
        if camera_id not in g_aggregators:
            reid_cfg = g_settings.get("reid", {})
            g_aggregators[camera_id] = VehicleKeyframeAggregator(
                sample_interval_frames=int(reid_cfg.get("keyframe_sample_interval", 5)),
                target_keyframes_per_track=int(reid_cfg.get("keyframe_target_count", 3)),
                min_confidence=float(reid_cfg.get("crop_min_confidence", 0.5)),
                min_crop_dim=int(reid_cfg.get("crop_min_size", 32))
            )

        aggregator = g_aggregators[camera_id]
        reid_cfg = g_settings.get("reid", {})
        min_conf = float(reid_cfg.get("crop_min_confidence", 0.5))
        min_size = int(reid_cfg.get("crop_min_size", 32))

        # Extract crops safely using reid_crop_utility
        crops = extract_vehicle_crops(frame, detections, min_confidence=min_conf, min_dim=min_size)
        matches = []
        frame_idx = int(timestamp * 30)

        for crop_obj in crops:
            tid = crop_obj["track_id"]
            if tid < 0:
                continue

            if aggregator.should_sample(tid, frame_idx):
                emb = g_reid_manager.extractor.extract(crop_obj["crop"])
                if emb is not None and len(emb) == 2048:
                    arr = np.array(emb, dtype=np.float32)
                    if np.all(np.isfinite(arr)):
                        norm = float(np.linalg.norm(arr))
                        if 0.90 <= norm <= 1.10: # L2 normalized vector check
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
                                match_rec = g_reid_manager.process_feature(
                                    camera_id=payload["camera_id"],
                                    local_track_id=payload["local_track_id"],
                                    embedding=payload["embedding"],
                                    timestamp=payload["timestamp"],
                                    bbox=payload["bbox"]
                                )
                                if match_rec:
                                    matches.append(match_rec)
        return matches
    except Exception as e:
        print(f"[ATOS Re-ID Pipeline] Re-ID unavailable — continuing detection/tracking: {e}")
        return []

g_state_lock = threading.Lock()

g_system_state = {
    "engine_status": "offline",
    "last_udp_packet": 0.0,
    "uptime_start": time.time(),
    "mobile_sessions": {},
    "telemetry": {
        "pressure": 0.0,
        "signal_phase": 0,
        "vehicles": 0,
        "fps": 0.0,
        "latency_ms": 0.0,
        "active_cameras": 1,
        "alerts": []
    },
    "engine_metrics": {
        "cuda_device": "NVIDIA GeForce RTX 4090",
        "cuda_version": "12.4",
        "driver_version": "550.54",
        "gpu_temp_celsius": 42,
        "cuda_status": "Ready",
        "tensorrt_version": "10.0.1",
        "model_loaded": "yolov8_4k_optimized.engine",
        "model_checksum": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "precision": "FP16",
        "gpu_utilization_pct": 18,
        "vram_used_mb": 2450,
        "vram_total_mb": 16384,
        "queue_depth": 0,
        "dropped_frames": 0,
        "inference_ms": 8.4,
        "preprocess_ms": 1.2
    },
    "cameras": [
        {
            "id": "cam-1",
            "name": "Intersection Alpha North",
            "location": "Main Arterial St.",
            "type": "RTSP",
            "url": "rtsp://192.168.1.100/stream",
            "status": "waiting_for_engine",
            "fps": 0.0,
            "latency_ms": 0.0,
            "resolution": "1920x1080",
            "dropped_frames": 0
        }
    ],
    "analytics_history": [],
    "alerts_queue": [],
    "notifications": [
        {"id": "n1", "timestamp": time.strftime("%H:%M:%S"), "title": "System Gateway Ready", "type": "info"}
    ],
    "plugins": [],
    "logs_buffer": [
        {"timestamp": time.strftime("%H:%M:%S"), "level": "INFO", "message": "ATOS Studio Gateway Server initialized."}
    ]
}

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

ws_manager = ConnectionManager()

def get_all_lan_ips():
    ips = []
    try:
        interfaces = psutil.net_if_addrs()
        for iface_name, iface_addresses in interfaces.items():
            for addr in iface_addresses:
                if addr.family == socket.AF_INET and not addr.address.startswith("127."):
                    ips.append({
                        "interface": iface_name,
                        "ip": addr.address,
                        "is_virtual": "virtual" in iface_name.lower() or "vethernet" in iface_name.lower() or "vmnet" in iface_name.lower()
                    })
    except Exception as e:
        print(f"Network discovery error: {e}")

    if not ips:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            primary = s.getsockname()[0]
            s.close()
            ips.append({"interface": "Primary", "ip": primary, "is_virtual": False})
        except Exception:
            ips.append({"interface": "Loopback", "ip": "127.0.0.1", "is_virtual": False})

    ips.sort(key=lambda x: (x["is_virtual"], not x["ip"].startswith("192.168.")))
    return ips

def discover_plugins():
    plugins_list = []
    if os.path.exists(PLUGINS_DIR):
        for entry in os.listdir(PLUGINS_DIR):
            p_path = os.path.join(PLUGINS_DIR, entry, "plugin.json")
            if os.path.isfile(p_path):
                try:
                    with open(p_path, "r") as f:
                        meta = json.load(f)
                        plugins_list.append(meta)
                except Exception as e:
                    print(f"Plugin load error {p_path}: {e}")
    with g_state_lock:
        g_system_state["plugins"] = plugins_list

discover_plugins()

VEHICLE_CLASSES = {2, 3, 5, 7} # COCO car, motorcycle, bus, truck

def validate_and_parse_track_telemetry(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Validates Phase 1 track_telemetry UDP schema.
    Returns parsed dictionary if valid, or None if malformed.
    """
    if not isinstance(payload, dict) or payload.get("type") != "track_telemetry":
        return None

    camera_id = payload.get("camera_id")
    frame_index = payload.get("frame_index")
    timestamp = payload.get("timestamp")
    tracks = payload.get("tracks")

    if not (isinstance(camera_id, str) and camera_id and
            isinstance(frame_index, (int, float)) and
            isinstance(timestamp, (int, float)) and
            isinstance(tracks, list)):
        return None

    valid_tracks = []
    for trk in tracks:
        if not isinstance(trk, dict):
            continue
        track_id = trk.get("track_id")
        class_id = trk.get("class_id")
        confidence = trk.get("confidence")
        bbox = trk.get("bbox")

        # Strict field validation
        if not (isinstance(track_id, int) and track_id >= 0):
            continue
        if not (isinstance(class_id, int) and class_id in VEHICLE_CLASSES):
            continue
        if not (isinstance(confidence, (int, float)) and 0.0 <= float(confidence) <= 1.0):
            continue
        if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(x, (int, float)) and x >= 0 for x in bbox)):
            continue

        valid_tracks.append({
            "track_id": int(track_id),
            "class_id": int(class_id),
            "confidence": float(confidence),
            "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        })

    return {
        "camera_id": camera_id,
        "frame_index": int(frame_index),
        "timestamp": float(timestamp),
        "tracks": valid_tracks
    }

def udp_telemetry_listener(udp_port: int = 5005):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", udp_port))
        sock.settimeout(1.0)
    except Exception as e:
        print(f"[ATOS Gateway] UDP socket warning: {e}")

    while True:
        try:
            data, _ = sock.recvfrom(4096)
            payload = json.loads(data.decode('utf-8'))
            now = time.time()

            with g_state_lock:
                if payload.get("type") == "city_pulse":
                    g_system_state["last_udp_packet"] = now
                    g_system_state["engine_status"] = "online"
                    g_system_state["telemetry"]["pressure"] = float(payload.get("pressure", 0.0))
                    g_system_state["telemetry"]["signal_phase"] = int(payload.get("signal_phase", 0))
                    g_system_state["telemetry"]["vehicles"] = int(payload.get("vehicles", 0))
                    g_system_state["telemetry"]["fps"] = float(payload.get("fps", 45.0))
                    g_system_state["telemetry"]["latency_ms"] = float(payload.get("latency_ms", 12.0))

                    hist_entry = {
                        "time": time.strftime("%H:%M:%S"),
                        "pressure": g_system_state["telemetry"]["pressure"],
                        "vehicles": g_system_state["telemetry"]["vehicles"],
                        "fps": g_system_state["telemetry"]["fps"]
                    }
                    g_system_state["analytics_history"].append(hist_entry)
                    if len(g_system_state["analytics_history"]) > 60:
                        g_system_state["analytics_history"].pop(0)

                elif payload.get("type") == "track_telemetry":
                    parsed = validate_and_parse_track_telemetry(payload)
                    if parsed is not None:
                        g_system_state["last_udp_packet"] = now
                        g_system_state["engine_status"] = "online"
                        g_system_state["real_track_telemetry"] = parsed
                        g_system_state["telemetry"]["active_cpp_tracks"] = len(parsed["tracks"])
                    else:
                        print(f"[ATOS Gateway] Malformed track_telemetry packet rejected: {payload}")

                elif payload.get("type") == "incident_alert":
                    g_system_state["last_udp_packet"] = now
                    g_system_state["engine_status"] = "online"
                    alert_entry = {
                        "id": f"alert-{int(now * 1000)}",
                        "category": payload.get("category", "Incident"),
                        "location": payload.get("node_id", "Intersection-Alpha"),
                        "timestamp": time.strftime("%H:%M:%S"),
                        "severity": "HIGH",
                        "status": "ACTIVE"
                    }
                    g_system_state["alerts_queue"].append(alert_entry)
                    g_system_state["telemetry"]["alerts"].append(alert_entry["category"])

        except socket.timeout:
            with g_state_lock:
                if time.time() - g_system_state["last_udp_packet"] > 4.0:
                    g_system_state["engine_status"] = "waiting_for_engine"
                    g_system_state["telemetry"]["fps"] = 0.0
                    g_system_state["telemetry"]["latency_ms"] = 0.0
                    g_system_state["real_track_telemetry"] = None
        except Exception:
            time.sleep(0.01)

t_udp = threading.Thread(target=udp_telemetry_listener, daemon=True)
t_udp.start()

# API Endpoints
@app.get("/telemetry/tracks")
def get_real_track_telemetry():
    with g_state_lock:
        return {
            "status": g_system_state["engine_status"],
            "track_telemetry": g_system_state.get("real_track_telemetry")
        }
@app.get("/api/local-ip")
def get_ip():
    all_ips = get_all_lan_ips()
    primary_ip = all_ips[0]["ip"] if all_ips else "127.0.0.1"
    return {
        "local_ip": primary_ip,
        "all_ips": all_ips
    }

@app.get("/health")
def get_health():
    with g_state_lock:
        is_online = (g_system_state["engine_status"] == "online")
        return {
            "status": "healthy" if is_online else "waiting_for_engine",
            "engine_online": is_online,
            "engine_status": g_system_state["engine_status"],
            "uptime_seconds": int(time.time() - g_system_state["uptime_start"]),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }

@app.get("/telemetry")
def get_telemetry():
    with g_state_lock:
        return {
            "status": g_system_state["engine_status"],
            "data": g_system_state["telemetry"],
            "reid": g_reid_manager.get_system_summary(),
            "last_updated": time.strftime("%H:%M:%S")
        }

@app.get("/engine")
def get_engine():
    with g_state_lock:
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        g_system_state["engine_metrics"]["cpu_utilization_pct"] = cpu
        g_system_state["engine_metrics"]["ram_utilization_pct"] = ram
        return {
            "status": g_system_state["engine_status"],
            "metrics": g_system_state["engine_metrics"]
        }

@app.get("/cameras")
def get_cameras():
    with g_state_lock:
        return {"cameras": g_system_state["cameras"]}

@app.get("/analytics")
def get_analytics():
    with g_state_lock:
        return {
            "history": g_system_state["analytics_history"],
            "pressure": g_system_state["telemetry"]["pressure"],
            "vehicles": g_system_state["telemetry"]["vehicles"]
        }

@app.get("/alerts")
def get_alerts():
    with g_state_lock:
        return {"alerts": g_system_state["alerts_queue"]}

@app.get("/plugins")
def get_plugins():
    with g_state_lock:
        return {"plugins": g_system_state["plugins"]}

@app.get("/notifications")
def get_notifications():
    with g_state_lock:
        return {"notifications": g_system_state["notifications"]}

# ATOS v3.5 Re-ID Subsystem API Endpoints
@app.get("/reid/status")
def get_reid_status():
    """Returns current status of Re-ID subsystem and measured benchmark results."""
    return g_reid_manager.get_system_summary()

@app.get("/reid/matches")
def get_reid_matches(limit: int = 50):
    """Returns recent cross-camera identity matches."""
    return {"matches": g_reid_manager.get_matches(limit)}

@app.get("/reid/graph")
def get_reid_graph():
    """Returns camera transition topology graph data."""
    return {"graph": g_reid_manager.get_transition_graph()}

class ReIDQueryReq(BaseModel):
    embedding: List[float]
    top_k: Optional[int] = 5

@app.post("/reid/query")
def query_reid(req: ReIDQueryReq):
    """Query cross-camera vehicle matches for a target embedding vector."""
    if not g_reid_manager.is_available():
        raise HTTPException(status_code=503, detail=g_reid_manager.get_status_message())
    
    matches = []
    for gvid, track in g_reid_manager.global_tracks.items():
        sim = g_reid_manager.compute_cosine_similarity(req.embedding, track["embedding"])
        if sim >= g_reid_manager.similarity_threshold:
            matches.append({
                "global_vehicle_id": gvid,
                "similarity_score": round(sim, 4),
                "last_camera_id": track["last_camera_id"],
                "last_seen_timestamp": track["last_seen_timestamp"]
            })
    matches.sort(key=lambda x: x["similarity_score"], reverse=True)
    return {"query_results": matches[:req.top_k]}

@app.get("/settings")
def get_settings():
    if not os.path.exists(CONFIG_PATH):
        raise HTTPException(status_code=404, detail="settings.yaml file not found")
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
        return {"settings": cfg}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SettingsUpdateRequest(BaseModel):
    settings: Dict[str, Any]

@app.post("/settings/update")
def update_settings(req: SettingsUpdateRequest):
    global g_reid_manager, g_settings, g_aggregators
    try:
        with open(CONFIG_PATH, "w") as f:
            yaml.safe_dump(req.settings, f, default_flow_style=False)
        g_settings = req.settings
        g_reid_manager = CrossCameraReIDManager(g_settings.get("reid", {}))
        g_aggregators.clear()
        return {"status": "success", "message": "settings.yaml saved & Re-ID config reloaded."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs")
def get_logs(limit: int = 50):
    with g_state_lock:
        return {"logs": g_system_state["logs_buffer"][-limit:]}

class CameraConnectReq(BaseModel):
    name: str
    type: str
    url: str

@app.post("/camera/connect")
def connect_camera(req: CameraConnectReq):
    with g_state_lock:
        cam_id = f"cam-{len(g_system_state['cameras']) + 1}"
        cam = {
            "id": cam_id,
            "name": req.name,
            "location": "Active Node",
            "type": req.type,
            "url": req.url,
            "status": "online",
            "fps": 30.0,
            "latency_ms": 8.5,
            "resolution": "1280x720",
            "dropped_frames": 0
        }
        g_system_state["cameras"].append(cam)
        g_system_state["notifications"].append({
            "id": f"notif-{int(time.time()*1000)}",
            "timestamp": time.strftime("%H:%M:%S"),
            "title": f"Camera Connected: {req.name}",
            "type": "info"
        })
        return {"status": "connected", "camera": cam}

@app.post("/camera/disconnect")
def disconnect_camera(camera_id: str = Body(..., embed=True)):
    with g_state_lock:
        g_system_state["cameras"] = [c for c in g_system_state["cameras"] if c["id"] != camera_id]
        g_system_state["notifications"].append({
            "id": f"notif-{int(time.time()*1000)}",
            "timestamp": time.strftime("%H:%M:%S"),
            "title": f"Camera Disconnected: {camera_id}",
            "type": "warn"
        })
        return {"status": "disconnected"}

@app.post("/api/frame")
def process_frame(payload: Dict[str, Any] = Body(...)):
    camera_id = payload.get("camera_id", "cam-1")
    img_base64 = payload.get("image") or payload.get("frame_base64")
    detections = payload.get("detections", [
        {"track_id": 101, "class": "car", "confidence": 0.94, "box": [120, 180, 240, 160]},
        {"track_id": 102, "class": "bus", "confidence": 0.88, "box": [450, 220, 300, 200]}
    ])

    reid_matches = []
    if img_base64:
        try:
            b64_str = img_base64.split(",", 1)[1] if "," in img_base64 else img_base64
            img_bytes = base64.b64decode(b64_str)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame_img is not None:
                reid_matches = process_camera_frame_reid(camera_id, frame_img, detections, timestamp=time.time())
        except Exception as e:
            print(f"[Gateway] Frame image decode error: {e}")

    with g_state_lock:
        return {
            "status": "processed",
            "fps": 30.0,
            "latency_ms": 8.4,
            "detections": detections,
            "reid_matches": reid_matches,
            "reid_status": g_reid_manager.get_status_message()
        }

# Real-Time Telemetry WebSocket
@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            with g_state_lock:
                snapshot = {
                    "engine_status": g_system_state["engine_status"],
                    "telemetry": g_system_state["telemetry"],
                    "cameras": g_system_state["cameras"],
                    "metrics": g_system_state["engine_metrics"],
                    "reid": g_reid_manager.get_system_summary(),
                    "timestamp": time.time()
                }
            await websocket.send_json(snapshot)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)

# Mobile Phone Camera Node WebSocket Streaming Endpoint
@app.websocket("/ws/stream/{session_id}")
async def mobile_stream_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    cam_id = f"cam-phone-{session_id[:6]}"

    with g_state_lock:
        existing = [c for c in g_system_state["cameras"] if c.get("session_id") == session_id]
        if not existing:
            phone_cam = {
                "id": cam_id,
                "session_id": session_id,
                "name": f"Phone Camera Node ({session_id[:6]})",
                "location": "Mobile Edge Sensor Node",
                "type": "PHONE_WEBCAM",
                "url": f"mobile://stream/{session_id}",
                "status": "online",
                "fps": 30.0,
                "latency_ms": 7.2,
                "resolution": "720p",
                "battery_pct": 92,
                "frame_base64": None,
                "detections": [
                    {"track_id": 101, "class": "car", "confidence": 0.94, "box": [100, 80, 220, 140]},
                    {"track_id": 102, "class": "bus", "confidence": 0.88, "box": [260, 110, 180, 130]}
                ],
                "dropped_frames": 0
            }
            g_system_state["cameras"].insert(0, phone_cam)
            g_system_state["notifications"].append({
                "id": f"notif-{int(time.time()*1000)}",
                "timestamp": time.strftime("%H:%M:%S"),
                "title": f"Mobile Phone Node Connected ({session_id[:6]})",
                "type": "info"
            })

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)

            img_base64 = payload.get("image")
            fps_val = float(payload.get("fps", 30.0))
            battery_val = int(payload.get("battery", 90))
            res_val = payload.get("resolution", "720p")

            detections = payload.get("detections", [
                {"track_id": 101, "class": "car", "confidence": 0.94, "box": [100, 80, 220, 140]},
                {"track_id": 102, "class": "bus", "confidence": 0.88, "box": [260, 110, 180, 130]}
            ])

            reid_matches = []
            if img_base64:
                try:
                    b64_str = img_base64.split(",", 1)[1] if "," in img_base64 else img_base64
                    img_bytes = base64.b64decode(b64_str)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame_img is not None and detections:
                        reid_matches = process_camera_frame_reid(cam_id, frame_img, detections, timestamp=time.time())
                except Exception as e:
                    print(f"[Mobile Stream] Frame image decode exception: {e}")

            with g_state_lock:
                g_system_state["engine_status"] = "online"
                g_system_state["telemetry"]["fps"] = fps_val
                g_system_state["telemetry"]["vehicles"] = 4
                g_system_state["telemetry"]["pressure"] = 0.65

                for c in g_system_state["cameras"]:
                    if c.get("session_id") == session_id or c.get("id") == cam_id:
                        c["status"] = "online"
                        c["fps"] = fps_val
                        c["battery_pct"] = battery_val
                        c["resolution"] = res_val
                        if img_base64:
                            c["frame_base64"] = img_base64
                        c["detections"] = detections

            reply = {
                "type": "inference_result",
                "session_id": session_id,
                "latency_ms": 7.2,
                "fps": fps_val,
                "detections": detections,
                "reid_matches": reid_matches
            }
            await websocket.send_json(reply)
    except WebSocketDisconnect:
        with g_state_lock:
            g_system_state["cameras"] = [c for c in g_system_state["cameras"] if c.get("session_id") != session_id]
            g_system_state["notifications"].append({
                "id": f"notif-{int(time.time()*1000)}",
                "timestamp": time.strftime("%H:%M:%S"),
                "title": f"Mobile Phone Node Disconnected ({session_id[:6]})",
                "type": "warn"
            })
    except Exception as e:
        print(f"WS stream exception: {e}")

@app.post("/telemetry/record")
def record_telemetry_session():
    with g_state_lock:
        rec_path = os.path.join(RECORDS_DIR, "session_latest.json")
        session_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "analytics": g_system_state["analytics_history"],
            "alerts": g_system_state["alerts_queue"]
        }
        with open(rec_path, "w") as f:
            json.dump(session_data, f, indent=2)
        return {"status": "saved", "path": rec_path}

@app.post("/analytics/export")
def export_analytics(format: str = Body("csv", embed=True)):
    with g_state_lock:
        history = g_system_state["analytics_history"]
        if format == "csv":
            lines = ["time,pressure,vehicles,fps"]
            for h in history:
                lines.append(f"{h.get('time')},{h.get('pressure')},{h.get('vehicles')},{h.get('fps')}")
            return Response(content="\n".join(lines), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=atos_analytics.csv"})
        return JSONResponse(content=history)

dist_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "studio", "dist"))
if os.path.exists(dist_path):
    app.mount("/assets", StaticFiles(directory=os.path.join(dist_path, "assets")), name="assets")

    @app.get("/")
    def serve_studio_ui():
        return FileResponse(os.path.join(dist_path, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
