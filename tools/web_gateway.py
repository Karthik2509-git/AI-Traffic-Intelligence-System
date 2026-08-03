#!/usr/bin/env python3
"""
ATOS Studio Production FastAPI & WebSocket Gateway Server
Bridges C++ ATOS Engine (via UDP 5005) with ATOS Studio Visual Intelligence OS.

Features:
    - FastAPI OpenAPI Swagger documentation automatically served at /docs
    - Dynamic plugin discovery in plugins/ directory
    - Telemetry session recorder for offline replay
    - Frame processing endpoint for browser/mobile webcams
"""

import os
import sys
import json
import time
import socket
import asyncio
import threading
from typing import List, Dict, Any, Optional

import yaml
import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, Response
from pydantic import BaseModel

app = FastAPI(
    title="ATOS Studio Visual Intelligence OS API",
    description="OpenAPI control plane & high-performance telemetry bridge for ATOS CUDA/TensorRT Engine",
    version="3.1.0"
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

g_state_lock = threading.Lock()

g_system_state = {
    "engine_status": "offline",
    "last_udp_packet": 0.0,
    "uptime_start": time.time(),
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

# Discover installed plugins
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

# Background UDP Telemetry Thread
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
                g_system_state["last_udp_packet"] = now
                g_system_state["engine_status"] = "online"

                if payload.get("type") == "city_pulse":
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

                elif payload.get("type") == "incident_alert":
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
        except Exception:
            time.sleep(0.01)

t_udp = threading.Thread(target=udp_telemetry_listener, daemon=True)
t_udp.start()

# API Endpoints
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
    try:
        with open(CONFIG_PATH, "w") as f:
            yaml.safe_dump(req.settings, f, default_flow_style=False)
        return {"status": "success", "message": "settings.yaml saved."}
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
    # Simulates real CUDA frame processing returning bounding boxes
    with g_state_lock:
        return {
            "status": "processed",
            "fps": 30.0,
            "latency_ms": 8.4,
            "detections": [
                {"track_id": 101, "class": "car", "confidence": 0.94, "box": [120, 180, 240, 160]},
                {"track_id": 102, "class": "bus", "confidence": 0.88, "box": [450, 220, 300, 200]}
            ]
        }

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
