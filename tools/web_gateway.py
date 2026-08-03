#!/usr/bin/env python3
"""
ATOS Studio Web Gateway & Real-Time Telemetry Server
Bridges the ATOS C++ Engine UDP telemetry stream to a modern browser-based web dashboard.

Usage:
    python tools/web_gateway.py --port 8080 --udp-port 5005
"""

import argparse
import json
import socket
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# Global state for latest telemetry snapshot
g_telemetry_state = {
    "status": "online",
    "last_update": time.time(),
    "pressure": 0.0,
    "signal_phase": 0,
    "vehicles": 0,
    "fps": 45.0,
    "latency_ms": 12.5,
    "alerts": []
}
g_lock = threading.Lock()

HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ATOS Studio — Real-Time Visual Intelligence Operating System</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #07090e;
            --card-bg: #0f131d;
            --border-color: #1e2638;
            --accent-cyan: #00f2fe;
            --accent-green: #00ff9d;
            --accent-orange: #ff9f43;
            --accent-red: #ff5252;
            --text-main: #f1f5f9;
            --text-muted: #64748b;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            height: 100vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        /* Top Navigation Header */
        header {
            height: 56px;
            background: #0b0f17;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 24px;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
            font-weight: 700;
            font-size: 1.1rem;
            letter-spacing: 0.5px;
        }

        .logo-badge {
            background: linear-gradient(135deg, #00f2fe, #4facfe);
            color: #000;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
        }

        .nav-status {
            display: flex;
            align-items: center;
            gap: 16px;
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-green);
            box-shadow: 0 0 10px var(--accent-green);
            display: inline-block;
        }

        /* Main Studio Layout */
        .studio-container {
            flex: 1;
            display: grid;
            grid-template-columns: 240px 1fr 340px;
            height: calc(100vh - 56px);
        }

        /* Left Navigation Sidebar */
        aside.left-sidebar {
            background: #0b0f19;
            border-right: 1px solid var(--border-color);
            padding: 16px 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .nav-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .nav-item:hover, .nav-item.active {
            background: #151c2c;
            color: var(--text-main);
        }

        .nav-item.active {
            border-left: 3px solid var(--accent-cyan);
        }

        /* Center Video & Intelligence Grid */
        main.viewport {
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 20px;
            overflow-y: auto;
        }

        .video-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            position: relative;
            aspect-ratio: 16/9;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .video-placeholder {
            text-align: center;
            color: var(--text-muted);
        }

        .hud-overlay {
            position: absolute;
            top: 16px;
            left: 16px;
            background: rgba(15, 19, 29, 0.85);
            backdrop-filter: blur(8px);
            padding: 8px 14px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            display: flex;
            gap: 16px;
        }

        .hud-val { color: var(--accent-cyan); font-weight: 700; }

        /* Metrics Summary Bar */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
        }

        .metric-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            padding: 18px;
            border-radius: 10px;
        }

        .metric-label {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Right Insights Panel */
        aside.right-panel {
            background: #0b0f19;
            border-left: 1px solid var(--border-color);
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .panel-title {
            font-size: 0.95rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
        }

        .alert-box {
            background: #1a1215;
            border: 1px solid #3d1c24;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 0.85rem;
            color: var(--accent-red);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .explainable-box {
            background: #111827;
            border: 1px solid #1f2937;
            padding: 14px;
            border-radius: 8px;
            font-size: 0.82rem;
            line-height: 1.5;
            color: #9ca3af;
        }
    </style>
</head>
<body>
    <header>
        <div class="logo">
            <span>ATOS STUDIO</span>
            <span class="logo-badge">ENGINE v3.1</span>
        </div>
        <div class="nav-status">
            <span><span class="status-dot"></span> Engine Online</span>
            <span>|</span>
            <span id="header-time">00:00:00</span>
        </div>
    </header>

    <div class="studio-container">
        <!-- Left Sidebar Navigation -->
        <aside class="left-sidebar">
            <div class="nav-item active">🎥 Live Video Grid</div>
            <div class="nav-item">📊 Traffic Analytics</div>
            <div class="nav-item">🚦 Signal Optimizer</div>
            <div class="nav-item">🔌 Plugin Store</div>
            <div class="nav-item">⚡ Automation (n8n)</div>
            <div class="nav-item">🏙️ 3D Digital Twin</div>
            <div class="nav-item">⚙️ Engine Settings</div>
        </aside>

        <!-- Main Viewport -->
        <main class="viewport">
            <div class="video-card">
                <div class="hud-overlay">
                    <div>CAM-01: <span class="hud-val">Intersection-Alpha</span></div>
                    <div>FPS: <span class="hud-val" id="hud-fps">45.0</span></div>
                    <div>LATENCY: <span class="hud-val" id="hud-latency">12.5 ms</span></div>
                </div>
                <div class="video-placeholder">
                    <h3 style="margin-bottom: 8px; color: var(--text-main);">Live Stream Feed Active</h3>
                    <p>C++ / CUDA / TensorRT Real-Time Processing Pipeline</p>
                </div>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Active Vehicles</div>
                    <div class="metric-value" id="val-vehicles" style="color: var(--accent-cyan);">0</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Traffic Pressure</div>
                    <div class="metric-value" id="val-pressure" style="color: var(--accent-green);">0.00</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Signal Extension</div>
                    <div class="metric-value" id="val-signal" style="color: var(--accent-orange);">10s</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">System Health</div>
                    <div class="metric-value" style="color: var(--accent-green);">100%</div>
                </div>
            </div>
        </main>

        <!-- Right Insights Panel -->
        <aside class="right-panel">
            <div class="panel-title">AI Real-Time Insights</div>
            
            <div class="alert-box" id="alert-container">
                <div>⚠️ No Critical Incident Alerts</div>
            </div>

            <div class="panel-title">Explainable AI Formula</div>
            <div class="explainable-box">
                <strong>Traffic Pressure Index:</strong><br>
                <code>P = min(ActiveTracks / IntersectionCapacity, 1.0)</code><br><br>
                Evaluated in C++ Core Analytics using zero-copy CUDA memory. Phase 0 extension triggered automatically when P > 0.20.
            </div>
        </aside>
    </div>

    <script>
        function updateClock() {
            const now = new Date();
            document.getElementById('header-time').innerText = now.toTimeString().split(' ')[0];
        }
        setInterval(updateClock, 1000);
        updateClock();

        async function fetchTelemetry() {
            try {
                const res = await fetch('/api/telemetry');
                if (!res.ok) return;
                const data = await res.json();
                
                document.getElementById('val-vehicles').innerText = data.vehicles || 0;
                document.getElementById('val-pressure').innerText = (data.pressure || 0).toFixed(2);
                document.getElementById('val-signal').innerText = (data.signal_phase === 0 ? '10s' : '30s');
                document.getElementById('hud-fps').innerText = (data.fps || 45.0).toFixed(1);
                document.getElementById('hud-latency').innerText = (data.latency_ms || 12.5).toFixed(1) + ' ms';

                if (data.alerts && data.alerts.length > 0) {
                    document.getElementById('alert-container').innerText = '🚨 ' + data.alerts[data.alerts.length - 1];
                }
            } catch (e) {
                console.error("Telemetry fetch error:", e);
            }
        }
        setInterval(fetchTelemetry, 250);
    </script>
</body>
</html>
"""

class StudioRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/telemetry':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            with g_lock:
                snapshot = dict(g_telemetry_state)
            self.wfile.write(json.dumps(snapshot).encode('utf-8'))
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_DASHBOARD.encode('utf-8'))

def udp_receiver_thread(udp_port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", udp_port))
    print(f"[ATOS Studio Gateway] Listening for C++ Engine UDP telemetry on 127.0.0.1:{udp_port}")

    while True:
        try:
            data, _ = sock.recvfrom(2048)
            msg = data.decode('utf-8')
            payload = json.loads(msg)

            with g_lock:
                g_telemetry_state["last_update"] = time.time()
                if payload.get("type") == "city_pulse":
                    g_telemetry_state["pressure"] = payload.get("pressure", 0.0)
                    g_telemetry_state["signal_phase"] = payload.get("signal_phase", 0)
                    g_telemetry_state["vehicles"] = payload.get("vehicles", 0)
                elif payload.get("type") == "incident_alert":
                    alert_str = f"{payload.get('category')} at node {payload.get('node_id')}"
                    g_telemetry_state["alerts"].append(alert_str)
                    if len(g_telemetry_state["alerts"]) > 10:
                        g_telemetry_state["alerts"].pop(0)

        except Exception as e:
            time.sleep(0.01)

def main():
    parser = argparse.ArgumentParser(description="ATOS Studio Web Gateway")
    parser.add_argument("--port", type=int, default=8080, help="Web server port")
    parser.add_argument("--udp-port", type=int, default=5005, help="UDP telemetry port")
    args = parser.parse_args()

    # Launch UDP receiver thread
    t = threading.Thread(target=udp_receiver_thread, args=(args.udp_port,), daemon=True)
    t.start()

    server = HTTPServer(("0.0.0.0", args.port), StudioRequestHandler)
    print(f"[ATOS Studio Gateway] Web UI online at http://localhost:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ATOS Studio Gateway] Stopping Web Server.")
        server.server_close()

if __name__ == "__main__":
    main()
