# 🔬 ATOS v3.5 Re-ID Post-Integration Architecture Audit

**Date:** August 20, 2026  
**Subsystem:** ATOS v3.5 Cross-Camera Vehicle Re-Identification (Re-ID)  
**Status:** Audit Complete — Codebase Inspected  
**Verdict:** **`ARCHITECTURE STATUS: INTEGRATION GAP DETECTED`**

---

## 1. Actual Runtime Call Graph & Data Flow

```
[SOURCE A: C++ Engine]
Video Source (data/test_4k_traffic.mp4)
  │
  ▼
VideoCaptureEngine (C++ pinned memory)
  │
  ▼
Detector::process (TensorRT FP16 yolov8_4k_optimized.engine)
  │
  ▼
CityController::updateTracks (C++ tracking state)
  │
  ▼
DigitalTwinBridge::syncState (UDP Port 5005)
  │
  └─► Broadcasts: {"type":"city_pulse", "pressure":..., "signal_phase":..., "vehicles":...}
      [GAP: NO track_id, NO bbox, NO vehicle class, NO frame crops sent over UDP!]

─────────────────────────────────────────────────────────────────────────────

[SOURCE B: Mobile Camera Node / Web Client]
Phone Camera / Client Payload
  │
  ▼
/ws/stream/{session_id} or /api/frame (tools/web_gateway.py)
  │
  ├─► Decodes JPEG base64 payload to OpenCV BGR numpy array
  │
  ├─► Inspects payload.get("detections")
  │     │
  │     ├─► If present: uses client-supplied detections
  │     └─► If missing: FALLS BACK TO STATIC MOCK DICTIONARIES:
  │           [{"track_id": 101, "box": [100,80,220,140]}, {"track_id": 102, "box": [260,110,180,130]}]
  │           [GAP: No gateway-side YOLOv8 + ByteTrack execution!]
  │
  ▼
process_camera_frame_reid() (tools/web_gateway.py)
  │
  ▼
extract_vehicle_crops() -> VehicleKeyframeAggregator -> ONNXReIDFeatureExtractor -> CrossCameraReIDManager
```

---

## 2. Answers to Specific Audit Questions (A – M)

| # | Question | Verified Source Code Answer |
| :--- | :--- | :--- |
| **A** | **Where is YOLOv8 actually executed?** | Executed in **C++** ([src/main.cpp](file:///c:/Users/KARTHIK%20V/OneDrive/Desktop/AI-Traffic-Intelligence-System/src/main.cpp) using `antigravity::engine::Detector` on TensorRT FP16) for C++ pipeline; in `scripts/test_two_camera_reid.py` for offline 2-camera harness; and in `legacy/src/pipeline.py`. **NOT** executed in `tools/web_gateway.py`. |
| **B** | **Where is ByteTrack actually executed?** | Executed in `scripts/test_two_camera_reid.py` and `legacy/src/pipeline.py`. In C++, `src/network/city_controller.cpp` manages tracking. **NOT** executed in `tools/web_gateway.py`. |
| **C** | **Is YOLOv8 executed in Python `web_gateway.py`?** | **NO.** `web_gateway.py` does not import or execute YOLOv8. |
| **D** | **Is ByteTrack executed in Python `web_gateway.py`?** | **NO.** `web_gateway.py` does not import or execute ByteTrack. |
| **E** | **Is YOLOv8/ByteTrack executed by `src/main.cpp` or another C++ engine?** | **YES.** `src/main.cpp` executes TensorRT YOLOv8 (`antigravity::engine::Detector`) and C++ tracking (`CityController`). |
| **F** | **What data structure crosses the C++ → Python/gateway boundary?** | Only lightweight UDP JSON packets containing aggregate counters: `{"type":"city_pulse", "pressure": float, "signal_phase": int, "vehicles": int}`. **Zero bounding boxes, zero track IDs, and zero image crops cross UDP.** |
| **G** | **Does `process_camera_frame_reid()` receive real detections/tracks?** | **NO** (when called via gateway streaming without external detections). It receives hardcoded fallback dictionaries (`track_id: 101`, `track_id: 102`). |
| **H** | **Are mock/synthetic detections still used by the live path?** | **YES.** `tools/web_gateway.py` lines 510-513 and 591-594 fall back to static hardcoded track dictionaries when client JSON payloads omit detections. |
| **I** | **Is YOLOv8 executed twice for the same frame?** | **NO.** `web_gateway.py` does not execute YOLOv8 at all. |
| **J** | **Is ByteTrack executed twice for the same frame?** | **NO.** ByteTrack is not executed in `web_gateway.py`. |
| **K** | **Is Re-ID connected to authoritative production track IDs?** | **NO.** C++ track IDs stay inside C++ memory and are not sent over UDP to `web_gateway.py`. |
| **L** | **Does mobile camera path reach the C++ pipeline?** | **NO.** Mobile streams land in Python `web_gateway.py` over WebSockets and are not routed to C++ or processed by Python YOLOv8. |
| **M** | **Does enabling `reid.enabled` process real production detections?** | **NO.** It processes gateway-local detections (falling back to static synthetic track dictionaries if client payloads omit detections). |

---

## 3. Performance Accounting Audit

- **Reported Numbers:**
  - C++ TensorRT YOLOv8 + ByteTrack: `8.40 ms` (148 FPS)
  - ONNX CPU Re-ID Model Inference: `2.14 ms / crop` (Batched 64)
- **Combination Check:**
  - `8.40 ms + 2.14 ms` is **NOT** a valid combined end-to-end latency estimate because the two stages currently run in separate, disconnected processes on different data streams.
- **Empirical Status:**
  ```
  END_TO_END_REID_LATENCY = NOT MEASURED
  ```

---

## 4. Discovered Architectural Problems & Integration Gaps

1. **Telemetry Serialization Deficit (C++ → Python):**
   - `DigitalTwinBridge::syncState` in [src/simulation/digital_twin.cpp](file:///c:/Users/KARTHIK%20V/OneDrive/Desktop/AI-Traffic-Intelligence-System/src/simulation/digital_twin.cpp) only serializes `city_pressure`, `signal_phase`, and `vehicle_count`.
   - It does not serialize tracked vehicle bounding boxes (`[x, y, w, h]`), `track_id`, or `class_id`.
2. **Mobile Gateway Detector Absence:**
   - [tools/web_gateway.py](file:///c:/Users/KARTHIK%20V/OneDrive/Desktop/AI-Traffic-Intelligence-System/tools/web_gateway.py) decodes incoming base64 frame images from `/ws/stream/{session_id}`, but does not run a detector/tracker (such as `ultralytics.YOLO` or C++ engine bindings) on the decoded image, falling back to static dummy detections (`track_id: 101`, `track_id: 102`).
3. **Safety for Real Two-Camera Testing:**
   - The standalone validation harness ([scripts/test_two_camera_reid.py](file:///c:/Users/KARTHIK%20V/OneDrive/Desktop/AI-Traffic-Intelligence-System/scripts/test_two_camera_reid.py)) **IS SAFE** for controlled offline two-camera testing because it directly invokes `ultralytics.YOLO` + ByteTrack on video files.
   - However, the **live web gateway (`web_gateway.py`) IS NOT YET READY** for real two-camera live streaming tests until either:
     a) The C++ engine serializes tracked vehicle bounding boxes and frame crops over UDP/IPC to `web_gateway.py`, OR
     b) `web_gateway.py` initializes a gateway-side detector/tracker for incoming mobile camera stream frames.

---

## 5. Concise Verdict

```
ARCHITECTURE STATUS: INTEGRATION GAP DETECTED
```
