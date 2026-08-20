# 🚀 ATOS v3.5 C++ to Python Track Telemetry Bridge Implementation (Phase 1)

**Subsystem:** ATOS v3.5 C++ → Python Track Telemetry Bridge  
**Phase:** Phase 1 (Metadata & Bounding-Box Track Telemetry Exposing)  
**Status:** Integrated & Validated (42/42 Unit Tests Passing)  
**Production Re-ID Mode:** `reid.enabled: false` (Safe Default Fallback Active)  

---

## 1. C++ Source of Truth

The C++ TensorRT inference engine and IoU tracking pipeline ([src/main.cpp](file:///c:/Users/KARTHIK%20V/OneDrive/Desktop/AI-Traffic-Intelligence-System/src/main.cpp) and [src/network/city_controller.cpp](file:///c:/Users/KARTHIK%20V/OneDrive/Desktop/AI-Traffic-Intelligence-System/src/network/city_controller.cpp)) serve as the sole authoritative source of truth for:
- Local vehicle tracking IDs (`traffic::Track::id`, starting at persistent ID `1000`)
- Vehicle class IDs (`traffic::Track::classId`, COCO indices `2=car`, `3=motorcycle`, `5=bus`, `7=truck`)
- Detection confidence scores (`traffic::Track::confidence`, `0.0` – `1.0`)
- Bounding box coordinates (`traffic::Track::bbox`, `[x, y, width, height]`)

Zero synthetic track IDs (`101`, `102`) or recalculated track IDs are generated for C++ telemetry.

---

## 2. Track Telemetry JSON Schema

```json
{
    "type": "track_telemetry",
    "camera_id": "cam-1",
    "frame_index": 1234,
    "timestamp": 1724174000.123,
    "tracks": [
        {
            "track_id": 1001,
            "class_id": 2,
            "confidence": 0.94,
            "bbox": [120, 180, 240, 160]
        },
        {
            "track_id": 1002,
            "class_id": 5,
            "confidence": 0.88,
            "bbox": [450, 220, 300, 200]
        }
    ]
}
```

---

## 3. UDP Transport & Architecture

- **Transport:** Non-blocking UDP Loopback (`127.0.0.1:5005`) via `DigitalTwinBridge::syncTracks`.
- **Backward Compatibility:** Preserves existing `"type": "city_pulse"` and `"type": "incident_alert"` semantics.
- **Serialization Method:** `DigitalTwinBridge::syncTracks(camera_id, frame_index, timestamp, tracks)` formats high-efficiency JSON output in C++ and broadcasts via `sendto()`.

---

## 4. C++ → Python Data Flow Diagram

```
C++ Pipeline (src/main.cpp)
  │
  ├── TensorRT YOLOv8 FP16 Detection -> IoU Tracker (CityController)
  │
  ├── g_cityController->getActiveTracks() (Thread-safe active tracks getter)
  │
  └── g_twinBridge->syncTracks("cam-1", pFrame->frameIndex, ts, tracks)
        │
        ▼
  UDP Socket (127.0.0.1:5005)
        │
        ▼
  Python Gateway (tools/web_gateway.py)
        │
        ├── udp_telemetry_listener() receives packet
        │
        ├── validate_and_parse_track_telemetry(payload)
        │     ├── Validates schema & field types
        │     └── Filters non-vehicle classes (retains 2, 3, 5, 7)
        │
        └── Updates g_system_state["real_track_telemetry"] & /telemetry/tracks REST endpoint
```

---

## 5. Validation Rules & Failure Safety

1. **Schema Validation:** `validate_and_parse_track_telemetry()` rejects packets missing `camera_id`, `frame_index`, `timestamp`, or `tracks` array.
2. **Vehicle Class Filtering:** Only COCO vehicle classes `2` (car), `3` (motorcycle), `5` (bus), and `7` (truck) are accepted. Non-vehicle tracks are safely dropped.
3. **Field Range Checks:**
   - `track_id`: Integer $\ge 0$
   - `confidence`: Float $0.0 \le c \le 1.0$
   - `bbox`: List of 4 non-negative integers $[x, y, w, h]$
4. **Malformed Packet Handling:** Rejects invalid JSON or corrupt schemas safely without crashing `web_gateway.py`.
5. **UDP Timeout & Retain Behavior:** If C++ stream pauses ($> 4.0$ s), `engine_status` switches to `"waiting_for_engine"`. No synthetic tracks or fallback IDs are ever fabricated.

---

## 6. Verification & Test Results

Executed command:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

**Results:**
- **42 / 42 tests passed (OK)** in 5.18 seconds.
- 10 new dedicated Phase 1 integration tests in [tests/test_reid_track_telemetry.py](file:///c:/Users/KARTHIK%20V/OneDrive/Desktop/AI-Traffic-Intelligence-System/tests/test_reid_track_telemetry.py) covering valid packets, multi-tracks, vehicle class filtering, invalid fields, malformed JSON, and zero synthetic track injection.

Frontend Vite Build:
```bash
cd studio && npm run build
```
- **Passed** with 0 errors.

---

## 7. Known Limitation & Current Status

> [!IMPORTANT]
> **Phase 1 Limitation:** Track telemetry currently transmits bounding boxes and metadata only. Image/crop transport is **NOT** implemented in Phase 1.

```
END-TO-END PRODUCTION RE-ID: NOT YET ACTIVE
reid.enabled: false (UNCHANGED)
```
