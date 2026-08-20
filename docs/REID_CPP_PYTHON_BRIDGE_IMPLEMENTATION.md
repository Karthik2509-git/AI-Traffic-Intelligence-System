# 🚀 ATOS v3.5 C++ to Python Track Telemetry & Vehicle Crop Bridge Implementation

**Subsystem:** ATOS v3.5 C++ → Python Track Telemetry & Crop Transport Bridge  
**Phase:** Phase 1 (Track Metadata UDP Bridge) & Phase 2 (TCP Vehicle Crop Transport Stream)  
**Status:** Integrated & Validated (58/58 Unit Tests Passing)  
**Production Re-ID Default Mode:** `reid.enabled: false` (Safe Default Fallback Active)  

---

## 1. C++ Source of Truth

The C++ TensorRT inference engine and IoU tracking pipeline ([src/main.cpp](file:///c:/Users/KARTHIK%20V/OneDrive/Desktop/AI-Traffic-Intelligence-System/src/main.cpp) and [src/network/city_controller.cpp](file:///c:/Users/KARTHIK%20V/OneDrive/Desktop/AI-Traffic-Intelligence-System/src/network/city_controller.cpp)) serve as the sole authoritative source of truth for:
- Local vehicle tracking IDs (`traffic::Track::id`, starting at persistent ID `1000`)
- Vehicle class IDs (`traffic::Track::classId`, COCO indices `2=car`, `3=motorcycle`, `5=bus`, `7=truck`)
- Detection confidence scores (`traffic::Track::confidence`, `0.0` – `1.0`)
- Bounding box coordinates (`traffic::Track::bbox`, `[x, y, width, height]`)
- Raw BGR camera image frames (`pFrame->frame`)

Zero synthetic track IDs (`101`, `102`) or recalculated track IDs are generated for C++ telemetry or crop processing.

---

## 2. Phase 1 Track Telemetry JSON Schema (UDP 5005)

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

## 3. Phase 2 Vehicle Crop Binary Protocol Schema (TCP 5006)

Each crop package transmitted over TCP loopback (`127.0.0.1:5006`) adheres to a 12-byte length-prefixed header layout:

```
[Header: 12 Bytes]
  - Magic Bytes: 4 Bytes ("CROP")
  - Metadata Length N: 4 Bytes (uint32_be)
  - Image Length M: 4 Bytes (uint32_be)

[N-Byte JSON Metadata]
  {"type": "vehicle_crop", "camera_id": "cam-1", "frame_index": 1234,
   "timestamp": 1724174000.123, "track_id": 1001, "class_id": 2,
   "confidence": 0.94, "bbox": [120, 180, 240, 160]}

[M-Byte Binary JPEG Buffer]
  [FF D8 ... Raw JPEG Image Stream ... FF D9]
```

---

## 4. C++ → Python Data Flow Diagram

```
C++ Main Pipeline (src/main.cpp)
  │
  ├── TensorRT YOLOv8 FP16 Detection -> IoU Tracker (CityController)
  │
  ├── 1. Track Metadata Broadcast -> DigitalTwinBridge::syncTracks (UDP 5005)
  │
  └── 2. Crop Extraction -> CropBridgeSender::sendCrop (TCP 5006)
        │  ├── Quality Filter: bbox bounds, crop >= 32x32 px, conf >= 0.50
        │  └── JPEG Encode: cv::imencode(".jpg", crop_mat, jpeg_bytes, [IMWRITE_JPEG_QUALITY, 85])
        │
        ▼
Python Gateway Stream Listener (tools/web_gateway.py)
  │
  ├── 1. udp_telemetry_listener() receives UDP track telemetry (Port 5005)
  │
  └── 2. tcp_crop_listener() receives TCP crop packages (Port 5006)
        │  ├── Header Decode ("CROP", N, M) & Metadata Validation
        │  └── JPEG Buffer Decode -> OpenCV BGR numpy array
        │
        ▼
Production Re-ID Pipeline (Behind reid.enabled)
  │
  ├── VehicleKeyframeAggregator (Sampling 1 frame every 5 frames per track)
  ├── ONNXReIDFeatureExtractor (models/fastreid_sbs_r50_ibn_veri776.onnx -> 2048-D)
  └── CrossCameraReIDManager (Spatiotemporal GVID correlation)
```

---

## 5. Validation Rules & Failure Safety

1. **Schema & Header Validation:** `validate_and_decode_crop_package()` verifies `'CROP'` magic header, non-empty JPEG buffer, and valid metadata fields.
2. **Vehicle Class & Confidence Filtering:** Accepts only COCO vehicle classes `2` (car), `3` (motorcycle), `5` (bus), and `7` (truck) with confidence $\ge 0.50$.
3. **Dimension Requirements:** Crops smaller than $32\times 32$ pixels or with empty width/height are rejected before processing.
4. **Failure Closed Safety:**
   - Any TCP connection failure, JPEG decode error, or ONNX runtime exception is caught safely inside `try...except` wrappers.
   - Primary single-camera YOLOv8 + ByteTrack pipeline and mobile WebSocket streaming continue uninterrupted.
   - Zero fake GVIDs or synthetic track IDs (`101`, `102`) are ever generated.

---

## 6. Verification & Test Results

Executed command:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

**Results:**
- **58 / 58 tests passed (OK)** in 6.69 seconds.
- 26 dedicated integration tests in [tests/test_reid_track_telemetry.py](file:///c:/Users/KARTHIK%20V/OneDrive/Desktop/AI-Traffic-Intelligence-System/tests/test_reid_track_telemetry.py) and [tests/test_reid_crop_transport.py](file:///c:/Users/KARTHIK%20V/OneDrive/Desktop/AI-Traffic-Intelligence-System/tests/test_reid_crop_transport.py) covering valid packets, multi-tracks, vehicle class filtering, invalid fields, malformed JSON, JPEG encode/decode, 2048-D embedding validation, and L2 normalization ($\|v\|_2 \approx 1.0$).

Frontend Vite Build:
```bash
cd studio && npm run build
```
- **Passed** with 0 errors.

---

## 7. Status & Tier Matrix Update

```
VeRi-776 Benchmark: Rank-1 = 88.08% | Rank-5 = 93.92% | mAP = 70.38% (PASS)
C++ Track Telemetry Bridge: PASS
C++ Vehicle Crop Transport Stream: PASS
Production Re-ID Mode: reid.enabled = false (Default Safe Fallback Active)
Tier 8 Status: PENDING — Real Two-Camera Field Test
End-to-End Latency: NOT MEASURED
```
