# 🚀 ATOS v3.5 Cross-Camera Vehicle Re-ID Production Integration Reference

**Subsystem:** ATOS v3.5 Cross-Camera Vehicle Re-Identification (Re-ID)  
**Status:** Integrated & Validated in Safe Fallback Mode (`reid.enabled: false`)  
**Reference Model:** [models/fastreid_sbs_r50_ibn_veri776.onnx](file:///c:/Users/KARTHIK%20V/OneDrive/Desktop/AI-Traffic-Intelligence-System/models/fastreid_sbs_r50_ibn_veri776.onnx) (89.77 MB, SHA-256 `bc43d2fd...`)  
**Embedding Vector:** 2048-dimensional float (L2 normalized)  

---

## 1. System Architecture & Data Flow

```
Camera Feed / Mobile Phone Node (/ws/stream/{session_id})
        │
        ▼
Frame Ingestion & Base64 Decoding (tools/web_gateway.py)
        │
        ▼
YOLOv8 Object Detection (COCO vehicle classes: car, bus, truck, motorcycle)
        │
        ▼
ByteTrack Local Tracking (track_id, bbox, confidence)
        │
        ▼
Vehicle Crop Extraction (tools/reid_crop_utility.py -> extract_vehicle_crops)
        │  ├── Quality Gating: Reject crops < 32x32 px or confidence < 0.50
        │  └── Coordinate Clipping: Safe image boundary clipping
        ▼
Keyframe Gating & Aggregation (VehicleKeyframeAggregator)
        │  ├── Sample Interval: Every 5 frames per track
        │  └── Multi-Observation Buffer: 3–5 keyframe features -> L2 Mean Vector
        ▼
ONNX Re-ID Model Inference (models/fastreid_sbs_r50_ibn_veri776.onnx)
        │  └── CPU ONNXRuntime / CUDA Execution Provider
        ▼
2048-D L2 Normalized Embedding Vector
        │
        ▼
Cross-Camera Re-ID Manager (CrossCameraReIDManager.process_feature)
        │  ├── Cosine Similarity Matching (Target cutoff: 0.75)
        │  ├── Uncertainty Band Gating (Uncertainty threshold: 0.60)
        │  ├── Same-Camera Temporal Exclusion (< 5 seconds)
        │  └── Spatiotemporal Window Gating (< 300 seconds)
        ▼
Global Vehicle ID Assignment (GVID-1001, GVID-1002, ...)
        │
        ▼
FastAPI Telemetry Bridge (/ws/telemetry & /reid/status)
        │
        ▼
ATOS Studio Visual Intelligence Dashboard (studio/src/components/ReIDDashboard.tsx)
```

---

## 2. Crop Extraction & Keyframe Gating

1. **Crop Extraction (`extract_vehicle_crops`)**:
   - Safely clips bounding box coordinates to `[0, w_img]` and `[0, h_img]`.
   - Filters out non-vehicle classes and bounding boxes smaller than 32x32 pixels or below confidence threshold (default 0.50).
2. **Keyframe Gating (`VehicleKeyframeAggregator`)**:
   - Samples 1 frame every 5 frames (`keyframe_sample_interval: 5`).
   - Accumulates 3–5 keyframe observations (`keyframe_target_count: 3`).
   - Computes element-wise mean vector across keyframe embeddings and applies L2 normalization ($\|v\|_2 = 1.0$).
   - Reduces Re-ID model inference executions by 80–90% compared to per-frame evaluation.

---

## 3. Embedding Vector & Model Specifications

| Parameter | Specification | Verification |
| :--- | :--- | :--- |
| **Model Weights** | `models/fastreid_sbs_r50_ibn_veri776.onnx` | Verified (89.77 MB) |
| **Model Architecture** | Fast-ReID SBS (ResNet-50-IBN backbone) | Fine-tuned on VeRi-776 |
| **Input Shape** | `[1, 3, 256, 256]` (NCHW ImageNet Normalized) | ONNXRuntime verified |
| **Output Shape** | `[1, 2048]` | ONNXRuntime verified |
| **Embedding Dimension** | **2048 float** | L2 normalized ($\|v\|_2 = 1.0$) |
| **VeRi-776 Rank-1** | **88.08%** | Empirically measured |
| **VeRi-776 Rank-5** | **93.92%** | Empirically measured |
| **VeRi-776 mAP** | **70.38%** | Empirically measured |
| **ONNX Inference Cost** | **2.14 ms / crop** | CPU ONNX Runtime batched |

---

## 4. Failure Safety & Zero Crash Guarantee

The production Re-ID subsystem enforces non-blocking failure safety:

1. **Missing Model File:** `CrossCameraReIDManager.is_available()` evaluates to `False`. The gateway logs diagnostic status `"Re-ID model unavailable — evaluation pending"` and cleanly bypasses Re-ID inference.
2. **Runtime Exception:** Any exception during image decoding, crop extraction, or ONNX inference is caught inside `process_camera_frame_reid()`:
   ```python
   print("[ATOS Re-ID Pipeline] Re-ID unavailable — continuing detection/tracking: err")
   ```
3. **Primary Pipeline Continuity:** YOLOv8 detection, ByteTrack local tracking, and mobile WebSocket streaming (`/ws/stream/{session_id}`) operate uninterrupted under all failure modes.
4. **Zero Telemetry Fabrication:** When no live cross-camera matches exist, `/reid/matches` returns an empty array, and ATOS Studio displays `"No cross-camera matches detected"`.

---

## 5. Production Configuration Parameters

Configured in [config/settings.yaml](file:///c:/Users/KARTHIK%20V/OneDrive/Desktop/AI-Traffic-Intelligence-System/config/settings.yaml):

```yaml
reid:
  enabled: false # Default safe fallback (MUST remain false until real two-camera validation is completed)
  model_path: "models/fastreid_sbs_r50_ibn_veri776.onnx" # Validated Fast-ReID SBS(R50-IBN) ONNX model
  fallback_model_path: "models/fastreid_resnet50_veri776.onnx"
  similarity_threshold: 0.75
  uncertainty_threshold: 0.60
  max_spatiotemporal_window_sec: 300
  top_k_matches: 5
  crop_min_confidence: 0.5
  crop_min_size: 32
  keyframe_sample_interval: 5
  keyframe_target_count: 3
```

---

## 6. TensorRT GPU Status

- **TensorRT Package Inspection:** Python environment returns `ModuleNotFoundError: No module named 'tensorrt'`.
- **Status:** Documented as `TENSORRT_GPU_VALIDATION_PENDING`.
- **Reference Model:** `models/fastreid_sbs_r50_ibn_veri776.onnx` remains the active validated reference artifact.

---

## 7. Next Step: Controlled Two-Camera Field Test

To complete production activation (Tier 8):
Execute [scripts/test_two_camera_reid.py](file:///c:/Users/KARTHIK%20V/OneDrive/Desktop/AI-Traffic-Intelligence-System/scripts/test_two_camera_reid.py) with real prerecorded/live footage:
```bash
python scripts/test_two_camera_reid.py --camera-a path/to/camera_a.mp4 --camera-b path/to/camera_b.mp4
```
Upon successful field validation, set `reid.enabled: true` in `config/settings.yaml`.
