# 🚗 ATOS v3.5 Cross-Camera Vehicle Re-Identification (Re-ID) Architecture & Integration Guide

## 📋 Overview

ATOS v3.5 introduces an optional, production-grade Cross-Camera Vehicle Re-Identification (Re-ID) subsystem designed to track vehicle identities across multiple distinct surveillance camera feeds.

The Re-ID subsystem operates as a secondary feature extraction layer on top of the primary YOLOv8 + ByteTrack single-camera detection pipeline.

---

## 🏗️ System Architecture & Data Contracts

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                             Single-Camera Node                             │
│  Camera Stream ──► YOLOv8 Detector (960x960 FP16) ──► ByteTrack (Local ID) │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                         [if reid_enabled == true]
                                      │
┌─────────────────────────────────────▼──────────────────────────────────────┐
│                       ReIDModelAdapter (C++ / Python)                      │
│   Crop BBox ──► Normalization (256x256) ──► Feature Extractor ──► L2 Vector│
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼──────────────────────────────────────┐
│                    CrossCameraReIDManager (Python Core)                    │
│   Spatiotemporal Window Gating ──► Cosine Similarity Matcher ──► Graph     │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼──────────────────────────────────────┐
│                    FastAPI Gateway & WebSocket Telemetry                   │
│   GET /reid/status • GET /reid/matches • GET /reid/graph • WS /ws/telemetry│
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼──────────────────────────────────────┐
│                          ATOS Studio UI (React)                            │
│     Re-ID Health Card • Cross-Camera Transition Graph • Identity Matcher   │
└────────────────────────────────────────────────────────────────────────────┘
```

### Data Contract Definitions

```cpp
// C++ Header Data Contract: include/reid/reid_types.hpp
namespace traffic {
struct ReIDFeature {
    std::string camera_id;
    int local_track_id;
    std::string global_vehicle_id;
    std::vector<float> embedding;   // Dynamically discovered dimension
    double timestamp;
    cv::Rect bbox;
    float match_confidence;
};
}
```

```json
// REST API Data Contract: GET /reid/status
{
  "reid_enabled": false,
  "model_loaded": false,
  "model_path": "models/reid_vehiclenet.engine",
  "status": "Re-ID model unavailable — evaluation pending",
  "active_global_tracks": 0,
  "total_matches_found": 0,
  "similarity_threshold": 0.75,
  "benchmark": {
    "status": "dataset_missing",
    "evaluated": false,
    "rank1": null,
    "rank5": null,
    "mAP": null,
    "false_match_rate": null,
    "false_non_match_rate": null,
    "inference_ms": null,
    "matching_ms": null,
    "vram_used_mb": null,
    "dataset_name": "veri776"
  }
}
```

---

## 📊 Dataset Specifications & Acquisition Guide

### 1. VeRi-776 (Primary Benchmark Target)
- **Identities**: 776 vehicles
- **Images**: 50,000+ cropped vehicle images
- **Cameras**: 20 surveillance cameras in unconstrained urban environment
- **Metadata**: Timestamps, camera IDs, vehicle type, color, license plate annotations
- **License**: Non-Commercial Academic Research License
- **Official URL**: [VeRi-776 Maintainers Page](https://vecam.github.io/VeRi/)
- **Setup Instructions**:
  1. Register and request dataset access at the official URL.
  2. Unpack `image_train`, `image_test`, and `image_query` into `datasets/reid/veri776/`.
  3. Verify directory contents using `python scripts/benchmark_reid.py`.

### 2. CityFlow-ReID (AI City Challenge)
- **Identities**: 666 vehicles
- **Images**: 229,147 images across 40 cameras
- **License**: NVIDIA AI City Challenge License
- **Official URL**: [AI City Challenge](https://www.aicitychallenge.org/)

---

## 🧠 Model Candidates & Input Specifications

| Model Candidate | Estimated Parameters | Target Resolution | Tensor Input | Embedding Normalization |
| :--- | :---: | :---: | :---: | :---: |
| **OSNet_x1_0 / MobileNetV3-ReID** | ~2.2M | $256 \times 256$ | `[1, 3, 256, 256]` | L2 Normalized ($\|e\|_2 = 1.0$) |
| **VehicleNet / ResNet50-IBN** | ~25.5M | $256 \times 256$ | `[1, 3, 256, 256]` | L2 Normalized ($\|e\|_2 = 1.0$) |

### Crop Image Preprocessing
1. Crop bounding box image: $I_{\text{crop}} = \text{Frame}[\text{ymin}:\text{ymax}, \text{xmin}:\text{xmax}]$.
2. Resize to $256 \times 256$ pixels.
3. Convert BGR to RGB channel order.
4. Scale pixels to $[0.0, 1.0]$.
5. Normalize using ImageNet statistics:
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
6. Run model forward pass and apply L2 norm: $\hat{e} = \frac{e}{\|e\|_2}$.

---

## ⚙️ Configuration & Safe Defaults

In `config/settings.yaml`:
```yaml
reid:
  enabled: false # Safe fallback default
  model_path: "models/reid_vehiclenet.engine"
  similarity_threshold: 0.75 # Configurable starting threshold
  embedding_dim: 512
  max_spatiotemporal_window_sec: 300 # Configurable 5-minute transition window
  top_k_matches: 5
```

---

## 🧪 Benchmark Runner Usage

To evaluate model accuracy on real dataset annotations:
```bash
python scripts/benchmark_reid.py --dataset veri776 --dataset-dir datasets/reid/veri776 --model models/reid_vehiclenet.onnx
```

Empirical results are output to `runs/reid_benchmark_results.json` and automatically reflected in ATOS Studio UI.
