# 📦 ATOS v3.5 Model & Dataset Readiness Report

**Subsystem:** ATOS v3.5 Cross-Camera Vehicle Re-Identification (Re-ID)  
**Phase:** Model & Dataset Selection & Verification  
**Repository Branch:** `main` | **Working Tree:** Clean  
**Safe Production Config:** `reid.enabled: false` (Active Fallback)

---

## 📊 Legitimate Vehicle Re-ID Model Candidates Evaluation

> [!NOTE]
> All parameter counts, embedding dimensions, inference costs, and accuracy figures in this evaluation matrix represent candidate/estimated values. Exact metrics will be empirically measured on host hardware (`scripts/check_reid_readiness.py` and `scripts/benchmark_reid.py`) once model weights are installed.

| Model Candidate | Architecture | Source / Maintainers | License | Input Shape | Output Vector | Est. Params | ONNX / TRT Compatible | Deployment Complexity |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **OSNet_x1_0 / MobileNetV3-ReID** *(Recommended Primary)* | Omni-Scale Feature Network / MobileNetV3 | TorchReID / Academic | MIT / Apache 2.0 | `[1, 3, 256, 256]` | `[1, 512]` | ~2.2M | **Yes** (Native ONNX/TensorRT FP16) | **Low** (Ideal for real-time edge execution) |
| **VehicleNet / ResNet50-IBN** *(Recommended Fallback)* | ResNet50 with Instance-Batch Normalization | VehicleNet / Academic | Non-Commercial Research | `[1, 3, 256, 256]` | `[1, 512]` / `[1, 2048]` | ~25.5M | **Yes** (ONNX export supported) | **Moderate** (Higher VRAM & latency overhead) |
| **TransReID / ViT-Base** | Vision Transformer (ViT) | TransReID / Academic | Apache 2.0 | `[1, 3, 256, 256]` | `[1, 768]` | ~86.0M | **Conditional** (Requires static shape ONNX) | **High** (Large memory footprint) |

---

## 🎯 Model Recommendations & Hardware Suitability Justification

### Primary Model Recommendation: **OSNet_x1_0 / MobileNetV3-ReID**
1. **Real-Time Edge Throughput**: Featuring a lightweight ~2.2M parameter footprint, OSNet enables sub-2ms FP16 inference per vehicle crop on NVIDIA RTX GPUs (such as RTX 5050 / RTX 4090 / Jetson edge devices).
2. **Multi-Stream Multi-Camera Scalability**: Low VRAM usage allows extracting feature embeddings from dozens of active vehicle crops per second across multiple concurrent camera feeds without stalling the main YOLOv8 960x960 FP16 detection pipeline.
3. **Open-Source License Compliance**: Released under standard permissive licenses (MIT / Apache 2.0).

### Fallback Model Recommendation: **VehicleNet / ResNet50-IBN**
1. **High Feature Discriminative Power**: Higher capacity (~25.5M parameters) provides enhanced feature representation for distinguishing subtle vehicle attribute differences.
2. **Standard ResNet ONNX Export**: Well-tested PyTorch-to-ONNX export pathway supported by TensorRT.

---

## 📋 VeRi-776 Dataset Acquisition & Structure Checklist

### Manual Dataset Acquisition Checklist

- [ ] **Official Dataset Registration**: Register and request access at the [Official VeRi-776 Homepage](https://vecam.github.io/VeRi/).
- [ ] **License Agreement Review**: Verify compliance with non-commercial academic research licensing terms.
- [ ] **Download Image Archives**: Obtain `image_train.zip`, `image_test.zip`, `image_query.zip`, and track list text files.
- [ ] **Target File Unpacking**: Unpack archives into `datasets/reid/veri776/` following the expected layout below.

### Expected Directory Layout (`scripts/benchmark_reid.py`)

```text
datasets/reid/veri776/
├── image_query/       # Query probe vehicle images (e.g. 0001_c001_00026030_0.jpg)
├── image_test/        # Gallery search pool images (e.g. 0001_c002_00026100_0.jpg)
├── image_train/       # Model training set images
├── name_query.txt     # Query list annotations
├── name_test.txt      # Gallery list annotations
└── name_train.txt     # Training list annotations
```

---

## 🔍 Validation Commands & Automated Integrity Checks

### Command 1: Run Model & Dataset Integrity Inspector
```bash
python scripts/check_reid_readiness.py --model models/reid_vehiclenet.onnx --dataset-dir datasets/reid/veri776
```
- **Expected Result (Current State)**:
  ```text
  --- MODEL INTEGRITY STATUS ---
  Present : False (MODEL_FILE_MISSING)
  --- DATASET INTEGRITY STATUS ---
  Present : False (DATASET_FILES_MISSING)
  ```

### Command 2: Run Re-ID Evaluation Harness
```bash
python scripts/benchmark_reid.py --dataset veri776 --dataset-dir datasets/reid/veri776 --model models/reid_vehiclenet.onnx
```
- **Expected Result (Current State)**:
  ```text
  [NOTICE] Dataset files not found at: .../datasets/reid/veri776
  [STATUS] Benchmark results written to runs/reid_benchmark_results.json (Status: dataset_missing).
  ```

### Command 3: Run Subsystem Unit Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```
- **Expected Result**: `Ran 4 tests ... OK`.

---

## 🏁 Current Status & Remaining Blockers

```text
[PASS] Tier 1: Implementation Complete (C++, Python, REST/WS, UI)
[PASS] Tier 2: Unit Tests Passing (4/4 tests OK)
[PASS] Tier 3: Integration Validated (reid_enabled: false safe fallback active)
[PENDING] Tier 4: Model Loaded (Requires manual placement of models/reid_vehiclenet.onnx)
[PENDING] Tier 5: Dataset Prepared (Requires manual extraction to datasets/reid/veri776/)
[PENDING] Tier 6: Benchmark Executed (Pending dataset & model files)
[PENDING] Tier 7: Accuracy Validated (Rank-1, Rank-5, mAP pending empirical run)
[PENDING] Tier 8: Real Two-Camera Field Test
[PASS] Tier 9: Performance Validated (Baseline YOLOv8 + ByteTrack at 148 FPS)
[SAFE FALLBACK ACTIVE] Tier 10: Production Ready (Default reid_enabled: false active)
```

**Next Action Required by User**: Obtain dataset access for VeRi-776 and place model weights in `models/reid_vehiclenet.onnx`. Once files are in place, run `python scripts/check_reid_readiness.py` to proceed to empirical accuracy benchmarking.
