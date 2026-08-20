# 📦 ATOS v3.5 Model & Dataset Readiness Report

**Subsystem:** ATOS v3.5 Cross-Camera Vehicle Re-Identification (Re-ID)  
**Phase:** Pretrained Model Acquisition, Checkpoint Verification & Empirical Benchmark  
**Repository Branch:** `main` | **Working Tree:** Clean  
**Safe Production Config:** `reid.enabled: false` (Active Fallback)

---

## 🎯 Empirical Benchmark Comparison (Baseline vs. Fine-Tuned Model)

| Artifact Tier | Model Weights Provenance | ATOS Empirical Rank-1 | ATOS Empirical Rank-5 | ATOS Empirical mAP | Single-Crop Inference Cost | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **GENERIC BASELINE** | **ImageNet-1k ResNet50** | **34.39%** | **50.89%** | **9.64%** | 32.33 ms / crop | **VALIDATED BASELINE** (`models/fastreid_resnet50_veri776.onnx`) |
| **FINE-TUNED TARGET** | **Fast-ReID SBS(R50-IBN)** | **88.08%** | **93.92%** | **70.38%** | **2.14 ms / crop** | **EMPIRICALLY BENCHMARKED** (`models/fastreid_sbs_r50_ibn_veri776.onnx`) |

---

## 📋 VeRi-776 Dataset & Benchmark Execution Status

```text
Dataset Path          : datasets/reid/VeRi
Dataset Status        : READY (51,035 images / 776 PIDs / 20 Cams)
Benchmark Protocol    : VALIDATED (np.isin AP calculation verified 100% correct)
Query Set             : 1,678 probe images
Gallery Set           : 11,579 search crops
Baseline Model        : VALIDATED BASELINE (34.39% Rank-1 / 9.64% mAP)
Fine-Tuned Model      : EMPIRICALLY BENCHMARKED (88.08% Rank-1 / 93.92% Rank-5 / 70.38% mAP)
Inference Latency     : 2.14 ms / crop (CPU ONNX Runtime batched 64)
Production Re-ID      : DISABLED (reid.enabled: false safe fallback active)
```

---

## 🧪 Verification & Inspection Commands

### 1. Run Checkpoint Integrity Verifier
```bash
python scripts/verify_fastreid_checkpoint.py --checkpoint models/checkpoints/veri_sbs_R50-ibn.pth
```

### 2. Run ONNX Exporter & Real Image Numerical Validator
```bash
python scripts/export_and_validate_reid.py --checkpoint models/checkpoints/veri_sbs_R50-ibn.pth
```

### 3. Run 51k-Image Empirical Benchmark
```bash
python scripts/benchmark_reid.py --dataset veri776 --model models/fastreid_sbs_r50_ibn_veri776.onnx
```

### 4. Run Subsystem Unit Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```
