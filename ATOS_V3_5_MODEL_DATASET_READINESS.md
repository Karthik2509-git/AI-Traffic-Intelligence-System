# 📦 ATOS v3.5 Model & Dataset Readiness Report

**Subsystem:** ATOS v3.5 Cross-Camera Vehicle Re-Identification (Re-ID)  
**Phase:** Pretrained Model Acquisition, Checkpoint Verification & ONNX Export  
**Repository Branch:** `main` | **Working Tree:** Clean  
**Safe Production Config:** `reid.enabled: false` (Active Fallback)

---

## 🎯 Model Selection & Baseline vs Target Matrix

| Artifact Tier | Model Weights Provenance | ATOS Empirical Rank-1 | ATOS Empirical mAP | Official Model Zoo Reference | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **GENERIC BASELINE** | **ImageNet-1k ResNet50** | **34.39%** | **9.64%** | N/A | **VALIDATED BASELINE** (`models/fastreid_resnet50_veri776.onnx`) |
| **FINE-TUNED TARGET** | **Fast-ReID SBS(R50-IBN)** | **BENCHMARK PENDING** | **BENCHMARK PENDING** | Rank-1: 97.0% \| mAP: 81.9% \| mINP: 46.3% | **CHECKPOINT VERIFIED & ONNX EXPORTED** (`models/fastreid_sbs_r50_ibn_veri776.onnx`) |

---

## 📋 VeRi-776 Dataset & Artifact Status

```text
Dataset Path          : datasets/reid/VeRi
Dataset Status        : READY (51,035 images / 776 PIDs / 20 Cams)
Benchmark Protocol    : VALIDATED (np.isin AP calculation verified correct)
Baseline ONNX Model   : VALIDATED BASELINE (34.39% Rank-1 / 9.64% mAP)
Fine-Tuned Checkpoint : VERIFIED (models/checkpoints/veri_sbs_R50-ibn.pth, 189.08 MB, 575 classes)
Fine-Tuned ONNX Model : EXPORTED & VALIDATED (models/fastreid_sbs_r50_ibn_veri776.onnx, 89.77 MB)
Numerical Validation  : PASSED (Finite values, 2048-dim, L2 norm 1.000000, self-cosine 1.000001)
Empirical Benchmark   : PENDING (Awaiting benchmark run on models/fastreid_sbs_r50_ibn_veri776.onnx)
Production Re-ID      : DISABLED (reid.enabled: false safe fallback active)
```

---

## 🧪 Verification & Inspection Commands

### 1. Run Checkpoint Integrity Verifier
```bash
python scripts/verify_fastreid_checkpoint.py --checkpoint models/checkpoints/veri_sbs_R50-ibn.pth
```
- **Output**: `VERIFIED_VERI776_CHECKPOINT` (575 vehicle training classes, SHA-256 `57fb9c17...`).

### 2. Run ONNX Exporter & Real Image Numerical Validator
```bash
python scripts/export_and_validate_reid.py --checkpoint models/checkpoints/veri_sbs_R50-ibn.pth
```
- **Output**: `[VALIDATION RESULT: PASSED]` (`models/fastreid_sbs_r50_ibn_veri776.onnx`, 89.77 MB, SHA-256 `bc43d2fd...`).

### 3. Run Subsystem Unit Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```
- **Output**: `Ran 5 tests ... OK`.
