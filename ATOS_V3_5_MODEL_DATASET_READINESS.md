# 📦 ATOS v3.5 Model & Dataset Readiness Report

**Subsystem:** ATOS v3.5 Cross-Camera Vehicle Re-Identification (Re-ID)  
**Phase:** Pretrained Model Acquisition, Checkpoint Verification & Benchmark Protocol  
**Repository Branch:** `main` | **Working Tree:** Clean  
**Safe Production Config:** `reid.enabled: false` (Active Fallback)

---

## 🎯 Model Selection & Baseline vs Target Matrix

| Artifact Tier | Model Weights Provenance | Rank-1 Accuracy | mAP Score | Status |
| :--- | :--- | :---: | :---: | :---: |
| **BASELINE ARTIFACT** | **Generic ImageNet-1k ResNet50** | **34.39%** | **9.64%** | **VALIDATED BASELINE** (`models/fastreid_resnet50_veri776.onnx`) |
| **TARGET ARTIFACT** | **VeRi-776 Fine-Tuned Fast-ReID (`veri_resnet50.pth`)** | **~88%+ (Expected)** | **~70%+ (Expected)** | **BENCHMARK PENDING** (`models/checkpoints/veri_resnet50.pth`) |

---

## 📋 VeRi-776 Dataset & Protocol Status

```text
Dataset Path         : datasets/reid/VeRi
Dataset Status       : READY
Total Images         : 51,035
Query Images         : 1,678
Gallery Images       : 11,579
Train Images         : 37,778
Unique Identities    : 776
Unique Cameras       : 20
Benchmark Protocol   : VALIDATED (np.isin AP calculation verified correct)
Baseline ONNX Model  : VALIDATED (ImageNet ResNet50 -> 34.39% Rank-1 / 9.64% mAP)
Fine-Tuned Checkpoint: PENDING (models/checkpoints/veri_resnet50.pth awaiting file placement)
Production Re-ID     : DISABLED (reid.enabled: false)
```

---

## 🧪 Verification & Inspection Tools

### 1. Checkpoint Integrity Verifier
```bash
python scripts/verify_fastreid_checkpoint.py --checkpoint models/checkpoints/veri_resnet50.pth
```

### 2. Model & Dataset Readiness Inspector
```bash
python scripts/check_reid_readiness.py
```

### 3. Subsystem Unit Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```
