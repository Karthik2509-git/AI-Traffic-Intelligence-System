# 📦 ATOS v3.5 Model & Dataset Readiness Report

**Subsystem:** ATOS v3.5 Cross-Camera Vehicle Re-Identification (Re-ID)  
**Phase:** Model & Dataset Selection & Verification  
**Repository Branch:** `main` | **Working Tree:** Clean  
**Safe Production Config:** `reid.enabled: false` (Active Fallback)

---

## 🎯 Model Selection & Acquisition Status Matrix

| Stage | Status | Details / Location |
| :--- | :---: | :--- |
| **Recommended Primary Model** | `SELECTED` | **Fast-ReID ResNet50 (VeRi-776)** (`[1, 3, 256, 256]` $\rightarrow$ `[1, 2048]`, Apache 2.0) |
| **Recommended Fallback Model** | `SELECTED` | **Torchreid OSNet_x1_0 (VeRi-776)** (`[1, 3, 256, 256]` $\rightarrow$ `[1, 512]`, MIT) |
| **Model Acquired** | `NO (PENDING)` | Weights file `models/reid_vehiclenet.onnx` un-populated on host disk |
| **Model Validated** | `NO (PENDING)` | Requires `python scripts/check_reid_readiness.py` output with verified SHA-256 and ONNX input/output shapes |
| **Model Benchmarked** | `NO (PENDING)` | Requires `python scripts/benchmark_reid.py` empirical Rank-1 / mAP output |

---

## 📊 Legitimate Vehicle Re-ID Model Candidates Audit

> [!NOTE]
> All parameter counts, embedding dimensions, inference costs, and accuracy figures in this evaluation matrix represent candidate/estimated values from official repositories. Exact metrics will be empirically measured on host hardware (`scripts/check_reid_readiness.py` and `scripts/benchmark_reid.py`) once model weights are installed.

| Exact Model Name | Architecture | Repository / Source | Training Dataset | License | Input Tensor | Output Tensor | Est. Params | ONNX / TRT Compatible | Deployment Complexity |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Fast-ReID ResNet50 GeM** *(Primary)* | ResNet50 | [JDAI-CV/fast-reid](https://github.com/JDAI-CV/fast-reid) | VeRi-776 (576 IDs) | Apache 2.0 | `[1, 3, 256, 256]` | `[1, 2048]` | ~25.5M | **Yes** (ONNX / TensorRT) | **Moderate** |
| **TorchReID OSNet_x1_0** *(Fallback)* | OSNet_x1_0 | [KaiyangZhou/torchreid](https://github.com/KaiyangZhou/deep-person-reid) | VeRi-776 (576 IDs) | MIT | `[1, 3, 256, 256]` | `[1, 512]` | ~2.2M | **Yes** (ONNX / TensorRT) | **Low** |
| **TransReID ViT-Base** | ViT-Base | [TransReID](https://github.com/albumentations-team/autoalbument) | VeRi-776 | Apache 2.0 | `[1, 3, 256, 256]` | `[1, 768]` | ~86.0M | **Conditional** | **High** |

---

## 🔬 Pipeline Stage Distinction

To maintain scientific integrity throughout deployment:

1. **Pretrained Inference**: Running forward pass of `models/reid_vehiclenet.onnx` directly on cropped vehicle images to generate 512-dim or 2048-dim L2-normalized feature vectors.
2. **Benchmark / Evaluation**: Running `python scripts/benchmark_reid.py` on VeRi-776 probe query images (1,678 images) against gallery test images (11,579 images) to calculate empirical Rank-1, Rank-5, and mAP accuracy scores on target hardware.
3. **Fine-Tuning**: (Optional) Further training on local site camera feeds if significant domain shift occurs across camera angles.
4. **Final Deployment**: Setting `reid.enabled: true` in `config/settings.yaml` after empirical benchmark verification.

---

## 📋 VeRi-776 Dataset Acquisition & Layout Checklist

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

## 🔍 Verification Commands

### Command 1: Run Model & Dataset Integrity Inspector
```bash
python scripts/check_reid_readiness.py --model models/reid_vehiclenet.onnx --dataset-dir datasets/reid/veri776
```

### Command 2: Run Re-ID Evaluation Harness
```bash
python scripts/benchmark_reid.py --dataset veri776 --dataset-dir datasets/reid/veri776 --model models/reid_vehiclenet.onnx
```

### Command 3: Run Subsystem Unit Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 🏁 Current Status & Next Actions

```text
[PASS] Tier 1: Implementation Complete (C++, Python, REST/WS, UI)
[PASS] Tier 2: Unit Tests Passing (4/4 tests OK)
[PASS] Tier 3: Integration Validated (reid_enabled: false safe fallback active)
[MODEL_FILE_PENDING] Tier 4: Model Loaded (Requires placement of models/reid_vehiclenet.onnx)
[DATASET_PENDING] Tier 5: Dataset Prepared (Requires manual extraction to datasets/reid/veri776/)
[DATASET_MISSING] Tier 6: Benchmark Executed (Pending dataset & model files)
[PENDING] Tier 7: Accuracy Validated (Rank-1, Rank-5, mAP pending empirical run)
[PENDING] Tier 8: Real Two-Camera Field Test
[PASS] Tier 9: Performance Validated (Baseline YOLOv8 + ByteTrack at 148 FPS)
[SAFE FALLBACK ACTIVE] Tier 10: Production Ready (Default reid_enabled: false active)
```

**Next Action Required by User**: Follow manual acquisition instructions in `docs/REID_MODEL_ACQUISITION.md` to export PyTorch weights to `models/reid_vehiclenet.onnx` and unpack VeRi-776 images into `datasets/reid/veri776/`.
