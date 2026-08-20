# 📥 ATOS v3.5 Fast-ReID Vehicle Re-ID Model Acquisition & Verification Guide

## 📋 Subsystem Overview

This document specifies the official provenance, acquisition procedure, checksum verification, expected file paths, licensing, and export requirements for the fine-tuned Fast-ReID VeRi-776 vehicle Re-Identification model checkpoint.

---

## 🎯 Baseline vs Target Checkpoint Distinction

| Artifact Tier | Model Weights Provenance | Rank-1 Accuracy | mAP Score | Status |
| :--- | :--- | :---: | :---: | :---: |
| **BASELINE ARTIFACT** | **Generic ImageNet-1k ResNet50** | **34.39%** | **9.64%** | **VALIDATED BASELINE** (`models/fastreid_resnet50_veri776.onnx`) |
| **TARGET ARTIFACT** | **VeRi-776 Fine-Tuned Fast-ReID (`veri_resnet50.pth`)** | **~88%+ (Expected)** | **~70%+ (Expected)** | **BENCHMARK PENDING** (`models/checkpoints/veri_resnet50.pth`) |

---

## 🎯 Official Checkpoint Provenance & Metadata

- **Official Maintainer**: JDAI-CV (JD AI Research)
- **Repository**: [https://github.com/JDAI-CV/fast-reid](https://github.com/JDAI-CV/fast-reid)
- **Model Config**: `configs/VeRi/bagtricks_R50.yml` or `configs/VeRi/sbs_R50.yml`
- **Official Checkpoint Name**: `veri_resnet50.pth`
- **Training Dataset**: VeRi-776 Train Split (37,778 images across 576 vehicle identities)
- **License**: Apache License 2.0 (Permissive Open-Source)
- **Expected Destination**: `models/checkpoints/veri_resnet50.pth`

---

## 📥 Official Acquisition Procedure

1. **Obtain Official Weights**:
   Download the pretrained Fast-ReID VeRi-776 weights checkpoint (`veri_resnet50.pth`) from the official JDAI-CV Fast-ReID repository Model Zoo:
   - Official Repo: [https://github.com/JDAI-CV/fast-reid](https://github.com/JDAI-CV/fast-reid)
   - Save the downloaded PyTorch checkpoint to:
     ```bash
     models/checkpoints/veri_resnet50.pth
     ```

2. **Verify Checkpoint Integrity**:
   Run the ATOS checkpoint integrity inspector:
   ```bash
   python scripts/verify_fastreid_checkpoint.py --checkpoint models/checkpoints/veri_resnet50.pth
   ```
   - **Verification Requirement**: The script checks that `state_dict` contains 576 output classifier classes (matching the 576 vehicle identities of VeRi-776) and confirms it is NOT a generic ImageNet backbone.

3. **Export & Validate ONNX Model**:
   Once verified, run the ONNX exporter:
   ```bash
   python scripts/export_and_validate_reid.py --checkpoint models/checkpoints/veri_resnet50.pth
   ```
   - Exports the fine-tuned weights to `models/fastreid_resnet50_veri776.onnx`.
   - Validates ONNXRuntime numerical inference and vector L2 normalization.

4. **Re-Run Empirical Benchmark**:
   ```bash
   python scripts/benchmark_reid.py --dataset veri776 --model models/fastreid_resnet50_veri776.onnx
   ```

---

## 🛡️ Git Safety Enforcement

PyTorch `.pth` checkpoint files and ONNX `.onnx` model weights are excluded via `.gitignore` (`models/` rule). Checkpoints will **NOT** be committed to Git repositories.
