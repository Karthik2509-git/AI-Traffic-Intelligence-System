# 📥 ATOS v3.5 Fast-ReID Vehicle Re-ID Model Acquisition & Verification Guide

## 📋 Subsystem Overview

This document specifies the official provenance, acquisition procedure, checksum verification, expected file paths, licensing, and export requirements for the fine-tuned Fast-ReID VeRi-776 vehicle Re-Identification model checkpoint (`veri_sbs_R50-ibn.pth`).

---

## 🎯 Baseline vs. Target Model Artifact Distinction

| Artifact Tier | Model Weights Provenance | ATOS Empirical Rank-1 | ATOS Empirical mAP | Official Model Zoo Reference | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **GENERIC BASELINE** | **ImageNet-1k ResNet50** | **34.39%** | **9.64%** | N/A | **VALIDATED BASELINE** (`models/fastreid_resnet50_veri776.onnx`) |
| **FINE-TUNED TARGET** | **Fast-ReID SBS(R50-IBN)** | **BENCHMARK PENDING** | **BENCHMARK PENDING** | Rank-1: 97.0% \| mAP: 81.9% \| mINP: 46.3% | **CHECKPOINT VERIFIED & ONNX EXPORTED** (`models/fastreid_sbs_r50_ibn_veri776.onnx`) |

> **Note**: Official Fast-ReID Model Zoo metrics (Rank-1 97.0%, mAP 81.9%) are reference targets only. ATOS empirical numbers remain explicitly `PENDING` until the benchmark runner evaluates `models/fastreid_sbs_r50_ibn_veri776.onnx` on the 51k-image dataset.

---

## 🎯 Verified Target Checkpoint & Artifact Provenance

- **Official Maintainer**: JDAI-CV (JD AI Research)
- **Repository**: [https://github.com/JDAI-CV/fast-reid](https://github.com/JDAI-CV/fast-reid)
- **Model Config**: `configs/VeRi/sbs_R50-ibn.yml`
- **Checkpoint File**: `models/checkpoints/veri_sbs_R50-ibn.pth` (189.08 MB)
- **Checkpoint SHA-256**: `57fb9c17d88911ea64390bf5427f43511435e7f88f6eed9dbc969d4b611e53cd`
- **Classifier Tensor Shape**: `heads.classifier.weight` $\rightarrow$ `[575, 2048]` (575 vehicle training classes)
- **Training Dataset**: VeRi-776 Train Split (37,778 images across vehicle identities)
- **License**: Apache License 2.0 (Permissive Open-Source)
- **Exported ONNX Artifact**: `models/fastreid_sbs_r50_ibn_veri776.onnx` (**89.77 MB**)
- **ONNX SHA-256**: `bc43d2fd8f39d1544da53de6a6556d12eacd6741710d2607f7d238cf577e6bb2`
- **Input Tensor**: `input` | `['batch_size', 3, 256, 256]` | `float32`
- **Output Tensor**: `embedding` | `['batch_size', 2048]` | `float32`
- **L2 Vector Normalization**: Verified ($\|e\|_2 = 1.000000$)
- **Deterministic Repeat Match**: Verified ($\text{Cosine Sim} = 1.000001$)

---

## 📥 Verification & Export Execution Procedure

1. **Verify Official Checkpoint**:
   ```bash
   python scripts/verify_fastreid_checkpoint.py --checkpoint models/checkpoints/veri_sbs_R50-ibn.pth
   ```
   - **Verification Status**: `VERIFIED_VERI776_CHECKPOINT` (575 vehicle identity classes verified).

2. **Export & Validate ONNX Model**:
   ```bash
   python scripts/export_and_validate_reid.py --checkpoint models/checkpoints/veri_sbs_R50-ibn.pth
   ```
   - Exports fine-tuned weights to `models/fastreid_sbs_r50_ibn_veri776.onnx`.
   - Validates ONNXRuntime numerical inference and vector L2 normalization.

3. **Run Empirical Benchmark (Next Stage)**:
   ```bash
   python scripts/benchmark_reid.py --dataset veri776 --model models/fastreid_sbs_r50_ibn_veri776.onnx
   ```

---

## 🛡️ Git Safety Enforcement

PyTorch `.pth` checkpoint files and ONNX `.onnx` model weights are excluded via `.gitignore` (`models/` rule). Checkpoints and model binaries will **NOT** be committed to Git repositories.
