# 📥 ATOS v3.5 Vehicle Re-ID Model Selection & Acquisition Guide

## 📋 Overview

This guide provides official sources, licensing, exact architecture specifications, and manual acquisition procedures for installing a trained Vehicle Re-Identification model into ATOS v3.5.

---

## 🎯 Verified Vehicle Re-ID Model Candidates

### 1. Primary Selected Model: **Fast-ReID ResNet50 (VeRi-776 Trained)**

| Property | Specification Detail |
| :--- | :--- |
| **Model Name** | Fast-ReID ResNet50 GeM (Circle Loss) |
| **Architecture** | ResNet50 with Generalized Mean (GeM) Pooling |
| **Official Repository** | [JDAI-CV / Fast-ReID](https://github.com/JDAI-CV/fast-reid) |
| **Official Checkpoint** | `veri_resnet50.pth` |
| **Training Dataset** | VeRi-776 Train Split (37,778 images across 576 vehicle identities) |
| **License** | Apache License 2.0 (Permissive Open Source) |
| **Input Tensor Shape** | `[1, 3, 256, 256]` (NCHW format, float32) |
| **Output Tensor Shape** | `[1, 2048]` float vector |
| **Embedding Dimension** | 2048 float (L2 Normalized) |
| **Parameter Count** | ~25.5 Million |
| **ONNX Availability** | Exportable via `tools/deploy/onnx_export.py` |
| **TensorRT Compatibility** | Compatible with `trtexec --onnx=veri_resnet50.onnx --fp16` |
| **Preprocessing** | Resize $256 \times 256$, Scale $[0, 1]$, ImageNet Mean $[0.485, 0.456, 0.406]$, Std $[0.229, 0.224, 0.225]$ |
| **Postprocessing** | L2 Normalization ($\hat{e} = \frac{e}{\|e\|_2}$) |
| **Deployment Complexity** | Moderate (Standard ResNet backbone, well-tested ONNX export) |

---

### 2. Fallback Selected Model: **TorchReID OSNet_x1_0 (VeRi-776 Trained)**

| Property | Specification Detail |
| :--- | :--- |
| **Model Name** | OSNet_x1_0 (Omni-Scale Feature Learning) |
| **Architecture** | OSNet_x1_0 |
| **Official Repository** | [Kaiyang Zhou / Torchreid](https://github.com/KaiyangZhou/deep-person-reid) |
| **Official Checkpoint** | `osnet_x1_0_veri776.pth` |
| **Training Dataset** | VeRi-776 Train Split (576 vehicle identities) |
| **License** | MIT License (Permissive Open Source) |
| **Input Tensor Shape** | `[1, 3, 256, 256]` (NCHW format, float32) |
| **Output Tensor Shape** | `[1, 512]` float vector |
| **Embedding Dimension** | 512 float (L2 Normalized) |
| **Parameter Count** | ~2.2 Million |
| **ONNX Availability** | Exportable via `torch.onnx.export` |
| **TensorRT Compatibility** | Compatible with `trtexec --onnx=osnet_x1_0.onnx --fp16` |
| **Preprocessing** | Resize $256 \times 256$, Scale $[0, 1]$, ImageNet Mean $[0.485, 0.456, 0.406]$, Std $[0.229, 0.224, 0.225]$ |
| **Postprocessing** | L2 Normalization ($\hat{e} = \frac{e}{\|e\|_2}$) |
| **Deployment Complexity** | Low (Lightweight footprint, ideal for real-time edge execution) |

---

## 🔬 Pipeline Stage Distinction

To maintain scientific integrity throughout deployment:

1. **Pretrained Inference**: Running forward pass of `models/reid_vehiclenet.onnx` directly on cropped vehicle images to generate 512-dim or 2048-dim L2-normalized feature vectors.
2. **Benchmark / Evaluation**: Running `python scripts/benchmark_reid.py` on VeRi-776 probe query images (1,678 images) against gallery test images (11,579 images) to calculate empirical Rank-1, Rank-5, and mAP accuracy scores on target hardware.
3. **Fine-Tuning**: (Optional) Further training on local site camera feeds if significant domain shift occurs across camera angles.
4. **Final Deployment**: Setting `reid.enabled: true` in `config/settings.yaml` after empirical benchmark verification.

---

## 🛠️ Step-by-Step Manual Acquisition & Conversion Procedure

> [!IMPORTANT]
> Do NOT commit model weights or dataset archives to git repository. Follow these manual steps to install weights locally on host disk.

### Step 1: Install PyTorch & Torchreid / Fast-ReID
```bash
pip install torch torchvision onnx onnxruntime
```

### Step 2: Export Model Weights to ONNX Format
Run the following script to export `osnet_x1_0` or `fast-reid` weights to `models/reid_vehiclenet.onnx`:

```python
import torch
import torchreid

# 1. Build model architecture
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=576,
    loss='softmax',
    pretrained=False
)

# 2. Load trained vehicle Re-ID weights
weight_path = 'osnet_x1_0_veri776.pth'
torchreid.utils.load_pretrained_weights(model, weight_path)
model.eval()

# 3. Export to ONNX format
dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(
    model,
    dummy_input,
    "models/reid_vehiclenet.onnx",
    input_names=["input"],
    output_names=["embedding"],
    dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
    opset_version=13
)
print("Successfully exported ONNX model to models/reid_vehiclenet.onnx")
```

### Step 3: Target File Verification & Integrity Check
Run the automated integrity checker to verify input/output tensor shapes and SHA-256 checksum:

```bash
python scripts/check_reid_readiness.py --model models/reid_vehiclenet.onnx --dataset-dir datasets/reid/veri776
```

### Expected Output after Model Installation
```text
--- MODEL INTEGRITY STATUS ---
Present           : True
Status            : MODEL_FILE_PRESENT
Path              : models/reid_vehiclenet.onnx
Input Shape       : [1, 3, 256, 256]
Output Shape      : [1, 512]
Runtime           : ONNXRuntime Validated
```
