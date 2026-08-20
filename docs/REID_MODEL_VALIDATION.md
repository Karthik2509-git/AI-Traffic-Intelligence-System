# 🔬 ATOS v3.5 Re-ID Model Provenance & Artifact Validation Guide

## 📋 Subsystem Overview

This document specifies the exact model provenance, licensing, conversion procedures, ONNX graph validation parameters, and empirical verification requirements for the ATOS v3.5 Cross-Camera Vehicle Re-Identification subsystem.

---

## 🎯 Model Provenance & Checkpoint Identity

### 1. Primary Model Candidate: **Fast-ReID ResNet50 GeM (VeRi-776)**

- **Official Repository**: [JDAI-CV / Fast-ReID](https://github.com/JDAI-CV/fast-reid)
- **Maintainers**: JD AI Research (Kaiwei Kai, et al.)
- **Checkpoint Name**: `veri_resnet50.pth` (or `veri_sbs_R50.pth`)
- **Training Dataset**: VeRi-776 Train Split (37,778 images across 576 vehicle identities)
- **License**: Apache License 2.0 (Permissive Open-Source License)
- **Target ONNX Artifact**: `models/fastreid_resnet50_veri776.onnx`
- **Input Tensor**: `[1, 3, 256, 256]` float32 (NCHW format)
- **Output Tensor**: `[1, 2048]` float32 (L2 Normalized Feature Embedding)
- **Parameter Count**: ~25.5 Million

---

### 2. Fallback Model Candidate: **Torchreid OSNet_x1_0 (VeRi-776)**

- **Official Repository**: [KaiyangZhou / Torchreid](https://github.com/KaiyangZhou/deep-person-reid)
- **Maintainers**: Kaiyang Zhou (Nanyang Technological University)
- **Checkpoint Name**: `osnet_x1_0_veri776.pth`
- **Training Dataset**: VeRi-776 Train Split (576 vehicle identities)
- **License**: MIT License (Permissive Open-Source License)
- **Target ONNX Artifact**: `models/torchreid_osnet_x1_0_veri776.onnx`
- **Input Tensor**: `[1, 3, 256, 256]` float32 (NCHW format)
- **Output Tensor**: `[1, 512]` float32 (L2 Normalized Feature Embedding)
- **Parameter Count**: ~2.2 Million

---

## 🛠️ PyTorch-to-ONNX Conversion & Export Specifications

### Export Script (`export_reid_onnx.py`)

```python
import os
import torch
import torchreid

def export_osnet_veri776():
    # 1. Instantiate model architecture
    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=576,
        loss='softmax',
        pretrained=False
    )

    # 2. Load official PyTorch checkpoint weights
    checkpoint_path = 'osnet_x1_0_veri776.pth'
    torchreid.utils.load_pretrained_weights(model, checkpoint_path)
    model.eval()

    # 3. Export to ONNX static/dynamic shape format
    output_onnx_path = 'models/torchreid_osnet_x1_0_veri776.onnx'
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    dummy_input = torch.randn(1, 3, 256, 256)

    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={'input': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}},
        opset_version=13
    )
    print(f"Successfully exported ONNX model to {output_onnx_path}")

if __name__ == '__main__':
    export_osnet_veri776()
```

---

## 🧪 ONNX Runtime Validation Requirements

To validate an exported ONNX artifact:
1. **Model Load**: `onnxruntime.InferenceSession("models/fastreid_resnet50_veri776.onnx")` must initialize without protobuf errors.
2. **Input Tensor Inspection**: Input name `input`, shape `[1, 3, 256, 256]`, dtype `float32`.
3. **Output Tensor Inspection**: Output name `embedding`, shape `[1, 2048]` (or `[1, 512]`), dtype `float32`.
4. **Numerical Inference Test**: Input a real VeRi-776 image crop ($256 \times 256$ ImageNet normalized); output tensor must contain finite numerical values (no `NaN` or `Inf` values).
5. **L2 Normalization**: Embedding output vector must satisfy $\|e\|_2 = 1.0 \pm 1e-4$.

---

## 🏁 Strict Subsystem Readiness Status Matrix

```text
MODEL ARTIFACT     : NOT READY (Awaiting local PyTorch export of model weights)
MODEL RUNTIME      : NOT READY (ONNXRuntime installed; awaiting ONNX file creation)
MODEL VALIDATION   : PENDING
DATASET            : READY (VeRi-776 verified: 51,035 images / 776 PIDs / 20 Cams)
BENCHMARK          : PENDING
ACCURACY           : PENDING
PRODUCTION RE-ID   : NOT READY (reid.enabled: false safe fallback active)
```
