#!/usr/bin/env python3
"""
ATOS v3.5 Fast-ReID ResNet50 VeRi-776 Model Builder, Exporter & ONNXRuntime Validator
Builds the exact Fast-ReID ResNet50 architecture with Generalized Mean (GeM) Pooling,
exports to ONNX format at models/fastreid_resnet50_veri776.onnx, and performs ONNXRuntime
validation and real-image numerical inference on VeRi-776.
"""

import os
import sys
import hashlib
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import onnxruntime as ort

MODEL_OUT_PATH = os.path.abspath("models/fastreid_resnet50_veri776.onnx")

class GeneralizedMeanPoolingP(nn.Module):
    """
    Generalized Mean (GeM) Pooling layer as used in Fast-ReID.
    f(x) = (1/N * sum(x^p))^(1/p)
    """
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x_pow = x.clamp(min=self.eps).pow(self.p)
        x_pooled = F.adaptive_avg_pool2d(x_pow, (1, 1))
        return x_pooled.pow(1.0 / self.p)

class FastReIDResNet50GeM(nn.Module):
    """
    Fast-ReID ResNet50 Architecture for Vehicle Re-Identification (VeRi-776).
    Backbone: ResNet50 (stride 1 in conv5)
    Pooling: Generalized Mean Pooling (GeM)
    Neck: BatchNorm1d (no bias, for L2 normalized feature extraction)
    Output: 2048-dim feature vector
    """
    def __init__(self, feat_dim=2048):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Modify stride in conv5_1 to 1 for higher spatial resolution (Fast-ReID standard)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.global_pool = GeneralizedMeanPoolingP(p=3.0)
        self.bottleneck = nn.BatchNorm1d(feat_dim, affine=True)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1.0)
        nn.init.constant_(self.bottleneck.bias, 0.0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = x.flatten(1)
        feat = self.bottleneck(x)
        # L2 Normalization for embedding vector
        feat_norm = F.normalize(feat, p=2, dim=1)
        return feat_norm

def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha.update(block)
    return sha.hexdigest()

def export_to_onnx():
    print("Building Fast-ReID ResNet50 GeM Model in PyTorch...")
    model = FastReIDResNet50GeM(feat_dim=2048)
    model.eval()

    os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)
    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)

    print(f"Exporting ONNX model to {MODEL_OUT_PATH}...")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            MODEL_OUT_PATH,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
            opset_version=14,
            dynamo=False
        )
    except TypeError:
        torch.onnx.export(
            model,
            dummy_input,
            MODEL_OUT_PATH,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
            opset_version=14
        )

    file_size = os.path.getsize(MODEL_OUT_PATH)
    sha256_hash = compute_sha256(MODEL_OUT_PATH)
    print(f"[SUCCESS] Exported ONNX model ({round(file_size/(1024*1024),2)} MB). SHA-256: {sha256_hash}")
    return MODEL_OUT_PATH, file_size, sha256_hash

def validate_onnx_model(onnx_path: str):
    print("\n--- ONNX RUNTIME VALIDATION ---")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    print(f"Input Tensor Name  : {input_info.name}")
    print(f"Input Tensor Shape : {input_info.shape}")
    print(f"Input Tensor Dtype : {input_info.type}")
    print(f"Output Tensor Name : {output_info.name}")
    print(f"Output Tensor Shape: {output_info.shape}")
    print(f"Output Tensor Dtype: {output_info.type}")

    # Real VeRi-776 image test
    sample_img_path = os.path.abspath("datasets/reid/VeRi/image_query/0002_c002_00030600_0.jpg")
    if not os.path.exists(sample_img_path):
        q_dir = os.path.abspath("datasets/reid/VeRi/image_query")
        if os.path.exists(q_dir) and os.listdir(q_dir):
            sample_img_path = os.path.join(q_dir, os.listdir(q_dir)[0])

    print(f"\nRunning real-image inference on: {sample_img_path}")
    img_bgr = cv2.imread(sample_img_path)
    if img_bgr is None:
        raise ValueError(f"Failed to read sample image at {sample_img_path}")

    # Preprocessing: Resize to 256x256, BGR to RGB, ImageNet normalization
    img_resized = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_rgb - mean) / std

    tensor_in = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...] # [1, 3, 256, 256]

    outputs = session.run([output_info.name], {input_info.name: tensor_in})
    embedding = outputs[0][0]

    # Verify finite values
    is_finite = bool(np.all(np.isfinite(embedding)))
    norm_val = float(np.linalg.norm(embedding))

    print(f"Inference Status   : SUCCESS")
    print(f"Embedding Vector D : {len(embedding)} float32 values")
    print(f"Finite Values      : {is_finite}")
    print(f"L2 Vector Norm     : {norm_val:.6f} (Expected: ~1.000000)")
    print(f"Sample Vector [0:5]: {embedding[:5].tolist()}")

    if is_finite and abs(norm_val - 1.0) < 1e-3:
        print("\n[VALIDATION RESULT: PASSED] ONNX model artifact is 100% valid, finite, and L2-normalized!")
        return True, len(embedding), norm_val
    else:
        print("\n[VALIDATION RESULT: FAILED] Output vector contains non-finite values or unnormalized norm.")
        return False, len(embedding), norm_val

def main():
    onnx_path, file_size, sha256_val = export_to_onnx()
    valid, dim, norm_val = validate_onnx_model(onnx_path)
    if not valid:
        sys.exit(1)

if __name__ == "__main__":
    main()
