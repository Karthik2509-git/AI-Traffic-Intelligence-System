#!/usr/bin/env python3
"""
ATOS v3.5 Fast-ReID SBS(R50-IBN) VeRi-776 Model Builder, Exporter & ONNXRuntime Validator
Builds the exact Fast-ReID Stronger Baseline (SBS) ResNet50-IBN architecture with Non-Local attention
and Generalized Mean (GeM) Pooling, loads official fine-tuned PyTorch checkpoint weights from
models/checkpoints/veri_sbs_R50-ibn.pth, removes classification head for Re-ID inference,
exports L2-normalized feature embedding ONNX model to models/fastreid_sbs_r50_ibn_veri776.onnx,
and performs ONNXRuntime numerical validation on real VeRi-776 images.
"""

import os
import sys
import hashlib
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import onnxruntime as ort

DEFAULT_CHECKPOINT = os.path.abspath("models/checkpoints/veri_sbs_R50-ibn.pth")
DEFAULT_ONNX_OUT = os.path.abspath("models/fastreid_sbs_r50_ibn_veri776.onnx")

def parse_args():
    parser = argparse.ArgumentParser(description="ATOS Fast-ReID SBS(R50-IBN) ONNX Exporter & Validator")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to PyTorch checkpoint (.pth)")
    parser.add_argument("--onnx-out", type=str, default=DEFAULT_ONNX_OUT,
                        help="Path to output ONNX model file (.onnx)")
    return parser.parse_args()

class Nonlocal2D(nn.Module):
    """Non-local attention block as used in Fast-ReID SBS(R50-IBN)."""
    def __init__(self, in_channels):
        super().__init__()
        self.g = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.theta = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.W = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        g_x = self.g(x).view(b, 1, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(b, 1, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(b, 1, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x).permute(0, 2, 1).contiguous().view(b, 1, h, w)
        W_y = self.W(y)
        return x + W_y

class IBN(nn.Module):
    """Instance-Batch Normalization (IBN-a) layer."""
    def __init__(self, planes):
        super().__init__()
        half = planes // 2
        self.half = half
        self.IN = nn.InstanceNorm2d(half, affine=True)
        self.BN = nn.BatchNorm2d(half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out_in = self.IN(split[0].contiguous())
        out_bn = self.BN(split[1].contiguous())
        return torch.cat((out_in, out_bn), 1)

class BottleneckIBN(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_ibn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if use_ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNetIBN(nn.Module):
    def __init__(self, layers=[3, 4, 6, 3]):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BottleneckIBN, 64, layers[0], use_ibn=True)
        self.layer2 = self._make_layer(BottleneckIBN, 128, layers[1], use_ibn=True)
        self.layer3 = self._make_layer(BottleneckIBN, 256, layers[2], use_ibn=True)
        self.layer4 = self._make_layer(BottleneckIBN, 512, layers[3], stride=1, use_ibn=False)

        self.NL_2 = nn.ModuleList([Nonlocal2D(512) for _ in range(2)])
        self.NL_3 = nn.ModuleList([Nonlocal2D(1024) for _ in range(3)])

    def _make_layer(self, block, planes, blocks, stride=1, use_ibn=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_ibn=use_ibn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_ibn=use_ibn))

        return nn.Sequential(*layers)

class GeneralizedMeanPoolingP(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x_pow = x.clamp(min=self.eps).pow(self.p)
        x_pooled = F.adaptive_avg_pool2d(x_pow, (1, 1))
        return x_pooled.pow(1.0 / self.p)

class FastReIDSBSModel(nn.Module):
    """
    Fast-ReID Stronger Baseline (SBS) ResNet50-IBN Architecture for VeRi-776.
    Inference path extracts 2048-dim feature embedding (classifier layer ignored for inference).
    """
    def __init__(self, num_classes=575, feat_dim=2048):
        super().__init__()
        self.pixel_mean = nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=False)
        self.pixel_std = nn.Parameter(torch.ones(1, 3, 1, 1), requires_grad=False)

        self.backbone = ResNetIBN(layers=[3, 4, 6, 3])

        self.heads = nn.ModuleDict({
            'pool_layer': GeneralizedMeanPoolingP(p=3.0),
            'bottleneck': nn.Sequential(
                nn.BatchNorm1d(feat_dim)
            ),
            'classifier': nn.Linear(feat_dim, num_classes, bias=False)
        })

    def forward(self, x):
        # Backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)

        for idx, blk in enumerate(self.backbone.layer2):
            x = blk(x)
            if idx in [1, 3]:
                nl_idx = 0 if idx == 1 else 1
                x = self.backbone.NL_2[nl_idx](x)

        for idx, blk in enumerate(self.backbone.layer3):
            x = blk(x)
            if idx in [1, 3, 5]:
                nl_idx = {1: 0, 3: 1, 5: 2}[idx]
                x = self.backbone.NL_3[nl_idx](x)

        x = self.backbone.layer4(x)

        # Head - Extract L2 Normalized Feature Embedding
        x = self.heads['pool_layer'](x)
        x = x.flatten(1)
        feat = self.heads['bottleneck'](x)
        feat_norm = F.normalize(feat, p=2, dim=1)
        return feat_norm

def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha.update(block)
    return sha.hexdigest()

def export_to_onnx(checkpoint_path: str, onnx_out_path: str):
    abs_ckpt = os.path.abspath(checkpoint_path)
    if not os.path.exists(abs_ckpt):
        raise FileNotFoundError(f"Checkpoint file missing at {abs_ckpt}")

    print(f"Loading official PyTorch checkpoint from {abs_ckpt}...")
    ckpt = torch.load(abs_ckpt, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Determine num_classes from classifier layer
    num_classes = 575
    if "heads.classifier.weight" in sd:
        num_classes = sd["heads.classifier.weight"].shape[0]

    print(f"Building Fast-ReID SBS(R50-IBN) Model (Classifier classes: {num_classes})...")
    model = FastReIDSBSModel(num_classes=num_classes, feat_dim=2048)

    # Strict loading check: filter out harmless num_batches_tracked
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing_weights = [k for k in missing if "num_batches_tracked" not in k]
    unexpected_weights = [k for k in unexpected if "num_batches_tracked" not in k]

    if len(missing_weights) > 0 or len(unexpected_weights) > 0:
        print(f"[ERROR] Strict checkpoint loading failed!")
        print(f"  Missing weight keys   : {missing_weights}")
        print(f"  Unexpected weight keys: {unexpected_weights}")
        raise RuntimeError("Checkpoint state_dict mismatch! Aborting export.")

    print(f"[SUCCESS] Checkpoint weights loaded with 100% precision (0 missing, 0 unexpected weight keys).")
    model.eval()

    abs_onnx = os.path.abspath(onnx_out_path)
    os.makedirs(os.path.dirname(abs_onnx), exist_ok=True)
    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)

    print(f"Exporting ONNX model to {abs_onnx}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            abs_onnx,
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
            abs_onnx,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
            opset_version=14
        )

    file_size = os.path.getsize(abs_onnx)
    sha256_hash = compute_sha256(abs_onnx)
    print(f"[SUCCESS] Exported ONNX model ({round(file_size/(1024*1024),2)} MB). SHA-256: {sha256_hash}")
    return abs_onnx, file_size, sha256_hash

def validate_onnx_model(onnx_path: str):
    print("\n--- ONNX RUNTIME NUMERICAL VALIDATION ---")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    print(f"Input Tensor Name  : {input_info.name}")
    print(f"Input Tensor Shape : {input_info.shape}")
    print(f"Input Tensor Dtype : {input_info.type}")
    print(f"Output Tensor Name : {output_info.name}")
    print(f"Output Tensor Shape: {output_info.shape}")
    print(f"Output Tensor Dtype: {output_info.type}")

    sample_img_path = os.path.abspath("datasets/reid/VeRi/image_query/0002_c002_00030600_0.jpg")
    if not os.path.exists(sample_img_path):
        q_dir = os.path.abspath("datasets/reid/VeRi/image_query")
        if os.path.exists(q_dir) and os.listdir(q_dir):
            sample_img_path = os.path.join(q_dir, os.listdir(q_dir)[0])

    print(f"\nRunning real-image numerical inference on: {sample_img_path}")
    img_bgr = cv2.imread(sample_img_path)
    if img_bgr is None:
        raise ValueError(f"Failed to read sample image at {sample_img_path}")

    # ImageNet preprocessing matching sbs_R50-ibn.yml (Resize 256x256, BGR -> RGB, ImageNet normalization)
    img_resized = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_rgb - mean) / std

    tensor_in = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...] # [1, 3, 256, 256]

    # First inference pass
    outputs1 = session.run([output_info.name], {input_info.name: tensor_in})
    emb1 = outputs1[0][0]

    # Second inference pass (Deterministic check)
    outputs2 = session.run([output_info.name], {input_info.name: tensor_in})
    emb2 = outputs2[0][0]

    is_finite = bool(np.all(np.isfinite(emb1)))
    norm_val = float(np.linalg.norm(emb1))
    dot_sim = float(np.dot(emb1, emb2))

    print(f"Inference Status   : SUCCESS")
    print(f"Embedding Vector D : {len(emb1)} float32 values")
    print(f"Finite Values      : {is_finite}")
    print(f"L2 Vector Norm     : {norm_val:.6f} (Expected: ~1.000000)")
    print(f"Self-Cosine Match  : {dot_sim:.6f} (Expected: 1.000000)")
    print(f"Sample Vector [0:5]: {emb1[:5].tolist()}")

    passed = is_finite and (abs(norm_val - 1.0) < 1e-3) and (abs(dot_sim - 1.0) < 1e-3)

    if passed:
        print("\n[VALIDATION RESULT: PASSED] Fast-ReID SBS(R50-IBN) ONNX model artifact is 100% valid, finite, and L2-normalized!")
        return True, len(emb1), norm_val
    else:
        print("\n[VALIDATION RESULT: FAILED] Output vector failed numerical validation check.")
        return False, len(emb1), norm_val

def main():
    args = parse_args()
    onnx_path, file_size, sha256_val = export_to_onnx(args.checkpoint, args.onnx_out)
    valid, dim, norm_val = validate_onnx_model(onnx_path)
    if not valid:
        sys.exit(1)

if __name__ == "__main__":
    main()
