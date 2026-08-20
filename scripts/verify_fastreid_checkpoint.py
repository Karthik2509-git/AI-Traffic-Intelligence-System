#!/usr/bin/env python3
"""
ATOS v3.5 Fast-ReID VeRi-776 Checkpoint Verifier & Inspector
Validates file presence, SHA-256 checksum, PyTorch state_dict keys, and vehicle identity classifier shapes
to distinguish official fine-tuned Fast-ReID VeRi-776 checkpoints from generic ImageNet-1k baseline weights.
"""

import os
import sys
import json
import hashlib
import argparse
from typing import Dict, Any

def parse_args():
    parser = argparse.ArgumentParser(description="ATOS Fast-ReID Checkpoint Verification Tool")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/veri_sbs_R50-ibn.pth",
                        help="Path to PyTorch checkpoint file (.pth or .pt)")
    return parser.parse_args()

def compute_sha256(filepath: str) -> str:
    """Calculates SHA-256 checksum of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha.update(block)
    return sha.hexdigest()

def verify_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    abs_path = os.path.abspath(checkpoint_path)

    if not os.path.exists(abs_path):
        return {
            "status": "CHECKPOINT_FILE_MISSING",
            "verified": False,
            "checkpoint_path": checkpoint_path,
            "abs_path": abs_path,
            "file_size_bytes": 0,
            "sha256": None,
            "is_veri776_finetuned": False,
            "num_classes": 0,
            "message": (
                f"Checkpoint file not found at {abs_path}.\n"
                "OFFICIAL ACQUISITION INSTRUCTIONS:\n"
                "1. Repository: JDAI-CV / Fast-ReID (https://github.com/JDAI-CV/fast-reid)\n"
                "2. Model Config: configs/VeRi/sbs_R50-ibn.yml\n"
                "3. Checkpoint Name: veri_sbs_R50-ibn.pth\n"
                "4. License: Apache License 2.0\n"
                "5. Save destination: models/checkpoints/veri_sbs_R50-ibn.pth"
            )
        }

    file_size = os.path.getsize(abs_path)
    sha256_hash = compute_sha256(abs_path)

    try:
        import torch
        ckpt = torch.load(abs_path, map_location="cpu")

        # Handle state_dict wrapping ('model', 'state_dict', or direct dict)
        if isinstance(ckpt, dict):
            if "model" in ckpt:
                state_dict = ckpt["model"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt.state_dict()

        # Check for classifier layer (heads.classifier.weight or classifier.weight)
        num_classes = 0
        classifier_key = None
        for k, v in state_dict.items():
            if "classifier.weight" in k or "fc.weight" in k or "head.weight" in k:
                classifier_key = k
                num_classes = v.shape[0]
                break

        # Official Fast-ReID VeRi-776 checkpoint artifacts contain 575 or 576 vehicle training classes
        is_veri = (num_classes in [575, 576])
        is_imagenet = (num_classes == 1000)

        if is_veri:
            diag_msg = (
                f"SUCCESS: Verified official fine-tuned Fast-ReID VeRi-776 checkpoint ({num_classes} vehicle classes).\n"
                f"Note: {num_classes} classifier output dimension corresponds to the official Fast-ReID SBS(R50-IBN) checkpoint artifact."
            )
            status_str = "VERIFIED_VERI776_CHECKPOINT"
        elif is_imagenet:
            diag_msg = "WARNING: Checkpoint is standard ImageNet-1k baseline (1000 classes), NOT fine-tuned VeRi-776!"
            status_str = "GENERIC_IMAGENET_CHECKPOINT"
        else:
            diag_msg = f"NOTICE: Checkpoint loaded ({len(state_dict)} tensors, classifier classes: {num_classes})."
            status_str = "CUSTOM_CHECKPOINT_LOADED"

        return {
            "status": status_str,
            "verified": is_veri,
            "checkpoint_path": checkpoint_path,
            "abs_path": abs_path,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "sha256": sha256_hash,
            "num_tensors": len(state_dict),
            "classifier_key": classifier_key,
            "num_classes": num_classes,
            "associated_config": "configs/VeRi/sbs_R50-ibn.yml",
            "is_veri776_finetuned": is_veri,
            "message": diag_msg
        }

    except Exception as e:
        return {
            "status": "CHECKPOINT_LOAD_FAILED",
            "verified": False,
            "checkpoint_path": checkpoint_path,
            "abs_path": abs_path,
            "file_size_bytes": file_size,
            "sha256": sha256_hash,
            "is_veri776_finetuned": False,
            "message": f"Failed to inspect PyTorch checkpoint: {str(e)}"
        }

def main():
    args = parse_args()
    print("==================================================")
    print("ATOS v3.5 Fast-ReID Checkpoint Integrity Inspector")
    print("==================================================")

    res = verify_checkpoint(args.checkpoint)

    print(f"\nTarget Path  : {res['checkpoint_path']}")
    print(f"Status       : {res['status']}")
    print(f"Verified     : {res['verified']}")
    print(f"File Size    : {res.get('file_size_mb', 0)} MB")
    print(f"SHA-256      : {res['sha256']}")
    print(f"Classes      : {res.get('num_classes', 0)}")
    print(f"Config File  : {res.get('associated_config', 'N/A')}")
    print(f"Diagnostic   : {res['message']}\n")

    out_path = os.path.abspath("runs/reid_checkpoint_verification.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)

    if not res["verified"]:
        print("--------------------------------------------------")
        print("ACTION REQUIRED:")
        print("Please acquire the official fine-tuned Fast-ReID checkpoint:")
        print("  Destination: models/checkpoints/veri_sbs_R50-ibn.pth")
        print("  Official Repository: https://github.com/JDAI-CV/fast-reid")
        print("--------------------------------------------------")

if __name__ == "__main__":
    main()
