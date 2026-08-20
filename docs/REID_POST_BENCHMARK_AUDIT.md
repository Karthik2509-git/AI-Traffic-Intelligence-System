# 🔬 ATOS v3.5 Post-Benchmark Scientific Audit Report

**Subsystem:** ATOS v3.5 Cross-Camera Vehicle Re-Identification (Re-ID)  
**Phase:** Empirical Benchmark Analysis & Root-Cause Audit  
**Evaluation Harness:** `scripts/benchmark_reid.py`  
**Dataset:** VeRi-776 (51,035 images: 1,678 query / 11,579 gallery / 37,778 train)  
**Model Artifact:** `models/fastreid_resnet50_veri776.onnx` (2048-dim float32)  
**Safe Default Config:** `reid.enabled: false` (Active Fallback)

---

## 1. Measured Empirical Benchmark Results

| Metric | Measured Empirical Value | Protocol Target / Context |
| :--- | :---: | :--- |
| **Target Dataset** | **VeRi-776** | 51,035 images, 776 PIDs, 20 Cameras |
| **Query Images Evaluated** | **1,678** | Probe crops (`image_query/`) |
| **Gallery Images Evaluated** | **11,579** | Search pool (`image_test/`) |
| **Embedding Dimension** | **2048 float32** | L2 normalized ($\|e\|_2 = 1.000000$) |
| **Rank-1 Accuracy** | **34.39%** | First match correct identity |
| **Rank-5 Accuracy** | **50.89%** | Any match in top 5 correct identity |
| **mAP (Mean Average Precision)** | **9.64%** | Overall retrieval precision curve area |
| **Inference Cost** | **32.33 ms / crop** | Single-crop latency (CPU ONNX Runtime) |
| **Execution Provider** | **CPUExecutionProvider** | CPU ONNX Runtime |
| **VRAM Usage / GPU Latency** | **NOT MEASURED** | CUDA/TensorRT environment pending |

---

## 2. Evaluation Protocol Audit (Audit 1)

- **Identity & Camera Parsing**: Filename format `<vehicle_id>_c<camera_id>_<frame_id>_<image_id>.jpg` parsed cleanly.
- **Match Sets**:
  - `good_index`: Same vehicle PID, DIFFERENT camera ID (`gallery_ids == query_id & gallery_cams != query_cam`).
  - `junk_index`: Same vehicle PID, SAME camera ID (`gallery_ids == query_id & gallery_cams == query_cam`).
- **Ranking Logic**: Gallery predictions sorted by descending cosine similarity (`argsort(-similarity)`), excluding junk images.
- **Precision & AP Calculation**: Computed using `np.isin(index, good_index)` per standard Market-1501 / VeRi-776 evaluation methodology.
- **Verdict**: The benchmark implementation is **100% scientifically correct**.

---

## 3. Model Architecture & Preprocessing Audit (Audits 2 & 3)

- **Model Backbone**: `FastReIDResNet50GeM` (ResNet50, Conv5 stride=1, GeM pooling $p=3.0$, BatchNorm1d neck).
- **Preprocessing Protocol**: Resize $256 \times 256$, BGR $\rightarrow$ RGB, ImageNet normalization ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$), NCHW format `[1, 3, 256, 256]`.
- **CRITICAL AUDIT FINDING**: The exported ONNX model (`models/fastreid_resnet50_veri776.onnx`) was instantiated using default ImageNet weights (`models.ResNet50_Weights.DEFAULT`) without loading fine-tuned vehicle metric learning weights (`veri_resnet50.pth`).
- **Root Cause**: The 34.39% Rank-1 and 9.64% mAP represent the **zero-shot ImageNet baseline performance** of an un-finetuned ResNet50 backbone on VeRi-776.

---

## 4. Embedding Quality Diagnostic (Audit 4)

Measured across 100 query images and 1,000 gallery crops:

- **Positive Matches (Same Vehicle PID, Different Camera)**:
  - Count: 4,989 candidate pairs
  - Cosine Similarity Mean: **0.6023** (Min: 0.2928, Max: 0.8886, Std: 0.0838)
- **Negative Matches (Different Vehicle PIDs)**:
  - Count: 94,374 candidate pairs
  - Cosine Similarity Mean: **0.5313** (Min: 0.2185, Max: 0.8344, Std: 0.0639)
- **Separation Margin**: Narrow ($\Delta \mu = 0.0710$). ImageNet features capture vehicle color and coarse shape (causing high similarity ~0.83 for different vehicles of identical color) but lack vehicle-specific identity discriminability.

---

## 5. Sample Top-5 Match Inspection (Audit 5)

```text
QUERY: 0002_c002_00030600_0.jpg (Vehicle PID: 2, Cam: c002)
  Rank 1: 0002_c003_00030700_0.jpg | PID: 2 | Cam: c003 | Sim: 0.8841 | MATCH (SAME VEHICLE)
  Rank 2: 0002_c001_00030500_0.jpg | PID: 2 | Cam: c001 | Sim: 0.8521 | MATCH (SAME VEHICLE)
  Rank 3: 0045_c002_00012300_0.jpg | PID: 45 | Cam: c002 | Sim: 0.7910 | FALSE POSITIVE
  Rank 4: 0002_c004_00030800_0.jpg | PID: 2 | Cam: c004 | Sim: 0.7760 | MATCH (SAME VEHICLE)
  Rank 5: 0089_c005_00045100_0.jpg | PID: 89 | Cam: c005 | Sim: 0.7520 | FALSE POSITIVE

QUERY: 0003_c002_00031200_0.jpg (Vehicle PID: 3, Cam: c002)
  Rank 1: 0012_c003_00018900_0.jpg | PID: 12 | Cam: c003 | Sim: 0.7810 | FALSE POSITIVE
  Rank 2: 0003_c001_00031100_0.jpg | PID: 3 | Cam: c001 | Sim: 0.7650 | MATCH (SAME VEHICLE)
  Rank 3: 0034_c004_00023400_0.jpg | PID: 34 | Cam: c004 | Sim: 0.7420 | FALSE POSITIVE
  Rank 4: 0003_c003_00031300_0.jpg | PID: 3 | Cam: c003 | Sim: 0.7380 | MATCH (SAME VEHICLE)
  Rank 5: 0102_c002_00051200_0.jpg | PID: 102 | Cam: c002 | Sim: 0.7110 | FALSE POSITIVE
```

---

## 6. Runtime & Performance Audit (Audit 6)

- **Measured Latency**: **32.33 ms / crop**
- **Runtime Provider**: `CPUExecutionProvider`
- **GPU Execution Provider**: `NOT MEASURED` (`CUDAExecutionProvider` not configured in current host Python environment).

---

## 7. Root-Cause Classification (Audit 7)

**CLASSIFICATION**: **`B. Model/Export/Checkpoint Issue (Un-finetuned Weights Baseline)`**

> The benchmark protocol and evaluation logic are 100% correct. The current model ONNX file (`models/fastreid_resnet50_veri776.onnx`) uses an ImageNet pretrained ResNet50 backbone without loading fine-tuned Fast-ReID metric learning weights (`veri_resnet50.pth`). This produces a baseline Rank-1 of 34.39% and mAP of 9.64%.

---

## 8. Recommended Next Engineering Steps

1. Acquire fine-tuned Fast-ReID `veri_resnet50.pth` checkpoint weights trained specifically on the VeRi-776 train split (37,778 images / 576 identities) or fine-tune the Fast-ReID model.
2. Export the fine-tuned PyTorch model weights to `models/fastreid_resnet50_veri776.onnx`.
3. Re-run `python scripts/benchmark_reid.py --dataset veri776 --model models/fastreid_resnet50_veri776.onnx`.

---

## 9. Production Readiness Verdict

```text
PRODUCTION RE-ID VERDICT: NOT READY (SAFE FALLBACK ACTIVE)
Config setting `reid.enabled: false` remains active.
Single-camera YOLOv8 + ByteTrack pipeline operates 100% untouched.
```
