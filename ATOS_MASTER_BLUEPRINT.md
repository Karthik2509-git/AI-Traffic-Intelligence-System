# ATOS — Project Master Blueprint (Interview Prep)

> **Project:** ATOS v3.1 (AI Traffic Intelligence System)  
> **Stack:** C++, CUDA 12.4, TensorRT 10, OpenCV 4.9, YOLOv8m  
> **Key Metric:** **60.3 FPS @ 960p** (up from 6.4 FPS in Python)

---

## 1. System Architecture
- **Inference Pipeline**: Multi-threaded capture with a thread-safe concurrent queue. Uses Pinned Memory (Host-Device) to eliminate copy bottlenecks.
- **Custom CUDA Preprocessing**: Fused kernel performing Resize + Color Swap + Normalization + Contrast Boost in one pass.
- **TensorRT Optimization**: Serialized FP16 engine running a YOLOv8m backbone with dynamic anchor parsing.
- **Temporal Stability**: IoU-based track retention that persists "missed" detections for 2 frames to prevent flickering.

## 2. Performance Evolution
| Milestone | Build | FPS | Improvement |
|-----------|-------|-----|-------------|
| **Legacy** | Python / PyTorch | 6.4 | Baseline |
| **v2.0** | C++ / TensorRT | 95.8 | **14.9x Speedup** |
| **v3.1 (Final)** | **Optimized 960p** | **60.3** | **Precision + Speed** |

## 3. Interview Case Studies (STAR)
### Challenge: Improving Accuracy Without Hitting Performance
- **Situation**: Small vehicles were missed at 640p.
- **Task**: Increase resolution + accuracy while staying above 20 FPS.
- **Action**: Upgraded to YOLOv8m, bumped to 960p, and added a fused contrast boost to the GPU preprocessing step.
- **Result**: Achieved 60 FPS (3x the target) with significantly higher detection recall.

---
*Review this file to explain the "How" and "Why" behind your architecture choices.*
