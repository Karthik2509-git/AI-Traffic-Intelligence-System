#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace atos {
namespace cuda {

/**
 * @brief IoU calculation helper.
 */
__device__ inline float box_iou(const float* a, const float* b) {
    float left = max(a[0], b[0]), top = max(a[1], b[1]);
    float right = min(a[2], b[2]), bottom = min(a[3], b[3]);
    float width = max(right - left, 0.0f), height = max(bottom - top, 0.0f);
    float inter_area = width * height;
    if (inter_area <= 0) return 0.0f;
    float area_a = (a[2] - a[0]) * (a[3] - a[1]);
    float area_b = (b[2] - b[0]) * (b[3] - b[1]);
    return inter_area / (area_a + area_b - inter_area);
}

/**
 * @brief Massive Parallel Non-Maximum Suppression.
 * 
 * Logic:
 * Each thread handles one box. It compares its box with all boxes 
 * that have a HIGHER confidence (lower index, assuming boxes are sorted).
 * If any higher-confidence box has a high IoU, the current box is suppressed.
 */
__global__ void nms_kernel_optimized(
    const float* __restrict__ boxes, // [x1, y1, x2, y2, conf, cls] ...
    bool* __restrict__ keep_mask,
    int count,
    float iou_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // By default, we keep the box
    keep_mask[idx] = true;

    // Check against all boxes with higher confidence
    for (int i = 0; i < idx; ++i) {
        if (box_iou(&boxes[idx * 6], &boxes[i * 6]) > iou_threshold) {
            keep_mask[idx] = false;
            break;
        }
    }
}

extern "C" void launch_nms(
    const float* d_boxes,
    bool* d_keep_mask,
    int count,
    float iou_threshold,
    cudaStream_t stream
) {
    if (count <= 0) return;
    int block = 256;
    int grid = (count + block - 1) / block;

    nms_kernel_optimized<<<grid, block, 0, stream>>>(d_boxes, d_keep_mask, count, iou_threshold);
}

} // namespace cuda
} // namespace atos
