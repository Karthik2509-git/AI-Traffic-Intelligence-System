#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

namespace atos {
namespace cuda {

/**
 * @brief Simple Check for Intersection over Union (IoU) on Device.
 */
__device__ inline float calculate_iou(const float* a, const float* b) {
    float left = max(a[0], b[0]), top = max(a[1], b[1]);
    float right = min(a[2], b[2]), bottom = min(a[3], b[3]);
    float width = max(right - left, 0.f), height = max(bottom - top, 0.f);
    float inter_area = width * height;
    if (inter_area == 0) return 0;
    float area_a = (a[2] - a[0]) * (a[3] - a[1]);
    float area_b = (b[2] - b[0]) * (b[3] - b[1]);
    return inter_area / (area_a + area_b - inter_area);
}

/**
 * @brief CUDA Kernel for filtering boxes via NMS.
 * 
 * Uses a simplified parallel suppression approach. This is significantly 
 * faster than CPU-based NMS for high-density vehicle detection.
 */
__global__ void nms_kernel(
    const float* boxes,   // [x1, y1, x2, y2, conf, cls] ...
    bool* mask,          // Output mask: true if box is kept
    int count, 
    float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // Greedy Parallel Suppression
    // For every box, check all higher-confidence boxes
    for (int i = 0; i < idx; ++i) {
        if (calculate_iou(&boxes[idx * 6], &boxes[i * 6]) > threshold) {
            mask[idx] = false;
            return;
        }
    }
    mask[idx] = true;
}

extern "C" void launch_nms(
    const float* d_boxes, 
    bool* d_mask, 
    int count, 
    float threshold,
    cudaStream_t stream
) {
    if (count <= 0) return;
    int block = 256;
    int grid = (count + block - 1) / block;

    nms_kernel<<<grid, block, 0, stream>>>(d_boxes, d_mask, count, threshold);
}

} // namespace cuda
} // namespace atos
