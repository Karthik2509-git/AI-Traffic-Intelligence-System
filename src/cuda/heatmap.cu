#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace traffic {
namespace cuda {

/**
 * @brief CUDA kernel for heatmap accumulation.
 * 
 * Instead of drawing circles on the CPU, this kernel increments pixel values
 * in a high-precision accumulation buffer based on vehicle centers.
 */
__global__ void heatmap_update_kernel(
    float* accumulator, 
    int width, 
    int height, 
    int cx, 
    int cy, 
    int radius, 
    float intensity
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float dx = (float)x - cx;
        float dy = (float)y - cy;
        float dist_sq = dx * dx + dy * dy;

        if (dist_sq <= radius * radius) {
            // Use atomicAdd to prevent race conditions when multiple 
            // threads update the same pixel (though unlikely with centers)
            atomicAdd(&accumulator[y * width + x], intensity);
        }
    }
}

/**
 * @brief CUDA kernel for temporal decay.
 */
__global__ void heatmap_decay_kernel(
    float* accumulator, 
    int width, 
    int height, 
    float decay_factor
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        accumulator[idx] *= decay_factor;
    }
}

extern "C" void launch_heatmap_update(
    float* d_accumulator, 
    int width, 
    int height, 
    int cx, 
    int cy, 
    int radius, 
    float intensity,
    cudaStream_t stream
) {
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    heatmap_update_kernel<<<grid, block, 0, stream>>>(d_accumulator, width, height, cx, cy, radius, intensity);
}

extern "C" void launch_heatmap_decay(
    float* d_accumulator, 
    int width, 
    int height, 
    float decay_factor,
    cudaStream_t stream
) {
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    heatmap_decay_kernel<<<grid, block, 0, stream>>>(d_accumulator, width, height, decay_factor);
}

} // namespace cuda
} // namespace traffic
