#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

namespace traffic {
namespace cuda {

/**
 * @brief CUDA kernel for image normalization and BGR -> RGB conversion.
 * 
 * Performs:
 * 1. Pixel normalization (0-255 -> 0.0-1.0)
 * 2. BGR to RGB channel swapping
 * 3. Layout conversion (HWC to CHW) for TensorRT input
 */
__global__ void preprocess_kernel(
    const uint8_t* src, 
    float* dst, 
    int width, 
    int height, 
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        int dst_area = width * height;

        // BGR to RGB and Normalize
        // TensorRT expects CHW format: [R, G, B]
        dst[0 * dst_area + y * width + x] = (float)src[idx + 2] / 255.0f; // R
        dst[1 * dst_area + y * width + x] = (float)src[idx + 1] / 255.0f; // G
        dst[2 * dst_area + y * width + x] = (float)src[idx + 0] / 255.0f; // B
    }
}

extern "C" void launch_preprocess(
    const uint8_t* d_src, 
    float* d_dst, 
    int width, 
    int height, 
    cudaStream_t stream
) {
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    preprocess_kernel<<<grid, block, 0, stream>>>(d_src, d_dst, width, height, 3);
}

} // namespace cuda
} // namespace traffic
