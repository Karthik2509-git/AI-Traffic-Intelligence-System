#pragma once

#include <string>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "core/memory.hpp"

namespace atos {
namespace engine {

/**
 * @brief High-Performance TensorRT Detector.
 * 
 * Optimized for multi-stream 4K processing. Uses asynchronous CUDA streams 
 * and fused kernels for zero-latency detection.
 */
class Detector {
public:
    struct Config {
        std::string engine_path;
        int input_w = 640;
        int input_h = 640;
        float conf_threshold = 0.45f;
        float nms_threshold = 0.50f;
    };

    Detector(const Config& config);
    ~Detector();

    /**
     * @brief Run inference on a 4K frame using Pinned Memory.
     */
    void process(const uint8_t* d_image_ptr, int src_w, int src_h);

    // Get results from GPU
    // (Detailed implementation of result fetching follows)

private:
    Config config;
    
    // TensorRT Core
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    // CUDA State
    cudaStream_t stream;
    void* bindings[2]; // [Input, Output]
    
    void initEngine();
};

} // namespace engine
} // namespace atos
