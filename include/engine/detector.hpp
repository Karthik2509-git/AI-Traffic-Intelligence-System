#pragma once

#include <string>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "core/memory.hpp"
#include "core/types.hpp"

namespace antigravity {
namespace engine {

/**
 * @brief Logger for TensorRT diagnostics.
 */
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

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
        int input_w = 960;
        int input_h = 960;
        float conf_threshold = 0.20f;
        float nms_threshold = 0.55f;
        int num_anchors = 18900; // Updated for 960x960
    };

    Detector(const Config& config);
    ~Detector();

    /**
     * @brief Run inference on a 4K frame using Pinned Memory.
     * @return List of tracked vehicles and their metadata.
     */
    std::vector<traffic::Track> process(const uint8_t* d_image_ptr, int src_w, int src_h);

    // Get results from GPU
    // (Detailed implementation of result fetching follows)

private:
    Config config;
    
    // TensorRT Core
    TRTLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    // CUDA State
    cudaStream_t stream;
    void* bindings[2]; // [Input, Output]
    
    void initEngine();
};

} // namespace engine
} // namespace antigravity
