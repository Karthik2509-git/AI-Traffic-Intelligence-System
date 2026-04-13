#pragma once

#include <string>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "core/types.hpp"

namespace traffic {

/**
 * @brief Logger for TensorRT engine.
 */
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override;
};

/**
 * @brief High-performance YOLOv8 detection engine using TensorRT.
 */
class Detector {
public:
    Detector(const std::string& enginePath);
    ~Detector();

    /**
     * @brief Run inference on a GPU frame.
     * @param d_input_ptr Pointer to the preprocessed GPU float buffer (NCHW).
     * @return Vector of detections.
     */
    std::vector<Track> detect(float* d_input_ptr);

    /**
     * @brief Run slicing-aided hyper inference (SAHI) for high accuracy.
     * @param frame Full-sized input frame.
     * @return Consolidated detections from all slices.
     */
    std::vector<Track> detectTiled(const cv::Mat& frame, int tile_size = 640, float overlap = 0.25f);

private:
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    
    cudaStream_t stream;
    void* buffers[2]; // Input and Output pointers on GPU
    
    int inputIndex;
    int outputIndex;
    int batchSize = 1;

    // Output buffer size calculation attributes
    size_t inputSize;
    size_t outputSize;
    float* h_outputPtr; // Host buffer for outputs

    void initEngine(const std::string& enginePath);
};

} // namespace traffic
