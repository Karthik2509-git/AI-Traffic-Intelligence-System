#pragma once

#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <NvInfer.h>
#include "core/memory.hpp"

namespace atos {
namespace network {

/**
 * @brief Signature representing a unique vehicle "fingerprint".
 */
using VehicleSignature = std::vector<float>;

/**
 * @brief Siamese Feature Extractor for Cross-Camera Re-ID.
 * 
 * Uses a lightweight CNN (e.g., OSNet or ResNet-50) to generate 
 * embeddings for vehicle images, facilitating global tracking.
 */
class ReIDEngine {
public:
    struct Config {
        std::string engine_path;
        int input_w = 128; // Standard Re-ID input size
        int input_h = 256;
    };

    ReIDEngine(const Config& config);
    ~ReIDEngine();

    /**
     * @brief Generate a signature for a vehicle crop.
     */
    VehicleSignature extractSignature(const cv::Mat& vehicleCrop);

    /**
     * @brief Compare two signatures and return a similarity score (0.0 - 1.0).
     */
    float compare(const VehicleSignature& s1, const VehicleSignature& s2);

private:
    Config config;
    
    // TensorRT Core (Re-ID models are usually separate from Detectors)
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    void initEngine();
};

} // namespace network
} // namespace atos
