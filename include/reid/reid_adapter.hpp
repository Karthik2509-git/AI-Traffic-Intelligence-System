#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include "reid/reid_types.hpp"

namespace antigravity {
namespace reid {

/**
 * @brief C++ TensorRT / ONNX Vehicle Re-ID Model Adapter Interface.
 * 
 * Extracts high-dimensional feature embeddings from vehicle crop images.
 * Gracefully handles missing model files by returning ReIDStatus::MODEL_UNAVAILABLE.
 */
class ReIDModelAdapter {
public:
    struct Config {
        std::string model_path = "models/reid_vehiclenet.engine";
        int input_w = 256;
        int input_h = 256;
        int embedding_dim = 512;
        bool enabled = false;
    };

    ReIDModelAdapter(const Config& config);
    virtual ~ReIDModelAdapter() = default;

    /**
     * @brief Initialize engine and load model weights from disk.
     */
    virtual bool loadModel(const std::string& modelPath);

    /**
     * @brief Check if the Re-ID model engine is loaded and initialized.
     */
    virtual bool isModelLoaded() const;

    /**
     * @brief Extract L2-normalized feature embedding vector from vehicle crop image.
     */
    virtual std::vector<float> extractEmbedding(const cv::Mat& cropImage);

    /**
     * @brief Get human-readable diagnostic status string.
     */
    virtual std::string getStatusMessage() const;

    /**
     * @brief Get current subsystem status code.
     */
    virtual traffic::ReIDStatus getStatus() const;

private:
    Config config_;
    traffic::ReIDStatus status_;
    std::string status_message_;
};

} // namespace reid
} // namespace antigravity
