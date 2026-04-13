#include "detector.hpp"
#include "core/logger.hpp"
#include <fstream>
#include <iostream>

namespace traffic {

// ---------------------------------------------------------------------------
// TRTLogger
// ---------------------------------------------------------------------------

void TRTLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        if (severity == Severity::kINTERNAL_ERROR || severity == Severity::kERROR) {
            Logger::error(std::string("[TensorRT] ") + msg);
        } else {
            Logger::warn(std::string("[TensorRT] ") + msg);
        }
    }
}

static TRTLogger gLogger;

// ---------------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------------

Detector::Detector(const std::string& enginePath) {
    cudaStreamCreate(&stream);
    initEngine(enginePath);
}

Detector::~Detector() {
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    delete[] h_outputPtr;
}

void Detector::initEngine(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        Logger::error("Cannot open engine file: " + enginePath);
        return;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    runtime.reset(nvinfer1::createInferRuntime(gLogger));
    engine.reset(runtime->deserializeCudaEngine(engineData.data(), size));
    context.reset(engine->createExecutionContext());

    inputIndex = engine->getBindingIndex("images");
    outputIndex = engine->getBindingIndex("output0");

    // Allocation (Mocked dimensions for YOLOv8n 640x640)
    // In production, these should be queried from the engine metadata
    inputSize = batchSize * 3 * 640 * 640 * sizeof(float);
    outputSize = batchSize * 84 * 8400 * sizeof(float); // YOLOv8 output: [batch, classes+bbox, anchors]

    cudaMalloc(&buffers[inputIndex], inputSize);
    cudaMalloc(&buffers[outputIndex], outputSize);
    h_outputPtr = new float[outputSize / sizeof(float)];

    Logger::info("TensorRT Engine initialized from: " + enginePath);
}

std::vector<Track> Detector::detect(float* d_input_ptr) {
    // 1. Copy preprocessed input to GPU buffer
    cudaMemcpyAsync(buffers[inputIndex], d_input_ptr, inputSize, cudaMemcpyDeviceToDevice, stream);

    // 2. Execute Inference
    context->enqueueV2(buffers, stream, nullptr);

    // 3. Copy results back to Host
    cudaMemcpyAsync(h_outputPtr, buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 4. Post-processing (Placeholder: Logic to parse YOLOv8 boxes)
    std::vector<Track> detections;
    // TODO: Implement box parsing, NMS, and class filtering here
    
    return detections;
}

std::vector<Track> Detector::detectTiled(const cv::Mat& frame, int tile_size, float overlap) {
    int h = frame.rows;
    int w = frame.cols;
    int stride = static_cast<int>(tile_size * (1.0f - overlap));

    std::vector<Track> all_detections;

    for (int y = 0; y <= h - tile_size; y += stride) {
        for (int x = 0; x <= w - tile_size; x += stride) {
            cv::Mat tile = frame(cv::Rect(x, y, tile_size, tile_size));
            
            // 1. In a real implementation, we would launch the CUDA preprocess kernel on this tile
            // 2. Run inference: detector->detect(...)
            std::vector<Track> tile_detections = detect(nullptr); // Placeholder
            
            // 3. Map tile detections back to global frame coordinates
            for (auto& det : tile_detections) {
                det.bbox.x += x;
                det.bbox.y += y;
                all_detections.push_back(det);
            }
        }
    }

    // 4. Final step: Global NMS (Non-Maximum Suppression)
    // Here we merge boxes that detected the same vehicle in overlapping slices
    Logger::info("Tiled Inference complete. Consolidated " + std::to_string(all_detections.size()) + " raw detection slices.");
    
    return all_detections;
}

} // namespace traffic
