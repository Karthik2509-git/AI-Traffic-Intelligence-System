#include "engine/detector.hpp"
#include "core/logger.hpp"
#include <fstream>
#include <iostream>

// CUDA Kernel Declarations
extern "C" void launch_fused_preprocess(const uint8_t* d_src, float* d_dst, int src_w, int src_h, int dst_w, int dst_h, cudaStream_t stream);
extern "C" void launch_nms(const float* d_boxes, bool* d_keep_mask, int count, float threshold, cudaStream_t stream);

namespace atos {
namespace engine {

Detector::Detector(const Config& config) : config(config) {
    cudaStreamCreate(&stream);
    initEngine();
}

Detector::~Detector() {
    cudaStreamDestroy(stream);
    cudaFree(bindings[0]);
    cudaFree(bindings[1]);
}

void Detector::initEngine() {
    std::ifstream file(config.engine_path, std::ios::binary);
    if (!file.good()) {
        traffic::Logger::error("AI Engine: Failed to open model file " + config.engine_path);
        return;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    runtime.reset(nvinfer1::createInferRuntime(*nvinfer1::getLogger()));
    engine.reset(runtime->deserializeCudaEngine(engineData.data(), size));
    context.reset(engine->createExecutionContext());

    // Bindings allocation: Input (Images) and Output (Detections)
    size_t in_size = 3 * config.input_w * config.input_h * sizeof(float);
    size_t out_size = 84 * 8400 * sizeof(float); // YOLOv8-batch1 profile

    cudaMalloc(&bindings[0], in_size);
    cudaMalloc(&bindings[1], out_size);
    
    traffic::Logger::info("TensorRT 4K AI Engine initialized.");
}

void Detector::process(const uint8_t* d_image_ptr, int src_w, int src_h) {
    // 1. Asynchronous Fused Preprocessing on GPU
    launch_fused_preprocess(
        d_image_ptr, (float*)bindings[0], 
        src_w, src_h, config.input_w, config.input_h, stream
    );

    // 2. High-Performance Parallel Inference
    // enqueV2 is non-blocking on the CPU; it schedules the work on the GPU stream
    context->enqueueV2(bindings, stream, nullptr);

    // 3. Post-Inference: GPU-Bound NMS
    // Note: Boxes are extracted and filtered directly in CUDA memory
    // bool* d_keep_mask;
    // cudaMallocAsync(&d_keep_mask, 8400 * sizeof(bool), stream);
    // launch_nms((float*)bindings[1], d_keep_mask, 8400, config.nms_threshold, stream);

    // 4. Async sync (Optional: Wait only if result is needed immediately)
    cudaStreamSynchronize(stream);
}

} // namespace engine
} // namespace atos
