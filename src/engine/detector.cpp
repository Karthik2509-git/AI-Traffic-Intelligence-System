#include "engine/detector.hpp"
#include "core/logger.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

// CUDA Kernel Declaration
extern "C" void launch_fused_preprocess(const uint8_t* d_src, float* d_dst, int src_w, int src_h, int dst_w, int dst_h, cudaStream_t stream);

static const int NUM_VEHICLE_CLASSES = 4;
static const int NUM_COCO_CLASSES = 80;

namespace antigravity {
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
        traffic::Logger::error("Detector: Cannot open engine file: " + config.engine_path);
        return;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime) throw std::runtime_error("TensorRT: Failed to create runtime.");

    engine.reset(runtime->deserializeCudaEngine(engineData.data(), size));
    if (!engine) throw std::runtime_error("TensorRT: Failed to deserialize engine. File may be corrupt or built for a different GPU.");

    context.reset(engine->createExecutionContext());
    if (!context) throw std::runtime_error("TensorRT: Failed to create execution context.");

    // Fetch output dimensions from engine
    auto outShape = engine->getTensorShape("output0");
    if (outShape.nbDims < 2) throw std::runtime_error("TensorRT: Invalid output shape.");
    
    // YOLOv8 output: [1, 84, anchors]
    int num_anchors = outShape.d[2];
    config.num_anchors = num_anchors;

    size_t in_size = 3 * config.input_w * config.input_h * sizeof(float);
    size_t out_size = (4 + NUM_COCO_CLASSES) * num_anchors * sizeof(float);

    cudaMalloc(&bindings[0], in_size);
    cudaMalloc(&bindings[1], out_size);

    context->setTensorAddress("images", bindings[0]);
    context->setTensorAddress("output0", bindings[1]);

    traffic::Logger::info("Detector: TensorRT engine loaded (" + std::to_string(size / 1024) + " KB).");
}

std::vector<::traffic::Track> Detector::process(const uint8_t* d_image_ptr, int src_w, int src_h) {
    if (!d_image_ptr) {
        traffic::Logger::error("Detector: Null image pointer.");
        return {};
    }

    // --- Step 1: GPU Preprocessing (resize + BGR→RGB + normalize) ---
    launch_fused_preprocess(
        d_image_ptr, (float*)bindings[0],
        src_w, src_h, config.input_w, config.input_h, stream
    );

    // --- Step 2: TensorRT Inference ---
    if (!context->enqueueV3(stream)) {
        traffic::Logger::error("Detector: TensorRT inference failed.");
        return {};
    }
    cudaStreamSynchronize(stream);

    // --- Step 3: Copy output to host ---
    // YOLOv8 output layout: [84 x anchors] where 84 = 4 (box) + 80 (class confidences)
    std::vector<float> h_out((4 + NUM_COCO_CLASSES) * config.num_anchors);
    cudaMemcpy(h_out.data(), bindings[1], h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // --- Step 4: Parse detections with class-aware confidence ---
    struct RawDetection {
        float cx, cy, w, h, score;
        int classId;
    };
    std::vector<RawDetection> raw;

    static const int VEHICLE_CLASSES[] = {2, 3, 5, 7}; // car, motorcycle, bus, truck

    for (int i = 0; i < config.num_anchors; ++i) {
        // Box parameters (rows 0-3)
        float cx = h_out[0 * config.num_anchors + i];
        float cy = h_out[1 * config.num_anchors + i];
        float bw = h_out[2 * config.num_anchors + i];
        float bh = h_out[3 * config.num_anchors + i];

        // Find max class confidence across all 80 classes (rows 4-83)
        float maxConf = 0.0f;
        int maxClassId = 0;
        for (int c = 0; c < NUM_COCO_CLASSES; ++c) {
            float conf = h_out[(4 + c) * config.num_anchors + i];
            if (conf > maxConf) {
                maxConf = conf;
                maxClassId = c;
            }
        }

        // Filter: must exceed threshold AND be a vehicle class
        if (maxConf < config.conf_threshold) continue;

        bool isVehicle = false;
        for (int v = 0; v < NUM_VEHICLE_CLASSES; ++v) {
            if (maxClassId == VEHICLE_CLASSES[v]) { isVehicle = true; break; }
        }
        if (!isVehicle) continue;

        raw.push_back({cx, cy, bw, bh, maxConf, maxClassId});
    }

    // --- Step 5: CPU NMS (greedy IoU suppression) ---
    // Sort by confidence descending
    std::sort(raw.begin(), raw.end(), [](const RawDetection& a, const RawDetection& b) {
        return a.score > b.score;
    });

    std::vector<bool> suppressed(raw.size(), false);

    auto iou = [](const RawDetection& a, const RawDetection& b) -> float {
        float ax1 = a.cx - a.w / 2, ay1 = a.cy - a.h / 2;
        float ax2 = a.cx + a.w / 2, ay2 = a.cy + a.h / 2;
        float bx1 = b.cx - b.w / 2, by1 = b.cy - b.h / 2;
        float bx2 = b.cx + b.w / 2, by2 = b.cy + b.h / 2;

        float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
        float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
        float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
        float inter = iw * ih;
        float areaA = a.w * a.h, areaB = b.w * b.h;
        return inter / (areaA + areaB - inter + 1e-6f);
    };

    for (size_t i = 0; i < raw.size(); ++i) {
        if (suppressed[i]) continue;
        for (size_t j = i + 1; j < raw.size(); ++j) {
            if (suppressed[j]) continue;
            if (iou(raw[i], raw[j]) > config.nms_threshold) {
                suppressed[j] = true;
            }
        }
    }

    // --- Step 6: Build final tracks ---
    std::vector<::traffic::Track> tracks;
    float sx = static_cast<float>(src_w) / config.input_w;
    float sy = static_cast<float>(src_h) / config.input_h;

    for (size_t i = 0; i < raw.size() && tracks.size() < 64; ++i) {
        if (suppressed[i]) continue;

        ::traffic::Track t;
        t.id = static_cast<int>(tracks.size());
        t.classId = raw[i].classId;
        t.confidence = raw[i].score;
        t.bbox = cv::Rect(
            static_cast<int>((raw[i].cx - raw[i].w / 2) * sx),
            static_cast<int>((raw[i].cy - raw[i].h / 2) * sy),
            static_cast<int>(raw[i].w * sx),
            static_cast<int>(raw[i].h * sy)
        );
        tracks.push_back(t);
    }

    return tracks;
}

} // namespace engine
} // namespace antigravity
