#include "core/logger.hpp"
#include "core/concurrent_queue.hpp"
#include "core/memory.hpp"
#include "core/stream_manager.hpp"
#include "core/thread_pool.hpp"
#include "engine/detector.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <thread>
#include <atomic>

using namespace atos::core;
using namespace atos::engine;

/**
 * @brief Antigravity Traffic Omni-System (ATOS) - Production Pipeline
 * 
 * Version: 1.0 (Master-Class Implementation)
 * Architecture: Asynchronous Multi-camera Data-Parallel Pipeline
 */

std::atomic<bool> g_running{true};

// ---------------------------------------------------------------------------
// Pipeline Structs
// ---------------------------------------------------------------------------
struct PipelineFrame {
    int streamId;
    std::shared_ptr<PinnedBuffer<uint8_t>> buffer;
    cv::Mat frame; // Wrapper for the pinned buffer
    int width, height;
    std::chrono::steady_clock::time_point timestamp;
};

// ---------------------------------------------------------------------------
// Global Queues (High-Throughput)
// ---------------------------------------------------------------------------
ConcurrentQueue<std::shared_ptr<PipelineFrame>> g_inferenceQueue(32);
ConcurrentQueue<std::shared_ptr<PipelineFrame>> g_analyticsQueue(32);

// ---------------------------------------------------------------------------
// Capture Worker (Producer)
// ---------------------------------------------------------------------------
void captureWorker(int streamId, std::string source) {
    auto& logger = traffic::Logger::getInstance();
    cv::VideoCapture cap(source);
    if (!cap.isOpened()) {
        logger.error("CaptureWorker: Failed to connect to " + source);
        return;
    }

    while (g_running) {
        auto pFrame = std::make_shared<PipelineFrame>();
        pFrame->streamId = streamId;
        pFrame->timestamp = std::chrono::steady_clock::now();

        cv::Mat temp;
        if (!cap.read(temp)) break;

        // Optimized Path: Use Pinned Memory for Zero-Copy H2D transfer
        pFrame->width = temp.cols;
        pFrame->height = temp.rows;
        pFrame->buffer = std::make_shared<PinnedBuffer<uint8_t>>(pFrame->width * pFrame->height * 3);
        
        // Wrap pinned memory in cv::Mat and Copy
        pFrame->frame = cv::Mat(pFrame->height, pFrame->width, CV_8UC3, pFrame->buffer->get());
        temp.copyTo(pFrame->frame);

        g_inferenceQueue.push(std::move(pFrame));
    }
}

// ---------------------------------------------------------------------------
// Main Orchestrator
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    auto& logger = traffic::Logger::getInstance();
    logger.info("ATOS Initializing: World-Class Traffic Intelligence Pipeline...");

    try {
        // 1. Setup GPU Engine
        Detector::Config detConfig;
        detConfig.engine_path = "data/yolov8_4k_optimized.engine";
        Detector detector(detConfig);

        // 2. Setup Multi-Camera Management
        auto& sm = StreamManager::getInstance();
        int s1 = sm.addStream("rtsp://192.168.1.10:8080/live"); // Cam 1
        int s2 = sm.addStream("rtsp://192.168.1.11:8080/live"); // Cam 2

        // 3. Launch Capture Threads
        std::thread t1(captureWorker, s1, "data/test_4k_traffic.mp4");
        std::thread t2(captureWorker, s2, "data/test_4k_highway.mp4");

        // 4. Main AI Inference Loop (The GPU Orchestrator)
        while (g_running) {
            std::shared_ptr<PipelineFrame> pFrame;
            if (g_inferenceQueue.pop(pFrame)) {
                auto start = std::chrono::steady_clock::now();

                // Advanced Phase 2 Call: Fused Kernel + TensorRT
                detector.process(pFrame->buffer->get(), pFrame->width, pFrame->height);

                auto end = std::chrono::steady_clock::now();
                auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - pFrame->timestamp).count();
                
                if (latency > 33) {
                   logger.warn("Pipeline Latency Spiked: " + std::to_string(latency) + "ms");
                }

                g_analyticsQueue.push(std::move(pFrame));
            }
        }

        t1.join();
        t2.join();

    } catch (const std::exception& e) {
        logger.error("ATOS Fatal Crash: " + std::string(e.what()));
        return -1;
    }

    return 0;
}
