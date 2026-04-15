#include "core/logger.hpp"
#include "core/concurrent_queue.hpp"
#include "core/memory.hpp"
#include "core/types.hpp"
#include "core/stream_manager.hpp"
#include "engine/detector.hpp"
#include "network/city_controller.hpp"
#include "simulation/digital_twin.hpp"

#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <thread>
#include <atomic>

/**
 * @brief Antigravity Traffic Omni-System (ATOS) - Production Pipeline
 * Version: 2.0 (Autonomous Global Intelligence Edition)
 */

std::atomic<bool> g_running{true};

struct PipelineFrame {
    int streamId;
    uint64_t frameIndex;
    std::shared_ptr<::antigravity::core::PinnedBuffer<uint8_t>> buffer;
    cv::Mat frame; 
    int width, height;
    std::chrono::steady_clock::time_point timestamp;
    std::vector<::traffic::Track> results;
};

// Global High-Throughput Buffers
::antigravity::core::ConcurrentQueue<std::shared_ptr<PipelineFrame>> g_inferenceQueue(32);

// Global Intelligence Nodes
std::shared_ptr<::antigravity::network::CityController> g_cityController;
std::shared_ptr<::antigravity::simulation::DigitalTwinBridge> g_twinBridge;

void captureWorker(int streamId, std::string source) {
    ::traffic::Logger::info("CaptureWorker: Attempting connection to " + source);
    uint64_t frameIndex = 0;

    while (g_running) {
        cv::VideoCapture cap(source);
        if (!cap.isOpened()) {
            ::traffic::Logger::error("CaptureWorker: Failed to connect to source. Retrying in 2 seconds...");
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }

        while (g_running) {
            auto pFrame = std::make_shared<PipelineFrame>();
            pFrame->streamId = streamId;
            pFrame->frameIndex = frameIndex++;
            pFrame->timestamp = std::chrono::steady_clock::now();

            cv::Mat temp;
            if (!cap.read(temp)) {
                ::traffic::Logger::warn("CaptureWorker: Stream Interrupted. Reconnecting...");
                break;
            }

            pFrame->width = temp.cols;
            pFrame->height = temp.rows;

            // Zero-Copy Memory Allocation (Pinned Memory for Direct GPU Access)
            pFrame->buffer = std::make_shared<::antigravity::core::PinnedBuffer<uint8_t>>(pFrame->width * pFrame->height * 3);
            pFrame->frame = cv::Mat(pFrame->height, pFrame->width, CV_8UC3, pFrame->buffer->get());
            temp.copyTo(pFrame->frame);

            g_inferenceQueue.push(std::move(pFrame));
        }
    }
}

void digitalTwinSyncWorker() {
    while (g_running) {
        if (g_cityController && g_twinBridge) {
            float pressure = g_cityController->getGlobalPressure();
            g_twinBridge->syncState(pressure, 0); // Fixed ID for demo
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 10Hz Heartbeat
    }
}

int main(int argc, char** argv) {
    ::traffic::Logger::info("ATOS 2.0: Initializing Autonomous Intelligence Node...");

    std::string videoSource = "data/test_4k_traffic.mp4"; // Default
    if (argc > 1) {
        videoSource = argv[1];
        ::traffic::Logger::info("Live Mode: Targeting Mobile Stream -> " + videoSource);
    }

    try {
        // 1. Setup City Strategy & 3. Initialize Global Intelligence Modules
        auto graph = std::make_shared<::antigravity::network::RoadGraph>();
        graph->addCameraNode(0, "Main Intersection 4K-Alpha");
        graph->addRoadConnection(0, 1, 500.0f); // Seed 500m exit segment for density tracking

        auto signals = std::make_shared<::antigravity::control::RLSignalController>("data/ppo_policy_4k.onnx");
        g_cityController = std::make_shared<::antigravity::network::CityController>(graph, signals);
        
        ::antigravity::simulation::DigitalTwinBridge::Config twinConfig;
        twinConfig.target_ip = "127.0.0.1";
        twinConfig.target_port = 5005;
        g_twinBridge = std::make_shared<::antigravity::simulation::DigitalTwinBridge>(twinConfig);

        // 3. Setup GPU Inference Engine (RTX 50-series optimized)
        ::antigravity::engine::Detector::Config detConfig;
        detConfig.engine_path = "data/yolov8_4k_optimized.engine";
        ::antigravity::engine::Detector detector(detConfig);

        // 4. Launch Support Threads
        std::thread t1(captureWorker, 0, videoSource);
        std::thread syncThread(digitalTwinSyncWorker);

        ::traffic::Logger::info("ATOS 2.0 Operational. High-Performance Autonomous Intelligence Active.");

        // 5. Main Autonomous AI Loop
        while (g_running) {
            std::shared_ptr<PipelineFrame> pFrame;
            if (g_inferenceQueue.pop(pFrame)) {
                // Perform High-Speed 4K Inference using Zero-Copy Device Pointer
                pFrame->results = detector.process(pFrame->buffer->getDevicePtr(), pFrame->width, pFrame->height);

                // Update the City Brain with new tracks
                g_cityController->updateTracks(pFrame->results);

                auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - pFrame->timestamp).count();
                
                ::traffic::Logger::info("Stream " + std::to_string(pFrame->streamId) + " | Seq: " + std::to_string(pFrame->frameIndex) + 
                         " | Latency: " + std::to_string(latency) + "ms | Global Pressure: " + 
                         std::to_string(g_cityController->getGlobalPressure()));
            }
        }

        t1.join();
        syncThread.join();

    } catch (const std::exception& e) {
        ::traffic::Logger::error("ATOS 2.0 Fatal Crash: " + std::string(e.what()));
        return -1;
    }

    return 0;
}
