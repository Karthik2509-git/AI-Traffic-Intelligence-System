#include "core/logger.hpp"
#include "core/concurrent_queue.hpp"
#include "core/memory.hpp"
#include "core/types.hpp"
#include "engine/detector.hpp"
#include "network/city_controller.hpp"
#include "simulation/digital_twin.hpp"

#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <thread>
#include <atomic>
#include <fstream>
#include <csignal>
#include <numeric>
#include <deque>
#include <iomanip>
#include <sstream>

// =========================================================================
// ATOS v3.0 — Validated Traffic Intelligence Engine
//
// Pipeline: Capture → Pinned Memory → TensorRT → Analytics → Annotate → Telemetry
// =========================================================================

std::atomic<bool> g_running{true};

// --- Signal handler for graceful Ctrl+C shutdown ---
void signalHandler(int) {
    g_running = false;
}

// --- COCO class name lookup ---
static const char* getClassName(int classId) {
    switch (classId) {
        case 2:  return "car";
        case 3:  return "motorcycle";
        case 5:  return "bus";
        case 7:  return "truck";
        default: return "vehicle";
    }
}

// --- Color per class for bounding boxes ---
static cv::Scalar getClassColor(int classId) {
    switch (classId) {
        case 2:  return cv::Scalar(0, 255, 0);     // car: green
        case 3:  return cv::Scalar(255, 165, 0);   // motorcycle: orange
        case 5:  return cv::Scalar(255, 0, 0);     // bus: blue
        case 7:  return cv::Scalar(0, 0, 255);     // truck: red
        default: return cv::Scalar(200, 200, 200); // gray
    }
}

struct PipelineFrame {
    int streamId;
    uint64_t frameIndex;
    std::shared_ptr<::antigravity::core::PinnedBuffer<uint8_t>> buffer;
    cv::Mat frame;
    int width, height;
    std::chrono::steady_clock::time_point timestamp;
    std::vector<::traffic::Track> results;
};

::antigravity::core::ConcurrentQueue<std::shared_ptr<PipelineFrame>> g_inferenceQueue(32);

std::shared_ptr<::antigravity::network::CityController> g_cityController;
std::shared_ptr<::antigravity::simulation::DigitalTwinBridge> g_twinBridge;

// =========================================================================
// Capture thread
// =========================================================================
void captureWorker(int streamId, std::string source) {
    ::traffic::Logger::info("Capture: Connecting to " + source);
    uint64_t frameIndex = 0;

    while (g_running) {
        cv::VideoCapture cap(source);
        if (!cap.isOpened()) {
            ::traffic::Logger::error("Capture: Cannot open source. Retrying in 2s...");
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }
        ::traffic::Logger::info("Capture: Stream opened (" +
            std::to_string((int)cap.get(cv::CAP_PROP_FRAME_WIDTH)) + "x" +
            std::to_string((int)cap.get(cv::CAP_PROP_FRAME_HEIGHT)) + " @ " +
            std::to_string((int)cap.get(cv::CAP_PROP_FPS)) + " fps).");

        while (g_running) {
            auto pFrame = std::make_shared<PipelineFrame>();
            pFrame->streamId = streamId;
            pFrame->frameIndex = frameIndex++;
            pFrame->timestamp = std::chrono::steady_clock::now();

            cv::Mat temp;
            if (!cap.read(temp)) {
                ::traffic::Logger::info("Capture: End of stream after " +
                    std::to_string(frameIndex) + " frames.");
                g_running = false;
                break;
            }

            pFrame->width = temp.cols;
            pFrame->height = temp.rows;

            pFrame->buffer = std::make_shared<::antigravity::core::PinnedBuffer<uint8_t>>(
                pFrame->width * pFrame->height * 3);
            pFrame->frame = cv::Mat(pFrame->height, pFrame->width, CV_8UC3, pFrame->buffer->get());
            temp.copyTo(pFrame->frame);

            g_inferenceQueue.push(std::move(pFrame));
        }
    }

    // Signal the queue to unblock the main loop
    g_inferenceQueue.stop();
}

// =========================================================================
// Telemetry thread
// =========================================================================
void telemetryWorker() {
    while (g_running) {
        if (g_cityController && g_twinBridge) {
            float pressure = g_cityController->getGlobalPressure();
            int count = g_cityController->getVehicleCount();
            g_twinBridge->syncState(pressure, 0, count);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// =========================================================================
// Draw detections on frame
// =========================================================================
void annotateFrame(cv::Mat& frame, const std::vector<::traffic::Track>& tracks,
                   uint64_t frameIdx, double fps) {
    for (const auto& t : tracks) {
        cv::Scalar color = getClassColor(t.classId);

        // Bounding box
        cv::rectangle(frame, t.bbox, color, 2);

        // Label: "car 0.87"
        std::stringstream ss;
        ss << getClassName(t.classId) << " " << std::fixed << std::setprecision(2) << t.confidence;
        std::string label = ss.str();

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::Point labelPos(t.bbox.x, t.bbox.y - 5);
        if (labelPos.y < 15) labelPos.y = t.bbox.y + 15;

        // Label background
        cv::rectangle(frame,
            cv::Point(labelPos.x, labelPos.y - textSize.height - 2),
            cv::Point(labelPos.x + textSize.width + 2, labelPos.y + 2),
            color, cv::FILLED);
        cv::putText(frame, label, labelPos, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0), 1);
    }

    // HUD: frame index + FPS + detection count
    std::stringstream hud;
    hud << "Frame: " << frameIdx
        << "  |  Det: " << tracks.size()
        << "  |  FPS: " << std::fixed << std::setprecision(1) << fps;

    cv::putText(frame, hud.str(), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
}

// =========================================================================
// Entry point
// =========================================================================
int main(int argc, char** argv) {
    // Graceful shutdown on Ctrl+C
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    ::traffic::Logger::info("ATOS v3.0 starting.");

    std::string videoSource = "data/test_4k_traffic.mp4";
    if (argc > 1) {
        videoSource = argv[1];
        ::traffic::Logger::info("Source: " + videoSource);
    }

    // Ensure output directory exists
    std::system("if not exist output mkdir output");

    try {
        // --- Initialize subsystems ---
        auto graph = std::make_shared<::antigravity::network::RoadGraph>();
        graph->addCameraNode(0, "Intersection-Alpha");
        graph->addRoadConnection(0, 1, 500.0f);

        auto signals = std::make_shared<::antigravity::control::SignalController>();
        g_cityController = std::make_shared<::antigravity::network::CityController>(graph, signals);

        ::antigravity::simulation::DigitalTwinBridge::Config twinConfig;
        twinConfig.target_ip = "127.0.0.1";
        twinConfig.target_port = 5005;
        g_twinBridge = std::make_shared<::antigravity::simulation::DigitalTwinBridge>(twinConfig);

        ::antigravity::engine::Detector::Config detConfig;
        detConfig.engine_path = "data/yolov8_4k_optimized.engine";
        ::antigravity::engine::Detector detector(detConfig);

        // --- Launch threads ---
        std::thread captureThread(captureWorker, 0, videoSource);
        std::thread telemetryThread(telemetryWorker);

        ::traffic::Logger::info("ATOS v3.0 operational. Processing...");

        // --- Metrics tracking ---
        std::ofstream metricsFile("output/metrics.csv");
        metricsFile << "frame,detections,latency_ms,fps_instant,fps_avg\n";

        auto globalStart = std::chrono::steady_clock::now();
        uint64_t totalFrames = 0;
        int totalDetections = 0;
        double latencySum = 0.0;
        double latencyMin = 1e9, latencyMax = 0.0;
        std::deque<double> fpsWindow;  // rolling window for instant FPS
        const size_t FPS_WINDOW = 30;

        cv::VideoWriter videoWriter;
        bool videoInitialized = false;

        // --- Main inference loop ---
        while (g_running) {
            std::shared_ptr<PipelineFrame> pFrame;
            if (!g_inferenceQueue.pop(pFrame)) break; // queue stopped

            // Inference
            pFrame->results = detector.process(
                pFrame->buffer->getDevicePtr(), pFrame->width, pFrame->height);

            // Analytics
            g_cityController->updateTracks(pFrame->results);

            // Latency
            auto now = std::chrono::steady_clock::now();
            double latency = std::chrono::duration<double, std::milli>(now - pFrame->timestamp).count();

            // FPS tracking
            totalFrames++;
            totalDetections += static_cast<int>(pFrame->results.size());
            latencySum += latency;
            if (latency < latencyMin) latencyMin = latency;
            if (latency > latencyMax) latencyMax = latency;

            double elapsed = std::chrono::duration<double>(now - globalStart).count();
            double globalFps = (elapsed > 0) ? totalFrames / elapsed : 0.0;

            // Rolling FPS (inverse of per-frame time)
            double instantFps = (latency > 0) ? 1000.0 / latency : 0.0;
            fpsWindow.push_back(instantFps);
            if (fpsWindow.size() > FPS_WINDOW) fpsWindow.pop_front();
            double rollingFps = 0.0;
            if (!fpsWindow.empty()) {
                rollingFps = std::accumulate(fpsWindow.begin(), fpsWindow.end(), 0.0) / fpsWindow.size();
            }

            // Write metrics CSV row
            metricsFile << pFrame->frameIndex << ","
                        << pFrame->results.size() << ","
                        << std::fixed << std::setprecision(1) << latency << ","
                        << std::setprecision(1) << rollingFps << ","
                        << std::setprecision(1) << globalFps << "\n";

            // --- Annotated output ---
            annotateFrame(pFrame->frame, pFrame->results, pFrame->frameIndex, rollingFps);

            // Initialize video writer on first frame
            if (!videoInitialized) {
                videoWriter.open("output/output_video.mp4",
                    cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                    25.0, cv::Size(pFrame->width, pFrame->height));
                videoInitialized = videoWriter.isOpened();
                if (!videoInitialized) {
                    ::traffic::Logger::warn("VideoWriter: Could not open. Falling back to frame output.");
                }
            }

            if (videoInitialized) {
                videoWriter.write(pFrame->frame);
            }

            // Save sample frames (every 30th frame)
            if (pFrame->frameIndex % 30 == 0) {
                std::string framePath = "output/annotated_" +
                    std::to_string(pFrame->frameIndex) + ".jpg";
                cv::imwrite(framePath, pFrame->frame);
            }

            // Console log (every 10th frame to avoid spam)
            if (pFrame->frameIndex % 10 == 0) {
                ::traffic::Logger::info(
                    "Seq:" + std::to_string(pFrame->frameIndex) +
                    " | Det:" + std::to_string(pFrame->results.size()) +
                    " | " + std::to_string((int)latency) + "ms" +
                    " | FPS:" + std::to_string((int)rollingFps) +
                    " | Pressure:" + std::to_string(g_cityController->getGlobalPressure()));
            }
        }

        // --- Shutdown ---
        ::traffic::Logger::info("Shutting down...");
        g_running = false;
        g_inferenceQueue.stop();

        captureThread.join();
        telemetryThread.join();

        if (videoInitialized) videoWriter.release();
        metricsFile.close();

        // --- Write benchmark report ---
        double totalElapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - globalStart).count();
        double avgFps = (totalElapsed > 0) ? totalFrames / totalElapsed : 0.0;
        double avgLatency = (totalFrames > 0) ? latencySum / totalFrames : 0.0;

        std::ofstream report("output/benchmark_report.txt");
        report << "========================================================\n";
        report << "  ATOS v3.0 — Performance Benchmark Report\n";
        report << "========================================================\n\n";

        report << "Source: " << videoSource << "\n";
        report << "Total Frames: " << totalFrames << "\n";
        report << "Total Time: " << std::fixed << std::setprecision(2) << totalElapsed << " s\n";
        report << "Total Detections: " << totalDetections << "\n\n";

        report << "--- Performance Metrics ---\n";
        report << "Average FPS:     " << std::setprecision(2) << avgFps << "\n";
        report << "Average Latency: " << std::setprecision(1) << avgLatency << " ms\n";
        report << "Min Latency:     " << std::setprecision(1) << latencyMin << " ms\n";
        report << "Max Latency:     " << std::setprecision(1) << latencyMax << " ms\n\n";

        report << "--- v1 vs v2 Comparison ---\n";
        report << "+------------------+------------------+------------------+\n";
        report << "| Metric           | v1 (Python)      | v3 (C++/TRT)     |\n";
        report << "+------------------+------------------+------------------+\n";
        report << "| FPS              | 6.43             | "
               << std::setw(16) << std::left << std::setprecision(2) << avgFps << " |\n";
        report << "| Avg Latency (ms) | 155.0            | "
               << std::setw(16) << std::left << std::setprecision(1) << avgLatency << " |\n";
        double speedup = (avgFps > 0) ? avgFps / 6.43 : 0.0;
        report << "| Speedup          | 1.0x             | "
               << std::setw(16) << std::left << std::setprecision(1) << speedup << "x |\n";
        report << "+------------------+------------------+------------------+\n\n";

        report << "--- Output Artifacts ---\n";
        report << "Annotated video: output/output_video.mp4\n";
        report << "Annotated frames: output/annotated_*.jpg\n";
        report << "Metrics CSV: output/metrics.csv\n";
        report.close();

        ::traffic::Logger::info("Benchmark report written to output/benchmark_report.txt");
        ::traffic::Logger::info("Final: " + std::to_string(totalFrames) + " frames, " +
            std::to_string((int)avgFps) + " FPS avg, " +
            std::to_string(totalDetections) + " total detections.");

    } catch (const std::exception& e) {
        ::traffic::Logger::error("Fatal: " + std::string(e.what()));
        return -1;
    }

    return 0;
}
