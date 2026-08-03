#include "core/logger.hpp"
#include "core/concurrent_queue.hpp"
#include "core/memory.hpp"
#include "core/types.hpp"
#include "core/config.hpp"
#include "capture/video_capture.hpp"
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
// ATOS v3.1 — Industrial Traffic Intelligence Engine
// Pipeline: VideoCaptureEngine -> Pinned Memory -> TensorRT -> Analytics -> Annotate -> Telemetry
// =========================================================================

std::atomic<bool> g_running{true};

void signalHandler(int) {
    g_running = false;
}

static const char* getClassName(int classId) {
    switch (classId) {
        case 2:  return "car";
        case 3:  return "motorcycle";
        case 5:  return "bus";
        case 7:  return "truck";
        default: return "vehicle";
    }
}

static cv::Scalar getClassColor(int classId) {
    switch (classId) {
        case 2:  return cv::Scalar(0, 255, 0);     // car: green
        case 3:  return cv::Scalar(255, 165, 0);   // motorcycle: orange
        case 5:  return cv::Scalar(255, 0, 0);     // bus: blue
        case 7:  return cv::Scalar(0, 0, 255);     // truck: red
        default: return cv::Scalar(200, 200, 200); // gray
    }
}

using PipelineFrame = antigravity::capture::FramePackage;

antigravity::core::ConcurrentQueue<std::shared_ptr<PipelineFrame>> g_captureQueue(32);

std::shared_ptr<antigravity::network::CityController> g_cityController;
std::shared_ptr<antigravity::simulation::DigitalTwinBridge> g_twinBridge;

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

void annotateFrame(cv::Mat& frame, const std::vector<traffic::Track>& tracks,
                   uint64_t frameIdx, double fps) {
    for (const auto& t : tracks) {
        cv::Scalar color = getClassColor(t.classId);

        cv::rectangle(frame, t.bbox, color, 2);

        std::stringstream ss;
        ss << getClassName(t.classId) << " " << std::fixed << std::setprecision(2) << t.confidence;
        std::string label = ss.str();

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::Point labelPos(t.bbox.x, t.bbox.y - 5);
        if (labelPos.y < 15) labelPos.y = t.bbox.y + 15;

        cv::rectangle(frame,
            cv::Point(labelPos.x, labelPos.y - textSize.height - 2),
            cv::Point(labelPos.x + textSize.width + 2, labelPos.y + 2),
            color, cv::FILLED);
        cv::putText(frame, label, labelPos, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0), 1);
    }

    std::stringstream hud;
    hud << "Frame: " << frameIdx
        << "  |  Det: " << tracks.size()
        << "  |  FPS: " << std::fixed << std::setprecision(1) << fps;

    cv::putText(frame, hud.str(), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
}

int main(int argc, char** argv) {
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    traffic::Logger::info("ATOS v3.1 Industrial Engine starting...");

    // 1. Load configuration from settings.yaml
    auto& configMgr = antigravity::core::ConfigManager::getInstance();
    configMgr.loadFromFile("config/settings.yaml");
    const auto& appConfig = configMgr.getConfig();

    std::string videoSource = appConfig.video.default_source;
    if (argc > 1) {
        videoSource = argv[1];
        traffic::Logger::info("Source override: " + videoSource);
    }

    std::system("if not exist output mkdir output");

    try {
        // 2. Initialize Subsystems from Configuration
        auto graph = std::make_shared<antigravity::network::RoadGraph>();
        graph->addCameraNode(0, "Intersection-Alpha");
        graph->addRoadConnection(0, 1, 500.0f);

        auto signals = std::make_shared<antigravity::control::SignalController>();
        g_cityController = std::make_shared<antigravity::network::CityController>(graph, signals);

        antigravity::simulation::DigitalTwinBridge::Config twinConfig;
        twinConfig.target_ip = appConfig.telemetry.target_ip;
        twinConfig.target_port = appConfig.telemetry.target_port;
        g_twinBridge = std::make_shared<antigravity::simulation::DigitalTwinBridge>(twinConfig);

        antigravity::engine::Detector::Config detConfig;
        detConfig.engine_path = appConfig.engine.model_path;
        detConfig.input_w = appConfig.engine.input_width;
        detConfig.input_h = appConfig.engine.input_height;
        detConfig.conf_threshold = appConfig.detection.confidence_threshold;
        detConfig.nms_threshold = appConfig.detection.nms_threshold;
        antigravity::engine::Detector detector(detConfig);

        // 3. Initialize Video Capture Engine
        antigravity::capture::VideoCaptureEngine::Config capConfig;
        capConfig.source = videoSource;
        capConfig.stream_id = 0;
        capConfig.queue_capacity = 32;
        capConfig.auto_reconnect = true;
        capConfig.drop_on_full = true;

        antigravity::capture::VideoCaptureEngine captureEngine(capConfig, g_captureQueue);
        captureEngine.start();

        std::thread telemetryThread(telemetryWorker);

        traffic::Logger::info("ATOS v3.1 operational. Processing...");

        // 4. Metrics Tracking
        std::ofstream metricsFile("output/metrics.csv");
        metricsFile << "frame,detections,latency_ms,fps_instant,fps_avg\n";

        auto globalStart = std::chrono::steady_clock::now();
        uint64_t totalFrames = 0;
        int totalDetections = 0;
        double latencySum = 0.0;
        double latencyMin = 1e9, latencyMax = 0.0;
        std::deque<double> fpsWindow;
        const size_t FPS_WINDOW = 30;

        cv::VideoWriter videoWriter;
        bool videoInitialized = false;

        // 5. Main Processing Loop
        while (g_running) {
            std::shared_ptr<PipelineFrame> pFrame;
            if (!g_captureQueue.pop(pFrame)) break;

            if (!pFrame || !pFrame->buffer) continue;

            pFrame->results = detector.process(
                pFrame->buffer->getDevicePtr(), pFrame->width, pFrame->height);

            g_cityController->updateTracks(pFrame->results);

            auto now = std::chrono::steady_clock::now();
            double latency = std::chrono::duration<double, std::milli>(now - pFrame->captureTimestamp).count();

            totalFrames++;
            totalDetections += static_cast<int>(pFrame->results.size());
            latencySum += latency;
            if (latency < latencyMin) latencyMin = latency;
            if (latency > latencyMax) latencyMax = latency;

            double elapsed = std::chrono::duration<double>(now - globalStart).count();
            double globalFps = (elapsed > 0) ? totalFrames / elapsed : 0.0;

            double instantFps = (latency > 0) ? 1000.0 / latency : 0.0;
            fpsWindow.push_back(instantFps);
            if (fpsWindow.size() > FPS_WINDOW) fpsWindow.pop_front();
            double rollingFps = 0.0;
            if (!fpsWindow.empty()) {
                rollingFps = std::accumulate(fpsWindow.begin(), fpsWindow.end(), 0.0) / fpsWindow.size();
            }

            metricsFile << pFrame->frameIndex << ","
                        << pFrame->results.size() << ","
                        << std::fixed << std::setprecision(1) << latency << ","
                        << std::setprecision(1) << rollingFps << ","
                        << std::setprecision(1) << globalFps << "\n";

            annotateFrame(pFrame->frame, pFrame->results, pFrame->frameIndex, rollingFps);

            if (!videoInitialized) {
                videoWriter.open("output/output_video.mp4",
                    cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                    25.0, cv::Size(pFrame->width, pFrame->height));
                videoInitialized = videoWriter.isOpened();
            }

            if (videoInitialized) {
                videoWriter.write(pFrame->frame);
            }

            if (pFrame->frameIndex % 30 == 0) {
                std::string framePath = "output/annotated_" + std::to_string(pFrame->frameIndex) + ".jpg";
                cv::imwrite(framePath, pFrame->frame);
            }

            if (pFrame->frameIndex % 10 == 0) {
                auto stats = captureEngine.getStats();
                traffic::Logger::info(
                    "Seq:" + std::to_string(pFrame->frameIndex) +
                    " | Det:" + std::to_string(pFrame->results.size()) +
                    " | Latency:" + std::to_string((int)latency) + "ms" +
                    " | FPS:" + std::to_string((int)rollingFps) +
                    " | CapFPS:" + std::to_string((int)stats.currentFps) +
                    " | Dropped:" + std::to_string(stats.totalDropped));
            }
        }

        // 6. Shutdown Procedure
        traffic::Logger::info("Shutting down ATOS pipeline...");
        g_running = false;
        captureEngine.stop();
        g_captureQueue.stop();

        telemetryThread.join();

        if (videoInitialized) videoWriter.release();
        metricsFile.close();

        // 7. Write Performance Benchmark Report
        double totalElapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - globalStart).count();
        double avgFps = (totalElapsed > 0) ? totalFrames / totalElapsed : 0.0;
        double avgLatency = (totalFrames > 0) ? latencySum / totalFrames : 0.0;

        auto stats = captureEngine.getStats();

        std::ofstream report("output/benchmark_report.txt");
        report << "========================================================\n";
        report << "  ATOS v3.1 — Production Benchmark & Audit Report\n";
        report << "========================================================\n\n";

        report << "Configuration File: config/settings.yaml\n";
        report << "Model Path: " << appConfig.engine.model_path << "\n";
        report << "Confidence Threshold: " << appConfig.detection.confidence_threshold << "\n";
        report << "NMS Threshold: " << appConfig.detection.nms_threshold << "\n";
        report << "Telemetry Destination: " << appConfig.telemetry.target_ip << ":" << appConfig.telemetry.target_port << "\n\n";

        report << "Source: " << videoSource << "\n";
        report << "Total Captured Frames: " << stats.totalCaptured << "\n";
        report << "Total Processed Frames: " << totalFrames << "\n";
        report << "Total Dropped Frames: " << stats.totalDropped << "\n";
        report << "Total Execution Time: " << std::fixed << std::setprecision(2) << totalElapsed << " s\n";
        report << "Total Detections: " << totalDetections << "\n\n";

        report << "--- Performance Metrics ---\n";
        report << "Average Processing FPS: " << std::setprecision(2) << avgFps << "\n";
        report << "Average Capture FPS:    " << std::setprecision(2) << stats.currentFps << "\n";
        report << "Average End-to-End Latency: " << std::setprecision(1) << avgLatency << " ms\n";
        report << "Min Latency: " << std::setprecision(1) << latencyMin << " ms\n";
        report << "Max Latency: " << std::setprecision(1) << latencyMax << " ms\n";
        report.close();

        traffic::Logger::info("Benchmark report written to output/benchmark_report.txt");

    } catch (const std::exception& e) {
        traffic::Logger::error("Fatal Exception: " + std::string(e.what()));
        return -1;
    }

    return 0;
}
