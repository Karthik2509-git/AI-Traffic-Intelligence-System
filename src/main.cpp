#include "core/logger.hpp"
#include "core/config_loader.hpp"
#include "engine/detector.hpp"
#include "analytics/density.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>

namespace traffic {

/**
 * @brief Main system orchestrator.
 * 
 * Implements a high-performance pipeline for AI Traffic Intelligence.
 * Ties together CUDA preprocessing, TensorRT inference, and real-time analytics.
 */
class TrafficSystem {
public:
    TrafficSystem(const std::string& configPath) {
        if (!ConfigLoader::getInstance().load(configPath)) {
            throw std::runtime_error("Config load failed.");
        }
        
        const auto& config = ConfigLoader::getInstance();
        detector = std::make_unique<Detector>(config.getModelPath());
        analyzer = std::make_unique<DensityAnalyzer>(config.getLanes());
    }

    void run(const std::string& videoSource) {
        cv::VideoCapture cap(videoSource);
        if (!cap.isOpened()) {
            Logger::error("Could not open video: " + videoSource);
            return;
        }

        cv::Mat frame;
        while (isRunning) {
            auto start = std::chrono::high_resolution_clock::now();
            
            if (!cap.read(frame)) break;

            // 1. Preprocessing & Inference (Placeholder for full integration)
            // In a full implementation, we'd pass 'frame' to the CUDA preprocessor
            // and then to the detector.
            std::vector<Track> tracks = detector->detect(nullptr); 

            // 2. Analytics
            double ts = std::chrono::duration<double, std::milli>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            FrameResult result = analyzer->update(tracks, ts);

            // 3. Visualization (OpenCV Overlay)
            renderHUD(frame, result);
            cv::imshow("AI Traffic Intelligence - C++/CUDA", frame);

            auto end = std::chrono::high_resolution_clock::now();
            float fps = 1000.0f / std::chrono::duration<float, std::milli>(end - start).count();
            
            if (frameIdx++ % 30 == 0) {
                Logger::info("System Running | FPS: " + std::to_string(fps));
            }

            if (cv::waitKey(1) == 27) break; 
        }
    }

private:
    std::unique_ptr<Detector> detector;
    std::unique_ptr<DensityAnalyzer> analyzer;
    bool isRunning = true;
    int frameIdx = 0;

    void renderHUD(cv::Mat& frame, const FrameResult& res) {
        // Draw Lane Overlays
        for (const auto& lane : ConfigLoader::getInstance().getLanes()) {
            std::vector<cv::Point> pts;
            for (const auto& p : lane.polygon) pts.push_back(cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)));
            
            std::vector<std::vector<cv::Point>> contours = {pts};
            cv::polylines(frame, contours, true, cv::Scalar(0, 255, 0), 2);
        }

        // Draw Stats Panel
        cv::rectangle(frame, cv::Point(10, 10), cv::Point(250, 120), cv::Scalar(30, 30, 30), -1);
        cv::putText(frame, "Vehicles: " + std::to_string(res.totalCount), cv::Point(20, 40), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(frame, "Congestion: " + std::to_string(static_cast<int>(res.congestionScore)) + "%", 
                    cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 165, 255), 2);
        cv::putText(frame, "Status: " + analyzer->getTrend(), cv::Point(20, 100), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 100), 2);
    }
};

} // namespace traffic

int main(int argc, char** argv) {
    try {
        traffic::TrafficSystem system("config.yaml");
        system.run(argc > 1 ? argv[1] : "0");
    } catch (const std::exception& e) {
        traffic::Logger::error("Fatal Error: " + std::string(e.what()));
        return -1;
    }
    return 0;
}
