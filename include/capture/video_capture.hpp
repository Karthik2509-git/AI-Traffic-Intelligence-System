#pragma once

#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "core/concurrent_queue.hpp"
#include "core/memory.hpp"
#include "core/types.hpp"

namespace antigravity {
namespace capture {

enum class SourceType {
    LOCAL_FILE,
    RTSP_STREAM,
    HTTP_WEBCAM,
    USB_CAMERA,
    UNKNOWN
};

struct FramePackage {
    int streamId = 0;
    uint64_t frameIndex = 0;
    std::chrono::steady_clock::time_point captureTimestamp;
    std::shared_ptr<core::PinnedBuffer<uint8_t>> buffer;
    cv::Mat frame;
    int width = 0;
    int height = 0;
    SourceType sourceType = SourceType::LOCAL_FILE;
    std::vector<traffic::Track> results;
};

struct CaptureStats {
    uint64_t totalCaptured = 0;
    uint64_t totalDropped = 0;
    double currentFps = 0.0;
    int width = 0;
    int height = 0;
    bool isConnected = false;
    std::string sourceUri;
};

class VideoCaptureEngine {
public:
    struct Config {
        std::string source = "data/sample_traffic.mp4";
        int stream_id = 0;
        size_t queue_capacity = 32;
        bool auto_reconnect = true;
        int max_reconnect_attempts = 10;
        int reconnect_interval_ms = 2000;
        bool drop_on_full = true;
    };

    VideoCaptureEngine(const Config& config, core::ConcurrentQueue<std::shared_ptr<FramePackage>>& targetQueue);
    ~VideoCaptureEngine();

    void start();
    void stop();

    bool isRunning() const { return running; }
    CaptureStats getStats() const;

    static SourceType detectSourceType(const std::string& source);

private:
    Config config;
    core::ConcurrentQueue<std::shared_ptr<FramePackage>>& queue;

    std::thread captureThread;
    std::atomic<bool> running{false};
    std::atomic<bool> stopRequested{false};

    mutable std::mutex statsMtx;
    CaptureStats stats;

    void workerLoop();
    bool openCapture(cv::VideoCapture& cap, SourceType sType);
};

} // namespace capture
} // namespace antigravity
