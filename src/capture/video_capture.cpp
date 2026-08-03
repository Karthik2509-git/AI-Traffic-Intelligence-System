#include "capture/video_capture.hpp"
#include "core/logger.hpp"

#include <iostream>
#include <deque>
#include <numeric>

namespace antigravity {
namespace capture {

SourceType VideoCaptureEngine::detectSourceType(const std::string& source) {
    if (source.rfind("rtsp://", 0) == 0 || source.rfind("rtsps://", 0) == 0) {
        return SourceType::RTSP_STREAM;
    }
    if (source.rfind("http://", 0) == 0 || source.rfind("https://", 0) == 0) {
        return SourceType::HTTP_WEBCAM;
    }
    bool isNumeric = !source.empty() && std::all_of(source.begin(), source.end(), ::isdigit);
    if (isNumeric) {
        return SourceType::USB_CAMERA;
    }
    return SourceType::LOCAL_FILE;
}

VideoCaptureEngine::VideoCaptureEngine(
    const Config& config,
    core::ConcurrentQueue<std::shared_ptr<FramePackage>>& targetQueue
) : config(config), queue(targetQueue) {
    stats.sourceUri = config.source;
}

VideoCaptureEngine::~VideoCaptureEngine() {
    stop();
}

void VideoCaptureEngine::start() {
    if (running) return;
    stopRequested = false;
    running = true;
    captureThread = std::thread(&VideoCaptureEngine::workerLoop, this);
    traffic::Logger::info("VideoCaptureEngine: Ingestion thread launched for source: " + config.source);
}

void VideoCaptureEngine::stop() {
    if (!running) return;
    stopRequested = true;
    running = false;
    if (captureThread.joinable()) {
        captureThread.join();
    }
    traffic::Logger::info("VideoCaptureEngine: Ingestion thread stopped.");
}

CaptureStats VideoCaptureEngine::getStats() const {
    std::lock_guard<std::mutex> lock(statsMtx);
    return stats;
}

bool VideoCaptureEngine::openCapture(cv::VideoCapture& cap, SourceType sType) {
    if (sType == SourceType::USB_CAMERA) {
        int devId = std::stoi(config.source);
        return cap.open(devId, cv::CAP_ANY);
    } else if (sType == SourceType::RTSP_STREAM) {
#ifdef HAVE_OPENCV_FFMPEG
        return cap.open(config.source, cv::CAP_FFMPEG);
#else
        return cap.open(config.source, cv::CAP_ANY);
#endif
    }
    return cap.open(config.source, cv::CAP_ANY);
}

void VideoCaptureEngine::workerLoop() {
    SourceType sType = detectSourceType(config.source);
    uint64_t frameIndex = 0;
    int reconnectAttempts = 0;

    std::deque<std::chrono::steady_clock::time_point> timestamps;

    while (!stopRequested) {
        cv::VideoCapture cap;
        bool opened = openCapture(cap, sType);

        if (!opened || !cap.isOpened()) {
            {
                std::lock_guard<std::mutex> lock(statsMtx);
                stats.isConnected = false;
            }

            traffic::Logger::error("VideoCaptureEngine: Failed to open source: " + config.source);

            if (!config.auto_reconnect || reconnectAttempts >= config.max_reconnect_attempts) {
                traffic::Logger::error("VideoCaptureEngine: Exceeded max reconnect attempts. Halting ingestion.");
                break;
            }

            reconnectAttempts++;
            traffic::Logger::warn("VideoCaptureEngine: Retrying connection (" +
                std::to_string(reconnectAttempts) + "/" + std::to_string(config.max_reconnect_attempts) + ") in " +
                std::to_string(config.reconnect_interval_ms) + " ms...");

            std::this_thread::sleep_for(std::chrono::milliseconds(config.reconnect_interval_ms));
            continue;
        }

        // Connection established
        reconnectAttempts = 0;
        int frameW = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frameH = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double sourceFps = cap.get(cv::CAP_PROP_FPS);

        {
            std::lock_guard<std::mutex> lock(statsMtx);
            stats.isConnected = true;
            stats.width = frameW;
            stats.height = frameH;
        }

        traffic::Logger::info("VideoCaptureEngine: Stream connected (" +
            std::to_string(frameW) + "x" + std::to_string(frameH) +
            " @ " + std::to_string(static_cast<int>(sourceFps)) + " FPS)");

        cv::Mat rawFrame;

        while (!stopRequested) {
            auto now = std::chrono::steady_clock::now();

            if (!cap.read(rawFrame) || rawFrame.empty()) {
                if (sType == SourceType::LOCAL_FILE) {
                    traffic::Logger::info("VideoCaptureEngine: End of local video file reached (" +
                        std::to_string(frameIndex) + " total frames read).");
                    stopRequested = true;
                    break;
                } else {
                    traffic::Logger::warn("VideoCaptureEngine: Stream read drop / disconnect detected.");
                    break; // Reconnect loop will trigger
                }
            }

            // Valid frame acquired
            int actualW = rawFrame.cols;
            int actualH = rawFrame.rows;

            auto pkg = std::make_shared<FramePackage>();
            pkg->streamId = config.stream_id;
            pkg->frameIndex = frameIndex++;
            pkg->captureTimestamp = now;
            pkg->width = actualW;
            pkg->height = actualH;
            pkg->sourceType = sType;

            // Allocate Pinned Memory buffer for zero-copy CUDA DMA
            pkg->buffer = std::make_shared<core::PinnedBuffer<uint8_t>>(actualW * actualH * 3);
            pkg->frame = cv::Mat(actualH, actualW, CV_8UC3, pkg->buffer->get());
            rawFrame.copyTo(pkg->frame);

            // Queue Backpressure & Dropped Frame Policy
            if (config.drop_on_full && queue.size() >= config.queue_capacity) {
                // Drop incoming frame to prevent capture thread latency accumulation
                {
                    std::lock_guard<std::mutex> lock(statsMtx);
                    stats.totalDropped++;
                }
            } else {
                queue.push(std::move(pkg));
            }

            // Capture FPS stats tracking (rolling 30-frame window)
            timestamps.push_back(now);
            if (timestamps.size() > 30) timestamps.pop_front();

            double currentFps = 0.0;
            if (timestamps.size() >= 2) {
                double spanSeconds = std::chrono::duration<double>(timestamps.back() - timestamps.front()).count();
                if (spanSeconds > 0) {
                    currentFps = (timestamps.size() - 1) / spanSeconds;
                }
            }

            {
                std::lock_guard<std::mutex> lock(statsMtx);
                stats.totalCaptured++;
                stats.currentFps = currentFps;
                stats.width = actualW;
                stats.height = actualH;
            }
        }

        cap.release();
    }

    {
        std::lock_guard<std::mutex> lock(statsMtx);
        stats.isConnected = false;
    }
    running = false;
}

} // namespace capture
} // namespace antigravity
