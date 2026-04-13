#pragma once

#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <string>
#include "logger.hpp"

namespace traffic {

/**
 * @brief High-performance asynchronous frame grabber.
 * 
 * Optimized for RTSP/Mobile camera streams. Runs in a separate thread
 * to ensure that the processing pipeline always has access to the 
 * latest frame without network-induced lag.
 */
class AsyncCapture {
public:
    AsyncCapture(const std::string& source) : source(source) {
        stop_flag = false;
        capture_thread = std::thread(&AsyncCapture::grabFrames, this);
    }

    ~AsyncCapture() {
        stop_flag = true;
        if (capture_thread.joinable()) {
            capture_thread.join();
        }
    }

    /**
     * @brief Get the most recent frame from the camera buffer.
     * @return cv::Mat Latest frame (empty if camera not ready).
     */
    cv::Mat getLatestFrame() {
        std::lock_guard<std::mutex> lock(mtx);
        return latest_frame.clone();
    }

    bool isReady() const {
        return ready;
    }

private:
    std::string source;
    std::thread capture_thread;
    std::mutex mtx;
    cv::Mat latest_frame;
    std::atomic<bool> stop_flag;
    std::atomic<bool> ready{false};

    void grabFrames() {
        cv::VideoCapture cap(source);
        if (!cap.isOpened()) {
            Logger::error("AsyncCapture: Failed to open source " + source);
            return;
        }

        // Optimization for RTSP: Disable internal buffer to get lowest latency
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

        while (!stop_flag) {
            cv::Mat frame;
            if (cap.read(frame)) {
                std::lock_guard<std::mutex> lock(mtx);
                latest_frame = frame;
                ready = true;
            } else {
                Logger::warn("AsyncCapture: Dropped frame or connection lost. Retrying...");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }
};

} // namespace traffic
