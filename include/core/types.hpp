#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <opencv2/core.hpp>

namespace traffic {

/**
 * @brief Raw AI Detection result.
 */
struct Detection {
    cv::Rect bbox;
    float confidence;
    int classId;
    std::string className;
};

/**
 * @brief Persistent Vehicle Track across multiple frames.
 */
struct Track {
    int id;
    int classId = 0;      // COCO class index (2=car, 3=motorcycle, 5=bus, 7=truck)
    cv::Rect bbox;
    float confidence;
    float velocity = 0.0f;
    std::vector<cv::Point2f> history;
    int missed_frames = 0; // Frames since last hit
    std::chrono::system_clock::time_point lastSeen;
};

} // namespace traffic
