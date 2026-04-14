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
    cv::Rect bbox;
    float velocity;
    std::vector<cv::Point2f> history;
    std::chrono::system_clock::time_point lastSeen;
};

} // namespace traffic
