#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <opencv2/core.hpp>

namespace traffic {

/**
 * @brief Represents a vehicle feature vector extracted for cross-camera Re-ID.
 */
struct ReIDFeature {
    std::string camera_id;
    int local_track_id = -1;
    std::string global_vehicle_id; // Assigned upon cross-camera correlation match
    std::vector<float> embedding;   // 256 or 512 float vector
    double timestamp = 0.0;
    cv::Rect bbox;
    float match_confidence = 0.0f;
};

/**
 * @brief Subsystem status flags for Re-ID.
 */
enum class ReIDStatus {
    DISABLED,
    MODEL_UNAVAILABLE,
    READY,
    ERROR
};

/**
 * @brief Cross-Camera Identity Match record.
 */
struct CrossCameraMatch {
    std::string global_vehicle_id;
    std::string source_camera_id;
    int source_local_id;
    std::string target_camera_id;
    int target_local_id;
    float similarity_score;
    double time_delta_sec;
};

} // namespace traffic
