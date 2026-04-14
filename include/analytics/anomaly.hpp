#pragma once

#include <vector>
#include <map>
#include <string>
#include <chrono>
#include <opencv2/core.hpp>
#include "core/types.hpp"

namespace antigravity {
namespace analytics {

/**
 * @brief High-Fidelity Anomaly & Safety Incident Detector.
 * 
 * Uses spatial-temporal trajectory analysis to detect accidents, 
 * wrong-way drivers, and stalled vehicles in real-time.
 */
class AnomalyDetector {
public:
    enum class AnomalyType {
        ACCIDENT_DETECTION,
        WRONG_WAY_DRIVER,
        STALLED_VEHICLE,
        ZONE_VIOLATION,
        NONE
    };

    struct Alert {
        AnomalyType type;
        int vehicleId;
        float confidence;
        std::string description;
        std::chrono::system_clock::time_point timestamp;
    };

    AnomalyDetector() = default;

    /**
     * @brief Analyze tracks for behavioral anomalies.
     */
    std::vector<Alert> monitor(const std::vector<traffic::Track>& tracks);

private:
    // Global trajectory history for incident reconstruction
    std::map<int, std::vector<cv::Point2f>> trajectory_cache;

    bool checkWrongWay(const std::vector<cv::Point2f>& traj);
    bool checkAccident(const std::vector<cv::Point2f>& traj);
};

} // namespace analytics
} // namespace antigravity
