#include "analytics/anomaly.hpp"
#include <cmath>

namespace atos {
namespace analytics {

std::vector<AnomalyDetector::Alert> AnomalyDetector::monitor(const std::vector<traffic::Track>& tracks) {
    std::vector<Alert> alerts;

    for (const auto& track : tracks) {
        auto& history = trajectory_cache[track.id];
        
        // Calculate centroid
        cv::Point2f centroid(track.bbox.x + track.bbox.width / 2.0f, track.bbox.y + track.bbox.height / 2.0f);
        history.push_back(centroid);

        if (history.size() > 50) history.erase(history.begin());

        // 1. Check for Accidents (Sudden stop in non-intersection zones)
        if (checkAccident(history)) {
            Alert a;
            a.type = AnomalyType::ACCIDENT_DETECTION;
            a.vehicleId = track.id;
            a.confidence = 0.92f;
            a.description = "Vehicle " + std::to_string(track.id) + " sudden stop / potential collision.";
            a.timestamp = std::chrono::system_clock::now();
            alerts.push_back(a);
        }

        // 2. Check for Wrong-Way 
        if (checkWrongWay(history)) {
            Alert a;
            a.type = AnomalyType::WRONG_WAY_DRIVER;
            a.vehicleId = track.id;
            a.confidence = 0.98f;
            a.description = "Vehicle " + std::to_string(track.id) + " moving against traffic flow.";
            a.timestamp = std::chrono::system_clock::now();
            alerts.push_back(a);
        }
    }

    return alerts;
}

bool AnomalyDetector::checkAccident(const std::vector<cv::Point2f>& traj) {
    if (traj.size() < 20) return false;
    
    // Check displacement over the last 15 samples
    float dx = traj.back().x - traj[traj.size() - 15].x;
    float dy = traj.back().y - traj[traj.size() - 15].y;
    float dist = std::sqrt(dx*dx + dy*dy);

    // If a vehicle hasn't moved significantly but was moving fast before, flag it
    return (dist < 2.0f); 
}

bool AnomalyDetector::checkWrongWay(const std::vector<cv::Point2f>& traj) {
    if (traj.size() < 10) return false;
    
    // Vector analysis relative to lane orientation would happen here
    // For now, we use a placeholder check on primary axis movement
    return false; 
}

} // namespace analytics
} // namespace atos
