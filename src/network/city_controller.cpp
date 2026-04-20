#include "network/city_controller.hpp"
#include "core/logger.hpp"

namespace antigravity {
namespace network {

CityController::CityController(std::shared_ptr<RoadGraph> graph,
                               std::shared_ptr<control::SignalController> signal_controller)
    : graph(graph), signal_controller(signal_controller) {
    traffic::Logger::info("CityController initialized.");
}

void CityController::updateTracks(const std::vector<::traffic::Track>& detections) {
    std::lock_guard<std::mutex> lock(mtx);

    // --- 1. IoU Matching & Persistence Logic ---
    auto get_iou = [](const cv::Rect& a, const cv::Rect& b) -> float {
        float inter = static_cast<float>((a & b).area());
        float union_area = static_cast<float>(a.area() + b.area() - inter);
        return inter / (union_area + 1e-6f);
    };

    std::vector<::traffic::Track> next_history;
    std::vector<bool> detection_used(detections.size(), false);

    // A. Match existing history with new detections
    for (auto& old_track : track_history) {
        float max_iou = 0.0f;
        int best_idx = -1;

        for (size_t i = 0; i < detections.size(); ++i) {
            if (detection_used[i]) continue;
            float iou = get_iou(old_track.bbox, detections[i].bbox);
            if (iou > 0.4f && iou > max_iou) {
                max_iou = iou;
                best_idx = static_cast<int>(i);
            }
        }

        if (best_idx != -1) {
            // Hit! Update track
            old_track.bbox = detections[best_idx].bbox;
            old_track.confidence = detections[best_idx].confidence;
            old_track.classId = detections[best_idx].classId;
            old_track.missed_frames = 0;
            old_track.lastSeen = std::chrono::system_clock::now();
            detection_used[best_idx] = true;
            next_history.push_back(old_track);
        } else {
            // Miss! Retain if within age limit
            old_track.missed_frames++;
            if (old_track.missed_frames <= max_missed_frames) {
                next_history.push_back(old_track);
            }
        }
    }

    // B. Add new detections as new tracks
    static int next_track_id = 1000;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!detection_used[i]) {
            ::traffic::Track t = detections[i];
            t.id = next_track_id++;
            t.missed_frames = 0;
            t.lastSeen = std::chrono::system_clock::now();
            next_history.push_back(t);
        }
    }

    track_history = std::move(next_history);
    const auto& tracks = track_history;

    // --- 2. Density metrics ---
    auto metrics = density_engine.analyze(tracks);
    vehicle_count = static_cast<int>(tracks.size()); // Total active tracks

    // --- 3. Update road graph ---
    float density = static_cast<float>(vehicle_count);
    graph->updateDensity(0, density);
    global_pressure = density;

    // --- 4. Signal controller ---
    control::SignalController::Observation obs;
    obs.lane_densities.push_back(density);
    obs.queue_lengths.push_back(static_cast<float>(vehicle_count));
    obs.time_of_day = 12.0f;

    auto action = signal_controller->computePolicy(obs);
    if (action.duration_extension > 0) {
        traffic::Logger::info("Signal: Phase " + std::to_string(action.phase_id) +
                  " extended " + std::to_string(action.duration_extension) + "s");
    }

    // --- 5. Anomaly detection ---
    auto alerts = anomaly_detector.monitor(tracks);
    for (const auto& alert : alerts) {
        traffic::Logger::warn("ALERT: " + alert.description);
    }
}

float CityController::getGlobalPressure() const {
    std::lock_guard<std::mutex> lock(mtx);
    return global_pressure;
}

int CityController::getVehicleCount() const {
    std::lock_guard<std::mutex> lock(mtx);
    return vehicle_count;
}

} // namespace network
} // namespace antigravity
