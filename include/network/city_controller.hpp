#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include "core/types.hpp"
#include "network/graph.hpp"
#include "analytics/anomaly.hpp"
#include "analytics/density_engine.hpp"
#include "control/signal_control.hpp"

namespace antigravity {
namespace network {

/**
 * @brief Central traffic intelligence coordinator.
 *
 * Receives detection results from the inference engine, updates the road graph,
 * runs anomaly detection, computes density metrics, and triggers signal adjustments.
 */
class CityController {
public:
    CityController(std::shared_ptr<RoadGraph> graph,
                   std::shared_ptr<control::SignalController> signal_controller);

    /** Process new vehicle detections: density, signals, anomalies. */
    void updateTracks(const std::vector<traffic::Track>& tracks);

    /** Get the current global traffic pressure. */
    float getGlobalPressure() const;

    /** Get the last detection count. */
    int getVehicleCount() const;

private:
    std::shared_ptr<RoadGraph> graph;
    std::shared_ptr<control::SignalController> signal_controller;
    analytics::AnomalyDetector anomaly_detector;
    analytics::DensityEngine density_engine;
    std::vector<::traffic::Track> track_history;
    const int max_missed_frames = 2; // Retain for 2 missing frames

    mutable std::mutex mtx;
    float global_pressure = 0.0f;
    int vehicle_count = 0;
};

} // namespace network
} // namespace antigravity
