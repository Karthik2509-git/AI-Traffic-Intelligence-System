#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include "core/types.hpp"
#include "network/graph.hpp"
#include "analytics/anomaly.hpp"
#include "control/signal_control.hpp"

namespace antigravity {
namespace network {

/**
 * @brief Unified City Intelligence Controller.
 * 
 * The "Brain" of the ATOS system. Coordinates between detection, 
 * behavior analytics, and autonomous signal control.
 */
class CityController {
public:
    CityController(std::shared_ptr<RoadGraph> graph, 
                   std::shared_ptr<control::RLSignalController> signal_controller);

    /**
     * @brief Update city state with new vehicle detections.
     */
    void updateTracks(const std::vector<traffic::Track>& tracks);

    /**
     * @brief Process and broadcast safety alerts.
     */
    void handleAlerts(const std::vector<analytics::AnomalyDetector::Alert>& alerts);

    /**
     * @brief Get global city pressure (sum of all node densities).
     */
    float getGlobalPressure() const;

private:
    std::shared_ptr<RoadGraph> graph;
    std::shared_ptr<control::RLSignalController> signal_controller;
    
    mutable std::mutex mtx;
    float global_pressure = 0.0f;

    void updateNodeMetrics();
};

} // namespace network
} // namespace antigravity
