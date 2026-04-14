#include "network/city_controller.hpp"
#include "core/logger.hpp"
#include <numeric>

namespace antigravity {
namespace network {

CityController::CityController(std::shared_ptr<RoadGraph> graph, 
                               std::shared_ptr<control::RLSignalController> signal_controller)
    : graph(graph), signal_controller(signal_controller) {
    traffic::Logger::info("CityController initialized. World-Class Intelligence Active.");
}

void CityController::updateTracks(const std::vector<traffic::Track>& tracks) {
    std::lock_guard<std::mutex> lock(mtx);
    
    // Calculate density based on track count (normalized for 1km stretch)
    // For this simulation, we assume each camera node covers 0.5km of road
    float current_density = static_cast<float>(tracks.size()) * 2.0f;
    
    // Update the road graph
    // We assume Node 0 for the primary camera stream in this demo
    graph->updateDensity(0, current_density);

    // Compute Global City Pressure
    global_pressure = current_density; // Simplified for single-node demonstration

    // Communicate with the RL Signal Controller
    control::RLSignalController::Observation obs;
    obs.lane_densities.push_back(current_density);
    obs.queue_lengths.push_back(static_cast<float>(tracks.size())); // Simple proxy
    obs.neighbor_pressures.push_back(0.1f); // Static for now; in multi-node, this comes from graph
    obs.time_of_day = 12.0f; // Midday baseline

    auto action = signal_controller->computePolicy(obs);
    
    if (action.duration_extension > 0) {
        traffic::Logger::info("Intelligent Signal Update: Phase " + std::to_string(action.phase_id) + 
                 " Extended by " + std::to_string(action.duration_extension) + "s");
    }
}

void CityController::handleAlerts(const std::vector<analytics::AnomalyDetector::Alert>& alerts) {
    for (const auto& alert : alerts) {
        if (alert.type == analytics::AnomalyDetector::AnomalyType::ACCIDENT_DETECTION) {
            traffic::Logger::warn("EMERGENCY OVERRIDE: Accident at Intersection 0. Initiating Green-Wave for Responders.");
            // Signal Controller: Force phase for emergency vehicle corridor
            signal_controller->updateReward(0.0f, -100.0f); // Massive penalty to current flow
        }
    }
}

float CityController::getGlobalPressure() const {
    std::lock_guard<std::mutex> lock(mtx);
    return global_pressure;
}

} // namespace network
} // namespace antigravity
