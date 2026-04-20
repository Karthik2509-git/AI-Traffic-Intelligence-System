#include "control/signal_control.hpp"
#include "core/logger.hpp"
#include <algorithm>

namespace antigravity {
namespace control {

SignalController::SignalController() {
    traffic::Logger::info("SignalController: Density-threshold heuristic active.");
}

SignalAction SignalController::computePolicy(const Observation& obs) {
    SignalAction action;
    action.phase_id = 0;
    action.duration_extension = 30; // default green extension
    action.skip_phase = false;

    if (obs.lane_densities.empty()) return action;

    // Find the lane with highest density
    auto it = std::max_element(obs.lane_densities.begin(), obs.lane_densities.end());
    int maxLane = static_cast<int>(std::distance(obs.lane_densities.begin(), it));
    float maxDensity = *it;

    action.phase_id = maxLane;

    // Threshold-based extension: higher density → longer green
    if (maxDensity > 20.0f) {
        action.duration_extension = 45;
    } else if (maxDensity > 10.0f) {
        action.duration_extension = 30;
    } else if (maxDensity > 5.0f) {
        action.duration_extension = 15;
    } else {
        action.duration_extension = 10;
    }

    return action;
}

void SignalController::updateReward(float, float) {
    // No-op. Placeholder for future online learning.
}

} // namespace control
} // namespace antigravity
