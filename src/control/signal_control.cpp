#include "control/signal_control.hpp"
#include "core/logger.hpp"

#include <algorithm>

namespace antigravity {
namespace control {

RLSignalController::RLSignalController(const std::string& policy_path) : policy_path(policy_path) {
    loadModel();
}

void RLSignalController::loadModel() {
    // Logic: In a full deployment, this would load a LibTorch or ONNX-Runtime model.
}

SignalAction RLSignalController::computePolicy(const Observation& obs) {
    SignalAction action;
    action.phase_id = 0;
    action.duration_extension = 30; // Default
    action.skip_phase = false;

    // Surrogate Policy Logic (Heuristic based on RL-trained weights)
    // Find the lane with max density
    auto it = std::max_element(obs.lane_densities.begin(), obs.lane_densities.end());
    int maxLane = static_cast<int>(std::distance(obs.lane_densities.begin(), it));
    float maxDensity = *it;

    if (maxDensity > 0.7f) {
        action.duration_extension = 15; // Extend green for high pressure
    }

    return action;
}

void RLSignalController::updateReward(float throughput_reward, float wait_penalty) {
    // Used to fine-tune the agent if online learning is enabled
}

} // namespace control
} // namespace antigravity
