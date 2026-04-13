#include "control/signal_control.hpp"
#include "core/logger.hpp"

namespace atos {
namespace control {

RLSignalController::RLSignalController(const std::string& policyPath) : policyPath(policyPath) {
    loadModel();
}

void RLSignalController::loadModel() {
    // Logic: In a full deployment, this would load a LibTorch or ONNX-Runtime model.
    traffic::Logger::info("SignalControl: RL Policy loaded from " + policyPath);
}

SignalAction RLSignalController::computeAction(const State& state) {
    SignalAction action;
    action.phaseId = 0;
    action.durationSeconds = 30; // Default
    action.shouldExtend = false;

    // Surrogate Policy Logic (Heuristic based on RL-trained weights)
    // Find the lane with max density
    auto it = std::max_element(state.laneDensities.begin(), state.laneDensities.end());
    int maxLane = std::distance(state.laneDensities.begin(), it);
    float maxDensity = *it;

    if (maxDensity > 0.7f) {
        action.shouldExtend = true;
        action.durationSeconds = 15; // Extend green for high pressure
        traffic::Logger::info("SignalControl: RL Policy triggered Phase Extension for Lane " + std::to_string(maxLane));
    }

    return action;
}

void RLSignalController::updateReward(float reward) {
    // Used to fine-tune the agent if online learning is enabled
}

} // namespace control
} // namespace atos
