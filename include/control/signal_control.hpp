#pragma once

#include <vector>
#include <string>
#include <memory>
#include <map>

namespace atos {
namespace control {

/**
 * @brief Adaptive Signal Control Action.
 */
struct SignalAction {
    int phase_id;
    int duration_extension; // in seconds
    bool skip_phase;
};

/**
 * @brief Reinforcement Learning-based Adaptive Signal Controller.
 * 
 * An autonomous agent that observes intersection pressure and computes 
 * optimal signal timings using a Proximal Policy Optimization (PPO) model.
 */
class RLSignalController {
public:
    struct Observation {
        std::vector<float> lane_densities;
        std::vector<float> queue_lengths;
        std::vector<float> neighbor_pressures;
        float time_of_day;
    };

    RLSignalController(const std::string& policy_path);

    /**
     * @brief Compute the optimal phase update based on current observations.
     */
    SignalAction computePolicy(const Observation& obs);

    /**
     * @brief Feed reward back to the environment (during training/sim).
     */
    void updateReward(float throughput_reward, float wait_penalty);

private:
    std::string policy_path;
    
    // LibTorch or ONNX Runtime Inference Engine for Policy Net
    // std::unique_ptr<InferenceEngine> model;

    void loadModel();
};

} // namespace control
} // namespace atos
