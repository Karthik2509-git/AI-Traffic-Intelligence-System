#pragma once

#include <vector>
#include <string>

namespace antigravity {
namespace control {

/**
 * @brief Signal phase action output.
 */
struct SignalAction {
    int phase_id;
    int duration_extension; // seconds
    bool skip_phase;
};

/**
 * @brief Density-threshold adaptive signal controller.
 *
 * Observes lane densities and queue lengths at an intersection,
 * then extends the green phase for the most congested lane.
 * This is a heuristic controller — no ML model is used.
 */
class SignalController {
public:
    struct Observation {
        std::vector<float> lane_densities;
        std::vector<float> queue_lengths;
        float time_of_day;
    };

    SignalController();

    /** Compute signal timing adjustment based on current densities. */
    SignalAction computePolicy(const Observation& obs);

    /** Placeholder for future online-learning reward feedback. */
    void updateReward(float throughput_reward, float wait_penalty);
};

} // namespace control
} // namespace antigravity
