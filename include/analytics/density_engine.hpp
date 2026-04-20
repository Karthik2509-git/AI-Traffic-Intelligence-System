#pragma once

#include <vector>
#include <map>
#include <string>
#include "core/types.hpp"

namespace antigravity {
namespace analytics {

/**
 * @brief High-Performance Density & Traffic Pressure Analyzer.
 * 
 * Computes lane-level and intersection-level metrics to inform the
 * signal controller and city pressure calculations.
 */
class DensityEngine {
public:
    struct Metrics {
        float totalDensity;
        float laneOccupancy;
        int vehicleCount;
        std::map<int, int> countPerClass;
    };

    DensityEngine() = default;

    /**
     * @brief Process active tracks to generate traffic metrics.
     */
    Metrics analyze(const std::vector<traffic::Track>& tracks);

private:
    float computeOccupancy(const std::vector<traffic::Track>& tracks);
};

} // namespace analytics
} // namespace antigravity
