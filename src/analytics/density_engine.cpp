#include "analytics/density_engine.hpp"
#include <algorithm>

namespace antigravity {
namespace analytics {

DensityEngine::Metrics DensityEngine::analyze(const std::vector<traffic::Track>& tracks) {
    Metrics m;
    m.vehicleCount = static_cast<int>(tracks.size());
    m.laneOccupancy = computeOccupancy(tracks);
    
    // Normalizing density (0.0 to 1.0) based on typical capacity
    const int intersectionCapacity = 200; 
    m.totalDensity = std::min(static_cast<float>(m.vehicleCount) / intersectionCapacity, 1.0f);

    return m;
}

float DensityEngine::computeOccupancy(const std::vector<traffic::Track>& tracks) {
    // Logic: Calculate total area of all vehicle bounding boxes relative to road area.
    // For this implementation, we sum the normalized areas.
    float totalArea = 0.0f;
    for (const auto& track : tracks) {
        totalArea += (track.bbox.width * track.bbox.height);
    }
    
    // In a production setup, this would be divided by a pre-configured ROI area.
    return std::min(totalArea / 8294400.0f, 1.0f); // Normalized to 4K resolution (3840x2160)
}

} // namespace analytics
} // namespace antigravity
