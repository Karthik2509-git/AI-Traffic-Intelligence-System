#pragma once

#include <vector>
#include <deque>
#include <string>
#include "core/types.hpp"

namespace traffic {

/**
 * @brief High-speed traffic density and congestion engine.
 * 
 * Ported from Python implementation with optimization for multi-lane 
 * spatial analysis and EMA smoothing.
 */
class DensityAnalyzer {
public:
    struct Config {
        float emaAlpha = 0.20f;
        int lowThreshold = 10;
        int highThreshold = 25;
        float fps = 30.0f;
    };

    DensityAnalyzer(const std::vector<Lane>& lanes, Config cfg = Config());

    /**
     * @brief Process tracks and generate density snapshot.
     */
    FrameResult update(const std::vector<Track>& tracks, double timestampMs);

    void reset();

    // Stats
    float getRollingAverage(int window = 30) const;
    std::string getTrend(int window = 15) const;

private:
    std::vector<Lane> lanes;
    Config cfg;

    float ema = 0.0f;
    int frameIdx = 0;
    std::deque<int> history;

    float computeOccupancy(const std::vector<Track>& tracks);
    float computeCongestion(int count, float occupancy);
};

} // namespace traffic
