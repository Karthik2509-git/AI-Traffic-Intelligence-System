#include "density.hpp"
#include <numeric>
#include <algorithm>
#include <cmath>

namespace traffic {

DensityAnalyzer::DensityAnalyzer(const std::vector<Lane>& lanes, Config cfg)
    : lanes(lanes), cfg(cfg) {}

void DensityAnalyzer::reset() {
    ema = 0.0f;
    frameIdx = 0;
    history.clear();
}

FrameResult DensityAnalyzer::update(const std::vector<Track>& tracks, double timestampMs) {
    FrameResult res;
    res.frameIndex = frameIdx++;
    res.timestampMs = timestampMs;
    res.totalCount = 0;

    // 1. Spatial aggregation per lane
    for (const auto& lane : lanes) {
        res.countsPerLane[lane.name] = 0;
    }

    for (const auto& track : tracks) {
        cv::Point2f center = track.getCenter();
        for (const auto& lane : lanes) {
            if (lane.contains(center)) {
                res.countsPerLane[lane.name]++;
            }
        }
    }

    // 2. Summation and EMA
    int currentTotal = 0;
    if (lanes.empty()) {
        currentTotal = static_cast<int>(tracks.size());
    } else {
        for (const auto& pair : res.countsPerLane) {
            currentTotal += pair.second;
        }
    }

    if (frameIdx == 1) {
        ema = static_cast<float>(currentTotal);
    } else {
        ema = cfg.emaAlpha * currentTotal + (1.0f - cfg.emaAlpha) * ema;
    }

    res.totalCount = currentTotal;

    // 3. Occupancy and Congestion
    res.occupancyRatio = computeOccupancy(tracks);
    res.congestionScore = computeCongestion(currentTotal, res.occupancyRatio);
    res.tracks = tracks;

    // 4. History for stats
    history.push_back(currentTotal);
    if (history.size() > 500) history.pop_front();

    return res;
}

float DensityAnalyzer::computeOccupancy(const std::vector<Track>& tracks) {
    float totalLaneArea = 0.0f;
    for (const auto& lane : lanes) totalLaneArea += lane.area;

    if (totalLaneArea <= 0) return 0.0f;

    float vehicleArea = 0.0f;
    for (const auto& track : tracks) {
        vehicleArea += track.bbox.width * track.bbox.height;
    }

    return std::min(vehicleArea / totalLaneArea, 1.0f);
}

float DensityAnalyzer::computeCongestion(int count, float occupancy) {
    float countNorm = std::min(static_cast<float>(count) / static_cast<float>(std::max(cfg.highThreshold, 1)), 1.0f);
    float raw = 0.60f * countNorm + 0.40f * occupancy;
    return raw * 100.0f;
}

float DensityAnalyzer::getRollingAverage(int window) const {
    if (history.empty()) return 0.0f;
    int actualWindow = std::min(static_cast<int>(history.size()), window);
    float sum = std::accumulate(history.end() - actualWindow, history.end(), 0.0f);
    return sum / actualWindow;
}

std::string DensityAnalyzer::getTrend(int window) const {
    if (history.size() < 5) return "stable";
    
    int n = std::min(static_cast<int>(history.size()), window);
    float x_sum = 0, y_sum = 0, xy_sum = 0, x2_sum = 0;
    
    for (int i = 0; i < n; ++i) {
        float y = history[history.size() - n + i];
        x_sum += i;
        y_sum += y;
        xy_sum += i * y;
        x2_sum += i * i;
    }
    
    float slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum);
    
    if (slope > 0.05f) return "rising";
    if (slope < -0.05f) return "falling";
    return "stable";
}

} // namespace traffic
