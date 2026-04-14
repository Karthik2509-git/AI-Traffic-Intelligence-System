#pragma once

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <mutex>

namespace antigravity {
namespace network {

/**
 * @brief Represents a road segment connecting two camera nodes.
 */
struct RoadSegment {
    int fromNode;
    int toNode;
    float distance;         // in meters
    float avgTravelTime;    // moving average in seconds
    float currentDensity;   // vehicles per km
};

/**
 * @brief The City-Level Global Traffic Graph.
 * 
 * Maps the relationship between multiple camera nodes and predicts 
 * flow based on vehicle transitions.
 */
class RoadGraph {
public:
    RoadGraph() = default;

    /**
     * @brief Add a camera node to the graph.
     */
    void addCameraNode(int id, const std::string& locationName);

    /**
     * @brief Connect two camera nodes with a road segment.
     */
    void addRoadConnection(int fromId, int toId, float distance);

    /**
     * @brief Update road state based on vehicle Re-ID sightings.
     */
    void updateTravelTime(int fromId, int toId, float timeObserved);

    /**
     * @brief Get predictive travel time using graph metrics.
     */
    float estimateTravelTime(int fromId, int toId) const;

    /**
     * @brief High-frequency update of lane density.
     */
    void updateDensity(int nodeId, float density);

    /**
     * @brief Get current density for a node.
     */
    float getDensity(int nodeId) const;

private:
    struct Node {
        int id;
        std::string name;
        std::vector<std::shared_ptr<RoadSegment>> outgoing;
    };

    std::map<int, std::shared_ptr<Node>> nodes;
    mutable std::mutex graph_mutex;
};

} // namespace network
} // namespace antigravity
