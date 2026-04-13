#include "network/graph.hpp"
#include "core/logger.hpp"

namespace atos {
namespace network {

using namespace traffic;

void RoadGraph::addCameraNode(int id, const std::string& locationName) {
    std::lock_guard<std::mutex> lock(graph_mutex);
    auto node = std::make_shared<Node>();
    node->id = id;
    node->name = locationName;
    nodes[id] = node;
    Logger::info("GraphEngine: Added Camera Node " + std::to_string(id) + " at " + locationName);
}

void RoadGraph::addRoadConnection(int fromId, int toId, float distance) {
    std::lock_guard<std::mutex> lock(graph_mutex);
    if (nodes.count(fromId) && nodes.count(toId)) {
        auto segment = std::make_shared<RoadSegment>();
        segment->fromNode = fromId;
        segment->toNode = toId;
        segment->distance = distance;
        segment->avgTravelTime = 0.0f; // Initialized as unknown
        segment->currentDensity = 0.0f;
        
        nodes[fromId]->outgoing.push_back(segment);
        Logger::info("GraphEngine: Connected Node " + std::to_string(fromId) + " -> " + std::to_string(toId));
    }
}

void RoadGraph::updateTravelTime(int fromId, int toId, float timeObserved) {
    std::lock_guard<std::mutex> lock(graph_mutex);
    if (!nodes.count(fromId)) return;

    for (auto& segment : nodes[fromId]->outgoing) {
        if (segment->toNode == toId) {
            // Use EMA (Exponential Moving Average) for travel time smoothing
            float alpha = 0.2f;
            if (segment->avgTravelTime == 0.0f) {
                segment->avgTravelTime = timeObserved;
            } else {
                segment->avgTravelTime = (alpha * timeObserved) + (1.0f - alpha) * segment->avgTravelTime;
            }
        }
    }
}

float RoadGraph::estimateTravelTime(int fromId, int toId) const {
    std::lock_guard<std::mutex> lock(graph_mutex);
    if (!nodes.count(fromId)) return -1.0f;

    for (auto& segment : nodes.at(fromId)->outgoing) {
        if (segment->toNode == toId) {
            return segment->avgTravelTime;
        }
    }
    return -1.0f;
}

} // namespace network
} // namespace atos
