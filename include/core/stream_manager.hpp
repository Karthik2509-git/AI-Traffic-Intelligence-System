#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <mutex>
#include <atomic>
#include "core/concurrent_queue.hpp"
#include "core/memory.hpp"

namespace antigravity {
namespace core {

/**
 * @brief Represents a single camera stream processing unit.
 */
struct StreamData {
    int id;
    std::string source;
    std::atomic<bool> active{true};
    // Per-stream buffers or metadata can be added here
};

/**
 * @brief Top-level orchestrator for multi-camera streams.
 * 
 * Manages the connection, buffering, and lifecycle of N cameras. 
 * Bridges the gap between raw video capture and the AI Processing engine.
 */
class StreamManager {
public:
    static StreamManager& getInstance() {
        static StreamManager instance;
        return instance;
    }

    /**
     * @brief Add a new camera source to the system.
     */
    int addStream(const std::string& source);

    /**
     * @brief Remove and stop a camera stream.
     */
    void removeStream(int streamId);

    /**
     * @brief Shutdown all active streams.
     */
    void shutdown();

    std::vector<int> getActiveStreamIds() const;

private:
    StreamManager() = default;
    
    std::map<int, std::unique_ptr<StreamData>> streams;
    mutable std::mutex streams_mutex;
    std::atomic<int> next_stream_id{0};
};

} // namespace core
} // namespace antigravity
