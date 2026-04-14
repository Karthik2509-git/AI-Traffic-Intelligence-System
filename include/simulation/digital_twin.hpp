#pragma once

#include <string>
#include <vector>
#include <memory>
#include "core/types.hpp"

namespace antigravity {
namespace simulation {

/**
 * @brief High-Performance Digital Twin Synchronization Bridge.
 * 
 * Streams real-time traffic telemetry from the 4K AI engine to 
 * external simulation environments (CARLA/SUMO) via UDP/JSON.
 */
class DigitalTwinBridge {
public:
    struct Config {
        std::string target_ip = "127.0.0.1";
        int target_port = 5005;
        int sync_rate_hz = 10;
    };

    DigitalTwinBridge(const Config& config);
    ~DigitalTwinBridge();

    /**
     * @brief Send a snapshot of the current city state to the virtual twin.
     */
    void syncState(float city_pressure, int active_phase);

    /**
     * @brief Broadcast a safety incident to the simulator for event simulation.
     */
    void broadcastIncident(const std::string& incident_type, int nodeId);

private:
    Config config;
    int socket_fd;

    void initSocket();
    void closeSocket();
};

} // namespace simulation
} // namespace antigravity
