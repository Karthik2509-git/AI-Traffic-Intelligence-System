#pragma once

// Must be before any Windows header to prevent min/max macro conflicts with std::
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include <string>
#include <vector>
#include <memory>
#include "core/types.hpp"

#ifdef _WIN32
#include <winsock2.h>
#endif

namespace antigravity {
namespace simulation {

/**
 * @brief UDP telemetry bridge.
 * 
 * Streams real-time JSON telemetry (city pressure, signal phase, vehicle count)
 * to a companion receiver (e.g. run_atos_telem_test.py) on a configurable port.
 */
class DigitalTwinBridge {
public:
    struct Config {
        std::string target_ip = "127.0.0.1";
        int target_port = 5005;
    };

    DigitalTwinBridge(const Config& config);
    ~DigitalTwinBridge();

    /** Send a city state snapshot via UDP. */
    void syncState(float city_pressure, int active_phase, int vehicle_count);

    /** Broadcast an incident alert via UDP. */
    void broadcastIncident(const std::string& incident_type, int nodeId);

private:
    Config config;
#ifdef _WIN32
    SOCKET socket_fd = INVALID_SOCKET;
#else
    int socket_fd = -1;
#endif

    void initSocket();
    void closeSocket();
};

} // namespace simulation
} // namespace antigravity
