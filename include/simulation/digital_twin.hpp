#pragma once

#include <vector>
#include <string>
#include <map>
#include "core/types.hpp"

namespace atos {
namespace simulation {

/**
 * @brief City-scale Digital Twin Synchronization Engine.
 * 
 * Synchronizes real-world traffic data (ID signatures, trajectories, signal states) 
 * with a virtual environment in CARLA or SUMO via gRPC or LibCarla.
 */
class DigitalTwinSync {
public:
    struct SimObject {
        int globalId;
        cv::Point3f position;
        float velocity;
        std::string vehicleType;
    };

    DigitalTwinSync(const std::string& host = "localhost", int port = 2000);

    /**
     * @brief High-frequency sync of the ATOS global state to the Digital Twin.
     */
    void sync(const std::vector<SimObject>& objects, int activeSignalPhase);

    /**
     * @brief Command the simulation to run a "what-if" congestion forecast.
     */
    void triggerProjection(float intensityMultiplier);

private:
    std::string sim_host;
    int sim_port;

    // Simulation client (LibCarla or TraCI)
    void connectSimulation();
};

} // namespace simulation
} // namespace atos
