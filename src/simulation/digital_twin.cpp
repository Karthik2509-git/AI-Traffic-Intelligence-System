#include "simulation/digital_twin.hpp"
#include "core/logger.hpp"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#endif

#include <iostream>
#include <sstream>

namespace antigravity {
namespace simulation {

DigitalTwinBridge::DigitalTwinBridge(const Config& config) : config(config), socket_fd(-1) {
    initSocket();
    traffic::Logger::info("DigitalTwinBridge Initialized. Streaming to " + config.target_ip + ":" + std::to_string(config.target_port));
}

DigitalTwinBridge::~DigitalTwinBridge() {
    closeSocket();
}

void DigitalTwinBridge::initSocket() {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        traffic::Logger::error("Winsock initialization failed.");
        return;
    }
#endif

    socket_fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (socket_fd < 0) {
        traffic::Logger::error("Socket creation failed.");
    }
}

void DigitalTwinBridge::syncState(float city_pressure, int active_phase) {
    if (socket_fd < 0) return;

    sockaddr_in target_addr;
    target_addr.sin_family = AF_INET;
    target_addr.sin_port = htons(config.target_port);
    inet_pton(AF_INET, config.target_ip.c_str(), &target_addr.sin_addr);

    // Construct a standard JSON telemetry packet
    std::stringstream ss;
    ss << "{\"type\":\"city_pulse\", \"pressure\":" << city_pressure 
       << ", \"signal_phase\":" << active_phase << "}";
    
    std::string payload = ss.str();
    sendto(socket_fd, payload.c_str(), static_cast<int>(payload.size()), 0, 
           (struct sockaddr*)&target_addr, sizeof(target_addr));
}

void DigitalTwinBridge::broadcastIncident(const std::string& incident_type, int nodeId) {
    if (socket_fd < 0) return;

    sockaddr_in target_addr;
    target_addr.sin_family = AF_INET;
    target_addr.sin_port = htons(config.target_port);
    inet_pton(AF_INET, config.target_ip.c_str(), &target_addr.sin_addr);

    std::stringstream ss;
    ss << "{\"type\":\"incident_alert\", \"category\":\"" << incident_type 
       << "\", \"node_id\":" << nodeId << "}";
    
    std::string payload = ss.str();
    sendto(socket_fd, payload.c_str(), static_cast<int>(payload.size()), 0, 
           (struct sockaddr*)&target_addr, sizeof(target_addr));
}

void DigitalTwinBridge::closeSocket() {
    if (socket_fd >= 0) {
#ifdef _WIN32
        closesocket(socket_fd);
        WSACleanup();
#else
        close(socket_fd);
#endif
    }
}

} // namespace simulation
} // namespace antigravity
