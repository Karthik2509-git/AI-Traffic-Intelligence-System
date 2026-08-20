#include "simulation/digital_twin.hpp"
#include "core/logger.hpp"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#endif

#include <sstream>

namespace antigravity {
namespace simulation {

DigitalTwinBridge::DigitalTwinBridge(const Config& config) : config(config) {
    initSocket();
    traffic::Logger::info("Telemetry bridge active -> " + config.target_ip + ":" + std::to_string(config.target_port));
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
    socket_fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (socket_fd == INVALID_SOCKET) {
        traffic::Logger::error("Socket creation failed.");
    }
#else
    socket_fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (socket_fd < 0) {
        traffic::Logger::error("Socket creation failed.");
    }
#endif
}

void DigitalTwinBridge::syncState(float city_pressure, int active_phase, int vehicle_count) {
#ifdef _WIN32
    if (socket_fd == INVALID_SOCKET) return;
#else
    if (socket_fd < 0) return;
#endif

    sockaddr_in target_addr{};
    target_addr.sin_family = AF_INET;
    target_addr.sin_port = htons(static_cast<u_short>(config.target_port));
    inet_pton(AF_INET, config.target_ip.c_str(), &target_addr.sin_addr);

    std::stringstream ss;
    ss << "{\"type\":\"city_pulse\""
       << ", \"pressure\":" << city_pressure
       << ", \"signal_phase\":" << active_phase
       << ", \"vehicles\":" << vehicle_count
       << "}";

    std::string payload = ss.str();
    sendto(socket_fd, payload.c_str(), static_cast<int>(payload.size()), 0,
           (struct sockaddr*)&target_addr, sizeof(target_addr));
}

void DigitalTwinBridge::broadcastIncident(const std::string& incident_type, int nodeId) {
#ifdef _WIN32
    if (socket_fd == INVALID_SOCKET) return;
#else
    if (socket_fd < 0) return;
#endif

    sockaddr_in target_addr{};
    target_addr.sin_family = AF_INET;
    target_addr.sin_port = htons(static_cast<u_short>(config.target_port));
    inet_pton(AF_INET, config.target_ip.c_str(), &target_addr.sin_addr);

    std::stringstream ss;
    ss << "{\"type\":\"incident_alert\""
       << ", \"category\":\"" << incident_type << "\""
       << ", \"node_id\":" << nodeId
       << "}";

    std::string payload = ss.str();
    sendto(socket_fd, payload.c_str(), static_cast<int>(payload.size()), 0,
           (struct sockaddr*)&target_addr, sizeof(target_addr));
}

void DigitalTwinBridge::syncTracks(const std::string& camera_id, uint64_t frame_index, double timestamp, const std::vector<traffic::Track>& tracks) {
#ifdef _WIN32
    if (socket_fd == INVALID_SOCKET) return;
#else
    if (socket_fd < 0) return;
#endif

    sockaddr_in target_addr{};
    target_addr.sin_family = AF_INET;
    target_addr.sin_port = htons(static_cast<u_short>(config.target_port));
    inet_pton(AF_INET, config.target_ip.c_str(), &target_addr.sin_addr);

    std::stringstream ss;
    ss << "{\"type\":\"track_telemetry\""
       << ", \"camera_id\":\"" << camera_id << "\""
       << ", \"frame_index\":" << frame_index
       << ", \"timestamp\":" << std::fixed << std::setprecision(3) << timestamp
       << ", \"tracks\":[";

    bool first = true;
    for (const auto& t : tracks) {
        // Vehicle classes COCO: 2=car, 3=motorcycle, 5=bus, 7=truck
        if (t.classId != 2 && t.classId != 3 && t.classId != 5 && t.classId != 7) continue;

        if (!first) ss << ",";
        first = false;

        ss << "{\"track_id\":" << t.id
           << ", \"class_id\":" << t.classId
           << ", \"confidence\":" << std::fixed << std::setprecision(2) << t.confidence
           << ", \"bbox\":[" << t.bbox.x << "," << t.bbox.y << "," << t.bbox.width << "," << t.bbox.height << "]}";
    }
    ss << "]}";

    std::string payload = ss.str();
    sendto(socket_fd, payload.c_str(), static_cast<int>(payload.size()), 0,
           (struct sockaddr*)&target_addr, sizeof(target_addr));
}

void DigitalTwinBridge::closeSocket() {
#ifdef _WIN32
    if (socket_fd != INVALID_SOCKET) {
        closesocket(socket_fd);
        WSACleanup();
    }
#else
    if (socket_fd >= 0) {
        close(socket_fd);
    }
#endif
}

} // namespace simulation
} // namespace antigravity
