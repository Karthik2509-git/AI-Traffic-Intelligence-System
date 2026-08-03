#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>

namespace antigravity {
namespace core {

struct EngineConfig {
    std::string model_path = "data/yolov8_4k_optimized.engine";
    int input_width = 640;
    int input_height = 640;
};

struct DetectionConfig {
    float confidence_threshold = 0.20f;
    float nms_threshold = 0.55f;
    int max_detections = 128;
    std::vector<int> vehicle_classes = {2, 3, 5, 7};
};

struct TelemetryConfig {
    std::string target_ip = "127.0.0.1";
    int target_port = 5005;
    int rate_hz = 10;
};

struct SignalConfig {
    float threshold_low = 5.0f;
    float threshold_medium = 10.0f;
    float threshold_high = 20.0f;
    float threshold_critical = 30.0f;
};

struct AnomalyConfig {
    int stall_frames = 15;
    float stall_displacement = 2.0f;
    int trajectory_window = 50;
};

struct VideoConfig {
    std::string default_source = "data/sample_traffic.mp4";
    std::string mobile_ip = "192.168.1.7";
    int mobile_port = 8080;
};

struct AppConfig {
    EngineConfig engine;
    DetectionConfig detection;
    TelemetryConfig telemetry;
    SignalConfig signal;
    AnomalyConfig anomaly;
    VideoConfig video;

    bool isValid = false;
    std::vector<std::string> validationErrors;
};

class ConfigManager {
public:
    static ConfigManager& getInstance();

    bool loadFromFile(const std::string& filepath = "config/settings.yaml");
    const AppConfig& getConfig() const;
    
    static AppConfig getDefaultConfig();
    static bool validateConfig(AppConfig& cfg);

private:
    ConfigManager();
    mutable std::mutex mtx;
    AppConfig currentConfig;
};

} // namespace core
} // namespace antigravity
