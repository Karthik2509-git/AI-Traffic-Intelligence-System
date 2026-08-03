#include "core/config.hpp"
#include "core/logger.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <iostream>

namespace antigravity {
namespace core {

static inline std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

ConfigManager& ConfigManager::getInstance() {
    static ConfigManager instance;
    return instance;
}

ConfigManager::ConfigManager() {
    currentConfig = getDefaultConfig();
    validateConfig(currentConfig);
}

AppConfig ConfigManager::getDefaultConfig() {
    AppConfig cfg;
    cfg.engine.model_path = "data/yolov8_4k_optimized.engine";
    cfg.engine.input_width = 640;
    cfg.engine.input_height = 640;

    cfg.detection.confidence_threshold = 0.20f;
    cfg.detection.nms_threshold = 0.55f;
    cfg.detection.max_detections = 128;
    cfg.detection.vehicle_classes = {2, 3, 5, 7};

    cfg.telemetry.target_ip = "127.0.0.1";
    cfg.telemetry.target_port = 5005;
    cfg.telemetry.rate_hz = 10;

    cfg.signal.threshold_low = 5.0f;
    cfg.signal.threshold_medium = 10.0f;
    cfg.signal.threshold_high = 20.0f;
    cfg.signal.threshold_critical = 30.0f;

    cfg.anomaly.stall_frames = 15;
    cfg.anomaly.stall_displacement = 2.0f;
    cfg.anomaly.trajectory_window = 50;

    cfg.video.default_source = "data/sample_traffic.mp4";
    cfg.video.mobile_ip = "192.168.1.7";
    cfg.video.mobile_port = 8080;

    return cfg;
}

bool ConfigManager::validateConfig(AppConfig& cfg) {
    cfg.validationErrors.clear();

    if (cfg.engine.input_width <= 0 || cfg.engine.input_width % 32 != 0) {
        cfg.validationErrors.push_back("engine.input_width must be positive and multiple of 32");
    }
    if (cfg.engine.input_height <= 0 || cfg.engine.input_height % 32 != 0) {
        cfg.validationErrors.push_back("engine.input_height must be positive and multiple of 32");
    }

    if (cfg.detection.confidence_threshold < 0.01f || cfg.detection.confidence_threshold > 1.0f) {
        cfg.validationErrors.push_back("detection.confidence_threshold must be in range [0.01, 1.0]");
    }
    if (cfg.detection.nms_threshold < 0.01f || cfg.detection.nms_threshold > 1.0f) {
        cfg.validationErrors.push_back("detection.nms_threshold must be in range [0.01, 1.0]");
    }
    if (cfg.detection.max_detections <= 0) {
        cfg.validationErrors.push_back("detection.max_detections must be > 0");
    }

    if (cfg.telemetry.target_port < 1 || cfg.telemetry.target_port > 65535) {
        cfg.validationErrors.push_back("telemetry.target_port must be in range [1, 65535]");
    }
    if (cfg.telemetry.rate_hz <= 0 || cfg.telemetry.rate_hz > 100) {
        cfg.validationErrors.push_back("telemetry.rate_hz must be in range [1, 100]");
    }

    if (cfg.anomaly.stall_frames <= 0) {
        cfg.validationErrors.push_back("anomaly.stall_frames must be > 0");
    }
    if (cfg.anomaly.trajectory_window <= 0) {
        cfg.validationErrors.push_back("anomaly.trajectory_window must be > 0");
    }

    cfg.isValid = cfg.validationErrors.empty();
    return cfg.isValid;
}

bool ConfigManager::loadFromFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        traffic::Logger::warn("ConfigManager: Cannot open " + filepath + ". Using default configuration.");
        std::lock_guard<std::mutex> lock(mtx);
        currentConfig = getDefaultConfig();
        validateConfig(currentConfig);
        return false;
    }

    AppConfig loaded = getDefaultConfig();
    std::string currentSection;
    std::string line;

    while (std::getline(file, line)) {
        // Strip comment
        auto commentPos = line.find('#');
        if (commentPos != std::string::npos) {
            line = line.substr(0, commentPos);
        }
        line = trim(line);
        if (line.empty()) continue;

        // Section header check
        if (line.back() == ':' && line.find_first_of(" \t") == std::string::npos) {
            currentSection = line.substr(0, line.size() - 1);
            continue;
        }

        // Key-value pair
        auto colonPos = line.find(':');
        if (colonPos == std::string::npos) continue;

        std::string key = trim(line.substr(0, colonPos));
        std::string value = trim(line.substr(colonPos + 1));

        // Strip surrounding quotes
        if (value.size() >= 2 && (value.front() == '"' || value.front() == '\'')) {
            value = value.substr(1, value.size() - 2);
        }

        try {
            if (currentSection == "engine") {
                if (key == "path") loaded.engine.model_path = value;
                else if (key == "input_width") loaded.engine.input_width = std::stoi(value);
                else if (key == "input_height") loaded.engine.input_height = std::stoi(value);
            } else if (currentSection == "detection") {
                if (key == "confidence_threshold") loaded.detection.confidence_threshold = std::stof(value);
                else if (key == "nms_threshold") loaded.detection.nms_threshold = std::stof(value);
                else if (key == "max_detections") loaded.detection.max_detections = std::stoi(value);
            } else if (currentSection == "telemetry") {
                if (key == "target_ip") loaded.telemetry.target_ip = value;
                else if (key == "target_port") loaded.telemetry.target_port = std::stoi(value);
                else if (key == "rate_hz") loaded.telemetry.rate_hz = std::stoi(value);
            } else if (currentSection == "signal") {
                // Section has sub-section thresholds
                if (key == "low") loaded.signal.threshold_low = std::stof(value);
                else if (key == "medium") loaded.signal.threshold_medium = std::stof(value);
                else if (key == "high") loaded.signal.threshold_high = std::stof(value);
                else if (key == "critical") loaded.signal.threshold_critical = std::stof(value);
            } else if (currentSection == "anomaly") {
                if (key == "stall_frames") loaded.anomaly.stall_frames = std::stoi(value);
                else if (key == "stall_displacement") loaded.anomaly.stall_displacement = std::stof(value);
                else if (key == "trajectory_window") loaded.anomaly.trajectory_window = std::stoi(value);
            } else if (currentSection == "video") {
                if (key == "default_source") loaded.video.default_source = value;
                else if (key == "mobile_ip") loaded.video.mobile_ip = value;
                else if (key == "mobile_port") loaded.video.mobile_port = std::stoi(value);
            }
        } catch (const std::exception& e) {
            traffic::Logger::warn("ConfigManager: Parsing error on key '" + key + "': " + e.what());
        }
    }

    file.close();

    bool valid = validateConfig(loaded);
    if (!valid) {
        traffic::Logger::error("ConfigManager: Loaded config has validation errors:");
        for (const auto& err : loaded.validationErrors) {
            traffic::Logger::error("  - " + err);
        }
        traffic::Logger::warn("ConfigManager: Falling back to safe default configuration.");
        loaded = getDefaultConfig();
        validateConfig(loaded);
    }

    std::lock_guard<std::mutex> lock(mtx);
    currentConfig = loaded;
    traffic::Logger::info("ConfigManager: Configuration loaded successfully from " + filepath);
    return true;
}

const AppConfig& ConfigManager::getConfig() const {
    std::lock_guard<std::mutex> lock(mtx);
    return currentConfig;
}

} // namespace core
} // namespace antigravity
