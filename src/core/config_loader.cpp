#include "config_loader.hpp"
#include "logger.hpp"
#include <iostream>

namespace traffic {

bool ConfigLoader::load(const std::string& configPath) {
    try {
        YAML::Node config = YAML::LoadFile(configPath);
        
        if (config["model"]) {
            modelPath = config["model"]["path"].as<std::string>("yolov8n.onnx");
            confThreshold = config["model"]["confidence_threshold"].as<float>(0.4f);
            inferenceSize = config["model"]["inference_size"].as<int>(640);
        }

        if (config["lanes"]) {
            parseLanes(config["lanes"]);
        }

        Logger::info("Config loaded successfully from: " + configPath);
        return true;
    } catch (const std::exception& e) {
        Logger::error("Failed to load config: " + std::string(e.what()));
        return false;
    }
}

void ConfigLoader::parseLanes(const YAML::Node& lanesNode) {
    lanes.clear();
    for (YAML::const_iterator it = lanesNode.begin(); it != lanesNode.end(); ++it) {
        std::string name = it->first.as<std::string>();
        std::vector<cv::Point2f> points;
        
        const auto& ptsNode = it->second["polygon"];
        for (const auto& p : ptsNode) {
            points.emplace_back(p[0].as<float>(), p[1].as<float>());
        }
        
        lanes.emplace_back(name, points);
    }
}

} // namespace traffic
