#pragma once

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include "types.hpp"

namespace traffic {

/**
 * @brief Singleton for managing system-wide configuration.
 */
class ConfigLoader {
public:
    static ConfigLoader& getInstance() {
        static ConfigLoader instance;
        return instance;
    }

    bool load(const std::string& configPath);

    // Getters for common config parameters
    std::string getModelPath() const { return modelPath; }
    float getConfThreshold() const { return confThreshold; }
    int getInferenceSize() const { return inferenceSize; }
    const std::vector<Lane>& getLanes() const { return lanes; }

private:
    ConfigLoader() = default;
    
    std::string modelPath;
    float confThreshold = 0.4f;
    int inferenceSize = 640;
    std::vector<Lane> lanes;

    void parseLanes(const YAML::Node& lanesNode);
};

} // namespace traffic
