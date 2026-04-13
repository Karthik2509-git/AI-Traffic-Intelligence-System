#pragma once

#include <string>
#include <vector>
#include <map>
#include <opencv2/core.hpp>

namespace traffic {

/**
 * @brief Represents vehicle types supported by the system.
 */
enum class VehicleClass {
    CAR,
    MOTORCYCLE,
    BUS,
    TRUCK,
    UNKNOWN
};

/**
 * @brief Helper to convert string to VehicleClass.
 */
inline VehicleClass stringToClass(const std::string& name) {
    if (name == "car") return VehicleClass::CAR;
    if (name == "motorcycle") return VehicleClass::MOTORCYCLE;
    if (name == "bus") return VehicleClass::BUS;
    if (name == "truck") return VehicleClass::TRUCK;
    return VehicleClass::UNKNOWN;
}

/**
 * @brief Represents a tracked vehicle object for a single frame.
 */
struct Track {
    int id;
    cv::Rect2f bbox;
    VehicleClass cls;
    float confidence;
    
    // Derived properties
    inline cv::Point2f getCenter() const {
        return cv::Point2f(bbox.x + bbox.width / 2.0f, bbox.y + bbox.height / 2.0f);
    }
};

/**
 * @brief Represents a polygonal region of interest (Lane).
 */
struct Lane {
    std::string name;
    std::vector<cv::Point2f> polygon;
    float area;

    Lane(std::string name, std::vector<cv::Point2f> pts) 
        : name(std::move(name)), polygon(std::move(pts)) {
        computeArea();
    }

    void computeArea() {
        if (polygon.size() < 3) {
            area = 0;
            return;
        }
        area = static_cast<float>(cv::contourArea(polygon));
    }

    bool contains(cv::Point2f pt) const {
        return cv::pointPolygonTest(polygon, pt, false) >= 0;
    }
};

/**
 * @brief Container for analytical results per frame.
 */
struct FrameResult {
    int frameIndex;
    double timestampMs;
    std::map<std::string, int> countsPerLane;
    int totalCount;
    float congestionScore;
    float occupancyRatio;
    std::vector<Track> tracks;
};

} // namespace traffic
