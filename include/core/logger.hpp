#pragma once

#include <iostream>
#include <string>
#include <mutex>
#include <fstream>
#include <chrono>
#include <iomanip>

namespace traffic {

/**
 * @brief Thread-safe Logger for high-performance traffic telemetry.
 */
class Logger {
public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    static void info(const std::string& msg) {
        getInstance().log("INFO", msg);
    }

    static void warn(const std::string& msg) {
        getInstance().log("WARN", msg);
    }

    static void error(const std::string& msg) {
        getInstance().log("ERROR", msg);
    }

private:
    std::mutex mtx;

    Logger() = default;
    
    void log(const std::string& level, const std::string& msg) {
        std::lock_guard<std::mutex> lock(mtx);
        
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);

        std::cout << "[" << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S") << "] "
                  << "[" << level << "] " << msg << std::endl;
    }
};

} // namespace traffic
