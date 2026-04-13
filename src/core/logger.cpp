#include "logger.hpp"
#include <chrono>
#include <iomanip>
#include <ctime>

namespace traffic {

std::mutex Logger::mtx;

void Logger::log(LogLevel level, const std::string& message) {
    std::lock_guard<std::mutex> lock(mtx);
    
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::cout << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") 
              << "." << std::setfill('0') << std::setw(3) << ms.count() << "] "
              << "[" << levelToString(level) << "] "
              << message << std::endl;
}

std::string Logger::levelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO";
        case LogLevel::WARNING: return "WARN";
        case LogLevel::ERROR:   return "ERROR";
        default:                return "UNKNOWN";
    }
}

} // namespace traffic
