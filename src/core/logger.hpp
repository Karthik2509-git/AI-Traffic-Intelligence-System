#pragma once

#include <string>
#include <iostream>
#include <mutex>

namespace traffic {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

class Logger {
public:
    static void log(LogLevel level, const std::string& message);
    
    static void debug(const std::string& msg) { log(LogLevel::DEBUG, msg); }
    static void info(const std::string& msg) { log(LogLevel::INFO, msg); }
    static void warn(const std::string& msg) { log(LogLevel::WARNING, msg); }
    static void error(const std::string& msg) { log(LogLevel::ERROR, msg); }

private:
    static std::mutex mtx;
    static std::string levelToString(LogLevel level);
};

} // namespace traffic
