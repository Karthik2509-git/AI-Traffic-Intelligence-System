cmake_minimum_required(VERSION 3.18)
project(ATOS_Traffic_Intelligence LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Build Settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Optimized for RTX 50-series (Blackwell Architecture)
# Note: Adjust compute capability if necessary (e.g., 9.0 for H100/Blackwell)
set(CMAKE_CUDA_ARCHITECTURES 89 90) 

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# CUDA
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# TensorRT (Manually specified for standard Windows installs)
set(TENSORRT_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT") # Placeholder path
include_directories("${TENSORRT_ROOT}/include")
link_directories("${TENSORRT_ROOT}/lib")

# yaml-cpp
find_package(yaml-cpp QUIET)

# ---------------------------------------------------------------------------
# Project Structure & Inclues
# ---------------------------------------------------------------------------
include_directories(include)

# ---------------------------------------------------------------------------
# Source Files
# ---------------------------------------------------------------------------
file(GLOB_RECURSE SOURCES 
    "src/core/*.cpp"
    "src/engine/*.cpp"
    "src/analytics/*.cpp"
    "src/control/*.cpp"
    "src/network/*.cpp"
    "src/simulation/*.cpp"
    "src/main.cpp"
)

file(GLOB_RECURSE CUDA_SOURCES 
    "src/cuda/*.cu"
)

# ---------------------------------------------------------------------------
# Build Executable
# ---------------------------------------------------------------------------
add_executable(atos_traffic_system ${SOURCES} ${CUDA_SOURCES})

# ---------------------------------------------------------------------------
# Linking
# ---------------------------------------------------------------------------
target_link_libraries(atos_traffic_system
    PRIVATE
    ${OpenCV_LIBS}
    nvinfer
    nvparsers
    nvonnxparser
    yaml-cpp
)

# ---------------------------------------------------------------------------
# Optimizations for RTX 5050
# ---------------------------------------------------------------------------
if(MSVC)
    target_compile_options(atos_traffic_system PRIVATE /O2 /Oi /Ot)
else()
    target_compile_options(atos_traffic_system PRIVATE -O3 -ffast-math)
endif()

message(STATUS "ATOS: System configured for RTX 5050 / CUDA 12.4")
