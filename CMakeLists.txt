cmake_minimum_required(VERSION 3.18)
project(ATOS_Traffic_Intelligence LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Build Settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Optimized for RTX 50-series (Blackwell Architecture)
set(CMAKE_CUDA_ARCHITECTURES 89 90) 
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# OpenCV (Manual Linkage for v18 compatibility)
set(OpenCV_INCLUDE_DIRS "C:/opencv/build/include")
set(OpenCV_LIBS "C:/opencv/build/x64/vc16/lib/opencv_world490.lib")
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories("C:/opencv/build/x64/vc16/lib")

# CUDA
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# TensorRT (Using verified local path)
set(TENSORRT_ROOT "C:/TensorRT") 
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
