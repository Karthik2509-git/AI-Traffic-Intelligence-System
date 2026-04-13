cmake_minimum_required(VERSION 3.18)
project(AITrafficIntelligenceSystem VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# OpenCV (Check for CUDA support if possible)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui cudaarithm cudaimgproc)
include_directories(${OpenCV_INCLUDE_DIRS})

# YAML-CPP for configuration parsing
# (Assuming installed via package manager or vcpkg)
find_package(yaml-cpp REQUIRED)

# TensorRT (For detection engine)
# Note: TensorRT usually needs manual path specification if not in system paths
# find_path(TENSORRT_INCLUDE_DIR NvInfer.h)
# find_library(TENSORRT_LIBRARY nvinfer)

# ---------------------------------------------------------------------------
# Directories & Sources
# ---------------------------------------------------------------------------

include_directories(src)

# Core Files
set(CORE_SOURCES
    src/core/logger.cpp
    src/core/config_loader.cpp
)

# Analytics Files
set(ANALYTICS_SOURCES
    src/analytics/density.cpp
)

# CUDA Sources
set(CUDA_SOURCES
    src/cuda/preprocess.cu
    src/cuda/heatmap.cu
)

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------

add_executable(traffic_system 
    src/main.cpp 
    ${CORE_SOURCES} 
    ${ANALYTICS_SOURCES}
    ${CUDA_SOURCES}
)

target_link_libraries(traffic_system 
    PRIVATE 
    ${OpenCV_LIBS} 
    ${CUDA_LIBRARIES} 
    yaml-cpp
    # nvinfer
)

# ---------------------------------------------------------------------------
# CUDA Configuration
# ---------------------------------------------------------------------------
set_target_properties(traffic_system PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(traffic_system PROPERTIES CUDA_ARCHITECTURES "75;80;86;89") # Target common modern GPUs
