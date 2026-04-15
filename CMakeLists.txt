cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Version Bypass for VS 2026 (Industrial Overdrive)
# ---------------------------------------------------------------------------
set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler")

project(ATOS_Traffic_Intelligence VERSION 2.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Hardware Dependency Paths (Hardware-Synchronized)
# ---------------------------------------------------------------------------
set(OPENCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR})

set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4")
find_package(CUDA REQUIRED)

set(TENSORRT_DIR "C:/TensorRT")
include_directories(${TENSORRT_DIR}/include ${CUDA_TOOLKIT_ROOT_DIR}/include include)
link_directories(${TENSORRT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

# ---------------------------------------------------------------------------
# Source Enumeration
# ---------------------------------------------------------------------------
file(GLOB_RECURSE SRC_FILES "src/*.cpp")
file(GLOB_RECURSE CUDA_FILES "src/cuda/*.cu")

# ---------------------------------------------------------------------------
# Build Target: ATOS Engine
# ---------------------------------------------------------------------------
cuda_add_executable(atos_traffic_system ${SRC_FILES} ${CUDA_FILES})

target_link_libraries(atos_traffic_system
    PRIVATE
    ${OpenCV_LIBS}
    nvinfer_10 nvinfer_plugin_10 nvonnxparser_10
    cudart
    ws2_32
)

# ---------------------------------------------------------------------------
# Output Configuration
# ---------------------------------------------------------------------------
set_target_properties(atos_traffic_system PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
