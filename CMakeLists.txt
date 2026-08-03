cmake_minimum_required(VERSION 3.18)

# Set CUDA flags before project initialization for compiler check
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

project(AI_Traffic_Intelligence_System 
    VERSION 3.1.0 
    DESCRIPTION "Industrial-Grade AI Traffic Observation System (ATOS)" 
    LANGUAGES CXX CUDA
)

# ------------------------------------------------------------------------------
# 1. C++ & CUDA Standard
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Output directory placement
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# ------------------------------------------------------------------------------
# 2. Compiler Optimization Flags (Language-Gated)
# ------------------------------------------------------------------------------
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:CXX>:/D_CRT_SECURE_NO_WARNINGS>
        $<$<COMPILE_LANGUAGE:CXX>:/DNOMINMAX>
        $<$<COMPILE_LANGUAGE:CXX>:/DWIN32_LEAN_AND_MEAN>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/MD>
    )

    # Flags passed explicitly to NVCC
    string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler /EHsc -Xcompiler /MD")
    string(APPEND CMAKE_CUDA_FLAGS_RELEASE " -Xcompiler /O2")
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3 -march=native>)
endif()

# ------------------------------------------------------------------------------
# 3. Dependency Discovery
# ------------------------------------------------------------------------------
# OpenCV
if(NOT OpenCV_DIR)
    if(EXISTS "C:/opencv/build/x64/vc16/lib")
        set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib" CACHE PATH "Path to OpenCV build")
    elseif(EXISTS "C:/opencv/build")
        set(OpenCV_DIR "C:/opencv/build" CACHE PATH "Path to OpenCV build")
    endif()
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio imgcodecs)
message(STATUS "[ATOS CMake] Found OpenCV: ${OpenCV_VERSION} (${OpenCV_INCLUDE_DIRS})")

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)
message(STATUS "[ATOS CMake] Found CUDA Toolkit: ${CUDAToolkit_VERSION}")

# TensorRT
set(TENSORRT_ROOT "" CACHE PATH "Path to TensorRT installation directory")

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS 
        ${TENSORRT_ROOT} 
        ENV TENSORRT_ROOT
        ENV TensorRT_ROOT
        "C:/TensorRT"
        "/usr/include/x86_64-linux-gnu"
        "/usr/local/tensorrt"
    PATH_SUFFIXES include
)

find_library(TENSORRT_LIBRARY 
    NAMES nvinfer_10 nvinfer nvinfer_10.lib nvinfer.lib
    HINTS 
        ${TENSORRT_ROOT} 
        ENV TENSORRT_ROOT
        ENV TensorRT_ROOT
        "C:/TensorRT"
        "/usr/lib/x86_64-linux-gnu"
        "/usr/local/tensorrt"
    PATH_SUFFIXES lib lib64 x64/lib
)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
    message(STATUS "[ATOS CMake] Found TensorRT include: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "[ATOS CMake] Found TensorRT library: ${TENSORRT_LIBRARY}")
else()
    message(FATAL_ERROR "[ATOS CMake] TensorRT NOT found. Please set -DTENSORRT_ROOT=<path_to_tensorrt>")
endif()

# ------------------------------------------------------------------------------
# 4. Source Files & Targets
# ------------------------------------------------------------------------------
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

set(CORE_CPP_SOURCES
    src/analytics/anomaly.cpp
    src/analytics/density_engine.cpp
    src/capture/video_capture.cpp
    src/control/signal_control.cpp
    src/core/config.cpp
    src/engine/detector.cpp
    src/network/city_controller.cpp
    src/network/graph.cpp
    src/simulation/digital_twin.cpp
)

set(CORE_CUDA_SOURCES
    src/cuda/kernel_fusion.cu
)

add_library(atos_core STATIC ${CORE_CPP_SOURCES} ${CORE_CUDA_SOURCES})

target_link_libraries(atos_core
    PUBLIC
        ${OpenCV_LIBS}
        CUDA::cudart
        ${TENSORRT_LIBRARY}
)

if(WIN32)
    target_link_libraries(atos_core PUBLIC ws2_32)
endif()

add_executable(atos_traffic_system src/main.cpp)

target_link_libraries(atos_traffic_system
    PRIVATE
        atos_core
        ${OpenCV_LIBS}
        CUDA::cudart
        ${TENSORRT_LIBRARY}
)

if(WIN32)
    target_link_libraries(atos_traffic_system PRIVATE ws2_32)
endif()

message(STATUS "============================================================")
message(STATUS " ATOS v3.1 Industrial AI Engine Configuration Complete")
message(STATUS " Target Executable: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/atos_traffic_system")
message(STATUS "============================================================")
