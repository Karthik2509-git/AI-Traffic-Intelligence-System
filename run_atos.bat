@echo off
echo [ATOS] Initializing High-Performance Environment...

:: Define Linkage Paths
set "OPENCV_BIN=C:\opencv\build\x64\vc16\bin"
set "TRT_BIN=C:\TensorRT\lib"
set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"

:: Update PATH locally
set "PATH=%OPENCV_BIN%;%TRT_BIN%;%CUDA_BIN%;%PATH%"

echo [ATOS] Launching 4K Traffic Intelligence Engine...
echo [ATOS] Target: NVIDIA RTX 5050 (Optimized)
echo [ATOS] Sources: Multi-Camera RTSP Simulation

bin\atos_traffic_system.exe

if %errorlevel% neq 0 (
    echo [ERROR] ATOS Engine crashed or failed to launch with code %errorlevel%
) else (
    echo [SUCCESS] ATOS Benchmark completed successfully.
)
pause
