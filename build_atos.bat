@echo off
set "VS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"

if not exist "%VS_PATH%" (
    echo [ERROR] Visual Studio Build Tools v18 not found at expected path.
    exit /b 1
)

echo [ATOS] Initializing x64 Build Environment...
set "NVCC_APPEND_FLAGS=-allow-unsupported-compiler"
call "%VS_PATH%" x64

if exist build rmdir /s /q build
mkdir build
cd build

echo [ATOS] Running CMake Configuration...
cmake .. -G "NMake Makefiles" ^
    -DOpenCV_DIR="C:/opencv/build" ^
    -DTENSORRT_ROOT="C:/TensorRT" ^
    -DCMAKE_BUILD_TYPE=Release

if %errorlevel% neq 0 (
    echo [ERROR] CMake Configuration failed.
    exit /b 1
)

echo [ATOS] Starting High-Performance Compilation...
nmake /f Makefile

if %errorlevel% neq 0 (
    echo [ERROR] Build failed.
    exit /b 1
)

echo [SUCCESS] ATOS Traffic Intelligence System build complete!
echo Executable located in: build/atos_traffic_system.exe
