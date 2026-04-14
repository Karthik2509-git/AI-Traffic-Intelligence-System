@echo off
set "VS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"

echo [ATOS] Initializing Iron-Clad Bootstrap Environment...
call "%VS_PATH%" x64

:: ---------------------------------------------------------------------------
:: Paths & Setup
:: ---------------------------------------------------------------------------
set "INCLUDES=/I include /I C:\opencv\build\include /I C:\TensorRT\include /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\include""
set "LIBS=/LIBPATH:C:\opencv\build\x64\vc16\lib /LIBPATH:C:\TensorRT\lib /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64""
set "LIB_FILES=opencv_world490.lib nvinfer_10.lib nvinfer_plugin_10.lib nvonnxparser_10.lib cudart.lib user32.lib gdi32.lib shell32.lib"
set "CFLAGS=/EHsc /O2 /MD /W3"

:: ---------------------------------------------------------------------------
:: Compilation Pass 1: C++ Modules
:: ---------------------------------------------------------------------------
echo [ATOS] Compiling Core Modules...
cl.exe %CFLAGS% %INCLUDES% /c src\main.cpp
cl.exe %CFLAGS% %INCLUDES% /c src\engine\detector.cpp
cl.exe %CFLAGS% %INCLUDES% /c src\core\stream_manager.cpp
cl.exe %CFLAGS% %INCLUDES% /c src\core\thread_pool.cpp
cl.exe %CFLAGS% %INCLUDES% /c src\analytics\anomaly.cpp
cl.exe %CFLAGS% %INCLUDES% /c src\analytics\density_engine.cpp
cl.exe %CFLAGS% %INCLUDES% /c src\network\graph.cpp
cl.exe %CFLAGS% %INCLUDES% /c src\analytics\forecasting.cpp
cl.exe %CFLAGS% %INCLUDES% /c src\control\signal_control.cpp

:: ---------------------------------------------------------------------------
:: Compilation Pass 2: CUDA Kernels
:: ---------------------------------------------------------------------------
echo [ATOS] Compiling CUDA Kernels...
nvcc -c src\cuda\kernel_fusion.cu -o kernel_fusion.obj --include-path include --include-path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\include" -allow-unsupported-compiler -Xcompiler /MD

:: ---------------------------------------------------------------------------
:: Final Linkage Pass
:: ---------------------------------------------------------------------------
echo [ATOS] Performing Final High-Performance Linkage...
cl.exe *.obj /Fe:atos_traffic_system.exe /link %LIBS% %LIB_FILES%

if %errorlevel% neq 0 (
    echo [ERROR] Bootstrap linkage failed.
) else (
    echo [SUCCESS] ATOS Traffic Engine is now operational: atos_traffic_system.exe
)
