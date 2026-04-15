@echo off
setlocal enabledelayedexpansion

:: ---------------------------------------------------------------------------
:: ATOS v2.0 Industrial Build Orchestrator (Hardware-Synchronized)
:: ---------------------------------------------------------------------------
echo [ATOS] Initializing Hardware-Synchronized Build Environment...

:: 1. Define Visual Studio Path (Using Absolute Variable)
set "VS_VARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"

:: 2. Invoke Environmental Handshake (Direct Call to avoid syntax conflicts)
echo [ATOS] Synchronizing with Visual Studio 2026 Build Tools...
call "%VS_VARS%" x64

:: 3. Hardware Dependencies
set "INCLUDES=/I include /I C:\opencv\build\include /I C:\TensorRT\include /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\include""
set "LIBS=/LIBPATH:C:\opencv\build\x64\vc16\lib /LIBPATH:C:\TensorRT\lib /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64""
set "LIB_FILES=opencv_world490.lib nvinfer_10.lib nvinfer_plugin_10.lib nvonnxparser_10.lib cudart.lib user32.lib gdi32.lib shell32.lib ws2_32.lib"
set "CFLAGS=/EHsc /O2 /MD /W3"

:: 4. Clear Legacy Artifacts
if exist *.obj del *.obj
if exist bin\atos_traffic_system.exe del bin\atos_traffic_system.exe

:: 5. Compilation Pass: Industrial Modules
echo [ATOS] Compiling Core Modules...
for /R src %%f in (*.cpp) do (
    echo [%%~nxf]
    cl.exe %CFLAGS% %INCLUDES% /c "%%f"
)

:: 6. Compilation Pass: CUDA Kernels (Bypassing Version Check)
echo [ATOS] Compiling CUDA Kernels...
set "NVCC_INCLUDES=-I include -I C:\opencv\build\include -I C:\TensorRT\include -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\include""
for /R src %%f in (*.cu) do (
    echo [%%~nxf]
    nvcc -c "%%f" -o "%%~nf.obj" %NVCC_INCLUDES% -allow-unsupported-compiler -Xcompiler /MD
)

:: 7. Final Linkage Pass
echo [ATOS] Performing Final High-Performance Linkage...
link.exe /OUT:bin\atos_traffic_system.exe *.obj %LIBS% %LIB_FILES% /SUBSYSTEM:CONSOLE

if %errorlevel% neq 0 (
    echo [ERROR] ATOS Industrial Forge failed.
    exit /b %errorlevel%
)

echo [SUCCESS] ATOS Real AI Engine is now operational: bin\atos_traffic_system.exe
del *.obj
