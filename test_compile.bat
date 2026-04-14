@echo off
set "VS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"

echo [DEBUG] Initializing Environment...
call "%VS_PATH%" x64

echo [DEBUG] Attempting manual compilation of main.cpp...
cl.exe /c /EHsc /I include /I C:\opencv\build\include /I C:\TensorRT\include /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\include" src\main.cpp

if %errorlevel% neq 0 (
    echo [DEBUG] Compilation failed with error code %errorlevel%
) else (
    echo [DEBUG] Compilation succeeded!
)
