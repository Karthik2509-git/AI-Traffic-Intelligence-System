@echo off
set "VS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
call "%VS_PATH%" x64
set "INCLUDES=/I include /I C:\opencv\build\include /I C:\TensorRT\include /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\include""
set "CFLAGS=/EHsc /O2 /MD /W3"
echo --- COMPILING MAIN ---
cl.exe %CFLAGS% %INCLUDES% /c src\main.cpp
echo --- COMPILING DETECTOR ---
cl.exe %CFLAGS% %INCLUDES% /c src\engine\detector.cpp
