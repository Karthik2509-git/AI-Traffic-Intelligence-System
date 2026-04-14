@echo off
setlocal enabledelayedexpansion

:: ---------------------------------------------------------------------------
:: ATOS v2.0 Live Mobile Bridge
:: ---------------------------------------------------------------------------
echo [ATOS] Initializing Live Mobile Hardware Bridge...

:: 1. Define Hardware Paths
set "PATH=C:\opencv\build\x64\vc16\bin;C:\TensorRT\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;%PATH%"

:: 2. Configuration
set "DEFAULT_IP=192.168.1.7"
set "DEFAULT_URL=http://!DEFAULT_IP!:8080/video"

:: 3. Prompt for Mobile IP/URL
echo.
echo [SETUP] Target Bridge: !DEFAULT_IP!
echo [HINT]  Ensure IP Webcam is running on your phone.
echo.
set /p MOBILE_URL="Enter Mobile URL [Default: !DEFAULT_URL!]: "

if "!MOBILE_URL!"=="" (
    set "MOBILE_URL=!DEFAULT_URL!"
)

echo [ATOS] Connecting to Live Mobile Feed: !MOBILE_URL!
.\bin\atos_traffic_system.exe "!MOBILE_URL!"

pause
