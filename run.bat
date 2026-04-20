@echo off
setlocal enabledelayedexpansion

:: ---------------------------------------------------------------------------
:: ATOS v2.0 Launcher
:: ---------------------------------------------------------------------------
set "PATH=C:\opencv\build\x64\vc16\bin;C:\TensorRT\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;%PATH%"

if "%~1"=="" (
    echo [ATOS] No argument. Use: run.bat [source]
    echo.
    echo   run.bat                          - Uses default test video
    echo   run.bat mobile                   - Connects to IP Webcam phone
    echo   run.bat http://ip:8080/video     - Custom URL
    echo   run.bat path\to\video.mp4        - Local file
    echo.
    echo Starting with default video...
    .\bin\atos_traffic_system.exe
    goto :end
)

if /I "%~1"=="mobile" (
    set "URL=http://192.168.1.7:8080/video"
    echo [ATOS] Mobile mode: !URL!
    echo [HINT] Ensure IP Webcam is running on your phone.
    .\bin\atos_traffic_system.exe "!URL!"
    goto :end
)

echo [ATOS] Source: %~1
.\bin\atos_traffic_system.exe "%~1"

:end
pause
