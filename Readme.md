# AI Traffic Intelligence System (ATOS v2.0)

A real-time traffic monitoring system built in C++/CUDA, using TensorRT-accelerated YOLOv8 inference on NVIDIA GPUs. Captures live video (IP Webcam or file), detects vehicles, computes traffic density, runs anomaly detection, and streams telemetry over UDP.

---

## Architecture

```
Video Source → OpenCV Capture → CUDA Pinned Memory → TensorRT YOLOv8 → Analytics → UDP Telemetry
     ↑                                                      ↓
  IP Webcam                                         Signal Controller
  or .mp4 file                                      Anomaly Detector
                                                    Density Engine
```

### Pipeline Stages

1. **Capture** — `cv::VideoCapture` reads frames from RTSP/HTTP streams or local files. Auto-reconnects on stream interruption.
2. **Preprocessing** — Fused CUDA kernel performs bilinear resize (source → 640×640), BGR→RGB conversion, and float normalization in a single GPU pass.
3. **Inference** — TensorRT 10 `enqueueV3` executes the serialized YOLOv8 engine. Detections are parsed with class-aware confidence extraction (vehicle classes only: car, motorcycle, bus, truck) and CPU-side NMS.
4. **Analytics** — `CityController` coordinates:
   - `DensityEngine`: vehicle count, lane occupancy, normalized density
   - `AnomalyDetector`: trajectory-based accident detection (stall detection)
   - `SignalController`: density-threshold heuristic for green phase timing
5. **Telemetry** — `DigitalTwinBridge` streams JSON via UDP to a Python receiver at 10 Hz.

---

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (developed on RTX 5050)
- **CUDA**: 12.4
- **TensorRT**: 10.x
- **OpenCV**: 4.9.0 (with prebuilt binaries)
- **Compiler**: Visual Studio 2022+ Build Tools (tested with VS 2026 v18)

---

## Build

```batch
build.bat
```

This script:
1. Initializes the VS build environment via `vcvarsall.bat`
2. Compiles all C++ sources with `cl.exe`
3. Compiles CUDA kernels with `nvcc -allow-unsupported-compiler`
4. Links against OpenCV, TensorRT, and CUDA runtime

Output: `bin/atos_traffic_system.exe`

---

## Run

```batch
:: Default test video
run.bat

:: Live phone camera (requires IP Webcam app on Android)
run.bat mobile

:: Custom source
run.bat http://192.168.1.50:8080/video
run.bat path\to\traffic_video.mp4
```

### Telemetry Receiver

In a separate terminal:
```
python run_atos_telem_test.py
```

This prints live JSON packets: pressure, signal phase, vehicle count.

---

## Project Structure

```
├── build.bat                     # Build script (VS2026 + CUDA 12.4)
├── run.bat                       # Unified launcher
├── run_atos_telem_test.py        # UDP telemetry receiver
├── include/                      # C++ headers
│   ├── core/                     # Logger, memory, types, concurrent queue
│   ├── engine/                   # TensorRT detector
│   ├── analytics/                # Anomaly detector, density engine
│   ├── control/                  # Signal controller
│   ├── network/                  # City controller, road graph
│   └── simulation/               # UDP telemetry bridge
├── src/                          # C++ source files
│   ├── main.cpp                  # Entry point and pipeline orchestrator
│   ├── engine/detector.cpp       # TensorRT inference + YOLOv8 parsing + NMS
│   ├── analytics/                # Anomaly detection, density calculation
│   ├── control/signal_control.cpp
│   ├── network/                  # City controller, road graph
│   ├── simulation/digital_twin.cpp
│   └── cuda/kernel_fusion.cu     # Fused GPU preprocessing kernel
├── data/                         # Model files and test data
│   ├── yolov8_4k_optimized.engine
│   ├── yolov8_4k_optimized.onnx
│   └── test_4k_traffic.mp4
├── config/settings.yaml          # Runtime configuration
├── tools/export_model.py         # ONNX model export utility
├── scripts/generate_synthetic_video.py
├── legacy/                       # Archived Python v1 system
└── output/                       # Runtime output (logs, frames)
```

---

## Configuration

Edit `config/settings.yaml` to adjust detection thresholds, telemetry target, and model paths.

---

## Legacy Python System

The `legacy/` directory contains the original Python-based system (YOLOv8 + Streamlit + SQLite). It is archived for reference. The current v2 system is the C++/CUDA pipeline described above.

---

## License

MIT — see [LICENSE](LICENSE).