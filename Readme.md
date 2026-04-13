# Industrial AI Traffic Intelligence System (AITIS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/Computer%20Vision-YOLOv8-orange.svg)](https://ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An industrial-grade, real-time traffic monitoring and optimization engine powered by computer vision and machine learning. Designed for urban infrastructure management, this system provides high-fidelity vehicle tracking, behavioral analytics, and autonomous signal coordination.

---

## 🚦 Key Capabilities

- **High-Precision Multi-Object Tracking (MOT)**: Advanced SORT implementation with Kalman filtering for robust vehicle persistence across occlusions.
- **Adaptive Signal Optimization**: Proportional-weighted phase allocation based on real-time demand, forecasted congestion, and starvation prevention heuristics.
- **Forecasting & Predictive Analytics**: Calibrated Gradient Boosting models for short-term congestion state prediction (Low/Medium/High).
- **Spatial Topology Analysis**: Lane-aware density monitoring with polygonal ROI segmentation and occupancy ratio calculation.
- **Telemetry & Behavioral Insights**: Real-time speed estimation, anomalous event detection (stagnation, flow-drops, surges), and historical trend analysis.
- **Enterprise-Ready Infrastructure**: SQLite/WAL persistence, multi-camera orchestration, and a centralized `settings.yaml` configuration architecture.

---

## 🏗️ System Architecture

### Vision Pipeline
The core pipeline (`TrafficPipeline`) orchestrates frames through:
1. **Detection Interface**: Multi-scale YOLOv8 inference with Tiled Inference support for high-density environments.
2. **Object Tracking**: Temporal association and trajectory management.
3. **Analytics Engines**:
   - `DensityAnalyzer`: Spatial volume and flow metrics.
   - `SpeedAnalyzer`: Multi-stage filtered telemetry (EMA + Median).
   - `AnomalyDetector`: Statistical monitoring for infrastructure incidents.
4. **Coordination Logic**: `SignalOptimizer` recommending real-time phase adjustments.

### Technology Stack
- **Vision**: Ultralytics (YOLOv8)
- **Analytics**: NumPy, SciPy, Pandas
- **Dashboard**: Streamlit, Altair
- **Forecasting**: Scikit-learn (Gradient Boosting)
- **Database**: SQLite3 (WAL Mode)

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA support (Recommended for high-res inference)

### Configuration
1. Clone the repository:
   ```bash
   git clone https://github.com/Karthik2509-git/AI-Traffic-Intelligence-System.git
   cd AI-Traffic-Intelligence-System
   ```
2. Set up a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/macOS
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Launch the Intelligence Dashboard
The primary interface for monitoring and system calibration.
```bash
streamlit run src/dashboard.py
```

### 2. Configure Operational Parameters
Tweak all thresholds, lane polygons, and model weights in `config/settings.yaml`.

### 3. Verification & CI
Run the comprehensive unit test suite (86 tests) to ensure system stability:
```bash
pytest test.py -v
```

---

## 📊 Feature Highlights

### High-Resolution Inference Mode (1280px)
Optimized for wide-field infrastructure cameras. Self-calibrates the model scale to maximize detection precision for distant vehicles.

### Tiled Inference Pipeline
Dynamically segments the input frame into overlapping tiles to detect small-scale vehicles in extremely dense traffic where global inference might fail.

### Long-Range Detection Handling
Heuristic-based filtering that suppresses uncertain or distant detections to maintain high-signal telemetry for signal timing decisions.

---

## 👨‍💻 Developer & Portfolio Context
This project was refactored for **industrial-grade technical credibility**, featuring:
- **Strict PEP 484 Type Hinting** across the entire codebase.
- **NumPy-style Docstrings** for all library and engine modules.
- **Clean Architecture** with clear separation between vision, orchestration, and UI layers.
- **Robust Error Handling** and factory-based configuration patterns.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.