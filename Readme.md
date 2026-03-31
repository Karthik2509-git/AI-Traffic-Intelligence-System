# 🚦 AI Traffic Intelligence System

End-to-end intelligent traffic monitoring and decision-making system using Computer Vision, Multi-Object Tracking, and Machine Learning.

---

## Architecture

```
Video Source / Image
      │
      ▼
YOLOv8 Detection        (src/detection.py)
  • Per-class bounding boxes
  • Confidence + NMS filtering
      │
      ▼
SORT Multi-Object Tracker  (src/tracker.py)
  • Kalman Filter per track
  • Hungarian assignment
  • Consistent vehicle IDs
      │
      ▼
Density Analyser        (src/density_analyzer.py)
  • Lane-aware counting
  • EMA temporal smoothing
  • Occupancy + congestion score
  • Flow rate (veh/min)
      │
      ▼
Congestion Predictor    (src/predictor.py)
  • Gradient Boosting + Platt calibration
  • 7 engineered features incl. time-of-day
  • Auto-retrains on rolling history
      │
      ▼
Signal Optimizer        (src/signal_optimizer.py)
  • Pressure-weighted green-time allocation
  • Starvation prevention
  • Hard safety clamps (min/max green)
      │
      ▼
Annotated Output + Metrics
Streamlit Dashboard     (src/dashboard.py)
```

---

## Project Structure

```
AI-Traffic-Intelligence-System/
├── config/
│   └── settings.yaml        # All tunable parameters
├── data/
│   └── test.jpg             # Default test image
├── output/                  # Annotated frames, logs, saved model
├── src/
│   ├── __init__.py
│   ├── detection.py         # YOLOv8 inference + annotated output
│   ├── tracker.py           # SORT multi-object tracker (Kalman + Hungarian)
│   ├── density_analyzer.py  # Lane-aware density + flow analysis
│   ├── predictor.py         # ML congestion classifier
│   ├── signal_optimizer.py  # Dynamic signal timing engine
│   ├── pipeline.py          # End-to-end video orchestrator
│   ├── dashboard.py         # Streamlit dashboard
│   └── utils.py             # Logging, config, buffers
├── test.py                  # Unit tests (pytest)
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run detection on a static image

```bash
python -m src.detection
# or
python -c "from src.detection import main; main()"
```

Reads `data/test.jpg`, writes `output/output.jpg`.

### 3. Run the full pipeline on a video

```python
from src.pipeline import TrafficPipeline, PipelineConfig
from pathlib import Path

config = PipelineConfig(
    model_name="yolov8n.pt",
    save_annotated=True,
    output_dir=Path("output"),
)

pipeline = TrafficPipeline(source="data/your_video.mp4", config=config)

for result in pipeline.run():
    print(result.metrics)
```

### 4. Launch the Streamlit dashboard

```bash
streamlit run src/dashboard.py
```

### 5. Run tests

```bash
pytest test.py -v --tb=short
pytest test.py -v --cov=src --cov-report=term-missing
```

---

## Configuration

All parameters are in `config/settings.yaml`. Key settings:

| Section | Key | Default | Description |
|---|---|---|---|
| `model` | `name` | `yolov8n.pt` | YOLO model size |
| `detection` | `confidence_threshold` | `0.40` | Min box confidence |
| `detection` | `frame_skip` | `1` | Process every N frames |
| `density_thresholds` | `low` / `high` | `10` / `25` | Count → label thresholds |
| `signal` | `cycle_time_s` | `120` | Total signal cycle |
| `signal` | `min_green_s` | `10` | Hard minimum green |
| `lanes` | _(list)_ | _(full frame)_ | Polygonal lane ROIs |

---

## Multi-Lane Setup

Define lane polygons in `config/settings.yaml`:

```yaml
lanes:
  - name: "Inbound"
    polygon: [[0, 0], [640, 0], [640, 720], [0, 720]]
  - name: "Outbound"
    polygon: [[640, 0], [1280, 0], [1280, 720], [640, 720]]
```

Or programmatically:

```python
from src.density_analyzer import Lane
import numpy as np

lanes = [
    Lane("North",  np.array([[0, 0],   [640, 0],   [640, 360], [0, 360]])),
    Lane("South",  np.array([[0, 360], [640, 360], [640, 720], [0, 720]])),
]
pipeline = TrafficPipeline(source="video.mp4", lanes=lanes)
```

---

## Edge-AI Mode

For Raspberry Pi / Jetson Nano — use the 320px inference size and nano model:

```python
from src.pipeline import PipelineConfig

config = PipelineConfig(
    model_name     = "yolov8n.pt",
    inference_size = 320,     # ~1.4× faster than 640px
    frame_skip     = 2,       # process every other frame
)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (ultralytics) |
| Video I/O | OpenCV |
| Multi-Object Tracking | SORT (Kalman + Hungarian, scipy) |
| Density Analysis | Custom EMA + polygon occupancy |
| Congestion Prediction | Gradient Boosting + Platt calibration (scikit-learn) |
| Signal Optimisation | Pressure-weighted proportional allocation |
| Dashboard | Streamlit + Altair |
| Configuration | PyYAML |
| Testing | pytest + pytest-cov |

---

## License

MIT