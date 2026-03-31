"""
dashboard.py - Streamlit-based real-time traffic intelligence dashboard.

Run
---
  streamlit run src/dashboard.py

Features
--------
  - Live vehicle count + density gauge
  - Per-class breakdown bar chart
  - Rolling congestion score time-series
  - Signal timing Gantt-style view per lane (using real SignalOptimizer)
  - Video upload + frame-by-frame processing with TrafficPipeline
  - Camera selector (multi-source simulation)
  - Model confidence / feature importance panel
  - Edge-AI vs cloud performance comparison with real benchmarking
"""

from __future__ import annotations

import time
import tempfile
from pathlib import Path
import sys

# -- Streamlit guard --------------------------------------------------------
try:
    import streamlit as st
    import altair as alt
    import pandas as pd
    import numpy as np
    import cv2
except ImportError as exc:
    print(f"[dashboard] Required package not installed: {exc}")
    print("Install with: pip install streamlit altair pandas opencv-python")
    sys.exit(1)

from src.detection import load_model, run_detection, VEHICLE_CLASSES
from src.pipeline import PipelineConfig, TrafficPipeline
from src.density_analyzer import make_full_frame_lane, FrameDensity
from src.predictor import CongestionPredictor, PredictionResult, build_feature_vector
from src.signal_optimizer import SignalOptimizer, LaneSignalInput
from src.multi_camera import MultiCameraManager, CameraSource
from src.utils import get_project_root, get_logger, load_config

logger = get_logger(__name__)
ROOT   = get_project_root()

# -- Load YAML defaults -----------------------------------------------------
try:
    _yaml_cfg = load_config()
except (FileNotFoundError, ValueError) as _exc:
    logger.warning("Could not load settings.yaml, using hardcoded defaults: %s", _exc)
    _yaml_cfg = {}

_cfg_model     = _yaml_cfg.get("model", {})
_cfg_detection = _yaml_cfg.get("detection", {})
_cfg_signal    = _yaml_cfg.get("signal", {})

_default_model = _cfg_model.get("name", "yolov8n.pt")
_default_conf  = _cfg_detection.get("confidence_threshold", 0.40)
_default_cycle = _cfg_signal.get("cycle_time_s", 120)
_default_min_green = _cfg_signal.get("min_green_s", 10)
_default_max_green = _cfg_signal.get("max_green_s", 90)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title  = "AI Traffic Intelligence System",
    page_icon   = "🚦",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("🚦 Traffic Intelligence")
st.sidebar.markdown("---")

_model_options = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
_model_index   = _model_options.index(_default_model) if _default_model in _model_options else 0

model_choice = st.sidebar.selectbox(
    "YOLO Model",
    options=_model_options,
    index=_model_index,
    help="Larger models -> higher accuracy, slower inference.",
)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.1, max_value=0.9,
    value=float(_default_conf), step=0.05,
)

edge_mode = st.sidebar.toggle(
    "Edge-AI Mode",
    value=False,
    help="Reduces inference resolution to 320px for on-device performance.",
)

cycle_time = st.sidebar.number_input(
    "Signal Cycle (s)", min_value=30, max_value=300, value=int(_default_cycle), step=10,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Input Source**")

input_mode = st.sidebar.radio(
    "Mode", ["Image", "Video"], index=0, horizontal=True,
)

if input_mode == "Image":
    uploaded = st.sidebar.file_uploader("Upload traffic image", type=["jpg", "jpeg", "png"])
    uploaded_video = None
else:
    uploaded_video = st.sidebar.file_uploader("Upload traffic video", type=["mp4", "avi", "mov"])
    uploaded = None

max_video_frames = st.sidebar.number_input(
    "Max video frames", min_value=10, max_value=500, value=90, step=10,
    help="Limit frames to process from the video (for speed).",
) if input_mode == "Video" else 90


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history: list[dict] = []

if "run_count" not in st.session_state:
    st.session_state.run_count = 0

if "video_results" not in st.session_state:
    st.session_state.video_results: list[dict] = []


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.title("🚦 AI Traffic Intelligence Dashboard")
col_status, col_total, col_density, col_cong = st.columns(4)

# -- Header KPI cards -------------------------------------------------------
status_box   = col_status.empty()
total_box    = col_total.empty()
density_box  = col_density.empty()
cong_box     = col_cong.empty()

tab_live, tab_trend, tab_signal, tab_perf, tab_multi = st.tabs(
    ["📸 Live Detection", "📈 Trend & Analytics", "🚦 Signal Optimizer", "⚡ Performance", "📹 Multi-Camera"]
)

# ---------------------------------------------------------------------------
# Tab: Live Detection
# ---------------------------------------------------------------------------

with tab_live:
    col_img, col_breakdown = st.columns([2, 1])

    with col_img:
        st.subheader("Annotated Output")
        img_placeholder = st.empty()

    with col_breakdown:
        st.subheader("Vehicle Breakdown")
        breakdown_chart = st.empty()


# ---------------------------------------------------------------------------
# Tab: Trend & Analytics
# ---------------------------------------------------------------------------

with tab_trend:
    st.subheader("Congestion Score Over Time")
    cong_chart_ph = st.empty()

    st.subheader("Vehicle Count (EMA Smoothed)")
    ema_chart_ph = st.empty()

    st.subheader("Traffic Trend")
    trend_ph = st.empty()


# ---------------------------------------------------------------------------
# Tab: Signal Optimizer
# ---------------------------------------------------------------------------

with tab_signal:
    st.subheader("Recommended Signal Durations")
    signal_ph = st.empty()
    signal_detail_ph = st.empty()
    notes_ph  = st.empty()


# ---------------------------------------------------------------------------
# Tab: Performance
# ---------------------------------------------------------------------------

with tab_perf:
    st.subheader("Inference Performance")
    perf_ph = st.empty()

    st.subheader("Edge-AI vs Cloud Comparison")
    compare_ph = st.empty()

    st.subheader("Detailed Benchmark")
    bench_ph = st.empty()


# ---------------------------------------------------------------------------
# Tab: Multi-Camera
# ---------------------------------------------------------------------------

with tab_multi:
    st.subheader("Multi-Camera Comparative Analysis")
    st.caption("Simulates multiple cameras by processing different regions of the input.")
    multi_summary_ph = st.empty()
    multi_table_ph   = st.empty()
    multi_chart_ph   = st.empty()


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_model(model_name: str):
    return load_model(model_name)


def _get_inference_size() -> int:
    return 320 if edge_mode else 640


def _run_detection_on_image(image_path: Path) -> dict:
    """Run detection on a static image and return structured result."""
    model  = _get_model(model_choice)
    out_p  = ROOT / "output" / "dashboard_output.jpg"
    result = run_detection(
        model                = model,
        input_path           = image_path,
        output_path          = out_p,
        confidence_threshold = conf_threshold,
    )
    result["annotated_path"] = str(out_p)
    return result


def _run_signal_optimizer(result: dict) -> dict:
    """
    Run the real SignalOptimizer on detection results.
    Returns signal info dict with green_time, red_time, advisory, etc.
    """
    total   = result.get("total_vehicles", 0)
    density = result.get("density", "Low")

    optimizer = SignalOptimizer(
        cycle_time_s = int(cycle_time),
        min_green_s  = int(_default_min_green),
        max_green_s  = int(_default_max_green),
    )

    # Build a FrameDensity for the optimizer
    fd = FrameDensity(
        frame_idx=0, timestamp_ms=0,
        counts_per_lane={"Lane 1": total},
        total_count=total,
        density_label=density,
        occupancy_ratio=min(total / 50.0, 1.0),
        congestion_score=min(total * 3.0, 100.0),
        ema_count=float(total),
    )

    # Build a PredictionResult
    label_idx = {"Low": 0, "Medium": 1, "High": 2}.get(density, 0)
    pred = PredictionResult(
        label=density, label_index=label_idx,
        probabilities={"Low": 0.1, "Medium": 0.3, "High": 0.6} if density == "High"
            else {"Low": 0.6, "Medium": 0.3, "High": 0.1} if density == "Low"
            else {"Low": 0.2, "Medium": 0.6, "High": 0.2},
        confidence=0.7,
    )

    lane_input = LaneSignalInput(
        lane_name="Lane 1", density=fd, prediction=pred, trend="stable",
    )

    try:
        schedule = optimizer.optimise([lane_input])
        lane_out = schedule.lanes[0]
        return {
            "green_time": lane_out.green_time_s,
            "red_time":   int(cycle_time) - lane_out.green_time_s,
            "pressure":   lane_out.pressure,
            "advisory":   lane_out.advisory,
            "notes":      schedule.notes,
            "schedule":   schedule,
        }
    except Exception as exc:
        logger.warning("Signal optimizer failed: %s", exc)
        green = int(cycle_time) // 2
        return {
            "green_time": green,
            "red_time":   int(cycle_time) - green,
            "pressure":   0.0,
            "advisory":   f"Fallback: equal split ({green}s green)",
            "notes":      [],
            "schedule":   None,
        }


def _run_edge_benchmark(image_path: Path) -> dict:
    """
    Actually run inference at both 320px and 640px and return real timing.
    """
    model = _get_model(model_choice)

    # Cloud mode: 640px
    t0 = time.perf_counter()
    model(str(image_path), conf=conf_threshold, imgsz=640, verbose=False)
    cloud_ms = (time.perf_counter() - t0) * 1000.0

    # Edge mode: 320px
    t0 = time.perf_counter()
    model(str(image_path), conf=conf_threshold, imgsz=320, verbose=False)
    edge_ms = (time.perf_counter() - t0) * 1000.0

    speedup = cloud_ms / edge_ms if edge_ms > 0 else 1.0

    return {
        "cloud_ms": round(cloud_ms, 1),
        "edge_ms":  round(edge_ms, 1),
        "speedup":  round(speedup, 2),
    }


def _process_video(video_path: Path, max_frames: int = 90) -> list[dict]:
    """Process a video with TrafficPipeline and return per-frame metrics."""
    cfg = PipelineConfig(
        model_name=model_choice,
        confidence_threshold=conf_threshold,
        inference_size=_get_inference_size(),
        frame_skip=2,
        save_annotated=False,
        display=False,
    )
    pipeline = TrafficPipeline(source=str(video_path), config=cfg)

    results = []
    progress = st.progress(0, text="Processing video...")
    frame_count = 0

    for result in pipeline.run():
        frame_count += 1
        if frame_count > max_frames:
            break

        metrics = result.metrics
        metrics["frame_idx"] = result.frame_idx
        results.append(metrics)

        progress.progress(
            min(frame_count / max_frames, 1.0),
            text=f"Frame {frame_count}/{max_frames} | Vehicles: {metrics.get('total_vehicles', 0)} | Density: {metrics.get('density_label', '?')}",
        )

    progress.empty()
    return results


def _simulate_multi_camera(image_path: Path) -> list[dict]:
    """
    Simulate 3 cameras by cropping different regions of the image.
    Returns per-camera detection results.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return []

    h, w = img.shape[:2]
    model = _get_model(model_choice)

    cameras = [
        {"name": "Camera A - North", "crop": (0, 0, w // 2, h // 2)},
        {"name": "Camera B - South", "crop": (w // 2, 0, w, h // 2)},
        {"name": "Camera C - Full",  "crop": (0, 0, w, h)},
    ]

    results = []
    for cam in cameras:
        x1, y1, x2, y2 = cam["crop"]
        crop = img[y1:y2, x1:x2].copy()

        # Run detection on crop
        tmp_in  = ROOT / "output" / f"_multi_{cam['name'].replace(' ', '_')}_in.jpg"
        tmp_out = ROOT / "output" / f"_multi_{cam['name'].replace(' ', '_')}_out.jpg"
        tmp_in.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(tmp_in), crop)

        try:
            det_result = run_detection(
                model=model, input_path=tmp_in, output_path=tmp_out,
                confidence_threshold=conf_threshold,
            )
            det_result["camera_name"] = cam["name"]
            results.append(det_result)
        except Exception as exc:
            logger.warning("Multi-camera detection failed for %s: %s", cam["name"], exc)
            results.append({
                "camera_name": cam["name"],
                "total_vehicles": 0,
                "density": "Low",
                "counts_per_class": {},
                "processing_time_ms": 0,
                "mean_confidence": 0,
            })

    return results


# ---------------------------------------------------------------------------
# UI update helpers
# ---------------------------------------------------------------------------

def _update_kpi_cards(result: dict) -> None:
    density = result.get("density", "--")
    density_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(density, "⚪")
    total   = result.get("total_vehicles", 0)
    ms      = result.get("processing_time_ms", 0.0)

    status_box.metric("Status",  f"{density_emoji} {density}")
    total_box.metric( "Vehicles", total)
    density_box.metric("Density", density)
    cong_box.metric(  "Processing", f"{ms:.0f} ms")


def _update_breakdown(counts: dict) -> None:
    df = pd.DataFrame(
        [{"class": k, "count": v} for k, v in counts.items()]
    )
    if df.empty:
        breakdown_chart.info("No vehicles detected.")
        return

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("class:N", axis=alt.Axis(title="")),
            y=alt.Y("count:Q", axis=alt.Axis(title="Count")),
            color=alt.Color(
                "class:N",
                scale=alt.Scale(
                    domain=["car", "motorcycle", "bus", "truck"],
                    range=["#56b4e9", "#e69f00", "#009e73", "#d55e00"],
                ),
                legend=None,
            ),
            tooltip=["class", "count"],
        )
        .properties(height=250)
    )
    breakdown_chart.altair_chart(chart, use_container_width=True)


def _update_trends() -> None:
    hist = st.session_state.history
    if len(hist) < 2:
        return

    df = pd.DataFrame(hist)

    # Congestion score
    if "congestion_score" in df.columns:
        cong_c = (
            alt.Chart(df)
            .mark_line(color="#d55e00", point=True)
            .encode(
                x=alt.X("run:Q", title="Run"),
                y=alt.Y("congestion_score:Q", title="Congestion Score (0-100)", scale=alt.Scale(domain=[0, 100])),
                tooltip=["run", "congestion_score"],
            )
            .properties(height=220)
        )
        cong_chart_ph.altair_chart(cong_c, use_container_width=True)

    # EMA count
    ema_c = (
        alt.Chart(df)
        .mark_area(line={"color": "#0072b2"}, color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="#0072b2", offset=0),
                alt.GradientStop(color="#56b4e9", offset=1),
            ],
            x1=1, x2=1, y1=1, y2=0,
        ))
        .encode(
            x=alt.X("run:Q", title="Run"),
            y=alt.Y("total_vehicles:Q", title="Total Vehicles"),
            tooltip=["run", "total_vehicles"],
        )
        .properties(height=220)
    )
    ema_chart_ph.altair_chart(ema_c, use_container_width=True)


def _update_signal_real(result: dict) -> None:
    """Use the real SignalOptimizer to compute signal timings."""
    sig = _run_signal_optimizer(result)

    green_s = sig["green_time"]
    red_s   = sig["red_time"]

    df_signal = pd.DataFrame([
        {"Phase": "Green", "Seconds": green_s, "Lane": "Lane 1"},
        {"Phase": "Red",   "Seconds": red_s,   "Lane": "Lane 1"},
    ])

    bar_chart = (
        alt.Chart(df_signal)
        .mark_bar(size=40)
        .encode(
            x=alt.X("Seconds:Q", title="Duration (s)", scale=alt.Scale(domain=[0, int(cycle_time)])),
            y=alt.Y("Lane:N", title=""),
            color=alt.Color(
                "Phase:N",
                scale=alt.Scale(
                    domain=["Green", "Red"],
                    range=["#009e73", "#d55e00"],
                ),
            ),
            tooltip=["Phase", "Seconds"],
        )
        .properties(height=120)
    )
    signal_ph.altair_chart(bar_chart, use_container_width=True)

    # Advisory and details
    signal_detail_ph.markdown(f"""
**Pressure Score:** `{sig['pressure']:.4f}`

**Advisory:** {sig['advisory']}
""")

    if sig["notes"]:
        for note in sig["notes"]:
            notes_ph.warning(note)
    else:
        notes_ph.info(
            f"Signal Optimizer: **{green_s}s green** / **{red_s}s red** "
            f"for **{result.get('density', 'Unknown')}** density "
            f"({result.get('total_vehicles', 0)} vehicles)"
        )


def _update_performance_real(image_path: Path, processing_ms: float) -> None:
    """Run actual benchmark at both resolutions."""
    perf_ph.metric("Current Inference Latency", f"{processing_ms:.0f} ms")

    try:
        bench = _run_edge_benchmark(image_path)

        df_perf = pd.DataFrame({
            "Mode":    ["Edge-AI (320px)", "Cloud (640px)"],
            "Latency": [bench["edge_ms"], bench["cloud_ms"]],
        })
        bar = (
            alt.Chart(df_perf)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Mode:N", axis=alt.Axis(title="")),
                y=alt.Y("Latency:Q", title="Inference Latency (ms)"),
                color=alt.Color(
                    "Mode:N",
                    scale=alt.Scale(
                        domain=["Edge-AI (320px)", "Cloud (640px)"],
                        range=["#56b4e9", "#0072b2"],
                    ),
                    legend=None,
                ),
                tooltip=["Mode", "Latency"],
            )
            .properties(height=200)
        )
        compare_ph.altair_chart(bar, use_container_width=True)

        bench_ph.markdown(f"""
| Metric | Edge-AI (320px) | Cloud (640px) |
|---|---|---|
| Latency | **{bench['edge_ms']:.1f} ms** | **{bench['cloud_ms']:.1f} ms** |
| Speedup | **{bench['speedup']:.2f}x** | 1.00x |
| Suitable for | Raspberry Pi, Jetson Nano | Desktop GPU, Cloud |
""")

    except Exception as exc:
        compare_ph.warning(f"Benchmark failed: {exc}")


def _update_video_trends(video_results: list[dict]) -> None:
    """Show frame-by-frame analytics from video processing."""
    if not video_results:
        return

    df = pd.DataFrame(video_results)

    with tab_trend:
        # Vehicle count over frames
        if "total_vehicles" in df.columns and "frame_idx" in df.columns:
            count_chart = (
                alt.Chart(df)
                .mark_line(color="#0072b2", point=False)
                .encode(
                    x=alt.X("frame_idx:Q", title="Frame"),
                    y=alt.Y("total_vehicles:Q", title="Vehicles Detected"),
                    tooltip=["frame_idx", "total_vehicles", "density_label"],
                )
                .properties(height=250, title="Vehicle Count Over Frames")
            )
            ema_chart_ph.altair_chart(count_chart, use_container_width=True)

        # Congestion score
        if "congestion_score" in df.columns:
            cong_chart = (
                alt.Chart(df)
                .mark_area(
                    line={"color": "#d55e00"},
                    color=alt.Gradient(
                        gradient="linear",
                        stops=[
                            alt.GradientStop(color="#d55e00", offset=0),
                            alt.GradientStop(color="#fee0d2", offset=1),
                        ],
                        x1=1, x2=1, y1=1, y2=0,
                    ),
                )
                .encode(
                    x=alt.X("frame_idx:Q", title="Frame"),
                    y=alt.Y("congestion_score:Q", title="Congestion (0-100)",
                             scale=alt.Scale(domain=[0, 100])),
                    tooltip=["frame_idx", "congestion_score"],
                )
                .properties(height=220, title="Congestion Score Over Frames")
            )
            cong_chart_ph.altair_chart(cong_chart, use_container_width=True)

        # Trend info
        if len(df) > 5:
            last_trend = df["trend"].iloc[-1] if "trend" in df.columns else "stable"
            avg_vehicles = df["total_vehicles"].mean() if "total_vehicles" in df.columns else 0
            trend_ph.markdown(f"""
**Video Analysis Summary:**
- Frames processed: **{len(df)}**
- Avg vehicles/frame: **{avg_vehicles:.1f}**
- Final trend: **{last_trend}**
- Peak vehicles: **{df['total_vehicles'].max() if 'total_vehicles' in df.columns else 0}**
""")


def _update_multi_camera(image_path: Path) -> None:
    """Run multi-camera simulation and display results."""
    cam_results = _simulate_multi_camera(image_path)
    if not cam_results:
        multi_summary_ph.warning("No cameras could be processed.")
        return

    # Summary table
    rows = []
    for cr in cam_results:
        rows.append({
            "Camera":     cr.get("camera_name", "?"),
            "Vehicles":   cr.get("total_vehicles", 0),
            "Density":    cr.get("density", "Low"),
            "Confidence": f"{cr.get('mean_confidence', 0):.2%}",
            "Latency":    f"{cr.get('processing_time_ms', 0):.0f} ms",
        })

    df_cam = pd.DataFrame(rows)
    multi_table_ph.dataframe(df_cam, use_container_width=True, hide_index=True)

    # Comparison chart
    df_chart = pd.DataFrame([
        {"Camera": cr.get("camera_name", ""), "Vehicles": cr.get("total_vehicles", 0)}
        for cr in cam_results
    ])

    if not df_chart.empty:
        chart = (
            alt.Chart(df_chart)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("Camera:N", axis=alt.Axis(title="")),
                y=alt.Y("Vehicles:Q", title="Total Vehicles"),
                color=alt.Color("Camera:N", legend=None,
                    scale=alt.Scale(scheme="tableau10")),
                tooltip=["Camera", "Vehicles"],
            )
            .properties(height=250)
        )
        multi_chart_ph.altair_chart(chart, use_container_width=True)

    # System status
    total_all = sum(cr.get("total_vehicles", 0) for cr in cam_results)
    busiest = max(cam_results, key=lambda x: x.get("total_vehicles", 0))
    avg_latency = np.mean([cr.get("processing_time_ms", 0) for cr in cam_results])

    if total_all > 30:
        status_color = "🔴 Critical"
    elif total_all > 15:
        status_color = "🟡 Elevated"
    else:
        status_color = "🟢 Normal"

    multi_summary_ph.markdown(f"""
### System Status: {status_color}

| Metric | Value |
|---|---|
| Total cameras | **{len(cam_results)}** |
| Total vehicles (all cameras) | **{total_all}** |
| Busiest camera | **{busiest.get('camera_name', '?')}** ({busiest.get('total_vehicles', 0)} vehicles) |
| Avg inference latency | **{avg_latency:.0f} ms** |
""")


# ---------------------------------------------------------------------------
# Trigger detection
# ---------------------------------------------------------------------------

run_button = st.button("▶ Run Detection", type="primary", use_container_width=True)

if run_button or uploaded or uploaded_video:
    with st.spinner("Running AI inference..."):
        try:
            # === VIDEO MODE ===
            if input_mode == "Video" and uploaded_video:
                # Save upload to temp
                tmp_video = ROOT / "output" / "_dashboard_upload_video.mp4"
                tmp_video.parent.mkdir(exist_ok=True)
                with open(tmp_video, "wb") as f:
                    f.write(uploaded_video.read())

                video_results = _process_video(tmp_video, max_frames=int(max_video_frames))
                st.session_state.video_results = video_results

                if video_results:
                    last = video_results[-1]
                    # KPI from last frame
                    _update_kpi_cards({
                        "density": last.get("density_label", "Low"),
                        "total_vehicles": last.get("total_vehicles", 0),
                        "processing_time_ms": 0,
                    })

                    _update_video_trends(video_results)

                    # Signal from average
                    avg_vehicles = int(np.mean([r.get("total_vehicles", 0) for r in video_results]))
                    from src.detection import classify_density
                    avg_density = classify_density(avg_vehicles)
                    _update_signal_real({
                        "total_vehicles": avg_vehicles,
                        "density": avg_density,
                    })

                    st.success(
                        f"Video processed: {len(video_results)} frames | "
                        f"Avg vehicles: {avg_vehicles} | Density: {avg_density}"
                    )

            # === IMAGE MODE ===
            else:
                if uploaded:
                    tmp_path = ROOT / "output" / "_dashboard_upload.jpg"
                    tmp_path.parent.mkdir(exist_ok=True)
                    with open(tmp_path, "wb") as f:
                        f.write(uploaded.read())
                    img_path = tmp_path
                else:
                    img_path = ROOT / "data" / "test.jpg"

                if not img_path.is_file():
                    st.error(f"Image not found: {img_path}")
                    st.stop()

                result = _run_detection_on_image(img_path)

                # Compute congestion score for history
                total = result["total_vehicles"]
                cong_score = min(total * 3.0, 100.0)
                result["congestion_score"] = cong_score

                # Append to rolling history
                st.session_state.run_count += 1
                hist_entry = {
                    "run":              st.session_state.run_count,
                    "total_vehicles":   result["total_vehicles"],
                    "density":          result["density"],
                    "processing_ms":    result["processing_time_ms"],
                    "congestion_score": cong_score,
                }
                st.session_state.history.append(hist_entry)

                # Update all UI sections
                _update_kpi_cards(result)

                ann_path = Path(result.get("annotated_path", ""))
                if ann_path.is_file():
                    img_placeholder.image(str(ann_path), use_container_width=True)

                _update_breakdown(result.get("counts_per_class", {}))
                _update_trends()
                _update_signal_real(result)
                _update_performance_real(img_path, result["processing_time_ms"])
                _update_multi_camera(img_path)

                st.success(
                    f"Detection complete: {result['total_vehicles']} vehicles "
                    f"({result['density']} density) in {result['processing_time_ms']:.0f} ms"
                )

        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.exception(e)