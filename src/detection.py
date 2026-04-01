"""
detection.py — Core vehicle detection engine.

Wraps YOLOv8 inference with:
  • Configurable confidence + NMS-IoU thresholds
  • Per-class count aggregation with confidence-weighted scoring
  • Structured result schema ready for downstream ML / analytics
  • Annotated-image export with per-class colour coding
  • Graceful error handling that never silently swallows exceptions
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from src.utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VEHICLE_CLASSES: frozenset[str] = frozenset({"car", "motorcycle", "bus", "truck"})

# BGR colours used when drawing bounding boxes for each vehicle class.
CLASS_COLOURS: dict[str, tuple[int, int, int]] = {
    "car":        (86,  180, 233),   # sky-blue
    "motorcycle": (230, 159,  0),    # orange
    "bus":        ( 0,  158, 115),   # teal-green
    "truck":      (213,  94,  0),    # vermilion
}

DENSITY_THRESHOLDS: dict[str, int] = {"low": 10, "medium": 25}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_model(model_name: str = "yolov8n.pt") -> YOLO:
    """Load a YOLOv8 model, raising a clear RuntimeError on failure."""
    logger.info("Loading YOLO model: %s", model_name)
    try:
        model = YOLO(model_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to load YOLO model '{model_name}': {exc}") from exc
    logger.info("Model loaded successfully.")
    return model


def classify_density(count: int) -> str:
    """
    Map total vehicle count → traffic density label.

    Thresholds (tweakable via DENSITY_THRESHOLDS):
      Low    : count < 10
      Medium : 10 ≤ count ≤ 25
      High   : count > 25
    """
    if count < DENSITY_THRESHOLDS["low"]:
        return "Low"
    if count <= DENSITY_THRESHOLDS["medium"]:
        return "Medium"
    return "High"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_names(prediction: Any, model: YOLO) -> dict[int, str]:
    """Return a {class_id: class_name} mapping from a YOLO prediction object."""
    names_map = getattr(prediction, "names", None) or getattr(model, "names", None)
    if isinstance(names_map, dict):
        return {int(k): str(v) for k, v in names_map.items()}
    if isinstance(names_map, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names_map)}
    raise RuntimeError("Cannot resolve YOLO class-id → name mapping from model or prediction.")


def _draw_custom_boxes(
    frame: np.ndarray,
    boxes_data: Any,
    names_map: dict[int, str],
    vehicle_classes: frozenset[str],
    confidence_threshold: float,
) -> np.ndarray:
    """
    Draw colour-coded bounding boxes on *frame* (modified in-place, copy returned).

    Each box includes:
      - A filled label strip with class name + confidence score
      - A 2-pixel border in the class-specific colour
    """
    out = frame.copy()
    if boxes_data is None or boxes_data.xyxy is None:
        return out

    xyxys = boxes_data.xyxy.cpu().numpy()
    confs  = boxes_data.conf.cpu().numpy()
    cls_ids = boxes_data.cls.cpu().numpy().astype(int)

    for xyxy, conf, cls_id in zip(xyxys, confs, cls_ids):
        if conf < confidence_threshold:
            continue
        class_name = names_map.get(cls_id)
        if class_name not in vehicle_classes:
            continue

        colour = CLASS_COLOURS.get(class_name, (200, 200, 200))
        x1, y1, x2, y2 = map(int, xyxy)

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        # Label strip
        label = f"{class_name} {conf:.2f}"
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        strip_y2 = max(y1, lh + baseline + 4)
        cv2.rectangle(out, (x1, y1 - lh - baseline - 4), (x1 + lw + 4, strip_y2), colour, -1)
        cv2.putText(
            out, label,
            (x1 + 2, strip_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

    return out


# ---------------------------------------------------------------------------
# Core detection routine
# ---------------------------------------------------------------------------

def run_tracking(
    model: YOLO,
    frame: np.ndarray,
    confidence_threshold: float = 0.40,
    iou_threshold: float = 0.45,
    inference_size: int = 640,
    tracker_type: str = "bytetrack.yaml",
) -> Any:
    """
    Run native SOTA tracking (ByteTrack) on a single frame.
    Returns the Results object from ultralytics.
    """
    results = model.track(
        source    = frame,
        conf      = confidence_threshold,
        iou       = iou_threshold,
        imgsz     = inference_size,
        tracker   = tracker_type,
        persist   = True,
        verbose   = False,
    )
    return results[0] if results else None


def run_detection(
    model: YOLO,
    input_path: Path,
    output_path: Path,
    vehicle_classes: frozenset[str] | None = None,
    confidence_threshold: float = 0.40,
    iou_threshold: float = 0.45,
) -> dict[str, Any]:
    """
    Run YOLOv8 detection on a single image, annotate it, and return metrics.

    Parameters
    ----------
    model               : Loaded ultralytics.YOLO instance.
    input_path          : Path to source image.
    output_path         : Path where the annotated image is written.
    vehicle_classes     : Set of COCO class names to track. Defaults to VEHICLE_CLASSES.
    confidence_threshold: Minimum box confidence (0–1). Boxes below this are ignored.
    iou_threshold       : IoU threshold for non-maximum suppression.

    Returns
    -------
    dict with keys:
        timestamp           : ISO-8601 string
        total_vehicles      : int
        density             : "Low" | "Medium" | "High"
        counts_per_class    : dict[class_name, int]
        mean_confidence     : float   (mean conf of accepted detections)
        processing_time_ms  : float
        output_path         : str
    """
    if vehicle_classes is None:
        vehicle_classes = VEHICLE_CLASSES

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input image not found: '{input_path}'. "
            "Ensure 'data/test.jpg' exists at the project root."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Running inference on '%s'", input_path)
    t_start = time.perf_counter()
    try:
        results = model(
            str(input_path),
            conf=confidence_threshold,
            iou=iou_threshold,
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(f"YOLO inference failed for '{input_path}': {exc}") from exc
    elapsed_ms = (time.perf_counter() - t_start) * 1_000.0

    if not results:
        raise RuntimeError("YOLO returned an empty result list.")

    prediction = results[0]
    names_map  = _resolve_names(prediction, model)

    counts: dict[str, int] = {name: 0 for name in sorted(vehicle_classes)}
    accepted_confidences: list[float] = []

    boxes = prediction.boxes
    if boxes is not None and boxes.cls is not None and len(boxes.cls):
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs   = boxes.conf.cpu().numpy().astype(float)

        if len(cls_ids) != len(confs):
            raise RuntimeError(
                f"Array length mismatch: cls_ids={len(cls_ids)}, confs={len(confs)}."
            )

        for cls_id, conf in zip(cls_ids, confs):
            class_name = names_map.get(int(cls_id))
            if class_name in vehicle_classes:
                counts[class_name] += 1
                accepted_confidences.append(float(conf))

    # --- Annotated output image -------------------------------------------
    raw_frame = cv2.imread(str(input_path))
    annotated  = _draw_custom_boxes(
        raw_frame, boxes, names_map, vehicle_classes, confidence_threshold
    )

    # Overlay HUD: density + counts
    total   = sum(counts.values())
    density = classify_density(total)
    _draw_hud(annotated, counts, total, density)

    if not cv2.imwrite(str(output_path), annotated):
        raise IOError(f"Failed to write annotated image to '{output_path}'.")
    logger.info("Annotated image saved to '%s'", output_path)

    mean_conf = float(np.mean(accepted_confidences)) if accepted_confidences else 0.0

    return {
        "timestamp":          datetime.now().isoformat(),
        "total_vehicles":     total,
        "density":            density,
        "counts_per_class":   counts,
        "mean_confidence":    round(mean_conf, 4),
        "processing_time_ms": round(elapsed_ms, 2),
        "output_path":        str(output_path),
    }


def _draw_hud(
    frame: np.ndarray,
    counts: dict[str, int],
    total: int,
    density: str,
) -> None:
    """Overlay a semi-transparent heads-up display on the annotated frame."""
    density_colours = {"Low": (0, 200, 100), "Medium": (0, 170, 255), "High": (0, 0, 220)}
    colour = density_colours.get(density, (200, 200, 200))

    lines = [f"{cls}: {cnt}" for cls, cnt in counts.items()] + [
        "─" * 18,
        f"Total : {total}",
        f"Traffic: {density}",
    ]
    pad, line_h, font_scale = 10, 22, 0.55
    panel_h = pad * 2 + line_h * len(lines)
    panel_w = 190

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, line in enumerate(lines):
        y = 8 + pad + (i + 1) * line_h - 4
        text_colour = colour if "Traffic" in line else (220, 220, 220)
        cv2.putText(
            frame, line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            text_colour, 1, cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _get_default_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    return root / "data" / "test.jpg", root / "output" / "output.jpg"


def main() -> dict[str, Any] | None:
    """Standalone entry point: detect vehicles in data/test.jpg."""
    input_path, output_path = _get_default_paths()

    try:
        model  = load_model("yolov8n.pt")
        result = run_detection(
            model=model,
            input_path=input_path,
            output_path=output_path,
        )
    except FileNotFoundError as e:
        logger.error("%s", e)
        return None
    except Exception as e:
        logger.exception("Unexpected detection failure: %s", e)
        return None

    _pretty_print(result)
    return result


def _pretty_print(result: dict[str, Any]) -> None:
    print("\n🚗  Vehicle Detection Summary")
    print("─" * 32)
    for cls, cnt in result["counts_per_class"].items():
        print(f"  {cls:<14} {cnt:>3}")
    print("─" * 32)
    print(f"  {'Total':<14} {result['total_vehicles']:>3}")
    print(f"  Traffic density  : {result['density']}")
    print(f"  Mean confidence  : {result['mean_confidence']:.2%}")
    print(f"  Processing time  : {result['processing_time_ms']:.1f} ms")
    print(f"  Output image     : {result['output_path']}")
    print()


if __name__ == "__main__":
    main()