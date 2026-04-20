"""
detection.py — Vehicle detection engine using YOLO and Tiled Inference.

Provides high-fidelity vehicle detection and tracking integration for 
traffic intelligence systems. Supports standard single-pass inference 
and high-resolution tiled inference for dense urban environments.
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
    "car":        (233, 180,  86),   # Sky-Blue
    "motorcycle": (  0, 159, 230),   # Orange
    "bus":        (115, 158,   0),   # Teal-Green
    "truck":      (156,  79, 118),   # Purple (new distinct color)
}

DENSITY_THRESHOLDS: dict[str, int] = {"low": 10, "medium": 25}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_model(model_name: str = "yolov8n.pt") -> YOLO:
    """
    Initialise a YOLOv8 detection model.

    Parameters
    ----------
    model_name : str
        The filename of the model (e.g., 'yolov8n.pt').

    Returns
    -------
    YOLO
        The loaded Ultralytics YOLO instance.

    Raises
    ------
    RuntimeError
        If the model file is invalid or cannot be loaded.
    """
    logger.info("Loading YOLO model: %s", model_name)
    try:
        model = YOLO(model_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to load YOLO model '{model_name}': {exc}") from exc
    logger.info("Model loaded successfully.")
    return model


def classify_density(count: int) -> str:
    """
    Classify traffic density based on total vehicle count.

    Parameters
    ----------
    count : int
        Current total number of vehicles in the frame.

    Returns
    -------
    str
        Density label ('Low', 'Medium', or 'High').
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
    """
    Extract the class-ID to class-name mapping from a YOLO prediction or model.

    Parameters
    ----------
    prediction : Any
        The Raw prediction result from YOLO.
    model : YOLO
        The YOLO model instance.

    Returns
    -------
    dict[int, str]
        Mapping of class-ID (int) to name (str).
    """
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
    Render colour-coded bounding boxes and labels onto the image frame.

    Parameters
    ----------
    frame : np.ndarray
        The BGR image frame to annotate.
    boxes_data : Any
        YOLO bounding box data.
    names_map : dict[int, str]
        Class ID mapping.
    vehicle_classes : frozenset[str]
        Set of classes considered for drawing.
    confidence_threshold : float
        Min confidence for a box to be rendered.

    Returns
    -------
    np.ndarray
        The annotated image frame (copy of original).
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

        # Draw main rectangle
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        # Add label strip
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


def run_tracking(
    model: YOLO,
    frame: np.ndarray,
    confidence_threshold: float = 0.40,
    iou_threshold: float = 0.45,
    inference_size: int = 640,
    tracker_type: str = "bytetrack.yaml",
    min_motorcycle_conf: float = 0.25,
) -> Any:
    """
    Execute object detection and temporal tracking on a single frame.

    Parameters
    ----------
    model : YOLO
        The pretrained YOLO model instance.
    frame : np.ndarray
        The current video frame (BGR).
    confidence_threshold : float
        Minimum confidence for car/truck/bus detection.
    iou_threshold : float
        NMS IoU threshold for overlapping detections.
    inference_size : int
        Resolution to resize the frame for model input.
    tracker_type : str
        Configuration filename for the tracker (e.g., 'bytetrack.yaml').
    min_motorcycle_conf : float
        Lower threshold specifically for detecting motorcycles.

    Returns
    -------
    Any
        The first Ultralytics 'Results' object for the frame.
    """
    run_conf = min(confidence_threshold, min_motorcycle_conf)
    
    results = model.track(
        source    = frame,
        conf      = run_conf,
        iou       = iou_threshold,
        imgsz     = inference_size,
        tracker   = tracker_type,
        persist   = True,
        verbose   = False,
    )
    return results[0] if results else None



# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Tiled Inference Pipeline (Industrial Detection)
# ---------------------------------------------------------------------------

def _get_tiles(frame: np.ndarray, tile_size: int = 640, overlap: float = 0.25) -> list[dict[str, Any]]:
    """
    Divide a frame into overlapping tiles for high-resolution inference.

    Parameters
    ----------
    frame : np.ndarray
        The input image frame.
    tile_size : int
        Dimension of each square tile.
    overlap : float
        Percentage overlap between adjacent tiles (0.0 - 1.0).

    Returns
    -------
    list[dict[str, Any]]
        List of tile dictionaries containing 'image', 'x_off', and 'y_off'.
    """
    h, w = frame.shape[:2]
    stride = int(tile_size * (1 - overlap))
    
    tiles = []
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = frame[y:y+tile_size, x:x+tile_size]
            tiles.append({"image": tile, "x_off": x, "y_off": y})
            
    # Add bottom-right tile if dimensions are not perfectly divisible
    if (h - tile_size) % stride != 0 or (w - tile_size) % stride != 0:
        y_end = h - tile_size
        x_end = w - tile_size
        tiles.append({"image": frame[y_end:h, x_end:w], "x_off": x_end, "y_off": y_end})
        
    return tiles


def _apply_global_nms(detections: list[dict[str, Any]], iou_threshold: float = 0.65) -> list[dict[str, Any]]:
    """
    Perform Non-Maximum Suppression (NMS) across detections from all tiles.

    Parameters
    ----------
    detections : list[dict[str, Any]]
        Consolidated list of detections with localized coordinates.
    iou_threshold : float
        Overlap threshold for suppression.

    Returns
    -------
    list[dict[str, Any]]
        Filtered list of unique detections.
    """
    if not detections:
        return []
        
    # Sort by confidence descending for greedy selection
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    keep = []
    
    while detections:
        best = detections.pop(0)
        keep.append(best)
        
        # Filter out overlapping boxes of the SAME class
        remaining = []
        for d in detections:
            if d["class_name"] != best["class_name"]:
                remaining.append(d)
                continue
                
            iou = _calculate_iou(best["bbox"], d["bbox"])
            if iou < iou_threshold:
                remaining.append(d)
        detections = remaining
        
    return keep


def _calculate_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """
    Calculate the Intersection-over-Union (IoU) of two bounding boxes.

    Parameters
    ----------
    box1, box2 : Sequence[float]
        Bounding boxes in [x1, y1, x2, y2] format.

    Returns
    -------
    float
        IoU value between 0.0 and 1.0.
    """
    x1, y1, x2, y2 = box1
    xx1, yy1, xx2, yy2 = box2
    
    ix1 = max(x1, xx1)
    iy1 = max(y1, yy1)
    ix2 = min(x2, xx2)
    iy2 = min(y2, yy2)
    
    w = max(0.0, ix2 - ix1)
    h = max(0.0, iy2 - iy1)
    inter = w * h
    
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (xx2 - xx1) * (yy2 - yy1)
    union = area1 + area2 - inter + 1e-6
    
    return float(inter / union)


def run_tiled_inference(
    model: YOLO,
    frame: np.ndarray,
    confidence_threshold: float = 0.20,
    iou_threshold: float = 0.65,
    tile_size: int = 640,
    overlap: float = 0.25,
) -> list[dict[str, Any]]:
    """
    Run high-resolution tiled inference to detect small objects in dense scenes.

    Parameters
    ----------
    model : YOLO
        The pretrained YOLO model instance.
    frame : np.ndarray
        The large image frame (BGR).
    confidence_threshold : float
        Min confidence for per-tile detections.
    iou_threshold : float
        Global NMS overlap threshold.
    tile_size : int
        Size of each square inference tile.
    overlap : float
        Percentage overlap between tiles.

    Returns
    -------
    list[dict[str, Any]]
        Consolidated detections with globally-mapped coordinates.
    """
    tiles = _get_tiles(frame, tile_size, overlap)
    raw_detections = []
    
    for tile in tiles:
        # Run inference on tile
        results = model.predict(
            tile["image"], 
            conf=confidence_threshold, 
            iou=iou_threshold, 
            verbose=False
        )
        prediction = results[0]
        if prediction.boxes is None:
            continue
            
        names_map = _resolve_names(prediction, model)
        xyxy = prediction.boxes.xyxy.cpu().numpy()
        conf = prediction.boxes.conf.cpu().numpy()
        clss = prediction.boxes.cls.cpu().numpy().astype(int)
        
        for i in range(len(clss)):
            class_name = names_map.get(clss[i], "unknown")
            if class_name not in VEHICLE_CLASSES:
                continue
                
            # Map box back to global coordinates
            global_box = [
                float(xyxy[i][0] + tile["x_off"]),
                float(xyxy[i][1] + tile["y_off"]),
                float(xyxy[i][2] + tile["x_off"]),
                float(xyxy[i][3] + tile["y_off"])
            ]
            raw_detections.append({
                "bbox": global_box,
                "confidence": float(conf[i]),
                "class_name": class_name
            })
            
    # Apply Global NMS to remove duplicates at tile boundaries
    return _apply_global_nms(raw_detections, iou_threshold)


def to_tracks(
    results: Any, 
    names_map: dict[int, str],
    confidence_threshold: float = 0.40,
    min_motorcycle_conf: float = 0.25,
) -> list[Track]:
    """
    Convert Ultralytics Results object into a list of internal Track objects.

    Acts as the bridge between the raw YOLO/ByteTrack detection output and 
    the pipeline's analytic modules. Filters by confidence and class.

    Parameters
    ----------
    results : Any
        The results object returned by model.track().
    names_map : dict[int, str]
        Mapping of class IDs to human-readable names.
    confidence_threshold : float
        Min confidence for main vehicle classes (car, truck, bus).
    min_motorcycle_conf : float
        Min confidence for motorcycle detections.

    Returns
    -------
    list[Track]
        List of confirmed/tentative tracks for the current frame.
    """
    from src.tracker import Track
    
    tracks = []
    if results is None or results.boxes is None:
        return tracks
        
    boxes = results.boxes
    # ByteTrack provides .id (tracking IDs)
    if boxes.id is None:
        return tracks
        
    # Extract data
    ids   = boxes.id.cpu().numpy().astype(int)
    xyxy  = boxes.xyxy.cpu().numpy().astype(float)
    confs = boxes.conf.cpu().numpy().astype(float)
    clss  = boxes.cls.cpu().numpy().astype(int)
    
    for i in range(len(ids)):
        cls_id = int(clss[i])
        class_name = names_map.get(cls_id, "unknown")
        conf = float(confs[i])
        
        # Class-specific confidence filtering
        if class_name == "motorcycle":
            if conf < min_motorcycle_conf: continue
        else:
            if conf < confidence_threshold: continue

        # Only include vehicle classes
        if class_name not in VEHICLE_CLASSES:
            continue
            
        tracks.append(Track(
            track_id   = int(ids[i]),
            bbox       = xyxy[i],
            class_name = class_name,
            confidence = conf,
            hit_streak = 5, # Mocked for downstream compatibility
            age        = 5, # Mocked for downstream compatibility
        ))
        
    return tracks



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