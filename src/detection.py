from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import time

import cv2
import numpy as np
from ultralytics import YOLO


# YOLOv8 default COCO vehicle class names we care about.
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}


def get_project_root() -> Path:
    """
    Return the absolute path to the project root directory.

    This assumes this file lives in `project_root/src/`.
    """
    return Path(__file__).resolve().parents[1]


def get_default_paths() -> Tuple[Path, Path]:
    """
    Provide default input and output image paths relative to the project root.

    - Input:  data/test.jpg
    - Output: output/output.jpg
    """
    root = get_project_root()
    input_path = root / "data" / "test.jpg"
    output_path = root / "output" / "output.jpg"
    return input_path, output_path


def load_model(model_name: str = "yolov8n.pt") -> YOLO:
    """
    Load a YOLOv8 model from the ultralytics package.

    Parameters
    ----------
    model_name:
        Model identifier or path understood by ultralytics.YOLO
        (e.g. 'yolov8n.pt', 'yolov8s.pt', or a custom .pt file).
    """
    try:
        model = YOLO(model_name)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to load YOLO model '{model_name}': {exc}") from exc
    return model


def ensure_output_directory(output_path: Path) -> None:
    """
    Ensure that the parent directory of the output path exists.
    """
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)


def classify_density(count: int) -> str:
    """
    Classify traffic density based on the vehicle count.

    Rules:
    - Low: < 10
    - Medium: 10–25
    - High: > 25
    """
    if count < 10:
        return "Low"
    if count <= 25:
        return "Medium"
    return "High"


def compute_total_vehicle_count(counts: Dict[str, int]) -> int:
    """Compute the total vehicle count from a per-class counts dictionary."""
    return sum(counts.values())


def print_detection_summary(
    counts: Dict[str, int],
    total: int,
    density: str,
    processing_time_ms: float,
) -> None:
    """Print a clean human-readable summary of detection + density and timing."""
    print("🚗 Vehicle Detection Summary:")
    # Keep stable order by iterating over the dictionary (it is created in sorted order).
    for cls_name, count in counts.items():
        print(f"{cls_name}: {count}")
    print(f"\nTotal vehicles: {total}")
    print(f"Traffic Density: {density}")
    print(f"Processing time: {processing_time_ms:.2f} ms")


def _get_names_map(prediction, model: YOLO) -> Dict[int, str]:
    """
    Resolve a YOLO class-id to class-name mapping.

    Ultralytics typically provides `results[0].names` (or falls back to `model.names`).
    """
    names_map = getattr(prediction, "names", None) or getattr(model, "names", None)

    # `names` is usually a dict[int, str], but handle list/other mapping shapes defensively.
    if isinstance(names_map, dict):
        return {int(k): str(v) for k, v in names_map.items()}

    if isinstance(names_map, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names_map)}

    raise RuntimeError("Could not resolve YOLO class-id to class-name mapping.")


def run_detection(
    model: YOLO,
    input_path: Path,
    output_path: Path,
    vehicle_classes: List[str] | None = None,
    confidence_threshold: float = 0.4,
) -> Dict[str, Any]:
    """
    Run object detection on a single image and save the annotated output.

    Parameters
    ----------
    model:
        A loaded ultralytics.YOLO model instance.
    input_path:
        Path to the input image.
    output_path:
        Path where the annotated image will be written.
    vehicle_classes:
        List of class names to treat as vehicles. If None, VEHICLE_CLASSES is used.
    confidence_threshold:
        Minimum confidence score for a detection to be counted.

    Returns
    -------
    Dict[str, int]
        Mapping from vehicle class name to count detected in the image.
    """
    if vehicle_classes is None:
        vehicle_classes = sorted(VEHICLE_CLASSES)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input image not found at '{input_path}'. "
            "Make sure 'data/test.jpg' exists relative to the project root."
        )

    # Run inference on the image and measure processing time.
    start_time = time.time()
    try:
        results = model(str(input_path))
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Model inference failed for '{input_path}': {exc}") from exc
    end_time = time.time()
    processing_time_ms = (end_time - start_time) * 1000.0

    if not results:
        raise RuntimeError("YOLO returned no results for the image.")

    prediction = results[0]

    # Prepare counting structure and map class ids to names.
    counts: Dict[str, int] = {name: 0 for name in vehicle_classes}
    names_map = _get_names_map(prediction, model)

    boxes = prediction.boxes
    if boxes is None or boxes.cls is None:
        # No detections at all; still generate and save an annotated image.
        annotated = prediction.plot()
        ensure_output_directory(output_path)
        if not cv2.imwrite(str(output_path), annotated):
            raise IOError(f"Failed to write annotated image to '{output_path}'.")
    else:
        # Extract class IDs and confidence scores.
        class_ids = boxes.cls.cpu().numpy().astype(int)

        # Use both class_id and confidence for filtering.
        if boxes.conf is None:
            raise RuntimeError("YOLO boxes.conf is missing; cannot apply confidence filtering.")
        confidences = boxes.conf.cpu().numpy().astype(float)

        if len(class_ids) != len(confidences):
            raise RuntimeError(
                f"Mismatched detection arrays: class_ids={len(class_ids)} vs confidences={len(confidences)}"
            )

        # Count only vehicle detections with sufficient confidence.
        for class_id, conf in zip(class_ids, confidences):
            if conf < confidence_threshold:
                continue

            class_name = names_map.get(int(class_id))
            if class_name in vehicle_classes:
                counts[class_name] += 1

        # Generate an annotated image and write it to disk.
        annotated_bgr: np.ndarray = prediction.plot()
        ensure_output_directory(output_path)
        if not cv2.imwrite(str(output_path), annotated_bgr):
            raise IOError(f"Failed to write annotated image to '{output_path}'.")

    # Build structured result data for downstream ML/analytics pipelines.
    total = compute_total_vehicle_count(counts)
    density = classify_density(total)
    result_data: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "total_vehicles": total,
        "density": density,
        "counts_per_class": counts,
        "processing_time_ms": processing_time_ms,
    }

    return result_data


def main() -> Dict[str, Any] | None:
    """
    Entry point for running vehicle detection on the default image.

    - Loads YOLOv8.
    - Runs detection on `data/test.jpg`.
    - Saves annotated output to `output/output.jpg`.
    - Prints per-class counts and a total vehicle count.
    """
    input_path, output_path = get_default_paths()

    try:
        model = load_model("yolov8n.pt")
        result_data = run_detection(
            model=model,
            input_path=input_path,
            output_path=output_path,
            vehicle_classes=None,  # Keep only supported vehicle classes
            confidence_threshold=0.4,
        )
    except FileNotFoundError as not_found_err:
        # Provide a clear, user-friendly message for missing input images.
        print(f"[ERROR] {not_found_err}")
        return None
    except Exception as exc:
        # Catch-all for unexpected errors to avoid silent failures.
        print(f"[ERROR] Unexpected failure during detection: {exc}")
        return None

    counts = result_data["counts_per_class"]
    total = result_data["total_vehicles"]
    density = result_data["density"]
    processing_time_ms = result_data["processing_time_ms"]

    print_detection_summary(
        counts=counts,
        total=total,
        density=density,
        processing_time_ms=processing_time_ms,
    )
    print(f"Annotated image saved to: {output_path}")
    print("Structured result data:", result_data)

    return result_data


if __name__ == "__main__":
    main()

