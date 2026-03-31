"""
Core package for the AI Traffic Intelligence System.

Public API
----------
  TrafficPipeline      : End-to-end video processing orchestrator.
  PipelineConfig       : All tunable pipeline parameters.
  pipeline_from_config : Factory to build a pipeline from settings.yaml.
  load_model           : Load a YOLOv8 model.
  run_detection        : Run detection on a single image.
  MultiCameraManager   : Manage multiple camera pipelines.
  CameraSource         : Definition of a single camera feed.
"""

from src.detection import load_model, run_detection
from src.pipeline import TrafficPipeline, PipelineConfig, pipeline_from_config
from src.multi_camera import MultiCameraManager, CameraSource

__all__ = [
    "TrafficPipeline",
    "PipelineConfig",
    "pipeline_from_config",
    "load_model",
    "run_detection",
    "MultiCameraManager",
    "CameraSource",
]
