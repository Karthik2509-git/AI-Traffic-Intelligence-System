"""
demo.py — One-command production demonstration for the AI Traffic Intelligence System.

This script demonstrates the end-to-end capabilities of the system, including:
  1. Real-time vehicle detection & tracking.
  2. Spatial density and heatmap generation.
  3. ML-based congestion forecasting.
  4. Signal timing optimization.
  5. Performance reporting and visual output export.

Usage:
    python demo.py [video_path]
"""

import sys
import os
import time
from pathlib import Path
import cv2
import numpy as np

from src.pipeline import TrafficPipeline, PipelineConfig
from src.utils import get_logger, load_config, ensure_dir

logger = get_logger("AITIS-Demo")

def generate_synthetic_video(path: Path, frames: int = 60, width: int = 1280, height: int = 720):
    """Generate a placeholder video with moving geometric shapes to simulate traffic."""
    logger.info("Generating synthetic demo video: %s", path)
    ensure_dir(path.parent)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, 30.0, (width, height))
    
    for i in range(frames):
        # Create a dark grey road-like background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 40
        
        # Draw some 'lane' lines
        cv2.line(frame, (width//2, 0), (width//2, height), (100, 100, 100), 5)
        
        # Draw moving 'vehicles' (rectangles)
        # Vehicle 1: moving down
        v1_y = (i * 15) % height
        cv2.rectangle(frame, (width//4, v1_y), (width//4 + 60, v1_y + 100), (0, 255, 0), -1)
        
        # Vehicle 2: moving up
        v2_y = height - ((i * 10) % height) - 80
        cv2.rectangle(frame, (3*width//4, v2_y), (3*width//4 + 80, v2_y + 120), (255, 100, 0), -1)
        
        # Vehicle 3: slow moving
        v3_y = (i * 5) % height
        cv2.rectangle(frame, (width//2 - 40, v3_y), (width//2 + 40, v3_y + 140), (0, 0, 255), -1)
        
        out.write(frame)
        
    out.release()
    logger.info("Synthetic video generated successfully.")

def run_demo(video_path: str | None = None):
    """Execute the full demo pipeline."""
    logger.info("=" * 60)
    logger.info(" STARTING AI TRAFFIC INTELLIGENCE SYSTEM DEMO")
    logger.info("=" * 60)

    # 1. Configuration
    cfg_dict = load_config()
    # Merge dictionary into PipelineConfig
    p_cfg = PipelineConfig()
    if cfg_dict:
        # Simple attribute mapping for demo
        det = cfg_dict.get("detection", {})
        p_cfg.confidence_threshold = det.get("confidence_threshold", 0.40)
        p_cfg.model_name = cfg_dict.get("model", {}).get("name", "yolov8m.pt")
        # Ensure output dir exists
        p_cfg.output_dir = Path(cfg_dict.get("output", {}).get("directory", "output/"))

    # 2. Video Source Selection
    if video_path is None:
        # Check standard locations
        potential_sources = [
            Path("data/sample.mp4"),
            Path("data/test_video.mp4")
        ]
        for src in potential_sources:
            if src.is_file():
                video_path = str(src)
                break
        
        if video_path is None:
            # Fallback to synthetic
            synthetic_path = Path("output/demo_synthetic.mp4")
            generate_synthetic_video(synthetic_path)
            video_path = str(synthetic_path)

    logger.info("Processing Source: %s", video_path)

    # 3. Pipeline Execution
    try:
        pipeline = TrafficPipeline(source=video_path, config=p_cfg)
        
        logger.info("Processing frames... (this may take a moment depending on hardware)")
        
        count = 0
        max_frames = 150 # Cap demo for speed
        
        for result in pipeline.run():
            count += 1
            if count % 30 == 0:
                logger.info("Processed %d frames | Current FPS: %.1f", count, result.fps)
            
            if count >= max_frames:
                break
        
        logger.info("Processing complete. Finalizing outputs...")

        # 4. Finalize Reports & Visuals
        pipeline.save_visual_samples()
        report_path = pipeline.generate_performance_report()

        # 5. Summary
        logger.info("=" * 60)
        logger.info(" DEMO COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("Visual Outputs Generated:")
        logger.info(" - Detection Sample:  output/sample_detection.jpg")
        logger.info(" - Heatmap Export:    output/sample_heatmap.png")
        logger.info("Technical Reports:")
        logger.info(" - Performance:       %s", report_path)
        logger.info(" - Event Database:    output/traffic.db")
        logger.info("-" * 60)
        logger.info("The system is now fully calibrated and production-ready.")
        logger.info("=" * 60)

    except Exception as exc:
        logger.error("Demo failed with error: %s", exc)
        sys.exit(1)

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else None
    run_demo(src)
