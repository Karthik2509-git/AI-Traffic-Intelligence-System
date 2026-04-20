"""
heatmap.py — Spatial traffic density visualization engine.

Generates temporal heatmaps using Gaussian kernel density estimation (KDE) 
over vehicle trajectories. Supports alpha-blended overlays and standalone 
statistical map exports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.tracker import Track
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HeatmapConfig:
    """Tunable parameters for heatmap generation."""
    blur_kernel:    int   = 51      # Gaussian blur kernel size (must be odd)
    decay_factor:   float = 0.96    # Temporal decay per frame (0 = no history, 1 = permanent)
    intensity:      float = 5.0     # Intensity per vehicle center point
    alpha:          float = 0.55    # Overlay blend factor (0 = transparent, 1 = opaque)
    colormap:       int   = cv2.COLORMAP_JET   # OpenCV colormap
    export_dir:     str   = "output/heatmaps"


# ---------------------------------------------------------------------------
# Heatmap Generator
# ---------------------------------------------------------------------------

class HeatmapGenerator:
    """
    Rolling spatial density calculator and visualization generator.

    Maintains a continuous accumulation buffer that tracks vehicle spatial 
    concentration over time. Applies Gaussian smoothing and temporal decay 
    to produce polished, analytical heatmaps.

    Parameters
    ----------
    frame_shape : tuple[int, int]
        The (height, width) dimensions of the target video stream.
    config : HeatmapConfig | None
        Configuration overrides. Defaults to global settings.
    """

    def __init__(
        self,
        frame_shape: tuple[int, int],
        config: HeatmapConfig | None = None,
    ) -> None:
        self.height, self.width = frame_shape
        self.cfg = config or HeatmapConfig()

        # Scale detection radius and blur by resolution (baseline 640px)
        self._scale = self.width / 640.0
        self._scaled_radius = max(4, int(12 * self._scale))
        self._scaled_blur = max(5, int(self.cfg.blur_kernel * self._scale))
        if self._scaled_blur % 2 == 0:
            self._scaled_blur += 1

        # High-precision accumulation buffer
        self._accumulator = np.zeros((self.height, self.width), dtype=np.float32)

        self._total_points = 0
        self._frame_count = 0

        logger.info(
            "HeatmapGenerator initialised: %dx%d (scale=%.2f, blur=%d)",
            self.width, self.height, self._scale, self._scaled_blur,
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, tracks: list[Track]) -> None:
        """
        Update the spatial density map with new vehicle positions.

        Parameters
        ----------
        tracks : list[Track]
            List of confirmed vehicle tracks for the current frame.
        """
        # Temporal decay for historical weight reduction
        self._accumulator *= self.cfg.decay_factor

        for track in tracks:
            cx, cy = int(track.centre[0]), int(track.centre[1])

            # Safety bounds check
            if 0 <= cx < self.width and 0 <= cy < self.height:
                cv2.circle(
                    self._accumulator,
                    (cx, cy),
                    radius=self._scaled_radius,
                    color=self.cfg.intensity,
                    thickness=-1,
                )
                self._total_points += 1

        self._frame_count += 1

    def update_from_detections(self, detections: Sequence[dict[str, Any]]) -> None:
        """
        Update the spatial density map directly from detection results.

        Useful for single-frame analysis or when temporal tracking is disabled.

        Parameters
        ----------
        detections : Sequence[dict[str, Any]]
            Detections containing 'bbox' coordinates in [x1, y1, x2, y2] format.
        """
        self._accumulator *= self.cfg.decay_factor

        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) == 4:
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                if 0 <= cx < self.width and 0 <= cy < self.height:
                    cv2.circle(
                        self._accumulator, (cx, cy),
                        radius=self._scaled_radius, color=self.cfg.intensity, thickness=-1,
                    )
                    self._total_points += 1

        self._frame_count += 1

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> np.ndarray:
        """
        Generate a standalone heatmap visualization.

        Applies Gaussian smoothing and colormapping to the accumulation buffer.

        Returns
        -------
        np.ndarray
            The rendered heatmap (BGR, uint8).
        """
        # Temporal smoothing for fluid visualization
        smoothed = cv2.GaussianBlur(self._accumulator, (self._scaled_blur, self._scaled_blur), 0)

        # Dynamic range normalization
        max_val = smoothed.max()
        if max_val > 0:
            normalized = (smoothed / max_val * 255).astype(np.uint8)
        else:
            normalized = np.zeros((self.height, self.width), dtype=np.uint8)

        # Map intensities to perceptual color space
        return cv2.applyColorMap(normalized, self.cfg.colormap)

    def overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Alpha-blend the heatmap over a source image frame.

        Parameters
        ----------
        frame : np.ndarray
            The reference BGR frame for the background.

        Returns
        -------
        np.ndarray
            The blended analytical frame.
        """
        heatmap = self.render()

        if heatmap.shape[:2] != frame.shape[:2]:
            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

        # High-transparency blend for background visibility
        blended = cv2.addWeighted(frame, 1 - self.cfg.alpha, heatmap, self.cfg.alpha, 0)

        # Add professional metadata overlay
        cv2.putText(
            blended, "Spatial Traffic Concentration",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            blended, f"Sample window: {self._frame_count} frames",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )

        return blended

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, output_path: str | Path | None = None) -> Path:
        """Export the current heatmap to a file."""
        if output_path is None:
            out_dir = Path(self.cfg.export_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"heatmap_frame_{self._frame_count}.jpg"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        heatmap = self.render()
        cv2.imwrite(str(output_path), heatmap)
        logger.info("Heatmap exported to: %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Reset & properties
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the accumulation buffer."""
        self._accumulator[:] = 0
        self._total_points = 0
        self._frame_count  = 0

    @property
    def total_points(self) -> int:
        return self._total_points

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def peak_intensity(self) -> float:
        return float(self._accumulator.max())
