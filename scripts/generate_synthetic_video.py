"""
generate_synthetic_video.py - Create a synthetic traffic video for testing.

Generates a short video with:
  - A realistic road background with lane markings
  - Moving rectangles simulating vehicles (cars, trucks, bikes)
  - Varying speed and direction per lane
  - Configurable duration, resolution, and vehicle density

Usage:
    python scripts/generate_synthetic_video.py
    python scripts/generate_synthetic_video.py --output data/sample_traffic.mp4 --duration 10 --fps 30
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Vehicle simulation
# ---------------------------------------------------------------------------

class SyntheticVehicle:
    """A rectangle that moves along a lane with a constant speed."""

    VEHICLE_TYPES = {
        "car":        {"w_range": (40, 60),  "h_range": (25, 35),  "colours": [(86, 180, 233), (200, 200, 220), (180, 180, 180), (100, 100, 240)]},
        "truck":      {"w_range": (70, 100), "h_range": (35, 45),  "colours": [(213, 94, 0), (60, 60, 200), (180, 140, 50)]},
        "bus":        {"w_range": (80, 110), "h_range": (30, 40),  "colours": [(0, 158, 115), (40, 100, 200)]},
        "motorcycle": {"w_range": (18, 25),  "h_range": (12, 18),  "colours": [(230, 159, 0), (150, 150, 150)]},
    }

    def __init__(self, lane_y: int, direction: int, frame_w: int) -> None:
        vtype = random.choice(["car", "car", "car", "motorcycle", "truck", "bus"])
        spec  = self.VEHICLE_TYPES[vtype]

        self.w = random.randint(*spec["w_range"])
        self.h = random.randint(*spec["h_range"])
        self.colour = random.choice(spec["colours"])
        self.vtype  = vtype

        self.speed = random.uniform(2.0, 6.0) * direction
        self.y     = lane_y + random.randint(-8, 8)

        if direction > 0:
            self.x = float(-self.w - random.randint(0, 200))
        else:
            self.x = float(frame_w + random.randint(0, 200))

        self.frame_w = frame_w

    def update(self) -> None:
        self.x += self.speed

    def is_off_screen(self) -> bool:
        if self.speed > 0:
            return self.x > self.frame_w + 50
        return self.x + self.w < -50

    def draw(self, frame: np.ndarray) -> None:
        x1, y1 = int(self.x), int(self.y)
        x2, y2 = x1 + self.w, y1 + self.h

        # Body
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colour, -1)
        # Border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), 1)

        # Windshield for cars/trucks/buses
        if self.vtype in ("car", "truck", "bus"):
            ws_w = int(self.w * 0.3)
            if self.speed > 0:
                cv2.rectangle(frame, (x2 - ws_w, y1 + 3), (x2 - 2, y2 - 3), (180, 220, 240), -1)
            else:
                cv2.rectangle(frame, (x1 + 2, y1 + 3), (x1 + ws_w, y2 - 3), (180, 220, 240), -1)


# ---------------------------------------------------------------------------
# Road renderer
# ---------------------------------------------------------------------------

def draw_road(frame: np.ndarray, lane_ys: list[int], lane_height: int) -> None:
    """Draw a multi-lane road with markings."""
    h, w = frame.shape[:2]

    # Asphalt background
    frame[:] = (60, 65, 60)  # dark grey road

    # Sidewalks
    top_sidewalk = max(0, lane_ys[0] - lane_height)
    bot_sidewalk = min(h, lane_ys[-1] + lane_height + 20)
    cv2.rectangle(frame, (0, 0), (w, top_sidewalk), (140, 145, 130), -1)
    cv2.rectangle(frame, (0, bot_sidewalk), (w, h), (140, 145, 130), -1)

    # Lane dividers (dashed white lines)
    for i in range(1, len(lane_ys)):
        divider_y = (lane_ys[i - 1] + lane_ys[i]) // 2 + lane_height // 2
        for x_start in range(0, w, 60):
            cv2.line(frame, (x_start, divider_y), (x_start + 30, divider_y), (200, 200, 200), 2)

    # Road edges (solid yellow)
    road_top = lane_ys[0] - 5
    road_bot = lane_ys[-1] + lane_height + 5
    cv2.line(frame, (0, road_top), (w, road_top), (0, 200, 255), 3)
    cv2.line(frame, (0, road_bot), (w, road_bot), (0, 200, 255), 3)

    # Centre divider (solid yellow, between opposing lanes)
    mid_lane = len(lane_ys) // 2
    if len(lane_ys) >= 2:
        centre_y = (lane_ys[mid_lane - 1] + lane_ys[mid_lane]) // 2 + lane_height // 2
        cv2.line(frame, (0, centre_y - 2), (w, centre_y - 2), (0, 180, 255), 2)
        cv2.line(frame, (0, centre_y + 2), (w, centre_y + 2), (0, 180, 255), 2)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_video(
    output_path: str = "data/sample_traffic.mp4",
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    duration_s: int = 10,
    num_lanes: int = 4,
    spawn_rate: float = 0.08,
) -> Path:
    """
    Generate a synthetic traffic video.

    Parameters
    ----------
    output_path : Output file path.
    width, height : Video resolution.
    fps : Frames per second.
    duration_s : Video duration in seconds.
    num_lanes : Number of traffic lanes.
    spawn_rate : Probability of spawning a vehicle per lane per frame.

    Returns
    -------
    Path to the generated video.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for '{out_path}'")

    # Lane setup
    lane_height = 50
    total_road   = num_lanes * (lane_height + 15) + 40
    road_start   = (height - total_road) // 2 + 40

    lane_ys     = [road_start + i * (lane_height + 15) for i in range(num_lanes)]
    directions  = [1 if i < num_lanes // 2 else -1 for i in range(num_lanes)]

    vehicles: list[SyntheticVehicle] = []
    total_frames = fps * duration_s

    for frame_idx in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw road
        draw_road(frame, lane_ys, lane_height)

        # Spawn vehicles
        for lane_idx, (ly, d) in enumerate(zip(lane_ys, directions)):
            # Vary spawn rate: higher in middle of video for density variation
            progress = frame_idx / total_frames
            dynamic_rate = spawn_rate * (1.0 + 1.5 * np.sin(2 * np.pi * progress))
            dynamic_rate = max(0.02, min(dynamic_rate, 0.20))

            if random.random() < dynamic_rate:
                vehicles.append(SyntheticVehicle(ly, d, width))

        # Update and draw vehicles
        alive: list[SyntheticVehicle] = []
        for v in vehicles:
            v.update()
            if not v.is_off_screen():
                v.draw(frame)
                alive.append(v)
        vehicles = alive

        # Add frame counter + info text
        cv2.putText(
            frame,
            f"Synthetic Traffic | Frame {frame_idx + 1}/{total_frames} | Vehicles: {len(vehicles)}",
            (10, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()
    print(f"[OK] Generated: {out_path} ({total_frames} frames, {duration_s}s @ {fps}fps)")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic traffic video for testing.")
    parser.add_argument("--output", type=str, default="data/sample_traffic.mp4", help="Output file path")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--duration", type=int, default=10, help="Duration in seconds")
    parser.add_argument("--lanes", type=int, default=4, help="Number of lanes")
    parser.add_argument("--spawn-rate", type=float, default=0.08, help="Vehicle spawn probability per lane per frame")
    args = parser.parse_args()

    generate_video(
        output_path=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration_s=args.duration,
        num_lanes=args.lanes,
        spawn_rate=args.spawn_rate,
    )


if __name__ == "__main__":
    main()
