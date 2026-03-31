"""
utils.py — Shared utilities across the AI-Traffic-Intelligence-System.

Provides:
  • Centralised logging factory (structured, coloured, file+console)
  • Project-root resolution
  • YAML config loader with schema validation
  • Frame-rate tracker for video processing
  • Simple thread-safe ring buffer for rolling metric windows
"""

from __future__ import annotations

import logging
import sys
import time
import threading
from collections import deque
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the absolute path to the project root (parent of src/)."""
    return Path(__file__).resolve().parents[1]


def get_output_dir() -> Path:
    d = get_project_root() / "output"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_data_dir() -> Path:
    return get_project_root() / "data"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class _ColourFormatter(logging.Formatter):
    """ANSI-coloured log formatter for console output."""

    LEVEL_COLOURS = {
        logging.DEBUG:    "\033[36m",   # cyan
        logging.INFO:     "\033[32m",   # green
        logging.WARNING:  "\033[33m",   # yellow
        logging.ERROR:    "\033[31m",   # red
        logging.CRITICAL: "\033[35m",   # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self.LEVEL_COLOURS.get(record.levelno, "")
        record.levelname = f"{colour}{record.levelname:8s}{self.RESET}"
        return super().format(record)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger with coloured console + rotating file handlers.

    Handlers are only added once even if *get_logger* is called multiple times
    for the same *name*.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)

    # Console handler (coloured)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(
        _ColourFormatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(ch)

    # File handler (plain text, in output/)
    log_path = get_output_dir() / "traffic_system.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# YAML config loader
# ---------------------------------------------------------------------------

_REQUIRED_KEYS: set[str] = {
    "model",
    "detection",
    "density_thresholds",
    "signal",
    "prediction",
}


def load_config(path: Path | None = None) -> dict[str, Any]:
    """
    Load and return the project YAML configuration.

    Falls back to config/settings.yaml relative to the project root
    when *path* is not provided.  Raises ValueError for missing top-level keys.
    """
    if path is None:
        path = get_project_root() / "config" / "settings.yaml"

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: '{path}'.")

    with open(path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    missing = _REQUIRED_KEYS - cfg.keys()
    if missing:
        raise ValueError(
            f"Config file '{path}' is missing required sections: {sorted(missing)}"
        )

    return cfg


# ---------------------------------------------------------------------------
# Frame-rate / throughput tracker
# ---------------------------------------------------------------------------

class FPSMeter:
    """
    Lightweight frames-per-second counter using a sliding window.

    Usage
    -----
    fps = FPSMeter(window=30)
    fps.tick()          # call after each frame is processed
    print(fps.get())    # current FPS estimate
    """

    def __init__(self, window: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()

    def tick(self) -> None:
        with self._lock:
            self._timestamps.append(time.perf_counter())

    def get(self) -> float:
        with self._lock:
            ts = list(self._timestamps)
        if len(ts) < 2:
            return 0.0
        return (len(ts) - 1) / (ts[-1] - ts[0])


# ---------------------------------------------------------------------------
# Rolling statistics buffer
# ---------------------------------------------------------------------------

class RollingBuffer:
    """
    Thread-safe fixed-length ring buffer for rolling aggregates.

    Parameters
    ----------
    maxlen : int
        Maximum number of samples retained.
    """

    def __init__(self, maxlen: int = 300) -> None:
        self._buf: deque[float] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, value: float) -> None:
        with self._lock:
            self._buf.append(value)

    def mean(self) -> float:
        with self._lock:
            data = list(self._buf)
        return float(sum(data) / len(data)) if data else 0.0

    def max(self) -> float:
        with self._lock:
            return max(self._buf) if self._buf else 0.0

    def to_list(self) -> list[float]:
        with self._lock:
            return list(self._buf)


# ---------------------------------------------------------------------------
# Convenience: ensure directory exists
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if it does not already exist. Return *path*."""
    path.mkdir(parents=True, exist_ok=True)
    return path