"""
utils.py — Shared infrastructure and utility functions.

Provides centralized logging, path resolution, configuration management, 
and performance monitoring tools used across the traffic intelligence system.
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
    """
    Resolve the absolute path to the core project directory.

    Returns
    -------
    Path
        Absolute path to the project root.
    """
    return Path(__file__).resolve().parents[1]


def get_output_dir() -> Path:
    """
    Resolve and ensure existence of the project output directory.

    Returns
    -------
    Path
        Path to the 'output/' directory.
    """
    d = get_project_root() / "output"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_data_dir() -> Path:
    """
    Resolve the path to the project data directory.

    Returns
    -------
    Path
        Path to the 'data/' directory.
    """
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
    Initialise a named logger with industrial-grade formatting.

    Configurations:
      - Console: Coloured ANSI output for immediate feedback.
      - File: Persistent log in 'output/traffic_system.log'.

    Parameters
    ----------
    name : str
        The name of the module or component.
    level : int
        Logging level (e.g., logging.INFO).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console handler (Coloured ANSI)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(
        _ColourFormatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(ch)

    # File handler (Persistent log)
    log_path = get_output_dir() / "traffic_intelligence.log"
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


# ── Configuration ──────────────────────────────────────────────────

_REQUIRED_SECTIONS: set[str] = {
    "model",
    "detection",
    "tracking",
    "analytics",
}


def load_config(path: Path | None = None) -> dict[str, Any]:
    """
    Load the project configuration from a YAML file.

    Prioritises 'config.yaml' at the root if no path is provided. Fallback 
    to 'config/settings.yaml' (legacy) if root config is missing.

    Parameters
    ----------
    path : Path | None
        Override path to the config file.

    Returns
    -------
    dict[str, Any]
        A structured dictionary of configuration parameters.

    Raises
    ------
    FileNotFoundError
        If no configuration file is found in any standard location.
    ValueError
        If required top-level sections are missing.
    """
    root = get_project_root()
    
    # Selection logic: Override -> Root 'config.yaml' -> Legacy 'config/settings.yaml'
    if path is None:
        root_path = root / "config.yaml"
        legacy_path = root / "config" / "settings.yaml"
        path = root_path if root_path.is_file() else legacy_path

    if not path.is_file():
        raise FileNotFoundError(f"No configuration file found at: '{path}'.")

    try:
        with open(path, encoding="utf-8") as fh:
            cfg: dict[str, Any] = yaml.safe_load(fh) or {}
    except Exception as exc:
        raise ValueError(f"Failed to parse config at '{path}': {exc}") from exc

    # Validate essential sections
    missing = _REQUIRED_SECTIONS - cfg.keys()
    if missing:
        # For legacy compatibility, we only warn if it's the legacy file
        if "config.yaml" in str(path):
            raise ValueError(
                f"Config file '{path}' is missing required sections: {sorted(missing)}"
            )
        else:
            logging.getLogger(__name__).warning("Legacy config missing new sections: %s", missing)

    return cfg


# ---------------------------------------------------------------------------
# Frame-rate / throughput tracker
# ---------------------------------------------------------------------------

class FPSMeter:
    """
    Sliding-window frames-per-second (FPS) monitor.

    Provides a thread-safe implementation for tracking real-time processing
    throughput using a configurable temporal window.

    Parameters
    ----------
    window : int
        Number of previous frames to consider for the FPS estimate.
    """

    def __init__(self, window: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()

    def tick(self) -> None:
        """Record the completion of a single frame processing step."""
        with self._lock:
            self._timestamps.append(time.perf_counter())

    def get(self) -> float:
        """
        Calculate the current heart-beat FPS estimate.

        Returns
        -------
        float
            Estimated frames per second (0.0 if insufficient data).
        """
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
    Thread-safe ring buffer for statistical analysis of time-series data.

    Maintains a fixed-size window of recent values and provides fast
    aggregation methods.

    Parameters
    ----------
    maxlen : int
        Maximum number of samples to retain in the buffer.
    """

    def __init__(self, maxlen: int = 300) -> None:
        self._buf: deque[float] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, value: float) -> None:
        """
        Append a new value to the buffer, evicting the oldest if full.

        Parameters
        ----------
        value : float
            The numeric value to add.
        """
        with self._lock:
            self._buf.append(value)

    def mean(self) -> float:
        """
        Calculate the arithmetic mean of all samples in the buffer.

        Returns
        -------
        float
            The mean value (0.0 if empty).
        """
        with self._lock:
            data = list(self._buf)
        return float(sum(data) / len(data)) if data else 0.0

    def max(self) -> float:
        """
        Identify the maximum value in the current window.

        Returns
        -------
        float
            The maximum value (0.0 if empty).
        """
        with self._lock:
            return max(self._buf) if self._buf else 0.0

    def to_list(self) -> list[float]:
        """
        Return a copy of the buffer contents as a standard list.

        Returns
        -------
        list[float]
            List of buffer values.
        """
        with self._lock:
            return list(self._buf)


# ---------------------------------------------------------------------------
# Convenience: ensure directory exists
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if it does not already exist. Return *path*."""
    path.mkdir(parents=True, exist_ok=True)
    return path