"""
anomaly_detector.py - Real-time traffic anomaly detection engine.

Monitors rolling traffic metrics and fires structured alerts when
statistical anomalies are detected, indicating potential incidents.

Detection Methods
-----------------
  1. Z-Score Spike   : Vehicle count deviates >N std-devs from rolling mean
  2. Sudden Drop     : Flow rate or count drops >X% within a short window
  3. Trend Reversal  : Rapid shift from stable/falling to rising (congestion wave)
  4. Congestion Surge: Congestion score exceeds critical threshold persistently
  5. Speed Anomaly   : Average speed drops suddenly (upstream accident)

Design
------
  - Stateless per-event: each event gets a UUID and is never duplicated
  - Configurable thresholds via settings.yaml
  - Cooldown periods to prevent alert fatigue
  - Severity classification: info, warning, critical
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AnomalyEvent:
    """A detected traffic anomaly."""
    event_id:         str
    timestamp:        datetime
    frame_idx:        int
    anomaly_type:     str        # "spike", "drop", "reversal", "congestion_surge", "speed_anomaly"
    severity:         str        # "info", "warning", "critical"
    description:      str
    confidence:       float      # 0.0 - 1.0
    metrics_snapshot: dict       # frozen metrics at time of event

    def to_dict(self) -> dict[str, Any]:
        d = {
            "event_id":     self.event_id,
            "timestamp":    self.timestamp.isoformat(),
            "frame_idx":    self.frame_idx,
            "anomaly_type": self.anomaly_type,
            "severity":     self.severity,
            "description":  self.description,
            "confidence":   round(self.confidence, 3),
        }
        return d


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AnomalyConfig:
    """Threshold configuration for anomaly detection."""
    # Z-Score spike detection
    zscore_threshold:     float = 2.5     # standard deviations for spike
    zscore_window:        int   = 30      # rolling window size for mean/std

    # Sudden drop detection
    drop_pct_threshold:   float = 0.50    # 50% drop = anomaly
    drop_window:          int   = 10      # frames to compare over

    # Congestion surge
    congestion_critical:  float = 80.0    # score threshold for critical alert
    congestion_persist:   int   = 5       # must persist N frames to trigger

    # Trend reversal
    reversal_accel:       float = 3.0     # acceleration threshold for reversal alert

    # Cooldown (frames between same-type alerts)
    cooldown_frames:      int   = 30      # minimum frames between same-type alerts

    # Speed anomaly
    speed_drop_pct:       float = 0.40    # 40% speed drop = anomaly


# ---------------------------------------------------------------------------
# Anomaly Detector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Statistical traffic anomaly detection engine.

    Feed it frame-level metrics and it will return anomaly events
    when unusual patterns are detected.

    Usage
    -----
    detector = AnomalyDetector()
    events = detector.analyse(frame_idx=42, metrics={...})
    for event in events:
        print(event.severity, event.description)
    """

    def __init__(self, config: AnomalyConfig | None = None) -> None:
        self.cfg = config or AnomalyConfig()

        # Rolling history buffers
        self._count_history: deque[float]   = deque(maxlen=max(self.cfg.zscore_window, 100))
        self._flow_history:  deque[float]   = deque(maxlen=max(self.cfg.drop_window * 3, 50))
        self._cong_history:  deque[float]   = deque(maxlen=max(self.cfg.congestion_persist * 3, 30))
        self._speed_history: deque[float]   = deque(maxlen=30)
        self._trend_history: deque[str]     = deque(maxlen=10)

        # Cooldown tracking: {anomaly_type: last_frame_idx}
        self._last_alert: dict[str, int] = {}

        # Statistics
        self._total_events = 0

        logger.info("AnomalyDetector initialised (zscore=%.1f, drop=%.0f%%)",
                     self.cfg.zscore_threshold, self.cfg.drop_pct_threshold * 100)

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyse(
        self,
        frame_idx: int,
        metrics: dict[str, Any],
    ) -> list[AnomalyEvent]:
        """
        Analyse a single frame's metrics for anomalies.

        Parameters
        ----------
        frame_idx : Current frame index.
        metrics   : Dict with keys: total_vehicles, congestion_score,
                    flow_per_min, trend, ema_count, avg_speed_kmh (optional).

        Returns
        -------
        List of AnomalyEvent objects (may be empty).
        """
        count     = float(metrics.get("total_vehicles", 0))
        cong      = float(metrics.get("congestion_score", 0))
        flow      = float(metrics.get("flow_per_min", 0))
        trend     = metrics.get("trend", "stable")
        avg_speed = metrics.get("avg_speed_kmh", None)

        # Push to history
        self._count_history.append(count)
        self._flow_history.append(flow)
        self._cong_history.append(cong)
        self._trend_history.append(trend)
        if avg_speed is not None:
            self._speed_history.append(float(avg_speed))

        events: list[AnomalyEvent] = []

        # Run detectors
        events.extend(self._check_zscore_spike(frame_idx, count, metrics))
        events.extend(self._check_sudden_drop(frame_idx, metrics))
        events.extend(self._check_congestion_surge(frame_idx, cong, metrics))
        events.extend(self._check_trend_reversal(frame_idx, metrics))
        events.extend(self._check_speed_anomaly(frame_idx, avg_speed, metrics))

        self._total_events += len(events)
        return events

    # ------------------------------------------------------------------
    # Detector: Z-Score Spike
    # ------------------------------------------------------------------

    def _check_zscore_spike(
        self, frame_idx: int, count: float, metrics: dict,
    ) -> list[AnomalyEvent]:
        if len(self._count_history) < self.cfg.zscore_window:
            return []
        if self._in_cooldown("spike", frame_idx):
            return []

        arr  = np.array(list(self._count_history))
        mean = arr.mean()
        std  = arr.std()

        if std < 0.5:
            return []

        z = (count - mean) / std

        if abs(z) >= self.cfg.zscore_threshold:
            direction = "above" if z > 0 else "below"
            severity  = "critical" if abs(z) > 3.5 else "warning"
            conf      = min(abs(z) / 5.0, 1.0)

            event = AnomalyEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                frame_idx=frame_idx,
                anomaly_type="spike",
                severity=severity,
                description=(
                    f"Vehicle count spike: {int(count)} vehicles "
                    f"({direction} mean {mean:.1f}, z={z:+.2f})"
                ),
                confidence=conf,
                metrics_snapshot=dict(metrics),
            )
            self._last_alert["spike"] = frame_idx
            logger.warning("ANOMALY [spike] frame=%d z=%.2f count=%d", frame_idx, z, int(count))
            return [event]

        return []

    # ------------------------------------------------------------------
    # Detector: Sudden Drop
    # ------------------------------------------------------------------

    def _check_sudden_drop(
        self, frame_idx: int, metrics: dict,
    ) -> list[AnomalyEvent]:
        window = self.cfg.drop_window
        if len(self._flow_history) < window * 2:
            return []
        if self._in_cooldown("drop", frame_idx):
            return []

        recent = list(self._flow_history)
        old_avg = np.mean(recent[-window*2:-window])
        new_avg = np.mean(recent[-window:])

        if old_avg <= 0:
            return []

        drop_pct = (old_avg - new_avg) / old_avg

        if drop_pct >= self.cfg.drop_pct_threshold:
            severity = "critical" if drop_pct > 0.70 else "warning"
            conf = min(drop_pct / 0.80, 1.0)

            event = AnomalyEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                frame_idx=frame_idx,
                anomaly_type="drop",
                severity=severity,
                description=(
                    f"Sudden flow drop: {drop_pct:.0%} decrease in {window} frames "
                    f"(from {old_avg:.1f} to {new_avg:.1f} veh/min)"
                ),
                confidence=conf,
                metrics_snapshot=dict(metrics),
            )
            self._last_alert["drop"] = frame_idx
            logger.warning("ANOMALY [drop] frame=%d drop=%.0f%%", frame_idx, drop_pct*100)
            return [event]

        return []

    # ------------------------------------------------------------------
    # Detector: Congestion Surge
    # ------------------------------------------------------------------

    def _check_congestion_surge(
        self, frame_idx: int, cong: float, metrics: dict,
    ) -> list[AnomalyEvent]:
        if len(self._cong_history) < self.cfg.congestion_persist:
            return []
        if self._in_cooldown("congestion_surge", frame_idx):
            return []

        recent = list(self._cong_history)[-self.cfg.congestion_persist:]
        all_critical = all(c >= self.cfg.congestion_critical for c in recent)

        if all_critical:
            avg = np.mean(recent)
            severity = "critical"
            conf = min(avg / 100.0, 1.0)

            event = AnomalyEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                frame_idx=frame_idx,
                anomaly_type="congestion_surge",
                severity=severity,
                description=(
                    f"Sustained critical congestion: score = {avg:.1f}/100 "
                    f"for {self.cfg.congestion_persist} consecutive frames"
                ),
                confidence=conf,
                metrics_snapshot=dict(metrics),
            )
            self._last_alert["congestion_surge"] = frame_idx
            logger.warning("ANOMALY [congestion_surge] frame=%d avg_cong=%.1f", frame_idx, avg)
            return [event]

        return []

    # ------------------------------------------------------------------
    # Detector: Trend Reversal
    # ------------------------------------------------------------------

    def _check_trend_reversal(
        self, frame_idx: int, metrics: dict,
    ) -> list[AnomalyEvent]:
        if len(self._trend_history) < 6:
            return []
        if self._in_cooldown("reversal", frame_idx):
            return []

        recent = list(self._trend_history)
        old_trends = recent[-6:-3]
        new_trends = recent[-3:]

        was_declining = all(t in ("falling", "stable") for t in old_trends)
        now_rising    = all(t == "rising" for t in new_trends)

        if was_declining and now_rising:
            # Check acceleration via count history
            if len(self._count_history) >= 6:
                counts = list(self._count_history)
                accel = (counts[-1] - counts[-3]) - (counts[-3] - counts[-5])
                if accel < self.cfg.reversal_accel:
                    return []
            else:
                accel = 0

            event = AnomalyEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                frame_idx=frame_idx,
                anomaly_type="reversal",
                severity="warning",
                description=(
                    f"Trend reversal detected: traffic shifted from declining "
                    f"to rapidly rising (accel={accel:.1f})"
                ),
                confidence=0.65,
                metrics_snapshot=dict(metrics),
            )
            self._last_alert["reversal"] = frame_idx
            logger.warning("ANOMALY [reversal] frame=%d accel=%.1f", frame_idx, accel)
            return [event]

        return []

    # ------------------------------------------------------------------
    # Detector: Speed Anomaly
    # ------------------------------------------------------------------

    def _check_speed_anomaly(
        self, frame_idx: int, avg_speed: float | None, metrics: dict,
    ) -> list[AnomalyEvent]:
        if avg_speed is None:
            return []
        if len(self._speed_history) < 10:
            return []
        if self._in_cooldown("speed_anomaly", frame_idx):
            return []

        history = list(self._speed_history)
        historical_avg = np.mean(history[:-3])
        current_avg    = np.mean(history[-3:])

        if historical_avg <= 5.0:
            return []

        drop_pct = (historical_avg - current_avg) / historical_avg

        if drop_pct >= self.cfg.speed_drop_pct:
            severity = "critical" if drop_pct > 0.60 else "warning"
            conf = min(drop_pct / 0.70, 1.0)

            event = AnomalyEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                frame_idx=frame_idx,
                anomaly_type="speed_anomaly",
                severity=severity,
                description=(
                    f"Speed anomaly: avg speed dropped from {historical_avg:.1f} km/h "
                    f"to {current_avg:.1f} km/h ({drop_pct:.0%} decrease)"
                ),
                confidence=conf,
                metrics_snapshot=dict(metrics),
            )
            self._last_alert["speed_anomaly"] = frame_idx
            logger.warning("ANOMALY [speed_anomaly] frame=%d drop=%.0f%%", frame_idx, drop_pct*100)
            return [event]

        return []

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _in_cooldown(self, anomaly_type: str, frame_idx: int) -> bool:
        """Check if this anomaly type is still in cooldown."""
        last = self._last_alert.get(anomaly_type, -999)
        return (frame_idx - last) < self.cfg.cooldown_frames

    def reset(self) -> None:
        """Clear all history and cooldowns."""
        self._count_history.clear()
        self._flow_history.clear()
        self._cong_history.clear()
        self._speed_history.clear()
        self._trend_history.clear()
        self._last_alert.clear()
        self._total_events = 0

    @property
    def total_events(self) -> int:
        return self._total_events
