"""
database.py - Persistent traffic data storage using SQLite.

Provides:
  - Structured schema for frame metrics, signal logs, and anomaly events
  - Session-based organisation for multi-run tracking
  - CSV / JSON batch export for downstream analysis
  - Historical trend queries across sessions
  - Thread-safe connection management

Schema
------
  frame_metrics   : Per-frame vehicle counts, density, congestion, flow rate
  signal_logs     : Signal optimizer recommendations per lane
  anomaly_events  : Detected traffic anomalies (incidents, spikes, drops)
  speed_records   : Per-vehicle speed measurements
  sessions        : Session metadata (start time, source, config hash)
"""

from __future__ import annotations

import csv
import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Sequence

from src.utils import get_logger, get_output_dir

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Session helper
# ---------------------------------------------------------------------------

def new_session_id() -> str:
    """Generate a unique session identifier."""
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- Session metadata
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    started_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source       TEXT,
    config_hash  TEXT,
    notes        TEXT
);

-- Per-frame traffic metrics
CREATE TABLE IF NOT EXISTS frame_metrics (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT NOT NULL,
    frame_idx        INTEGER NOT NULL,
    timestamp_ms     REAL,
    total_vehicles   INTEGER DEFAULT 0,
    cars             INTEGER DEFAULT 0,
    motorcycles      INTEGER DEFAULT 0,
    buses            INTEGER DEFAULT 0,
    trucks           INTEGER DEFAULT 0,
    density_label    TEXT,
    congestion_score REAL DEFAULT 0.0,
    ema_count        REAL DEFAULT 0.0,
    occupancy        REAL DEFAULT 0.0,
    flow_per_min     REAL DEFAULT 0.0,
    trend            TEXT DEFAULT 'stable',
    processing_ms    REAL DEFAULT 0.0,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Signal optimizer recommendations
CREATE TABLE IF NOT EXISTS signal_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    frame_idx     INTEGER,
    lane_name     TEXT,
    green_time_s  INTEGER,
    red_time_s    INTEGER,
    pressure      REAL,
    advisory      TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Anomaly events
CREATE TABLE IF NOT EXISTS anomaly_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT NOT NULL,
    event_id          TEXT UNIQUE,
    frame_idx         INTEGER,
    anomaly_type      TEXT,
    severity          TEXT,
    description       TEXT,
    confidence        REAL,
    metrics_snapshot  TEXT,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Per-vehicle speed records
CREATE TABLE IF NOT EXISTS speed_records (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    frame_idx    INTEGER,
    track_id     INTEGER,
    speed_kmh    REAL,
    direction_deg REAL,
    speed_class  TEXT,
    is_violation INTEGER DEFAULT 0,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Performance indices
CREATE INDEX IF NOT EXISTS idx_frame_session ON frame_metrics(session_id, frame_idx);
CREATE INDEX IF NOT EXISTS idx_signal_session ON signal_logs(session_id, frame_idx);
CREATE INDEX IF NOT EXISTS idx_anomaly_session ON anomaly_events(session_id, frame_idx);
CREATE INDEX IF NOT EXISTS idx_speed_session ON speed_records(session_id, frame_idx);
"""


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class TrafficDatabase:
    """
    Persistent SQLite storage for traffic intelligence data.

    Usage
    -----
    db = TrafficDatabase()                       # default: output/traffic.db
    sid = db.start_session("video.mp4")
    db.log_frame(sid, frame_idx=0, metrics={...})
    db.export_csv(sid, "output/report.csv")
    db.close()

    Thread safety: each method acquires its own connection via context manager.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            db_path = get_output_dir() / "traffic.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create schema
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

        logger.info("TrafficDatabase initialised at: %s", self._db_path)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def start_session(
        self,
        source: str = "",
        config_hash: str = "",
        notes: str = "",
    ) -> str:
        """Create a new analysis session and return its ID."""
        sid = new_session_id()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, source, config_hash, notes) "
                "VALUES (?, ?, ?, ?)",
                (sid, source, config_hash, notes),
            )
        logger.info("Started session: %s (source=%s)", sid, source)
        return sid

    def list_sessions(self, limit: int = 20) -> list[dict]:
        """Return recent sessions ordered by start time (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, started_at, source, notes "
                "FROM sessions ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Frame metrics
    # ------------------------------------------------------------------

    def log_frame(
        self,
        session_id: str,
        frame_idx: int,
        metrics: dict[str, Any],
    ) -> None:
        """Persist per-frame traffic metrics."""
        counts = metrics.get("counts_per_class", {})
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO frame_metrics
                (session_id, frame_idx, timestamp_ms, total_vehicles,
                 cars, motorcycles, buses, trucks,
                 density_label, congestion_score, ema_count,
                 occupancy, flow_per_min, trend, processing_ms)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    session_id,
                    frame_idx,
                    metrics.get("timestamp_ms", 0),
                    metrics.get("total_vehicles", 0),
                    counts.get("car", 0),
                    counts.get("motorcycle", 0),
                    counts.get("bus", 0),
                    counts.get("truck", 0),
                    metrics.get("density_label", "Low"),
                    metrics.get("congestion_score", 0.0),
                    metrics.get("ema_count", 0.0),
                    metrics.get("occupancy", 0.0),
                    metrics.get("flow_per_min", 0.0),
                    metrics.get("trend", "stable"),
                    metrics.get("processing_time_ms", 0.0),
                ),
            )

    def log_frames_batch(
        self,
        session_id: str,
        frames: Sequence[tuple[int, dict[str, Any]]],
    ) -> None:
        """Batch-insert multiple frames for efficiency."""
        rows = []
        for frame_idx, metrics in frames:
            counts = metrics.get("counts_per_class", {})
            rows.append((
                session_id,
                frame_idx,
                metrics.get("timestamp_ms", 0),
                metrics.get("total_vehicles", 0),
                counts.get("car", 0),
                counts.get("motorcycle", 0),
                counts.get("bus", 0),
                counts.get("truck", 0),
                metrics.get("density_label", "Low"),
                metrics.get("congestion_score", 0.0),
                metrics.get("ema_count", 0.0),
                metrics.get("occupancy", 0.0),
                metrics.get("flow_per_min", 0.0),
                metrics.get("trend", "stable"),
                metrics.get("processing_time_ms", 0.0),
            ))
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO frame_metrics
                (session_id, frame_idx, timestamp_ms, total_vehicles,
                 cars, motorcycles, buses, trucks,
                 density_label, congestion_score, ema_count,
                 occupancy, flow_per_min, trend, processing_ms)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
        logger.debug("Batch-inserted %d frame records for session %s.", len(rows), session_id)

    def get_frame_metrics(
        self,
        session_id: str,
        limit: int = 1000,
    ) -> list[dict]:
        """Retrieve frame metrics for a session."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM frame_metrics WHERE session_id = ? "
                "ORDER BY frame_idx LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Signal logs
    # ------------------------------------------------------------------

    def log_signal(
        self,
        session_id: str,
        frame_idx: int,
        lane_name: str,
        green_time_s: int,
        red_time_s: int,
        pressure: float,
        advisory: str,
    ) -> None:
        """Persist a signal optimization recommendation."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO signal_logs
                (session_id, frame_idx, lane_name, green_time_s,
                 red_time_s, pressure, advisory)
                VALUES (?,?,?,?,?,?,?)""",
                (session_id, frame_idx, lane_name, green_time_s,
                 red_time_s, pressure, advisory),
            )

    def get_signal_logs(self, session_id: str, limit: int = 500) -> list[dict]:
        """Retrieve signal logs for a session."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM signal_logs WHERE session_id = ? "
                "ORDER BY frame_idx LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Anomaly events
    # ------------------------------------------------------------------

    def log_anomaly(
        self,
        session_id: str,
        event_id: str,
        frame_idx: int,
        anomaly_type: str,
        severity: str,
        description: str,
        confidence: float,
        metrics_snapshot: dict | None = None,
    ) -> None:
        """Persist an anomaly detection event."""
        snapshot_json = json.dumps(metrics_snapshot) if metrics_snapshot else "{}"
        with self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO anomaly_events
                (session_id, event_id, frame_idx, anomaly_type,
                 severity, description, confidence, metrics_snapshot)
                VALUES (?,?,?,?,?,?,?,?)""",
                (session_id, event_id, frame_idx, anomaly_type,
                 severity, description, confidence, snapshot_json),
            )

    def get_anomalies(self, session_id: str, limit: int = 200) -> list[dict]:
        """Retrieve anomaly events for a session."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM anomaly_events WHERE session_id = ? "
                "ORDER BY frame_idx LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Speed records
    # ------------------------------------------------------------------

    def log_speed(
        self,
        session_id: str,
        frame_idx: int,
        track_id: int,
        speed_kmh: float,
        direction_deg: float,
        speed_class: str,
        is_violation: bool,
    ) -> None:
        """Persist a vehicle speed measurement."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO speed_records
                (session_id, frame_idx, track_id, speed_kmh,
                 direction_deg, speed_class, is_violation)
                VALUES (?,?,?,?,?,?,?)""",
                (session_id, frame_idx, track_id, speed_kmh,
                 direction_deg, speed_class, int(is_violation)),
            )

    def log_speeds_batch(
        self,
        session_id: str,
        records: Sequence[tuple],
    ) -> None:
        """Batch-insert speed records: [(frame_idx, track_id, speed_kmh, dir, cls, viol), ...]."""
        rows = [(session_id, *r) for r in records]
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO speed_records
                (session_id, frame_idx, track_id, speed_kmh,
                 direction_deg, speed_class, is_violation)
                VALUES (?,?,?,?,?,?,?)""",
                rows,
            )

    # ------------------------------------------------------------------
    # Aggregation & analytics
    # ------------------------------------------------------------------

    def get_session_summary(self, session_id: str) -> dict[str, Any]:
        """Compute aggregate statistics for a session."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT
                    COUNT(*)              AS total_frames,
                    COALESCE(AVG(total_vehicles), 0) AS avg_vehicles,
                    COALESCE(MAX(total_vehicles), 0) AS peak_vehicles,
                    COALESCE(AVG(congestion_score), 0) AS avg_congestion,
                    COALESCE(MAX(congestion_score), 0) AS peak_congestion,
                    COALESCE(AVG(processing_ms), 0) AS avg_latency_ms,
                    COALESCE(AVG(flow_per_min), 0) AS avg_flow_rate
                FROM frame_metrics WHERE session_id = ?""",
                (session_id,),
            ).fetchone()

            anomaly_count = conn.execute(
                "SELECT COUNT(*) FROM anomaly_events WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]

            speed_row = conn.execute(
                """SELECT
                    COALESCE(AVG(speed_kmh), 0) AS avg_speed,
                    COALESCE(MAX(speed_kmh), 0) AS max_speed,
                    COALESCE(SUM(is_violation), 0) AS violations
                FROM speed_records WHERE session_id = ?""",
                (session_id,),
            ).fetchone()

        summary = dict(row) if row else {}
        summary["anomaly_count"] = anomaly_count
        if speed_row:
            summary.update(dict(speed_row))
        return summary

    def get_density_distribution(self, session_id: str) -> dict[str, int]:
        """Get count of frames per density label."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT density_label, COUNT(*) as cnt
                FROM frame_metrics WHERE session_id = ?
                GROUP BY density_label""",
                (session_id,),
            ).fetchall()
        return {r["density_label"]: r["cnt"] for r in rows}

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(
        self,
        session_id: str,
        output_path: Path | str,
    ) -> Path:
        """Export frame metrics for a session to a CSV file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        frames = self.get_frame_metrics(session_id, limit=100_000)
        if not frames:
            logger.warning("No frame data to export for session %s.", session_id)
            return path

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=frames[0].keys())
            writer.writeheader()
            writer.writerows(frames)

        logger.info("Exported %d rows to CSV: %s", len(frames), path)
        return path

    def export_json(
        self,
        session_id: str,
        output_path: Path | str,
    ) -> Path:
        """Export all session data (frames + signals + anomalies) to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "session_id": session_id,
            "summary": self.get_session_summary(session_id),
            "frame_metrics": self.get_frame_metrics(session_id, limit=100_000),
            "signal_logs": self.get_signal_logs(session_id),
            "anomaly_events": self.get_anomalies(session_id),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Exported session data to JSON: %s", path)
        return path

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def delete_session(self, session_id: str) -> None:
        """Remove all data for a session."""
        with self._connect() as conn:
            for table in ("frame_metrics", "signal_logs", "anomaly_events", "speed_records", "sessions"):
                conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
        logger.info("Deleted session: %s", session_id)

    def close(self) -> None:
        """No-op for compatibility — connections are opened/closed per operation."""
        pass
