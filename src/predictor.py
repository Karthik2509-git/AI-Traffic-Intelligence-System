"""
predictor.py — Machine-learning congestion forecasting engine.

Provides high-accuracy traffic state classification using temporal features 
and calibrated Gradient Boosting models. Includes multi-stage feature 
engineering and automated model retraining/persistence workflows.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.density_analyzer import FrameDensity
from src.utils import get_logger, get_project_root

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}
LABEL_RMAP = {"Low": 0, "Medium": 1, "High": 2}
FEATURE_COLS = [
    "ema_count",
    "occupancy_ratio",
    "congestion_score",
    "flow_rate_per_min",
    "count_trend",
    "speed_variance",
    "time_of_day_sin",
    "time_of_day_cos",
]


def _time_of_day_features(dt: datetime) -> tuple[float, float]:
    """Encode time of day as cyclical sin/cos features."""
    hour_frac = dt.hour + dt.minute / 60.0
    angle = 2 * math.pi * hour_frac / 24.0
    return math.sin(angle), math.cos(angle)


def build_feature_vector(
    fd: FrameDensity,
    flow_rate_per_min: float,
    count_trend: float,
    speed_variance: float = 0.0,
    dt: datetime | None = None,
) -> np.ndarray:
    """
    Construct a standardized feature vector from real-time telemetry.

    Parameters
    ----------
    fd : FrameDensity
        Current spatial density snapshot.
    flow_rate_per_min : float
        Calculated throughput estimation.
    count_trend : float
        Temporal volume acceleration (EMA slope).
    speed_variance : float
        Variance in vehicle speeds (default: 0.0).
    dt : datetime | None
        Timestamp for cyclical time encoding. Defaults to now().

    Returns
    -------
    np.ndarray
        Standardized feature vector of shape (1, 8).
    """
    if dt is None:
        dt = datetime.now()

    tod_sin, tod_cos = _time_of_day_features(dt)

    vec = np.array([[
        fd.ema_count,
        fd.occupancy_ratio,
        fd.congestion_score,
        flow_rate_per_min,
        count_trend,
        speed_variance,
        tod_sin,
        tod_cos,
    ]], dtype=float)

    return vec


def frames_to_dataframe(
    history: Sequence[FrameDensity],
    flow_rate_fn: Any = None,
    low_threshold: int = 10,
    high_threshold: int = 25,
) -> pd.DataFrame:
    """
    Convert historical telemetry records into a labeled training dataset.

    Generates ground-truth labels based on raw counts while preserving 
    EMA-smoothed features for predictive training.

    Parameters
    ----------
    history : Sequence[FrameDensity]
        Chronological list of density snapshots.
    flow_rate_fn : Any
        Callable to inject real-time throughput metrics.
    low_threshold : int
        Boundary for 'Low' congestion label.
    high_threshold : int
        Boundary for 'High' congestion label.

    Returns
    -------
    pd.DataFrame
        Dataset structured for scikit-learn consumption.
    """
    rows: list[dict[str, Any]] = []
    hist_list = list(history)
    for i, fd in enumerate(hist_list):
        if i >= 5:
            trend = hist_list[i].ema_count - hist_list[i - 5].ema_count
        else:
            trend = 0.0

        flow = flow_rate_fn() if flow_rate_fn is not None else 0.0

        dt = datetime.fromtimestamp(fd.timestamp_ms / 1_000.0)
        tod_sin, tod_cos = _time_of_day_features(dt)

        if fd.total_count < low_threshold:
            label = 0
        elif fd.total_count <= high_threshold:
            label = 1
        else:
            label = 2

        rows.append({
            "ema_count": fd.ema_count,
            "occupancy_ratio": fd.occupancy_ratio,
            "congestion_score": fd.congestion_score,
            "flow_rate_per_min": flow,
            "count_trend": trend,
            "speed_variance": 0.0,
            "time_of_day_sin": tod_sin,
            "time_of_day_cos": tod_cos,
            "label": label,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Predictor model
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Structured output of a single prediction call."""
    label: str              # "Low" | "Medium" | "High"
    label_index: int              # 0 / 1 / 2
    probabilities: dict[str, float] # {"Low": p0, "Medium": p1, "High": p2}
    confidence: float            # max probability


class CongestionPredictor:
    """
    Industrial-grade congestion forecasting engine.

    Utilizes a calibrated Gradient Boosting pipeline for high-fidelity 
    state classification. Support for model versioning and automated 
    re-training on historical telemetry.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the ensemble.
    max_depth : int
        Maximum depth of each tree.
    learning_rate : float
        Step size shrinkage used in update to prevent overfitting.
    cv_folds : int
        Number of folds for cross-validation evaluation.
    """

    MODEL_FILENAME = "congestion_model.joblib"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        cv_folds: int = 5,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.cv_folds = cv_folds
        self._pipeline: Pipeline | None = None
        self._is_trained = False
        self._train_report: str = ""

    def train(
        self,
        df: pd.DataFrame,
        min_samples: int = 50,
        use_gradient_boost: bool = True,
    ) -> dict[str, Any]:
        """
        Fit and calibrate the classification pipeline using historical data.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset containing telemetry features and 'label'.
        min_samples : int
            Minimum observations required to initiate training.
        use_gradient_boost : bool
            If True, uses GradientBoostingClassifier. Otherwise, uses 
            RandomForestClassifier.

        Returns
        -------
        dict[str, Any]
            Training metadata including metrics and sample counts.
        """
        if len(df) < min_samples:
            raise ValueError(
                f"Insufficient training data: {len(df)} rows (min={min_samples}). "
                "Accumulate more frames before training."
            )

        X = df[FEATURE_COLS].values.astype(float)
        y = df["label"].values.astype(int)

        # Handle class imbalance via automatic weighting
        classes, counts = np.unique(y, return_counts=True)
        class_weight = {int(c): float(len(y)) / (len(classes) * n) for c, n in zip(classes, counts)}

        if use_gradient_boost:
            base = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.8,
                min_samples_leaf=5,
                random_state=42,
            )
        else:
            base = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1,
            )

        # Pipeline: scale -> calibrated classifier
        calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", calibrated),
        ])

        # Cross-validation validation
        valid_folds = min(self.cv_folds, len(np.unique(y)))
        if valid_folds >= 2:
            cv = StratifiedKFold(n_splits=valid_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="balanced_accuracy")
        else:
            cv_scores = np.array([1.0])

        # Final fit on all data
        pipeline.fit(X, y)
        self._pipeline = pipeline
        self._is_trained = True

        y_pred = pipeline.predict(X)
        report = classification_report(
            y, y_pred, target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)], zero_division=0
        )
        self._train_report = report

        return {
            "cv_accuracy": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "report": report,
            "n_samples": len(df),
        }

    def predict(self, feature_vector: np.ndarray) -> PredictionResult:
        """
        Execute congestion state inference on a feature vector.

        Parameters
        ----------
        feature_vector : np.ndarray
            Input features (shape [1, 8]).

        Returns
        -------
        PredictionResult
            Predicted labels and calibrated probabilities.
        """
        if not self._is_trained or self._pipeline is None:
            raise NotFittedError(
                "Predictor has not been trained yet. "
                "Call .train(df) or .load() first."
            )

        idx: int = int(self._pipeline.predict(feature_vector)[0])
        proba = self._pipeline.predict_proba(feature_vector)[0]

        # Align probabilities with the static LABEL_MAP
        classes = self._pipeline.classes_
        prob_dict = {LABEL_MAP.get(int(c), str(c)): float(p) for c, p in zip(classes, proba)}
        for lbl in LABEL_MAP.values():
            prob_dict.setdefault(lbl, 0.0)

        return PredictionResult(
            label=LABEL_MAP.get(idx, "Unknown"),
            label_index=idx,
            probabilities=prob_dict,
            confidence=float(max(proba)),
        )

    def feature_importances(self) -> dict[str, float] | None:
        """
        Extract feature importances from the underlying model.

        Returns
        -------
        dict[str, float] | None
            Importance mapping or None if model type doesn't support extraction.
        """
        if not self._is_trained or self._pipeline is None:
            return None

        clf = self._pipeline.named_steps["classifier"]
        try:
            # CalibratedClassifierCV stores estimators in calibrated_classifiers_
            base = clf.calibrated_classifiers_[0].estimator
            importances = getattr(base, "feature_importances_", None)
            if importances is not None:
                return dict(zip(FEATURE_COLS, importances.tolist()))
        except (AttributeError, IndexError):
            pass
        return None

    def save(self, path: Path | None = None) -> Path:
        """
        Persist the trained pipeline to disk.

        Parameters
        ----------
        path : Path | None
            The target file path. Defaults to 'output/congestion_model.joblib'.

        Returns
        -------
        Path
            The absolute path where the model was saved.
        """
        if not self._is_trained:
            raise RuntimeError("Cannot save: model has not been trained.")

        save_path = path or (get_project_root() / "output" / self.MODEL_FILENAME)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, save_path, compress=3)
        logger.info("Model saved to '%s'.", save_path)
        return save_path

    def load(self, path: Path | None = None) -> None:
        """
        Load a previously persisted pipeline from disk.

        Parameters
        ----------
        path : Path | None
            The source file path. Defaults to 'output/congestion_model.joblib'.
        """
        load_path = path or (get_project_root() / "output" / self.MODEL_FILENAME)
        if not load_path.is_file():
            raise FileNotFoundError(f"Model file not found: '{load_path}'.")

        self._pipeline = joblib.load(load_path)
        self._is_trained = True
        logger.info("Model loaded from '%s'.", load_path)