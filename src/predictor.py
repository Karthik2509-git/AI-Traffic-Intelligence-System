"""
predictor.py — Machine-learning congestion prediction module.

Pipeline
--------
1. Feature engineering from rolling traffic metrics
2. GradientBoostingClassifier (primary) / RandomForestClassifier (fallback)
3. Probability calibration via CalibratedClassifierCV (Platt scaling)
4. Threshold-tuned class labels for operational use
5. Model persistence (joblib) with versioning

Features used
-------------
  ema_count          : Smoothed vehicle count
  occupancy          : Fractional area occupied by vehicles
  congestion_score   : Composite 0–100 score from DensityAnalyzer
  flow_rate_per_min  : Throughput over rolling window
  count_trend        : Slope of EMA over last N frames
  time_of_day_sin    : sin(2π * hour / 24)   — encode cyclical time
  time_of_day_cos    : cos(2π * hour / 24)

Labels (target)
---------------
  0 → Low    (count < low_threshold)
  1 → Medium (low ≤ count ≤ high_threshold)
  2 → High   (count > high_threshold)
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

LABEL_MAP   = {0: "Low", 1: "Medium", 2: "High"}
LABEL_RMAP  = {"Low": 0, "Medium": 1, "High": 2}
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
    hour_frac = dt.hour + dt.minute / 60.0
    angle     = 2 * math.pi * hour_frac / 24.0
    return math.sin(angle), math.cos(angle)


def build_feature_vector(
    fd:                FrameDensity,
    flow_rate_per_min: float,
    count_trend:       float,
    speed_variance:    float = 0.0,
    dt:                datetime | None = None,
) -> np.ndarray:
    """
    Construct a 1-D feature vector from a FrameDensity snapshot.

    Parameters
    ----------
    fd                : Current frame density from DensityAnalyzer.
    flow_rate_per_min : Throughput estimate from DensityAnalyzer.
    count_trend       : Slope of EMA over recent frames (+ = rising, − = falling).
    dt                : Timestamp for time-of-day features; defaults to now.

    Returns
    -------
    np.ndarray of shape (1, len(FEATURE_COLS))
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

    return vec  # shape (1, 8)


def frames_to_dataframe(
    history:          list[FrameDensity],
    flow_rate_fn:     Any = None,          # callable() → float, injected by pipeline
    low_threshold:    int = 10,
    high_threshold:   int = 25,
) -> pd.DataFrame:
    """
    Convert a list of FrameDensity records into a labelled DataFrame for training.

    Target label is derived from *total_count* (not ema_count) to avoid label
    leakage from the smoothed feature.
    """
    rows: list[dict] = []
    for i, fd in enumerate(history):
        # Approximate trend as difference over 5 frames
        if i >= 5:
            trend = history[i].ema_count - history[i - 5].ema_count
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
            "ema_count":         fd.ema_count,
            "occupancy_ratio":   fd.occupancy_ratio,
            "congestion_score":  fd.congestion_score,
            "flow_rate_per_min": flow,
            "count_trend":       trend,
            "speed_variance":    0.0,  # historical speed variance estimation (placeholder)
            "time_of_day_sin":   tod_sin,
            "time_of_day_cos":   tod_cos,
            "label":             label,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Predictor model
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Structured output of a single prediction call."""
    label:         str              # "Low" | "Medium" | "High"
    label_index:   int              # 0 / 1 / 2
    probabilities: dict[str, float] # {"Low": p0, "Medium": p1, "High": p2}
    confidence:    float            # max probability


class CongestionPredictor:
    """
    Scikit-learn congestion prediction pipeline.

    Training is intentionally separated from inference so the predictor
    can be retrained periodically (e.g. hourly) without restarting the app.

    Usage
    -----
    predictor = CongestionPredictor()
    predictor.train(history_df)
    result = predictor.predict(feature_vector)
    predictor.save()
    predictor.load()
    """

    MODEL_FILENAME = "congestion_model.joblib"

    def __init__(
        self,
        n_estimators:  int   = 200,
        max_depth:     int   = 5,
        learning_rate: float = 0.05,
        cv_folds:      int   = 5,
    ) -> None:
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.cv_folds      = cv_folds
        self._pipeline: Pipeline | None = None
        self._is_trained   = False
        self._train_report: str = ""

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df:                 pd.DataFrame,
        min_samples:        int  = 50,
        use_gradient_boost: bool = True,
    ) -> dict[str, Any]:
        """
        Fit and calibrate the classification pipeline.

        Parameters
        ----------
        df              : DataFrame with columns in FEATURE_COLS + 'label'.
        min_samples     : Minimum rows required to proceed.
        use_gradient_boost : If False, uses RandomForest (faster, lower accuracy).

        Returns
        -------
        dict with 'cv_accuracy', 'report', 'n_samples'.
        """
        if len(df) < min_samples:
            raise ValueError(
                f"Insufficient training data: {len(df)} rows (min={min_samples}). "
                "Accumulate more frames before training."
            )

        X = df[FEATURE_COLS].values.astype(float)
        y = df["label"].values.astype(int)

        # --- Handle class imbalance via class_weight ----------------------
        classes, counts = np.unique(y, return_counts=True)
        class_weight    = {int(c): float(len(y)) / (len(classes) * n) for c, n in zip(classes, counts)}

        # --- Base estimator -----------------------------------------------
        if use_gradient_boost:
            base = GradientBoostingClassifier(
                n_estimators  = self.n_estimators,
                max_depth     = self.max_depth,
                learning_rate = self.learning_rate,
                subsample     = 0.8,
                min_samples_leaf = 5,
                random_state  = 42,
            )
        else:
            base = RandomForestClassifier(
                n_estimators = self.n_estimators,
                max_depth    = self.max_depth,
                class_weight = class_weight,
                random_state = 42,
                n_jobs       = -1,
            )

        # --- Pipeline: scale → calibrated classifier ----------------------
        calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3)

        pipeline = Pipeline([
            ("scaler",     StandardScaler()),
            ("classifier", calibrated),
        ])

        # --- Cross-validation ---------------------------------------------
        cv = StratifiedKFold(n_splits=min(self.cv_folds, len(np.unique(y))), shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="balanced_accuracy")
        logger.info(
            "CV balanced accuracy: %.3f ± %.3f", cv_scores.mean(), cv_scores.std()
        )

        # --- Final fit on all data ----------------------------------------
        pipeline.fit(X, y)
        self._pipeline    = pipeline
        self._is_trained  = True

        y_pred = pipeline.predict(X)
        report = classification_report(
            y, y_pred, target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)], zero_division=0
        )
        self._train_report = report
        logger.info("Training complete.\n%s", report)

        return {
            "cv_accuracy": float(cv_scores.mean()),
            "cv_std":      float(cv_scores.std()),
            "report":      report,
            "n_samples":   len(df),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, feature_vector: np.ndarray) -> PredictionResult:
        """
        Predict congestion level from a feature vector.

        Parameters
        ----------
        feature_vector : np.ndarray of shape (1, 7) from build_feature_vector().

        Returns
        -------
        PredictionResult
        """
        if not self._is_trained or self._pipeline is None:
            raise NotFittedError(
                "Predictor has not been trained yet. "
                "Call .train(df) or .load() first."
            )

        idx: int   = int(self._pipeline.predict(feature_vector)[0])
        proba      = self._pipeline.predict_proba(feature_vector)[0]

        # Align probabilities with LABEL_MAP — ensure all three keys present
        classes    = self._pipeline.classes_
        prob_dict  = {LABEL_MAP.get(int(c), str(c)): float(p) for c, p in zip(classes, proba)}
        for lbl in LABEL_MAP.values():
            prob_dict.setdefault(lbl, 0.0)

        return PredictionResult(
            label         = LABEL_MAP.get(idx, "Unknown"),
            label_index   = idx,
            probabilities = prob_dict,
            confidence    = float(max(proba)),
        )

    # ------------------------------------------------------------------
    # Feature importances
    # ------------------------------------------------------------------

    def feature_importances(self) -> dict[str, float] | None:
        """
        Return feature importances if the base estimator exposes them.
        Returns None for calibrated pipelines without direct access.
        """
        if not self._is_trained or self._pipeline is None:
            return None

        clf = self._pipeline.named_steps["classifier"]
        base = getattr(clf, "estimator", None) or getattr(clf, "base_estimator", None)
        if base is None:
            # Try to access through calibrated estimators list
            try:
                base = clf.calibrated_classifiers_[0].estimator
            except (AttributeError, IndexError):
                return None

        importances = getattr(base, "feature_importances_", None)
        if importances is None:
            return None

        return dict(zip(FEATURE_COLS, importances.tolist()))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> Path:
        """Serialise trained pipeline to disk."""
        if not self._is_trained:
            raise RuntimeError("Cannot save: model has not been trained.")

        save_path = path or (get_project_root() / "output" / self.MODEL_FILENAME)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, save_path, compress=3)
        logger.info("Model saved to '%s'.", save_path)
        return save_path

    def load(self, path: Path | None = None) -> None:
        """Load a previously saved pipeline from disk."""
        load_path = path or (get_project_root() / "output" / self.MODEL_FILENAME)
        if not load_path.is_file():
            raise FileNotFoundError(f"Model file not found: '{load_path}'.")

        self._pipeline   = joblib.load(load_path)
        self._is_trained = True
        logger.info("Model loaded from '%s'.", load_path)