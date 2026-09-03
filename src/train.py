"""Leakage-safe model construction for fraud-risk experiments."""
from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

from src.features import engineer_features
from src.preprocessing import build_preprocessor


def build_logistic_regression(*, C: float, max_iter: int, class_weight: str | None = "balanced") -> LogisticRegression:
    """Create the reproducible baseline classifier."""
    return LogisticRegression(C=C, max_iter=max_iter, class_weight=class_weight, solver="liblinear")


def build_logistic_pipeline(X_train, parameters: dict[str, Any]) -> Pipeline:
    """Create an unfitted baseline pipeline; it must be fit on training rows only."""
    engineered_train = engineer_features(X_train)
    return Pipeline([
        ("feature_engineering", FunctionTransformer(engineer_features, validate=False)),
        ("preprocessing", build_preprocessor(engineered_train)),
        ("classifier", build_logistic_regression(**parameters)),
    ])


def build_xgboost_pipeline(X_train, parameters: dict[str, Any], *, scale_pos_weight: float) -> Pipeline:
    """Create an unfitted XGBoost pipeline with training-derived class weighting."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("XGBoost is not installed; use the documented fallback.") from exc
    classifier = XGBClassifier(
        **parameters,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    engineered_train = engineer_features(X_train)
    return Pipeline([
        ("feature_engineering", FunctionTransformer(engineer_features, validate=False)),
        ("preprocessing", build_preprocessor(engineered_train)),
        ("classifier", classifier),
    ])


def class_imbalance_weight(y_train) -> float:
    """Return legitimate/fraud ratio calculated solely from training labels."""
    counts = y_train.value_counts()
    if 0 not in counts or 1 not in counts or counts[1] == 0:
        raise ValueError("Training labels must contain legitimate (0) and fraud (1) examples.")
    return float(counts[0] / counts[1])
