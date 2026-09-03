"""Prediction input/output contracts shared by future UI and batch workflows."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import joblib

import numpy as np
import pandas as pd

from src.risk_engine import RiskAssessment, probability_to_risk


class ProbabilityModel(Protocol):
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


def validate_prediction_input(frame: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    """Ensure one or more prediction rows match the trained feature schema."""
    if frame.empty:
        raise ValueError("At least one transaction row is required for prediction.")
    missing = [column for column in expected_columns if column not in frame.columns]
    extras = [column for column in frame.columns if column not in expected_columns]
    if missing or extras:
        details = []
        if missing:
            details.append(f"missing columns: {missing}")
        if extras:
            details.append(f"unexpected columns: {extras}")
        raise ValueError("Prediction schema mismatch (" + "; ".join(details) + ").")
    return frame.loc[:, expected_columns].copy()


def score_transactions(
    model: ProbabilityModel,
    frame: pd.DataFrame,
    *,
    expected_columns: list[str],
    thresholds: dict[str, float],
) -> list[RiskAssessment]:
    """Return one risk assessment per validated transaction without mutating input."""
    validated = validate_prediction_input(frame, expected_columns)
    probabilities = np.asarray(model.predict_proba(validated))
    if probabilities.ndim != 2 or probabilities.shape[0] != len(validated) or probabilities.shape[1] < 2:
        raise ValueError("Model predict_proba must return an (n_rows, n_classes) array.")
    return [probability_to_risk(float(value), thresholds) for value in probabilities[:, 1]]


def load_prediction_artifacts(models_dir: Path | str) -> tuple[ProbabilityModel, dict, float]:
    """Load the saved pipeline, feature metadata, and validation-selected threshold."""
    directory = Path(models_dir)
    model = joblib.load(directory / "final_model.joblib")
    metadata = json.loads((directory / "model_metadata.json").read_text(encoding="utf-8"))
    threshold = float(json.loads((directory / "threshold.json").read_text(encoding="utf-8"))["fraud_decision_threshold"])
    return model, metadata, threshold


def score_raw_transactions(model: ProbabilityModel, raw_frame: pd.DataFrame, metadata: dict, thresholds: dict[str, float]) -> list[RiskAssessment]:
    """Score raw rows; the saved pipeline owns feature engineering and preprocessing."""
    required = metadata["raw_feature_columns"]
    validated_raw = validate_prediction_input(raw_frame, required)
    return score_transactions(model, validated_raw, expected_columns=required, thresholds=thresholds)
