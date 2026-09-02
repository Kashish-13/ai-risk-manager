"""Configurable conversion from model probability to human-facing risk."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RiskAssessment:
    probability: float
    score: int
    category: str


def probability_to_risk(probability: float, thresholds: Mapping[str, int]) -> RiskAssessment:
    """Map a probability in [0, 1] to an integer score and configured category."""
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1 inclusive.")
    required = ("low_max", "medium_max", "high_max")
    if any(key not in thresholds for key in required):
        raise ValueError(f"Thresholds must provide: {', '.join(required)}")
    low, medium, high = (int(thresholds[key]) for key in required)
    if not 0 <= low < medium < high < 100:
        raise ValueError("Risk thresholds must satisfy 0 <= low < medium < high < 100.")
    score = round(probability * 100)
    category = "LOW" if score <= low else "MEDIUM" if score <= medium else "HIGH" if score <= high else "CRITICAL"
    return RiskAssessment(probability=float(probability), score=score, category=category)

