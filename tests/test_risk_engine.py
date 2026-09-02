import pytest

from src.risk_engine import probability_to_risk


THRESHOLDS = {"low_max": 29, "medium_max": 59, "high_max": 84}


@pytest.mark.parametrize(
    ("probability", "score", "category"),
    [(0.0, 0, "LOW"), (0.30, 30, "MEDIUM"), (0.60, 60, "HIGH"), (0.85, 85, "CRITICAL")],
)
def test_probability_maps_to_configured_band(probability: float, score: int, category: str) -> None:
    assessment = probability_to_risk(probability, THRESHOLDS)
    assert assessment.score == score
    assert assessment.category == category


def test_probability_must_be_bounded() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        probability_to_risk(1.1, THRESHOLDS)

