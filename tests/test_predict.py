import numpy as np
import pandas as pd
import pytest

from src.predict import score_transactions, validate_prediction_input


class DummyModel:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.repeat(0.9, len(X)), np.repeat(0.1, len(X))])


THRESHOLDS = {"low_max": 29, "medium_max": 59, "high_max": 84}


def test_prediction_contract_returns_one_assessment_per_input() -> None:
    frame = pd.DataFrame({"Time": [1.0, 2.0], "Amount": [10.0, 20.0]})
    assessments = score_transactions(
        DummyModel(), frame, expected_columns=["Time", "Amount"], thresholds=THRESHOLDS
    )
    assert len(assessments) == 2
    assert all(item.category == "LOW" for item in assessments)


def test_prediction_contract_rejects_schema_mismatch() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        validate_prediction_input(pd.DataFrame({"Time": [1.0]}), ["Time", "Amount"])

