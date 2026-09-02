import json

import joblib
import numpy as np
import pandas as pd

from src.predict import load_prediction_artifacts
from src.train import class_imbalance_weight


class StoredDummyModel:
    def predict_proba(self, X):
        return np.tile([[0.9, 0.1]], (len(X), 1))


def test_class_imbalance_weight_uses_training_labels_only() -> None:
    assert class_imbalance_weight(pd.Series([0, 0, 0, 1])) == 3.0


def test_saved_prediction_artifacts_can_be_loaded(tmp_path) -> None:
    joblib.dump(StoredDummyModel(), tmp_path / "final_model.joblib")
    (tmp_path / "model_metadata.json").write_text(json.dumps({"raw_feature_columns": ["Amount"]}), encoding="utf-8")
    (tmp_path / "threshold.json").write_text(json.dumps({"fraud_decision_threshold": 0.42}), encoding="utf-8")
    model, metadata, threshold = load_prediction_artifacts(tmp_path)
    assert metadata["raw_feature_columns"] == ["Amount"]
    assert threshold == 0.42
    assert model.predict_proba(pd.DataFrame({"Amount": [1.0]})).shape == (1, 2)
