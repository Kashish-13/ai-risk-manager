import numpy as np

from src.evaluate import evaluate_predictions, select_f1_threshold


def test_threshold_is_selected_from_validation_scores() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.01, 0.30, 0.70, 0.95])
    threshold, details = select_f1_threshold(labels, scores)
    assert 0 < threshold < 1
    assert details["validation_f1_at_selected_threshold"] == 1.0


def test_evaluation_returns_real_metric_contract() -> None:
    metrics = evaluate_predictions(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    assert metrics["support"] == {"legitimate": 2, "fraud": 2}
    assert metrics["pr_auc"] == 1.0

