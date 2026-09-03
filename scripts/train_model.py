"""Train and evaluate fraud-risk models using a validation-selected threshold."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Keep Matplotlib's generated font cache inside the writable project workspace.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reports" / ".matplotlib"))

from src.config import load_config
from src.data_loader import load_csv, split_features_target
from src.evaluate import evaluate_predictions, save_comparison_plot, save_evaluation_plots, save_threshold_plot, select_f1_threshold
from src.explain import ranked_feature_importance
from src.preprocessing import split_train_validation_test
from src.train import build_logistic_pipeline, build_xgboost_pipeline, class_imbalance_weight
from src.validation import validate_dataframe


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _probabilities(pipeline, features) -> np.ndarray:
    return np.asarray(pipeline.predict_proba(features))[:, 1]


def _duplicate_summary(frame, target_column: str) -> dict[str, int]:
    features = frame.drop(columns=[target_column])
    return {
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "feature_duplicate_rows": int(features.duplicated().sum()),
        "feature_groups_with_label_conflicts": int(frame.groupby(features.columns.tolist(), dropna=False)[target_column].nunique().gt(1).sum()),
    }


def run_training(data_path: Path) -> dict[str, Any]:
    """Run selection on validation data; make one final held-out test evaluation."""
    config = load_config()
    frame = load_csv(data_path)
    report = validate_dataframe(frame, config.target_column)
    if not report.is_valid:
        raise ValueError("Invalid training data: " + "; ".join(report.errors))
    duplicate_summary = _duplicate_summary(frame, config.target_column)
    handling = config.duplicate_handling  # benchmark duplicates are not silently removed.
    if handling != "retain":
        raise ValueError(f"Unsupported duplicate handling '{handling}'; only 'retain' is currently implemented.")
    X_raw, y = split_features_target(frame, config.target_column)
    duplicate_groups = pd.util.hash_pandas_object(X_raw, index=False)
    splits = split_train_validation_test(X_raw, y, test_size=config.test_size, validation_size=config.validation_size, random_seed=config.random_seed, stratify=True, groups=duplicate_groups)
    train_weight = class_imbalance_weight(splits.y_train)

    baseline = build_logistic_pipeline(splits.X_train, config.model_params["logistic_regression"])
    baseline.fit(splits.X_train, splits.y_train)
    baseline_probabilities = _probabilities(baseline, splits.X_validation)
    baseline_metrics = evaluate_predictions(splits.y_validation.to_numpy(), baseline_probabilities, 0.5)

    selection_note = "XGBoost selected by validation PR-AUC."
    try:
        candidate = build_xgboost_pipeline(splits.X_train, config.model_params["xgboost"], scale_pos_weight=train_weight)
        candidate_name = "XGBoost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.pipeline import Pipeline
        from src.preprocessing import build_preprocessor
        candidate = Pipeline([("preprocessing", build_preprocessor(splits.X_train)), ("classifier", HistGradientBoostingClassifier(random_state=config.random_seed))])
        candidate_name, selection_note = "HistGradientBoostingClassifier", "XGBoost unavailable; fallback assessed by validation PR-AUC."
    candidate.fit(splits.X_train, splits.y_train)
    candidate_probabilities = _probabilities(candidate, splits.X_validation)
    threshold, threshold_details = select_f1_threshold(splits.y_validation.to_numpy(), candidate_probabilities)
    candidate_metrics = evaluate_predictions(splits.y_validation.to_numpy(), candidate_probabilities, threshold)
    comparison = {"logistic_regression": baseline_metrics, candidate_name.lower(): candidate_metrics}

    final_pipeline, model_name, final_probabilities, final_validation = candidate, candidate_name, candidate_probabilities, candidate_metrics
    if baseline_metrics["pr_auc"] > candidate_metrics["pr_auc"]:
        final_pipeline, model_name, final_probabilities = baseline, "LogisticRegression", baseline_probabilities
        threshold, threshold_details = select_f1_threshold(splits.y_validation.to_numpy(), baseline_probabilities)
        final_validation = evaluate_predictions(splits.y_validation.to_numpy(), baseline_probabilities, threshold)
        selection_note = "Logistic Regression selected because its validation PR-AUC exceeded the tree model."

    # Test data is only used here, after final model and threshold selection.
    test_probabilities = _probabilities(final_pipeline, splits.X_test)
    test_metrics = evaluate_predictions(splits.y_test.to_numpy(), test_probabilities, threshold)
    models_dir, metrics_dir, figures_dir = ROOT / "models", ROOT / "reports/metrics", ROOT / "reports/figures"
    models_dir.mkdir(exist_ok=True); metrics_dir.mkdir(parents=True, exist_ok=True); figures_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, models_dir / "final_model.joblib")
    joblib.dump(final_pipeline.named_steps["preprocessing"], models_dir / "preprocessing.joblib")
    engineered_columns = final_pipeline.named_steps["feature_engineering"].transform(splits.X_train.head(1)).columns.tolist()
    metadata = {"dataset": "OpenML dataset 1597", "dataset_rows": len(frame), "dataset_fraud_count": int(y.sum()), "target_column": config.target_column, "raw_feature_columns": X_raw.columns.tolist(), "engineered_feature_columns": engineered_columns, "random_seed": config.random_seed, "duplicate_handling": handling, "duplicate_summary": duplicate_summary, "split_method": "stratified_group_split_by_identical_feature_rows", "split_sizes": {"train": len(splits.X_train), "validation": len(splits.X_validation), "test": len(splits.X_test)}, "split_fraud_counts": {"train": int(splits.y_train.sum()), "validation": int(splits.y_validation.sum()), "test": int(splits.y_test.sum())}, "model": model_name, "selection_note": selection_note, "class_imbalance_weight_from_train": train_weight, "threshold_selection": threshold_details}
    _write_json(models_dir / "model_metadata.json", metadata)
    _write_json(models_dir / "threshold.json", {"fraud_decision_threshold": threshold, **threshold_details})
    _write_json(metrics_dir / "validation_metrics.json", {"baseline_logistic_regression": baseline_metrics, "selected_model": model_name, "selected_model_validation": final_validation, "threshold_selection": threshold_details})
    _write_json(metrics_dir / "test_metrics.json", {"label": "HELD-OUT TEST SET RESULTS", "selected_model": model_name, **test_metrics})
    _write_json(metrics_dir / "classification_report.json", {"label": "HELD-OUT TEST SET RESULTS", **test_metrics["classification_report"]})
    _write_json(metrics_dir / "model_comparison.json", {"primary_selection_metric": "PR-AUC", "validation_only": comparison, "selected_model": model_name, "selection_note": selection_note})
    _write_json(metrics_dir / "feature_importance.json", {"wording": "Top risk indicators associated with this model prediction; importance is not causation.", "features": ranked_feature_importance(final_pipeline).to_dict(orient="records")})
    demo_payload = {}
    for label, name, chooser in ((0, "low_risk", np.argmin), (1, "high_risk", np.argmax)):
        positions = np.flatnonzero(splits.y_test.to_numpy() == label)
        selected = int(positions[chooser(test_probabilities[positions])])
        demo_payload[name] = {
            "source": "held-out test set (saved after final evaluation; inference demo only)",
            "actual_label": int(label),
            "features": {key: float(value) for key, value in splits.X_test.iloc[selected].items()},
        }
    _write_json(models_dir / "demo_samples.json", demo_payload)
    save_evaluation_plots(splits.y_test.to_numpy(), test_probabilities, threshold, figures_dir)
    save_comparison_plot(comparison, figures_dir / "model_comparison.png")
    save_threshold_plot(splits.y_validation.to_numpy(), final_probabilities, threshold, figures_dir / "risk_threshold_analysis.png")
    return {"metadata": metadata, "baseline": baseline_metrics, "final_validation": final_validation, "test": test_metrics, "threshold": threshold}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="CSV path; defaults to config primary_file.")
    args = parser.parse_args()
    result = run_training(args.data or load_config().primary_file)
    print(f"Dataset: {result['metadata']['dataset']}")
    print(f"Duplicate handling: {result['metadata']['duplicate_handling']} ({result['metadata']['duplicate_summary']})")
    print(f"Split sizes / fraud: {result['metadata']['split_sizes']} / {result['metadata']['split_fraud_counts']}")
    print(f"Logistic Regression validation: {result['baseline']}")
    print(f"Selected model: {result['metadata']['model']}")
    print(f"Selected threshold (validation only): {result['threshold']:.6f}")
    print("HELD-OUT TEST SET RESULTS")
    print(result["test"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
