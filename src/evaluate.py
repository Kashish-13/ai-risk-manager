"""Metrics, threshold calibration, and presentation-ready evaluation plots."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

# Training runs non-interactively (including CI and Streamlit deployment).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Choose the validation threshold with maximum F1; ties use higher recall.

    This documented rule is applied only to validation predictions. It treats recall
    and precision symmetrically while avoiding an unjustified default of 0.5.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_f1 = np.nanmax(f1)
    candidates = np.flatnonzero(np.isclose(f1, best_f1))
    best_index = candidates[np.argmax(recall[:-1][candidates])]
    return float(thresholds[best_index]), {
        "selection_rule": "maximum validation F1; tie-breaker: higher validation recall",
        "validation_f1_at_selected_threshold": float(f1[best_index]),
        "validation_precision_at_selected_threshold": float(precision[best_index]),
        "validation_recall_at_selected_threshold": float(recall[best_index]),
    }


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    """Calculate genuine classification and ranking metrics at a fixed threshold."""
    labels = (np.asarray(probabilities) >= threshold).astype(int)
    report = classification_report(y_true, labels, output_dict=True, zero_division=0)
    return {
        "decision_threshold": float(threshold),
        "precision": float(precision_score(y_true, labels, zero_division=0)),
        "recall": float(recall_score(y_true, labels, zero_division=0)),
        "f1": float(f1_score(y_true, labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, labels, labels=[0, 1]).tolist(),
        "classification_report": report,
        "support": {"legitimate": int((np.asarray(y_true) == 0).sum()), "fraud": int((np.asarray(y_true) == 1).sum())},
    }


def save_evaluation_plots(y_true: np.ndarray, probabilities: np.ndarray, threshold: float, output_dir: Path) -> None:
    """Write legible held-out evaluation figures without modifying model state."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = (np.asarray(probabilities) >= threshold).astype(int)
    matrix = confusion_matrix(y_true, labels, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"], xlabel="Predicted", ylabel="Actual", title="Held-out Test Confusion Matrix")
    for row in range(2):
        for column in range(2): axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout(); figure.savefig(output_dir / "confusion_matrix.png", dpi=160); plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 4)); PrecisionRecallDisplay.from_predictions(y_true, probabilities, ax=axis); axis.set_title("Held-out Test Precision–Recall Curve"); figure.tight_layout(); figure.savefig(output_dir / "precision_recall_curve.png", dpi=160); plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 4)); RocCurveDisplay.from_predictions(y_true, probabilities, ax=axis); axis.set_title("Held-out Test ROC Curve"); figure.tight_layout(); figure.savefig(output_dir / "roc_curve.png", dpi=160); plt.close(figure)


def save_comparison_plot(comparison: dict[str, dict[str, Any]], output_path: Path) -> None:
    names = list(comparison); values = [comparison[name]["pr_auc"] for name in names]
    figure, axis = plt.subplots(figsize=(6, 4)); axis.bar(names, values, color=["#64748b", "#2563eb"][:len(names)]); axis.set_ylim(0, 1); axis.set_ylabel("PR-AUC"); axis.set_title("Validation Model Comparison")
    for index, value in enumerate(values): axis.text(index, value, f"{value:.4f}", ha="center", va="bottom")
    figure.tight_layout(); figure.savefig(output_path, dpi=160); plt.close(figure)


def save_threshold_plot(y_true: np.ndarray, probabilities: np.ndarray, chosen: float, output_path: Path) -> None:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    figure, axis = plt.subplots(figsize=(7, 4)); axis.plot(thresholds, f1, label="Validation F1"); axis.axvline(chosen, color="#dc2626", linestyle="--", label=f"Selected: {chosen:.4f}"); axis.set(xlabel="Fraud decision threshold", ylabel="F1", title="Validation Threshold Calibration"); axis.legend(); figure.tight_layout(); figure.savefig(output_path, dpi=160); plt.close(figure)
