"""Classification metrics, including Cohen's kappa implemented in-house.

``sklearn.metrics.cohen_kappa_score`` is available, but the pipeline computes
kappa from the confusion matrix itself so the definition used in the reports is
explicit and independently checkable:

.. math::
    \\kappa = \\frac{p_o - p_e}{1 - p_e}, \\qquad
    p_o = \\frac{\\sum_i n_{ii}}{N}, \\qquad
    p_e = \\frac{\\sum_i a_i b_i}{N^2}

where ``a_i`` is the row total (true counts) and ``b_i`` the column total
(predicted counts).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = [
    "confusion_matrix",
    "cohens_kappa",
    "per_class_metrics",
    "macro_average",
    "classification_report",
    "matthews_corrcoef",
]


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> np.ndarray:
    """Confusion matrix with rows = truth, columns = prediction (label order given)."""
    index = {label: position for position, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true_label, predicted_label in zip(y_true, y_pred):
        if true_label in index and predicted_label in index:
            matrix[index[true_label], index[predicted_label]] += 1
    return matrix


def cohens_kappa(matrix: np.ndarray) -> float:
    """Cohen's kappa computed from a confusion matrix."""
    matrix = np.asarray(matrix, dtype=float)
    total = matrix.sum()
    if total == 0:
        return float("nan")
    observed = float(np.trace(matrix) / total)
    row_totals = matrix.sum(axis=1)
    column_totals = matrix.sum(axis=0)
    expected = float(np.sum(row_totals * column_totals) / (total * total))
    if abs(1.0 - expected) < 1e-12:
        return 1.0 if observed >= 1.0 - 1e-12 else 0.0
    return float((observed - expected) / (1.0 - expected))


def matthews_corrcoef(matrix: np.ndarray) -> float:
    """Multiclass Matthews correlation coefficient (a balanced measure)."""
    matrix = np.asarray(matrix, dtype=float)
    total = matrix.sum()
    if total == 0:
        return float("nan")
    row_totals = matrix.sum(axis=1)
    column_totals = matrix.sum(axis=0)
    correct = float(np.trace(matrix))
    numerator = correct * total - float(np.dot(row_totals, column_totals))
    denominator = np.sqrt(
        (total**2 - float(np.dot(column_totals, column_totals)))
        * (total**2 - float(np.dot(row_totals, row_totals)))
    )
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def per_class_metrics(
    matrix: np.ndarray, labels: Sequence[str], zero_division: bool = False
) -> dict[str, dict[str, Any]]:
    """One-vs-rest precision / recall / F1 / support for every class."""
    matrix = np.asarray(matrix, dtype=float)
    total = matrix.sum()
    result: dict[str, dict[str, Any]] = {}
    for index, label in enumerate(labels):
        true_positive = float(matrix[index, index])
        predicted_positive = float(matrix[:, index].sum())
        actual_positive = float(matrix[index, :].sum())
        precision = true_positive / predicted_positive if predicted_positive else float("nan")
        recall = true_positive / actual_positive if actual_positive else float("nan")
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
            f1 = float(2 * precision * recall / (precision + recall))
        else:
            f1 = 0.0 if zero_division else float("nan")
        result[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(actual_positive),
            "predicted": int(predicted_positive),
            "prevalence": float(actual_positive / total) if total else float("nan"),
        }
    return result


def macro_average(per_class: dict[str, dict[str, Any]], key: str) -> float:
    """Unweighted mean of a per-class metric (classes with no support excluded)."""
    values = [
        metrics[key]
        for metrics in per_class.values()
        if metrics.get("support", 0) > 0 and np.isfinite(metrics.get(key, float("nan")))
    ]
    return float(np.mean(values)) if values else float("nan")


def classification_report(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict[str, Any]:
    """Full metric bundle for one prediction set."""
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    matrix = confusion_matrix(y_true, y_pred, labels)
    per_class = per_class_metrics(matrix, labels)
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else float("nan")
    supports = [per_class[label]["support"] for label in labels]
    recalls = [per_class[label]["recall"] for label in labels]
    balanced = float(np.mean([r for r, n in zip(recalls, supports) if n > 0])) if any(
        s > 0 for s in supports
    ) else float("nan")
    return {
        "n_samples": total,
        "labels": list(labels),
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "macro_precision": macro_average(per_class, "precision"),
        "macro_recall": macro_average(per_class, "recall"),
        "macro_f1": macro_average(per_class, "f1"),
        "weighted_f1": _weighted(per_class, "f1"),
        "cohens_kappa": cohens_kappa(matrix),
        "matthews_corrcoef": matthews_corrcoef(matrix),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "predicted_distribution": {
            label: int(matrix[:, index].sum()) for index, label in enumerate(labels)
        },
        "true_distribution": {
            label: int(matrix[index, :].sum()) for index, label in enumerate(labels)
        },
    }


def _weighted(per_class: dict[str, dict[str, Any]], key: str) -> float:
    total = sum(metrics.get("support", 0) for metrics in per_class.values())
    if total == 0:
        return float("nan")
    return float(
        sum(
            metrics[key] * metrics.get("support", 0)
            for metrics in per_class.values()
            if np.isfinite(metrics.get(key, float("nan")))
        )
        / total
    )
