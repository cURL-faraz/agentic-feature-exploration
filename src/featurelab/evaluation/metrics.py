"""Metrics, with the *subject* as the unit of inference.

Epochs from one night are strongly correlated, so 40 000 epochs are not 40 000
independent observations.  Every significance test here is computed on one
number per subject - otherwise p-values are absurdly small and meaningless
("pseudo-replication").
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from ..data.cohort import STAGE_CODE, STAGES


def target_f1(y_true: np.ndarray, y_pred: np.ndarray, targets: list[str]) -> float:
    """Macro F1 over the target stages present in y_true (5-class predictions)."""
    labels = [STAGE_CODE[t] for t in targets if (y_true == STAGE_CODE[t]).any()]
    if not labels:
        return float("nan")
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def per_subject_scores(y, pred, subject, targets) -> dict[str, float]:
    return {s: target_f1(y[subject == s], pred[subject == s], targets) for s in np.unique(subject)}


def confusion_pair_errors(y, pred, pairs) -> dict[str, int]:
    """How many epochs of A were called B, plus B called A, for each pair."""
    out = {}
    for a, b in pairs:
        ca, cb = STAGE_CODE[a], STAGE_CODE[b]
        out[f"{a}<->{b}"] = int(((y == ca) & (pred == cb)).sum() + ((y == cb) & (pred == ca)).sum())
    return out


def summary(y, pred, targets) -> dict:
    per_class = f1_score(y, pred, labels=list(range(5)), average=None, zero_division=0)
    return {
        "accuracy": float((y == pred).mean()),
        "macro_f1": float(np.mean(per_class)),
        "kappa": float(cohen_kappa_score(y, pred)),
        "target_f1": target_f1(y, pred, targets),
        "per_class_f1": {STAGES[i]: round(float(v), 4) for i, v in enumerate(per_class)},
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(5))).tolist(),
    }


def paired_test(base: dict[str, float], cand: dict[str, float]) -> dict:
    """One-sided Wilcoxon signed-rank test on per-subject gains (H1: candidate better)."""
    from scipy.stats import wilcoxon

    subjects = [s for s in base if s in cand and np.isfinite(base[s]) and np.isfinite(cand[s])]
    delta = np.array([cand[s] - base[s] for s in subjects])
    result = {"n_subjects": len(subjects), "mean_delta": float(delta.mean()) if delta.size else 0.0,
              "median_delta": float(np.median(delta)) if delta.size else 0.0,
              "n_improved": int((delta > 0).sum()), "n_worse": int((delta < 0).sum())}
    if delta.size < 5 or np.allclose(delta, 0):
        result["p_value"] = 1.0
    else:
        result["p_value"] = float(wilcoxon(delta, alternative="greater", zero_method="zsplit").pvalue)
    rng = np.random.default_rng(0)
    if delta.size:
        boots = rng.choice(delta, size=(2000, delta.size), replace=True).mean(axis=1)
        result["ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return result


def holm(p_values: dict[str, float], alpha: float) -> dict[str, bool]:
    """Holm step-down: controls the family-wise error rate across finalists."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m, out, still = len(ordered), {}, True
    for rank, (key, p) in enumerate(ordered):
        still = still and p <= alpha / (m - rank)
        out[key] = still
    return out
