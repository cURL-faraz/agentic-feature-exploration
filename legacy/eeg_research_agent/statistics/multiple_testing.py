"""Multiple-comparison corrections.

Every p-value produced by the pipeline is stored twice: raw and adjusted, and the
adjustment procedure is recorded with the result so a reader can always tell
which family the correction was applied to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

import numpy as np

__all__ = ["CorrectionResult", "correct_pvalues", "benjamini_hochberg", "bonferroni", "holm"]

Method = Literal["fdr_bh", "holm", "bonferroni", "none"]


@dataclass
class CorrectionResult:
    """Raw and adjusted p-values for one family of tests."""

    method: str
    labels: list[str]
    p_raw: list[float]
    p_adjusted: list[float]
    reject: list[bool]
    alpha: float
    n_tests: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "n_tests": self.n_tests,
            "n_rejected": int(sum(self.reject)),
            "results": [
                {
                    "label": label,
                    "p_raw": float(p_raw),
                    "p_adjusted": float(p_adj),
                    "significant": bool(flag),
                }
                for label, p_raw, p_adj, flag in zip(
                    self.labels, self.p_raw, self.p_adjusted, self.reject
                )
            ],
        }

    def as_frame(self) -> list[dict[str, Any]]:
        return self.to_dict()["results"]


def _as_float_array(pvalues: Sequence[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    if values.size == 0:
        return values
    if np.nanmin(values) < 0.0 or np.nanmax(values) > 1.0:
        raise ValueError("p-values must lie in [0, 1]")
    return values


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR step-up adjustment.

    NaNs are preserved and excluded from the ranking (a test that could not be
    computed must never silently shrink the family of tests).
    """
    values = _as_float_array(pvalues)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return adjusted
    p = values[finite]
    n = p.size
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    scaled = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity (step-up)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    monotone = np.clip(monotone, 0.0, 1.0)
    restored = np.empty(n, dtype=float)
    restored[order] = monotone
    adjusted[finite] = restored
    return adjusted


def bonferroni(pvalues: Sequence[float]) -> np.ndarray:
    """Bonferroni correction (family-wise error rate)."""
    values = _as_float_array(pvalues)
    n = int(np.isfinite(values).sum())
    if n == 0:
        return np.full(values.shape, np.nan, dtype=float)
    return np.clip(values * n, 0.0, 1.0)


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down correction (uniformly more powerful than Bonferroni)."""
    values = _as_float_array(pvalues)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return adjusted
    p = values[finite]
    n = p.size
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    scaled = ranked * (n - np.arange(n))
    monotone = np.maximum.accumulate(scaled)
    monotone = np.clip(monotone, 0.0, 1.0)
    restored = np.empty(n, dtype=float)
    restored[order] = monotone
    adjusted[finite] = restored
    return adjusted


_ADJUSTERS = {
    "fdr_bh": benjamini_hochberg,
    "holm": holm,
    "bonferroni": bonferroni,
}


def correct_pvalues(
    pvalues: Sequence[float],
    labels: Iterable[str] | None = None,
    method: Method = "fdr_bh",
    alpha: float = 0.05,
) -> CorrectionResult:
    """Apply ``method`` and return raw/adjusted p-values with a reject flag."""
    method = (method or "none").lower()  # type: ignore[assignment]
    if method not in {*_ADJUSTERS, "none"}:
        raise ValueError(f"unknown correction method: {method}")
    values = _as_float_array(pvalues)
    names = list(labels) if labels is not None else [f"test_{i}" for i in range(values.size)]
    if len(names) != values.size:
        raise ValueError("labels and p-values must have the same length")
    if values.size == 0:
        return CorrectionResult(method, [], [], [], [], alpha, 0)
    adjusted = values.copy() if method == "none" else _ADJUSTERS[method](values)
    reject = [bool(np.isfinite(a) and a <= alpha) for a in adjusted]
    return CorrectionResult(
        method=method,
        labels=names,
        p_raw=[float(v) for v in values],
        p_adjusted=[float(v) for v in adjusted],
        reject=reject,
        alpha=alpha,
        n_tests=int(values.size),
    )
