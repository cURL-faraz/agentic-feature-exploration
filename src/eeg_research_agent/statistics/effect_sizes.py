"""Effect sizes and confidence intervals.

The project never reports a p-value without an effect size.  Small p-values with
negligible effect sizes are common in EEG because epoch counts are large, and
epochs are *not* independent observations - so the effective sample size is much
smaller than the row count.  Effect sizes are computed on the unit of analysis
that the comparison actually uses (usually subject-level aggregates) and are
reported with bootstrap confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    "EffectSize",
    "cohens_d",
    "hedges_g",
    "glass_delta",
    "rank_biserial",
    "cliffs_delta",
    "eta_squared_from_h",
    "epsilon_squared_from_h",
    "eta_squared_from_f",
    "omega_squared_from_f",
    "bootstrap_ci",
    "interpret_effect",
]


@dataclass
class EffectSize:
    """One effect-size estimate with an optional confidence interval."""

    measure: str
    value: float
    ci_low: float | None = None
    ci_high: float | None = None
    n_a: int | None = None
    n_b: int | None = None
    interpretation: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "measure": self.measure,
            "value": None if self.value is None or not np.isfinite(self.value) else float(self.value),
            "ci_low": None if self.ci_low is None or not np.isfinite(self.ci_low) else float(self.ci_low),
            "ci_high": None if self.ci_high is None or not np.isfinite(self.ci_high) else float(self.ci_high),
            "n_a": self.n_a,
            "n_b": self.n_b,
            "interpretation": self.interpretation,
            "notes": self.notes,
        }


def interpret_effect(value: float, measure: str = "d") -> str:
    """Cohen's conventional verbal buckets (reported *as convention*, not truth)."""
    if value is None or not np.isfinite(value):
        return "undefined"
    magnitude = abs(float(value))
    if measure in {"eta_squared", "epsilon_squared", "partial_eta_squared"}:
        cuts = ((0.01, "negligible"), (0.06, "small"), (0.14, "medium"), (float("inf"), "large"))
    elif measure in {"rank_biserial", "cliffs_delta"}:
        cuts = ((0.147, "negligible"), (0.33, "small"), (0.474, "medium"), (float("inf"), "large"))
    else:  # d / g / delta
        cuts = ((0.2, "negligible"), (0.5, "small"), (0.8, "medium"), (float("inf"), "large"))
    for threshold, label in cuts:
        if magnitude < threshold:
            return label
    return "large"


def _clean(a: Sequence[float]) -> np.ndarray:
    values = np.asarray(a, dtype=float).ravel()
    return values[np.isfinite(values)]


def _pooled_sd(a: np.ndarray, b: np.ndarray) -> float:
    n_a, n_b = a.size, b.size
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    return float(np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)))


def cohens_d(a: Sequence[float], b: Sequence[float]) -> EffectSize:
    """Standardised mean difference using the pooled standard deviation."""
    x, y = _clean(a), _clean(b)
    sd = _pooled_sd(x, y)
    if not np.isfinite(sd) or sd == 0.0:
        return EffectSize("cohens_d", float("nan"), n_a=x.size, n_b=y.size, interpretation="undefined")
    value = float((np.mean(x) - np.mean(y)) / sd)
    return EffectSize("cohens_d", value, n_a=x.size, n_b=y.size, interpretation=interpret_effect(value, "d"))


def hedges_g(a: Sequence[float], b: Sequence[float]) -> EffectSize:
    """Bias-corrected Cohen's d (small-sample correction factor J)."""
    x, y = _clean(a), _clean(b)
    d = cohens_d(x, y)
    n = x.size + y.size
    if n < 4 or not np.isfinite(d.value):
        return EffectSize("hedges_g", float("nan"), n_a=x.size, n_b=y.size, interpretation="undefined")
    correction = 1.0 - 3.0 / (4.0 * n - 9.0)
    value = float(d.value * correction)
    return EffectSize("hedges_g", value, n_a=x.size, n_b=y.size, interpretation=interpret_effect(value, "g"))


def glass_delta(a: Sequence[float], b: Sequence[float]) -> EffectSize:
    """Mean difference standardised by the *control* (second group) SD."""
    x, y = _clean(a), _clean(b)
    if y.size < 2:
        return EffectSize("glass_delta", float("nan"), n_a=x.size, n_b=y.size, interpretation="undefined")
    sd = float(np.std(y, ddof=1))
    if sd == 0.0:
        return EffectSize("glass_delta", float("nan"), n_a=x.size, n_b=y.size, interpretation="undefined")
    value = float((np.mean(x) - np.mean(y)) / sd)
    return EffectSize(
        "glass_delta", value, n_a=x.size, n_b=y.size, interpretation=interpret_effect(value, "delta")
    )


def rank_biserial(u_statistic: float, n_a: int, n_b: int, direction: float = 1.0) -> EffectSize:
    """Rank-biserial correlation from a Mann-Whitney U statistic.

    ``r = 2U/(n_a n_b) - 1`` (positive when group A tends to be larger).
    """
    if n_a <= 0 or n_b <= 0:
        return EffectSize("rank_biserial", float("nan"), n_a=n_a, n_b=n_b, interpretation="undefined")
    value = float(direction * (2.0 * u_statistic / (n_a * n_b) - 1.0))
    return EffectSize(
        "rank_biserial", value, n_a=n_a, n_b=n_b, interpretation=interpret_effect(value, "rank_biserial")
    )


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> EffectSize:
    """Cliff's delta: P(X > Y) - P(X < Y), computed from ranks (O(n log n))."""
    x, y = _clean(a), _clean(b)
    if x.size == 0 or y.size == 0:
        return EffectSize("cliffs_delta", float("nan"), n_a=x.size, n_b=y.size, interpretation="undefined")
    combined = np.concatenate([x, y])
    ranks = _average_ranks(combined)
    rank_x = ranks[: x.size].sum()
    u = rank_x - x.size * (x.size + 1) / 2.0
    value = float(2.0 * u / (x.size * y.size) - 1.0)
    return EffectSize(
        "cliffs_delta", value, n_a=x.size, n_b=y.size, interpretation=interpret_effect(value, "cliffs_delta")
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average (midrank) ranking with tie handling, matching scipy's convention."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    i = 0
    while i < sorted_values.size:
        j = i
        while j + 1 < sorted_values.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def eta_squared_from_h(h: float, n: int, k: int) -> float:
    """Rank-based eta-squared for Kruskal-Wallis: ``(H - k + 1) / (n - k)``."""
    if n <= k:
        return float("nan")
    return float(max(0.0, (h - k + 1.0) / (n - k)))


def epsilon_squared_from_h(h: float, n: int) -> float:
    """Rank-based epsilon-squared for Kruskal-Wallis: ``H / ((n^2 - 1)/(n + 1))``."""
    if n <= 1:
        return float("nan")
    return float(max(0.0, h / ((n * n - 1.0) / (n + 1.0))))


def eta_squared_from_f(f: float, df_between: int, df_within: int) -> float:
    """Classical eta-squared from an F statistic."""
    denominator = f * df_between + df_within
    if denominator <= 0:
        return float("nan")
    return float(max(0.0, f * df_between / denominator))


def omega_squared_from_f(f: float, df_between: int, df_within: int) -> float:
    """Omega-squared: a less biased population estimate than eta-squared."""
    denominator = f * df_between + df_within + 1.0
    if denominator <= 0:
        return float("nan")
    return float(max(0.0, (f * df_between - df_within) / denominator))


def bootstrap_ci(
    statistic: Callable[[np.ndarray, np.ndarray], float],
    a: Sequence[float],
    b: Sequence[float] | None = None,
    iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 20240517,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a two-sample statistic.

    Resampling is done **within each group**, which is the correct scheme for an
    independent-groups statistic and preserves the group sizes.
    """
    x = _clean(a)
    y = _clean(b) if b is not None else None
    if x.size < 2 or (y is not None and y.size < 2):
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sample_x = x[rng.integers(0, x.size, x.size)]
        sample_y = y[rng.integers(0, y.size, y.size)] if y is not None else None
        estimates[index] = statistic(sample_x, sample_y)  # type: ignore[arg-type]
    estimates = estimates[np.isfinite(estimates)]
    if estimates.size == 0:
        return (float("nan"), float("nan"))
    return (
        float(np.percentile(estimates, 100.0 * alpha / 2.0)),
        float(np.percentile(estimates, 100.0 * (1.0 - alpha / 2.0))),
    )


def hedges_g_with_ci(
    a: Sequence[float],
    b: Sequence[float],
    iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 20240517,
) -> EffectSize:
    """Hedges' g with a bootstrap confidence interval."""
    effect = hedges_g(a, b)

    def _statistic(x: np.ndarray, y: np.ndarray) -> float:
        return hedges_g(x, y).value

    low, high = bootstrap_ci(_statistic, a, b, iterations=iterations, alpha=alpha, seed=seed)
    effect.ci_low, effect.ci_high = low, high
    return effect
