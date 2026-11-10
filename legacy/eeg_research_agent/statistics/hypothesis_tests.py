"""Deterministic hypothesis testing for feature -> sleep-stage association.

Design decisions (all enforced in code, not by an LLM)
------------------------------------------------------
1. **Normality** is assessed with D'Agostino-Pearson (n >= 8) or Shapiro-Wilk
   (small n) and **variance homogeneity** with Levene's test.  The two-sample and
   k-sample tests are then chosen by an explicit decision table.
2. **Repeated measures.**  Epochs are clustered within subjects, so the primary
   analysis runs on *subject x stage* means.  When the design is complete (every
   subject contributes every stage) a Friedman test with paired Wilcoxon
   post-hoc is used; otherwise an unpaired Kruskal-Wallis is used on the same
   aggregates.  The epoch-level test is still reported, but explicitly flagged as
   pseudo-replicated.
3. **Post-hoc** comparisons always carry a multiple-comparison correction and a
   rank-based effect size.
4. **Permutation tests** resample whole *subjects* (cluster permutation), which
   is the appropriate null for clustered data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .effect_sizes import (
    EffectSize,
    cliffs_delta,
    epsilon_squared_from_h,
    eta_squared_from_f,
    eta_squared_from_h,
    hedges_g,
    rank_biserial,
)
from .multiple_testing import correct_pvalues

__all__ = [
    "AssumptionCheck",
    "TestResult",
    "ComparisonResult",
    "check_normality",
    "check_variance_homogeneity",
    "choose_two_sample_test",
    "choose_k_sample_test",
    "welch_anova",
    "two_sample_test",
    "k_sample_test",
    "dunn_posthoc",
    "clustered_permutation_test",
    "compare_feature",
]

MIN_NORMALITY_N = 8


@dataclass
class AssumptionCheck:
    """Outcome of one assumption test, with the decision it drove."""

    name: str
    statistic: float | None
    p_value: float | None
    satisfied: bool | None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "statistic": None if self.statistic is None else float(self.statistic),
            "p_value": None if self.p_value is None else float(self.p_value),
            "satisfied": self.satisfied,
            "detail": self.detail,
        }


@dataclass
class TestResult:
    """One omnibus or pairwise test."""

    test: str
    statistic: float
    p_value: float
    df: tuple[float, ...] = ()
    n: int = 0
    group_sizes: dict[str, int] = field(default_factory=dict)
    effect_size: EffectSize | None = None
    p_adjusted: float | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "test": self.test,
            "statistic": None if not np.isfinite(self.statistic) else float(self.statistic),
            "p_value": None if not np.isfinite(self.p_value) else float(self.p_value),
            "p_adjusted": None if self.p_adjusted is None or not np.isfinite(self.p_adjusted) else float(self.p_adjusted),
            "df": [float(d) for d in self.df],
            "n": int(self.n),
            "group_sizes": dict(self.group_sizes),
            "effect_size": self.effect_size.to_dict() if self.effect_size else None,
            "notes": self.notes,
        }


@dataclass
class ComparisonResult:
    """Complete comparison of one feature across sleep stages."""

    feature: str
    stages: list[str]
    unit_of_analysis: str
    primary: TestResult | None = None
    epoch_level: TestResult | None = None
    permutation: TestResult | None = None
    posthoc: list[TestResult] = field(default_factory=list)
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    group_summary: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "stages": list(self.stages),
            "unit_of_analysis": self.unit_of_analysis,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "primary": self.primary.to_dict() if self.primary else None,
            "epoch_level": self.epoch_level.to_dict() if self.epoch_level else None,
            "permutation": self.permutation.to_dict() if self.permutation else None,
            "posthoc": [p.to_dict() for p in self.posthoc],
            "assumptions": [a.to_dict() for a in self.assumptions],
            "group_summary": self.group_summary,
        }


# --------------------------------------------------------------------------- #
# assumption checks
# --------------------------------------------------------------------------- #
def check_normality(values: Sequence[float], alpha: float = 0.05, label: str = "") -> AssumptionCheck:
    """D'Agostino-Pearson / Shapiro-Wilk normality check on one group."""
    array = np.asarray(values, dtype=float).ravel()
    array = array[np.isfinite(array)]
    if array.size < 3:
        return AssumptionCheck(f"normality:{label}", None, None, None, "too few observations")
    if np.allclose(array, array[0]):
        return AssumptionCheck(f"normality:{label}", None, None, False, "constant group")
    try:
        if array.size >= MIN_NORMALITY_N:
            statistic, p_value = stats.normaltest(array)
            name = "D'Agostino-Pearson"
        else:
            statistic, p_value = stats.shapiro(array)
            name = "Shapiro-Wilk"
    except Exception as exc:  # pragma: no cover - numerical edge case
        return AssumptionCheck(f"normality:{label}", None, None, None, f"test failed: {exc}")
    return AssumptionCheck(
        f"normality:{label}",
        float(statistic),
        float(p_value),
        bool(p_value > alpha),
        f"{name}; H0 = normally distributed",
    )


def check_variance_homogeneity(groups: Sequence[Sequence[float]], alpha: float = 0.05) -> AssumptionCheck:
    """Levene's test (median-centred = Brown-Forsythe) for equal variances."""
    cleaned = [np.asarray(g, dtype=float) for g in groups]
    cleaned = [g[np.isfinite(g)] for g in cleaned if np.isfinite(g).sum() >= 2]
    if len(cleaned) < 2:
        return AssumptionCheck("variance_homogeneity", None, None, None, "fewer than two usable groups")
    try:
        statistic, p_value = stats.levene(*cleaned, center="median")
    except Exception as exc:  # pragma: no cover
        return AssumptionCheck("variance_homogeneity", None, None, None, f"test failed: {exc}")
    return AssumptionCheck(
        "variance_homogeneity",
        float(statistic),
        float(p_value),
        bool(p_value > alpha),
        "Brown-Forsythe Levene; H0 = equal variances",
    )


def choose_two_sample_test(
    normal_a: bool | None, normal_b: bool | None, equal_variance: bool | None
) -> str:
    """Deterministic two-sample test decision table."""
    if normal_a and normal_b:
        return "student_t" if equal_variance else "welch_t"
    return "mann_whitney_u"


def choose_k_sample_test(normal_all: bool | None, equal_variance: bool | None) -> str:
    """Deterministic k-sample test decision table."""
    if normal_all:
        return "one_way_anova" if equal_variance else "welch_anova"
    return "kruskal_wallis"


# --------------------------------------------------------------------------- #
# omnibus tests
# --------------------------------------------------------------------------- #
def welch_anova(groups: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    """Welch's ANOVA for unequal variances (implemented here, not imported).

    Returns ``(F, df_between, df_within, p)``.

    .. math::
        F=\\frac{\\sum_i w_i(\\bar x_i-\\bar x)^2/(k-1)}
                {1+\\frac{2(k-2)}{k^2-1}\\sum_i\\frac{(1-w_i/W)^2}{n_i-1}},
        \\quad w_i=n_i/s_i^2
    """
    cleaned = [np.asarray(g, dtype=float) for g in groups]
    cleaned = [g[np.isfinite(g)] for g in cleaned]
    if any(g.size < 2 for g in cleaned):
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    k = len(cleaned)
    if k < 2:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    n_i = np.array([g.size for g in cleaned], dtype=float)
    mean_i = np.array([g.mean() for g in cleaned], dtype=float)
    var_i = np.array([g.var(ddof=1) for g in cleaned], dtype=float)
    if np.any(var_i <= 0):
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    w_i = n_i / var_i
    total_w = w_i.sum()
    grand_mean = float(np.sum(w_i * mean_i) / total_w)
    numerator = float(np.sum(w_i * (mean_i - grand_mean) ** 2) / (k - 1))
    term = np.sum((1.0 - w_i / total_w) ** 2 / (n_i - 1.0))
    denominator = 1.0 + (2.0 * (k - 2) / (k * k - 1.0)) * term
    f_statistic = numerator / denominator if denominator > 0 else float("nan")
    df_between = float(k - 1)
    if term <= 0:
        df_within = float("nan")
    else:
        df_within = float(1.0 / ((3.0 / (k * k - 1.0)) * term))
    if not np.isfinite(f_statistic) or not np.isfinite(df_within):
        return (f_statistic, df_between, df_within, float("nan"))
    p_value = float(stats.f.sf(f_statistic, df_between, df_within))
    return (float(f_statistic), df_between, df_within, p_value)


def two_sample_test(
    a: Sequence[float],
    b: Sequence[float],
    label_a: str = "A",
    label_b: str = "B",
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """Run the appropriate two-sample test, chosen by the decision table."""
    x = np.asarray(a, dtype=float)
    x = x[np.isfinite(x)]
    y = np.asarray(b, dtype=float)
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return TestResult(
            "insufficient_data", float("nan"), float("nan"), n=int(x.size + y.size),
            group_sizes={label_a: int(x.size), label_b: int(y.size)},
            notes="at least two observations per group are required",
        )
    normal_x = check_normality(x, alpha, label_a).satisfied
    normal_y = check_normality(y, alpha, label_b).satisfied
    equal_var = check_variance_homogeneity([x, y], alpha).satisfied
    chosen = choose_two_sample_test(normal_x, normal_y, equal_var)

    if chosen == "student_t":
        statistic, p_value = stats.ttest_ind(x, y, equal_var=True, alternative=alternative)
        effect = hedges_g(x, y)
        df = (float(x.size + y.size - 2),)
    elif chosen == "welch_t":
        statistic, p_value = stats.ttest_ind(x, y, equal_var=False, alternative=alternative)
        effect = hedges_g(x, y)
        var_x, var_y = x.var(ddof=1) / x.size, y.var(ddof=1) / y.size
        df = ((var_x + var_y) ** 2 / (var_x**2 / (x.size - 1) + var_y**2 / (y.size - 1)),)
    else:
        alternative_map = {"two-sided": "two-sided", "greater": "greater", "less": "less"}
        statistic, p_value = stats.mannwhitneyu(
            x, y, alternative=alternative_map.get(alternative, "two-sided")
        )
        direction = 1.0 if alternative != "less" else -1.0
        effect = rank_biserial(float(statistic), x.size, y.size, direction=direction)
        effect.ci_low, effect.ci_high = None, None
        df = ()
    return TestResult(
        chosen,
        float(statistic),
        float(p_value),
        df=df,
        n=int(x.size + y.size),
        group_sizes={label_a: int(x.size), label_b: int(y.size)},
        effect_size=effect,
        notes=f"chosen by decision table (normality {label_a}={normal_x}, {label_b}={normal_y}, equal_var={equal_var})",
    )


def k_sample_test(
    groups: dict[str, Sequence[float]],
    alpha: float = 0.05,
) -> tuple[TestResult, list[AssumptionCheck]]:
    """Run the appropriate k-sample omnibus test, chosen by the decision table."""
    labels = list(groups)
    arrays = {label: np.asarray(groups[label], dtype=float) for label in labels}
    arrays = {label: values[np.isfinite(values)] for label, values in arrays.items()}
    usable = {label: values for label, values in arrays.items() if values.size >= 2}
    checks: list[AssumptionCheck] = []
    if len(usable) < 2:
        return (
            TestResult(
                "insufficient_data",
                float("nan"),
                float("nan"),
                n=int(sum(v.size for v in arrays.values())),
                group_sizes={label: int(v.size) for label, v in arrays.items()},
                notes="fewer than two groups with at least two observations",
            ),
            checks,
        )

    pooled = np.concatenate(list(usable.values()))
    if pooled.size and float(np.ptp(pooled)) == 0.0:
        # SciPy's rank tests raise on fully tied data; a constant feature carries
        # no information, so it is reported as untestable rather than crashing.
        checks.append(
            AssumptionCheck("variance", None, None, False, "all values are identical across groups")
        )
        return (
            TestResult(
                "constant_data",
                float("nan"),
                float("nan"),
                n=int(pooled.size),
                group_sizes={label: int(v.size) for label, v in arrays.items()},
                notes="every observation is identical; no test is defined",
            ),
            checks,
        )

    normality_results = {label: check_normality(values, alpha, label) for label, values in usable.items()}
    checks.extend(normality_results.values())
    normal_all = all(result.satisfied for result in normality_results.values())
    homogeneity = check_variance_homogeneity(list(usable.values()), alpha)
    checks.append(homogeneity)
    chosen = choose_k_sample_test(normal_all, homogeneity.satisfied)
    values = list(usable.values())
    sizes = {label: int(v.size) for label, v in usable.items()}
    n_total = int(sum(sizes.values()))

    if chosen == "one_way_anova":
        statistic, p_value = stats.f_oneway(*values)
        df_between = float(len(values) - 1)
        df_within = float(n_total - len(values))
        effect = _effect_from_f(float(statistic), df_between, df_within)
        test = TestResult(
            chosen, float(statistic), float(p_value), df=(df_between, df_within),
            n=n_total, group_sizes=sizes, effect_size=effect,
        )
    elif chosen == "welch_anova":
        statistic, df_between, df_within, p_value = welch_anova(values)
        effect = _effect_from_f(float(statistic), df_between, df_within)
        test = TestResult(
            chosen, float(statistic), float(p_value), df=(df_between, df_within),
            n=n_total, group_sizes=sizes, effect_size=effect,
        )
    else:
        statistic, p_value = stats.kruskal(*values)
        effect = EffectSize(
            "epsilon_squared",
            epsilon_squared_from_h(float(statistic), n_total),
            interpretation="",
        )
        effect_eta = eta_squared_from_h(float(statistic), n_total, len(values))
        effect.notes = f"rank-based; eta-squared_H={effect_eta:.4f}"
        test = TestResult(
            chosen, float(statistic), float(p_value), df=(float(len(values) - 1),),
            n=n_total, group_sizes=sizes, effect_size=effect,
        )
    test.notes = f"chosen by decision table (all normal={normal_all}, equal_var={homogeneity.satisfied})"
    return test, checks


def _effect_from_f(f: float, df_between: float, df_within: float) -> EffectSize:
    eta = eta_squared_from_f(f, df_between, df_within)
    from .effect_sizes import interpret_effect

    return EffectSize(
        "eta_squared",
        eta,
        interpretation=interpret_effect(eta, "eta_squared"),
        notes="omega-squared is reported alongside in the results JSON where relevant",
    )


# --------------------------------------------------------------------------- #
# post-hoc
# --------------------------------------------------------------------------- #
def dunn_posthoc(
    groups: dict[str, Sequence[float]],
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> list[TestResult]:
    """Dunn's pairwise post-hoc test following a Kruskal-Wallis omnibus test.

    Ties are handled with the standard tie correction ``T = sum(t^3 - t)``.
    """
    labels = [label for label in groups if np.isfinite(np.asarray(groups[label], dtype=float)).sum() >= 2]
    if len(labels) < 2:
        return []
    arrays = {label: np.asarray(groups[label], dtype=float) for label in labels}
    arrays = {label: values[np.isfinite(values)] for label, values in arrays.items()}
    combined = np.concatenate([arrays[label] for label in labels])
    n_total = combined.size
    ranks = stats.rankdata(combined, method="average")
    offset = 0
    mean_ranks: dict[str, float] = {}
    for label in labels:
        size = arrays[label].size
        mean_ranks[label] = float(ranks[offset : offset + size].mean())
        offset += size
    _, counts = np.unique(combined, return_counts=True)
    ties = float(np.sum(counts**3 - counts))
    sigma2 = (n_total * (n_total + 1) / 12.0) - ties / (12.0 * (n_total - 1))

    pair_labels: list[str] = []
    raw_p: list[float] = []
    z_scores: list[float] = []
    sizes: list[tuple[int, int]] = []
    for a, b in combinations(labels, 2):
        n_a, n_b = arrays[a].size, arrays[b].size
        denominator = np.sqrt(sigma2 * (1.0 / n_a + 1.0 / n_b))
        if denominator <= 0:
            z = float("nan")
            p = float("nan")
        else:
            z = float((mean_ranks[a] - mean_ranks[b]) / denominator)
            p = float(2.0 * stats.norm.sf(abs(z)))
        pair_labels.append(f"{a}_vs_{b}")
        z_scores.append(z)
        raw_p.append(p)
        sizes.append((n_a, n_b))

    correction = correct_pvalues(raw_p, pair_labels, method=method, alpha=alpha)
    results: list[TestResult] = []
    for index, label in enumerate(pair_labels):
        n_a, n_b = sizes[index]
        delta = cliffs_delta(arrays[label.split("_vs_")[0]], arrays[label.split("_vs_")[1]])
        results.append(
            TestResult(
                "dunn",
                z_scores[index],
                raw_p[index],
                n=n_a + n_b,
                group_sizes={label.split("_vs_")[0]: n_a, label.split("_vs_")[1]: n_b},
                effect_size=delta,
                p_adjusted=correction.p_adjusted[index],
                notes=f"Dunn z = {z_scores[index]:.4f}; mean ranks "
                f"{label.split('_vs_')[0]}={mean_ranks[label.split('_vs_')[0]]:.1f}, "
                f"{label.split('_vs_')[1]}={mean_ranks[label.split('_vs_')[1]]:.1f}",
            )
        )
    return results


def paired_posthoc(
    matrix: dict[str, Sequence[float]],
    method: str = "holm",
    alpha: float = 0.05,
) -> list[TestResult]:
    """Pairwise Wilcoxon signed-rank tests on aligned (paired) subject vectors."""
    labels = list(matrix)
    pair_labels: list[str] = []
    raw_p: list[float] = []
    payload: list[tuple[float, int, list[float], str, str]] = []
    for a, b in combinations(labels, 2):
        x = np.asarray(matrix[a], dtype=float)
        y = np.asarray(matrix[b], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size < 5 or np.allclose(x, y):
            continue
        try:
            statistic, p_value = stats.wilcoxon(x, y, zero_method="wilcox")
        except Exception:  # pragma: no cover - degenerate sample
            continue
        pair_labels.append(f"{a}_vs_{b}")
        raw_p.append(float(p_value))
        payload.append((float(statistic), int(x.size), list(x - y), a, b))
    if not pair_labels:
        return []
    correction = correct_pvalues(raw_p, pair_labels, method=method, alpha=alpha)
    results: list[TestResult] = []
    for index, label in enumerate(pair_labels):
        statistic, n_pairs, differences, first, second = payload[index]
        diffs = np.asarray(differences, dtype=float)
        nz = diffs[diffs != 0]
        if nz.size == 0:
            effect = EffectSize("rank_biserial", float("nan"), notes="all differences zero")
        else:
            positive = float(np.sum(stats.rankdata(np.abs(nz))[nz > 0]))
            total = float(np.sum(stats.rankdata(np.abs(nz))))
            effect = EffectSize(
                "rank_biserial",
                2.0 * positive / total - 1.0 if total else float("nan"),
                notes="matched-pairs rank-biserial",
            )
        results.append(
            TestResult(
                "wilcoxon_signed_rank",
                statistic,
                raw_p[index],
                n=n_pairs,
                group_sizes={first: n_pairs, second: n_pairs},
                effect_size=effect,
                p_adjusted=correction.p_adjusted[index],
                notes="paired (same subjects contribute both stages)",
            )
        )
    return results


# --------------------------------------------------------------------------- #
# permutation test for clustered data
# --------------------------------------------------------------------------- #
def clustered_permutation_test(
    table: pd.DataFrame,
    feature: str,
    stages: Sequence[str],
    statistic: str = "kruskal_h",
    iterations: int = 1000,
    seed: int = 20240517,
) -> TestResult:
    """Permutation test that permutes *stage labels within each subject*.

    Shuffling whole subjects' label vectors preserves the within-subject
    dependence structure, so the null distribution reflects the real design
    instead of pretending every epoch is independent.

    The inner loop is pure NumPy: rebuilding a DataFrame per iteration made the
    test ~20x slower for no benefit.
    """
    subset = table.loc[table["stage"].isin(stages), ["subject_id", "stage", feature]].copy()
    subset = subset.replace([np.inf, -np.inf], np.nan).dropna(subset=[feature])
    if subset.empty:
        return TestResult("permutation", float("nan"), float("nan"), notes="no data")
    stage_index = {stage: index for index, stage in enumerate(stages)}
    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    for _, group in subset.groupby("subject_id", observed=True):
        values = group[feature].to_numpy(dtype=float)
        codes = np.array([stage_index[s] for s in group["stage"]], dtype=np.intp)
        if values.size >= 2:
            blocks.append((values, codes))
    if not blocks:
        return TestResult("permutation", float("nan"), float("nan"), notes="no subjects")

    observed = _kruskal_h_blocks(blocks)
    if not np.isfinite(observed):
        return TestResult("permutation", float("nan"), float("nan"), notes="statistic undefined")
    rng = np.random.default_rng(seed)
    exceed = 0
    counted = 0
    for _ in range(iterations):
        blocks_shuffled = [(values, rng.permutation(codes)) for values, codes in blocks]
        statistic_value = _kruskal_h_blocks(blocks_shuffled)
        if np.isfinite(statistic_value):
            counted += 1
            exceed += int(statistic_value >= observed)
    p_value = float((exceed + 1) / (counted + 1)) if counted else float("nan")
    return TestResult(
        f"permutation_{statistic}",
        observed,
        p_value,
        n=int(len(subset)),
        group_sizes={str(k): int(v) for k, v in subset["stage"].value_counts().items()},
        notes=f"cluster permutation ({iterations} iterations, stage labels permuted within subject)",
    )


def _kruskal_h_blocks(blocks: Sequence[tuple[np.ndarray, np.ndarray]]) -> float:
    """Kruskal-Wallis H on flat (values, stage codes), with tie correction.

    ``H = (12 / (N(N+1))) * sum_i R_i^2/n_i - 3(N+1)`` divided by the tie
    correction ``1 - T/(N^3 - N)``, where ``T = sum(t^3 - t)``.
    """
    if not blocks:
        return float("nan")
    values = np.concatenate([block[0] for block in blocks])
    codes = np.concatenate([block[1] for block in blocks])
    if values.size == 0:
        return float("nan")
    k = int(codes.max()) + 1 if codes.size else 0
    n_i = np.bincount(codes, minlength=k).astype(float)
    if k < 2 or np.sum(n_i >= 2) < 2:
        return float("nan")
    ranks = _average_ranks(values.astype(float))
    rank_sums = np.bincount(codes, weights=ranks, minlength=k)
    n_total = float(values.size)
    h = 12.0 / (n_total * (n_total + 1.0)) * np.sum(rank_sums**2 / np.maximum(n_i, 1.0)) - 3.0 * (n_total + 1.0)
    _, counts = np.unique(values, return_counts=True)
    ties = float(np.sum(counts**3 - counts))
    correction = 1.0 - ties / (n_total**3 - n_total) if n_total > 1 else 1.0
    if correction <= 0:
        return float("nan")
    return float(h / correction)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Midrank (average ranks with ties), matching scipy's ``rankdata``."""
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


# --------------------------------------------------------------------------- #
# top-level comparison
# --------------------------------------------------------------------------- #
def compare_feature(
    table: pd.DataFrame,
    feature: str,
    config: Any,
    stages: Sequence[str] | None = None,
) -> ComparisonResult:
    """Run the full comparison of one feature across sleep stages.

    Returns both the epoch-level (pseudo-replicated, reported for completeness)
    and the subject-level (primary) analysis, the chosen post-hoc for the
    primary analysis, and every assumption check that drove the choices.
    """
    from .descriptive import describe_series, stage_order, subject_stage_means

    statistics_config = config.analysis.statistics
    alpha = float(statistics_config.alpha)
    min_group = int(statistics_config.min_group_size)
    order = list(stages) if stages is not None else stage_order(table)
    feature_values = table[feature].astype(float)
    finite_values = feature_values[np.isfinite(feature_values)].to_numpy(dtype=float)
    present = [
        stage
        for stage in order
        if int(np.isfinite(feature_values[table["stage"] == stage]).sum()) >= min_group
    ]
    result = ComparisonResult(
        feature=feature, stages=present, unit_of_analysis="subject", assumptions=[]
    )
    if finite_values.size and float(np.ptp(finite_values)) == 0.0:
        result.skipped = True
        result.skip_reason = "feature is constant; there is no variance to compare across stages"
        return result
    if len(present) < 2:
        result.skipped = True
        result.skip_reason = f"fewer than two stages with >= {min_group} finite epochs"
        return result

    epoch_groups = {
        stage: feature_values[table["stage"] == stage].to_numpy(dtype=float) for stage in present
    }
    epoch_test, epoch_checks = k_sample_test(epoch_groups, alpha)
    epoch_test.notes = (
        epoch_test.notes
        + " | EPOCH LEVEL: pseudo-replicated, epochs within a subject are not independent"
    )
    result.epoch_level = epoch_test
    result.assumptions.extend(epoch_checks)
    result.group_summary = {
        stage: describe_series(epoch_groups[stage]) for stage in present
    }

    aggregates = subject_stage_means(table, feature, min_epochs=1, stages=present)
    if aggregates.empty:
        result.skipped = True
        result.skip_reason = "no subject-level aggregates available"
        return result

    stage_vectors: dict[str, list[float]] = {}
    for stage in present:
        rows = aggregates.loc[aggregates["stage"] == stage]
        if len(rows) >= 3:
            stage_vectors[stage] = list(rows["mean"].to_numpy(dtype=float))
    if len(stage_vectors) < 2:
        result.skipped = True
        result.skip_reason = "fewer than two stages have >= 3 subject aggregates"
        return result

    result.stages = list(stage_vectors)
    complete = _complete_block_subjects(aggregates, list(stage_vectors))
    if len(complete) >= 4 and statistics_config.use_subject_aggregation:
        matrix = {
            stage: aggregates.loc[
                (aggregates["stage"] == stage) & (aggregates["subject_id"].isin(complete)), "mean"
            ].to_numpy(dtype=float)
            for stage in stage_vectors
        }
        ordered = {stage: matrix[stage] for stage in sorted(matrix) if matrix[stage].size == len(complete)}
        if len(ordered) >= 2:
            statistic, p_value = stats.friedmanchisquare(*ordered.values())
            kendall_w = float(statistic / (len(complete) * (len(ordered) - 1))) if len(complete) else float("nan")
            primary = TestResult(
                "friedman",
                float(statistic),
                float(p_value),
                df=(float(len(ordered) - 1),),
                n=int(len(complete) * len(ordered)),
                group_sizes={stage: int(len(complete)) for stage in ordered},
                effect_size=EffectSize(
                    "kendall_w",
                    kendall_w,
                    notes="Kendall's W = chi2 / (n_subjects * (k - 1))",
                ),
                notes=(
                    "PRIMARY: repeated-measures (Friedman) on subject x stage means; "
                    f"{len(complete)} subjects with a complete block"
                ),
            )
            result.unit_of_analysis = "subject (repeated measures, complete blocks)"
            result.posthoc = paired_posthoc(
                {stage: list(ordered[stage]) for stage in ordered},
                method="holm",
                alpha=alpha,
            )
        else:
            primary, checks = k_sample_test(stage_vectors, alpha)
            primary.notes = "PRIMARY: unpaired test on subject x stage means; " + primary.notes
            result.assumptions.extend(checks)
            result.unit_of_analysis = "subject (unpaired aggregates)"
            result.posthoc = dunn_posthoc(stage_vectors, method="holm", alpha=alpha)
    else:
        primary, checks = k_sample_test(stage_vectors, alpha)
        primary.notes = (
            "PRIMARY: unpaired test on subject x stage means"
            + (f"; only {len(complete)} complete blocks" if complete else "; design not balanced")
            + " | "
            + primary.notes
        )
        result.assumptions.extend(checks)
        result.unit_of_analysis = "subject (unpaired aggregates)"
        result.posthoc = dunn_posthoc(stage_vectors, method="holm", alpha=alpha)
    result.primary = primary

    permutation = clustered_permutation_test(
        table,
        feature,
        present,
        iterations=min(1000, int(statistics_config.permutation_iterations)),
        seed=int(statistics_config.random_state),
    )
    result.permutation = permutation
    return result


def _complete_block_subjects(aggregates: pd.DataFrame, stages: Sequence[str]) -> list[str]:
    """Subjects that contribute every stage (needed for a paired design)."""
    if not stages:
        return []
    pivot = aggregates.pivot_table(
        index="subject_id", columns="stage", values="mean", aggfunc="mean"
    )
    available = [stage for stage in stages if stage in pivot.columns]
    if len(available) < len(stages):
        return []
    mask = pivot[list(stages)].notna().all(axis=1)
    return list(pivot.index[mask])
