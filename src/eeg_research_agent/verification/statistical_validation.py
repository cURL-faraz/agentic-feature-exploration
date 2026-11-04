"""Verification of the statistical machinery against known ground truth.

Statistical code that is never tested against a case with a known answer is a
liability.  Every check here builds synthetic data with a *known* property and
asserts that the implementation recovers it:

* the automatic two-sample test keeps its type-I error at the nominal level,
* a known mean difference yields the analytically expected Cohen's d,
* Benjamini-Hochberg reproduces the reference implementation and controls the FDR,
* Cohen's kappa is 1 for perfect agreement and ~0 for chance agreement,
* the clustered permutation test is calibrated under the null,
* the decision table routes non-normal data to rank-based tests,
* Dunn's z statistics match the closed-form expression.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "verify_null_calibration",
    "verify_effect_size_recovery",
    "verify_multiple_testing_control",
    "verify_kappa",
    "verify_permutation_calibration",
    "verify_assumption_gating",
    "verify_dunn_against_reference",
    "run_statistical_validation",
]


def _check(name: str, passed: bool, detail: str, observed: Any = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail, "observed": _plain(observed)}


def _plain(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else round(float(value), 6)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [None if not np.isfinite(v) else round(float(v), 6) for v in value.tolist()]
    return value


def verify_null_calibration(
    alpha: float = 0.05, n_simulations: int = 400, seed: int = 20240517
) -> dict[str, Any]:
    """Under a true null the p-value distribution must be uniform."""
    from ..statistics.hypothesis_tests import two_sample_test

    rng = np.random.default_rng(seed)
    p_values = np.empty(n_simulations)
    for index in range(n_simulations):
        a = rng.normal(0.0, 1.0, 40)
        b = rng.normal(0.0, 1.0, 40)
        p_values[index] = two_sample_test(a, b).p_value
    rejection_rate = float(np.mean(p_values <= alpha))
    ks = stats.kstest(p_values, "uniform")
    tolerance = 3.0 * np.sqrt(alpha * (1.0 - alpha) / n_simulations)
    within = abs(rejection_rate - alpha) <= tolerance
    return {
        "ok": bool(within and ks.pvalue > 0.001),
        "checks": [
            _check(
                "type_i_error_rate_near_alpha",
                within,
                f"empirical type-I error {rejection_rate:.3f} vs alpha={alpha} "
                f"(Monte-Carlo tolerance {tolerance:.3f} at 3 SE, n={n_simulations})",
                rejection_rate,
            ),
            _check(
                "p_values_uniform_under_null",
                bool(ks.pvalue > 0.001),
                f"Kolmogorov-Smirnov test against U(0,1) gives p={ks.pvalue:.4f}",
                {"ks_statistic": float(ks.statistic), "ks_pvalue": float(ks.pvalue)},
            ),
        ],
    }


def verify_effect_size_recovery(seed: int = 20240517) -> dict[str, Any]:
    """Hedges' g must recover a known standardised mean difference."""
    from ..statistics.effect_sizes import cohens_d, hedges_g, rank_biserial

    rng = np.random.default_rng(seed)
    delta = 0.8
    a = rng.normal(delta / 2, 1.0, 20000)
    b = rng.normal(-delta / 2, 1.0, 20000)
    d = cohens_d(a, b)
    g = hedges_g(a, b)
    expected_rank_biserial = 2.0 * stats.norm.cdf(delta / np.sqrt(2.0)) - 1.0
    u_statistic = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    r = rank_biserial(float(u_statistic), a.size, b.size)
    d_scaled = cohens_d(a * 10.0, b * 10.0)
    return {
        "ok": bool(
            abs(d.value - delta) < 0.05
            and abs(g.value - delta) < 0.05
            and abs(r.value - expected_rank_biserial) < 0.02
            and abs(d_scaled.value - d.value) < 1e-6
        ),
        "checks": [
            _check(
                "cohens_d_recovers_known_effect",
                abs(d.value - delta) < 0.05,
                f"Cohen's d={d.value:.4f} for a population effect of {delta}",
                d.value,
            ),
            _check(
                "hedges_g_recovers_known_effect",
                abs(g.value - delta) < 0.05,
                f"Hedges' g={g.value:.4f}",
                g.value,
            ),
            _check(
                "rank_biserial_matches_analytic_value",
                abs(r.value - expected_rank_biserial) < 0.02,
                f"rank-biserial={r.value:.4f} vs analytic 2*Phi(d/sqrt2)-1={expected_rank_biserial:.4f}",
                r.value,
            ),
            _check(
                "standardised_effect_is_scale_invariant",
                abs(d_scaled.value - d.value) < 1e-6,
                "multiplying both groups by 10 must not change Cohen's d",
                {"d": d.value, "d_scaled": d_scaled.value},
            ),
        ],
    }


def verify_multiple_testing_control(
    n_simulations: int = 400, seed: int = 20240517
) -> dict[str, Any]:
    """BH must match the reference implementation and control the FDR."""
    from scipy.stats import false_discovery_control

    from ..statistics.multiple_testing import benjamini_hochberg, bonferroni, correct_pvalues, holm

    rng = np.random.default_rng(seed)
    raw = rng.uniform(0, 1, 50)
    reference = false_discovery_control(raw, method="bh")
    bh = benjamini_hochberg(raw)
    monotone = bool(np.all(np.diff(benjamini_hochberg(np.sort(raw))) >= -1e-12))
    bounded = bool(np.all((bh >= raw - 1e-12) & (bh <= 1.0 + 1e-12)))

    n_tests = 100
    discoveries = 0
    for _ in range(n_simulations):
        p_values = rng.uniform(0, 1, n_tests)
        result = correct_pvalues(p_values, method="fdr_bh", alpha=0.05)
        # under the global null every rejection is a false discovery
        discoveries += int(sum(result.reject))
    fdr = discoveries / (n_tests * n_simulations)
    bonf = bonferroni(raw)
    holm_values = holm(raw)
    return {
        "ok": bool(
            np.allclose(bh, reference, atol=1e-12)
            and monotone
            and bounded
            and np.all(holm_values >= bonf - 1e-12)
            and fdr <= 0.05 + 3.0 * np.sqrt(0.05 * 0.95 / (n_tests * n_simulations))
        ),
        "checks": [
            _check(
                "benjamini_hochberg_matches_reference",
                np.allclose(bh, reference, atol=1e-12),
                "BH adjustment matches scipy's false_discovery_control(method='bh')",
                {"max_abs_difference": float(np.max(np.abs(bh - reference)))},
            ),
            _check("bh_monotone", monotone, "adjusted p-values must be monotone in the raw p-values"),
            _check(
                "adjusted_geq_raw",
                bounded,
                "adjusted p-values must never be smaller than the raw p-values",
            ),
            _check(
                "holm_geq_bonferroni",
                bool(np.all(holm_values >= bonf - 1e-12)),
                "Holm is uniformly less conservative than Bonferroni",
            ),
            _check(
                "fdr_controlled_under_global_null",
                fdr <= 0.05 + 3.0 * np.sqrt(0.05 * 0.95 / (n_tests * n_simulations)),
                f"proportion of false discoveries {fdr:.5f} under the global null must stay near alpha=0.05",
                fdr,
            ),
        ],
    }


def verify_kappa() -> dict[str, Any]:
    """Cohen's kappa must be 1 for perfect agreement and ~0 for independence."""
    from ..evaluation.metrics import classification_report, cohens_kappa, confusion_matrix

    labels = ["W", "N1", "N2", "N3", "REM"]
    truth = np.repeat(labels, 40)
    perfect = cohens_kappa(confusion_matrix(truth, truth, labels))
    rng = np.random.default_rng(20240517)
    chance = cohens_kappa(confusion_matrix(truth, rng.choice(labels, truth.size), labels))
    degenerate = cohens_kappa(np.array([[10.0, 10.0], [10.0, 10.0]]))
    report = classification_report(truth, truth, labels)
    return {
        "ok": bool(
            abs(perfect - 1.0) < 1e-9
            and abs(chance) < 0.1
            and np.isfinite(degenerate)
            and report["balanced_accuracy"] > 0.999
        ),
        "checks": [
            _check("kappa_one_for_perfect_agreement", abs(perfect - 1.0) < 1e-9, f"kappa={perfect:.6f}", perfect),
            _check("kappa_near_zero_for_chance", abs(chance) < 0.1, f"kappa={chance:.6f} on random labels", chance),
            _check(
                "kappa_handles_degenerate_matrix",
                np.isfinite(degenerate),
                "an all-equal confusion matrix (p_o == p_e) must not produce NaN",
                degenerate,
            ),
            _check(
                "perfect_prediction_gives_full_metrics",
                report["balanced_accuracy"] > 0.999 and report["macro_f1"] > 0.999,
                "a perfect predictor must score 1.0 on balanced accuracy and macro F1",
                {
                    "balanced_accuracy": report["balanced_accuracy"],
                    "macro_f1": report["macro_f1"],
                },
            ),
        ],
    }


def verify_permutation_calibration(
    table: pd.DataFrame | None = None, n_simulations: int = 60, seed: int = 20240517
) -> dict[str, Any]:
    """The clustered permutation test must be calibrated under the null."""
    from ..statistics.hypothesis_tests import clustered_permutation_test

    rng = np.random.default_rng(seed)
    stages = ["W", "N1", "N2", "N3", "REM"]
    subjects = [f"s{index}" for index in range(8)]
    p_values: list[float] = []
    for _ in range(n_simulations):
        rows = []
        for subject in subjects:
            codes = rng.choice(stages, 60)
            values = rng.normal(0, 1, 60)
            rows.extend(
                {"subject_id": subject, "stage": stage, "value": value}
                for stage, value in zip(codes, values)
            )
        frame = pd.DataFrame(rows).rename(columns={"value": "signal"})
        p_values.append(
            clustered_permutation_test(
                frame, "signal", stages, iterations=200, seed=int(rng.integers(1 << 30))
            ).p_value
        )
    array = np.asarray([value for value in p_values if np.isfinite(value)])
    if array.size < 10:
        return {
            "ok": False,
            "checks": [
                _check("permutation_produced_results", False, "too few finite p-values", int(array.size))
            ],
        }
    type_i = float(np.mean(array <= 0.05))
    tolerance = 3.0 * np.sqrt(0.05 * 0.95 / array.size)
    return {
        "ok": bool(type_i <= 0.05 + tolerance),
        "checks": [
            _check(
                "clustered_permutation_type_i_error",
                type_i <= 0.05 + tolerance,
                f"empirical type-I error {type_i:.3f} over {array.size} null datasets "
                f"(tolerance {0.05 + tolerance:.3f} including Monte-Carlo noise)",
                type_i,
            )
        ],
        "n_simulations": int(array.size),
    }


def verify_assumption_gating() -> dict[str, Any]:
    """The decision table must route to rank-based tests for skewed data."""
    from ..statistics.hypothesis_tests import (
        check_normality,
        choose_k_sample_test,
        choose_two_sample_test,
        k_sample_test,
    )

    rng = np.random.default_rng(20240517)
    normal_check = check_normality(rng.normal(0, 1, 400))
    skewed_check = check_normality(rng.exponential(1.0, 400))
    route_two = choose_two_sample_test(False, True, False)
    route_k = choose_k_sample_test(False, True)
    skewed_test, _ = k_sample_test({f"g{i}": rng.exponential(1.0, 60) for i in range(4)})
    return {
        "ok": bool(
            normal_check.satisfied
            and not skewed_check.satisfied
            and route_two == "mann_whitney_u"
            and route_k == "kruskal_wallis"
            and skewed_test.test == "kruskal_wallis"
        ),
        "checks": [
            _check(
                "normality_accepts_gaussian_data",
                bool(normal_check.satisfied),
                f"D'Agostino-Pearson p={normal_check.p_value} on Gaussian data",
                normal_check.p_value,
            ),
            _check(
                "normality_rejects_skewed_data",
                not bool(skewed_check.satisfied),
                f"D'Agostino-Pearson p={skewed_check.p_value} on exponential data",
                skewed_check.p_value,
            ),
            _check(
                "decision_table_routes_non_normal_to_rank_test",
                route_two == "mann_whitney_u",
                f"non-normal, unequal variance -> {route_two}",
                route_two,
            ),
            _check(
                "decision_table_routes_k_sample",
                route_k == "kruskal_wallis",
                f"non-normal k-sample -> {route_k}",
                route_k,
            ),
            _check(
                "end_to_end_route_on_skewed_groups",
                skewed_test.test == "kruskal_wallis",
                f"automatic selection chose {skewed_test.test}",
                skewed_test.test,
            ),
        ],
    }


def verify_dunn_against_reference() -> dict[str, Any]:
    """Dunn's z statistics must match the closed-form expression."""
    from scipy.stats import rankdata

    from ..statistics.hypothesis_tests import dunn_posthoc

    rng = np.random.default_rng(20240517)
    groups = {
        "g0": rng.normal(0, 1, 40),
        "g1": rng.normal(0.8, 1, 45),
        "g2": rng.normal(-0.4, 1, 50),
    }
    results = dunn_posthoc(groups, method="none")
    labels = list(groups)
    combined = np.concatenate([groups[label] for label in labels])
    ranks = rankdata(combined, method="average")
    offsets = np.cumsum([0, *[groups[label].size for label in labels]])
    mean_ranks = {
        label: float(ranks[offsets[index] : offsets[index + 1]].mean())
        for index, label in enumerate(labels)
    }
    n_total = combined.size
    _, counts = np.unique(combined, return_counts=True)
    ties = float(np.sum(counts**3 - counts))
    sigma2 = n_total * (n_total + 1) / 12.0 - ties / (12.0 * (n_total - 1))
    observed: dict[str, float] = {}
    for item in results:
        pair = list(item.group_sizes)
        if len(pair) == 2:
            observed[f"{pair[0]}_vs_{pair[1]}"] = item.statistic
    matches = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            denominator = np.sqrt(sigma2 * (1.0 / groups[a].size + 1.0 / groups[b].size))
            expected = (mean_ranks[a] - mean_ranks[b]) / denominator
            matches[f"{a}_vs_{b}"] = abs(observed.get(f"{a}_vs_{b}", float("nan")) - expected) < 1e-9
    return {
        "ok": bool(matches and all(matches.values())),
        "checks": [
            _check(
                "dunn_z_matches_closed_form",
                all(matches.values()),
                "Dunn's z statistics reproduce the analytic expression with tie correction",
                matches,
            ),
            _check(
                "dunn_returns_all_pairs",
                len(results) == 3,
                "three groups must produce three pairwise comparisons",
                len(results),
            ),
        ],
    }


def run_statistical_validation(table: pd.DataFrame | None = None) -> dict[str, Any]:
    """Run every statistical verification."""
    sections = {
        "null_calibration": verify_null_calibration(),
        "effect_size_recovery": verify_effect_size_recovery(),
        "multiple_testing_control": verify_multiple_testing_control(),
        "kappa": verify_kappa(),
        "permutation_calibration": verify_permutation_calibration(table=table),
        "assumption_gating": verify_assumption_gating(),
        "dunn_reference": verify_dunn_against_reference(),
    }
    checks = [check for section in sections.values() for check in section.get("checks", [])]
    return {
        "ok": all(section.get("ok", True) for section in sections.values()),
        "checks": checks,
        "sections": sections,
        "failed_checks": [check["name"] for check in checks if not check["passed"]],
    }
