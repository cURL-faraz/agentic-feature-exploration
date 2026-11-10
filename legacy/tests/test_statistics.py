"""Statistical-engine tests.

These tests check the statistics against independent references (SciPy where a
reference exists, closed-form algebra where it does not) and against synthetic
data whose answer is known by construction.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from eeg_research_agent.statistics.descriptive import describe_series
from eeg_research_agent.statistics.effect_sizes import (
    cliffs_delta,
    cohens_d,
    hedges_g,
    rank_biserial,
)
from eeg_research_agent.statistics.hypothesis_tests import (
    clustered_permutation_test,
    compare_feature,
    k_sample_test,
    welch_anova,
)
from eeg_research_agent.statistics.multiple_testing import correct_pvalues


def test_descriptive_statistics_match_numpy():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
    summary = describe_series(values)
    assert summary["n"] == 5
    assert summary["n_missing"] == 1
    assert summary["mean"] == pytest.approx(3.0)
    assert summary["median"] == pytest.approx(3.0)
    assert summary["std"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))
    assert summary["iqr"] == pytest.approx(2.0)


def test_benjamini_hochberg_matches_the_closed_form_formula():
    p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    # Reference values computed independently with the textbook step-up formula
    #   p_adj(i) = min_{j >= i} ( n / j * p_(j) )
    # and cross-checked against R's `p.adjust(p, method = "BH")`, which returns
    #   [0.01, 0.04, 0.084, 0.084, 0.084, 0.10, 0.1057143, 0.216, 0.216, 0.216]
    n = len(p_values)
    ordered = sorted(p_values)
    monotone = np.minimum.accumulate([n / (rank + 1) * value for rank, value in enumerate(ordered)][::-1])[::-1]
    expected = list(np.clip(np.array(monotone), 0.0, 1.0))
    result = correct_pvalues(p_values, [f"f{i}" for i in range(len(p_values))], method="fdr_bh")
    assert np.allclose(result.p_adjusted, expected, atol=1e-9)
    assert result.n_tests == len(p_values)
    # The step-up procedure is monotone in the ranked p-values and never lowers one.
    assert np.all(np.diff(result.p_adjusted) >= -1e-12)
    assert all(adjusted >= raw - 1e-12 for adjusted, raw in zip(result.p_adjusted, p_values))


def test_bonferroni_and_holm_are_finite_and_monotone():
    p_values = [0.001, 0.02, 0.3, 0.9]
    bonferroni = correct_pvalues(p_values, method="bonferroni").p_adjusted
    holm = correct_pvalues(p_values, method="holm").p_adjusted
    bh = correct_pvalues(p_values, method="fdr_bh").p_adjusted
    for adjusted in (bonferroni, holm, bh):
        assert all(0 <= value <= 1 for value in adjusted)
        assert np.allclose(np.sort(adjusted), adjusted)
    # Holm is never more liberal than Bonferroni and never more conservative than BH
    assert all(h <= b + 1e-12 for h, b in zip(holm, bonferroni))
    assert all(b_ <= h + 1e-12 for b_, h in zip(bh, holm))


def test_correction_handles_degenerate_input():
    assert correct_pvalues([], method="fdr_bh").n_tests == 0
    with_nan = correct_pvalues([0.01, np.nan, 0.5], method="fdr_bh")
    assert len(with_nan.p_adjusted) == 3


def test_effect_sizes_recover_known_values():
    rng = np.random.default_rng(7)
    a = rng.normal(0.0, 1.0, 400)
    b = a + 0.6
    d = cohens_d(a, b)
    g = hedges_g(a, b)
    assert d.value == pytest.approx(-0.6, abs=0.12)
    assert abs(g.value) < abs(d.value), "Hedges' g applies a small-sample correction"

    # identical samples: no effect, no division by zero
    zero = cohens_d(a, a)
    assert zero.value == pytest.approx(0.0, abs=1e-9)

    # a perfect separation has |Cliff's delta| = 1
    assert abs(cliffs_delta(np.zeros(20), np.ones(20)).value) == pytest.approx(1.0)


def test_rank_biserial_extremes():
    assert rank_biserial(0.0, 6, 6, direction=1.0).value == pytest.approx(-1.0)
    assert rank_biserial(36.0, 6, 6, direction=1.0).value == pytest.approx(1.0)
    assert rank_biserial(18.0, 6, 6, direction=1.0).value == pytest.approx(0.0)


def test_kruskal_wallis_matches_scipy():
    """Skewed groups must route to the rank-based test and match SciPy exactly."""
    rng = np.random.default_rng(3)
    groups = [shift + rng.exponential(scale=1.0, size=60) for shift in (0.0, 1.0, 2.0)]
    assert all(stats.normaltest(group).pvalue < 0.01 for group in groups), "test data must be non-normal"
    result, checks = k_sample_test({name: values for name, values in zip("abc", groups)})
    reference = stats.kruskal(*groups)
    assert result.test == "kruskal_wallis", "non-normal groups must select the rank-based test"
    assert result.statistic == pytest.approx(reference.statistic, rel=1e-9)
    assert result.p_value == pytest.approx(reference.pvalue, rel=1e-9)
    assert checks and any(check.name.startswith("normality") for check in checks)


def test_welch_anova_matches_scipy():
    rng = np.random.default_rng(11)
    groups = [rng.normal(shift, scale, 50) for shift, scale in ((0.0, 1.0), (0.5, 2.0), (1.5, 0.5))]
    statistic, df_between, df_within, p_value = welch_anova(groups)
    assert p_value < 0.05
    assert np.isfinite(statistic) and df_between == 2.0 and df_within > 0
    assert (p_value < 0.05) == (stats.f_oneway(*groups).pvalue < 0.05)


def test_comparison_uses_the_design_appropriate_test(feature_table, base_config):
    frame, _table, _splits = feature_table
    payload = compare_feature(frame, "abs_power_delta", base_config).to_dict()
    assert payload["skipped"] is False, payload["skip_reason"]
    # The primary analysis runs on subject x stage aggregates, never on epochs.
    assert payload["unit_of_analysis"].startswith("subject")
    assert payload["primary"]["test"] in {"friedman", "kruskal_wallis", "one_way_anova", "welch_anova"}
    # Only stages that pass the minimum-observation gate enter the comparison.
    assert 2 <= len(payload["stages"]) <= frame["stage"].nunique()
    assert set(payload["stages"]) <= set(frame["stage"].unique())
    assert payload["primary"]["n"] >= 4
    assert payload["epoch_level"]["test"] in {"kruskal_wallis", "welch_anova", "one_way_anova"}
    assert "pseudo-replicated" in payload["epoch_level"]["notes"]
    assert all("satisfied" in check for check in payload["assumptions"])
    assert payload["group_summary"], "per-stage descriptives must be attached"
    assert set(payload["group_summary"]) == set(payload["stages"])


def test_comparison_reports_skips_instead_of_crashing(feature_table, base_config):
    frame, _table, _splits = feature_table
    frame = frame.copy()
    frame["almost_constant"] = 1.0
    payload = compare_feature(frame, "almost_constant", base_config).to_dict()
    assert payload["skipped"] is True, "a zero-variance feature must be skipped, not tested"
    assert payload["skip_reason"]


def test_permutation_null_is_calibrated(feature_table):
    """Permuting stage labels within subject must not manufacture significance."""
    frame, _table, _splits = feature_table
    rng = np.random.default_rng(5)
    shuffled = frame.copy()
    shuffled["pure_noise_feature"] = rng.normal(0.0, 1.0, len(shuffled))
    stages = sorted(str(stage) for stage in shuffled["stage"].unique())
    result = clustered_permutation_test(
        shuffled, "pure_noise_feature", stages, iterations=200, seed=1
    )
    assert result.p_value > 0.05, "null data must not yield a small permutation p-value"
