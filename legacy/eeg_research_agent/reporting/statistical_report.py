"""``reports/statistical_analysis.md``.

Every number in this document is read from ``artifacts/tables/statistical_results.json``
and registered as a claim with an exact locator, so the verification stage can
prove the prose matches the computation.

The report deliberately separates:

* the **primary** repeated-measures analysis (subject x stage means - the correct
  unit when epochs are clustered inside subjects), from
* the **epoch-level** analysis (pseudo-replicated; reported for transparency only).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .context import ReportContext
from .markdown_utils import format_value, heading, render_key_values, render_list, render_table

__all__ = ["write_statistical_report"]

SOURCE = "tables/statistical_results.json"


def _comparison_rows(comparisons: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for item in comparisons:
        primary = item.get("primary") or {}
        epoch = item.get("epoch_level") or {}
        effect = primary.get("effect_size") or {}
        rows.append(
            {
                "feature": item.get("feature"),
                "n": primary.get("n"),
                "primary test": primary.get("test"),
                "statistic": primary.get("statistic"),
                "p (raw)": primary.get("p_value"),
                "p (FDR)": primary.get("p_adjusted"),
                "significant": primary.get("significant_after_correction"),
                "effect": effect.get("measure"),
                "effect value": effect.get("value"),
                "epoch-level test": epoch.get("test"),
                "epoch-level p": epoch.get("p_value"),
            }
        )
    rows.sort(
        key=lambda row: (
            row["p (FDR)"] is None,
            row["p (FDR)"] if row["p (FDR)"] is not None else 1.0,
        )
    )
    return rows[:limit] if limit else rows


def _assumption_rows(item: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "feature": item.get("feature"),
            "check": check.get("name"),
            "statistic": check.get("statistic"),
            "p": check.get("p_value"),
            "assumption met": check.get("satisfied"),
            "note": check.get("detail"),
        }
        for check in (item.get("assumptions") or [])
    ]


def _posthoc_rows(comparisons: list[dict[str, Any]], limit: int = 25) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in comparisons:
        for entry in item.get("posthoc") or []:
            groups = list((entry.get("group_sizes") or {}).keys())
            rows.append(
                {
                    "feature": item.get("feature"),
                    "stages": " vs ".join(groups),
                    "test": entry.get("test"),
                    "statistic": entry.get("statistic"),
                    "p": entry.get("p_value"),
                    "p (FDR)": entry.get("p_adjusted"),
                    "effect": (entry.get("effect_size") or {}).get("measure"),
                    "effect value": (entry.get("effect_size") or {}).get("value"),
                }
            )
    rows.sort(key=lambda row: (row["p (FDR)"] is None, row["p (FDR)"] or 1.0))
    return rows[:limit]


def _group_summary_tables(comparisons: list[dict[str, Any]], limit: int = 6) -> list[str]:
    blocks: list[str] = []
    for item in comparisons[:limit]:
        summary = item.get("group_summary") or {}
        if not summary:
            continue
        rows = [
            {
                "stage": stage,
                "n": values.get("n"),
                "missing": values.get("missing_proportion"),
                "mean": values.get("mean"),
                "median": values.get("median"),
                "std": values.get("std"),
                "IQR": values.get("iqr"),
                "min": values.get("min"),
                "max": values.get("max"),
            }
            for stage, values in summary.items()
        ]
        blocks.append(heading(4, str(item.get("feature"))))
        blocks.append(render_table(rows, columns=["stage", "n", "missing", "mean", "median", "std", "IQR", "min", "max"]))
    return blocks


def write_statistical_report(context: ReportContext) -> str:
    stats = context.statistical
    comparisons = list(stats.get("comparisons") or [])
    alpha = stats.get("alpha")
    correction = stats.get("correction")
    family = stats.get("family_correction") or {}
    selection = stats.get("feature_selection") or {}
    associations = list(stats.get("associations") or [])

    n_epochs = stats.get("n_epochs")
    n_subjects = stats.get("n_subjects")
    n_features = stats.get("n_features")
    n_tested = family.get("n_tests")
    n_rejected = family.get("n_rejected")
    for claim_id, description, value, locator, unit in (
        ("stats.n_epochs", "epochs entering the statistical analysis", n_epochs, "n_epochs", "epochs"),
        ("stats.n_subjects", "subjects entering the statistical analysis", n_subjects, "n_subjects", "subjects"),
        ("stats.n_features", "features tested", n_features, "n_features", "features"),
        ("stats.n_tests", "tests entering the multiple-comparison correction", n_tested, "family_correction.n_tests", "tests"),
        ("stats.n_rejected", "tests rejected after correction", n_rejected, "family_correction.n_rejected", "tests"),
    ):
        if value is not None:
            context.add_claim(claim_id, description, value, locator, SOURCE, unit=unit)

    lines: list[str] = [
        heading(1, "Statistical Analysis of EEG Features Across Sleep Stages"),
        "",
        context.origin_caveat(),
        "",
        heading(2, "1. Design and decision rules"),
        render_key_values(
            {
                "epochs": n_epochs,
                "subjects": n_subjects,
                "features tested": n_features,
                "unit of analysis (primary)": stats.get("unit_of_analysis"),
                "family-wise correction": correction,
                "alpha": alpha,
                "stages": stats.get("stages"),
                "stage counts": stats.get("stage_counts"),
                "minimum group size": (stats.get("descriptive") or {}).get("min_group_size"),
            }
        ),
        "",
        "Test selection is deterministic, not model-chosen: normality is screened with "
        "D'Agostino-Pearson (Shapiro-Wilk for small samples), variance homogeneity with "
        "Levene, and the resulting decision table selects the test. With repeated measures "
        "across stages within the same subject, the primary analysis is Friedman with "
        "paired Wilcoxon post-hoc comparisons; the epoch-level Kruskal-Wallis result is "
        "reported only to show the effect of pseudo-replication.",
        "",
        heading(2, "2. Multiple-comparison correction"),
        render_key_values(
            {
                "method": family.get("method"),
                "tests": n_tested,
                "rejected": n_rejected,
                "alpha": family.get("alpha"),
            }
        ),
        "",
        "Both raw and adjusted p-values are stored in `artifacts/tables/feature_stage_comparisons.csv`. "
        "Adjusted values are computed across the family of tested features, so the adjusted p-value "
        "of a single feature depends on how many features entered this run.",
        "",
        heading(2, "3. Stage-wise descriptive statistics"),
    ]
    if comparisons:
        lines += _group_summary_tables(comparisons, limit=6)
    else:
        lines.append("_No comparisons were produced._")
    lines += [
        "",
        heading(2, "4. Inferential comparisons"),
        render_table(
            _comparison_rows(comparisons, limit=25),
            columns=[
                "feature",
                "n",
                "primary test",
                "statistic",
                "p (raw)",
                "p (FDR)",
                "significant",
                "effect",
                "effect value",
                "epoch-level p",
            ],
        ),
        "",
    ]
    skipped = [item for item in comparisons if item.get("skipped")]
    if skipped:
        lines += [
            heading(3, "Comparisons that could not be run"),
            render_table(
                [{"feature": item.get("feature"), "reason": item.get("skip_reason")} for item in skipped]
            ),
            "",
        ]

    lines += [heading(2, "5. Assumption checks"), ""]
    assumption_rows: list[dict[str, Any]] = []
    for item in comparisons[:8]:
        assumption_rows.extend(_assumption_rows(item))
    if assumption_rows:
        lines.append(render_table(assumption_rows, columns=["feature", "check", "statistic", "p", "assumption met", "note"]))
    else:
        lines.append("_No assumption checks were recorded._")
    lines += [
        "",
        heading(2, "6. Post-hoc pairwise comparisons"),
        render_table(
            _posthoc_rows(comparisons),
            columns=["feature", "stages", "test", "statistic", "p", "p (FDR)", "effect", "effect value"],
        ),
        "",
        heading(2, "7. Order-free association measures"),
        "",
        "Multiclass labels are never treated as a continuous variable. Reported here are "
        "Kruskal-Wallis epsilon-squared and mutual information (nats); both are invariant to "
        "how the stage labels happen to be ordered.",
        "",
        render_table(
            [
                {
                    "feature": entry.get("feature"),
                    "unit": entry.get("unit"),
                    "n subjects": entry.get("n_subjects"),
                    "epsilon^2": (entry.get("kruskal_epsilon_squared") or {}).get("value"),
                    "MI (nats)": (entry.get("mutual_information_nats") or {}).get("value"),
                }
                for entry in associations[:25]
            ],
            columns=["feature", "unit", "n subjects", "epsilon^2", "MI (nats)"],
        ),
        "",
        heading(2, "8. Feature selection and interactions (training subjects only)"),
    ]
    if selection.get("skipped"):
        lines.append(f"Feature selection was skipped: {selection.get('reason')}")
    elif selection:
        mi = selection.get("mutual_information") or {}
        top_mi = sorted(mi.items(), key=lambda item: item[1], reverse=True)[:15]
        lines += [
            render_key_values(
                {
                    "training rows": selection.get("n_train_rows"),
                    "training subjects": selection.get("n_train_subjects"),
                    "candidate features": len(selection.get("candidate_features") or []),
                    "redundancy clusters": (selection.get("redundancy") or {}).get("n_clusters"),
                    "features dropped as redundant": (selection.get("redundancy") or {}).get("dropped"),
                    "RFE selection": selection.get("rfe"),
                }
            ),
            "",
            heading(3, "Top mutual-information features"),
            render_table([{"feature": name, "MI (nats)": value} for name, value in top_mi]),
            "",
        ]
        interactions = list(selection.get("interactions") or [])
        lines += [heading(3, "Interaction candidates (exploratory)")]
        if interactions:
            lines.append(
                render_table(
                    [
                        {
                            "pair": entry.get("interaction"),
                            "role": entry.get("role"),
                            "best parent MI": entry.get("parent_mi_max"),
                            "CV MI gain": entry.get("cv_mi_gain"),
                            "passes screen": entry.get("passes_heuristic_screen"),
                        }
                        for entry in interactions[:15]
                    ],
                    columns=["pair", "role", "best parent MI", "CV MI gain", "passes screen"],
                )
            )
            lines.append(
                "\nInteraction search is bounded to a prespecified candidate list and is scored by "
                "cross-validated mutual-information gain; any surviving pair is labelled "
                "*exploratory* until it is confirmed on held-out subjects in the predictive stage."
            )
        else:
            lines.append("_No interaction candidates were generated._")
        lines.append("")
    else:
        lines.append("_Feature selection was not run (no training rows supplied)._")

    if context.figures:
        lines += [heading(2, "9. Figures"), render_list([f"`{path}`" for path in sorted(context.figures.values())]), ""]
    lines += [
        heading(2, "10. Interpretation limits"),
        render_list(
            [
                "A significant association is not a causal marker and is not by itself evidence of predictive value.",
                "The epoch-level p-values are pseudo-replicated and are strongly anti-conservative; use the primary analysis.",
                "Monte-Carlo p-values (permutation, bootstrap) have a floor of 1/(n_iterations+1); a p-value at that floor means `at least this small`.",
                "Effect sizes are reported with the test statistic because p-values depend on sample size.",
                "Statistical results here describe this corpus "
                + ("(synthetic)" if context.synthetic else "(the selected public subjects)")
                + "; they do not transfer automatically.",
            ]
        ),
        "",
    ]
    return "\n".join(lines).strip() + "\n"
