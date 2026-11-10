"""Orchestration of the statistical analysis stage.

The engine turns a feature table into one machine-readable result document.  All
numbers in the generated reports come from this document (or from its siblings),
which is what lets the verification agent check that no figure in a report was
invented.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..config import ProjectConfig
from ..logging_utils import get_logger
from .correlations import feature_stage_associations
from .descriptive import describe_table, stage_order
from .feature_selection import (
    anova_f_ranking,
    evaluate_interactions,
    interaction_candidates,
    l1_ranking,
    mutual_information_ranking,
    permutation_importance_scores,
    prune_redundant,
    random_forest_importance,
    recursive_feature_elimination,
    select_top_k,
)
from .hypothesis_tests import compare_feature
from .multiple_testing import correct_pvalues

__all__ = ["StatisticalAnalysis", "run_statistical_analysis", "stage_pairs"]

logger = get_logger(__name__)


def stage_pairs(stages: Sequence[str]) -> list[tuple[str, str]]:
    """All unordered stage pairs, ordered by the canonical stage order."""
    order = {stage: index for index, stage in enumerate(["W", "N1", "N2", "N3", "REM"])}
    ordered = sorted(set(stages), key=lambda stage: (order.get(stage, 99), stage))
    return [(ordered[i], ordered[j]) for i in range(len(ordered)) for j in range(i + 1, len(ordered))]


@dataclass
class StatisticalAnalysis:
    """Everything the statistical stage produced, plus where it was written."""

    results: dict[str, Any] = field(default_factory=dict)
    written: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        comparisons = self.results.get("comparisons", [])
        significant = [
            c["feature"]
            for c in comparisons
            if (c.get("primary") or {}).get("p_adjusted") is not None
            and c["primary"]["p_adjusted"] <= self.results["alpha"]
        ]
        return {
            "n_features_tested": len(comparisons),
            "n_significant_after_fdr": len(significant),
            "significant_features": significant[:25],
            "unit_of_analysis": self.results.get("unit_of_analysis"),
            "correction": self.results.get("correction"),
        }


def run_statistical_analysis(
    table: pd.DataFrame,
    config: ProjectConfig,
    features: Sequence[str] | None = None,
    train_mask: pd.Series | np.ndarray | None = None,
    write: bool = True,
) -> StatisticalAnalysis:
    """Run the complete descriptive + inferential + association analysis.

    Parameters
    ----------
    table:
        Feature table produced by :func:`..eeg.feature_extraction.extract_features`.
    features:
        Feature columns to test; defaults to every numeric, non-metadata column.
    train_mask:
        Rows that may be used for *feature selection*.  Inferential statistics are
        reported on all rows (that is what "association with stage" means) but
        selection/interaction search stays on training rows.
    """
    from ..eeg.feature_extraction import feature_columns

    statistics_config = config.analysis.statistics
    selection_config = config.analysis.features_selection
    alpha = float(statistics_config.alpha)
    correction = str(statistics_config.correction)
    seed = int(statistics_config.random_state)

    frame = table.copy()
    feature_list = list(features) if features is not None else feature_columns(frame)
    frame = frame.replace([np.inf, -np.inf], np.nan)

    analysis = StatisticalAnalysis()
    analysis.results["alpha"] = alpha
    analysis.results["correction"] = correction
    analysis.results["stages"] = stage_order(frame)
    analysis.results["stage_counts"] = {
        str(k): int(v) for k, v in frame["stage"].value_counts().items()
    }
    analysis.results["n_epochs"] = int(len(frame))
    analysis.results["n_subjects"] = int(frame["subject_id"].nunique())
    analysis.results["n_features"] = len(feature_list)
    analysis.results["unit_of_analysis"] = (
        "subject x stage means (repeated measures where the design allows)"
        if statistics_config.use_subject_aggregation
        else "epoch (pseudo-replicated)"
    )

    # ---- descriptive ------------------------------------------------------ #
    analysis.results["descriptive"] = describe_table(frame, feature_list, statistics_config.min_group_size)

    # ---- per-feature comparisons ------------------------------------------ #
    comparisons: list[dict[str, Any]] = []
    for feature in feature_list:
        try:
            result = compare_feature(frame, feature, config)
        except Exception as exc:  # noqa: BLE001 - one feature must not abort the stage
            analysis.errors.append(f"{feature}: {type(exc).__name__}: {exc}")
            logger.warning("comparison failed for %s: %s", feature, exc)
            continue
        payload = result.to_dict()
        comparisons.append(payload)

    primary_p = [
        (item["primary"] or {}).get("p_value")
        if item.get("primary") and not item.get("skipped")
        else None
        for item in comparisons
    ]
    finite_index = [i for i, value in enumerate(primary_p) if value is not None and np.isfinite(value)]
    if finite_index:
        family_correction = correct_pvalues(
            [primary_p[i] for i in finite_index],
            [comparisons[i]["feature"] for i in finite_index],
            method=correction,  # type: ignore[arg-type]
            alpha=alpha,
        )
        for position, index in enumerate(finite_index):
            comparisons[index]["primary"]["p_adjusted"] = float(family_correction.p_adjusted[position])
            comparisons[index]["primary"]["significant_after_correction"] = bool(
                family_correction.reject[position]
            )
    analysis.results["comparisons"] = comparisons
    analysis.results["family_correction"] = {
        "method": correction,
        "alpha": alpha,
        "n_tests": len(finite_index),
        "n_rejected": int(
            sum(1 for i in finite_index if comparisons[i].get("primary", {}).get("significant_after_correction"))
        ),
    }

    # ---- associations (order-free) ---------------------------------------- #
    try:
        analysis.results["associations"] = feature_stage_associations(
            frame,
            feature_list,
            min_group_size=max(5, statistics_config.min_group_size // 3),
            n_neighbors=int(selection_config.mutual_information_neighbors),
            seed=seed,
            aggregate_by_subject=bool(statistics_config.use_subject_aggregation),
        )
    except Exception as exc:  # noqa: BLE001
        analysis.errors.append(f"associations: {type(exc).__name__}: {exc}")
        logger.warning("association analysis failed: %s", exc)

    # ---- selection: train rows only --------------------------------------- #
    if train_mask is not None:
        mask = np.asarray(train_mask)
        train_frame = frame.loc[mask].copy() if mask.dtype == bool else frame.iloc[mask].copy()
    else:
        train_frame = frame.copy()
    if train_frame["stage"].nunique() < 2 or len(train_frame) < 20:
        analysis.results["feature_selection"] = {"skipped": True, "reason": "training rows insufficient"}
    else:
        analysis.results["feature_selection"] = _run_selection(
            train_frame, feature_list, config
        )

    if write:
        analysis.written = _write_artifacts(analysis, config)
    return analysis


def _run_selection(
    train_frame: pd.DataFrame, feature_list: Sequence[str], config: ProjectConfig
) -> dict[str, Any]:
    selection_config = config.analysis.features_selection
    hypotheses = config.analysis.hypotheses
    labels = train_frame["stage"].astype(str).to_numpy()
    frame = train_frame.loc[:, [f for f in feature_list if f in train_frame.columns]].astype(float)
    report: dict[str, Any] = {
        "n_train_rows": int(len(train_frame)),
        "n_train_subjects": int(train_frame["subject_id"].nunique()),
        "candidate_features": list(frame.columns),
    }
    try:
        information = mutual_information_ranking(
            frame, labels, seed=int(config.seed), n_neighbors=int(selection_config.mutual_information_neighbors)
        )
        report["mutual_information"] = information
        report["anova_f"] = anova_f_ranking(frame, labels)
        report["random_forest_importance"] = random_forest_importance(frame, labels, seed=int(config.seed))
        report["permutation_importance"] = permutation_importance_scores(
            frame, labels, seed=int(config.seed)
        )
        top_for_l1 = select_top_k(information, int(selection_config.max_features_for_multivariable))
        report["l1"] = l1_ranking(frame.loc[:, top_for_l1], labels, seed=int(config.seed)) if top_for_l1 else {}
        report["rfe"] = recursive_feature_elimination(
            frame.loc[:, top_for_l1], labels, n_features=min(20, max(2, len(top_for_l1) // 2)),
            step=float(selection_config.rfe_step), seed=int(config.seed),
        ) if top_for_l1 else []

        redundancy = prune_redundant(
            frame, list(frame.columns), threshold=float(selection_config.correlation_threshold),
            importance=information,
        )
        report["redundancy"] = redundancy.to_dict()
        report["top_k_mutual_information"] = select_top_k(information, int(selection_config.max_features_for_multivariable))

        seeds = report["top_k_mutual_information"][: max(2, min(6, len(report["top_k_mutual_information"])))]
        interactions, names = interaction_candidates(
            frame, seeds, max_pairs=int(hypotheses.interaction_pairs_max)
        )
        report["interaction_candidates"] = names
        report["interactions"] = evaluate_interactions(
            frame.join(interactions),
            labels,
            names,
            baseline=information,
            seed=int(config.seed),
            alpha=float(config.analysis.statistics.alpha),
            method=str(config.analysis.statistics.correction),
        )
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("feature-selection stage failed: %s", exc)
    return report


def _write_artifacts(analysis: StatisticalAnalysis, config: ProjectConfig) -> dict[str, str]:
    tables = config.path("tables")
    tables.mkdir(parents=True, exist_ok=True)
    metadata = config.path("metadata")
    metadata.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    json_path = tables / "statistical_results.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(analysis.results, handle, indent=2, default=_json_default)
    written["statistical_results_json"] = str(json_path)

    comparisons = analysis.results.get("comparisons", [])
    rows = []
    for item in comparisons:
        primary = item.get("primary") or {}
        epoch = item.get("epoch_level") or {}
        permutation = item.get("permutation") or {}
        effect = (primary.get("effect_size") or {})
        rows.append(
            {
                "feature": item["feature"],
                "stages_compared": ",".join(item.get("stages", [])),
                "unit_of_analysis": item.get("unit_of_analysis"),
                "skipped": item.get("skipped"),
                "skip_reason": item.get("skip_reason"),
                "primary_test": primary.get("test"),
                "primary_statistic": primary.get("statistic"),
                "primary_p_raw": primary.get("p_value"),
                "primary_p_adjusted": primary.get("p_adjusted"),
                "primary_significant": primary.get("significant_after_correction"),
                "effect_measure": effect.get("measure"),
                "effect_value": effect.get("value"),
                "effect_interpretation": effect.get("interpretation"),
                "epoch_level_test": epoch.get("test"),
                "epoch_level_p": epoch.get("p_value"),
                "permutation_p": permutation.get("p_value"),
                "n_subjects": None,
            }
        )
    comparison_frame = pd.DataFrame(rows)
    if not comparison_frame.empty:
        comparison_frame = comparison_frame.sort_values(
            "primary_p_adjusted", na_position="last", kind="mergesort"
        )
    csv_path = tables / "feature_stage_comparisons.csv"
    comparison_frame.to_csv(csv_path, index=False)
    written["comparisons_csv"] = str(csv_path)

    associations = analysis.results.get("associations", [])
    association_frame = pd.DataFrame(
        [
            {
                "feature": item["feature"],
                "unit": item["unit"],
                "n_subjects": item["n_subjects"],
                "kruskal_epsilon_squared": item["kruskal_epsilon_squared"]["value"],
                "kruskal_p": item["kruskal_epsilon_squared"]["p_value"],
                "mutual_information_nats": item["mutual_information_nats"]["value"],
            }
            for item in associations
        ]
    )
    association_path = tables / "feature_stage_associations.csv"
    association_frame.to_csv(association_path, index=False)
    written["associations_csv"] = str(association_path)

    posthoc_rows = []
    for item in comparisons:
        for pair in item.get("posthoc", []):
            groups = list((pair.get("group_sizes") or {}).keys())
            posthoc_rows.append(
                {
                    "feature": item["feature"],
                    "test": pair.get("test"),
                    "comparison": " vs ".join(groups) if len(groups) == 2 else "",
                    "statistic": pair.get("statistic"),
                    "p_raw": pair.get("p_value"),
                    "p_adjusted": pair.get("p_adjusted"),
                    "effect_measure": (pair.get("effect_size") or {}).get("measure"),
                    "effect_value": (pair.get("effect_size") or {}).get("value"),
                }
            )
    posthoc_path = tables / "posthoc_comparisons.csv"
    pd.DataFrame(posthoc_rows).to_csv(posthoc_path, index=False)
    written["posthoc_csv"] = str(posthoc_path)

    summary_path = metadata / "statistical_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary": analysis.summary(),
                "errors": analysis.errors,
                "artifacts": written,
            },
            handle,
            indent=2,
            default=_json_default,
        )
    written["summary_json"] = str(summary_path)
    return written


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)
