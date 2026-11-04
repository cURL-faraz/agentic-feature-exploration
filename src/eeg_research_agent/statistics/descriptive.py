"""Descriptive statistics at both the epoch and the subject level.

The subject-level aggregation is not optional decoration: epochs within a subject
are strongly correlated, so the subject is the effective unit of analysis for any
inferential claim.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "describe_series",
    "describe_feature_by_stage",
    "subject_stage_means",
    "describe_table",
    "stage_order",
]


def stage_order(table: pd.DataFrame) -> list[str]:
    """Canonical stage order for tables and plots (W, N1, N2, N3, REM)."""
    canonical = ["W", "N1", "N2", "N3", "REM"]
    present = set(table["stage"].dropna().unique())
    ordered = [stage for stage in canonical if stage in present]
    ordered += sorted(present - set(ordered))
    return ordered


def describe_series(values: Sequence[float]) -> dict[str, Any]:
    """Distribution summary of one numeric vector (NaN-aware)."""
    array = np.asarray(values, dtype=float).ravel()
    total = array.size
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "n": 0,
            "n_missing": int(total),
            "missing_proportion": 1.0 if total else float("nan"),
            "mean": None,
            "median": None,
            "std": None,
            "iqr": None,
            "q25": None,
            "q75": None,
            "min": None,
            "max": None,
            "skew": None,
            "kurtosis": None,
        }
    q25, q75 = np.percentile(finite, [25, 75])
    centered = finite - finite.mean()
    std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    if finite.size > 2 and std > 0:
        skew = float(np.mean(centered**3) / std**3)
        kurtosis = float(np.mean(centered**4) / std**4 - 3.0)
    else:
        skew = kurtosis = float("nan")
    return {
        "n": int(finite.size),
        "n_missing": int(total - finite.size),
        "missing_proportion": float(1.0 - finite.size / total) if total else float("nan"),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "std": std,
        "iqr": float(q75 - q25),
        "q25": float(q25),
        "q75": float(q75),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "skew": skew,
        "kurtosis": kurtosis,
    }


def describe_feature_by_stage(
    table: pd.DataFrame,
    feature: str,
    stages: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-stage distribution summary of one feature (epoch level)."""
    order = list(stages) if stages is not None else stage_order(table)
    result: dict[str, dict[str, Any]] = {}
    for stage in order:
        subset = table.loc[table["stage"] == stage, feature]
        result[stage] = describe_series(subset.to_numpy(dtype=float))
    return result


def subject_stage_means(
    table: pd.DataFrame,
    feature: str,
    min_epochs: int = 1,
    stages: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Mean feature value per (subject, stage) - the repeated-measures unit.

    Subjects without any epoch of a stage simply have no row for it; the caller
    decides whether that makes the design unbalanced.
    """
    order = list(stages) if stages is not None else stage_order(table)
    subset = table.loc[table["stage"].isin(order), ["subject_id", "stage", feature]].copy()
    subset = subset.replace([np.inf, -np.inf], np.nan).dropna(subset=[feature])
    if subset.empty:
        return pd.DataFrame(columns=["subject_id", "stage", "mean", "n_epochs"])
    grouped = (
        subset.groupby(["subject_id", "stage"], observed=True)[feature]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"count": "n_epochs"})
    )
    return grouped.loc[grouped["n_epochs"] >= max(1, min_epochs)].reset_index(drop=True)


def describe_table(
    table: pd.DataFrame,
    features: Sequence[str],
    min_group_size: int = 15,
) -> dict[str, Any]:
    """Descriptive statistics for many features, grouped by stage."""
    report: dict[str, Any] = {
        "n_epochs": int(len(table)),
        "n_subjects": int(table["subject_id"].nunique()),
        "stage_counts": {str(k): int(v) for k, v in table["stage"].value_counts().items()},
        "min_group_size": min_group_size,
        "features": {},
    }
    for feature in features:
        report["features"][feature] = {
            "epoch_level": describe_feature_by_stage(table, feature),
            "subject_level": _subject_level_summary(table, feature, min_group_size),
        }
    return report


def _subject_level_summary(table: pd.DataFrame, feature: str, min_group_size: int) -> dict[str, Any]:
    aggregates = subject_stage_means(table, feature, min_epochs=1)
    if aggregates.empty:
        return {}
    summary: dict[str, Any] = {}
    for stage, group in aggregates.groupby("stage", observed=True):
        if len(group) < max(3, min_group_size // 3):
            summary[str(stage)] = {"n_subjects": int(len(group)), "insufficient_for_inference": True}
            continue
        values = group["mean"].to_numpy(dtype=float)
        stats = describe_series(values)
        stats["n_subjects"] = int(len(group))
        stats["mean_epochs_per_subject"] = float(group["n_epochs"].mean())
        summary[str(stage)] = stats
    return summary
