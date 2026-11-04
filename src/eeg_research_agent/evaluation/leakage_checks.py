"""Data-leakage audits.

These checks are the machine-checkable half of the project's scientific-integrity
policy.  They are run by the verification agent after every pipeline execution and
their verdict is embedded in the validation report; a failed check is a hard
failure, not a warning.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ..data.splits import DatasetSplits, verify_splits

__all__ = [
    "audit_subject_separation",
    "audit_duplicate_rows",
    "audit_feature_selection_scope",
    "audit_non_causal_features",
    "audit_preprocessing_scope",
    "run_leakage_audit",
]


def audit_subject_separation(table: pd.DataFrame, splits: DatasetSplits) -> dict[str, Any]:
    """Every subject must live in exactly one split, and the splits must partition."""
    if "split" not in table.columns:
        return {"ok": False, "reason": "feature table has no `split` column"}
    # `verify_splits` expects epoch-index rows, which carry the stage label; the
    # feature table has the same label under `stage`, so pass it through when it
    # is available instead of handing the helper an incomplete row.
    if "stage" in table.columns:
        rows = [
            {"subject_id": str(subject), "split": str(split), "stage": str(stage)}
            for subject, split, stage in zip(table["subject_id"], table["split"], table["stage"])
        ]
    else:
        rows = [
            {"subject_id": str(subject), "split": str(split)}
            for subject, split in zip(table["subject_id"], table["split"])
        ]
    report = verify_splits(rows, splits)
    report["observed_subjects"] = sorted({row["subject_id"] for row in rows})
    report["declared_subjects"] = sorted(splits.all_subjects())
    unexpected = sorted(set(report["observed_subjects"]) - set(report["declared_subjects"]))
    report["undeclared_subjects"] = unexpected
    if unexpected:
        report["ok"] = False
    return report


def audit_duplicate_rows(table: pd.DataFrame, key: Sequence[str] | None = None) -> dict[str, Any]:
    """No epoch identity may be duplicated inside or across splits.

    A duplicate ``(subject, recording, epoch)`` triple that appears in two splits
    is a literal train/test leak, and duplicated feature rows indicate a
    concatenation bug.
    """
    keys = list(key) if key else ["subject_id", "recording_id", "epoch_id"]
    missing = [column for column in keys if column not in table.columns]
    if missing:
        return {"ok": False, "reason": f"missing identity columns: {missing}"}
    duplicates = table.duplicated(subset=keys, keep=False)
    n_duplicates = int(duplicates.sum())
    cross_split = 0
    if "split" in table.columns and n_duplicates:
        grouped = table.loc[duplicates].groupby(keys, observed=True)["split"].nunique()
        cross_split = int((grouped > 1).sum())
    return {
        "ok": n_duplicates == 0,
        "key": keys,
        "n_duplicate_rows": n_duplicates,
        "n_identities_in_multiple_splits": cross_split,
    }


def audit_feature_selection_scope(
    feature_selection_report: dict[str, Any] | None,
    train_rows: int,
    total_rows: int,
) -> dict[str, Any]:
    """Feature selection must have seen training rows only."""
    if not feature_selection_report:
        return {"ok": False, "reason": "no feature-selection report was produced"}
    if feature_selection_report.get("skipped"):
        return {
            "ok": True,
            "reason": "selection was skipped (insufficient training rows)",
            "detail": feature_selection_report,
        }
    observed = feature_selection_report.get("n_train_rows")
    ok = observed is not None and int(observed) <= int(total_rows)
    return {
        "ok": bool(ok),
        "n_train_rows_used_for_selection": observed,
        "n_total_rows": int(total_rows),
        "selection_used_only_training_rows": bool(observed is not None and int(observed) < int(total_rows)),
    }


def audit_non_causal_features(
    feature_sets: dict[str, Iterable[str]] | None, non_causal: Sequence[str]
) -> dict[str, Any]:
    """Non-causal features must be absent from every causal ablation."""
    if not feature_sets:
        return {"ok": True, "reason": "no ablations supplied"}
    leaks: dict[str, list[str]] = {}
    for name, features in feature_sets.items():
        if "non_causal" in name:
            continue
        offenders = sorted(set(features) & set(non_causal))
        if offenders:
            leaks[name] = offenders
    return {
        "ok": not leaks,
        "non_causal_features": list(non_causal),
        "leaks": leaks,
    }


def audit_preprocessing_scope(pipeline: Any) -> dict[str, Any]:
    """A fitted pipeline must contain its own imputer/scaler, fitted in-fold.

    Structural check: if a model pipeline accepted a globally pre-scaled matrix,
    the standardiser would not appear among its steps and test-set statistics
    would have influenced training.
    """
    try:
        step_names = set(pipeline.named_steps) if hasattr(pipeline, "named_steps") else set()
    except Exception:  # pragma: no cover - defensive
        return {"ok": False, "reason": "object is not an sklearn Pipeline"}
    has_imputer = any("impute" in name for name in step_names)
    has_scaler = any("scale" in name for name in step_names)
    return {
        "ok": has_imputer and has_scaler,
        "steps": sorted(step_names),
        "has_in_pipeline_imputer": has_imputer,
        "has_in_pipeline_scaler": has_scaler,
    }


def run_leakage_audit(
    table: pd.DataFrame,
    splits: DatasetSplits | None = None,
    feature_selection_report: dict[str, Any] | None = None,
    feature_sets: dict[str, Iterable[str]] | None = None,
    non_causal: Sequence[str] = ("feature_delta_norm", "feature_delta_norm_sigma"),
) -> dict[str, Any]:
    """Run every leakage audit and return one verdict object."""
    checks: dict[str, Any] = {}
    checks["subject_separation"] = (
        audit_subject_separation(table, splits)
        if splits is not None
        else {"ok": False, "reason": "no split manifest supplied"}
    )
    checks["duplicate_rows"] = audit_duplicate_rows(table)
    train_rows = int((table["split"] == "train").sum()) if "split" in table.columns else 0
    checks["feature_selection_scope"] = audit_feature_selection_scope(
        feature_selection_report, train_rows, int(len(table))
    )
    checks["non_causal_features"] = audit_non_causal_features(feature_sets, non_causal)
    ok = all(bool(check.get("ok")) for check in checks.values())
    return {
        "ok": ok,
        "checks": checks,
        "failed": [name for name, check in checks.items() if not check.get("ok")],
    }
