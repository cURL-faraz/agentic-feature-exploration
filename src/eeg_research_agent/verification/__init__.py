"""Verification and scientific validation.

Pipeline stage: ``verify``.  Writes ``artifacts/metadata/verification_report.json``
and feeds ``reports/validation_report.md`` (via the reporting stage).

``synthetic_signals``      reference signals and feature-behaviour checks
``feature_validation``     mathematics: invertibility, determinism, stability
``statistical_validation`` statistics against known ground truth
``research_validation``    evidence integrity: claims, citations, novelty labels
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .feature_validation import run_feature_validation
from .research_validation import run_research_validation
from .statistical_validation import run_statistical_validation
from .synthetic_signals import run_synthetic_signal_checks

__all__ = [
    "run_verification_suite",
    "run_feature_validation",
    "run_statistical_validation",
    "run_research_validation",
    "run_synthetic_signal_checks",
    "run_ml_validation",
]

logger = get_logger(__name__)


def run_ml_validation(table: pd.DataFrame, config: Any) -> dict[str, Any]:
    """Machine-learning specific verification: splits, leakage, reproducibility."""
    from ..data.splits import DatasetSplits
    from ..evaluation.leakage_checks import (
        audit_duplicate_rows,
        audit_non_causal_features,
        audit_subject_separation,
    )

    checks: list[dict[str, Any]] = []
    if "split" not in table.columns:
        return {
            "ok": False,
            "checks": [
                {
                    "name": "split_column_present",
                    "passed": False,
                    "detail": "the feature table carries no `split` column",
                }
            ],
            "failed_checks": ["split_column_present"],
        }

    split_path = config.path("metadata") / "subject_splits.json"
    splits = DatasetSplits.load(split_path) if split_path.is_file() else None
    if splits is None:
        checks.append(
            {
                "name": "split_manifest_available",
                "passed": False,
                "detail": f"split manifest not found at {split_path}",
            }
        )
    else:
        separation = audit_subject_separation(table, splits)
        checks.append(
            {
                "name": "subject_separation",
                "passed": bool(separation.get("ok")),
                "detail": "no subject appears in more than one split",
                "observed": separation.get("overlaps"),
            }
        )
        checks.extend(_reproducibility_check(table, config))

    duplicates = audit_duplicate_rows(table)
    checks.append(
        {
            "name": "no_duplicate_epoch_identities",
            "passed": bool(duplicates.get("ok")),
            "detail": "epoch identities are unique and never span splits",
            "observed": duplicates,
        }
    )
    causal = [
        column
        for column in table.columns
        if column not in {"feature_delta_norm", "feature_delta_norm_sigma"}
    ]
    non_causal_audit = audit_non_causal_features(
        {"all_features": causal}, ("feature_delta_norm", "feature_delta_norm_sigma")
    )
    checks.append(
        {
            "name": "non_causal_features_excluded_from_causal_sets",
            "passed": bool(non_causal_audit.get("ok")),
            "detail": "sequence features stay out of the causal ablation",
            "observed": non_causal_audit.get("leaks"),
        }
    )
    return {
        "ok": all(check["passed"] for check in checks),
        "checks": checks,
        "failed_checks": [check["name"] for check in checks if not check["passed"]],
    }


def _reproducibility_check(table: pd.DataFrame, config: Any) -> list[dict[str, Any]]:
    """Train the same model twice and confirm identical predictions."""
    from ..eeg.feature_extraction import feature_columns
    from ..evaluation.classification import available_classifiers, build_pipeline

    columns = [c for c in feature_columns(table) if not c.startswith("feature_delta_norm")]
    train = table.loc[table["split"] == "train"]
    test = table.loc[table["split"] == "test"]
    if len(train) < 20 or len(test) == 0 or train["stage"].nunique() < 2 or not columns:
        return [
            {
                "name": "reproducible_predictions",
                "passed": False,
                "detail": "not enough rows to re-fit a model twice",
            }
        ]
    spec = available_classifiers(["random_forest"], seed=int(config.seed))[0]
    first = build_pipeline(spec, seed=int(config.seed))
    second = build_pipeline(spec, seed=int(config.seed))
    first.fit(train.loc[:, columns], train["stage"])
    second.fit(train.loc[:, columns], train["stage"])
    same = bool(
        np.array_equal(
            first.predict(test.loc[:, columns]), second.predict(test.loc[:, columns])
        )
    )
    return [
        {
            "name": "reproducible_predictions",
            "passed": same,
            "detail": "two identically seeded models must produce identical predictions",
            "observed": {"identical_predictions": same, "n_test": int(len(test))},
        }
    ]


def run_verification_suite(
    table: pd.DataFrame,
    config: Any,
    registry: Any | None = None,
    splits: Any = None,
    feature_selection_report: dict[str, Any] | None = None,
    feature_sets: dict[str, Any] | None = None,
    claims: Sequence[Any] = (),
    papers: Sequence[dict[str, Any]] = (),
    report_paths: Sequence[str | Path] = (),
    report_text: str = "",
    results: dict[str, Any] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Run every verification suite and persist one combined verdict."""
    import json as _json

    if registry is None:
        from ..eeg.feature_registry import load_default_registry

        registry = load_default_registry()

    signal_checks_raw = run_synthetic_signal_checks(registry, config)
    evidence_raw = run_research_validation(
        claims=claims,
        results=results,
        papers=papers,
        report_paths=report_paths,
        report_text=report_text,
        artifacts_dir=config.path("artifacts"),
    )
    sections = {
        "synthetic_signals": {
            "ok": all(check.passed for check in signal_checks_raw),
            "checks": [check.to_dict() for check in signal_checks_raw],
            "failed_checks": [check.name for check in signal_checks_raw if not check.passed],
        },
        "feature_mathematics": run_feature_validation(registry, config),
        "statistics": run_statistical_validation(table=table),
        "machine_learning": run_ml_validation(table, config),
        "leakage": _leakage_audit(table, splits, feature_selection_report, feature_sets),
        # The evidence report nests its own per-check verdicts; flatten it into the
        # same shape as every other section so the aggregation below stays uniform.
        "evidence": {
            "ok": bool(evidence_raw.get("ok")),
            "checks": [
                {"name": name, "passed": bool(check.get("ok")), "detail": check}
                for name, check in (evidence_raw.get("checks") or {}).items()
            ],
            "failed_checks": list(evidence_raw.get("failed_checks") or []),
        },
    }
    failed = sorted({name for name, section in sections.items() if not section.get("ok", True)})
    report = {
        "ok": not failed,
        "failed_checks": failed,
        "sections": sections,
        "n_checks": sum(len(section.get("checks", [])) for section in sections.values()),
        "n_failed_checks": sum(
            1
            for section in sections.values()
            for check in section.get("checks", [])
            if not check.get("passed", True)
        ),
        "data_origin": (
            str(table["data_origin"].iloc[0]) if "data_origin" in table.columns else "unknown"
        ),
        "stage_level_failures": {
            name: section.get("failed_checks", []) for name, section in sections.items()
        },
    }
    if write:
        path = config.path("metadata") / "verification_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            _json.dump(report, handle, indent=2, default=str)
        report["written"] = str(path)
    return report


def _leakage_audit(
    table: pd.DataFrame,
    splits: Any,
    feature_selection_report: dict[str, Any] | None,
    feature_sets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from ..evaluation.leakage_checks import run_leakage_audit

    if splits is None:
        logger.warning("no splits object supplied to the leakage audit")
        return {
            "ok": False,
            "failed_checks": ["split_manifest"],
            "checks": [
                {
                    "name": "split_manifest",
                    "passed": False,
                    "detail": "the split manifest was not supplied to the verification suite",
                }
            ],
        }
    audit = run_leakage_audit(
        table,
        splits,
        feature_selection_report=feature_selection_report,
        feature_sets=feature_sets,
    )
    checks = [
        {"name": name, "passed": bool(detail.get("ok")), "detail": detail}
        for name, detail in audit["checks"].items()
    ]
    return {
        "ok": all(check["passed"] for check in checks) and bool(audit["ok"]),
        "checks": checks,
        "failed_checks": [check["name"] for check in checks if not check["passed"]],
    }
