"""Report-generation tests.

The point of these tests is the *contract* between the reports and the saved
computation: every number a report prints is registered as a claim with an
artifact locator, and the claim validator must be able to re-read the artifact
and confirm the number.  A report that drifts from its artifacts fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

from eeg_research_agent.reporting import generate_all_reports
from eeg_research_agent.verification.research_validation import validate_claims


class _Artifacts:
    """Minimal stand-in for the pipeline's artifact bag."""

    def __init__(self, frame, definitions) -> None:
        self.feature_table = frame
        self.feature_definitions = definitions


def _statistical_results(subjects: int, epochs: int, features: int) -> dict:
    return {
        "alpha": 0.05,
        "correction": "fdr_bh",
        "stages": ["W", "N1", "N2"],
        "stage_counts": {"W": 10, "N1": 10, "N2": 20},
        "n_epochs": epochs,
        "n_subjects": subjects,
        "n_features": features,
        "unit_of_analysis": "subject x stage means",
        "descriptive": {
            "n_epochs": epochs,
            "n_subjects": subjects,
            "min_group_size": 5,
            "features": {},
        },
        "comparisons": [
            {
                "feature": "abs_power_delta",
                "stages": ["W", "N1", "N2"],
                "unit_of_analysis": "subject x stage means",
                "skipped": False,
                "skip_reason": "",
                "primary": {
                    "test": "friedman",
                    "statistic": 7.0,
                    "p_value": 0.03,
                    "p_adjusted": 0.03,
                    "df": [2.0],
                    "n": subjects * 3,
                    "group_sizes": {"W": subjects, "N1": subjects, "N2": subjects},
                    "effect_size": {"measure": "kendall_w", "value": 0.7, "interpretation": ""},
                    "significant_after_correction": True,
                    "notes": "primary",
                },
                "epoch_level": {
                    "test": "kruskal_wallis",
                    "statistic": 12.0,
                    "p_value": 0.001,
                    "group_sizes": {"W": 10, "N1": 10, "N2": 20},
                    "effect_size": None,
                    "notes": "pseudo-replicated",
                },
                "permutation": {
                    "test": "permutation_kruskal_h",
                    "statistic": 12.0,
                    "p_value": 0.01,
                    "group_sizes": {},
                    "notes": "cluster permutation",
                },
                "posthoc": [
                    {
                        "test": "wilcoxon_signed_rank",
                        "statistic": 0.0,
                        "p_value": 0.03,
                        "p_adjusted": 0.06,
                        "group_sizes": {"W": subjects, "N2": subjects},
                        "effect_size": {"measure": "rank_biserial", "value": -0.6},
                    }
                ],
                "assumptions": [
                    {
                        "name": "normality:W",
                        "statistic": 1.0,
                        "p_value": 0.01,
                        "satisfied": False,
                        "detail": "D'Agostino-Pearson",
                    }
                ],
                "group_summary": {
                    "W": {"n": 10, "missing_proportion": 0.0, "mean": 1.0, "median": 1.0, "std": 0.1, "iqr": 0.2, "min": 0.5, "max": 2.0},
                    "N2": {"n": 20, "missing_proportion": 0.0, "mean": 3.0, "median": 3.0, "std": 0.4, "iqr": 0.5, "min": 2.0, "max": 4.0},
                },
            }
        ],
        "family_correction": {"method": "fdr_bh", "alpha": 0.05, "n_tests": 1, "n_rejected": 1},
        "associations": [
            {
                "feature": "abs_power_delta",
                "unit": "subject_stage_mean",
                "n_subjects": subjects,
                "kruskal_epsilon_squared": {"value": 0.8},
                "mutual_information_nats": {"value": 0.9},
            }
        ],
        "feature_selection": {
            "n_train_rows": epochs // 2,
            "n_train_subjects": max(1, subjects // 2),
            "candidate_features": ["abs_power_delta"],
            "mutual_information": {"abs_power_delta": 0.9},
            "top_k_mutual_information": ["abs_power_delta"],
            "redundancy": {"threshold": 0.9, "n_clusters": 1, "clusters": [["abs_power_delta"]], "dropped": []},
            "rfe": ["abs_power_delta"],
            "interaction_candidates": ["interaction::a*b"],
            "interactions": [
                {
                    "interaction": "interaction::a*b",
                    "role": "exploratory",
                    "parent_mi_max": 0.9,
                    "cv_mi_gain": 0.01,
                    "passes_heuristic_screen": False,
                }
            ],
        },
    }


def _evaluation() -> dict:
    return {
        "results": [
            {
                "feature_set": "all_features",
                "classifier": "logistic_regression",
                "split": "test",
                "n_features": 5,
                "accuracy": 0.6,
                "balanced_accuracy": 0.55,
                "macro_f1": 0.5,
                "cohens_kappa": 0.3,
                "per_class": {"N1": {"f1": 0.2}, "N2": {"f1": 0.6}, "REM": {"f1": 0.4}},
                "confusion_matrix": [[2, 1, 0], [1, 3, 1], [0, 1, 4]],
                "labels": ["W", "N1", "N2"],
            }
        ],
        "errors": [],
        "artifacts": {},
    }


def _write_inputs(config, feature_table, registry) -> dict:
    tables = config.path("tables")
    tables.mkdir(parents=True, exist_ok=True)
    frame, _table, _splits = feature_table
    statistical = _statistical_results(
        subjects=int(frame["subject_id"].nunique()),
        epochs=int(len(frame)),
        features=5,
    )
    (tables / "statistical_results.json").write_text(json.dumps(statistical), encoding="utf-8")
    (tables / "predictive_evaluation.json").write_text(json.dumps(_evaluation()), encoding="utf-8")
    metadata = config.path("metadata")
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "dataset_summary.json").write_text(
        json.dumps(
            {
                "dataset": "synthetic-psg",
                "data_origin": "synthetic",
                "selected_channel": "EEG Fpz-Cz",
                "sampling_frequency": 100.0,
                "epoch_duration": 30.0,
                "subject_count": int(frame["subject_id"].nunique()),
                "epoch_count": int(len(frame)),
                "available_stages": sorted(frame["stage"].unique().tolist()),
                "class_distribution": frame["stage"].value_counts().to_dict(),
                "quality_report": {"epochs_total": int(len(frame)), "epochs_excluded": 0},
            }
        ),
        encoding="utf-8",
    )
    return statistical


def test_reports_are_written_and_their_claims_verify(base_config, feature_table, registry):
    frame, _table, _splits = feature_table
    _write_inputs(base_config, feature_table, registry)
    artifacts = _Artifacts(frame, [definition.describe() for definition in registry])
    results = {
        "inspect": json.loads((base_config.path("metadata") / "dataset_summary.json").read_text()),
        "verify": {"ok": True, "n_checks": 42, "n_failed_checks": 0, "failed_checks": [], "sections": {}},
    }
    written = generate_all_reports(base_config, artifacts, results)

    for key in (
        "literature_review",
        "feature_catalog",
        "statistical_analysis",
        "validation_report",
        "final_research_report",
        "claims_json",
    ):
        path = Path(written[key])
        assert path.is_file(), key
        assert path.read_text(encoding="utf-8").strip(), key

    claims = json.loads(Path(written["claims_json"]).read_text())["claims"]
    assert claims, "the reports must register their numbers as claims"
    verification = validate_claims(claims, results, artifacts_dir=base_config.path("artifacts"))
    assert verification["ok"], verification["failures"]
    assert verification["n_verified"] == len(claims)

    # the report must state the provenance of a synthetic corpus prominently
    final = Path(written["final_research_report"]).read_text(encoding="utf-8")
    assert "synthetic" in final
    assert "must not be read as a statement about human sleep" in final


def test_corrupted_artifact_fails_claim_verification(base_config, feature_table, registry):
    frame, _table, _splits = feature_table
    _write_inputs(base_config, feature_table, registry)
    artifacts = _Artifacts(frame, [definition.describe() for definition in registry])
    results = {"verify": {"ok": True, "n_checks": 1, "n_failed_checks": 0, "sections": {}}}
    written = generate_all_reports(base_config, artifacts, results)

    statistical_path = base_config.path("tables") / "statistical_results.json"
    payload = json.loads(statistical_path.read_text())
    payload["n_epochs"] = payload["n_epochs"] + 1  # drift the artifact from the report
    statistical_path.write_text(json.dumps(payload))

    claims = json.loads(Path(written["claims_json"]).read_text())["claims"]
    verification = validate_claims(claims, results, artifacts_dir=base_config.path("artifacts"))
    assert not verification["ok"]
    assert any(failure["claim_id"] == "stats.n_epochs" for failure in verification["failures"])


def test_report_stage_degrades_gracefully_without_inputs(base_config):
    """With no statistical or predictive artifacts the reports must still be honest."""
    for name in ("statistical_results.json", "predictive_evaluation.json"):
        path = base_config.path("tables") / name
        if path.is_file():
            path.unlink()
    artifacts = _Artifacts(None, [])
    written = generate_all_reports(base_config, artifacts, {"verify": {}})
    final = Path(written["final_research_report"]).read_text(encoding="utf-8")
    assert "# Single-Channel EEG Feature Discovery for Sleep Staging" in final
    assert "_No rows._" in Path(written["statistical_analysis"]).read_text()
