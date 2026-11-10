"""Subject-level split and leakage-audit tests.

The single most damaging error in EEG sleep staging is splitting *epochs* instead
of *subjects*: neighbouring 30 s epochs from one night are almost duplicates, so
an epoch-level split reports a test score that says nothing about a new subject.
These tests make that failure mode detectable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eeg_research_agent.data.splits import (
    assign_splits,
    build_epoch_index,
    make_subject_splits,
    verify_splits,
)
from eeg_research_agent.evaluation.leakage_checks import (
    audit_duplicate_rows,
    audit_feature_selection_scope,
    audit_non_causal_features,
    audit_subject_separation,
    run_leakage_audit,
)


def test_subjects_are_partitioned_across_splits():
    subjects = [f"S{i:02d}" for i in range(10)]
    splits = make_subject_splits(subjects, test_fraction=0.3, validation_fraction=0.2, seed=1)
    groups = [set(splits.train), set(splits.validation), set(splits.test)]
    assert set().union(*groups) == set(subjects), "every subject must be assigned"
    for index, first in enumerate(groups):
        for second in groups[index + 1 :]:
            assert not (first & second), "splits must not overlap"


def test_split_assignment_is_reproducible():
    subjects = [f"S{i:02d}" for i in range(12)]
    first = make_subject_splits(subjects, seed=99)
    second = make_subject_splits(subjects, seed=99)
    assert first.train == second.train
    assert first.test == second.test
    assert sorted(first.all_subjects()) == subjects
    assert first.split_for(subjects[0]) in {"train", "validation", "test"}


def test_verify_splits_passes_on_a_clean_manifest(epoch_sets):
    sets, _reports = epoch_sets
    rows = build_epoch_index(sets)
    splits = make_subject_splits(sorted({row["subject_id"] for row in rows}), seed=5)
    rows = assign_splits(rows, splits)
    report = verify_splits(rows, splits)
    assert report["ok"], report["checks"]
    assert report["overlaps"] == {}
    assert all(report["checks"].values())


def test_verify_splits_detects_a_leaking_subject(epoch_sets):
    """A subject appearing under two split labels is the leak the audit must catch."""
    sets, _reports = epoch_sets
    rows = build_epoch_index(sets)
    subjects = sorted({row["subject_id"] for row in rows})
    splits = make_subject_splits(subjects, seed=5)
    rows = assign_splits(rows, splits)
    victim = splits.test[0] if splits.test else subjects[0]
    # Deliberately leak: one epoch of a held-out subject is also filed under train.
    leaked = dict(next(row for row in rows if row["subject_id"] == victim))
    leaked["split"] = "train"
    rows.append(leaked)
    report = verify_splits(rows, splits)
    assert not report["ok"]
    assert victim in report["overlaps"]
    assert report["checks"]["no_subject_in_multiple_splits"] is False


def test_subject_separation_audit_on_the_feature_table(feature_table):
    frame, _table, splits = feature_table
    audit = audit_subject_separation(frame, splits)
    assert audit["ok"], audit
    assert audit["undeclared_subjects"] == []


def test_duplicate_identity_audit_detects_a_duplicated_epoch(feature_table):
    frame, _table, _splits = feature_table
    duplicated = pd.concat([frame, frame.head(3)], ignore_index=True)
    audit = audit_duplicate_rows(duplicated)
    assert not audit["ok"]
    assert audit["n_duplicate_rows"] >= 3


def test_feature_selection_scope_audit():
    assert audit_feature_selection_scope(None, 10, 100)["ok"] is False
    assert audit_feature_selection_scope({"skipped": True}, 10, 100)["ok"] is True
    good = audit_feature_selection_scope({"n_train_rows": 60}, 60, 100)
    assert good["ok"] and good["selection_used_only_training_rows"]
    assert audit_feature_selection_scope({}, 10, 100)["ok"] is False


def test_non_causal_features_are_rejected_from_causal_ablations():
    non_causal = ["feature_delta_norm"]
    clean = audit_non_causal_features({"all_features": ["abs_power_delta"]}, non_causal)
    assert clean["ok"]
    leaky = audit_non_causal_features({"all_features": ["feature_delta_norm"]}, non_causal)
    assert not leaky["ok"]
    assert "all_features" in leaky["leaks"]
    # an explicitly non-causal set is allowed to contain them
    allowed = audit_non_causal_features({"non_causal_only": ["feature_delta_norm"]}, non_causal)
    assert allowed["ok"]


def test_full_leakage_audit_over_the_real_split(feature_table):
    """The end-to-end audit passes when selection sees training rows and causal sets stay causal."""
    frame, _table, splits = feature_table
    from eeg_research_agent.eeg.feature_extraction import feature_columns
    from eeg_research_agent.statistics.feature_selection import mutual_information_ranking

    features = feature_columns(frame)
    causal = [name for name in features if "delta_norm" not in name]
    train = frame.loc[frame["split"] == "train"]
    subset = causal[:12]
    selection_report = {
        "n_train_rows": int(len(train)),
        "n_train_subjects": int(train["subject_id"].nunique()),
        "candidate_features": list(subset),
        "mutual_information": mutual_information_ranking(
            train.loc[:, subset].astype(float), train["stage"].astype(str).to_numpy(), seed=1
        ),
    }
    feature_sets = {"all_features": causal, "non_causal_only": ["feature_delta_norm"]}
    audit = run_leakage_audit(frame, splits, feature_selection_report=selection_report, feature_sets=feature_sets)
    assert audit["ok"], audit
    assert audit["failed"] == []
    assert audit["checks"]["feature_selection_scope"]["selection_used_only_training_rows"] is True

    # Control: smuggling a non-causal feature into a causal set must flip the verdict.
    leaky = run_leakage_audit(
        frame,
        splits,
        feature_selection_report=selection_report,
        feature_sets={"all_features": causal + ["feature_delta_norm"]},
    )
    assert not leaky["ok"]
    assert "non_causal_features" in leaky["failed"]


def test_random_epoch_split_would_have_leaked(feature_table):
    """A negative control: splitting by epoch does leak subject identity."""
    frame, _table, splits = feature_table
    shuffled = frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
    shuffled["split"] = np.where(np.arange(len(shuffled)) < len(shuffled) // 2, "train", "test")
    audit = audit_subject_separation(shuffled, splits)
    assert not audit["ok"], "an epoch-level random split must be detected as leaking"
