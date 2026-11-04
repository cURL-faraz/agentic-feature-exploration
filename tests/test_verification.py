"""Verification-suite tests, including the *negative* controls.

An integrity check that never fails is worthless.  So every claim/evidence check
here is exercised twice: once on honest input, and once on input that has been
deliberately corrupted, where the check must catch it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eeg_research_agent.verification.feature_validation import run_feature_validation
from eeg_research_agent.verification.research_validation import (
    Claim,
    validate_citations,
    validate_claims,
    validate_novelty_labels,
    validate_report_files,
)
from eeg_research_agent.verification.statistical_validation import run_statistical_validation
from eeg_research_agent.verification.synthetic_signals import run_synthetic_signal_checks


def test_synthetic_signal_checks_all_pass(registry, base_config):
    checks = run_synthetic_signal_checks(registry, base_config)
    assert len(checks) >= 20, "the synthetic suite should cover many behaviours"
    failed = [check.name for check in checks if not check.passed]
    assert not failed, f"synthetic-signal checks failed: {failed}"


def test_feature_implementation_validation_passes(registry, base_config):
    report = run_feature_validation(registry, base_config)
    assert report["ok"], report.get("failed_checks")
    assert report["n_features"] == len(registry)


def test_feature_implementation_validation_catches_a_broken_implementation(registry, base_config):
    """Injecting a wrong formula must be caught by the amplitude-scaling check."""
    import numpy as np

    from eeg_research_agent.eeg.feature_registry import FeatureDefinition

    broken = registry.subset(["abs_power_delta"])
    original = broken.get("abs_power_delta")
    broken.register(
        FeatureDefinition(
            feature_id="abs_power_delta_broken",
            name="deliberately broken power",
            category="spectral",
            family="spectral",
            function=lambda context: float(np.mean(np.abs(context.epoch))),
            mathematical_definition="mean absolute amplitude (wrong on purpose)",
            expected_range=(None, None),
            novelty_status="established",
            validation_status="unvalidated",
        )
    )
    del original
    report = run_feature_validation(broken, base_config)
    assert not report["ok"], "a linear-in-amplitude 'power' must fail the quadratic-scaling check"


def test_statistical_validation_passes(feature_table):
    frame, _table, _splits = feature_table
    report = run_statistical_validation(table=frame)
    assert report["ok"], report.get("failed_checks")


def _claims(tmp_path: Path) -> tuple[list[dict], dict]:
    artifact = tmp_path / "computation.json"
    artifact.write_text(json.dumps({"n_significant": 7, "effect": 0.42, "label": "N2"}))
    payload = {
        "claims": [
            {
                "claim_id": "c1",
                "description": "seven features survive correction",
                "value": 7,
                "locator": "n_significant",
                "source": "computation.json",
            },
            {
                "claim_id": "c2",
                "description": "the effect size is 0.42",
                "value": 0.42,
                "locator": "effect",
                "source": "computation.json",
            },
        ]
    }
    return payload["claims"], payload


def test_claim_validation_verifies_honest_claims(tmp_path):
    claims, _ = _claims(tmp_path)
    result = validate_claims(claims, {}, artifacts_dir=tmp_path)
    assert result["ok"], result["failures"]
    assert result["n_verified"] == 2


def test_claim_validation_catches_a_wrong_number(tmp_path):
    claims, _ = _claims(tmp_path)
    claims[0]["value"] = 8  # the report says something the artifact does not
    result = validate_claims(claims, {}, artifacts_dir=tmp_path)
    assert not result["ok"]
    assert result["failures"][0]["claim_id"] == "c1"
    assert result["failures"][0]["reason"] == "value mismatch"


def test_claim_validation_catches_a_missing_locator(tmp_path):
    claims, _ = _claims(tmp_path)
    claims[1]["locator"] = "effect_size"
    result = validate_claims(claims, {}, artifacts_dir=tmp_path)
    assert not result["ok"]
    assert "locator not found" in result["failures"][0]["reason"]


def test_claim_validation_catches_an_unreadable_artifact(tmp_path):
    claims, _ = _claims(tmp_path)
    (tmp_path / "computation.json").write_text("{not json")
    result = validate_claims(claims, {}, artifacts_dir=tmp_path)
    assert not result["ok"]
    assert "unreadable" in result["failures"][0]["reason"]


def test_citation_validation_flags_incomplete_verified_records():
    result = validate_citations(
        [{"title": "A verified paper", "year": 2020, "source": "journal", "identifier_type": "doi"}]
    )
    assert not result["ok"]
    assert result["n_incomplete"] == 1


def test_citation_validation_accepts_labelled_offline_records_with_a_recorded_gap():
    result = validate_citations(
        [
            {
                "title": "Anchor document",
                "year": None,
                "source": "project_brief",
                "identifier_type": "url",
                "identifier": "https://example.invalid/doc",
                "evidence_level": "unverified-offline",
            }
        ]
    )
    assert result["ok"], "an explicitly unverified record must be reported, not fabricated"
    assert result["n_incomplete_unverified"] == 1
    assert result["n_unverified_offline"] == 1


def test_citation_validation_rejects_a_malformed_doi():
    result = validate_citations(
        [
            {
                "title": "Paper",
                "year": 2021,
                "source": "journal",
                "identifier_type": "doi",
                "identifier": "not-a-doi",
            }
        ]
    )
    assert not result["ok"]


def test_novelty_labelling_catches_overstated_language():
    claim = Claim(
        claim_id="n1",
        description="the composite score is a confirmed marker",
        value=1.0,
        locator="x",
        source="",
        level="novel-hypothesis",
    )
    report = "Our analysis shows the composite score is a confirmed marker of N2."
    result = validate_novelty_labels([claim], report)
    assert not result["ok"]
    assert result["offenders"][0]["offending_phrase"] in {"confirmed", "known biomarker"}


def test_report_file_validation_catches_placeholders(tmp_path):
    good = tmp_path / "good.md"
    good.write_text("# Title\n\n" + "content " * 60)
    assert validate_report_files([good])["ok"]

    bad = tmp_path / "bad.md"
    bad.write_text("# Title\n\nTODO: fill this in\n" + "content " * 40)
    result = validate_report_files([bad])
    assert not result["ok"]
    assert any("todo" in problem["reason"] for problem in result["problems"])

    missing = validate_report_files([tmp_path / "nope.md"])
    assert not missing["ok"]
