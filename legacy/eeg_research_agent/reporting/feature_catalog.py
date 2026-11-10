"""``reports/feature_catalog.md``.

One entry per registered feature, generated from the registry metadata rather
than from prose: name, formula, motivation, references, units, cost, novelty
status and validation status.  Novel hypotheses are grouped separately and
carry the label required by the brief, because an invented equation is not a
scientific result.
"""

from __future__ import annotations

from typing import Any

from .context import ReportContext
from .markdown_utils import heading, render_key_values, render_list, render_table

__all__ = ["write_feature_catalog"]

SOURCE = "tables/feature_catalog.json"

NOVELTY_LABEL = {
    "established": "Established - standard EEG/sleep feature with a published definition",
    "literature-derived": "Literature-inspired - adapted from published relationships",
    "novel-hypothesis": "Novel hypothesis - proposed by this project; NOT a validated finding",
}


def _features(context: ReportContext) -> list[dict[str, Any]]:
    return [dict(item) for item in (context.catalog.get("features") or [])]


def _definition_table(features: list[dict[str, Any]]) -> str:
    return render_table(
        [
            {
                "feature": entry.get("feature_id"),
                "name": entry.get("name"),
                "family": entry.get("family"),
                "definition": entry.get("mathematical_definition"),
                "units": entry.get("output_units"),
                "cost": entry.get("computational_cost"),
                "novelty": entry.get("novelty_status"),
                "validation": entry.get("validation_status"),
            }
            for entry in features
        ],
        columns=["feature", "name", "family", "definition", "units", "cost", "novelty", "validation"],
    )


def write_feature_catalog(context: ReportContext) -> str:
    catalog = context.catalog
    features = _features(context)
    by_novelty = catalog.get("by_novelty_status") or {}
    by_family = catalog.get("by_family") or {}
    by_validation = catalog.get("by_validation_status") or {}

    context.add_claim(
        "catalog.n_features",
        "number of features registered for extraction",
        catalog.get("n_features"),
        "n_features",
        SOURCE,
        unit="features",
    )
    for name, count in sorted(by_novelty.items()):
        context.add_claim(
            f"catalog.novelty.{name}",
            f"number of features with novelty status {name}",
            count,
            f"by_novelty_status.{name}",
            SOURCE,
            level="novel-hypothesis" if name == "novel-hypothesis" else "established",
            unit="features",
        )

    lines: list[str] = [
        heading(1, "Feature Catalog"),
        context.origin_caveat(),
        heading(2, "Composition"),
        render_key_values(
            {
                "registered features": catalog.get("n_features"),
                "by novelty status": by_novelty,
                "by family": by_family,
                "by validation status": by_validation,
            }
        ),
        heading(2, "How validation status is assigned"),
        render_list(
            [
                "`unit-tested` - the implementation has a dedicated unit test.",
                "`synthetic-validated` - behaviour verified on synthetic signals with known properties "
                "(amplitude scaling, sampling-frequency sensitivity, expected range, numerical stability).",
                "`reproduced-from-literature` - the definition follows a published definition and the "
                "implementation reproduces the documented qualitative behaviour.",
                "`unvalidated` - implemented and exercised, but no dedicated validation evidence yet.",
            ]
        ),
        "Validation status describes the *implementation*, never the *usefulness* of a feature. "
        "A synthetic-validated feature can still have no predictive value.",
        heading(2, "Category A and B - established and literature-derived features"),
        _definition_table(
            [entry for entry in features if entry.get("novelty_status") != "novel-hypothesis"]
        ),
        heading(2, "Category C - novel hypotheses"),
        (
            "The features below are **proposed by this project**, not taken from the literature. "
            "Each has an explicit equation, an executable implementation, unit tests and "
            "synthetic-signal validation. None of them is a validated biomarker, and a small "
            "p-value for one of them does not make it one."
        ),
    ]
    novel = [entry for entry in features if entry.get("novelty_status") == "novel-hypothesis"]
    if novel:
        for entry in novel:
            lines += [
                heading(3, str(entry.get("name"))),
                render_table(
                    [
                        {"field": "export label", "value": NOVELTY_LABEL["novel-hypothesis"]},
                        {"field": "feature id", "value": entry.get("feature_id")},
                        {"field": "equation", "value": entry.get("mathematical_definition")},
                        {"field": "motivation", "value": entry.get("biological_motivation") or "n/a"},
                        {"field": "references", "value": entry.get("literature_references")},
                        {"field": "expected range", "value": entry.get("expected_range")},
                        {"field": "units", "value": entry.get("output_units")},
                        {"field": "computational cost", "value": entry.get("computational_cost")},
                        {"field": "validation", "value": entry.get("validation_status")},
                        {"field": "numerical notes", "value": entry.get("notes") or "n/a"},
                    ],
                    columns=["field", "value"],
                ),
            ]
    else:
        lines.append("_No novel-hypothesis features are registered in this run._")

    lines += [
        heading(2, "Requirements and failure modes"),
        render_table(
            [
                {
                    "feature": entry.get("feature_id"),
                    "min sfreq (Hz)": entry.get("required_sampling_frequency"),
                    "max sfreq (Hz)": entry.get("max_required_sampling_frequency"),
                    "min samples": entry.get("min_signal_samples"),
                    "expected range": entry.get("expected_range"),
                    "notes": entry.get("notes"),
                }
                for entry in features
            ],
            columns=[
                "feature",
                "min sfreq (Hz)",
                "max sfreq (Hz)",
                "min samples",
                "expected range",
                "notes",
            ],
        ),
        heading(2, "Non-causal features"),
        (
            "`feature_delta_norm` and `feature_delta_norm_sigma` compare an epoch with the preceding "
            "epoch of the same recording. They are *offline / non-causal* when the preceding epoch is "
            "unavailable (the first epoch of a recording is `n/a`), and they are excluded from every "
            "causal ablation and from every subject-grouped predictive evaluation."
        ),
    ]
    return "\n".join(lines).strip() + "\n"
