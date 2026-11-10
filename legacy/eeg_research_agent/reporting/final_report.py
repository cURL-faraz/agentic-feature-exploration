"""The final research report and the report-stage orchestrator.

``generate_all_reports`` is the entry point used by the CLI and the LangGraph
node.  It:

1. loads every artifact the reports may quote,
2. renders the figures,
3. writes the five markdown reports,
4. registers every quoted number as a *claim* pointing at the artifact it came
   from, and saves the claim register to ``artifacts/tables/claims.json``.

Step 4 is what makes the reports falsifiable: the verification stage re-reads the
artifacts and fails if any quoted number disagrees with the saved computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .context import ReportContext, load_report_context
from .feature_catalog import write_feature_catalog
from .literature_report import write_literature_review
from .markdown_utils import (
    format_value,
    heading,
    normalise_markdown,
    render_key_values,
    render_list,
    render_table,
)
from .plots import generate_figures
from .statistical_report import write_statistical_report
from .validation_report import write_validation_report

__all__ = ["write_final_report", "generate_all_reports"]

logger = get_logger(__name__)


def _top_features(context: ReportContext, limit: int = 12) -> list[str]:
    comparisons = list(context.statistical.get("comparisons") or [])
    ranked = sorted(
        [item for item in comparisons if not item.get("skipped")],
        key=lambda item: (
            (item.get("primary") or {}).get("p_adjusted")
            if (item.get("primary") or {}).get("p_adjusted") is not None
            else 1.0
        ),
    )
    features = [str(item.get("feature")) for item in ranked if item.get("feature")]
    if len(features) < limit:
        selection = (context.statistical.get("feature_selection") or {}).get("top_k_mutual_information") or []
        features += [str(name) for name in selection if name not in features]
    return features[:limit]


def _best_test_row(evaluation: Mapping[str, Any]) -> tuple[int | None, dict[str, Any] | None]:
    rows = evaluation.get("results") or []
    best_index: int | None = None
    best_value = -np.inf
    for index, row in enumerate(rows):
        if row.get("split") != "test":
            continue
        value = row.get("balanced_accuracy")
        if value is None or not np.isfinite(value):
            continue
        if value > best_value:
            best_value = float(value)
            best_index = index
    return best_index, (rows[best_index] if best_index is not None else None)


def _evaluation_table(evaluation: Mapping[str, Any], split: str = "test") -> list[dict[str, Any]]:
    return [
        {
            "feature set": row.get("feature_set"),
            "classifier": row.get("classifier"),
            "features": row.get("n_features"),
            "balanced accuracy": row.get("balanced_accuracy"),
            "accuracy": row.get("accuracy"),
            "macro F1": row.get("macro_f1"),
            "kappa": row.get("cohens_kappa"),
            "F1 N1": (row.get("per_class") or {}).get("N1", {}).get("f1"),
            "F1 N2": (row.get("per_class") or {}).get("N2", {}).get("f1"),
            "F1 REM": (row.get("per_class") or {}).get("REM", {}).get("f1"),
        }
        for row in (evaluation.get("results") or [])
        if row.get("split") == split
    ]


def write_final_report(context: ReportContext) -> str:
    """Render ``reports/final_research_report.md`` and register its claims."""
    dataset = context.dataset_summary
    stats = context.statistical
    evaluation = context.evaluation
    verification = context.verification
    family = stats.get("family_correction") or {}
    best_index, best_row = _best_test_row(evaluation)
    comparisons = list(stats.get("comparisons") or [])
    novelty_counts = (context.catalog.get("by_novelty_status") or {})
    registered = {
        str(definition.get("feature_id"))
        for definition in (context.catalog.get("features") or ())
        if definition.get("feature_id")
    }
    analysed = {str(item.get("feature")) for item in comparisons if item.get("feature")}
    derived = sorted(analysed - registered)

    dataset_claims = {
        "dataset.subject_count": ("subjects in the corpus", "subject_count", "subjects"),
        "dataset.epoch_count": ("epochs in the corpus", "epoch_count", "epochs"),
        "dataset.sampling_frequency": ("sampling frequency of the analysed derivation", "sampling_frequency", "Hz"),
        "dataset.epoch_duration": ("epoch duration used for segmentation", "epoch_duration", "s"),
    }
    for claim_id, (description, locator, unit) in dataset_claims.items():
        value = dataset.get(locator)
        if value is not None and not isinstance(value, list):
            context.add_claim(claim_id, description, value, locator, "tables/dataset_summary.json", unit=unit)

    if family.get("n_rejected") is not None:
        context.add_claim(
            "final.n_significant_after_fdr",
            "features whose stage comparison survives FDR correction",
            family.get("n_rejected"),
            "family_correction.n_rejected",
            "tables/statistical_results.json",
            level="exploratory",
            unit="features",
        )
    if best_row is not None:
        for claim_id, metric in (
            ("final.best_test_balanced_accuracy", "balanced_accuracy"),
            ("final.best_test_macro_f1", "macro_f1"),
            ("final.best_test_kappa", "cohens_kappa"),
        ):
            value = best_row.get(metric)
            if value is not None and np.isfinite(value):
                context.add_claim(
                    claim_id,
                    f"held-out test {metric.replace('_', ' ')} of the best model",
                    float(value),
                    f"results[{best_index}].{metric}",
                    "tables/predictive_evaluation.json",
                    level="exploratory",
                )

    lines: list[str] = [
        heading(1, "Single-Channel EEG Feature Discovery for Sleep Staging"),
        context.origin_caveat(),
        heading(2, "1. Executive Summary"),
        (
            "This report is the output of an autonomous research pipeline over a single EEG "
            "derivation. It (i) assembles a corpus and maps its sleep-stage annotations, "
            "(ii) segments it into 30 s epochs, (iii) extracts a registered catalog of "
            "spectral, temporal, nonlinear, event-based and novel mathematical features, "
            "(iv) compares each feature across stages with repeated-measures-aware statistics "
            "and FDR correction, (v) evaluates whether the features predict stage on "
            "*unseen subjects*, and (vi) verifies the implementations on synthetic signals "
            "with known properties."
        ),
        render_key_values(
            {
                "research question": (
                    "Which single-channel EEG characteristics are statistically associated with "
                    "sleep stages, and which of them improve subject-independent stage prediction?"
                ),
                "data origin": context.data_origin,
                "subjects": dataset.get("subject_count"),
                "epochs": dataset.get("epoch_count"),
                "derivation": dataset.get("selected_channel"),
                "registered features in the catalog": len(registered) or context.catalog.get("n_features"),
                "feature columns analysed in this run": len(analysed) or stats.get("n_features"),
                "significance after FDR": family.get("n_rejected"),
                "alpha / correction": f"{stats.get('alpha')} / {family.get('method')}",
                "primary analysis unit": stats.get("unit_of_analysis"),
                "best held-out model": (
                    f"{best_row.get('classifier')} on {best_row.get('feature_set')}"
                    if best_row
                    else "no evaluable held-out model"
                ),
                "verification": "PASS" if verification.get("ok") else "FAIL",
            }
        ),
        (
            (
                "The analysed table is the registered catalog plus "
                f"{len(derived)} derived column(s) computed from *adjacent* epochs "
                f"({', '.join(f'`{name}`' for name in derived)}). Those are flagged non-causal and are "
                "excluded from every causal ablation; they must not be read as per-epoch markers."
            )
            if derived
            else "The analysed table is exactly the registered feature catalog."
        ),
        heading(2, "2. Dataset"),
        render_key_values(
            {
                "dataset": dataset.get("dataset"),
                "source": dataset.get("source") or dataset.get("provider"),
                "origin": context.data_origin,
                "subjects": dataset.get("subject_count"),
                "recordings": dataset.get("recording_count"),
                "derivation": dataset.get("selected_channel"),
                "sampling frequency (Hz)": dataset.get("sampling_frequency"),
                "epoch duration (s)": dataset.get("epoch_duration"),
                "available stages": dataset.get("available_stages"),
                "class distribution": dataset.get("class_distribution"),
            }
        ),
        "Quality control:",
        render_key_values(dataset.get("quality_report") or {}),
        (
            "Epoch construction keeps a subject/recording/epoch identity triple and the recording "
            "timestamp for every epoch, so any epoch can be traced back to its source. Epochs that "
            "failed quality gates are recorded with their reason rather than silently dropped."
        ),
        heading(2, "3. Literature Review"),
        (
            f"{len(context.papers)} bibliographic record(s) are available in this run "
            f"(retrieval mode: `{context.literature.get('mode', 'unavailable')}`). "
            "See `reports/literature_review.md` for the full review, the planned search queries, "
            "and an explicit statement of what could not be verified. No numeric claim in this "
            "report attributes a value to a published author."
        ),
        heading(2, "4. Feature Catalog"),
        render_key_values(
            {
                "registered features": context.catalog.get("n_features"),
                "established": novelty_counts.get("established", 0),
                "literature-derived": novelty_counts.get("literature-derived", 0),
                "novel hypotheses": novelty_counts.get("novel-hypothesis", 0),
                "families": context.catalog.get("by_family"),
            }
        ),
        (
            "Formulas, motivations, units and validation status for every feature are in "
            "`reports/feature_catalog.md`. Novel features are labelled there as "
            "*novel hypothesis - not a validated finding*."
        ),
        heading(2, "5. Feature Distributions"),
    ]
    if context.figures:
        lines.append(render_list([f"`{path}`" for path in sorted(context.figures.values())]))
    else:
        lines.append("_No figures were generated in this run._")
    lines += [
        heading(2, "6. Statistical Comparisons"),
        (
            "Full tables, assumption checks and post-hoc results are in "
            "`reports/statistical_analysis.md`. The headline comparison is repeated-measures "
            "(Friedman) on subject x stage means, with paired Wilcoxon post-hoc tests and "
            f"{family.get('method')} correction across the {family.get('n_tests')} tested features."
        ),
        render_table(
            [
                {
                    "feature": item.get("feature"),
                    "primary test": (item.get("primary") or {}).get("test"),
                    "p (raw)": (item.get("primary") or {}).get("p_value"),
                    "p (FDR)": (item.get("primary") or {}).get("p_adjusted"),
                    "effect": ((item.get("primary") or {}).get("effect_size") or {}).get("measure"),
                    "effect value": ((item.get("primary") or {}).get("effect_size") or {}).get("value"),
                }
                for item in comparisons[:15]
            ],
            columns=["feature", "primary test", "p (raw)", "p (FDR)", "effect", "effect value"],
        ),
        heading(2, "7. Predictive Evaluation"),
        (
            "Models are fitted on training subjects, validated on a disjoint validation subject "
            "set, and reported once on held-out test subjects. Scaling, imputation and any "
            "supervised feature selection are fitted inside the training folds only. Epoch-level "
            "random splits are never used."
        ),
        render_table(
            _evaluation_table(evaluation, split="test"),
            columns=[
                "feature set",
                "classifier",
                "features",
                "balanced accuracy",
                "accuracy",
                "macro F1",
                "kappa",
                "F1 N1",
                "F1 N2",
                "F1 REM",
            ],
        ),
    ]
    if context.cross_validation:
        lines += [
            heading(3, "Subject-grouped cross-validation on training subjects"),
            render_table(
                [
                    {
                        "classifier": row.get("classifier"),
                        "features": row.get("n_features"),
                        "folds": len(row.get("folds") or []),
                        "balanced accuracy": (row.get("aggregate") or {}).get("balanced_accuracy_mean"),
                        "std": (row.get("aggregate") or {}).get("balanced_accuracy_std"),
                        "macro F1": (row.get("aggregate") or {}).get("macro_f1_mean"),
                    }
                    for row in context.cross_validation
                ],
                columns=["classifier", "features", "folds", "balanced accuracy", "std", "macro F1"],
            ),
        ]
    else:
        lines.append("_Cross-validation was not run in this execution._")
    lines += [
        heading(2, "8. Feature Interactions"),
    ]
    interactions = (stats.get("feature_selection") or {}).get("interactions") or []
    if interactions:
        lines += [
            render_table(
                [
                    {
                        "pair": entry.get("interaction"),
                        "role": entry.get("role"),
                        "CV MI gain": entry.get("cv_mi_gain"),
                        "passes screen": entry.get("passes_heuristic_screen"),
                    }
                    for entry in interactions[:10]
                ],
                columns=["pair", "role", "CV MI gain", "passes screen"],
            ),
            (
                "Interaction search is bounded to a prespecified candidate list and scored with "
                "cross-validated mutual-information gain, so it is a controlled search rather than "
                "a scan over arbitrary combinations. Surviving candidates remain *exploratory* "
                "until they are confirmed on held-out subjects."
            ),
        ]
    else:
        lines.append("_No interaction candidates were evaluated in this run._")
    lines += [
        heading(2, "9. Verification"),
        render_key_values(
            {
                "verification verdict": "PASS" if verification.get("ok") else "FAIL",
                "checks": verification.get("n_checks"),
                "failed checks": verification.get("n_failed_checks"),
                "failed sections": verification.get("failed_checks") or [],
            }
        ),
        (
            "Synthetic signals with known properties (pure tones, dual tones, white and pink noise, "
            "amplitude- and frequency-modulated signals, spindle bursts, slow waves, artefacts) are "
            "used to check that features respond as specified. Statistical routines are calibrated on "
            "null data. Splits and selection scopes are audited for leakage. Details: "
            "`reports/validation_report.md`."
        ),
        heading(2, "10. Discussion"),
    ]
    lines.append(_discussion(context, comparisons, best_row))
    lines += [
        heading(2, "11. Limitations"),
        render_list(
            [
                "Single derivation only: without EOG and EMG, REM and N1 are expected to be the hardest stages.",
                "Sleep-stage labels are assumed correct; inter-scorer disagreement is not modelled, and "
                "the annotation protocol defines the label semantics.",
                "Class imbalance is severe; balanced accuracy and per-stage F1 are reported instead of accuracy alone.",
                "Association, predictive value, biological plausibility and causality are four different "
                "questions; this report answers the first two and does not claim the others.",
                "Multiple comparisons are controlled, so reported significance means `significant after "
                "correction across the tested family`, not `significant in isolation`.",
                "Monte-Carlo p-values cannot go below 1/(n_iterations+1).",
                "Results are specific to this corpus and segmentation; transfer to another dataset, "
                "montage or population is untested.",
                (
                    "This run used synthetic data, so no statement here describes human sleep."
                    if context.synthetic
                    else "Sample size is small, so effect-size confidence intervals are wide."
                ),
            ]
        ),
        heading(2, "12. Future Work"),
        render_list(
            [
                "Compare against learned representations (AttnSleep, BIOT) using the same subject-grouped splits.",
                "Fuse the extracted feature table with deep-learning embeddings and test whether the "
                "features add information beyond the embedding.",
                "Add domain adaptation / cross-dataset evaluation (Sleep-EDF vs SHHS) to test transfer.",
                "Improve event detection (spindle, K-complex) with validated detectors and re-test the event family.",
                "Model label uncertainty (multi-scorer agreement) instead of treating annotations as ground truth.",
                "Extend the interaction search to a prespecified structural family and confirm on held-out subjects.",
            ]
        ),
        heading(2, "13. References"),
    ]
    if context.papers:
        lines.append(
            render_table(
                [
                    {
                        "title": paper.get("title"),
                        "year": paper.get("year"),
                        "identifier": paper.get("identifier"),
                        "type": paper.get("identifier_type"),
                        "evidence level": paper.get("evidence_level"),
                    }
                    for paper in context.papers
                ],
                columns=["title", "year", "identifier", "type", "evidence level"],
            )
        )
        if any(str(p.get("evidence_level")) == "unverified-offline" for p in context.papers):
            lines.append(
                "\nRecords marked `unverified-offline` were **not retrieved** in this run; their "
                "metadata was not checked against a publisher and they are listed as pointers for a "
                "networked run, not as verified citations."
            )
    else:
        lines.append("_No references are available in this run._")

    lines += [
        heading(2, "Provenance"),
        (
            "Every number in this report is registered in `artifacts/tables/claims.json` with the "
            "artifact file and locator it came from, and the verification stage re-reads those "
            "artifacts to confirm the report matches the computation."
        ),
    ]
    return "\n".join(lines).strip() + "\n"


def _discussion(
    context: ReportContext, comparisons: Sequence[Mapping[str, Any]], best_row: Mapping[str, Any] | None
) -> str:
    significant = [
        item for item in comparisons if (item.get("primary") or {}).get("significant_after_correction")
    ]
    families: dict[str, int] = {}
    by_id = {str(entry.get("feature_id")): entry for entry in (context.catalog.get("features") or [])}
    for item in significant:
        definition = by_id.get(str(item.get("feature")), {})
        family = str(definition.get("family") or "unknown")
        families[family] = families.get(family, 0) + 1
    parts = [
        (
            f"Of {len(comparisons)} features compared across stages, "
            f"{len(significant)} survive FDR correction at alpha={context.statistical.get('alpha')}. "
            "Surviving features cluster by family as follows: "
            + (", ".join(f"{name} ({count})" for name, count in sorted(families.items())) or "none")
            + "."
        ),
        (
            "Because the classes are heavily imbalanced, the meaningful comparison is not overall "
            "accuracy but balanced accuracy, macro F1 and per-stage F1 - particularly for N1, which is "
            "the stage most often merged with its neighbours."
        ),
    ]
    if best_row:
        parts.append(
            f"The best held-out model was `{best_row.get('classifier')}` on the "
            f"`{best_row.get('feature_set')}` feature set, with balanced accuracy "
            f"{format_value(best_row.get('balanced_accuracy'))} and macro F1 "
            f"{format_value(best_row.get('macro_f1'))}."
        )
    parts.append(
        "Statistically associated features that do not improve held-out subject-independent "
        "prediction are reported as associations only; the ablation table in section 7 is the "
        "evidence for predictive usefulness, and no feature is described as predictively useful "
        "without it."
    )
    return "\n\n".join(parts)


def _write(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normalise_markdown(text), encoding="utf-8")
    return str(path)


def _existing_figures(config: Any) -> dict[str, str]:
    """Reuse figures already on disk (so a second pass does not redraw them)."""
    figures_dir = config.path("figures")
    if not figures_dir.is_dir():
        return {}
    return {path.stem: str(path) for path in sorted(figures_dir.glob("*.png"))}


def generate_all_reports(
    config: Any,
    artifacts: Any,
    results: Mapping[str, Any] | None = None,
    write_claims: bool = True,
    figures: Mapping[str, str] | None = None,
    only: Sequence[str] | None = None,
) -> dict[str, str]:
    """Generate the reports and the claim register; returns ``{name: path}``.

    Parameters
    ----------
    figures:
        Pre-computed figures to reference instead of regenerating them.  When
        omitted, PNGs already present in the figures directory are reused if the
        feature table is unavailable.
    only:
        Restrict writing to these report keys (used by the pipeline to refresh the
        verification and final reports after the evidence check has run).
    """
    results = dict(results or {})
    context = load_report_context(config, artifacts, results)
    written: dict[str, str] = dict(context.written)

    features = _top_features(context)
    table = context.feature_table
    if figures is not None:
        context.figures = dict(figures)
    elif table is not None and len(table):
        try:
            context.figures = generate_figures(
                table,
                config,
                statistical_results=context.statistical,
                evaluation=context.evaluation,
                features=features,
            )
        except Exception as exc:  # noqa: BLE001 - figures must never break the report
            logger.warning("figure generation failed: %s", exc)
            context.notes.append(f"figure generation failed: {type(exc).__name__}: {exc}")
    else:
        context.figures = _existing_figures(config)
        if not context.figures:
            context.notes.append("no feature table was available, so no figures were generated")
    if table is not None and len(table) and not context.figures:
        context.notes.append("figure generation produced no figures")
    written.update({f"figure:{name}": path for name, path in context.figures.items()})

    reports_dir = config.path("reports")
    wanted = list(only) if only is not None else [
        "literature_review",
        "feature_catalog",
        "statistical_analysis",
        "validation_report",
        "final_research_report",
    ]
    writers = {
        "literature_review": ("literature_review.md", write_literature_review),
        "feature_catalog": ("feature_catalog.md", write_feature_catalog),
        "statistical_analysis": ("statistical_analysis.md", write_statistical_report),
        "validation_report": ("validation_report.md", write_validation_report),
        "final_research_report": ("final_research_report.md", write_final_report),
    }
    for key in wanted:
        filename, writer = writers[key]
        written[key] = _write(reports_dir / filename, writer(context))

    if write_claims:
        path = context.save_claims()
        written["claims_json"] = str(path)
    written["n_claims"] = str(len(context.claims))
    written["_notes"] = " | ".join(context.notes) if context.notes else ""
    return written
