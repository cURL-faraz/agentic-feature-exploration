"""Deterministic research pipeline.

This module is the *single source of truth* for what the project actually
computes.  The CLI calls it directly, and the LangGraph agent nodes call the
same functions - there is no second, "agentic" implementation of the science.
That separation is deliberate: an LLM may choose *which* analyses to prioritise
and how to describe them, but it can never be the thing that computes a p-value.

Every stage
-----------
* writes its artifacts under the configured paths,
* emits a structured summary into the JSONL event log,
* is cache-friendly: re-running a stage does not re-do expensive work when the
  cache key matches,
* records failures instead of hiding them.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import ProjectConfig, load_config
from .logging_utils import EventLogger, ExecutionManifest, get_logger

__all__ = ["PipelineArtifacts", "ResearchPipeline", "MODE_PRESETS"]

logger = get_logger(__name__)


#: Execution-mode presets.  ``smoke`` keeps CI and review runs to a couple of
#: minutes; ``full`` runs the complete statistical machinery.
MODE_PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "max_features_statistics": 24,
        "permutation_iterations": 200,
        "bootstrap_iterations": 200,
        "include_events": True,
    },
    "standard": {
        "max_features_statistics": None,
        "permutation_iterations": 300,
        "bootstrap_iterations": 400,
        "include_events": True,
    },
    "full": {
        "max_features_statistics": None,
        "permutation_iterations": 1000,
        "bootstrap_iterations": 1000,
        "include_events": True,
    },
}


@dataclass
class PipelineArtifacts:
    """In-memory state carried between pipeline stages."""

    config: ProjectConfig
    events: EventLogger | None = None
    manifest: ExecutionManifest | None = None
    acquisition: Any = None
    recordings: list[Any] = field(default_factory=list)
    preprocessed: list[Any] = field(default_factory=list)
    epoch_sets: list[Any] = field(default_factory=list)
    segmentation_reports: list[Any] = field(default_factory=list)
    splits: Any = None
    feature_table: pd.DataFrame | None = None
    feature_definitions: list[dict[str, Any]] = field(default_factory=list)
    feature_report: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def data_origin(self) -> str:
        if self.acquisition is not None:
            return str(self.acquisition.origin)
        return "unknown"

    def to_summary(self) -> dict[str, Any]:
        return {
            "data_origin": self.data_origin,
            "n_recordings": len(self.recordings),
            "n_epochs": int(0 if self.feature_table is None else len(self.feature_table)),
            "n_features": len(self.feature_definitions),
            "stages_completed": sorted(self.results),
            "errors": list(self.errors),
        }


class ResearchPipeline:
    """Runs the deterministic stages in order, with caching and logging."""

    def __init__(
        self,
        config: ProjectConfig | None = None,
        origin: str | None = None,
        mode: str | None = None,
        use_cache: bool | None = None,
        max_subjects: int | None = None,
        quiet: bool = False,
    ) -> None:
        self.config = config or load_config()
        self.quiet = quiet
        self.mode = (mode or self.config.settings.execution.mode or "standard").lower()
        if self.mode not in MODE_PRESETS:
            raise ValueError(f"unknown execution mode: {self.mode}")
        self.preset = MODE_PRESETS[self.mode]
        self.origin = origin or self.config.dataset.data_origin
        self.use_cache = (
            self.config.settings.execution.use_cache if use_cache is None else bool(use_cache)
        )
        self.max_subjects = int(
            max_subjects
            if max_subjects is not None
            else self.config.settings.execution.max_subjects
        )
        self.artifacts = PipelineArtifacts(config=self.config)
        self._statistics_config = self.config.analysis.statistics.model_copy(
            update={
                "permutation_iterations": int(self.preset["permutation_iterations"]),
                "bootstrap_iterations": int(self.preset["bootstrap_iterations"]),
            }
        )
        self.artifacts.config = self.config.model_copy(
            update={
                "analysis": self.config.analysis.model_copy(
                    update={"statistics": self._statistics_config}
                )
            }
        )

    # ------------------------------------------------------------------ setup #
    def initialise(self) -> dict[str, Any]:
        """Create directories, logging and the execution manifest."""
        from .logging_utils import setup_logging

        created = self.config.ensure_directories()
        setup_logging(
            level=self.config.settings.logging.level,
            fmt=self.config.settings.logging.format,
            log_file=self.config.path("logs") / "pipeline.log",
            quiet=self.quiet,
        )
        events_path = self.config.settings.logging.json_events
        events_file = self.config.root / events_path
        self.artifacts.events = EventLogger(events_file)
        self.artifacts.manifest = ExecutionManifest(
            run_id=time.strftime("%Y%m%dT%H%M%S"),
            objective=(
                "Discover, evaluate and statistically verify single-channel EEG features "
                "associated with sleep stages"
            ),
        )
        summary = {
            "mode": self.mode,
            "config_root": str(self.config.root),
            "directories": {key: str(value) for key, value in created.items()},
            "use_cache": self.use_cache,
            "max_subjects": self.max_subjects,
            "llm_provider": self.config.settings.llm.provider,
        }
        self.artifacts.results["initialise"] = summary
        self._event("pipeline_initialised", **summary)
        return summary

    def _event(self, name: str, **payload: Any) -> None:
        if self.artifacts.events is not None:
            self.artifacts.events.emit(name, **payload)

    # ------------------------------------------------------------- stage 1+2 #
    def acquire(self, origin: str | None = None) -> dict[str, Any]:
        """Download Sleep-EDF (when available) or generate the synthetic corpus."""
        from .data.acquisition import acquire_dataset, write_acquisition_manifest

        requested = origin or self.origin
        with self._stage("acquire_dataset", origin=requested):
            result = acquire_dataset(self.config, origin=requested)
            recordings = list(result.recordings)
            if self.max_subjects and len({r.subject_id for r in recordings}) > self.max_subjects:
                keep = sorted({r.subject_id for r in recordings})[: self.max_subjects]
                recordings = [r for r in recordings if r.subject_id in keep]
                result.recordings = recordings
            self.artifacts.acquisition = result
            self.artifacts.recordings = recordings
            manifest_path = write_acquisition_manifest(result, self.config)
        summary = {
            "origin": result.origin,
            "requested_origin": requested,
            "n_recordings": len(recordings),
            "subjects": sorted({r.subject_id for r in recordings}),
            "manifest": str(manifest_path),
            "detail": {k: v for k, v in result.manifest.items() if k != "recordings"},
        }
        self.artifacts.results["acquire"] = summary
        self._event("dataset_acquired", **summary)
        return summary

    def inspect(self) -> dict[str, Any]:
        """Summarise the dataset as the dataset-intelligence agent would."""
        from .data.inspection import inspect_dataset

        recordings = self.artifacts.recordings or self.artifacts.preprocessed
        if not recordings:
            raise RuntimeError("no recordings acquired; run acquire() first")
        with self._stage("inspect_dataset", n_recordings=len(recordings)):
            summary = inspect_dataset(recordings, self.config)
        output = self.config.path("metadata") / "dataset_summary.json"
        _write_json(output, summary)
        payload = dict(summary)
        payload["written"] = str(output)
        self.artifacts.results["inspect"] = payload
        self._event(
            "dataset_inspected",
            dataset=summary["dataset"],
            data_origin=summary["data_origin"],
            selected_channel=summary["selected_channel"],
            sampling_frequency=summary["sampling_frequency"],
            subject_count=summary["subject_count"],
            class_distribution=summary["class_distribution"],
        )
        return payload

    def prepare(self) -> dict[str, Any]:
        """Preprocess, epoch, split by subject and extract every feature."""
        return {
            "preprocess": self.preprocess(),
            "segment": self.segment_and_split()["segment"],
            "split": self.artifacts.results["split"],
            "features": self.extract_features(),
        }

    def preprocess(self) -> dict[str, Any]:
        """Condition every recording and record exactly what was applied."""
        from .data.preprocessing import preprocess_recording

        if not self.artifacts.recordings:
            raise RuntimeError("no recordings acquired; run acquire() first")

        with self._stage("preprocess", n_recordings=len(self.artifacts.recordings)):
            preprocessed = []
            reports = []
            for recording in self.artifacts.recordings:
                cleaned, report = preprocess_recording(recording, self.config)
                preprocessed.append(cleaned)
                reports.append(report)
            self.artifacts.preprocessed = preprocessed
        preprocessing_summary = {
            "n_recordings": len(preprocessed),
            "mean_amplitude_scale_uv": (
                float(np.mean([report.amplitude_scale_uv for report in reports])) if reports else None
            ),
            "max_clipping_fraction": (
                float(np.max([report.clipping_fraction for report in reports])) if reports else None
            ),
            "detail": [report.to_dict() for report in reports],
        }
        self.artifacts.results["preprocess"] = preprocessing_summary
        self._event("preprocessing_completed", n_recordings=len(preprocessed))
        return preprocessing_summary

    def segment_and_split(self) -> dict[str, Any]:
        """Segment into epochs and partition subjects into train/validation/test."""
        from .data.splits import assign_splits, build_epoch_index, make_subject_splits, verify_splits
        from .eeg.segmentation import segment_recordings

        preprocessed = self.artifacts.preprocessed
        if not preprocessed:
            raise RuntimeError("no preprocessed recordings; run preprocess() first")

        with self._stage("segment", n_recordings=len(preprocessed)):
            epoch_sets, segmentation_reports = segment_recordings(preprocessed, self.config)
            self.artifacts.epoch_sets = epoch_sets
            self.artifacts.segmentation_reports = segmentation_reports
        segmentation_summary = {
            "n_epoch_sets": len(epoch_sets),
            "n_epochs": int(sum(es.n_epochs for es in epoch_sets)),
            "exclusions": {
                reason: int(
                    sum(
                        item["reason"] == reason
                        for report in segmentation_reports
                        for item in report.exclusions
                    )
                )
                for reason in sorted(
                    {item["reason"] for report in segmentation_reports for item in report.exclusions}
                )
            },
            "n_windows": int(sum(report.n_windows for report in segmentation_reports)),
            "exclusion_fraction": (
                float(
                    sum(len(report.exclusions) for report in segmentation_reports)
                    / max(1, sum(report.n_windows for report in segmentation_reports))
                )
            ),
            "stage_counts": {
                stage: int(sum(es.stage_counts().get(stage, 0) for es in epoch_sets))
                for stage in self.config.dataset.stages.supported
            },
        }
        self.artifacts.results["segment"] = segmentation_summary
        self._event("epochs_segmented", **{k: v for k, v in segmentation_summary.items() if k != "exclusions"})

        # ---- subject-level splits ----------------------------------------- #
        subjects = sorted({es.subject_id for es in epoch_sets})
        if len(subjects) < 3:
            self.artifacts.errors.append(
                f"only {len(subjects)} subject(s) available; subject-level evaluation is not possible"
            )
        predictive = self.config.analysis.predictive
        with self._stage("split", n_subjects=len(subjects)):
            splits = make_subject_splits(
                subjects,
                test_fraction=float(predictive.test_fraction),
                validation_fraction=float(predictive.validation_fraction),
                seed=int(self.config.seed),
            )
            rows = assign_splits(build_epoch_index(epoch_sets), splits)
            split_report = verify_splits(rows, splits)
            self.artifacts.splits = splits
            split_path = splits.save(self.config.path("metadata") / "subject_splits.json")
        self.artifacts.results["split"] = {
            "subjects_per_split": split_report["subjects_per_split"],
            "epochs_per_split": split_report["epochs_per_split"],
            "stage_distribution": split_report["stage_distribution"],
            "verification": split_report,
            "manifest": str(split_path),
        }
        self._event("subject_splits_created", ok=split_report["ok"], **split_report["subjects_per_split"])
        if not split_report["ok"]:
            self.artifacts.errors.append("subject split verification FAILED")
        return {"segment": segmentation_summary, "split": self.artifacts.results["split"], "rows": rows}

    def extract_features(self) -> dict[str, Any]:
        """Extract the whole registered feature catalog (cached by content hash)."""
        from .data.splits import assign_splits, build_epoch_index
        from .eeg.feature_extraction import (
            add_sequence_features,
            cache_key,
            extract_features,
            feature_columns,
            load_feature_table,
            save_feature_table,
        )

        epoch_sets = self.artifacts.epoch_sets
        if not epoch_sets:
            raise RuntimeError("no epoch sets; run segment_and_split() first")
        rows = assign_splits(
            build_epoch_index(epoch_sets), self.artifacts.splits
        ) if self.artifacts.splits is not None else build_epoch_index(epoch_sets)
        # ---- feature extraction (cached) ---------------------------------- #
        registry = self._registry()
        feature_ids = registry.ids()
        key = cache_key(epoch_sets, self.config, feature_ids)
        cache_path = self.config.path("features") / f"features_{key}.csv.gz"
        table: pd.DataFrame | None = None
        if self.use_cache and cache_path.is_file():
            try:
                table = load_feature_table(self.config, name=f"features_{key}")
                logger.info("feature table cache hit (%s)", cache_path.name)
                self._event("feature_cache_hit", path=str(cache_path))
            except Exception as exc:  # noqa: BLE001 - fall back to recomputation
                logger.warning("feature cache unreadable (%s); recomputing", exc)
        if table is None:
            with self._stage("feature_extraction", n_epochs=sum(es.n_epochs for es in epoch_sets)):
                feature_table = extract_features(epoch_sets, self.config)
                save_feature_table(feature_table, self.config, name=f"features_{key}")
                table = feature_table.table
                self.artifacts.feature_definitions = feature_table.feature_definitions
                self.artifacts.feature_report = feature_table.report.to_dict()
            self.artifacts.feature_report["cache_key"] = key
        if not self.artifacts.feature_definitions:
            definitions_path = self.config.path("metadata") / f"features_{key}_definitions.json"
            if definitions_path.is_file():
                with open(definitions_path, encoding="utf-8") as handle:
                    self.artifacts.feature_definitions = json.load(handle)
            else:
                self.artifacts.feature_definitions = [f.describe() for f in registry]

        split_lookup = {f"{row['subject_id']}|{row['recording_id']}|{row['epoch_id']}": row["split"] for row in rows}
        table = table.copy()
        table["split"] = [
            split_lookup.get(f"{subject}|{recording}|{epoch}", "unknown")
            for subject, recording, epoch in zip(table.subject_id, table.recording_id, table.epoch_id)
        ]
        table = add_sequence_features(table)
        self.artifacts.feature_table = table

        features_path = self.config.path("features") / "features_with_seq.csv.gz"
        table.to_csv(features_path, index=False, compression="gzip")
        self.artifacts.results["features"] = {
            "n_epochs": int(len(table)),
            "n_features": len(feature_columns(table)),
            "failed_feature_computations": sum(
                sum(v.values()) for v in self.artifacts.feature_report.get("failures", {}).values()
            ),
            "features_with_nans": {
                k: v for k, v in self.artifacts.feature_report.get("nan_counts", {}).items() if v
            },
            "table": str(features_path),
            "report": self.artifacts.feature_report,
        }
        return self.artifacts.results["features"]

    # -------------------------------------------------------------- stage 3 #
    def statistics(self) -> dict[str, Any]:
        """Run the statistical analysis over the feature table."""
        from .statistics.engine import run_statistical_analysis

        table = self._require_table()
        features = self._features_for_statistics(table)
        train_mask = (table["split"] == "train").to_numpy() if "split" in table.columns else None
        with self._stage("statistics", n_features=len(features)):
            analysis = run_statistical_analysis(
                table, self.artifacts.config, features=features, train_mask=train_mask, write=True
            )
        summary = analysis.summary()
        summary["artifacts"] = analysis.written
        summary["errors"] = analysis.errors
        # The verification stage audits *how* selection was scoped, so the raw
        # selection report (training rows only, plus the interaction search) is
        # carried forward rather than only its headline counts.
        summary["feature_selection"] = analysis.results.get("feature_selection")
        summary["family_correction"] = analysis.results.get("family_correction")
        summary["n_features_available"] = len(self._all_features(table))
        summary["feature_subset_applied"] = len(features) < len(self._all_features(table))
        self.artifacts.results["statistics"] = summary
        self._event("statistics_completed", **{k: v for k, v in summary.items() if k != "significant_features"})
        return summary

    # -------------------------------------------------------------- stage 4 #
    def literature(self, target_papers: int | None = None, force_offline: bool = False) -> dict[str, Any]:
        """Search the literature (live) or emit the labelled offline bibliography.

        This stage never fabricates a source: when retrieval is unavailable it
        records *why*, marks every record ``unverified-offline`` and writes the
        search plan that a networked run would execute.
        """
        from .research.literature import run_literature_review

        question = (
            "Which single-channel EEG features are associated with sleep stages, and which "
            "combinations improve N1/N2/REM discrimination?"
        )
        with self._stage("literature_review"):
            review = run_literature_review(
                self.artifacts.config,
                question,
                target_papers=int(target_papers or 25),
                force_offline=force_offline,
            )
        payload = review.to_dict()
        self.artifacts.results["literature"] = {
            "mode": review.mode,
            "retrieval_available": review.retrieval_available,
            "n_papers": len(review.records),
            "retrieval_blockers": review.retrieval_blockers,
            "validation_ok": (review.validation or {}).get("ok"),
            "artifacts": review.written,
            "errors": review.errors,
        }
        self._event(
            "literature_review_completed",
            mode=review.mode,
            n_papers=len(review.records),
            retrieval_available=review.retrieval_available,
        )
        return payload

    # -------------------------------------------------------------- stage 4 #
    def evaluate(self) -> dict[str, Any]:
        """Run the predictive evaluation and subject-grouped cross-validation."""
        from .evaluation.classification import run_predictive_evaluation
        from .evaluation.cross_validation import cross_validate
        from .evaluation.classification import available_classifiers

        table = self._require_table()
        with self._stage("predictive_evaluation", n_epochs=len(table)):
            evaluation = run_predictive_evaluation(
                table, self.artifacts.config, registry=self._registry(), write=True
            )
        best = evaluation.best("balanced_accuracy", "test")
        summary: dict[str, Any] = {
            "n_models": len(evaluation.results),
            "best_test_model": best,
            "artifacts": evaluation.written,
            "errors": evaluation.errors,
        }

        # ---- cross-validation on training subjects ------------------------- #
        train = table.loc[table["split"] == "train"] if "split" in table.columns else table
        specs = available_classifiers(
            [name for name in self.config.analysis.predictive.classifiers if name != "majority"],
            seed=int(self.config.seed),
        )
        cv_rows: list[dict[str, Any]] = []
        causal = [f for f in self._all_features(table) if not f.startswith("feature_delta_norm")]
        for spec in specs[:2]:
            result = cross_validate(train, causal, spec, self.artifacts.config)
            cv_rows.append(result.to_dict())
            if result.errors:
                summary.setdefault("cv_errors", []).extend(
                    f"{spec.name}: {message}" for message in result.errors
                )
        summary["cross_validation"] = cv_rows
        cv_path = self.config.path("tables") / "cross_validation.json"
        _write_json(cv_path, cv_rows)
        summary["artifacts"]["cross_validation_json"] = str(cv_path)
        self.artifacts.results["evaluate"] = summary
        self._event("predictive_evaluation_completed", n_models=summary["n_models"])
        return summary

    # -------------------------------------------------------------- stage 5 #
    def verify(self) -> dict[str, Any]:
        """Run every verification suite (mathematical, statistical, ML, evidence)."""
        from .verification import run_verification_suite

        table = self._require_table()
        feature_sets = self._feature_sets_from_disk()
        with self._stage("verification", n_epochs=len(table)):
            report = run_verification_suite(
                table=table,
                config=self.artifacts.config,
                registry=self._registry(),
                splits=self.artifacts.splits,
                feature_selection_report=(self.artifacts.results.get("statistics") or {}).get(
                    "feature_selection"
                ),
                feature_sets=feature_sets,
            )
        self.artifacts.results["verify"] = report
        self._event(
            "verification_completed",
            ok=report.get("ok"),
            failed=report.get("failed_checks", []),
        )
        return report

    def _feature_sets_from_disk(self) -> dict[str, list[str]]:
        """The ablation feature sets, read back from the evaluation artifact.

        The leakage audit needs the *actual* feature lists each model was trained
        on in order to prove that the non-causal sequence features are absent from
        every causal ablation.
        """
        path = self.config.path("tables") / "predictive_evaluation.json"
        if not path.is_file():
            return {}
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:  # noqa: BLE001 - a missing artifact only weakens the audit
            self.artifacts.errors.append(f"predictive_evaluation.json unreadable: {exc}")
            return {}
        sets: dict[str, list[str]] = {}
        for row in payload.get("results") or []:
            name = row.get("feature_set")
            ids = row.get("feature_ids")
            if name and isinstance(ids, list):
                sets[str(name)] = [str(item) for item in ids]
        return sets

    # -------------------------------------------------------------- stage 6 #
    def report(self) -> dict[str, Any]:
        """Generate every markdown report from the saved artifacts."""
        from .reporting import generate_all_reports
        from .reporting.final_report import _existing_figures

        with self._stage("reporting"):
            written = generate_all_reports(
                config=self.artifacts.config,
                artifacts=self.artifacts,
                results=self.artifacts.results,
            )

        # ---- evidence-integrity check on the reports we just wrote --------- #
        evidence = self._validate_report_evidence(written)
        if evidence is not None:
            refreshed = generate_all_reports(
                config=self.artifacts.config,
                artifacts=self.artifacts,
                results=self.artifacts.results,
                write_claims=False,
                figures=_existing_figures(self.artifacts.config),
                only=["validation_report", "final_research_report"],
            )
            written.update(refreshed)

        self.artifacts.results["report"] = written
        self._event(
            "reports_generated",
            **{k: str(v) for k, v in written.items() if not k.startswith("_")},
        )
        return written

    def _validate_report_evidence(self, written: dict[str, Any]) -> dict[str, Any] | None:
        """Re-read the saved artifacts and prove the reports match the computation."""
        from .verification.research_validation import run_research_validation

        report_keys = (
            "literature_review",
            "feature_catalog",
            "statistical_analysis",
            "validation_report",
            "final_research_report",
        )
        report_paths = [str(written[key]) for key in report_keys if key in written]
        claims: list[dict[str, Any]] = []
        claims_path = written.get("claims_json")
        if claims_path and Path(str(claims_path)).is_file():
            with open(str(claims_path), encoding="utf-8") as handle:
                claims = list(json.load(handle).get("claims") or [])
        papers: list[dict[str, Any]] = []
        papers_path = self.config.path("metadata") / "papers.json"
        if papers_path.is_file():
            try:
                with open(papers_path, encoding="utf-8") as handle:
                    papers = list(json.load(handle).get("papers") or [])
            except Exception as exc:  # noqa: BLE001 - unreadable bibliography is reported
                self.artifacts.errors.append(f"papers.json unreadable: {type(exc).__name__}: {exc}")
        report_text = "\n".join(
            Path(path).read_text(encoding="utf-8") for path in report_paths if Path(path).is_file()
        )
        evidence = run_research_validation(
            claims=claims,
            results=self.artifacts.results,
            papers=papers,
            report_paths=report_paths,
            report_text=report_text,
            artifacts_dir=self.config.path("artifacts"),
        )
        verification = dict(self.artifacts.results.get("verify") or {})
        if not verification:
            return evidence
        sections = dict(verification.get("sections") or {})
        sections["evidence"] = {
            "ok": bool(evidence.get("ok")),
            "checks": [
                {"name": name, "passed": bool(check.get("ok")), "detail": check}
                for name, check in (evidence.get("checks") or {}).items()
            ],
            "failed_checks": list(evidence.get("failed_checks") or []),
        }
        failed = sorted({name for name, section in sections.items() if not section.get("ok", True)})
        verification.update(
            {
                "sections": sections,
                "ok": not failed,
                "failed_checks": failed,
                "n_checks": sum(len(section.get("checks") or []) for section in sections.values()),
                "n_failed_checks": sum(
                    1
                    for section in sections.values()
                    for check in (section.get("checks") or [])
                    if not check.get("passed", True)
                ),
                "report_evidence": evidence,
            }
        )
        self.artifacts.results["verify"] = verification
        try:
            _write_json(self.config.path("metadata") / "verification_report.json", verification)
        except Exception as exc:  # noqa: BLE001
            self.artifacts.errors.append(f"could not persist the merged verification report: {exc}")
        self._event(
            "report_evidence_validated",
            ok=evidence.get("ok"),
            claims=evidence.get("checks", {}).get("claims", {}).get("n_claims"),
            verified=evidence.get("checks", {}).get("claims", {}).get("n_verified"),
        )
        return evidence

    # ------------------------------------------------------------- run-all --- #
    def run_all(self, stages: Sequence[str] | None = None) -> dict[str, Any]:
        """Run the complete pipeline; returns the stage results plus the manifest."""
        requested = list(stages) if stages else [
            "initialise",
            "acquire",
            "inspect",
            "prepare",
            "literature",
            "statistics",
            "evaluate",
            "verify",
            "report",
        ]
        started = time.time()
        for stage in requested:
            handler = getattr(self, stage, None)
            if handler is None:
                raise ValueError(f"unknown pipeline stage: {stage}")
            try:
                handler()
            except Exception as exc:  # noqa: BLE001 - surface, then re-raise with context
                self.artifacts.errors.append(f"{stage}: {type(exc).__name__}: {exc}")
                self._event("stage_failed", stage=stage, error=str(exc))
                if self.artifacts.manifest is not None:
                    self.artifacts.manifest.errors.append(f"{stage}: {exc}")
                raise
        duration = time.time() - started
        if self.artifacts.manifest is not None:
            self.artifacts.manifest.finished_at = time.time()
            self.artifacts.manifest.data_origin = self.artifacts.data_origin
            self.artifacts.manifest.metrics = {
                "n_epochs": int(0 if self.artifacts.feature_table is None else len(self.artifacts.feature_table)),
                "n_features": len(self.artifacts.feature_definitions),
                "statistics": (self.artifacts.results.get("statistics") or {}).get(
                    "n_significant_after_fdr"
                ),
                "best_test_balanced_accuracy": (
                    (self.artifacts.results.get("evaluate") or {}).get("best_test_model") or {}
                ).get("balanced_accuracy"),
            }
            manifest_path = self.artifacts.manifest.save(
                self.config.path("logs") / "execution_manifest.json"
            )
        else:  # pragma: no cover - initialise always runs first
            manifest_path = None
        summary = self.artifacts.to_summary()
        summary["duration_s"] = round(duration, 2)
        summary["manifest"] = str(manifest_path) if manifest_path else None
        self._event("pipeline_finished", duration_s=round(duration, 2), **{
            k: v for k, v in summary.items() if k in {"n_epochs", "n_features", "data_origin"}
        })
        return summary

    # ------------------------------------------------------------- helpers -- #
    def _registry(self) -> Any:
        from .eeg.feature_registry import load_default_registry

        return load_default_registry()

    def _require_table(self) -> pd.DataFrame:
        if self.artifacts.feature_table is None:
            raise RuntimeError("feature table not available; run prepare() first")
        return self.artifacts.feature_table

    @staticmethod
    def _all_features(table: pd.DataFrame) -> list[str]:
        from .eeg.feature_extraction import feature_columns

        return feature_columns(table)

    def _features_for_statistics(self, table: pd.DataFrame) -> list[str]:
        features = self._all_features(table)
        limit = self.preset.get("max_features_statistics")
        if limit is None or len(features) <= int(limit):
            return features
        registry = self._registry()
        # Keep one representative per family so no family is silently dropped.
        from .statistics.feature_selection import mutual_information_ranking, select_top_k

        train = table.loc[table["split"] == "train"] if "split" in table.columns else table
        ranking = mutual_information_ranking(train.loc[:, features], train["stage"], seed=int(self.config.seed))
        chosen = select_top_k(ranking, int(limit))
        for family in registry.families():
            members = [f.feature_id for f in registry.by_family(family) if f.feature_id in features]
            if members and not any(member in chosen for member in members):
                best = max(members, key=lambda name: ranking.get(name, -np.inf))
                chosen.append(best)
        return sorted(set(chosen))

    def _stage(self, name: str, **payload: Any):
        from contextlib import contextmanager

        @contextmanager
        def _manager():
            started = time.time()
            self._event("stage_start", stage=name, **payload)
            try:
                yield
            finally:
                duration = time.time() - started
                self._event("stage_end", stage=name, duration_s=round(duration, 3))
                if self.artifacts.manifest is not None:
                    self.artifacts.manifest.add_stage(name, duration, **payload)

        return _manager()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)
