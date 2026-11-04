"""Shared state for report generation.

The context loads every artifact the reports are allowed to quote, so that
**no number in a report is typed by hand**.  Each number registered through
:meth:`ReportContext.add_claim` carries the artifact file it came from and a
locator inside that file; ``verification.research_validation.validate_claims``
then re-reads the artifacts and proves the report matches the computation.

Loading is defensive on purpose: a missing optional artifact yields ``{}`` and a
recorded note rather than an exception, because the report stage runs last and
should still produce an honest (if shorter) document.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..logging_utils import get_logger
from ..verification.research_validation import Claim

__all__ = ["ReportContext", "load_report_context"]

logger = get_logger(__name__)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001 - a corrupt artifact is reported, not fatal
        logger.warning("could not read %s: %s", path, exc)
        return {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read %s: %s", path, exc)
        return pd.DataFrame()


@dataclass
class ReportContext:
    """Everything the report writers may quote, plus the claim register."""

    config: Any
    artifacts: Any
    results: dict[str, Any]
    dataset_summary: dict[str, Any] = field(default_factory=dict)
    statistical: dict[str, Any] = field(default_factory=dict)
    comparisons: pd.DataFrame = field(default_factory=pd.DataFrame)
    evaluation: dict[str, Any] = field(default_factory=dict)
    cross_validation: list[dict[str, Any]] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    literature: dict[str, Any] = field(default_factory=dict)
    catalog: dict[str, Any] = field(default_factory=dict)
    figures: dict[str, str] = field(default_factory=dict)
    claims: list[Claim] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    written: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ data #
    @property
    def data_origin(self) -> str:
        summary_origin = self.dataset_summary.get("data_origin")
        if summary_origin:
            return str(summary_origin)
        return str((self.results.get("acquire") or {}).get("origin") or "unknown")

    @property
    def synthetic(self) -> bool:
        return self.data_origin.startswith("synthetic")

    @property
    def feature_table(self) -> pd.DataFrame | None:
        return getattr(self.artifacts, "feature_table", None)

    @property
    def papers(self) -> list[dict[str, Any]]:
        return list(self.literature.get("papers") or [])

    def origin_caveat(self) -> str:
        if self.synthetic:
            return (
                "> **Data provenance.** This run used a *synthetic* polysomnography surrogate "
                f"(origin: `{self.data_origin}`), not human recordings. Every association "
                "reported below is a property of that simulation and must not be read as a "
                "statement about human sleep. Re-run with network access to use Sleep-EDF."
            )
        return (
            "> **Data provenance.** This run used a public human dataset "
            f"(origin: `{self.data_origin}`). Conclusions are limited to the recorded "
            "population, montage and annotation protocol."
        )

    # ---------------------------------------------------------------- claims #
    def add_claim(
        self,
        claim_id: str,
        description: str,
        value: Any,
        locator: str,
        source: str,
        level: str = "established",
        unit: str = "",
        tolerance: float = 1e-6,
    ) -> Claim:
        claim = Claim(
            claim_id=claim_id,
            description=description,
            value=value,
            locator=locator,
            source=source,
            level=level,
            unit=unit,
            tolerance=tolerance,
        )
        self.claims.append(claim)
        return claim

    def save_claims(self) -> Path:
        path = self.config.path("tables") / "claims.json"
        _write_json(
            path,
            {
                "n_claims": len(self.claims),
                "claims": [claim.to_dict() for claim in self.claims],
                "notes": list(self.notes),
            },
        )
        self.written["claims_json"] = str(path)
        return path


def _build_catalog(artifacts: Any, results: Mapping[str, Any]) -> dict[str, Any]:
    definitions = [dict(item) for item in (getattr(artifacts, "feature_definitions", None) or [])]
    if not definitions:
        definitions = [
            dict(item)
            for item in ((results.get("features") or {}).get("definitions") or [])
        ]
    by_novelty: dict[str, int] = {}
    by_family: dict[str, int] = {}
    by_validation: dict[str, int] = {}
    for definition in definitions:
        for key, bucket in (
            ("novelty_status", by_novelty),
            ("family", by_family),
            ("validation_status", by_validation),
        ):
            label = str(definition.get(key, "unknown"))
            bucket[label] = bucket.get(label, 0) + 1
    return {
        "n_features": len(definitions),
        "by_novelty_status": by_novelty,
        "by_family": by_family,
        "by_validation_status": by_validation,
        "features": definitions,
    }


def load_report_context(config: Any, artifacts: Any, results: Mapping[str, Any]) -> ReportContext:
    """Assemble the report context from saved artifacts."""
    results = dict(results or {})
    context = ReportContext(config=config, artifacts=artifacts, results=results)
    tables = config.path("tables")
    metadata = config.path("metadata")

    dataset_summary = _read_json(metadata / "dataset_summary.json")
    if not isinstance(dataset_summary, dict) or not dataset_summary:
        dataset_summary = {k: v for k, v in (results.get("inspect") or {}).items() if k != "written"}
        if dataset_summary:
            context.notes.append("dataset summary was reconstructed from the in-memory stage result")
    _write_json(tables / "dataset_summary.json", dataset_summary)
    context.dataset_summary = dict(dataset_summary)
    context.written["dataset_summary_json"] = str(tables / "dataset_summary.json")

    statistical = _read_json(tables / "statistical_results.json")
    context.statistical = statistical if isinstance(statistical, dict) else {}
    context.comparisons = _read_csv(tables / "feature_stage_comparisons.csv")
    evaluation = _read_json(tables / "predictive_evaluation.json")
    context.evaluation = evaluation if isinstance(evaluation, dict) else {}
    cross_validation = _read_json(tables / "cross_validation.json")
    if isinstance(cross_validation, list):  # pragma: no cover - legacy layout
        context.cross_validation = list(cross_validation)
    else:
        context.cross_validation = list(cross_validation.get("value") or [])

    verification = results.get("verify") or _read_json(metadata / "verification_report.json")
    context.verification = dict(verification) if isinstance(verification, dict) else {}
    _write_json(tables / "verification_snapshot.json", context.verification)
    context.written["verification_snapshot_json"] = str(tables / "verification_snapshot.json")

    literature = _read_json(metadata / "papers.json")
    context.literature = literature if isinstance(literature, dict) else {}
    # Always snapshot the bibliography - even when it is empty - so the literature
    # claims ("n papers") always resolve against a real artifact instead of
    # dangling when retrieval produced nothing.
    context.literature.setdefault("n_papers", len(context.literature.get("papers") or []))
    _write_json(tables / "papers.json", context.literature)
    context.written["papers_snapshot_json"] = str(tables / "papers.json")
    context.catalog = _build_catalog(artifacts, results)
    _write_json(tables / "feature_catalog.json", context.catalog)
    context.written["feature_catalog_json"] = str(tables / "feature_catalog.json")
    return context
