"""Orchestration of the literature stage.

Two modes, both first-class:

``live``
    The network is reachable, so every provider in :mod:`paper_search` is queried
    for every topic in the brief's search plan, results are deduplicated, open
    access texts are fetched where they exist, evidence is extracted and the
    bibliography is written to ``data/metadata/papers.json``.

``offline``
    No network.  The stage emits the brief's anchor sources at
    ``evidence_level: unverified-offline``, records the exact reason retrieval did
    not happen, and produces a literature review that is explicit about what it
    could not check.  It never invents a paper, an author list or a DOI.

Either way the *machine-readable output schema is identical*, so downstream code
([feature hypotheses, the report generator, the citation validator]) does not
need a special case.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..config import ProjectConfig
from ..data.acquisition import network_available
from ..logging_utils import get_logger
from .citation_validation import cross_source_consistency, validate_bibliography
from .evidence_extraction import annotate_relevance, extract_evidence
from .paper_retrieval import fetch_open_access_text, summarise_access
from .paper_search import PaperRecord, deduplicate, record_from_dict, search_all
from .seed_sources import SEARCH_QUERIES, SEED_SOURCES

__all__ = ["LiteratureResult", "run_literature_review"]

logger = get_logger(__name__)


@dataclass
class LiteratureResult:
    """Outcome of the literature stage."""

    mode: str
    records: list[PaperRecord] = field(default_factory=list)
    query_log: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    access: dict[str, Any] = field(default_factory=dict)
    retrieval_available: bool = False
    retrieval_blockers: list[str] = field(default_factory=list)
    written: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "retrieval_available": self.retrieval_available,
            "retrieval_blockers": self.retrieval_blockers,
            "n_papers": len(self.records),
            "query_log": self.query_log,
            "validation": self.validation,
            "access": self.access,
            "artifacts": self.written,
            "errors": self.errors,
            "papers": [record.to_dict() for record in self.records],
        }


def run_literature_review(
    config: ProjectConfig,
    research_question: str,
    target_papers: int = 25,
    providers: Sequence[str] | None = None,
    force_offline: bool = False,
    write: bool = True,
) -> LiteratureResult:
    """Run the literature stage end to end."""
    papers_path = config.path("metadata") / "papers.json"
    live = (not force_offline) and config.settings.offline_mode != "always" and network_available()

    if live:
        result = _run_live(config, research_question, target_papers, providers)
    else:
        result = _run_offline(config, research_question)
        if not force_offline:
            result.retrieval_blockers.append(
                "no network access was detected, so live bibliographic retrieval was skipped"
            )
        else:
            result.retrieval_blockers.append("live retrieval was disabled by request (force_offline)")
        if papers_path.is_file():
            result.retrieval_blockers.append(
                f"a previously retrieved bibliography exists at {papers_path} and was merged in"
            )
            try:
                with open(papers_path, encoding="utf-8") as handle:
                    cached = json.load(handle)
                cached_records = [record_from_dict(item) for item in cached.get("papers", [])]
                if cached_records:
                    result.records = deduplicate([*result.records, *cached_records])
                    result.errors.append(
                        "merged cached bibliography; check its evidence_level before citing"
                    )
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"cached bibliography unreadable: {type(exc).__name__}: {exc}")
        result = _finalise(config, result, research_question, write=write)
        return result

    return _finalise(config, result, research_question, write=write)


def _run_live(
    config: ProjectConfig,
    research_question: str,
    target_papers: int,
    providers: Sequence[str] | None,
) -> LiteratureResult:
    from .http import HttpClient

    result = LiteratureResult(mode="live", retrieval_available=True)
    cache_path = config.path("logs") / "literature_http_cache.json"
    client = HttpClient(cache_path=cache_path, min_interval_s=1.0)
    limit = max(5, min(25, target_papers // max(1, len(SEARCH_QUERIES))))
    records, query_log = search_all(SEARCH_QUERIES, providers=providers, limit_per_query=limit, client=client)
    result.query_log = query_log
    result.records = deduplicate(records)
    failures = [entry for entry in query_log if entry["status"] == "error"]
    if failures:
        result.errors.extend(
            f"{entry['provider']} ({entry['query']}): {entry.get('error')}" for entry in failures[:10]
        )
        if len(failures) == len(query_log):
            result.retrieval_blockers.append(
                "every provider call failed; the live bibliography is empty"
            )

    result.records = [extract_evidence(record) for record in result.records]
    result.records = annotate_relevance(result.records, research_question)

    full_texts: dict[str, Any] = {}
    for record in result.records:
        if not record.full_text_available:
            continue
        if len(full_texts) >= 10:  # bounded: full-text retrieval is the expensive step
            break
        document = fetch_open_access_text(record, client=client)
        full_texts[record.identifier or record.title] = document
        if document.ok:
            record.evidence_level = "full-text"
    result.access = summarise_access(result.records, full_texts)
    return result


def _run_offline(config: ProjectConfig, research_question: str) -> LiteratureResult:
    result = LiteratureResult(mode="offline", retrieval_available=False)
    result.records = [record_from_dict(dict(source)) for source in SEED_SOURCES]
    result.records = [extract_evidence(record) for record in result.records]
    result.records = annotate_relevance(result.records, research_question)
    result.access = {
        "n_records": len(result.records),
        "evidence_levels": {"unverified-offline": len(result.records)},
        "n_full_texts_retrieved": 0,
        "n_full_text_failures": 0,
        "note": (
            "offline mode: no open-access retrieval was attempted; the search plan below lists "
            "the queries to run once network access is available"
        ),
    }
    result.query_log = [
        {**entry, "provider": "not_queried_offline", "n_results": 0, "status": "skipped"}
        for entry in SEARCH_QUERIES
    ]
    return result


def _finalise(
    config: ProjectConfig,
    result: LiteratureResult,
    research_question: str,
    write: bool,
) -> LiteratureResult:
    result.validation = validate_bibliography(result.records)
    result.validation["cross_source_consistency"] = cross_source_consistency(result.records)
    result.validation["research_question"] = research_question
    if write:
        metadata = config.path("metadata")
        metadata.mkdir(parents=True, exist_ok=True)
        papers_path = metadata / "papers.json"
        with open(papers_path, "w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2, default=str)
        result.written["papers_json"] = str(papers_path)
        search_plan_path = metadata / "literature_search_plan.json"
        with open(search_plan_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "research_question": research_question,
                    "mode": result.mode,
                    "retrieval_available": result.retrieval_available,
                    "queries": SEARCH_QUERIES,
                    "query_log": result.query_log,
                    "retrieval_blockers": result.retrieval_blockers,
                },
                handle,
                indent=2,
                default=str,
            )
        result.written["search_plan_json"] = str(search_plan_path)
    return result
