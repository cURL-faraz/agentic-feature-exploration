"""Citation validation.

Two levels of checking:

``structural``
    Deterministic, offline.  Every citation must have a title, a year, a source
    and a persistent identifier of an accepted form.  DOIs must match the
    ``10.NNNN/...`` pattern, arXiv ids the ``NNNN.NNNNN`` pattern, PMIDs must be
    digits only.

``cross_source``
    Only possible with network access: the same work found through two providers
    must agree on title and year.  Disagreements are reported, never resolved
    silently by picking a winner.

A record that was never retrieved from a live source is *never* certified; it is
listed under ``unverified`` with the reason.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from .paper_search import PaperRecord

__all__ = ["validate_record", "validate_bibliography", "cross_source_consistency"]

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
#: accepts a bare arXiv id ("2305.10351"), the "arXiv:" prefix, and an arXiv
#: abstract URL, because sources legitimately quote any of those forms
_ARXIV_RE = re.compile(
    r"^(?:https?://arxiv\.org/(?:abs|pdf)/)?(?:arXiv:)?\d{4}\.\d{4,5}(?:v\d+)?(?:\.pdf)?$",
    re.IGNORECASE,
)
_PMID_RE = re.compile(r"^\d{1,9}$")
_URL_RE = re.compile(r"^https?://\S+$")


def validate_record(record: PaperRecord) -> dict[str, Any]:
    """Structural validation of one citation."""
    problems: list[str] = []
    warnings: list[str] = []
    if not record.title.strip():
        problems.append("missing title")
    elif len(record.title.split()) < 3:
        warnings.append("title is suspiciously short")
    if record.year is None:
        warnings.append("no publication year")
    elif not 1900 <= int(record.year) <= 2100:
        problems.append(f"implausible publication year: {record.year}")
    if not record.source.strip():
        problems.append("missing source attribution")
    identifier = (record.identifier or "").strip()
    if not identifier:
        warnings.append("no persistent identifier")
    else:
        checks = {
            "doi": (_DOI_RE, "DOI"),
            "arxiv": (_ARXIV_RE, "arXiv identifier"),
            "pmid": (_PMID_RE, "PMID"),
            "url": (_URL_RE, "URL"),
            "openalex": (_URL_RE, "OpenAlex id"),
        }
        pattern, label = checks.get(record.identifier_type, (None, record.identifier_type))
        if pattern is not None and not pattern.match(identifier):
            problems.append(f"{label} does not match the expected pattern: {identifier!r}")
        elif record.identifier_type == "unknown":
            warnings.append(f"identifier type not recognised: {identifier!r}")
    if record.evidence_level == "full-text" and not record.full_text_available:
        problems.append("claims full-text evidence but no full text is available")
    if not record.authors:
        warnings.append("no author list available")
    return {
        "title": record.title,
        "identifier": identifier,
        "identifier_type": record.identifier_type,
        "source": record.source,
        "ok": not problems,
        "problems": problems,
        "warnings": warnings,
        "evidence_level": record.evidence_level,
        "verified_online": record.evidence_level not in {"unverified-offline"},
    }


def validate_bibliography(records: Sequence[PaperRecord]) -> dict[str, Any]:
    """Validate every citation and summarise the bibliography's provenance."""
    results = [validate_record(record) for record in records]
    invalid = [result for result in results if not result["ok"]]
    unverified = [result for result in results if not result["verified_online"]]
    without_identifier = [result for result in results if not result["identifier"]]
    with_identifier = [result for result in results if result["identifier"]]
    return {
        "ok": not invalid,
        "n_records": len(results),
        "n_invalid": len(invalid),
        "invalid": invalid,
        "n_unverified": len(unverified),
        "unverified": [
            {"title": result["title"], "reason": "built without live retrieval"} for result in unverified
        ],
        "n_without_persistent_identifier": len(without_identifier),
        "fraction_with_persistent_identifier": (len(with_identifier) / len(results)) if results else 0.0,
        "fraction_verified_online": (
            sum(1 for result in results if result["verified_online"]) / len(results)
        )
        if results
        else 0.0,
        "results": results,
        "note": (
            "records with evidence_level 'unverified-offline' were assembled from the project "
            "brief without live retrieval and are NOT certified citations; the literature review "
            "labels them explicitly"
        ),
    }


def cross_source_consistency(records: Sequence[PaperRecord]) -> dict[str, Any]:
    """Compare records that share an identifier across providers (needs network data)."""
    by_identifier: dict[tuple[str, str], list[PaperRecord]] = {}
    for record in records:
        if record.identifier:
            by_identifier.setdefault((record.identifier_type, record.identifier.lower()), []).append(record)
    disagreements: list[dict[str, Any]] = []
    for (identifier_type, identifier), group in by_identifier.items():
        if len(group) < 2:
            continue
        sources = {record.source for record in group}
        if len(sources) < 2:
            continue
        titles = {_normalise(record.title) for record in group if record.title}
        years = {record.year for record in group if record.year}
        if len(titles) > 1 or len(years) > 1:
            disagreements.append(
                {
                    "identifier_type": identifier_type,
                    "identifier": identifier,
                    "sources": sorted(sources),
                    "titles": sorted({record.title for record in group}),
                    "years": sorted(years),
                }
            )
    return {
        "ok": not disagreements,
        "n_identifiers_checked": len(by_identifier),
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
        "note": (
            "disagreements are reported verbatim; the pipeline never silently picks a winning "
            "title or year"
        ),
    }


def _normalise(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
