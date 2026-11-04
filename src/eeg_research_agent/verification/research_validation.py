"""Evidence-integrity verification.

The brief's hardest requirement is that an LLM must never be able to put an
invented number or an invented citation into a report.  The mechanism used here
is deliberately mechanical:

1. Report generation emits a machine-readable **claims file**: every number that
   appears in a report is registered with the artifact path and key it came from.
2. :func:`validate_claims` walks that file and re-reads the artifact.  A claim
   whose value does not exist in the saved computation output is a hard failure.
3. :func:`validate_citations` refuses to certify a bibliographic record that has
   not been retrieved from a real source, and marks records built while offline
   as explicitly unverified.
4. :func:`validate_novelty_labels` fails if a speculative feature is described
   with the language of an established finding.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "Claim",
    "validate_claims",
    "validate_citations",
    "validate_novelty_labels",
    "validate_report_files",
    "run_research_validation",
]

REQUIRED_CITATION_FIELDS = ("title", "year", "source", "identifier_type", "identifier")
OFFLINE_EVIDENCE_LEVELS = {"unverified-offline", "metadata-only", "abstract-only"}
ESTABLISHED_LANGUAGE = (
    "established",
    "well documented",
    "known biomarker",
    "confirmed",
    "validated biomarker",
    "proven",
)


@dataclass
class Claim:
    """One number asserted by a report, with the artifact it came from."""

    claim_id: str
    description: str
    value: Any
    locator: str
    source: str
    level: str = "established"
    unit: str = ""
    tolerance: float = 1e-6

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "description": self.description,
            "value": _plain(self.value),
            "locator": self.locator,
            "source": self.source,
            "level": self.level,
            "unit": self.unit,
            "tolerance": self.tolerance,
        }


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return "<missing>"


_MISSING = _Missing()


def _plain(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _resolve(payload: Any, locator: str) -> Any:
    """Resolve a dotted/bracketed locator such as ``a.b[0].c`` against a payload."""
    if not locator:
        return payload
    tokens = re.findall(r"[^.\[\]]+|\[\d+\]", locator)
    current = payload
    for token in tokens:
        if token.startswith("["):
            if not isinstance(current, (list, tuple)):
                return _MISSING
            index = int(token[1:-1])
            if index >= len(current):
                return _MISSING
            current = current[index]
            continue
        if isinstance(current, dict):
            if token not in current:
                return _MISSING
            current = current[token]
            continue
        return _MISSING
    return current


def validate_claims(
    claims: Sequence[Claim | dict[str, Any]],
    results: dict[str, Any],
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Check every report claim against the saved computation output."""
    claims = list(claims)
    failures: list[dict[str, Any]] = []
    verified = 0
    for raw in claims:
        claim = raw if isinstance(raw, Claim) else Claim(**raw)
        payload: Any = results
        path_note = "in-memory results"
        if claim.source and artifacts_dir is not None:
            candidate = Path(artifacts_dir) / claim.source
            if not candidate.is_file():
                candidate = Path(claim.source)
            if candidate.is_file():
                try:
                    with open(candidate, encoding="utf-8") as handle:
                        payload = json.load(handle)
                    path_note = str(candidate)
                except Exception as exc:  # noqa: BLE001 - an unreadable artifact invalidates the claim
                    failures.append(
                        {
                            "claim_id": claim.claim_id,
                            "reason": f"source artifact unreadable: {type(exc).__name__}",
                            "source": str(candidate),
                        }
                    )
                    continue
        observed = _resolve(payload, claim.locator)
        if observed is _MISSING:
            failures.append(
                {
                    "claim_id": claim.claim_id,
                    "reason": "locator not found in the saved computation output",
                    "locator": claim.locator,
                    "source": path_note,
                }
            )
            continue
        if isinstance(claim.value, (int, float)) and isinstance(observed, (int, float)):
            matches = bool(
                np.isclose(float(claim.value), float(observed), rtol=0.0, atol=float(claim.tolerance))
            )
        else:
            matches = claim.value == observed
        if matches:
            verified += 1
        else:
            failures.append(
                {
                    "claim_id": claim.claim_id,
                    "reason": "value mismatch",
                    "claimed": _plain(claim.value),
                    "observed": _plain(observed),
                    "locator": claim.locator,
                    "source": path_note,
                }
            )
    return {
        "ok": not failures,
        "n_claims": len(claims),
        "n_verified": verified,
        "failures": failures,
    }


def validate_citations(papers: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Check that every cited paper is traceable and honestly labelled.

    A record that is *presented as a verified source* must be complete: that is a
    hard requirement, and an incomplete verified record fails the check.  A record
    whose ``evidence_level`` says it was never retrieved (``unverified-offline``,
    ``metadata-only``, ``abstract-only``) is held to the weaker standard of being
    *honestly labelled*: its gaps are reported so a reader can see exactly what is
    missing, but the pipeline refuses to invent a year, DOI or author list to
    satisfy a schema.  Failing on those would make the check permanently red for
    offline runs and therefore useless.
    """
    papers = list(papers)
    incomplete: list[dict[str, Any]] = []
    incomplete_unverified: list[dict[str, Any]] = []
    unverified: list[str] = []
    for index, paper in enumerate(papers):
        missing = [field for field in REQUIRED_CITATION_FIELDS if not paper.get(field)]
        identifier = str(paper.get("identifier") or "")
        if paper.get("identifier_type") == "doi" and identifier and not re.match(r"^10\.\d{4,9}/", identifier):
            missing.append("identifier (DOI does not match the 10.NNNN/... pattern)")
        if missing:
            record = {
                "index": index,
                "title": paper.get("title"),
                "evidence_level": paper.get("evidence_level"),
                "missing": sorted(set(missing)),
            }
            offline = str(paper.get("evidence_level", "")) in OFFLINE_EVIDENCE_LEVELS
            (incomplete_unverified if offline else incomplete).append(record)
        if str(paper.get("evidence_level", "")) in OFFLINE_EVIDENCE_LEVELS:
            unverified.append(str(paper.get("title") or paper.get("identifier") or f"paper_{index}"))
    return {
        "ok": not incomplete,
        "n_papers": len(papers),
        "n_incomplete": len(incomplete),
        "incomplete": incomplete,
        "n_incomplete_unverified": len(incomplete_unverified),
        "incomplete_unverified": incomplete_unverified,
        "n_unverified_offline": len(unverified),
        "unverified_offline_titles": unverified,
        "note": (
            "records whose evidence_level is unverified-offline/metadata-only were built without "
            "live retrieval (or from abstracts alone): their metadata gaps are listed under "
            "`incomplete_unverified` and they must never be presented as fully verified sources. "
            "Only incomplete records that claim a verified evidence level fail this check."
        ),
    }


def validate_novelty_labels(
    claims: Sequence[Claim | dict[str, Any]], text: str = ""
) -> dict[str, Any]:
    """Speculative findings must not be described with established-result language."""
    offenders: list[dict[str, Any]] = []
    lowered = text.lower()
    for raw in claims:
        claim = raw if isinstance(raw, Claim) else Claim(**raw)
        if claim.level not in {"novel-hypothesis", "exploratory"}:
            continue
        description = (claim.description or "").lower()
        if not description or description not in lowered:
            continue
        index = lowered.index(description)
        window = lowered[max(0, index - 160) : index + 200]
        for phrase in ESTABLISHED_LANGUAGE:
            if phrase in window:
                offenders.append(
                    {
                        "claim_id": claim.claim_id,
                        "level": claim.level,
                        "offending_phrase": phrase,
                        "context": window.strip()[:200],
                    }
                )
                break
    return {
        "ok": not offenders,
        "n_offenders": len(offenders),
        "offenders": offenders,
        "note": (
            "only claim descriptions that appear verbatim in the report text are checked; this "
            "check is conservative and finding nothing is not proof of careful wording"
        ),
    }


def validate_report_files(report_paths: Iterable[str | Path]) -> dict[str, Any]:
    """Every report must exist, be non-trivial, and contain no obvious placeholders."""
    problems: list[dict[str, Any]] = []
    checked: list[str] = []
    for path in report_paths:
        candidate = Path(path)
        if not candidate.is_file():
            problems.append({"path": str(candidate), "reason": "missing"})
            continue
        text = candidate.read_text(encoding="utf-8")
        checked.append(str(candidate))
        if len(text.strip()) < 200:
            problems.append({"path": str(candidate), "reason": "suspiciously short"})
        lowered = text.lower()
        for placeholder in ("todo", "fixme", "{{", "xxx", "lorem ipsum", "<placeholder>"):
            if placeholder in lowered:
                problems.append(
                    {"path": str(candidate), "reason": f"contains placeholder {placeholder!r}"}
                )
        if "#" not in text:
            problems.append({"path": str(candidate), "reason": "no markdown headings"})
    return {"ok": not problems, "n_checked": len(checked), "problems": problems, "checked": checked}


def run_research_validation(
    claims: Sequence[Claim | dict[str, Any]] = (),
    results: dict[str, Any] | None = None,
    papers: Sequence[dict[str, Any]] = (),
    report_paths: Sequence[str | Path] = (),
    report_text: str = "",
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run every evidence-integrity check."""
    checks: dict[str, Any] = {}
    checks["claims"] = (
        validate_claims(claims, results or {}, artifacts_dir=artifacts_dir)
        if claims
        else {"ok": True, "note": "no claims registered"}
    )
    checks["citations"] = (
        validate_citations(papers) if papers else {"ok": True, "note": "no papers retrieved"}
    )
    checks["novelty_labelling"] = validate_novelty_labels(claims, report_text)
    checks["report_files"] = (
        validate_report_files(report_paths)
        if report_paths
        else {"ok": True, "note": "no report files supplied"}
    )
    return {
        "ok": all(bool(check.get("ok")) for check in checks.values()),
        "checks": checks,
        "failed_checks": [name for name, check in checks.items() if not check.get("ok")],
    }
