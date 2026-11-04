"""Open-access full-text retrieval.

Only genuinely open resources are used: arXiv (preprints), Europe PMC's open
subset, and Unpaywall's index of legal open-access copies.  Nothing here attempts
to bypass a paywall; when no open copy exists the record keeps
``evidence_level: abstract-only`` and the review says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..logging_utils import get_logger
from .http import HttpClient
from .paper_search import PaperRecord

__all__ = ["FullText", "fetch_open_access_text", "summarise_access"]

logger = get_logger(__name__)

#: A document longer than this is truncated; the point is evidence extraction,
#: not archiving somebody else's paper.
MAX_CHARS = 200_000


@dataclass
class FullText:
    """Retrieved full text plus its provenance."""

    paper_identifier: str
    url: str
    text: str
    source: str
    truncated: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.text.strip()) and self.error is None


def _strip_html(text: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_open_access_text(
    record: PaperRecord, client: HttpClient | None = None, contact_email: str = "reproducibility@example.invalid"
) -> FullText:
    """Try, in order, arXiv, Europe PMC and Unpaywall for an open copy."""
    client = client or HttpClient()
    if record.identifier_type == "arxiv" and record.identifier:
        response = client.get_text(f"https://arxiv.org/abs/{record.identifier}")
        if response.ok and isinstance(response.payload, str):
            text = _strip_html(response.payload)
            return FullText(
                paper_identifier=record.identifier,
                url=f"https://arxiv.org/abs/{record.identifier}",
                text=text[:MAX_CHARS],
                source="arxiv-abstract-page",
                truncated=len(text) > MAX_CHARS,
                metadata={"note": "arXiv abstract page; the PDF body was not parsed"},
            )
        return FullText(
            paper_identifier=record.identifier,
            url=f"https://arxiv.org/abs/{record.identifier}",
            text="",
            source="arxiv",
            error=response.error or f"HTTP {response.status}",
        )

    pmcid = (record.extra or {}).get("pmcid")
    if pmcid:
        response = client.get_text(
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
        )
        if response.ok and isinstance(response.payload, str):
            text = _strip_html(response.payload)
            return FullText(
                paper_identifier=record.identifier,
                url=f"https://europepmc.org/article/PMC/{pmcid}",
                text=text[:MAX_CHARS],
                source="europe-pmc-open-access",
                truncated=len(text) > MAX_CHARS,
            )

    if record.identifier_type == "doi" and record.identifier:
        response = client.get_json(f"https://api.unpaywall.org/v2/{record.identifier}", params={"email": contact_email})
        if response.ok and isinstance(response.payload, dict):
            best = response.payload.get("best_oa_location") or {}
            url = best.get("url_for_pdf") or best.get("url")
            if url:
                document = client.get_text(url)
                if document.ok and isinstance(document.payload, str):
                    text = _strip_html(document.payload)
                    return FullText(
                        paper_identifier=record.identifier,
                        url=url,
                        text=text[:MAX_CHARS],
                        source=f"unpaywall:{best.get('host_type', 'unknown')}",
                        truncated=len(text) > MAX_CHARS,
                        metadata={"license": best.get("license")},
                    )
            return FullText(
                paper_identifier=record.identifier,
                url=record.url or "",
                text="",
                source="unpaywall",
                error="no open-access location reported by Unpaywall",
            )
        return FullText(
            paper_identifier=record.identifier,
            url=record.url or "",
            text="",
            source="unpaywall",
            error=response.error or f"HTTP {response.status}",
        )

    return FullText(
        paper_identifier=record.identifier or record.title,
        url=record.url or "",
        text="",
        source="none",
        error="no open-access route available for this identifier type",
    )


def summarise_access(records: list[PaperRecord], full_texts: dict[str, FullText]) -> dict[str, Any]:
    """How much of the bibliography was actually readable."""
    by_level: dict[str, int] = {}
    for record in records:
        by_level[record.evidence_level] = by_level.get(record.evidence_level, 0) + 1
    return {
        "n_records": len(records),
        "evidence_levels": by_level,
        "n_full_texts_retrieved": sum(1 for text in full_texts.values() if text.ok),
        "n_full_text_failures": sum(1 for text in full_texts.values() if not text.ok),
        "note": (
            "records without an open-access copy stay at abstract-only/metadata-only evidence "
            "level; the review states this explicitly for each claim it makes"
        ),
    }
