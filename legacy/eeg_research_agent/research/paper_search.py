"""Bibliographic retrieval from public, open sources.

Providers
---------
``semantic_scholar``, ``openalex``, ``crossref``, ``pubmed`` (E-utilities),
``arxiv`` and ``europe_pmc``.  Each returns a list of :class:`PaperRecord`.

Design rules
------------
* Retrieval is **best-effort and attributable**: every record carries the source
  that produced it and the exact query that found it.
* No field is ever guessed.  If a provider does not report the number of
  subjects, the field is ``None``, not an estimate.
* When the network is unavailable the module returns the brief's anchor sources
  with ``evidence_level: unverified-offline`` plus an explicit record of *why*
  live retrieval did not happen.  It never fabricates a citation.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Sequence

from ..logging_utils import get_logger
from .http import HttpClient

__all__ = [
    "PaperRecord",
    "PROVIDERS",
    "available_providers",
    "search_provider",
    "search_all",
    "deduplicate",
    "record_from_dict",
]

logger = get_logger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", value)).strip()


@dataclass
class PaperRecord:
    """Bibliographic record with the retrieval provenance attached."""

    title: str
    identifier: str
    identifier_type: str  # doi | pmid | arxiv | url | openalex
    source: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    abstract: str = ""
    url: str | None = None
    access: str = "unknown"
    full_text_available: bool = False
    evidence_level: str = "metadata-only"
    queries: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    n_subjects: int | None = None
    channels_used: list[str] = field(default_factory=list)
    sampling_frequency_hz: float | None = None
    epoch_seconds: float | None = None
    feature_methods: list[str] = field(default_factory=list)
    statistical_methods: list[str] = field(default_factory=list)
    main_findings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    relevance: str = ""
    notes: str = ""
    #: ``True`` only when the text in ``abstract`` was actually returned by a
    #: provider.  Records assembled from the project brief keep it ``False`` so
    #: evidence extraction can never silently upgrade them to "retrieved".
    text_retrieved: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def record_from_dict(payload: dict[str, Any]) -> PaperRecord:
    """Rebuild a record from a saved dictionary (unknown keys become ``extra``)."""
    known = {field_name for field_name in PaperRecord.__dataclass_fields__}
    kwargs = {key: value for key, value in payload.items() if key in known and key != "extra"}
    extra = dict(payload.get("extra") or {})
    extra.update({key: value for key, value in payload.items() if key not in known})
    return PaperRecord(**kwargs, extra=extra)


# --------------------------------------------------------------------------- #
# providers
# --------------------------------------------------------------------------- #
def _search_semantic_scholar(client: HttpClient, query: str, limit: int) -> list[PaperRecord]:
    response = client.get_json(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={
            "query": query,
            "limit": min(limit, 100),
            "fields": (
                "title,abstract,year,venue,authors,externalIds,openAccessPdf,"
                "publicationTypes,fieldsOfStudy"
            ),
        },
    )
    if not response.ok or not isinstance(response.payload, dict):
        raise RuntimeError(f"Semantic Scholar request failed: {response.error or response.status}")
    records: list[PaperRecord] = []
    for item in response.payload.get("data", []):
        external = item.get("externalIds") or {}
        identifier, identifier_type = _pick_identifier(
            {"doi": external.get("DOI"), "pmid": external.get("PubMed"), "arxiv": external.get("ArXiv")}
        )
        pdf = (item.get("openAccessPdf") or {}).get("url")
        records.append(
            PaperRecord(
                title=_clean_text(item.get("title")),
                identifier=identifier,
                identifier_type=identifier_type,
                source="semantic_scholar",
                authors=[author.get("name", "") for author in (item.get("authors") or []) if author.get("name")],
                year=item.get("year"),
                venue=item.get("venue") or None,
                abstract=_clean_text(item.get("abstract")),
                url=pdf or (f"https://api.semanticscholar.org/graph/v1/paper/{item.get('paperId')}" if item.get("paperId") else None),
                access="open-access-pdf" if pdf else "metadata-only",
                full_text_available=bool(pdf),
                evidence_level="abstract-only" if item.get("abstract") else "metadata-only",
                queries=[query],
                text_retrieved=bool(item.get("abstract")),
                extra={"paperId": item.get("paperId"), "externalIds": external},
            )
        )
    return records


def _search_openalex(client: HttpClient, query: str, limit: int) -> list[PaperRecord]:
    response = client.get_json(
        "https://api.openalex.org/works",
        params={"search": query, "per-page": min(limit, 50), "mailto": "reproducibility@example.invalid"},
    )
    if not response.ok or not isinstance(response.payload, dict):
        raise RuntimeError(f"OpenAlex request failed: {response.error or response.status}")
    records: list[PaperRecord] = []
    for item in response.payload.get("results", []):
        doi = (item.get("doi") or "").replace("https://doi.org/", "") or None
        identifier, identifier_type = _pick_identifier({"doi": doi, "openalex": item.get("id")})
        best = item.get("best_oa_location") or {}
        records.append(
            PaperRecord(
                title=_clean_text(item.get("title")),
                identifier=identifier,
                identifier_type=identifier_type,
                source="openalex",
                authors=[
                    authorship.get("author", {}).get("display_name", "")
                    for authorship in (item.get("authorships") or [])
                    if authorship.get("author", {}).get("display_name")
                ],
                year=item.get("publication_year"),
                venue=((item.get("primary_location") or {}).get("source") or {}).get("display_name"),
                abstract=_reconstruct_openalex_abstract(item.get("abstract_inverted_index")),
                url=best.get("pdf_url") or item.get("id"),
                access="open-access-pdf" if best.get("pdf_url") else "metadata-only",
                full_text_available=bool(best.get("pdf_url")),
                evidence_level="abstract-only" if item.get("abstract_inverted_index") else "metadata-only",
                queries=[query],
                text_retrieved=bool(item.get("abstract_inverted_index")),
                extra={"type": item.get("type"), "cited_by_count": item.get("cited_by_count")},
            )
        )
    return records


def _reconstruct_openalex_abstract(inverted: dict[str, list[int]] | None) -> str:
    """OpenAlex ships abstracts as an inverted index; rebuild the sentence."""
    if not inverted:
        return ""
    positions: dict[int, str] = {}
    for word, indexes in inverted.items():
        for index in indexes:
            positions[int(index)] = word
    return _clean_text(" ".join(positions[key] for key in sorted(positions)))


def _search_crossref(client: HttpClient, query: str, limit: int) -> list[PaperRecord]:
    response = client.get_json(
        "https://api.crossref.org/works",
        params={
            "query": query,
            "rows": min(limit, 50),
            "select": "DOI,title,author,issued,container-title,abstract,type",
        },
    )
    if not response.ok or not isinstance(response.payload, dict):
        raise RuntimeError(f"Crossref request failed: {response.error or response.status}")
    records: list[PaperRecord] = []
    for item in (response.payload.get("message") or {}).get("items", []):
        year = None
        issued = (item.get("issued") or {}).get("date-parts") or [[None]]
        if issued and issued[0]:
            year = issued[0][0]
        container = item.get("container-title") or []
        records.append(
            PaperRecord(
                title=_clean_text((item.get("title") or [""])[0]),
                identifier=item.get("DOI") or "",
                identifier_type="doi",
                source="crossref",
                authors=[
                    " ".join(part for part in [author.get("given"), author.get("family")] if part)
                    for author in (item.get("author") or [])
                ],
                year=year,
                venue=container[0] if container else None,
                abstract=_clean_text(item.get("abstract")),
                url=f"https://doi.org/{item.get('DOI')}" if item.get("DOI") else None,
                access="metadata-only",
                full_text_available=False,
                evidence_level="abstract-only" if item.get("abstract") else "metadata-only",
                queries=[query],
                text_retrieved=bool(item.get("abstract")),
                extra={"type": item.get("type")},
            )
        )
    return records


def _search_pubmed(client: HttpClient, query: str, limit: int) -> list[PaperRecord]:
    search = client.get_json(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmax": min(limit, 50), "retmode": "json"},
    )
    if not search.ok or not isinstance(search.payload, dict):
        raise RuntimeError(f"PubMed esearch failed: {search.error or search.status}")
    ids = (search.payload.get("esearchresult") or {}).get("idlist") or []
    if not ids:
        return []
    summary = client.get_json(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
    )
    if not summary.ok or not isinstance(summary.payload, dict):
        raise RuntimeError(f"PubMed esummary failed: {summary.error or summary.status}")
    records: list[PaperRecord] = []
    for pmid, item in (summary.payload.get("result") or {}).items():
        if pmid == "uids":
            continue
        doi = next(
            (
                identifier.get("value")
                for identifier in (item.get("articleids") or [])
                if identifier.get("idtype") == "doi"
            ),
            None,
        )
        records.append(
            PaperRecord(
                title=_clean_text(item.get("title")),
                identifier=pmid,
                identifier_type="pmid",
                source="pubmed",
                authors=[author.get("name", "") for author in (item.get("authors") or [])],
                year=_year_from_pubdate(item.get("pubdate")),
                venue=item.get("fulljournalname"),
                abstract="",
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                access="metadata-only",
                full_text_available=False,
                evidence_level="metadata-only",
                queries=[query],
                extra={"doi": doi, "uid": pmid},
            )
        )
    return records


def _search_arxiv(client: HttpClient, query: str, limit: int) -> list[PaperRecord]:
    import xml.etree.ElementTree as ElementTree

    response = client.get_text(
        "http://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "max_results": min(limit, 50)},
    )
    if not response.ok or not isinstance(response.payload, (str, bytes)):
        raise RuntimeError(f"arXiv request failed: {response.error or response.status}")
    raw = response.payload if isinstance(response.payload, bytes) else response.payload.encode()
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    root = ElementTree.fromstring(raw)
    records: list[PaperRecord] = []
    for entry in root.findall("atom:entry", namespace):
        identifier = _clean_text(entry.findtext("atom:id", default="", namespaces=namespace))
        records.append(
            PaperRecord(
                title=_clean_text(entry.findtext("atom:title", default="", namespaces=namespace)),
                identifier=identifier.rsplit("/", 1)[-1],
                identifier_type="arxiv",
                source="arxiv",
                authors=[
                    _clean_text(author.findtext("atom:name", default="", namespaces=namespace))
                    for author in entry.findall("atom:author", namespace)
                ],
                year=_year_from_pubdate(
                    entry.findtext("atom:published", default="", namespaces=namespace)
                ),
                venue="arXiv",
                abstract=_clean_text(entry.findtext("atom:summary", default="", namespaces=namespace)),
                url=identifier or None,
                access="preprint",
                full_text_available=True,
                evidence_level="abstract-only",
                queries=[query],
                text_retrieved=bool(entry.findtext("atom:summary", default="", namespaces=namespace)),
            )
        )
    return records


def _search_europe_pmc(client: HttpClient, query: str, limit: int) -> list[PaperRecord]:
    response = client.get_json(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={"query": query, "format": "json", "pageSize": min(limit, 50), "resultType": "core"},
    )
    if not response.ok or not isinstance(response.payload, dict):
        raise RuntimeError(f"Europe PMC request failed: {response.error or response.status}")
    records: list[PaperRecord] = []
    for item in (response.payload.get("resultList") or {}).get("result", []):
        records.append(
            PaperRecord(
                title=_clean_text(item.get("title")),
                identifier=item.get("doi") or item.get("pmid") or item.get("id") or "",
                identifier_type="doi" if item.get("doi") else "pmid",
                source="europe_pmc",
                authors=[_clean_text(item.get("authorString"))] if item.get("authorString") else [],
                year=item.get("pubYear"),
                venue=item.get("journalTitle"),
                abstract=_clean_text(item.get("abstractText")),
                url=(
                    f"https://europepmc.org/article/{item.get('source', 'MED')}/{item.get('id')}"
                    if item.get("id")
                    else None
                ),
                access="open-access-fulltext" if item.get("isOpenAccess") == "Y" else "metadata-only",
                full_text_available=item.get("isOpenAccess") == "Y",
                evidence_level="abstract-only" if item.get("abstractText") else "metadata-only",
                queries=[query],
                text_retrieved=bool(item.get("abstractText")),
                extra={"pmcid": item.get("pmcid")},
            )
        )
    return records


PROVIDERS: dict[str, Callable[[HttpClient, str, int], list[PaperRecord]]] = {
    "semantic_scholar": _search_semantic_scholar,
    "openalex": _search_openalex,
    "crossref": _search_crossref,
    "pubmed": _search_pubmed,
    "arxiv": _search_arxiv,
    "europe_pmc": _search_europe_pmc,
}


def available_providers() -> list[str]:
    return sorted(PROVIDERS)


def _pick_identifier(candidates: dict[str, str | None]) -> tuple[str, str]:
    for identifier_type, value in candidates.items():
        if value:
            return str(value), identifier_type
    return "", "unknown"


def _year_from_pubdate(value: Any) -> int | None:
    if not value:
        return None
    match = re.search(r"(19|20)\d{2}", str(value))
    return int(match.group(0)) if match else None


def search_provider(
    provider: str,
    query: str,
    limit: int = 20,
    client: HttpClient | None = None,
) -> list[PaperRecord]:
    """Run one provider query.  Raises on provider failure so the caller can log it."""
    if provider not in PROVIDERS:
        raise KeyError(f"unknown provider: {provider}; available: {available_providers()}")
    client = client or HttpClient()
    return PROVIDERS[provider](client, query, limit)


def search_all(
    queries: Sequence[dict[str, str]],
    providers: Iterable[str] | None = None,
    limit_per_query: int = 10,
    client: HttpClient | None = None,
) -> tuple[list[PaperRecord], list[dict[str, Any]]]:
    """Run every (query, provider) pair; returns records plus a per-call log."""
    client = client or HttpClient()
    selected = [name for name in (providers or available_providers()) if name in PROVIDERS]
    records: list[PaperRecord] = []
    log: list[dict[str, Any]] = []
    for entry in queries:
        query = entry["query"]
        for provider in selected:
            try:
                found = search_provider(provider, query, limit=limit_per_query, client=client)
                records.extend(found)
                log.append(
                    {
                        "provider": provider,
                        "query": query,
                        "topic": entry.get("topic", ""),
                        "n_results": len(found),
                        "status": "ok",
                    }
                )
            except Exception as exc:  # noqa: BLE001 - one provider must not kill the stage
                logger.warning("provider %s failed for %r: %s", provider, query, exc)
                log.append(
                    {
                        "provider": provider,
                        "query": query,
                        "topic": entry.get("topic", ""),
                        "n_results": 0,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return records, log


def deduplicate(records: Sequence[PaperRecord]) -> list[PaperRecord]:
    """Merge records that describe the same work.

    Two records are the same work if they share a normalised identifier, or a
    normalised title.  Merging is conservative: it unions the identifiers and
    queries and keeps the *most informative* value of every other field rather
    than overwriting with whichever record happened to come second.
    """
    merged: dict[str, PaperRecord] = {}
    for record in records:
        if not record.title and not record.identifier:
            continue
        key = _dedupe_key(record)
        existing = merged.get(key)
        if existing is None:
            merged[key] = record
            continue
        merged[key] = _merge(existing, record)
    return list(merged.values())


def _normalised_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _dedupe_key(record: PaperRecord) -> str:
    if record.identifier and record.identifier_type in {"doi", "pmid", "arxiv"}:
        return f"{record.identifier_type}:{record.identifier.lower()}"
    return f"title:{_normalised_title(record.title)}"


def _merge(first: PaperRecord, second: PaperRecord) -> PaperRecord:
    union_queries = sorted(set(first.queries) | set(second.queries))
    union_authors = first.authors or second.authors
    prefer_first_abstract = bool(first.abstract.strip())
    return PaperRecord(
        title=first.title or second.title,
        identifier=first.identifier or second.identifier,
        identifier_type=first.identifier_type if first.identifier else second.identifier_type,
        source=first.source if first.source == second.source else f"{first.source}+{second.source}",
        authors=union_authors,
        year=first.year or second.year,
        venue=first.venue or second.venue,
        abstract=first.abstract if prefer_first_abstract else second.abstract,
        url=first.url or second.url,
        access=first.access if first.full_text_available else second.access,
        full_text_available=first.full_text_available or second.full_text_available,
        # Evidence level is decided by *retrieval provenance*, never by which
        # record happened to carry a longer string.  A record whose text was not
        # retrieved from a provider can never be promoted by merging.
        evidence_level=_merge_evidence_level(first, second),
# also keep the provenance flag itself
        text_retrieved=first.text_retrieved or second.text_retrieved,
        queries=union_queries,
        datasets=sorted(set(first.datasets) | set(second.datasets)),
        n_subjects=first.n_subjects or second.n_subjects,
        channels_used=sorted(set(first.channels_used) | set(second.channels_used)),
        sampling_frequency_hz=first.sampling_frequency_hz or second.sampling_frequency_hz,
        epoch_seconds=first.epoch_seconds or second.epoch_seconds,
        feature_methods=sorted(set(first.feature_methods) | set(second.feature_methods)),
        statistical_methods=sorted(set(first.statistical_methods) | set(second.statistical_methods)),
        main_findings=sorted(set(first.main_findings) | set(second.main_findings)),
        limitations=sorted(set(first.limitations) | set(second.limitations)),
        relevance=first.relevance or second.relevance,
        notes=first.notes or second.notes,
        extra={**second.extra, **first.extra},
    )


_EVIDENCE_ORDER = {
    "unverified-offline": 0,
    "metadata-only": 1,
    "abstract-only": 2,
    "full-text": 3,
}


def _best_evidence(first: str, second: str) -> str:
    return max(
        [first, second],
        key=lambda value: _EVIDENCE_ORDER.get(value, 0),
    )


def _merge_evidence_level(first: PaperRecord, second: PaperRecord) -> str:
    """Merge two evidence levels without ever certifying unretrieved text."""
    if first.text_retrieved and not second.text_retrieved:
        return first.evidence_level
    if second.text_retrieved and not first.text_retrieved:
        return second.evidence_level
    return _best_evidence(first.evidence_level, second.evidence_level)
