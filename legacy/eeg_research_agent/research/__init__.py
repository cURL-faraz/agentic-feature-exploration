"""Academic literature retrieval, evidence extraction and citation validation.

``http``                 rate-limited, retrying, caching HTTP client
``paper_search``         Semantic Scholar / OpenAlex / Crossref / PubMed / arXiv / Europe PMC
``paper_retrieval``      open-access full-text retrieval (arXiv, Europe PMC, Unpaywall)
``evidence_extraction``  conservative, regex-anchored structured extraction
``citation_validation``  structural + cross-source citation checks
``seed_sources``         the anchor documents named in the project brief
``literature``           stage orchestration (live and offline modes)
"""

from .citation_validation import cross_source_consistency, validate_bibliography, validate_record
from .evidence_extraction import annotate_relevance, extract_evidence
from .literature import LiteratureResult, run_literature_review
from .paper_retrieval import fetch_open_access_text, summarise_access
from .paper_search import PaperRecord, available_providers, deduplicate, search_all, search_provider
from .seed_sources import SEARCH_QUERIES, SEED_SOURCES

__all__ = [
    "LiteratureResult",
    "run_literature_review",
    "PaperRecord",
    "available_providers",
    "deduplicate",
    "search_all",
    "search_provider",
    "fetch_open_access_text",
    "summarise_access",
    "extract_evidence",
    "annotate_relevance",
    "validate_bibliography",
    "validate_record",
    "cross_source_consistency",
    "SEED_SOURCES",
    "SEARCH_QUERIES",
]
