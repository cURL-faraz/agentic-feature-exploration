"""Literature retrieval from OpenAlex (free, no API key) -> inspiration cards.

Retrieval is deterministic code; only the *reading* is done by the LLM.  The
LLM never chooses or invents a source: it may only cite paper ids that the
retrieval step handed to it (checked in :func:`roles.scout_cards`).

Two rankings per query, as you suggested: the most-cited papers of all time
(established knowledge) and the most-cited recent papers (trends).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

from .roles import scout_cards
from .schemas import InspirationCard

API = "https://api.openalex.org/works"


def _abstract(inverted: dict | None) -> str:
    if not inverted:
        return ""
    positions = [(pos, word) for word, places in inverted.items() for pos in places]
    return " ".join(word for _, word in sorted(positions))


def search(query: str, per_page: int, from_year: int | None = None) -> list[dict]:
    # `title_and_abstract.search` is much stricter than the generic `search` parameter:
    # citation-sorting a loose search surfaces famous but off-topic papers.
    filters = ["has_abstract:true", f"title_and_abstract.search:{query}"]
    if from_year:
        filters.append(f"from_publication_date:{from_year}-01-01")
    params = {
        "filter": ",".join(filters), "sort": "cited_by_count:desc",
        "per-page": str(per_page),
        "select": "id,doi,title,publication_year,cited_by_count,abstract_inverted_index",
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "featurelab/0.2"})
    with urllib.request.urlopen(request, timeout=60) as response:
        results = json.loads(response.read().decode())["results"]
    return [
        {"id": r["id"].rsplit("/", 1)[-1], "doi": r.get("doi"), "title": r.get("title") or "",
         "year": r.get("publication_year"), "cited_by": r.get("cited_by_count", 0),
         "abstract": _abstract(r.get("abstract_inverted_index"))}
        for r in results
    ]


def is_relevant(paper: dict) -> bool:
    """Deterministic relevance guard: must be about sleep AND EEG."""
    text = f"{paper['title']} {paper['abstract']}".lower()
    return "sleep" in text and ("eeg" in text or "electroencephalogra" in text)


def collect_papers(config: dict) -> list[dict]:
    lit = config["literature"]
    seen: dict[str, dict] = {}
    for query in lit["queries"]:
        for year in (None, int(lit["recent_from_year"])):
            for paper in search(query, int(lit["per_query"]), year):
                if paper["abstract"] and paper["id"] not in seen and is_relevant(paper):
                    paper["found_by"] = f"{query} ({'recent' if year else 'all-time'})"
                    seen[paper["id"]] = paper
    papers = sorted(seen.values(), key=lambda p: -p["cited_by"])
    return papers[: int(lit["max_papers"])]


def build_inspiration(config: dict, client, force: bool = False, batch_size: int = 8,
                      log=print) -> list[InspirationCard]:
    root = Path(config["data"]["root"])
    papers_path, cards_path = root / "literature" / "papers.json", root / "literature" / "cards.json"
    if cards_path.is_file() and not force:
        return [InspirationCard.model_validate(c) for c in json.loads(cards_path.read_text())]
    papers_path.parent.mkdir(parents=True, exist_ok=True)
    if papers_path.is_file() and not force:
        papers = json.loads(papers_path.read_text())
    else:
        papers = collect_papers(config)
        papers_path.write_text(json.dumps(papers, indent=1))
    log(f"scout: {len(papers)} papers with abstracts")
    cards: list[InspirationCard] = []
    for start in range(0, len(papers), batch_size):
        try:
            cards += scout_cards(client, papers[start : start + batch_size])
        except Exception as exc:  # noqa: BLE001 - one bad batch must not stop the scout
            log(f"scout: batch {start // batch_size} failed: {exc}")
    cards_path.write_text(json.dumps([c.model_dump() for c in cards], indent=1))
    log(f"scout: {len(cards)} grounded inspiration cards")
    return cards
