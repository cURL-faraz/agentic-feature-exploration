"""``reports/literature_review.md``.

The report is written from the saved bibliography (``data/metadata/papers.json``)
and the saved search plan.  Nothing is cited that is not in that file, and the
evidence level of every record is printed next to it.  In offline runs the
document says so at the top rather than pretending the literature was verified.
"""

from __future__ import annotations

from typing import Any

from .context import ReportContext
from .markdown_utils import heading, render_key_values, render_list, render_records, render_table

__all__ = ["write_literature_review"]

SOURCE = "tables/papers.json"

#: The review structure requested by the project brief: (section title, planned
#: query, keyword tags used to match records that were stored without queries).
TOPICS: list[tuple[str, str, tuple[str, ...]]] = [
    (
        "Overview of single-channel EEG sleep staging",
        "single-channel EEG automatic sleep staging",
        ("single-channel", "sleep staging", "sleep stage"),
    ),
    (
        "Conventional EEG features",
        "spectral power features delta theta alpha sigma sleep stage EEG",
        ("spectral", "time_domain"),
    ),
    (
        "N1/N2/REM classification difficulty",
        "N1 sleep stage classification difficulty automatic scoring",
        ("n1", "rem", "confusion", "sleep staging"),
    ),
    (
        "Nonlinear feature engineering",
        "entropy fractal dimension EEG sleep stage classification",
        ("nonlinear",),
    ),
    (
        "Event-based EEG features",
        "sleep spindle K-complex detection algorithm EEG",
        ("events", "spindle", "k-complex", "slow wave"),
    ),
    (
        "Feature selection techniques",
        "EEG feature selection sleep staging machine learning",
        ("feature selection", "machine learning", "interpretab"),
    ),
    (
        "Statistical evaluation methods",
        "sleep staging statistical evaluation cross validation",
        ("statistic", "cross-validation", "mixed-effects", "subject-wise"),
    ),
    (
        "Deep-learning approaches",
        "deep learning sleep staging single channel EEG",
        ("deep_learning", "transformer", "attention", "neural network"),
    ),
    (
        "Research gaps",
        "limitations automatic sleep staging generalization",
        ("limitation", "generalization", "domain"),
    ),
    (
        "Candidate experimental hypotheses",
        "sleep staging feature hypothesis testing",
        ("hypothesis", "feature"),
    ),
]


def _query_index(search_plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for entry in search_plan.get("queries") or []:
        if isinstance(entry, dict) and entry.get("query"):
            index[str(entry["query"])] = entry
    return index


def _papers_for_query(
    context: ReportContext, query: str, tags: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    """Match a topic to records by planned query, recorded queries, tags or text."""
    matches = []
    for paper in context.papers:
        haystack = " ".join(
            [
                str(paper.get("title") or ""),
                str(paper.get("abstract") or ""),
                str(paper.get("relevance") or ""),
                " ".join(str(item) for item in (paper.get("queries") or [])),
                " ".join(str(item) for item in (paper.get("datasets") or [])),
            ]
        ).lower()
        query_match = any(
            query.lower() in str(entry).lower() or str(entry).lower() in query.lower()
            for entry in (paper.get("queries") or [])
        )
        tag_match = any(tag in haystack for tag in tags)
        if query_match or tag_match:
            matches.append(paper)
    return matches


def write_literature_review(context: ReportContext) -> str:
    """Render the literature review and register its numeric claims."""
    literature = context.literature
    papers = context.papers
    mode = str(literature.get("mode") or "unavailable")
    search_plan = literature.get("search_plan") or {}
    validation = literature.get("validation") or {}
    access = literature.get("access") or {}
    blockers = list(literature.get("retrieval_blockers") or [])

    context.add_claim(
        "lit.n_papers",
        "number of bibliographic records in the saved bibliography",
        literature.get("n_papers", len(papers)),
        "n_papers",
        SOURCE,
        unit="records",
    )
    if isinstance(validation.get("n_unverified_offline"), int):
        context.add_claim(
            "lit.n_unverified_offline",
            "number of bibliography records whose evidence level is offline or metadata-only",
            validation["n_unverified_offline"],
            "validation.n_unverified_offline",
            SOURCE,
            unit="records",
        )

    lines: list[str] = [
        heading(1, "Literature Review: Single-Channel EEG Features for Sleep Staging"),
        "",
        context.origin_caveat(),
        "",
        heading(2, "Retrieval status"),
        render_key_values(
            {
                "retrieval mode": mode,
                "records": len(papers),
                "live retrieval available": bool(literature.get("retrieval_available")),
                "full texts retrieved": access.get("n_full_texts_retrieved", 0),
                "queries planned": len(search_plan.get("queries") or []),
                "queries executed": len(
                    [entry for entry in (literature.get("query_log") or []) if entry.get("status") != "skipped"]
                ),
                "citation validation": "ok" if validation.get("ok") else validation.get("ok"),
            }
        ),
        "",
    ]
    if blockers:
        lines += [heading(3, "What could not be verified"), render_list(blockers), ""]
    if mode != "live":
        lines += [
            (
                "This review was produced **without live bibliographic retrieval**. The records below "
                "are the anchor sources named in the project brief plus any previously cached "
                "bibliography; their metadata has not been checked against the publisher in this run. "
                "No statement in this document should be read as a verified literature finding, and no "
                "numeric quantity is attributed to any author without a retrieved full text."
            ),
            "",
        ]

    # -- bibliography ------------------------------------------------------- #
    lines += [heading(2, "Bibliography"), ""]
    if papers:
        lines.append(
            render_table(
                [
                    {
                        "title": paper.get("title"),
                        "year": paper.get("year"),
                        "venue": paper.get("venue"),
                        "identifier": paper.get("identifier"),
                        "id type": paper.get("identifier_type"),
                        "evidence": paper.get("evidence_level"),
                    }
                    for paper in papers
                ],
                columns=["title", "year", "venue", "identifier", "id type", "evidence"],
            )
        )
    else:
        lines.append("_No bibliographic records are available in this run._")
    lines.append("")

    unresolved = list(validation.get("incomplete") or []) + list(
        validation.get("incomplete_unverified") or []
    )
    if unresolved:
        lines += [
            heading(3, "Records with incomplete metadata"),
            render_table(unresolved),
            (
                "Records whose evidence level is `unverified-offline` are allowed to be "
                "incomplete: the project will not invent a publication year, DOI or author list to "
                "satisfy a schema. They are listed here so the gap is visible, and they are never "
                "presented as verified sources."
            ),
            "",
        ]

    # -- topic sections ----------------------------------------------------- #
    query_index = _query_index(search_plan)
    for title, query, tags in TOPICS:
        lines.append(heading(2, title))
        lines.append("")
        entry = query_index.get(query)
        if entry:
            lines.append(f"Planned query: `{query}` - status in this run: `{entry.get('status', 'n/a')}`.")
            lines.append("")
            if entry.get("why"):
                lines.append(f"*Why this topic matters here:* {entry['why']}")
                lines.append("")
        matches = _papers_for_query(context, query, tags)
        retrieved = [paper for paper in matches if paper.get("full_text_available") or paper.get("text_retrieved")]
        if retrieved:
            lines.append(
                render_records(
                    retrieved[:6],
                    ["title", "year", "identifier", "evidence_level", "relevance", "main_findings", "limitations"],
                )
            )
        elif matches:
            lines.append(
                "Records matched to this topic were retrieved at metadata level only, so no claim "
                "from their full text is used here:"
            )
            lines.append("")
            lines.append(render_list(_paper_labels(matches[:6])))
        else:
            lines.append(
                "No retrieved source in this run addresses this topic directly. The associated "
                "feature families are therefore treated as *literature-inspired but unverified in "
                "this run* and are labelled as such in the feature catalog."
            )
        lines.append("")

    # -- gaps and hypotheses ------------------------------------------------ #
    lines += [
        heading(2, "Research gaps this project targets"),
        render_list(
            [
                "Single-channel montages discard EOG/EMG evidence, so REM is separated from N1/N2 by EEG alone.",
                "N1 is short, transitional and unbalanced; per-stage metrics are reported rather than accuracy alone.",
                "Event features (spindles, K-complexes, slow waves) are often conflated with band power; both are computed here and compared.",
                "Nonlinear and information-theoretic features are rarely validated on synthetic signals with known properties.",
                "Epoch-level random splits leak subject identity; subject-grouped splits with a saved manifest are used here.",
            ]
        ),
        "",
        heading(2, "Candidate experimental hypotheses"),
        render_list(
            [
                "H1: spectral concentration measures separate N2 from N1/REM more strongly than absolute band power.",
                "H2: within-epoch spectral instability is higher in transitional stages (N1, and wake-sleep transitions).",
                "H3: event-based features (spindle, slow-wave density) carry information not recoverable from band power alone.",
                "H4: nonlinear complexity features are stage-dependent even after amplitude normalisation.",
                "H5: combining spectral, nonlinear and event families improves per-stage balanced accuracy over any single family.",
                "H6: any predictive advantage of a novel feature family survives subject-grouped held-out evaluation.",
            ]
        ),
        "",
        heading(2, "How each hypothesis is tested"),
        (
            "H1-H4 are tested with the repeated-measures comparisons in "
            "`reports/statistical_analysis.md` (subject x stage means, FDR-corrected). "
            "H5 is tested by the ablation study in `reports/final_research_report.md`. "
            "H6 is tested on the held-out test subjects and is the only basis on which a "
            "feature is described as predictively useful."
        ),
        "",
    ]
    return "\n".join(lines).strip() + "\n"


def _paper_labels(papers: list[dict[str, Any]]) -> list[str]:
    labels = []
    for paper in papers:
        identifier = paper.get("identifier") or "no identifier"
        labels.append(f"{paper.get('title')} ({paper.get('year') or 'n.d.'}; {identifier})")
    return labels
