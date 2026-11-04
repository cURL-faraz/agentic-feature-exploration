"""Prompts and structured-output contracts for the LLM-facing steps.

Two rules shape this module:

1. **The LLM is a reasoner over *supplied* facts, never a source of facts.**  Every
   prompt carries its payload explicitly, and the expected output is either prose
   about that payload or a *selection* from a list the deterministic code already
   produced.
2. **Prompts are declarative objects.**  A prompt is built by a function that
   returns ``(system, user, metadata)`` so the same call works for a networked
   provider and for the offline template renderer, which dispatches on
   ``metadata["task"]``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

__all__ = [
    "TASK_NAMES",
    "SYSTEM_RESEARCHER",
    "SYSTEM_CRITIC",
    "build_dataset_summary_prompt",
    "build_search_topics_prompt",
    "build_feature_hypotheses_prompt",
    "build_interpretation_prompt",
    "build_report_section_prompt",
]

#: Task identifiers understood by the offline provider.
TASK_NAMES = (
    "dataset_summary",
    "search_topics",
    "feature_hypotheses",
    "interpretation",
    "report_section",
)

SYSTEM_RESEARCHER = (
    "You are a biomedical signal-processing researcher assisting a reproducible EEG sleep-staging "
    "study. Use ONLY the facts supplied in the user message. Never invent numbers, citations, "
    "dataset properties or results. If something is unavailable, say so explicitly. State the "
    "difference between statistical association, predictive usefulness and biological plausibility."
)

SYSTEM_CRITIC = (
    "You are an adversarial reviewer. Your job is to find the ways the supplied analysis could be "
    "wrong or overstated: leakage, pseudo-replication, multiple comparisons, unvalidated features, "
    "unsupported biological claims. Be specific and cite the supplied fields. Do not soften the "
    "critique and do not introduce facts that are not in the payload."
)


def _payload(**kwargs: Any) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}


def build_dataset_summary_prompt(dataset_summary: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Ask for a plain-language reading of the dataset-intelligence output."""
    system = SYSTEM_RESEARCHER
    user = (
        "Summarise this dataset inspection for a scientific reader. Report the derivation, sampling "
        "frequency, epoch duration, subject and epoch counts, stage distribution and quality flags. "
        "Do not add any number that is not present in the payload.\n\n"
        f"payload:\n{dataset_summary}"
    )
    return system, user, {"task": "dataset_summary", "payload": _payload(dataset_summary=dict(dataset_summary))}


def build_search_topics_prompt(queries: Sequence[Mapping[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    """Ask which planned literature queries matter most for this research question."""
    system = SYSTEM_RESEARCHER
    user = (
        "Below is the planned literature search for a single-channel EEG sleep-staging feature study. "
        "Prioritise the queries by how directly they can inform feature design or statistical method. "
        "Do not invent papers.\n\n"
        f"queries:\n{list(queries)}"
    )
    return system, user, {
        "task": "search_topics",
        "payload": _payload(queries=[dict(entry) for entry in queries]),
    }


def build_feature_hypotheses_prompt(
    candidates: Sequence[Mapping[str, Any]],
    literature_summary: Mapping[str, Any] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Ask the model to *select* from registered features, never to write code."""
    system = SYSTEM_RESEARCHER
    user = (
        "These are feature specifications generated from the project's feature registry. Each entry "
        "has a feature_id, a category, a novelty status and tunable parameters with their allowed "
        "ranges. Select the subset worth running in this study and explain the choice. You may NOT "
        "invent new feature ids or parameters outside the declared ranges: the selection is validated "
        "against the registry afterwards, and anything unregistered is rejected.\n\n"
        f"candidates:\n{list(candidates)}\n\nliterature summary:\n{literature_summary or {}}"
    )
    return system, user, {
        "task": "feature_hypotheses",
        "payload": _payload(
            candidates=[dict(entry) for entry in candidates],
            selected=[entry.get("feature_id") for entry in candidates],
            notes=literature_summary,
        ),
    }


def build_interpretation_prompt(findings: Sequence[Mapping[str, Any]], question: str) -> tuple[str, str, dict[str, Any]]:
    """Ask for a cautious reading of saved statistical output."""
    system = SYSTEM_RESEARCHER
    user = (
        "Interpret these saved statistical results. For each, state whether it is an association, a "
        "predictive result, or neither. Flag anything that is exploratory or based on a novel "
        "unvalidated feature. Do not restate a p-value without its correction status and effect size.\n\n"
        f"question:\n{question}\n\nfindings:\n{list(findings)}"
    )
    return system, user, {
        "task": "interpretation",
        "payload": _payload(question=question, findings=[dict(item) for item in findings]),
    }


def build_report_section_prompt(
    title: str, body_markdown: str = "", data: Mapping[str, Any] | None = None
) -> tuple[str, str, dict[str, Any]]:
    """Ask for a report section rendered from already-computed content."""
    system = SYSTEM_RESEARCHER
    user = (
        f"Write the '{title}' section of a research report using only the supplied content. Keep every "
        "number exactly as given.\n\n"
        f"content:\n{body_markdown or data or {}}"
    )
    return system, user, {
        "task": "report_section",
        "payload": _payload(title=title, body_markdown=body_markdown, data=dict(data or {})),
    }
