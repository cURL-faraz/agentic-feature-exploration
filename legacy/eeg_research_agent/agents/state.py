"""Explicit, typed graph state.

The state is deliberately *data only*: it carries the identifiers and summaries
of what each stage produced, plus the paths to the artifacts that hold the real
payloads.  Nothing scientific lives in the state as a computed value that cannot
be re-derived from an artifact - that is what stops the workflow from becoming a
second, unauditable implementation of the analysis.

``errors`` and ``stages_completed`` use ``operator.add`` reducers so parallel or
retried branches accumulate rather than overwrite each other.
"""

from __future__ import annotations

import dataclasses
import operator
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ResearchState", "StageRecord", "initial_state", "jsonable"]


def jsonable(value: Any) -> Any:
    """Normalise a value into msgpack/JSON-friendly Python primitives.

    LangGraph serialises checkpointed state with msgpack, which rejects numpy
    scalars and arrays.  Node outputs cross exactly one boundary into the graph,
    so they are converted here rather than sprinkling casts throughout the node
    bodies.  Unknown objects are returned untouched: silently mangling a value
    would be worse than letting the serialiser complain about it.
    """
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value

    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is a hard dependency
        np = None  # type: ignore[assignment]

    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return [jsonable(item) for item in value.tolist()]

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    if isinstance(value, BaseModel):
        return jsonable(value.model_dump())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value))
    return value


class StageRecord(BaseModel):
    """One completed graph node, with timing and provenance."""

    model_config = ConfigDict(extra="allow")

    stage: str
    ok: bool = True
    duration_s: float | None = None
    detail: str = ""


class ResearchState(BaseModel):
    """The complete workflow state."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    # -- problem definition ------------------------------------------------- #
    research_question: str = (
        "Which single-channel EEG characteristics are associated with sleep stages, and which "
        "combinations of features improve N1/N2/REM discrimination on unseen subjects?"
    )
    mode: Literal["smoke", "standard", "full"] = "standard"
    origin: str | None = None

    # -- data --------------------------------------------------------------- #
    dataset_metadata: dict[str, Any] = Field(default_factory=dict)
    downloaded_files: list[str] = Field(default_factory=list)
    preprocessing_config: dict[str, Any] = Field(default_factory=dict)
    subject_splits: dict[str, Any] = Field(default_factory=dict)

    # -- literature --------------------------------------------------------- #
    literature_results: dict[str, Any] = Field(default_factory=dict)
    paper_evidence: dict[str, Any] = Field(default_factory=dict)

    # -- features ----------------------------------------------------------- #
    candidate_features: list[dict[str, Any]] = Field(default_factory=list)
    approved_features: list[dict[str, Any]] = Field(default_factory=list)
    rejected_features: list[dict[str, Any]] = Field(default_factory=list)
    feature_tests: dict[str, Any] = Field(default_factory=dict)
    feature_extraction_results: dict[str, Any] = Field(default_factory=dict)
    feature_distributions: dict[str, Any] = Field(default_factory=dict)

    # -- analysis ----------------------------------------------------------- #
    statistical_results: dict[str, Any] = Field(default_factory=dict)
    feature_selection_results: dict[str, Any] = Field(default_factory=dict)
    predictive_results: dict[str, Any] = Field(default_factory=dict)
    verification_results: dict[str, Any] = Field(default_factory=dict)
    research_report: dict[str, Any] = Field(default_factory=dict)

    # -- bookkeeping -------------------------------------------------------- #
    agent_notes: dict[str, str] = Field(default_factory=dict)
    llm: dict[str, Any] = Field(default_factory=dict)
    execution_metadata: dict[str, Any] = Field(default_factory=dict)
    stages_completed: Annotated[list[str], operator.add] = Field(default_factory=list)
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)


def initial_state(
    mode: str = "standard",
    origin: str | None = None,
    research_question: str | None = None,
    **overrides: Any,
) -> ResearchState:
    """Build the starting state for a run."""
    payload: dict[str, Any] = {"mode": mode, "origin": origin}
    if research_question:
        payload["research_question"] = research_question
    payload.update(overrides)
    return ResearchState(**payload)
