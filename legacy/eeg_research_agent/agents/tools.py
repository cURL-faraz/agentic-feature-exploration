"""Narrowly scoped tools exposed to the agent nodes.

Design rule from the brief: *agents call tools; agents never write executable
code*.  Every tool here is a thin, typed wrapper around a deterministic function
that already exists in the project, and each declares who may call it.

The tool table is data, not prose: :data:`AGENT_TOOLS` maps a tool name to its
callable, the agent roles allowed to use it, and whether it mutates the project
state.  The graph builder uses it to grant permissions, so a node physically
cannot call a tool its role does not own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..logging_utils import get_logger

__all__ = [
    "ToolSpec",
    "AGENT_TOOLS",
    "ToolNotPermitted",
    "ResearchToolbox",
    "tool_table",
]

logger = get_logger(__name__)


class ToolNotPermitted(RuntimeError):
    """Raised when an agent role attempts to use a tool it does not own."""


@dataclass(frozen=True)
class ToolSpec:
    """One callable capability with an explicit permission list."""

    name: str
    function: Callable[..., Any]
    roles: tuple[str, ...]
    mutates_project_state: bool = False
    description: str = ""

    def call(self, role: str, **kwargs: Any) -> Any:
        if role not in self.roles:
            raise ToolNotPermitted(
                f"role {role!r} may not call {self.name!r}; allowed roles: {', '.join(self.roles)}"
            )
        return self.function(**kwargs)


# --------------------------------------------------------------------------- #
# EEG tools
# --------------------------------------------------------------------------- #
def _inspect_recording(recording: Any) -> dict[str, Any]:
    from ..data.inspection import inspect_recording

    return dict(inspect_recording(recording))


def _get_channel_metadata(recordings: Sequence[Any], preference: Sequence[str] | None = None) -> dict[str, Any]:
    from ..data.inspection import pick_channel

    channels = sorted({str(recording.ch_name) for recording in recordings})
    if preference is None:
        # Fall back to whatever candidate channels the recordings themselves declare.
        preference = sorted({c for recording in recordings for c in getattr(recording, "channel_candidates", [])})
    return {
        "available_channels": channels,
        "selected": pick_channel(channels, list(preference)),
        "n_recordings": len(recordings),
    }


def _calculate_band_power(epoch: Any, sfreq: float, band: str, config: Any) -> float:
    from ..eeg.feature_registry import SignalContext

    return SignalContext(epoch=epoch, sfreq=sfreq, config=config).band_power(band)


def _detect_spindles(context: Any) -> dict[str, Any]:
    from ..eeg.sleep_events import detect_spindles

    events = detect_spindles(context)
    return {
        "count": events.count,
        "density_per_min": events.density_per_min,
        "mean_duration_s": events.mean_duration,
        "mean_amplitude_v": events.mean_amplitude,
        "mean_frequency_hz": events.mean_frequency,
        "occupancy": events.occupancy,
    }


# --------------------------------------------------------------------------- #
# statistical tools
# --------------------------------------------------------------------------- #
def _describe_feature_distribution(table: Any, feature: str) -> dict[str, Any]:
    from ..statistics.descriptive import describe_feature_by_stage

    return describe_feature_by_stage(table, feature)


def _correct_multiple_tests(p_values: Sequence[float], labels: Sequence[str], method: str) -> dict[str, Any]:
    from ..statistics.multiple_testing import correct_pvalues

    result = correct_pvalues(list(p_values), list(labels), method=method)
    return result.to_dict()


def _calculate_effect_size(group_a: Sequence[float], group_b: Sequence[float], paired: bool = False) -> dict[str, Any]:
    from ..statistics.effect_sizes import cohens_d, hedges_g

    return {
        "paired_requested": bool(paired),
        "cohens_d": cohens_d(group_a, group_b).to_dict(),
        "hedges_g": hedges_g(group_a, group_b).to_dict(),
    }


# --------------------------------------------------------------------------- #
# verification tools
# --------------------------------------------------------------------------- #
def _generate_synthetic_signal(kind: str, **kwargs: Any) -> Any:
    """Build one of the controlled test signals used by the verification suite."""
    from ..verification import synthetic_signals as synth

    generators = {
        "sine": synth.sine,
        "multi_tone": synth.multi_tone,
        "white_noise": synth.white_noise,
        "pink_noise": synth.pink_noise,
        "amplitude_modulated": synth.amplitude_modulated,
        "frequency_modulated": synth.frequency_modulated,
        "spindle_burst": synth.spindle_burst,
        "slow_wave": synth.slow_wave,
        "with_artifact": synth.with_artifact,
        "eeg_like_background": synth.eeg_like_background,
        "kcomplex_like": synth.kcomplex_like,
        "slow_wave_transient": synth.slow_wave_transient,
    }
    if kind not in generators:
        raise ValueError(f"unknown synthetic signal kind: {kind!r}; known kinds: {sorted(generators)}")
    return generators[kind](**kwargs)


def _verify_split_integrity(rows: Sequence[dict[str, Any]], splits: Any) -> dict[str, Any]:
    from ..data.splits import verify_splits

    return verify_splits(list(rows), splits)


def _check_data_leakage(table: Any, splits: Any) -> dict[str, Any]:
    from ..evaluation.leakage_checks import run_leakage_audit

    return run_leakage_audit(table, splits)


def _validate_citations(papers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from ..verification.research_validation import validate_citations

    return validate_citations(papers)


def _validate_statistical_results(claims: Sequence[Mapping[str, Any]], results: Mapping[str, Any]) -> dict[str, Any]:
    from ..verification.research_validation import validate_claims

    return validate_claims(claims, dict(results))


# --------------------------------------------------------------------------- #
# literature tools
# --------------------------------------------------------------------------- #
def _search_papers(query: str, limit: int = 5, providers: Sequence[str] | None = None) -> dict[str, Any]:
    """Search the configured providers; returns an empty result set offline."""

    from ..data.acquisition import network_available
    from ..research.paper_search import search_all

    if not network_available():
        return {"query": query, "available": False, "records": [], "reason": "no network access"}
    records, query_log = search_all(
        [{"topic": "ad_hoc", "query": query}], providers=providers, limit_per_query=limit
    )
    return {
        "query": query,
        "available": True,
        "n_records": len(records),
        "records": [record.to_dict() for record in records],
        "log": query_log,
    }


def _retrieve_open_access_text(record: Mapping[str, Any]) -> dict[str, Any]:
    from ..data.acquisition import network_available
    from ..research.paper_retrieval import fetch_open_access_text

    if not network_available():
        return {"ok": False, "reason": "no network access; open-access retrieval unavailable"}
    from ..research.paper_search import record_from_dict

    document = fetch_open_access_text(record_from_dict(dict(record)))
    return {
        "ok": bool(getattr(document, "ok", False)),
        "source": getattr(document, "source", None),
        "text_chars": len(getattr(document, "text", "") or ""),
        "error": getattr(document, "error", None),
    }


def _extract_paper_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    from ..research.evidence_extraction import extract_evidence
    from ..research.paper_search import record_from_dict

    enriched = extract_evidence(record_from_dict(dict(record)))
    return {
        "title": enriched.title,
        "evidence_level": enriched.evidence_level,
        "text_retrieved": enriched.text_retrieved,
        "main_findings": list(enriched.main_findings),
        "limitations": list(enriched.limitations),
        "feature_methods": list(enriched.feature_methods),
        "statistical_methods": list(enriched.statistical_methods),
        "datasets": list(enriched.datasets),
        "notes": enriched.notes,
    }


# --------------------------------------------------------------------------- #
# reporting tools
# --------------------------------------------------------------------------- #
def _create_feature_catalog(config: Any, artifacts: Any, results: Mapping[str, Any]) -> dict[str, Any]:
    from ..reporting.context import load_report_context

    context = load_report_context(config, artifacts, dict(results))
    return {
        "n_features": context.catalog.get("n_features"),
        "by_novelty_status": context.catalog.get("by_novelty_status"),
        "by_family": context.catalog.get("by_family"),
    }


def _generate_final_report(config: Any, artifacts: Any, results: Mapping[str, Any]) -> dict[str, Any]:
    from ..reporting import generate_all_reports

    return generate_all_reports(config=config, artifacts=artifacts, results=dict(results))


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #
AGENT_TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec("inspect_recording", _inspect_recording, ("dataset", "verification")),
        ToolSpec("get_channel_metadata", _get_channel_metadata, ("dataset",)),
        ToolSpec("calculate_band_power", _calculate_band_power, ("feature", "verification")),
        ToolSpec("detect_spindles", _detect_spindles, ("feature", "verification")),
        ToolSpec("describe_feature_distribution", _describe_feature_distribution, ("statistics",)),
        ToolSpec("correct_multiple_tests", _correct_multiple_tests, ("statistics", "verification")),
        ToolSpec("calculate_effect_size", _calculate_effect_size, ("statistics", "verification")),
        ToolSpec("generate_synthetic_signal", _generate_synthetic_signal, ("verification",)),
        ToolSpec("verify_split_integrity", _verify_split_integrity, ("verification",)),
        ToolSpec("check_data_leakage", _check_data_leakage, ("verification",)),
        ToolSpec("validate_citations", _validate_citations, ("verification", "literature")),
        ToolSpec("validate_statistical_results", _validate_statistical_results, ("verification",)),
        ToolSpec("search_papers", _search_papers, ("literature",)),
        ToolSpec("retrieve_open_access_text", _retrieve_open_access_text, ("literature",)),
        ToolSpec("extract_paper_evidence", _extract_paper_evidence, ("literature",)),
        ToolSpec("create_feature_catalog", _create_feature_catalog, ("reporting",)),
        ToolSpec("generate_final_report", _generate_final_report, ("reporting",)),
    )
}


def tool_table() -> list[dict[str, Any]]:
    """Serialisable view of the tool permissions (used by docs and tests)."""
    return [
        {
            "name": spec.name,
            "roles": list(spec.roles),
            "mutates_project_state": spec.mutates_project_state,
        }
        for spec in sorted(AGENT_TOOLS.values(), key=lambda item: item.name)
    ]


class ResearchToolbox:
    """Role-scoped facade: ``toolbox.for_role("statistics").correct_multiple_tests(...)``."""

    def __init__(self, roles: Sequence[str] | None = None, calls: list[dict[str, Any]] | None = None) -> None:
        self._roles = list(roles) if roles else sorted({role for spec in AGENT_TOOLS.values() for role in spec.roles})
        # ``calls`` is shared with any role-scoped view so the graph's tool log
        # records every call, not just the ones made through the root toolbox.
        self.calls: list[dict[str, Any]] = calls if calls is not None else []

    def __getattr__(self, name: str) -> Callable[..., Any]:  # pragma: no cover - thin dispatch
        spec = AGENT_TOOLS.get(name)
        if spec is None:
            raise AttributeError(f"unknown tool: {name}")
        if not set(spec.roles) & set(self._roles):
            raise ToolNotPermitted(
                f"a toolbox scoped to {self._roles} may not call {name!r} (owned by {spec.roles})"
            )

        def _call(**kwargs: Any) -> Any:
            self.calls.append({"tool": name, **{k: type(v).__name__ for k, v in kwargs.items()}})
            return spec.function(**kwargs)

        return _call

    def for_role(self, role: str) -> "ResearchToolbox":
        """A toolbox restricted to one role's permissions, sharing the call log."""

        return ResearchToolbox(roles=[role], calls=self.calls)

    def to_dict(self) -> dict[str, Any]:
        return {"roles": self._roles, "n_calls": len(self.calls), "calls": self.calls}
