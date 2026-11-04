"""Graph nodes: one function per stage of the research workflow.

Two invariants hold across every node:

* **A node never computes science.**  It calls the same
  :class:`~..pipeline.ResearchPipeline` method the CLI calls.  The graph changes
  *sequencing, routing and narration*, not the mathematics.
* **A node never hides a failure.**  Exceptions are caught, appended to
  ``state.errors`` with the node name, and recorded in the stage log; the router
  then decides whether the run can continue.  A node that failed never reports
  success.

The LLM is used for exactly three things: reading a dataset summary, selecting
from registry-generated feature candidates, and interpreting saved results.
Nothing it returns is ever executed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..logging_utils import get_logger
from . import prompts
from .state import ResearchState, jsonable
from .tools import ResearchToolbox

__all__ = ["AgentContext", "NODE_NAMES", "build_nodes", "node_names"]

logger = get_logger(__name__)

NODE_NAMES = (
    "initialize_project",
    "acquire_dataset",
    "inspect_dataset",
    "prepare_dataset",
    "search_literature",
    "extract_paper_evidence",
    "generate_feature_hypotheses",
    "validate_feature_specifications",
    "implement_candidate_features",
    "run_feature_tests",
    "extract_features",
    "analyze_feature_distributions",
    "run_statistical_tests",
    "run_feature_selection",
    "evaluate_predictive_models",
    "verify_results",
    "generate_report",
    "publish_artifacts",
)


@dataclass
class AgentContext:
    """Everything a node needs: the deterministic pipeline, the LLM and the tools."""

    pipeline: Any
    llm: Any = None
    toolbox: ResearchToolbox = field(default_factory=ResearchToolbox)
    trace: list[dict[str, Any]] = field(default_factory=list)
    stage_trace: list[dict[str, Any]] = field(default_factory=list)

    @property
    def config(self) -> Any:
        return self.pipeline.config

    @property
    def artifacts(self) -> Any:
        return self.pipeline.artifacts

    def ask(self, system: str, user: str, metadata: dict[str, Any]) -> Any:
        """One LLM call; records the task and never raises on provider failure."""
        from ..llm.provider import LLMMessage, LLMRequest

        if self.llm is None:
            from ..llm import build_llm_client

            self.llm = build_llm_client(self.config)
        task = str(metadata.get("task", "generic"))
        try:
            reply = self.llm.complete(
                LLMRequest(messages=[LLMMessage("user", user)], system=system, metadata=metadata)
            )
        except Exception as exc:  # noqa: BLE001 - an LLM outage must not stop the science
            self.trace.append({"task": task, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            logger.warning("LLM task %s failed: %s", task, exc)
            return None
        self.trace.append(
            {
                "task": task,
                "ok": reply.ok,
                "provider": reply.provider,
                "model": reply.model,
                "from_cache": reply.from_cache,
                "chars": len(reply.text or ""),
                "used_model_reasoning": bool(reply.metadata.get("reasoning", True)),
            }
        )
        return reply


def _node(name: str, role: str) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
    """Wrap a node body with timing, error capture, stage bookkeeping and tracing."""

    def decorator(function: Callable[[ResearchState, AgentContext], dict[str, Any]]):
        def wrapped(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
            started = time.time()
            try:
                update = dict(jsonable(function(state, ctx) or {}))
            except Exception as exc:  # noqa: BLE001 - the router decides what a failure means
                ctx.stage_trace.append(
                    {"node": name, "role": role, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
                )
                logger.error("node %s failed: %s", name, exc)
                return {
                    "errors": [f"{name}: {type(exc).__name__}: {exc}"],
                    "stages_completed": [name],
                    "execution_metadata": {f"{name}_failed": True},
                }
            duration = round(time.time() - started, 3)
            ctx.stage_trace.append({"node": name, "role": role, "ok": True, "duration_s": duration})
            completed = list(update.get("stages_completed") or [])
            completed.append(name)
            update["stages_completed"] = completed
            metadata = dict(update.get("execution_metadata") or {})
            metadata[f"{name}_seconds"] = duration
            update["execution_metadata"] = metadata
            return update

        wrapped.__name__ = name
        wrapped.__doc__ = function.__doc__
        return wrapped

    return decorator


def _registry() -> Any:
    from ..eeg.feature_registry import load_default_registry

    return load_default_registry()


# --------------------------------------------------------------------------- #
# data nodes
# --------------------------------------------------------------------------- #
@_node("initialize_project", "dataset")
def initialize_project(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
    """Create the directory layout, the event log and the execution manifest."""
    summary = ctx.pipeline.initialise()
    return {
        "execution_metadata": {"initialise": summary},
        "llm": {
            "provider": getattr(ctx.llm, "name", state.llm.get("provider", "offline")),
            "model": getattr(ctx.llm, "model", state.llm.get("model", "")),
        },
    }


@_node("acquire_dataset", "dataset")
def acquire_dataset(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
    """Download the public dataset, or fall back to the synthetic surrogate."""
    summary = ctx.pipeline.acquire(origin=state.origin or None)
    files = [str(summary["manifest"])] if summary.get("manifest") else []
    return {
        "dataset_metadata": {"acquisition": summary},
        "downloaded_files": files,
    }


@_node("inspect_dataset", "dataset")
def inspect_dataset(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
    """Describe the corpus schema, channel, annotations and class balance."""
    summary = ctx.pipeline.inspect()
    preference = list(getattr(ctx.config.dataset.signal, "channel_preference", []) or [])
    ctx.toolbox.for_role("dataset").get_channel_metadata(
        recordings=ctx.artifacts.recordings, preference=preference
    )
    system, user, metadata = prompts.build_dataset_summary_prompt(summary)
    reply = ctx.ask(system, user, metadata)
    note = reply.text if reply is not None and reply.ok else "dataset-summary narration unavailable"
    return {
        "dataset_metadata": summary,
        "agent_notes": {"inspect_dataset": note},
    }


@_node("prepare_dataset", "dataset")
def prepare_dataset(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
    """Preprocess, epoch and create the subject-grouped split manifest."""
    preprocessing = ctx.pipeline.preprocess()
    segmentation = ctx.pipeline.segment_and_split()
    splits = ctx.artifacts.splits
    return {
        "preprocessing_config": preprocessing,
        "subject_splits": splits.to_dict() if splits is not None else {},
        "dataset_metadata": {
            "segmentation": {
                key: value for key, value in segmentation["segment"].items() if key != "exclusions"
            }
        },
    }


# --------------------------------------------------------------------------- #
# literature nodes
# --------------------------------------------------------------------------- #
@_node("search_literature", "literature")
def search_literature(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
    """Run the literature stage; offline it emits a labelled, unverified plan."""
    payload = ctx.pipeline.literature()
    system, user, metadata = prompts.build_search_topics_prompt(payload.get("query_log") or [])
    reply = ctx.ask(system, user, metadata)
    note = reply.text if reply is not None and reply.ok else "search-plan narration unavailable"
    return {
        "literature_results": {key: value for key, value in payload.items() if key != "papers"},
        "agent_notes": {"search_literature": note},
    }


@_node("extract_paper_evidence", "literature")
def extract_paper_evidence(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
    """Summarise retrievable evidence and validate every citation structurally."""
    from collections import Counter

    papers = list((state.literature_results or {}).get("papers") or [])
    path = ctx.config.path("metadata") / "papers.json"
    if not papers and path.is_file():
        with open(path, encoding="utf-8") as handle:
            papers = list(json.load(handle).get("papers") or [])
    levels = Counter(str(paper.get("evidence_level")) for paper in papers)
    with_text = sum(1 for paper in papers if paper.get("full_text_available") or paper.get("text_retrieved"))
    citations = ctx.toolbox.for_role("literature").validate_citations(papers=papers)
    return {
        "paper_evidence": {
            "n_papers": len(papers),
            "evidence_levels": dict(levels),
            "n_with_retrieved_text": with_text,
            "citation_validation": citations,
            "identified": [paper.get("title") for paper in papers if paper.get("identifier")],
        }
    }


# --------------------------------------------------------------------------- #
# feature nodes
# --------------------------------------------------------------------------- #
@_node("generate_feature_hypotheses", "feature")
def generate_feature_hypotheses(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
    """Build registry-derived candidates, then let the model *select* among them."""
    registry = _registry()
    candidates = [
        {
            "feature_id": definition.feature_id,
            "name": definition.name,
            "category": definition.category,
            "family": definition.family,
            "novelty_status": definition.novelty_status,
            "parameters": dict(definition.parameters),
            "output_units": definition.output_units,
        }
        for definition in registry
    ]
    system, user, metadata = prompts.build_feature_hypotheses_prompt(
        candidates, literature_summary=state.paper_evidence or None
    )
    reply = ctx.ask(system, user, metadata)
    selected = [entry["feature_id"] for entry in candidates]
    rationale = "deterministic default: every registered feature is a candidate"
    if reply is not None and reply.ok:
        try:
            payload = json.loads(reply.text)
            proposed = [str(item) for item in (payload.get("selected") or [])]
            if proposed:
                selected = proposed
            rationale = str(payload.get("rationale") or rationale)
        except Exception:  # noqa: BLE001 - non-JSON prose is still useful narration
            rationale = reply.text[:1200]
    keep = set(selected)
    return {
        "candidate_features": [
            {
                "feature_id": entry["feature_id"],
                "category": entry["category"],
                "family": entry["family"],
                "novelty_status": entry["novelty_status"],
                "parameters": entry["parameters"],
                "requested_by_model": entry["feature_id"] in keep,
            }
            for entry in candidates
        ],
        "agent_notes": {"generate_feature_hypotheses": rationale},
    }


@_node("validate_feature_specifications", "feature")
def validate_feature_specifications(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
    """Validate every candidate against the registry - the LLM boundary."""
    registry = _registry()
    sfreq = float((state.dataset_metadata or {}).get("sampling_frequency") or 100.0)
    specification = {
        "features": [
            {"feature_id": entry.get("feature_id"), "parameters": entry.get("parameters") or {}}
            for entry in (state.candidate_features or [])
        ]
    }
    outcome = registry.validate_specification(specification, sfreq)
    return {"approved_features": outcome["approved"], "rejected_features": outcome["rejected"]}


@_node("implement_candidate_features", "feature")
def implement_candidate_features(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
    """Confirm approved specs map onto executable implementations.

    No code is generated here (or anywhere else in the graph): a feature enters
    the pipeline only if it is already registered with a tested implementation.
    """
    registry = _registry()
    implemented: list[dict[str, Any]] = []
    rejected = list(state.rejected_features or [])
    for entry in state.approved_features or []:
        feature_id = str(entry.get("feature_id"))
        if feature_id not in registry:
            rejected.append({"feature_id": feature_id, "reason": "no registered implementation"})
            continue
        definition = registry.get(feature_id)
        implemented.append(
            {
                **entry,
                "validation_status": definition.validation_status,
                "is_novel": definition.novelty_status == "novel-hypothesis",
                "state": "registered-implementation-confirmed",
            }
        )
    return {"approved_features": implemented, "rejected_features": rejected}


@_node("run_feature_tests", "verification")
def run_feature_tests(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
    """Run the implementation-validation suite over the approved features."""
    from ..verification.feature_validation import run_feature_validation

    registry = _registry()
    approved_ids = [str(entry.get("feature_id")) for entry in (state.approved_features or [])]
    subset = registry.subset(approved_ids) if approved_ids else registry
    report = run_feature_validation(subset, ctx.config)
    failed = [check.get("name") for check in (report.get("checks") or []) if not check.get("passed", True)]
    return {"feature_tests": {**report, "n_features_tested": len(subset), "failed_checks": failed}}


@_node("extract_features", "feature")
def extract_features_node(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
    """Extract the feature table (cached by a content hash of the epochs)."""
    return {"feature_extraction_results": ctx.pipeline.extract_features()}


# --------------------------------------------------------------------------- #
# analysis nodes
# --------------------------------------------------------------------------- #
@_node("analyze_feature_distributions", "statistics")
def analyze_feature_distributions(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
    """Per-family distribution and missingness summary of the extracted table."""
    import numpy as np
    import pandas as pd

    from ..eeg.feature_extraction import feature_columns

    table = ctx.artifacts.feature_table
    if table is None:
        raise RuntimeError("no feature table; run extract_features first")
    registry = _registry()
    rows: list[dict[str, Any]] = []
    for feature in feature_columns(table):
        values = pd.to_numeric(table[feature], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        family = registry.get(feature).family if feature in registry else "sequence"
        rows.append(
            {
                "feature": feature,
                "family": family,
                "n_finite": int(finite.size),
                "missing_fraction": round(float(1 - finite.size / max(1, values.size)), 6),
                "mean": float(np.mean(finite)) if finite.size else None,
                "std": float(np.std(finite)) if finite.size else None,
                "min": float(np.min(finite)) if finite.size else None,
                "max": float(np.max(finite)) if finite.size else None,
            }
        )
    by_family: dict[str, dict[str, float]] = {}
    for row in rows:
        stats = by_family.setdefault(str(row["family"]), {"n_features": 0, "mean_missing_fraction": 0.0})
        stats["n_features"] += 1
        stats["mean_missing_fraction"] += float(row["missing_fraction"] or 0.0)
    for stats in by_family.values():
        stats["mean_missing_fraction"] = round(
            stats["mean_missing_fraction"] / max(1, stats["n_features"]), 6
        )
    return {
        "feature_distributions": {
            "n_features": len(rows),
            "by_family": by_family,
            "degenerate_features": [row["feature"] for row in rows if row["n_finite"] == 0],
            "highest_missing_fraction": sorted(
                rows, key=lambda row: row["missing_fraction"] or 0, reverse=True
            )[:10],
        }
    }


@_node("run_statistical_tests", "statistics")
def run_statistical_tests(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
    """Compare every feature across stages with assumption-gated, corrected tests."""
    return {"statistical_results": ctx.pipeline.statistics()}


@_node("run_feature_selection", "statistics")
def run_feature_selection(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
    """Extract the training-only feature-selection and interaction results."""
    path = ctx.config.path("tables") / "statistical_results.json"
    selection: dict[str, Any] = {}
    if path.is_file():
        with open(path, encoding="utf-8") as handle:
            selection = dict(json.load(handle).get("feature_selection") or {})
    interactions = list(selection.get("interactions") or [])
    return {
        "feature_selection_results": {
            "n_train_rows": selection.get("n_train_rows"),
            "n_train_subjects": selection.get("n_train_subjects"),
            "top_k_mutual_information": selection.get("top_k_mutual_information"),
            "redundancy_clusters": (selection.get("redundancy") or {}).get("n_clusters"),
            "rfe": selection.get("rfe"),
            "n_interaction_candidates": len(selection.get("interaction_candidates") or []),
            "n_interactions_passing_screen": sum(
                1 for entry in interactions if entry.get("passes_heuristic_screen")
            ),
        }
    }


@_node("evaluate_predictive_models", "statistics")
def evaluate_predictive_models(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
    """Subject-grouped ablation study evaluated on held-out subjects."""
    return {"predictive_results": ctx.pipeline.evaluate()}


@_node("verify_results", "verification")
def verify_results(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
    """Run the adversarial verification suite, then ask for a critique of it."""
    report = ctx.pipeline.verify()
    top: list[dict[str, Any]] = []
    results_path = ctx.config.path("tables") / "statistical_results.json"
    if results_path.is_file():
        with open(results_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        top = [
            {
                "feature": item.get("feature"),
                "test": (item.get("primary") or {}).get("test"),
                "p_adjusted": (item.get("primary") or {}).get("p_adjusted"),
                "effect": ((item.get("primary") or {}).get("effect_size") or {}).get("value"),
                "significant": (item.get("primary") or {}).get("significant_after_correction"),
            }
            for item in (payload.get("comparisons") or [])[:10]
        ]
    top.append(
        {"verification": {key: report.get(key) for key in ("ok", "failed_checks", "n_failed_checks")}}
    )
    system, user, metadata = prompts.build_interpretation_prompt(top, state.research_question)
    reply = ctx.ask(system, user, metadata)
    return {
        "verification_results": report,
        "agent_notes": {
            "verify_results": reply.text if reply is not None and reply.ok else "critique unavailable"
        },
    }


@_node("generate_report", "reporting")
def generate_report(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
    """Write the five reports plus the claim register from saved artifacts."""
    return {"research_report": dict(ctx.pipeline.report())}


@_node("publish_artifacts", "reporting")
def publish_artifacts(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
    """Persist the execution trace and tool log; never pushes to a remote."""
    logs = ctx.config.path("logs")
    logs.mkdir(parents=True, exist_ok=True)
    trace_path = logs / "graph_run.json"
    payload = {
        "research_question": state.research_question,
        "mode": state.mode,
        "origin": state.origin,
        "stages_completed": list(state.stages_completed),
        "errors": list(state.errors),
        "llm_calls": list(ctx.trace),
        "stage_trace": list(ctx.stage_trace),
        "tools": ctx.toolbox.to_dict(),
        "reports": state.research_report,
        "data_origin": ctx.artifacts.data_origin,
        "note": (
            "This trace is written locally. Publishing to GitHub is a separate, explicit step and "
            "no credentials are ever written here."
        ),
    }
    with open(trace_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    manifest = logs / "execution_manifest.json"
    return {
        "execution_metadata": {
            "graph_trace": str(trace_path),
            "execution_manifest": str(manifest) if manifest.is_file() else None,
            "n_llm_calls": len(ctx.trace),
            "tool_calls": len(ctx.toolbox.calls),
        }
    }


def build_nodes() -> dict[str, Callable[[ResearchState, AgentContext], dict[str, Any]]]:
    """Map node name -> callable, in workflow order."""
    mapping = {
        "initialize_project": initialize_project,
        "acquire_dataset": acquire_dataset,
        "inspect_dataset": inspect_dataset,
        "prepare_dataset": prepare_dataset,
        "search_literature": search_literature,
        "extract_paper_evidence": extract_paper_evidence,
        "generate_feature_hypotheses": generate_feature_hypotheses,
        "validate_feature_specifications": validate_feature_specifications,
        "implement_candidate_features": implement_candidate_features,
        "run_feature_tests": run_feature_tests,
        "extract_features": extract_features_node,
        "analyze_feature_distributions": analyze_feature_distributions,
        "run_statistical_tests": run_statistical_tests,
        "run_feature_selection": run_feature_selection,
        "evaluate_predictive_models": evaluate_predictive_models,
        "verify_results": verify_results,
        "generate_report": generate_report,
        "publish_artifacts": publish_artifacts,
    }
    missing = set(NODE_NAMES) - set(mapping)
    assert not missing, f"nodes missing from the table: {sorted(missing)}"
    return mapping


def node_names() -> list[str]:
    """Workflow order used by the graph builder and the tests."""
    return list(NODE_NAMES)
