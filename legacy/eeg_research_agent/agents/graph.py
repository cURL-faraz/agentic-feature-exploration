"""The LangGraph research workflow.

Why a graph and not a script that calls an LLM a few times?

* **The workflow is stateful and resumable.**  Stages are expensive (dataset
  acquisition, feature extraction, statistical machinery), so execution state is
  checkpointed and a failed run can be re-entered without repeating work.
* **Routing is explicit and conditional.**  A failed acquisition, an unusable
  split, or an empty approved-feature set each take a *different* path that the
  reader can inspect - instead of a try/except buried inside a prompt loop.
* **Responsibilities are separated.**  Each node has one job and a bounded tool
  set; the graph, not the model, decides what runs next.
* **The science stays deterministic.**  Nodes call :class:`ResearchPipeline`; the
  LLM never sees a number it is allowed to change.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Any, Mapping

from ..logging_utils import get_logger
from ..pipeline import ResearchPipeline
from .nodes import NODE_NAMES, AgentContext, build_nodes
from .state import ResearchState, initial_state

__all__ = [
    "build_research_graph",
    "run_research_workflow",
    "route_after_acquire",
    "route_after_split",
    "route_after_validation",
]

logger = get_logger(__name__)


def _failed(state: ResearchState, node: str) -> bool:
    return bool((state.execution_metadata or {}).get(f"{node}_failed"))


def route_after_acquire(state: ResearchState) -> str:
    """A dataset that could not be obtained cannot be inspected."""
    return "publish_artifacts" if _failed(state, "acquire_dataset") else "inspect_dataset"


def route_after_split(state: ResearchState) -> str:
    """A split that does not partition subjects invalidates every later metric."""
    if _failed(state, "prepare_dataset"):
        return "publish_artifacts"
    verification = (state.subject_splits or {}).get("verification") or {}
    if verification and not verification.get("ok", True):
        return "publish_artifacts"
    return "search_literature"


def route_after_validation(state: ResearchState) -> str:
    """With no approved specifications there is nothing to implement or test."""
    if _failed(state, "validate_feature_specifications"):
        return "extract_features"
    return "implement_candidate_features" if state.approved_features else "extract_features"


def build_research_graph(
    config: Any | None = None,
    pipeline: ResearchPipeline | None = None,
    llm: Any | None = None,
    mode: str = "standard",
    origin: str | None = None,
    checkpointer: Any | None = None,
):
    """Compile the workflow.  Returns ``(graph, context)``.

    The context is returned so the caller can inspect the LLM and tool traces
    after the run.
    """
    from langgraph.graph import END, StateGraph

    try:  # pragma: no cover - import location varies across langgraph releases
        from langgraph.checkpoint.memory import InMemorySaver
    except Exception:  # pragma: no cover
        InMemorySaver = None  # type: ignore[assignment]

    pipeline = pipeline or ResearchPipeline(config=config, mode=mode, origin=origin)
    context = AgentContext(pipeline=pipeline, llm=llm)

    graph = StateGraph(ResearchState)
    for name, function in build_nodes().items():
        graph.add_node(name, partial(function, ctx=context))

    graph.set_entry_point("initialize_project")
    graph.add_edge("initialize_project", "acquire_dataset")
    graph.add_conditional_edges(
        "acquire_dataset",
        route_after_acquire,
        {"inspect_dataset": "inspect_dataset", "publish_artifacts": "publish_artifacts"},
    )
    graph.add_edge("inspect_dataset", "prepare_dataset")
    graph.add_conditional_edges(
        "prepare_dataset",
        route_after_split,
        {"search_literature": "search_literature", "publish_artifacts": "publish_artifacts"},
    )
    graph.add_edge("search_literature", "extract_paper_evidence")
    graph.add_edge("extract_paper_evidence", "generate_feature_hypotheses")
    graph.add_edge("generate_feature_hypotheses", "validate_feature_specifications")
    graph.add_conditional_edges(
        "validate_feature_specifications",
        route_after_validation,
        {
            "implement_candidate_features": "implement_candidate_features",
            "extract_features": "extract_features",
        },
    )
    graph.add_edge("implement_candidate_features", "run_feature_tests")
    # Feature tests report; they never block the run.  A failing check is recorded
    # and re-surfaced by the verification stage, which can fail the whole run.
    graph.add_edge("run_feature_tests", "extract_features")
    graph.add_edge("extract_features", "analyze_feature_distributions")
    graph.add_edge("analyze_feature_distributions", "run_statistical_tests")
    graph.add_edge("run_statistical_tests", "run_feature_selection")
    graph.add_edge("run_feature_selection", "evaluate_predictive_models")
    graph.add_edge("evaluate_predictive_models", "verify_results")
    graph.add_edge("verify_results", "generate_report")
    graph.add_edge("generate_report", "publish_artifacts")
    graph.add_edge("publish_artifacts", END)

    if checkpointer is None and InMemorySaver is not None:
        checkpointer = InMemorySaver()
    compiled = graph.compile(checkpointer=checkpointer) if checkpointer is not None else graph.compile()
    return compiled, context


def run_research_workflow(
    config: Any | None = None,
    mode: str = "standard",
    origin: str | None = None,
    llm: Any | None = None,
    pipeline: ResearchPipeline | None = None,
    research_question: str | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Run the whole workflow once and return a JSON-serialisable summary."""
    graph, context = build_research_graph(
        config=config, pipeline=pipeline, llm=llm, mode=mode, origin=origin
    )
    state = initial_state(mode=mode, origin=origin, research_question=research_question)
    run_config = {"configurable": {"thread_id": thread_id or f"run-{int(time.time())}"}}
    started = time.time()
    final = graph.invoke(state, config=run_config)
    if isinstance(final, ResearchState):
        state = final
    else:  # pragma: no cover - older langgraph returns a dict
        state = ResearchState(**{k: v for k, v in dict(final).items() if k in ResearchState.model_fields})
    duration = round(time.time() - started, 2)
    summary = {
        "duration_s": duration,
        "stages_completed": list(state.stages_completed),
        "n_stages_expected": len(NODE_NAMES),
        "errors": list(state.errors),
        "data_origin": state.dataset_metadata.get("data_origin"),
        "n_epochs": (state.feature_extraction_results or {}).get("n_epochs"),
        "n_features_extracted": (state.feature_extraction_results or {}).get("n_features"),
        "n_features_approved": len(state.approved_features),
        "n_features_rejected": len(state.rejected_features),
        "n_papers": (state.paper_evidence or {}).get("n_papers"),
        "n_significant_after_fdr": (state.statistical_results or {}).get("n_significant_after_fdr"),
        "best_test_model": (state.predictive_results or {}).get("best_test_model"),
        "verification_ok": (state.verification_results or {}).get("ok"),
        "reports": state.research_report,
        "llm_calls": list(context.trace),
        "stage_trace": list(context.stage_trace),
        "tools": context.toolbox.to_dict(),
    }
    summary["ok"] = not summary["errors"] and bool(summary["verification_ok"])
    return summary
