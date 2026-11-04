"""Workflow tests: routing, node contracts, tool permissions and a real run.

The LLM is mocked (the offline provider is the default), so these tests need no
network and no API key.  The graph is exercised twice: once against a stub
pipeline (fast, checks routing and error handling) and once end to end on the
small synthetic corpus (marked ``slow``).
"""

from __future__ import annotations

import pytest

from eeg_research_agent.agents.graph import (
    build_research_graph,
    route_after_acquire,
    route_after_split,
    route_after_validation,
)
from eeg_research_agent.agents.nodes import NODE_NAMES, AgentContext, build_nodes
from eeg_research_agent.agents.state import ResearchState, initial_state
from eeg_research_agent.agents.tools import AGENT_TOOLS, ResearchToolbox, ToolNotPermitted, tool_table


class StubPipeline:
    """A minimal stand-in that records the calls the nodes make."""

    def __init__(self, config, fail_on: set[str] | None = None, split_ok: bool = True) -> None:
        self.config = config
        self.calls: list[str] = []
        self.fail_on = set(fail_on or ())
        self.split_ok = split_ok
        self.results: dict[str, object] = {}
        self.feature_table = None

        class _Artifacts:
            feature_table = None
            data_origin = "synthetic"
            splits = None
            recordings: list[object] = []

        self.artifacts = _Artifacts()

    def _record(self, name: str) -> None:
        self.calls.append(name)
        if name in self.fail_on:
            raise RuntimeError(f"stub failure in {name}")

    def initialise(self):
        self._record("initialise")
        return {"mode": "smoke"}

    def acquire(self, origin=None):
        self._record("acquire")
        return {"origin": "synthetic", "n_recordings": 1, "manifest": "m.json", "subjects": ["S1"]}

    def inspect(self):
        self._record("inspect")
        return {"dataset": "d", "data_origin": "synthetic", "sampling_frequency": 100.0}

    def preprocess(self):
        self._record("preprocess")
        return {"n_recordings": 1}

    def segment_and_split(self):
        self._record("segment_and_split")
        return {"segment": {"n_epochs": 10}, "split": {"verification": {"ok": self.split_ok}}}

    def literature(self, **kwargs):
        self._record("literature")
        return {"mode": "offline", "n_papers": 0, "retrieval_available": False, "query_log": [], "papers": []}


def test_node_table_covers_every_declared_node():
    assert set(build_nodes()) == set(NODE_NAMES)
    assert len(NODE_NAMES) == 18, "the brief's workflow has eighteen stages"


def test_routing_decisions():
    healthy = initial_state(mode="smoke")
    assert route_after_acquire(healthy) == "inspect_dataset"
    assert route_after_split(healthy) == "search_literature"
    assert route_after_validation(healthy) == "extract_features", "no approvals means nothing to implement"

    broken = initial_state(mode="smoke")
    broken.execution_metadata["acquire_dataset_failed"] = True
    assert route_after_acquire(broken) == "publish_artifacts"
    broken_split = initial_state(mode="smoke")
    broken_split.subject_splits = {"verification": {"ok": False}}
    assert route_after_split(broken_split) == "publish_artifacts"
    approved = initial_state(mode="smoke")
    approved.approved_features = [{"feature_id": "abs_power_delta"}]
    assert route_after_validation(approved) == "implement_candidate_features"


def test_failing_node_is_recorded_not_swallowed(base_config):
    pipeline = StubPipeline(base_config, fail_on={"acquire"})
    context = AgentContext(pipeline=pipeline)
    nodes = build_nodes()
    update = nodes["acquire_dataset"](initial_state(mode="smoke"), context)
    assert update["errors"] and "acquire_dataset" in update["errors"][0]
    assert update["execution_metadata"]["acquire_dataset_failed"] is True
    assert update["stages_completed"] == ["acquire_dataset"]


def test_tool_permissions_are_enforced():
    table = tool_table()
    assert any(entry["name"] == "search_papers" for entry in table)
    assert "statistics" not in AGENT_TOOLS["search_papers"].roles

    statistics_tools = ResearchToolbox(roles=["statistics"])
    distribution = statistics_tools.describe_feature_distribution.__name__
    assert distribution  # the tool is reachable for this role
    with pytest.raises(AttributeError):
        statistics_tools.not_a_tool()
    verification_only = ResearchToolbox(roles=["statistics"])
    with pytest.raises(ToolNotPermitted):
        verification_only.for_role("statistics").generate_synthetic_signal(kind="sine", freq=1.0, n_samples=100, sfreq=100.0)


def test_llm_task_returns_narration_without_a_key(base_config, offline_llm, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    context = AgentContext(pipeline=StubPipeline(base_config), llm=offline_llm)
    from eeg_research_agent.agents import prompts

    system, user, metadata = prompts.build_dataset_summary_prompt(
        {"dataset": "d", "data_origin": "synthetic", "subject_count": 2, "epoch_count": 5}
    )
    reply = context.ask(system, user, metadata)
    assert reply is not None and reply.ok
    assert "synthetic" in reply.text
    assert context.trace[-1]["task"] == "dataset_summary"


def test_graph_construction_and_dry_route(base_config):
    pipeline = StubPipeline(base_config)
    graph, context = build_research_graph(config=base_config, pipeline=pipeline)
    assert graph is not None
    assert context.pipeline is pipeline
    state = initial_state(mode="smoke")
    nodes = build_nodes()
    for name in ("initialize_project", "acquire_dataset", "inspect_dataset"):
        state_dict = nodes[name](state, context)
        assert "stages_completed" in state_dict
    assert pipeline.calls[:3] == ["initialise", "acquire", "inspect"]


@pytest.mark.slow
def test_full_agentic_workflow_runs_offline(base_config):
    """End-to-end graph run on the small synthetic corpus with the offline LLM."""
    from eeg_research_agent.agents.graph import run_research_workflow
    from eeg_research_agent.pipeline import ResearchPipeline

    pipeline = ResearchPipeline(config=base_config, mode="smoke", origin="synthetic")
    summary = run_research_workflow(config=base_config, pipeline=pipeline, mode="smoke", origin="synthetic")
    assert summary["errors"] == [], summary["errors"]
    assert "publish_artifacts" in summary["stages_completed"]
    assert summary["n_features_approved"] > 0
    assert summary["n_features_rejected"] == 0
    assert summary["verification_ok"] is True
    assert summary["reports"], "the workflow must write reports"
    assert summary["llm_calls"], "the LLM steps must run (offline provider by default)"
    assert all(call.get("provider") == "offline" for call in summary["llm_calls"])


@pytest.mark.slow
def test_state_accumulates_errors_and_stage_names(tmp_path, base_config):
    from eeg_research_agent.agents.nodes import initialize_project, inspect_dataset

    def apply(state: ResearchState, update: dict) -> ResearchState:
        """Merge a node update the way the graph does: reducers concatenate."""
        data = state.model_dump()
        for key, value in update.items():
            if key in ("stages_completed", "errors"):
                data[key] = list(data.get(key) or []) + list(value)
            else:
                data[key] = value
        return ResearchState(**data)

    pipeline = StubPipeline(base_config, fail_on={"inspect"})
    context = AgentContext(pipeline=pipeline)
    state = initial_state(mode="smoke")
    state = apply(state, initialize_project(state, context))
    update = inspect_dataset(state, context)
    merged = apply(state, update)
    assert any("inspect_dataset" in message for message in merged.errors)
    assert "initialize_project" in merged.stages_completed
    assert "inspect_dataset" in merged.stages_completed
