"""Agentic orchestration layer.

``state``    the typed ``ResearchState`` carried through the graph
``prompts``  the prompt contracts for the four LLM-facing tasks
``tools``    narrow, role-scoped tools (agents never execute generated code)
``nodes``    one node per workflow stage; each calls the deterministic pipeline
``graph``    the LangGraph workflow with explicit conditional routing

Everything scientific lives outside this package: nodes delegate to
``eeg_research_agent.pipeline``.  Removing the LLM entirely (the default
``offline`` provider) leaves the workflow fully functional.
"""

from .graph import (
    build_research_graph,
    route_after_acquire,
    route_after_split,
    route_after_validation,
    run_research_workflow,
)
from .nodes import NODE_NAMES, AgentContext, build_nodes, node_names
from .state import ResearchState, StageRecord, initial_state
from .tools import AGENT_TOOLS, ResearchToolbox, ToolNotPermitted, ToolSpec, tool_table

__all__ = [
    "ResearchState",
    "StageRecord",
    "initial_state",
    "AgentContext",
    "NODE_NAMES",
    "build_nodes",
    "node_names",
    "build_research_graph",
    "run_research_workflow",
    "route_after_acquire",
    "route_after_split",
    "route_after_validation",
    "AGENT_TOOLS",
    "ResearchToolbox",
    "ToolNotPermitted",
    "ToolSpec",
    "tool_table",
]
