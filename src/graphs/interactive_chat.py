"""Minimal LangGraph scaffold for the interactive chat workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.agents.contracts import RouteDecision, VerificationReport

from .events import GraphEvent
from .models import GraphNodeName, GraphWorkflowKind
from .state import GraphState

try:  # pragma: no cover - optional dependency during incremental rollout
    from langgraph.graph import END, START, StateGraph
except Exception as exc:  # pragma: no cover - import guard for local dev without dependency
    START = "__start__"  # type: ignore[assignment]
    END = "__end__"  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]
    _LANGGRAPH_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised when dependency is installed
    _LANGGRAPH_IMPORT_ERROR = None


class LangGraphUnavailableError(RuntimeError):
    """Raised when the local environment does not have LangGraph installed."""


NodeHandler = Callable[[GraphState], dict[str, Any]]


@dataclass(frozen=True)
class GraphNodeSpec:
    """Single graph node description plus its state handler."""

    name: GraphNodeName
    handler: NodeHandler
    description: str = ""


@dataclass(frozen=True)
class GraphEdgeSpec:
    """Directed graph edge."""

    source: str
    target: str


@dataclass(frozen=True)
class InteractiveChatGraphBlueprint:
    """Portable description of the interactive chat graph."""

    workflow_kind: GraphWorkflowKind = "interactive_chat"
    nodes: tuple[GraphNodeSpec, ...] = field(default_factory=tuple)
    edges: tuple[GraphEdgeSpec, ...] = field(default_factory=tuple)
    entrypoint: str = "preflight"
    finish_point: str = "finalize"

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_kind": self.workflow_kind,
            "entrypoint": self.entrypoint,
            "finish_point": self.finish_point,
            "nodes": [
                {
                    "name": node.name,
                    "description": node.description,
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                }
                for edge in self.edges
            ],
        }

    def compile(self) -> Any:
        """Compile the blueprint into a LangGraph graph when available."""

        if StateGraph is None:
            raise LangGraphUnavailableError(
                "LangGraph is not installed in this environment."
            ) from _LANGGRAPH_IMPORT_ERROR

        graph = StateGraph(GraphState)
        for node in self.nodes:
            graph.add_node(node.name, node.handler)
        for edge in self.edges:
            graph.add_edge(edge.source, edge.target)
        graph.add_edge(START, self.entrypoint)
        graph.add_edge(self.finish_point, END)
        graph.add_conditional_edges(
            "deliberation",
            _choose_execution_path,
            {
                "fast_direct": "fast_direct",
                "route": "route",
            },
        )
        graph.add_conditional_edges(
            "verify",
            _choose_verification_path,
            {
                "repair": "repair",
                "synthesize": "synthesize",
            },
        )
        return graph.compile()


def build_interactive_chat_graph_blueprint() -> InteractiveChatGraphBlueprint:
    """Create the production-oriented interactive chat graph description."""

    return InteractiveChatGraphBlueprint(
        nodes=(
            GraphNodeSpec("preflight", _preflight_node, "Normalize incoming state."),
            GraphNodeSpec("deliberation", _deliberation_node, "Derive routing policy."),
            GraphNodeSpec("route", _route_node, "Resolve the chosen path."),
            GraphNodeSpec("fast_direct", _fast_direct_node, "Short direct path."),
            GraphNodeSpec("solve", _solve_node, "Specialist solve step."),
            GraphNodeSpec("verify", _verify_node, "Check solve output."),
            GraphNodeSpec("repair", _repair_node, "Optional repair step."),
            GraphNodeSpec("synthesize", _synthesize_node, "Produce the response."),
            GraphNodeSpec("finalize", _finalize_node, "Mark completion."),
        ),
        edges=(
            GraphEdgeSpec("preflight", "deliberation"),
            GraphEdgeSpec("route", "solve"),
            GraphEdgeSpec("solve", "verify"),
            GraphEdgeSpec("repair", "synthesize"),
            GraphEdgeSpec("synthesize", "finalize"),
            GraphEdgeSpec("fast_direct", "finalize"),
        ),
    )


def build_interactive_chat_graph() -> Any:
    """Build the interactive chat graph or return a blueprint when LangGraph is absent."""

    blueprint = build_interactive_chat_graph_blueprint()
    if StateGraph is None:
        return blueprint
    return blueprint.compile()


def _stamp_event(state: GraphState, *, phase: str, status: str, message: str) -> GraphEvent:
    return GraphEvent(
        event_type="state.updated",
        workflow_kind=str(state.get("workflow_kind") or "interactive_chat"),
        phase=phase,
        status=status,
        agent_role="triage" if phase in {"preflight", "deliberation", "route"} else None,
        node=phase,
        message=message,
        payload={"selected_domains": list(state.get("selected_domains") or [])},
    )


def _node_patch(state: GraphState, *, phase: str, message: str, **updates: Any) -> dict[str, Any]:
    patch: dict[str, Any] = {"graph_events": [_stamp_event(state, phase=phase, status="completed", message=message)]}
    patch.update(updates)
    return patch


def _preflight_node(state: GraphState) -> dict[str, Any]:
    return _node_patch(
        state,
        phase="preflight",
        message="Turn state normalized.",
        status="running",
    )


def _deliberation_node(state: GraphState) -> dict[str, Any]:
    route = state.get("route") or RouteDecision(
        selected_domains=list(state.get("selected_domains") or ["core"]),
        reason="graph_scaffold_default",
    )
    return _node_patch(
        state,
        phase="deliberation",
        message="Deliberation scaffold completed.",
        route=route,
        selected_domains=list(route.selected_domains),
    )


def _route_node(state: GraphState) -> dict[str, Any]:
    selected_domains = list(state.get("selected_domains") or ["core"])
    return _node_patch(
        state,
        phase="route",
        message="Routing scaffold completed.",
        selected_domains=selected_domains,
    )


def _choose_execution_path(state: GraphState) -> str:
    selected_domains = [str(token).strip() for token in state.get("selected_domains") or [] if str(token).strip()]
    selected_tool_names = [
        str(token).strip() for token in state.get("selected_tool_names") or [] if str(token).strip()
    ]
    workflow_hint = dict(state.get("workflow_hint") or {})
    workflow_id = str(workflow_hint.get("id") or "").strip().lower()
    if workflow_id in {"direct", "fast_direct"}:
        return "fast_direct"
    if not selected_domains:
        return "fast_direct"
    if len(selected_domains) == 1 and selected_domains[0] == "core" and not selected_tool_names:
        return "fast_direct"
    return "route"


def _fast_direct_node(state: GraphState) -> dict[str, Any]:
    return _node_patch(state, phase="fast_direct", message="Fast-direct scaffold completed.")


def _solve_node(state: GraphState) -> dict[str, Any]:
    return _node_patch(state, phase="solve", message="Specialist solve scaffold completed.")


def _verify_node(state: GraphState) -> dict[str, Any]:
    verification = state.get("verification") or VerificationReport(
        passed=True,
        notes=["graph_scaffold_placeholder"],
    )
    return _node_patch(
        state,
        phase="verify",
        message="Verification scaffold completed.",
        verification=verification,
    )


def _repair_node(state: GraphState) -> dict[str, Any]:
    return _node_patch(state, phase="repair", message="Repair scaffold completed.")


def _synthesize_node(state: GraphState) -> dict[str, Any]:
    response_text = str(state.get("response_text") or "").strip()
    if not response_text:
        response_text = "LangGraph scaffold synthesized the turn state."
    return _node_patch(
        state,
        phase="synthesize",
        message="Synthesis scaffold completed.",
        response_text=response_text,
    )


def _finalize_node(state: GraphState) -> dict[str, Any]:
    return _node_patch(state, phase="finalize", message="Graph execution finalized.", status="succeeded")


def _choose_verification_path(state: GraphState) -> str:
    verification = state.get("verification")
    if isinstance(verification, VerificationReport) and not verification.passed:
        return "repair"
    return "synthesize"
