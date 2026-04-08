"""Typed state and reducer helpers for LangGraph execution."""

from __future__ import annotations

from operator import add
from typing import Any, TypedDict
from typing import Annotated

from src.agents.contracts import (
    AgentResult,
    KnowledgeContext,
    RouteDecision,
    ToolResultEnvelope,
    TurnIntent,
    VerificationReport,
)

from .events import GraphEvent
from .models import GraphArtifactRef, GraphExecutionStatus, GraphTaskRef, GraphWorkflowKind


class GraphState(TypedDict, total=False):
    """Mutable graph state shared across nodes and checkpoints."""

    thread_id: str
    run_id: str
    user_id: str
    conversation_id: str
    workflow_kind: GraphWorkflowKind
    status: GraphExecutionStatus
    turn_intent: TurnIntent
    route: RouteDecision
    selected_domains: Annotated[list[str], _merge_unique_strings]
    selected_tool_names: Annotated[list[str], _merge_unique_strings]
    workflow_hint: Annotated[dict[str, Any], _merge_dicts]
    knowledge_context: KnowledgeContext
    tool_state: Annotated[dict[str, Any], _merge_dicts]
    messages: Annotated[list[dict[str, str]], add]
    messages_with_context: Annotated[list[dict[str, str]], add]
    agent_results: Annotated[dict[str, AgentResult], _merge_dicts]
    verification: VerificationReport
    tool_results: Annotated[dict[str, ToolResultEnvelope], _merge_dicts]
    response_text: str
    graph_events: Annotated[list[GraphEvent], append_graph_events]
    artifacts: Annotated[list[GraphArtifactRef], add]
    tasks: Annotated[list[GraphTaskRef], add]
    pending_hitl: Annotated[dict[str, Any], _merge_dicts]
    latest_checkpoint_id: str
    metadata: Annotated[dict[str, Any], _merge_dicts]
    retry_count: int


def append_graph_events(
    existing: list[GraphEvent] | None, incoming: list[GraphEvent] | None
) -> list[GraphEvent]:
    """Append-only reducer for event streams and checkpoints."""

    combined = list(existing or [])
    combined.extend(list(incoming or []))
    return combined


def _merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = dict(left or {})
    merged.update(dict(right or {}))
    return merged


def merge_graph_state(base: dict[str, Any] | None, patch: dict[str, Any] | None) -> dict[str, Any]:
    """Shallow merge helper for graph node outputs."""

    merged: dict[str, Any] = dict(base or {})
    for key, value in dict(patch or {}).items():
        if value is None:
            continue
        if key == "graph_events" and isinstance(value, list):
            merged[key] = append_graph_events(
                merged.get(key) if isinstance(merged.get(key), list) else None,
                value,
            )
            continue
        if key == "metadata" and isinstance(value, dict):
            merged[key] = _merge_dicts(dict(merged.get(key) or {}), value)
            continue
        if key == "pending_hitl" and isinstance(value, dict):
            merged[key] = _merge_dicts(dict(merged.get(key) or {}), value)
            continue
        if key in {"workflow_hint", "tool_state", "agent_results", "tool_results"} and isinstance(
            value, dict
        ):
            merged[key] = _merge_dicts(dict(merged.get(key) or {}), value)
            continue
        if key in {"selected_domains", "selected_tool_names"} and isinstance(value, list):
            merged[key] = _merge_unique_strings(
                list(merged.get(key) or []) + [str(item).strip() for item in value if str(item).strip()]
            )
            continue
        merged[key] = value
    return merged


def new_graph_state(**kwargs: Any) -> GraphState:
    """Create a normalized graph state payload."""

    state: GraphState = {
        "status": "queued",
        "workflow_kind": "interactive_chat",
        "selected_domains": [],
        "selected_tool_names": [],
        "workflow_hint": {},
        "tool_state": {},
        "messages": [],
        "messages_with_context": [],
        "agent_results": {},
        "tool_results": {},
        "graph_events": [],
        "artifacts": [],
        "tasks": [],
        "pending_hitl": {},
        "retry_count": 0,
    }
    return merge_graph_state(state, kwargs)  # type: ignore[return-value]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in values:
        token = str(item or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _merge_unique_strings(left: list[str] | None, right: list[str] | None) -> list[str]:
    return _dedupe_preserve_order(list(left or []) + list(right or []))
