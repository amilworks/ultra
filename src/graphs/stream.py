"""Stream and SSE adapters for graph-native execution."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field

from .events import GraphEvent, GraphStreamEnvelope, coerce_graph_event


class StreamFrame(BaseModel):
    """Normalized frame emitted to the client or a worker bridge."""

    event: str
    data: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None
    retry: int | None = None


def graph_stream_event_to_dict(event: GraphEvent | dict[str, Any]) -> dict[str, Any]:
    """Convert a graph event into a JSON-serializable record."""

    graph_event = coerce_graph_event(event) if not isinstance(event, GraphEvent) else event
    payload = graph_event.model_dump(mode="json")
    payload["event"] = "run_event"
    payload["data"] = {
        "event_type": graph_event.event_type,
        "phase": graph_event.phase,
        "status": graph_event.status,
        "agent_role": graph_event.agent_role,
        "node": graph_event.node,
        "domain_id": graph_event.domain_id,
        "task_id": graph_event.task_id,
        "checkpoint_id": graph_event.checkpoint_id,
        "scope_id": graph_event.scope_id,
        "message": graph_event.message,
        "payload": graph_event.payload,
        "ts": payload.pop("ts"),
    }
    return payload


def graph_event_to_sse_frame(event: GraphEvent | dict[str, Any]) -> str:
    """Serialize one graph event as an SSE frame."""

    frame = graph_stream_event_to_dict(event)
    return format_sse_frame("run_event", frame)


def format_sse_frame(event: str, data: dict[str, Any], *, id: str | None = None) -> str:
    """Serialize a server-sent event frame."""

    lines: list[str] = []
    if id is not None:
        lines.append(f"id: {id}")
    lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data, default=str, separators=(',', ':'))}")
    return "\n".join(lines) + "\n\n"


def iter_sse_frames(frames: Iterable[GraphEvent | StreamFrame | GraphStreamEnvelope | dict[str, Any]]) -> str:
    """Join multiple graph or token frames into an SSE stream body."""

    chunks: list[str] = []
    for frame in frames:
        if isinstance(frame, GraphEvent):
            chunks.append(graph_event_to_sse_frame(frame))
            continue
        if isinstance(frame, GraphStreamEnvelope):
            chunks.append(format_sse_frame(frame.event, frame.data, id=frame.id))
            continue
        if isinstance(frame, StreamFrame):
            chunks.append(format_sse_frame(frame.event, frame.data, id=frame.id))
            continue
        event_name = str(frame.get("event") or "run_event")
        chunks.append(format_sse_frame(event_name, dict(frame)))
    return "".join(chunks)
