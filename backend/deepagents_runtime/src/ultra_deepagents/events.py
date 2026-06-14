from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from ultra_deepagents.context import AgentRunContext


@dataclass(frozen=True)
class RunEvent:
    run_id: str
    event_kind: str
    payload: dict[str, Any]
    thread_id: str | None = None
    event_type: str | None = None
    node_name: str | None = None
    task_id: str | None = None
    checkpoint_id: str | None = None
    scope_id: str | None = None
    agent_role: str | None = None
    level: str | None = None
    message: str | None = None
    ts: str = field(default_factory=lambda: _utc_timestamp())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_message_delta(
    context: AgentRunContext,
    text: str,
    *,
    source: str = "coordinator",
) -> dict[str, Any]:
    return RunEvent(
        run_id=context.run_id,
        thread_id=context.thread_id,
        event_kind="message.delta",
        event_type="message",
        node_name=source,
        agent_role=source,
        level="info",
        message=text,
        payload={"text": text, "source": source},
    ).to_dict()


def normalize_reasoning_delta(
    context: AgentRunContext,
    text: str,
    *,
    status: str = "running",
    source: str = "coordinator",
) -> dict[str, Any]:
    """Coalesced model-thinking text, streamed so the UI can show progress
    during the otherwise-silent reasoning phase. Never part of the answer."""
    return RunEvent(
        run_id=context.run_id,
        thread_id=context.thread_id,
        event_kind="trace.reasoning.delta",
        event_type="trace",
        node_name="reasoning",
        agent_role="reasoning",
        level="debug",
        message=text,
        payload={"text": text, "source": source, "status": status},
    ).to_dict()


def normalize_tool_call(
    context: AgentRunContext,
    tool_name: str,
    status: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event_payload = {"tool_name": tool_name, "status": status}
    event_payload.update(_json_safe_payload(payload))
    return RunEvent(
        run_id=context.run_id,
        thread_id=context.thread_id,
        event_kind=f"tool_call.{status}",
        event_type="tool_call",
        node_name=f"tool:{tool_name}",
        agent_role="tool",
        level="info" if status not in {"failed", "error"} else "error",
        message=f"{tool_name} {status}",
        payload=event_payload,
    ).to_dict()


def normalize_subagent_status(
    context: AgentRunContext,
    name: str,
    task_id: str,
    status: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event_payload = {"name": name, "status": status}
    event_payload.update(_json_safe_payload(payload))
    return RunEvent(
        run_id=context.run_id,
        thread_id=context.thread_id,
        event_kind=f"subagent.{status}",
        event_type="subagent",
        node_name=name,
        task_id=task_id,
        agent_role=name,
        level="info" if status not in {"failed", "error"} else "error",
        message=f"{name} {status}",
        payload=event_payload,
    ).to_dict()


def normalize_subagent_message_delta(
    context: AgentRunContext,
    name: str,
    text: str,
    *,
    task_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event_payload = {"text": text, "source": name}
    event_payload.update(_json_safe_payload(payload))
    return RunEvent(
        run_id=context.run_id,
        thread_id=context.thread_id,
        event_kind="subagent.message.delta",
        event_type="subagent_message",
        node_name=name,
        task_id=task_id,
        agent_role=name,
        level="info",
        message=text,
        payload=event_payload,
    ).to_dict()


def normalize_artifact_created(
    context: AgentRunContext,
    *,
    artifact_id: str,
    path: str,
    title: str = "",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event_payload = {
        "artifact_id": artifact_id,
        "path": path,
        "title": title,
    }
    event_payload.update(_json_safe_payload(payload))
    return RunEvent(
        run_id=context.run_id,
        thread_id=context.thread_id,
        event_kind="artifact.created",
        event_type="artifact",
        node_name="artifact",
        level="info",
        message=title or path,
        payload=event_payload,
    ).to_dict()


def normalize_run_completed(
    context: AgentRunContext,
    response_text: str,
    *,
    conversation_title: str = "",
    title_generation: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {"response_text": response_text}
    if conversation_title.strip():
        payload["conversation_title"] = conversation_title.strip()
    if title_generation:
        payload["title_generation"] = _json_safe_payload(title_generation)
    if usage:
        payload["usage"] = _json_safe_payload(usage)
    return RunEvent(
        run_id=context.run_id,
        thread_id=context.thread_id,
        event_kind="run.completed",
        event_type="run",
        node_name="coordinator",
        agent_role="coordinator",
        level="info",
        message=response_text,
        payload=payload,
    ).to_dict()


def normalize_token_usage(
    context: AgentRunContext,
    usage: dict[str, Any],
    *,
    model: str = "",
    usage_event_id: str = "",
) -> dict[str, Any]:
    payload = {
        "usage_event_id": usage_event_id,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }
    if model:
        payload["model"] = model
    return RunEvent(
        run_id=context.run_id,
        thread_id=context.thread_id,
        event_kind="run.token_usage",
        event_type="run",
        node_name="coordinator",
        agent_role="coordinator",
        level="debug",
        message="",
        payload=payload,
    ).to_dict()


def normalize_run_failed(context: AgentRunContext, error: BaseException | str) -> dict[str, Any]:
    message = str(error)
    error_type = type(error).__name__ if isinstance(error, BaseException) else "Error"
    return RunEvent(
        run_id=context.run_id,
        thread_id=context.thread_id,
        event_kind="run.failed",
        event_type="run",
        node_name="coordinator",
        agent_role="coordinator",
        level="error",
        message=message,
        payload={"error": message, "error_type": error_type},
    ).to_dict()


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _json_safe_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    return json.loads(json.dumps(payload, default=str))
