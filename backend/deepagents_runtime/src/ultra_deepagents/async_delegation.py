"""Ultra run-scope + tenant-isolation wrapper around the deep agents async-subagent tools.

EXPERIMENTAL / dormant. Async subagents federate to an EXTERNAL Agent Protocol /
LangGraph deployment (``langgraph_sdk`` client). Ultra runs systemd + NATS + a Go
control plane and does NOT host a LangGraph deployment, so this is off by default
(``ULTRA_DEEPAGENTS_ENABLE_ASYNC_SUBAGENTS`` unset) and has never been validated
against a live server. ASGI transport (``url=None``) is impossible without a
co-deployed graph; the only viable config is HTTP to a remote deployment that does
not currently exist. For in-Ultra background work, prefer the NATS run backbone
(see ``rarespot/tools.py``). See ``planning/`` async-subagents review for the path
to a NATS-native background-job tool trio.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlparse

from langchain.agents.middleware.types import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph_sdk import get_client, get_sync_client
from langgraph_sdk.client import LangGraphClient, SyncLangGraphClient

from ultra_deepagents.config import allow_private_async_subagent_url, is_local_http_host
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.context_tools import public_selected_resource_descriptor

logger = logging.getLogger(__name__)

# Shared with the context-URI guards so the server-URL check cannot drift.
_is_local_http_host = is_local_http_host

_ASYNC_TASK_TOOLS_WITH_REMOTE_RUNS = {"start_async_task", "update_async_task"}
_ASYNC_TASK_TOOLS_REQUIRING_TRACKED_SUBAGENT = {
    "check_async_task",
    "update_async_task",
    "cancel_async_task",
}
_ASYNC_TASK_TOOLS_REQUIRING_SYNC_URL = {
    "cancel_async_task",
    "check_async_task",
    "update_async_task",
}
_ASYNC_TASK_LIST_TOOL = "list_async_tasks"
_TERMINAL_ASYNC_TASK_STATUSES = {
    "cancelled",
    "canceled",
    "complete",
    "completed",
    "error",
    "failed",
    "failure",
    "interrupted",
    "success",
    "succeeded",
    "timeout",
}
_FAILED_TERMINAL_ASYNC_TASK_STATUSES = {
    "error",
    "failed",
    "failure",
    "interrupted",
    "timeout",
}
_FAILED_ASYNC_RUN_ID = "__ultra_async_launch_failed__"
_ASYNC_TASK_REQUIRED_FIELDS = (
    "task_id",
    "agent_name",
    "thread_id",
    "run_id",
    "status",
    "created_at",
    "last_checked_at",
    "last_updated_at",
)
_ASYNC_CONTEXT_RESOURCE_DESCRIPTOR_FIELDS = {
    "type",
    "artifact_id",
    "output_id",
    "run_id",
    "kind",
    "title",
    "path",
    "relative_path",
    "mime_type",
    "size_bytes",
    "sha256",
    "tool_name",
    "deepagents_path",
    "remote_storage_uri",
}
_ASYNC_CONTEXT_PATH_FIELDS = {"path", "relative_path", "deepagents_path"}
_ASYNC_CONTEXT_VIRTUAL_PATH_PREFIXES = ("/outputs/", "/workspace/")
_ASYNC_CONTEXT_REMOTE_STORAGE_SCHEMES = {
    "abfs",
    "abfss",
    "az",
    "gs",
    "gcs",
    "http",
    "https",
    "s3",
}
_ASYNC_CONTEXT_REFERENCE_URI_SCHEMES = {
    *_ASYNC_CONTEXT_REMOTE_STORAGE_SCHEMES,
    "bisque",
    "dataset",
    "resource",
}
_ASYNC_CONTEXT_NESTED_FIELDS = (
    "knowledge_context",
    "selection_context",
    "workflow_hint",
    "benchmark",
    "response_contract",
    "budget",
    "sandbox_policy",
)
_ASYNC_CONTEXT_SECRET_KEY_MARKERS = (
    "authorization",
    "cookie",
    "password",
    "secret",
    "credential",
    "api_key",
    "apikey",
)
_ASYNC_CONTEXT_SECRET_KEY_NAMES = {
    "auth_token",
    "access_token",
    "refresh_token",
    "id_token",
    "session_token",
    "token",
}
_ASYNC_CONTEXT_PUBLIC_TOKEN_COUNT_KEYS = {
    "max_tokens",
    "min_tokens",
    "token_budget",
    "token_limit",
    "token_count",
}
_DROP_ASYNC_CONTEXT_VALUE = object()


class UltraAsyncSubagentContextMiddleware(AgentMiddleware[Any, AgentRunContext, Any]):
    """Add Ultra run scope to remote Deep Agents async subagent launches."""

    tools = ()

    def __init__(self, async_subagents: Sequence[dict[str, Any]]) -> None:
        agent_map: dict[str, dict[str, Any]] = {}
        seen_names: set[str] = set()
        for spec in async_subagents:
            if not isinstance(spec, dict):
                raise ValueError("Async subagent specs must be objects")
            name = _required_async_subagent_string(
                spec.get("name"),
                field="name",
                name="(unknown)",
            )
            normalized_name = name.lower()
            if normalized_name in seen_names:
                raise ValueError(f"Duplicate async subagent name: {name}")
            seen_names.add(normalized_name)
            agent_map[name] = _normalized_async_subagent_spec(spec, name=name)
        self._agent_map = agent_map
        self._clients = _AsyncSubagentClientCache(self._agent_map)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        tool_name = _tool_call_name(request)
        if tool_name in _ASYNC_TASK_TOOLS_REQUIRING_TRACKED_SUBAGENT:
            failed_launch = _failed_launch_task_message(request)
            if failed_launch:
                return _tool_message(failed_launch, request, status="error")
            tracked = self._resolve_configured_tracked_task(request)
            if isinstance(tracked, str):
                return _tool_message(tracked, request, status="error")
            if tool_name in {"cancel_async_task", "check_async_task", "update_async_task"}:
                cached_terminal = _cached_terminal_task_check_message(tracked)
                if cached_terminal:
                    return _tool_message(
                        cached_terminal,
                        request,
                        status=(
                            "error"
                            if tracked["status"].lower()
                            in _FAILED_TERMINAL_ASYNC_TASK_STATUSES
                            else "success"
                        ),
                    )
            if tool_name in _ASYNC_TASK_TOOLS_REQUIRING_SYNC_URL:
                error = self._validate_sync_url(tracked["agent_name"])
                if error:
                    return _tool_message(error, request, status="error")
        if tool_name == _ASYNC_TASK_LIST_TOOL:
            cached_terminal_list = _cached_terminal_task_list_message(request)
            if cached_terminal_list:
                return _tool_message(
                    cached_terminal_list,
                    request,
                    status=_cached_terminal_task_list_status(request),
                )
            error = self._validate_listable_tracked_tasks(request, require_sync_url=True)
            if error:
                return _tool_message(error, request, status="error")
            return self._list_async_tasks(request)
        if tool_name not in _ASYNC_TASK_TOOLS_WITH_REMOTE_RUNS:
            result = handler(request)
            if tool_name == "check_async_task":
                return (
                    _check_failure_command_from_result(request, result)
                    or _check_error_detail_command_from_result(result)
                    or result
                )
            if tool_name == "cancel_async_task":
                return _cancel_failure_command_from_result(request, result) or result
            return result
        context = _runtime_context(request)
        if context is None:
            return _tool_message(_missing_context_error(tool_name), request, status="error")
        if tool_name == "start_async_task":
            return self._start_async_task(request, context)
        return self._update_async_task(request, context)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        tool_name = _tool_call_name(request)
        if tool_name in _ASYNC_TASK_TOOLS_REQUIRING_TRACKED_SUBAGENT:
            failed_launch = _failed_launch_task_message(request)
            if failed_launch:
                return _tool_message(failed_launch, request, status="error")
            tracked = self._resolve_configured_tracked_task(request)
            if isinstance(tracked, str):
                return _tool_message(tracked, request, status="error")
            if tool_name in {"cancel_async_task", "check_async_task", "update_async_task"}:
                cached_terminal = _cached_terminal_task_check_message(tracked)
                if cached_terminal:
                    return _tool_message(
                        cached_terminal,
                        request,
                        status=(
                            "error"
                            if tracked["status"].lower()
                            in _FAILED_TERMINAL_ASYNC_TASK_STATUSES
                            else "success"
                        ),
                    )
        if tool_name == _ASYNC_TASK_LIST_TOOL:
            cached_terminal_list = _cached_terminal_task_list_message(request)
            if cached_terminal_list:
                return _tool_message(
                    cached_terminal_list,
                    request,
                    status=_cached_terminal_task_list_status(request),
                )
            error = self._validate_listable_tracked_tasks(request, require_sync_url=False)
            if error:
                return _tool_message(error, request, status="error")
            return await self._alist_async_tasks(request)
        if tool_name not in _ASYNC_TASK_TOOLS_WITH_REMOTE_RUNS:
            result = await handler(request)
            if tool_name == "check_async_task":
                return (
                    _check_failure_command_from_result(request, result)
                    or _check_error_detail_command_from_result(result)
                    or result
                )
            if tool_name == "cancel_async_task":
                return _cancel_failure_command_from_result(request, result) or result
            return result
        context = _runtime_context(request)
        if context is None:
            return _tool_message(_missing_context_error(tool_name), request, status="error")
        if tool_name == "start_async_task":
            return await self._astart_async_task(request, context)
        return await self._aupdate_async_task(request, context)

    def _start_async_task(
        self,
        request: ToolCallRequest,
        context: AgentRunContext,
    ) -> ToolMessage | Command[Any]:
        args = _tool_call_args(request)
        subagent_type = str(args.get("subagent_type") or "").strip()
        description, description_error = _required_tool_arg_text(
            args,
            field="description",
            tool_name="start_async_task",
        )
        error = self._validate_subagent_type(subagent_type)
        if error:
            return _tool_message(error, request, status="error")
        if description_error:
            return _tool_message(description_error, request, status="error")
        spec = self._agent_map[subagent_type]
        task_id = ""
        try:
            client = self._clients.get_sync(subagent_type)
            thread = client.threads.create(
                metadata=async_subagent_thread_metadata(
                    context,
                    subagent_name=subagent_type,
                    graph_id=str(spec["graph_id"]),
                ),
                graph_id=str(spec["graph_id"]),
            )
            task_id = _thread_id_from_response(thread)
            run = client.runs.create(
                thread_id=task_id,
                assistant_id=str(spec["graph_id"]),
                input={"messages": [{"role": "user", "content": description}]},
                context=async_subagent_context_payload(context, subagent_name=subagent_type),
                metadata=async_subagent_run_metadata(
                    context,
                    subagent_name=subagent_type,
                    graph_id=str(spec["graph_id"]),
                ),
                durability="async",
            )
            run_id = _run_id_from_response(run)
        except Exception as exc:  # noqa: BLE001 - LangGraph SDK raises untyped errors
            logger.warning("Failed to launch async subagent '%s': %s", subagent_type, exc)
            if task_id:
                return _async_task_error_command(
                    request,
                    message=f"Failed to launch async subagent '{subagent_type}': {exc}",
                    task_id=task_id,
                    agent_name=subagent_type,
                    error=str(exc),
                )
            return _tool_message(
                f"Failed to launch async subagent '{subagent_type}': {exc}",
                request,
                status="error",
            )
        return _async_task_command(
            request,
            message=f"Launched async subagent. task_id: {task_id}",
            task_id=task_id,
            agent_name=subagent_type,
            run_id=run_id,
        )

    async def _astart_async_task(
        self,
        request: ToolCallRequest,
        context: AgentRunContext,
    ) -> ToolMessage | Command[Any]:
        args = _tool_call_args(request)
        subagent_type = str(args.get("subagent_type") or "").strip()
        description, description_error = _required_tool_arg_text(
            args,
            field="description",
            tool_name="start_async_task",
        )
        error = self._validate_subagent_type(subagent_type)
        if error:
            return _tool_message(error, request, status="error")
        if description_error:
            return _tool_message(description_error, request, status="error")
        spec = self._agent_map[subagent_type]
        task_id = ""
        try:
            client = self._clients.get_async(subagent_type)
            thread = await client.threads.create(
                metadata=async_subagent_thread_metadata(
                    context,
                    subagent_name=subagent_type,
                    graph_id=str(spec["graph_id"]),
                ),
                graph_id=str(spec["graph_id"]),
            )
            task_id = _thread_id_from_response(thread)
            run = await client.runs.create(
                thread_id=task_id,
                assistant_id=str(spec["graph_id"]),
                input={"messages": [{"role": "user", "content": description}]},
                context=async_subagent_context_payload(context, subagent_name=subagent_type),
                metadata=async_subagent_run_metadata(
                    context,
                    subagent_name=subagent_type,
                    graph_id=str(spec["graph_id"]),
                ),
                durability="async",
            )
            run_id = _run_id_from_response(run)
        except Exception as exc:  # noqa: BLE001 - LangGraph SDK raises untyped errors
            logger.warning("Failed to launch async subagent '%s': %s", subagent_type, exc)
            if task_id:
                return _async_task_error_command(
                    request,
                    message=f"Failed to launch async subagent '{subagent_type}': {exc}",
                    task_id=task_id,
                    agent_name=subagent_type,
                    error=str(exc),
                )
            return _tool_message(
                f"Failed to launch async subagent '{subagent_type}': {exc}",
                request,
                status="error",
            )
        return _async_task_command(
            request,
            message=f"Launched async subagent. task_id: {task_id}",
            task_id=task_id,
            agent_name=subagent_type,
            run_id=run_id,
        )

    def _update_async_task(
        self,
        request: ToolCallRequest,
        context: AgentRunContext,
    ) -> ToolMessage | Command[Any]:
        tracked = self._resolve_configured_tracked_task(request)
        if isinstance(tracked, str):
            return _tool_message(tracked, request, status="error")
        spec = self._agent_map[tracked["agent_name"]]
        message, message_error = _required_tool_arg_text(
            _tool_call_args(request),
            field="message",
            tool_name="update_async_task",
        )
        if message_error:
            return _tool_message(message_error, request, status="error")
        try:
            client = self._clients.get_sync(tracked["agent_name"])
            run = client.runs.create(
                thread_id=tracked["thread_id"],
                assistant_id=str(spec["graph_id"]),
                input={"messages": [{"role": "user", "content": message}]},
                context=async_subagent_context_payload(
                    context,
                    subagent_name=tracked["agent_name"],
                ),
                metadata=async_subagent_run_metadata(
                    context,
                    subagent_name=tracked["agent_name"],
                    graph_id=str(spec["graph_id"]),
                ),
                durability="async",
                multitask_strategy="interrupt",
            )
            run_id = _run_id_from_response(run)
        except Exception as exc:  # noqa: BLE001 - LangGraph SDK raises untyped errors
            logger.warning("Failed to update async subagent '%s': %s", tracked["agent_name"], exc)
            return _async_task_update_error_command(
                request,
                tracked=tracked,
                error=str(exc),
            )
        return _async_task_command(
            request,
            message=f"Updated async subagent. task_id: {tracked['task_id']}",
            task_id=tracked["task_id"],
            agent_name=tracked["agent_name"],
            thread_id=tracked["thread_id"],
            run_id=run_id,
            created_at=tracked["created_at"],
            last_checked_at=tracked["last_checked_at"],
        )

    async def _aupdate_async_task(
        self,
        request: ToolCallRequest,
        context: AgentRunContext,
    ) -> ToolMessage | Command[Any]:
        tracked = self._resolve_configured_tracked_task(request)
        if isinstance(tracked, str):
            return _tool_message(tracked, request, status="error")
        spec = self._agent_map[tracked["agent_name"]]
        message, message_error = _required_tool_arg_text(
            _tool_call_args(request),
            field="message",
            tool_name="update_async_task",
        )
        if message_error:
            return _tool_message(message_error, request, status="error")
        try:
            client = self._clients.get_async(tracked["agent_name"])
            run = await client.runs.create(
                thread_id=tracked["thread_id"],
                assistant_id=str(spec["graph_id"]),
                input={"messages": [{"role": "user", "content": message}]},
                context=async_subagent_context_payload(
                    context,
                    subagent_name=tracked["agent_name"],
                ),
                metadata=async_subagent_run_metadata(
                    context,
                    subagent_name=tracked["agent_name"],
                    graph_id=str(spec["graph_id"]),
                ),
                durability="async",
                multitask_strategy="interrupt",
            )
            run_id = _run_id_from_response(run)
        except Exception as exc:  # noqa: BLE001 - LangGraph SDK raises untyped errors
            logger.warning("Failed to update async subagent '%s': %s", tracked["agent_name"], exc)
            return _async_task_update_error_command(
                request,
                tracked=tracked,
                error=str(exc),
            )
        return _async_task_command(
            request,
            message=f"Updated async subagent. task_id: {tracked['task_id']}",
            task_id=tracked["task_id"],
            agent_name=tracked["agent_name"],
            thread_id=tracked["thread_id"],
            run_id=run_id,
            created_at=tracked["created_at"],
            last_checked_at=tracked["last_checked_at"],
        )

    def _list_async_tasks(self, request: ToolCallRequest) -> Command[Any] | ToolMessage:
        selected = _selected_tracked_tasks(request)
        if not selected:
            return _tool_message("No async subagent tasks tracked.", request)
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        entries: list[str] = []
        updates: dict[str, dict[str, str]] = {}
        for tracked in selected:
            if tracked["status"].lower() in _TERMINAL_ASYNC_TASK_STATUSES:
                task = _async_task_list_entry(tracked, status=tracked["status"])
            else:
                try:
                    client = self._clients.get_sync(tracked["agent_name"])
                    run = client.runs.get(
                        thread_id=tracked["thread_id"],
                        run_id=tracked["run_id"],
                    )
                    task = _async_task_list_entry(
                        tracked,
                        status=_run_status_from_response(run, fallback=tracked["status"]),
                        checked_at=now,
                        error=_run_error_from_response(run),
                    )
                except Exception as exc:  # noqa: BLE001 - LangGraph SDK raises untyped errors
                    logger.warning(
                        "Failed to fetch live async subagent status for task %s: %s",
                        tracked["task_id"],
                        exc,
                    )
                    task = _async_task_list_entry(
                        tracked,
                        status=tracked["status"],
                        updated_at=now,
                        error=str(exc),
                    )
            updates[task["task_id"]] = task
            entries.append(_format_async_task_list_entry(task))
        return Command(
            update={
                "messages": [
                    _tool_message(
                        f"{len(entries)} tracked task(s):\n" + "\n".join(entries),
                        request,
                        status=(
                            "error"
                            if any(_async_task_list_entry_is_error(task) for task in updates.values())
                            else "success"
                        ),
                    )
                ],
                "async_tasks": updates,
            }
        )

    async def _alist_async_tasks(self, request: ToolCallRequest) -> Command[Any] | ToolMessage:
        selected = _selected_tracked_tasks(request)
        if not selected:
            return _tool_message("No async subagent tasks tracked.", request)
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        entries: list[str] = []
        updates: dict[str, dict[str, str]] = {}
        for tracked in selected:
            if tracked["status"].lower() in _TERMINAL_ASYNC_TASK_STATUSES:
                task = _async_task_list_entry(tracked, status=tracked["status"])
            else:
                try:
                    client = self._clients.get_async(tracked["agent_name"])
                    run = await client.runs.get(
                        thread_id=tracked["thread_id"],
                        run_id=tracked["run_id"],
                    )
                    task = _async_task_list_entry(
                        tracked,
                        status=_run_status_from_response(run, fallback=tracked["status"]),
                        checked_at=now,
                        error=_run_error_from_response(run),
                    )
                except Exception as exc:  # noqa: BLE001 - LangGraph SDK raises untyped errors
                    logger.warning(
                        "Failed to fetch live async subagent status for task %s: %s",
                        tracked["task_id"],
                        exc,
                    )
                    task = _async_task_list_entry(
                        tracked,
                        status=tracked["status"],
                        updated_at=now,
                        error=str(exc),
                    )
            updates[task["task_id"]] = task
            entries.append(_format_async_task_list_entry(task))
        return Command(
            update={
                "messages": [
                    _tool_message(
                        f"{len(entries)} tracked task(s):\n" + "\n".join(entries),
                        request,
                        status=(
                            "error"
                            if any(_async_task_list_entry_is_error(task) for task in updates.values())
                            else "success"
                        ),
                    )
                ],
                "async_tasks": updates,
            }
        )

    def _validate_subagent_type(self, subagent_type: str) -> str:
        if subagent_type in self._agent_map:
            return ""
        if not self._agent_map:
            return "No async subagents are configured."
        available = ", ".join(sorted(self._agent_map))
        return (
            f"Unknown async subagent type '{subagent_type}'. Available async subagents: {available}"
        )

    def _resolve_configured_tracked_task(
        self,
        request: ToolCallRequest,
    ) -> dict[str, str] | str:
        tracked = _resolve_tracked_task(
            str(_tool_call_args(request).get("task_id") or ""),
            request,
        )
        if isinstance(tracked, str):
            return tracked
        error = self._validate_subagent_type(tracked["agent_name"])
        if error:
            return error
        return tracked

    def _validate_sync_url(self, subagent_type: str) -> str:
        spec = self._agent_map.get(subagent_type)
        if spec is None or spec.get("url") is not None:
            return ""
        return (
            f"Async subagent '{subagent_type}' has no url configured. "
            "ASGI transport (url=None) requires async invocation."
        )

    def _validate_listable_tracked_tasks(
        self,
        request: ToolCallRequest,
        *,
        require_sync_url: bool,
    ) -> str:
        tasks = request.state.get("async_tasks") or {}
        if not tasks:
            return ""
        if not isinstance(tasks, dict):
            return "Async subagent task is missing required state: list_async_tasks"
        status_filter = str(_tool_call_args(request).get("status_filter") or "all").strip()
        for key, task in tasks.items():
            if not isinstance(task, dict):
                return f"Async subagent task is missing required state: {key or '(unknown)'}"
            task_id = str(task.get("task_id") or key or "").strip()
            status = str(task.get("status") or "").strip()
            if status_filter and status_filter != "all" and status != status_filter:
                continue
            if not all(str(task.get(field) or "").strip() for field in _ASYNC_TASK_REQUIRED_FIELDS):
                return f"Async subagent task is missing required state: {task_id or key or '(unknown)'}"
            agent_name = str(task.get("agent_name") or "").strip()
            if status.lower() in _TERMINAL_ASYNC_TASK_STATUSES:
                continue
            error = self._validate_subagent_type(agent_name)
            if error:
                return error
            if require_sync_url:
                error = self._validate_sync_url(agent_name)
                if error:
                    return error
        return ""


class _AsyncSubagentClientCache:
    def __init__(self, agents: dict[str, dict[str, Any]]) -> None:
        self._agents = agents
        self._sync: dict[tuple[str | None, frozenset[tuple[str, str]]], SyncLangGraphClient] = {}
        self._async: dict[tuple[str | None, frozenset[tuple[str, str]]], LangGraphClient] = {}

    def get_sync(self, name: str) -> SyncLangGraphClient:
        spec = self._agents[name]
        if spec.get("url") is None:
            raise ValueError(
                f"Async subagent '{name}' has no url configured. "
                "ASGI transport (url=None) requires async invocation."
            )
        key = _client_cache_key(spec)
        if key not in self._sync:
            self._sync[key] = get_sync_client(
                url=_optional_string(spec.get("url")),
                headers=_resolve_headers(spec),
            )
        return self._sync[key]

    def get_async(self, name: str) -> LangGraphClient:
        spec = self._agents[name]
        key = _client_cache_key(spec)
        if key not in self._async:
            self._async[key] = get_client(
                url=_optional_string(spec.get("url")),
                headers=_resolve_headers(spec),
            )
        return self._async[key]


def _normalized_async_subagent_spec(spec: dict[str, Any], *, name: str) -> dict[str, Any]:
    normalized = dict(spec)
    normalized["graph_id"] = _required_async_subagent_string(
        normalized.get("graph_id"),
        field="graph_id",
        name=name,
    )
    if normalized.get("url") is not None:
        normalized["url"] = _validate_async_subagent_url(normalized.get("url"), name=name)
    if "headers" in normalized:
        headers = _validate_async_subagent_headers(normalized.get("headers"), name=name)
        if headers:
            normalized["headers"] = headers
        else:
            normalized.pop("headers", None)
    return normalized


def _required_async_subagent_string(value: Any, *, field: str, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Async subagent '{name}' {field} must be a non-empty string")
    value = value.strip()
    if not value:
        raise ValueError(f"Async subagent '{name}' {field} is required")
    return value


def _validate_async_subagent_headers(value: Any, *, name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Async subagent '{name}' headers must be an object when provided")
    headers: dict[str, str] = {}
    seen_header_names: set[str] = set()
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            raise ValueError(f"Async subagent '{name}' headers contains a non-string header name")
        key = raw_key.strip()
        if not key:
            raise ValueError(f"Async subagent '{name}' headers contains an empty header name")
        normalized_key = key.lower()
        if normalized_key in seen_header_names:
            raise ValueError(f"Duplicate async subagent header name: {key}")
        seen_header_names.add(normalized_key)
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(f"Async subagent '{name}' headers.{key} must be a non-empty string")
        headers[key] = raw_value.strip()
    return headers


def _validate_async_subagent_url(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Async subagent '{name}' url must be a non-empty string when provided")
    url = value.strip()
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            f"Async subagent '{name}' url must be an http:// or https:// endpoint when provided"
        )
    if parsed.username or parsed.password:
        raise ValueError(f"Async subagent '{name}' url must not include credentials")
    if is_local_http_host(parsed.hostname) and not allow_private_async_subagent_url():
        raise ValueError(
            f"Async subagent '{name}' url must not target a localhost/private/link-local "
            "host (the run context + task egress to it); set "
            "ULTRA_DEEPAGENTS_ALLOW_PRIVATE_ASYNC_SUBAGENT_URL=1 for local dev"
        )
    return url


def _resolve_headers(spec: dict[str, Any]) -> dict[str, str]:
    raw_headers = spec.get("headers")
    headers = dict(raw_headers) if isinstance(raw_headers, dict) else {}
    header_names = {str(key).lower() for key in headers}
    if "x-auth-scheme" not in header_names:
        headers["x-auth-scheme"] = "langsmith"
    return {str(key): str(value) for key, value in headers.items()}


def _client_cache_key(spec: dict[str, Any]) -> tuple[str | None, frozenset[tuple[str, str]]]:
    return (_optional_string(spec.get("url")), frozenset(_resolve_headers(spec).items()))


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _run_id_from_response(run: Any) -> str:
    value = run.get("run_id") if isinstance(run, dict) else getattr(run, "run_id", None)
    run_id = str(value or "").strip()
    if not run_id:
        raise ValueError("Agent Protocol run response is missing run_id")
    return run_id


def _thread_id_from_response(thread: Any) -> str:
    value = (
        thread.get("thread_id") if isinstance(thread, dict) else getattr(thread, "thread_id", None)
    )
    thread_id = str(value or "").strip()
    if not thread_id:
        raise ValueError("Agent Protocol thread response is missing thread_id")
    return thread_id


def _run_status_from_response(run: Any, *, fallback: str) -> str:
    value = run.get("status") if isinstance(run, dict) else getattr(run, "status", None)
    status = str(value or "").strip()
    return status or fallback


def _run_error_from_response(run: Any) -> str:
    value = run.get("error") if isinstance(run, dict) else getattr(run, "error", None)
    return str(value or "").strip()


def _resolve_tracked_task(task_id: str, request: ToolCallRequest) -> dict[str, str] | str:
    task_id = task_id.strip()
    if not task_id:
        return "task_id is required."
    tasks = request.state.get("async_tasks") or {}
    if not isinstance(tasks, dict):
        return "No async subagent tasks tracked."
    task = tasks.get(task_id)
    if not isinstance(task, dict):
        return f"Unknown async subagent task: {task_id}"
    resolved = {key: str(task.get(key) or "") for key in _ASYNC_TASK_REQUIRED_FIELDS}
    if not all(resolved.values()):
        return f"Async subagent task is missing required state: {task_id}"
    last_error = str(task.get("last_error") or "").strip()
    if last_error:
        resolved["last_error"] = last_error
    return resolved


def _failed_launch_task_message(request: ToolCallRequest) -> str:
    task_id = str(_tool_call_args(request).get("task_id") or "").strip()
    if not task_id:
        return ""
    tasks = request.state.get("async_tasks") or {}
    if not isinstance(tasks, dict):
        return ""
    task = tasks.get(task_id)
    if not isinstance(task, dict):
        return ""
    if str(task.get("run_id") or "") != _FAILED_ASYNC_RUN_ID:
        return ""
    if str(task.get("status") or "").lower() != "error":
        return ""
    agent_name = str(task.get("agent_name") or "unknown").strip()
    error = str(task.get("last_error") or "remote run creation failed").strip()
    return (
        f"Async subagent task {task_id} ({agent_name}) failed before a remote run "
        f"was created: {error}"
    )


def _cached_terminal_task_list_message(request: ToolCallRequest) -> str:
    tasks = request.state.get("async_tasks") or {}
    if not isinstance(tasks, dict) or not tasks:
        return ""
    status_filter = str(_tool_call_args(request).get("status_filter") or "all").strip()
    selected: list[dict[str, Any]] = []
    for key, task in tasks.items():
        if not isinstance(task, dict):
            return ""
        status = str(task.get("status") or "").strip()
        if status_filter and status_filter != "all" and status != status_filter:
            continue
        if status.lower() not in _TERMINAL_ASYNC_TASK_STATUSES:
            return ""
        if not all(str(task.get(field) or "").strip() for field in _ASYNC_TASK_REQUIRED_FIELDS):
            return ""
        selected.append(task)
    if not selected:
        return ""
    entries = []
    for task in selected:
        task_id = str(task.get("task_id") or "").strip()
        agent_name = str(task.get("agent_name") or "").strip()
        status = str(task.get("status") or "").strip()
        entry = f"- task_id: {task_id}  agent: {agent_name}  status: {status}"
        error = str(task.get("last_error") or "").strip()
        if error:
            entry += f"  error: {error}"
        entries.append(entry)
    return f"{len(entries)} tracked task(s):\n" + "\n".join(entries)


def _cached_terminal_task_list_status(
    request: ToolCallRequest,
) -> Literal["success", "error"]:
    tasks = request.state.get("async_tasks") or {}
    if not isinstance(tasks, dict):
        return "success"
    status_filter = str(_tool_call_args(request).get("status_filter") or "all").strip()
    for task in tasks.values():
        if not isinstance(task, dict):
            continue
        status = str(task.get("status") or "").strip()
        if status_filter and status_filter != "all" and status != status_filter:
            continue
        if _async_task_list_entry_is_error(task):
            return "error"
    return "success"


def _cached_terminal_task_check_message(tracked: dict[str, str]) -> str:
    if tracked["status"].lower() not in _TERMINAL_ASYNC_TASK_STATUSES:
        return ""
    task = _async_task_list_entry(tracked, status=tracked["status"])
    return "Cached async subagent task status:\n" + _format_async_task_list_entry(task)


def _selected_tracked_tasks(request: ToolCallRequest) -> list[dict[str, str]]:
    tasks = request.state.get("async_tasks") or {}
    if not isinstance(tasks, dict):
        return []
    status_filter = str(_tool_call_args(request).get("status_filter") or "all").strip()
    selected: list[dict[str, str]] = []
    for key, task in tasks.items():
        if not isinstance(task, dict):
            continue
        status = str(task.get("status") or "").strip()
        if status_filter and status_filter != "all" and status != status_filter:
            continue
        tracked = {
            field: str(task.get(field) or "").strip() for field in _ASYNC_TASK_REQUIRED_FIELDS
        }
        if not all(tracked.values()):
            continue
        last_error = str(task.get("last_error") or "").strip()
        if last_error:
            tracked["last_error"] = last_error
        selected.append(tracked)
    return selected


def _async_task_list_entry(
    tracked: dict[str, str],
    *,
    status: str,
    checked_at: str | None = None,
    updated_at: str | None = None,
    error: str = "",
) -> dict[str, str]:
    normalized_status = str(status or tracked["status"]).strip()
    task = {
        "task_id": tracked["task_id"],
        "agent_name": tracked["agent_name"],
        "thread_id": tracked["thread_id"],
        "run_id": tracked["run_id"],
        "status": normalized_status,
        "created_at": tracked["created_at"],
        "last_checked_at": checked_at or tracked["last_checked_at"],
        "last_updated_at": (
            updated_at
            or (
                checked_at if normalized_status != tracked["status"] else tracked["last_updated_at"]
            )
            or tracked["last_updated_at"]
        ),
    }
    error_text = str(error or "").strip()
    if error_text:
        task["last_error"] = error_text
    elif normalized_status.lower() in _FAILED_TERMINAL_ASYNC_TASK_STATUSES:
        tracked_error = str(tracked.get("last_error") or "").strip()
        if tracked_error:
            task["last_error"] = tracked_error
    return task


def _format_async_task_list_entry(task: dict[str, str]) -> str:
    entry = f"- task_id: {task['task_id']}  agent: {task['agent_name']}  status: {task['status']}"
    error = str(task.get("last_error") or "").strip()
    if error:
        entry += f"  error: {error}"
    return entry


def _async_task_list_entry_is_error(task: dict[str, str]) -> bool:
    return bool(task.get("last_error")) or task.get("status", "").lower() in (
        _FAILED_TERMINAL_ASYNC_TASK_STATUSES
    )


def _cancel_failure_command_from_result(
    request: ToolCallRequest,
    result: ToolMessage | Command[Any],
) -> Command[Any] | None:
    if not isinstance(result, ToolMessage):
        return None
    error = _cancel_failure_text(str(result.content or ""))
    if not error:
        return None
    tracked = _resolve_tracked_task(str(_tool_call_args(request).get("task_id") or ""), request)
    if isinstance(tracked, str):
        return None
    return _async_task_cancel_error_command(request, tracked=tracked, error=error)


def _check_failure_command_from_result(
    request: ToolCallRequest,
    result: ToolMessage | Command[Any],
) -> Command[Any] | None:
    if not isinstance(result, ToolMessage):
        return None
    error = _check_failure_text(str(result.content or ""))
    if not error:
        return None
    tracked = _resolve_tracked_task(str(_tool_call_args(request).get("task_id") or ""), request)
    if isinstance(tracked, str):
        return None
    return _async_task_check_error_command(request, tracked=tracked, error=error)


def _check_error_detail_command_from_result(
    result: ToolMessage | Command[Any],
) -> Command[Any] | None:
    if not isinstance(result, Command):
        return None
    update = result.update
    if not isinstance(update, dict):
        return None
    messages = update.get("messages")
    async_tasks = update.get("async_tasks")
    if not isinstance(messages, list) or not isinstance(async_tasks, dict):
        return None
    error_text = _check_result_error_text(messages)
    if not error_text:
        return None
    enriched_tasks: dict[str, Any] = {}
    changed = False
    for task_id, task in async_tasks.items():
        if not isinstance(task, dict):
            enriched_tasks[task_id] = task
            continue
        next_task = dict(task)
        if str(next_task.get("status") or "").strip().lower() == "error":
            next_task["last_error"] = error_text
            changed = True
        enriched_tasks[task_id] = next_task
    if not changed:
        return None
    enriched_update = dict(update)
    enriched_update["async_tasks"] = enriched_tasks
    enriched_update["messages"] = [
        message.model_copy(update={"status": "error"})
        if isinstance(message, ToolMessage)
        else message
        for message in messages
    ]
    return Command(update=enriched_update)


def _check_result_error_text(messages: list[Any]) -> str:
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        content = str(message.content or "").strip()
        if not content:
            continue
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if str(parsed.get("status") or "").strip().lower() != "error":
            continue
        error = str(parsed.get("error") or "").strip()
        if error:
            return error
    return ""


def _cancel_failure_text(text: str) -> str:
    stripped = " ".join(str(text or "").strip().split())
    prefix = "Failed to cancel run:"
    if stripped.lower().startswith(prefix.lower()):
        return stripped[len(prefix) :].strip() or stripped
    return ""


def _check_failure_text(text: str) -> str:
    stripped = " ".join(str(text or "").strip().split())
    prefix = "Failed to get run status:"
    if stripped.lower().startswith(prefix.lower()):
        return stripped[len(prefix) :].strip() or stripped
    return ""


def async_subagent_context_payload(
    context: AgentRunContext,
    *,
    subagent_name: str,
) -> dict[str, Any]:
    payload = context.to_payload()
    for field in _ASYNC_CONTEXT_NESTED_FIELDS:
        value = payload.get(field)
        payload[field] = _sanitize_async_context_mapping(value if isinstance(value, dict) else {})
    # Notes authority is coordinator-only. Even sanitized note ids/revisions must
    # not become usable context for an async run with a different lease.
    selection_context = payload.get("selection_context")
    if isinstance(selection_context, dict):
        selection_context.pop("note_access", None)
    payload["selected_resource_uris"] = _sanitize_async_context_references(
        context.selected_resource_uris
    )
    payload["selected_dataset_uris"] = _sanitize_async_context_references(
        context.selected_dataset_uris
    )
    payload["auth_claims"] = {}
    payload["resource_descriptors"] = _sanitize_async_resource_descriptors(
        context.resource_descriptors
    )
    payload["run_metadata"] = {
        "delegation": {
            "mode": "async_subagent",
            "parent_run_id": context.run_id,
            "parent_thread_id": context.thread_id,
            "subagent_name": subagent_name,
        }
    }
    payload["workspace_root"] = "/workspace"
    payload["artifact_root"] = "/outputs"
    return payload


def _sanitize_async_context_references(values: Sequence[Any]) -> list[str]:
    sanitized: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and _is_async_context_safe_path(text):
            sanitized.append(text)
    return sanitized


def _sanitize_async_resource_descriptors(
    descriptors: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            continue
        if str(descriptor.get("type") or "").strip() == "selected_resource":
            selected = public_selected_resource_descriptor(descriptor)
            if selected:
                sanitized.append(selected)
            continue
        public: dict[str, Any] = {}
        for key in _ASYNC_CONTEXT_RESOURCE_DESCRIPTOR_FIELDS:
            if key not in descriptor:
                continue
            value = descriptor.get(key)
            if key in _ASYNC_CONTEXT_PATH_FIELDS and not _is_async_context_safe_path(value):
                continue
            if key == "remote_storage_uri" and not _is_async_context_safe_remote_storage_uri(value):
                continue
            public[key] = value
        remote_storage_uri = _safe_async_remote_storage_uri(descriptor.get("storage_uri"))
        if remote_storage_uri:
            public["remote_storage_uri"] = remote_storage_uri
        if public:
            sanitized.append(public)
    return sanitized


def _sanitize_async_context_mapping(value: dict[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize_async_context_value(value)
    return sanitized if isinstance(sanitized, dict) else {}


def _sanitize_async_context_value(value: Any, *, key: str = "") -> Any:
    if key and _is_async_context_secret_key(key):
        return _DROP_ASYNC_CONTEXT_VALUE
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for raw_key, raw_child in value.items():
            child_key = str(raw_key)
            child = _sanitize_async_context_value(raw_child, key=child_key)
            if child is not _DROP_ASYNC_CONTEXT_VALUE:
                sanitized[child_key] = child
        return sanitized
    if isinstance(value, list | tuple):
        sanitized_items = []
        for item in value:
            child = _sanitize_async_context_value(item)
            if child is not _DROP_ASYNC_CONTEXT_VALUE:
                sanitized_items.append(child)
        return sanitized_items
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        if not _is_async_context_safe_path(text):
            return _DROP_ASYNC_CONTEXT_VALUE
        return value
    if isinstance(value, int | float | bool) or value is None:
        return value
    return str(value)


def _is_async_context_secret_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    if normalized in _ASYNC_CONTEXT_PUBLIC_TOKEN_COUNT_KEYS:
        return False
    return (
        normalized in _ASYNC_CONTEXT_SECRET_KEY_NAMES
        or normalized.endswith("_token")
        or any(marker in normalized for marker in _ASYNC_CONTEXT_SECRET_KEY_MARKERS)
    )


def _is_async_context_safe_path(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if text.startswith(_ASYNC_CONTEXT_VIRTUAL_PATH_PREFIXES):
        return True
    if "\\" in text:
        return False
    if len(text) >= 2 and text[1] == ":" and text[0].isalpha():
        return False
    if text.startswith("/") or text.lower().startswith("file://"):
        return False
    if _is_async_context_url_like(text):
        return _is_async_context_safe_reference_uri(text)
    return ".." not in text.split("/")


def _is_async_context_url_like(text: str) -> bool:
    parsed = urlparse(text)
    return bool(parsed.scheme and ("://" in text or parsed.scheme.lower() == "file"))


def _is_async_context_safe_reference_uri(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    parsed = urlparse(text)
    scheme = parsed.scheme.lower()
    if scheme not in _ASYNC_CONTEXT_REFERENCE_URI_SCHEMES or not parsed.netloc:
        return False
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return False
    if "\\" in parsed.path or ".." in parsed.path.split("/"):
        return False
    if scheme in {"http", "https"} and _is_local_http_host(parsed.hostname):
        return False
    return True


def _safe_async_remote_storage_uri(value: Any) -> str:
    text = str(value or "").strip()
    if not _is_async_context_safe_remote_storage_uri(text):
        return ""
    return text


def _is_async_context_safe_remote_storage_uri(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    parsed = urlparse(text)
    scheme = parsed.scheme.lower()
    if scheme not in _ASYNC_CONTEXT_REMOTE_STORAGE_SCHEMES or not parsed.netloc:
        return False
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return False
    if ".." in parsed.path.split("/"):
        return False
    if scheme in {"http", "https"} and _is_local_http_host(parsed.hostname):
        return False
    return True


def async_subagent_run_metadata(
    context: AgentRunContext,
    *,
    subagent_name: str,
    graph_id: str | None = None,
) -> dict[str, str]:
    metadata = {
        "ultra_delegation": "async_subagent",
        "ultra_parent_run_id": context.run_id,
        "ultra_parent_thread_id": context.thread_id,
        "ultra_subagent_name": subagent_name,
        "ultra_org_id": context.org_id,
        "ultra_user_id": context.user_id,
        "ultra_project_id": context.project_id,
    }
    if graph_id is not None:
        metadata["ultra_subagent_graph_id"] = graph_id
    return metadata


def async_subagent_thread_metadata(
    context: AgentRunContext,
    *,
    subagent_name: str,
    graph_id: str,
) -> dict[str, str]:
    return async_subagent_run_metadata(
        context,
        subagent_name=subagent_name,
        graph_id=graph_id,
    )


def _async_task_command(
    request: ToolCallRequest,
    *,
    message: str,
    task_id: str,
    agent_name: str,
    run_id: str,
    thread_id: str | None = None,
    created_at: str | None = None,
    last_checked_at: str | None = None,
) -> Command[Any]:
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    created = created_at or now
    checked = last_checked_at or now
    task = {
        "task_id": task_id,
        "agent_name": agent_name,
        "thread_id": thread_id or task_id,
        "run_id": run_id,
        "status": "running",
        "created_at": created,
        "last_checked_at": checked,
        "last_updated_at": now,
    }
    return Command(
        update={
            "messages": [_tool_message(message, request)],
            "async_tasks": {task_id: task},
        }
    )


def _async_task_error_command(
    request: ToolCallRequest,
    *,
    message: str,
    task_id: str,
    agent_name: str,
    error: str,
) -> Command[Any]:
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    error_text = str(error or message).strip()
    task = {
        "task_id": task_id,
        "agent_name": agent_name,
        "thread_id": task_id,
        "run_id": _FAILED_ASYNC_RUN_ID,
        "status": "error",
        "created_at": now,
        "last_checked_at": now,
        "last_updated_at": now,
        "last_error": error_text,
    }
    visible_message = f"{message} task_id: {task_id} status: error error: {error_text}"
    return Command(
        update={
            "messages": [_tool_message(visible_message, request, status="error")],
            "async_tasks": {task_id: task},
        }
    )


def _async_task_update_error_command(
    request: ToolCallRequest,
    *,
    tracked: dict[str, str],
    error: str,
) -> Command[Any]:
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    error_text = str(error or "remote update failed").strip()
    task_id = tracked["task_id"]
    task = {
        "task_id": task_id,
        "agent_name": tracked["agent_name"],
        "thread_id": tracked["thread_id"],
        "run_id": tracked["run_id"],
        "status": tracked["status"],
        "created_at": tracked["created_at"],
        "last_checked_at": tracked["last_checked_at"],
        "last_updated_at": now,
        "last_error": error_text,
    }
    visible_message = (
        f"Failed to update async subagent: {error_text} "
        f"task_id: {task_id} status: {tracked['status']} error: {error_text}"
    )
    return Command(
        update={
            "messages": [_tool_message(visible_message, request, status="error")],
            "async_tasks": {task_id: task},
        }
    )


def _async_task_cancel_error_command(
    request: ToolCallRequest,
    *,
    tracked: dict[str, str],
    error: str,
) -> Command[Any]:
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    error_text = str(error or "remote cancel failed").strip()
    task_id = tracked["task_id"]
    task = {
        "task_id": task_id,
        "agent_name": tracked["agent_name"],
        "thread_id": tracked["thread_id"],
        "run_id": tracked["run_id"],
        "status": tracked["status"],
        "created_at": tracked["created_at"],
        "last_checked_at": tracked["last_checked_at"],
        "last_updated_at": now,
        "last_error": error_text,
    }
    visible_message = (
        f"Failed to cancel run: {error_text} "
        f"task_id: {task_id} status: {tracked['status']} error: {error_text}"
    )
    return Command(
        update={
            "messages": [_tool_message(visible_message, request, status="error")],
            "async_tasks": {task_id: task},
        }
    )


def _async_task_check_error_command(
    request: ToolCallRequest,
    *,
    tracked: dict[str, str],
    error: str,
) -> Command[Any]:
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    error_text = str(error or "remote status check failed").strip()
    task_id = tracked["task_id"]
    task = {
        "task_id": task_id,
        "agent_name": tracked["agent_name"],
        "thread_id": tracked["thread_id"],
        "run_id": tracked["run_id"],
        "status": tracked["status"],
        "created_at": tracked["created_at"],
        "last_checked_at": tracked["last_checked_at"],
        "last_updated_at": now,
        "last_error": error_text,
    }
    visible_message = (
        f"Failed to get run status: {error_text} "
        f"task_id: {task_id} status: {tracked['status']} error: {error_text}"
    )
    return Command(
        update={
            "messages": [_tool_message(visible_message, request, status="error")],
            "async_tasks": {task_id: task},
        }
    )


def _tool_call_name(request: ToolCallRequest) -> str:
    return str(request.tool_call.get("name") or "")


def _tool_call_args(request: ToolCallRequest) -> dict[str, Any]:
    args = request.tool_call.get("args")
    return args if isinstance(args, dict) else {}


def _required_tool_arg_text(
    args: dict[str, Any],
    *,
    field: str,
    tool_name: str,
) -> tuple[str, str | None]:
    value = str(args.get(field) or "").strip()
    if value:
        return value, None
    return "", f"{tool_name} {field} is required for async subagent delegation."


def _runtime_context(request: ToolCallRequest) -> AgentRunContext | None:
    context = getattr(request.runtime, "context", None)
    return context if isinstance(context, AgentRunContext) else None


def _missing_context_error(tool_name: str) -> str:
    return (
        "AgentRunContext is required for "
        f"{tool_name} so Ultra can propagate scoped tenant, artifact, "
        "workspace, and authorization context to async subagents."
    )


def _tool_message(
    content: str,
    request: ToolCallRequest,
    *,
    status: Literal["success", "error"] = "success",
) -> ToolMessage:
    return ToolMessage(
        content,
        tool_call_id=str(request.tool_call.get("id") or ""),
        status=status,
    )
