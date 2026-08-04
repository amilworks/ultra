"""Todo state-echo + staleness nudge for the coordinator's system prompt.

Live finding (2026-08-04, planets run): the model plans well with ``write_todos``
but never maintains the list — exactly 2 rewrites in an 81-tool-call run: the
initial plan, then one batch mark-all-completed right before the final answer.
The tool description already demands real-time updates; prose alone loses to
momentum once the model is heads-down in tool loops.

The missing mechanics (both present in Claude Code's harness, both absent
here) are re-showing the CURRENT list to the model and nudging when it goes
stale. This middleware supplies them the same way ``UltraAttemptLedgerMiddleware``
supplies failure memory: a per-request system-prompt append derived from durable
state, never from message history — so the plan survives SummarizationMiddleware
compaction BY CONSTRUCTION (compaction rewrites ``messages``; ``todos`` state and
the per-request prompt are untouched). Injecting into ``messages`` instead would
be self-defeating: the next compaction would erase the plan exactly when a
long run needs it most.

Advisory only, never a gate: nothing blocks a model that ignores the nudge —
the progress-stall and completion guards keep their own jurisdiction, and
``write_todos`` deliberately stays out of ``PROGRESS_TOOL_NAMES`` (updating a
plan is bookkeeping, not progress). Runs that never call ``write_todos`` pay
nothing.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware import ModelRequest
from langchain.agents.middleware.types import AgentMiddleware, ModelResponse
from langchain_core.messages import AIMessage, ToolMessage

DEFAULT_STALE_AFTER_TOOL_RESULTS = 12

_ECHO_HEADER = "## Current todo list (write_todos state)"


def _tool_results_since_last_write_todos(messages: Sequence[Any]) -> int:
    """Count ToolMessages after the most recent AIMessage that called write_todos.

    Counts backwards so the cost is proportional to the stale window, not the
    history length. When no write_todos AIMessage survives in history (the plan
    predates the last compaction), every visible ToolMessage counts — the echo
    is then the plan's only remaining trace, which is precisely the case the
    echo exists for.
    """
    count = 0
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            tool_calls = getattr(message, "tool_calls", None) or []
            if any(call.get("name") == "write_todos" for call in tool_calls):
                return count
        elif isinstance(message, ToolMessage):
            count += 1
    return count


def build_todo_reminder_fragment(
    todos: Sequence[Mapping[str, Any]] | None,
    messages: Sequence[Any] | None,
    *,
    stale_after_tool_results: int = DEFAULT_STALE_AFTER_TOOL_RESULTS,
) -> str:
    """Render the state echo (+ staleness nudge) or "" when there is nothing to say.

    Empty list -> "" (no plan, no overhead). All-completed -> "" (a closed list
    steers nothing). Otherwise the current list is always echoed, and the nudge
    line is added once the stale window reaches ``stale_after_tool_results``.
    """
    if not todos:
        return ""
    statuses = [str(todo.get("status", "")) for todo in todos]
    if all(status == "completed" for status in statuses):
        return ""

    lines = [_ECHO_HEADER]
    for todo, status in zip(todos, statuses):
        content = str(todo.get("content", "")).strip()
        lines.append(f"- [{status or 'pending'}] {content}")

    stale_count = _tool_results_since_last_write_todos(messages or [])
    if stale_count >= max(1, stale_after_tool_results):
        lines.append(
            f"\n{stale_count} tool results have landed since this list was last "
            "updated — statuses above may be stale. If reality has moved, call "
            "write_todos now: mark finished items completed, keep only the step "
            "you are actually on as in_progress, and add or drop items the work "
            "has revealed. If the list is still accurate, continue."
        )
    return "\n".join(lines)


class UltraTodoReminderMiddleware(AgentMiddleware[Any, Any, Any]):
    """Append the current todo list (and staleness nudge) to each model request."""

    def __init__(
        self, *, stale_after_tool_results: int = DEFAULT_STALE_AFTER_TOOL_RESULTS
    ) -> None:
        super().__init__()
        self._stale_after_tool_results = stale_after_tool_results

    def _fragment(self, request: ModelRequest) -> str:
        state = getattr(request, "state", None) or {}
        return build_todo_reminder_fragment(
            state.get("todos"),
            state.get("messages"),
            stale_after_tool_results=self._stale_after_tool_results,
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        fragment = self._fragment(request)
        if not fragment:
            return handler(request)
        return handler(
            request.override(
                system_message=append_to_system_message(request.system_message, fragment)
            )
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        fragment = self._fragment(request)
        if not fragment:
            return await handler(request)
        return await handler(
            request.override(
                system_message=append_to_system_message(request.system_message, fragment)
            )
        )
