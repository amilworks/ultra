"""Todo state-echo + staleness nudge (todo_reminders.py).

The live gap this closes (planets run, 2026-08-04): the model plans well with
write_todos but never maintains the list — the current list must be re-shown
from durable state each model call, with a nudge once the status ages past a
tool-result window. Advisory only; empty/closed lists must cost nothing.
"""

from __future__ import annotations

from typing import Any, cast

from langchain.agents.middleware import ModelRequest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ultra_deepagents.todo_reminders import (
    UltraTodoReminderMiddleware,
    build_todo_reminder_fragment,
)

_TODOS = [
    {"content": "Inspect current source", "status": "completed"},
    {"content": "Add planets to scene", "status": "in_progress"},
    {"content": "Verify in headless Chromium", "status": "pending"},
]


def _write_todos_turn() -> list[Any]:
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": "write_todos", "args": {"todos": []}, "id": "call-1"}],
        ),
        ToolMessage(content="Updated todo list to [...]", tool_call_id="call-1"),
    ]


def _tool_turns(count: int) -> list[Any]:
    turns: list[Any] = []
    for index in range(count):
        turns.append(
            AIMessage(
                content="",
                tool_calls=[{"name": "execute", "args": {}, "id": f"exec-{index}"}],
            )
        )
        turns.append(ToolMessage(content="ok", tool_call_id=f"exec-{index}"))
    return turns


def test_no_todos_renders_nothing() -> None:
    assert build_todo_reminder_fragment(None, []) == ""
    assert build_todo_reminder_fragment([], []) == ""


def test_all_completed_renders_nothing() -> None:
    todos = [{"content": "done thing", "status": "completed"}]
    assert build_todo_reminder_fragment(todos, []) == ""


def test_open_list_is_echoed_with_statuses() -> None:
    fragment = build_todo_reminder_fragment(_TODOS, _write_todos_turn())
    assert "## Current todo list (write_todos state)" in fragment
    assert "- [completed] Inspect current source" in fragment
    assert "- [in_progress] Add planets to scene" in fragment
    assert "- [pending] Verify in headless Chromium" in fragment


def test_fresh_list_has_no_nudge() -> None:
    # Note the ToolMessage for write_todos itself lands AFTER the AIMessage that
    # called it, so a just-updated list carries a count of 1 — below any sane
    # threshold, and the reason the threshold floor is enforced at >= 1.
    messages = _write_todos_turn() + _tool_turns(3)
    fragment = build_todo_reminder_fragment(_TODOS, messages, stale_after_tool_results=12)
    assert "statuses above may be stale" not in fragment


def test_stale_list_gets_nudge_with_count() -> None:
    # 1 write_todos ToolMessage + 12 tool results after the plan turn.
    messages = _write_todos_turn() + _tool_turns(12)
    fragment = build_todo_reminder_fragment(_TODOS, messages, stale_after_tool_results=12)
    assert "13 tool results have landed" in fragment
    assert "call write_todos now" in fragment


def test_count_restarts_at_the_latest_write_todos() -> None:
    # Old stale stretch, then a fresh write_todos: only post-update results count.
    messages = (
        _write_todos_turn() + _tool_turns(30) + _write_todos_turn() + _tool_turns(2)
    )
    fragment = build_todo_reminder_fragment(_TODOS, messages, stale_after_tool_results=12)
    assert "statuses above may be stale" not in fragment


def test_compacted_history_still_echoes_and_counts_all_results() -> None:
    # Compaction erased the write_todos turn entirely; todos state survives.
    # The echo is the plan's only remaining trace and every visible tool result
    # counts toward staleness.
    messages = [HumanMessage(content="summary of earlier work")] + _tool_turns(12)
    fragment = build_todo_reminder_fragment(_TODOS, messages, stale_after_tool_results=12)
    assert "## Current todo list (write_todos state)" in fragment
    assert "12 tool results have landed" in fragment


def _request(state: dict[str, Any]) -> ModelRequest:
    return ModelRequest(
        model=cast(BaseChatModel, object()),
        messages=[],
        system_message=SystemMessage(content="base prompt"),
        runtime=cast(Any, None),
        state=cast(Any, state),
    )


def test_middleware_appends_echo_to_system_message() -> None:
    middleware = UltraTodoReminderMiddleware()
    captured: list[Any] = []

    def handler(request: ModelRequest) -> ModelResponse:
        captured.append(request.system_message)
        return ModelResponse(result=[AIMessage(content="ok")])

    middleware.wrap_model_call(
        _request({"todos": _TODOS, "messages": _write_todos_turn()}), handler
    )
    text = str(captured[0].content)
    assert "base prompt" in text
    assert "## Current todo list (write_todos state)" in text


def test_middleware_without_todos_leaves_prompt_untouched() -> None:
    middleware = UltraTodoReminderMiddleware()
    captured: list[Any] = []

    def handler(request: ModelRequest) -> ModelResponse:
        captured.append(request.system_message)
        return ModelResponse(result=[AIMessage(content="ok")])

    middleware.wrap_model_call(_request({"messages": []}), handler)
    assert str(captured[0].content) == "base prompt"
