"""Tests for SubagentFailureIsolationMiddleware (Phase 1 of the subagents hardening).

Proves the core property: a failing/slow `task` subagent becomes a degraded ToolMessage instead
of a raise, so one bad subagent in a parallel `asyncio.gather` fan-out cannot cancel its siblings
and abort the run — while control-flow signals (GraphInterrupt/HITL, cancellation) still propagate.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp, GraphInterrupt
from ultra_deepagents.subagent_resilience import SubagentFailureIsolationMiddleware


def _req(name: str = "task", *, subagent: str | None = "code-runner", call_id: str = "call_1"):
    args = {"subagent_type": subagent} if subagent is not None else {}
    return SimpleNamespace(tool_call={"name": name, "args": args, "id": call_id})


def _run(coro):
    return asyncio.run(coro)


# --- async path (Ultra's live runner) ---------------------------------------------------------


def test_failing_task_becomes_degraded_message():
    mw = SubagentFailureIsolationMiddleware()

    async def handler(_req):
        raise RuntimeError("kaboom")

    out = _run(mw.awrap_tool_call(_req(), handler))
    assert isinstance(out, ToolMessage)
    assert out.status == "error"
    assert out.tool_call_id == "call_1" and out.name == "task"
    assert "code-runner" in out.content  # names the target subagent
    assert "RuntimeError" in out.content and "isolated" in out.content.lower()
    assert "UNAFFECTED" in out.content  # tells the coordinator siblings are fine


def test_successful_task_passes_through_unchanged():
    mw = SubagentFailureIsolationMiddleware()
    sentinel = ToolMessage("done", tool_call_id="call_1")

    async def handler(_req):
        return sentinel

    assert _run(mw.awrap_tool_call(_req(), handler)) is sentinel


def test_non_task_tool_failure_is_isolated():
    """A failing ordinary tool (e.g. bisque_upload_files hitting a control-plane error) is degraded
    to a ToolMessage instead of propagating and killing the run. The degraded message names the
    actual tool (not 'task') and carries no subagent-fan-out language."""
    mw = SubagentFailureIsolationMiddleware()

    async def handler(_req):
        raise RuntimeError("upload failed")

    out = _run(mw.awrap_tool_call(_req(name="bisque_upload_files", subagent=None), handler))
    assert isinstance(out, ToolMessage)
    assert out.status == "error"
    assert out.name == "bisque_upload_files" and out.tool_call_id == "call_1"
    assert "RuntimeError" in out.content and "isolated" in out.content.lower()
    assert "UNAFFECTED" not in out.content  # not subagent-fan-out phrasing


def test_non_task_tool_has_no_deadline():
    """The per-task deadline must never apply to an ordinary tool — only isolation does."""
    mw = SubagentFailureIsolationMiddleware(timeout_seconds=0.05)
    sentinel = ToolMessage("ok", tool_call_id="call_1")

    async def handler(_req):
        await asyncio.sleep(0.2)  # longer than the timeout; must NOT be cut off for a non-task tool
        return sentinel

    assert _run(mw.awrap_tool_call(_req(name="search_resources", subagent=None), handler)) is sentinel


def test_graph_interrupt_propagates_not_swallowed():
    """GraphInterrupt is an Exception subclass — a naive `except Exception` would break HITL."""
    mw = SubagentFailureIsolationMiddleware()

    async def handler(_req):
        raise GraphInterrupt()

    with pytest.raises(GraphInterrupt):
        _run(mw.awrap_tool_call(_req(), handler))


def test_graph_bubble_up_propagates():
    mw = SubagentFailureIsolationMiddleware()

    async def handler(_req):
        raise GraphBubbleUp()

    with pytest.raises(GraphBubbleUp):
        _run(mw.awrap_tool_call(_req(), handler))


def test_cancellation_propagates():
    mw = SubagentFailureIsolationMiddleware()

    async def handler(_req):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        _run(mw.awrap_tool_call(_req(), handler))


def test_timeout_becomes_degraded_message_for_bounded_subagent():
    mw = SubagentFailureIsolationMiddleware(timeout_seconds=0.05)

    async def handler(_req):
        await asyncio.sleep(5)  # exceeds the deadline

    out = _run(mw.awrap_tool_call(_req(subagent="vision-reasoner"), handler))
    assert isinstance(out, ToolMessage) and out.status == "error"
    assert "deadline" in out.content.lower()


def test_deadline_excluded_for_compute_subagent():
    """code-runner/builder legitimately run 30-67 min — the per-task deadline must NOT apply to
    them (only to bounded subagents), or it would false-kill real training/sim work."""
    mw = SubagentFailureIsolationMiddleware(timeout_seconds=0.05)
    sentinel = ToolMessage("training done", tool_call_id="call_1")

    async def handler(_req):
        await asyncio.sleep(0.2)  # 4x the deadline, but code-runner is excluded so it is NOT cut
        return sentinel

    assert _run(mw.awrap_tool_call(_req(subagent="code-runner"), handler)) is sentinel
    # ...but isolation of an actual CRASH still applies to compute subagents:
    async def crashing(_req):
        raise RuntimeError("real crash")

    out = _run(mw.awrap_tool_call(_req(subagent="code-runner"), crashing))
    assert isinstance(out, ToolMessage) and out.status == "error" and "RuntimeError" in out.content


def test_no_timeout_when_disabled():
    """timeout_seconds=0 => no per-task timer; a slow-but-completing subagent is not cut off."""
    mw = SubagentFailureIsolationMiddleware(timeout_seconds=0.0)
    sentinel = ToolMessage("slow ok", tool_call_id="call_1")

    async def handler(_req):
        await asyncio.sleep(0.01)
        return sentinel

    assert _run(mw.awrap_tool_call(_req(), handler)) is sentinel


def test_degraded_message_default_target_when_unspecified():
    mw = SubagentFailureIsolationMiddleware()

    async def handler(_req):
        raise ValueError("x")

    out = _run(mw.awrap_tool_call(_req(subagent=None), handler))
    assert isinstance(out, ToolMessage) and "'subagent'" in out.content


# --- sync path (invoke/stream) ---------------------------------------------------------------


def test_sync_failing_task_becomes_degraded_message():
    mw = SubagentFailureIsolationMiddleware()

    def handler(_req):
        raise ValueError("nope")

    out = mw.wrap_tool_call(_req(), handler)
    assert isinstance(out, ToolMessage) and out.status == "error" and "ValueError" in out.content


def test_sync_graph_interrupt_propagates():
    mw = SubagentFailureIsolationMiddleware()

    def handler(_req):
        raise GraphInterrupt()

    with pytest.raises(GraphInterrupt):
        mw.wrap_tool_call(_req(), handler)


def test_sync_non_task_tool_failure_is_isolated():
    mw = SubagentFailureIsolationMiddleware()

    def handler(_req):
        raise RuntimeError("boom")

    out = mw.wrap_tool_call(_req(name="read_file", subagent=None), handler)
    assert isinstance(out, ToolMessage)
    assert out.status == "error" and out.name == "read_file" and "RuntimeError" in out.content
