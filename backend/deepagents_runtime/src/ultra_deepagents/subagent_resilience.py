"""Tool-call failure isolation + a per-task wall-clock deadline for delegated subagents.

langgraph's ``ToolNode`` is built (in deepagents' ``create_agent``) WITHOUT ``handle_tool_errors``,
so its default handler re-raises anything that is not a ``ToolInvocationError``. Any tool that raises
then propagates out of the node and fails the WHOLE run. Two ways this bites:
  * a fanned-out ``task()`` subagent that raises (recursion-limit ``GraphRecursionError``, a
    structured-output ``OutputParserException``, an uncaught model/httpx error, or our per-task
    timeout) is run via ``asyncio.gather(*coros)`` with **no** ``return_exceptions`` — so it also
    cancels its in-flight siblings, negating parallel fan-out where partial success should be usable;
  * an ordinary coordinator tool that raises (e.g. ``bisque_upload_files`` hitting a control-plane
    HTTP error) kills the run outright — exactly how the OME-Zarr otsu run died on a 500.

``SubagentFailureIsolationMiddleware`` wraps EVERY coordinator tool call: any failure becomes a
degraded ``ToolMessage`` (``status="error"``) the coordinator can reason about, instead of a raise.
The per-task wall-clock deadline is scoped narrowly — only to the ``task`` tool targeting a BOUNDED
subagent (see ``DEADLINE_BOUNDED_SUBAGENTS``); every other tool gets isolation but no timer.
Control-flow signals are re-raised untouched: ``GraphBubbleUp`` (the parent of ``GraphInterrupt``,
so human-in-the-loop ``interrupt_on`` and other graph bubble-ups still work) and cancellation
(``CancelledError``/``KeyboardInterrupt``) must never be swallowed.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)

TASK_TOOL_NAME = "task"

# Subagents whose work is BOUNDED (image analysis, paper review) and so should never run for
# many minutes — a long silence from one is a hang, not progress. Only these get the per-task
# wall-clock deadline. The COMPUTATIONAL delegates (code-runner, qwen-code-runner, builder)
# legitimately run 30-67 min on real training/sims (measured), so a total-time deadline would
# false-kill them; they rely on the run-level idle watchdog + the sandbox wall-clock cap instead.
# Failure ISOLATION (crash -> degraded) still applies to ALL subagents regardless of this set.
DEADLINE_BOUNDED_SUBAGENTS = frozenset({"vision-reasoner", "literature-reviewer"})


def _tool_name(request: ToolCallRequest) -> str:
    return str(request.tool_call.get("name") or "")


def _target_subagent(request: ToolCallRequest) -> str:
    """The name of the subagent the failed task() targeted, for the degraded message."""
    args = request.tool_call.get("args")
    if isinstance(args, dict):
        for key in ("subagent_type", "name"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return "subagent"


def _degraded_message(request: ToolCallRequest, detail: str) -> ToolMessage:
    name = _tool_name(request) or "tool"
    tool_call_id = str(request.tool_call.get("id") or "")
    if name == TASK_TOOL_NAME:
        target = _target_subagent(request)
        content = (
            f"Delegation to the '{target}' subagent failed and was isolated: {detail}. "
            "This single subagent did not complete; any sibling subagents launched in the same "
            "turn are UNAFFECTED and their results stand. Decide whether to retry this piece, "
            "route it differently, or proceed with the partial results you have."
        )
    else:
        content = (
            f"The '{name}' tool call failed and was isolated: {detail}. The run is still alive — "
            "only this tool call did not complete, nothing else was aborted. Decide whether to "
            "retry it, take a different approach, or proceed with what you already have."
        )
    return ToolMessage(content=content, tool_call_id=tool_call_id, name=name, status="error")


class SubagentFailureIsolationMiddleware(AgentMiddleware[Any, Any, Any]):
    """Contain a failing/slow tool call to a degraded ``ToolMessage`` so one bad tool — a crashing
    subagent in a parallel fan-out, or an ordinary tool hitting an error — cannot cancel siblings or
    abort the run. Isolation applies to EVERY coordinator tool; control-flow bubble-ups and
    cancellation always propagate.

    Optional per-task wall-clock deadline (``timeout_seconds``, ``0`` disables), scoped to the
    ``task`` tool targeting a bounded subagent only. The clock is execution-relative — it starts when
    this wrapper actually runs the subagent (i.e. when the gathered coroutine executes), so it
    composes correctly with concurrent fan-out and any future width cap. The deadline applies to the
    async path only (``asyncio.timeout`` is async-only); the sync path still gets full failure
    isolation, just no per-task timer.
    """

    tools = ()

    def __init__(
        self,
        *,
        timeout_seconds: float = 0.0,
        deadline_subagents: frozenset[str] = DEADLINE_BOUNDED_SUBAGENTS,
    ) -> None:
        super().__init__()
        self._timeout: float | None = (
            timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
        )
        # The per-task wall-clock deadline applies ONLY to these (bounded) subagents; isolation
        # of a raising subagent applies to all. An empty set disables the deadline entirely.
        self._deadline_subagents = deadline_subagents

    def _deadline_applies(self, request: ToolCallRequest) -> bool:
        return (
            self._timeout is not None
            and _tool_name(request) == TASK_TOOL_NAME
            and _target_subagent(request) in self._deadline_subagents
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        try:
            return handler(request)
        except GraphBubbleUp:
            raise  # GraphInterrupt (HITL) + other control-flow bubble-ups MUST propagate
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise  # cancellation MUST propagate (BaseException, but be explicit)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: isolate ANY tool crash
            logger.warning(
                "Isolated a failed '%s' tool call: %s",
                _tool_name(request) or "?",
                exc,
                exc_info=True,
            )
            return _degraded_message(request, f"{type(exc).__name__}: {exc}")

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        deadline: asyncio.Timeout | None = None
        try:
            if self._deadline_applies(request):
                deadline = asyncio.timeout(self._timeout)
                async with deadline:
                    return await handler(request)
            return await handler(request)
        except GraphBubbleUp:
            raise
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        except Exception as exc:  # noqa: BLE001
            if (
                isinstance(exc, TimeoutError)
                and deadline is not None
                and deadline.expired()
                and self._timeout is not None
            ):
                # asyncio.timeout converts its own cancellation into TimeoutError at the boundary;
                # an EXTERNAL cancel surfaces as CancelledError above and is re-raised, not caught.
                logger.warning(
                    "Subagent '%s' exceeded the %.0fs per-task deadline; isolating it",
                    _target_subagent(request),
                    self._timeout,
                )
                return _degraded_message(
                    request, f"exceeded the {self._timeout:.0f}s per-task wall-clock deadline"
                )
            if isinstance(exc, TimeoutError):
                logger.warning(
                    "Isolated a failed '%s' tool call: TimeoutError",
                    _tool_name(request) or "?",
                )
                return _degraded_message(request, "TimeoutError")
            logger.warning(
                "Isolated a failed '%s' tool call: %s",
                _tool_name(request) or "?",
                exc,
                exc_info=True,
            )
            return _degraded_message(request, f"{type(exc).__name__}: {exc}")
