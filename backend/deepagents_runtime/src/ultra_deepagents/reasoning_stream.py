"""Streams model thinking as coalesced trace events.

Reasoning deltas do not survive the LangGraph v3 stream protocol: the
chat-completions content-block translator maps only ``content`` text and tool
calls, so thinking chunks (held in ``additional_kwargs`` by
``UltraChatOpenAI``) produce no stream events at all and the UI is silent for
the entire reasoning phase. LangChain callbacks see every raw chunk before
that translation, independent of stream protocol, so the worker taps
``on_llm_new_token`` instead.
"""
from __future__ import annotations

import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler

from ultra_deepagents.events import normalize_reasoning_delta

logger = logging.getLogger(__name__)

# Coalesce per-token thinking into at most a few trace events per second so
# the UI shows live progress without per-token event-ingest pressure.
REASONING_FLUSH_MAX_CHARS = 160
REASONING_FLUSH_MAX_SECONDS = 0.4

# The deepagents coordinator graph runs its LLM in a node named "model"
# (callback metadata), while v3 stream events label it "agent"/"coordinator".
_COORDINATOR_NODES = {"agent", "coordinator", "main", "model", "ultra-research-agent"}
_COORDINATOR_AGENT_NAMES = {"ultra-research-agent", "main-agent", "coordinator"}


def _coordinator_model_call(metadata: dict[str, Any] | None) -> bool:
    """Mirror the runner's coordinator gate for callback metadata: subagent
    model calls carry their own agent name and run in a nested checkpoint
    namespace (segments joined with "|")."""
    metadata = metadata or {}
    node = str(metadata.get("langgraph_node") or "").lower()
    agent_name = str(metadata.get("lc_agent_name") or "").lower()
    checkpoint_ns = str(metadata.get("langgraph_checkpoint_ns") or "")
    if node and node not in _COORDINATOR_NODES:
        return False
    if agent_name and agent_name not in _COORDINATOR_AGENT_NAMES:
        return False
    return "|" not in checkpoint_ns


def _chunk_reasoning_delta(chunk: Any) -> str:
    message = getattr(chunk, "message", None)
    kwargs = getattr(message, "additional_kwargs", None)
    if not isinstance(kwargs, dict):
        return ""
    value = kwargs.get("reasoning_content") or kwargs.get("reasoning")
    return value if isinstance(value, str) else ""


class ReasoningEventStreamer(AsyncCallbackHandler):
    """Publishes coordinator thinking as ``trace.reasoning.delta`` events.

    Buffered text flushes with ``status="running"`` while the model thinks; a
    closing flush with ``status="completed"`` fires when visible content
    starts, the model call ends, or the attempt finishes, so the UI step can
    leave its spinner state. Publishing is best-effort: trace events must
    never fail or stall a run.
    """

    def __init__(self, *, context: Any, sequencer: Any, publish_event: Any) -> None:
        super().__init__()
        self._context = context
        self._sequencer = sequencer
        self._publish_event = publish_event
        self._eligible_runs: set[UUID] = set()
        self._parts: list[str] = []
        self._chars = 0
        self._deadline = 0.0
        self._round_open = False

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if _coordinator_model_call(metadata):
            self._eligible_runs.add(run_id)

    async def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Any = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if run_id not in self._eligible_runs:
            return
        reasoning = _chunk_reasoning_delta(chunk)
        if reasoning:
            if not self._parts:
                self._deadline = time.monotonic() + REASONING_FLUSH_MAX_SECONDS
            self._parts.append(reasoning)
            self._chars += len(reasoning)
            self._round_open = True
            if self._chars >= REASONING_FLUSH_MAX_CHARS or time.monotonic() >= self._deadline:
                await self._flush()
            return
        if token and self._round_open:
            # First visible answer token of this round: close the thinking
            # step before the corresponding message delta reaches the stream.
            await self._flush(status="completed")

    async def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._eligible_runs.discard(run_id)
        await self._flush(status="completed")

    async def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._eligible_runs.discard(run_id)
        await self._flush(status="completed")

    async def aclose(self) -> None:
        """Final closing flush after the agent stream ends."""
        await self._flush(status="completed")

    async def _flush(self, status: str = "running") -> None:
        closing = status != "running" and self._round_open
        if not self._parts and not closing:
            return
        text = "".join(self._parts)
        self._parts = []
        self._chars = 0
        self._round_open = status == "running"
        stamped = self._sequencer.stamp(
            normalize_reasoning_delta(self._context, text, status=status)
        )
        allocated_sequence = stamped.get("sequence")
        try:
            await self._publish_event(stamped)
        except Exception:
            # The reasoning trace is best-effort, but stamp() already consumed a
            # source_sequence. Dropping the event silently would leave a
            # permanent hole that stalls the strict per-run ingest gate. Roll the
            # counter back so the next event reuses this number — but only when
            # nothing else has stamped since (i.e. we are still the latest
            # allocation); otherwise the number is genuinely spent and the
            # control plane's bounded-retry backstop absorbs the gap.
            if (
                isinstance(allocated_sequence, int)
                and self._sequencer.sequence == allocated_sequence
            ):
                self._sequencer.sequence = allocated_sequence - 1
            logger.warning(
                "Reasoning trace publish failed; continuing run without it.",
                extra={"run_id": getattr(self._context, "run_id", "")},
                exc_info=True,
            )
