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

import asyncio
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler

from ultra_deepagents.events import normalize_reasoning_delta

logger = logging.getLogger(__name__)

# Coalesce per-token thinking into at most a few trace events per second so
# the UI shows live progress without per-token event-ingest pressure.
REASONING_FLUSH_MAX_CHARS = 160
REASONING_FLUSH_MAX_SECONDS = 0.4
REASONING_ACTIVITY_SIGNAL_MIN_SECONDS = 0.25

# A continuously streaming provider is alive from the transport watchdog's
# perspective even when generation has collapsed into a punctuation/token
# loop. One live DeepSeek incident exceeded 350 dash-like characters per 10K
# characters (and eventually emitted U+FFFD); another repeated a four-token
# phrase for 820K characters. In 2K-character tails the latter repeated one
# trigram 109 times at 1.9% token diversity, while 13 neighboring runs peaked
# at 5 repeats and remained above 36% diversity. These deliberately
# conservative bounds require a sustained local-window pathology, not an
# isolated stylistic em dash or ordinary scientific terminology.
REASONING_QUALITY_WINDOW_CHARS = 4096
REASONING_QUALITY_MIN_CHARS = 2048
REASONING_QUALITY_MIN_DASHLIKE = 64
REASONING_QUALITY_MIN_DASHLIKE_RATIO = 0.025
REASONING_QUALITY_SEVERE_MIN_CHARS = 512
REASONING_QUALITY_SEVERE_MIN_DASHLIKE = 32
REASONING_QUALITY_SEVERE_MIN_DASHLIKE_RATIO = 0.06
REASONING_QUALITY_MIN_REPLACEMENT_CHARS = 4
REASONING_QUALITY_REPETITION_WINDOW_CHARS = 2048
REASONING_QUALITY_MIN_REPEATED_TRIGRAM = 48
REASONING_QUALITY_MAX_TOKEN_DIVERSITY = 0.08

_DASHLIKE_RE = re.compile(r"[\u2011\u2013\u2014]")
_WORD_RE = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)

# The deepagents coordinator graph runs its LLM in a node named "model"
# (callback metadata), while v3 stream events label it "agent"/"coordinator".
_COORDINATOR_NODES = {"agent", "coordinator", "main", "model", "ultra-research-agent"}
_COORDINATOR_AGENT_NAMES = {"ultra-research-agent", "main-agent", "coordinator"}


@dataclass(frozen=True)
class ReasoningDegenerationVerdict:
    signal: str
    window_chars: int
    dashlike_count: int
    dashlike_ratio: float
    replacement_char_count: int
    max_repeated_trigram: int
    token_diversity: float


class ReasoningDegenerationError(RuntimeError):
    """A model turn is producing sustained malformed/repetitive output.

    The runner fills the attempt-local fields before bounded recovery, just as
    it does for a dead-open model stream.  No raw model text is carried in the
    exception or its durable diagnostic event.
    """

    def __init__(
        self,
        verdict: ReasoningDegenerationVerdict,
        *,
        observed_chars: int,
    ) -> None:
        super().__init__("Deep Agents model output entered a malformed repetitive generation loop.")
        self.verdict = verdict
        self.partial_result: Any | None = None
        self.reasoning_chars = max(0, int(observed_chars))
        self.active_tool_call = False


def _reasoning_window_metrics(window: str) -> tuple[int, float, int, int, float]:
    window_chars = len(window)
    dashlike_count = len(_DASHLIKE_RE.findall(window))
    dashlike_ratio = dashlike_count / window_chars if window_chars else 0.0
    replacement_char_count = window.count("\ufffd")
    tokens = [token.lower() for token in _WORD_RE.findall(window)]
    token_diversity = len(set(tokens)) / len(tokens) if tokens else 1.0
    max_repeated_trigram = 0
    if len(tokens) >= 3:
        trigrams = Counter(zip(tokens, tokens[1:], tokens[2:], strict=False))
        max_repeated_trigram = max(trigrams.values(), default=0)
    return (
        dashlike_count,
        dashlike_ratio,
        replacement_char_count,
        max_repeated_trigram,
        token_diversity,
    )


def _reasoning_degeneration_verdict(text: str) -> ReasoningDegenerationVerdict | None:
    window = text[-REASONING_QUALITY_WINDOW_CHARS:]
    (
        dashlike_count,
        dashlike_ratio,
        replacement_char_count,
        max_repeated_trigram,
        token_diversity,
    ) = _reasoning_window_metrics(window)

    signal = ""
    if len(window) >= 256 and replacement_char_count >= REASONING_QUALITY_MIN_REPLACEMENT_CHARS:
        signal = "replacement_characters"
    else:
        repetition_window = window[-REASONING_QUALITY_REPETITION_WINDOW_CHARS:]
        (
            repetition_dashlike_count,
            repetition_dashlike_ratio,
            repetition_replacement_char_count,
            repetition_max_repeated_trigram,
            repetition_token_diversity,
        ) = _reasoning_window_metrics(repetition_window)
        if (
            len(repetition_window) >= REASONING_QUALITY_REPETITION_WINDOW_CHARS
            and repetition_max_repeated_trigram >= REASONING_QUALITY_MIN_REPEATED_TRIGRAM
            and repetition_token_diversity <= REASONING_QUALITY_MAX_TOKEN_DIVERSITY
        ):
            signal = "lexical_repetition"
            window = repetition_window
            dashlike_count = repetition_dashlike_count
            dashlike_ratio = repetition_dashlike_ratio
            replacement_char_count = repetition_replacement_char_count
            max_repeated_trigram = repetition_max_repeated_trigram
            token_diversity = repetition_token_diversity
        else:
            severe_window = window[-REASONING_QUALITY_SEVERE_MIN_CHARS:]
            (
                severe_dashlike_count,
                severe_dashlike_ratio,
                severe_replacement_char_count,
                severe_max_repeated_trigram,
                severe_token_diversity,
            ) = _reasoning_window_metrics(severe_window)
            if (
                len(severe_window) >= REASONING_QUALITY_SEVERE_MIN_CHARS
                and severe_dashlike_count >= REASONING_QUALITY_SEVERE_MIN_DASHLIKE
                and severe_dashlike_ratio >= REASONING_QUALITY_SEVERE_MIN_DASHLIKE_RATIO
            ):
                signal = "dashlike_density"
                window = severe_window
                dashlike_count = severe_dashlike_count
                dashlike_ratio = severe_dashlike_ratio
                replacement_char_count = severe_replacement_char_count
                max_repeated_trigram = severe_max_repeated_trigram
                token_diversity = severe_token_diversity
            elif (
                len(window) >= REASONING_QUALITY_MIN_CHARS
                and dashlike_count >= REASONING_QUALITY_MIN_DASHLIKE
                and dashlike_ratio >= REASONING_QUALITY_MIN_DASHLIKE_RATIO
            ):
                signal = "dashlike_density"
    if not signal:
        return None
    return ReasoningDegenerationVerdict(
        signal=signal,
        window_chars=len(window),
        dashlike_count=dashlike_count,
        dashlike_ratio=dashlike_ratio,
        replacement_char_count=replacement_char_count,
        max_repeated_trigram=max_repeated_trigram,
        token_diversity=token_diversity,
    )


class ReasoningQualityGuard(AsyncCallbackHandler):
    """Fail a coordinator model call closed on sustained malformed generation."""

    raise_error = True

    def __init__(self) -> None:
        super().__init__()
        self._eligible_runs: set[Any] = set()
        self._windows: dict[Any, str] = {}
        self._observed_chars: dict[Any, int] = {}

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: Any,
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if _coordinator_model_call(metadata):
            self._eligible_runs.add(run_id)
            self._windows[run_id] = ""
            self._observed_chars[run_id] = 0

    async def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Any = None,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if run_id not in self._eligible_runs:
            return
        reasoning = _chunk_reasoning_delta(chunk)
        text = reasoning or (token if isinstance(token, str) else "")
        if not text:
            return
        self._observed_chars[run_id] = self._observed_chars.get(run_id, 0) + len(text)
        window = (self._windows.get(run_id, "") + text)[-REASONING_QUALITY_WINDOW_CHARS:]
        self._windows[run_id] = window
        verdict = _reasoning_degeneration_verdict(window)
        if verdict is not None:
            raise ReasoningDegenerationError(
                verdict,
                observed_chars=self._observed_chars[run_id],
            )

    async def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._forget(run_id)

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._forget(run_id)

    def _forget(self, run_id: Any) -> None:
        self._eligible_runs.discard(run_id)
        self._windows.pop(run_id, None)
        self._observed_chars.pop(run_id, None)


@dataclass
class _ReasoningRoundBuffer:
    parts: list[str] = field(default_factory=list)
    chars: int = 0
    deadline: float = 0.0
    round_open: bool = False


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
        self._eligible_runs: set[Any] = set()
        # Callback handlers are shared by every coordinator model call in an
        # attempt.  Keep buffers per callback run id so overlapping callbacks
        # can never splice two model streams into one displayed thought.
        self._buffers: dict[Any, _ReasoningRoundBuffer] = {}
        self._observed_chars = 0
        self._last_observed_monotonic = 0.0
        self._last_activity_signal_monotonic = 0.0
        self._activity_version = 0
        self._activity_event = asyncio.Event()

    @property
    def observed_chars(self) -> int:
        """Total coordinator reasoning characters observed during this attempt."""

        return self._observed_chars

    @property
    def last_observed_monotonic(self) -> float:
        """Monotonic timestamp of the latest coordinator reasoning chunk."""

        return self._last_observed_monotonic

    @property
    def activity_version(self) -> int:
        return self._activity_version

    async def wait_for_activity_after(self, version: int) -> int:
        """Wait until reasoning activity advances beyond ``version``.

        The version check around ``Event.clear`` prevents a token arriving in
        that narrow window from being lost. Signals are rate-limited, while
        ``last_observed_monotonic`` still records every raw chunk.
        """

        while self._activity_version <= version:
            self._activity_event.clear()
            if self._activity_version > version:
                break
            await self._activity_event.wait()
        return self._activity_version

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
            self._buffers.setdefault(run_id, _ReasoningRoundBuffer())

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
        state = self._buffers.setdefault(run_id, _ReasoningRoundBuffer())
        reasoning = _chunk_reasoning_delta(chunk)
        if reasoning:
            now = time.monotonic()
            self._last_observed_monotonic = now
            if (
                self._activity_version == 0
                or now - self._last_activity_signal_monotonic
                >= REASONING_ACTIVITY_SIGNAL_MIN_SECONDS
            ):
                self._last_activity_signal_monotonic = now
                self._activity_version += 1
                self._activity_event.set()
            if not state.parts:
                state.deadline = now + REASONING_FLUSH_MAX_SECONDS
            state.parts.append(reasoning)
            state.chars += len(reasoning)
            self._observed_chars += len(reasoning)
            state.round_open = True
            if state.chars >= REASONING_FLUSH_MAX_CHARS or time.monotonic() >= state.deadline:
                await self._flush(run_id)
            return
        if token and state.round_open:
            # First visible answer token of this round: close the thinking
            # step before the corresponding message delta reaches the stream.
            await self._flush(run_id, status="completed")

    async def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        await self._flush(run_id, status="completed")
        self._eligible_runs.discard(run_id)
        self._buffers.pop(run_id, None)

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        await self._flush(run_id, status="completed")
        self._eligible_runs.discard(run_id)
        self._buffers.pop(run_id, None)

    async def aclose(self) -> None:
        """Final closing flush after the agent stream ends."""
        for run_id in sorted(self._buffers, key=str):
            await self._flush(run_id, status="completed")
        self._eligible_runs.clear()
        self._buffers.clear()

    async def _flush(self, run_id: Any, status: str = "running") -> None:
        state = self._buffers.get(run_id)
        if state is None:
            return
        closing = status != "running" and state.round_open
        if not state.parts and not closing:
            return
        text = "".join(state.parts)
        state.parts = []
        state.chars = 0
        state.round_open = status == "running"
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
