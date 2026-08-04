"""Tier-1 compressed-horizon harness: week-scale runs in CI minutes.

A "week" is not wall-clock time to the runtime — it is counts: N coordinator
turns, M compaction cycles, K checkpoint writes, R attempt/requeue boundaries.
This harness drives the REAL production machinery at those counts by replacing
exactly two things at the outer edges, both deterministic:

- ``ScriptedChatModel``: a policy-driven chat model (no network, no sampling).
- ``ScriptedSandbox``: a deterministic ``execute()`` backend (no docker).

Everything in between is production code under test: ``run_job``'s attempt
loop, ``build_research_agent``'s full deepagents graph (middleware, subagent
wiring, skills routes), summarization/compaction driven by the model profile
window, the progress-stall guard, the completion guard, the idle watchdog,
the attempt ledger, event normalization + the run sequencer, and the durable
checkpointer.

Compression comes from ``RuntimeSettings`` knobs, never from patched logic:
a tiny ``model_max_input_tokens`` makes compaction cycle every few turns; a
small ``progress_stall_threshold`` trips the livelock breaker in seconds; a
sub-second ``model_stream_idle_timeout_seconds`` exercises the watchdog arm.

Worker restarts are simulated the same way ``test_checkpoint_resume`` does:
one shared ``InMemoryCheckpointStateStore`` plays the durable Postgres row,
while each ``LongHorizonWorld.run()`` invocation gets a fresh
``DurableCheckpointer`` (fresh process memory) and a sequence floor above all
previously published events (the control plane's redelivery contract).
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest import mock

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool

from ultra_deepagents.agent import build_research_agent
from ultra_deepagents.checkpointing import DurableCheckpointer, InMemoryCheckpointStateStore
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.runner import run_job
from ultra_deepagents.schemas import RunJobEnvelope

# Stable substrings of the runner's corrective prompts (asserted in runner.py's
# own text; policies key on them to react the way a real model would).
IDLE_RECOVERY_MARKER = "model stream went idle"
STALL_RECOVERY_MARKER = "Progress stall detected"
STALL_EXHAUSTED_MARKER = "Progress-stall guard"

# Canary tokens planted in goals/outputs to prove constraint retention across
# compaction. The scripted summarizer echoes every canary it can still see, so
# a canary present in a post-compaction model call travelled the real
# summarize -> keep -> next-prompt pipeline.
CANARY_PATTERN = re.compile(r"ULTRA_CONSTRAINT_[A-Z0-9]+")
SUMMARY_STAMP = "[scripted-summary]"

_SUMMARY_MARKER: str = ""


def _summary_prompt_marker() -> str:
    """A literal chunk of deepagents' summary prompt template, used to classify
    incoming model calls as summarization calls without guessing shapes."""
    global _SUMMARY_MARKER
    if _SUMMARY_MARKER:
        return _SUMMARY_MARKER
    from deepagents.middleware.summarization import DEEPAGENTS_DEFAULT_SUMMARY_PROMPT

    literal = DEEPAGENTS_DEFAULT_SUMMARY_PROMPT.split("{", 1)[0].strip()
    # A short distinctive slice is enough; an empty literal would misclassify
    # every call as a summary, so fail loudly instead.
    if len(literal) < 12:
        raise AssertionError(
            "deepagents summary prompt no longer starts with a literal chunk; "
            "update _summary_prompt_marker()"
        )
    _SUMMARY_MARKER = literal[:80]
    return _SUMMARY_MARKER


def _message_text(message: BaseMessage | dict[str, Any]) -> str:
    content = message.get("content") if isinstance(message, dict) else message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content or "")


def _is_system(message: BaseMessage) -> bool:
    return getattr(message, "type", "") == "system"


@dataclass(frozen=True)
class ModelCallRecord:
    """Content-bearing record of one model invocation (test-side observability)."""

    index: int
    invocation: int
    kind: str  # "turn" | "summary"
    message_count: int
    approx_total_tokens: int
    approx_conversation_tokens: int  # non-system messages only (what compaction bounds)
    prompt_text: str
    tool_result_count: int


@dataclass(frozen=True)
class TurnRequest:
    """What a policy sees when the coordinator asks for the next step."""

    invocation: int
    call_index: int
    tool_result_count: int  # ToolMessages visible in this call's context
    messages: tuple[BaseMessage, ...]
    last_text: str  # newest non-system message's text
    full_text: str  # all message text, system included

    def saw(self, marker: str) -> bool:
        return marker in self.full_text

    def last_saw(self, marker: str) -> bool:
        return marker in self.last_text


@dataclass(frozen=True)
class TurnDecision:
    """What a policy tells the scripted model to do for one call."""

    text: str = ""
    execute_command: str | None = None
    # Generic tool dispatch for scripting task/map_task/etc.; mutually
    # exclusive with execute_command (which remains the ergonomic shorthand).
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    sleep_seconds: float = 0.0
    raise_error: BaseException | None = None


TurnPolicy = Callable[[TurnRequest], TurnDecision]


class ScriptedChatModel(BaseChatModel):
    """Deterministic policy-driven model with a declared context-window profile.

    The profile's ``max_input_tokens`` is what flips deepagents summarization to
    adaptive fraction-based compaction (trigger 85% / keep 10% of the window) —
    the same mechanism production enables via
    ``RuntimeSettings.model_max_input_tokens``.

    Summarization calls (recognized by deepagents' own summary-prompt literal)
    are answered by a rule-based summarizer that echoes every canary token still
    visible in the to-summarize content, so constraint retention is measured on
    the real transport, not on a clever fake.
    """

    policy: Any = None
    world: Any = None
    window_tokens: int = 0
    profile: dict[str, Any] | None = None
    model_name: str = "scripted-longhorizon"

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "scripted-longhorizon"

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: Any | None = None,
        **kwargs: Any,
    ) -> Runnable:
        del tool_choice  # deterministic model ignores forcing; graph never sets it
        return self.bind(tools=[convert_to_openai_tool(tool) for tool in tools], **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        return ChatResult(
            generations=[ChatGeneration(message=self._decide_message(messages))]
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ):
        """Streaming surface: production models stream (``stream_usage=True``),
        and the v3 protocol only carries per-call usage on the streaming path's
        ``message-finish`` — so the scripted model streams one chunk shaped like
        a real final chunk (text/tool-call deltas + usage metadata)."""
        del stop, run_manager, kwargs
        message = self._decide_message(messages)
        chunk = AIMessageChunk(
            content=message.content,
            tool_call_chunks=[
                {
                    "name": call["name"],
                    "args": json.dumps(call["args"]),
                    "id": call["id"],
                    "index": position,
                    "type": "tool_call_chunk",
                }
                for position, call in enumerate(message.tool_calls)
            ],
            usage_metadata=message.usage_metadata,
            response_metadata=message.response_metadata,
        )
        yield ChatGenerationChunk(message=chunk)

    def _decide_message(self, messages: list[BaseMessage]) -> AIMessage:
        world: LongHorizonWorld = self.world
        full_text = "\n".join(_message_text(message) for message in messages)
        conversation = [message for message in messages if not _is_system(message)]
        tool_result_count = sum(isinstance(message, ToolMessage) for message in conversation)
        is_summary = _summary_prompt_marker() in full_text
        record = ModelCallRecord(
            index=len(world.model_calls) + 1,
            invocation=world.invocation,
            kind="summary" if is_summary else "turn",
            message_count=len(messages),
            approx_total_tokens=count_tokens_approximately(messages),
            approx_conversation_tokens=count_tokens_approximately(conversation)
            if conversation
            else 0,
            prompt_text=full_text,
            tool_result_count=tool_result_count,
        )
        world.model_calls.append(record)

        if is_summary:
            canaries = sorted(set(CANARY_PATTERN.findall(full_text)))
            text = (
                f"{SUMMARY_STAMP} Prior work condensed deterministically. "
                f"Constraints still in force: {' '.join(canaries) if canaries else 'none'}."
            )
            return self._finalize(messages, AIMessage(content=text))

        request = TurnRequest(
            invocation=world.invocation,
            call_index=record.index,
            tool_result_count=tool_result_count,
            messages=tuple(messages),
            last_text=_message_text(conversation[-1]) if conversation else "",
            full_text=full_text,
        )
        decision: TurnDecision = self.policy(request)
        if decision.sleep_seconds > 0:
            # Runs on the executor thread BaseChatModel._agenerate uses, so the
            # event stream is genuinely silent while we sleep — exactly the
            # dead-transport shape the idle watchdog exists to catch.
            time.sleep(decision.sleep_seconds)
        if decision.raise_error is not None:
            raise decision.raise_error
        tool_name = decision.tool_name
        tool_args = decision.tool_args
        if decision.execute_command is not None:
            tool_name, tool_args = "execute", {"command": decision.execute_command}
        if tool_name is not None:
            message = AIMessage(
                content=decision.text,
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": tool_args or {},
                        "id": f"call_{world.invocation}_{record.index}",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            message = AIMessage(content=decision.text)
        return self._finalize(messages, message)

    def _finalize(self, prompt: list[BaseMessage], message: AIMessage) -> AIMessage:
        input_tokens = count_tokens_approximately(prompt)
        output_tokens = max(1, len(_message_text(message)) // 4)
        message.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        message.response_metadata = {"model_name": self.model_name}
        return message


class ScriptedSandbox(BaseSandbox):
    """Deterministic sandbox: ``behavior(command, nth_call)`` -> ExecuteResponse.

    Records every executed command — the harness's side-effect ledger. Duplicate
    side effects after a crash/resume boundary show up here as repeated
    commands, which is precisely the durability claim under test.
    """

    def __init__(
        self,
        behavior: Callable[[str, int], ExecuteResponse] | None = None,
    ) -> None:
        self._behavior = behavior
        self.calls: list[str] = []

    @property
    def id(self) -> str:
        return "scripted-sandbox"

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        del timeout
        self.calls.append(command)
        if self._behavior is not None:
            return self._behavior(command, len(self.calls))
        return ExecuteResponse(output=f"ok #{len(self.calls)}: {command}", exit_code=0)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [FileUploadResponse(path=path, error=None) for path, _ in files]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return [
            FileDownloadResponse(path=path, content=None, error="file_not_found")
            for path in paths
        ]


class EventLog:
    """In-memory control plane: captures every published, sequencer-stamped event."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def publish(self, event: dict[str, Any]) -> None:
        self.events.append(copy.deepcopy(event))

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event.get("event_kind") == kind]

    def kinds(self) -> list[str]:
        return [str(event.get("event_kind")) for event in self.events]

    def max_sequence(self) -> int:
        return max((int(event.get("sequence") or 0) for event in self.events), default=0)

    def tool_events(self, tool_name: str = "") -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for event in self.events:
            payload = event.get("payload")
            if not isinstance(payload, dict) or "tool_name" not in payload:
                continue
            if tool_name and str(payload.get("tool_name")) != tool_name:
                continue
            selected.append(event)
        return selected


@dataclass
class CompressedConfig:
    """The knobs that compress a week into a CI run. All map 1:1 onto
    RuntimeSettings fields — compression is configuration, never patched logic.

    ``context_window_tokens`` feeds the model profile's ``max_input_tokens``,
    which drives deepagents' adaptive compaction (trigger 85% / keep 10%). The
    trigger counts messages PLUS the system prompt and tool schemas — with the
    full Ultra agent that fixed overhead is ~12-14k approx-tokens, so windows
    below it run in a permanent-compaction regime (summarize every turn) while
    windows above it produce the production sawtooth. The default sits in the
    sawtooth regime; scenarios opt down deliberately to stress the other."""

    context_window_tokens: int = 24_000
    progress_stall_threshold: int = 0  # 0 disables (scenario opts in)
    progress_stall_max_recoveries: int = 1
    idle_timeout_seconds: float = 3600.0
    idle_max_recoveries: int = 1
    completion_max_continuations: int = 2
    recursion_limit: int = 20_000


@dataclass
class LongHorizonWorld:
    """Shared durable world across simulated worker restarts of one run."""

    tmp: Path
    run_id: str = "run-longhorizon-1"
    behavior: Callable[[str, int], ExecuteResponse] | None = None
    store: InMemoryCheckpointStateStore = field(default_factory=InMemoryCheckpointStateStore)
    log: EventLog = field(default_factory=EventLog)
    model_calls: list[ModelCallRecord] = field(default_factory=list)
    invocation: int = 0

    def __post_init__(self) -> None:
        self.sandbox = ScriptedSandbox(self.behavior)

    def settings(self, config: CompressedConfig) -> RuntimeSettings:
        uploads = self.tmp / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        return RuntimeSettings(
            openai_base_url="http://scripted.invalid/v1",
            openai_model="scripted-longhorizon",
            model_max_input_tokens=config.context_window_tokens,
            workspace_root=str(self.tmp / "workspaces"),
            artifact_root=str(self.tmp / "artifacts"),
            memory_root=str(self.tmp / "memory"),
            rarespot_upload_roots=(str(uploads),),
            title_generation_enabled=False,
            model_stream_idle_timeout_seconds=config.idle_timeout_seconds,
            model_stream_idle_max_recoveries=config.idle_max_recoveries,
            progress_stall_threshold=config.progress_stall_threshold,
            progress_stall_max_recoveries=config.progress_stall_max_recoveries,
            completion_max_continuations=config.completion_max_continuations,
            langgraph_recursion_limit=config.recursion_limit,
        )

    async def run(
        self,
        policy: TurnPolicy,
        *,
        goal: str,
        config: CompressedConfig,
    ) -> str:
        """One worker-process lifetime: fresh checkpointer memory + fresh agent,
        durable store and event floor carried over — the redelivery contract."""
        self.invocation += 1
        settings = self.settings(config)
        model = ScriptedChatModel(
            policy=policy,
            world=self,
            window_tokens=config.context_window_tokens,
            profile={"max_input_tokens": config.context_window_tokens},
        )

        def factory(factory_settings: RuntimeSettings, **kwargs: Any) -> Any:
            return build_research_agent(factory_settings, model=model, **kwargs)

        checkpointer = DurableCheckpointer(self.store)
        job = RunJobEnvelope(
            run_id=self.run_id,
            thread_id="thread-longhorizon",
            user_id="longhorizon-tester",
            goal=goal,
        )
        with mock.patch(
            "ultra_deepagents.agent.build_sandbox_backend",
            return_value=self.sandbox,
        ):
            try:
                return await run_job(
                    job,
                    settings,
                    publish_event=self.log.publish,
                    agent_factory=factory,
                    checkpointer=checkpointer,
                    sequence_floor=self.log.max_sequence(),
                )
            finally:
                await checkpointer.flush()

    def run_sync(self, policy: TurnPolicy, *, goal: str, config: CompressedConfig) -> str:
        return asyncio.run(self.run(policy, goal=goal, config=config))

    async def checkpoint_encoded_bytes(self) -> int:
        blob = await self.store.load(self.run_id)
        return len(blob) if blob else 0


def staged_pipeline_policy(
    world: LongHorizonWorld,
    rounds: int,
    *,
    final_answer: str,
    command_prefix: str = "python stage.py --index",
    crash_on_round: int = 0,
    crash_only_in_invocation: int = 1,
    crash_error: Callable[[], BaseException] | None = None,
) -> TurnPolicy:
    """Drive ``rounds`` distinct execute stages, then answer.

    Progress is derived from the sandbox's side-effect ledger (commands that
    actually executed), never from closure counters or visible context: context
    is rewritten by compaction (old tool results get summarized away) and
    replayed on crash/redelivery resume, but the external world is monotone.
    A resumed run therefore continues at the next unexecuted stage — and if the
    runtime ever wrongly re-executed a completed stage, the ledger (and this
    policy) would surface it as a duplicate command.
    """

    def policy(request: TurnRequest) -> TurnDecision:
        next_round = len(world.sandbox.calls) + 1
        if (
            crash_on_round
            and next_round == crash_on_round
            and request.invocation == crash_only_in_invocation
        ):
            error = crash_error() if crash_error is not None else RuntimeError(
                "simulated worker crash mid-run"
            )
            return TurnDecision(raise_error=error)
        if next_round <= rounds:
            return TurnDecision(execute_command=f"{command_prefix} {next_round}")
        return TurnDecision(text=final_answer)

    return policy
