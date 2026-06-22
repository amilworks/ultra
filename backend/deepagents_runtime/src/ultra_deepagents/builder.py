"""The Builder - a model-agnostic autonomous-coding sub-coordinator.

The coordinator delegates a GOAL (specific, with a machine-checkable DONE-WHEN and a
budget). The Builder OWNS the full build loop for that goal - plan -> implement -> run ->
(self-check) -> evaluate -> iterate - using its own tools and, because it is itself a
deep agent (CompiledSubAgent), its own `task` tool + general-purpose worker for context
isolation. It does not finish until the goal is met or its iteration budget is spent.

Design (verified against deepagents 0.6.8 docs + source):
- A subagent CAN be a full deep agent via ``CompiledSubAgent(runnable=create_deep_agent(...))``.
- The "work until a condition is met" loop is a middleware (``GoalLoopMiddleware``) whose
  ``after_agent`` returns ``{messages, jump_to:'model'}`` to continue or a plain dict to
  finish - the exact mechanism ``RubricMiddleware`` uses.
- MODEL-AGNOSTIC: the loop model is ``build_builder_model(settings)`` pointed at any
  OpenAI-compatible endpoint; the engineering is the asset, not the model choice.

The Builder's loop contract: it must end a turn with a single line
``GOAL_OUTCOME: {json}`` carrying ``met`` (bool) + the structured result the coordinator
reads. ``GoalLoopMiddleware`` parses it; on ``met=false``/missing (within budget) it injects
the gap and jumps back to the model. This v1 enforces the STRUCTURE + don't-quit-early +
iterate-on-unmet; an independent semantic verifier is a documented hardening follow-up.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Annotated, Any

from deepagents import CompiledSubAgent, create_deep_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    PrivateStateAttr,
    hook_config,
)
from langchain_core.messages import AIMessage, HumanMessage
from typing_extensions import NotRequired

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.model import build_builder_model

logger = logging.getLogger(__name__)

GOAL_LOOP_SOURCE = "goal_loop"
# The injection count lives in this DURABLE STATE field, never derived from message history.
# The deep-agent stack includes SummarizationMiddleware, which on context overflow rewrites
# `messages` with RemoveMessage(REMOVE_ALL_MESSAGES) + a fresh summary — pruning the goal_loop
# nudge tags and silently resetting any history-derived count to ~0. Since the Builder exists
# FOR long, context-accumulating runs, that compaction is the COMMON path, so a message-derived
# pathology cap would not hold. A state field is untouched by the messages rewrite.
GOAL_LOOP_INJECTIONS_KEY = "goal_loop_injections"
_GOAL_OUTCOME_RE = re.compile(r"GOAL_OUTCOME:\s*(\{.*\})", re.DOTALL)


class GoalLoopState(AgentState[Any]):
    """Builder agent state extended with the durable GoalLoop injection counter."""

    goal_loop_injections: NotRequired[Annotated[int, PrivateStateAttr]]

BUILDER_NAME = "builder"
BUILDER_DESCRIPTION = (
    "Autonomous build-until-goal coding sub-coordinator for HEAVY or ITERATIVE work - model "
    "training, optimization, multi-step quantitative analysis, pipeline/repro engineering, "
    "anything needing implement -> run -> evaluate -> iterate. Delegate it the GOAL (with a "
    "machine-checkable done-when) plus the relevant DATA/ARTIFACT PATHS, EARLY, and let it do "
    "its OWN exploration, prep, and smaller targeted follow-up analyses in its isolated context "
    "and its own sub-workers; it returns a structured result only when the goal is verified met "
    "or its budget is spent. Do NOT pre-stage everything or re-run pipelines yourself first - "
    "hand the Builder the goal and the paths. Use code-runner instead only for a SINGLE focused "
    "one-shot computation or plot."
)

# Coordinator-facing delegation discipline (added to the system prompt only when the Builder is
# enabled). A live trace showed the coordinator over-prepping in its OWN context (re-running a
# pipeline whose outputs already existed, reading every file, ballooning to 527K tokens) BEFORE
# delegating - undercutting the whole point of an isolated, goal-owning sub-coordinator.
BUILDER_DELEGATION_GUIDANCE = """
A `builder` subagent is available - an autonomous coding sub-coordinator that owns a whole
build/analysis loop in its OWN isolated context. For any request that needs SUBSTANTIAL or
ITERATIVE coding, data analysis, model training, optimization, or a multi-step investigation,
DELEGATE IT TO THE BUILDER EARLY: hand it the GOAL (state a checkable done-when) plus the relevant
data/artifact PATHS, and let IT do the exploration, implementation, and its own smaller targeted
follow-up analyses. Do NOT first stage everything into your own context, read every file, or
re-run a pipeline whose outputs already exist - reuse existing artifacts and point the Builder at
them. Stay the reasoner: write a clear goal, then synthesize the Builder's structured result;
if something needs verifying or extending, delegate to the Builder again rather than doing the
heavy work yourself. Reserve code-runner for a single focused one-shot computation or plot.
""".strip()

BUILDER_SYSTEM_PROMPT = """You are the Builder - an autonomous coding agent that OWNS a goal end to end.

The coordinator has given you a GOAL with a success condition (DONE-WHEN) and a budget. Your job is to
keep building until the goal is VERIFIED met or the budget is spent - you are not done just because code
runs once.

How to work:
1. Restate the goal and its DONE-WHEN as a concrete, MEASURABLE check you can run (e.g. a held-out
   evaluation, a benchmark, a passing test). If the DONE-WHEN is not directly measurable, define the
   closest measurable proxy and say so.
2. Plan with write_todos. Establish the BASELINE first (measure the current state you must beat).
3. Implement and RUN in the sandbox. Inspect outputs, fix errors, iterate. For long or repetitive
   sub-runs (a training trial, a sweep point), delegate to your general-purpose worker via the task tool
   so your own context stays clean - collect only the metric back.
4. VERIFY against the measurable check every time you think you are close. Never claim success from a
   single run or an unverified number. Read the metric from STDOUT or a results file (CSV/JSON) - a
   NUMBER, not a picture.
5. Your model is TEXT-ONLY unless `inspect_images` is in your tool list. Do NOT open image files
   (.png/.jpg/.svg) with read_file - a text-only model cannot view them and it will hard-fail the run;
   they are output artifacts for the user, not inputs for you. ONLY if you actually have `inspect_images`
   may you visually self-check your own rendered figures / training curves to catch defects that pass
   every assertion (inverted colormap, an off-by-one axis, a loss that is technically decreasing but
   diverging). Otherwise rely entirely on the numeric outputs.

END-OF-TURN CONTRACT (required): when you stop, the LAST line of your message MUST be exactly one
machine-readable verdict:
GOAL_OUTCOME: {"met": true|false, "summary": "...", "metric": <value-or-string>, "baseline": <value-or-string>, "target": "...", "what_changed": "...", "artifacts": ["/outputs/..."], "evidence": "the command/eval you ran and its result", "limitations": "..."}
Set "met": true ONLY when your measurable check confirms the goal; otherwise "met": false and keep
working if you have budget. Do not fabricate the verdict - it must reflect an executed verification.
"""


class GoalLoopMiddleware(AgentMiddleware):
    """The "/goal until a condition is met" engine.

    Fires when the Builder reaches a natural stop. If the Builder did not declare the goal
    verified-met (a parseable ``GOAL_OUTCOME`` with ``met=true``) and the iteration budget
    remains, it injects the gap and jumps back to the model. ``max_iterations`` is a
    PATHOLOGY cap (catches a never-converging loop), never a depth/quality cap - a build
    making real progress toward the goal is never throttled by it.

    The budget is counted in DURABLE STATE (``goal_loop_injections``), NOT by scanning the
    message history: SummarizationMiddleware (in the deep-agent stack) compacts ``messages``
    on overflow and would prune the nudge tags, silently resetting a history-derived count
    and defeating the cap on exactly the long runs the Builder is built for. The message scan
    is kept only as a never-undercount lower bound.
    """

    state_schema = GoalLoopState  # type: ignore[assignment]

    def __init__(self, *, max_iterations: int = 6) -> None:
        super().__init__()
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive int")
        self.max_iterations = max_iterations

    @staticmethod
    def _state_injections(state: Any) -> int:  # noqa: ANN401
        raw = (
            state.get(GOAL_LOOP_INJECTIONS_KEY, 0)
            if isinstance(state, dict)
            else getattr(state, GOAL_LOOP_INJECTIONS_KEY, 0)
        )
        try:
            return max(0, int(raw or 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _message_injections(messages: Any) -> int:  # noqa: ANN401
        return sum(
            1 for m in messages
            if (getattr(m, "additional_kwargs", {}) or {}).get("lc_source") == GOAL_LOOP_SOURCE
        )

    @staticmethod
    def _message_text(content: Any) -> str:
        """Normalize an AIMessage's content to text. Via the worker's streaming path the
        content is a LIST of content blocks (not a plain str as the in-process path gives),
        so a naive regex on it raises 'expected string ... got list' (a real live-stack bug)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                    parts.append(str(block["text"]))
            return " ".join(parts)
        return str(content or "")

    @classmethod
    def _parse_outcome(cls, content: Any) -> dict[str, Any] | None:
        m = _GOAL_OUTCOME_RE.search(cls._message_text(content))
        if not m:
            return None
        try:
            obj = json.loads(m.group(1))
            return obj if isinstance(obj, dict) else None
        except (ValueError, TypeError):
            return None

    @hook_config(can_jump_to=["model"])
    def after_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:  # noqa: ANN401
        messages = (
            state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        )
        # iteration budget = how many goal-loop nudges we have already injected. Read from
        # DURABLE STATE (survives the SummarizationMiddleware messages-rewrite); the message
        # scan is a never-undercount lower bound for the pre-summarization path.
        injections = max(self._state_injections(state), self._message_injections(messages))
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        outcome = self._parse_outcome(getattr(last_ai, "content", "") if last_ai else "")

        # Goal verified met -> finish.
        if outcome is not None and outcome.get("met") is True:
            return None
        # Budget exhausted -> finish (the unmet/partial GOAL_OUTCOME is returned to the coordinator).
        if injections >= self.max_iterations:
            logger.warning(
                "GoalLoopMiddleware exhausted max_iterations=%d without a verified goal",
                self.max_iterations,
            )
            return None
        # Otherwise: keep going.
        if outcome is None:
            nudge = (
                "You stopped without a verdict. Verify against the goal's DONE-WHEN by RUNNING the "
                "measurable check, then end your turn with a single line "
                'GOAL_OUTCOME: {"met": true|false, "summary": ..., "metric": ..., "baseline": ..., '
                '"target": ..., "what_changed": ..., "artifacts": [...], "evidence": ..., "limitations": ...}. '
                "If the goal is not yet met, keep building first - you still have budget."
            )
        else:
            nudge = (
                f"GOAL NOT MET (metric={outcome.get('metric')!r} vs target={outcome.get('target')!r}). "
                "The goal still stands and you have budget remaining - improve your approach, run the "
                "measurable check again, and only finish when it confirms the goal (met=true) or you are "
                "out of budget. Do not stop here."
            )
        return {
            "messages": [
                HumanMessage(
                    content=nudge,
                    name=GOAL_LOOP_SOURCE,
                    additional_kwargs={"lc_source": GOAL_LOOP_SOURCE},
                )
            ],
            "jump_to": "model",
            # Persist the bumped count in durable state so the cap holds even after a
            # SummarizationMiddleware compaction prunes the nudge tags from `messages`.
            GOAL_LOOP_INJECTIONS_KEY: injections + 1,
        }


def _is_image_block(block: Any) -> bool:  # noqa: ANN401
    if not isinstance(block, dict):
        return False
    if "image_url" in block:
        return True
    return "image" in str(block.get("type", "")).lower()


def _strip_image_content(messages: list[Any]) -> list[Any]:
    """Drop image content blocks from messages, leaving a text breadcrumb. A text-only model
    hard-400s on ANY image block (e.g. a stray read_file('*.png')); this makes that impossible
    without mutating stored state — only what is sent to the model."""
    cleaned: list[Any] = []
    for m in messages:
        content = getattr(m, "content", None)
        if isinstance(content, list) and any(_is_image_block(b) for b in content):
            kept = [b for b in content if not _is_image_block(b)]
            kept.append({"type": "text", "text": "[image omitted: the Builder model is text-only]"})
            cleaned.append(m.model_copy(update={"content": kept}) if hasattr(m, "model_copy") else m)
        else:
            cleaned.append(m)
    return cleaned


class StripImagesForTextModelMiddleware(AgentMiddleware):
    """Defense-in-depth for a TEXT-ONLY Builder model: strip image content blocks from the
    OUTGOING model request so a stray image in history (a read_file on a .png it rendered) can
    never hard-fail the run with a provider 'not a multimodal model' 400. Attached only when
    the Builder has no inspect_images. The prompt already steers away from reading images; this
    makes it deterministic rather than a matter of the model obeying."""

    def wrap_model_call(self, request: Any, handler: Any) -> Any:  # noqa: ANN401
        return handler(request.override(messages=_strip_image_content(request.messages)))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:  # noqa: ANN401
        return await handler(request.override(messages=_strip_image_content(request.messages)))


def build_builder_subagent(
    settings: RuntimeSettings,
    *,
    tools: list[Any],
    backend: Any,
    vision_tools: list[Any] | None = None,
) -> CompiledSubAgent | None:
    """Build the Builder as a CompiledSubAgent (a full deep agent with its own loop).

    MODEL-AGNOSTIC: ``build_builder_model`` resolves the configured endpoint (or falls back
    to the coordinator's model). The Builder inherits the coordinator's coding tools +
    filesystem backend, gets ``inspect_images`` when configured multimodal (visual
    self-check), and runs the ``GoalLoopMiddleware`` termination loop. Returns ``None`` when
    the Builder is disabled.
    """
    if not settings.builder_enabled:
        return None
    builder_is_multimodal = bool(settings.builder_multimodal and vision_tools)
    builder_tools = list(tools)
    if builder_is_multimodal:
        builder_tools = [*builder_tools, *vision_tools]
    middleware: list[Any] = [GoalLoopMiddleware(max_iterations=settings.builder_goal_max_iterations)]
    if not builder_is_multimodal:
        # The fallback/coordinator model is text-only: guarantee no image block ever reaches it.
        middleware.append(StripImagesForTextModelMiddleware())
    inner = create_deep_agent(
        model=build_builder_model(settings),
        tools=builder_tools,
        system_prompt=BUILDER_SYSTEM_PROMPT,
        backend=backend,
        middleware=middleware,
    ).with_config({"recursion_limit": settings.builder_recursion_limit})
    return CompiledSubAgent(
        name=BUILDER_NAME,
        description=BUILDER_DESCRIPTION,
        runnable=inner,
    )
