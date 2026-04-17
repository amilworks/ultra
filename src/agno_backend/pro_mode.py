from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from agno.agent import Agent
from agno.workflow import Parallel, Step, Workflow
from agno.workflow.condition import Condition
from agno.workflow.router import Router
from agno.workflow.steps import Steps
from agno.workflow.types import StepInput, StepOutput
from pydantic import BaseModel, Field

from src.tooling.calculator import numpy_calculator

from .pro_mode_prompts import ROLE_SYSTEM_PROMPTS

ProModeRole = Literal[
    "problem_framer",
    "decomposer",
    "mechanist",
    "formalist",
    "empiricist",
    "contrarian",
    "socratic_crux_examiner",
    "tool_broker",
    "synthesizer",
]
ProModeRoute = Literal["direct_response", "deep_reasoning", "tool_workflow"]
ExecutionRegime = Literal[
    "fast_dialogue",
    "validated_tool",
    "iterative_research",
    "autonomous_cycle",
    "focused_team",
    "proof_workflow",
    "reasoning_solver",
    "expert_council",
]
TaskRegime = Literal[
    "phatic_or_small_talk",
    "closed_form_grounded",
    "rigorous_proof",
    "self_contained_reasoning",
    "artifact_interpretation",
    "dataset_or_catalog_research",
    "iterative_multimodal_research",
    "conceptual_high_uncertainty",
]
ConfidenceLevel = Literal["low", "medium", "high"]
ConsensusLevel = Literal["low", "medium", "high"]
TaskType = Literal["quantitative", "mechanistic", "conceptual", "mixed"]
CouncilVote = Literal["agree", "agree_with_reservation", "needs_revision"]
MessageKind = Literal["private_memo", "critique", "rebuttal", "evidence", "synthesis", "verifier"]

PRIVATE_FIRST_ROLES: tuple[ProModeRole, ...] = (
    "problem_framer",
    "decomposer",
    "mechanist",
    "formalist",
    "empiricist",
    "contrarian",
)
DISCUSSION_ROLES: tuple[ProModeRole, ...] = PRIVATE_FIRST_ROLES
REVIEW_ROLES: tuple[ProModeRole, ...] = (
    "socratic_crux_examiner",
    "tool_broker",
)
ALL_COUNCIL_ROLES: tuple[ProModeRole, ...] = (*PRIVATE_FIRST_ROLES, *REVIEW_ROLES, "synthesizer")
PRO_MODE_PHASE_ORDER: tuple[str, ...] = (
    "intake",
    "context_policy",
    "execution_router",
    "direct_response",
    "tool_workflow",
    "proof_workflow",
    "autonomous_cycle",
    "preflight",
    "context_hydrate",
    "private_memos",
    "socratic_review",
    "critique_round_1",
    "tool_broker",
    "calculator_evidence",
    "critique_round_2",
    "synthesis",
    "verifier",
    "retry_round",
    "finalize",
)
LEGACY_DEFAULT_RUNTIME_SECONDS = 900
DEFAULT_PRO_MODE_MAX_RUNTIME_SECONDS = 7200

ROLE_LABELS: dict[ProModeRole, str] = {
    "problem_framer": "Problem Framer",
    "decomposer": "Decomposer",
    "mechanist": "Mechanist",
    "formalist": "Formalist",
    "empiricist": "Empiricist",
    "contrarian": "Contrarian",
    "socratic_crux_examiner": "Socratic Crux Examiner",
    "tool_broker": "Tool Broker",
    "synthesizer": "Synthesizer",
}

ROLE_BRIEFS: dict[ProModeRole, str] = {
    "problem_framer": "Restate the objective, constraints, and success condition.",
    "decomposer": "Break the task into subproblems and dependency order.",
    "mechanist": "Track the causal, mechanistic, or process-level explanation.",
    "formalist": "Check equations, conventions, symbolic consistency, and edge cases.",
    "empiricist": "Ask what evidence or observation would settle uncertainty.",
    "contrarian": "Stress-test the leading answer by surfacing the strongest alternative explanation or failure mode.",
    "socratic_crux_examiner": "Expose hidden assumptions, weak causal jumps, undefined terms, and the highest-value crux questions.",
    "tool_broker": "Request deterministic calculator work only when it materially resolves the answer.",
    "synthesizer": "Write the final, polished answer only after the discussion converges.",
}

ROUND_ONE_TARGETS: dict[ProModeRole, tuple[ProModeRole, ...]] = {
    "problem_framer": ("decomposer", "socratic_crux_examiner"),
    "decomposer": ("problem_framer", "mechanist"),
    "mechanist": ("formalist", "contrarian"),
    "formalist": ("mechanist", "empiricist"),
    "empiricist": ("mechanist", "formalist"),
    "contrarian": ("mechanist", "formalist", "empiricist"),
    "socratic_crux_examiner": ("problem_framer", "formalist", "empiricist", "contrarian"),
    "tool_broker": ("formalist", "empiricist", "socratic_crux_examiner"),
    "synthesizer": (),
}

SOCRATIC_CRUX_EXAMINER_GUIDANCE = """
You are the Socratic Crux Examiner in a multi-agent scientific reasoning system.

Your purpose is not to solve the problem first.
Your purpose is to improve the team's reasoning by exposing hidden assumptions, unclear definitions, weak causal jumps, missing evidence, and unresolved cruxes.

Your goal:
1. Identify the few highest-value questions that determine whether the team's current reasoning is sound.
2. Force ambiguous claims into precise, testable statements.
3. Surface the strongest alternative explanations or interpretations.
4. Clarify what evidence, calculation, or tool call would actually resolve the disagreement.
5. Improve the final answer by making the reasoning sharper, more honest, and more falsifiable.

Behavioral constraints:
- Ask only high-leverage questions.
- Prefer one decisive crux over many shallow objections.
- Stop attacking once the central cruxes are resolved.
- Do not generate theatrical skepticism or endless philosophical questions.
- Do not produce a final answer unless explicitly asked.
- Do not resolve your own central cruxes without evidence, calculation, or explicit cross-role agreement.
""".strip()


class ProModeTaskProfile(BaseModel):
    objective: str
    task_type: TaskType = "mixed"
    success_criteria: list[str] = Field(default_factory=list)
    calculator_candidate: bool = False
    exactness_priority: bool = False
    report_requested: bool = False
    notes: list[str] = Field(default_factory=list)


class ProModeContextPolicy(BaseModel):
    load_memory: bool = False
    load_knowledge: bool = False
    history_window: int = 0
    artifact_handles_to_expose: list[str] = Field(default_factory=list)
    compression_required: bool = False


class ProModeIntakeDecision(BaseModel):
    route: ProModeRoute = "deep_reasoning"
    execution_regime: ExecutionRegime = "reasoning_solver"
    task_regime: TaskRegime = "conceptual_high_uncertainty"
    reason: str = ""
    direct_response: str | None = None
    selected_tool_names: list[str] = Field(default_factory=list)
    tool_plan_category: str | None = None
    strict_tool_validation: bool = False
    load_memory: bool = False
    load_knowledge: bool = False
    recent_history_turns: int = 2
    context_policy: ProModeContextPolicy = Field(default_factory=ProModeContextPolicy)


class ProModeRoleMemo(BaseModel):
    role: str
    headline: str
    claims: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = "medium"


class ProModeCalculatorRequest(BaseModel):
    purpose: str
    expression: str
    variables: dict[str, Any] = Field(default_factory=dict)


class ProModeCalculatorResult(BaseModel):
    purpose: str
    expression: str
    success: bool
    formatted_result: str
    result: Any = None
    error: str | None = None


class CouncilMessage(BaseModel):
    message_id: str
    round_index: int
    sender_role: str
    recipient_roles: list[str] = Field(default_factory=list)
    reply_to_ids: list[str] = Field(default_factory=list)
    message_kind: MessageKind
    content: str
    claims: list[str] = Field(default_factory=list)
    objections: list[str] = Field(default_factory=list)
    requested_actions: list[str] = Field(default_factory=list)
    ready_to_finalize: bool = False
    vote: CouncilVote = "needs_revision"
    central_cruxes: list[str] = Field(default_factory=list)
    resolved_cruxes: list[str] = Field(default_factory=list)
    tool_requests: list[ProModeCalculatorRequest] = Field(default_factory=list)
    confidence: ConfidenceLevel = "medium"
    ts: float


class CouncilRound(BaseModel):
    round_index: int
    messages: list[CouncilMessage] = Field(default_factory=list)
    central_cruxes: list[str] = Field(default_factory=list)
    resolved_cruxes: list[str] = Field(default_factory=list)
    unresolved_cruxes: list[str] = Field(default_factory=list)
    agreement_snapshot: dict[str, CouncilVote] = Field(default_factory=dict)


class ConvergenceState(BaseModel):
    per_role_vote: dict[str, CouncilVote] = Field(default_factory=dict)
    central_blockers: list[str] = Field(default_factory=list)
    ready: bool = False
    consensus_level: ConsensusLevel = "medium"


class CouncilBlackboard(BaseModel):
    task_profile: ProModeTaskProfile
    context_pre_read: str = ""
    evidence_items: list[str] = Field(default_factory=list)
    calculator_results: list[ProModeCalculatorResult] = Field(default_factory=list)
    claim_map: dict[str, list[str]] = Field(default_factory=dict)
    crux_map: dict[str, list[str]] = Field(default_factory=dict)
    convergence: ConvergenceState = Field(default_factory=ConvergenceState)


class ProModeDiscussionReply(BaseModel):
    content: str
    claims: list[str] = Field(default_factory=list)
    objections: list[str] = Field(default_factory=list)
    requested_actions: list[str] = Field(default_factory=list)
    ready_to_finalize: bool = False
    vote: CouncilVote = "needs_revision"
    central_cruxes: list[str] = Field(default_factory=list)
    resolved_cruxes: list[str] = Field(default_factory=list)
    tool_requests: list[ProModeCalculatorRequest] = Field(default_factory=list)
    confidence: ConfidenceLevel = "medium"


class ProModeSynthesis(BaseModel):
    response_text: str
    settlement_summary: str
    consensus_level: ConsensusLevel = "medium"
    confidence: ConfidenceLevel = "medium"
    unresolved_points: list[str] = Field(default_factory=list)
    minority_view: str | None = None
    agreement_reached: bool = False


class ProModeVerifierReport(BaseModel):
    passed: bool = True
    issues: list[str] = Field(default_factory=list)
    suggested_changes: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = "medium"


@dataclass
class ProModeWorkflowResult:
    response_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_invocations: list[dict[str, Any]] = field(default_factory=list)
    runtime_status: str = "completed"
    runtime_error: str | None = None


class ProModeWorkflowRunner:
    """True multi-agent Pro Mode workflow with per-role agents and real message passing."""

    @staticmethod
    def _step_soft_cap_seconds(
        *,
        max_runtime_seconds: int,
        fraction: float,
        floor: int,
        ceiling: int,
    ) -> int:
        requested = int(max(1.0, float(max_runtime_seconds)) * max(0.0, float(fraction)))
        return max(floor, min(ceiling, requested))

    def __init__(
        self,
        *,
        model_builder: Callable[..., Any],
        fallback_model_builder: Callable[..., Any] | None = None,
        enable_expert_council: bool = False,
    ) -> None:
        self._model_builder = model_builder
        self._fallback_model_builder = fallback_model_builder
        self._enable_expert_council = bool(enable_expert_council)

    def _default_deep_execution_regime(self) -> ExecutionRegime:
        return "reasoning_solver"

    def _normalize_intake_decision(self, decision: ProModeIntakeDecision) -> ProModeIntakeDecision:
        normalized = decision.model_copy(deep=True)
        if normalized.route == "direct_response":
            normalized.execution_regime = "fast_dialogue"
            if normalized.task_regime == "conceptual_high_uncertainty":
                normalized.task_regime = "phatic_or_small_talk"
            if (
                normalized.context_policy.history_window != 0
                or normalized.context_policy.load_memory
            ):
                normalized.context_policy = ProModeContextPolicy(
                    load_memory=False,
                    load_knowledge=False,
                    history_window=0,
                    artifact_handles_to_expose=[],
                    compression_required=False,
                )
        elif normalized.route == "tool_workflow":
            if normalized.execution_regime not in {"validated_tool", "iterative_research"}:
                normalized.execution_regime = "validated_tool"
            if normalized.context_policy.history_window <= 0:
                normalized.context_policy.history_window = max(
                    1, int(normalized.recent_history_turns or 1)
                )
        else:
            if not self._enable_expert_council and normalized.execution_regime == "expert_council":
                normalized.execution_regime = "reasoning_solver"
            if normalized.execution_regime not in {
                "expert_council",
                "reasoning_solver",
                "proof_workflow",
                "focused_team",
                "autonomous_cycle",
            }:
                normalized.execution_regime = self._default_deep_execution_regime()
            if normalized.context_policy.history_window <= 0:
                normalized.context_policy.history_window = max(
                    1, int(normalized.recent_history_turns or 2)
                )
        normalized.load_memory = bool(normalized.context_policy.load_memory)
        normalized.load_knowledge = bool(normalized.context_policy.load_knowledge)
        normalized.recent_history_turns = int(
            normalized.context_policy.history_window or normalized.recent_history_turns or 0
        )
        return normalized

    async def intake(
        self,
        *,
        messages: list[dict[str, Any]],
        latest_user_text: str,
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        reasoning_mode: str | None,
        max_runtime_seconds: int,
        debug: bool,
    ) -> ProModeIntakeDecision:
        prompt = self._intake_prompt(messages=messages, latest_user_text=latest_user_text)
        fallback = self._fallback_intake_decision(
            messages=messages, latest_user_text=latest_user_text
        )
        decision = await self._run_structured_phase(
            role="intake",
            phase_name="intake",
            schema=ProModeIntakeDecision,
            prompt=prompt,
            fallback=fallback,
            conversation_id=conversation_id,
            run_id=run_id,
            user_id=user_id,
            reasoning_mode="fast"
            if str(reasoning_mode or "").strip().lower() == "fast"
            else "auto",
            max_runtime_seconds=max(20, min(int(max_runtime_seconds), 90)),
            debug=debug,
        )
        return self._normalize_intake_decision(decision)

    async def execute(
        self,
        *,
        messages: list[dict[str, Any]],
        latest_user_text: str,
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        reasoning_mode: str | None,
        max_runtime_seconds: int,
        event_callback: Callable[[dict[str, Any]], None] | None,
        debug: bool,
        intake_decision: ProModeIntakeDecision | None = None,
        shared_context: dict[str, Any] | None = None,
    ) -> ProModeWorkflowResult:
        effective_max_runtime_seconds = max(
            60,
            int(max_runtime_seconds)
            if int(max_runtime_seconds) != LEGACY_DEFAULT_RUNTIME_SECONDS
            else DEFAULT_PRO_MODE_MAX_RUNTIME_SECONDS,
        )
        deadline = time.monotonic() + max(60.0, float(effective_max_runtime_seconds))
        tool_invocations: list[dict[str, Any]] = []
        phase_timings: dict[str, float] = {}
        workflow_lock = asyncio.Lock()
        local_state: dict[str, Any] = {}
        phase_started: set[str] = set()
        phase_completed: set[str] = set()
        phase_counters: dict[str, int] = {}
        provided_intake = self._normalize_intake_decision(
            intake_decision
            or self._fallback_intake_decision(
                messages=messages,
                latest_user_text=latest_user_text,
            )
        )
        shared_context = dict(shared_context or {})

        discussion_step_cap_seconds = self._step_soft_cap_seconds(
            max_runtime_seconds=effective_max_runtime_seconds,
            fraction=0.25,
            floor=600,
            ceiling=1800,
        )
        synthesis_step_cap_seconds = self._step_soft_cap_seconds(
            max_runtime_seconds=effective_max_runtime_seconds,
            fraction=1.0 / 3.0,
            floor=900,
            ceiling=2400,
        )
        verifier_step_cap_seconds = self._step_soft_cap_seconds(
            max_runtime_seconds=effective_max_runtime_seconds,
            fraction=0.25,
            floor=600,
            ceiling=1800,
        )

        def remaining_seconds(soft_cap: int) -> int:
            remaining = int(max(20.0, deadline - time.monotonic()))
            return max(20, min(int(soft_cap), remaining))

        def role_label(role: ProModeRole) -> str:
            return ROLE_LABELS[role]

        def emit(payload: dict[str, Any]) -> None:
            if callable(event_callback):
                event_callback(payload)

        async def emit_phase_started(phase: str, message: str) -> None:
            async with workflow_lock:
                if phase in phase_started:
                    return
                phase_started.add(phase)
            emit(
                {
                    "kind": "graph",
                    "event_type": "pro_mode.phase_started",
                    "phase": phase,
                    "status": "started",
                    "message": message,
                }
            )

        async def emit_phase_completed(
            phase: str, message: str, payload: dict[str, Any] | None = None
        ) -> None:
            async with workflow_lock:
                if phase in phase_completed:
                    return
                phase_completed.add(phase)
            record = {
                "kind": "graph",
                "event_type": "pro_mode.phase_completed",
                "phase": phase,
                "status": "completed",
                "message": message,
            }
            if payload:
                record["payload"] = payload
            emit(record)

        async def mark_parallel_progress(
            phase: str,
            *,
            total: int,
            message: str,
            payload: dict[str, Any] | None = None,
        ) -> None:
            async with workflow_lock:
                phase_counters[phase] = int(phase_counters.get(phase, 0)) + 1
                count = phase_counters[phase]
            if count >= total:
                await emit_phase_completed(phase, message, payload)

        def state_from_input(step_input: StepInput | None = None) -> dict[str, Any]:
            if step_input is not None and step_input.workflow_session is not None:
                session_data = step_input.workflow_session.session_data or {}
                step_input.workflow_session.session_data = session_data
                pro_mode_state = session_data.setdefault("pro_mode_state", {})
                if isinstance(pro_mode_state, dict):
                    return pro_mode_state
            return local_state

        async def append_message(step_input: StepInput | None, message: CouncilMessage) -> None:
            message_phase = (
                "private_memos"
                if message.message_kind == "private_memo"
                else "socratic_review"
                if message.sender_role == ROLE_LABELS["socratic_crux_examiner"]
                and int(message.round_index) == 0
                else "tool_broker"
                if message.sender_role == ROLE_LABELS["tool_broker"]
                and int(message.round_index) == 0
                else f"critique_round_{message.round_index}"
            )
            async with workflow_lock:
                state = state_from_input(step_input)
                state.setdefault("messages", []).append(message.model_dump(mode="json"))
                state.setdefault("claim_map", {})[message.sender_role] = list(message.claims)
                state.setdefault("crux_map", {})[message.sender_role] = list(message.central_cruxes)
            emit(
                {
                    "kind": "graph",
                    "event_type": "pro_mode.role_message",
                    "phase": message_phase,
                    "status": "completed",
                    "message": f"{message.sender_role} posted a {message.message_kind.replace('_', ' ')}.",
                    "role": message.sender_role,
                    "payload": {
                        "message": message.model_dump(mode="json"),
                    },
                }
            )

        async def record_model_call(
            step_input: StepInput | None,
            *,
            role: str,
            phase: str,
            elapsed: float,
        ) -> None:
            async with workflow_lock:
                state = state_from_input(step_input)
                state["model_call_count"] = int(state.get("model_call_count", 0)) + 1
                role_stats = state.setdefault("role_stats", {})
                current = dict(role_stats.get(role) or {})
                current["calls"] = int(current.get("calls", 0)) + 1
                current["last_phase"] = phase
                current["last_elapsed_seconds"] = round(float(elapsed), 3)
                current["total_elapsed_seconds"] = round(
                    float(current.get("total_elapsed_seconds", 0.0)) + float(elapsed),
                    3,
                )
                role_stats[role] = current

        def current_messages(step_input: StepInput | None) -> list[CouncilMessage]:
            state = state_from_input(step_input)
            return [
                CouncilMessage.model_validate(item) for item in list(state.get("messages") or [])
            ]

        def current_rounds(step_input: StepInput | None) -> list[CouncilRound]:
            state = state_from_input(step_input)
            return [CouncilRound.model_validate(item) for item in list(state.get("rounds") or [])]

        def calculator_results(step_input: StepInput | None) -> list[ProModeCalculatorResult]:
            state = state_from_input(step_input)
            return [
                ProModeCalculatorResult.model_validate(item)
                for item in list(state.get("calculator_results") or [])
            ]

        def current_convergence(step_input: StepInput | None) -> ConvergenceState:
            state = state_from_input(step_input)
            return ConvergenceState.model_validate(dict(state.get("convergence") or {}))

        def task_profile(step_input: StepInput | None) -> ProModeTaskProfile:
            state = state_from_input(step_input)
            return ProModeTaskProfile.model_validate(dict(state.get("task_profile") or {}))

        def context_pre_read(step_input: StepInput | None) -> str:
            state = state_from_input(step_input)
            return str(state.get("context_pre_read") or "").strip()

        def latest_round(step_input: StepInput | None) -> CouncilRound | None:
            rounds = current_rounds(step_input)
            return rounds[-1] if rounds else None

        def latest_message_from_role(
            step_input: StepInput | None, role: ProModeRole
        ) -> CouncilMessage | None:
            target = ROLE_LABELS[role]
            for message in reversed(current_messages(step_input)):
                if message.sender_role == target:
                    return message
            return None

        async def set_round(step_input: StepInput | None, round_data: CouncilRound) -> None:
            async with workflow_lock:
                state = state_from_input(step_input)
                rounds = list(state.get("rounds") or [])
                rounds = [
                    item
                    for item in rounds
                    if int(dict(item).get("round_index") or -1) != round_data.round_index
                ]
                rounds.append(round_data.model_dump(mode="json"))
                rounds.sort(key=lambda item: int(dict(item).get("round_index") or 0))
                state["rounds"] = rounds
                state["round_count"] = len(rounds)
                state["convergence"] = self._convergence_from_messages(
                    round_data.messages
                ).model_dump(mode="json")

        async def set_calculator_results(
            step_input: StepInput | None,
            results: list[ProModeCalculatorResult],
            *,
            phase: str,
        ) -> None:
            if not results:
                return
            evidence_lines = [
                f"{item.purpose}: {item.formatted_result or item.error or 'No result'}"
                for item in results
            ]
            evidence_message = CouncilMessage(
                message_id=str(uuid4()),
                round_index=(
                    latest_round(step_input).round_index if latest_round(step_input) else 1
                ),
                sender_role=ROLE_LABELS["tool_broker"],
                recipient_roles=[ROLE_LABELS[role] for role in DISCUSSION_ROLES],
                reply_to_ids=[],
                message_kind="evidence",
                content="Calculator evidence posted to the council.",
                claims=evidence_lines,
                objections=[],
                requested_actions=[],
                ready_to_finalize=True,
                vote="agree_with_reservation",
                central_cruxes=[],
                resolved_cruxes=[item.purpose for item in results if item.success],
                tool_requests=[],
                confidence="high" if all(item.success for item in results) else "medium",
                ts=time.time(),
            )
            async with workflow_lock:
                state = state_from_input(step_input)
                state["calculator_results"] = [item.model_dump(mode="json") for item in results]
                state.setdefault("evidence_items", []).extend(evidence_lines)
            await append_message(step_input, evidence_message)
            emit(
                {
                    "kind": "graph",
                    "event_type": "pro_mode.tool_completed",
                    "phase": phase,
                    "status": "completed",
                    "message": (
                        f"Completed {len(results)} calculator check{'s' if len(results) != 1 else ''}."
                    ),
                    "payload": {
                        "calculator_results": [item.model_dump(mode="json") for item in results],
                    },
                }
            )

        async def set_synthesis(step_input: StepInput | None, synthesis: ProModeSynthesis) -> None:
            async with workflow_lock:
                state = state_from_input(step_input)
                state["synthesis"] = synthesis.model_dump(mode="json")

        async def set_verifier(step_input: StepInput | None, report: ProModeVerifierReport) -> None:
            async with workflow_lock:
                state = state_from_input(step_input)
                state["verifier"] = report.model_dump(mode="json")

        async def initialize_state(
            step_input: StepInput, profile: ProModeTaskProfile, snapshot: str
        ) -> None:
            async with workflow_lock:
                state = state_from_input(step_input)
                state["task_profile"] = profile.model_dump(mode="json")
                state["context_pre_read"] = snapshot
                state.setdefault("messages", [])
                state.setdefault("rounds", [])
                state.setdefault("calculator_results", [])
                state.setdefault("evidence_items", [])
                state.setdefault("claim_map", {})
                state.setdefault("crux_map", {})
                state.setdefault("model_call_count", 0)
                state.setdefault("role_stats", {})
                state.setdefault(
                    "phase_order",
                    list(PRO_MODE_PHASE_ORDER),
                )

        async def set_intake_decision_state(
            step_input: StepInput, decision: ProModeIntakeDecision
        ) -> None:
            async with workflow_lock:
                state = state_from_input(step_input)
                state["intake_decision"] = decision.model_dump(mode="json")
                state.setdefault("messages", [])
                state.setdefault("rounds", [])
                state.setdefault("calculator_results", [])
                state.setdefault("evidence_items", [])
                state.setdefault("claim_map", {})
                state.setdefault("crux_map", {})
                state.setdefault("model_call_count", 1)
                state.setdefault("role_stats", {})
                state.setdefault("phase_order", list(PRO_MODE_PHASE_ORDER))

        def intake_from_state(step_input: StepInput | None) -> ProModeIntakeDecision:
            state = state_from_input(step_input)
            return ProModeIntakeDecision.model_validate(dict(state.get("intake_decision") or {}))

        def role_step_name(prefix: str, role: ProModeRole) -> str:
            return f"{prefix}_{role}"

        async def intake_step(step_input: StepInput) -> StepOutput:
            await emit_phase_started("intake", "Triaging the turn before expert deliberation.")
            await initialize_state(step_input, self._profile_task(latest_user_text), "")
            await set_intake_decision_state(step_input, provided_intake)
            await emit_phase_completed(
                "intake",
                (
                    "The front-door triage answered directly."
                    if provided_intake.route == "direct_response"
                    else "The front-door triage escalated to the expert council."
                ),
                payload=provided_intake.model_dump(mode="json"),
            )
            return StepOutput(content=provided_intake.model_dump(mode="json"))

        async def direct_response(step_input: StepInput) -> StepOutput:
            await emit_phase_started(
                "direct_response", "Answering directly without expert escalation."
            )
            decision = intake_from_state(step_input)
            text = str(decision.direct_response or "").strip() or "Hello! How can I help you today?"
            synthesis_result = ProModeSynthesis(
                response_text=text,
                settlement_summary="Answered at the intake gate without invoking the expert council.",
                consensus_level="high",
                confidence="high",
                unresolved_points=[],
                minority_view=None,
                agreement_reached=True,
            )
            verifier_result = ProModeVerifierReport(
                passed=True,
                issues=[],
                suggested_changes=[],
                confidence="high",
            )
            await set_synthesis(step_input, synthesis_result)
            await set_verifier(step_input, verifier_result)
            await emit_phase_completed(
                "direct_response",
                "Returned a lightweight direct answer.",
                payload={"route": decision.route, "used_council": False},
            )
            return StepOutput(content=synthesis_result.model_dump(mode="json"))

        async def context_policy_step(step_input: StepInput) -> StepOutput:
            decision = intake_from_state(step_input)
            await emit_phase_started(
                "context_policy", "Applying the explicit Pro Mode context policy."
            )
            await emit_phase_completed(
                "context_policy",
                "Context loading policy resolved for this turn.",
                payload=decision.context_policy.model_dump(mode="json"),
            )
            return StepOutput(content=decision.context_policy.model_dump(mode="json"))

        async def execution_router_step(step_input: StepInput) -> StepOutput:
            decision = intake_from_state(step_input)
            await emit_phase_started(
                "execution_router", "Choosing the internal Pro Mode execution regime."
            )
            await emit_phase_completed(
                "execution_router",
                f"Selected the `{decision.execution_regime}` regime.",
                payload={
                    "route": decision.route,
                    "execution_regime": decision.execution_regime,
                    "task_regime": decision.task_regime,
                },
            )
            return StepOutput(
                content={
                    "route": decision.route,
                    "execution_regime": decision.execution_regime,
                    "task_regime": decision.task_regime,
                }
            )

        async def preflight(step_input: StepInput) -> StepOutput:
            await emit_phase_started("preflight", "Profiling the problem for Pro Mode.")
            profile = self._profile_task(latest_user_text)
            await initialize_state(step_input, profile, "")
            await emit_phase_completed(
                "preflight",
                "Problem profile ready.",
                payload={"task_profile": profile.model_dump(mode="json")},
            )
            return StepOutput(content=profile.model_dump(mode="json"))

        async def context_hydrate(step_input: StepInput) -> StepOutput:
            await emit_phase_started("context_hydrate", "Hydrating shared context for the council.")
            profile = task_profile(step_input)
            snapshot = self._render_context_snapshot(messages, shared_context=shared_context)
            await initialize_state(step_input, profile, snapshot)
            await emit_phase_completed(
                "context_hydrate",
                "Shared context prepared for the council.",
                payload={"context_length": len(snapshot)},
            )
            return StepOutput(content={"context_pre_read": snapshot})

        def make_private_memo_step(role: ProModeRole) -> Step:
            async def _executor(step_input: StepInput) -> StepOutput:
                await emit_phase_started(
                    "private_memos", "Exploring distinct internal perspectives."
                )
                profile = task_profile(step_input)
                snapshot = context_pre_read(step_input)
                start = time.monotonic()
                memo = await self._run_private_memo_agent(
                    role=role,
                    latest_user_text=latest_user_text,
                    context_snapshot=snapshot,
                    profile=profile,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    reasoning_mode=reasoning_mode,
                    max_runtime_seconds=remaining_seconds(discussion_step_cap_seconds),
                    debug=debug,
                )
                elapsed = time.monotonic() - start
                await record_model_call(
                    step_input, role=role_label(role), phase="private_memos", elapsed=elapsed
                )
                message = CouncilMessage(
                    message_id=str(uuid4()),
                    round_index=0,
                    sender_role=role_label(role),
                    recipient_roles=[],
                    reply_to_ids=[],
                    message_kind="private_memo",
                    content=memo.headline,
                    claims=list(memo.claims),
                    objections=[],
                    requested_actions=list(memo.open_questions),
                    ready_to_finalize=False,
                    vote="needs_revision",
                    central_cruxes=list(
                        memo.open_questions[:2]
                        if role != "socratic_crux_examiner"
                        else memo.open_questions[:3]
                    ),
                    resolved_cruxes=[],
                    tool_requests=[],
                    confidence=memo.confidence,
                    ts=time.time(),
                )
                await append_message(step_input, message)
                await mark_parallel_progress(
                    "private_memos",
                    total=len(DISCUSSION_ROLES),
                    message="Private perspectives prepared.",
                    payload={"count": len(DISCUSSION_ROLES)},
                )
                return StepOutput(content=message.model_dump(mode="json"))

            return Step(name=role_step_name("private_memo", role), executor=_executor)

        async def socratic_review(step_input: StepInput) -> StepOutput:
            await emit_phase_started("socratic_review", "Running the Socratic crux review.")
            profile = task_profile(step_input)
            private_memos = [
                message
                for message in current_messages(step_input)
                if message.message_kind == "private_memo"
            ]
            start = time.monotonic()
            reply = await self._run_discussion_agent(
                role="socratic_crux_examiner",
                round_index=0,
                latest_user_text=latest_user_text,
                profile=profile,
                context_snapshot=context_pre_read(step_input),
                memo=None,
                target_messages=private_memos,
                calculator_results=[],
                verifier_feedback=[],
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                reasoning_mode=reasoning_mode,
                max_runtime_seconds=remaining_seconds(discussion_step_cap_seconds),
                debug=debug,
            )
            elapsed = time.monotonic() - start
            await record_model_call(
                step_input,
                role=ROLE_LABELS["socratic_crux_examiner"],
                phase="socratic_review",
                elapsed=elapsed,
            )
            message = CouncilMessage(
                message_id=str(uuid4()),
                round_index=0,
                sender_role=ROLE_LABELS["socratic_crux_examiner"],
                recipient_roles=[item.sender_role for item in private_memos],
                reply_to_ids=[item.message_id for item in private_memos[:6]],
                message_kind="critique",
                content=reply.content,
                claims=list(reply.claims),
                objections=list(reply.objections),
                requested_actions=list(reply.requested_actions),
                ready_to_finalize=bool(reply.ready_to_finalize),
                vote=reply.vote,
                central_cruxes=list(reply.central_cruxes),
                resolved_cruxes=list(reply.resolved_cruxes),
                tool_requests=[],
                confidence=reply.confidence,
                ts=time.time(),
            )
            await append_message(step_input, message)
            await emit_phase_completed(
                "socratic_review",
                "Socratic crux review completed.",
                payload={"recipient_count": len(private_memos)},
            )
            return StepOutput(content=message.model_dump(mode="json"))

        def make_discussion_step(role: ProModeRole, round_index: int, retry: bool = False) -> Step:
            async def _executor(step_input: StepInput) -> StepOutput:
                phase_name = "retry_round" if retry else f"critique_round_{round_index}"
                await emit_phase_started(
                    phase_name,
                    "Running a retry discussion round."
                    if retry
                    else f"Running discussion round {round_index}.",
                )
                profile = task_profile(step_input)
                memo_message = self._private_memo_for_role(
                    messages=current_messages(step_input), role_label=role_label(role)
                )
                target_messages = self._discussion_inputs_for_role(
                    role=role,
                    round_index=round_index,
                    messages=current_messages(step_input),
                    convergence=current_convergence(step_input),
                )
                calc_results = calculator_results(step_input)
                verifier_feedback = self._verifier_issues(step_input) if retry else []
                start = time.monotonic()
                reply = await self._run_discussion_agent(
                    role=role,
                    round_index=round_index,
                    latest_user_text=latest_user_text,
                    profile=profile,
                    context_snapshot=context_pre_read(step_input),
                    memo=memo_message,
                    target_messages=target_messages,
                    calculator_results=calc_results,
                    verifier_feedback=verifier_feedback,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    reasoning_mode=reasoning_mode,
                    max_runtime_seconds=remaining_seconds(discussion_step_cap_seconds),
                    debug=debug,
                )
                elapsed = time.monotonic() - start
                await record_model_call(
                    step_input, role=role_label(role), phase=phase_name, elapsed=elapsed
                )
                message = CouncilMessage(
                    message_id=str(uuid4()),
                    round_index=round_index,
                    sender_role=role_label(role),
                    recipient_roles=self._recipient_roles_for_round(
                        role=role,
                        round_index=round_index,
                        messages=current_messages(step_input),
                    ),
                    reply_to_ids=self._reply_to_ids_for_round(
                        role=role,
                        round_index=round_index,
                        messages=current_messages(step_input),
                    ),
                    message_kind="rebuttal" if round_index > 1 else "critique",
                    content=reply.content,
                    claims=list(reply.claims),
                    objections=list(reply.objections),
                    requested_actions=list(reply.requested_actions),
                    ready_to_finalize=bool(reply.ready_to_finalize),
                    vote=reply.vote,
                    central_cruxes=list(reply.central_cruxes),
                    resolved_cruxes=list(reply.resolved_cruxes),
                    tool_requests=self._normalize_tool_requests(
                        role=role, requests=reply.tool_requests
                    ),
                    confidence=reply.confidence,
                    ts=time.time(),
                )
                await append_message(step_input, message)
                total = len(DISCUSSION_ROLES)
                completion_note = (
                    "Retry discussion round completed."
                    if retry
                    else f"Discussion round {round_index} completed."
                )
                await mark_parallel_progress(
                    phase_name,
                    total=total,
                    message=completion_note,
                    payload={"round_index": round_index, "count": total},
                )
                return StepOutput(content=message.model_dump(mode="json"))

            return Step(
                name=role_step_name(f"discussion_round_{round_index}", role), executor=_executor
            )

        async def tool_broker_review(step_input: StepInput) -> StepOutput:
            await emit_phase_started("tool_broker", "Deciding whether external evidence is needed.")
            profile = task_profile(step_input)
            target_messages = [
                *(
                    [socratic_message]
                    if (
                        socratic_message := latest_message_from_role(
                            step_input, "socratic_crux_examiner"
                        )
                    )
                    is not None
                    else []
                ),
                *self._messages_for_round(current_messages(step_input), round_index=1),
            ]
            start = time.monotonic()
            reply = await self._run_discussion_agent(
                role="tool_broker",
                round_index=0,
                latest_user_text=latest_user_text,
                profile=profile,
                context_snapshot=context_pre_read(step_input),
                memo=None,
                target_messages=target_messages,
                calculator_results=calculator_results(step_input),
                verifier_feedback=[],
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                reasoning_mode=reasoning_mode,
                max_runtime_seconds=remaining_seconds(discussion_step_cap_seconds),
                debug=debug,
            )
            elapsed = time.monotonic() - start
            await record_model_call(
                step_input,
                role=ROLE_LABELS["tool_broker"],
                phase="tool_broker",
                elapsed=elapsed,
            )
            message = CouncilMessage(
                message_id=str(uuid4()),
                round_index=0,
                sender_role=ROLE_LABELS["tool_broker"],
                recipient_roles=[item.sender_role for item in target_messages[:6]],
                reply_to_ids=[item.message_id for item in target_messages[:6]],
                message_kind="critique",
                content=reply.content,
                claims=list(reply.claims),
                objections=list(reply.objections),
                requested_actions=list(reply.requested_actions),
                ready_to_finalize=bool(reply.ready_to_finalize),
                vote=reply.vote,
                central_cruxes=list(reply.central_cruxes),
                resolved_cruxes=list(reply.resolved_cruxes),
                tool_requests=self._normalize_tool_requests(
                    role="tool_broker", requests=reply.tool_requests
                ),
                confidence=reply.confidence,
                ts=time.time(),
            )
            await append_message(step_input, message)
            await emit_phase_completed(
                "tool_broker",
                "Tool / Evidence Broker completed its action decision.",
                payload={"requested_actions": len(message.tool_requests)},
            )
            return StepOutput(content=message.model_dump(mode="json"))

        async def collect_round_one(step_input: StepInput) -> StepOutput:
            round_messages = self._messages_for_round(current_messages(step_input), round_index=1)
            round_data = self._build_round(round_index=1, messages=round_messages)
            await set_round(step_input, round_data)
            emit(
                {
                    "kind": "graph",
                    "event_type": "pro_mode.convergence_updated",
                    "phase": "critique_round_1",
                    "status": "completed",
                    "message": "Updated the council convergence state.",
                    "payload": {
                        "convergence": current_convergence(step_input).model_dump(mode="json"),
                    },
                }
            )
            return StepOutput(content=round_data.model_dump(mode="json"))

        async def calculator_evidence(step_input: StepInput) -> StepOutput:
            await emit_phase_started("calculator_evidence", "Applying numeric checks where needed.")
            requests = self._tool_requests_from_broker_message(
                latest_message_from_role(step_input, "tool_broker")
            )
            if not requests:
                await emit_phase_completed(
                    "calculator_evidence",
                    "No numeric check was needed.",
                    payload={"calculator_used": False},
                )
                return StepOutput(content={"results": []})
            emit(
                {
                    "kind": "graph",
                    "event_type": "pro_mode.tool_requested",
                    "phase": "calculator_evidence",
                    "status": "started",
                    "message": f"Tool Broker requested {len(requests)} calculator check{'s' if len(requests) != 1 else ''}.",
                    "payload": {"requests": [item.model_dump(mode="json") for item in requests]},
                }
            )
            results = self._execute_calculator_requests(
                requests=requests,
                tool_invocations=tool_invocations,
                event_callback=event_callback,
            )
            await set_calculator_results(step_input, results, phase="calculator_evidence")
            await emit_phase_completed(
                "calculator_evidence",
                (
                    f"Completed {len(results)} numeric check{'s' if len(results) != 1 else ''}."
                    if results
                    else "No numeric check was needed."
                ),
                payload={
                    "calculator_used": bool(results),
                    "results": [item.model_dump(mode="json") for item in results],
                },
            )
            return StepOutput(
                content={"results": [item.model_dump(mode="json") for item in results]}
            )

        def should_run_round_two(step_input: StepInput) -> bool:
            round_payload = step_input.get_step_content("collect_round_one")
            round_data = (
                CouncilRound.model_validate(dict(round_payload or {}))
                if isinstance(round_payload, dict)
                else latest_round(step_input)
            )
            calculator_payload = step_input.get_step_content("calculator_evidence")
            calculator_count = 0
            if isinstance(calculator_payload, dict):
                calculator_count = len(list(calculator_payload.get("results") or []))
            return bool(round_data and round_data.unresolved_cruxes) or calculator_count > 0

        async def collect_round_two(step_input: StepInput) -> StepOutput:
            round_messages = self._messages_for_round(current_messages(step_input), round_index=2)
            if not round_messages:
                return StepOutput(content={})
            round_data = self._build_round(round_index=2, messages=round_messages)
            await set_round(step_input, round_data)
            emit(
                {
                    "kind": "graph",
                    "event_type": "pro_mode.convergence_updated",
                    "phase": "critique_round_2",
                    "status": "completed",
                    "message": "Updated the council convergence state after round 2.",
                    "payload": {
                        "convergence": current_convergence(step_input).model_dump(mode="json"),
                    },
                }
            )
            return StepOutput(content=round_data.model_dump(mode="json"))

        async def synthesis(step_input: StepInput) -> StepOutput:
            await emit_phase_started("synthesis", "Synthesizing the council result.")
            profile = task_profile(step_input)
            convergence = current_convergence(step_input)
            start = time.monotonic()
            synthesis_result = await self._run_synthesizer(
                latest_user_text=latest_user_text,
                profile=profile,
                blackboard=self._blackboard_from_state(step_input),
                round_data=latest_round(step_input),
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                reasoning_mode=reasoning_mode,
                max_runtime_seconds=remaining_seconds(synthesis_step_cap_seconds),
                debug=debug,
                verifier_feedback=[],
            )
            elapsed = time.monotonic() - start
            await record_model_call(
                step_input, role=ROLE_LABELS["synthesizer"], phase="synthesis", elapsed=elapsed
            )
            await set_synthesis(step_input, synthesis_result)
            await emit_phase_completed(
                "synthesis",
                "Synthesizer prepared the final draft.",
                payload={
                    "agreement_reached": synthesis_result.agreement_reached or convergence.ready
                },
            )
            return StepOutput(content=synthesis_result.model_dump(mode="json"))

        async def verifier(step_input: StepInput) -> StepOutput:
            await emit_phase_started("verifier", "Verifying the council draft.")
            synthesis_payload = ProModeSynthesis.model_validate(
                dict(state_from_input(step_input).get("synthesis") or {})
            )
            start = time.monotonic()
            report = await self._run_verifier(
                latest_user_text=latest_user_text,
                synthesis=synthesis_payload,
                blackboard=self._blackboard_from_state(step_input),
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                reasoning_mode=reasoning_mode,
                max_runtime_seconds=remaining_seconds(verifier_step_cap_seconds),
                debug=debug,
            )
            elapsed = time.monotonic() - start
            await record_model_call(step_input, role="Verifier", phase="verifier", elapsed=elapsed)
            await set_verifier(step_input, report)
            emit(
                {
                    "kind": "graph",
                    "event_type": "pro_mode.verifier_result",
                    "phase": "verifier",
                    "status": "completed",
                    "message": "Verifier completed.",
                    "payload": report.model_dump(mode="json"),
                }
            )
            await emit_phase_completed(
                "verifier",
                "Verification completed.",
                payload={"passed": report.passed},
            )
            return StepOutput(content=report.model_dump(mode="json"))

        def retry_needed(step_input: StepInput) -> bool:
            verifier_payload = step_input.get_step_content("verifier")
            if isinstance(verifier_payload, dict):
                report = ProModeVerifierReport.model_validate(verifier_payload)
            else:
                report = ProModeVerifierReport.model_validate(
                    dict(state_from_input(step_input).get("verifier") or {})
                )
            return not report.passed

        async def collect_retry_round(step_input: StepInput) -> StepOutput:
            round_messages = self._messages_for_round(current_messages(step_input), round_index=3)
            if not round_messages:
                return StepOutput(content={})
            round_data = self._build_round(round_index=3, messages=round_messages)
            await set_round(step_input, round_data)
            emit(
                {
                    "kind": "graph",
                    "event_type": "pro_mode.convergence_updated",
                    "phase": "retry_round",
                    "status": "completed",
                    "message": "Updated the council convergence state after the retry round.",
                    "payload": {
                        "convergence": current_convergence(step_input).model_dump(mode="json"),
                    },
                }
            )
            return StepOutput(content=round_data.model_dump(mode="json"))

        async def calculator_evidence_retry(step_input: StepInput) -> StepOutput:
            return StepOutput(content={"results": []})

        async def synthesis_retry(step_input: StepInput) -> StepOutput:
            profile = task_profile(step_input)
            feedback = self._verifier_issues(step_input)
            start = time.monotonic()
            synthesis_result = await self._run_synthesizer(
                latest_user_text=latest_user_text,
                profile=profile,
                blackboard=self._blackboard_from_state(step_input),
                round_data=latest_round(step_input),
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                reasoning_mode=reasoning_mode,
                max_runtime_seconds=remaining_seconds(synthesis_step_cap_seconds),
                debug=debug,
                verifier_feedback=feedback,
            )
            elapsed = time.monotonic() - start
            await record_model_call(
                step_input, role=ROLE_LABELS["synthesizer"], phase="retry_round", elapsed=elapsed
            )
            await set_synthesis(step_input, synthesis_result)
            return StepOutput(content=synthesis_result.model_dump(mode="json"))

        async def verifier_retry(step_input: StepInput) -> StepOutput:
            synthesis_payload = ProModeSynthesis.model_validate(
                dict(state_from_input(step_input).get("synthesis") or {})
            )
            start = time.monotonic()
            report = await self._run_verifier(
                latest_user_text=latest_user_text,
                synthesis=synthesis_payload,
                blackboard=self._blackboard_from_state(step_input),
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                reasoning_mode=reasoning_mode,
                max_runtime_seconds=remaining_seconds(verifier_step_cap_seconds),
                debug=debug,
            )
            elapsed = time.monotonic() - start
            await record_model_call(
                step_input, role="Verifier", phase="retry_round", elapsed=elapsed
            )
            await set_verifier(step_input, report)
            emit(
                {
                    "kind": "graph",
                    "event_type": "pro_mode.verifier_result",
                    "phase": "retry_round",
                    "status": "completed",
                    "message": "Verifier re-check completed.",
                    "payload": report.model_dump(mode="json"),
                }
            )
            await emit_phase_completed(
                "retry_round",
                "Retry round completed.",
                payload={"passed": report.passed},
            )
            return StepOutput(content=report.model_dump(mode="json"))

        async def finalize(step_input: StepInput) -> StepOutput:
            await emit_phase_started("finalize", "Preparing the final Pro Mode answer.")
            state = state_from_input(step_input)
            synthesis_payload = ProModeSynthesis.model_validate(dict(state.get("synthesis") or {}))
            verifier_payload = ProModeVerifierReport.model_validate(
                dict(state.get("verifier") or {})
            )
            finalized = self._finalize_payload(
                synthesis=synthesis_payload,
                verifier_report=verifier_payload,
                phase_timings=phase_timings,
                tool_invocations=tool_invocations,
                active_roles=(
                    ["Front Door Triage"]
                    if intake_from_state(step_input).route == "direct_response"
                    else [ROLE_LABELS[role] for role in ALL_COUNCIL_ROLES]
                ),
                round_count=int(state.get("round_count") or 0),
                blackboard=self._blackboard_from_state(step_input),
                messages=current_messages(step_input),
                rounds=current_rounds(step_input),
                model_call_count=int(state.get("model_call_count") or 0),
                role_stats=dict(state.get("role_stats") or {}),
                debug=debug,
                intake_decision=intake_from_state(step_input),
            )
            await emit_phase_completed(
                "finalize",
                "Final answer prepared.",
                payload={"round_count": finalized["metadata"]["pro_mode"]["round_count"]},
            )
            return StepOutput(content=finalized)

        def timed_step(name: str, executor: Callable[[StepInput], Any]) -> Step:
            async def _wrapped(step_input: StepInput) -> StepOutput:
                start = time.monotonic()
                output = await executor(step_input)
                phase_timings[name] = round(time.monotonic() - start, 3)
                return output

            return Step(name=name, executor=_wrapped)

        direct_branch = Steps(
            name="direct_response_branch",
            steps=[timed_step("direct_response", direct_response)],
        )
        deep_branch = Steps(
            name="deep_reasoning_branch",
            steps=[
                timed_step("preflight", preflight),
                timed_step("context_hydrate", context_hydrate),
                Parallel(
                    "private_memos",
                    *[make_private_memo_step(role) for role in PRIVATE_FIRST_ROLES],
                ),
                timed_step("socratic_review", socratic_review),
                Parallel(
                    "critique_round_1",
                    *[make_discussion_step(role, round_index=1) for role in DISCUSSION_ROLES],
                ),
                timed_step("collect_round_one", collect_round_one),
                timed_step("tool_broker", tool_broker_review),
                timed_step("calculator_evidence", calculator_evidence),
                Condition(
                    name="maybe_critique_round_2",
                    evaluator=should_run_round_two,
                    steps=[
                        Parallel(
                            "critique_round_2",
                            *[
                                make_discussion_step(role, round_index=2)
                                for role in DISCUSSION_ROLES
                            ],
                        ),
                        timed_step("collect_round_two", collect_round_two),
                    ],
                    else_steps=[],
                ),
                timed_step("synthesis", synthesis),
                timed_step("verifier", verifier),
                Condition(
                    name="maybe_retry_round",
                    evaluator=retry_needed,
                    steps=[
                        Parallel(
                            "retry_round",
                            *[
                                make_discussion_step(role, round_index=3, retry=True)
                                for role in DISCUSSION_ROLES
                            ],
                        ),
                        timed_step("collect_retry_round", collect_retry_round),
                        timed_step("calculator_evidence_retry", calculator_evidence_retry),
                        timed_step("synthesis_retry", synthesis_retry),
                        timed_step("verifier_retry", verifier_retry),
                    ],
                    else_steps=[],
                ),
            ],
        )

        def execution_route_selector(
            step_input: StepInput, session_state: dict[str, Any] | None = None
        ):
            payload = step_input.get_step_content("intake")
            if isinstance(payload, dict) and payload:
                decision = ProModeIntakeDecision.model_validate(payload)
            else:
                pro_mode_state = dict((session_state or {}).get("pro_mode_state") or {})
                decision = ProModeIntakeDecision.model_validate(
                    dict(pro_mode_state.get("intake_decision") or {})
                )
            return direct_branch if decision.route == "direct_response" else deep_branch

        workflow = Workflow(
            name="pro_mode_workflow",
            description="True multi-agent scientific deliberation workflow.",
            steps=[
                timed_step("intake", intake_step),
                timed_step("context_policy", context_policy_step),
                timed_step("execution_router", execution_router_step),
                Router(
                    name="execution_route",
                    selector=execution_route_selector,
                    choices=[direct_branch, deep_branch],
                ),
                timed_step("finalize", finalize),
            ],
            stream=False,
            telemetry=False,
            store_events=False,
            debug_mode=bool(debug),
            session_state={"pro_mode_state": {}},
            metadata={"mode": "pro_mode", "phase_order": list(PRO_MODE_PHASE_ORDER)},
        )

        try:
            workflow_output = await workflow.arun(
                input={"messages": messages, "user_text": latest_user_text},
                user_id=user_id,
                run_id=run_id,
                session_id=self._phase_session_token(
                    conversation_id=conversation_id,
                    run_id=run_id,
                    phase_name="workflow",
                ),
            )
            content = dict(getattr(workflow_output, "content", {}) or {})
            return ProModeWorkflowResult(
                response_text=str(content.get("response_text") or "").strip(),
                metadata=dict(content.get("metadata") or {}),
                tool_invocations=list(content.get("tool_invocations") or tool_invocations),
                runtime_status="completed",
                runtime_error=None,
            )
        except Exception as exc:
            return ProModeWorkflowResult(
                response_text=(
                    "Pro Mode could not complete a stable internal deliberation pass. "
                    "Please retry or use the default scientist mode."
                ),
                metadata={
                    "pro_mode": {
                        "active_roles": [ROLE_LABELS[role] for role in ALL_COUNCIL_ROLES],
                        "phase_order": list(PRO_MODE_PHASE_ORDER),
                        "summary": "Internal deliberation did not complete cleanly.",
                    },
                    "debug": {
                        "path": "pro_mode",
                        "agent_mode": "workflow",
                        "prompt_profile": "pro_mode",
                    },
                },
                tool_invocations=tool_invocations,
                runtime_status="error",
                runtime_error=str(exc or exc.__class__.__name__),
            )

    async def _run_private_memo_agent(
        self,
        *,
        role: ProModeRole,
        latest_user_text: str,
        context_snapshot: str,
        profile: ProModeTaskProfile,
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        reasoning_mode: str | None,
        max_runtime_seconds: int,
        debug: bool,
    ) -> ProModeRoleMemo:
        prompt = self._private_memo_prompt(
            role=role,
            latest_user_text=latest_user_text,
            context_snapshot=context_snapshot,
            profile=profile,
        )
        fallback = self._fallback_private_memo(role=role, latest_user_text=latest_user_text)
        return await self._run_structured_phase(
            role=role,
            phase_name="private_memos",
            schema=ProModeRoleMemo,
            prompt=prompt,
            fallback=fallback,
            conversation_id=conversation_id,
            run_id=run_id,
            user_id=user_id,
            reasoning_mode=reasoning_mode,
            max_runtime_seconds=max_runtime_seconds,
            debug=debug,
        )

    async def _run_discussion_agent(
        self,
        *,
        role: ProModeRole,
        round_index: int,
        latest_user_text: str,
        profile: ProModeTaskProfile,
        context_snapshot: str,
        memo: CouncilMessage | None,
        target_messages: list[CouncilMessage],
        calculator_results: list[ProModeCalculatorResult],
        verifier_feedback: list[str],
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        reasoning_mode: str | None,
        max_runtime_seconds: int,
        debug: bool,
    ) -> ProModeDiscussionReply:
        prompt = self._discussion_prompt(
            role=role,
            round_index=round_index,
            latest_user_text=latest_user_text,
            profile=profile,
            context_snapshot=context_snapshot,
            memo=memo,
            target_messages=target_messages,
            calculator_results=calculator_results,
            verifier_feedback=verifier_feedback,
        )
        fallback = self._fallback_discussion_reply(
            role=role, round_index=round_index, target_messages=target_messages
        )
        return await self._run_structured_phase(
            role=role,
            phase_name=f"critique_round_{round_index}" if round_index < 3 else "retry_round",
            schema=ProModeDiscussionReply,
            prompt=prompt,
            fallback=fallback,
            conversation_id=conversation_id,
            run_id=run_id,
            user_id=user_id,
            reasoning_mode=reasoning_mode,
            max_runtime_seconds=max_runtime_seconds,
            debug=debug,
        )

    async def _run_synthesizer(
        self,
        *,
        latest_user_text: str,
        profile: ProModeTaskProfile,
        blackboard: CouncilBlackboard,
        round_data: CouncilRound | None,
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        reasoning_mode: str | None,
        max_runtime_seconds: int,
        debug: bool,
        verifier_feedback: list[str],
    ) -> ProModeSynthesis:
        prompt = self._synthesis_prompt(
            latest_user_text=latest_user_text,
            profile=profile,
            blackboard=blackboard,
            round_data=round_data,
            verifier_feedback=verifier_feedback,
        )
        fallback = self._fallback_synthesis(
            latest_user_text=latest_user_text, blackboard=blackboard
        )
        return await self._run_structured_phase(
            role="synthesizer",
            phase_name="synthesis" if not verifier_feedback else "retry_round",
            schema=ProModeSynthesis,
            prompt=prompt,
            fallback=fallback,
            conversation_id=conversation_id,
            run_id=run_id,
            user_id=user_id,
            reasoning_mode=("deep" if verifier_feedback else reasoning_mode),
            max_runtime_seconds=max_runtime_seconds,
            debug=debug,
        )

    async def _run_verifier(
        self,
        *,
        latest_user_text: str,
        synthesis: ProModeSynthesis,
        blackboard: CouncilBlackboard,
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        reasoning_mode: str | None,
        max_runtime_seconds: int,
        debug: bool,
    ) -> ProModeVerifierReport:
        prompt = self._verifier_prompt(
            latest_user_text=latest_user_text,
            synthesis=synthesis,
            blackboard=blackboard,
        )
        fallback = ProModeVerifierReport(
            passed=bool(synthesis.agreement_reached),
            issues=list(blackboard.convergence.central_blockers[:3]),
            suggested_changes=list(synthesis.unresolved_points[:3]),
            confidence=synthesis.confidence,
        )
        return await self._run_structured_phase(
            role="verifier",
            phase_name="verifier",
            schema=ProModeVerifierReport,
            prompt=prompt,
            fallback=fallback,
            conversation_id=conversation_id,
            run_id=run_id,
            user_id=user_id,
            reasoning_mode="deep"
            if str(reasoning_mode or "").strip().lower() != "fast"
            else "fast",
            max_runtime_seconds=max_runtime_seconds,
            debug=debug,
        )

    async def _run_structured_phase(
        self,
        *,
        role: str,
        phase_name: str,
        schema: type[BaseModel],
        prompt: str,
        fallback: BaseModel,
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        reasoning_mode: str | None,
        max_runtime_seconds: int,
        debug: bool,
    ) -> Any:
        normalized_role = str(role).strip().lower().replace(" ", "_")
        response_reasoning_mode = self._role_reasoning_mode(
            phase_name=phase_name,
            role=normalized_role,
            requested=reasoning_mode,
        )
        response_reasoning_effort = self._role_reasoning_effort_override(
            phase_name=phase_name,
            role=normalized_role,
            requested=reasoning_mode,
        )
        reasoning_bounds = self._role_reasoning_step_bounds(
            phase_name=phase_name,
            role=normalized_role,
            requested=reasoning_mode,
        )

        def _build_phase_agent(model_builder: Callable[..., Any]) -> Agent:
            agent_kwargs: dict[str, Any] = {
                "name": f"pro-mode-{phase_name}-{normalized_role}",
                "model": model_builder(
                    reasoning_mode=response_reasoning_mode,
                    reasoning_effort_override=response_reasoning_effort,
                    max_runtime_seconds=max_runtime_seconds,
                ),
                "instructions": [
                    "You are participating in Pro Mode, a structured scientific council workflow.",
                    "Return only structured content matching the requested schema.",
                    "Be precise, grounded, and concise.",
                    "Do not reveal hidden chain-of-thought or workflow commentary.",
                ],
                "output_schema": schema,
                "structured_outputs": True,
                "use_json_mode": True,
                "parse_response": True,
                "markdown": False,
                "telemetry": False,
                "retries": 0,
                "store_events": False,
                "store_history_messages": False,
                "add_datetime_to_context": False,
                "add_location_to_context": False,
                "debug_mode": bool(debug),
            }
            if reasoning_bounds is not None:
                min_steps, max_steps = reasoning_bounds
                agent_kwargs.update(
                    {
                        "reasoning_model": model_builder(
                            reasoning_mode="deep",
                            reasoning_effort_override="high",
                            max_runtime_seconds=max_runtime_seconds,
                        ),
                        "reasoning": True,
                        "reasoning_min_steps": min_steps,
                        "reasoning_max_steps": max_steps,
                    }
                )
            return Agent(**agent_kwargs)

        agent = _build_phase_agent(self._model_builder)
        try:
            result = await agent.arun(
                prompt,
                stream=False,
                user_id=user_id,
                session_id=self._phase_session_token(
                    conversation_id=conversation_id,
                    run_id=run_id,
                    phase_name=f"{phase_name}:{str(role).strip().lower().replace(' ', '_')}",
                ),
                debug_mode=bool(debug),
            )
        except Exception:
            if self._fallback_model_builder is None:
                return fallback
            try:
                fallback_result = await _build_phase_agent(self._fallback_model_builder).arun(
                    prompt,
                    stream=False,
                    user_id=user_id,
                    session_id=self._phase_session_token(
                        conversation_id=conversation_id,
                        run_id=run_id,
                        phase_name=f"{phase_name}:{str(role).strip().lower().replace(' ', '_')}:fallback",
                    ),
                    debug_mode=bool(debug),
                )
            except Exception:
                return fallback
            return self._coerce_schema_output(
                schema=schema, result=fallback_result, fallback=fallback
            )
        return self._coerce_schema_output(schema=schema, result=result, fallback=fallback)

    @staticmethod
    def _coerce_schema_output(*, schema: type[BaseModel], result: Any, fallback: BaseModel) -> Any:
        for candidate_name in ("content", "parsed", "final_output"):
            candidate = getattr(result, candidate_name, None)
            if isinstance(candidate, schema):
                return candidate
            if isinstance(candidate, BaseModel):
                try:
                    return schema.model_validate(candidate.model_dump(mode="json"))
                except Exception:
                    continue
            if isinstance(candidate, dict):
                try:
                    return schema.model_validate(candidate)
                except Exception:
                    continue
        text_candidates: list[str] = []
        for candidate_name in ("content", "final_output", "parsed"):
            candidate = getattr(result, candidate_name, None)
            if isinstance(candidate, str) and candidate.strip():
                text_candidates.append(candidate)
        if isinstance(result, str) and result.strip():
            text_candidates.append(result)
        for text in text_candidates:
            try:
                return schema.model_validate_json(text)
            except Exception:
                try:
                    return schema.model_validate(json.loads(text))
                except Exception:
                    continue
        return fallback

    @staticmethod
    def _phase_session_token(
        *, conversation_id: str | None, run_id: str | None, phase_name: str
    ) -> str:
        root = str(conversation_id or "").strip() or str(run_id or "").strip() or "ephemeral"
        return f"{root}::pro::{phase_name}"

    @staticmethod
    def _role_reasoning_mode(*, phase_name: str, role: str, requested: str | None) -> str:
        normalized = str(requested or "auto").strip().lower() or "auto"
        if phase_name == "intake" or role == "intake":
            return "fast"
        if normalized == "fast":
            return "fast"
        if role == "verifier":
            return "deep"
        return normalized if normalized in {"auto", "deep"} else "auto"

    @staticmethod
    def _role_reasoning_effort_override(
        *, phase_name: str, role: str, requested: str | None
    ) -> str | None:
        normalized = str(requested or "auto").strip().lower() or "auto"
        if phase_name == "intake" or role == "intake":
            return "low"
        if normalized == "fast":
            return None
        if (
            phase_name == "private_memos"
            or phase_name == "socratic_review"
            or phase_name == "synthesis"
            or phase_name == "verifier"
            or phase_name == "retry_round"
            or phase_name.startswith("critique_round_")
        ):
            return "high"
        if role in {"formalist", "contrarian", "tool_broker", "verifier"}:
            return "high"
        return None

    @classmethod
    def _role_reasoning_step_bounds(
        cls,
        *,
        phase_name: str,
        role: str,
        requested: str | None,
    ) -> tuple[int, int] | None:
        normalized = str(requested or "auto").strip().lower() or "auto"
        if phase_name == "intake" or role == "intake" or normalized == "fast":
            return None
        if phase_name in {"synthesis", "verifier", "retry_round"}:
            return (3, 8)
        if phase_name.startswith("critique_round_"):
            return (2, 6)
        if phase_name in {"private_memos", "socratic_review", "tool_broker"}:
            return (2, 5)
        return (2, 4)

    @staticmethod
    def _render_context_snapshot(
        messages: list[dict[str, Any]],
        *,
        shared_context: dict[str, Any] | None = None,
    ) -> str:
        lines: list[str] = []
        for raw in messages[-6:]:
            role = str(raw.get("role") or "user").strip().lower()
            content = str(raw.get("content") or "").strip()
            if not content:
                continue
            lines.append(f"{role.upper()}: {content}")
        context_payload = dict(shared_context or {})
        extra_sections: list[str] = []
        for field_name, heading in (
            ("memory_messages", "Retrieved memory"),
            ("knowledge_messages", "Retrieved knowledge"),
        ):
            rendered_blocks: list[str] = []
            for raw in list(context_payload.get(field_name) or []):
                if isinstance(raw, dict):
                    body = str(raw.get("content") or "").strip()
                else:
                    body = str(raw or "").strip()
                if body:
                    rendered_blocks.append(body)
            if rendered_blocks:
                extra_sections.append(f"{heading}:\n" + "\n\n".join(rendered_blocks))
        return "\n\n".join([*lines, *extra_sections]).strip()[:8000]

    @staticmethod
    def _profile_task(user_text: str) -> ProModeTaskProfile:
        text = str(user_text or "").strip()
        lowered = text.lower()
        report_requested = bool(
            re.search(
                r"(report|write up|write a report|research summary|survey|overview|backgrounder|deep dive|primer)",
                lowered,
            )
        )
        calculator_candidate = bool(
            re.search(
                r"(calculate|compute|derive|sum|integral|amplitude|nmr|energy|rate|phase|probability|count|evaluate|wave number|dipole|scattering)",
                lowered,
            )
        )
        exactness_priority = bool(
            re.search(
                r"(exact|distinct|how many|count|calculate|which product|identify|derive|balance)",
                lowered,
            )
        )
        if calculator_candidate and re.search(r"(product|reaction|mechanism|identify)", lowered):
            task_type: TaskType = "mechanistic"
        elif calculator_candidate:
            task_type = "quantitative"
        elif re.search(r"(why|mechanism|pathway|cause|boiling|binding|folding)", lowered):
            task_type = "mechanistic"
        elif re.search(r"(is |are |explain|compare|difference|define)", lowered):
            task_type = "conceptual"
        else:
            task_type = "mixed"
        notes: list[str] = []
        if "brief" in lowered:
            notes.append("Favor a concise final explanation.")
        if exactness_priority:
            notes.append("Prioritize exact conventions and explicit caveats.")
        if report_requested:
            notes.append(
                "Deliver a substantive report with clear takeaways, core taxonomy, tradeoffs, limitations, and current directions."
            )
        success_criteria = [
            "Answer the user's question directly.",
            "Preserve scientific correctness and uncertainty.",
            "Do not leak internal council details.",
        ]
        if report_requested:
            success_criteria.extend(
                [
                    "Organize the answer like a professional report rather than a shallow overview.",
                    "Cover the core landscape, important distinctions, and material limitations.",
                ]
            )
        return ProModeTaskProfile(
            objective=text[:240] or "Answer the scientific question.",
            task_type=task_type,
            success_criteria=success_criteria,
            calculator_candidate=calculator_candidate,
            exactness_priority=exactness_priority,
            report_requested=report_requested,
            notes=notes,
        )

    @staticmethod
    def _messages_for_round(
        messages: list[CouncilMessage], *, round_index: int
    ) -> list[CouncilMessage]:
        return [
            message
            for message in messages
            if int(message.round_index) == round_index and message.message_kind != "private_memo"
        ]

    @staticmethod
    def _private_memo_for_role(
        messages: list[CouncilMessage], *, role_label: str
    ) -> CouncilMessage | None:
        for message in reversed(messages):
            if message.message_kind == "private_memo" and message.sender_role == role_label:
                return message
        return None

    @staticmethod
    def _normalize_tool_requests(
        *,
        role: ProModeRole,
        requests: list[ProModeCalculatorRequest],
    ) -> list[ProModeCalculatorRequest]:
        if role != "tool_broker":
            return []
        normalized: list[ProModeCalculatorRequest] = []
        for request in list(requests or [])[:4]:
            expression = str(request.expression or "").strip()
            purpose = str(request.purpose or "").strip()
            if not expression or not purpose:
                continue
            normalized.append(
                ProModeCalculatorRequest(
                    purpose=purpose,
                    expression=expression,
                    variables=dict(request.variables or {}),
                )
            )
        return normalized

    def _discussion_inputs_for_role(
        self,
        *,
        role: ProModeRole,
        round_index: int,
        messages: list[CouncilMessage],
        convergence: ConvergenceState,
    ) -> list[CouncilMessage]:
        if round_index == 1:
            targets = {ROLE_LABELS[target] for target in ROUND_ONE_TARGETS.get(role, ())}
            private_targets = [
                message
                for message in messages
                if message.message_kind == "private_memo" and message.sender_role in targets
            ]
            socratic_message = next(
                (
                    message
                    for message in reversed(messages)
                    if message.sender_role == ROLE_LABELS["socratic_crux_examiner"]
                    and int(message.round_index) == 0
                ),
                None,
            )
            return ([socratic_message] if socratic_message is not None else []) + private_targets[
                :4
            ]
        own_label = ROLE_LABELS[role]
        direct = [
            message
            for message in messages
            if own_label in message.recipient_roles and message.round_index == round_index - 1
        ]
        if direct:
            return direct[:4]
        unresolved = set(convergence.central_blockers)
        if not unresolved:
            return [message for message in messages if message.round_index == round_index - 1][-4:]
        return [
            message
            for message in messages
            if message.round_index == round_index - 1
            and any(crux in unresolved for crux in message.central_cruxes)
        ][:4]

    def _recipient_roles_for_round(
        self,
        *,
        role: ProModeRole,
        round_index: int,
        messages: list[CouncilMessage],
    ) -> list[str]:
        if round_index == 1:
            peer_targets = [ROLE_LABELS[target] for target in ROUND_ONE_TARGETS.get(role, ())]
            return [ROLE_LABELS["socratic_crux_examiner"], *peer_targets]
        own_label = ROLE_LABELS[role]
        addressed_by = {
            message.sender_role
            for message in messages
            if own_label in message.recipient_roles and message.round_index == round_index - 1
        }
        return (
            sorted(addressed_by)
            if addressed_by
            else [ROLE_LABELS[target] for target in ROUND_ONE_TARGETS.get(role, ())]
        )

    @staticmethod
    def _reply_to_ids_for_round(
        *, role: ProModeRole, round_index: int, messages: list[CouncilMessage]
    ) -> list[str]:
        if round_index == 1:
            targets = {ROLE_LABELS[target] for target in ROUND_ONE_TARGETS.get(role, ())}
            reply_ids = [
                message.message_id
                for message in messages
                if message.message_kind == "private_memo" and message.sender_role in targets
            ][:4]
            socratic_message = next(
                (
                    message
                    for message in reversed(messages)
                    if message.sender_role == ROLE_LABELS["socratic_crux_examiner"]
                    and int(message.round_index) == 0
                ),
                None,
            )
            return (
                [socratic_message.message_id] if socratic_message is not None else []
            ) + reply_ids
        own_label = ROLE_LABELS[role]
        return [
            message.message_id
            for message in messages
            if own_label in message.recipient_roles and message.round_index == round_index - 1
        ][:4]

    @staticmethod
    def _tool_requests_from_broker_message(
        message: CouncilMessage | None,
    ) -> list[ProModeCalculatorRequest]:
        if message is None or message.sender_role != ROLE_LABELS["tool_broker"]:
            return []
        return list(message.tool_requests or [])[:4]

    @staticmethod
    def _build_round(*, round_index: int, messages: list[CouncilMessage]) -> CouncilRound:
        convergence = ProModeWorkflowRunner._convergence_from_messages(messages)
        resolved = ProModeWorkflowRunner._unique_text(
            item for message in messages for item in message.resolved_cruxes
        )
        unresolved = convergence.central_blockers
        return CouncilRound(
            round_index=round_index,
            messages=messages,
            central_cruxes=unresolved,
            resolved_cruxes=resolved,
            unresolved_cruxes=unresolved,
            agreement_snapshot=convergence.per_role_vote,
        )

    @staticmethod
    def _convergence_from_messages(messages: list[CouncilMessage]) -> ConvergenceState:
        votes: dict[str, CouncilVote] = {}
        blockers = ProModeWorkflowRunner._unique_text(
            item
            for message in messages
            for item in message.central_cruxes
            if str(item or "").strip()
        )
        for message in messages:
            current = message.vote
            sender = message.sender_role
            previous = votes.get(sender)
            if previous == "needs_revision":
                continue
            if current == "needs_revision":
                votes[sender] = current
                continue
            if previous is None or previous == "agree":
                votes[sender] = current
        ready = (
            bool(votes)
            and all(vote in {"agree", "agree_with_reservation"} for vote in votes.values())
            and not blockers
        )
        consensus_level: ConsensusLevel = "low"
        if ready and all(vote == "agree" for vote in votes.values()):
            consensus_level = "high"
        elif ready:
            consensus_level = "medium"
        elif votes and sum(1 for vote in votes.values() if vote != "needs_revision") >= max(
            1, len(votes) - 2
        ):
            consensus_level = "medium"
        return ConvergenceState(
            per_role_vote=votes,
            central_blockers=blockers,
            ready=ready,
            consensus_level=consensus_level,
        )

    @staticmethod
    def _unique_text(items: Any) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for raw in items:
            token = " ".join(str(raw or "").strip().split())
            if not token:
                continue
            if token.lower() in seen:
                continue
            seen.add(token.lower())
            ordered.append(token)
        return ordered

    def _blackboard_from_state(self, step_input: StepInput | None) -> CouncilBlackboard:
        state = {}
        if step_input is not None and step_input.workflow_session is not None:
            state = dict(
                (step_input.workflow_session.session_data or {}).get("pro_mode_state") or {}
            )
        profile = ProModeTaskProfile.model_validate(dict(state.get("task_profile") or {}))
        messages = [
            CouncilMessage.model_validate(item) for item in list(state.get("messages") or [])
        ]
        claim_map = dict(state.get("claim_map") or {})
        crux_map = dict(state.get("crux_map") or {})
        evidence_items = list(state.get("evidence_items") or [])
        calc_results = [
            ProModeCalculatorResult.model_validate(item)
            for item in list(state.get("calculator_results") or [])
        ]
        convergence = self._convergence_from_messages(
            [message for message in messages if message.message_kind != "private_memo"]
        )
        return CouncilBlackboard(
            task_profile=profile,
            context_pre_read=str(state.get("context_pre_read") or "").strip(),
            evidence_items=evidence_items,
            calculator_results=calc_results,
            claim_map=claim_map,
            crux_map=crux_map,
            convergence=convergence,
        )

    def _verifier_issues(self, step_input: StepInput | None) -> list[str]:
        if step_input is None or step_input.workflow_session is None:
            return []
        state = dict((step_input.workflow_session.session_data or {}).get("pro_mode_state") or {})
        verifier = (
            ProModeVerifierReport.model_validate(dict(state.get("verifier") or {}))
            if state.get("verifier")
            else None
        )
        if verifier is None:
            return []
        return list(verifier.issues or verifier.suggested_changes)

    def _intake_prompt(
        self,
        *,
        messages: list[dict[str, Any]],
        latest_user_text: str,
    ) -> str:
        recent_chat = self._render_recent_chat(messages)
        return (
            "You are the front-door triage agent for Pro Mode.\n"
            "Decide whether the current user turn should be answered immediately or escalated "
            "to either the tool workflow or the full expert council.\n\n"
            "Also classify the internal execution regime and explicit context policy.\n\n"
            "Routing rules:\n"
            "- Choose `direct_response` for greetings, thanks, acknowledgements, brief conversational turns, "
            "or short conceptual questions that can be answered safely without expert deliberation, tools, "
            "formal verification, or long context.\n"
            "- Choose `tool_workflow` for explicit tool use, code execution, image analysis, file/data interpretation, "
            "optimization loops, BisQue operations, or tasks that should be grounded by an external computation "
            "or scientific tool before answering.\n"
            "- Choose `deep_reasoning` for technical analysis, quantitative or exact scientific reasoning, "
            "multi-step derivations, unresolved prior technical context, or any case where direct answering "
            "would risk skipping validation.\n"
            "- If earlier conversation was technical but the current turn is just a greeting or acknowledgement, "
            "respond to the current turn directly and do not continue the old problem.\n"
            "- Set `load_memory` to true only if prior conversation state is genuinely needed.\n"
            "- Set `load_knowledge` to true only if curated or project knowledge is genuinely needed.\n"
            "- Prefer `reasoning_solver` for self-contained hard questions and `proof_workflow` for rigorous proofs.\n"
            "- Use `autonomous_cycle` for hard, open-ended scientific reasoning that benefits from bounded Think -> Act -> Analyze loops, typed checkpoints, or follow-up resumability.\n"
            "- Use `expert_council` only when explicit multi-perspective debate is truly necessary.\n"
            "- Use `focused_team` for open-ended report-quality synthesis when a smaller, more stable multi-agent pass is better than the full expert council.\n"
            "- Set `execution_regime` to one of: `fast_dialogue`, `validated_tool`, `iterative_research`, "
            "`autonomous_cycle`, `focused_team`, `reasoning_solver`, `proof_workflow`, `expert_council`.\n"
            "- Set `task_regime` to one of: `phatic_or_small_talk`, `closed_form_grounded`, "
            "`artifact_interpretation`, `dataset_or_catalog_research`, `iterative_multimodal_research`, "
            "`conceptual_high_uncertainty`.\n"
            "- `context_policy.history_window` should match how much recent chat should be forwarded.\n"
            "- `context_policy.artifact_handles_to_expose` should be a short list such as "
            "`uploaded_files`, `dataset_uris`, `resource_uris`, `prediction_json_paths`, or `preview_paths`.\n"
            "- Set `context_policy.compression_required=true` only for multi-step tool or research workflows.\n"
            "- When `route=direct_response`, provide the final user-facing answer in `direct_response`.\n"
            "- When `route` is `deep_reasoning` or `tool_workflow`, `direct_response` must be null.\n\n"
            f"Current user turn:\n{latest_user_text}\n\n"
            f"Recent raw chat:\n{recent_chat or 'No recent chat.'}\n"
        )

    @staticmethod
    def _render_recent_chat(messages: list[dict[str, Any]], *, limit: int = 4) -> str:
        lines: list[str] = []
        for raw in messages[-limit:]:
            role = str(raw.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = str(raw.get("content") or "").strip()
            if not content:
                continue
            lines.append(f"{role.upper()}: {content[:1200]}")
        return "\n\n".join(lines).strip()

    def _fallback_intake_decision(
        self,
        *,
        messages: list[dict[str, Any]],
        latest_user_text: str,
    ) -> ProModeIntakeDecision:
        del messages
        text = " ".join(str(latest_user_text or "").strip().split())
        lowered = text.lower()
        if lowered in {"hi", "hello", "hey", "hello!", "hi!", "hey!"}:
            return ProModeIntakeDecision(
                route="direct_response",
                execution_regime="fast_dialogue",
                task_regime="phatic_or_small_talk",
                reason="Simple greeting.",
                direct_response="Hello! How can I help you today?",
                load_memory=False,
                load_knowledge=False,
                recent_history_turns=0,
                context_policy=ProModeContextPolicy(
                    load_memory=False,
                    load_knowledge=False,
                    history_window=0,
                    artifact_handles_to_expose=[],
                    compression_required=False,
                ),
            )
        if lowered in {"thanks", "thank you", "thx", "thanks!"}:
            return ProModeIntakeDecision(
                route="direct_response",
                execution_regime="fast_dialogue",
                task_regime="phatic_or_small_talk",
                reason="Simple acknowledgement.",
                direct_response="You're welcome. What would you like to work on next?",
                load_memory=False,
                load_knowledge=False,
                recent_history_turns=0,
                context_policy=ProModeContextPolicy(
                    load_memory=False,
                    load_knowledge=False,
                    history_window=0,
                    artifact_handles_to_expose=[],
                    compression_required=False,
                ),
            )
        history_needed = bool(
            re.search(r"\b(that|this|it|those|these|previous|earlier|above)\b", lowered)
        )
        return ProModeIntakeDecision(
            route="deep_reasoning",
            execution_regime=self._default_deep_execution_regime(),
            task_regime="conceptual_high_uncertainty",
            reason=(
                "Default to the single-reasoner workflow when the turn is not obviously lightweight."
                if not self._enable_expert_council
                else "Default to the expert workflow when the turn is not obviously lightweight."
            ),
            direct_response=None,
            load_memory=history_needed,
            load_knowledge=False,
            recent_history_turns=2,
            context_policy=ProModeContextPolicy(
                load_memory=history_needed,
                load_knowledge=False,
                history_window=2,
                artifact_handles_to_expose=[],
                compression_required=False,
            ),
        )

    def _private_memo_prompt(
        self,
        *,
        role: ProModeRole,
        latest_user_text: str,
        context_snapshot: str,
        profile: ProModeTaskProfile,
    ) -> str:
        report_overlay = (
            "\n\nThis is a report-style request.\n"
            "Surface section-worthy claims, distinctions, and omissions that would materially affect report quality.\n"
            "Do not settle for a textbook-definition summary."
            if profile.report_requested
            else ""
        )
        return (
            f"{ROLE_SYSTEM_PROMPTS[role]}\n\n"
            "You are in the private-first memo phase.\n"
            "This is your private proposal, not the final answer.\n"
            f"User question:\n{latest_user_text}\n\n"
            f"Shared context pre-read:\n{context_snapshot or 'No extra context.'}\n\n"
            f"Task profile JSON:\n{profile.model_dump_json(indent=2)}\n\n"
            "Return a compact private memo from your role's perspective only.\n"
            "The memo must help later cross-role discussion.\n"
            "Do not imitate the other roles.\n"
            "Be concrete, specific, and evidence-aware.\n"
            "Do not produce the final answer."
            f"{report_overlay}"
        )

    def _discussion_prompt(
        self,
        *,
        role: ProModeRole,
        round_index: int,
        latest_user_text: str,
        profile: ProModeTaskProfile,
        context_snapshot: str,
        memo: CouncilMessage | None,
        target_messages: list[CouncilMessage],
        calculator_results: list[ProModeCalculatorResult],
        verifier_feedback: list[str],
    ) -> str:
        target_block = json.dumps(
            [item.model_dump(mode="json") for item in target_messages], ensure_ascii=False, indent=2
        )
        calc_block = json.dumps(
            [item.model_dump(mode="json") for item in calculator_results],
            ensure_ascii=False,
            indent=2,
        )
        feedback_block = json.dumps(verifier_feedback, ensure_ascii=False, indent=2)
        tool_rule = (
            "You may request up to three calculator calls only if they materially resolve the central crux."
            if role == "tool_broker"
            else "You may not request calculator calls. tool_requests must be empty."
        )
        phase_label = (
            "the Socratic crux review phase"
            if role == "socratic_crux_examiner" and round_index <= 0
            else "the Tool / Evidence Broker action-selection phase"
            if role == "tool_broker" and round_index <= 0
            else f"discussion round {round_index}"
        )
        report_overlay = (
            "Because the user asked for a report, prioritize section-worthy insights, comparisons, and missing pieces that would make the final write-up feel shallow or incomplete.\n"
            if profile.report_requested
            else ""
        )
        return (
            f"{ROLE_SYSTEM_PROMPTS[role]}\n\n"
            f"You are in {phase_label} of Pro Mode.\n"
            f"{tool_rule}\n\n"
            f"{report_overlay}"
            f"User question:\n{latest_user_text}\n\n"
            f"Task profile JSON:\n{profile.model_dump_json(indent=2)}\n\n"
            f"Shared context pre-read:\n{context_snapshot or 'No extra context.'}\n\n"
            f"Your private memo JSON:\n{json.dumps((memo.model_dump(mode='json') if memo else {}), ensure_ascii=False, indent=2)}\n\n"
            f"Target messages JSON:\n{target_block}\n\n"
            f"Calculator results JSON:\n{calc_block}\n\n"
            f"Verifier feedback JSON:\n{feedback_block}\n\n"
            "Write one public council message.\n"
            "- Respond to the specific target messages, not to the ether.\n"
            "- If you identify a central crux, keep it concise and discriminating.\n"
            "- Set vote to agree, agree_with_reservation, or needs_revision.\n"
            "- ready_to_finalize may be true only if your central blockers are empty."
        )

    @staticmethod
    def _synthesis_prompt(
        *,
        latest_user_text: str,
        profile: ProModeTaskProfile,
        blackboard: CouncilBlackboard,
        round_data: CouncilRound | None,
        verifier_feedback: list[str],
    ) -> str:
        report_overlay = (
            "This is a report-style request. Produce a substantive report, not a generic overview.\n"
            "Default structure unless the user asked otherwise: Executive Summary; Core Architecture or Taxonomy; Key Variants or Mechanisms; Applications and Tradeoffs; Limitations and Open Questions; Bottom Line.\n"
            "Make concrete distinctions explicit, and use a compact comparison table when it materially clarifies the landscape.\n"
            "The final report should feel worthy of Pro Mode: specific, synthesized, and decision-useful.\n\n"
            if profile.report_requested
            else ""
        )
        return (
            f"{ROLE_SYSTEM_PROMPTS['synthesizer']}\n\n"
            "Write the final answer only after reading the entire council blackboard.\n"
            "Do not mention internal experts, councils, tools, or workflow machinery.\n"
            "Make the answer polished, coherent, and appropriately detailed for the user's request.\n\n"
            f"{report_overlay}"
            f"User question:\n{latest_user_text}\n\n"
            f"Task profile JSON:\n{profile.model_dump_json(indent=2)}\n\n"
            f"Blackboard JSON:\n{blackboard.model_dump_json(indent=2)}\n\n"
            f"Latest round JSON:\n{round_data.model_dump_json(indent=2) if round_data is not None else '{}'}\n\n"
            f"Verifier feedback JSON:\n{json.dumps(verifier_feedback, ensure_ascii=False, indent=2)}\n\n"
            "Return the final answer draft and a compact settlement summary."
        )

    @staticmethod
    def _verifier_prompt(
        *,
        latest_user_text: str,
        synthesis: ProModeSynthesis,
        blackboard: CouncilBlackboard,
    ) -> str:
        return (
            "You are the verifier for Pro Mode.\n"
            "Check the final draft against the council blackboard and evidence.\n"
            "Flag issues only if they materially affect correctness, clarity, or evidentiary support.\n\n"
            f"User question:\n{latest_user_text}\n\n"
            f"Synthesis JSON:\n{synthesis.model_dump_json(indent=2)}\n\n"
            f"Blackboard JSON:\n{blackboard.model_dump_json(indent=2)}\n\n"
            "Return a verifier report. Do not rewrite the answer yourself."
        )

    def _fallback_private_memo(
        self, *, role: ProModeRole, latest_user_text: str
    ) -> ProModeRoleMemo:
        return ProModeRoleMemo(
            role=ROLE_LABELS[role],
            headline=f"{ROLE_LABELS[role]} fallback memo for: {latest_user_text[:100]}",
            claims=[ROLE_BRIEFS[role]],
            assumptions=["The role could not generate a richer memo."],
            open_questions=["What is the smallest next check needed to stabilize the answer?"],
            confidence="low",
        )

    def _fallback_discussion_reply(
        self,
        *,
        role: ProModeRole,
        round_index: int,
        target_messages: list[CouncilMessage],
    ) -> ProModeDiscussionReply:
        recipient_roles = (
            ", ".join(item.sender_role for item in target_messages[:2]) or "the relevant peers"
        )
        return ProModeDiscussionReply(
            content=f"{ROLE_LABELS[role]} responds to {recipient_roles} with a conservative fallback.",
            claims=[f"Review the key claim from {recipient_roles} before finalizing."],
            objections=(["One central check remains unresolved."] if round_index == 1 else []),
            requested_actions=["Tighten the argument using the strongest available evidence."],
            ready_to_finalize=(round_index > 1),
            vote=("agree_with_reservation" if round_index > 1 else "needs_revision"),
            central_cruxes=(
                [] if round_index > 1 else ["Fallback path could not resolve the main crux."]
            ),
            resolved_cruxes=[],
            tool_requests=[],
            confidence="low",
        )

    @staticmethod
    def _fallback_synthesis(
        *, latest_user_text: str, blackboard: CouncilBlackboard
    ) -> ProModeSynthesis:
        blockers = list(blackboard.convergence.central_blockers[:2])
        answer = (
            "I could not fully stabilize the internal council, so this answer carries explicit uncertainty."
            if blockers
            else "I could not fully stabilize the internal council, but the current answer is the best available draft."
        )
        if latest_user_text:
            answer += f" The question was: {latest_user_text[:180]}"
        return ProModeSynthesis(
            response_text=answer,
            settlement_summary="Fallback synthesis used after the council did not complete cleanly.",
            consensus_level=blackboard.convergence.consensus_level,
            confidence="low",
            unresolved_points=blockers,
            minority_view=(
                "The council still disagreed on one or more central points." if blockers else None
            ),
            agreement_reached=bool(blackboard.convergence.ready),
        )

    def _repair_calculator_request(
        self,
        request: ProModeCalculatorRequest,
    ) -> ProModeCalculatorRequest | None:
        expression = str(request.expression or "").strip()
        variables = dict(request.variables or {})
        alias_map = {"hc": "hbar_c", "hbarc": "hbar_c", "m_ec2": "m_e_c2", "mec2": "m_e_c2"}
        repaired = expression
        changed = False
        for bad, good in alias_map.items():
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(bad)}(?![A-Za-z0-9_])", repaired):
                repaired = re.sub(
                    rf"(?<![A-Za-z0-9_]){re.escape(bad)}(?![A-Za-z0-9_])", good, repaired
                )
                changed = True
        if not changed:
            return None
        return ProModeCalculatorRequest(
            purpose=request.purpose, expression=repaired, variables=variables
        )

    def _execute_calculator_requests(
        self,
        *,
        requests: list[ProModeCalculatorRequest],
        tool_invocations: list[dict[str, Any]],
        event_callback: Callable[[dict[str, Any]], None] | None,
    ) -> list[ProModeCalculatorResult]:
        results: list[ProModeCalculatorResult] = []
        for request in list(requests or [])[:4]:
            if callable(event_callback):
                event_callback(
                    {
                        "kind": "tool",
                        "phase": "tool",
                        "status": "started",
                        "tool_name": "numpy_calculator",
                        "message": f"Running calculator for {request.purpose}.",
                        "payload": {
                            "tool": "numpy_calculator",
                            "args": request.model_dump(mode="json"),
                        },
                    }
                )
            raw_result = numpy_calculator(
                expression=request.expression, variables=dict(request.variables or {})
            )
            if not bool(raw_result.get("success")):
                repaired_request = self._repair_calculator_request(request)
                if repaired_request is not None:
                    raw_result = numpy_calculator(
                        expression=repaired_request.expression,
                        variables=dict(repaired_request.variables or {}),
                    )
            result = ProModeCalculatorResult(
                purpose=request.purpose,
                expression=str(raw_result.get("expression") or request.expression),
                success=bool(raw_result.get("success")),
                formatted_result=str(raw_result.get("formatted_result") or "").strip(),
                result=raw_result.get("result"),
                error=str(raw_result.get("error") or "").strip() or None,
            )
            results.append(result)
            tool_invocations.append(
                {
                    "tool": "numpy_calculator",
                    "status": ("completed" if result.success else "error"),
                    "args": request.model_dump(mode="json"),
                    "output_envelope": raw_result if isinstance(raw_result, dict) else {},
                    "output_summary": {
                        "purpose": request.purpose,
                        "formatted_result": result.formatted_result or None,
                    },
                    "output_preview": json.dumps(raw_result, ensure_ascii=False, default=str)[:800],
                }
            )
            if callable(event_callback):
                event_callback(
                    {
                        "kind": "tool",
                        "phase": "tool",
                        "status": ("completed" if result.success else "error"),
                        "tool_name": "numpy_calculator",
                        "message": (
                            f"Calculator completed for {request.purpose}."
                            if result.success
                            else f"Calculator failed for {request.purpose}."
                        ),
                        "payload": {
                            "tool": "numpy_calculator",
                            "args": request.model_dump(mode="json"),
                            "result": result.model_dump(mode="json"),
                        },
                    }
                )
        return results

    def _finalize_payload(
        self,
        *,
        synthesis: ProModeSynthesis,
        verifier_report: ProModeVerifierReport,
        phase_timings: dict[str, float],
        tool_invocations: list[dict[str, Any]],
        active_roles: list[str],
        round_count: int,
        blackboard: CouncilBlackboard,
        messages: list[CouncilMessage],
        rounds: list[CouncilRound],
        model_call_count: int,
        role_stats: dict[str, Any],
        debug: bool,
        intake_decision: ProModeIntakeDecision,
    ) -> dict[str, Any]:
        if intake_decision.route == "direct_response":
            phase_order = [
                "intake",
                "context_policy",
                "execution_router",
                "direct_response",
                "finalize",
            ]
        elif intake_decision.route == "tool_workflow":
            phase_order = [
                "intake",
                "context_policy",
                "execution_router",
                "tool_workflow",
                "finalize",
            ]
        else:
            phase_order = [
                phase
                for phase in PRO_MODE_PHASE_ORDER
                if phase not in {"direct_response", "tool_workflow"}
            ]
        if intake_decision.route == "direct_response":
            summary = "Answered directly at the intake gate without expert-council escalation."
        elif intake_decision.route == "tool_workflow":
            summary = "Delegated to the tool workflow after intake triage."
        else:
            summary = (
                "Internal council converged and verification passed."
                if verifier_report.passed
                else "Internal council completed with remaining verifier concerns."
            )
        if tool_invocations:
            summary += " Tool evidence was used."
        dev_conversation = (
            self._build_dev_conversation(
                messages=messages,
                rounds=rounds,
                blackboard=blackboard,
                synthesis=synthesis,
                verifier_report=verifier_report,
                model_call_count=model_call_count,
            )
            if debug
            else None
        )
        return {
            "response_text": str(synthesis.response_text or "").strip(),
            "tool_invocations": tool_invocations,
            "metadata": {
                "debug": {
                    "path": "pro_mode",
                    "agent_mode": "workflow",
                    "prompt_profile": "pro_mode",
                    "active_roles": active_roles,
                    "model_call_count": model_call_count,
                },
                "pro_mode": {
                    "route": intake_decision.route,
                    "execution_path": intake_decision.route,
                    "execution_regime": intake_decision.execution_regime,
                    "reasoning_effort": (
                        "high"
                        if intake_decision.execution_regime == "expert_council"
                        else "low"
                        if intake_decision.execution_regime == "fast_dialogue"
                        else "medium"
                    ),
                    "task_regime": intake_decision.task_regime,
                    "context_policy": intake_decision.context_policy.model_dump(mode="json"),
                    "intake": intake_decision.model_dump(mode="json"),
                    "active_roles": active_roles,
                    "phase_order": phase_order,
                    "phase_timings": dict(phase_timings),
                    "round_count": (
                        0
                        if intake_decision.route in {"direct_response", "tool_workflow"}
                        else max(1, int(round_count or len(rounds) or 1))
                    ),
                    "discussion_round_count": (
                        0
                        if intake_decision.route in {"direct_response", "tool_workflow"}
                        else max(1, int(round_count or len(rounds) or 1))
                    ),
                    "model_call_count": (
                        max(1, int(model_call_count or 0))
                        if intake_decision.route in {"direct_response", "tool_workflow"}
                        else max(0, int(model_call_count or 0))
                    ),
                    "convergence": blackboard.convergence.model_dump(mode="json"),
                    "role_stats": role_stats,
                    "calculator": {
                        "used": bool(tool_invocations),
                        "call_count": len(tool_invocations),
                        "results": [
                            item.model_dump(mode="json") for item in blackboard.calculator_results
                        ],
                    },
                    "verifier": verifier_report.model_dump(mode="json"),
                    "summary": summary,
                    **(
                        {"dev_conversation": dev_conversation}
                        if isinstance(dev_conversation, dict)
                        else {}
                    ),
                },
            },
        }

    def _build_dev_conversation(
        self,
        *,
        messages: list[CouncilMessage],
        rounds: list[CouncilRound],
        blackboard: CouncilBlackboard,
        synthesis: ProModeSynthesis,
        verifier_report: ProModeVerifierReport,
        model_call_count: int,
    ) -> dict[str, Any]:
        structured = {
            "messages": [message.model_dump(mode="json") for message in messages],
            "rounds": [round_item.model_dump(mode="json") for round_item in rounds],
            "calculator_results": [
                item.model_dump(mode="json") for item in blackboard.calculator_results
            ],
            "synthesis": synthesis.model_dump(mode="json"),
            "verifier": verifier_report.model_dump(mode="json"),
            "convergence": blackboard.convergence.model_dump(mode="json"),
            "model_call_count": model_call_count,
        }
        structured["markdown"] = self._render_dev_conversation_markdown(
            messages=messages,
            rounds=rounds,
            blackboard=blackboard,
            synthesis=synthesis,
            verifier_report=verifier_report,
        )
        return structured

    def _render_dev_conversation_markdown(
        self,
        *,
        messages: list[CouncilMessage],
        rounds: list[CouncilRound],
        blackboard: CouncilBlackboard,
        synthesis: ProModeSynthesis,
        verifier_report: ProModeVerifierReport,
    ) -> str:
        lines: list[str] = [
            "## Pro Mode Internal Conversation",
            "",
            f"- Task type: `{blackboard.task_profile.task_type}`",
            f"- Exactness priority: `{str(blackboard.task_profile.exactness_priority).lower()}`",
            f"- Calculator candidate: `{str(blackboard.task_profile.calculator_candidate).lower()}`",
            "",
            "### Context Pre-read",
            blackboard.context_pre_read or "_No extra context._",
        ]
        private_memos = [message for message in messages if message.message_kind == "private_memo"]
        if private_memos:
            lines.extend(["", "### Private Memos"])
            for message in private_memos:
                lines.extend(
                    [
                        "",
                        f"#### {message.sender_role}",
                        f"- Headline: {message.content}",
                    ]
                )
                if message.claims:
                    lines.append("- Claims:")
                    lines.extend(f"  - {item}" for item in message.claims[:3])
                if message.requested_actions:
                    lines.append("- Open questions:")
                    lines.extend(f"  - {item}" for item in message.requested_actions[:4])
        for round_item in rounds:
            lines.extend(
                [
                    "",
                    f"### Round {round_item.round_index}",
                    f"- Consensus snapshot: `{blackboard.convergence.consensus_level}`",
                ]
            )
            if round_item.unresolved_cruxes:
                lines.append("- Central cruxes:")
                lines.extend(f"  - {item}" for item in round_item.unresolved_cruxes[:5])
            for message in round_item.messages:
                lines.extend(
                    [
                        "",
                        f"#### {message.sender_role}",
                        f"- Recipients: {', '.join(message.recipient_roles) or 'controller'}",
                        f"- Vote: `{message.vote}`",
                        f"- Content: {message.content}",
                    ]
                )
                if message.objections:
                    lines.append("- Objections:")
                    lines.extend(f"  - {item}" for item in message.objections[:4])
                if message.requested_actions:
                    lines.append("- Requested actions:")
                    lines.extend(f"  - {item}" for item in message.requested_actions[:4])
        lines.extend(
            [
                "",
                "### Calculator Results",
            ]
        )
        if blackboard.calculator_results:
            for item in blackboard.calculator_results:
                lines.append(
                    f"- {item.purpose}: {item.formatted_result or item.error or 'No result'}"
                )
        else:
            lines.append("- No calculator call was used.")
        lines.extend(
            [
                "",
                "### Synthesis",
                synthesis.response_text or "_No final answer draft._",
                "",
                "### Verifier",
                f"- Passed: `{str(verifier_report.passed).lower()}`",
            ]
        )
        if verifier_report.issues:
            lines.append("- Issues:")
            lines.extend(f"  - {item}" for item in verifier_report.issues[:4])
        return "\n".join(lines).strip()
