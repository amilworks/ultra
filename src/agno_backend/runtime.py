from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

import httpx
from agno.agent import Agent
from agno.compression.manager import CompressionManager
from agno.db.postgres import PostgresDb
from agno.db.sqlite import SqliteDb
from agno.knowledge.document.base import Document
from agno.models.base import Model as AgnoModel
from agno.models.message import Message
from agno.models.openai.like import OpenAILike
from agno.models.response import ToolExecution
from agno.run.agent import RunOutput, RunOutputEvent
from agno.run.requirement import RunRequirement
from agno.team.team import Team, TeamMode
from agno.tools import Function
from agno.tools.knowledge import KnowledgeTools
from agno.tools.memory import MemoryTools
from agno.tools.reasoning import ReasoningTools
from agno.tools.workflow import WorkflowTools
from agno.workflow import Loop, Parallel, Step, Workflow
from agno.workflow.types import StepInput, StepOutput
from pydantic import BaseModel, Field
from sqlalchemy import create_engine

from src.agentic.db import AgenticDb
from src.agentic.policies import APPROVAL_REQUIRED_TOOL_NAMES, route_scientist_turn
from src.agentic.repositories import ScientificNoteRepository, SessionRepository
from src.auth import get_request_bisque_auth, reset_request_bisque_auth, set_request_bisque_auth
from src.config import Settings
from src.science.viewer import build_hdf5_viewer_manifest
from src.tooling.calculator import numpy_calculator
from src.tooling.domains import (
    ANALYSIS_TOOL_SCHEMAS,
    BISQUE_TOOL_SCHEMAS,
    CHEMISTRY_TOOL_SCHEMAS,
    CODE_EXECUTION_TOOL_SCHEMAS,
    VISION_TOOL_SCHEMAS,
)
from src.tooling.engine import _progress_summary_from_result
from src.tools import AVAILABLE_TOOLS, execute_tool_call, extract_scientific_image_paths_from_text

from .codeexec_reasoning import build_codeexec_reasoning_agent
from .knowledge import ScientificKnowledgeContext, ScientificKnowledgeHub, ScientificKnowledgeScope
from .learning import ScientificLearningJournal
from .memory import (
    ScientificMemoryContext,
    ScientificMemoryPolicy,
    ScientificMemoryService,
    ScientificMemoryUpdate,
)
from .pro_mode import (
    ProModeIntakeDecision,
    ProModeSynthesis,
    ProModeVerifierReport,
    ProModeWorkflowResult,
    ProModeWorkflowRunner,
)
from .pro_mode_prompts import build_pro_mode_final_writer_prompt

ALL_TOOL_SCHEMAS = [
    *BISQUE_TOOL_SCHEMAS,
    *VISION_TOOL_SCHEMAS,
    *ANALYSIS_TOOL_SCHEMAS,
    *CODE_EXECUTION_TOOL_SCHEMAS,
    *CHEMISTRY_TOOL_SCHEMAS,
]
TOOL_SCHEMA_MAP = {
    str(schema.get("function", {}).get("name") or "").strip(): schema
    for schema in ALL_TOOL_SCHEMAS
    if isinstance(schema, dict)
}

DEFAULT_STANDARD_MAX_RUNTIME_SECONDS = 900
DEFAULT_PRO_MODE_MAX_RUNTIME_SECONDS = 7200
MAX_POST_TOOL_IDLE_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class AgnoTurnIntent:
    user_text: str
    uploaded_files: list[str] = field(default_factory=list)
    selected_tool_names: list[str] = field(default_factory=list)
    workflow_hint: dict[str, Any] = field(default_factory=dict)
    selection_context: dict[str, Any] = field(default_factory=dict)
    knowledge_context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgnoRouteDecision:
    primary_domain: str
    selected_domains: list[str]
    reason: str
    available_tool_names: list[str] = field(default_factory=list)
    requires_approval: bool = False
    approval_tool_names: list[str] = field(default_factory=list)


@dataclass
class AgnoChatRuntimeResult:
    response_text: str
    selected_domains: list[str]
    domain_outputs: dict[str, str]
    tool_calls: int
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProModeToolPlan:
    category: str
    selected_tool_names: list[str] = field(default_factory=list)
    strict_validation: bool = False
    reason: str = ""


class ValidatedNumericExpression(BaseModel):
    label: str
    expression: str
    purpose: str = ""


class ValidatedNumericPlan(BaseModel):
    target_quantity: str
    target_units: str = ""
    primary_convention: str = ""
    calculation_expressions: list[ValidatedNumericExpression] = Field(default_factory=list)
    alternative_conventions: list[str] = Field(default_factory=list)
    final_expression_label: str = ""


class ValidatedNumericSynthesis(BaseModel):
    response_text: str
    primary_result_label: str = ""
    alternative_result_labels: list[str] = Field(default_factory=list)


RESEARCH_PROGRAM_TOOL_BUNDLES: dict[str, tuple[str, ...]] = {
    "catalog": (
        "bisque_find_assets",
        "search_bisque_resources",
        "load_bisque_resource",
        "bisque_download_dataset",
        "bisque_download_resource",
    ),
    "vision": (
        "bioio_load_image",
        "yolo_detect",
        "segment_image_megaseg",
        "segment_image_sam2",
        "estimate_depth_pro",
    ),
    "acquisition": ("analyze_prediction_stability",),
    "analysis": (
        "quantify_objects",
        "quantify_segmentation_masks",
        "plot_quantified_detections",
        "compare_conditions",
        "numpy_calculator",
    ),
    "code": (
        "codegen_python_plan",
        "execute_python_job",
    ),
}

RESEARCH_PROGRAM_SAFE_TOOL_NAMES: tuple[str, ...] = tuple(
    dict.fromkeys(
        tool_name for bundle in RESEARCH_PROGRAM_TOOL_BUNDLES.values() for tool_name in bundle
    )
)

PROOF_METHOD_GUIDANCE = (
    "General proof method:\n"
    "1. Start with a Cartesian frame: classify the problem as perfectly understood or imperfectly understood; "
    "list the known conditions, unknown targets, and any missing bridges.\n"
    "2. State the simplest object, relation, substitution, or transformed variable that can be understood first.\n"
    "3. Name the representation in which the statement becomes easiest to manipulate.\n"
    "4. Prefer exact reductions to local certificates, local obstructions, valuations, residues, carry counts, "
    "monotone quantities, or surrogate bounds before attacking the whole theorem directly.\n"
    "5. If different mechanisms govern different ranges, split the proof into regimes explicitly and say why the split helps.\n"
    "6. For existence results, consider whether a constructive family, density argument, or good-set minus bad-set count "
    "can replace direct search.\n"
    "7. Keep the endgame explicit: show how the reduced statement returns to the original theorem and yields infinitude, "
    "contradiction, or impossibility.\n"
    "8. Run a Cartesian omission audit: ask what cases, dependencies, local-to-global bridges, or asymptotic steps remain unclear."
)

PROOF_WORKFLOW_ROLE_INSTRUCTIONS: dict[str, str] = {
    "proof_planner": (
        "You are the Proof Planner. Restate the theorem precisely, classify the proof target, "
        "choose the simplest starting substitution, reduction, or invariant, and propose a lemma order from "
        "simplest to hardest. Explicitly track the main proof obligations and the endgame needed to close the theorem. "
        "If the proof uses base-p digits, valuations, residues, or carries, name the exact local lemma that would justify "
        "any componentwise comparison; do not treat ordinary order as coordinatewise order by default. "
        "Do not claim the theorem is proved unless the remaining gaps and open obligations are empty. "
        + PROOF_METHOD_GUIDANCE
    ),
    "proof_constructor": (
        "You are the Proof Constructor. Extend the strongest current proof path from the simplest "
        "verified step. Add only steps that can plausibly be made rigorous. Prefer reductions, "
        "lemmas, explicit inequalities, and clean local-to-global transitions over broad narrative. "
        "Advance one open obligation at a time and make the dependency chain explicit. "
        "Do not infer digitwise, coefficientwise, or residuewise monotonicity from an ordinary inequality unless it is separately proved. "
        "If a hard quantity can be controlled by a simpler surrogate that still suffices, make that surrogate explicit and prove the bridge."
    ),
    "proof_obligation_closer": (
        "You are the Obligation Closer. Focus only on the single most important unresolved obligation. "
        "Either close it with a rigorous local argument, or explain exactly why it remains blocked. "
        "Do not move to later obligations until the current dependency is explicit. "
        "Prefer the smallest exact local statement that would unblock the larger proof."
    ),
    "proof_example_finder": (
        "You are the Example Finder. Search for tiny examples, edge cases, or arithmetic configurations that could falsify the current local lemma or expose a hidden missing hypothesis. "
        "If no counterexample is evident, say what feature of the claim survived the stress test."
    ),
    "proof_checker": (
        "You are the Formal Gap Checker. Reduce the current proof to load-bearing claims and identify "
        "any unjustified implications, hidden quantifiers, omitted ranges, asymptotic jumps, local-to-global jumps, "
        "or infinitude/endgame gaps. Check whether every open obligation is actually discharged. "
        "Attack any step that turns a global inequality into componentwise digit, coefficient, or residue bounds; such steps are suspect unless independently justified. "
        "Also attack any step that tries to derive prime-by-prime divisibility, valuation signs, or coefficientwise conclusions from a coarse global logarithmic estimate without a verified bridge. "
        "Check that the proof's chosen representation, regime split, and endgame really cover the original theorem."
    ),
    "proof_skeptic": (
        "You are the Skeptic. Attack the strongest current proof path at its weakest point. Look for "
        "omitted cases, invalid reductions, unjustified asymptotics, broken case splits, or places where the argument "
        "silently assumes the conclusion. Attack endgame and infinitude claims especially hard. "
        "If a local monotonicity, carry, digitwise, or valuation claim looks fragile, try to break it with a tiny counterexample or keep the blocker active. "
        "Do not accept a proof that converts a small weighted sum or O(log n) slack into universal primewise sign information unless the missing arithmetic bridge is explicitly proved. "
        "If the current path survives, say so plainly. "
        "Ask whether the proof has actually produced a good object, ruled out all bad objects, or only established a suggestive average."
    ),
}

PROOF_REPAIR_ROLE_INSTRUCTIONS: dict[str, str] = {
    "proof_reframer": (
        "You are the Proof Reframer. The current proof path is stuck or underpowered. "
        "Find the smallest better reduction, auxiliary object, invariant, or equivalent formulation "
        "that could unblock the argument without changing the theorem. Prefer simpler formulations to clever ones. "
        "In particular, look for a cleaner local certificate or a simpler surrogate quantity."
    ),
    "proof_case_splitter": (
        "You are the Case Splitter. If the current proof is stalled, identify the cleanest case distinction "
        "or easy-versus-hard decomposition that makes the obligations easier to discharge. "
        "Do not multiply cases without a real gain in tractability. "
        "Prefer regime splits that change the dominant mechanism, not cosmetic partitions."
    ),
    "proof_endgame_closer": (
        "You are the Endgame Closer. Focus only on the last nontrivial implication: "
        "how the local lemmas, reductions, or case bounds would close the theorem, especially "
        "for infinitude, contradiction, or impossibility conclusions. Identify what exact bridge is still missing. "
        "If the theorem is existential, ask whether a good-set versus bad-set count or density argument would finish it."
    ),
}
VALIDATED_TOOL_CATEGORIES: frozenset[str] = frozenset(
    {
        "validated_numeric",
        "code_execution",
        "image_metadata",
        "uploaded_file_analysis",
        "bisque_management",
        "depth_analysis",
        "detection",
        "segmentation",
    }
)

FOCUSED_TEAM_MEMBER_SPECS: tuple[dict[str, str], ...] = (
    {
        "name": "Cartesian Planner",
        "role": "Turn the user request into a report contract using a Cartesian decomposition of the problem.",
        "instructions": (
            "Apply a Cartesian approach. Restate the real question, break it into the fewest useful subproblems, "
            "order them from foundational and simple to more complex or consequential, and check for completeness. "
            "Name the highest-value distinctions, likely omissions, and the cleanest section structure for a strong report. "
            "For student-facing technical explanations, prefer an arc like intuition, notation, key mechanism, tradeoffs, and takeaways. "
            "Do not write the whole report."
        ),
    },
    {
        "name": "Core Analyst",
        "role": "Explain the core technical landscape, mechanisms, and major variants.",
        "instructions": (
            "Produce the strongest conceptual explanation of the topic. "
            "Define the key terms, explain the core architecture or mechanism, and distinguish the major variants in terms of what changes and why. "
            "Prefer causal and structural explanation over generic textbook summary. "
            "When using equations, define notation before using it, include only equations that earn their place, and explain in plain English what each key equation is doing."
        ),
    },
    {
        "name": "Crux Examiner",
        "role": "Expose hidden assumptions, unresolved ambiguities, decisive tradeoffs, and the few cruxes that determine whether the answer is sound.",
        "instructions": (
            "Act like a compressed Socratic Crux Examiner. "
            "Your purpose is not to solve the problem first but to improve the team's reasoning. "
            "Identify the few highest-value questions that determine whether the current story is sound, force ambiguous claims into precise and testable statements, "
            "surface the strongest alternative interpretations, and name what evidence, caveat, or distinction would resolve uncertainty. "
            "Prefer one decisive crux over many shallow objections, and avoid theatrical skepticism. "
            "Flag abrupt notation, missing derivation bridges, unjustified rules of thumb, and practical advice that sounds broader than the evidence supports."
        ),
    },
    {
        "name": "Prose Synthesist",
        "role": "Turn the analysis into coherent, elegant explanatory prose with strong structure, transitions, and reader guidance.",
        "instructions": (
            "You are the team's prose specialist. "
            "Focus on coherence, paragraph flow, section order, and the reader's experience of understanding the topic. "
            "Write in a clarity-first explanatory mode: lead paragraphs with the governing idea, move from intuition to mechanism to implication, "
            "use concrete examples or analogies when they genuinely clarify an abstraction, use contrasts to sharpen distinctions, and keep the tone confident but not inflated. "
            "Prefer vivid but disciplined prose over generic summary language. "
            "Highlight which distinctions actually change practice and which are mostly conceptual. "
            "Write for advanced undergraduate, graduate, and PhD-level readers: smart, motivated, and technical, but often still learning the conceptual map."
        ),
    },
)

PROSE_STYLE_GUIDELINES: tuple[str, ...] = (
    "Lead with the governing idea of the section or paragraph so the reader knows why the point matters.",
    "Move from intuition to mechanism to implication rather than dropping technical facts in isolation.",
    "Use concrete examples, analogies, or miniature thought experiments only when they genuinely clarify an abstraction.",
    "Sharpen distinctions with contrastive framing: explain not just what something is, but what it is not and why the difference matters.",
    "When figures, tables, or artifacts are available, orient the reader to what to inspect first and why it matters.",
    "Prefer clean, direct sentences with occasional longer sentences for synthesis; vary rhythm without becoming ornate.",
    "Use confident, precise claims, but qualify them where the evidence or scope requires restraint.",
    "Favor strong verbs, concrete nouns, and explicit transitions over vague meta-language.",
    "Allow light wit or surprise only when it clarifies the idea; do not perform cleverness for its own sake.",
)

STUDENT_EXPLANATION_GUIDELINES: tuple[str, ...] = (
    "Treat the reader as intelligent but not omniscient: explain without patronizing and do not skip the load-bearing bridge.",
    "Define notation before or immediately when it is introduced, and keep symbols consistent.",
    "For mathematically dense sections, use a teaching rhythm: intuition, formal definition or equation, plain-English interpretation, then consequence.",
    "Keep equations that do real explanatory work; omit decorative or redundant formulas.",
    "After each important equation, explain what the terms are doing and why the equation matters.",
    "Name common confusions, edge cases, or misleading intuitions where a student reader is likely to stumble.",
    "Make practical advice robust and conditional; avoid brittle defaults unless the scope and assumptions are explicit.",
)


class ToolProgramAction(BaseModel):
    tool_name: str
    purpose: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolProgramIterationPlan(BaseModel):
    objective: str
    reasoning_summary: str = ""
    actions: list[ToolProgramAction] = Field(default_factory=list)
    ready_to_answer: bool = False
    answer_outline: list[str] = Field(default_factory=list)
    remaining_questions: list[str] = Field(default_factory=list)


class ToolProgramSynthesis(BaseModel):
    response_text: str
    evidence_basis: list[str] = Field(default_factory=list)
    unresolved_points: list[str] = Field(default_factory=list)
    confidence: str = "medium"


class ToolProgramReportPacket(BaseModel):
    title: str = "Scientific Report"
    direct_answer: str = ""
    executive_summary: list[str] = Field(default_factory=list)
    measured_findings: list[str] = Field(default_factory=list)
    comparative_findings: list[str] = Field(default_factory=list)
    interpretation: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)


class ResearchPresentationEvidence(BaseModel):
    source: str = ""
    summary: str = ""
    artifact: str | None = None
    run_id: str | None = None


class ResearchPresentationMeasurement(BaseModel):
    name: str = ""
    value: Any = None
    unit: str | None = None
    summary: str | None = None


class ResearchPresentationStatistic(BaseModel):
    label: str = ""
    summary: str = ""


class ResearchPresentationConfidence(BaseModel):
    level: Literal["low", "medium", "high"] = "medium"
    why: list[str] = Field(default_factory=list)


class ResearchPresentationNextStep(BaseModel):
    action: str = ""


class ResearchPresentationContract(BaseModel):
    result: str = ""
    evidence: list[ResearchPresentationEvidence] = Field(default_factory=list)
    measurements: list[ResearchPresentationMeasurement] = Field(default_factory=list)
    statistical_analysis: list[ResearchPresentationStatistic] = Field(default_factory=list)
    confidence: ResearchPresentationConfidence = Field(
        default_factory=ResearchPresentationConfidence
    )
    qc_warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_steps: list[ResearchPresentationNextStep] = Field(default_factory=list)


AutonomousCycleAction = Literal[
    "reasoning_solver",
    "focused_team",
    "tool_workflow",
    "finalize",
    "checkpoint",
]

RTDSkill = Literal[
    "deterministic_numeric_workflow",
    "evidence_review_workflow",
    "proof_derivation_workflow",
    "programmatic_experiment_workflow",
    "focused_synthesis_team_workflow",
    "counterfactual_verification_workflow",
]


class RTDProblemFrame(BaseModel):
    objective: str = ""
    task_type: str = "conceptual_high_uncertainty"
    target_quantity: str = ""
    assumptions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class RTDCandidate(BaseModel):
    answer_text: str = ""
    rationale: str = ""
    confidence: float = 0.0


class RTDCandidateSet(BaseModel):
    leader: RTDCandidate = Field(default_factory=RTDCandidate)
    challenger: RTDCandidate = Field(default_factory=RTDCandidate)


class RTDVerificationRecord(BaseModel):
    cycle_index: int = 0
    workflow: str = ""
    outcome: str = ""
    corrected_error: bool = False
    leader_survived: bool = False
    notes: str = ""
    surviving_cruxes: list[str] = Field(default_factory=list)


class RTDCheckpoint(BaseModel):
    cycle_index: int = 0
    stop_reason: str = ""
    resume_readiness: str = "ready"
    next_best_actions: list[str] = Field(default_factory=list)


class RTDBudgetState(BaseModel):
    cycles_used: int = 0
    tool_families_used: list[str] = Field(default_factory=list)
    model_calls: int = 0
    verification_count: int = 0
    watchdog_triggered: bool = False
    watchdog_reasons: list[str] = Field(default_factory=list)


class RTDEpistemicState(BaseModel):
    controller_mode: str = "rtd_v1"
    problem_frame: RTDProblemFrame = Field(default_factory=RTDProblemFrame)
    candidate_set: RTDCandidateSet = Field(default_factory=RTDCandidateSet)
    obligation_ledger: list[str] = Field(default_factory=list)
    evidence_ledger: list[str] = Field(default_factory=list)
    verification_ledger: list[RTDVerificationRecord] = Field(default_factory=list)
    checkpoint: RTDCheckpoint = Field(default_factory=RTDCheckpoint)
    budget_state: RTDBudgetState = Field(default_factory=RTDBudgetState)
    evidence_sufficiency_score: float = 0.0
    continuation_fidelity: float = 0.0


class AutonomousCyclePlan(BaseModel):
    objective: str
    think_plan: str = ""
    selected_action: AutonomousCycleAction = "reasoning_solver"
    selected_skill: RTDSkill | None = None
    action_rationale: str = ""
    candidate_answer: str = ""
    candidate_set: RTDCandidateSet = Field(default_factory=RTDCandidateSet)
    problem_frame: RTDProblemFrame = Field(default_factory=RTDProblemFrame)
    open_obligations: list[str] = Field(default_factory=list)
    obligation_ledger: list[str] = Field(default_factory=list)
    evidence_ledger: list[str] = Field(default_factory=list)
    verification_ledger: list[RTDVerificationRecord] = Field(default_factory=list)
    next_best_actions: list[str] = Field(default_factory=list)
    selected_tool_names: list[str] = Field(default_factory=list)
    request_checkpoint: bool = False


class AutonomousCycleAnalysis(BaseModel):
    should_continue: bool = False
    should_finalize: bool = False
    evidence_sufficiency_score: float = 0.5
    self_correction_needed: bool = False
    stop_reason: str = ""
    resume_readiness: str = "ready"
    open_obligations: list[str] = Field(default_factory=list)
    obligation_ledger: list[str] = Field(default_factory=list)
    next_best_actions: list[str] = Field(default_factory=list)
    candidate_answer: str = ""
    candidate_set: RTDCandidateSet = Field(default_factory=RTDCandidateSet)
    verification_ledger: list[RTDVerificationRecord] = Field(default_factory=list)
    critique: str = ""


class AutonomousCycleWorkflowEnvelope(BaseModel):
    response_text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    tool_invocations: list[dict[str, Any]] = Field(default_factory=list)
    runtime_status: str = "completed"
    runtime_error: str | None = None


class AutonomousCycleControllerEnvelope(BaseModel):
    response_text: str = ""
    controller_summary: str = ""
    workflow_ran: bool = False
    workflow_output: dict[str, Any] = Field(default_factory=dict)
    workflow_status: str = "unknown"
    toolkits_used: list[str] = Field(default_factory=list)


class FocusedTeamMemberNote(BaseModel):
    role: str = ""
    headline: str = ""
    key_points: list[str] = Field(default_factory=list)
    important_differences: list[str] = Field(default_factory=list)
    recommended_sections: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    follow_up_checks: list[str] = Field(default_factory=list)
    confidence: str = "medium"


class FocusedTeamReview(BaseModel):
    major_issues: list[str] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    claims_to_qualify: list[str] = Field(default_factory=list)
    missing_differences: list[str] = Field(default_factory=list)
    passed: bool = True
    confidence: str = "medium"


class MetadataSpecialistSummary(BaseModel):
    direct_answer: str = ""
    verified_findings: list[str] = Field(default_factory=list)
    filename_inferences: list[str] = Field(default_factory=list)
    scientific_context: list[str] = Field(default_factory=list)
    equipment_context: list[str] = Field(default_factory=list)
    derived_location_context: list[str] = Field(default_factory=list)
    missing_metadata: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    location_present: bool = False


class _AutonomousSharedContextKnowledge:
    """Lightweight lexical knowledge adapter for Agno KnowledgeTools.

    This keeps the controller compatible with Agno's explicit search tool
    pattern without introducing a new vector database dependency into the
    gated autonomy path.
    """

    def __init__(self, *, snippets: list[str]) -> None:
        self._snippets = [str(item or "").strip() for item in snippets if str(item or "").strip()]

    @staticmethod
    def _tokenize(value: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[A-Za-z0-9_]+", str(value or "").lower())
            if len(token) >= 3
        }

    def search(self, query: str, max_results: int | None = None, **_: Any) -> list[Document]:
        query_tokens = self._tokenize(query)
        ranked: list[tuple[int, int, str]] = []
        for index, snippet in enumerate(self._snippets):
            snippet_tokens = self._tokenize(snippet)
            score = len(query_tokens & snippet_tokens)
            if query_tokens and score <= 0:
                continue
            ranked.append((score, -index, snippet))
        ranked.sort(reverse=True)
        limit = max(1, int(max_results or 5))
        documents: list[Document] = []
        for rank, (_score, _neg_index, snippet) in enumerate(ranked[:limit], start=1):
            documents.append(
                Document(
                    id=f"autonomy-shared-context-{rank}",
                    name="autonomy_shared_context",
                    content=snippet,
                    meta_data={"source": "shared_context"},
                )
            )
        return documents

    metadata_richness: str = "minimal"


class ProofProblemFrame(BaseModel):
    understanding_status: str = "imperfectly_understood"
    goal_type: str = ""
    known_conditions: list[str] = Field(default_factory=list)
    unknown_targets: list[str] = Field(default_factory=list)
    missing_conditions: list[str] = Field(default_factory=list)
    simplest_first_object: str = ""
    canonical_representation: str = ""
    dependency_order: list[str] = Field(default_factory=list)
    candidate_local_certificate: str = ""
    candidate_regime_split: str = ""
    candidate_endgame: str = ""


class ProofObligation(BaseModel):
    label: str = ""
    statement: str = ""
    kind: str = "lemma"
    status: str = "open"


class ProofRoleMemo(BaseModel):
    role: str = ""
    summary: str = ""
    goal_type: str = ""
    simplest_anchor: str = ""
    canonical_reduction: str = ""
    candidate_direction: str = ""
    proposed_lemmas: list[str] = Field(default_factory=list)
    obligations: list[ProofObligation] = Field(default_factory=list)
    resolved_obligations: list[str] = Field(default_factory=list)
    case_splits: list[str] = Field(default_factory=list)
    endgame_strategy: str = ""
    verified_steps: list[str] = Field(default_factory=list)
    blocker_gaps: list[str] = Field(default_factory=list)
    attack_points: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    ready_to_finalize: bool = False
    confidence: str = "medium"


class ProofWorkflowState(BaseModel):
    goal_type: str = ""
    current_direction: str = ""
    simplest_anchor: str = ""
    canonical_reduction: str = ""
    proof_outline: list[str] = Field(default_factory=list)
    obligations: list[ProofObligation] = Field(default_factory=list)
    resolved_obligations: list[str] = Field(default_factory=list)
    case_splits: list[str] = Field(default_factory=list)
    endgame_strategy: str = ""
    verified_steps: list[str] = Field(default_factory=list)
    blocker_gaps: list[str] = Field(default_factory=list)
    attack_points: list[str] = Field(default_factory=list)
    next_iteration_focus: str = ""
    ready_to_finalize: bool = False
    unresolved_points: list[str] = Field(default_factory=list)
    sanity_findings: list[str] = Field(default_factory=list)
    confidence: str = "medium"
    progress_score: float = 0.0
    quality_flags: list[str] = Field(default_factory=list)


class ProofWorkflowSynthesis(BaseModel):
    response_text: str = ""
    proof_status: str = "partial"
    canonical_reduction: str = ""
    verified_steps: list[str] = Field(default_factory=list)
    resolved_obligations: list[str] = Field(default_factory=list)
    unresolved_points: list[str] = Field(default_factory=list)
    confidence: str = "medium"
    progress_score: float = 0.0


class AgnoChatRuntime:
    """Small Agno-native scientist runtime.

    The default path is intentionally simple:
    one Agno Agent call, optional explicitly selected tools, and Agno-native
    pause/continue handling for approval-gated tools.
    """

    IMAGE_LIKE_SUFFIXES: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
        ".bmp",
        ".gif",
        ".webp",
        ".dcm",
        ".dicom",
        ".nii",
        ".nii.gz",
        ".nd2",
        ".czi",
        ".svs",
        ".zarr",
        ".ome.zarr",
        ".ome.tif",
        ".ome.tiff",
    )
    RESEARCH_PROGRAM_IMAGE_BATCH_SIZE: int = 24

    @staticmethod
    def _setting(settings: Any, name: str, default: Any = None) -> Any:
        return getattr(settings, name, default)

    @staticmethod
    def _normalize_proof_sanity_text(value: Any) -> str:
        text = str(value or "")
        text = (
            text.replace("\\le", "<=").replace("\\ge", ">=").replace("≤", "<=").replace("≥", ">=")
        )
        text = text.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
        return re.sub(r"\s+", " ", text).strip().lower()

    @classmethod
    def _proof_sanity_findings_for_texts(cls, texts: list[str]) -> list[str]:
        corpus = " ".join(
            cls._normalize_proof_sanity_text(text)
            for text in list(texts or [])
            if cls._normalize_proof_sanity_text(text)
        )
        if not corpus:
            return []
        findings: list[str] = []
        componentwise_pattern = re.search(
            r"\bbecause\b[^.]{0,180}<=[^.]{0,180}\b[a-z]_[a-z0-9]+\b[^.]{0,40}<=[^.]{0,40}\b[a-z]_[a-z0-9]+\b",
            corpus,
        )
        digit_context = any(
            token in corpus
            for token in ("digit", "base-p", "base p", "carry", "coefficient", "residue")
        )
        if componentwise_pattern and digit_context:
            findings.append(
                "A global inequality is being treated as componentwise digit, coefficient, or residue monotonicity without a separate proof."
            )
        if (
            digit_context
            and "subset of those for" in corpus
            and "carry" in corpus
            and any(token in corpus for token in ("digit", "base-p", "base p"))
        ):
            findings.append(
                "A carry-comparison claim is being justified by a loose digitwise subset argument rather than a verified local lemma."
            )
        if (
            digit_context
            and any(token in corpus for token in ("for every digit", "each digit", "every digit"))
            and any(
                token in corpus
                for token in ("because a<=", "because x<=", "because m<=", "because n<=")
            )
        ):
            findings.append(
                "The proof appears to infer per-digit bounds directly from an integer inequality, which is not automatically valid."
            )
        if (
            "if any coefficient were positive" in corpus
            and "at least log 2" in corpus
            and "contradict" in corpus
        ):
            findings.append(
                "A weighted logarithmic sum is being used to force coefficientwise nonpositivity without a valid separation argument."
            )
        if (
            any(
                token in corpus
                for token in ("for every prime", "for all primes", "for all p", "for every p")
            )
            and "log n" in corpus
            and any(
                token in corpus
                for token in (
                    "slack dominates",
                    "forces",
                    "forcing",
                    "possible deficit",
                    "uniform control",
                )
            )
            and any(
                token in corpus
                for token in ("valuation inequality", "digit-sum inequality", "divisibility")
            )
        ):
            findings.append(
                "A coarse logarithmic slack is being used to conclude a universal primewise valuation or divisibility claim without a verified per-prime bridge."
            )
        return findings[:4]

    @classmethod
    def _effective_runtime_budget_seconds(
        cls,
        *,
        max_runtime_seconds: int,
        workflow_hint: dict[str, Any] | None = None,
    ) -> int:
        requested = max(1, int(max_runtime_seconds))
        workflow_id = str((workflow_hint or {}).get("id") or "").strip().lower()
        if workflow_id == "pro_mode" and requested == DEFAULT_STANDARD_MAX_RUNTIME_SECONDS:
            return DEFAULT_PRO_MODE_MAX_RUNTIME_SECONDS
        return requested

    def __init__(self, *, settings: Settings) -> None:
        self.settings = settings
        self.model = str(
            self._setting(settings, "resolved_llm_model")
            or self._setting(settings, "llm_model")
            or self._setting(settings, "openai_model")
            or "gpt-oss-120b"
        )
        self.base_url = str(
            self._setting(settings, "resolved_llm_base_url")
            or self._setting(settings, "llm_base_url")
            or self._setting(settings, "openai_base_url")
            or self._setting(settings, "ollama_base_url")
            or "http://localhost:8000/v1"
        )
        raw_api_key = (
            self._setting(settings, "resolved_llm_api_key")
            or self._setting(settings, "llm_api_key")
            or self._setting(settings, "openai_api_key")
        )
        provider = str(self._setting(settings, "llm_provider", "vllm") or "vllm").strip().lower()
        self.llm_provider = provider if provider in {"vllm", "openai", "ollama"} else "vllm"
        self.api_key = (
            str(raw_api_key).strip()
            if raw_api_key is not None and str(raw_api_key).strip()
            else ("EMPTY" if self.llm_provider in {"vllm", "ollama"} else None)
        )
        self.response_verbosity = str(
            self._setting(settings, "llm_response_verbosity", "balanced") or "balanced"
        )
        self.agno_db = self._build_agno_db()
        db_target = str(
            self._setting(settings, "run_store_path", "data/runs.db") or "data/runs.db"
        ).strip()
        self.persistence_db = AgenticDb(db_target)
        self.session_repository = SessionRepository(self.persistence_db)
        self.note_repository = ScientificNoteRepository(self.persistence_db)
        self.memory_service = ScientificMemoryService(
            sessions=self.session_repository,
            notes=self.note_repository,
        )
        self.knowledge_hub = ScientificKnowledgeHub(notes=self.note_repository)
        self.learning_journal = ScientificLearningJournal(notes=self.note_repository)
        self.pro_mode = ProModeWorkflowRunner(
            model_builder=self._build_pro_mode_agent_model,
            fallback_model_builder=self._build_model,
            enable_expert_council=self._pro_mode_expert_council_enabled(),
        )

    def _pro_mode_expert_council_enabled(self) -> bool:
        return bool(self._setting(self.settings, "pro_mode_expert_council_enabled", False))

    def _pro_mode_autonomous_cycle_enabled(self) -> bool:
        return bool(self._setting(self.settings, "pro_mode_autonomous_cycle_enabled", False))

    def _pro_mode_autonomous_cycle_shadow_enabled(self) -> bool:
        return bool(self._setting(self.settings, "pro_mode_autonomous_cycle_shadow_enabled", False))

    def _pro_mode_autonomous_cycle_agno_controller_enabled(self) -> bool:
        return bool(
            self._setting(self.settings, "pro_mode_autonomous_cycle_agno_controller_enabled", True)
        )

    def _pro_mode_autonomous_cycle_max_cycles(self) -> int:
        try:
            value = int(
                self._setting(self.settings, "pro_mode_autonomous_cycle_max_cycles", 12) or 12
            )
        except Exception:
            value = 12
        return max(1, min(32, value))

    def _pro_mode_autonomous_cycle_watchdog_runtime_seconds(self) -> int:
        try:
            value = int(
                self._setting(
                    self.settings,
                    "pro_mode_autonomous_cycle_watchdog_runtime_seconds",
                    1800,
                )
                or 1800
            )
        except Exception:
            value = 1800
        return max(60, min(7200, value))

    def _pro_mode_autonomous_cycle_watchdog_tool_calls(self) -> int:
        try:
            value = int(
                self._setting(
                    self.settings,
                    "pro_mode_autonomous_cycle_watchdog_tool_calls",
                    48,
                )
                or 48
            )
        except Exception:
            value = 48
        return max(1, min(256, value))

    def _pro_mode_autonomous_cycle_phase_timeout_seconds(self) -> int:
        try:
            value = int(
                self._setting(
                    self.settings,
                    "pro_mode_autonomous_cycle_phase_timeout_seconds",
                    240,
                )
                or 240
            )
        except Exception:
            value = 240
        return max(30, min(1800, value))

    def _pro_mode_autonomous_cycle_transport_watchdog_seconds(self) -> int:
        try:
            value = int(
                self._setting(
                    self.settings,
                    "pro_mode_autonomous_cycle_transport_watchdog_seconds",
                    1800,
                )
                or 1800
            )
        except Exception:
            value = 1800
        return max(60, min(7200, value))

    def _build_pro_mode_agent_model(
        self,
        *,
        model_id: str | None = None,
        reasoning_mode: str | None = None,
        reasoning_effort_override: str | None = None,
        max_runtime_seconds: int = 900,
    ) -> AgnoModel:
        if self._uses_published_pro_mode_api():
            return self._build_model(
                model_id=model_id,
                reasoning_mode=reasoning_mode,
                reasoning_effort_override=reasoning_effort_override,
                max_runtime_seconds=max_runtime_seconds,
            )
        return self._build_pro_mode_model(
            model_id=model_id,
            reasoning_mode=reasoning_mode,
            reasoning_effort_override=reasoning_effort_override,
            max_runtime_seconds=max_runtime_seconds,
        )

    def _pro_mode_fallback_enabled(self) -> bool:
        return bool(self._setting(self.settings, "pro_mode_fallback_enabled", True))

    def _uses_dedicated_pro_mode_model(self) -> bool:
        if self._pro_mode_transport() == "aws_bedrock_claude":
            return True
        return any(
            str(self._setting(self.settings, name, "") or "").strip()
            for name in (
                "pro_mode_base_url",
                "pro_mode_api_key",
                "pro_mode_api_key_header",
                "pro_mode_model",
                "pro_mode_default_headers",
                "pro_mode_default_query",
            )
        )

    def _resolved_pro_mode_timeout_seconds(self) -> float:
        try:
            value = float(
                self._setting(self.settings, "resolved_pro_mode_timeout_seconds", 60) or 60
            )
        except Exception:
            value = 60.0
        return max(1.0, value)

    def _pro_mode_transport(self) -> str:
        transport = (
            str(
                self._setting(self.settings, "pro_mode_transport", "openai_compatible")
                or "openai_compatible"
            )
            .strip()
            .lower()
        )
        if transport not in {"openai_compatible", "bedrock_published_api", "aws_bedrock_claude"}:
            return "openai_compatible"
        return transport

    def _uses_published_pro_mode_api(self) -> bool:
        return self._pro_mode_transport() == "bedrock_published_api"

    def _uses_native_bedrock_claude(self) -> bool:
        return self._pro_mode_transport() == "aws_bedrock_claude"

    @staticmethod
    def _string_dict(value: Any) -> dict[str, str]:
        if value is None:
            return {}
        parsed = value
        if isinstance(parsed, str):
            text = parsed.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except Exception:
                return {}
        if not isinstance(parsed, Mapping):
            return {}
        normalized: dict[str, str] = {}
        for raw_key, raw_value in parsed.items():
            key = str(raw_key or "").strip()
            if not key or raw_value is None:
                continue
            normalized[key] = str(raw_value)
        return normalized

    def _resolved_pro_mode_client_overrides(
        self,
        *,
        api_key: str | None,
    ) -> tuple[str, dict[str, str] | None, dict[str, str] | None, str | None]:
        header_name = str(self._setting(self.settings, "pro_mode_api_key_header", "") or "").strip()
        header_prefix_raw = self._setting(self.settings, "pro_mode_api_key_prefix", None)
        headers = self._string_dict(self._setting(self.settings, "pro_mode_default_headers", {}))
        query = self._string_dict(self._setting(self.settings, "pro_mode_default_query", {}))
        effective_api_key = str(api_key or "").strip() or "EMPTY"

        if header_name:
            if header_name.lower() == "authorization" and header_prefix_raw in (
                None,
                "",
                "Bearer",
                "bearer",
            ):
                return effective_api_key, headers or None, query or None, "Authorization"
            prefix = ""
            if header_prefix_raw is None:
                prefix = "Bearer" if header_name.lower() == "authorization" else ""
            else:
                prefix = str(header_prefix_raw).strip()
            if api_key:
                headers.setdefault(
                    header_name,
                    f"{prefix} {str(api_key).strip()}".strip() if prefix else str(api_key).strip(),
                )
            effective_api_key = "EMPTY"

        return effective_api_key, headers or None, query or None, header_name or None

    @staticmethod
    def _published_api_reasoning_enabled(
        *,
        reasoning_mode: str | None,
        reasoning_effort_override: str | None = None,
    ) -> bool:
        normalized_mode = str(reasoning_mode or "").strip().lower()
        normalized_effort = str(reasoning_effort_override or "").strip().lower()
        return normalized_mode == "deep" or normalized_effort == "high"

    def _native_claude_sampling_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        max_tokens = self._setting(self.settings, "pro_mode_max_tokens", None)
        if max_tokens is not None:
            with suppress(TypeError, ValueError):
                kwargs["max_tokens"] = max(256, int(max_tokens))
        temperature = self._setting(self.settings, "pro_mode_temperature", None)
        if temperature is not None:
            with suppress(TypeError, ValueError):
                kwargs["temperature"] = max(0.0, min(1.0, float(temperature)))
        top_p = self._setting(self.settings, "pro_mode_top_p", None)
        if top_p is not None:
            with suppress(TypeError, ValueError):
                kwargs["top_p"] = max(1e-6, min(1.0, float(top_p)))
        top_k = self._setting(self.settings, "pro_mode_top_k", None)
        if top_k is not None:
            with suppress(TypeError, ValueError):
                kwargs["top_k"] = max(1, int(top_k))
        return kwargs

    def _native_claude_thinking_config(
        self,
        *,
        reasoning_mode: str | None,
        reasoning_effort_override: str | None,
    ) -> dict[str, Any] | None:
        if not bool(self._setting(self.settings, "pro_mode_claude_thinking_enabled", True)):
            return None
        if not self._published_api_reasoning_enabled(
            reasoning_mode=reasoning_mode,
            reasoning_effort_override=reasoning_effort_override,
        ):
            return None
        thinking: dict[str, Any] = {"type": "enabled"}
        budget_tokens = self._setting(
            self.settings,
            "pro_mode_claude_thinking_budget_tokens",
            4096,
        )
        try:
            thinking["budget_tokens"] = max(1024, int(budget_tokens))
        except Exception:
            thinking["budget_tokens"] = 4096
        display = (
            str(self._setting(self.settings, "pro_mode_claude_thinking_display", "") or "")
            .strip()
            .lower()
        )
        if display in {"summarized", "omitted"}:
            thinking["display"] = display
        return thinking

    @staticmethod
    def _published_api_extract_text(payload: dict[str, Any] | None) -> str:
        root = dict(payload or {})
        message = root.get("message")
        if isinstance(message, dict):
            root = message
        content_blocks = list(root.get("content") or [])
        parts: list[str] = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("contentType") or "").strip().lower()
            if block_type == "text":
                body = str(block.get("body") or "").strip()
                if body:
                    parts.append(body)
            elif block_type == "reasoning":
                continue
        return "\n".join(part for part in parts if part)

    async def _run_published_pro_mode_prompt(
        self,
        *,
        phase_name: str,
        prompt: str,
        reasoning_mode: str | None,
        reasoning_effort_override: str | None,
        max_runtime_seconds: int,
    ) -> tuple[str, dict[str, Any]]:
        raw_api_key = self._setting(self.settings, "resolved_pro_mode_api_key")
        api_key = str(raw_api_key or "").strip() or None
        _effective_api_key, default_headers, default_query, auth_header_name = (
            self._resolved_pro_mode_client_overrides(api_key=api_key)
        )
        base_url = (
            str(
                self._setting(self.settings, "resolved_pro_mode_base_url", self.base_url)
                or self.base_url
            )
            .strip()
            .rstrip("/")
        )
        resolved_model = (
            str(
                self._setting(self.settings, "resolved_pro_mode_model", self.model) or self.model
            ).strip()
            or self.model
        )
        request_timeout = max(
            self._resolved_pro_mode_timeout_seconds(),
            float(max_runtime_seconds) + 15.0,
        )
        headers = dict(default_headers or {})
        params = dict(default_query or {})
        payload = {
            "message": {
                "model": resolved_model,
                "content": [{"contentType": "text", "body": str(prompt or "")}],
            },
            "continueGenerate": False,
            "enableReasoning": self._published_api_reasoning_enabled(
                reasoning_mode=reasoning_mode,
                reasoning_effort_override=reasoning_effort_override,
            ),
        }

        async with httpx.AsyncClient(timeout=request_timeout) as client:
            create_response = await client.post(
                f"{base_url}/conversation",
                headers=headers,
                params=params,
                json=payload,
            )
            create_response.raise_for_status()
            create_payload = create_response.json()
            conversation_id = str(create_payload.get("conversationId") or "").strip()
            message_id = str(create_payload.get("messageId") or "").strip()
            if not conversation_id or not message_id:
                raise RuntimeError(
                    "Published Pro Mode API did not return conversationId/messageId."
                )
            deadline = time.monotonic() + max(5.0, float(max_runtime_seconds))
            last_payload: dict[str, Any] | None = None
            while True:
                try:
                    message_response = await client.get(
                        f"{base_url}/conversation/{conversation_id}/{message_id}",
                        headers=headers,
                        params=params,
                    )
                    if message_response.status_code == 404:
                        message_response = None
                    else:
                        message_response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    if exc.response is not None and exc.response.status_code == 404:
                        message_response = None
                    else:
                        raise
                if message_response is not None:
                    last_payload = message_response.json()
                    response_text = self._published_api_extract_text(last_payload)
                    if response_text.strip():
                        return response_text, {
                            "conversation_id": conversation_id,
                            "message_id": message_id,
                            "auth_header_name": str(auth_header_name or "").strip()
                            or "Authorization",
                        }
                if time.monotonic() >= deadline:
                    break
                await asyncio.sleep(0.75)
        raise TimeoutError(
            f"Published Pro Mode API timed out waiting for {phase_name} message completion."
        )

    async def _arun_text_phase_with_optional_pro_mode_transport(
        self,
        *,
        phase_name: str,
        prompt: str,
        build_agent: Callable[[Callable[..., AgnoModel]], Agent],
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        debug: bool | None,
        reasoning_mode: str | None,
        reasoning_effort_override: str | None,
        max_runtime_seconds: int,
    ) -> tuple[Any, dict[str, Any]]:
        if not self._uses_published_pro_mode_api():
            return await self._arun_with_optional_pro_mode_fallback(
                phase_name=phase_name,
                prompt=prompt,
                build_agent=build_agent,
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                debug=debug,
            )
        configured_model = (
            str(
                self._setting(self.settings, "resolved_pro_mode_model", self.model) or self.model
            ).strip()
            or self.model
        )
        try:
            response_text, published_meta = await self._run_published_pro_mode_prompt(
                phase_name=phase_name,
                prompt=prompt,
                reasoning_mode=reasoning_mode,
                reasoning_effort_override=reasoning_effort_override,
                max_runtime_seconds=max_runtime_seconds,
            )
            return response_text, {
                **self._pro_mode_model_route_metadata(
                    fallback_used=False,
                    active_model=configured_model,
                ),
                "published_api": published_meta,
            }
        except Exception as exc:
            failure_code = self._classify_pro_mode_failure(exc) or "published_api_failure"
            if not self._pro_mode_fallback_enabled():
                raise
            fallback_agent = build_agent(self._build_model)
            result = await fallback_agent.arun(
                prompt,
                stream=False,
                user_id=user_id,
                session_id=self._scope_session_id(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    run_id=f"{str(run_id or '').strip()}::{phase_name}:published_fallback"
                    if run_id
                    else f"{phase_name}:published_fallback",
                ),
                debug_mode=bool(debug),
            )
            return result, self._pro_mode_model_route_metadata(
                fallback_used=True,
                failure_code=failure_code,
                active_model=self.model,
            )

    @staticmethod
    def _pro_mode_failure_text(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        text = re.sub(r"\s+", " ", text)
        return text.lower()

    @classmethod
    def _classify_pro_mode_failure(cls, value: Any) -> str | None:
        text = cls._pro_mode_failure_text(value)
        if not text:
            return None
        if bool(
            re.search(
                r"\b("
                r"api connection error|connection error|failed to connect|connection refused|"
                r"no route to host|name or service not known|dns|timeout|timed out|read timeout|"
                r"temporarily unavailable|service unavailable|bad gateway|gateway timeout|"
                r"502|503|504|model not found|unknown model|provider returned 5|"
                r"unable to locate credentials|credentials not found|aws credentials not found|"
                r"security token|access denied|forbidden|unauthorized|invalidclienttokenid|"
                r"expiredtoken|profile .* could not be found"
                r")\b",
                text,
            )
        ):
            return "transport_or_availability_failure"
        return None

    def _pro_mode_model_route_metadata(
        self,
        *,
        fallback_used: bool,
        failure_code: str | None = None,
        active_model: str | None = None,
    ) -> dict[str, Any]:
        configured_model = (
            str(
                self._setting(self.settings, "resolved_pro_mode_model", self.model) or self.model
            ).strip()
            or self.model
        )
        auth_header_name = (
            str(self._setting(self.settings, "pro_mode_api_key_header", "") or "").strip()
            or "Authorization"
        )
        if self._uses_native_bedrock_claude():
            auth_header_name = "AWS SigV4 / IAM"
        return {
            "configured_model": configured_model,
            "active_model": str(active_model or configured_model).strip() or configured_model,
            "uses_dedicated_model": self._uses_dedicated_pro_mode_model(),
            "fallback_enabled": self._pro_mode_fallback_enabled(),
            "fallback_used": bool(fallback_used),
            "fallback_reason": str(failure_code or "").strip() or None,
            "transport": self._pro_mode_transport(),
            "auth_header_name": auth_header_name,
            "aws_region_configured": bool(
                str(self._setting(self.settings, "resolved_pro_mode_aws_region", "") or "").strip()
            ),
            "aws_profile_configured": bool(
                str(self._setting(self.settings, "resolved_pro_mode_aws_profile", "") or "").strip()
            ),
            "custom_headers_configured": bool(
                self._string_dict(self._setting(self.settings, "pro_mode_default_headers", {}))
            ),
            "custom_query_configured": bool(
                self._string_dict(self._setting(self.settings, "pro_mode_default_query", {}))
            ),
        }

    @staticmethod
    def _structured_phase_model_routes(
        compression_stats: Mapping[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        routes: dict[str, dict[str, Any]] = {}
        if not isinstance(compression_stats, Mapping):
            return routes
        for raw_phase, raw_payload in compression_stats.items():
            phase_name = str(raw_phase or "").strip()
            if not phase_name or not isinstance(raw_payload, Mapping):
                continue
            route_payload = raw_payload.get("model_route")
            if isinstance(route_payload, Mapping) and route_payload:
                routes[phase_name] = dict(route_payload)
        return routes

    async def _arun_with_optional_pro_mode_fallback(
        self,
        *,
        phase_name: str,
        prompt: str,
        build_agent: Callable[[Callable[..., AgnoModel]], Agent],
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        debug: bool | None,
    ) -> tuple[Any, dict[str, Any]]:
        agent = build_agent(self._build_pro_mode_model)
        try:
            result = await agent.arun(
                prompt,
                stream=False,
                user_id=user_id,
                session_id=self._scope_session_id(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    run_id=f"{str(run_id or '').strip()}::{phase_name}" if run_id else phase_name,
                ),
                debug_mode=bool(debug),
            )
            return result, self._pro_mode_model_route_metadata(
                fallback_used=False,
                active_model=str(
                    self._setting(self.settings, "resolved_pro_mode_model", self.model)
                    or self.model
                ),
            )
        except Exception as exc:
            failure_code = self._classify_pro_mode_failure(exc)
            if not (
                failure_code
                and self._uses_dedicated_pro_mode_model()
                and self._pro_mode_fallback_enabled()
            ):
                raise
            fallback_agent = build_agent(self._build_model)
            result = await fallback_agent.arun(
                prompt,
                stream=False,
                user_id=user_id,
                session_id=self._scope_session_id(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    run_id=f"{str(run_id or '').strip()}::{phase_name}:fallback"
                    if run_id
                    else f"{phase_name}:fallback",
                ),
                debug_mode=bool(debug),
            )
            return result, self._pro_mode_model_route_metadata(
                fallback_used=True,
                failure_code=failure_code,
                active_model=self.model,
            )

    @staticmethod
    def _rtd_requires_challenger(user_text: str, *, task_regime: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        if not lowered:
            return False
        return bool(
            re.search(
                r"\b(relativistic|collider|beam|kinetic energy|total energy|convention|assumption|"
                r"alternative interpretation|counterfactual|which statement|which option|mcq|multiple choice)\b",
                lowered,
            )
        )

    @staticmethod
    def _rtd_problem_frame_for_turn(
        *,
        latest_user_text: str,
        task_regime: str,
    ) -> RTDProblemFrame:
        lowered = str(latest_user_text or "").strip().lower()
        target_quantity = ""
        if re.search(
            r"\b(energy|mass|probability|count|rate|area|volume|distance|time)\b", lowered
        ):
            match = re.search(
                r"\b(energy|mass|probability|count|rate|area|volume|distance|time)\b", lowered
            )
            target_quantity = str(match.group(1) if match else "").strip()
        assumptions: list[str] = []
        if "energy" in lowered and "kinetic" not in lowered and "relativistic" in lowered:
            assumptions.append(
                "Interpret bare `energy` as total relativistic energy unless the prompt says kinetic."
            )
        if "defined as" in lowered or "knowing that" in lowered:
            assumptions.append(
                "Infer quantities from the composition explicitly given in the prompt before using external conventions."
            )
        constraints: list[str] = []
        precision_match = re.search(r"precision[^0-9]*([0-9]+e[-+]?[0-9]+|1e[-+]?[0-9]+)", lowered)
        if precision_match:
            constraints.append(
                f"Respect the requested numeric precision: {precision_match.group(1)}."
            )
        return RTDProblemFrame(
            objective=str(latest_user_text or "").strip(),
            task_type=str(task_regime or "conceptual_high_uncertainty").strip()
            or "conceptual_high_uncertainty",
            target_quantity=target_quantity,
            assumptions=assumptions[:6],
            constraints=constraints[:6],
        )

    @classmethod
    def _rtd_skill_from_context(
        cls,
        *,
        latest_user_text: str,
        task_regime: str,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
    ) -> RTDSkill:
        lowered = str(latest_user_text or "").strip().lower()
        has_artifacts = bool(uploaded_files) or bool(
            dict(selection_context or {}).get("resource_uris")
            or dict(selection_context or {}).get("dataset_uris")
        )
        if task_regime == "rigorous_proof" or bool(
            re.search(
                r"\b(prove|proof|derive rigorously|show that|theorem|lemma|corollary)\b", lowered
            )
        ):
            return "proof_derivation_workflow"
        if bool(
            re.search(
                r"\b(simulat|enumerat|sweep|search over|optimi[sz]e|table|program|write code)\b",
                lowered,
            )
        ):
            return "programmatic_experiment_workflow"
        if has_artifacts or task_regime in {
            "artifact_interpretation",
            "dataset_or_catalog_research",
            "iterative_multimodal_research",
        }:
            return "evidence_review_workflow"
        if bool(
            re.search(
                r"\b(report|concise report|write a report|survey|compare approaches|landscape|synthesis)\b",
                lowered,
            )
        ):
            return "focused_synthesis_team_workflow"
        if bool(
            re.search(
                r"\b(calculate|compute|derive|solve for|probability|energy|mass|count|rate|evaluate|exact)\b",
                lowered,
            )
        ):
            return "deterministic_numeric_workflow"
        return "evidence_review_workflow"

    @staticmethod
    def _rtd_skill_to_action(skill: RTDSkill) -> AutonomousCycleAction:
        if skill == "focused_synthesis_team_workflow":
            return "focused_team"
        if skill in {
            "deterministic_numeric_workflow",
            "evidence_review_workflow",
            "programmatic_experiment_workflow",
        }:
            return "tool_workflow"
        return "reasoning_solver"

    @staticmethod
    def _rtd_leader_text(candidate_set: RTDCandidateSet | dict[str, Any] | None) -> str:
        if isinstance(candidate_set, RTDCandidateSet):
            return str(candidate_set.leader.answer_text or "").strip()
        payload = dict(candidate_set or {})
        return str(dict(payload.get("leader") or {}).get("answer_text") or "").strip()

    @staticmethod
    def _rtd_verification_satisfied(
        state: RTDEpistemicState,
        *,
        requires_challenger: bool,
    ) -> bool:
        if not requires_challenger:
            return True
        if not list(state.verification_ledger or []):
            return False
        latest = state.verification_ledger[-1]
        return bool(latest.leader_survived)

    def _coerce_rtd_state(
        self,
        *,
        autonomy_state_seed: dict[str, Any] | None,
        latest_user_text: str,
        task_regime: str,
        cycles_completed: int,
        tool_families_used: list[str],
    ) -> RTDEpistemicState:
        seed = dict(autonomy_state_seed or {})
        raw_v2 = dict(seed.get("autonomy_state_v2") or {})
        if raw_v2:
            try:
                return RTDEpistemicState.model_validate(raw_v2)
            except Exception:
                pass
        raw_autonomy = self._saved_autonomy_state(seed)
        candidate_answer = str(raw_autonomy.get("candidate_answer") or "").strip()
        leader = RTDCandidate(
            answer_text=candidate_answer,
            rationale="Recovered from saved autonomy state.",
            confidence=0.8 if candidate_answer else 0.0,
        )
        challenger = RTDCandidate()
        requires_challenger = self._rtd_requires_challenger(
            latest_user_text, task_regime=task_regime
        )
        if requires_challenger and not challenger.answer_text:
            challenger = RTDCandidate(
                answer_text="",
                rationale="Reserved for counterfactual or alternative-convention testing.",
                confidence=0.0,
            )
        return RTDEpistemicState(
            controller_mode="rtd_v1",
            problem_frame=self._rtd_problem_frame_for_turn(
                latest_user_text=latest_user_text,
                task_regime=task_regime,
            ),
            candidate_set=RTDCandidateSet(leader=leader, challenger=challenger),
            obligation_ledger=[
                str(item or "").strip()
                for item in list(raw_autonomy.get("open_obligations") or [])
                if str(item or "").strip()
            ][:12],
            evidence_ledger=[
                str(item or "").strip()
                for item in list(raw_autonomy.get("evidence_ledger") or [])
                if str(item or "").strip()
            ][-24:],
            verification_ledger=[],
            checkpoint=RTDCheckpoint(
                cycle_index=int(raw_autonomy.get("checkpoint_index") or cycles_completed),
                stop_reason=str(raw_autonomy.get("stop_reason") or "").strip(),
                resume_readiness=str(raw_autonomy.get("resume_readiness") or "").strip() or "ready",
                next_best_actions=[
                    str(item or "").strip()
                    for item in list(raw_autonomy.get("next_best_actions") or [])
                    if str(item or "").strip()
                ][:8],
            ),
            budget_state=RTDBudgetState(
                cycles_used=int(raw_autonomy.get("cycles_completed") or cycles_completed),
                tool_families_used=list(tool_families_used),
                model_calls=0,
                verification_count=0,
            ),
            evidence_sufficiency_score=0.0,
            continuation_fidelity=float(raw_autonomy.get("continuation_fidelity") or 0.0),
        )

    def _default_pro_mode_deep_execution_regime(self) -> str:
        return "reasoning_solver"

    def _build_agno_db(self) -> SqliteDb | PostgresDb:
        db_target = str(
            self._setting(self.settings, "run_store_path", "data/runs.db") or "data/runs.db"
        ).strip()
        table_kwargs = {
            "session_table": "agno_agent_sessions",
            "memory_table": "agno_agent_memories",
            "metrics_table": "agno_agent_metrics",
            "approvals_table": "agno_agent_approvals",
        }
        if db_target.lower().startswith(("postgres://", "postgresql://")):
            normalized_target = db_target
            if normalized_target.startswith("postgres://"):
                normalized_target = (
                    "postgresql+psycopg://" + normalized_target[len("postgres://") :]
                )
            elif normalized_target.startswith("postgresql://"):
                normalized_target = (
                    "postgresql+psycopg://" + normalized_target[len("postgresql://") :]
                )
            return PostgresDb(db_engine=create_engine(normalized_target), **table_kwargs)
        return SqliteDb(db_file=db_target, **table_kwargs)

    @staticmethod
    def _normalize_selected_tool_names(selected_tool_names: list[str] | None) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for raw in selected_tool_names or []:
            token = str(raw or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            ordered.append(token)
        return ordered

    @classmethod
    def _is_image_like_path(cls, path: str | None) -> bool:
        token = str(path or "").strip().lower()
        if not token:
            return False
        return any(token.endswith(suffix) for suffix in cls.IMAGE_LIKE_SUFFIXES)

    @classmethod
    def _image_like_files(cls, paths: list[str] | None) -> list[str]:
        return [
            str(path or "").strip()
            for path in list(paths or [])
            if str(path or "").strip() and cls._is_image_like_path(str(path or ""))
        ]

    @staticmethod
    def _prompt_image_paths(user_text: str) -> list[str]:
        return extract_scientific_image_paths_from_text(user_text)

    @classmethod
    def _has_direct_image_target(
        cls,
        *,
        user_text: str,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
    ) -> bool:
        return bool(
            cls._image_like_files(uploaded_files)
            or cls._prompt_image_paths(user_text)
            or cls._selection_context_supports_direct_image_analysis(
                user_text=user_text,
                selection_context=selection_context,
            )
        )

    @staticmethod
    def _latest_user_text(messages: list[dict[str, Any]] | list[Message]) -> str:
        for raw in reversed(messages):
            role = ""
            content: Any = ""
            if isinstance(raw, Message):
                role = str(raw.role or "")
                content = raw.content
            elif isinstance(raw, dict):
                role = str(raw.get("role") or "")
                content = raw.get("content") or ""
            if role.strip().lower() != "user":
                continue
            text = str(content or "").strip()
            if text:
                return text
        return ""

    @classmethod
    def _reasoning_effort_for_mode(cls, reasoning_mode: str | None) -> str:
        normalized = str(reasoning_mode or "auto").strip().lower()
        if normalized == "fast":
            return "low"
        if normalized == "deep":
            # Medium is a better stability/latency tradeoff for our vLLM-backed gpt-oss path.
            return "medium"
        return "low"

    def _verbosity_for_response(self) -> str:
        normalized = self.response_verbosity.strip().lower()
        if normalized == "concise":
            return "low"
        if normalized == "balanced":
            return "medium"
        return "high"

    def _build_model(
        self,
        *,
        model_id: str | None = None,
        reasoning_mode: str | None = None,
        reasoning_effort_override: str | None = None,
        max_runtime_seconds: int = 900,
    ) -> AgnoModel:
        timeout_seconds = max(
            float(self._setting(self.settings, "openai_timeout", 60) or 60),
            float(max_runtime_seconds) + 15.0,
        )
        reasoning_effort = (
            str(reasoning_effort_override or self._reasoning_effort_for_mode(reasoning_mode))
            .strip()
            .lower()
        )
        if reasoning_effort not in {"low", "medium", "high"}:
            reasoning_effort = self._reasoning_effort_for_mode(reasoning_mode)
        return OpenAILike(
            id=str(
                model_id
                or self.model
                or self._setting(self.settings, "openai_model", "gpt-oss-120b")
            ),
            api_key=self.api_key or "EMPTY",
            base_url=self.base_url,
            timeout=timeout_seconds,
            max_retries=0,
            reasoning_effort=reasoning_effort,
            verbosity=self._verbosity_for_response(),
            max_tokens=None,
            max_completion_tokens=None,
        )

    def _build_native_bedrock_claude_model(
        self,
        *,
        model_id: str | None = None,
        reasoning_mode: str | None = None,
        reasoning_effort_override: str | None = None,
        max_runtime_seconds: int = 900,
    ) -> AgnoModel:
        try:
            from agno.models.aws import Claude as AwsBedrockClaude
        except ImportError as exc:  # pragma: no cover - exercised in smoke/local env
            raise RuntimeError(
                "Native AWS Bedrock Claude transport requires boto3 and anthropic[bedrock]. "
                "Run `uv sync` after pulling the updated production repo."
            ) from exc

        profile = (
            str(self._setting(self.settings, "resolved_pro_mode_aws_profile", "") or "").strip()
            or None
        )
        region = (
            str(self._setting(self.settings, "resolved_pro_mode_aws_region", "") or "").strip()
            or None
        )
        access_key = (
            str(
                self._setting(self.settings, "resolved_pro_mode_aws_access_key_id", "") or ""
            ).strip()
            or None
        )
        secret_key = (
            str(
                self._setting(self.settings, "resolved_pro_mode_aws_secret_access_key", "") or ""
            ).strip()
            or None
        )
        session_token = (
            str(
                self._setting(self.settings, "resolved_pro_mode_aws_session_token", "") or ""
            ).strip()
            or None
        )
        bearer_token = (
            str(
                self._setting(self.settings, "resolved_pro_mode_aws_bearer_token", "") or ""
            ).strip()
            or None
        )
        use_sso = bool(self._setting(self.settings, "pro_mode_aws_sso_auth", False))
        timeout_seconds = max(
            self._resolved_pro_mode_timeout_seconds(),
            float(max_runtime_seconds) + 15.0,
        )
        resolved_model = (
            str(
                self._setting(self.settings, "resolved_pro_mode_model", self.model) or self.model
            ).strip()
            or self.model
        )

        session = None
        if profile or use_sso:
            try:
                from boto3.session import Session
            except ImportError as exc:  # pragma: no cover - exercised in smoke/local env
                raise RuntimeError(
                    "AWS profile support for native Bedrock Claude requires boto3. "
                    "Run `uv sync` after pulling the updated production repo."
                ) from exc
            session_kwargs: dict[str, Any] = {}
            if profile:
                session_kwargs["profile_name"] = profile
            if region:
                session_kwargs["region_name"] = region
            session = Session(**session_kwargs)
        use_bearer_token = bool(
            bearer_token
            and session is None
            and not access_key
            and not secret_key
            and not session_token
        )

        kwargs: dict[str, Any] = {
            "id": str(model_id or resolved_model),
            "timeout": timeout_seconds,
        }
        kwargs.update(self._native_claude_sampling_kwargs())
        thinking = self._native_claude_thinking_config(
            reasoning_mode=reasoning_mode,
            reasoning_effort_override=reasoning_effort_override,
        )
        if thinking is not None:
            kwargs["thinking"] = thinking
        if session is not None:
            kwargs["session"] = session
            if region:
                kwargs["aws_region"] = region
        elif use_bearer_token:
            try:
                from anthropic.lib.bedrock import AnthropicBedrock, AsyncAnthropicBedrock
            except ImportError as exc:  # pragma: no cover - exercised in smoke/local env
                raise RuntimeError(
                    "Native AWS Bedrock bearer-token auth requires anthropic[bedrock]. "
                    "Run `uv sync` after pulling the updated production repo."
                ) from exc
            client_kwargs: dict[str, Any] = {
                "api_key": bearer_token,
                "timeout": timeout_seconds,
            }
            if region:
                client_kwargs["aws_region"] = region
            kwargs["client"] = AnthropicBedrock(**client_kwargs)
            kwargs["async_client"] = AsyncAnthropicBedrock(**client_kwargs)
            if region:
                kwargs["aws_region"] = region
        else:
            if region:
                kwargs["aws_region"] = region
            if access_key:
                kwargs["aws_access_key"] = access_key
            if secret_key:
                kwargs["aws_secret_key"] = secret_key
            if session_token:
                kwargs["aws_session_token"] = session_token
        return AwsBedrockClaude(**kwargs)

    def _build_pro_mode_model(
        self,
        *,
        model_id: str | None = None,
        reasoning_mode: str | None = None,
        reasoning_effort_override: str | None = None,
        max_runtime_seconds: int = 900,
    ) -> AgnoModel:
        if self._uses_native_bedrock_claude():
            return self._build_native_bedrock_claude_model(
                model_id=model_id,
                reasoning_mode=reasoning_mode,
                reasoning_effort_override=reasoning_effort_override,
                max_runtime_seconds=max_runtime_seconds,
            )
        timeout_seconds = max(
            self._resolved_pro_mode_timeout_seconds(),
            float(max_runtime_seconds) + 15.0,
        )
        reasoning_effort = (
            str(reasoning_effort_override or self._reasoning_effort_for_mode(reasoning_mode))
            .strip()
            .lower()
        )
        if reasoning_effort not in {"low", "medium", "high"}:
            reasoning_effort = self._reasoning_effort_for_mode(reasoning_mode)
        raw_api_key = self._setting(self.settings, "resolved_pro_mode_api_key")
        if raw_api_key is not None and str(raw_api_key).strip():
            api_key = str(raw_api_key).strip()
        else:
            api_key = (
                "EMPTY" if self.llm_provider in {"vllm", "ollama"} else (self.api_key or "EMPTY")
            )
        effective_api_key, default_headers, default_query, _auth_header_name = (
            self._resolved_pro_mode_client_overrides(api_key=api_key)
        )
        base_url = (
            str(
                self._setting(self.settings, "resolved_pro_mode_base_url", self.base_url)
                or self.base_url
            ).strip()
            or self.base_url
        )
        resolved_model = (
            str(
                self._setting(self.settings, "resolved_pro_mode_model", self.model) or self.model
            ).strip()
            or self.model
        )
        return OpenAILike(
            id=str(model_id or resolved_model),
            api_key=effective_api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=0,
            reasoning_effort=reasoning_effort,
            verbosity=self._verbosity_for_response(),
            max_tokens=None,
            max_completion_tokens=None,
            default_headers=default_headers,
            default_query=default_query,
        )

    @staticmethod
    def _numeric_tool_guidance(user_text: str) -> list[str]:
        lowered = str(user_text or "").strip().lower()
        guidance: list[str] = []
        if not lowered:
            return guidance
        guidance.append(
            "For closed-form scientific numeric questions, do a brief assumption audit before calculating: "
            "identify the target quantity, the governing formula, the constants/conventions implied by the prompt, "
            "and whether the user asked for total energy, kinetic energy, a difference, or a rate."
        )
        if re.search(
            r"\b(relativistic|collider|beam|rhic|lhc|synchrotron|beta|gamma|0\.\d+\s*c|speed .*c)\b",
            lowered,
        ):
            guidance.append(
                "For relativistic beam or collider questions, treat plain `energy` as total relativistic energy "
                "`E = gamma m c^2` unless the user explicitly asks for kinetic energy."
            )
        if re.search(r"\b(nucleus|nucl(eus|ide)|ion|isotope|neutron|proton)\b", lowered):
            guidance.append(
                "If a nucleus or ion is specified only by proton/neutron counts or elemental identity plus neutron count, "
                "prefer the composition-based mass convention supported directly by the prompt instead of importing a tabulated isotope mass. "
                "For beam-energy back-of-the-envelope questions where no isotope mass is supplied, it is usually better to use the mass-number / nucleon approximation directly implied by the prompt, i.e. `m ≈ A * m_n`, and say that approximation explicitly rather than substituting `A * u`."
            )
        if re.search(
            r"\b(relativistic|collider|beam|rhic|lhc|synchrotron)\b", lowered
        ) and re.search(r"\b(nucleus|ion|isotope|neutron|proton)\b", lowered):
            guidance.append(
                "For textbook relativistic beam questions that specify a nucleus by composition but do not provide an exact isotope mass, "
                "default to the rounded nucleon rest-energy constant `m_n c^2 ≈ 939.5 MeV` per nucleon unless the user explicitly asks for higher-fidelity mass modeling."
            )
        if re.search(r"\b(precision|1e-|10\^-\d+|significant|decimal places?)\b", lowered):
            guidance.append(
                "Carry intermediate constants with enough precision to satisfy the requested tolerance, and round only in the final reported value."
            )
        return guidance

    @staticmethod
    def _prefers_textbook_nucleon_rest_energy(user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        if not lowered:
            return False
        has_relativistic_beam_context = bool(
            re.search(
                r"\b(relativistic|collider|beam|rhic|lhc|synchrotron|beta|gamma|speed .*c|0\.\d+\s*c)\b",
                lowered,
            )
        )
        has_nuclear_composition = bool(
            re.search(r"\b(nucleus|ion|isotope|neutron|proton)\b", lowered)
        )
        asks_for_exact_mass_model = bool(
            re.search(
                r"\b(exact mass|precise mass|tabulated mass|atomic mass|nuclear mass|mass defect|binding energy|amu|u\b)\b",
                lowered,
            )
        )
        return (
            has_relativistic_beam_context
            and has_nuclear_composition
            and not asks_for_exact_mass_model
        )

    @classmethod
    def _normalize_validated_numeric_expression(cls, expression: str, *, user_text: str) -> str:
        normalized = str(expression or "").strip()
        if not normalized:
            return normalized
        if cls._prefers_textbook_nucleon_rest_energy(user_text):
            normalized = re.sub(r"\b939\.565(?:4133?)?\b", "939.5", normalized)
        return normalized

    @classmethod
    def _validated_numeric_system_message(cls, user_text: str) -> str:
        guidance = cls._numeric_tool_guidance(user_text)
        if not guidance:
            return ""
        return "Closed-form numeric workflow guidance:\n" + "\n".join(
            f"- {item}" for item in guidance
        )

    def _tool_instructions(self, tool_names: list[str], *, user_text: str = "") -> list[str]:
        if not tool_names:
            return []
        instructions = [
            "Only call a tool when it directly helps answer the current request.",
            "Only use the explicitly enabled tools for this turn.",
            "After tool use, explain the result in scientist-friendly prose.",
        ]
        if "numpy_calculator" in tool_names:
            instructions.append(
                "Use numpy_calculator for deterministic numeric work when the formula is known. "
                "You may call it multiple times for intermediate computations, but keep the final answer concise."
            )
            instructions.extend(self._numeric_tool_guidance(user_text))
        if "formula_balance_check" in tool_names:
            instructions.append(
                "For formula balance checks, evaluate the equation exactly as written first. "
                "If it is unbalanced, report that directly. Only search for corrected coefficients "
                "if the user explicitly asks for a balanced form."
            )
        if {
            "search_bisque_resources",
            "bisque_find_assets",
            "load_bisque_resource",
            "bisque_download_dataset",
            "bisque_download_resource",
            "bisque_fetch_xml",
        }.intersection(tool_names):
            instructions.extend(
                [
                    "For BisQue catalog answers, use only the tool results from this turn and do not invent public collections, sample datasets, or repository contents.",
                    "When presenting a BisQue asset to the user, prefer client_view_url when available and avoid raw data_service resource URIs in visible prose unless the user explicitly asks for the raw URI.",
                    "If a BisQue search returns zero results, say that directly and keep the answer short instead of padding it with generic portal instructions.",
                ]
            )
        return instructions

    @staticmethod
    def _is_report_like_request(user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        return bool(
            re.search(
                r"\b("
                r"report|write up|write a report|scientist-facing|ecological analysis|"
                r"research summary|survey|overview|backgrounder|deep dive|primer|"
                r"technical analysis|methods discussion|methods section|discussion section|"
                r"technical note|graduate-level analysis"
                r")\b",
                lowered,
            )
        )

    @staticmethod
    def _is_math_explanation_request(user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        return bool(
            re.search(
                r"\b("
                r"math(?:ematics)?(?:\s+behind)?|mathematical|equation|equations|notation|derive|derivation|"
                r"formal(?:ly)?|proof|show the math|why the math|mechanics of the formula|"
                r"attention formula|loss function|complexity analysis"
                r")\b",
                lowered,
            )
        )

    def _image_like_uploaded_names(self, uploaded_files: list[str]) -> set[str]:
        image_names: set[str] = set()
        for path in list(uploaded_files or []):
            value = str(path or "").strip()
            if not value:
                continue
            lowered = value.lower()
            if any(lowered.endswith(suffix) for suffix in self.IMAGE_LIKE_SUFFIXES):
                image_names.add(os.path.basename(value))
        return image_names

    @staticmethod
    def _display_artifact_name(value: str | None) -> str:
        name = os.path.basename(str(value or "").strip())
        if not name:
            return ""
        parts = [segment for segment in name.split("__") if segment]
        while len(parts) > 1 and re.fullmatch(r"(?=.*\d)[A-Za-z0-9-]{8,}", parts[0] or ""):
            parts = parts[1:]
        return "__".join(parts) if parts else name

    @staticmethod
    def _report_text_has_internal_orchestration(text: str) -> bool:
        lowered = str(text or "").strip().lower()
        return bool(
            re.search(
                r"\b(provisional|blocker|canonical branch|remaining blocker|best current answer|iterative research program)\b",
                lowered,
            )
        )

    def _base_instructions(self, *, tool_names: list[str], user_text: str = "") -> list[str]:
        instructions = [
            "You are a scientist-facing research assistant.",
            "Answer directly and clearly, with enough detail to match the question.",
            "Match the response length to the user's request.",
            "Use one or two sentences for simple acknowledgements, recall questions, and straightforward factual prompts unless the user explicitly asks for more detail.",
            "When the user asks for a final answer, state it explicitly near the beginning.",
            "If no tools are enabled, solve the task directly with the model.",
        ]
        instructions.extend(self._tool_instructions(tool_names, user_text=user_text))
        return instructions

    def _scope_session_id(
        self,
        *,
        conversation_id: str | None,
        user_id: str | None,
        run_id: str | None,
    ) -> str:
        root = str(conversation_id or "").strip() or str(run_id or "").strip() or "ephemeral"
        owner = str(user_id or "").strip() or "anonymous"
        return f"{owner}::{root}"

    @staticmethod
    def _analysis_state_metadata_session_id(
        *,
        conversation_id: str | None,
        user_id: str | None,
        run_id: str | None,
    ) -> str | None:
        root = str(conversation_id or "").strip() or str(run_id or "").strip()
        if not root:
            return None
        owner = str(user_id or "").strip() or "anonymous"
        return f"analysis::{owner}::{root}"

    @staticmethod
    def _pro_mode_state_metadata_session_id(
        *,
        conversation_id: str | None,
        user_id: str | None,
        run_id: str | None,
    ) -> str | None:
        root = str(conversation_id or "").strip() or str(run_id or "").strip()
        if not root:
            return None
        owner = str(user_id or "").strip() or "anonymous"
        return f"pro-mode::{owner}::{root}"

    def _pro_mode_state_cache_path(self, *, session_id: str) -> Path:
        db_target = str(
            self._setting(self.settings, "run_store_path", "data/runs.db") or "data/runs.db"
        ).strip()
        if db_target.lower().startswith(("postgres://", "postgresql://", "postgresql+psycopg://")):
            base_dir = (Path.cwd() / "data").resolve()
        else:
            base_dir = Path(db_target).expanduser().resolve().parent
        target_dir = base_dir / "pro_mode_state_cache"
        target_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(str(session_id or "").encode("utf-8")).hexdigest()[:24]
        return target_dir / f"{digest}.json"

    def _load_pro_mode_state_cache(self, *, session_id: str) -> dict[str, Any]:
        try:
            payload = json.loads(
                self._pro_mode_state_cache_path(session_id=session_id).read_text(encoding="utf-8")
            )
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        cached_session_id = str(payload.get("session_id") or "").strip()
        if cached_session_id and cached_session_id != str(session_id or "").strip():
            return {}
        state = payload.get("state")
        return dict(state or {}) if isinstance(state, dict) else {}

    def _save_pro_mode_state_cache(self, *, session_id: str, state: dict[str, Any]) -> None:
        if not str(session_id or "").strip() or not isinstance(state, dict) or not state:
            return
        try:
            self._pro_mode_state_cache_path(session_id=session_id).write_text(
                json.dumps(
                    {
                        "session_id": str(session_id or "").strip(),
                        "saved_at": time.time(),
                        "state": dict(state),
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        except Exception:
            return

    def _build_turn_intent(
        self,
        *,
        user_text: str,
        uploaded_files: list[str],
        selected_tool_names: list[str] | None,
        workflow_hint: dict[str, Any] | None,
        selection_context: dict[str, Any] | None,
        knowledge_context: dict[str, Any] | None,
    ) -> AgnoTurnIntent:
        return AgnoTurnIntent(
            user_text=str(user_text or "").strip(),
            uploaded_files=[
                str(item or "").strip() for item in uploaded_files if str(item or "").strip()
            ],
            selected_tool_names=self._normalize_selected_tool_names(selected_tool_names),
            workflow_hint=dict(workflow_hint or {}),
            selection_context=dict(selection_context or {}),
            knowledge_context=dict(knowledge_context or {}),
        )

    @staticmethod
    def _is_phatic_turn(user_text: str) -> bool:
        normalized = " ".join(str(user_text or "").strip().lower().split())
        return normalized in {
            "hi",
            "hello",
            "hello!",
            "hi!",
            "hey",
            "hey!",
            "thanks",
            "thanks!",
            "thank you",
            "thank you!",
            "thx",
            "ok",
            "okay",
            "sounds good",
        }

    @staticmethod
    def _is_lightweight_numeric_request(user_text: str) -> bool:
        lowered = " ".join(str(user_text or "").strip().lower().split())
        if not lowered:
            return False
        if re.search(
            r"\b("
            r"csv|tsv|parquet|dataframe|dataset|spreadsheet|table|file|files|folder|directory|"
            r"plot|figure|chart|save|export|report|image|images|mask|segmentation|random forest|"
            r"cross[- ]?validation|train|fit|classifier|regressor|cluster|clustering|pca|umap|"
            r"opencv|scikit-learn|sklearn|pandas|scipy|pipeline|batch"
            r")\b",
            lowered,
        ):
            return False
        asks_for_closed_form_value = bool(
            re.search(
                r"\b("
                r"calculate|compute|derive|evaluate|solve|determine|find|what is|what's|"
                r"mean|average|median|sum|difference|product|variance|standard deviation|std|"
                r"covariance|correlation|dot product|norm|determinant|eigenvalue|integral|derivative"
                r")\b",
                lowered,
            )
        )
        has_inline_expression = bool(
            re.search(r"(\[[^\]]+\]|\([^\)]*\)|\{[^\}]*\}|[-+]?\d+(?:\.\d+)?|=)", lowered)
        )
        return asks_for_closed_form_value and has_inline_expression and len(lowered) <= 220

    @staticmethod
    def _is_code_execution_request(user_text: str) -> bool:
        lowered = " ".join(str(user_text or "").strip().lower().split())
        if not lowered or AgnoChatRuntime._is_lightweight_numeric_request(lowered):
            return False
        explicit_code_signal = bool(
            re.search(
                r"\b("
                r"write|generate|create|produce|run|execute|debug|fix|test|profile|benchmark|optimi[sz]e"
                r")\s+(python|code|script|program|notebook)\b",
                lowered,
            )
        )
        heavy_analysis_signal = bool(
            re.search(
                r"\b("
                r"random forest|xgboost|lightgbm|svm|logistic regression|linear regression|"
                r"cross[- ]?validation|grid search|hyperparameter|feature importance|confusion matrix|"
                r"roc|auc|precision-recall|bootstrap|monte carlo|principal component|pca|umap|"
                r"cluster(?:ing)?|classifier|regressor|image processing|computer vision|opencv|"
                r"scikit-learn|sklearn"
                r")\b",
                lowered,
            )
        )
        scientific_stack_signal = bool(
            re.search(
                r"\b("
                r"python|script|sandbox|notebook|numpy|pandas|scipy|matplotlib|seaborn|"
                r"scikit-learn|sklearn|opencv"
                r")\b",
                lowered,
            )
        )
        dataset_or_artifact_signal = bool(
            re.search(
                r"\b("
                r"csv|tsv|parquet|dataframe|dataset|spreadsheet|table|file|files|folder|directory|"
                r"plot|figure|chart|report|artifact|png|pdf"
                r")\b",
                lowered,
            )
        )
        workflow_signal = bool(
            re.search(
                r"\b("
                r"fit|train|benchmark|profile|process|analy[sz]e|classify|segment|detect|featurize|"
                r"preprocess|clean|aggregate|transform|simulate|save|export|compare"
                r")\b",
                lowered,
            )
        )
        return explicit_code_signal or heavy_analysis_signal or (
            workflow_signal and (scientific_stack_signal or dataset_or_artifact_signal)
        )

    @staticmethod
    def _is_plot_or_visual_analysis_request(user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        return bool(
            re.search(
                r"\b("
                r"plot|plots|chart|charts|figure|figures|graph|graphs|histogram|histograms|"
                r"bar chart|bar plot|scatter|scatterplot|box plot|violin plot|heatmap|"
                r"visuali[sz]e|visuali[sz]ation|seaborn|matplotlib"
                r")\b",
                lowered,
            )
        )

    @staticmethod
    def _plot_request_prefers_detection_artifacts(user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        if not AgnoChatRuntime._is_plot_or_visual_analysis_request(lowered):
            return False
        detection_signals = bool(
            re.search(
                r"\b("
                r"confidence|confidences|low-confidence|uncertain prediction|uncertain predictions|"
                r"prediction|predictions|detection|detections|bounding box|bounding boxes|"
                r"review threshold|false positive|false positives|false negative|false negatives"
                r")\b",
                lowered,
            )
        )
        image_structure_signals = bool(
            re.search(
                r"\b("
                r"intensity|pixel|pixels|channel|channels|rgb|grayscale|greyscale|brightness|"
                r"illumination|contrast|value distribution|low-value|low value|tail|outlier|"
                r"outliers|pca|principal component|principal-component|image region|image regions|"
                r"region comparison|histogram of the image"
                r")\b",
                lowered,
            )
        )
        return detection_signals and not image_structure_signals

    @staticmethod
    def _selection_context_has_artifact_handles(
        selection_context: dict[str, Any] | None,
    ) -> bool:
        payload = dict(selection_context or {})
        raw_handles = payload.get("artifact_handles")
        if not isinstance(raw_handles, dict):
            return False
        return any(
            bool(list(value or [])) for value in raw_handles.values() if isinstance(value, list)
        )

    @classmethod
    def _selection_context_image_handle_paths(
        cls,
        selection_context: dict[str, Any] | None,
    ) -> list[str]:
        payload = dict(selection_context or {})
        raw_handles = payload.get("artifact_handles")
        if not isinstance(raw_handles, dict):
            return []
        candidates: list[str] = []
        for key in ("image_files", "downloaded_files", "preview_paths"):
            values = raw_handles.get(key)
            if not isinstance(values, list):
                continue
            for raw_value in values:
                token = str(raw_value or "").strip()
                if token and cls._is_image_like_path(token) and token not in candidates:
                    candidates.append(token)
        return candidates

    @classmethod
    def _selection_context_supports_direct_image_analysis(
        cls,
        *,
        user_text: str,
        selection_context: dict[str, Any] | None,
    ) -> bool:
        if cls._selection_context_image_handle_paths(selection_context):
            return True
        payload = dict(selection_context or {})
        resource_uris = [
            str(value or "").strip()
            for value in list(payload.get("resource_uris") or [])
            if str(value or "").strip()
        ]
        dataset_uris = [
            str(value or "").strip()
            for value in list(payload.get("dataset_uris") or [])
            if str(value or "").strip()
        ]
        if not resource_uris or dataset_uris:
            return False
        lowered = str(user_text or "").strip().lower()
        if (
            cls._is_depth_request(user_text)
            or cls._is_segmentation_request(user_text)
            or re.search(
                r"\b(yolo|detect|detection|bounding box|bounding boxes|bbox|spectral|instability|active learning|hard sample)\b",
                lowered,
            )
        ):
            return True
        if cls._is_image_metadata_request(user_text):
            return bool(
                re.search(
                    r"\b("
                    r"image|images|photo|picture|frame|scan|slide|micrograph|microscopy|"
                    r"camera|exif|gps|geo(?:tag|location)?|location|coordinates?|latitude|longitude"
                    r")\b",
                    lowered,
                )
            )
        return False

    @staticmethod
    def _requested_confidence_threshold(user_text: str, *, default: float = 0.6) -> float:
        lowered = str(user_text or "").strip().lower()
        percent_match = re.search(
            r"\b(?:below|under|less than|underneath|highlight(?:ing)?|flag(?:ging)?).{0,32}?(\d{1,3}(?:\.\d+)?)\s*%",
            lowered,
        )
        if percent_match:
            try:
                value = float(percent_match.group(1)) / 100.0
                return max(0.0, min(1.0, value))
            except Exception:
                return default
        decimal_match = re.search(
            r"\b(?:below|under|less than|highlight(?:ing)?|flag(?:ging)?).{0,32}?(0(?:\.\d+)?|1(?:\.0+)?)\b",
            lowered,
        )
        if decimal_match:
            try:
                value = float(decimal_match.group(1))
                return max(0.0, min(1.0, value))
            except Exception:
                return default
        return default

    @staticmethod
    def _should_prefer_inline_code_execution(
        user_text: str,
        *,
        handles: dict[str, Any] | None = None,
    ) -> bool:
        handle_map = dict(handles or {})
        if AgnoChatRuntime._is_plot_or_visual_analysis_request(user_text) and (
            handle_map.get("analysis_table_paths")
            or handle_map.get("prediction_json_paths")
            or handle_map.get("image_files")
        ):
            return True
        lowered = str(user_text or "").strip().lower()
        return bool(
            re.search(
                r"\b(histogram|distribution|box plot|boxplot|scatter|scatterplot|bar chart|bar plot|review plot|confidence plot)\b",
                lowered,
            )
            and (
                handle_map.get("analysis_table_paths")
                or handle_map.get("prediction_json_paths")
                or handle_map.get("image_files")
            )
        )

    @staticmethod
    def _has_bisque_language_signal(user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        return bool(
            re.search(
                r"\b("
                r"bisque|dataset|resource|catalog|search bisque|find assets|upload to bisque|"
                r"download resource|create dataset|add to dataset|bisque module"
                r")\b",
                lowered,
            )
        )

    @classmethod
    def _is_bisque_contextual_follow_up_request(cls, user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        if not lowered:
            return False
        if cls._has_bisque_language_signal(user_text):
            return True
        return bool(
            re.search(
                r"\b("
                r"download|save|export|delete|remove|tag|metadata|annotation|roi|gobject|"
                r"view|show|open|preview|inspect|search|find|list|latest|recent|most recent|"
                r"details?|xml|add|append|update|put|run|module|pipeline|workflow"
                r")\b",
                lowered,
            )
            and (
                cls._has_follow_up_reference(user_text)
                or re.search(
                    r"\b("
                    r"this|that|these|those|selected|current|same|it|them|"
                    r"resource|resources|dataset|datasets|upload|uploads|image|images|"
                    r"file|files|table|tables|asset|assets"
                    r")\b",
                    lowered,
                )
            )
        )

    @classmethod
    def _is_bisque_management_request(
        cls,
        user_text: str,
        *,
        selection_context: dict[str, Any] | None = None,
    ) -> bool:
        if cls._has_bisque_language_signal(user_text):
            return True
        selection_context = dict(selection_context or {})
        if not (selection_context.get("resource_uris") or selection_context.get("dataset_uris")):
            return False
        return cls._is_bisque_contextual_follow_up_request(user_text)

    @staticmethod
    def _is_bisque_connectivity_request(user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        return bool(
            re.search(
                r"\b("
                r"ping bisque|check bisque|test bisque|bisque auth|bisque login|whoami|"
                r"am i authenticated|am i logged in|connection status|connectivity"
                r")\b",
                lowered,
            )
        )

    @staticmethod
    def _is_bisque_upload_action_request(
        user_text: str,
        *,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None = None,
    ) -> bool:
        lowered = str(user_text or "").strip().lower()
        selection_context = dict(selection_context or {})
        resource_uris = list(selection_context.get("resource_uris") or [])
        dataset_uris = list(selection_context.get("dataset_uris") or [])

        # Treat "latest upload", "my uploads", or "uploaded files" as catalog/search
        # language unless the user is explicitly asking us to upload something.
        if re.search(
            r"\b("
            r"latest upload|recent upload|most recent upload|last upload|"
            r"my uploads|your uploads|uploaded files|uploaded images|uploaded resources"
            r")\b",
            lowered,
        ) and re.search(r"\b(show|find|list|search|what|which|see|browse|open)\b", lowered):
            return False

        explicit_upload_intent = bool(
            re.search(
                r"\b("
                r"upload to bisque|upload into bisque|store in bisque|store into bisque|"
                r"ingest into bisque|send to bisque"
                r")\b",
                lowered,
            )
            or re.search(
                r"\bupload\s+("
                r"this|these|it|them|the file|the files|my file|my files|"
                r"selected|current|that|those|the image|the images|the dataset|the resource"
                r")\b",
                lowered,
            )
            or re.search(r"\bupload\b.*\b(to|into)\s+(bisque|dataset)\b", lowered)
            or re.search(r"\bingest\b", lowered)
        )

        if explicit_upload_intent:
            return True

        if not re.search(r"\bupload\b", lowered):
            return False

        # A plain "upload" without imperative phrasing only counts as an action when
        # there is concrete material in hand that makes a write action likely.
        return bool(uploaded_files or resource_uris or dataset_uris)

    @staticmethod
    def _bisque_management_tool_plan(
        *,
        user_text: str,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
        inferred_tool_names: list[str],
    ) -> ProModeToolPlan:
        lowered = str(user_text or "").strip().lower()
        selection_context = dict(selection_context or {})
        resource_uris = list(selection_context.get("resource_uris") or [])
        dataset_uris = list(selection_context.get("dataset_uris") or [])
        inferred = AgnoChatRuntime._normalize_selected_tool_names(inferred_tool_names)

        def _has_any(patterns: tuple[str, ...]) -> bool:
            return any(re.search(pattern, lowered) for pattern in patterns)

        if AgnoChatRuntime._is_bisque_connectivity_request(user_text):
            return ProModeToolPlan(
                category="bisque_management",
                selected_tool_names=["bisque_ping"],
                strict_validation=True,
                reason="Prompt explicitly asks about BisQue connectivity or authentication state.",
            )

        if _has_any((r"\brun module\b", r"\bbisque module\b", r"\bpipeline\b", r"\bworkflow\b")):
            return ProModeToolPlan(
                category="bisque_management",
                selected_tool_names=["run_bisque_module"],
                strict_validation=True,
                reason="Prompt asks to run a BisQue module or pipeline.",
            )

        if _has_any((r"\badvanced search\b", r"\blucene\b", r"\bquery xml\b")):
            return ProModeToolPlan(
                category="bisque_management",
                selected_tool_names=["bisque_advanced_search"],
                strict_validation=True,
                reason="Prompt asks for advanced BisQue search behavior.",
            )

        if _has_any((r"\bxml\b", r"\braw metadata\b", r"\bmex\b")):
            return ProModeToolPlan(
                category="bisque_management",
                selected_tool_names=["bisque_fetch_xml"],
                strict_validation=True,
                reason="Prompt asks for raw BisQue XML or metadata export.",
            )

        if _has_any(
            (r"\bgobject\b", r"\bgobjects\b", r"\bpolygon annotation\b", r"\badd annotations\b")
        ):
            return ProModeToolPlan(
                category="bisque_management",
                selected_tool_names=["bisque_add_gobjects"],
                strict_validation=True,
                reason="Prompt asks to add BisQue geometric annotations.",
            )

        if _has_any((r"\bdelete\b", r"\bremove\b", r"\btrash\b")) and "dataset" not in lowered:
            return ProModeToolPlan(
                category="bisque_management",
                selected_tool_names=["delete_bisque_resource"],
                strict_validation=True,
                reason="Prompt asks to delete a BisQue resource.",
            )

        if _has_any(
            (r"\badd tag\b", r"\btag resource\b", r"\bmetadata tag\b", r"\bannotate metadata\b")
        ):
            return ProModeToolPlan(
                category="bisque_management",
                selected_tool_names=["add_tags_to_resource"],
                strict_validation=True,
                reason="Prompt asks to mutate BisQue resource metadata.",
            )

        if _has_any((r"\bcreate dataset\b", r"\bnew dataset\b")):
            return ProModeToolPlan(
                category="bisque_management",
                selected_tool_names=["bisque_create_dataset"],
                strict_validation=True,
                reason="Prompt asks to create a BisQue dataset.",
            )

        if dataset_uris and _has_any((r"\bdownload\b", r"\bsave\b", r"\bexport\b")):
            return ProModeToolPlan(
                category="bisque_management",
                selected_tool_names=["bisque_download_dataset"],
                strict_validation=True,
                reason="Prompt asks to download a selected BisQue dataset.",
            )

        if dataset_uris and _has_any((r"\badd\b", r"\bappend\b", r"\bupdate\b", r"\bput\b")):
            return ProModeToolPlan(
                category="bisque_management",
                selected_tool_names=["bisque_add_to_dataset"],
                strict_validation=True,
                reason="Prompt asks to update a selected BisQue dataset.",
            )

        if AgnoChatRuntime._is_bisque_upload_action_request(
            user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
        ):
            return ProModeToolPlan(
                category="bisque_management",
                selected_tool_names=["upload_to_bisque"],
                strict_validation=True,
                reason="Prompt asks to upload data into BisQue.",
            )

        if resource_uris and _has_any((r"\bdownload\b", r"\bsave\b", r"\bexport\b")):
            return ProModeToolPlan(
                category="bisque_management",
                selected_tool_names=["bisque_download_resource"],
                strict_validation=True,
                reason="Prompt asks to download a selected BisQue resource.",
            )

        inspect_or_metadata = _has_any(
            (
                r"\binspect\b",
                r"\bview\b",
                r"\bshow metadata\b",
                r"\bmetadata\b",
                r"\bdimensions?\b",
                r"\bheader\b",
                r"\bdetails\b",
            )
        )
        if resource_uris and inspect_or_metadata:
            return ProModeToolPlan(
                category="bisque_management",
                selected_tool_names=["load_bisque_resource"],
                strict_validation=True,
                reason="Prompt asks to inspect a selected BisQue resource.",
            )

        catalog_search = _has_any(
            (
                r"\bsearch\b",
                r"\bfind\b",
                r"\blist\b",
                r"\bshow me\b",
                r"\brecent\b",
                r"\blatest\b",
                r"\bwhat images\b",
                r"\bwhich images\b",
                r"\bwhat files\b",
                r"\bwhich files\b",
                r"\bassets\b",
                r"\bresources\b",
            )
        )
        if not catalog_search:
            catalog_search = bool(
                re.search(r"\b(do i have|have any|are there any|named)\b", lowered)
                and re.search(
                    r"\b(dataset|datasets|resource|resources|asset|assets|image|images|table|tables)\b",
                    lowered,
                )
            )
        if catalog_search or dataset_uris or resource_uris:
            exact_lookup = bool(
                re.search(
                    r"\b(uri|exact|confirm|named|dataset resource|resource uri|dataset uri)\b",
                    lowered,
                )
                or re.search(r'"[^"]+"', user_text)
            )
            preferred_search_tool = "search_bisque_resources"
            if not exact_lookup:
                preferred_search_tool = (
                    "search_bisque_resources"
                    if "search_bisque_resources" in inferred or "bisque_find_assets" not in inferred
                    else "bisque_find_assets"
                )
            return ProModeToolPlan(
                category="bisque_management",
                selected_tool_names=[preferred_search_tool],
                strict_validation=True,
                reason="Prompt asks to search or list BisQue resources.",
            )

        preferred_non_ping = [
            tool_name
            for tool_name in inferred
            if tool_name in TOOL_SCHEMA_MAP and tool_name != "bisque_ping"
        ]
        if preferred_non_ping:
            return ProModeToolPlan(
                category="bisque_management",
                selected_tool_names=[preferred_non_ping[0]],
                strict_validation=True,
                reason="BisQue request routed to the highest-value non-connectivity tool.",
            )

        return ProModeToolPlan(
            category="bisque_management",
            selected_tool_names=["bisque_find_assets"],
            strict_validation=True,
            reason="BisQue request defaulted to catalog search rather than connectivity ping.",
        )

    @staticmethod
    def _is_image_metadata_request(user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        return bool(
            re.search(
                r"\b("
                r"dimensions?|shape|size|resolution|metadata|header|spacing|voxel|channel|channels|"
                r"time points?|z[- ]slices?|dtype|pixel size|width|height|"
                r"exif|gps|geo(?:tag|location)?|location|coordinates?|latitude|longitude|"
                r"camera|captured|timestamp|filename"
                r")\b",
                lowered,
            )
        )

    @staticmethod
    def _segmentation_request_needs_quantification(user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        return bool(
            re.search(
                r"\b("
                r"quantit(?:ative|atively)|quantif(?:y|ication)|result|results|measure|measurement|"
                r"count|counts|coverage|area|areas|"
                r"perimeter|overlap|distance|statistics|stats|metrics|morpholog|regionprops|"
                r"report|summary|compare|analysis|analyze"
                r")\b",
                lowered,
            )
        )

    @staticmethod
    def _is_depth_request(user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        return bool(re.search(r"\b(depth|depth map|depth estimation|depthpro)\b", lowered))

    @staticmethod
    def _is_segmentation_request(user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        return bool(
            re.search(r"\b(segment|segmentation|sam2|sam3|megaseg|dynunet|mask)\b", lowered)
        )

    @staticmethod
    def _prefers_megaseg_segmentation(user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        return bool(re.search(r"\b(megaseg|dynunet)\b", lowered))

    @classmethod
    def _preferred_segmentation_tool_names(cls, user_text: str) -> list[str]:
        tool_names = [
            "segment_image_megaseg"
            if cls._prefers_megaseg_segmentation(user_text)
            else "segment_image_sam2"
        ]
        if cls._segmentation_request_needs_quantification(user_text):
            tool_names.append("quantify_segmentation_masks")
        return tool_names

    @classmethod
    def _is_depth_segmentation_request(cls, user_text: str) -> bool:
        return cls._is_depth_request(user_text) and cls._is_segmentation_request(user_text)

    @staticmethod
    def _is_hdf5_like_path(path: str | os.PathLike[str] | None) -> bool:
        lowered = str(path or "").strip().lower()
        return bool(
            lowered and lowered.endswith((".h5", ".hdf5", ".hdf", ".he5", ".dream3d", ".h5ebsd"))
        )

    @staticmethod
    def _is_structured_artifact_introspection_request(user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        return bool(
            re.search(
                r"\b("
                r"keys?|groups?|datasets?|columns?|headers?|variables?|fields?|schema|structure|layout|"
                r"contents?|inside|top[- ]level|what(?:'s| is)? in|what(?: does)? .* have"
                r")\b",
                lowered,
            )
        )

    @classmethod
    def _inspect_hdf5_artifact_summary(
        cls, file_path: str | os.PathLike[str] | None
    ) -> dict[str, Any] | None:
        source = Path(str(file_path or "")).expanduser()
        if not source.exists() or not cls._is_hdf5_like_path(source):
            return None
        try:
            manifest = build_hdf5_viewer_manifest(
                file_id=f"local::{source.name}",
                file_path=str(source),
                original_name=str(source.name),
                enabled=True,
            )
        except Exception as exc:
            return {
                "success": False,
                "kind": "hdf5_structure",
                "file_path": str(source),
                "file_name": str(source.name),
                "error": str(exc),
            }
        hdf_payload = manifest.get("hdf5") if isinstance(manifest.get("hdf5"), dict) else {}
        summary = hdf_payload.get("summary") if isinstance(hdf_payload.get("summary"), dict) else {}
        return {
            "success": True,
            "kind": "hdf5_structure",
            "file_path": str(source),
            "file_name": str(source.name),
            "root_keys": [
                str(item) for item in list(hdf_payload.get("root_keys") or []) if str(item).strip()
            ],
            "default_dataset_path": (
                str(hdf_payload.get("default_dataset_path")).strip()
                if str(hdf_payload.get("default_dataset_path") or "").strip()
                else None
            ),
            "group_count": int(summary.get("group_count") or 0),
            "dataset_count": int(summary.get("dataset_count") or 0),
            "dataset_kinds": dict(summary.get("dataset_kinds") or {}),
        }

    @classmethod
    def _collect_research_program_artifact_introspection_results(
        cls,
        *,
        latest_user_text: str,
        handles: dict[str, list[str]],
        existing_invocations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not cls._is_structured_artifact_introspection_request(latest_user_text):
            return []
        existing_hdf5_paths = {
            str(dict(item.get("args") or {}).get("file_path") or "").strip()
            for item in list(existing_invocations or [])
            if str(item.get("tool") or "").strip() == "inspect_local_hdf5"
        }
        candidate_paths: list[str] = []
        for key in ("uploaded_files", "downloaded_files"):
            for raw_path in list(handles.get(key) or []):
                token = str(raw_path or "").strip()
                if token and cls._is_hdf5_like_path(token) and token not in candidate_paths:
                    candidate_paths.append(token)
        for file_path in candidate_paths:
            if file_path in existing_hdf5_paths:
                continue
            summary = cls._inspect_hdf5_artifact_summary(file_path)
            if not isinstance(summary, dict):
                continue
            return [
                {
                    "tool": "inspect_local_hdf5",
                    "status": "completed" if summary.get("success", True) else "failed",
                    "args": {"file_path": file_path},
                    "output_summary": summary,
                    "output_envelope": dict(summary),
                }
            ]
        return []

    @staticmethod
    def _build_pro_mode_tool_plan(
        *,
        user_text: str,
        uploaded_files: list[str],
        selected_tool_names: list[str],
        selection_context: dict[str, Any] | None,
        inferred_tool_names: list[str],
        prior_pro_mode_state: dict[str, Any] | None = None,
    ) -> ProModeToolPlan | None:
        explicit = [
            tool_name
            for tool_name in AgnoChatRuntime._normalize_selected_tool_names(selected_tool_names)
            if tool_name in TOOL_SCHEMA_MAP
        ]
        lowered = str(user_text or "").strip().lower()
        has_direct_image_target = AgnoChatRuntime._has_direct_image_target(
            user_text=user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
        )
        if AgnoChatRuntime._requires_rigorous_proof_workflow(
            user_text=user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
            tool_plan=None,
        ):
            return None
        early_bisque_plan: ProModeToolPlan | None = None
        if AgnoChatRuntime._is_bisque_management_request(
            user_text, selection_context=selection_context
        ):
            early_bisque_plan = AgnoChatRuntime._bisque_management_tool_plan(
                user_text=user_text,
                uploaded_files=uploaded_files,
                selection_context=selection_context,
                inferred_tool_names=inferred_tool_names,
            )
            if any(
                tool_name
                not in {"search_bisque_resources", "bisque_find_assets", "load_bisque_resource"}
                for tool_name in list(early_bisque_plan.selected_tool_names or [])
            ):
                return early_bisque_plan
        if AgnoChatRuntime._is_code_execution_request(user_text):
            return ProModeToolPlan(
                category="code_execution",
                selected_tool_names=["codegen_python_plan", "execute_python_job"],
                strict_validation=True,
                reason="Prompt explicitly requests code generation and sandbox execution.",
            )
        if AgnoChatRuntime._requires_iterative_research_program(
            user_text=user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
            prior_pro_mode_state=prior_pro_mode_state,
        ):
            return ProModeToolPlan(
                category="research_program",
                selected_tool_names=AgnoChatRuntime._research_program_tool_bundle(
                    user_text=user_text,
                    uploaded_files=uploaded_files,
                    selection_context=selection_context,
                    prior_pro_mode_state=prior_pro_mode_state,
                ),
                strict_validation=False,
                reason=(
                    "Prompt requires a multi-step scientific evidence program rather than a single-tool pass."
                ),
            )
        if AgnoChatRuntime._requires_validated_numeric_workflow(
            user_text=user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
        ):
            return ProModeToolPlan(
                category="validated_numeric",
                selected_tool_names=["numpy_calculator"],
                strict_validation=True,
                reason="Closed-form quantitative prompt should be grounded with deterministic calculation.",
            )

        if has_direct_image_target:
            if re.search(
                r"\b(spectral|frequency|instability|active learning|hard sample|uncertain(?:ty)? ranking)\b",
                lowered,
            ):
                return ProModeToolPlan(
                    category="spectral_analysis",
                    selected_tool_names=["analyze_prediction_stability"],
                    strict_validation=True,
                    reason="Image-targeted prompt explicitly asks for spectral instability or active-learning ranking.",
                )
            if AgnoChatRuntime._is_depth_segmentation_request(user_text):
                depth_segmentation_tools = ["estimate_depth_pro", "segment_image_sam2"]
                if AgnoChatRuntime._segmentation_request_needs_quantification(user_text):
                    depth_segmentation_tools.append("quantify_segmentation_masks")
                return ProModeToolPlan(
                    category="depth_segmentation",
                    selected_tool_names=depth_segmentation_tools,
                    strict_validation=True,
                    reason=(
                        "Image-targeted prompt explicitly asks for depth estimation, segmentation of the derived depth map, and quantitative follow-up."
                        if "quantify_segmentation_masks" in depth_segmentation_tools
                        else "Image-targeted prompt explicitly asks for depth estimation followed by segmentation of the derived depth map."
                    ),
                )
            if AgnoChatRuntime._is_depth_request(user_text):
                return ProModeToolPlan(
                    category="depth_analysis",
                    selected_tool_names=["estimate_depth_pro"],
                    strict_validation=True,
                    reason="Image-targeted prompt explicitly asks for depth estimation.",
                )
            if re.search(r"\b(yolo|detect|detection|bounding box|bbox)\b", lowered):
                return ProModeToolPlan(
                    category="detection",
                    selected_tool_names=["yolo_detect", "quantify_objects"],
                    strict_validation=True,
                    reason="Image-targeted prompt explicitly asks for detection.",
                )
            if AgnoChatRuntime._is_segmentation_request(user_text):
                segmentation_tools = AgnoChatRuntime._preferred_segmentation_tool_names(user_text)
                return ProModeToolPlan(
                    category="segmentation",
                    selected_tool_names=segmentation_tools,
                    strict_validation=True,
                    reason=(
                        "Image-targeted prompt explicitly asks for segmentation with quantitative follow-up."
                        if len(segmentation_tools) > 1
                        else "Image-targeted prompt explicitly asks for object segmentation."
                    ),
                )
            if AgnoChatRuntime._is_image_metadata_request(user_text):
                return ProModeToolPlan(
                    category="image_metadata",
                    selected_tool_names=["bioio_load_image"],
                    strict_validation=True,
                    reason="Image-targeted prompt asks for metadata or dimensions.",
                )
            if inferred_tool_names:
                return ProModeToolPlan(
                    category="uploaded_file_analysis",
                    selected_tool_names=inferred_tool_names,
                    strict_validation=False,
                    reason="Image-targeted prompt requires tool-based analysis.",
                )

        if early_bisque_plan is not None:
            return early_bisque_plan

        if explicit:
            return ProModeToolPlan(
                category="explicit_selection",
                selected_tool_names=explicit,
                strict_validation=False,
                reason="Falling back to explicitly selected tools after Pro Mode routing heuristics.",
            )

        return None

    async def _run_pro_mode_fast_dialogue(
        self,
        *,
        latest_user_text: str,
        intake_direct_response: str | None,
        task_regime: str | None,
        shared_context: dict[str, Any],
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        max_runtime_seconds: int,
        debug: bool | None,
    ) -> ProModeWorkflowResult:
        context_sections: list[str] = []
        memory_messages = [
            str(item or "").strip()
            for item in list((shared_context or {}).get("memory_messages") or [])
            if str(item or "").strip()
        ]
        knowledge_messages = [
            str(item or "").strip()
            for item in list((shared_context or {}).get("knowledge_messages") or [])
            if str(item or "").strip()
        ]
        analysis_brief_lines = [
            str(item or "").strip()
            for item in list((shared_context or {}).get("analysis_brief_lines") or [])
            if str(item or "").strip()
        ]
        if memory_messages:
            context_sections.append(
                "Relevant prior context:\n" + "\n".join(f"- {item}" for item in memory_messages[:4])
            )
        if knowledge_messages:
            context_sections.append(
                "Relevant knowledge context:\n"
                + "\n".join(f"- {item}" for item in knowledge_messages[:4])
            )
        if analysis_brief_lines:
            context_sections.append(
                "Relevant running summary:\n"
                + "\n".join(f"- {item}" for item in analysis_brief_lines[:4])
            )

        prompt_sections = [
            "Answer the following scientist-facing request directly.",
            "Keep the answer concise, accurate, and natural to read.",
            "Start with a 1-2 sentence bottom line before any structure or elaboration.",
            "Prefer short paragraphs over heading-heavy frameworks unless the user explicitly asked for a checklist or protocol.",
            "Do not mention internal workflows, routes, tools, or hidden reasoning.",
            "If the user asked for a direct comparison or explanation, prioritize clarity over exhaustiveness.",
        ]
        if str(task_regime or "").strip().lower() == "closed_form_grounded":
            prompt_sections.append(
                "Prefer the standard interpretation and state the answer near the beginning."
            )
        draft = str(intake_direct_response or "").strip()
        if draft:
            prompt_sections.extend(
                [
                    "Here is the current triage draft to improve for accuracy and polish:",
                    draft,
                ]
            )
        prompt_sections.extend(context_sections)
        prompt_sections.append(f"User request: {latest_user_text}")
        prompt = "\n\n".join(section for section in prompt_sections if section)

        def _build_fast_dialogue_agent(model_builder: Callable[..., AgnoModel]) -> Agent:
            return Agent(
                name="pro-mode-fast-dialogue",
                model=model_builder(
                    reasoning_mode="fast",
                    reasoning_effort_override="low",
                    max_runtime_seconds=max_runtime_seconds,
                ),
                instructions=[
                    "You answer directly and clearly.",
                    "Be concise without sounding terse.",
                    "Lead with the answer, then add short explanatory paragraphs.",
                    "Avoid unnecessary formatting unless it improves clarity.",
                    "Prefer paragraphs to long heading stacks or nested bullet hierarchies.",
                    "Do not fabricate missing details.",
                ],
                markdown=True,
                telemetry=False,
                retries=0,
                store_events=False,
                store_history_messages=False,
                session_state={
                    "pro_mode_context": {
                        "memory_messages": memory_messages[:4],
                        "knowledge_messages": knowledge_messages[:4],
                        "analysis_brief_lines": analysis_brief_lines[:4],
                    }
                },
                add_session_state_to_context=True,
                debug_mode=bool(debug),
            )

        model_route = self._pro_mode_model_route_metadata(
            fallback_used=False,
            active_model=str(
                self._setting(self.settings, "resolved_pro_mode_model", self.model) or self.model
            ),
        )
        try:
            result, model_route = await self._arun_text_phase_with_optional_pro_mode_transport(
                phase_name="fast_dialogue",
                prompt=prompt,
                build_agent=_build_fast_dialogue_agent,
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                debug=debug,
                reasoning_mode="fast",
                reasoning_effort_override="low",
                max_runtime_seconds=max_runtime_seconds,
            )
        except Exception as exc:
            failure_code = self._classify_pro_mode_failure(exc) or "published_api_failure"
            fallback_text = draft or str(exc or exc.__class__.__name__).strip()
            return ProModeWorkflowResult(
                response_text=fallback_text,
                metadata={
                    "pro_mode": {
                        "execution_path": "direct_response",
                        "runtime_status": "failed",
                        "model_route": self._pro_mode_model_route_metadata(
                            fallback_used=False,
                            failure_code=failure_code,
                            active_model=self.model,
                        ),
                    }
                },
                runtime_status="failed",
                runtime_error=str(exc or exc.__class__.__name__),
            )
        response_text, _source = self._coerce_visible_output(result)
        normalized_response_text = str(response_text or "").strip() or draft
        return ProModeWorkflowResult(
            response_text=normalized_response_text,
            metadata={
                "pro_mode": {
                    "execution_path": "direct_response",
                    "runtime_status": "completed",
                    "model_route": model_route,
                }
            },
            runtime_status="completed",
            runtime_error=None,
        )

    @staticmethod
    def _has_follow_up_reference(user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        return bool(
            re.search(
                r"\b("
                r"given everything|based on what we discussed|using the dataset|using that dataset|"
                r"that dataset|that image|that file|that table|that hdf5|this dataset|this image|"
                r"this file|this table|this hdf5|follow[- ]up|continue|"
                r"what you found|what we found|from that|from this|now given|using everything"
                r")\b",
                lowered,
            )
        )

    @classmethod
    def _has_cross_turn_artifact_reference(cls, user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        if cls._has_follow_up_reference(user_text):
            return True
        return bool(
            re.search(
                r"\b(using|based on|from|with|given|compare)\b.{0,80}\b("
                r"dataset|image|images|file|files|table|tables|hdf5|resource|resources|scan|scans|tile|tiles|"
                r"results|report|analysis|statistics|findings"
                r")\b",
                lowered,
            )
            or re.search(
                r"\b(this|that|these|those|previous|prior|earlier)\b.{0,80}\b("
                r"dataset|image|images|file|files|table|tables|hdf5|resource|resources|report|analysis|results|findings"
                r")\b",
                lowered,
            )
        )

    @staticmethod
    def _prompt_explicitly_requests_prior_results_load(user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        if not lowered:
            return False
        return bool(
            re.search(
                r"\b(load|reuse|use|open|show|continue with|work from)\b.{0,32}\b(previous|prior|cached|existing)\b",
                lowered,
            )
            or re.search(
                r"\b(load|reuse|use)\b.{0,24}\b(results?|analysis|run|output|outputs)\b",
                lowered,
            )
        )

    @staticmethod
    def _prompt_requests_cross_turn_comparison(user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        if not lowered:
            return False
        return bool(
            re.search(
                r"\b(compare|comparison|versus|vs\.?|against|relative to)\b.{0,48}\b(previous|prior|earlier|existing|baseline|results?|analysis|run|output|outputs)\b",
                lowered,
            )
            or re.search(
                r"\b(previous|prior|earlier|existing|baseline)\b.{0,48}\b(compare|comparison|versus|vs\.?|against)\b",
                lowered,
            )
        )

    @classmethod
    def _uploaded_image_target_is_authoritative(
        cls,
        *,
        user_text: str,
        uploaded_files: list[str],
    ) -> bool:
        if not cls._image_like_files(uploaded_files):
            return False
        if cls._prompt_explicitly_requests_prior_results_load(user_text):
            return False
        lowered = str(user_text or "").strip().lower()
        return bool(
            cls._is_depth_request(user_text)
            or cls._is_segmentation_request(user_text)
            or cls._is_image_metadata_request(user_text)
            or cls._is_plot_or_visual_analysis_request(user_text)
            or re.search(
                r"\b("
                r"detect|detection|bounding box|bounding boxes|bbox|yolo|spectral|instability|"
                r"active learning|hard sample|uncertainty|confidence|measure|quantif(?:y|ication)|"
                r"analy[sz]e|report|summary"
                r")\b",
                lowered,
            )
        )

    @classmethod
    def _selection_context_replaces_saved_image_target(
        cls,
        *,
        user_text: str,
        selection_context: dict[str, Any] | None,
        prior_state: dict[str, Any] | None,
    ) -> bool:
        if cls._prompt_explicitly_requests_prior_results_load(user_text):
            return False
        payload = dict(selection_context or {})
        current_resource_uris = {
            str(value or "").strip()
            for value in list(payload.get("resource_uris") or [])
            if str(value or "").strip()
        }
        if not current_resource_uris:
            return False
        if not cls._selection_context_supports_direct_image_analysis(
            user_text=user_text,
            selection_context=selection_context,
        ):
            return False
        prior_handles = dict(dict(prior_state or {}).get("handles") or {})
        prior_resource_uris = {
            str(value or "").strip()
            for value in list(prior_handles.get("resource_uris") or [])
            if str(value or "").strip()
        }
        return current_resource_uris != prior_resource_uris

    @classmethod
    def _current_turn_replaces_saved_image_target(
        cls,
        *,
        user_text: str,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
        prior_state: dict[str, Any] | None,
    ) -> bool:
        return cls._uploaded_image_target_is_authoritative(
            user_text=user_text,
            uploaded_files=uploaded_files,
        ) or cls._selection_context_replaces_saved_image_target(
            user_text=user_text,
            selection_context=selection_context,
            prior_state=prior_state,
        )

    @classmethod
    def _normalize_selection_context_for_current_turn(
        cls,
        *,
        user_text: str,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
        prior_state: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        current = dict(selection_context or {})
        if not current:
            return selection_context
        if (
            (current.get("resource_uris") or current.get("dataset_uris"))
            and not cls._has_bisque_language_signal(user_text)
            and not cls._is_bisque_contextual_follow_up_request(user_text)
        ):
            sanitized = dict(current)
            for key in (
                "resource_uris",
                "dataset_uris",
                "artifact_handles",
                "focused_file_ids",
                "context_id",
                "source",
                "originating_message_id",
                "originating_user_text",
                "suggested_domain",
                "suggested_tool_names",
            ):
                sanitized.pop(key, None)
            return sanitized or None
        if not cls._uploaded_image_target_is_authoritative(
            user_text=user_text,
            uploaded_files=uploaded_files,
        ):
            return selection_context
        sanitized = dict(current)
        for key in (
            "resource_uris",
            "dataset_uris",
            "artifact_handles",
            "focused_file_ids",
            "context_id",
        ):
            sanitized.pop(key, None)
        return sanitized or None

    @classmethod
    def _has_prior_analysis_handles(cls, prior_state: dict[str, Any] | None) -> bool:
        state = dict(prior_state or {})
        handles = dict(state.get("handles") or {})
        if not handles:
            return False
        meaningful_keys = (
            "mask_paths",
            "depth_map_paths",
            "depth_npy_paths",
            "prediction_json_paths",
            "analysis_table_paths",
            "job_ids",
            "download_dirs",
            "downloaded_files",
            "image_files",
            "preview_paths",
            "array_paths",
            "dataset_uris",
            "resource_uris",
        )
        return any(bool(list(handles.get(key) or [])) for key in meaningful_keys) or bool(
            list(state.get("evidence_summaries") or [])
        )

    @classmethod
    def _is_prior_analysis_follow_up_request(
        cls,
        user_text: str,
        *,
        prior_state: dict[str, Any] | None,
    ) -> bool:
        if not cls._has_prior_analysis_handles(prior_state):
            return False
        lowered = str(user_text or "").strip().lower()
        if cls._has_cross_turn_artifact_reference(user_text):
            return True
        plot_like = cls._is_plot_or_visual_analysis_request(user_text)
        analysis_like = bool(
            re.search(
                r"\b("
                r"distribution|confidence|uncertain|uncertainty|threshold|low[- ]confidence|"
                r"highlight|review candidates|hard sample|summari[sz]e|compare|statistics|"
                r"quantif(?:y|ication)|counts?|measurements?|patterns?"
                r")\b",
                lowered,
            )
        )
        return plot_like or analysis_like

    @staticmethod
    def _saved_proof_workflow_state(pro_mode_state: dict[str, Any] | None) -> dict[str, Any]:
        state = dict(pro_mode_state or {})
        proof_workflow = dict(state.get("proof_workflow") or {})
        proof_state = dict(proof_workflow.get("proof_state") or {})
        if proof_state:
            return {
                "proof_state": proof_state,
                "iteration_summaries": list(proof_workflow.get("iteration_summaries") or []),
                "iterations": int(proof_workflow.get("iterations") or 0),
                "compression_stats": dict(proof_workflow.get("compression_stats") or {}),
                "stagnant_iterations": int(proof_workflow.get("stagnant_iterations") or 0),
                "last_blocker_signature": str(
                    proof_workflow.get("last_blocker_signature") or ""
                ).strip(),
                "progress_score": float(proof_state.get("progress_score") or 0.0),
                "quality_flags": list(proof_state.get("quality_flags") or []),
                "proof_frame": dict(proof_workflow.get("proof_frame") or {}),
                "proof_digest": list(state.get("proof_digest") or []),
            }
        return {}

    @staticmethod
    def _saved_autonomy_state(pro_mode_state: dict[str, Any] | None) -> dict[str, Any]:
        state = dict(pro_mode_state or {})
        autonomy_state = dict(state.get("autonomy_state") or {})
        if not autonomy_state:
            autonomy_state = dict(state.get("autonomous_cycle") or {})
        if not autonomy_state:
            return {}
        autonomy_state_v2 = dict(state.get("autonomy_state_v2") or {})
        return {
            "cycle_id": str(autonomy_state.get("cycle_id") or "").strip(),
            "checkpoint_index": int(autonomy_state.get("checkpoint_index") or 0),
            "open_obligations": [
                str(item or "").strip()
                for item in list(autonomy_state.get("open_obligations") or [])
                if str(item or "").strip()
            ][:12],
            "evidence_ledger": [
                str(item or "").strip()
                for item in list(autonomy_state.get("evidence_ledger") or [])
                if str(item or "").strip()
            ][-20:],
            "candidate_answer": str(autonomy_state.get("candidate_answer") or "").strip(),
            "stop_reason": str(autonomy_state.get("stop_reason") or "").strip(),
            "resume_readiness": str(autonomy_state.get("resume_readiness") or "").strip()
            or "ready",
            "next_best_actions": [
                str(item or "").strip()
                for item in list(autonomy_state.get("next_best_actions") or [])
                if str(item or "").strip()
            ][:8],
            "cycles_completed": int(autonomy_state.get("cycles_completed") or 0),
            "tool_families_used": [
                str(item or "").strip()
                for item in list(autonomy_state.get("tool_families_used") or [])
                if str(item or "").strip()
            ][:8],
            "continuation_fidelity": float(autonomy_state.get("continuation_fidelity") or 0.0),
            "autonomy_state_v2": autonomy_state_v2,
        }

    @classmethod
    def _has_saved_autonomy_state(cls, pro_mode_state: dict[str, Any] | None) -> bool:
        return bool(cls._saved_autonomy_state(pro_mode_state))

    @classmethod
    def _is_autonomy_follow_up_turn(
        cls, user_text: str, pro_mode_state: dict[str, Any] | None
    ) -> bool:
        if not cls._has_saved_autonomy_state(pro_mode_state):
            return False
        if cls._is_phatic_turn(user_text):
            return False
        lowered = str(user_text or "").strip().lower()
        if not lowered:
            return False
        if cls._has_follow_up_reference(user_text):
            return True
        return bool(
            re.search(
                r"\b("
                r"continue|resume|pick up|pick this up|next step|what next|tighten|refine|"
                r"carry on|follow[- ]up|revise|update the plan|stress[- ]test|ablation|"
                r"benchmark|what about|does that mean|how would you extend"
                r")\b",
                lowered,
            )
        )

    @classmethod
    def _is_autonomy_resume_synthesis_turn(
        cls, user_text: str, pro_mode_state: dict[str, Any] | None
    ) -> bool:
        saved_state = cls._saved_autonomy_state(pro_mode_state)
        if not saved_state:
            return False
        if list(saved_state.get("open_obligations") or []):
            return False
        if not str(saved_state.get("candidate_answer") or "").strip():
            return False
        lowered = str(user_text or "").strip().lower()
        if not lowered or cls._is_phatic_turn(user_text):
            return False
        return bool(
            re.search(
                r"\b("
                r"resume|continue|follow[- ]up|"
                r"restate|restat(?:e|ing)|rewrite|rephrase|reword|summari[sz]e|"
                r"compact|concise|short paragraph|one paragraph|one short paragraph|"
                r"final answer|state the answer|give the answer|derivation|"
                r"clarify|explain again|say it more clearly|briefly"
                r")\b",
                lowered,
            )
        )

    @classmethod
    def _has_saved_proof_state(cls, pro_mode_state: dict[str, Any] | None) -> bool:
        state = dict(pro_mode_state or {})
        if str(state.get("task_regime") or "").strip() == "rigorous_proof":
            return True
        return bool(cls._saved_proof_workflow_state(state))

    @classmethod
    def _is_proof_follow_up_turn(
        cls, user_text: str, pro_mode_state: dict[str, Any] | None
    ) -> bool:
        if not cls._has_saved_proof_state(pro_mode_state):
            return False
        if cls._is_phatic_turn(user_text):
            return False
        lowered = str(user_text or "").strip().lower()
        if not lowered:
            return False
        return bool(
            re.search(
                r"\b("
                r"continue|carry on|go on|finish|complete|resume|follow[- ]up|"
                r"close the gap|fill the gap|justify|clarify|why does|why is|"
                r"that proof|this proof|the proof|that argument|this argument|"
                r"that step|this step|that lemma|this lemma|remaining blocker|"
                r"remaining gap|what remains|is this rigorous"
                r")\b",
                lowered,
            )
        )

    @staticmethod
    def _task_regime_for_turn(
        *,
        user_text: str,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
        tool_plan: ProModeToolPlan | None,
    ) -> str:
        lowered = str(user_text or "").strip().lower()
        selection_context = dict(selection_context or {})
        has_catalog_handles = bool(
            selection_context.get("resource_uris") or selection_context.get("dataset_uris")
        )
        has_artifact_handles = AgnoChatRuntime._selection_context_has_artifact_handles(
            selection_context
        )
        if AgnoChatRuntime._is_phatic_turn(user_text):
            return "phatic_or_small_talk"
        if AgnoChatRuntime._requires_rigorous_proof_workflow(
            user_text=user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
            tool_plan=tool_plan,
        ):
            return "rigorous_proof"
        if AgnoChatRuntime._requires_self_contained_reasoning_solver(
            user_text=user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
            tool_plan=tool_plan,
        ):
            return "self_contained_reasoning"
        if tool_plan is not None:
            if tool_plan.category == "research_program":
                if uploaded_files and (
                    has_catalog_handles
                    or re.search(r"\b(dataset|bisque|report|compare|given everything)\b", lowered)
                ):
                    return "iterative_multimodal_research"
                if has_catalog_handles or re.search(
                    r"\b(dataset|bisque|catalog|resource)\b", lowered
                ):
                    return "dataset_or_catalog_research"
                return "artifact_interpretation"
            if tool_plan.category in {"validated_numeric", "code_execution"}:
                return "closed_form_grounded"
            if tool_plan.category in {
                "image_metadata",
                "uploaded_file_analysis",
                "depth_analysis",
                "detection",
                "segmentation",
            }:
                return "artifact_interpretation"
            if tool_plan.category == "bisque_management":
                return "dataset_or_catalog_research"
        if uploaded_files or has_catalog_handles or has_artifact_handles:
            return "artifact_interpretation"
        if AgnoChatRuntime._is_report_like_request(user_text):
            return "conceptual_high_uncertainty"
        if AgnoChatRuntime._requires_deep_reasoning(user_text):
            return "conceptual_high_uncertainty"
        return "closed_form_grounded"

    def _execution_regime_for_decision(
        self,
        *,
        route: str,
        tool_plan: ProModeToolPlan | None,
    ) -> str:
        if route == "direct_response":
            return "fast_dialogue"
        if route == "tool_workflow":
            return "validated_tool"
        return self._default_pro_mode_deep_execution_regime()

    @staticmethod
    def _normalize_forced_execution_regime(benchmark: dict[str, Any] | None) -> str | None:
        regime = (
            str(dict(benchmark or {}).get("force_pro_mode_execution_regime") or "").strip().lower()
        )
        if regime in {
            "fast_dialogue",
            "validated_tool",
            "iterative_research",
            "autonomous_cycle",
            "focused_team",
            "proof_workflow",
            "reasoning_solver",
            "expert_council",
        }:
            return regime
        return None

    @staticmethod
    def _context_policy_from_decision(
        *,
        route: str,
        execution_regime: str,
        task_regime: str,
        latest_user_text: str,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
        prior_pro_mode_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        selection_context = dict(selection_context or {})
        handles: list[str] = []
        if uploaded_files:
            handles.append("uploaded_files")
        if selection_context.get("dataset_uris"):
            handles.append("dataset_uris")
        if selection_context.get("resource_uris"):
            handles.append("resource_uris")
        if execution_regime == "fast_dialogue":
            return {
                "load_memory": False,
                "load_knowledge": False,
                "history_window": 0,
                "artifact_handles_to_expose": [],
                "compression_required": False,
            }
        if execution_regime == "validated_tool":
            return {
                "load_memory": bool(AgnoChatRuntime._has_follow_up_reference(latest_user_text)),
                "load_knowledge": False,
                "history_window": 1 if task_regime == "closed_form_grounded" else 2,
                "artifact_handles_to_expose": handles,
                "compression_required": False,
            }
        if execution_regime == "reasoning_solver":
            return {
                "load_memory": bool(
                    AgnoChatRuntime._has_follow_up_reference(latest_user_text)
                    or AgnoChatRuntime._has_saved_proof_state(prior_pro_mode_state)
                ),
                "load_knowledge": False,
                "history_window": 1,
                "artifact_handles_to_expose": [],
                "compression_required": False,
            }
        if execution_regime == "autonomous_cycle":
            lowered = str(latest_user_text or "").strip().lower()
            knowledge_signal = bool(
                re.search(
                    r"\b(paper|benchmark|ablation|docs|documentation|prior work|literature|reference|report)\b",
                    lowered,
                )
            )
            has_saved_autonomy = AgnoChatRuntime._has_saved_autonomy_state(prior_pro_mode_state)
            return {
                "load_memory": bool(
                    AgnoChatRuntime._has_follow_up_reference(latest_user_text) or has_saved_autonomy
                ),
                "load_knowledge": knowledge_signal,
                "history_window": 3 if has_saved_autonomy else 2,
                "artifact_handles_to_expose": handles,
                "compression_required": True,
            }
        if execution_regime == "focused_team":
            return {
                "load_memory": bool(AgnoChatRuntime._has_follow_up_reference(latest_user_text)),
                "load_knowledge": False,
                "history_window": 2,
                "artifact_handles_to_expose": handles,
                "compression_required": False,
            }
        if execution_regime == "expert_council":
            return {
                "load_memory": bool(AgnoChatRuntime._has_follow_up_reference(latest_user_text)),
                "load_knowledge": False,
                "history_window": 2,
                "artifact_handles_to_expose": handles,
                "compression_required": False,
            }
        if execution_regime == "proof_workflow":
            return {
                "load_memory": bool(
                    AgnoChatRuntime._has_follow_up_reference(latest_user_text)
                    or AgnoChatRuntime._has_saved_proof_state(prior_pro_mode_state)
                ),
                "load_knowledge": False,
                "history_window": 2
                if AgnoChatRuntime._has_saved_proof_state(prior_pro_mode_state)
                else 1,
                "artifact_handles_to_expose": [],
                "compression_required": True,
            }
        if execution_regime == "iterative_research":
            extra_handles = list(handles)
            for token in ("prediction_json_paths", "preview_paths", "downloaded_files"):
                if token not in extra_handles:
                    extra_handles.append(token)
            return {
                "load_memory": True,
                "load_knowledge": False,
                "history_window": 3,
                "artifact_handles_to_expose": extra_handles,
                "compression_required": True,
            }
        return {
            "load_memory": bool(AgnoChatRuntime._has_follow_up_reference(latest_user_text)),
            "load_knowledge": False,
            "history_window": 2,
            "artifact_handles_to_expose": handles,
            "compression_required": False,
        }

    def _force_pro_mode_execution_regime(
        self,
        *,
        decision: ProModeIntakeDecision,
        forced_execution_regime: str,
        latest_user_text: str,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
        prior_pro_mode_state: dict[str, Any] | None = None,
    ) -> ProModeIntakeDecision:
        overridden = decision.model_copy(deep=True)
        regime = str(forced_execution_regime or "").strip().lower()
        if not regime:
            return overridden
        if regime == "fast_dialogue":
            overridden.route = "direct_response"
            overridden.direct_response = (
                overridden.direct_response or str(latest_user_text or "").strip()
            )
            task_regime = "phatic_or_small_talk"
        elif regime in {"validated_tool", "iterative_research"}:
            overridden.route = "tool_workflow"
            overridden.direct_response = None
            task_regime = self._task_regime_for_turn(
                user_text=latest_user_text,
                uploaded_files=uploaded_files,
                selection_context=selection_context,
                tool_plan=None,
            )
        elif regime == "proof_workflow":
            overridden.route = "deep_reasoning"
            overridden.direct_response = None
            task_regime = "rigorous_proof"
        elif regime == "reasoning_solver":
            overridden.route = "deep_reasoning"
            overridden.direct_response = None
            task_regime = "self_contained_reasoning"
        else:
            overridden.route = "deep_reasoning"
            overridden.direct_response = None
            task_regime = "conceptual_high_uncertainty"
        policy = self._context_policy_from_decision(
            route=overridden.route,
            execution_regime=regime,
            task_regime=task_regime,
            latest_user_text=latest_user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
            prior_pro_mode_state=prior_pro_mode_state,
        )
        overridden.execution_regime = regime
        overridden.task_regime = task_regime
        overridden.context_policy = overridden.context_policy.model_validate(policy)
        overridden.load_memory = bool(policy.get("load_memory"))
        overridden.load_knowledge = bool(policy.get("load_knowledge"))
        overridden.recent_history_turns = max(0, int(policy.get("history_window") or 0))
        overridden.reason = f"Benchmark override forced the `{regime}` Pro Mode regime."
        return overridden

    @staticmethod
    def _requires_iterative_research_program(
        *,
        user_text: str,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
        prior_pro_mode_state: dict[str, Any] | None = None,
    ) -> bool:
        lowered = str(user_text or "").strip().lower()
        selection_context = dict(selection_context or {})
        if AgnoChatRuntime._is_bisque_connectivity_request(user_text):
            return False
        if AgnoChatRuntime._is_bisque_upload_action_request(
            user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
        ):
            return False
        has_artifact_handles = AgnoChatRuntime._selection_context_has_artifact_handles(
            selection_context
        )
        has_direct_image_target = AgnoChatRuntime._has_direct_image_target(
            user_text=user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
        )
        simple_single_image_segmentation = bool(
            has_direct_image_target
            and AgnoChatRuntime._is_segmentation_request(user_text)
            and not AgnoChatRuntime._is_depth_request(user_text)
            and not bool(selection_context.get("dataset_uris"))
            and not has_artifact_handles
        )
        if AgnoChatRuntime._is_depth_segmentation_request(user_text) and (
            has_direct_image_target or has_artifact_handles
        ):
            return False
        if simple_single_image_segmentation:
            return False
        plot_request = AgnoChatRuntime._is_plot_or_visual_analysis_request(user_text)
        prior_analysis_follow_up = AgnoChatRuntime._is_prior_analysis_follow_up_request(
            user_text,
            prior_state=prior_pro_mode_state,
        )
        has_bisque_context = bool(
            selection_context.get("resource_uris") or selection_context.get("dataset_uris")
        )
        multi_stage_signal = bool(
            re.search(
                r"\b("
                r"report|survey|analysis|analyze|investigate|synthesize|compare|summarize|"
                r"count|average|distribution|statistics|population|ecological|optimi[sz]e|"
                r"until|benchmark|evaluate|given everything|follow[- ]up"
                r")\b",
                lowered,
            )
        )
        data_signal = bool(
            uploaded_files
            or has_bisque_context
            or has_artifact_handles
            or has_direct_image_target
            or prior_analysis_follow_up
            or re.search(r"\b(dataset|bisque|resource|resources|catalog|ct scan|scan)\b", lowered)
        )
        introspection_signal = bool(
            AgnoChatRuntime._is_structured_artifact_introspection_request(user_text)
            and (
                has_bisque_context
                or any(
                    AgnoChatRuntime._is_hdf5_like_path(path) for path in list(uploaded_files or [])
                )
            )
        )
        report_signal = bool(
            re.search(
                r"\b(report|write up|write a report|ecological analysis|research summary)\b",
                lowered,
            )
        )
        numeric_signal = bool(
            re.search(
                r"\b(count|average|mean|median|statistics|distribution|size|box|burrow|population|density)\b",
                lowered,
            )
        )
        return bool(
            data_signal
            and (
                introspection_signal
                or (plot_request and has_artifact_handles)
                or report_signal
                or prior_analysis_follow_up
                or numeric_signal
                or AgnoChatRuntime._has_follow_up_reference(user_text)
                or (multi_stage_signal and len(lowered) > 120)
            )
        )

    @staticmethod
    def _requires_rigorous_proof_workflow(
        *,
        user_text: str,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
        tool_plan: ProModeToolPlan | None,
    ) -> bool:
        if uploaded_files:
            return False
        selection_context = dict(selection_context or {})
        if selection_context.get("resource_uris") or selection_context.get("dataset_uris"):
            return False
        if tool_plan is not None:
            return False
        lowered = str(user_text or "").strip().lower()
        if AgnoChatRuntime._has_follow_up_reference(user_text):
            return False
        proof_signal = bool(
            re.search(
                r"\b(prove|proof|rigorous|self-contained|lemma|theorem|corollary|contradiction|suppose|show that)\b",
                lowered,
            )
        )
        proof_task_signal = bool(
            re.search(
                r"\b(either prove|either show|determine whether|do not assume|fully self-contained and rigorous)\b",
                lowered,
            )
        )
        open_ended_signal = bool(
            re.search(
                r"\b(explain|summari[sz]e|report|research|survey|literature|compare carefully)\b",
                lowered,
            )
        )
        return (proof_signal or proof_task_signal) and not open_ended_signal

    @staticmethod
    def _requires_self_contained_reasoning_solver(
        *,
        user_text: str,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
        tool_plan: ProModeToolPlan | None,
    ) -> bool:
        if uploaded_files:
            return False
        selection_context = dict(selection_context or {})
        if selection_context.get("resource_uris") or selection_context.get("dataset_uris"):
            return False
        if tool_plan is not None:
            return False
        lowered = str(user_text or "").strip().lower()
        if AgnoChatRuntime._has_follow_up_reference(user_text):
            return False
        if AgnoChatRuntime._requires_rigorous_proof_workflow(
            user_text=user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
            tool_plan=tool_plan,
        ):
            return False
        if AgnoChatRuntime._requests_collective_reasoning_council(user_text):
            return False
        if not AgnoChatRuntime._requires_deep_reasoning(user_text):
            return False
        if AgnoChatRuntime._requires_validated_numeric_workflow(
            user_text=user_text,
            uploaded_files=[],
            selection_context=None,
        ):
            return False
        open_ended_request = bool(
            re.search(
                r"\b(explain|why|interpret|summari[sz]e|report|research|proposal|design an experiment|discuss|compare carefully)\b",
                lowered,
            )
        )
        single_answer_request = bool(
            re.search(
                r"\b(indicate|determine|find|identify|which|what is|what's|how many|how much|total number|answer)\b",
                lowered,
            )
        )
        formal_signal = bool(
            re.search(
                r"\d|[a-z]\d|\bmixture\b|\bmolecule\b|\bcompound\b|\bhydrogenation\b|\bhydrocarbon\b|\bsolvent\b|"
                r"\bbromine\b|\bplatinum\b|\bdecolori[sz]e\b|\bmass fraction\b|\bnmr\b|\bdistinct\b|"
                r"\benergy\b|\bphase\b|\bprobability\b|\brate\b|\bmechanism\b",
                lowered,
            )
        )
        return single_answer_request and formal_signal and not open_ended_request

    @classmethod
    def _requires_autonomous_cycle(
        cls,
        *,
        user_text: str,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
        tool_plan: ProModeToolPlan | None,
        prior_pro_mode_state: dict[str, Any] | None,
    ) -> bool:
        if tool_plan is not None:
            return False
        selection_context = dict(selection_context or {})
        if cls._is_autonomy_follow_up_turn(user_text, prior_pro_mode_state):
            return True
        if uploaded_files:
            return False
        if selection_context.get("resource_uris") or selection_context.get("dataset_uris"):
            return False
        if cls._selection_context_has_artifact_handles(selection_context):
            return False
        if cls._requires_rigorous_proof_workflow(
            user_text=user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
            tool_plan=tool_plan,
        ):
            return False
        if cls._requires_self_contained_reasoning_solver(
            user_text=user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
            tool_plan=tool_plan,
        ):
            return False
        if cls._is_report_like_request(user_text):
            return False
        lowered = str(user_text or "").strip().lower()
        if not cls._requires_deep_reasoning(user_text):
            return False
        long_cycle_signal = bool(
            re.search(
                r"\b("
                r"agentic|autonomous|long[- ]cycle|long thinking|multi-step|cross-turn|"
                r"resume|checkpoint|benchmark|ablation|paper|neurips|architecture|"
                r"system design|workflow|controller|evaluation plan|research agenda|"
                r"roadmap|tradeoff analysis|design a framework|come up with a plan"
                r")\b",
                lowered,
            )
        )
        open_ended_signal = bool(
            re.search(
                r"\b(plan|proposal|evaluate|compare|assess|review|synthesize|design|stress[- ]test)\b",
                lowered,
            )
        )
        return bool(long_cycle_signal and open_ended_signal and len(lowered) >= 220)

    @staticmethod
    def _is_option_based_reasoning_request(user_text: str) -> bool:
        text = str(user_text or "")
        lowered = text.strip().lower()
        if not lowered:
            return False
        option_lines = len(re.findall(r"(?m)^[ \t]*[A-D][\.\)]\s+", text))
        if option_lines >= 4:
            return True
        has_option_language = bool(re.search(r"\b(option|choice|statement|answer)\b", lowered))
        discriminative_language = bool(
            re.search(
                r"\b(correct|incorrect|except|best|most likely|least likely|false|true)\b", lowered
            )
        )
        return has_option_language and discriminative_language

    @staticmethod
    def _requests_collective_reasoning_council(user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        if not lowered:
            return False
        return bool(
            re.search(
                r"\b("
                r"collective reasoning|expert council|multi-agent|multi agent|multi-perspective|"
                r"multiple perspectives|debate|consensus|argue both sides|stress-test|"
                r"socratic crux|contrarian pass|red team"
                r")\b",
                lowered,
            )
        )

    @staticmethod
    def _research_program_tool_bundle(
        *,
        user_text: str,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
        prior_pro_mode_state: dict[str, Any] | None = None,
    ) -> list[str]:
        lowered = str(user_text or "").strip().lower()
        selection_context = dict(selection_context or {})
        chosen: list[str] = []

        def _extend(names: tuple[str, ...]) -> None:
            for tool_name in names:
                if tool_name in TOOL_SCHEMA_MAP and tool_name not in chosen:
                    chosen.append(tool_name)

        direct_image_analysis = bool(
            AgnoChatRuntime._has_direct_image_target(
                user_text=user_text,
                uploaded_files=uploaded_files,
                selection_context=selection_context,
            )
            or re.search(
                r"\b(attached|uploaded|new image|new aerial image|this image|analy[sz]e it|analy[sz]e this|look at this|local image)\b",
                lowered,
            )
        )
        quantitative_request = bool(
            re.search(
                r"\b(count|average|mean|median|statistics|distribution|compare|population|density|box size|annotation|bounding[- ]box|bbox)\b",
                lowered,
            )
        )
        report_like_request = bool(
            re.search(r"\b(report|summary|analysis|ecological|summarize|compare)\b", lowered)
        )
        artifact_handle_context = AgnoChatRuntime._selection_context_has_artifact_handles(
            selection_context
        )
        data_context = bool(
            selection_context.get("resource_uris")
            or selection_context.get("dataset_uris")
            or artifact_handle_context
            or AgnoChatRuntime._is_prior_analysis_follow_up_request(
                user_text,
                prior_state=prior_pro_mode_state,
            )
            or re.search(r"\b(bisque|dataset|resource|catalog)\b", lowered)
        )
        object_pattern_request = bool(
            re.search(
                r"\b(prairie dog|burrow|object|bbox|bounding[- ]box|box size|detection|detect|class pattern|population)\b",
                lowered,
            )
        )
        acquisition_request = bool(
            re.search(
                r"\b(spectral|frequency|instability|active learning|hard sample|uncertain(?:ty)? ranking|acquisition)\b",
                lowered,
            )
        )
        plot_request = AgnoChatRuntime._is_plot_or_visual_analysis_request(user_text)

        if direct_image_analysis:
            _extend(RESEARCH_PROGRAM_TOOL_BUNDLES["vision"])
        if data_context:
            _extend(RESEARCH_PROGRAM_TOOL_BUNDLES["catalog"])
        if data_context and object_pattern_request:
            _extend(RESEARCH_PROGRAM_TOOL_BUNDLES["vision"])
        if acquisition_request and (direct_image_analysis or data_context):
            _extend(RESEARCH_PROGRAM_TOOL_BUNDLES["acquisition"])
        if quantitative_request or report_like_request or direct_image_analysis:
            _extend(RESEARCH_PROGRAM_TOOL_BUNDLES["analysis"])
        if plot_request or re.search(
            r"\b(code|python|script|parse|optimi[sz]e|benchmark|custom analysis|xml|csv|json)\b",
            lowered,
        ):
            _extend(RESEARCH_PROGRAM_TOOL_BUNDLES["code"])
        if not chosen:
            _extend(RESEARCH_PROGRAM_TOOL_BUNDLES["analysis"])
            _extend(RESEARCH_PROGRAM_TOOL_BUNDLES["code"])
        return chosen

    @staticmethod
    def _requires_tool_workflow(
        *,
        user_text: str,
        uploaded_files: list[str],
        selected_tool_names: list[str],
        selection_context: dict[str, Any] | None,
        inferred_tool_names: list[str],
    ) -> bool:
        if selected_tool_names:
            return True
        if uploaded_files:
            return True
        selection_context = dict(selection_context or {})
        if selection_context.get("resource_uris") or selection_context.get("dataset_uris"):
            return True
        if AgnoChatRuntime._selection_context_has_artifact_handles(selection_context):
            return True
        lowered = str(user_text or "").strip().lower()
        conceptual_discussion_signal = bool(
            re.search(
                r"\b("
                r"compare|comparison|difference|different|tradeoff|trade-off|"
                r"assumption|assumptions|limitation|limitations|pros and cons|"
                r"when does|when do|why|explain|overview|what is|how does"
                r")\b",
                lowered,
            )
        )
        operational_tool_signal = bool(
            re.search(
                r"\b("
                r"run|use|execute|apply|segment this|detect this|process this|"
                r"analyze this|analyze that|upload|download|search|find|list|show|"
                r"open|load|browse|create|add|delete|annotate|tag|module"
                r")\b",
                lowered,
            )
        )
        if conceptual_discussion_signal and not operational_tool_signal:
            return False
        if AgnoChatRuntime._is_code_execution_request(user_text):
            return True
        explicit_tool_signal = bool(
            re.search(
                r"\b("
                r"run|use|execute|tool|sandbox|python|code|script|optimi[sz]e|benchmark|mAP|"
                r"yolo|sam2|sam3|segment|segmentation|detect|detection|depth|depthpro|"
                r"bisque|dataset|resource|image|images|file|files|upload|download|csv|table|"
                r"analyze this image|process this image|module"
                r")\b",
                lowered,
            )
        )
        return explicit_tool_signal and bool(inferred_tool_names)

    @staticmethod
    def _requires_deep_reasoning(user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        if re.search(
            r"\b(previous|earlier|above|that|this|those|these|continue|follow[- ]up)\b", lowered
        ):
            return True
        if re.search(
            r"\b("
            r"calculate|compute|derive|prove|show that|evaluate|integral|sum|count|how many|"
            r"rate|energy|phase|amplitude|probability|distinct|exact|nmr|mechanism|"
            r"justify|analyze carefully|work through|step by step|compare carefully"
            r")\b",
            lowered,
        ):
            return True
        if re.search(r"\d", lowered) and re.search(
            r"[=<>]|10\^|e[-+]?\d+|mev|ev|hz|nm|cm|kg|mol|s\b", lowered
        ):
            return True
        sentence_count = len(re.findall(r"[.!?]+", lowered))
        return len(lowered) > 280 or sentence_count >= 3

    @staticmethod
    def _requires_codeexec_reasoning_agent(user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        if not lowered or not AgnoChatRuntime._is_code_execution_request(lowered):
            return False
        return AgnoChatRuntime._requires_deep_reasoning(lowered) or bool(
            re.search(
                r"\b("
                r"bootstrap|cross[- ]?validation|diagnostic plot|diagnostic plots|"
                r"compare .*model|compare .*famil|model famil(?:y|ies)|nonlinear model|"
                r"confidence interval|uncertainty|justify|best model|why .* preferable|"
                r"quality[- ]control report|qc report|outlier"
                r")\b",
                lowered,
            )
        )

    @staticmethod
    def _requires_validated_numeric_workflow(
        *,
        user_text: str,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
    ) -> bool:
        if uploaded_files:
            return False
        selection_context = dict(selection_context or {})
        if selection_context.get("resource_uris") or selection_context.get("dataset_uris"):
            return False
        lowered = str(user_text or "").strip().lower()
        if re.search(
            r"\b(why|explain|mechanism|pathway|cause|interpret|research|paper)\b", lowered
        ):
            return False
        if re.search(r"\b(prove|proof|theorem|lemma|corollary|rigorous|self-contained)\b", lowered):
            return False
        asks_for_numeric_answer = bool(
            re.search(
                r"\b(calculate|compute|derive|evaluate|solve|determine|find|estimate|how large|how much|how many|what is|what's)\b",
                lowered,
            )
        )
        formal_signal = bool(
            re.search(
                r"\d|[a-z]\d|\be\d\b|τ|tau|energy|difference|separation|rate|phase|amplitude|probability|frequency|wavelength|linewidth|lifetime|resolved|resolution",
                lowered,
            )
        )
        open_ended_research = bool(
            re.search(
                r"\b(research question|open problem|hypothesis|proposal|design an experiment)\b",
                lowered,
            )
        )
        return asks_for_numeric_answer and formal_signal and not open_ended_research

    def _infer_pro_mode_tool_subset(
        self,
        *,
        messages: list[dict[str, Any]],
        uploaded_files: list[str],
        selected_tool_names: list[str] | None,
    ) -> list[str]:
        explicit = self._normalize_selected_tool_names(selected_tool_names)
        if explicit:
            return [tool_name for tool_name in explicit if tool_name in TOOL_SCHEMA_MAP]
        from src.tooling.tool_selection import _select_tool_subset

        selected_schemas = _select_tool_subset(
            messages=[
                {"role": str(item.get("role") or ""), "content": str(item.get("content") or "")}
                for item in messages
            ],
            uploaded_files=uploaded_files,
            all_tools=ALL_TOOL_SCHEMAS,
        )
        inferred: list[str] = []
        for schema in selected_schemas:
            tool_name = str(schema.get("function", {}).get("name") or "").strip()
            if tool_name == "repro_report":
                continue
            if tool_name and tool_name in TOOL_SCHEMA_MAP and tool_name not in inferred:
                inferred.append(tool_name)
        latest_user_text = str((messages or [{}])[-1].get("content") or "").strip().lower()
        if uploaded_files and not re.search(
            r"\b(bisque|dataset|resource|upload to bisque|download|catalog|module|annotate|tag resource)\b",
            latest_user_text,
        ):
            bisque_management_tools = {
                "bisque_find_assets",
                "upload_to_bisque",
                "bisque_ping",
                "bisque_download_resource",
                "search_bisque_resources",
                "load_bisque_resource",
                "bisque_download_dataset",
                "bisque_create_dataset",
                "bisque_add_to_dataset",
                "delete_bisque_resource",
                "add_tags_to_resource",
                "bisque_fetch_xml",
                "bisque_add_gobjects",
                "bisque_advanced_search",
                "run_bisque_module",
            }
            inferred = [
                tool_name for tool_name in inferred if tool_name not in bisque_management_tools
            ]
        if self._prefers_megaseg_segmentation(latest_user_text):
            inferred = [
                tool_name
                for tool_name in inferred
                if tool_name
                not in {"segment_image_sam2", "segment_image_sam3", "sam2_prompt_image"}
            ]
            if "segment_image_megaseg" not in inferred:
                inferred.insert(0, "segment_image_megaseg")
        return inferred

    @staticmethod
    def _tool_workflow_required_tools(
        *,
        user_text: str,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
        selected_tool_names: list[str],
    ) -> tuple[list[str], bool]:
        normalized_selected = AgnoChatRuntime._normalize_selected_tool_names(selected_tool_names)
        if not normalized_selected:
            return [], False
        lowered = str(user_text or "").strip().lower()
        has_direct_image_target = AgnoChatRuntime._has_direct_image_target(
            user_text=user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
        )
        if AgnoChatRuntime._is_code_execution_request(user_text):
            return ["codegen_python_plan", "execute_python_job"], True
        if AgnoChatRuntime._requires_validated_numeric_workflow(
            user_text=user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
        ):
            return ["numpy_calculator"], True
        if has_direct_image_target:
            if re.search(
                r"\b(spectral|frequency|instability|active learning|hard sample|uncertain(?:ty)? ranking)\b",
                lowered,
            ):
                return ["analyze_prediction_stability"], True
            if AgnoChatRuntime._is_depth_segmentation_request(user_text):
                required = ["estimate_depth_pro", "segment_image_sam2"]
                if AgnoChatRuntime._segmentation_request_needs_quantification(user_text):
                    required.append("quantify_segmentation_masks")
                return required, True
            if AgnoChatRuntime._is_depth_request(user_text):
                return ["estimate_depth_pro"], True
            if re.search(r"\b(yolo|detect|detection|bounding box|bbox)\b", lowered):
                return ["yolo_detect"], True
            if AgnoChatRuntime._is_segmentation_request(user_text):
                required = AgnoChatRuntime._preferred_segmentation_tool_names(user_text)
                return required, True
            if AgnoChatRuntime._is_image_metadata_request(user_text):
                return ["bioio_load_image"], True
        return normalized_selected, False

    @staticmethod
    def _tool_workflow_satisfied(
        *,
        tool_invocations: list[dict[str, Any]],
        required_tool_names: list[str],
        strict_validation: bool,
    ) -> bool:
        used_tool_names = {
            str(invocation.get("tool") or "").strip()
            for invocation in list(tool_invocations or [])
            if str(invocation.get("status") or "").strip().lower() == "completed"
        }
        required = [tool_name for tool_name in required_tool_names if tool_name]
        if not required:
            return bool(used_tool_names) or not strict_validation
        if strict_validation:
            return set(required).issubset(used_tool_names)
        return bool(used_tool_names.intersection(required))

    @staticmethod
    def _completed_tool_names(tool_invocations: list[dict[str, Any]]) -> set[str]:
        return {
            str(invocation.get("tool") or "").strip()
            for invocation in list(tool_invocations or [])
            if str(invocation.get("status") or "").strip().lower() == "completed"
            and str(invocation.get("tool") or "").strip()
        }

    @classmethod
    def _missing_required_tool_names(
        cls,
        *,
        tool_invocations: list[dict[str, Any]],
        required_tool_names: list[str],
    ) -> list[str]:
        completed = cls._completed_tool_names(tool_invocations)
        return [
            str(tool_name or "").strip()
            for tool_name in list(required_tool_names or [])
            if str(tool_name or "").strip() and str(tool_name or "").strip() not in completed
        ]

    @staticmethod
    def _latest_result_refs_from_tool_invocations(
        tool_invocations: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        refs: dict[str, Any] = {}
        for invocation in list(tool_invocations or []):
            if str(invocation.get("status") or "").strip().lower() != "completed":
                continue
            envelope = invocation.get("output_envelope")
            if not isinstance(envelope, dict):
                continue
            raw_refs = envelope.get("latest_result_refs")
            if isinstance(raw_refs, dict):
                for key, value in raw_refs.items():
                    if value is None:
                        continue
                    refs[str(key)] = value
            result_group_id = (
                str(envelope.get("result_group_id") or "").strip()
                or str(refs.get("latest_segmentation_result_group_id") or "").strip()
            )
            if result_group_id:
                refs.setdefault("latest_segmentation_result_group_id", result_group_id)
        return refs

    def _stabilize_pro_mode_intake_decision(
        self,
        *,
        decision: ProModeIntakeDecision,
        messages: list[dict[str, Any]],
        latest_user_text: str,
        uploaded_files: list[str],
        selected_tool_names: list[str] | None,
        selection_context: dict[str, Any] | None,
        prior_pro_mode_state: dict[str, Any] | None = None,
    ) -> ProModeIntakeDecision:
        stabilized = decision.model_copy(deep=True)
        collective_council_requested = self._requests_collective_reasoning_council(latest_user_text)
        selection_context_payload = dict(selection_context or {})
        text_only_collective_council = bool(
            collective_council_requested
            and not uploaded_files
            and not bool(
                selection_context_payload.get("resource_uris")
                or selection_context_payload.get("dataset_uris")
            )
            and not self._selection_context_has_artifact_handles(selection_context_payload)
            and not self._normalize_selected_tool_names(selected_tool_names)
        )
        inferred_tool_names = self._infer_pro_mode_tool_subset(
            messages=messages,
            uploaded_files=uploaded_files,
            selected_tool_names=selected_tool_names,
        )
        tool_plan = self._build_pro_mode_tool_plan(
            user_text=latest_user_text,
            uploaded_files=uploaded_files,
            selected_tool_names=self._normalize_selected_tool_names(selected_tool_names),
            selection_context=selection_context,
            inferred_tool_names=inferred_tool_names,
            prior_pro_mode_state=prior_pro_mode_state,
        )

        def _apply_policy(
            *,
            route: str,
            tool_plan_override: ProModeToolPlan | None = None,
            regime_override: str | None = None,
            task_regime_override: str | None = None,
        ) -> None:
            computed_task_regime = self._task_regime_for_turn(
                user_text=latest_user_text,
                uploaded_files=uploaded_files,
                selection_context=selection_context,
                tool_plan=tool_plan_override,
            )
            regime = str(
                regime_override
                or self._execution_regime_for_decision(route=route, tool_plan=tool_plan_override)
            ).strip()
            task_regime = str(task_regime_override or computed_task_regime).strip()
            policy = self._context_policy_from_decision(
                route=route,
                execution_regime=regime,
                task_regime=task_regime,
                latest_user_text=latest_user_text,
                uploaded_files=uploaded_files,
                selection_context=selection_context,
                prior_pro_mode_state=prior_pro_mode_state,
            )
            stabilized.execution_regime = regime
            stabilized.task_regime = task_regime
            stabilized.load_memory = bool(policy.get("load_memory"))
            stabilized.load_knowledge = bool(policy.get("load_knowledge"))
            stabilized.recent_history_turns = max(0, int(policy.get("history_window") or 0))
            stabilized.context_policy = stabilized.context_policy.model_validate(policy)

        if self._is_phatic_turn(latest_user_text):
            stabilized.route = "direct_response"
            stabilized.selected_tool_names = []
            stabilized.tool_plan_category = None
            stabilized.strict_tool_validation = False
            if not str(stabilized.direct_response or "").strip():
                if "thank" in str(latest_user_text or "").strip().lower():
                    stabilized.direct_response = (
                        "You're welcome. What would you like to work on next?"
                    )
                else:
                    stabilized.direct_response = "Hello! How can I assist you today?"
            _apply_policy(route="direct_response", tool_plan_override=None)
            return stabilized
        if (
            self._is_proof_follow_up_turn(latest_user_text, prior_pro_mode_state)
            and tool_plan is None
        ):
            stabilized.route = "deep_reasoning"
            stabilized.direct_response = None
            stabilized.selected_tool_names = []
            stabilized.tool_plan_category = None
            stabilized.strict_tool_validation = False
            stabilized.reason = (
                "Continue the saved rigorous proof state with the single frontier reasoning solver."
            )
            _apply_policy(
                route="deep_reasoning",
                tool_plan_override=None,
                regime_override="reasoning_solver",
                task_regime_override="rigorous_proof",
            )
            return stabilized
        if text_only_collective_council:
            stabilized.route = "deep_reasoning"
            stabilized.direct_response = None
            stabilized.selected_tool_names = []
            stabilized.tool_plan_category = None
            stabilized.strict_tool_validation = False
            stabilized.reason = "Explicit collective-reasoning cues were routed into the single frontier reasoning solver for production stability."
            collective_task_regime = (
                "self_contained_reasoning"
                if (
                    self._requires_deep_reasoning(latest_user_text)
                    and not self._is_report_like_request(latest_user_text)
                )
                else None
            )
            _apply_policy(
                route="deep_reasoning",
                tool_plan_override=None,
                regime_override="reasoning_solver",
                task_regime_override=collective_task_regime,
            )
            return stabilized
        if tool_plan is not None:
            if tool_plan.category == "code_execution" and self._requires_codeexec_reasoning_agent(
                latest_user_text
            ):
                stabilized.route = "deep_reasoning"
                stabilized.reason = (
                    str(tool_plan.reason or stabilized.reason or "").strip()
                    + " Hard computational prompt upgraded to the dedicated code-execution reasoning agent."
                ).strip()
                stabilized.direct_response = None
                stabilized.selected_tool_names = list(tool_plan.selected_tool_names)
                stabilized.tool_plan_category = str(tool_plan.category or "").strip() or None
                stabilized.strict_tool_validation = bool(tool_plan.strict_validation)
                _apply_policy(
                    route="deep_reasoning",
                    tool_plan_override=tool_plan,
                    regime_override="reasoning_solver",
                    task_regime_override="self_contained_reasoning",
                )
                return stabilized
            stabilized.route = "tool_workflow"
            stabilized.reason = str(tool_plan.reason or stabilized.reason or "").strip()
            stabilized.direct_response = None
            stabilized.selected_tool_names = list(tool_plan.selected_tool_names)
            stabilized.tool_plan_category = str(tool_plan.category or "").strip() or None
            stabilized.strict_tool_validation = bool(tool_plan.strict_validation)
            _apply_policy(route="tool_workflow", tool_plan_override=tool_plan)
            return stabilized
        if self._requires_rigorous_proof_workflow(
            user_text=latest_user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
            tool_plan=tool_plan,
        ):
            stabilized.route = "deep_reasoning"
            stabilized.direct_response = None
            stabilized.selected_tool_names = []
            stabilized.tool_plan_category = None
            stabilized.strict_tool_validation = False
            _apply_policy(
                route="deep_reasoning",
                tool_plan_override=None,
                regime_override="reasoning_solver",
                task_regime_override="rigorous_proof",
            )
            return stabilized
        if collective_council_requested:
            stabilized.route = "deep_reasoning"
            stabilized.direct_response = None
            stabilized.selected_tool_names = []
            stabilized.tool_plan_category = None
            stabilized.strict_tool_validation = False
            stabilized.reason = "Explicit collective-reasoning cues were routed into the single frontier reasoning solver for production stability."
            collective_task_regime = (
                "self_contained_reasoning"
                if (
                    not uploaded_files
                    and not bool(
                        dict(selection_context or {}).get("resource_uris")
                        or dict(selection_context or {}).get("dataset_uris")
                    )
                    and not self._selection_context_has_artifact_handles(selection_context)
                    and self._requires_deep_reasoning(latest_user_text)
                    and not self._is_report_like_request(latest_user_text)
                )
                else None
            )
            _apply_policy(
                route="deep_reasoning",
                tool_plan_override=None,
                regime_override="reasoning_solver",
                task_regime_override=collective_task_regime,
            )
            return stabilized
        if self._pro_mode_autonomous_cycle_enabled() and self._requires_autonomous_cycle(
            user_text=latest_user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
            tool_plan=tool_plan,
            prior_pro_mode_state=prior_pro_mode_state,
        ):
            stabilized.route = "deep_reasoning"
            stabilized.direct_response = None
            stabilized.selected_tool_names = []
            stabilized.tool_plan_category = None
            stabilized.strict_tool_validation = False
            stabilized.reason = "The turn looks like an open-ended, long-cycle reasoning problem that benefits from checkpointed Think -> Act -> Analyze loops."
            _apply_policy(
                route="deep_reasoning",
                tool_plan_override=None,
                regime_override="autonomous_cycle",
                task_regime_override="conceptual_high_uncertainty",
            )
            return stabilized
        if self._requires_self_contained_reasoning_solver(
            user_text=latest_user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
            tool_plan=tool_plan,
        ):
            stabilized.route = "deep_reasoning"
            stabilized.direct_response = None
            stabilized.selected_tool_names = []
            stabilized.tool_plan_category = None
            stabilized.strict_tool_validation = False
            _apply_policy(
                route="deep_reasoning",
                tool_plan_override=None,
                regime_override="reasoning_solver",
                task_regime_override="self_contained_reasoning",
            )
            return stabilized
        if (
            tool_plan is None
            and not uploaded_files
            and not bool(
                dict(selection_context or {}).get("resource_uris")
                or dict(selection_context or {}).get("dataset_uris")
            )
            and not self._selection_context_has_artifact_handles(selection_context)
            and self._is_report_like_request(latest_user_text)
        ):
            stabilized.route = "deep_reasoning"
            stabilized.direct_response = None
            stabilized.selected_tool_names = []
            stabilized.tool_plan_category = None
            stabilized.strict_tool_validation = False
            stabilized.reason = "Open-ended report requests stay on the frontier reasoning solver in production; the focused team is benchmark-only."
            _apply_policy(
                route="deep_reasoning",
                tool_plan_override=None,
                regime_override="reasoning_solver",
            )
            return stabilized
        if self._requires_tool_workflow(
            user_text=latest_user_text,
            uploaded_files=uploaded_files,
            selected_tool_names=self._normalize_selected_tool_names(selected_tool_names),
            selection_context=selection_context,
            inferred_tool_names=inferred_tool_names,
        ):
            stabilized.route = "tool_workflow"
            stabilized.direct_response = None
            stabilized.selected_tool_names = inferred_tool_names
            stabilized.tool_plan_category = None
            stabilized.strict_tool_validation = False
            _apply_policy(route="tool_workflow", tool_plan_override=None)
            return stabilized
        if self._requires_deep_reasoning(latest_user_text):
            stabilized.route = "deep_reasoning"
            stabilized.direct_response = None
            stabilized.selected_tool_names = []
            stabilized.tool_plan_category = None
            stabilized.strict_tool_validation = False
            _apply_policy(route="deep_reasoning", tool_plan_override=None)
            return stabilized
        stabilized.selected_tool_names = []
        stabilized.tool_plan_category = None
        stabilized.strict_tool_validation = False
        _apply_policy(route=stabilized.route, tool_plan_override=None)
        return stabilized

    def _select_route(self, turn_intent: AgnoTurnIntent) -> AgnoRouteDecision:
        explicit_tools = set(self._normalize_selected_tool_names(turn_intent.selected_tool_names))
        workflow_id = str(turn_intent.workflow_hint.get("id") or "").strip().lower()
        if workflow_id == "pro_mode":
            return AgnoRouteDecision(
                primary_domain="core",
                selected_domains=["core"],
                reason="workflow_hint",
                available_tool_names=sorted(TOOL_SCHEMA_MAP.keys()),
                requires_approval=False,
                approval_tool_names=[],
            )
        if workflow_id == "detect_prairie_dog" or "yolo_detect" in explicit_tools:
            return AgnoRouteDecision(
                primary_domain="ecology",
                selected_domains=["ecology"],
                reason="selected_tool_constraint",
                available_tool_names=list(explicit_tools or {"yolo_detect"}),
                requires_approval=bool(explicit_tools & APPROVAL_REQUIRED_TOOL_NAMES),
                approval_tool_names=sorted(explicit_tools & APPROVAL_REQUIRED_TOOL_NAMES),
            )
        if workflow_id == "chemistry_workbench" or (
            explicit_tools and explicit_tools.issubset(self._CORE_TOOL_NAMES)
        ):
            return AgnoRouteDecision(
                primary_domain="core",
                selected_domains=["core"],
                reason="selected_tool_constraint",
                available_tool_names=sorted(explicit_tools),
                requires_approval=bool(explicit_tools & APPROVAL_REQUIRED_TOOL_NAMES),
                approval_tool_names=sorted(explicit_tools & APPROVAL_REQUIRED_TOOL_NAMES),
            )

        routed = route_scientist_turn(
            user_text=turn_intent.user_text,
            file_ids=list(turn_intent.selection_context.get("focused_file_ids") or []),
            resource_uris=list(turn_intent.selection_context.get("resource_uris") or []),
            dataset_uris=list(turn_intent.selection_context.get("dataset_uris") or []),
            selected_tool_names=turn_intent.selected_tool_names,
            knowledge_context=turn_intent.knowledge_context,
            selection_context=turn_intent.selection_context,
            workflow_hint=turn_intent.workflow_hint,
        )
        return AgnoRouteDecision(
            primary_domain=str(routed.primary_domain or "core"),
            selected_domains=list(routed.selected_domains or ["core"]),
            reason=str(routed.route_reason or "heuristic"),
            available_tool_names=list(routed.available_tool_names or []),
            requires_approval=bool(routed.requires_approval),
            approval_tool_names=list(routed.approval_tool_names or []),
        )

    def _allowed_tools_for_route(
        self,
        *,
        route: AgnoRouteDecision,
        selected_tool_names: list[str],
    ) -> list[str]:
        allowed = set(route.available_tool_names or [])
        explicit = self._normalize_selected_tool_names(selected_tool_names)
        filtered = [tool_name for tool_name in explicit if tool_name in TOOL_SCHEMA_MAP]
        if allowed:
            filtered = [tool_name for tool_name in filtered if tool_name in allowed] or filtered
        return filtered

    @staticmethod
    def _looks_like_bisque_catalog_question(user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        if not lowered or "bisque" not in lowered:
            return False
        if not re.search(
            r"\b(dataset|datasets|resource|resources|image|images|file|files|table|tables|hdf5|h5|dream3d)\b",
            lowered,
        ):
            return False
        return bool(
            re.search(
                r"\b(search|find|list|browse|show me|look up|which|what|any|recent|latest|most recent)\b",
                lowered,
            )
            or "are there any" in lowered
            or "do i have any" in lowered
            or "what do i have" in lowered
        )

    def _infer_default_tool_names_for_turn(
        self,
        *,
        turn_intent: AgnoTurnIntent,
        route: AgnoRouteDecision,
        explicit_tool_names: list[str],
    ) -> list[str]:
        if explicit_tool_names:
            return []
        workflow_id = str(turn_intent.workflow_hint.get("id") or "").strip().lower()
        if workflow_id:
            return []
        if not route.available_tool_names:
            return []
        available = set(route.available_tool_names or [])
        if "search_bisque_resources" in available and self._looks_like_bisque_catalog_question(
            turn_intent.user_text
        ):
            return ["search_bisque_resources"]
        return []

    def _json_result(self, value: Any) -> tuple[Any, str]:
        if isinstance(value, dict):
            return value, json.dumps(value, ensure_ascii=False)
        raw = str(value or "").strip()
        if not raw:
            return {}, raw
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}, raw
        return parsed, raw

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

    def _summarize_tool_output(self, tool_name: str, parsed: Any, raw_text: str) -> dict[str, Any]:
        if isinstance(parsed, dict):
            if tool_name == "bisque_download_dataset":
                download_rows = (
                    parsed.get("download_rows")
                    if isinstance(parsed.get("download_rows"), list)
                    else []
                )
                if not download_rows and isinstance(parsed.get("results"), list):
                    download_rows = [
                        {
                            "resource_uri": item.get("resource_uri") or item.get("uri"),
                            "output_path": item.get("output_path") or item.get("path"),
                            "success": item.get("success"),
                        }
                        for item in parsed.get("results") or []
                        if isinstance(item, dict)
                    ]
                return {
                    "success": bool(parsed.get("success")),
                    "kind": "bisque_dataset_download",
                    "dataset_uri": parsed.get("dataset_uri"),
                    "output_dir": parsed.get("output_dir"),
                    "total_members": parsed.get("total_members"),
                    "downloaded": parsed.get("downloaded"),
                    "sample_downloads": [
                        {
                            "resource_uri": item.get("resource_uri"),
                            "output_path": item.get("output_path"),
                        }
                        for item in download_rows[:5]
                        if isinstance(item, dict)
                    ],
                }
            if tool_name == "load_bisque_resource":
                resource = (
                    parsed.get("resource") if isinstance(parsed.get("resource"), dict) else {}
                )
                tags = resource.get("tags") if isinstance(resource.get("tags"), list) else []
                dimensions = (
                    resource.get("dimensions")
                    if isinstance(resource.get("dimensions"), dict)
                    else {}
                )
                return {
                    "success": bool(parsed.get("success")),
                    "kind": "bisque_resource",
                    "resource_uri": resource.get("uri"),
                    "resource_name": resource.get("name"),
                    "resource_type": resource.get("resource_type"),
                    "created": resource.get("created"),
                    "tag_count": len(tags),
                    "tags": [
                        {
                            "name": item.get("name"),
                            "value": item.get("value"),
                        }
                        for item in tags[:12]
                        if isinstance(item, dict)
                    ],
                    "dimensions": dimensions,
                }
            if tool_name == "bioio_load_image":
                dimensions = (
                    parsed.get("dimensions") if isinstance(parsed.get("dimensions"), dict) else {}
                )
                if not dimensions and isinstance(parsed.get("axis_sizes"), dict):
                    dimensions = {
                        str(axis): int(value)
                        for axis, value in parsed.get("axis_sizes", {}).items()
                        if str(axis).strip() and isinstance(value, (int, float))
                    }
                metadata = (
                    parsed.get("metadata") if isinstance(parsed.get("metadata"), dict) else {}
                )
                header = metadata.get("header") if isinstance(metadata.get("header"), dict) else {}
                exif = metadata.get("exif") if isinstance(metadata.get("exif"), dict) else {}
                geo = metadata.get("geo") if isinstance(metadata.get("geo"), dict) else {}
                filename_hints = (
                    metadata.get("filename_hints")
                    if isinstance(metadata.get("filename_hints"), dict)
                    else {}
                )
                captured_at = (
                    str(
                        parsed.get("captured_at")
                        or exif.get("DateTimeOriginal")
                        or exif.get("DateTime")
                        or ""
                    ).strip()
                    or None
                )
                return {
                    "success": bool(parsed.get("success")),
                    "kind": "bioio_image",
                    "file_path": parsed.get("file_path"),
                    "reader": parsed.get("reader"),
                    "dims_order": parsed.get("dims_order"),
                    "array_shape": list(parsed.get("array_shape") or [])[:8]
                    if isinstance(parsed.get("array_shape"), list)
                    else [],
                    "dimensions": dimensions,
                    "preview_path": parsed.get("preview_path"),
                    "array_path": parsed.get("array_path"),
                    "header": {
                        key: value
                        for key, value in header.items()
                        if key in {"Format", "Color mode", "Frame count", "Bit depth"}
                    },
                    "exif": {
                        key: value
                        for key, value in exif.items()
                        if key
                        in {
                            "DateTimeOriginal",
                            "DateTime",
                            "Make",
                            "Model",
                            "Software",
                            "LensModel",
                            "ExposureTime",
                            "FNumber",
                            "ISOSpeedRatings",
                            "ImageDescription",
                        }
                    },
                    "geo": {
                        key: value
                        for key, value in geo.items()
                        if key in {"latitude", "longitude", "altitude_m", "coordinate_reference"}
                    },
                    "filename_hints": {
                        str(key): value
                        for key, value in list(filename_hints.items())[:12]
                        if str(key).strip()
                    },
                    "captured_at": captured_at,
                }
            if tool_name == "yolo_detect":
                scientific_summary = (
                    parsed.get("scientific_summary")
                    if isinstance(parsed.get("scientific_summary"), dict)
                    else {}
                )
                overall = (
                    scientific_summary.get("overall")
                    if isinstance(scientific_summary.get("overall"), dict)
                    else {}
                )
                prediction_records = (
                    parsed.get("prediction_image_records")
                    if isinstance(parsed.get("prediction_image_records"), list)
                    else []
                )
                stability_audit = (
                    parsed.get("prediction_stability_audit")
                    if isinstance(parsed.get("prediction_stability_audit"), dict)
                    else {}
                )
                compact_records: list[dict[str, Any]] = []
                for item in prediction_records[:5]:
                    if not isinstance(item, dict):
                        continue
                    compact_records.append(
                        {
                            "source_name": self._display_artifact_name(item.get("source_name")),
                            "box_count": item.get("box_count"),
                            "class_counts": item.get("class_counts"),
                            "image_width": item.get("image_width"),
                            "image_height": item.get("image_height"),
                        }
                    )
                return {
                    "success": bool(parsed.get("success")),
                    "kind": "yolo",
                    "model_name": parsed.get("model_name"),
                    "predictions_json": parsed.get("predictions_json"),
                    "counts_by_class": parsed.get("counts_by_class"),
                    "total_boxes": overall.get("total_boxes")
                    or parsed.get("metrics", {}).get("total_boxes"),
                    "scientific_summary": {"overall": overall} if overall else {},
                    "per_image": compact_records,
                    "prediction_stability": {
                        "summary": stability_audit.get("summary"),
                        "review_candidates": list(stability_audit.get("review_candidates") or [])[
                            :5
                        ],
                        "backend_method": stability_audit.get("backend_method"),
                    }
                    if stability_audit
                    else {},
                }
            if tool_name in {"score_spectral_instability", "analyze_prediction_stability"}:
                summary = parsed.get("summary") if isinstance(parsed.get("summary"), dict) else {}
                top_ranked = (
                    summary.get("top_ranked") if isinstance(summary.get("top_ranked"), list) else []
                )
                compact_ranked: list[dict[str, Any]] = []
                for item in top_ranked[:5]:
                    if not isinstance(item, dict):
                        continue
                    compact_ranked.append(
                        {
                            "file_name": self._display_artifact_name(item.get("file_name")),
                            "score": item.get("score"),
                        }
                    )
                return {
                    "success": bool(parsed.get("success")),
                    "kind": "prediction_stability",
                    "analysis_kind": parsed.get("analysis_kind"),
                    "method": parsed.get("method"),
                    "backend_method": parsed.get("backend_method"),
                    "method_version": parsed.get("method_version"),
                    "model_name": parsed.get("model_name"),
                    "scores_json": parsed.get("scores_json"),
                    "image_count": summary.get("image_count"),
                    "nonzero_score_count": summary.get("nonzero_score_count"),
                    "max_score": summary.get("max_score"),
                    "mean_score": summary.get("mean_score"),
                    "median_score": summary.get("median_score"),
                    "top_ranked": compact_ranked,
                    "review_candidate_count": summary.get("review_candidate_count"),
                    "review_candidates": list(parsed.get("review_candidates") or [])[:5],
                    "active_learning_note": parsed.get("active_learning_note"),
                }
            if tool_name == "quantify_objects":
                distribution_summary = (
                    parsed.get("distribution_summary")
                    if isinstance(parsed.get("distribution_summary"), dict)
                    else {}
                )
                return {
                    "success": bool(parsed.get("success")),
                    "kind": "quantify_objects",
                    "total_objects": parsed.get("total_objects"),
                    "counts_by_class": parsed.get("counts_by_class"),
                    "class_summary": list(parsed.get("class_summary") or [])[:8],
                    "distribution_summary": {
                        key: value
                        for key, value in distribution_summary.items()
                        if key
                        in {"bbox_area_px", "width_px", "height_px", "equivalent_diameter_px"}
                    },
                    "object_row_count": parsed.get("object_row_count"),
                }
            if tool_name == "codegen_python_plan":
                return {
                    "success": bool(parsed.get("success")),
                    "kind": "codegen_python_plan",
                    "job_id": parsed.get("job_id"),
                    "attempt_index": parsed.get("attempt_index"),
                    "message": parsed.get("message"),
                }
            if tool_name == "execute_python_job":
                key_measurements = (
                    parsed.get("key_measurements")
                    if isinstance(parsed.get("key_measurements"), list)
                    else []
                )
                analysis_outputs = (
                    parsed.get("analysis_outputs")
                    if isinstance(parsed.get("analysis_outputs"), list)
                    else []
                )
                return {
                    "success": bool(parsed.get("success")),
                    "kind": "execute_python_job",
                    "job_id": parsed.get("job_id"),
                    "result_path": parsed.get("result_path"),
                    "durable_execution": parsed.get("durable_execution"),
                    "durable_status": parsed.get("durable_status"),
                    "error_class": parsed.get("error_class"),
                    "error_message": parsed.get("error_message"),
                    "key_measurements": list(key_measurements)[:12],
                    "analysis_outputs": [
                        {
                            "path": item.get("path"),
                            "parse_status": item.get("parse_status"),
                        }
                        for item in analysis_outputs[:6]
                        if isinstance(item, dict)
                    ],
                }
            if tool_name == "plot_quantified_detections":
                summary = parsed.get("summary") if isinstance(parsed.get("summary"), dict) else {}
                return {
                    "success": bool(parsed.get("success")),
                    "kind": "plot_quantified_detections",
                    "summary": {
                        key: value
                        for key, value in summary.items()
                        if key
                        in {
                            "image_name",
                            "total_detections",
                            "confidence_threshold",
                            "low_confidence_count",
                            "low_confidence_fraction",
                            "counts_by_class",
                            "low_confidence_by_class",
                            "generated_plot_count",
                        }
                    },
                    "output_directory": parsed.get("output_directory"),
                    "output_files": list(parsed.get("output_files") or [])[:8],
                }
        if isinstance(parsed, dict):
            engine_summary = _progress_summary_from_result(tool_name, parsed)
            if isinstance(engine_summary, dict) and engine_summary:
                return engine_summary
        summary: dict[str, Any] = {}
        if isinstance(parsed, dict):
            for key in (
                "success",
                "processed",
                "total_files",
                "total_masks_generated",
                "prairie_dog_count",
                "burrow_count",
                "output_directory",
                "result_group_id",
            ):
                if key in parsed:
                    summary[key] = parsed[key]
            if tool_name == "yolo_detect" and isinstance(parsed.get("class_counts"), list):
                summary["classes"] = list(parsed.get("class_counts") or [])[:16]
        if not summary and raw_text:
            summary["preview"] = raw_text[:400]
        return summary

    def _build_tool_functions(
        self,
        *,
        tool_names: list[str],
        uploaded_files: list[str],
        user_text: str,
        selection_context: dict[str, Any] | None = None,
    ) -> list[Function]:
        functions: list[Function] = []
        latest_result_refs: dict[str, Any] = {}
        per_turn_tool_cache: dict[tuple[str, str], str] = {}

        def _cache_key(selected_name: str, kwargs: dict[str, Any]) -> tuple[str, str] | None:
            if selected_name not in {"search_bisque_resources", "bisque_find_assets"}:
                return None
            try:
                normalized = json.dumps(kwargs, sort_keys=True, default=str)
            except TypeError:
                normalized = json.dumps(
                    json.loads(json.dumps(kwargs, default=str)),
                    sort_keys=True,
                )
            return (selected_name, normalized)

        for tool_name in tool_names:
            schema = TOOL_SCHEMA_MAP.get(tool_name)
            entrypoint = AVAILABLE_TOOLS.get(tool_name)
            if schema is None or entrypoint is None:
                continue
            function_schema = dict(schema.get("function") or {})
            parameters = dict(function_schema.get("parameters") or {})
            description = str(function_schema.get("description") or tool_name)
            requires_confirmation = tool_name in APPROVAL_REQUIRED_TOOL_NAMES

            def _make_entrypoint(selected_name: str) -> Callable[..., str]:
                def _tool_entrypoint(**kwargs: Any) -> str:
                    cached_key = _cache_key(selected_name, kwargs)
                    if cached_key is not None:
                        cached = per_turn_tool_cache.get(cached_key)
                        if cached is not None:
                            return cached
                    raw_result = execute_tool_call(
                        selected_name,
                        kwargs,
                        uploaded_files=list(uploaded_files or []),
                        user_text=user_text,
                        latest_result_refs=dict(latest_result_refs),
                        selection_context=dict(selection_context or {}),
                    )
                    try:
                        parsed = json.loads(str(raw_result or ""))
                    except Exception:
                        parsed = None
                    if isinstance(parsed, dict):
                        refs = parsed.get("latest_result_refs")
                        if isinstance(refs, dict):
                            for key, value in refs.items():
                                if value is None:
                                    continue
                                latest_result_refs[str(key)] = value
                    if cached_key is not None:
                        per_turn_tool_cache[cached_key] = raw_result
                    return raw_result

                _tool_entrypoint.__name__ = selected_name
                return _tool_entrypoint

            functions.append(
                Function(
                    name=tool_name,
                    description=description,
                    parameters=parameters,
                    entrypoint=_make_entrypoint(tool_name),
                    requires_confirmation=requires_confirmation,
                )
            )
        return functions

    def _build_agent(
        self,
        *,
        domain_id: str,
        selected_tool_names: list[str],
        memory_policy: ScientificMemoryPolicy | None = None,
        history_message_count: int = 0,
        model: str | None = None,
        reasoning_mode: str | None = None,
        max_tool_calls: int = 12,
        max_runtime_seconds: int = 900,
        uploaded_files: list[str] | None = None,
        user_text: str = "",
        selection_context: dict[str, Any] | None = None,
        analysis_state: dict[str, Any] | None = None,
        debug: bool = False,
    ) -> Agent:
        tool_names = self._normalize_selected_tool_names(selected_tool_names)
        resolved_memory_policy = self.memory_service.normalize_policy(memory_policy)
        memory_features = self.memory_service.agno_feature_settings(
            policy=resolved_memory_policy,
            history_message_count=history_message_count,
        )
        analysis_session_state = self._analysis_session_state_payload(
            analysis_state=analysis_state,
            selection_context=selection_context,
            uploaded_files=list(uploaded_files or []),
        )
        instructions = self._base_instructions(tool_names=tool_names, user_text=user_text)
        if tool_names:
            instructions.append(
                "At least one enabled tool is required for this turn. Use the appropriate tool before answering."
            )
        return Agent(
            name=f"{domain_id}-scientist",
            model=self._build_model(
                model_id=model,
                reasoning_mode=reasoning_mode,
                max_runtime_seconds=max_runtime_seconds,
            ),
            db=self.agno_db,
            add_history_to_context=bool(memory_features["add_history_to_context"]),
            search_past_sessions=bool(memory_features["search_past_sessions"]),
            enable_session_summaries=bool(memory_features["enable_session_summaries"]),
            enable_agentic_memory=bool(memory_features["enable_agentic_memory"]),
            update_memory_on_run=bool(memory_features["update_memory_on_run"]),
            add_datetime_to_context=False,
            add_location_to_context=False,
            markdown=True,
            tools=self._build_tool_functions(
                tool_names=tool_names,
                uploaded_files=list(uploaded_files or []),
                user_text=user_text,
                selection_context=selection_context,
            )
            or None,
            tool_call_limit=max(1, int(max_tool_calls)),
            tool_choice=("required" if tool_names else None),
            instructions=instructions,
            session_state=dict(analysis_session_state or {}),
            add_session_state_to_context=bool(analysis_session_state),
            debug_mode=bool(debug),
            telemetry=False,
            retries=0,
            store_events=False,
            store_history_messages=True,
        )

    @classmethod
    def _extract_text_from_content(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = [cls._extract_text_from_content(item) for item in value]
            return "".join(part for part in parts if part)
        if isinstance(value, dict):
            for key in ("text", "content", "output_text", "result"):
                candidate = cls._extract_text_from_content(value.get(key))
                if candidate:
                    return candidate
            return ""
        text = getattr(value, "text", None)
        if isinstance(text, str) and text.strip():
            return text
        content = getattr(value, "content", None)
        if content is not None and content is not value:
            candidate = cls._extract_text_from_content(content)
            if candidate:
                return candidate
        output = getattr(value, "output", None)
        if output is not None and output is not value:
            candidate = cls._extract_text_from_content(output)
            if candidate:
                return candidate
        raw_item = getattr(value, "raw_item", None)
        if raw_item is not None and raw_item is not value:
            candidate = cls._extract_text_from_content(raw_item)
            if candidate:
                return candidate
        final_output = getattr(value, "final_output", None)
        if final_output is not None and final_output is not value:
            candidate = cls._extract_text_from_content(final_output)
            if candidate:
                return candidate
        if hasattr(value, "model_dump"):
            try:
                dumped = value.model_dump()
            except Exception:
                dumped = None
            if dumped is not None and dumped is not value:
                return cls._extract_text_from_content(dumped)
        return ""

    @classmethod
    def _coerce_visible_output(cls, output: Any) -> tuple[str, str]:
        for source_name in ("final_output", "content", "parsed"):
            candidate = getattr(output, source_name, None)
            text = cls._extract_text_from_content(candidate)
            if text:
                return text, source_name

        for container_name in ("new_items", "raw_responses", "messages"):
            items = getattr(output, container_name, None)
            if not isinstance(items, list):
                continue
            for item in items:
                text = cls._extract_text_from_content(item)
                if text:
                    return text, container_name

        if isinstance(output, str):
            return output, "str"
        return "", "none"

    @classmethod
    def _coerce_visible_output_text(cls, output: Any) -> str:
        text, _source = cls._coerce_visible_output(output)
        return text

    def _messages_to_agno(self, messages: list[dict[str, Any]]) -> list[Message]:
        agno_messages: list[Message] = []
        for raw in messages:
            role = str(raw.get("role") or "user").strip().lower()
            if role not in {"system", "user", "assistant", "tool"}:
                role = "user"
            content = str(raw.get("content") or "")
            if not content.strip():
                continue
            agno_messages.append(Message(role=role, content=content))
        return agno_messages

    @staticmethod
    def _empty_memory_context(policy: ScientificMemoryPolicy) -> ScientificMemoryContext:
        return ScientificMemoryContext(
            policy=policy,
            agno_features=ScientificMemoryService.agno_feature_settings(
                policy=policy,
                history_message_count=0,
            ),
        )

    @staticmethod
    def _empty_knowledge_context(scope: ScientificKnowledgeScope) -> ScientificKnowledgeContext:
        return ScientificKnowledgeContext(
            scope=scope,
            namespaces_searched=list(scope.namespaces),
            project_id=str(scope.project_id or "").strip() or None,
        )

    def _pro_mode_analysis_brief_lines(
        self,
        *,
        analysis_state: dict[str, Any] | None,
        selection_context: dict[str, Any] | None,
        uploaded_files: list[str] | None,
    ) -> list[str]:
        payload = self._analysis_session_state_payload(
            analysis_state=analysis_state,
            selection_context=selection_context,
            uploaded_files=list(uploaded_files or []),
        )
        state = dict(payload.get("analysis_state") or {})
        lines: list[str] = []
        last_objective = str(state.get("last_objective") or "").strip()
        last_answer_summary = str(state.get("last_answer_summary") or "").strip()
        if last_objective:
            lines.append(
                f"Last objective: {self._presentation_text(last_objective, max_chars=180)}"
            )
        if last_answer_summary:
            lines.append(
                f"Last answer: {self._presentation_text(last_answer_summary, max_chars=220)}"
            )
        for item in list(state.get("key_measurements") or [])[:3]:
            rendered = self._presentation_text(item, max_chars=160)
            if rendered:
                lines.append(f"Key measurement: {rendered}")
        active_result_group_id = str(state.get("active_result_group_id") or "").strip()
        if active_result_group_id:
            lines.append(
                f"Active result group: {self._presentation_text(active_result_group_id, max_chars=120)}"
            )
        active_report_handle = str(state.get("active_report_handle") or "").strip()
        if active_report_handle:
            lines.append(
                f"Active report: {self._presentation_text(Path(active_report_handle).name, max_chars=120)}"
            )
        for item in list(state.get("active_selected_files") or [])[:2]:
            rendered = self._presentation_text(Path(str(item)).name, max_chars=120)
            if rendered:
                lines.append(f"Selected file: {rendered}")
        for item in list(state.get("recommended_next_steps") or [])[:2]:
            rendered = self._presentation_text(item, max_chars=180)
            if rendered:
                lines.append(f"Open next step: {rendered}")
        return lines[:6]

    def _pro_mode_shared_context_payload(
        self,
        *,
        memory_context: ScientificMemoryContext,
        knowledge_result: ScientificKnowledgeContext,
        analysis_state: dict[str, Any] | None,
        selection_context: dict[str, Any] | None,
        uploaded_files: list[str] | None,
    ) -> dict[str, Any]:
        return {
            "memory_messages": list(memory_context.system_messages or []),
            "knowledge_messages": list(knowledge_result.system_messages or []),
            "analysis_brief_lines": self._pro_mode_analysis_brief_lines(
                analysis_state=analysis_state,
                selection_context=selection_context,
                uploaded_files=uploaded_files,
            ),
            "memory_metadata": memory_context.metadata(),
            "knowledge_metadata": knowledge_result.metadata(),
        }

    async def _run_pro_mode_focused_team(
        self,
        *,
        latest_user_text: str,
        shared_context: dict[str, Any],
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        max_runtime_seconds: int,
        debug: bool | None,
    ) -> ProModeWorkflowResult:
        math_explainer_request = self._is_math_explanation_request(latest_user_text)
        context_sections: list[str] = []
        memory_messages = [
            str(item or "").strip()
            for item in list((shared_context or {}).get("memory_messages") or [])
            if str(item or "").strip()
        ]
        knowledge_messages = [
            str(item or "").strip()
            for item in list((shared_context or {}).get("knowledge_messages") or [])
            if str(item or "").strip()
        ]
        analysis_brief_lines = [
            str(item or "").strip()
            for item in list((shared_context or {}).get("analysis_brief_lines") or [])
            if str(item or "").strip()
        ]
        if memory_messages:
            context_sections.append(
                "Relevant prior context:\n" + "\n".join(f"- {item}" for item in memory_messages[:6])
            )
        if knowledge_messages:
            context_sections.append(
                "Relevant knowledge context:\n"
                + "\n".join(f"- {item}" for item in knowledge_messages[:6])
            )
        if analysis_brief_lines:
            context_sections.append(
                "Relevant running summary:\n"
                + "\n".join(f"- {item}" for item in analysis_brief_lines[:4])
            )

        team_session_state = {
            "pro_mode_context": {
                "memory_messages": memory_messages[:6],
                "knowledge_messages": knowledge_messages[:6],
                "analysis_brief_lines": analysis_brief_lines[:4],
            }
        }

        def _fallback_member_note(role_name: str) -> FocusedTeamMemberNote:
            return FocusedTeamMemberNote(
                role=role_name,
                headline=f"{role_name} fallback note",
                key_points=[f"{role_name} could not produce a full structured note."],
                caveats=["The member response was incomplete or unavailable."],
                confidence="low",
            )

        member_agents = [
            Agent(
                name=f"focused-team-{spec['name'].lower().replace(' ', '-')}",
                role=spec["role"],
                model=self._build_model(
                    reasoning_mode="fast",
                    reasoning_effort_override="low",
                    max_runtime_seconds=max_runtime_seconds,
                ),
                reasoning_model=self._build_model(
                    reasoning_mode="deep",
                    reasoning_effort_override="high",
                    max_runtime_seconds=max_runtime_seconds,
                ),
                instructions=[
                    "You are part of a focused scientific response team.",
                    "Return only the requested structured output.",
                    "Be concrete, selective, and report-oriented.",
                    "Do not write the full answer.",
                    spec["instructions"],
                ],
                output_schema=FocusedTeamMemberNote,
                structured_outputs=True,
                use_json_mode=True,
                parse_response=True,
                markdown=False,
                telemetry=False,
                retries=0,
                store_events=False,
                store_history_messages=False,
                session_state=dict(team_session_state),
                add_session_state_to_context=True,
                reasoning=True,
                reasoning_min_steps=2,
                reasoning_max_steps=8,
                debug_mode=bool(debug),
            )
            for spec in FOCUSED_TEAM_MEMBER_SPECS
        ]

        team_prompt = "\n\n".join(
            [
                "Produce a report-quality scientific response to the user request.",
                "Delegate to every member exactly once and gather all four perspectives before concluding.",
                "The members should cover: Cartesian decomposition and structure, core technical explanation, decisive cruxes and overclaims, and practical implications.",
                "Use a Cartesian method: decompose the topic into manageable parts, move from foundations to harder distinctions, and check that nothing material is skipped.",
                "Use one Socratic crux pass: surface only the highest-value unresolved assumptions, ambiguities, and overclaims.",
                "The prose specialist should make the eventual answer read like disciplined explanatory nonfiction: coherent, elegant, and easy to follow without becoming flowery.",
                "Assume the audience is primarily advanced undergraduate, graduate, and PhD-level students.",
                *(
                    [
                        "This is a math-heavy explanatory request. Keep the equations, but make each one earn its place.",
                        "For mathematical sections, use a teaching cadence: intuition first, then notation and equation, then a plain-English interpretation, then why it matters.",
                        "Prefer a smaller number of load-bearing equations over a maximal catalog of formulas.",
                    ]
                    if math_explainer_request
                    else []
                ),
                "Keep the collaboration stable: no repeated debate loops, no theatrical disagreement, and no invented citations or facts.",
                *context_sections,
                f"User request: {latest_user_text}",
            ]
        )
        team = Team(
            name="pro-mode-focused-team",
            model=self._build_model(
                reasoning_mode="fast",
                reasoning_effort_override="low",
                max_runtime_seconds=max_runtime_seconds,
            ),
            members=member_agents,
            mode=TeamMode.broadcast,
            delegate_to_all_members=True,
            determine_input_for_members=False,
            tools=[ReasoningTools(add_instructions=True, add_few_shot=True)],
            instructions=[
                "You are the lead of a small focused reasoning team for open-ended scientific responses.",
                "Always gather every member's view before settling on a draft.",
                "Optimize for a polished, report-worthy answer rather than a generic summary.",
                "Use Cartesian decomposition for structure and a Socratic crux standard for critique.",
                "Make the final draft read like high-quality explanatory prose: idea-first, coherent, and concrete.",
                "Optimize for technical clarity for student readers, not just completeness for experts.",
                "Keep delegation tight and stable. One pass only.",
            ],
            markdown=True,
            telemetry=False,
            retries=0,
            store_events=False,
            store_history_messages=False,
            store_member_responses=False,
            show_members_responses=True,
            stream_member_events=bool(debug) or self._pro_mode_autonomous_cycle_shadow_enabled(),
            session_state=dict(team_session_state),
            add_session_state_to_context=True,
            reasoning=True,
            reasoning_min_steps=2,
            reasoning_max_steps=8,
            debug_mode=bool(debug),
        )
        try:
            team_result = await team.arun(
                team_prompt,
                stream=False,
                user_id=user_id,
                session_id=self._scope_session_id(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    run_id=f"{str(run_id or '').strip()}::focused_team"
                    if run_id
                    else "focused_team",
                ),
                debug_mode=bool(debug),
            )
        except Exception as exc:
            return ProModeWorkflowResult(
                response_text="",
                metadata={
                    "pro_mode": {
                        "execution_path": "focused_team",
                        "runtime_status": "failed",
                    }
                },
                runtime_status="failed",
                runtime_error=str(exc or exc.__class__.__name__),
            )

        leader_summary = self._coerce_visible_output_text(team_result)
        member_responses = list(getattr(team_result, "member_responses", []) or [])
        member_notes: list[FocusedTeamMemberNote] = []
        for index, spec in enumerate(FOCUSED_TEAM_MEMBER_SPECS):
            response = member_responses[index] if index < len(member_responses) else None
            note = self._coerce_schema_output(
                schema=FocusedTeamMemberNote,
                result=response,
                fallback=_fallback_member_note(spec["name"]),
            )
            if not str(note.role or "").strip():
                note = note.model_copy(update={"role": spec["name"]})
            member_notes.append(note)

        focused_review_fallback = FocusedTeamReview(
            major_issues=[],
            must_include=[
                item
                for note in member_notes
                for item in list(note.important_differences or [])[:1]
                if str(item or "").strip()
            ][:4],
            claims_to_qualify=[
                item
                for note in member_notes
                for item in list(note.caveats or [])[:1]
                if str(item or "").strip()
            ][:4],
            missing_differences=[],
            passed=True,
            confidence="medium",
        )
        focused_review_prompt = "\n".join(
            [
                "You are the controlled critique round for a focused scientific response team.",
                "This is the only critique pass. Do not restart the whole analysis.",
                "Act like a compressed Socratic Crux Examiner rather than a full debating council.",
                "Identify only the highest-value unresolved cruxes, overclaims, weak transitions, or missing distinctions that would noticeably improve the final response.",
                "Force vague, absolute, or ambiguous claims into precise and qualified statements.",
                "Check Cartesian completeness: did the analysis decompose the problem cleanly, proceed from foundations to harder distinctions, and cover the material parts of the request?",
                "Name what caveat, distinction, or evidence would resolve each issue.",
                *(
                    [
                        "Audit the pedagogy as well as the correctness: would a strong graduate student understand why each important equation is present and what it means?",
                        "Flag equations that appear without explanatory payoff, notation that arrives too late, or practical advice that is too brittle or under-justified.",
                    ]
                    if math_explainer_request
                    else []
                ),
                "If the team already has enough to write a strong answer, mark the review as passed.",
                "",
                f"User request: {latest_user_text}",
                "",
                "Team leader draft summary:",
                leader_summary or "No team leader summary.",
                "",
                "Structured member notes:",
                json.dumps(
                    [note.model_dump(mode="json") for note in member_notes],
                    ensure_ascii=False,
                    indent=2,
                ),
            ]
        )
        focused_review = await self._run_tool_program_phase(
            phase_name="focused_team_review",
            schema=FocusedTeamReview,
            prompt=focused_review_prompt,
            fallback=focused_review_fallback,
            session_state=team_session_state,
            conversation_id=conversation_id,
            run_id=run_id,
            user_id=user_id,
            reasoning_mode="deep",
            reasoning_effort_override="high",
            max_runtime_seconds=min(max_runtime_seconds, 120),
            debug=debug,
        )

        synthesis_fallback = ProModeSynthesis(
            response_text=leader_summary
            or (
                "I could not yet produce a stable focused-team synthesis for this report-style request."
            ),
            settlement_summary="Focused team fallback synthesis used.",
            consensus_level="medium",
            confidence="medium",
            unresolved_points=list(focused_review.major_issues or []),
            minority_view=None,
            agreement_reached=bool(focused_review.passed),
        )
        synthesis_prompt = "\n".join(
            [
                "Write the final user-facing answer from the focused team's work.",
                "This should read like a polished Pro Mode response: specific, synthesized, and structurally clear.",
                "Use the member notes as the content base and the critique round as the single revision pass.",
                "Do not mention the team, members, workflow, or internal process.",
                "For report-style requests, prefer a compact executive summary followed by the clearest sections needed for the topic.",
                "When it helps, organize from foundational or simple ideas to harder distinctions and practical consequences.",
                "Make decisive cruxes explicit and qualify claims where the critique round found overreach or ambiguity.",
                "Aim for disciplined explanatory prose: idea-first paragraphs, strong transitions, concrete examples where helpful, and enough stylistic energy to keep the writing alive.",
                *(
                    [
                        "Because this is a mathematical explanation, keep the response selective and teachable rather than encyclopedic.",
                        "Define notation before using it or at first use, and follow each important equation with a brief interpretation in plain scientific English.",
                        "Do not let the equations replace the explanation; the prose should tell the reader what to notice in the math.",
                    ]
                    if math_explainer_request
                    else []
                ),
                "Prose techniques to follow:",
                *[f"- {item}" for item in PROSE_STYLE_GUIDELINES],
                *(
                    [
                        "Student-facing explanation techniques to follow:",
                        *[f"- {item}" for item in STUDENT_EXPLANATION_GUIDELINES],
                    ]
                    if math_explainer_request
                    else []
                ),
                "",
                f"User request: {latest_user_text}",
                "",
                "Team leader draft summary:",
                leader_summary or "No team leader summary.",
                "",
                "Structured member notes:",
                json.dumps(
                    [note.model_dump(mode="json") for note in member_notes],
                    ensure_ascii=False,
                    indent=2,
                ),
                "",
                "Focused critique round:",
                focused_review.model_dump_json(indent=2),
            ]
        )
        synthesis = await self._run_tool_program_phase(
            phase_name="focused_team_synthesis",
            schema=ProModeSynthesis,
            prompt=synthesis_prompt,
            fallback=synthesis_fallback,
            session_state=team_session_state,
            conversation_id=conversation_id,
            run_id=run_id,
            user_id=user_id,
            reasoning_mode="deep",
            reasoning_effort_override="high",
            max_runtime_seconds=min(max_runtime_seconds, 150),
            debug=debug,
        )

        verifier_fallback = ProModeVerifierReport(
            passed=bool(synthesis.response_text),
            issues=list(focused_review.major_issues or []),
            suggested_changes=list(focused_review.must_include or []),
            confidence="medium",
        )
        verifier_prompt = "\n".join(
            [
                "Verify the focused-team synthesis against the structured member notes and the critique round.",
                "Flag only material issues that would affect correctness, completeness, or report quality.",
                "Check that the answer resolves or cleanly qualifies the decisive cruxes and that the structure covers the problem without skipping a material subpart.",
                "",
                f"User request: {latest_user_text}",
                "",
                "Synthesis JSON:",
                synthesis.model_dump_json(indent=2),
                "",
                "Member notes JSON:",
                json.dumps(
                    [note.model_dump(mode="json") for note in member_notes],
                    ensure_ascii=False,
                    indent=2,
                ),
                "",
                "Critique JSON:",
                focused_review.model_dump_json(indent=2),
            ]
        )
        verifier = await self._run_tool_program_phase(
            phase_name="focused_team_verifier",
            schema=ProModeVerifierReport,
            prompt=verifier_prompt,
            fallback=verifier_fallback,
            session_state=team_session_state,
            conversation_id=conversation_id,
            run_id=run_id,
            user_id=user_id,
            reasoning_mode="deep",
            reasoning_effort_override="high",
            max_runtime_seconds=min(max_runtime_seconds, 120),
            debug=debug,
        )

        return ProModeWorkflowResult(
            response_text=str(synthesis.response_text or "").strip(),
            metadata={
                "pro_mode": {
                    "execution_path": "focused_team",
                    "execution_regime": "focused_team",
                    "active_roles": [
                        "Focused Team Lead",
                        *[spec["name"] for spec in FOCUSED_TEAM_MEMBER_SPECS],
                        "Focused Critique",
                        "Focused Synthesizer",
                        "Focused Verifier",
                    ],
                    "phase_order": [
                        "intake",
                        "context_policy",
                        "execution_router",
                        "focused_team",
                        "finalize",
                    ],
                    "round_count": 1,
                    "discussion_round_count": 1,
                    "model_call_count": len(member_notes) + 4,
                    "convergence": {
                        "per_role_vote": {},
                        "central_blockers": list(verifier.issues or []),
                        "ready": bool(verifier.passed),
                        "consensus_level": ("high" if verifier.passed else "medium"),
                    },
                    "role_stats": {},
                    "calculator": {"used": False, "call_count": 0, "results": []},
                    "verifier": verifier.model_dump(mode="json"),
                    "focused_team": {
                        "leader_summary": leader_summary,
                        "member_notes": [note.model_dump(mode="json") for note in member_notes],
                        "review": focused_review.model_dump(mode="json"),
                        "member_count": len(member_notes),
                    },
                    "summary": "Completed a focused multi-agent team pass with Cartesian planning and one controlled Socratic crux round.",
                }
            },
            runtime_status="completed",
        )

    async def _run_pro_mode_codeexec_reasoning_solver(
        self,
        *,
        latest_user_text: str,
        task_regime: str | None,
        shared_context: dict[str, Any],
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
        selected_tool_names: list[str],
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        max_runtime_seconds: int,
        debug: bool | None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> ProModeWorkflowResult:
        tool_names = [
            tool_name
            for tool_name in self._normalize_selected_tool_names(selected_tool_names)
            if tool_name in {"codegen_python_plan", "execute_python_job"}
        ] or ["codegen_python_plan", "execute_python_job"]
        memory_messages = [
            str(item or "").strip()
            for item in list((shared_context or {}).get("memory_messages") or [])
            if str(item or "").strip()
        ]
        knowledge_messages = [
            str(item or "").strip()
            for item in list((shared_context or {}).get("knowledge_messages") or [])
            if str(item or "").strip()
        ]
        analysis_brief_lines = [
            str(item or "").strip()
            for item in list((shared_context or {}).get("analysis_brief_lines") or [])
            if str(item or "").strip()
        ]
        context_sections: list[str] = []
        if memory_messages:
            context_sections.append(
                "Relevant prior context:\n" + "\n".join(f"- {item}" for item in memory_messages[:6])
            )
        if knowledge_messages:
            context_sections.append(
                "Relevant knowledge context:\n"
                + "\n".join(f"- {item}" for item in knowledge_messages[:6])
            )
        if analysis_brief_lines:
            context_sections.append(
                "Relevant running summary:\n"
                + "\n".join(f"- {item}" for item in analysis_brief_lines[:4])
            )

        prompt = "\n\n".join(
            [
                "Solve the following hard computational research request.",
                "Use reasoning first to decide the minimum code workflow needed.",
                "When execution is needed, you must ground the answer in produced artifacts or structured tool outputs.",
                "If the execution fails or yields incomplete evidence, say so explicitly and do not fabricate measured outputs.",
                "Return a scientist-facing answer with methods, key findings, interpretation, and limitations.",
                *context_sections,
                f"Request: {latest_user_text}",
            ]
        )
        session_payload = {
            "pro_mode_context": {
                "memory_messages": memory_messages[:6],
                "knowledge_messages": knowledge_messages[:6],
                "analysis_brief_lines": analysis_brief_lines[:4],
            }
        }
        tool_functions = self._build_tool_functions(
            tool_names=tool_names,
            uploaded_files=uploaded_files,
            user_text=latest_user_text,
            selection_context=selection_context,
        )

        def _build_agent(model_builder: Callable[..., AgnoModel]) -> Agent:
            return build_codeexec_reasoning_agent(
                model_builder=model_builder,
                tools=tool_functions,
                max_runtime_seconds=max_runtime_seconds,
                session_state=session_payload,
                debug_mode=bool(debug),
            )

        model_route = self._pro_mode_model_route_metadata(
            fallback_used=False,
            active_model=str(
                self._setting(self.settings, "resolved_pro_mode_model", self.model) or self.model
            ),
        )
        try:
            if self._uses_published_pro_mode_api():
                result = await _build_agent(self._build_model).arun(
                    prompt,
                    stream=False,
                    user_id=user_id,
                    session_id=self._scope_session_id(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        run_id=f"{str(run_id or '').strip()}::codeexec_reasoning_solver"
                        if run_id
                        else "codeexec_reasoning_solver",
                    ),
                    debug_mode=bool(debug),
                )
                model_route = self._pro_mode_model_route_metadata(
                    fallback_used=True,
                    failure_code="structured_phase_requires_tool_capable_model",
                    active_model=self.model,
                )
            else:
                result, model_route = await self._arun_with_optional_pro_mode_fallback(
                    phase_name="codeexec_reasoning_solver",
                    prompt=prompt,
                    build_agent=_build_agent,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    debug=debug,
                )
        except Exception as exc:
            return ProModeWorkflowResult(
                response_text="",
                metadata={
                    "pro_mode": {
                        "execution_path": "codeexec_reasoning_solver",
                        "runtime_status": "failed",
                        "model_route": model_route,
                    },
                    "tool_invocations": [],
                },
                runtime_status="failed",
                runtime_error=str(exc or exc.__class__.__name__),
            )

        run_output = result if isinstance(result, RunOutput) else None
        tool_invocations = self._tool_invocations_from_run_output(run_output)
        response_text, _source = self._coerce_visible_output(result)
        normalized_response_text = str(response_text or "").strip()
        deterministic_fallback_meta: dict[str, Any] | None = None
        if not self._tool_workflow_satisfied(
            tool_invocations=tool_invocations,
            required_tool_names=tool_names,
            strict_validation=True,
        ):
            missing_required_tool_names = self._missing_required_tool_names(
                tool_invocations=tool_invocations,
                required_tool_names=tool_names,
            )
            if missing_required_tool_names:
                deterministic_invocations = await self._execute_tool_program_actions(
                    phase="reasoning_solver",
                    phase_label="code execution reasoning solver",
                    actions=[
                        ToolProgramAction(
                            tool_name=tool_name,
                            purpose=(
                                "Running the required code-execution tool directly because the "
                                "reasoning agent did not execute it."
                            ),
                            args={},
                        )
                        for tool_name in missing_required_tool_names
                    ],
                    uploaded_files=uploaded_files,
                    latest_user_text=latest_user_text,
                    selection_context=selection_context,
                    event_callback=event_callback,
                    latest_result_refs_seed=self._latest_result_refs_from_tool_invocations(
                        tool_invocations
                    ),
                    request_bisque_auth=get_request_bisque_auth(),
                )
                if deterministic_invocations:
                    tool_invocations = self._merge_tool_invocations(
                        tool_invocations,
                        deterministic_invocations,
                    )
                    deterministic_fallback_meta = {
                        "required_tool_names": list(tool_names),
                        "missing_required_tool_names": list(missing_required_tool_names),
                        "tool_invocation_count": len(deterministic_invocations),
                    }
                    normalized_response_text = (
                        self._tool_invocation_fallback_text(tool_invocations)
                        or normalized_response_text
                    )
        fail_closed_text = self._code_execution_fail_closed_text(tool_invocations)
        if fail_closed_text and (
            not normalized_response_text
            or not self._tool_workflow_satisfied(
                tool_invocations=tool_invocations,
                required_tool_names=tool_names,
                strict_validation=True,
            )
        ):
            normalized_response_text = fail_closed_text
        if not normalized_response_text and not tool_invocations:
            return ProModeWorkflowResult(
                response_text="",
                metadata={
                    "pro_mode": {
                        "execution_path": "codeexec_reasoning_solver",
                        "runtime_status": "failed",
                        "model_route": model_route,
                    },
                    "tool_invocations": [],
                },
                runtime_status="failed",
                runtime_error="Code-execution reasoning agent returned no answer and no tool outputs.",
            )
        verifier_issues = self._code_execution_fail_closed_reservations(tool_invocations)
        satisfied = self._tool_workflow_satisfied(
            tool_invocations=tool_invocations,
            required_tool_names=tool_names,
            strict_validation=True,
        )
        verifier_report = ProModeVerifierReport(
            passed=bool(satisfied and not verifier_issues),
            issues=list(verifier_issues),
            suggested_changes=[],
            confidence="medium" if tool_invocations else "low",
        )
        pro_mode_metadata = {
            "execution_path": "codeexec_reasoning_solver",
            "task_regime": str(task_regime or "self_contained_reasoning").strip().lower()
            or "self_contained_reasoning",
            "model_route": model_route,
            "verifier": verifier_report.model_dump(mode="json"),
        }
        if deterministic_fallback_meta:
            pro_mode_metadata["deterministic_required_tool_fallback"] = (
                deterministic_fallback_meta
            )
        return ProModeWorkflowResult(
            response_text=normalized_response_text,
            metadata={
                "pro_mode": pro_mode_metadata,
                "tool_invocations": tool_invocations,
            },
            runtime_status="completed" if satisfied else "failed",
            runtime_error=None if satisfied else "required_code_execution_tools_did_not_complete",
        )

    async def _run_pro_mode_reasoning_solver(
        self,
        *,
        latest_user_text: str,
        task_regime: str | None,
        shared_context: dict[str, Any],
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        max_runtime_seconds: int,
        debug: bool | None,
    ) -> ProModeWorkflowResult:
        context_sections: list[str] = []
        normalized_task_regime = str(task_regime or "conceptual_high_uncertainty").strip().lower()
        proof_like_request = normalized_task_regime == "rigorous_proof"
        report_like_request = self._is_report_like_request(latest_user_text)
        counterfactual_verification_requested = bool(
            re.search(
                r"\bcounterfactual verification pass\b|\bstrongest alternative\b|\bassume the current leading answer may be wrong\b",
                str(latest_user_text or "").strip().lower(),
            )
        )
        memory_messages = [
            str(item or "").strip()
            for item in list((shared_context or {}).get("memory_messages") or [])
            if str(item or "").strip()
        ]
        knowledge_messages = [
            str(item or "").strip()
            for item in list((shared_context or {}).get("knowledge_messages") or [])
            if str(item or "").strip()
        ]
        analysis_brief_lines = [
            str(item or "").strip()
            for item in list((shared_context or {}).get("analysis_brief_lines") or [])
            if str(item or "").strip()
        ]
        if memory_messages:
            context_sections.append(
                "Relevant prior context:\n" + "\n".join(f"- {item}" for item in memory_messages[:6])
            )
        if knowledge_messages:
            context_sections.append(
                "Relevant knowledge context:\n"
                + "\n".join(f"- {item}" for item in knowledge_messages[:6])
            )
        if analysis_brief_lines:
            context_sections.append(
                "Relevant running summary:\n"
                + "\n".join(f"- {item}" for item in analysis_brief_lines[:4])
            )

        prompt = "\n\n".join(
            [
                "Solve the following hard, self-contained scientific question.",
                *(
                    [
                        "Treat this as a rigorous proof-style request.",
                        "State the theorem or claim status first, then the load-bearing argument, and name any unresolved gap instead of hand-waving it away.",
                    ]
                    if proof_like_request
                    else []
                ),
                *(
                    [
                        "Treat this as a report-style synthesis request.",
                        "Organize the answer into a compact but substantive explanatory report rather than a terse solver stub.",
                    ]
                    if report_like_request
                    else []
                ),
                "Use deep reasoning internally, but return only the final answer and a concise derivation.",
                "If the question asks for a single value or count, state it clearly near the beginning.",
                "Prefer the most standard interpretation unless the prompt explicitly requires another convention.",
                *(
                    [
                        "This request is a counterfactual verification pass.",
                        "Do not defend the current candidate answer by default.",
                        "First assume the current candidate answer may be wrong, identify the strongest alternative, and only keep the incumbent answer if it survives an explicit falsification attempt.",
                    ]
                    if counterfactual_verification_requested
                    else []
                ),
                "Do not mention internal workflows, routes, councils, tools, or hidden reasoning.",
                *context_sections,
                f"Question: {latest_user_text}",
            ]
        )

        def _build_reasoning_solver_agent(model_builder: Callable[..., AgnoModel]) -> Agent:
            return Agent(
                name="pro-mode-reasoning-solver",
                model=model_builder(
                    reasoning_mode="fast",
                    reasoning_effort_override="low",
                    max_runtime_seconds=max_runtime_seconds,
                ),
                reasoning_model=model_builder(
                    reasoning_mode="deep",
                    reasoning_effort_override="high",
                    max_runtime_seconds=max_runtime_seconds,
                ),
                instructions=[
                    "You answer self-contained scientific questions with rigorous but concise reasoning.",
                    "Lead with the answer when the user asked for a concrete result.",
                    "Show only the minimum derivation needed for confidence and readability.",
                    "Do not fabricate missing data or claim certainty beyond what the prompt supports.",
                    *(
                        [
                            "For proof-style requests, prefer explicit lemma-to-conclusion bridges over rhetorical confidence.",
                            "If a proof gap remains, say so directly instead of implying the proof is complete.",
                        ]
                        if proof_like_request
                        else []
                    ),
                    *(
                        [
                            "For report-style requests, preserve real synthesis and structure rather than collapsing to a short direct answer.",
                        ]
                        if report_like_request
                        else []
                    ),
                    *(
                        [
                            "When the prompt indicates a counterfactual verification pass, treat it as a falsification challenge.",
                            "Actively test the strongest competing explanation and revise the answer if the incumbent claim fails that challenge.",
                        ]
                        if counterfactual_verification_requested
                        else []
                    ),
                ],
                markdown=True,
                telemetry=False,
                retries=0,
                store_events=False,
                store_history_messages=False,
                session_state={
                    "pro_mode_context": {
                        "memory_messages": memory_messages[:6],
                        "knowledge_messages": knowledge_messages[:6],
                        "analysis_brief_lines": analysis_brief_lines[:4],
                    }
                },
                add_session_state_to_context=True,
                reasoning=True,
                reasoning_min_steps=2,
                reasoning_max_steps=12,
                debug_mode=bool(debug),
            )

        model_route = self._pro_mode_model_route_metadata(
            fallback_used=False,
            active_model=str(
                self._setting(self.settings, "resolved_pro_mode_model", self.model) or self.model
            ),
        )
        try:
            result, model_route = await self._arun_text_phase_with_optional_pro_mode_transport(
                phase_name="reasoning_solver",
                prompt=prompt,
                build_agent=_build_reasoning_solver_agent,
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                debug=debug,
                reasoning_mode="deep",
                reasoning_effort_override="high",
                max_runtime_seconds=max_runtime_seconds,
            )
        except Exception as exc:
            return ProModeWorkflowResult(
                response_text="",
                metadata={
                    "pro_mode": {
                        "execution_path": "reasoning_solver",
                        "runtime_status": "failed",
                        "model_route": model_route,
                    }
                },
                runtime_status="failed",
                runtime_error=str(exc or exc.__class__.__name__),
            )
        response_text, _source = self._coerce_visible_output(result)
        normalized_response_text = str(response_text or "").strip()
        failure_code = self._classify_pro_mode_failure(normalized_response_text)
        if (
            failure_code
            and not bool(model_route.get("fallback_used"))
            and self._uses_dedicated_pro_mode_model()
            and self._pro_mode_fallback_enabled()
        ):
            try:
                fallback_agent = _build_reasoning_solver_agent(self._build_model)
                result = await fallback_agent.arun(
                    prompt,
                    stream=False,
                    user_id=user_id,
                    session_id=self._scope_session_id(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        run_id=f"{str(run_id or '').strip()}::reasoning_solver:fallback_text"
                        if run_id
                        else "reasoning_solver:fallback_text",
                    ),
                    debug_mode=bool(debug),
                )
                model_route = self._pro_mode_model_route_metadata(
                    fallback_used=True,
                    failure_code=failure_code,
                    active_model=self.model,
                )
                response_text, _source = self._coerce_visible_output(result)
                normalized_response_text = str(response_text or "").strip()
                failure_code = self._classify_pro_mode_failure(normalized_response_text)
            except Exception:
                pass
        if failure_code:
            return ProModeWorkflowResult(
                response_text="",
                metadata={
                    "pro_mode": {
                        "execution_path": "reasoning_solver",
                        "runtime_status": "failed",
                        "model_route": model_route,
                    }
                },
                runtime_status="failed",
                runtime_error=normalized_response_text,
            )
        verifier_report = ProModeVerifierReport(
            passed=bool(normalized_response_text),
            issues=[],
            suggested_changes=[],
            confidence="high" if normalized_response_text else "low",
        )
        if normalized_response_text and (
            proof_like_request or report_like_request or counterfactual_verification_requested
        ):
            verifier_prompt = "\n".join(
                [
                    "Review the draft answer for correctness, unsupported leaps, and missing caveats.",
                    "Return only structured output.",
                    "Mark `passed=false` if the answer overstates confidence, skips a proof bridge, or hides a material limitation.",
                    "",
                    f"User request: {latest_user_text}",
                    "",
                    "Draft answer:",
                    normalized_response_text,
                ]
            )
            verifier_report = await self._run_tool_program_phase(
                phase_name="reasoning_solver_verifier",
                schema=ProModeVerifierReport,
                prompt=verifier_prompt,
                fallback=verifier_report,
                session_state={
                    "pro_mode_context": {
                        "memory_messages": memory_messages[:6],
                        "knowledge_messages": knowledge_messages[:6],
                    }
                },
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                reasoning_mode="deep",
                reasoning_effort_override="high",
                use_reasoning_agent=False,
                max_runtime_seconds=max(30, min(max_runtime_seconds, 90)),
                debug=debug,
            )
        return ProModeWorkflowResult(
            response_text=normalized_response_text,
            metadata={
                "pro_mode": {
                    "execution_path": "reasoning_solver",
                    "task_regime": normalized_task_regime or "conceptual_high_uncertainty",
                    "model_route": model_route,
                    "verifier": verifier_report.model_dump(mode="json"),
                }
            },
            runtime_status="completed",
        )

    def _autonomous_cycle_controller_toolkits(
        self,
        *,
        workflow: Workflow,
        shared_context: dict[str, Any],
        autonomy_state_seed: dict[str, Any] | None,
        user_id: str | None,
    ) -> tuple[list[Any], list[str]]:
        toolkits: list[Any] = [ReasoningTools(add_instructions=True, add_few_shot=True)]
        toolkit_names = ["ReasoningTools"]

        shared_snippets = [
            str(item or "").strip()
            for item in [
                *list((shared_context or {}).get("memory_messages") or []),
                *list((shared_context or {}).get("knowledge_messages") or []),
                *list(
                    dict((autonomy_state_seed or {}).get("autonomy_state") or {}).get(
                        "evidence_ledger"
                    )
                    or []
                ),
                *list(
                    dict((autonomy_state_seed or {}).get("autonomy_state") or {}).get(
                        "open_obligations"
                    )
                    or []
                ),
            ]
            if str(item or "").strip()
        ]
        if shared_snippets:
            toolkits.append(
                KnowledgeTools(
                    knowledge=_AutonomousSharedContextKnowledge(snippets=shared_snippets),
                    enable_think=False,
                    enable_analyze=False,
                    add_instructions=False,
                )
            )
            toolkit_names.append("KnowledgeTools")

        if str(user_id or "").strip():
            toolkits.append(
                MemoryTools(
                    db=self.agno_db,
                    enable_get_memories=True,
                    enable_add_memory=False,
                    enable_update_memory=False,
                    enable_delete_memory=False,
                    enable_think=False,
                    enable_analyze=False,
                    add_instructions=False,
                )
            )
            toolkit_names.append("MemoryTools")

        toolkits.append(
            WorkflowTools(
                workflow=workflow,
                enable_run_workflow=True,
                enable_think=False,
                enable_analyze=False,
                add_instructions=False,
                async_mode=True,
            )
        )
        toolkit_names.append("WorkflowTools")
        return toolkits, toolkit_names

    async def _run_pro_mode_autonomous_cycle_via_agno_controller(
        self,
        *,
        workflow: Workflow,
        latest_user_text: str,
        shared_context: dict[str, Any],
        autonomy_state_seed: dict[str, Any] | None,
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        max_runtime_seconds: int,
        debug: bool | None,
    ) -> ProModeWorkflowResult:
        del max_runtime_seconds
        controller_tools, toolkit_names = self._autonomous_cycle_controller_toolkits(
            workflow=workflow,
            shared_context=shared_context,
            autonomy_state_seed=autonomy_state_seed,
            user_id=user_id,
        )
        controller_timeout_seconds = max(
            self._pro_mode_autonomous_cycle_phase_timeout_seconds(),
            min(
                self._pro_mode_autonomous_cycle_watchdog_runtime_seconds(),
                self._pro_mode_autonomous_cycle_phase_timeout_seconds() * 2,
            ),
        )
        saved_autonomy_state = self._saved_autonomy_state(autonomy_state_seed)
        controller_prompt = "\n\n".join(
            [
                "You are the Agno-first RTD controller for a bounded long-cycle scientific reasoning workflow.",
                "Use the explicit Think -> Act -> Analyze discipline.",
                "Before answering, you must think, optionally inspect shared knowledge or user memory if it would materially improve the result, and then call `run_workflow()` exactly once with the current user request as input.",
                "Set `workflow_ran=true` only if `run_workflow()` actually executed and returned output. Do not fabricate workflow output.",
                "After the workflow returns, analyze whether the result is coherent and sufficient, then return only the requested structured output.",
                "Do not expose hidden chain-of-thought in the visible response.",
                f"User request: {latest_user_text}",
                "Saved autonomy state JSON:",
                json.dumps(saved_autonomy_state, ensure_ascii=False, indent=2),
                "Shared context JSON:",
                json.dumps(
                    {
                        "memory_messages": list(
                            (shared_context or {}).get("memory_messages") or []
                        )[:6],
                        "knowledge_messages": list(
                            (shared_context or {}).get("knowledge_messages") or []
                        )[:6],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            ]
        )
        fallback = AutonomousCycleControllerEnvelope(
            response_text="",
            controller_summary="The Agno controller did not return a complete structured result.",
            workflow_ran=False,
            workflow_output={},
            workflow_status="failed",
            toolkits_used=list(toolkit_names),
        )
        agent = Agent(
            name="pro-mode-autonomous-cycle-controller",
            model=self._build_model(
                reasoning_mode="fast",
                reasoning_effort_override="low",
                max_runtime_seconds=controller_timeout_seconds,
            ),
            reasoning_model=self._build_model(
                reasoning_mode="deep",
                reasoning_effort_override="high",
                max_runtime_seconds=controller_timeout_seconds,
            ),
            instructions=[
                "You are the controller for a long-cycle scientific reasoning system.",
                "Use the shared think/analyze scratchpad deliberately.",
                "The only valid scratchpad tool names are `think` and `analyze`; never invent aliases such as `analyze_output`.",
                "Run the workflow exactly once, then synthesize the workflow output into the final structured answer.",
                "Your output is invalid unless `workflow_ran=true` and `workflow_output` contains the workflow result.",
                "Return the workflow output dictionary faithfully enough that downstream code can recover the response and metadata.",
                "Do not embellish file paths, citations, or measurements not present in the workflow result.",
            ],
            output_schema=AutonomousCycleControllerEnvelope,
            structured_outputs=True,
            use_json_mode=True,
            parse_response=True,
            markdown=False,
            telemetry=False,
            retries=0,
            store_events=False,
            store_history_messages=False,
            tools=controller_tools,
            session_state={
                "pro_mode_context": {
                    "memory_messages": list((shared_context or {}).get("memory_messages") or [])[
                        :6
                    ],
                    "knowledge_messages": list(
                        (shared_context or {}).get("knowledge_messages") or []
                    )[:6],
                }
            },
            add_session_state_to_context=True,
            reasoning=False,
            debug_mode=bool(debug),
        )
        try:
            result = await agent.arun(
                controller_prompt,
                stream=False,
                user_id=user_id,
                session_id=self._scope_session_id(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    run_id=f"{str(run_id or '').strip()}::autonomous_cycle_controller"
                    if run_id
                    else "autonomous_cycle_controller",
                ),
                debug_mode=bool(debug),
            )
        except Exception:
            return ProModeWorkflowResult(
                response_text="",
                metadata={
                    "pro_mode": {
                        "execution_path": "autonomous_cycle",
                        "execution_regime": "autonomous_cycle",
                        "controller_toolkits": list(toolkit_names),
                        "controller_mode": "rtd_v1",
                        "runtime_status": "failed",
                    }
                },
                runtime_status="failed",
                runtime_error="agno_controller_failed",
            )

        controller_output = self._coerce_schema_output(
            schema=AutonomousCycleControllerEnvelope,
            result=result,
            fallback=fallback,
        )
        workflow_output = dict(controller_output.workflow_output or {})
        workflow_content = dict(workflow_output.get("content") or {})
        if not bool(controller_output.workflow_ran) or not workflow_content:
            return ProModeWorkflowResult(
                response_text="",
                metadata={
                    "pro_mode": {
                        "execution_path": "autonomous_cycle",
                        "execution_regime": "autonomous_cycle",
                        "controller_toolkits": list(toolkit_names),
                        "controller_mode": "rtd_v1",
                        "runtime_status": "failed",
                    }
                },
                runtime_status="failed",
                runtime_error="agno_controller_workflow_not_executed",
            )
        envelope = AutonomousCycleWorkflowEnvelope.model_validate(
            {
                "response_text": workflow_content.get("response_text")
                or controller_output.response_text
                or "",
                "metadata": workflow_content.get("metadata") or {},
                "tool_invocations": workflow_content.get("tool_invocations") or [],
                "runtime_status": workflow_content.get("runtime_status")
                or controller_output.workflow_status
                or "completed",
                "runtime_error": workflow_content.get("runtime_error"),
            }
        )
        pro_mode_meta = dict(envelope.metadata.get("pro_mode") or {})
        if not str(envelope.response_text or "").strip() or not pro_mode_meta:
            return ProModeWorkflowResult(
                response_text="",
                metadata={
                    "pro_mode": {
                        "execution_path": "autonomous_cycle",
                        "execution_regime": "autonomous_cycle",
                        "controller_toolkits": list(toolkit_names),
                        "controller_mode": "rtd_v1",
                        "runtime_status": "failed",
                    }
                },
                runtime_status="failed",
                runtime_error="agno_controller_missing_workflow_output",
            )
        pro_mode_meta["controller_mode"] = "rtd_v1"
        pro_mode_meta["controller_toolkits"] = list(toolkit_names)
        pro_mode_meta["controller_summary"] = str(
            controller_output.controller_summary or ""
        ).strip()
        return ProModeWorkflowResult(
            response_text=str(envelope.response_text or "").strip(),
            metadata={"pro_mode": pro_mode_meta},
            tool_invocations=list(envelope.tool_invocations or []),
            runtime_status=str(envelope.runtime_status or "completed").lower(),
            runtime_error=envelope.runtime_error,
        )

    async def _run_pro_mode_autonomous_cycle(
        self,
        *,
        messages: list[dict[str, Any]],
        latest_user_text: str,
        uploaded_files: list[str],
        shared_context: dict[str, Any],
        selection_context: dict[str, Any] | None,
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        max_tool_calls: int,
        max_runtime_seconds: int,
        reasoning_mode: str | None,
        autonomy_state_seed: dict[str, Any] | None,
        event_callback: Callable[[dict[str, Any]], None] | None,
        benchmark: dict[str, Any] | None = None,
        debug: bool | None,
    ) -> ProModeWorkflowResult:
        del max_tool_calls, max_runtime_seconds
        benchmark = dict(benchmark or {})
        watchdog_max_tool_calls = self._pro_mode_autonomous_cycle_watchdog_tool_calls()
        watchdog_runtime_seconds = self._pro_mode_autonomous_cycle_watchdog_runtime_seconds()
        use_agno_controller = bool(benchmark.get("use_autonomy_agno_controller"))
        if not use_agno_controller:
            use_agno_controller = self._pro_mode_autonomous_cycle_agno_controller_enabled()
        engine_benchmark = dict(benchmark)
        engine_benchmark["use_autonomy_agno_controller"] = False
        selection_context_payload = dict(selection_context or {})
        direct_engine_preferred = bool(
            self._is_option_based_reasoning_request(latest_user_text)
            and not uploaded_files
            and not bool(
                selection_context_payload.get("resource_uris")
                or selection_context_payload.get("dataset_uris")
            )
            and not self._selection_context_has_artifact_handles(selection_context_payload)
        )
        if not use_agno_controller or direct_engine_preferred:
            return await self._run_pro_mode_autonomous_cycle_engine(
                messages=messages,
                latest_user_text=latest_user_text,
                uploaded_files=uploaded_files,
                shared_context=shared_context,
                selection_context=selection_context,
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                max_tool_calls=watchdog_max_tool_calls,
                max_runtime_seconds=watchdog_runtime_seconds,
                reasoning_mode=reasoning_mode,
                autonomy_state_seed=autonomy_state_seed,
                event_callback=event_callback,
                benchmark=engine_benchmark,
                debug=debug,
            )

        async def _autonomous_cycle_engine_step(step_input: StepInput) -> StepOutput:
            del step_input
            engine_result = await self._run_pro_mode_autonomous_cycle_engine(
                messages=messages,
                latest_user_text=latest_user_text,
                uploaded_files=uploaded_files,
                shared_context=shared_context,
                selection_context=selection_context,
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                max_tool_calls=watchdog_max_tool_calls,
                max_runtime_seconds=watchdog_runtime_seconds,
                reasoning_mode=reasoning_mode,
                autonomy_state_seed=autonomy_state_seed,
                event_callback=event_callback,
                benchmark=engine_benchmark,
                debug=debug,
            )
            return StepOutput(
                content=AutonomousCycleWorkflowEnvelope(
                    response_text=str(engine_result.response_text or "").strip(),
                    metadata=dict(engine_result.metadata or {}),
                    tool_invocations=list(engine_result.tool_invocations or []),
                    runtime_status=str(engine_result.runtime_status or "completed"),
                    runtime_error=engine_result.runtime_error,
                ).model_dump(mode="json")
            )

        workflow = Workflow(
            name="pro_mode_autonomous_cycle_workflow",
            description="Agno-first controller workflow wrapper for the bounded autonomous-cycle engine.",
            steps=[Step(name="autonomous_cycle_engine", executor=_autonomous_cycle_engine_step)],
            stream=False,
            telemetry=False,
            store_events=False,
            debug_mode=bool(debug),
            session_state={"autonomous_cycle_controller": {}},
        )
        controller_result = await self._run_pro_mode_autonomous_cycle_via_agno_controller(
            workflow=workflow,
            latest_user_text=latest_user_text,
            shared_context=shared_context,
            autonomy_state_seed=autonomy_state_seed,
            conversation_id=conversation_id,
            run_id=run_id,
            user_id=user_id,
            max_runtime_seconds=watchdog_runtime_seconds,
            debug=debug,
        )
        if controller_result.runtime_status == "failed":
            return await self._run_pro_mode_autonomous_cycle_engine(
                messages=messages,
                latest_user_text=latest_user_text,
                uploaded_files=uploaded_files,
                shared_context=shared_context,
                selection_context=selection_context,
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                max_tool_calls=watchdog_max_tool_calls,
                max_runtime_seconds=watchdog_runtime_seconds,
                reasoning_mode=reasoning_mode,
                autonomy_state_seed=autonomy_state_seed,
                event_callback=event_callback,
                benchmark=engine_benchmark,
                debug=debug,
            )
        return controller_result

    async def _run_pro_mode_autonomous_cycle_engine(
        self,
        *,
        messages: list[dict[str, Any]],
        latest_user_text: str,
        uploaded_files: list[str],
        shared_context: dict[str, Any],
        selection_context: dict[str, Any] | None,
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        max_tool_calls: int,
        max_runtime_seconds: int,
        reasoning_mode: str | None,
        autonomy_state_seed: dict[str, Any] | None,
        event_callback: Callable[[dict[str, Any]], None] | None,
        benchmark: dict[str, Any] | None = None,
        debug: bool | None,
    ) -> ProModeWorkflowResult:
        benchmark = dict(benchmark or {})
        disable_memory_knowledge = bool(benchmark.get("disable_autonomy_memory_knowledge"))
        disable_focused_team_delegate = bool(
            benchmark.get("disable_autonomy_focused_team_delegate")
        )
        disable_resume = bool(benchmark.get("disable_autonomy_resume"))
        disable_falsifier = bool(benchmark.get("disable_autonomy_falsifier"))
        raw_max_cycles = benchmark.get("autonomy_max_cycles")
        try:
            configured_watchdog_max_cycles = (
                int(raw_max_cycles)
                if raw_max_cycles is not None
                else self._pro_mode_autonomous_cycle_max_cycles()
            )
        except Exception:
            configured_watchdog_max_cycles = self._pro_mode_autonomous_cycle_max_cycles()
        watchdog_max_cycles = max(1, min(32, configured_watchdog_max_cycles))
        watchdog_max_tool_calls = max(1, int(max_tool_calls))
        watchdog_runtime_seconds = max(60, int(max_runtime_seconds))
        phase_timeout_seconds = max(
            30,
            min(
                self._pro_mode_autonomous_cycle_phase_timeout_seconds(),
                watchdog_runtime_seconds,
            ),
        )
        watchdog_started = time.monotonic()
        watchdog_deadline = watchdog_started + float(watchdog_runtime_seconds)
        controller_tools = [ReasoningTools(add_instructions=True, add_few_shot=True)]
        counterfactual_verification_required = bool(
            self._is_option_based_reasoning_request(latest_user_text)
            and not uploaded_files
            and not bool(
                dict(selection_context or {}).get("resource_uris")
                or dict(selection_context or {}).get("dataset_uris")
            )
            and not self._selection_context_has_artifact_handles(selection_context)
        )
        prior_autonomy_state = (
            {} if disable_resume else self._saved_autonomy_state(autonomy_state_seed)
        )
        resumed = bool(prior_autonomy_state)
        prior_cycles_completed = int(prior_autonomy_state.get("cycles_completed") or 0)
        phase_timings: dict[str, float] = {}
        tool_invocations: list[dict[str, Any]] = []
        action_history: list[dict[str, Any]] = []
        reasoning_trace_summary: list[dict[str, Any]] = []
        tool_families_used = [
            str(item or "").strip()
            for item in list(prior_autonomy_state.get("tool_families_used") or [])
            if str(item or "").strip()
        ]
        initial_open_obligations = [
            str(item or "").strip()
            for item in list(prior_autonomy_state.get("open_obligations") or [])
            if str(item or "").strip()
        ]
        inferred_task_regime = self._task_regime_for_turn(
            user_text=latest_user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
            tool_plan=None,
        )
        requires_dual_candidates = self._rtd_requires_challenger(
            latest_user_text,
            task_regime=inferred_task_regime,
        )
        rtd_state = self._coerce_rtd_state(
            autonomy_state_seed=autonomy_state_seed,
            latest_user_text=latest_user_text,
            task_regime=inferred_task_regime,
            cycles_completed=prior_cycles_completed,
            tool_families_used=list(tool_families_used),
        )

        def _legacy_autonomy_state_from_rtd(
            state: RTDEpistemicState,
            *,
            counterfactual_required: bool,
            counterfactual_pending: bool,
            counterfactual_completed: bool,
        ) -> dict[str, Any]:
            return {
                "cycle_id": str(prior_autonomy_state.get("cycle_id") or uuid4().hex[:16]).strip(),
                "checkpoint_index": int(state.checkpoint.cycle_index or prior_cycles_completed),
                "open_obligations": list(state.obligation_ledger or []),
                "evidence_ledger": list(state.evidence_ledger or []),
                "candidate_answer": str(state.candidate_set.leader.answer_text or "").strip(),
                "stop_reason": str(state.checkpoint.stop_reason or "").strip(),
                "resume_readiness": str(state.checkpoint.resume_readiness or "").strip() or "ready",
                "next_best_actions": list(state.checkpoint.next_best_actions or []),
                "cycles_completed": int(state.budget_state.cycles_used or prior_cycles_completed),
                "tool_families_used": list(state.budget_state.tool_families_used or []),
                "continuation_fidelity": float(state.continuation_fidelity or 0.0),
                "watchdog_triggered": bool(state.budget_state.watchdog_triggered),
                "watchdog_reasons": list(state.budget_state.watchdog_reasons or []),
                "counterfactual_verification_required": bool(counterfactual_required),
                "counterfactual_verification_pending": bool(counterfactual_pending),
                "counterfactual_verification_completed": bool(counterfactual_completed),
            }

        def _persist_autonomy_state_snapshot(autonomy_metadata: dict[str, Any]) -> None:
            self._save_pro_mode_conversation_state(
                conversation_id=conversation_id,
                user_id=user_id,
                run_id=run_id,
                state=self._extract_pro_mode_conversation_state(
                    latest_user_text=latest_user_text,
                    uploaded_files=uploaded_files,
                    selection_context=selection_context,
                    task_regime=inferred_task_regime,
                    tool_invocations=list(tool_invocations),
                    research_program_meta=None,
                    autonomy_meta=autonomy_metadata,
                ),
                title=(latest_user_text[:120].strip() or "Pro Mode conversation state"),
            )

        memory_messages = [
            str(item or "").strip()
            for item in list((shared_context or {}).get("memory_messages") or [])
            if str(item or "").strip()
        ]
        knowledge_messages = [
            str(item or "").strip()
            for item in list((shared_context or {}).get("knowledge_messages") or [])
            if str(item or "").strip()
        ]
        if disable_memory_knowledge:
            memory_messages = []
            knowledge_messages = []
        session_state: dict[str, Any] = {
            "pro_mode_context": {
                "memory_messages": memory_messages[:6],
                "knowledge_messages": knowledge_messages[:6],
            },
            "autonomy_state_v2": rtd_state.model_dump(mode="json"),
        }
        session_state["autonomy_state"] = _legacy_autonomy_state_from_rtd(
            rtd_state,
            counterfactual_required=bool(
                (
                    prior_autonomy_state.get("counterfactual_verification_required")
                    if "counterfactual_verification_required" in prior_autonomy_state
                    else counterfactual_verification_required
                )
                and not disable_falsifier
            ),
            counterfactual_pending=bool(
                prior_autonomy_state.get("counterfactual_verification_pending")
            ),
            counterfactual_completed=bool(
                prior_autonomy_state.get("counterfactual_verification_completed")
            ),
        )
        if resumed:
            self._emit_event(
                event_callback,
                {
                    "kind": "graph",
                    "event_type": "autonomy.resumed",
                    "phase": "autonomous_cycle",
                    "status": "resumed",
                    "message": "Resumed the saved long-cycle autonomy state.",
                    "payload": {
                        "cycle_id": session_state["autonomy_state"]["cycle_id"],
                        "checkpoint_index": session_state["autonomy_state"]["checkpoint_index"],
                        "open_obligation_count": len(initial_open_obligations),
                    },
                },
            )

        def _remaining_watchdog_seconds(*, desired: int, floor: int = 30) -> int:
            remaining = int(max(1.0, watchdog_deadline - time.monotonic()))
            if remaining <= int(floor):
                return max(1, remaining)
            return max(floor, min(int(desired), remaining))

        def _watchdog_reasons() -> list[str]:
            reasons: list[str] = []
            if time.monotonic() >= watchdog_deadline:
                reasons.append("runtime_watchdog")
            if len(tool_invocations) >= watchdog_max_tool_calls:
                reasons.append("tool_call_watchdog")
            if int(local_state.get("iteration_count") or 0) >= watchdog_max_cycles:
                reasons.append("cycle_watchdog")
            return reasons

        def _dedupe_preserve(items: list[str], *, limit: int) -> list[str]:
            ordered: list[str] = []
            seen: set[str] = set()
            for item in items:
                token = str(item or "").strip()
                if not token or token in seen:
                    continue
                seen.add(token)
                ordered.append(token)
            return ordered[-limit:]

        def _response_excerpt(value: Any, *, limit: int = 900) -> str:
            text = str(value or "").strip()
            if len(text) <= limit:
                return text
            return text[: limit - 3].rstrip() + "..."

        def _build_action_request(
            *,
            plan: AutonomousCyclePlan,
            selected_action: str,
            cycle_index: int,
            autonomy_state: dict[str, Any],
        ) -> str:
            rtd_state = RTDEpistemicState.model_validate(
                dict(session_state.get("autonomy_state_v2") or {}) or {}
            )
            candidate_answer = (
                str(plan.candidate_set.leader.answer_text or "").strip()
                or str(plan.candidate_answer or "").strip()
                or str(rtd_state.candidate_set.leader.answer_text or "").strip()
                or str(autonomy_state.get("candidate_answer") or "").strip()
                or str(local_state.get("last_response_text") or "").strip()
            )
            challenger_answer = (
                str(plan.candidate_set.challenger.answer_text or "").strip()
                or str(rtd_state.candidate_set.challenger.answer_text or "").strip()
            )
            open_obligations = _dedupe_preserve(
                [
                    *[str(item or "").strip() for item in list(plan.obligation_ledger or [])],
                    *[str(item or "").strip() for item in list(plan.open_obligations or [])],
                    *[str(item or "").strip() for item in list(rtd_state.obligation_ledger or [])],
                    *[
                        str(item or "").strip()
                        for item in list(autonomy_state.get("open_obligations") or [])
                    ],
                ],
                limit=6,
            )
            next_best_actions = _dedupe_preserve(
                [
                    *[
                        str(item or "").strip()
                        for item in list(rtd_state.checkpoint.next_best_actions or [])
                    ],
                    *[str(item or "").strip() for item in list(plan.next_best_actions or [])],
                    *[
                        str(item or "").strip()
                        for item in list(autonomy_state.get("next_best_actions") or [])
                    ],
                ],
                limit=4,
            )
            sections = [
                "You are handling one bounded step inside an autonomous scientific reasoning cycle.",
                f"Cycle index: {cycle_index}",
                f"Assigned action: {selected_action}",
                "Primary user request:",
                str(latest_user_text or "").strip(),
            ]
            counterfactual_pending = bool(autonomy_state.get("counterfactual_verification_pending"))
            if candidate_answer:
                sections.extend(
                    [
                        "",
                        "Current candidate answer or working hypothesis:",
                        candidate_answer,
                    ]
                )
            if challenger_answer:
                sections.extend(
                    [
                        "",
                        "Current challenger or strongest alternative:",
                        challenger_answer,
                    ]
                )
            if counterfactual_pending:
                sections.extend(
                    [
                        "",
                        "This is the counterfactual verification pass.",
                        "Assume the current leading answer may be wrong. Actively try to falsify it, identify the strongest alternative, and only keep the current answer if it survives that challenge.",
                    ]
                )
            if str(plan.action_rationale or "").strip():
                sections.extend(
                    [
                        "",
                        "Why this step was selected:",
                        str(plan.action_rationale or "").strip(),
                    ]
                )
            if open_obligations:
                sections.append("")
                sections.append("Unresolved obligations to address in this step:")
                sections.extend(f"- {item}" for item in open_obligations)
            if next_best_actions:
                sections.append("")
                sections.append("Specific checks or refinements to pursue now:")
                sections.extend(f"- {item}" for item in next_best_actions)
            sections.append("")
            if selected_action == "tool_workflow":
                sections.append(
                    "Task for this step: gather only the deterministic evidence that could change the current answer, then return the strongest evidence-backed update."
                )
            elif selected_action == "focused_team":
                sections.append(
                    "Task for this step: synthesize the current problem clearly, but pressure-test the strongest alternative interpretation before settling on a final answer."
                )
            else:
                sections.append(
                    "Task for this step: resolve the decisive crux, test the strongest competing explanation, and revise the current answer if it fails."
                )
            sections.append(
                "Do not restart from scratch or ignore the working state unless the current candidate answer is clearly wrong."
            )
            return "\n".join(sections).strip()

        async def _run_rtd_skill_workflow(
            skill_name: RTDSkill,
            *,
            selected_action_hint: str,
            selected_tool_names: list[str],
            action_request: str,
        ) -> tuple[str, dict[str, Any], list[dict[str, Any]], str, RTDSkill]:
            workflow_messages = list(messages[-4:]) or [
                {"role": "user", "content": latest_user_text}
            ]
            if workflow_messages and workflow_messages[-1].get("role") == "user":
                workflow_messages = [
                    *workflow_messages[:-1],
                    {"role": "user", "content": action_request},
                ]
            else:
                workflow_messages.append({"role": "user", "content": action_request})

            async def _skill_executor(_step_input: StepInput) -> StepOutput:
                if skill_name == "focused_synthesis_team_workflow":
                    result = await self._run_pro_mode_focused_team(
                        latest_user_text=action_request,
                        shared_context={
                            "memory_messages": memory_messages,
                            "knowledge_messages": knowledge_messages,
                        },
                        conversation_id=conversation_id,
                        run_id=run_id,
                        user_id=user_id,
                        max_runtime_seconds=_remaining_watchdog_seconds(
                            desired=phase_timeout_seconds,
                            floor=60,
                        ),
                        debug=debug,
                    )
                    return StepOutput(
                        content={
                            "response_text": str(result.response_text or "").strip(),
                            "metadata": dict(result.metadata or {}),
                            "tool_invocations": list(result.tool_invocations or []),
                            "selected_action": "focused_team",
                        }
                    )
                if skill_name == "proof_derivation_workflow":
                    result = await self._run_pro_mode_proof_workflow(
                        latest_user_text=action_request,
                        shared_context={
                            "memory_messages": memory_messages,
                            "knowledge_messages": knowledge_messages,
                        },
                        proof_state_seed=autonomy_state_seed,
                        conversation_id=conversation_id,
                        run_id=run_id,
                        user_id=user_id,
                        max_runtime_seconds=_remaining_watchdog_seconds(
                            desired=phase_timeout_seconds,
                            floor=60,
                        ),
                        debug=debug,
                    )
                    return StepOutput(
                        content={
                            "response_text": str(result.response_text or "").strip(),
                            "metadata": dict(result.metadata or {}),
                            "tool_invocations": list(result.tool_invocations or []),
                            "selected_action": "reasoning_solver",
                        }
                    )
                if skill_name == "deterministic_numeric_workflow":
                    numeric_result = None
                    try:
                        numeric_result = await self._run_validated_numeric_workflow(
                            latest_user_text=action_request,
                            conversation_id=conversation_id,
                            run_id=run_id,
                            user_id=user_id,
                            max_runtime_seconds=_remaining_watchdog_seconds(
                                desired=phase_timeout_seconds,
                                floor=45,
                            ),
                            reasoning_mode=reasoning_mode,
                            debug=debug,
                        )
                    except Exception:
                        numeric_result = None
                    if numeric_result:
                        return StepOutput(
                            content={
                                "response_text": str(
                                    numeric_result.get("response_text") or ""
                                ).strip(),
                                "metadata": dict(numeric_result.get("metadata") or {}),
                                "tool_invocations": list(
                                    numeric_result.get("tool_invocations") or []
                                ),
                                "selected_action": "tool_workflow",
                            }
                        )
                if (
                    selected_action_hint == "reasoning_solver"
                    or skill_name == "counterfactual_verification_workflow"
                ):
                    result = await self._run_pro_mode_reasoning_solver(
                        latest_user_text=action_request,
                        task_regime=inferred_task_regime,
                        shared_context={
                            "memory_messages": memory_messages,
                            "knowledge_messages": knowledge_messages,
                        },
                        conversation_id=conversation_id,
                        run_id=run_id,
                        user_id=user_id,
                        max_runtime_seconds=_remaining_watchdog_seconds(
                            desired=phase_timeout_seconds,
                            floor=60,
                        ),
                        debug=debug,
                    )
                    return StepOutput(
                        content={
                            "response_text": str(result.response_text or "").strip(),
                            "metadata": dict(result.metadata or {}),
                            "tool_invocations": list(result.tool_invocations or []),
                            "selected_action": "reasoning_solver",
                        }
                    )
                if skill_name in {
                    "evidence_review_workflow",
                    "programmatic_experiment_workflow",
                    "deterministic_numeric_workflow",
                }:
                    effective_tools = list(selected_tool_names or [])
                    if skill_name == "programmatic_experiment_workflow" and not effective_tools:
                        effective_tools = ["codegen_python_plan", "execute_python_job"]
                    elif skill_name == "deterministic_numeric_workflow" and not effective_tools:
                        effective_tools = ["numpy_calculator"]
                    elif skill_name == "evidence_review_workflow" and not effective_tools:
                        effective_tools = self._research_program_tool_bundle(
                            user_text=latest_user_text,
                            uploaded_files=uploaded_files,
                            selection_context=selection_context,
                            prior_pro_mode_state=autonomy_state_seed,
                        )
                    tool_result = await self._run_pro_mode_tool_workflow(
                        messages=workflow_messages,
                        latest_user_text=action_request,
                        uploaded_files=uploaded_files,
                        max_tool_calls=watchdog_max_tool_calls,
                        max_runtime_seconds=_remaining_watchdog_seconds(
                            desired=phase_timeout_seconds,
                            floor=60,
                        ),
                        reasoning_mode=reasoning_mode,
                        conversation_id=conversation_id,
                        run_id=run_id,
                        user_id=user_id,
                        event_callback=event_callback,
                        selected_tool_names=list(effective_tools or []),
                        tool_plan_category=None,
                        strict_tool_validation=False,
                        selection_context=selection_context,
                        knowledge_context=None,
                        shared_context={
                            "memory_messages": memory_messages,
                            "knowledge_messages": knowledge_messages,
                        },
                        conversation_state_seed=autonomy_state_seed,
                        debug=debug,
                        allow_research_program=True,
                    )
                    return StepOutput(
                        content={
                            "response_text": str(tool_result.get("response_text") or "").strip(),
                            "metadata": dict(tool_result.get("metadata") or {}),
                            "tool_invocations": list(tool_result.get("tool_invocations") or []),
                            "selected_action": "tool_workflow",
                        }
                    )
                result = await self._run_pro_mode_reasoning_solver(
                    latest_user_text=action_request,
                    task_regime=inferred_task_regime,
                    shared_context={
                        "memory_messages": memory_messages,
                        "knowledge_messages": knowledge_messages,
                    },
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    max_runtime_seconds=_remaining_watchdog_seconds(
                        desired=phase_timeout_seconds,
                        floor=60,
                    ),
                    debug=debug,
                )
                return StepOutput(
                    content={
                        "response_text": str(result.response_text or "").strip(),
                        "metadata": dict(result.metadata or {}),
                        "tool_invocations": list(result.tool_invocations or []),
                        "selected_action": "reasoning_solver",
                    }
                )

            workflow = Workflow(
                name=f"rtd_{skill_name}",
                description="Single-skill RTD workflow step.",
                steps=[Step(name=str(skill_name), executor=_skill_executor)],
                stream=False,
                telemetry=False,
                store_events=False,
                debug_mode=bool(debug),
            )
            output = await workflow.arun(
                input={"skill": skill_name, "request": action_request},
                user_id=user_id,
                run_id=f"{str(run_id or '').strip()}::{skill_name}" if run_id else None,
                session_id=self._scope_session_id(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    run_id=f"{str(run_id or '').strip()}::{skill_name}"
                    if run_id
                    else str(skill_name),
                ),
            )
            content = dict(getattr(output, "content", {}) or {})
            return (
                str(content.get("response_text") or "").strip(),
                dict(content.get("metadata") or {}),
                list(content.get("tool_invocations") or []),
                str(content.get("selected_action") or self._rtd_skill_to_action(skill_name)).strip()
                or self._rtd_skill_to_action(skill_name),
                skill_name,
            )

        local_state: dict[str, Any] = {
            "iteration_count": 0,
            "cycle_offset": prior_cycles_completed,
            "current_cycle_index": prior_cycles_completed,
            "current_plan": {},
            "current_action": {},
            "current_analysis": {},
            "last_response_text": str(prior_autonomy_state.get("candidate_answer") or "").strip(),
            "stop_reason": str(prior_autonomy_state.get("stop_reason") or "").strip(),
            "self_correction_count": 0,
            "cycles_completed": prior_cycles_completed,
            "evidence_sufficiency_score": float(rtd_state.evidence_sufficiency_score or 0.0),
            "loop_should_stop": False,
            "counterfactual_verification_required": bool(
                session_state["autonomy_state"].get("counterfactual_verification_required")
            ),
            "counterfactual_verification_pending": bool(
                session_state["autonomy_state"].get("counterfactual_verification_pending")
            ),
            "counterfactual_verification_completed": bool(
                session_state["autonomy_state"].get("counterfactual_verification_completed")
            ),
            "current_skill": "",
        }

        # If a prior autonomous cycle already converged to a stable candidate answer,
        # treat follow-up reframing requests as answer-shaping turns rather than
        # sending them back through the full planner loop.
        if self._is_autonomy_resume_synthesis_turn(latest_user_text, autonomy_state_seed):
            next_cycle_index = (
                max(
                    int(prior_autonomy_state.get("checkpoint_index") or 0),
                    prior_cycles_completed,
                )
                + 1
            )
            prior_rtd_payload = dict(session_state.get("autonomy_state_v2") or {})
            prior_rtd_state = RTDEpistemicState.model_validate(prior_rtd_payload or {})
            prior_rtd_state.checkpoint = RTDCheckpoint(
                cycle_index=next_cycle_index,
                stop_reason="resume_synthesis",
                resume_readiness="not_needed",
                next_best_actions=[],
            )
            prior_rtd_state.budget_state.cycles_used = next_cycle_index
            final_autonomy_state = {
                "cycle_id": str(prior_autonomy_state.get("cycle_id") or uuid4().hex[:16]).strip(),
                "checkpoint_index": next_cycle_index,
                "open_obligations": [],
                "evidence_ledger": list(prior_autonomy_state.get("evidence_ledger") or []),
                "candidate_answer": str(prior_autonomy_state.get("candidate_answer") or "").strip(),
                "stop_reason": "resume_synthesis",
                "resume_readiness": "not_needed",
                "next_best_actions": [],
                "cycles_completed": next_cycle_index,
                "tool_families_used": list(tool_families_used),
                "continuation_fidelity": 1.0,
                "counterfactual_verification_required": bool(
                    prior_autonomy_state.get("counterfactual_verification_required")
                ),
                "counterfactual_verification_pending": False,
                "counterfactual_verification_completed": bool(
                    prior_autonomy_state.get("counterfactual_verification_completed")
                ),
            }
            session_state["autonomy_state"] = dict(final_autonomy_state)
            session_state["autonomy_state_v2"] = prior_rtd_state.model_dump(mode="json")
            supporting_points = _dedupe_preserve(
                [
                    *[
                        str(item or "").strip()
                        for item in list(prior_autonomy_state.get("evidence_ledger") or [])
                        if str(item or "").strip()
                    ],
                    str(prior_autonomy_state.get("candidate_answer") or "").strip(),
                ],
                limit=16,
            )
            rewritten_response, _writer_stats = await self._run_pro_mode_final_writer(
                latest_user_text=latest_user_text,
                draft_response_text=str(prior_autonomy_state.get("candidate_answer") or "").strip(),
                execution_regime="autonomous_cycle",
                task_regime="conceptual_high_uncertainty",
                supporting_points=supporting_points,
                reservations=[],
                session_state={
                    "pro_mode_context": dict(session_state.get("pro_mode_context") or {}),
                    "pro_mode": {"autonomy_state": dict(final_autonomy_state)},
                },
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                max_runtime_seconds=min(max_runtime_seconds, 60),
                debug=debug,
            )
            final_response_text = str(
                rewritten_response or prior_autonomy_state.get("candidate_answer") or ""
            ).strip()
            cycle_metrics = {
                "cycles_completed": int(next_cycle_index),
                "tool_families_used": list(tool_families_used),
                "self_correction_count": 0,
                "stop_reason": "resume_synthesis",
                "checkpoint_coverage": round(
                    float(next_cycle_index) / float(max(watchdog_max_cycles, 1)), 3
                ),
                "evidence_sufficiency_score": 1.0,
                "continuation_fidelity": 1.0,
                "counterfactual_verification_completed": bool(
                    final_autonomy_state.get("counterfactual_verification_completed")
                ),
                "converged": True,
            }
            reasoning_trace_summary.append(
                {
                    "cycle_index": next_cycle_index,
                    "action": "resume_synthesis",
                    "evidence_sufficiency_score": 1.0,
                    "open_obligation_count": 0,
                    "self_correction_needed": False,
                }
            )
            self._emit_event(
                event_callback,
                {
                    "kind": "graph",
                    "event_type": "autonomy.converged",
                    "phase": "autonomous_cycle",
                    "status": "completed",
                    "message": "Autonomous cycle reused the saved answer state and completed a follow-up synthesis.",
                    "payload": {
                        "cycle_index": next_cycle_index,
                        "stop_reason": "resume_synthesis",
                        "resume_fast_path": True,
                    },
                },
            )
            resume_metadata = {
                "pro_mode": {
                    "execution_path": "autonomous_cycle",
                    "execution_regime": "autonomous_cycle",
                    "autonomous_cycle": {
                        **dict(final_autonomy_state),
                        "action_history": [
                            {
                                "cycle_index": next_cycle_index,
                                "selected_action": "resume_synthesis",
                                "action_rationale": "A stable saved answer existed, so the follow-up was handled as a direct reframing step.",
                                "stop_reason": "resume_synthesis",
                                "evidence_sufficiency_score": 1.0,
                                "open_obligation_count": 0,
                            }
                        ],
                    },
                    "autonomy_state": dict(final_autonomy_state),
                    "autonomy_state_v2": prior_rtd_state.model_dump(mode="json"),
                    "candidate_set": prior_rtd_state.candidate_set.model_dump(mode="json"),
                    "obligation_ledger": [],
                    "verification_ledger": [
                        item.model_dump(mode="json") for item in prior_rtd_state.verification_ledger
                    ],
                    "cycle_metrics": cycle_metrics,
                    "cycle_metrics_v2": {
                        **cycle_metrics,
                        "verification_count": int(
                            prior_rtd_state.budget_state.verification_count or 0
                        ),
                    },
                    "stop_decision": {
                        "reason": "resume_synthesis",
                        "checkpoint_index": int(next_cycle_index),
                    },
                    "resume_decision": {
                        "readiness": "not_needed",
                        "next_best_actions": [],
                    },
                    "reasoning_trace_summary": list(reasoning_trace_summary),
                    "phase_timings": {},
                    "controller_toolkits": ["ReasoningTools"],
                    "controller_mode": "rtd_v1",
                    "engine_mode": "workflow_native",
                    "summary": "Reused a converged autonomous-cycle answer for a follow-up synthesis turn.",
                }
            }
            _persist_autonomy_state_snapshot(dict(resume_metadata.get("pro_mode") or {}))
            return ProModeWorkflowResult(
                response_text=final_response_text,
                metadata=resume_metadata,
                tool_invocations=[],
                runtime_status="completed",
            )

        async def initialize(step_input: StepInput) -> StepOutput:
            del step_input
            return StepOutput(content=dict(session_state.get("autonomy_state") or {}))

        async def think_plan(step_input: StepInput) -> StepOutput:
            del step_input
            local_state["iteration_count"] = int(local_state.get("iteration_count") or 0) + 1
            cycle_index = int(local_state.get("cycle_offset") or 0) + int(
                local_state.get("iteration_count") or 0
            )
            local_state["current_cycle_index"] = cycle_index
            state = dict(session_state.get("autonomy_state") or {})
            rtd_state_payload = dict(session_state.get("autonomy_state_v2") or {})
            rtd_state = RTDEpistemicState.model_validate(rtd_state_payload or {})
            state["checkpoint_index"] = cycle_index
            rtd_state.checkpoint.cycle_index = cycle_index
            session_state["autonomy_state"] = state
            session_state["autonomy_state_v2"] = rtd_state.model_dump(mode="json")
            self._emit_event(
                event_callback,
                {
                    "kind": "graph",
                    "event_type": "reasoning.think",
                    "phase": "autonomous_cycle",
                    "status": "started",
                    "message": f"Planning autonomous cycle {cycle_index}.",
                    "payload": {
                        "cycle_index": cycle_index,
                        "open_obligations": list(state.get("open_obligations") or []),
                    },
                },
            )
            plan_fallback = AutonomousCyclePlan(
                objective=str(latest_user_text or "").strip(),
                think_plan="Use one bounded action, then analyze whether more evidence or synthesis is needed.",
                selected_action=self._rtd_skill_to_action(
                    self._rtd_skill_from_context(
                        latest_user_text=latest_user_text,
                        task_regime=inferred_task_regime,
                        uploaded_files=uploaded_files,
                        selection_context=selection_context,
                    )
                ),
                selected_skill=self._rtd_skill_from_context(
                    latest_user_text=latest_user_text,
                    task_regime=inferred_task_regime,
                    uploaded_files=uploaded_files,
                    selection_context=selection_context,
                ),
                action_rationale="Fallback action selected because the controller plan was unavailable.",
                candidate_answer=str(
                    rtd_state.candidate_set.leader.answer_text
                    or state.get("candidate_answer")
                    or ""
                ).strip(),
                candidate_set=rtd_state.candidate_set,
                problem_frame=rtd_state.problem_frame,
                open_obligations=list(
                    rtd_state.obligation_ledger or state.get("open_obligations") or []
                ),
                obligation_ledger=list(rtd_state.obligation_ledger or []),
                evidence_ledger=list(
                    rtd_state.evidence_ledger or state.get("evidence_ledger") or []
                )[-12:],
                verification_ledger=list(rtd_state.verification_ledger or [])[-4:],
                next_best_actions=list(
                    rtd_state.checkpoint.next_best_actions or state.get("next_best_actions") or []
                )[:4],
                selected_tool_names=[],
                request_checkpoint=False,
            )
            plan_prompt = "\n".join(
                [
                    "You are the RTD controller for a bounded long-cycle scientific reasoning workflow.",
                    "Use a Think -> Act -> Analyze discipline with exactly one skill workflow per cycle.",
                    "Choose exactly one next action: `reasoning_solver`, `focused_team`, `tool_workflow`, `finalize`, or `checkpoint`.",
                    "Also choose exactly one skill workflow: `deterministic_numeric_workflow`, `evidence_review_workflow`, `proof_derivation_workflow`, `programmatic_experiment_workflow`, `focused_synthesis_team_workflow`, or `counterfactual_verification_workflow`.",
                    "Routine numeric questions should start with `deterministic_numeric_workflow`. Artifact-grounded tasks should use `evidence_review_workflow`. Proofs should use `proof_derivation_workflow`. Simulation, enumeration, or code-heavy tasks should use `programmatic_experiment_workflow`. Report-quality synthesis should use `focused_synthesis_team_workflow`.",
                    "For option questions, convention-sensitive numeric questions, and ambiguous scientific interpretations, maintain both a leader and a challenger answer. If the leader has not yet survived falsification, schedule `counterfactual_verification_workflow` before finalizing.",
                    "Use `finalize` only when obligations are empty, evidence sufficiency is high, and the latest verification says the leader survived. Use `checkpoint` only when a bounded partial answer plus explicit next steps is better than pretending to converge.",
                    "Update the problem frame, candidate set, and ledgers. Return typed output only and do not expose hidden chain-of-thought.",
                    "",
                    f"User request: {latest_user_text}",
                    "",
                    "Saved autonomy state JSON:",
                    json.dumps(state, ensure_ascii=False, indent=2),
                    "",
                    "RTD epistemic state JSON:",
                    rtd_state.model_dump_json(indent=2),
                    "",
                    "Shared context JSON:",
                    json.dumps(
                        session_state.get("pro_mode_context") or {}, ensure_ascii=False, indent=2
                    ),
                ]
            )
            plan_started = time.monotonic()
            plan = await self._run_tool_program_phase(
                phase_name=f"autonomous_cycle_think_{cycle_index}",
                schema=AutonomousCyclePlan,
                prompt=plan_prompt,
                fallback=plan_fallback,
                tools=controller_tools,
                session_state=session_state,
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                reasoning_mode="deep",
                reasoning_effort_override="high",
                use_reasoning_agent=False,
                max_runtime_seconds=_remaining_watchdog_seconds(
                    desired=phase_timeout_seconds,
                    floor=30,
                ),
                debug=debug,
            )
            phase_timings[f"autonomous_cycle_think_{cycle_index}"] = round(
                time.monotonic() - plan_started, 3
            )
            local_state["current_plan"] = plan.model_dump(mode="json")
            return StepOutput(content=plan.model_dump(mode="json"))

        async def run_action(step_input: StepInput) -> StepOutput:
            plan_payload = dict(
                step_input.get_step_content("autonomous_cycle_think_plan")
                or local_state.get("current_plan")
                or {}
            )
            plan = AutonomousCyclePlan.model_validate(plan_payload or {})
            cycle_index = int(local_state.get("current_cycle_index") or 0)
            selected_action = (
                str(plan.selected_action or "reasoning_solver").strip().lower()
                or "reasoning_solver"
            )
            selected_skill = str(plan.selected_skill or "").strip()
            if not selected_skill:
                if selected_action == "focused_team":
                    selected_skill = "focused_synthesis_team_workflow"
                elif selected_action == "reasoning_solver":
                    if bool(local_state.get("counterfactual_verification_pending")) or bool(
                        re.search(
                            r"counterfactual verification|strongest alternative|falsif",
                            (
                                str(plan.think_plan or "") + " " + str(plan.action_rationale or "")
                            ).lower(),
                        )
                    ):
                        selected_skill = "counterfactual_verification_workflow"
                    elif "proof" in str(latest_user_text or "").lower():
                        selected_skill = "proof_derivation_workflow"
                    else:
                        selected_skill = "evidence_review_workflow"
                elif selected_action == "tool_workflow":
                    if list(plan.selected_tool_names or []):
                        selected_skill = "evidence_review_workflow"
                    else:
                        selected_skill = self._rtd_skill_from_context(
                            latest_user_text=latest_user_text,
                            task_regime=inferred_task_regime,
                            uploaded_files=uploaded_files,
                            selection_context=selection_context,
                        )
                else:
                    selected_skill = self._rtd_skill_from_context(
                        latest_user_text=latest_user_text,
                        task_regime=inferred_task_regime,
                        uploaded_files=uploaded_files,
                        selection_context=selection_context,
                    )
            if disable_focused_team_delegate and selected_action == "focused_team":
                selected_action = "reasoning_solver"
                if selected_skill == "focused_synthesis_team_workflow":
                    selected_skill = "counterfactual_verification_workflow"
            if selected_action not in {
                "reasoning_solver",
                "focused_team",
                "tool_workflow",
                "finalize",
                "checkpoint",
            }:
                selected_action = (
                    self._rtd_skill_to_action(selected_skill)
                    if selected_action not in {"finalize", "checkpoint"}
                    else "reasoning_solver"
                )
            state = dict(session_state.get("autonomy_state") or {})
            response_text_for_cycle = str(
                plan.candidate_answer or state.get("candidate_answer") or ""
            ).strip()
            action_metadata: dict[str, Any] = {}
            action_tool_invocations: list[dict[str, Any]] = []
            if selected_action not in {"finalize", "checkpoint"}:
                selected_tools = list(plan.selected_tool_names or [])
                action_request = _build_action_request(
                    plan=plan,
                    selected_action=selected_action,
                    cycle_index=cycle_index,
                    autonomy_state=state,
                )
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "event_type": "autonomy.skill_selected",
                        "phase": "autonomous_cycle",
                        "status": "started",
                        "message": f"Selected `{selected_skill}` for autonomous cycle {cycle_index}.",
                        "payload": {
                            "cycle_index": cycle_index,
                            "selected_skill": selected_skill,
                            "selected_action": selected_action,
                        },
                    },
                )
                if selected_skill == "counterfactual_verification_workflow":
                    self._emit_event(
                        event_callback,
                        {
                            "kind": "graph",
                            "event_type": "autonomy.verification_started",
                            "phase": "autonomous_cycle",
                            "status": "started",
                            "message": f"Running challenger verification in cycle {cycle_index}.",
                            "payload": {
                                "cycle_index": cycle_index,
                                "selected_skill": selected_skill,
                            },
                        },
                    )
                action_started = time.monotonic()
                (
                    response_text_for_cycle,
                    action_metadata,
                    action_tool_invocations,
                    selected_action,
                    selected_skill,
                ) = await _run_rtd_skill_workflow(
                    selected_skill,
                    selected_action_hint=selected_action,
                    selected_tool_names=selected_tools,
                    action_request=action_request,
                )
                phase_timings[f"autonomous_cycle_run_{cycle_index}"] = round(
                    time.monotonic() - action_started, 3
                )
                tool_invocations.extend(action_tool_invocations)
                for invocation in action_tool_invocations:
                    family = self._research_program_tool_family(
                        str(invocation.get("tool") or "").strip()
                    )
                    if family and family not in tool_families_used:
                        tool_families_used.append(family)
            local_state["last_response_text"] = (
                _response_excerpt(response_text_for_cycle)
                or str(local_state.get("last_response_text") or "").strip()
            )
            local_state["current_action"] = {
                "selected_action": selected_action,
                "selected_skill": selected_skill,
                "response_text": str(response_text_for_cycle or "").strip(),
                "metadata": dict(action_metadata),
                "tool_invocations": list(action_tool_invocations),
            }
            local_state["current_skill"] = str(selected_skill or "")
            return StepOutput(content=dict(local_state.get("current_action") or {}))

        async def analyze_result(step_input: StepInput) -> StepOutput:
            plan_payload = dict(
                step_input.get_step_content("autonomous_cycle_think_plan")
                or local_state.get("current_plan")
                or {}
            )
            plan = AutonomousCyclePlan.model_validate(plan_payload or {})
            action_payload = dict(
                step_input.get_step_content("autonomous_cycle_run_action")
                or local_state.get("current_action")
                or {}
            )
            selected_action = (
                str(action_payload.get("selected_action") or "reasoning_solver").strip().lower()
                or "reasoning_solver"
            )
            selected_skill = str(
                action_payload.get("selected_skill")
                or local_state.get("current_skill")
                or plan.selected_skill
                or ""
            ).strip() or self._rtd_skill_from_context(
                latest_user_text=latest_user_text,
                task_regime=inferred_task_regime,
                uploaded_files=uploaded_files,
                selection_context=selection_context,
            )
            response_text_for_cycle = str(action_payload.get("response_text") or "").strip()
            state = dict(session_state.get("autonomy_state") or {})
            rtd_state = RTDEpistemicState.model_validate(
                dict(session_state.get("autonomy_state_v2") or {}) or {}
            )
            cycle_index = int(local_state.get("current_cycle_index") or 0)
            combined_ledger = _dedupe_preserve(
                [
                    *[
                        str(item or "").strip()
                        for item in list(
                            rtd_state.evidence_ledger or state.get("evidence_ledger") or []
                        )
                    ],
                    *[str(item or "").strip() for item in list(plan.evidence_ledger or [])],
                    *(
                        [_response_excerpt(response_text_for_cycle)]
                        if str(response_text_for_cycle or "").strip()
                        else []
                    ),
                ],
                limit=24,
            )
            analysis_fallback = AutonomousCycleAnalysis(
                should_continue=bool(selected_action not in {"finalize", "checkpoint"}),
                should_finalize=bool(selected_action == "finalize"),
                evidence_sufficiency_score=(
                    0.8
                    if selected_action in {"finalize", "checkpoint"}
                    else 0.65
                    if str(response_text_for_cycle or "").strip()
                    else 0.35
                ),
                self_correction_needed=False,
                stop_reason=(
                    "controller_requested_finalize"
                    if selected_action == "finalize"
                    else "checkpoint_requested"
                    if selected_action == "checkpoint"
                    else "safety_watchdog_pending"
                    if bool(_watchdog_reasons())
                    else "continue"
                ),
                resume_readiness=("ready" if selected_action == "checkpoint" else "not_needed"),
                open_obligations=list(
                    plan.obligation_ledger
                    or plan.open_obligations
                    or rtd_state.obligation_ledger
                    or []
                ),
                obligation_ledger=list(
                    plan.obligation_ledger
                    or plan.open_obligations
                    or rtd_state.obligation_ledger
                    or []
                ),
                next_best_actions=list(plan.next_best_actions or []),
                candidate_answer=str(response_text_for_cycle or "").strip(),
                candidate_set=plan.candidate_set
                if (
                    plan.candidate_set.leader.answer_text
                    or plan.candidate_set.challenger.answer_text
                )
                else rtd_state.candidate_set,
                verification_ledger=list(rtd_state.verification_ledger or [])[-4:],
                critique="Fallback analyzer result.",
            )
            analysis_prompt = "\n".join(
                [
                    "Analyze the result of the latest RTD cycle.",
                    "Decide whether the workflow should continue, finalize, or stop at a checkpoint.",
                    "Update the candidate set, obligation ledger, and verification ledger.",
                    "Do not finalize unless obligations are empty, evidence sufficiency is at least 0.8, and the latest verification says the leader survived when a challenger is required.",
                    "Keep the analysis typed and concise. Do not expose hidden chain-of-thought.",
                    "",
                    f"User request: {latest_user_text}",
                    "",
                    "Controller plan JSON:",
                    plan.model_dump_json(indent=2),
                    "",
                    "Action result summary:",
                    _response_excerpt(response_text_for_cycle, limit=1200) or "No visible output.",
                    "",
                    "Current evidence ledger JSON:",
                    json.dumps(combined_ledger, ensure_ascii=False, indent=2),
                    "",
                    "Current autonomy state JSON:",
                    json.dumps(state, ensure_ascii=False, indent=2),
                    "",
                    "Current RTD epistemic state JSON:",
                    rtd_state.model_dump_json(indent=2),
                ]
            )
            self._emit_event(
                event_callback,
                {
                    "kind": "graph",
                    "event_type": "reasoning.analyze",
                    "phase": "autonomous_cycle",
                    "status": "started",
                    "message": f"Analyzing autonomous cycle {cycle_index}.",
                    "payload": {"cycle_index": cycle_index, "selected_action": selected_action},
                },
            )
            analysis_started = time.monotonic()
            analysis = await self._run_tool_program_phase(
                phase_name=f"autonomous_cycle_analyze_{cycle_index}",
                schema=AutonomousCycleAnalysis,
                prompt=analysis_prompt,
                fallback=analysis_fallback,
                tools=controller_tools,
                session_state=session_state,
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                reasoning_mode="deep",
                reasoning_effort_override="high",
                use_reasoning_agent=False,
                max_runtime_seconds=_remaining_watchdog_seconds(
                    desired=phase_timeout_seconds,
                    floor=30,
                ),
                debug=debug,
            )
            phase_timings[f"autonomous_cycle_analyze_{cycle_index}"] = round(
                time.monotonic() - analysis_started, 3
            )
            if analysis.self_correction_needed:
                local_state["self_correction_count"] = (
                    int(local_state.get("self_correction_count") or 0) + 1
                )
            should_finalize = bool(analysis.should_finalize or selected_action == "finalize")
            should_checkpoint = bool(selected_action == "checkpoint" or plan.request_checkpoint)
            if should_finalize and not list(analysis.open_obligations or []):
                open_obligations = []
            else:
                open_obligations = _dedupe_preserve(
                    [
                        *[
                            str(item or "").strip()
                            for item in list(analysis.obligation_ledger or [])
                        ],
                        *[
                            str(item or "").strip()
                            for item in list(analysis.open_obligations or [])
                        ],
                        *[str(item or "").strip() for item in list(plan.obligation_ledger or [])],
                        *[str(item or "").strip() for item in list(plan.open_obligations or [])],
                        *[
                            str(item or "").strip()
                            for item in list(rtd_state.obligation_ledger or [])
                        ],
                    ],
                    limit=12,
                )
            if should_finalize and not list(analysis.next_best_actions or []):
                next_best_actions = []
            else:
                next_best_actions = _dedupe_preserve(
                    [
                        *[
                            str(item or "").strip()
                            for item in list(analysis.next_best_actions or [])
                        ],
                        *[str(item or "").strip() for item in list(plan.next_best_actions or [])],
                    ],
                    limit=8,
                )
            candidate_answer = (
                str(analysis.candidate_answer or "").strip()
                or str(response_text_for_cycle or "").strip()
                or str(
                    rtd_state.candidate_set.leader.answer_text
                    or state.get("candidate_answer")
                    or ""
                ).strip()
            )
            prior_candidate_set = rtd_state.candidate_set.model_copy(deep=True)
            candidate_set = (
                analysis.candidate_set.model_copy(deep=True)
                if (
                    analysis.candidate_set.leader.answer_text
                    or analysis.candidate_set.challenger.answer_text
                )
                else plan.candidate_set.model_copy(deep=True)
                if (
                    plan.candidate_set.leader.answer_text
                    or plan.candidate_set.challenger.answer_text
                )
                else rtd_state.candidate_set.model_copy(deep=True)
            )
            if candidate_answer and not str(candidate_set.leader.answer_text or "").strip():
                candidate_set.leader = RTDCandidate(
                    answer_text=candidate_answer,
                    rationale=str(analysis.critique or response_text_for_cycle or "").strip(),
                    confidence=max(
                        0.05, min(1.0, float(analysis.evidence_sufficiency_score or 0.5))
                    ),
                )
            elif (
                candidate_answer
                and str(candidate_set.leader.answer_text or "").strip() != candidate_answer
            ):
                previous_leader = candidate_set.leader.model_copy(deep=True)
                candidate_set.leader = RTDCandidate(
                    answer_text=candidate_answer,
                    rationale=str(analysis.critique or response_text_for_cycle or "").strip(),
                    confidence=max(
                        0.05, min(1.0, float(analysis.evidence_sufficiency_score or 0.5))
                    ),
                )
                if (
                    not str(candidate_set.challenger.answer_text or "").strip()
                    and str(previous_leader.answer_text or "").strip()
                ):
                    candidate_set.challenger = previous_leader
            if (
                requires_dual_candidates
                and not str(candidate_set.challenger.answer_text or "").strip()
            ):
                seed_answer = str(prior_candidate_set.leader.answer_text or "").strip()
                if (
                    seed_answer
                    and seed_answer != str(candidate_set.leader.answer_text or "").strip()
                ):
                    candidate_set.challenger = RTDCandidate(
                        answer_text=seed_answer,
                        rationale="Preserved as the strongest previously leading alternative.",
                        confidence=max(0.05, float(prior_candidate_set.leader.confidence or 0.4)),
                    )
                else:
                    candidate_set.challenger = RTDCandidate(
                        answer_text="",
                        rationale="Reserved for a challenger or alternative-convention pass.",
                        confidence=0.0,
                    )
            evidence_sufficiency_score = max(
                0.0, min(1.0, float(analysis.evidence_sufficiency_score or 0.0))
            )
            continuation_overlap = 0
            if initial_open_obligations and open_obligations:
                prior_terms = {item.lower() for item in initial_open_obligations}
                current_terms = {item.lower() for item in open_obligations}
                continuation_overlap = len(prior_terms & current_terms)
            continuation_fidelity = (
                1.0
                if not initial_open_obligations
                else min(1.0, continuation_overlap / max(1, len(initial_open_obligations)))
            )
            counterfactual_required = bool(
                state.get("counterfactual_verification_required")
                or local_state.get("counterfactual_verification_required")
                or requires_dual_candidates
            )
            if disable_falsifier:
                counterfactual_required = False
            counterfactual_pending = bool(
                state.get("counterfactual_verification_pending")
                or local_state.get("counterfactual_verification_pending")
            )
            counterfactual_completed = bool(
                state.get("counterfactual_verification_completed")
                or local_state.get("counterfactual_verification_completed")
            )
            if counterfactual_pending:
                counterfactual_pending = False
                counterfactual_completed = True
                local_state["counterfactual_verification_pending"] = False
                local_state["counterfactual_verification_completed"] = True
            verification_ledger = [
                item.model_copy(deep=True) for item in list(rtd_state.verification_ledger or [])
            ]
            if list(analysis.verification_ledger or []):
                verification_ledger.extend(
                    [
                        item.model_copy(deep=True)
                        for item in list(analysis.verification_ledger or [])
                    ][-4:]
                )
            if selected_skill == "counterfactual_verification_workflow":
                verification_ledger.append(
                    RTDVerificationRecord(
                        cycle_index=cycle_index,
                        workflow=selected_skill,
                        outcome="leader_replaced"
                        if bool(analysis.self_correction_needed)
                        else "leader_survived",
                        corrected_error=bool(analysis.self_correction_needed),
                        leader_survived=not bool(analysis.self_correction_needed),
                        notes=str(analysis.critique or response_text_for_cycle or "").strip(),
                        surviving_cruxes=list(open_obligations or [])[:4],
                    )
                )
                counterfactual_completed = True
                counterfactual_pending = False
                local_state["counterfactual_verification_completed"] = True
            latest_verification_ok = self._rtd_verification_satisfied(
                RTDEpistemicState(
                    controller_mode="rtd_v1",
                    problem_frame=rtd_state.problem_frame,
                    candidate_set=candidate_set,
                    obligation_ledger=list(open_obligations),
                    evidence_ledger=list(combined_ledger),
                    verification_ledger=verification_ledger,
                    checkpoint=rtd_state.checkpoint,
                    budget_state=rtd_state.budget_state,
                    evidence_sufficiency_score=evidence_sufficiency_score,
                    continuation_fidelity=continuation_fidelity,
                ),
                requires_challenger=requires_dual_candidates,
            )
            needs_counterfactual_verification = bool(
                counterfactual_required
                and not counterfactual_completed
                and str(candidate_answer or "").strip()
                and not should_checkpoint
                and not bool(_watchdog_reasons())
            )
            analysis_stop_reason = str(analysis.stop_reason or "").strip()
            if should_finalize and (
                open_obligations or evidence_sufficiency_score < 0.8 or not latest_verification_ok
            ):
                should_finalize = False
                if open_obligations:
                    analysis_stop_reason = "obligations_remaining"
                elif evidence_sufficiency_score < 0.8:
                    analysis_stop_reason = "insufficient_evidence"
                else:
                    analysis_stop_reason = "verification_pending"
                    needs_counterfactual_verification = (
                        counterfactual_required
                        and not should_checkpoint
                        and not bool(_watchdog_reasons())
                    )
            if needs_counterfactual_verification:
                counterfactual_pending = True
                should_finalize = False
                evidence_sufficiency_score = min(evidence_sufficiency_score, 0.82)
                local_state["counterfactual_verification_pending"] = True
                open_obligations = _dedupe_preserve(
                    [
                        *open_obligations,
                        "Run one counterfactual verification pass against the current leading answer.",
                    ],
                    limit=12,
                )
                next_best_actions = _dedupe_preserve(
                    [
                        *next_best_actions,
                        "Assume the current leading answer is wrong, test the strongest alternative, and revise the answer if it fails.",
                    ],
                    limit=8,
                )
                analysis_stop_reason = "counterfactual_verification_pending"
            watchdog_reasons = _watchdog_reasons()
            watchdog_triggered = bool(watchdog_reasons) and not should_finalize
            if watchdog_triggered:
                should_finalize = False
                should_checkpoint = True
                analysis_stop_reason = "safety_watchdog_triggered"
                local_state["watchdog_triggered"] = True
                local_state["watchdog_reasons"] = list(watchdog_reasons)
                open_obligations = _dedupe_preserve(
                    [
                        *open_obligations,
                        "Resume from the saved checkpoint to continue the RTD loop after the current safety watchdog window.",
                    ],
                    limit=12,
                )
                next_best_actions = _dedupe_preserve(
                    [
                        *next_best_actions,
                        "Resume the autonomous cycle from the persisted RTD state and continue resolving the remaining obligations.",
                    ],
                    limit=8,
                )
            updated_rtd_state = RTDEpistemicState(
                controller_mode="rtd_v1",
                problem_frame=plan.problem_frame
                if str(plan.problem_frame.objective or "").strip()
                else rtd_state.problem_frame,
                candidate_set=candidate_set,
                obligation_ledger=list(open_obligations),
                evidence_ledger=list(combined_ledger),
                verification_ledger=verification_ledger[-12:],
                checkpoint=RTDCheckpoint(
                    cycle_index=cycle_index,
                    stop_reason=analysis_stop_reason,
                    resume_readiness=str(analysis.resume_readiness or "").strip() or "ready",
                    next_best_actions=list(next_best_actions),
                ),
                budget_state=RTDBudgetState(
                    cycles_used=cycle_index,
                    tool_families_used=list(tool_families_used),
                    model_calls=max(0, (cycle_index * 2) + len(tool_invocations)),
                    verification_count=sum(
                        1
                        for item in verification_ledger[-12:]
                        if str(item.workflow or "").strip()
                        == "counterfactual_verification_workflow"
                    ),
                    watchdog_triggered=bool(watchdog_triggered),
                    watchdog_reasons=list(watchdog_reasons),
                ),
                evidence_sufficiency_score=evidence_sufficiency_score,
                continuation_fidelity=continuation_fidelity,
            )
            session_state["autonomy_state_v2"] = updated_rtd_state.model_dump(mode="json")
            session_state["autonomy_state"] = _legacy_autonomy_state_from_rtd(
                updated_rtd_state,
                counterfactual_required=counterfactual_required,
                counterfactual_pending=counterfactual_pending,
                counterfactual_completed=counterfactual_completed,
            )
            if (
                str(prior_candidate_set.leader.answer_text or "").strip()
                != str(candidate_set.leader.answer_text or "").strip()
                or str(prior_candidate_set.challenger.answer_text or "").strip()
                != str(candidate_set.challenger.answer_text or "").strip()
            ):
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "event_type": "autonomy.branch_updated",
                        "phase": "autonomous_cycle",
                        "status": "completed",
                        "message": f"Updated leader/challenger state in cycle {cycle_index}.",
                        "payload": {
                            "cycle_index": cycle_index,
                            "leader": candidate_set.leader.model_dump(mode="json"),
                            "challenger": candidate_set.challenger.model_dump(mode="json"),
                        },
                    },
                )
            action_history.append(
                {
                    "cycle_index": cycle_index,
                    "selected_action": selected_action,
                    "selected_skill": selected_skill,
                    "action_rationale": str(plan.action_rationale or "").strip(),
                    "stop_reason": analysis_stop_reason,
                    "evidence_sufficiency_score": evidence_sufficiency_score,
                    "open_obligation_count": len(open_obligations),
                }
            )
            reasoning_trace_summary.append(
                {
                    "cycle_index": cycle_index,
                    "action": selected_action,
                    "skill": selected_skill,
                    "evidence_sufficiency_score": evidence_sufficiency_score,
                    "open_obligation_count": len(open_obligations),
                    "self_correction_needed": bool(analysis.self_correction_needed),
                }
            )
            self._emit_event(
                event_callback,
                {
                    "kind": "graph",
                    "event_type": "autonomy.checkpoint_saved",
                    "phase": "autonomous_cycle",
                    "status": "completed",
                    "message": f"Saved checkpoint after autonomous cycle {cycle_index}.",
                    "payload": dict(session_state.get("autonomy_state") or {}),
                },
            )
            local_state["cycles_completed"] = cycle_index
            local_state["stop_reason"] = analysis_stop_reason
            local_state["evidence_sufficiency_score"] = evidence_sufficiency_score
            should_continue = bool(
                analysis.should_continue
                and not bool(_watchdog_reasons())
                and not should_finalize
                and not should_checkpoint
            )
            if needs_counterfactual_verification:
                should_continue = True
            if should_finalize:
                local_state["stop_reason"] = (
                    str(local_state.get("stop_reason") or "").strip() or "converged"
                )
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "event_type": "autonomy.converged",
                        "phase": "autonomous_cycle",
                        "status": "completed",
                        "message": "Autonomous cycle converged to a stable answer.",
                        "payload": {
                            "cycle_index": cycle_index,
                            "stop_reason": str(local_state.get("stop_reason") or "").strip(),
                            "evidence_sufficiency_score": evidence_sufficiency_score,
                        },
                    },
                )
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "event_type": "autonomy.finalized",
                        "phase": "autonomous_cycle",
                        "status": "completed",
                        "message": "RTD finalized a leader answer that survived verification.",
                        "payload": {
                            "cycle_index": cycle_index,
                            "leader": candidate_set.leader.model_dump(mode="json"),
                        },
                    },
                )
                local_state["loop_should_stop"] = True
            elif should_checkpoint or not should_continue:
                if not str(local_state.get("stop_reason") or "").strip():
                    local_state["stop_reason"] = (
                        "checkpoint_requested"
                        if should_checkpoint
                        else "safety_watchdog_triggered"
                        if bool(_watchdog_reasons())
                        else "analyzer_requested_stop"
                    )
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "event_type": "autonomy.stopped",
                        "phase": "autonomous_cycle",
                        "status": "completed",
                        "message": (
                            "Autonomous cycle stopped at a safety-watchdog checkpoint."
                            if str(local_state.get("stop_reason") or "").strip()
                            == "safety_watchdog_triggered"
                            else "Autonomous cycle stopped at a bounded checkpoint."
                        ),
                        "payload": {
                            "cycle_index": cycle_index,
                            "stop_reason": str(local_state.get("stop_reason") or "").strip(),
                            "resume_readiness": str(
                                dict(session_state.get("autonomy_state") or {}).get(
                                    "resume_readiness"
                                )
                                or "ready"
                            ),
                        },
                    },
                )
                local_state["loop_should_stop"] = True
            local_state["current_analysis"] = analysis.model_dump(mode="json")
            return StepOutput(
                content={
                    "analysis": analysis.model_dump(mode="json"),
                    "loop_should_stop": bool(local_state.get("loop_should_stop")),
                    "autonomy_state": dict(session_state.get("autonomy_state") or {}),
                }
            )

        def _loop_end_condition(_step_outputs: list[StepOutput]) -> bool:
            return bool(local_state.get("loop_should_stop"))

        async def final_synthesis(step_input: StepInput) -> StepOutput:
            del step_input
            final_autonomy_state = dict(session_state.get("autonomy_state") or {})
            final_rtd_state = RTDEpistemicState.model_validate(
                dict(session_state.get("autonomy_state_v2") or {}) or {}
            )
            last_response_text = str(local_state.get("last_response_text") or "").strip()
            stop_reason = str(local_state.get("stop_reason") or "").strip()
            cycles_completed = int(local_state.get("cycles_completed") or prior_cycles_completed)
            evidence_sufficiency_score = float(local_state.get("evidence_sufficiency_score") or 0.0)
            if not str(last_response_text or "").strip():
                open_obligations = list(final_autonomy_state.get("open_obligations") or [])
                next_best_actions = list(final_autonomy_state.get("next_best_actions") or [])
                last_response_text = "I made bounded progress but stopped at a checkpoint rather than pretending to have fully converged."
                if open_obligations:
                    last_response_text += (
                        " The main remaining obligations are: "
                        + "; ".join(open_obligations[:4])
                        + "."
                    )
                if next_best_actions:
                    last_response_text += (
                        " The best next actions are: " + "; ".join(next_best_actions[:3]) + "."
                    )
            elif str(stop_reason or "").strip() == "safety_watchdog_triggered":
                open_obligations = list(final_autonomy_state.get("open_obligations") or [])
                next_best_actions = list(final_autonomy_state.get("next_best_actions") or [])
                last_response_text = str(last_response_text).rstrip()
                last_response_text += (
                    " I saved a checkpoint instead of forcing a low-confidence stop."
                )
                if open_obligations:
                    last_response_text += (
                        " Remaining obligations: " + "; ".join(open_obligations[:4]) + "."
                    )
                if next_best_actions:
                    last_response_text += (
                        " Best next actions: " + "; ".join(next_best_actions[:3]) + "."
                    )
            convergence_ready = bool(
                not list(final_autonomy_state.get("open_obligations") or [])
                and str(stop_reason or "").strip()
                in {"converged", "controller_requested_finalize", "finalize", "finalized"}
            )
            checkpoint_coverage = round(
                float(int(local_state.get("iteration_count") or 0))
                / float(max(watchdog_max_cycles, 1)),
                3,
            )
            cycle_metrics = {
                "cycles_completed": int(cycles_completed),
                "tool_families_used": list(tool_families_used),
                "self_correction_count": int(local_state.get("self_correction_count") or 0),
                "stop_reason": str(stop_reason or "").strip() or "analyzer_requested_stop",
                "checkpoint_coverage": checkpoint_coverage,
                "evidence_sufficiency_score": round(float(evidence_sufficiency_score), 3),
                "continuation_fidelity": round(
                    float(final_autonomy_state.get("continuation_fidelity") or 0.0), 3
                ),
                "counterfactual_verification_completed": bool(
                    final_autonomy_state.get("counterfactual_verification_completed")
                ),
                "converged": bool(convergence_ready),
                "stop_policy": "semantic"
                if not bool(final_rtd_state.budget_state.watchdog_triggered)
                else "watchdog",
            }
            cycle_metrics_v2 = {
                **cycle_metrics,
                "verification_count": int(final_rtd_state.budget_state.verification_count or 0),
                "leader_confidence": round(
                    float(final_rtd_state.candidate_set.leader.confidence or 0.0), 3
                ),
                "has_challenger": bool(
                    str(final_rtd_state.candidate_set.challenger.answer_text or "").strip()
                ),
                "watchdog_triggered": bool(final_rtd_state.budget_state.watchdog_triggered),
                "watchdog_reasons": list(final_rtd_state.budget_state.watchdog_reasons or []),
            }
            pro_mode_metadata = {
                "execution_path": "autonomous_cycle",
                "execution_regime": "autonomous_cycle",
                "controller_mode": "rtd_v1",
                "autonomous_cycle": {
                    "cycle_id": str(final_autonomy_state.get("cycle_id") or "").strip(),
                    "checkpoint_index": int(
                        final_autonomy_state.get("checkpoint_index") or cycles_completed
                    ),
                    "open_obligations": list(final_autonomy_state.get("open_obligations") or []),
                    "evidence_ledger": list(final_autonomy_state.get("evidence_ledger") or []),
                    "candidate_answer": str(
                        final_autonomy_state.get("candidate_answer") or ""
                    ).strip(),
                    "stop_reason": str(stop_reason or "").strip() or "analyzer_requested_stop",
                    "resume_readiness": str(
                        final_autonomy_state.get("resume_readiness") or ""
                    ).strip()
                    or "ready",
                    "next_best_actions": list(final_autonomy_state.get("next_best_actions") or []),
                    "cycles_completed": int(cycles_completed),
                    "tool_families_used": list(tool_families_used),
                    "continuation_fidelity": round(
                        float(final_autonomy_state.get("continuation_fidelity") or 0.0), 3
                    ),
                    "counterfactual_verification_required": bool(
                        final_autonomy_state.get("counterfactual_verification_required")
                    ),
                    "counterfactual_verification_pending": bool(
                        final_autonomy_state.get("counterfactual_verification_pending")
                    ),
                    "counterfactual_verification_completed": bool(
                        final_autonomy_state.get("counterfactual_verification_completed")
                    ),
                    "action_history": action_history,
                },
                "autonomy_state": dict(final_autonomy_state),
                "autonomy_state_v2": final_rtd_state.model_dump(mode="json"),
                "candidate_set": final_rtd_state.candidate_set.model_dump(mode="json"),
                "obligation_ledger": list(final_rtd_state.obligation_ledger or []),
                "verification_ledger": [
                    item.model_dump(mode="json")
                    for item in list(final_rtd_state.verification_ledger or [])
                ],
                "cycle_metrics": cycle_metrics,
                "cycle_metrics_v2": cycle_metrics_v2,
                "safety_watchdog": {
                    "runtime_seconds": int(watchdog_runtime_seconds),
                    "tool_call_limit": int(watchdog_max_tool_calls),
                    "cycle_limit": int(watchdog_max_cycles),
                    "phase_timeout_seconds": int(phase_timeout_seconds),
                    "triggered": bool(final_rtd_state.budget_state.watchdog_triggered),
                    "reasons": list(final_rtd_state.budget_state.watchdog_reasons or []),
                },
                "stop_decision": {
                    "reason": str(stop_reason or "").strip() or "analyzer_requested_stop",
                    "checkpoint_index": int(
                        final_autonomy_state.get("checkpoint_index") or cycles_completed
                    ),
                    "policy": "semantic"
                    if not bool(final_rtd_state.budget_state.watchdog_triggered)
                    else "watchdog",
                },
                "resume_decision": {
                    "readiness": str(final_autonomy_state.get("resume_readiness") or "").strip()
                    or "ready",
                    "next_best_actions": list(final_autonomy_state.get("next_best_actions") or []),
                },
                "reasoning_trace_summary": reasoning_trace_summary,
                "phase_timings": dict(phase_timings),
                "controller_toolkits": ["ReasoningTools"],
                "engine_mode": "workflow_native",
            }
            _persist_autonomy_state_snapshot(dict(pro_mode_metadata))
            return StepOutput(
                content={
                    "response_text": str(last_response_text or "").strip(),
                    "metadata": {"pro_mode": pro_mode_metadata},
                    "tool_invocations": list(tool_invocations),
                    "runtime_status": "completed",
                    "runtime_error": None,
                }
            )

        workflow = Workflow(
            name="pro_mode_autonomous_cycle_engine",
            description="Native Agno workflow for bounded long-cycle scientific autonomy.",
            steps=[
                Step(name="autonomous_cycle_initialize", executor=initialize),
                Loop(
                    name="autonomous_cycle_loop",
                    max_iterations=watchdog_max_cycles,
                    end_condition=_loop_end_condition,
                    steps=[
                        Step(name="autonomous_cycle_think_plan", executor=think_plan),
                        Step(name="autonomous_cycle_run_action", executor=run_action),
                        Step(name="autonomous_cycle_analyze_result", executor=analyze_result),
                    ],
                ),
                Step(name="autonomous_cycle_final_synthesis", executor=final_synthesis),
            ],
            stream=False,
            telemetry=False,
            store_events=False,
            debug_mode=bool(debug),
            session_state={"autonomous_cycle_state": {}},
        )
        try:
            output = await workflow.arun(
                input={"messages": messages, "user_text": latest_user_text},
                user_id=user_id,
                run_id=f"{str(run_id or '').strip()}::autonomous_cycle_engine" if run_id else None,
                session_id=self._scope_session_id(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    run_id=f"{str(run_id or '').strip()}::autonomous_cycle_engine"
                    if run_id
                    else "autonomous_cycle_engine",
                ),
            )
        except Exception as exc:
            return ProModeWorkflowResult(
                response_text="",
                metadata={
                    "pro_mode": {
                        "execution_path": "autonomous_cycle",
                        "execution_regime": "autonomous_cycle",
                        "engine_mode": "workflow_native",
                        "runtime_status": "failed",
                    }
                },
                runtime_status="failed",
                runtime_error=str(exc or exc.__class__.__name__),
            )
        content = dict(getattr(output, "content", {}) or {})
        return ProModeWorkflowResult(
            response_text=str(content.get("response_text") or "").strip(),
            metadata=dict(content.get("metadata") or {}),
            tool_invocations=list(content.get("tool_invocations") or []),
            runtime_status=str(content.get("runtime_status") or "completed"),
            runtime_error=content.get("runtime_error"),
        )

    async def _run_pro_mode_proof_workflow(
        self,
        *,
        latest_user_text: str,
        shared_context: dict[str, Any],
        proof_state_seed: dict[str, Any] | None,
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        max_runtime_seconds: int,
        debug: bool | None,
    ) -> ProModeWorkflowResult:
        saved_seed = self._saved_proof_workflow_state(proof_state_seed)
        seeded_proof_state = dict(saved_seed.get("proof_state") or {})
        seeded_proof_frame = dict(saved_seed.get("proof_frame") or {})
        initial_proof_state = (
            ProofWorkflowState.model_validate(seeded_proof_state)
            if seeded_proof_state
            else ProofWorkflowState(
                goal_type="undetermined",
                current_direction="undetermined",
                simplest_anchor="",
                canonical_reduction="",
                proof_outline=[],
                obligations=[
                    ProofObligation(
                        label="main",
                        statement="Classify the theorem target and identify the simplest rigorous reduction.",
                        kind="reduction",
                        status="open",
                    )
                ],
                resolved_obligations=[],
                case_splits=[],
                endgame_strategy="",
                verified_steps=[],
                blocker_gaps=[
                    "Classify the theorem target and find the simplest viable reduction."
                ],
                attack_points=[],
                next_iteration_focus="Identify the simplest valid reduction or substitution.",
                ready_to_finalize=False,
                unresolved_points=["No rigorous proof skeleton has been validated yet."],
                confidence="low",
            )
        )
        initial_progress_score = 0.0
        initial_quality_flags: list[str] = []
        if not seeded_proof_state:
            initial_quality_flags = [
                "goal type not fixed",
                "canonical reduction missing",
                "proof obligations missing"
                if not list(initial_proof_state.obligations or [])
                else "",
                "no verified steps yet",
                "endgame strategy not explicit",
                "blocker gaps remain",
            ]
            initial_quality_flags = [item for item in initial_quality_flags if item][:6]
        local_state: dict[str, Any] = {
            "iteration": int(saved_seed.get("iterations") or 0),
            "proof_state": initial_proof_state.model_copy(
                update={
                    "progress_score": float(
                        saved_seed.get("progress_score") or initial_progress_score
                    ),
                    "quality_flags": list(saved_seed.get("quality_flags") or initial_quality_flags),
                }
            ).model_dump(mode="json"),
            "proof_frame": seeded_proof_frame,
            "iteration_summaries": list(saved_seed.get("iteration_summaries") or [])[-10:],
            "stagnant_iterations": int(saved_seed.get("stagnant_iterations") or 0),
            "last_blocker_signature": str(saved_seed.get("last_blocker_signature") or "").strip(),
            "compression_stats": dict(saved_seed.get("compression_stats") or {}),
        }
        phase_timings: dict[str, float] = {}
        if int(max_runtime_seconds) >= 14400:
            max_iterations = 16
        elif int(max_runtime_seconds) >= 7200:
            max_iterations = 12
        elif int(max_runtime_seconds) >= 3600:
            max_iterations = 8
        elif int(max_runtime_seconds) >= 1800:
            max_iterations = 6
        elif int(max_runtime_seconds) >= 600:
            max_iterations = 4
        else:
            max_iterations = 3
        stagnation_limit = (
            5
            if int(max_runtime_seconds) >= 7200
            else 4
            if int(max_runtime_seconds) >= 3600
            else 3
            if int(max_runtime_seconds) >= 1200
            else 2
        )
        sequential_phase_count = (max_iterations * 5) + 2
        step_cap = max(20, min(120, int(max_runtime_seconds / max(sequential_phase_count, 1))))
        workflow_deadline = time.monotonic() + max(1.0, float(max_runtime_seconds))
        memory_messages = [
            str(item or "").strip()
            for item in list((shared_context or {}).get("memory_messages") or [])
            if str(item or "").strip()
        ]
        knowledge_messages = [
            str(item or "").strip()
            for item in list((shared_context or {}).get("knowledge_messages") or [])
            if str(item or "").strip()
        ]

        def _current_proof_state() -> ProofWorkflowState:
            return ProofWorkflowState.model_validate(dict(local_state.get("proof_state") or {}))

        def _remaining_proof_budget(
            preferred: int | None = None, *, reserve_seconds: int = 0
        ) -> int:
            remaining = max(
                1.0, workflow_deadline - time.monotonic() - float(max(0, reserve_seconds))
            )
            candidate = int(preferred if preferred is not None else step_cap)
            return max(1, min(candidate, int(remaining)))

        def _proof_role_fallback(role: str) -> ProofRoleMemo:
            current = _current_proof_state()
            return ProofRoleMemo(
                role=role,
                summary=current.next_iteration_focus or current.simplest_anchor,
                goal_type=current.goal_type,
                simplest_anchor=current.simplest_anchor,
                canonical_reduction=current.canonical_reduction,
                candidate_direction=current.current_direction,
                proposed_lemmas=list(current.proof_outline[:4]),
                obligations=list(current.obligations[:5]),
                resolved_obligations=list(current.resolved_obligations[:4]),
                case_splits=list(current.case_splits[:4]),
                endgame_strategy=current.endgame_strategy,
                verified_steps=list(current.verified_steps[:4]),
                blocker_gaps=list(current.blocker_gaps[:4]),
                attack_points=list(current.attack_points[:4]),
                next_actions=list(current.unresolved_points[:3]),
                ready_to_finalize=False,
                confidence=current.confidence,
            )

        def _current_proof_frame() -> ProofProblemFrame:
            payload = dict(local_state.get("proof_frame") or {})
            if payload:
                return ProofProblemFrame.model_validate(payload)
            return ProofProblemFrame()

        def _checkpoint_proof_state() -> None:
            proof_meta = {
                "iterations": int(local_state.get("iteration") or 0),
                "reasoning_effort": "high",
                "proof_frame": dict(local_state.get("proof_frame") or {}),
                "proof_state": dict(local_state.get("proof_state") or {}),
                "iteration_summaries": list(local_state.get("iteration_summaries") or [])[-10:],
                "compression_stats": dict(local_state.get("compression_stats") or {}),
                "stagnant_iterations": int(local_state.get("stagnant_iterations") or 0),
                "last_blocker_signature": str(
                    local_state.get("last_blocker_signature") or ""
                ).strip(),
                "proof_status": (
                    "proved"
                    if not list(
                        dict(local_state.get("proof_state") or {}).get("blocker_gaps") or []
                    )
                    and bool(dict(local_state.get("proof_state") or {}).get("ready_to_finalize"))
                    else "partial"
                ),
            }
            state = self._extract_pro_mode_conversation_state(
                latest_user_text=latest_user_text,
                uploaded_files=[],
                selection_context=None,
                task_regime="rigorous_proof",
                tool_invocations=[],
                research_program_meta=None,
                proof_workflow_meta=proof_meta,
            )
            self._save_pro_mode_conversation_state(
                conversation_id=conversation_id,
                user_id=user_id,
                run_id=run_id,
                state=state,
                title="Pro Mode proof workflow",
            )

        def _merge_unique(items: list[str], limit: int = 10) -> list[str]:
            merged: list[str] = []
            seen: set[str] = set()
            for item in items:
                normalized = str(item or "").strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                merged.append(normalized)
                if len(merged) >= limit:
                    break
            return merged

        def _merge_obligations(
            obligations: list[ProofObligation], limit: int = 8
        ) -> list[ProofObligation]:
            merged: list[ProofObligation] = []
            seen: dict[str, ProofObligation] = {}
            status_rank = {"verified": 3, "resolved": 3, "in_progress": 2, "blocked": 1, "open": 0}
            for obligation in obligations:
                statement = str(obligation.statement or "").strip()
                if not statement:
                    continue
                key = statement.lower()
                normalized = ProofObligation(
                    label=str(obligation.label or "").strip() or f"obligation_{len(seen) + 1}",
                    statement=statement,
                    kind=str(obligation.kind or "").strip() or "lemma",
                    status=str(obligation.status or "").strip() or "open",
                )
                existing = seen.get(key)
                if existing is None or status_rank.get(normalized.status, 0) > status_rank.get(
                    existing.status, 0
                ):
                    seen[key] = normalized
            for obligation in seen.values():
                merged.append(obligation)
                if len(merged) >= limit:
                    break
            return merged

        def _apply_sanity_findings(
            state: ProofWorkflowState,
            *,
            source_texts: list[str],
        ) -> ProofWorkflowState:
            findings = self._proof_sanity_findings_for_texts(source_texts)
            if not findings:
                return state.model_copy(
                    update={"sanity_findings": list(state.sanity_findings or [])[:6]}
                )
            blocked_sanity_obligations = [
                ProofObligation(
                    label=f"sanity_check_{index + 1}",
                    statement=finding,
                    kind="sanity_check",
                    status="blocked",
                )
                for index, finding in enumerate(findings)
            ]
            return state.model_copy(
                update={
                    "obligations": _merge_obligations(
                        [*list(state.obligations or []), *blocked_sanity_obligations],
                        limit=10,
                    ),
                    "blocker_gaps": _merge_unique(
                        [*list(state.blocker_gaps or []), *findings], limit=8
                    ),
                    "attack_points": _merge_unique(
                        [*list(state.attack_points or []), *findings], limit=8
                    ),
                    "unresolved_points": _merge_unique(
                        [*list(state.unresolved_points or []), *findings], limit=8
                    ),
                    "sanity_findings": _merge_unique(
                        [*list(state.sanity_findings or []), *findings], limit=6
                    ),
                    "ready_to_finalize": False,
                    "confidence": "low",
                }
            )

        def _proof_progress_snapshot(state: ProofWorkflowState) -> tuple[float, list[str]]:
            score = 0.0
            flags: list[str] = []
            if str(state.goal_type or "").strip():
                score += 0.10
            else:
                flags.append("goal type not fixed")
            if str(state.canonical_reduction or "").strip():
                score += 0.20
            else:
                flags.append("canonical reduction missing")
            if list(state.obligations or []):
                score += 0.15
            else:
                flags.append("proof obligations missing")
            if list(state.verified_steps or []):
                score += 0.15
            else:
                flags.append("no verified steps yet")
            if list(state.case_splits or []):
                score += 0.10
            if str(state.endgame_strategy or "").strip():
                score += 0.10
            else:
                flags.append("endgame strategy not explicit")
            if list(state.resolved_obligations or []):
                score += 0.10
            unresolved_obligations = [
                item
                for item in list(state.obligations or [])
                if str(item.status or "").strip().lower() not in {"verified", "resolved"}
            ]
            if not unresolved_obligations:
                score += 0.05
            else:
                flags.append("open obligations remain")
            if not list(state.blocker_gaps or []):
                score += 0.05
            else:
                flags.append("blocker gaps remain")
            if list(state.sanity_findings or []):
                flags.append("sanity findings remain")
            if bool(state.ready_to_finalize):
                score += 0.10
            score -= 0.03 * min(len(list(state.blocker_gaps or [])), 4)
            score -= 0.12 * min(len(list(state.sanity_findings or [])), 3)
            score = max(0.0, min(1.0, round(score, 3)))
            return score, flags[:6]

        def _proof_state_fallback(memos: list[ProofRoleMemo]) -> ProofWorkflowState:
            current = _current_proof_state()
            goal_type = next(
                (
                    str(item.goal_type or "").strip()
                    for item in memos
                    if str(item.goal_type or "").strip()
                ),
                current.goal_type,
            )
            candidate_directions = [
                str(item.candidate_direction or "").strip()
                for item in memos
                if str(item.candidate_direction or "").strip()
            ]
            current_direction = (
                candidate_directions[0] if candidate_directions else current.current_direction
            )
            simplest_anchor = next(
                (
                    str(item.simplest_anchor or "").strip()
                    for item in memos
                    if str(item.simplest_anchor or "").strip()
                ),
                current.simplest_anchor,
            )
            canonical_reduction = next(
                (
                    str(item.canonical_reduction or "").strip()
                    for item in memos
                    if str(item.canonical_reduction or "").strip()
                ),
                current.canonical_reduction,
            )
            proof_outline = _merge_unique(
                [
                    *list(current.proof_outline or []),
                    *[lemma for memo in memos for lemma in list(memo.proposed_lemmas or [])],
                ],
                limit=10,
            )
            obligations = _merge_obligations(
                [
                    *list(current.obligations or []),
                    *[item for memo in memos for item in list(memo.obligations or [])],
                ],
                limit=10,
            )
            resolved_obligations = _merge_unique(
                [
                    *list(current.resolved_obligations or []),
                    *[item for memo in memos for item in list(memo.resolved_obligations or [])],
                ],
                limit=10,
            )
            case_splits = _merge_unique(
                [
                    *list(current.case_splits or []),
                    *[item for memo in memos for item in list(memo.case_splits or [])],
                ],
                limit=8,
            )
            endgame_strategy = next(
                (
                    str(item.endgame_strategy or "").strip()
                    for item in memos
                    if str(item.endgame_strategy or "").strip()
                ),
                current.endgame_strategy,
            )
            verified_steps = _merge_unique(
                [
                    *list(current.verified_steps or []),
                    *[step for memo in memos for step in list(memo.verified_steps or [])],
                ],
                limit=10,
            )
            blocker_gaps = _merge_unique(
                [gap for memo in memos for gap in list(memo.blocker_gaps or [])],
                limit=8,
            )
            attack_points = _merge_unique(
                [
                    *list(current.attack_points or []),
                    *[point for memo in memos for point in list(memo.attack_points or [])],
                ],
                limit=8,
            )
            unresolved_points = _merge_unique(
                [
                    *blocker_gaps,
                    *[
                        obligation.statement
                        for obligation in list(obligations or [])
                        if str(obligation.status or "").strip().lower()
                        not in {"verified", "resolved"}
                    ],
                    *[item for memo in memos for item in list(memo.next_actions or [])],
                ],
                limit=8,
            )
            all_obligations_closed = bool(obligations) and all(
                str(item.status or "").strip().lower() in {"verified", "resolved"}
                for item in obligations
            )
            next_iteration_focus = (
                blocker_gaps[0]
                if blocker_gaps
                else next(
                    (
                        str(item.next_actions[0] or "").strip()
                        for item in memos
                        if list(item.next_actions or []) and str(item.next_actions[0] or "").strip()
                    ),
                    current.next_iteration_focus,
                )
            )
            ready_to_finalize = (
                bool(verified_steps)
                and all_obligations_closed
                and not blocker_gaps
                and any(bool(item.ready_to_finalize) for item in memos)
            )
            confidence = "high" if ready_to_finalize else ("medium" if verified_steps else "low")
            candidate_state = ProofWorkflowState(
                goal_type=goal_type or current.goal_type or "undetermined",
                current_direction=current_direction or "undetermined",
                simplest_anchor=simplest_anchor,
                canonical_reduction=canonical_reduction,
                proof_outline=proof_outline,
                obligations=obligations,
                resolved_obligations=resolved_obligations,
                case_splits=case_splits,
                endgame_strategy=endgame_strategy,
                verified_steps=verified_steps,
                blocker_gaps=blocker_gaps,
                attack_points=attack_points,
                next_iteration_focus=next_iteration_focus or current.next_iteration_focus,
                ready_to_finalize=ready_to_finalize,
                unresolved_points=unresolved_points,
                sanity_findings=list(current.sanity_findings or []),
                confidence=confidence,
            )
            progress_score, quality_flags = _proof_progress_snapshot(candidate_state)
            return candidate_state.model_copy(
                update={
                    "progress_score": progress_score,
                    "quality_flags": quality_flags,
                }
            )

        async def initialize(step_input: StepInput) -> StepOutput:
            del step_input
            if not dict(local_state.get("proof_frame") or {}):
                current = _current_proof_state()
                fallback_frame = ProofProblemFrame(
                    understanding_status="imperfectly_understood",
                    goal_type=str(current.goal_type or "").strip() or "undetermined",
                    known_conditions=[],
                    unknown_targets=[
                        "Classify the theorem target and identify the decisive reduction."
                    ],
                    missing_conditions=list(current.blocker_gaps or [])[:4]
                    or ["Need a canonical reduction, explicit obligations, and an endgame."],
                    simplest_first_object=str(current.simplest_anchor or "").strip(),
                    canonical_representation=str(current.canonical_reduction or "").strip(),
                    dependency_order=[str(current.next_iteration_focus or "").strip()]
                    if str(current.next_iteration_focus or "").strip()
                    else [],
                    candidate_local_certificate="",
                    candidate_regime_split="",
                    candidate_endgame=str(current.endgame_strategy or "").strip(),
                )
                frame_prompt = "\n".join(
                    [
                        "Frame the theorem before attempting the proof.",
                        "Classify the problem as perfectly understood or imperfectly understood.",
                        "List known conditions, unknown targets, and missing conditions.",
                        "State the simplest object or relation the team can know first.",
                        "Name the representation in which the problem becomes easiest to manipulate.",
                        "Order subproblems from simplest to most complex.",
                        "If natural, identify a local certificate, a useful regime split, and an endgame template.",
                        "Return only structured output.",
                        "",
                        "Theorem/problem:",
                        latest_user_text,
                    ]
                )
                frame = await self._run_tool_program_phase(
                    phase_name="proof_problem_frame",
                    schema=ProofProblemFrame,
                    prompt=frame_prompt,
                    fallback=fallback_frame,
                    session_state=local_state,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    reasoning_mode="deep",
                    reasoning_effort_override="high",
                    max_runtime_seconds=_remaining_proof_budget(step_cap, reserve_seconds=30),
                    debug=debug,
                )
                local_state["proof_frame"] = frame.model_dump(mode="json")
                seeded = current.model_copy(
                    update={
                        "goal_type": (
                            str(frame.goal_type or "").strip()
                            if str(current.goal_type or "").strip() in {"", "undetermined"}
                            and str(frame.goal_type or "").strip()
                            else current.goal_type
                        ),
                        "simplest_anchor": current.simplest_anchor
                        or str(frame.simplest_first_object or "").strip(),
                        "canonical_reduction": current.canonical_reduction
                        or str(frame.canonical_representation or "").strip(),
                        "proof_outline": list(current.proof_outline or [])
                        or list(frame.dependency_order or [])[:8],
                        "case_splits": list(current.case_splits or [])
                        or (
                            [str(frame.candidate_regime_split or "").strip()]
                            if str(frame.candidate_regime_split or "").strip()
                            else []
                        ),
                        "endgame_strategy": current.endgame_strategy
                        or str(frame.candidate_endgame or "").strip(),
                        "next_iteration_focus": current.next_iteration_focus
                        or next(
                            (
                                str(item or "").strip()
                                for item in list(frame.dependency_order or [])
                                if str(item or "").strip()
                            ),
                            current.next_iteration_focus,
                        ),
                        "unresolved_points": _merge_unique(
                            [
                                *list(current.unresolved_points or []),
                                *list(frame.missing_conditions or []),
                            ],
                            limit=8,
                        ),
                    }
                )
                seeded_progress_score, seeded_quality_flags = _proof_progress_snapshot(seeded)
                local_state["proof_state"] = seeded.model_copy(
                    update={
                        "progress_score": seeded_progress_score,
                        "quality_flags": seeded_quality_flags,
                    }
                ).model_dump(mode="json")
                _checkpoint_proof_state()
            return StepOutput(content=dict(local_state.get("proof_state") or {}))

        def _make_proof_role_step(role: str) -> Step:
            async def _executor(step_input: StepInput) -> StepOutput:
                del step_input
                current = _current_proof_state()
                role_prompt = "\n".join(
                    [
                        PROOF_WORKFLOW_ROLE_INSTRUCTIONS[role],
                        "Return only structured output.",
                        "The theorem/problem:",
                        latest_user_text,
                        "",
                        "Proof problem frame:",
                        json.dumps(
                            _current_proof_frame().model_dump(mode="json"),
                            ensure_ascii=False,
                            indent=2,
                        ),
                        "",
                        "Current proof state:",
                        json.dumps(current.model_dump(mode="json"), ensure_ascii=False, indent=2),
                        "",
                        "Relevant context:",
                        json.dumps(
                            {
                                "memory_messages": memory_messages[:6],
                                "knowledge_messages": knowledge_messages[:6],
                                "iteration": int(local_state.get("iteration") or 0) + 1,
                                "expectations": {
                                    "track_goal_type": True,
                                    "track_reduction": True,
                                    "track_obligations": True,
                                    "track_case_splits": True,
                                    "track_endgame": True,
                                },
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                    ]
                )
                started = time.monotonic()
                memo = await self._run_tool_program_phase(
                    phase_name=f"{role}_proof_round_{int(local_state.get('iteration') or 0) + 1}",
                    schema=ProofRoleMemo,
                    prompt=role_prompt,
                    fallback=_proof_role_fallback(role),
                    session_state=local_state,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    reasoning_mode="deep",
                    reasoning_effort_override="high",
                    max_runtime_seconds=_remaining_proof_budget(step_cap, reserve_seconds=30),
                    debug=debug,
                )
                phase_timings[f"{role}_round_{int(local_state.get('iteration') or 0) + 1}"] = round(
                    time.monotonic() - started, 3
                )
                return StepOutput(content=memo.model_dump(mode="json"))

            return Step(name=role, executor=_executor)

        async def collect_iteration(step_input: StepInput) -> StepOutput:
            current = _current_proof_state()
            memos: list[ProofRoleMemo] = []
            for role in PROOF_WORKFLOW_ROLE_INSTRUCTIONS:
                payload = step_input.get_step_content(role)
                if isinstance(payload, dict):
                    memos.append(ProofRoleMemo.model_validate(payload))
            started = time.monotonic()
            fallback_state = _proof_state_fallback(memos)
            integration_prompt = "\n".join(
                [
                    "Integrate the parallel proof-role outputs into the current best proof state.",
                    "Set ready_to_finalize to true only if the theorem is now supported by a rigorous proof or rigorous disproof with no blocker gaps.",
                    "Keep the smallest viable proof skeleton. Do not inflate the outline with decorative commentary.",
                    "",
                    "Theorem/problem:",
                    latest_user_text,
                    "",
                    "Previous proof state:",
                    json.dumps(current.model_dump(mode="json"), ensure_ascii=False, indent=2),
                    "",
                    "Role outputs:",
                    json.dumps(
                        [memo.model_dump(mode="json") for memo in memos],
                        ensure_ascii=False,
                        indent=2,
                    ),
                ]
            )
            new_state = await self._run_tool_program_phase(
                phase_name=f"proof_integrate_round_{int(local_state.get('iteration') or 0) + 1}",
                schema=ProofWorkflowState,
                prompt=integration_prompt,
                fallback=fallback_state,
                session_state=local_state,
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                reasoning_mode="deep",
                reasoning_effort_override="high",
                max_runtime_seconds=_remaining_proof_budget(step_cap, reserve_seconds=30),
                debug=debug,
            )
            sanity_source_texts = [
                str(new_state.canonical_reduction or "").strip(),
                str(new_state.simplest_anchor or "").strip(),
                str(new_state.endgame_strategy or "").strip(),
                *list(new_state.proof_outline or []),
                *list(new_state.verified_steps or []),
                *list(new_state.case_splits or []),
                *list(new_state.attack_points or []),
                *[
                    piece
                    for memo in memos
                    for piece in [
                        str(memo.summary or "").strip(),
                        str(memo.simplest_anchor or "").strip(),
                        str(memo.canonical_reduction or "").strip(),
                        str(memo.endgame_strategy or "").strip(),
                        *list(memo.proposed_lemmas or []),
                        *list(memo.verified_steps or []),
                        *list(memo.attack_points or []),
                        *list(memo.next_actions or []),
                    ]
                    if str(piece or "").strip()
                ],
            ]
            new_state = _apply_sanity_findings(new_state, source_texts=sanity_source_texts)
            progress_score, quality_flags = _proof_progress_snapshot(new_state)
            new_state = new_state.model_copy(
                update={
                    "progress_score": progress_score,
                    "quality_flags": quality_flags,
                }
            )
            phase_timings[f"proof_integrate_round_{int(local_state.get('iteration') or 0) + 1}"] = (
                round(time.monotonic() - started, 3)
            )
            blocker_signature = " || ".join(list(new_state.blocker_gaps or [])[:6])
            previous_signature = str(local_state.get("last_blocker_signature") or "")
            previous_verified = set(
                str(item or "").strip() for item in list(current.verified_steps or [])
            )
            current_verified = set(
                str(item or "").strip() for item in list(new_state.verified_steps or [])
            )
            previous_resolved = set(
                str(item or "").strip() for item in list(current.resolved_obligations or [])
            )
            current_resolved = set(
                str(item or "").strip() for item in list(new_state.resolved_obligations or [])
            )
            previous_progress = float(current.progress_score or 0.0)
            current_progress = float(new_state.progress_score or 0.0)
            no_structural_progress = (
                current_verified.issubset(previous_verified)
                and current_resolved.issubset(previous_resolved)
                and current_progress <= previous_progress + 0.02
            )
            if (
                blocker_signature
                and blocker_signature == previous_signature
                and no_structural_progress
            ):
                local_state["stagnant_iterations"] = (
                    int(local_state.get("stagnant_iterations") or 0) + 1
                )
            else:
                local_state["stagnant_iterations"] = 0
            local_state["last_blocker_signature"] = blocker_signature
            local_state["proof_state"] = new_state.model_dump(mode="json")
            local_state["iteration"] = int(local_state.get("iteration") or 0) + 1
            summaries = list(local_state.get("iteration_summaries") or [])
            summaries.append(
                {
                    "iteration": int(local_state.get("iteration") or 0),
                    "goal_type": new_state.goal_type,
                    "direction": new_state.current_direction,
                    "simplest_anchor": new_state.simplest_anchor,
                    "canonical_reduction": new_state.canonical_reduction,
                    "resolved_obligations": list(new_state.resolved_obligations or [])[:4],
                    "verified_steps": list(new_state.verified_steps or [])[:6],
                    "blocker_gaps": list(new_state.blocker_gaps or [])[:6],
                    "sanity_findings": list(new_state.sanity_findings or [])[:4],
                    "case_splits": list(new_state.case_splits or [])[:4],
                    "endgame_strategy": new_state.endgame_strategy,
                    "next_focus": new_state.next_iteration_focus,
                    "ready_to_finalize": bool(new_state.ready_to_finalize),
                    "progress_score": float(new_state.progress_score or 0.0),
                }
            )
            local_state["iteration_summaries"] = summaries[-10:]
            _checkpoint_proof_state()
            return StepOutput(content=new_state.model_dump(mode="json"))

        async def repair_iteration(step_input: StepInput) -> StepOutput:
            del step_input
            current = _current_proof_state()
            needs_repair = bool(current.blocker_gaps) and (
                int(local_state.get("stagnant_iterations") or 0) >= 1
                or (int(local_state.get("iteration") or 0) >= 2 and not current.ready_to_finalize)
            )
            if not needs_repair:
                return StepOutput(content=current.model_dump(mode="json"))

            async def _run_repair_role(role: str) -> ProofRoleMemo:
                role_prompt = "\n".join(
                    [
                        PROOF_REPAIR_ROLE_INSTRUCTIONS[role],
                        "Return only structured output.",
                        "Theorem/problem:",
                        latest_user_text,
                        "",
                        "Proof problem frame:",
                        json.dumps(
                            _current_proof_frame().model_dump(mode="json"),
                            ensure_ascii=False,
                            indent=2,
                        ),
                        "",
                        "Current proof state:",
                        json.dumps(current.model_dump(mode="json"), ensure_ascii=False, indent=2),
                        "",
                        "Recent iteration summaries:",
                        json.dumps(
                            list(local_state.get("iteration_summaries") or [])[-6:],
                            ensure_ascii=False,
                            indent=2,
                        ),
                    ]
                )
                started = time.monotonic()
                memo = await self._run_tool_program_phase(
                    phase_name=f"{role}_repair_round_{int(local_state.get('iteration') or 0)}",
                    schema=ProofRoleMemo,
                    prompt=role_prompt,
                    fallback=_proof_role_fallback(role),
                    session_state=local_state,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    reasoning_mode="deep",
                    reasoning_effort_override="high",
                    max_runtime_seconds=_remaining_proof_budget(step_cap, reserve_seconds=30),
                    debug=debug,
                )
                phase_timings[f"{role}_repair_round_{int(local_state.get('iteration') or 0)}"] = (
                    round(time.monotonic() - started, 3)
                )
                return memo

            repair_memos = list(
                await asyncio.gather(
                    *[_run_repair_role(role) for role in PROOF_REPAIR_ROLE_INSTRUCTIONS],
                    return_exceptions=False,
                )
            )
            fallback_state = _proof_state_fallback(repair_memos)
            integration_prompt = "\n".join(
                [
                    "The proof workflow is stalled or still carries blocker-level gaps.",
                    "Integrate the repair memos into the current proof state.",
                    "Prefer the smallest structural change that lowers blocker count, sharpens the reduction, or clarifies the endgame.",
                    "Do not discard verified steps unless the repair memos show that they are invalid.",
                    "",
                    "Theorem/problem:",
                    latest_user_text,
                    "",
                    "Current proof state:",
                    json.dumps(current.model_dump(mode="json"), ensure_ascii=False, indent=2),
                    "",
                    "Repair memos:",
                    json.dumps(
                        [memo.model_dump(mode="json") for memo in repair_memos],
                        ensure_ascii=False,
                        indent=2,
                    ),
                ]
            )
            started = time.monotonic()
            repaired_state = await self._run_tool_program_phase(
                phase_name=f"proof_repair_integrate_round_{int(local_state.get('iteration') or 0)}",
                schema=ProofWorkflowState,
                prompt=integration_prompt,
                fallback=fallback_state,
                session_state=local_state,
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                reasoning_mode="deep",
                reasoning_effort_override="high",
                max_runtime_seconds=_remaining_proof_budget(step_cap, reserve_seconds=30),
                debug=debug,
            )
            repair_source_texts = [
                str(repaired_state.canonical_reduction or "").strip(),
                str(repaired_state.simplest_anchor or "").strip(),
                str(repaired_state.endgame_strategy or "").strip(),
                *list(repaired_state.proof_outline or []),
                *list(repaired_state.verified_steps or []),
                *list(repaired_state.case_splits or []),
                *list(repaired_state.attack_points or []),
                *[
                    piece
                    for memo in repair_memos
                    for piece in [
                        str(memo.summary or "").strip(),
                        str(memo.simplest_anchor or "").strip(),
                        str(memo.canonical_reduction or "").strip(),
                        str(memo.endgame_strategy or "").strip(),
                        *list(memo.proposed_lemmas or []),
                        *list(memo.verified_steps or []),
                        *list(memo.attack_points or []),
                        *list(memo.next_actions or []),
                    ]
                    if str(piece or "").strip()
                ],
            ]
            repaired_state = _apply_sanity_findings(
                repaired_state, source_texts=repair_source_texts
            )
            validation_roles = ("proof_checker", "proof_skeptic", "proof_obligation_closer")

            async def _run_repair_validation_role(role: str) -> ProofRoleMemo:
                validation_prompt = "\n".join(
                    [
                        PROOF_WORKFLOW_ROLE_INSTRUCTIONS[role],
                        "You are validating a repaired proof state. Do not invent a wholly new proof path.",
                        "Only assess whether the repaired state truly closes the key obligations, and keep blocker gaps open if the bridge is still missing.",
                        "Return only structured output.",
                        "Theorem/problem:",
                        latest_user_text,
                        "",
                        "Repaired proof state to audit:",
                        json.dumps(
                            repaired_state.model_dump(mode="json"), ensure_ascii=False, indent=2
                        ),
                        "",
                        "Recent repair memos:",
                        json.dumps(
                            [memo.model_dump(mode="json") for memo in repair_memos],
                            ensure_ascii=False,
                            indent=2,
                        ),
                    ]
                )
                return await self._run_tool_program_phase(
                    phase_name=f"{role}_repair_validate_round_{int(local_state.get('iteration') or 0)}",
                    schema=ProofRoleMemo,
                    prompt=validation_prompt,
                    fallback=_proof_role_fallback(role),
                    session_state=local_state,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    reasoning_mode="deep",
                    reasoning_effort_override="high",
                    max_runtime_seconds=_remaining_proof_budget(step_cap, reserve_seconds=30),
                    debug=debug,
                )

            validation_started = time.monotonic()
            validation_memos = list(
                await asyncio.gather(
                    *[_run_repair_validation_role(role) for role in validation_roles],
                    return_exceptions=False,
                )
            )
            phase_timings[
                f"proof_repair_validate_round_{int(local_state.get('iteration') or 0)}"
            ] = round(time.monotonic() - validation_started, 3)
            validation_obligations = [
                obligation
                for memo in validation_memos
                for obligation in list(memo.obligations or [])
                if str(obligation.status or "").strip().lower() not in {"verified", "resolved"}
            ]
            validation_blockers = [
                item
                for memo in validation_memos
                for item in list(memo.blocker_gaps or [])
                if str(item or "").strip()
            ]
            validation_attacks = [
                item
                for memo in validation_memos
                for item in list(memo.attack_points or [])
                if str(item or "").strip()
            ]
            validation_next_actions = [
                item
                for memo in validation_memos
                for item in list(memo.next_actions or [])
                if str(item or "").strip()
            ]
            repaired_state = repaired_state.model_copy(
                update={
                    "obligations": _merge_obligations(
                        [*list(repaired_state.obligations or []), *validation_obligations],
                        limit=10,
                    ),
                    "blocker_gaps": _merge_unique(
                        [*list(repaired_state.blocker_gaps or []), *validation_blockers],
                        limit=8,
                    ),
                    "attack_points": _merge_unique(
                        [*list(repaired_state.attack_points or []), *validation_attacks],
                        limit=8,
                    ),
                    "unresolved_points": _merge_unique(
                        [
                            *list(repaired_state.unresolved_points or []),
                            *validation_blockers,
                            *validation_next_actions,
                        ],
                        limit=8,
                    ),
                    "ready_to_finalize": bool(repaired_state.ready_to_finalize)
                    and not validation_blockers,
                    "confidence": (
                        "low" if validation_blockers else str(repaired_state.confidence or "medium")
                    ),
                }
            )
            validation_source_texts = [
                *repair_source_texts,
                *[
                    piece
                    for memo in validation_memos
                    for piece in [
                        str(memo.summary or "").strip(),
                        str(memo.simplest_anchor or "").strip(),
                        str(memo.canonical_reduction or "").strip(),
                        str(memo.endgame_strategy or "").strip(),
                        *list(memo.proposed_lemmas or []),
                        *list(memo.verified_steps or []),
                        *list(memo.attack_points or []),
                        *list(memo.next_actions or []),
                        *list(memo.blocker_gaps or []),
                    ]
                    if str(piece or "").strip()
                ],
            ]
            repaired_state = _apply_sanity_findings(
                repaired_state, source_texts=validation_source_texts
            )
            progress_score, quality_flags = _proof_progress_snapshot(repaired_state)
            repaired_state = repaired_state.model_copy(
                update={
                    "progress_score": progress_score,
                    "quality_flags": quality_flags,
                }
            )
            phase_timings[
                f"proof_repair_integrate_round_{int(local_state.get('iteration') or 0)}"
            ] = round(time.monotonic() - started, 3)
            previous_signature = str(local_state.get("last_blocker_signature") or "")
            repaired_signature = " || ".join(list(repaired_state.blocker_gaps or [])[:6])
            previous_verified = set(
                str(item or "").strip() for item in list(current.verified_steps or [])
            )
            repaired_verified = set(
                str(item or "").strip() for item in list(repaired_state.verified_steps or [])
            )
            previous_resolved = set(
                str(item or "").strip() for item in list(current.resolved_obligations or [])
            )
            repaired_resolved = set(
                str(item or "").strip() for item in list(repaired_state.resolved_obligations or [])
            )
            if (
                repaired_signature != previous_signature
                or repaired_verified != previous_verified
                or repaired_resolved != previous_resolved
                or float(repaired_state.progress_score or 0.0)
                > float(current.progress_score or 0.0) + 0.02
            ):
                local_state["stagnant_iterations"] = 0
            local_state["last_blocker_signature"] = repaired_signature
            local_state["proof_state"] = repaired_state.model_dump(mode="json")
            summaries = list(local_state.get("iteration_summaries") or [])
            summaries.append(
                {
                    "iteration": int(local_state.get("iteration") or 0),
                    "repair_applied": True,
                    "goal_type": repaired_state.goal_type,
                    "canonical_reduction": repaired_state.canonical_reduction,
                    "resolved_obligations": list(repaired_state.resolved_obligations or [])[:4],
                    "blocker_gaps": list(repaired_state.blocker_gaps or [])[:6],
                    "sanity_findings": list(repaired_state.sanity_findings or [])[:4],
                    "case_splits": list(repaired_state.case_splits or [])[:4],
                    "endgame_strategy": repaired_state.endgame_strategy,
                    "next_focus": repaired_state.next_iteration_focus,
                    "progress_score": float(repaired_state.progress_score or 0.0),
                }
            )
            local_state["iteration_summaries"] = summaries[-10:]
            _checkpoint_proof_state()
            return StepOutput(content=repaired_state.model_dump(mode="json"))

        def _loop_end_condition(step_input: StepInput) -> bool:
            del step_input
            state = _current_proof_state()
            if bool(state.ready_to_finalize):
                return True
            if int(local_state.get("stagnant_iterations") or 0) >= int(stagnation_limit):
                return True
            return (workflow_deadline - time.monotonic()) <= max(20.0, float(step_cap))

        async def synthesize(step_input: StepInput) -> StepOutput:
            del step_input
            current = _current_proof_state()
            fallback = ProofWorkflowSynthesis(
                response_text=(
                    "I could not yet justify a fully rigorous proof. "
                    "The strongest current structure is summarized below.\n\n"
                    + (
                        f"Current reduction: {current.canonical_reduction}\n\n"
                        if str(current.canonical_reduction or "").strip()
                        else ""
                    )
                    + "\n".join(
                        f"- {item}"
                        for item in list(current.verified_steps or current.proof_outline or [])[:8]
                    )
                    + (
                        "\n\nRemaining gaps:\n"
                        + "\n".join(
                            f"- {item}"
                            for item in list(
                                current.blocker_gaps or current.unresolved_points or []
                            )[:6]
                        )
                        if list(current.blocker_gaps or current.unresolved_points or [])
                        else ""
                    )
                ),
                proof_status="proved" if current.ready_to_finalize else "partial",
                canonical_reduction=current.canonical_reduction,
                verified_steps=list(current.verified_steps or [])[:10],
                resolved_obligations=list(current.resolved_obligations or [])[:8],
                unresolved_points=list(current.blocker_gaps or current.unresolved_points or [])[:8],
                confidence=current.confidence,
                progress_score=float(current.progress_score or 0.0),
            )
            synthesis_prompt = "\n".join(
                [
                    "Write the strongest rigorous proof draft currently supported by the proof workflow state.",
                    "If the state is not ready to finalize, say so plainly and identify the remaining blockers.",
                    "Do not claim the theorem is solved unless blocker_gaps is empty, all obligations are closed, and ready_to_finalize is true.",
                    "State the main reduction clearly if one has been found.",
                    "",
                    "Theorem/problem:",
                    latest_user_text,
                    "",
                    "Proof problem frame:",
                    json.dumps(
                        _current_proof_frame().model_dump(mode="json"), ensure_ascii=False, indent=2
                    ),
                    "",
                    "Proof workflow state:",
                    json.dumps(current.model_dump(mode="json"), ensure_ascii=False, indent=2),
                    "",
                    "Iteration summaries:",
                    json.dumps(
                        list(local_state.get("iteration_summaries") or []),
                        ensure_ascii=False,
                        indent=2,
                    ),
                ]
            )
            started = time.monotonic()
            synthesis = await self._run_tool_program_phase(
                phase_name="proof_workflow_synthesis",
                schema=ProofWorkflowSynthesis,
                prompt=synthesis_prompt,
                fallback=fallback,
                session_state=local_state,
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                reasoning_mode="deep",
                reasoning_effort_override="high",
                max_runtime_seconds=_remaining_proof_budget(max(step_cap, 45)),
                debug=debug,
            )
            phase_timings["proof_workflow_synthesis"] = round(time.monotonic() - started, 3)
            local_state["proof_synthesis"] = synthesis.model_dump(mode="json")
            return StepOutput(content=synthesis.model_dump(mode="json"))

        workflow = Workflow(
            name="pro_mode_proof_workflow",
            description="Iterative proof workflow for rigorous self-contained theorem proving.",
            steps=[
                Step(name="proof_initialize", executor=initialize),
                Loop(
                    name="proof_loop",
                    max_iterations=max_iterations,
                    end_condition=_loop_end_condition,
                    steps=[
                        Parallel(
                            "proof_parallel_roles",
                            _make_proof_role_step("proof_planner"),
                            _make_proof_role_step("proof_constructor"),
                            _make_proof_role_step("proof_checker"),
                            _make_proof_role_step("proof_skeptic"),
                        ),
                        Step(name="proof_collect", executor=collect_iteration),
                        Step(name="proof_repair", executor=repair_iteration),
                    ],
                ),
                Step(name="proof_synthesize", executor=synthesize),
            ],
            stream=False,
            telemetry=False,
            store_events=False,
            debug_mode=bool(debug),
            session_state={"proof_workflow_state": {}},
        )
        try:
            output = await workflow.arun(
                input={"messages": [{"role": "user", "content": latest_user_text}]},
                user_id=user_id,
                run_id=f"{str(run_id or '').strip()}::proof_workflow" if run_id else None,
                session_id=self._scope_session_id(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    run_id=f"{str(run_id or '').strip()}::proof_workflow"
                    if run_id
                    else "proof_workflow",
                ),
            )
            content = dict(getattr(output, "content", {}) or {})
            synthesis = ProofWorkflowSynthesis.model_validate(content or {})
            return ProModeWorkflowResult(
                response_text=str(synthesis.response_text or "").strip(),
                metadata={
                    "pro_mode": {
                        "execution_path": "proof_workflow",
                        "runtime_status": "completed",
                        "proof_workflow": {
                            "iterations": int(local_state.get("iteration") or 0),
                            "reasoning_effort": "high",
                            "proof_frame": dict(local_state.get("proof_frame") or {}),
                            "proof_state": dict(local_state.get("proof_state") or {}),
                            "iteration_summaries": list(
                                local_state.get("iteration_summaries") or []
                            ),
                            "compression_stats": dict(local_state.get("compression_stats") or {}),
                            "phase_timings": phase_timings,
                            "stagnant_iterations": int(local_state.get("stagnant_iterations") or 0),
                            "last_blocker_signature": str(
                                local_state.get("last_blocker_signature") or ""
                            ),
                            "proof_status": synthesis.proof_status,
                        },
                    }
                },
                runtime_status="completed",
            )
        except Exception as exc:
            current = _current_proof_state()
            return ProModeWorkflowResult(
                response_text="",
                metadata={
                    "pro_mode": {
                        "execution_path": "proof_workflow",
                        "runtime_status": "failed",
                        "proof_workflow": {
                            "iterations": int(local_state.get("iteration") or 0),
                            "reasoning_effort": "high",
                            "proof_frame": dict(local_state.get("proof_frame") or {}),
                            "proof_state": current.model_dump(mode="json"),
                            "iteration_summaries": list(
                                local_state.get("iteration_summaries") or []
                            ),
                            "compression_stats": dict(local_state.get("compression_stats") or {}),
                            "phase_timings": phase_timings,
                            "stagnant_iterations": int(local_state.get("stagnant_iterations") or 0),
                            "last_blocker_signature": str(
                                local_state.get("last_blocker_signature") or ""
                            ),
                        },
                    }
                },
                runtime_status="failed",
                runtime_error=str(exc or exc.__class__.__name__),
            )

    async def _run_validated_numeric_workflow(
        self,
        *,
        latest_user_text: str,
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        max_runtime_seconds: int,
        reasoning_mode: str | None,
        debug: bool | None,
    ) -> dict[str, Any] | None:
        plan_prompt = "\n".join(
            [
                "Prepare a deterministic calculation plan for a closed-form scientific question.",
                "Do a short assumption audit before choosing formulas.",
                "Return only structured output.",
                "For relativistic beam or collider questions, if the prompt asks for `energy` and does not say `kinetic`, treat the primary target as total relativistic energy.",
                "If a nucleus or ion is specified by element identity plus neutron count and no isotope mass table is supplied, infer the mass number from the composition and use the nucleon approximation `m ≈ A * m_n` as the primary back-of-the-envelope convention.",
                "For textbook relativistic beam questions without an exact isotope mass, use the rounded nucleon rest-energy constant `939.5 MeV` per nucleon for the primary convention rather than a higher-fidelity tabulated nucleon mass.",
                "If another convention is plausible and materially changes the result, include one alternative cross-check expression.",
                "Every `expression` field must be directly executable by `numpy_calculator` as plain Python numeric math.",
                "Use only numeric literals, ASCII operators, parentheses, and supported math functions such as `sqrt`.",
                "Do not include equal signs, variable assignments, prose, units, Greek letters, or symbolic placeholders in `expression`.",
                "Inline constants numerically. Example expressions: `1/sqrt(1-0.96**2)` and `(1/sqrt(1-0.96**2)) * (6 * 939.5)`.",
                "",
                f"Question: {latest_user_text}",
            ]
        )
        plan_fallback = ValidatedNumericPlan(
            target_quantity="Closed-form numeric quantity requested by the user.",
            target_units="",
            primary_convention="Use the governing closed-form equation directly.",
            calculation_expressions=[],
            alternative_conventions=[],
            final_expression_label="",
        )
        plan = await self._run_tool_program_phase(
            phase_name="validated_numeric_plan",
            schema=ValidatedNumericPlan,
            prompt=plan_prompt,
            fallback=plan_fallback,
            tools=None,
            session_state={"validated_numeric": {}},
            conversation_id=conversation_id,
            run_id=run_id,
            user_id=user_id,
            reasoning_mode=reasoning_mode or "deep",
            reasoning_effort_override="high",
            use_reasoning_agent=False,
            max_runtime_seconds=max(30, min(max_runtime_seconds, 90)),
            debug=debug,
        )
        calculations = [
            item
            for item in list(plan.calculation_expressions or [])
            if str(item.expression or "").strip()
        ][:4]
        if not calculations:
            return None

        tool_invocations: list[dict[str, Any]] = []
        rendered_results: list[dict[str, Any]] = []
        used_textbook_rounding = False
        for item in calculations:
            original_expression = str(item.expression or "").strip()
            expression = self._normalize_validated_numeric_expression(
                original_expression,
                user_text=latest_user_text,
            )
            if expression != original_expression:
                used_textbook_rounding = True
            raw_result = numpy_calculator(expression=expression, variables={})
            tool_invocations.append(
                {
                    "tool": "numpy_calculator",
                    "status": "completed" if bool(raw_result.get("success")) else "failed",
                    "args": {"expression": expression, "variables": {}},
                    "output_envelope": dict(raw_result),
                    "output_summary": {"success": bool(raw_result.get("success"))},
                    "output_preview": json.dumps(raw_result, ensure_ascii=False),
                }
            )
            rendered_results.append(
                {
                    "label": str(item.label or "").strip(),
                    "purpose": str(item.purpose or "").strip(),
                    "expression": expression,
                    "original_expression": original_expression,
                    "success": bool(raw_result.get("success")),
                    "formatted_result": str(raw_result.get("formatted_result") or "").strip(),
                    "result": raw_result.get("result"),
                }
            )
        if not any(bool(item.get("success")) for item in rendered_results):
            return None

        primary_label = str(plan.final_expression_label or calculations[0].label or "").strip()
        fallback_primary = next(
            (item for item in rendered_results if bool(item.get("success"))), rendered_results[0]
        )
        fallback_units = str(plan.target_units or "").strip()
        fallback_text = (
            f"{str(plan.target_quantity or 'Computed result').strip()}: "
            f"{str(fallback_primary.get('formatted_result') or fallback_primary.get('result') or '').strip()}"
            f"{(' ' + fallback_units) if fallback_units else ''}."
        )
        synthesis_prompt = "\n".join(
            [
                "Write the final scientist-facing answer from the deterministic numeric results below.",
                "State the final answer explicitly near the beginning.",
                "Use the primary convention chosen in the plan unless the results clearly show it is inconsistent with the prompt.",
                "If an alternative convention differs materially, mention it briefly as an alternative rather than replacing the primary answer silently.",
                (
                    "If the primary convention uses the rounded textbook nucleon rest-energy constant `939.5 MeV` per nucleon, "
                    "report the final GeV answer using conventional textbook rounding rather than spurious extra digits."
                    if used_textbook_rounding
                    else "Match the reported precision to the constants used in the calculation."
                ),
                "Do not invent constants or calculations beyond the provided results.",
                "",
                f"Question: {latest_user_text}",
                "",
                "Validated numeric plan JSON:",
                plan.model_dump_json(indent=2),
                "",
                "Calculator results JSON:",
                json.dumps(rendered_results, ensure_ascii=False, indent=2),
            ]
        )
        synthesis = await self._run_tool_program_phase(
            phase_name="validated_numeric_synthesis",
            schema=ValidatedNumericSynthesis,
            prompt=synthesis_prompt,
            fallback=ValidatedNumericSynthesis(
                response_text=fallback_text,
                primary_result_label=primary_label,
                alternative_result_labels=[],
            ),
            tools=None,
            session_state={"validated_numeric": {"plan": plan.model_dump(mode="json")}},
            conversation_id=conversation_id,
            run_id=run_id,
            user_id=user_id,
            reasoning_mode=reasoning_mode or "deep",
            reasoning_effort_override="high",
            use_reasoning_agent=False,
            max_runtime_seconds=max(30, min(max_runtime_seconds, 90)),
            debug=debug,
        )
        return {
            "response_text": str(synthesis.response_text or fallback_text).strip(),
            "tool_invocations": tool_invocations,
            "metadata": {
                "tool_invocations": tool_invocations,
                "validated_numeric": {
                    "plan": plan.model_dump(mode="json"),
                    "results": rendered_results,
                    "used_textbook_rounding": used_textbook_rounding,
                    "primary_result_label": str(
                        synthesis.primary_result_label or primary_label
                    ).strip(),
                    "alternative_result_labels": list(synthesis.alternative_result_labels or []),
                },
            },
            "model": self.model,
            "selected_domains": ["core"],
            "runtime_status": "completed",
            "runtime_error": None,
        }

    async def _run_pro_mode_tool_workflow(
        self,
        *,
        messages: list[dict[str, Any]],
        latest_user_text: str,
        uploaded_files: list[str],
        max_tool_calls: int,
        max_runtime_seconds: int,
        reasoning_mode: str | None,
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        event_callback: Callable[[dict[str, Any]], None] | None,
        selected_tool_names: list[str],
        tool_plan_category: str | None,
        strict_tool_validation: bool,
        selection_context: dict[str, Any] | None,
        knowledge_context: dict[str, Any] | None,
        shared_context: dict[str, Any] | None,
        conversation_state_seed: dict[str, Any] | None,
        debug: bool | None,
        allow_research_program: bool = False,
    ) -> dict[str, Any]:
        shared_context = dict(shared_context or {})
        prepared_messages = [
            *list(shared_context.get("memory_messages") or []),
            *list(shared_context.get("knowledge_messages") or []),
            *list(messages),
        ]
        initial_tool_names = self._normalize_selected_tool_names(selected_tool_names)
        inferred_tool_names = self._infer_pro_mode_tool_subset(
            messages=prepared_messages,
            uploaded_files=uploaded_files,
            selected_tool_names=initial_tool_names,
        )
        tool_plan: ProModeToolPlan | None = None
        normalized_plan_category = str(tool_plan_category or "").strip() or None
        if normalized_plan_category and initial_tool_names:
            tool_plan = ProModeToolPlan(
                category=normalized_plan_category,
                selected_tool_names=list(initial_tool_names),
                strict_validation=bool(strict_tool_validation),
                reason="Preserved the Pro Mode intake tool plan for execution.",
            )
        else:
            tool_plan = self._build_pro_mode_tool_plan(
                user_text=latest_user_text,
                uploaded_files=uploaded_files,
                selected_tool_names=initial_tool_names,
                selection_context=selection_context,
                inferred_tool_names=inferred_tool_names,
                prior_pro_mode_state=conversation_state_seed,
            )
        if (
            allow_research_program
            and tool_plan is not None
            and tool_plan.category == "research_program"
        ):
            return await self._run_pro_mode_research_program_workflow(
                messages=prepared_messages,
                latest_user_text=latest_user_text,
                uploaded_files=uploaded_files,
                max_runtime_seconds=max_runtime_seconds,
                reasoning_mode=reasoning_mode,
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                event_callback=event_callback,
                selected_tool_names=list(tool_plan.selected_tool_names or initial_tool_names),
                selection_context=selection_context,
                conversation_state_seed=conversation_state_seed,
                debug=debug,
            )
        if (
            tool_plan is not None
            and tool_plan.category == "validated_numeric"
            and self._normalize_selected_tool_names(tool_plan.selected_tool_names)
            == ["numpy_calculator"]
        ):
            validated_numeric_result = await self._run_validated_numeric_workflow(
                latest_user_text=latest_user_text,
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                max_runtime_seconds=max_runtime_seconds,
                reasoning_mode=reasoning_mode,
                debug=debug,
            )
            if validated_numeric_result is not None:
                validated_numeric_result["selected_tool_names"] = ["numpy_calculator"]
                validated_numeric_result.setdefault("metadata", {})
                validated_numeric_result["metadata"]["attempted_tool_sets"] = [["numpy_calculator"]]
                return validated_numeric_result
        if tool_plan is not None and tool_plan.strict_validation and tool_plan.selected_tool_names:
            required_tool_names = list(tool_plan.selected_tool_names)
            strict_validation = True
        else:
            required_tool_names, strict_validation = self._tool_workflow_required_tools(
                user_text=latest_user_text,
                uploaded_files=uploaded_files,
                selection_context=selection_context,
                selected_tool_names=initial_tool_names,
            )

        last_result: dict[str, Any] | None = None
        attempted_tool_sets: list[list[str]] = []
        candidate_tool_sets: list[list[str]] = [initial_tool_names]
        request_bisque_auth = get_request_bisque_auth()
        if strict_validation and required_tool_names and required_tool_names != initial_tool_names:
            candidate_tool_sets.append(required_tool_names)
        tool_prepared_messages = list(prepared_messages)
        if tool_plan is not None and tool_plan.category == "validated_numeric":
            numeric_system_message = self._validated_numeric_system_message(latest_user_text)
            if numeric_system_message:
                tool_prepared_messages = [
                    {"role": "system", "content": numeric_system_message},
                    *tool_prepared_messages,
                ]

        for attempt_index, effective_tool_names in enumerate(candidate_tool_sets[:2], start=1):
            effective_tool_names = self._normalize_selected_tool_names(effective_tool_names)
            attempted_tool_sets.append(list(effective_tool_names))
            nested_done: dict[str, Any] | None = None
            nested_run_id = (
                f"{str(run_id or '').strip()}::tool_workflow:{attempt_index}"
                if str(run_id or "").strip()
                else None
            )
            context_token = (
                set_request_bisque_auth(request_bisque_auth) if request_bisque_auth else None
            )
            try:
                async for event in self.stream(
                    messages=tool_prepared_messages,
                    uploaded_files=list(uploaded_files or []),
                    max_tool_calls=max_tool_calls,
                    max_runtime_seconds=max_runtime_seconds,
                    reasoning_mode=reasoning_mode,
                    conversation_id=conversation_id,
                    run_id=nested_run_id,
                    user_id=user_id,
                    event_callback=event_callback,
                    selected_tool_names=effective_tool_names,
                    workflow_hint=None,
                    selection_context=selection_context,
                    knowledge_context=knowledge_context,
                    memory_policy={"mode": "off"},
                    knowledge_scope={
                        "mode": "project_notebook",
                        "project_id": None,
                        "namespaces": [],
                        "include_curated_packs": False,
                        "include_uploads": False,
                        "include_project_notes": False,
                    },
                    debug=debug,
                ):
                    if str(event.get("event") or "").strip().lower() == "done":
                        nested_done = dict(event.get("data") or {})
            finally:
                if context_token is not None:
                    reset_request_bisque_auth(context_token)
            if not nested_done:
                last_result = {
                    "response_text": "I couldn't complete the requested tool workflow.",
                    "tool_invocations": [],
                    "metadata": {},
                    "model": self.model,
                    "selected_domains": ["core"],
                    "runtime_status": "error",
                    "runtime_error": "tool_workflow_missing_done_event",
                    "selected_tool_names": effective_tool_names,
                }
                continue
            nested_metadata = dict(nested_done.get("metadata") or {})
            last_result = {
                "response_text": str(nested_done.get("response_text") or "").strip(),
                "tool_invocations": list(nested_metadata.get("tool_invocations") or []),
                "metadata": nested_metadata,
                "model": str(nested_done.get("model") or self.model),
                "selected_domains": list(nested_done.get("selected_domains") or ["core"]),
                "runtime_status": str(
                    (nested_metadata.get("debug") or {}).get("runtime_status") or "completed"
                ),
                "runtime_error": str(
                    (nested_metadata.get("debug") or {}).get("runtime_error") or ""
                ).strip()
                or None,
                "selected_tool_names": effective_tool_names,
            }
            if self._tool_workflow_satisfied(
                tool_invocations=list(last_result.get("tool_invocations") or []),
                required_tool_names=required_tool_names,
                strict_validation=strict_validation,
            ):
                break
            if attempt_index < len(candidate_tool_sets):
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "event_type": "pro_mode.phase_started",
                        "phase": "tool_workflow",
                        "status": "started",
                        "message": "Retrying the tool workflow with a narrower required tool set.",
                        "payload": {
                            "attempt_index": attempt_index + 1,
                            "required_tool_names": list(required_tool_names),
                        },
                    },
                )
        if last_result is None:
            last_result = {
                "response_text": "I couldn't complete the requested tool workflow.",
                "tool_invocations": [],
                "metadata": {},
                "model": self.model,
                "selected_domains": ["core"],
                "runtime_status": "error",
                "runtime_error": "tool_workflow_no_attempt",
                "selected_tool_names": initial_tool_names,
            }
        if (
            strict_validation
            and required_tool_names
            and not self._tool_workflow_satisfied(
                tool_invocations=list(last_result.get("tool_invocations") or []),
                required_tool_names=required_tool_names,
                strict_validation=True,
            )
        ):
            existing_tool_invocations = list(last_result.get("tool_invocations") or [])
            missing_required_tool_names = self._missing_required_tool_names(
                tool_invocations=existing_tool_invocations,
                required_tool_names=required_tool_names,
            )
            fallback_actions = [
                ToolProgramAction(
                    tool_name=tool_name,
                    purpose=(
                        "Running the required tool directly because the initial model response "
                        "did not execute the requested action."
                    ),
                    args={},
                )
                for tool_name in missing_required_tool_names
                if str(tool_name or "").strip()
            ]
            if fallback_actions:
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "event_type": "pro_mode.phase_started",
                        "phase": "tool_workflow",
                        "status": "started",
                        "message": "Running the required tool directly because the earlier answer skipped it.",
                        "payload": {
                            "required_tool_names": list(required_tool_names),
                            "missing_required_tool_names": missing_required_tool_names,
                            "tool_plan_category": str(
                                tool_plan.category if tool_plan is not None else ""
                            ),
                        },
                    },
                )
                deterministic_invocations = await self._execute_tool_program_actions(
                    phase="tool_workflow",
                    phase_label="tool workflow",
                    actions=fallback_actions,
                    uploaded_files=uploaded_files,
                    latest_user_text=latest_user_text,
                    selection_context=selection_context,
                    event_callback=event_callback,
                    latest_result_refs_seed=self._latest_result_refs_from_tool_invocations(
                        existing_tool_invocations
                    ),
                    request_bisque_auth=request_bisque_auth,
                )
                if deterministic_invocations:
                    merged_tool_invocations = self._merge_tool_invocations(
                        existing_tool_invocations,
                        deterministic_invocations,
                    )
                    deterministic_satisfied = self._tool_workflow_satisfied(
                        tool_invocations=merged_tool_invocations,
                        required_tool_names=required_tool_names,
                        strict_validation=True,
                    )
                    deterministic_response_text = (
                        self._tool_invocation_fallback_text(merged_tool_invocations)
                        or self._tool_invocation_fallback_text(deterministic_invocations)
                        or str(last_result.get("response_text") or "").strip()
                        or "Completed the required tool workflow."
                    )
                    deterministic_metadata = dict(last_result.get("metadata") or {})
                    deterministic_metadata["tool_invocations"] = merged_tool_invocations
                    deterministic_tool_workflow_meta = dict(
                        deterministic_metadata.get("tool_workflow") or {}
                    )
                    deterministic_tool_workflow_meta["deterministic_required_tool_fallback"] = {
                        "required_tool_names": list(required_tool_names),
                        "missing_required_tool_names": missing_required_tool_names,
                        "tool_invocation_count": len(deterministic_invocations),
                    }
                    deterministic_metadata["tool_workflow"] = deterministic_tool_workflow_meta
                    last_result = {
                        **last_result,
                        "response_text": deterministic_response_text,
                        "tool_invocations": merged_tool_invocations,
                        "metadata": deterministic_metadata,
                        "runtime_status": "completed" if deterministic_satisfied else "error",
                        "runtime_error": None
                        if deterministic_satisfied
                        else "required_tool_execution_failed",
                    }
        if (
            allow_research_program
            and strict_validation
            and not self._tool_workflow_satisfied(
                tool_invocations=list(last_result.get("tool_invocations") or []),
                required_tool_names=required_tool_names,
                strict_validation=True,
            )
        ):
            self._emit_event(
                event_callback,
                {
                    "kind": "graph",
                    "event_type": "pro_mode.phase_started",
                    "phase": "research_program",
                    "status": "started",
                    "message": (
                        "Escalating to the iterative research workflow because the planned required "
                        "tool action did not execute."
                    ),
                    "payload": {
                        "required_tool_names": list(required_tool_names),
                        "tool_plan_category": str(
                            tool_plan.category if tool_plan is not None else ""
                        ),
                    },
                },
            )
            return await self._run_pro_mode_research_program_workflow(
                messages=prepared_messages,
                latest_user_text=latest_user_text,
                uploaded_files=uploaded_files,
                max_runtime_seconds=max_runtime_seconds,
                reasoning_mode=reasoning_mode,
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                event_callback=event_callback,
                selected_tool_names=self._research_program_tool_bundle(
                    user_text=latest_user_text,
                    uploaded_files=uploaded_files,
                    selection_context=selection_context,
                    prior_pro_mode_state=conversation_state_seed,
                ),
                selection_context=selection_context,
                conversation_state_seed=conversation_state_seed,
                debug=debug,
            )
        if allow_research_program and self._should_escalate_validated_tool_to_research_program(
            latest_user_text=latest_user_text,
            tool_plan=tool_plan,
            tool_invocations=list(last_result.get("tool_invocations") or []),
            response_text=str(last_result.get("response_text") or ""),
        ):
            self._emit_event(
                event_callback,
                {
                    "kind": "graph",
                    "event_type": "pro_mode.phase_started",
                    "phase": "research_program",
                    "status": "started",
                    "message": "Escalating from a narrow validated-tool pass to an iterative research workflow.",
                    "payload": {
                        "prior_tool_names": list(last_result.get("selected_tool_names") or []),
                        "tool_plan_category": str(
                            tool_plan.category if tool_plan is not None else ""
                        ),
                    },
                },
            )
            return await self._run_pro_mode_research_program_workflow(
                messages=prepared_messages,
                latest_user_text=latest_user_text,
                uploaded_files=uploaded_files,
                max_runtime_seconds=max_runtime_seconds,
                reasoning_mode=reasoning_mode,
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                event_callback=event_callback,
                selected_tool_names=self._research_program_tool_bundle(
                    user_text=latest_user_text,
                    uploaded_files=uploaded_files,
                    selection_context=selection_context,
                    prior_pro_mode_state=conversation_state_seed,
                ),
                selection_context=selection_context,
                debug=debug,
            )
        last_result["attempted_tool_sets"] = attempted_tool_sets
        return last_result

    @staticmethod
    def _research_program_tool_family(tool_name: str) -> str:
        normalized = str(tool_name or "").strip()
        for family, names in RESEARCH_PROGRAM_TOOL_BUNDLES.items():
            if normalized in names:
                return family
        return "analysis"

    @staticmethod
    def _research_program_requirements(
        *,
        latest_user_text: str,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
        prior_pro_mode_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        lowered = str(latest_user_text or "").strip().lower()
        selection_context = dict(selection_context or {})
        required_families: list[str] = []
        required_measurements: list[str] = []

        def _require(family: str) -> None:
            if family not in required_families:
                required_families.append(family)

        direct_image_analysis = bool(
            AgnoChatRuntime._has_direct_image_target(
                user_text=latest_user_text,
                uploaded_files=uploaded_files,
                selection_context=selection_context,
            )
            or re.search(
                r"\b(attached|uploaded|new image|new aerial image|this image|analy[sz]e it|analy[sz]e this|look at this|local image)\b",
                lowered,
            )
        )
        data_context = bool(
            selection_context.get("resource_uris")
            or selection_context.get("dataset_uris")
            or AgnoChatRuntime._is_prior_analysis_follow_up_request(
                latest_user_text,
                prior_state=prior_pro_mode_state,
            )
            or re.search(r"\b(dataset|bisque|resource|catalog)\b", lowered)
        )
        quantitative_request = bool(
            re.search(
                r"\b(count|average|mean|median|statistics|distribution|population|density|box size|annotation|bounding[- ]box|bbox)\b",
                lowered,
            )
        )
        object_pattern_request = bool(
            re.search(
                r"\b(prairie dog|burrow|object|bbox|bounding[- ]box|box size|detection|detect|class pattern|population)\b",
                lowered,
            )
        )
        acquisition_request = bool(
            re.search(
                r"\b(spectral|frequency|instability|active learning|hard sample|uncertain(?:ty)? ranking|acquisition)\b",
                lowered,
            )
        )
        report_like_request = bool(
            re.search(r"\b(report|ecological|summary|analysis|summarize|compare)\b", lowered)
        )
        plot_request = AgnoChatRuntime._is_plot_or_visual_analysis_request(latest_user_text)
        introspection_request = AgnoChatRuntime._is_structured_artifact_introspection_request(
            latest_user_text
        )
        if direct_image_analysis:
            _require("vision")
        if data_context:
            _require("catalog")
        if data_context and object_pattern_request:
            _require("vision")
        if acquisition_request and (direct_image_analysis or data_context):
            _require("acquisition")
        if (
            quantitative_request
            or report_like_request
            or direct_image_analysis
            or introspection_request
        ):
            _require("analysis")
        if plot_request or re.search(
            r"\b(code|python|script|parse|optimi[sz]e|benchmark|csv|json|xml)\b", lowered
        ):
            _require("code")
        if quantitative_request:
            required_measurements.append("quantitative_summary")
        if report_like_request:
            required_measurements.append("narrative_summary")
        if plot_request:
            required_measurements.append("plot_artifacts")
        return {
            "required_families": required_families,
            "required_measurements": required_measurements,
        }

    @staticmethod
    def _research_program_has_enough_evidence(
        *,
        evidence_summaries: list[str],
        executed_families: list[str],
        requirements: dict[str, Any],
    ) -> bool:
        evidence_blob = " ".join(str(item or "") for item in list(evidence_summaries or [])).lower()
        needed_families = {
            str(item)
            for item in list(requirements.get("required_families") or [])
            if str(item).strip()
        }
        seen_families = {str(item) for item in list(executed_families or []) if str(item).strip()}
        if needed_families and not needed_families.issubset(seen_families):
            return False
        strong_tokens = (
            "counts_by_class",
            "class_summary",
            "scientific_summary",
            "distribution_summary",
            "total_members",
            "downloaded",
            "download_dirs",
            "resource_uri",
            "dataset_uri",
            "predictions_json",
            "depth_summary",
            "segmentation",
            "quantify_objects",
            "total_objects",
            "object_row_count",
            "key_measurements",
            "hdf5_structure",
            "root_keys",
            "default_dataset_path",
        )
        return any(token in evidence_blob for token in strong_tokens)

    @staticmethod
    def _should_escalate_validated_tool_to_research_program(
        *,
        latest_user_text: str,
        tool_plan: ProModeToolPlan | None,
        tool_invocations: list[dict[str, Any]],
        response_text: str,
    ) -> bool:
        if tool_plan is None or tool_plan.category not in VALIDATED_TOOL_CATEGORIES:
            return False
        lowered = str(latest_user_text or "").strip().lower()
        asks_for_multi_step_analysis = bool(
            re.search(
                r"\b(report|analysis|analy[sz]e|summarize|statistics|population|distribution|compare|ecological|given everything)\b",
                lowered,
            )
        )
        if not asks_for_multi_step_analysis:
            return False
        completed_tools = {
            str(invocation.get("tool") or "").strip()
            for invocation in list(tool_invocations or [])
            if str(invocation.get("status") or "").strip().lower() == "completed"
        }
        if not completed_tools:
            return True
        if completed_tools.issubset(
            {
                "bioio_load_image",
                "load_bisque_resource",
                "search_bisque_resources",
                "bisque_find_assets",
            }
        ):
            return True
        return not str(response_text or "").strip()

    def _tool_catalog_for_names(self, tool_names: list[str]) -> list[dict[str, str]]:
        catalog: list[dict[str, str]] = []
        for tool_name in self._normalize_selected_tool_names(tool_names):
            schema = TOOL_SCHEMA_MAP.get(tool_name)
            if not schema:
                continue
            function_schema = dict(schema.get("function") or {})
            catalog.append(
                {
                    "tool_name": tool_name,
                    "family": self._research_program_tool_family(tool_name),
                    "description": str(function_schema.get("description") or "").strip(),
                }
            )
        return catalog

    @staticmethod
    def _tool_program_recent_context(messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for raw in list(messages or [])[-8:]:
            role = str(raw.get("role") or "user").strip().lower()
            content = str(raw.get("content") or "").strip()
            if content:
                lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines).strip()

    def _tool_program_handles_from_invocations(
        self,
        tool_invocations: list[dict[str, Any]],
        *,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
    ) -> dict[str, list[str]]:
        normalized_uploaded = [
            str(path) for path in list(uploaded_files or []) if str(path).strip()
        ]
        handles: dict[str, list[str]] = {
            "uploaded_files": normalized_uploaded,
            "image_files": AgnoChatRuntime._image_like_files(normalized_uploaded),
            "resource_uris": [
                str(uri)
                for uri in list((selection_context or {}).get("resource_uris") or [])
                if str(uri).strip()
            ],
            "dataset_uris": [
                str(uri)
                for uri in list((selection_context or {}).get("dataset_uris") or [])
                if str(uri).strip()
            ],
            "download_dirs": [],
            "downloaded_files": [],
            "mask_paths": [],
            "depth_map_paths": [],
            "depth_npy_paths": [],
            "prediction_json_paths": [],
            "analysis_table_paths": [],
            "preview_paths": [],
            "array_paths": [],
            "job_ids": [],
            "report_paths": [],
            "summary_csv_paths": [],
        }

        for key, raw_values in dict(
            (selection_context or {}).get("artifact_handles") or {}
        ).items():
            handle_key = str(key or "").strip()
            if handle_key not in handles:
                continue
            values = raw_values if isinstance(raw_values, list) else [raw_values]
            for raw_value in values:
                token = str(raw_value or "").strip()
                if token and token not in handles[handle_key]:
                    handles[handle_key].append(token)

        def _append(key: str, value: Any) -> None:
            token = str(value or "").strip()
            if token and token not in handles[key]:
                handles[key].append(token)

        def _download_rows(envelope: dict[str, Any]) -> list[dict[str, Any]]:
            rows = envelope.get("download_rows")
            if isinstance(rows, list):
                return [item for item in rows if isinstance(item, dict)]
            results = envelope.get("results")
            if isinstance(results, list):
                return [item for item in results if isinstance(item, dict)]
            return []

        for invocation in list(tool_invocations or []):
            envelope = invocation.get("output_envelope")
            if not isinstance(envelope, dict):
                continue
            summary = invocation.get("output_summary")
            summary = dict(summary or {}) if isinstance(summary, dict) else {}
            tool_name = str(invocation.get("tool") or "").strip()
            if tool_name in {"search_bisque_resources", "bisque_find_assets"}:
                for item in list(envelope.get("resources") or []):
                    if not isinstance(item, dict):
                        continue
                    resource_uri = item.get("resource_uri") or item.get("uri")
                    resource_type = str(item.get("resource_type") or "").strip().lower()
                    if resource_type == "dataset":
                        _append("dataset_uris", resource_uri)
                    else:
                        _append("resource_uris", resource_uri)
            elif tool_name == "load_bisque_resource":
                resource = envelope.get("resource")
                if isinstance(resource, dict):
                    resource_uri = resource.get("uri") or resource.get("resource_uri")
                    resource_type = str(resource.get("resource_type") or "").strip().lower()
                    if resource_type == "dataset":
                        _append("dataset_uris", resource_uri)
                    else:
                        _append("resource_uris", resource_uri)
            elif tool_name == "bisque_download_dataset":
                _append("download_dirs", envelope.get("download_dir") or envelope.get("output_dir"))
                for item in list(envelope.get("downloaded_files") or []):
                    _append("downloaded_files", item)
                    if AgnoChatRuntime._is_image_like_path(item):
                        _append("image_files", item)
                for row in _download_rows(envelope):
                    output_path = row.get("output_path") or row.get("path") or row.get("local_path")
                    _append("downloaded_files", output_path)
                    if AgnoChatRuntime._is_image_like_path(output_path):
                        _append("image_files", output_path)
                    _append("resource_uris", row.get("resource_uri") or row.get("uri"))
                _append("dataset_uris", envelope.get("dataset_uri"))
            elif tool_name == "bisque_download_resource":
                local_path = envelope.get("local_path")
                _append("downloaded_files", local_path)
                if AgnoChatRuntime._is_image_like_path(local_path):
                    _append("image_files", local_path)
                for row in _download_rows(envelope):
                    output_path = row.get("output_path") or row.get("path") or row.get("local_path")
                    _append("downloaded_files", output_path)
                    if AgnoChatRuntime._is_image_like_path(output_path):
                        _append("image_files", output_path)
                _append("resource_uris", envelope.get("resource_uri"))
            elif tool_name == "bioio_load_image":
                _append("image_files", envelope.get("file_path"))
                _append("preview_paths", envelope.get("preview_path"))
                _append("array_paths", envelope.get("array_path"))
            elif tool_name == "estimate_depth_pro":
                for item in list(envelope.get("depth_map_paths") or []):
                    _append("depth_map_paths", item)
                for item in list(envelope.get("depth_npy_paths") or []):
                    _append("depth_npy_paths", item)
            elif tool_name in {"segment_image_megaseg", "segment_image_sam2", "segment_image_sam3"}:
                for item in list(envelope.get("preferred_upload_paths") or []):
                    _append("mask_paths", item)
                for row in list(envelope.get("files_processed") or []):
                    if not isinstance(row, dict):
                        continue
                    for key in ("preferred_upload_path", "mask_path", "mask_volume_path"):
                        _append("mask_paths", row.get(key))
                _append("report_paths", envelope.get("report_path"))
                _append("summary_csv_paths", envelope.get("summary_csv_path"))
            elif tool_name == "yolo_detect":
                _append(
                    "prediction_json_paths",
                    envelope.get("predictions_json") or summary.get("predictions_json"),
                )
                for item in list(envelope.get("prediction_images_raw") or []):
                    _append("image_files", item)
                for item in list(envelope.get("prediction_image_records") or []):
                    if isinstance(item, dict):
                        _append(
                            "image_files", item.get("raw_source_path") or item.get("source_path")
                        )
                for item in list(envelope.get("prediction_images") or []):
                    _append("preview_paths", item)
            elif tool_name == "codegen_python_plan":
                _append("job_ids", envelope.get("job_id") or summary.get("job_id"))
            elif tool_name == "quantify_objects":
                object_table = envelope.get("object_table")
                if isinstance(object_table, list) and object_table:
                    persisted_path = self._persist_pro_mode_structured_rows(
                        rows=[item for item in object_table if isinstance(item, dict)],
                        prefix="quantify-objects",
                    )
                    _append("analysis_table_paths", persisted_path)
            elif tool_name == "quantify_segmentation_masks":
                rows = envelope.get("rows")
                if isinstance(rows, list) and rows:
                    persisted_path = self._persist_pro_mode_structured_rows(
                        rows=[item for item in rows if isinstance(item, dict)],
                        prefix="quantify-segmentation",
                    )
                    _append("analysis_table_paths", persisted_path)
        return handles

    @staticmethod
    def _merge_handle_maps(*maps: dict[str, Any] | None) -> dict[str, list[str]]:
        merged: dict[str, list[str]] = {}
        for raw_map in maps:
            if not isinstance(raw_map, dict):
                continue
            for key, raw_values in raw_map.items():
                bucket = merged.setdefault(str(key), [])
                values = raw_values if isinstance(raw_values, list) else [raw_values]
                for raw_value in values:
                    token = str(raw_value or "").strip()
                    if token and token not in bucket:
                        bucket.append(token)
        return merged

    def _persist_pro_mode_structured_rows(
        self,
        *,
        rows: list[dict[str, Any]],
        prefix: str,
    ) -> str | None:
        normalized_rows = [dict(item) for item in list(rows or []) if isinstance(item, dict)]
        if not normalized_rows:
            return None
        try:
            serialized = json.dumps(
                normalized_rows,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]
            db_target = str(
                self._setting(self.settings, "run_store_path", "data/runs.db") or "data/runs.db"
            ).strip()
            base_dir = Path(db_target).resolve().parent if db_target else (Path.cwd() / "data")
            target_dir = base_dir / "pro_mode_state"
            target_dir.mkdir(parents=True, exist_ok=True)
            output_path = target_dir / f"{prefix}-{digest}.json"
            if not output_path.exists():
                output_path.write_text(serialized, encoding="utf-8")
            return str(output_path)
        except Exception:
            return None

    def _extract_analysis_conversation_state(
        self,
        *,
        latest_user_text: str,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
        task_regime: str,
        tool_invocations: list[dict[str, Any]],
        research_program_meta: dict[str, Any] | None,
        proof_workflow_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        research_program_meta = dict(research_program_meta or {})
        proof_workflow_meta = dict(proof_workflow_meta or {})
        handles = self._merge_handle_maps(
            self._tool_program_handles_from_invocations(
                tool_invocations,
                uploaded_files=uploaded_files,
                selection_context=selection_context,
            ),
            research_program_meta.get("handles"),
        )
        evidence_summaries: list[str] = []
        for invocation in list(tool_invocations or []):
            summary = invocation.get("output_summary")
            if isinstance(summary, dict) and summary:
                evidence_summaries.append(
                    f"{invocation.get('tool')}: {json.dumps(summary, ensure_ascii=False)}"
                )
        evidence_summaries.extend(
            str(item or "").strip()
            for item in list(research_program_meta.get("evidence_summaries") or [])
            if str(item or "").strip()
        )
        executed_families = list(
            dict.fromkeys(
                [
                    *(
                        self._research_program_tool_family(
                            str(invocation.get("tool") or "").strip()
                        )
                        for invocation in list(tool_invocations or [])
                        if str(invocation.get("tool") or "").strip()
                    ),
                    *[
                        str(item or "").strip()
                        for item in list(research_program_meta.get("executed_families") or [])
                        if str(item or "").strip()
                    ],
                ]
            )
        )
        processed_image_files = list(
            dict.fromkeys(
                [
                    *[
                        str(path or "").strip()
                        for invocation in list(tool_invocations or [])
                        if str(invocation.get("tool") or "").strip() == "yolo_detect"
                        and bool(dict(invocation.get("output_summary") or {}).get("success", True))
                        for path in list(dict(invocation.get("args") or {}).get("file_paths") or [])
                        if str(path or "").strip()
                    ],
                    *[
                        str(dict(invocation.get("args") or {}).get("file_path") or "").strip()
                        for invocation in list(tool_invocations or [])
                        if str(invocation.get("tool") or "").strip() == "bioio_load_image"
                        and bool(dict(invocation.get("output_summary") or {}).get("success", True))
                        and str(dict(invocation.get("args") or {}).get("file_path") or "").strip()
                    ],
                    *[
                        str(item or "").strip()
                        for item in list(research_program_meta.get("processed_image_files") or [])
                        if str(item or "").strip()
                    ],
                ]
            )
        )
        return {
            "last_objective": str(latest_user_text or "").strip(),
            "task_regime": str(task_regime or "").strip(),
            "handles": handles,
            "evidence_summaries": evidence_summaries[-30:],
            "executed_families": executed_families,
            "processed_image_files": processed_image_files[-5000:],
            "updated_at": time.time(),
        }

    def _extract_pro_mode_conversation_state(
        self,
        *,
        latest_user_text: str,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
        task_regime: str,
        tool_invocations: list[dict[str, Any]],
        research_program_meta: dict[str, Any] | None,
        proof_workflow_meta: dict[str, Any] | None = None,
        autonomy_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self._extract_analysis_conversation_state(
            latest_user_text=latest_user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
            task_regime=task_regime,
            tool_invocations=tool_invocations,
            research_program_meta=research_program_meta,
        )
        proof_workflow_meta = dict(proof_workflow_meta or {})
        autonomy_meta = dict(autonomy_meta or {})
        proof_state = dict(proof_workflow_meta.get("proof_state") or {})
        if proof_state:
            proof_digest = [
                *(
                    [f"Goal type: {str(proof_state.get('goal_type') or '').strip()}"]
                    if str(proof_state.get("goal_type") or "").strip()
                    else []
                ),
                *(
                    [f"Reduction: {str(proof_state.get('canonical_reduction') or '').strip()}"]
                    if str(proof_state.get("canonical_reduction") or "").strip()
                    else []
                ),
                *[
                    f"Verified: {str(item or '').strip()}"
                    for item in list(proof_state.get("verified_steps") or [])[:4]
                    if str(item or "").strip()
                ],
                *[
                    f"Open gap: {str(item or '').strip()}"
                    for item in list(proof_state.get("blocker_gaps") or [])[:3]
                    if str(item or "").strip()
                ],
                *(
                    [f"Next focus: {str(proof_state.get('next_iteration_focus') or '').strip()}"]
                    if str(proof_state.get("next_iteration_focus") or "").strip()
                    else []
                ),
            ]
            state["proof_digest"] = proof_digest[:10]
            state["proof_workflow"] = {
                "iterations": int(proof_workflow_meta.get("iterations") or 0),
                "proof_frame": dict(proof_workflow_meta.get("proof_frame") or {}),
                "proof_state": proof_state,
                "iteration_summaries": list(proof_workflow_meta.get("iteration_summaries") or [])[
                    -10:
                ],
                "compression_stats": dict(proof_workflow_meta.get("compression_stats") or {}),
                "stagnant_iterations": int(proof_workflow_meta.get("stagnant_iterations") or 0),
                "last_blocker_signature": str(
                    proof_workflow_meta.get("last_blocker_signature") or ""
                ).strip(),
                "proof_status": str(proof_workflow_meta.get("proof_status") or "").strip()
                or "partial",
            }
        autonomy_state = dict(
            autonomy_meta.get("autonomy_state") or autonomy_meta.get("autonomous_cycle") or {}
        )
        autonomy_state_v2 = dict(autonomy_meta.get("autonomy_state_v2") or {})
        if autonomy_state:
            state["autonomy_state"] = {
                "cycle_id": str(autonomy_state.get("cycle_id") or "").strip(),
                "checkpoint_index": int(autonomy_state.get("checkpoint_index") or 0),
                "open_obligations": [
                    str(item or "").strip()
                    for item in list(autonomy_state.get("open_obligations") or [])
                    if str(item or "").strip()
                ][:12],
                "evidence_ledger": [
                    str(item or "").strip()
                    for item in list(autonomy_state.get("evidence_ledger") or [])
                    if str(item or "").strip()
                ][-20:],
                "candidate_answer": str(autonomy_state.get("candidate_answer") or "").strip(),
                "stop_reason": str(autonomy_state.get("stop_reason") or "").strip(),
                "resume_readiness": str(autonomy_state.get("resume_readiness") or "").strip()
                or "ready",
                "next_best_actions": [
                    str(item or "").strip()
                    for item in list(autonomy_state.get("next_best_actions") or [])
                    if str(item or "").strip()
                ][:8],
                "cycles_completed": int(autonomy_state.get("cycles_completed") or 0),
                "tool_families_used": [
                    str(item or "").strip()
                    for item in list(autonomy_state.get("tool_families_used") or [])
                    if str(item or "").strip()
                ][:8],
                "continuation_fidelity": float(autonomy_state.get("continuation_fidelity") or 0.0),
            }
            state["autonomy_digest"] = [
                *[
                    f"Open obligation: {str(item or '').strip()}"
                    for item in list(autonomy_state.get("open_obligations") or [])[:4]
                    if str(item or "").strip()
                ],
                *(
                    [f"Stop reason: {str(autonomy_state.get('stop_reason') or '').strip()}"]
                    if str(autonomy_state.get("stop_reason") or "").strip()
                    else []
                ),
                *[
                    f"Next action: {str(item or '').strip()}"
                    for item in list(autonomy_state.get("next_best_actions") or [])[:3]
                    if str(item or "").strip()
                ],
            ][:10]
        if autonomy_state_v2:
            state["autonomy_state_v2"] = autonomy_state_v2
            state["candidate_set"] = dict(
                autonomy_meta.get("candidate_set") or autonomy_state_v2.get("candidate_set") or {}
            )
            state["obligation_ledger"] = list(
                autonomy_meta.get("obligation_ledger")
                or autonomy_state_v2.get("obligation_ledger")
                or []
            )[:12]
            state["verification_ledger"] = list(
                autonomy_meta.get("verification_ledger")
                or autonomy_state_v2.get("verification_ledger")
                or []
            )[-10:]
        return state

    def _load_analysis_conversation_state(
        self,
        *,
        conversation_id: str | None,
        user_id: str | None,
        run_id: str | None,
    ) -> dict[str, Any]:
        session_id = self._analysis_state_metadata_session_id(
            conversation_id=conversation_id,
            user_id=user_id,
            run_id=run_id,
        )
        if session_id:
            row = self.session_repository.get_session_any(session_id=session_id) or {}
            metadata = dict(row.get("metadata") or {})
            state = metadata.get("analysis_state")
            if isinstance(state, dict) and state:
                return dict(state)
        legacy_state = self._load_pro_mode_conversation_state(
            conversation_id=conversation_id,
            user_id=user_id,
            run_id=run_id,
        )
        return dict(legacy_state or {}) if self._has_prior_analysis_handles(legacy_state) else {}

    def _load_pro_mode_conversation_state(
        self,
        *,
        conversation_id: str | None,
        user_id: str | None,
        run_id: str | None,
    ) -> dict[str, Any]:
        session_id = self._pro_mode_state_metadata_session_id(
            conversation_id=conversation_id,
            user_id=user_id,
            run_id=run_id,
        )
        if not session_id:
            return {}
        row = self.session_repository.get_session_any(session_id=session_id) or {}
        metadata = dict(row.get("metadata") or {})
        state = metadata.get("pro_mode_state")
        if isinstance(state, dict) and state:
            return dict(state)
        return self._load_pro_mode_state_cache(session_id=session_id)

    def _save_analysis_conversation_state(
        self,
        *,
        conversation_id: str | None,
        user_id: str | None,
        run_id: str | None,
        state: dict[str, Any] | None,
        title: str,
    ) -> None:
        resolved_state = dict(state or {})
        if not self._has_prior_analysis_handles(resolved_state):
            return
        session_id = self._analysis_state_metadata_session_id(
            conversation_id=conversation_id,
            user_id=user_id,
            run_id=run_id,
        )
        if not session_id:
            return
        owner = str(user_id or "").strip() or "anonymous"
        existing = self.session_repository.get_session_any(session_id=session_id) or {}
        metadata = dict(existing.get("metadata") or {})
        metadata["analysis_state"] = resolved_state
        self.session_repository.ensure_session(
            session_id=session_id,
            user_id=owner,
            title=title or "Scientific analysis state",
            metadata=metadata,
        )
        self.session_repository.update_session(
            session_id=session_id,
            user_id=owner,
            title=title or "Scientific analysis state",
            metadata=metadata,
        )

    def _save_pro_mode_conversation_state(
        self,
        *,
        conversation_id: str | None,
        user_id: str | None,
        run_id: str | None,
        state: dict[str, Any] | None,
        title: str,
    ) -> None:
        resolved_state = dict(state or {})
        if not resolved_state:
            return
        session_id = self._pro_mode_state_metadata_session_id(
            conversation_id=conversation_id,
            user_id=user_id,
            run_id=run_id,
        )
        if not session_id:
            return
        self._save_pro_mode_state_cache(session_id=session_id, state=resolved_state)
        owner = str(user_id or "").strip() or "anonymous"
        try:
            existing = self.session_repository.get_session_any(session_id=session_id) or {}
            metadata = dict(existing.get("metadata") or {})
            metadata["pro_mode_state"] = resolved_state
            self.session_repository.ensure_session(
                session_id=session_id,
                user_id=owner,
                title=title or "Pro Mode conversation state",
                metadata=metadata,
            )
            self.session_repository.update_session(
                session_id=session_id,
                user_id=owner,
                title=title or "Pro Mode conversation state",
                metadata=metadata,
            )
        except Exception:
            return

    def _persist_analysis_state(
        self,
        *,
        conversation_id: str | None,
        user_id: str | None,
        run_id: str | None,
        title: str,
        latest_user_text: str,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
        task_regime: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        normalized_metadata = dict(metadata or {})
        research_program_meta = dict(normalized_metadata.get("research_program") or {})
        if not research_program_meta:
            pro_mode = dict(normalized_metadata.get("pro_mode") or {})
            research_program_meta = dict(pro_mode.get("research_program") or {})
        tool_invocations = list(normalized_metadata.get("tool_invocations") or [])
        state = self._extract_analysis_conversation_state(
            latest_user_text=latest_user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
            task_regime=task_regime,
            tool_invocations=tool_invocations,
            research_program_meta=research_program_meta,
        )
        answer_summary = self._analysis_state_answer_summary(
            str(normalized_metadata.get("answer_summary") or "")
        )
        if answer_summary:
            state["last_answer_summary"] = answer_summary
            evidence_summaries = [
                str(item or "").strip()
                for item in list(state.get("evidence_summaries") or [])
                if str(item or "").strip()
            ]
            evidence_summaries.append(f"answer_summary: {answer_summary}")
            state["evidence_summaries"] = evidence_summaries[-30:]
        contract = self._coerce_research_presentation_contract_payload(
            normalized_metadata.get("contract")
            if isinstance(normalized_metadata.get("contract"), dict)
            else None,
            fallback_result=answer_summary,
        )
        contract_brief = self._analysis_state_contract_brief(contract)
        if contract_brief["measurements"]:
            state["last_measurements"] = contract_brief["measurements"][-6:]
        if contract_brief["next_steps"]:
            state["last_next_steps"] = contract_brief["next_steps"][-4:]
        if contract_brief["limitations"]:
            state["last_limitations"] = contract_brief["limitations"][-4:]
        scientific_handles = self._scientific_result_handles_from_tool_invocations(tool_invocations)
        if scientific_handles["active_result_group_id"]:
            state["active_result_group_id"] = scientific_handles["active_result_group_id"]
        if scientific_handles["active_report_handle"]:
            state["active_report_handle"] = scientific_handles["active_report_handle"]
        if scientific_handles["active_summary_csv_handle"]:
            state["active_summary_csv_handle"] = scientific_handles["active_summary_csv_handle"]
        if scientific_handles["active_selected_files"]:
            state["active_selected_files"] = scientific_handles["active_selected_files"][-8:]
        self._save_analysis_conversation_state(
            conversation_id=conversation_id,
            user_id=user_id,
            run_id=run_id,
            state=state,
            title=title,
        )

    @staticmethod
    def _scientific_result_handles_from_tool_invocations(
        tool_invocations: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        active_result_group_id = ""
        active_report_handle = ""
        active_summary_csv_handle = ""
        active_selected_files: list[str] = []
        for invocation in list(tool_invocations or []):
            if str(invocation.get("status") or "").strip().lower() != "completed":
                continue
            tool_name = str(invocation.get("tool") or "").strip()
            if tool_name not in {
                "segment_image_megaseg",
                "quantify_segmentation_masks",
                "segment_image_sam2",
                "segment_image_sam3",
            }:
                continue
            envelope = invocation.get("output_envelope")
            if not isinstance(envelope, dict):
                continue
            refs = envelope.get("latest_result_refs")
            refs_dict = dict(refs) if isinstance(refs, dict) else {}
            candidate_group_id = (
                str(envelope.get("result_group_id") or "").strip()
                or str(refs_dict.get("latest_segmentation_result_group_id") or "").strip()
            )
            if candidate_group_id:
                active_result_group_id = candidate_group_id
            candidate_report = (
                str(envelope.get("report_path") or "").strip()
                or str(refs_dict.get("segment_image_megaseg.report_path") or "").strip()
            )
            if candidate_report:
                active_report_handle = candidate_report
            candidate_summary_csv = (
                str(envelope.get("summary_csv_path") or "").strip()
                or str(refs_dict.get("segment_image_megaseg.summary_csv_path") or "").strip()
            )
            if candidate_summary_csv:
                active_summary_csv_handle = candidate_summary_csv
            for row in list(envelope.get("files_processed") or []):
                if not isinstance(row, dict):
                    continue
                file_label = str(row.get("file") or row.get("path") or "").strip()
                if file_label and file_label not in active_selected_files:
                    active_selected_files.append(file_label)
        return {
            "active_result_group_id": active_result_group_id,
            "active_report_handle": active_report_handle,
            "active_summary_csv_handle": active_summary_csv_handle,
            "active_selected_files": active_selected_files,
        }

    def _analysis_session_state_payload(
        self,
        *,
        analysis_state: dict[str, Any] | None,
        selection_context: dict[str, Any] | None,
        uploaded_files: list[str],
    ) -> dict[str, Any]:
        state = dict(analysis_state or {})
        selection = dict(selection_context or {})
        merged_handles = self._merge_handle_maps(
            dict(state.get("handles") or {}),
            dict(selection.get("artifact_handles") or {}),
            {
                "dataset_uris": list(selection.get("dataset_uris") or []),
                "resource_uris": list(selection.get("resource_uris") or []),
            },
        )
        if not merged_handles and not state.get("evidence_summaries") and not uploaded_files:
            return {}
        handle_counts = {
            key: len(list(values or []))
            for key, values in merged_handles.items()
            if list(values or [])
        }
        current_focus: dict[str, Any] = {}
        if uploaded_files:
            current_focus["uploaded_files"] = [
                Path(path).name for path in list(uploaded_files or [])[:5]
            ]
        if selection.get("dataset_uris"):
            current_focus["dataset_uris"] = [
                str(item or "").strip()
                for item in list(selection.get("dataset_uris") or [])[:3]
                if str(item or "").strip()
            ]
        if selection.get("resource_uris"):
            current_focus["resource_uris"] = [
                str(item or "").strip()
                for item in list(selection.get("resource_uris") or [])[:5]
                if str(item or "").strip()
            ]
        payload = {
            "analysis_state": {
                "task_regime": str(state.get("task_regime") or "").strip() or None,
                "last_objective": str(state.get("last_objective") or "").strip() or None,
                "executed_families": [
                    str(item or "").strip()
                    for item in list(state.get("executed_families") or [])[:6]
                    if str(item or "").strip()
                ],
                "evidence_summaries": [
                    str(item or "").strip()
                    for item in list(state.get("evidence_summaries") or [])[-6:]
                    if str(item or "").strip()
                ],
                "last_answer_summary": str(state.get("last_answer_summary") or "").strip() or None,
                "key_measurements": [
                    str(item or "").strip()
                    for item in list(state.get("last_measurements") or [])[:4]
                    if str(item or "").strip()
                ],
                "recommended_next_steps": [
                    str(item or "").strip()
                    for item in list(state.get("last_next_steps") or [])[:3]
                    if str(item or "").strip()
                ],
                "open_limits": [
                    str(item or "").strip()
                    for item in list(state.get("last_limitations") or [])[:3]
                    if str(item or "").strip()
                ],
                "active_result_group_id": (
                    str(state.get("active_result_group_id") or "").strip() or None
                ),
                "active_report_handle": (
                    str(state.get("active_report_handle") or "").strip() or None
                ),
                "active_summary_csv_handle": (
                    str(state.get("active_summary_csv_handle") or "").strip() or None
                ),
                "active_selected_files": [
                    str(item or "").strip()
                    for item in list(state.get("active_selected_files") or [])[:4]
                    if str(item or "").strip()
                ],
                "handle_counts": handle_counts,
                "current_focus": current_focus,
            }
        }
        payload["analysis_state"] = {
            key: value
            for key, value in payload["analysis_state"].items()
            if value not in (None, "", [], {})
        }
        return payload if payload["analysis_state"] else {}

    @staticmethod
    def _analysis_state_answer_summary(response_text: str) -> str:
        text = re.sub(r"```.*?```", " ", str(response_text or ""), flags=re.DOTALL)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""
        sentences = [
            segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()
        ]
        summary = " ".join(sentences[:2]).strip() or text
        if len(summary) > 420:
            summary = summary[:417].rstrip() + "..."
        return summary

    @staticmethod
    def _presentation_text(value: Any, *, max_chars: int = 220) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if len(text) > max_chars:
            return text[: max_chars - 3].rstrip() + "..."
        return text

    @classmethod
    def _presentation_result_summary(cls, response_text: str) -> str:
        text = re.sub(r"```.*?```", " ", str(response_text or ""), flags=re.DOTALL)
        bottom_line_match = re.search(
            r"(?:bottom line|takeaway|direct answer|key point|main conclusion|primary suspect|most likely failure mode|answer)\s*:\s*(.+?)(?:\n|$)",
            text,
            flags=re.IGNORECASE,
        )
        if bottom_line_match:
            return cls._presentation_text(bottom_line_match.group(1), max_chars=320)
        candidates: list[str] = []
        for raw_line in text.splitlines():
            line = re.sub(r"^\s*#{1,6}\s*", "", raw_line)
            line = re.sub(r"^\s*(?:[-*+]|\d+\.)\s*", "", line)
            line = re.sub(r"\s+", " ", line).strip()
            if not line:
                continue
            if len(line.split()) <= 6 and not re.search(r"[.!?]", line):
                continue
            candidates.append(line)
        joined = " ".join(candidates[:6]).strip()
        if not joined:
            joined = cls._analysis_state_answer_summary(text)
        sentences = [
            segment.strip() for segment in re.split(r"(?<=[.!?])\s+", joined) if segment.strip()
        ]
        summary = " ".join(sentences[:2]).strip() or joined
        return cls._presentation_text(summary, max_chars=320)

    @staticmethod
    def _presentation_label(value: str) -> str:
        normalized = re.sub(r"[_\-]+", " ", str(value or "")).strip()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized or "measurement"

    @classmethod
    def _presentation_unit_for_key(cls, key: str) -> str | None:
        lowered = str(key or "").strip().lower()
        if not lowered:
            return None
        if (
            lowered.endswith("percent")
            or lowered.endswith("_percent")
            or lowered.endswith("percentage")
        ):
            return "%"
        if lowered.endswith("_count") or lowered.endswith("count") or lowered.startswith("num_"):
            return "count"
        if lowered.endswith("_seconds") or lowered.endswith("seconds") or lowered.endswith("_secs"):
            return "s"
        if lowered.endswith("_minutes") or lowered.endswith("minutes"):
            return "min"
        if lowered.endswith("_hours") or lowered.endswith("hours"):
            return "h"
        if lowered.endswith("_voxels") or lowered.endswith("voxels"):
            return "voxels"
        if lowered.endswith("_slices") or lowered.endswith("slices"):
            return "slices"
        return None

    @classmethod
    def _presentation_scalar_measurements(
        cls,
        record: dict[str, Any] | None,
        *,
        tool_name: str = "",
        max_items: int = 8,
    ) -> list[dict[str, Any]]:
        payload = dict(record or {})
        if not payload:
            return []
        ignored_keys = {
            "success",
            "status",
            "error",
            "message",
            "details",
            "run_id",
            "tool",
            "model",
            "artifact",
            "artifacts",
        }
        ignored_fragments = (
            "path",
            "paths",
            "file",
            "files",
            "url",
            "uri",
            "directory",
            "artifact",
            "preview",
            "report",
            "mask",
            "overlay",
            "summary",
            "warning",
            "error",
        )
        measurements: list[dict[str, Any]] = []
        seen: set[str] = set()

        def _maybe_add(name: str, value: Any) -> None:
            if len(measurements) >= max_items:
                return
            key = str(name or "").strip()
            if not key:
                return
            lowered = key.lower()
            if lowered in ignored_keys or any(
                fragment in lowered for fragment in ignored_fragments
            ):
                return
            if isinstance(value, bool) or value is None:
                return
            if isinstance(value, (int, float)):
                label = cls._presentation_label(key)
                identity = f"{label.lower()}::{cls._presentation_unit_for_key(key) or ''}"
                if identity in seen:
                    return
                seen.add(identity)
                measurements.append(
                    {
                        "name": label,
                        "value": value,
                        "unit": cls._presentation_unit_for_key(key),
                        "summary": (
                            f"Derived from {cls._presentation_label(tool_name)}."
                            if str(tool_name or "").strip()
                            else None
                        ),
                    }
                )
                return
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    _maybe_add(f"{key} {nested_key}", nested_value)
                return
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped or len(stripped) > 40:
                    return
                if not re.fullmatch(r"-?\d+(?:\.\d+)?", stripped):
                    return
                parsed_value: int | float
                parsed_value = float(stripped) if "." in stripped else int(stripped)
                _maybe_add(key, parsed_value)

        for item_key, item_value in payload.items():
            _maybe_add(str(item_key or ""), item_value)
            if len(measurements) >= max_items:
                break
        return measurements

    @classmethod
    def _coerce_research_presentation_contract_payload(
        cls,
        payload: dict[str, Any] | None,
        *,
        fallback_result: str,
    ) -> dict[str, Any]:
        source = dict(payload or {})
        confidence_payload = dict(source.get("confidence") or {})
        level = str(confidence_payload.get("level") or "").strip().lower()
        if level not in {"low", "medium", "high"}:
            level = "medium"
        contract = ResearchPresentationContract(
            result=cls._presentation_text(source.get("result") or fallback_result, max_chars=600),
            evidence=[
                ResearchPresentationEvidence(
                    source=cls._presentation_text(item.get("source"), max_chars=80),
                    summary=cls._presentation_text(item.get("summary"), max_chars=220),
                    artifact=cls._presentation_text(item.get("artifact"), max_chars=180) or None,
                    run_id=cls._presentation_text(item.get("run_id"), max_chars=80) or None,
                )
                for item in list(source.get("evidence") or [])
                if isinstance(item, dict)
                and (
                    cls._presentation_text(item.get("source"), max_chars=80)
                    or cls._presentation_text(item.get("summary"), max_chars=220)
                )
            ][:4],
            measurements=[
                ResearchPresentationMeasurement(
                    name=cls._presentation_label(item.get("name")),
                    value=item.get("value"),
                    unit=cls._presentation_text(item.get("unit"), max_chars=24) or None,
                    summary=cls._presentation_text(item.get("summary"), max_chars=180) or None,
                )
                for item in list(source.get("measurements") or [])
                if isinstance(item, dict)
                and cls._presentation_label(item.get("name"))
                and item.get("value") not in (None, "")
            ][:6],
            statistical_analysis=[
                ResearchPresentationStatistic(
                    label=cls._presentation_label(item.get("label")),
                    summary=cls._presentation_text(item.get("summary"), max_chars=220),
                )
                for item in list(source.get("statistical_analysis") or [])
                if isinstance(item, dict)
                and cls._presentation_label(item.get("label"))
                and cls._presentation_text(item.get("summary"), max_chars=220)
            ][:4],
            confidence=ResearchPresentationConfidence(
                level=level,
                why=[
                    cls._presentation_text(item, max_chars=180)
                    for item in list(confidence_payload.get("why") or [])
                    if cls._presentation_text(item, max_chars=180)
                ][:3],
            ),
            qc_warnings=[
                cls._presentation_text(item, max_chars=200)
                for item in list(source.get("qc_warnings") or [])
                if cls._presentation_text(item, max_chars=200)
            ][:4],
            limitations=[
                cls._presentation_text(item, max_chars=220)
                for item in list(source.get("limitations") or [])
                if cls._presentation_text(item, max_chars=220)
            ][:4],
            next_steps=[
                ResearchPresentationNextStep(
                    action=cls._presentation_text(
                        item.get("action") if isinstance(item, dict) else item,
                        max_chars=220,
                    )
                )
                for item in list(source.get("next_steps") or [])
                if cls._presentation_text(
                    item.get("action") if isinstance(item, dict) else item,
                    max_chars=220,
                )
            ][:4],
        )
        if not str(contract.result or "").strip():
            contract.result = cls._presentation_text(fallback_result, max_chars=600)
        if not list(contract.next_steps or []):
            contract.next_steps = [
                ResearchPresentationNextStep(
                    action="Continue with the next highest-impact validation or comparison step."
                )
            ]
        return contract.model_dump(mode="json")

    @classmethod
    def _fallback_research_presentation_contract(
        cls,
        *,
        user_text: str,
        response_text: str,
        metadata: dict[str, Any] | None,
        tool_invocations: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        normalized_response = cls._presentation_result_summary(response_text)
        normalized_metadata = dict(metadata or {})
        normalized_invocations = [
            dict(item or {}) for item in list(tool_invocations or []) if isinstance(item, dict)
        ]
        evidence: list[dict[str, Any]] = []
        measurements: list[dict[str, Any]] = []
        qc_warnings: list[str] = []
        limitations: list[str] = []
        seen_evidence: set[str] = set()
        seen_measurements: set[str] = set()

        def _append_evidence(
            source: str, summary: str, artifact: str | None = None, run_id: str | None = None
        ) -> None:
            rendered_source = cls._presentation_text(source, max_chars=80)
            rendered_summary = cls._presentation_text(summary, max_chars=220)
            if not rendered_source and not rendered_summary:
                return
            identity = f"{rendered_source.lower()}::{rendered_summary.lower()}"
            if identity in seen_evidence:
                return
            seen_evidence.add(identity)
            evidence.append(
                {
                    "source": rendered_source or "Evidence",
                    "summary": rendered_summary,
                    "artifact": cls._presentation_text(artifact, max_chars=180) or None,
                    "run_id": cls._presentation_text(run_id, max_chars=80) or None,
                }
            )

        def _append_measurement(item: dict[str, Any]) -> None:
            name = cls._presentation_label(item.get("name"))
            value = item.get("value")
            if not name or value in (None, ""):
                return
            identity = (
                f"{name.lower()}::{cls._presentation_text(item.get('unit'), max_chars=24).lower()}"
            )
            if identity in seen_measurements:
                return
            seen_measurements.add(identity)
            measurements.append(
                {
                    "name": name,
                    "value": value,
                    "unit": cls._presentation_text(item.get("unit"), max_chars=24) or None,
                    "summary": cls._presentation_text(item.get("summary"), max_chars=180) or None,
                }
            )

        for invocation in normalized_invocations[:8]:
            tool_name = cls._presentation_label(str(invocation.get("tool") or "analysis"))
            status = str(invocation.get("status") or "").strip().lower()
            summary_payload = invocation.get("output_summary")
            rendered_summary = ""
            if isinstance(summary_payload, str):
                rendered_summary = cls._presentation_text(summary_payload, max_chars=220)
            elif isinstance(summary_payload, dict):
                rendered_summary = cls._presentation_text(
                    ", ".join(
                        f"{cls._presentation_label(key)}={value}"
                        for key, value in list(summary_payload.items())[:4]
                        if value not in (None, "", [], {})
                    ),
                    max_chars=220,
                )
                for item in cls._presentation_scalar_measurements(
                    summary_payload,
                    tool_name=tool_name,
                ):
                    _append_measurement(item)
            output_envelope = invocation.get("output_envelope")
            if isinstance(output_envelope, dict):
                for summary_key in ("summary", "metrics", "measurements"):
                    nested_payload = output_envelope.get(summary_key)
                    if isinstance(nested_payload, dict):
                        for item in cls._presentation_scalar_measurements(
                            nested_payload,
                            tool_name=tool_name,
                        ):
                            _append_measurement(item)
                if not rendered_summary:
                    rendered_summary = cls._presentation_text(
                        output_envelope.get("summary"),
                        max_chars=220,
                    )
            artifact = next(
                (
                    cls._presentation_text(item, max_chars=180)
                    for item in list(invocation.get("preferred_upload_paths") or [])
                    if cls._presentation_text(item, max_chars=180)
                ),
                None,
            )
            run_id = cls._presentation_text(invocation.get("run_id"), max_chars=80) or None
            if rendered_summary:
                _append_evidence(tool_name, rendered_summary, artifact=artifact, run_id=run_id)
            if status in {"error", "failed"}:
                warning_text = cls._presentation_text(
                    invocation.get("output_summary")
                    or invocation.get("error")
                    or f"{tool_name} failed.",
                    max_chars=220,
                )
                if warning_text and warning_text not in qc_warnings:
                    qc_warnings.append(warning_text)

        pro_mode_metadata = dict(normalized_metadata.get("pro_mode") or {})
        research_program_meta = dict(pro_mode_metadata.get("research_program") or {})
        for item in list(research_program_meta.get("evidence_summaries") or [])[:4]:
            summary_text = cls._presentation_text(item, max_chars=220)
            if summary_text:
                _append_evidence("Research program", summary_text)

        metadata_specialist = dict(pro_mode_metadata.get("metadata_specialist") or {})
        for item in list(metadata_specialist.get("verified_findings") or [])[:3]:
            summary_text = cls._presentation_text(item, max_chars=220)
            if summary_text:
                _append_evidence("Metadata review", summary_text)
        for item in [
            *list(metadata_specialist.get("missing_metadata") or []),
            *list(metadata_specialist.get("caveats") or []),
        ]:
            rendered = cls._presentation_text(item, max_chars=220)
            if rendered and rendered not in limitations:
                limitations.append(rendered)

        verifier = dict(pro_mode_metadata.get("verifier") or {})
        for item in list(verifier.get("issues") or [])[:3]:
            rendered = cls._presentation_text(item, max_chars=220)
            if rendered and rendered not in limitations:
                limitations.append(rendered)

        missing_families = list(
            research_program_meta.get("requirements", {}).get("missing_families") or []
        )
        if missing_families:
            limitations.append(
                "Missing evidence families: "
                + ", ".join(
                    cls._presentation_label(str(item or ""))
                    for item in missing_families[:4]
                    if cls._presentation_label(str(item or ""))
                )
            )

        if not limitations and not evidence and not measurements:
            limitations.append(
                "This response is primarily interpretive and does not include newly measured artifacts from this turn."
            )

        confidence_level = "medium"
        confidence_why: list[str] = []
        if qc_warnings:
            confidence_level = "low"
            confidence_why.append(
                "One or more workflow steps reported warnings or incomplete outputs."
            )
        elif measurements or len(evidence) >= 2:
            confidence_level = "high" if len(measurements) >= 2 else "medium"
            confidence_why.append(
                "The answer is anchored by measured outputs or explicit evidence summaries from this turn."
            )
        else:
            confidence_why.append(
                "The answer is grounded in the current response text but has limited structured evidence."
            )

        next_steps: list[dict[str, str]] = []
        if measurements:
            next_steps.append(
                {
                    "action": "Compare the key measurements against a control, replicate, or alternative condition."
                }
            )
        if evidence:
            next_steps.append(
                {
                    "action": "Review the leading figures or artifacts to confirm that the interpretation matches the visual evidence."
                }
            )
        if not next_steps:
            next_steps.append(
                {
                    "action": "Apply this interpretation to a representative example and check whether the main criteria hold in practice."
                }
            )
        if "follow" not in str(user_text or "").strip().lower():
            next_steps.append(
                {
                    "action": "Ask a focused follow-up question if you want a deeper comparison, validation plan, or implementation detail."
                }
            )

        return cls._coerce_research_presentation_contract_payload(
            {
                "result": normalized_response
                or cls._presentation_text(response_text, max_chars=600),
                "evidence": evidence[:4],
                "measurements": measurements[:6],
                "statistical_analysis": [],
                "confidence": {
                    "level": confidence_level,
                    "why": confidence_why[:3],
                },
                "qc_warnings": qc_warnings[:4],
                "limitations": limitations[:4],
                "next_steps": next_steps[:4],
            },
            fallback_result=response_text,
        )

    @classmethod
    def _analysis_state_contract_brief(
        cls,
        contract: dict[str, Any] | None,
    ) -> dict[str, list[str]]:
        normalized = cls._coerce_research_presentation_contract_payload(
            contract,
            fallback_result=str((contract or {}).get("result") or ""),
        )
        measurement_lines = [
            cls._presentation_text(
                f"{cls._presentation_label(item.get('name'))}: {item.get('value')}"
                + (
                    f" {cls._presentation_text(item.get('unit'), max_chars=24)}"
                    if cls._presentation_text(item.get("unit"), max_chars=24)
                    else ""
                ),
                max_chars=140,
            )
            for item in list(normalized.get("measurements") or [])[:4]
            if isinstance(item, dict)
            and cls._presentation_label(item.get("name"))
            and item.get("value") not in (None, "")
        ]
        next_step_lines = [
            cls._presentation_text(
                item.get("action") if isinstance(item, dict) else item,
                max_chars=180,
            )
            for item in list(normalized.get("next_steps") or [])[:3]
            if cls._presentation_text(
                item.get("action") if isinstance(item, dict) else item,
                max_chars=180,
            )
        ]
        limitation_lines = [
            cls._presentation_text(item, max_chars=180)
            for item in list(normalized.get("limitations") or [])[:3]
            if cls._presentation_text(item, max_chars=180)
        ]
        return {
            "measurements": measurement_lines,
            "next_steps": next_step_lines,
            "limitations": limitation_lines,
        }

    @classmethod
    def _merge_selection_context_with_analysis_state(
        cls,
        *,
        user_text: str,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
        analysis_state: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        current = dict(selection_context or {})
        state = dict(analysis_state or {})
        handles = dict(state.get("handles") or {})
        if not handles:
            return selection_context
        current_turn_replaces_saved_image_target = cls._current_turn_replaces_saved_image_target(
            user_text=user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
            prior_state=analysis_state,
        )
        prompt_requests_cross_turn_comparison = cls._prompt_requests_cross_turn_comparison(
            user_text
        )
        if not cls._should_reuse_analysis_state(
            user_text=user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
            prior_state=analysis_state,
        ):
            return selection_context
        for key in ("resource_uris", "dataset_uris"):
            merged = list(current.get(key) or [])
            for token in list(handles.get(key) or []):
                value = str(token or "").strip()
                if value and value not in merged:
                    merged.append(value)
            if merged:
                current[key] = merged
        artifact_handles = dict(current.get("artifact_handles") or {})
        for key in (
            "download_dirs",
            "downloaded_files",
            "mask_paths",
            "depth_map_paths",
            "depth_npy_paths",
            "prediction_json_paths",
            "analysis_table_paths",
            "preview_paths",
            "array_paths",
            "job_ids",
            "image_files",
            "report_paths",
            "summary_csv_paths",
        ):
            if current_turn_replaces_saved_image_target and key in {
                "download_dirs",
                "downloaded_files",
                "mask_paths",
                "depth_map_paths",
                "depth_npy_paths",
                "preview_paths",
                "array_paths",
                "image_files",
            }:
                continue
            if (
                current_turn_replaces_saved_image_target
                and not prompt_requests_cross_turn_comparison
                and key in {"prediction_json_paths", "analysis_table_paths", "job_ids"}
            ):
                continue
            merged = list(artifact_handles.get(key) or [])
            for token in list(handles.get(key) or []):
                value = str(token or "").strip()
                if value and value not in merged:
                    merged.append(value)
            if merged:
                artifact_handles[key] = merged
        if artifact_handles:
            current["artifact_handles"] = artifact_handles
        return current

    @classmethod
    def _merge_selection_context_with_pro_mode_state(
        cls,
        *,
        user_text: str,
        selection_context: dict[str, Any] | None,
        pro_mode_state: dict[str, Any] | None,
        uploaded_files: list[str] | None = None,
    ) -> dict[str, Any] | None:
        return cls._merge_selection_context_with_analysis_state(
            user_text=user_text,
            uploaded_files=list(uploaded_files or []),
            selection_context=selection_context,
            analysis_state=pro_mode_state,
        )

    @classmethod
    def _should_reuse_analysis_state(
        cls,
        *,
        user_text: str,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
        prior_state: dict[str, Any] | None,
    ) -> bool:
        state = dict(prior_state or {})
        if not state:
            return False
        handles = dict(state.get("handles") or {})
        has_prior_handles = any(bool(list(handles.get(key) or [])) for key in handles)
        if not has_prior_handles:
            return False
        current = dict(selection_context or {})
        current_dataset_uris = {
            str(item or "").strip()
            for item in list(current.get("dataset_uris") or [])
            if str(item or "").strip()
        }
        prior_dataset_uris = {
            str(item or "").strip()
            for item in list(handles.get("dataset_uris") or [])
            if str(item or "").strip()
        }
        current_resource_uris = {
            str(item or "").strip()
            for item in list(current.get("resource_uris") or [])
            if str(item or "").strip()
        }
        prior_resource_uris = {
            str(item or "").strip()
            for item in list(handles.get("resource_uris") or [])
            if str(item or "").strip()
        }
        current_turn_replaces_saved_image_target = cls._current_turn_replaces_saved_image_target(
            user_text=user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
            prior_state=prior_state,
        )
        if (
            current_turn_replaces_saved_image_target
            and not cls._prompt_requests_cross_turn_comparison(user_text)
        ):
            return False
        if current_dataset_uris and current_dataset_uris & prior_dataset_uris:
            return True
        if current_resource_uris and current_resource_uris & prior_resource_uris:
            return True
        if cls._is_prior_analysis_follow_up_request(user_text, prior_state=prior_state):
            return True
        if uploaded_files:
            return cls._has_cross_turn_artifact_reference(user_text)
        return cls._has_cross_turn_artifact_reference(user_text)

    @classmethod
    def _should_reuse_pro_mode_state(
        cls,
        *,
        user_text: str,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
        prior_state: dict[str, Any] | None,
    ) -> bool:
        return cls._should_reuse_analysis_state(
            user_text=user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
            prior_state=prior_state,
        )

    def _initial_research_program_state(
        self,
        *,
        latest_user_text: str,
        messages: list[dict[str, Any]],
        tool_names: list[str],
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
        existing_state: dict[str, Any] | None,
        seed_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        current_handles = self._tool_program_handles_from_invocations(
            [],
            uploaded_files=uploaded_files,
            selection_context=selection_context,
        )
        carry_states: list[dict[str, Any]] = []
        for candidate in (existing_state, seed_state):
            if self._should_reuse_analysis_state(
                user_text=latest_user_text,
                uploaded_files=uploaded_files,
                selection_context=selection_context,
                prior_state=candidate,
            ):
                carry_states.append(dict(candidate or {}))
        merged_handles = self._merge_handle_maps(
            *(dict(candidate.get("handles") or {}) for candidate in carry_states),
            current_handles,
        )
        evidence_summaries: list[str] = []
        for candidate in carry_states:
            evidence_summaries.extend(
                str(item or "").strip()
                for item in list(candidate.get("evidence_summaries") or [])
                if str(item or "").strip()
            )
        executed_families = list(
            dict.fromkeys(
                str(item or "").strip()
                for candidate in carry_states
                for item in list(candidate.get("executed_families") or [])
                if str(item or "").strip()
            )
        )
        compression_stats: dict[str, Any] = {}
        for candidate in carry_states:
            compression_stats.update(dict(candidate.get("compression_stats") or {}))
        processed_image_files = list(
            dict.fromkeys(
                str(item or "").strip()
                for candidate in carry_states
                for item in list(candidate.get("processed_image_files") or [])
                if str(item or "").strip()
            )
        )
        current_uploaded_files = [
            str(path or "").strip()
            for path in list(uploaded_files or [])
            if str(path or "").strip()
        ]
        return {
            "iteration": 0,
            "objective": latest_user_text,
            "recent_context": self._tool_program_recent_context(messages),
            "tool_catalog": self._tool_catalog_for_names(tool_names),
            "evidence_summaries": evidence_summaries[-30:],
            "handles": merged_handles,
            "tool_invocations": [],
            "requirements": self._research_program_requirements(
                latest_user_text=latest_user_text,
                uploaded_files=uploaded_files,
                selection_context=selection_context,
                prior_pro_mode_state=seed_state,
            ),
            "executed_families": executed_families,
            "processed_image_files": processed_image_files,
            "current_uploaded_files": current_uploaded_files,
            "compression_stats": compression_stats,
        }

    def _tool_program_plan_prompt(
        self,
        *,
        latest_user_text: str,
        recent_context: str,
        tool_catalog: list[dict[str, str]],
        evidence_summaries: list[str],
        handles: dict[str, list[str]],
        iteration: int,
    ) -> str:
        return "\n".join(
            [
                "You are planning the next iteration of a scientific evidence program.",
                "Choose the smallest set of tool actions that will materially improve the answer.",
                "Prefer inspecting and measuring real data over speculation.",
                "Only request tools from the available catalog.",
                "Do not invent file paths, URIs, or job ids. Reuse only handles that are explicitly available.",
                "If the evidence is already sufficient, set ready_to_answer=true and propose no further actions.",
                "Keep the plan compact: at most 3 actions in one iteration.",
                "",
                f"Iteration: {iteration}",
                f"User request: {latest_user_text}",
                "",
                "Recent conversation:",
                recent_context or "No prior conversation context.",
                "",
                "Available tool catalog:",
                json.dumps(tool_catalog, ensure_ascii=False, indent=2),
                "",
                "Available handles:",
                json.dumps(handles, ensure_ascii=False, indent=2),
                "",
                "Compressed evidence so far:",
                json.dumps(evidence_summaries[-10:], ensure_ascii=False, indent=2),
                "",
                "Planning rules:",
                "- Use capability families intentionally: catalog for discovery, vision for measured image evidence, analysis for summaries/counts, and code only when deterministic parsing or computation is needed.",
                "- For catalog questions, search first, then inspect or download only if that unlocks a missing fact.",
                "- For image-analysis questions, prefer substantive measurement tools over impressionistic reading from metadata alone.",
                "- For dataset statistics that are not directly exposed by one tool, use code generation + execution after acquiring the relevant files.",
                "- For plot, chart, or visualization requests grounded in prior measurements, prefer deterministic code generation + execution from the saved analysis artifacts instead of rewriting the analysis in prose only.",
                "- When using code execution, request codegen_python_plan first and then execute_python_job only after a job_id is available.",
                "- For report-writing tasks, gather enough measured evidence first; do not synthesize from a single weak clue.",
            ]
        )

    def _fallback_tool_program_plan(
        self,
        *,
        latest_user_text: str,
        handles: dict[str, list[str]],
        available_tool_names: list[str],
    ) -> ToolProgramIterationPlan:
        lowered = str(latest_user_text or "").strip().lower()
        available = set(available_tool_names)
        actions: list[ToolProgramAction] = []
        quantitative_request = bool(
            re.search(
                r"\b(counts?|average|mean|median|statistics|distribution|population|density|measurements?|box size|annotation|bounding[- ]box|bbox)\b",
                lowered,
            )
        )
        object_pattern_request = bool(
            re.search(
                r"\b(prairie dog|burrow|object|bbox|bounding[- ]box|box size|detection|detect|class pattern|population)\b",
                lowered,
            )
        )
        report_like_request = bool(
            re.search(r"\b(report|summary|analysis|ecological|summarize|compare)\b", lowered)
        )
        plot_request = self._is_plot_or_visual_analysis_request(latest_user_text)
        plot_prefers_detection_artifacts = self._plot_request_prefers_detection_artifacts(
            latest_user_text
        )
        requested_confidence_threshold = self._requested_confidence_threshold(latest_user_text)
        introspection_request = self._is_structured_artifact_introspection_request(latest_user_text)
        image_files = [
            str(path or "").strip()
            for path in list(handles.get("image_files") or [])
            if str(path or "").strip()
        ]
        mask_paths = [
            str(path or "").strip()
            for path in list(handles.get("mask_paths") or [])
            if str(path or "").strip()
        ]
        depth_map_paths = [
            str(path or "").strip()
            for path in list(handles.get("depth_map_paths") or [])
            if str(path or "").strip()
        ]
        downloaded_files = [
            str(path or "").strip()
            for path in list(handles.get("downloaded_files") or [])
            if str(path or "").strip()
        ]
        downloaded_hdf5_files = [path for path in downloaded_files if self._is_hdf5_like_path(path)]
        uploaded_hdf5_files = [
            str(path or "").strip()
            for path in list(handles.get("uploaded_files") or [])
            if str(path or "").strip() and self._is_hdf5_like_path(path)
        ]
        selected_resource_uri = next(
            (
                str(item or "").strip()
                for item in list(handles.get("resource_uris") or [])
                if str(item or "").strip()
            ),
            "",
        )
        if (
            introspection_request
            and selected_resource_uri
            and not (downloaded_hdf5_files or uploaded_hdf5_files)
            and "bisque_download_resource" in available
        ):
            actions.append(
                ToolProgramAction(
                    tool_name="bisque_download_resource",
                    purpose="Download the selected scientific container so its internal structure can be inspected deterministically.",
                    args={"resource_uri": selected_resource_uri},
                )
            )
        if (
            handles.get("dataset_uris")
            and not handles.get("download_dirs")
            and "load_bisque_resource" in available
        ):
            actions.append(
                ToolProgramAction(
                    tool_name="load_bisque_resource",
                    purpose="Inspect the known BisQue dataset metadata before deeper analysis.",
                    args={"resource_uri": handles["dataset_uris"][0]},
                )
            )
        if (
            handles.get("dataset_uris")
            and not handles.get("download_dirs")
            and not downloaded_files
            and "bisque_download_dataset" in available
        ):
            actions.append(
                ToolProgramAction(
                    tool_name="bisque_download_dataset",
                    purpose="Download the known BisQue dataset so it can be analyzed deterministically.",
                    args={"dataset_uri": handles["dataset_uris"][0], "limit": 200},
                )
            )
        elif (
            re.search(r"\b(dataset|bisque|resource)\b", lowered)
            and "search_bisque_resources" in available
        ):
            actions.append(
                ToolProgramAction(
                    tool_name="search_bisque_resources",
                    purpose="Discover relevant BisQue resources before deeper analysis.",
                    args={"resource_type": "dataset", "limit": 10},
                )
            )
        if (
            handles.get("prediction_json_paths")
            and "quantify_objects" in available
            and (quantitative_request or report_like_request or object_pattern_request)
        ):
            actions.append(
                ToolProgramAction(
                    tool_name="quantify_objects",
                    purpose="Convert detection outputs into measurement-ready counts and size summaries.",
                    args={"predictions_json_path": handles["prediction_json_paths"][-1]},
                )
            )
        segmentation_tool_name = (
            "segment_image_sam2"
            if "segment_image_sam2" in available
            else "segment_image_sam3"
            if "segment_image_sam3" in available
            else ""
        )
        if (
            depth_map_paths
            and segmentation_tool_name
            and (
                self._is_segmentation_request(latest_user_text)
                or (
                    self._is_depth_request(latest_user_text)
                    and (
                        quantitative_request
                        or report_like_request
                        or self._has_follow_up_reference(latest_user_text)
                    )
                )
            )
        ):
            actions.append(
                ToolProgramAction(
                    tool_name=segmentation_tool_name,
                    purpose="Segment object-like structures on the derived depth map so the depth output can be measured quantitatively.",
                    args={
                        "depth_map_paths": depth_map_paths[: self.RESEARCH_PROGRAM_IMAGE_BATCH_SIZE]
                    },
                )
            )
        if (
            mask_paths
            and "quantify_segmentation_masks" in available
            and (quantitative_request or report_like_request)
        ):
            actions.append(
                ToolProgramAction(
                    tool_name="quantify_segmentation_masks",
                    purpose="Measure coverage, connected components, and basic morphology from the available segmentation masks.",
                    args={"mask_paths": mask_paths[:8]},
                )
            )
        if (
            image_files
            and "yolo_detect" in available
            and (
                object_pattern_request
                or (
                    report_like_request
                    and quantitative_request
                    and (not plot_request or plot_prefers_detection_artifacts)
                )
            )
        ):
            actions.append(
                ToolProgramAction(
                    tool_name="yolo_detect",
                    purpose="Measure visible objects in the available scientific image files.",
                    args={"file_paths": image_files, "model_name": "yolov5_rarespot"},
                )
            )
        elif image_files and "bioio_load_image" in available:
            actions.append(
                ToolProgramAction(
                    tool_name="bioio_load_image",
                    purpose="Inspect the available image metadata and create preview handles.",
                    args={"file_path": image_files[0], "include_array": False},
                )
            )
        if (
            plot_request
            and plot_prefers_detection_artifacts
            and "plot_quantified_detections" in available
            and (handles.get("analysis_table_paths") or handles.get("prediction_json_paths"))
        ):
            action_args: dict[str, Any] = {
                "confidence_threshold": requested_confidence_threshold,
            }
            if handles.get("analysis_table_paths"):
                action_args["object_table_path"] = handles["analysis_table_paths"][-1]
            elif handles.get("prediction_json_paths"):
                action_args["predictions_json_path"] = handles["prediction_json_paths"][-1]
            if image_files:
                action_args["source_image_path"] = image_files[0]
            actions.append(
                ToolProgramAction(
                    tool_name="plot_quantified_detections",
                    purpose="Generate deterministic matplotlib/seaborn plots from the measured detection outputs and highlight low-confidence predictions.",
                    args=action_args,
                )
            )
        elif (
            handles.get("job_ids")
            and "execute_python_job" in available
            and (not plot_request or plot_prefers_detection_artifacts)
        ):
            execute_args: dict[str, Any] = {
                "job_id": handles["job_ids"][-1],
                "wait_for_completion": True,
            }
            if self._should_prefer_inline_code_execution(
                latest_user_text,
                handles=handles,
            ):
                execute_args["durable_execution"] = False
            actions.append(
                ToolProgramAction(
                    tool_name="execute_python_job",
                    purpose="Execute the prepared deterministic analysis job and harvest its measured outputs.",
                    args=execute_args,
                )
            )
        elif (
            (handles.get("prediction_json_paths") or handles.get("analysis_table_paths"))
            and "codegen_python_plan" in available
            and plot_request
            and plot_prefers_detection_artifacts
        ):
            prediction_json_path = (
                str(handles.get("prediction_json_paths", [])[-1] or "").strip()
                if handles.get("prediction_json_paths")
                else ""
            )
            analysis_table_path = (
                str(handles.get("analysis_table_paths", [])[-1] or "").strip()
                if handles.get("analysis_table_paths")
                else ""
            )
            inputs: list[dict[str, Any]] = []
            if prediction_json_path:
                inputs.append(
                    {
                        "path": prediction_json_path,
                        "kind": "file",
                        "description": "Detector predictions JSON from the prior measured analysis.",
                    }
                )
            elif analysis_table_path:
                inputs.append(
                    {
                        "path": analysis_table_path,
                        "kind": "file",
                        "description": "Per-detection quantification table from the prior measured analysis.",
                    }
                )
            if image_files:
                inputs.append(
                    {
                        "path": image_files[0],
                        "kind": "file",
                        "description": "Associated scientific image for figure labels and optional contextual annotations.",
                    }
                )
            actions.append(
                ToolProgramAction(
                    tool_name="codegen_python_plan",
                    purpose="Prepare a deterministic Python job that converts the measured detection outputs into publication-style plots and a concise quantitative summary.",
                    args={
                        "task_summary": (
                            "Load the prior measured detection outputs and produce a grounded plotting report for the prior scientific analysis. "
                            "Save result.json plus descriptive PNG figures using matplotlib or seaborn. "
                            "Include a class-count distribution plot, a confidence distribution plot, and a plot or tabular summary that highlights detections below "
                            f"{requested_confidence_threshold:.2f} confidence. "
                            "If per-detection geometry is available, summarize width, height, area, and centroid-derived spatial patterns, and report exact counts for low-confidence detections."
                        ),
                        "inputs": inputs,
                        "constraints": {
                            "expected_outputs": [
                                "result.json",
                                "plots/class_distribution.png",
                                "plots/confidence_distribution.png",
                                "plots/low_confidence_review.png",
                            ],
                            "preferred_libraries": ["matplotlib", "seaborn", "pandas"],
                            "confidence_threshold": requested_confidence_threshold,
                        },
                    },
                )
            )
        elif image_files and "codegen_python_plan" in available and plot_request:
            inputs: list[dict[str, Any]] = [
                {
                    "path": path,
                    "kind": "file",
                    "description": "Scientific image file available from the current or previous turn.",
                }
                for path in image_files[: self.RESEARCH_PROGRAM_IMAGE_BATCH_SIZE]
            ]
            code_task_summary = (
                "Load the referenced scientific image file or files and produce grounded exploratory plots and a concise quantitative summary. "
                "Use matplotlib or seaborn. Save result.json plus descriptive PNG figures. "
                "Include an intensity or value distribution plot, a channel- or image-level comparison plot when applicable, "
                "and a PCA-style or dimensionality-reduction-style summary only when it is meaningful for the available data; otherwise substitute a clearly labeled exploratory plot that explains the dominant variation without inventing unsupported structure. "
                "Report only measurements that are actually computed from the input image data."
            )
            actions.append(
                ToolProgramAction(
                    tool_name="codegen_python_plan",
                    purpose="Prepare a deterministic Python job that generates exploratory scientific plots from the available image data.",
                    args={
                        "task_summary": code_task_summary,
                        "inputs": inputs,
                        "constraints": {
                            "expected_outputs": [
                                "result.json",
                                "plots/intensity_distribution.png",
                                "plots/image_comparison.png",
                            ],
                            "preferred_libraries": [
                                "matplotlib",
                                "seaborn",
                                "numpy",
                                "pillow",
                                "scikit-image",
                            ],
                        },
                    },
                )
            )
        elif (
            handles.get("download_dirs")
            and "codegen_python_plan" in available
            and re.search(r"\b(code|python|script|parse|custom analysis|xml|csv|json)\b", lowered)
        ):
            actions.append(
                ToolProgramAction(
                    tool_name="codegen_python_plan",
                    purpose="Prepare a deterministic Python job for structured dataset analysis.",
                    args={
                        "task_summary": (
                            "Analyze the downloaded scientific dataset directory and report grounded summary statistics only. "
                            "Count files, summarize extension/type distribution, compute image dimension summaries when readable, "
                            "and report annotation or bounding-box statistics only if structured labels are actually present."
                        ),
                        "inputs": [
                            {
                                "path": handles["download_dirs"][-1],
                                "kind": "directory",
                                "description": "Downloaded scientific dataset directory.",
                            }
                        ],
                    },
                )
            )
        return ToolProgramIterationPlan(
            objective=str(latest_user_text or "").strip()
            or "Carry out the requested scientific analysis.",
            reasoning_summary="Fallback planning path selected because structured planning was unavailable.",
            actions=actions[:3],
            ready_to_answer=not actions,
            answer_outline=[],
            remaining_questions=[]
            if actions
            else ["No safe next action could be planned from the available evidence."],
        )

    def _next_research_program_image_batch(
        self,
        *,
        handles: dict[str, list[str]],
        session_state: dict[str, Any] | None,
    ) -> list[str]:
        processed = {
            str(path or "").strip()
            for path in list((session_state or {}).get("processed_image_files") or [])
            if str(path or "").strip()
        }
        current_uploaded = [
            str(path or "").strip()
            for path in list((session_state or {}).get("current_uploaded_files") or [])
            if self._is_image_like_path(path)
        ]
        prioritized = [path for path in current_uploaded if path not in processed]
        candidates = [
            str(path or "").strip()
            for path in list(handles.get("image_files") or [])
            if str(path or "").strip()
        ]
        if not candidates:
            candidates = [
                str(path or "").strip()
                for path in list(handles.get("downloaded_files") or [])
                if self._is_image_like_path(path)
            ]
        pending = [path for path in candidates if path not in processed]
        ordered: list[str] = []
        for path in [*prioritized, *pending]:
            if path and path not in ordered:
                ordered.append(path)
        if not ordered and candidates:
            for path in [*current_uploaded, *candidates]:
                if path and path not in ordered:
                    ordered.append(path)
        return ordered[: self.RESEARCH_PROGRAM_IMAGE_BATCH_SIZE]

    def _normalize_research_program_action(
        self,
        *,
        action: ToolProgramAction,
        handles: dict[str, list[str]],
        session_state: dict[str, Any] | None,
        latest_user_text: str,
        available_tool_names: list[str],
    ) -> ToolProgramAction | None:
        tool_name = str(action.tool_name or "").strip()
        if not tool_name or tool_name not in set(available_tool_names):
            return None
        args = dict(action.args or {})
        image_batch = self._next_research_program_image_batch(
            handles=handles,
            session_state=session_state,
        )

        if tool_name == "load_bisque_resource":
            if not str(args.get("resource_uri") or "").strip():
                resource_uri = next(
                    (
                        str(item or "").strip()
                        for key in ("resource_uris", "dataset_uris")
                        for item in list(handles.get(key) or [])
                        if str(item or "").strip()
                    ),
                    "",
                )
                if not resource_uri:
                    return None
                args["resource_uri"] = resource_uri
        elif tool_name == "bisque_download_dataset":
            dataset_uri = str(args.get("dataset_uri") or "").strip()
            if not dataset_uri:
                dataset_uri = next(
                    (
                        str(item or "").strip()
                        for item in list(handles.get("dataset_uris") or [])
                        if str(item or "").strip()
                    ),
                    "",
                )
                if not dataset_uri:
                    return None
                args["dataset_uri"] = dataset_uri
            args.setdefault("limit", 200)
        elif tool_name == "bisque_download_resource":
            if not str(args.get("resource_uri") or "").strip():
                resource_uri = next(
                    (
                        str(item or "").strip()
                        for item in list(handles.get("resource_uris") or [])
                        if str(item or "").strip()
                    ),
                    "",
                )
                if not resource_uri:
                    return None
                args["resource_uri"] = resource_uri
        elif tool_name == "search_bisque_resources":
            args.setdefault("limit", 10)
        elif tool_name == "bioio_load_image":
            file_path = str(args.get("file_path") or "").strip()
            if not file_path:
                file_path = image_batch[0] if image_batch else ""
                if not file_path:
                    return None
                args["file_path"] = file_path
            args.setdefault("include_array", False)
        elif tool_name == "yolo_detect":
            raw_paths = args.get("file_paths")
            if isinstance(raw_paths, (str, os.PathLike)):
                file_paths = [str(raw_paths).strip()]
            elif isinstance(raw_paths, (list, tuple, set)):
                file_paths = [
                    str(item or "").strip() for item in raw_paths if str(item or "").strip()
                ]
            else:
                file_paths = []
            prioritized_uploaded = [
                str(path or "").strip()
                for path in list((session_state or {}).get("current_uploaded_files") or [])
                if self._is_image_like_path(path)
            ]
            should_replace_with_batch = bool(
                image_batch
                and (
                    not file_paths
                    or len(file_paths) > len(image_batch)
                    or (
                        prioritized_uploaded
                        and any(path in image_batch for path in prioritized_uploaded)
                        and not any(path in file_paths for path in prioritized_uploaded)
                    )
                )
            )
            if should_replace_with_batch:
                file_paths = list(image_batch)
            if not file_paths:
                file_paths = list(image_batch)
            if not file_paths:
                return None
            args["file_paths"] = file_paths[: self.RESEARCH_PROGRAM_IMAGE_BATCH_SIZE]
            if not str(args.get("model_name") or "").strip() and re.search(
                r"\b(prairie dog|burrow)\b", str(latest_user_text or "").lower()
            ):
                args["model_name"] = "yolov5_rarespot"
        elif tool_name == "quantify_objects":
            predictions_json_path = str(args.get("predictions_json_path") or "").strip()
            if not predictions_json_path:
                predictions_json_path = next(
                    (
                        str(item or "").strip()
                        for item in reversed(list(handles.get("prediction_json_paths") or []))
                        if str(item or "").strip()
                    ),
                    "",
                )
                if not predictions_json_path:
                    return None
                args["predictions_json_path"] = predictions_json_path
        elif tool_name == "plot_quantified_detections":
            if not str(args.get("object_table_path") or "").strip():
                object_table_path = next(
                    (
                        str(item or "").strip()
                        for item in reversed(list(handles.get("analysis_table_paths") or []))
                        if str(item or "").strip()
                    ),
                    "",
                )
                if object_table_path:
                    args["object_table_path"] = object_table_path
            if (
                not str(args.get("object_table_path") or "").strip()
                and not str(args.get("predictions_json_path") or "").strip()
            ):
                predictions_json_path = next(
                    (
                        str(item or "").strip()
                        for item in reversed(list(handles.get("prediction_json_paths") or []))
                        if str(item or "").strip()
                    ),
                    "",
                )
                if not predictions_json_path:
                    return None
                args["predictions_json_path"] = predictions_json_path
            if not str(args.get("source_image_path") or "").strip():
                image_context_path = next(
                    (
                        str(item or "").strip()
                        for item in list(handles.get("image_files") or [])
                        if str(item or "").strip()
                    ),
                    "",
                )
                if image_context_path:
                    args["source_image_path"] = image_context_path
            if "confidence_threshold" not in args:
                args["confidence_threshold"] = self._requested_confidence_threshold(
                    latest_user_text
                )
        elif tool_name == "execute_python_job":
            job_id = str(args.get("job_id") or "").strip()
            if not job_id:
                job_id = next(
                    (
                        str(item or "").strip()
                        for item in reversed(list(handles.get("job_ids") or []))
                        if str(item or "").strip()
                    ),
                    "",
                )
                if not job_id:
                    return None
                args["job_id"] = job_id
            args.setdefault("wait_for_completion", True)
            if self._should_prefer_inline_code_execution(
                latest_user_text,
                handles=handles,
            ):
                args.setdefault("durable_execution", False)
        elif tool_name == "codegen_python_plan":
            inputs = list(args.get("inputs") or [])
            if not inputs:
                download_dir = next(
                    (
                        str(item or "").strip()
                        for item in reversed(list(handles.get("download_dirs") or []))
                        if str(item or "").strip()
                    ),
                    "",
                )
                if download_dir:
                    args["inputs"] = [
                        {
                            "path": download_dir,
                            "kind": "directory",
                            "description": "Downloaded scientific dataset directory.",
                        }
                    ]
                else:
                    prediction_json_path = next(
                        (
                            str(item or "").strip()
                            for item in reversed(list(handles.get("prediction_json_paths") or []))
                            if str(item or "").strip()
                        ),
                        "",
                    )
                    if prediction_json_path:
                        prepared_inputs = [
                            {
                                "path": prediction_json_path,
                                "kind": "file",
                                "description": "Detector predictions JSON from the measured analysis.",
                            }
                        ]
                        image_context_path = next(
                            (
                                str(item or "").strip()
                                for item in list(handles.get("image_files") or [])
                                if str(item or "").strip()
                            ),
                            "",
                        )
                        if image_context_path:
                            prepared_inputs.append(
                                {
                                    "path": image_context_path,
                                    "kind": "file",
                                    "description": "Associated scientific image for figure labels and contextual review.",
                                }
                            )
                        args["inputs"] = prepared_inputs
                    else:
                        analysis_table_path = next(
                            (
                                str(item or "").strip()
                                for item in reversed(
                                    list(handles.get("analysis_table_paths") or [])
                                )
                                if str(item or "").strip()
                            ),
                            "",
                        )
                        if analysis_table_path:
                            prepared_inputs = [
                                {
                                    "path": analysis_table_path,
                                    "kind": "file",
                                    "description": "Per-detection quantification table from the measured analysis.",
                                }
                            ]
                            image_context_path = next(
                                (
                                    str(item or "").strip()
                                    for item in list(handles.get("image_files") or [])
                                    if str(item or "").strip()
                                ),
                                "",
                            )
                            if image_context_path:
                                prepared_inputs.append(
                                    {
                                        "path": image_context_path,
                                        "kind": "file",
                                        "description": "Associated scientific image for figure labels and contextual review.",
                                    }
                                )
                            args["inputs"] = prepared_inputs
                        elif image_batch:
                            prepared_inputs = [
                                {
                                    "path": path,
                                    "kind": "file",
                                    "description": "Scientific image file available for deterministic plotting or quantitative analysis.",
                                }
                                for path in image_batch[: self.RESEARCH_PROGRAM_IMAGE_BATCH_SIZE]
                            ]
                            args["inputs"] = prepared_inputs
                        else:
                            return None
        return ToolProgramAction(
            tool_name=tool_name, purpose=str(action.purpose or "").strip(), args=args
        )

    def _stabilize_research_program_iteration_actions(
        self,
        actions: list[ToolProgramAction],
        *,
        session_state: dict[str, Any] | None,
        handles: dict[str, list[str]] | None,
        latest_user_text: str,
    ) -> list[ToolProgramAction]:
        normalized_actions = [
            action for action in list(actions or []) if isinstance(action, ToolProgramAction)
        ]
        if not normalized_actions:
            return []
        tool_names = [str(action.tool_name or "").strip() for action in normalized_actions]
        current_uploaded_present = any(
            str(path or "").strip()
            for path in list((session_state or {}).get("current_uploaded_files") or [])
            if str(path or "").strip()
        )
        if (
            current_uploaded_present
            and "yolo_detect" in tool_names
            and "quantify_objects" in tool_names
        ):
            return [
                action
                for action in normalized_actions
                if str(action.tool_name or "").strip() != "quantify_objects"
            ]
        if (
            self._is_plot_or_visual_analysis_request(latest_user_text)
            and self._plot_request_prefers_detection_artifacts(latest_user_text)
            and not self._is_code_execution_request(latest_user_text)
            and "plot_quantified_detections" in tool_names
            and any(
                bool(list((handles or {}).get(key) or []))
                for key in ("analysis_table_paths", "prediction_json_paths")
            )
        ):
            return [
                action
                for action in normalized_actions
                if str(action.tool_name or "").strip()
                not in {"codegen_python_plan", "execute_python_job"}
            ]
        return normalized_actions

    async def _run_tool_program_phase(
        self,
        *,
        phase_name: str,
        schema: type[BaseModel],
        prompt: str,
        fallback: BaseModel,
        tools: list[Any] | None = None,
        session_state: dict[str, Any] | None,
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        reasoning_mode: str | None,
        reasoning_effort_override: str | None = None,
        use_reasoning_agent: bool = True,
        max_runtime_seconds: int,
        debug: bool | None,
    ) -> Any:
        compression_stats: dict[str, Any] = {}

        def _build_tool_program_agent(model_builder: Callable[..., AgnoModel]) -> Agent:
            compression_manager = CompressionManager(
                model=model_builder(
                    reasoning_mode=reasoning_mode,
                    reasoning_effort_override=reasoning_effort_override,
                    max_runtime_seconds=max_runtime_seconds,
                ),
                compress_tool_results=True,
                stats=compression_stats,
            )
            return Agent(
                name=f"pro-mode-tool-program-{phase_name}",
                model=model_builder(
                    reasoning_mode=reasoning_mode,
                    reasoning_effort_override=reasoning_effort_override,
                    max_runtime_seconds=max_runtime_seconds,
                ),
                instructions=[
                    "You are operating inside a scientific workflow.",
                    "Return only the requested structured output.",
                    "Be evidence-seeking, concise, and tool-aware.",
                    "Do not invent measurements, metadata, coordinates, counts, or file handles.",
                    "If you use scratchpad tools, the only valid tool names are `think` and `analyze`; never invent aliases such as `analyze_output`.",
                ],
                output_schema=schema,
                structured_outputs=True,
                use_json_mode=True,
                parse_response=True,
                markdown=False,
                telemetry=False,
                retries=0,
                store_events=False,
                store_history_messages=False,
                tools=list(tools or []) or None,
                session_state=dict(session_state or {}),
                add_session_state_to_context=True,
                compress_tool_results=True,
                compression_manager=compression_manager,
                reasoning=bool(use_reasoning_agent),
                reasoning_max_steps=8 if use_reasoning_agent else None,
                debug_mode=bool(debug),
            )

        model_route = self._pro_mode_model_route_metadata(
            fallback_used=False,
            active_model=str(
                self._setting(self.settings, "resolved_pro_mode_model", self.model) or self.model
            ),
        )
        try:
            if self._uses_published_pro_mode_api():
                agent = _build_tool_program_agent(self._build_model)
                result = await agent.arun(
                    prompt,
                    stream=False,
                    user_id=user_id,
                    session_id=self._scope_session_id(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        run_id=f"{str(run_id or '').strip()}::{phase_name}:tool_capable_fallback"
                        if run_id
                        else f"{phase_name}:tool_capable_fallback",
                    ),
                    debug_mode=bool(debug),
                )
                model_route = self._pro_mode_model_route_metadata(
                    fallback_used=True,
                    failure_code="structured_phase_requires_tool_capable_model",
                    active_model=self.model,
                )
            else:
                result, model_route = await self._arun_with_optional_pro_mode_fallback(
                    phase_name=phase_name,
                    prompt=prompt,
                    build_agent=_build_tool_program_agent,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    debug=debug,
                )
        except Exception:
            return fallback
        if isinstance(session_state, dict):
            compression_state = dict(session_state.get("compression_stats") or {})
            compression_state[phase_name] = {
                **dict(compression_stats),
                "model_route": dict(model_route),
            }
            session_state["compression_stats"] = compression_state
        return self._coerce_schema_output(schema=schema, result=result, fallback=fallback)

    @staticmethod
    def _metadata_summary_has_content(summary: dict[str, Any] | None) -> bool:
        if not isinstance(summary, dict):
            return False
        for key in ("header", "exif", "geo", "filename_hints", "captured_at", "dimensions", "tags"):
            value = summary.get(key)
            if isinstance(value, dict) and value:
                return True
            if isinstance(value, list) and value:
                return True
            if isinstance(value, str) and value.strip():
                return True
        return False

    @staticmethod
    def _decimal_to_dms(value: float, *, positive_label: str, negative_label: str) -> str:
        absolute = abs(float(value))
        degrees = int(absolute)
        minutes_float = (absolute - degrees) * 60.0
        minutes = int(minutes_float)
        seconds = (minutes_float - minutes) * 60.0
        label = positive_label if value >= 0 else negative_label
        return f"{degrees}° {minutes}' {seconds:.2f}\" {label}"

    @staticmethod
    def _metadata_field_explanations() -> list[dict[str, str]]:
        return [
            {"field": "Make", "meaning": "manufacturer label recorded by the file"},
            {"field": "Model", "meaning": "device or camera model label recorded by the file"},
            {
                "field": "Software",
                "meaning": "software or firmware label that last wrote the metadata",
            },
            {
                "field": "DateTimeOriginal",
                "meaning": "original capture timestamp when the file provides it",
            },
            {
                "field": "DateTime",
                "meaning": "file timestamp written by the device or software; it may differ from capture time",
            },
            {
                "field": "latitude/longitude",
                "meaning": "embedded GPS coordinates if the file contains geotags",
            },
            {
                "field": "altitude_m",
                "meaning": "embedded altitude value from the file metadata when available",
            },
        ]

    @staticmethod
    def _reverse_geocode_coordinates(latitude: float, longitude: float) -> dict[str, Any]:
        query = urlencode(
            {
                "format": "jsonv2",
                "lat": f"{float(latitude):.8f}",
                "lon": f"{float(longitude):.8f}",
                "zoom": "10",
                "addressdetails": "1",
            }
        )
        request = Request(
            f"https://nominatim.openstreetmap.org/reverse?{query}",
            headers={
                "User-Agent": "bisque-ultra/metadata-specialist",
                "Accept": "application/json",
            },
        )
        with urlopen(request, timeout=3.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            return {}
        address = payload.get("address") if isinstance(payload.get("address"), dict) else {}
        coarse_parts = [
            str(address.get(key) or "").strip()
            for key in ("hamlet", "village", "town", "city", "county", "state", "country")
            if str(address.get(key) or "").strip()
        ]
        coarse_location = ", ".join(dict.fromkeys(coarse_parts))
        return {
            "coarse_location": coarse_location or str(payload.get("display_name") or "").strip(),
            "display_name": str(payload.get("display_name") or "").strip(),
            "source": "openstreetmap_nominatim",
        }

    async def _metadata_location_enrichment(self, geo: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(geo, dict):
            return {}
        latitude = geo.get("latitude")
        longitude = geo.get("longitude")
        if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
            return {}
        enrichment: dict[str, Any] = {
            "decimal_coordinates": f"{float(latitude):.8f}, {float(longitude):.8f}",
            "dms_coordinates": {
                "latitude": self._decimal_to_dms(
                    float(latitude), positive_label="N", negative_label="S"
                ),
                "longitude": self._decimal_to_dms(
                    float(longitude), positive_label="E", negative_label="W"
                ),
            },
        }
        altitude = geo.get("altitude_m")
        if isinstance(altitude, (int, float)):
            enrichment["altitude_m"] = float(altitude)
        try:
            reverse_payload = await asyncio.to_thread(
                self._reverse_geocode_coordinates,
                float(latitude),
                float(longitude),
            )
        except Exception:
            reverse_payload = {}
        if isinstance(reverse_payload, dict) and reverse_payload:
            enrichment["reverse_geocode"] = reverse_payload
        return enrichment

    def _build_metadata_specialist_packet(
        self,
        *,
        tool_invocations: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        image_summary: dict[str, Any] | None = None
        resource_summary: dict[str, Any] | None = None
        for invocation in list(tool_invocations or []):
            summary = invocation.get("output_summary")
            if not isinstance(summary, dict) or not summary.get("success", True):
                continue
            tool_name = str(invocation.get("tool") or "").strip()
            if tool_name == "bioio_load_image":
                image_summary = summary
            elif tool_name == "load_bisque_resource":
                resource_summary = summary
        if not self._metadata_summary_has_content(
            image_summary
        ) and not self._metadata_summary_has_content(resource_summary):
            return None
        image_file_path = str((image_summary or {}).get("file_path") or "").strip()
        display_name = self._display_artifact_name(image_file_path) if image_file_path else ""
        if not display_name:
            display_name = str((resource_summary or {}).get("resource_name") or "").strip()
        return {
            "file_name": display_name or None,
            "image_metadata": dict(image_summary or {}),
            "bisque_resource": dict(resource_summary or {}),
            "field_explanations": self._metadata_field_explanations(),
        }

    def _should_run_metadata_specialist(
        self,
        *,
        latest_user_text: str,
        tool_invocations: list[dict[str, Any]],
    ) -> bool:
        lowered = str(latest_user_text or "").strip().lower()
        if self._is_image_metadata_request(latest_user_text):
            return True
        if re.search(r"\b(exif|gps|location|geotag|metadata|header|filename|captured)\b", lowered):
            return True
        if self._is_report_like_request(latest_user_text):
            return any(
                str(invocation.get("tool") or "").strip()
                in {"bioio_load_image", "load_bisque_resource"}
                for invocation in list(tool_invocations or [])
            )
        return False

    async def _run_metadata_specialist(
        self,
        *,
        latest_user_text: str,
        tool_invocations: list[dict[str, Any]],
        session_state: dict[str, Any] | None,
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        max_runtime_seconds: int,
        debug: bool | None,
    ) -> MetadataSpecialistSummary:
        evidence_packet = self._build_metadata_specialist_packet(tool_invocations=tool_invocations)
        fallback = MetadataSpecialistSummary()
        if evidence_packet is None:
            return fallback
        image_metadata = (
            evidence_packet.get("image_metadata")
            if isinstance(evidence_packet.get("image_metadata"), dict)
            else {}
        )
        geo = image_metadata.get("geo") if isinstance(image_metadata.get("geo"), dict) else {}
        if isinstance(geo, dict) and geo:
            enrichment = await self._metadata_location_enrichment(geo)
            if enrichment:
                evidence_packet["location_enrichment"] = enrichment
        prompt = "\n".join(
            [
                "You are the metadata specialist for a scientific image workflow.",
                "Summarize only metadata that is explicitly present in the evidence packet.",
                "Separate embedded image metadata from BisQue upload metadata and from filename-derived hints.",
                "If EXIF GPS or other location metadata is absent, say that clearly.",
                "When technical fields appear, define them in plain language the first time they matter.",
                "If the file reports Make, Model, or Software, explain them as manufacturer/device/software identifiers unless the evidence packet supports a stronger statement.",
                "If latitude and longitude are present, you may mention any derived place name from the location_enrichment block, but label it clearly as mapped from the coordinates rather than stored in the file.",
                "Filename hints may encode scientific context, but they are hints, not embedded metadata.",
                "Do not infer geolocation from filenames, scene names, or run identifiers.",
                "",
                f"User request: {latest_user_text}",
                "",
                "Evidence packet:",
                json.dumps(evidence_packet, ensure_ascii=False, indent=2),
            ]
        )
        return await self._run_tool_program_phase(
            phase_name="metadata_specialist",
            schema=MetadataSpecialistSummary,
            prompt=prompt,
            fallback=fallback,
            session_state=session_state,
            conversation_id=conversation_id,
            run_id=run_id,
            user_id=user_id,
            reasoning_mode="fast",
            reasoning_effort_override="low",
            max_runtime_seconds=min(max_runtime_seconds, 45),
            debug=debug,
        )

    @staticmethod
    def _fallback_report_packet(
        *,
        synthesis: ToolProgramSynthesis,
        evidence_summaries: list[str],
    ) -> ToolProgramReportPacket:
        response_text = str(synthesis.response_text or "").strip()
        paragraphs = [
            part.strip() for part in response_text.split("\n\n") if str(part or "").strip()
        ]
        lead = paragraphs[0] if paragraphs else response_text
        measured_findings = [
            item for item in list(synthesis.evidence_basis or []) if str(item or "").strip()
        ]
        limitations = [
            item for item in list(synthesis.unresolved_points or []) if str(item or "").strip()
        ]
        if not measured_findings:
            measured_findings = [
                str(item or "").strip()
                for item in list(evidence_summaries or [])[-6:]
                if str(item or "").strip()
            ]
        return ToolProgramReportPacket(
            direct_answer=lead,
            executive_summary=([lead] if lead else []),
            measured_findings=measured_findings[:8],
            comparative_findings=[],
            interpretation=paragraphs[1:3],
            limitations=limitations[:6],
            recommended_next_steps=[],
        )

    async def _run_research_report_writer(
        self,
        *,
        latest_user_text: str,
        report_packet: ToolProgramReportPacket,
        evidence_summaries: list[str],
        handles: dict[str, Any],
        session_state: dict[str, Any] | None,
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        max_runtime_seconds: int,
        debug: bool | None,
    ) -> str:
        compression_stats: dict[str, Any] = {}

        def _build_report_writer_agent(model_builder: Callable[..., AgnoModel]) -> Agent:
            compression_manager = CompressionManager(
                model=model_builder(
                    reasoning_mode="fast",
                    reasoning_effort_override="low",
                    max_runtime_seconds=max_runtime_seconds,
                ),
                compress_tool_results=True,
                stats=compression_stats,
            )
            return Agent(
                name="pro-mode-report-writer",
                model=model_builder(
                    reasoning_mode="fast",
                    reasoning_effort_override="low",
                    max_runtime_seconds=max_runtime_seconds,
                ),
                reasoning_model=model_builder(
                    reasoning_mode="deep",
                    reasoning_effort_override="high",
                    max_runtime_seconds=max_runtime_seconds,
                ),
                instructions=[
                    "You write comprehensive, scientist-facing reports from measured evidence.",
                    "Lead with the strongest supported takeaway, then organize the report into clear sections when helpful.",
                    "Prefer connected prose with a few short headings over bullet spam.",
                    "Ground every quantitative statement in the provided packet or evidence summaries.",
                    "When perturbation-stability results and detector confidence scores are both present, explain them as complementary but different signals.",
                    "Stability describes robustness of the prediction set under perturbation; confidence describes per-detection model scoring and is not a calibrated probability.",
                    "Do not frame a zero instability score as proof that every detection is correct.",
                    "Do not dump raw per-box confidence lists unless the user explicitly asks for every confidence value.",
                    "When multiple images or artifacts are being compared, discuss the most important differences explicitly.",
                    "State limitations plainly, but do not let them crowd out the main findings.",
                    "Do not mention internal workflows, canonical branches, blockers, council roles, or tool names.",
                    "Do not invent measurements, coordinates, acquisition metadata, species presence, or counts.",
                ],
                markdown=True,
                telemetry=False,
                retries=0,
                store_events=False,
                store_history_messages=False,
                session_state=dict(session_state or {}),
                add_session_state_to_context=True,
                compress_tool_results=True,
                compression_manager=compression_manager,
                reasoning=True,
                reasoning_min_steps=2,
                reasoning_max_steps=10,
                debug_mode=bool(debug),
            )

        prompt = "\n".join(
            [
                "Write the final scientist-facing report from the structured evidence below.",
                "If the user asked for a report, make it substantive and complete, not a stub.",
                "Use only supported measurements from the packet and evidence summaries.",
                "If comparisons across multiple images are available, include them.",
                "If the evidence is partial, say so cleanly in a limitations section and continue with the best supported report.",
                "",
                f"User request: {latest_user_text}",
                "",
                "Structured report packet:",
                json.dumps(report_packet.model_dump(mode="json"), ensure_ascii=False, indent=2),
                "",
                "Compressed evidence summaries:",
                json.dumps(list(evidence_summaries or [])[-12:], ensure_ascii=False, indent=2),
                "",
                "Relevant handles:",
                json.dumps(handles, ensure_ascii=False, indent=2),
            ]
        )
        model_route = self._pro_mode_model_route_metadata(
            fallback_used=False,
            active_model=str(
                self._setting(self.settings, "resolved_pro_mode_model", self.model) or self.model
            ),
        )
        try:
            result, model_route = await asyncio.wait_for(
                self._arun_text_phase_with_optional_pro_mode_transport(
                    phase_name="research_program_report",
                    prompt=prompt,
                    build_agent=_build_report_writer_agent,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    debug=debug,
                    reasoning_mode="fast",
                    reasoning_effort_override="low",
                    max_runtime_seconds=max_runtime_seconds,
                ),
                timeout=max(10.0, min(float(max_runtime_seconds), 60.0)),
            )
        except Exception:
            return ""
        if isinstance(session_state, dict):
            compression_state = dict(session_state.get("compression_stats") or {})
            compression_state["research_program_report_writer"] = {
                **dict(compression_stats),
                "model_route": dict(model_route),
            }
            session_state["compression_stats"] = compression_state
        response_text, _source = self._coerce_visible_output(result)
        return str(response_text or "").strip()

    async def _run_pro_mode_final_writer(
        self,
        *,
        latest_user_text: str,
        draft_response_text: str,
        execution_regime: str,
        task_regime: str | None,
        supporting_points: list[str] | None,
        reservations: list[str] | None,
        session_state: dict[str, Any] | None,
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        max_runtime_seconds: int,
        debug: bool | None,
    ) -> tuple[str, dict[str, Any]]:
        normalized_draft = str(draft_response_text or "").strip()
        if not normalized_draft:
            return "", {}
        code_execution_failed = any(
            "Code execution failed in this turn." in str(item or "")
            for item in list(reservations or [])
        )
        code_execution_turn = self._is_code_execution_request(latest_user_text) or any(
            "code execution" in str(item or "").strip().lower() for item in list(reservations or [])
        )
        math_explainer_request = self._is_math_explanation_request(latest_user_text)
        proof_writer_instructions: list[str] = []
        report_writer_instructions: list[str] = []
        math_writer_instructions: list[str] = []
        codeexec_writer_instructions: list[str] = []
        if (
            execution_regime == "proof_workflow"
            or str(task_regime or "").strip().lower() == "rigorous_proof"
        ):
            proof_writer_instructions = [
                "For proofs, organize the answer pedagogically: verdict or current status first, then the main reduction, then the load-bearing steps, then the endgame or remaining gap.",
                "Make dependency order explicit, and do not skip from a local lemma to the final theorem without naming the bridge.",
            ]
        if self._is_report_like_request(latest_user_text):
            report_writer_instructions = [
                "For report-style answers, make the response feel like a substantive professional report, not a generic summary.",
                "Use short section headings when they improve scanability, and make the executive takeaway explicit near the top.",
                "Name the most important distinctions, comparisons, tradeoffs, and limitations rather than listing isolated facts.",
                "If the topic is broad, synthesize the landscape into a clear structure instead of repeating encyclopedia-style bullets.",
                "Write in a clarity-first explanatory mode: idea-first paragraphs, clean transitions, concrete examples when they clarify, and contrastive framing when distinctions matter.",
            ]
        if math_explainer_request:
            math_writer_instructions = [
                "For mathematically explanatory answers, keep the equations but make them teachable and selective.",
                "Introduce notation before use or at first use, and make symbol meanings easy to recover.",
                "After each important equation, add a brief interpretation that tells the reader what the equation is doing and why it matters.",
                "Prefer a smaller set of load-bearing equations over a long catalog of formulas.",
                "Treat the reader as a strong student: smart and technical, but not already in possession of the entire conceptual map.",
                "Turn brittle prescriptions into conditional guidance with assumptions and scope made explicit.",
            ]
        if code_execution_turn:
            codeexec_writer_instructions = [
                "For code-execution answers, separate: methods used, key quantitative findings, interpretation, and limitations.",
                "Name the exact artifact classes produced, such as CSV, PNG, JSON, or report outputs, when they exist.",
                "Prefer technically literate prose for PhD-level readers: concise, explicit, and evidence-led.",
                "If a caveat materially changes the conclusion, place it in the same paragraph as the claim.",
            ]
        scientific_result_surface_active = execution_regime == "tool_workflow"
        explicit_full_chat_report = bool(
            re.search(
                r"\b(full|detailed|complete|manuscript[- ]style)\b.{0,24}\b(report|write-up|writeup)\b",
                str(latest_user_text or ""),
                re.IGNORECASE | re.DOTALL,
            )
        )

        def _compact_scientific_surface_response(text: str) -> str:
            normalized = str(text or "").strip()
            if not normalized:
                return ""
            blocks = [
                block.strip()
                for block in re.split(r"\n\s*\n+", normalized)
                if str(block or "").strip()
            ]
            preferred_blocks: list[str] = []
            for block in blocks:
                stripped = block.strip()
                if not stripped:
                    continue
                if stripped.startswith("```"):
                    continue
                if re.match(r"^#{1,6}\s+", stripped):
                    continue
                if re.match(r"^\s*(?:[-*+]\s|\d+\.\s)", stripped):
                    continue
                if "|" in stripped and "\n" in stripped:
                    continue
                preferred_blocks.append(stripped)
                if len(preferred_blocks) >= 2:
                    break
            selected_blocks = preferred_blocks or blocks[:2]
            return "\n\n".join(selected_blocks[:2]).strip()
        compression_stats: dict[str, Any] = {}

        def _build_final_writer_agent(model_builder: Callable[..., AgnoModel]) -> Agent:
            compression_manager = CompressionManager(
                model=model_builder(
                    reasoning_mode="fast",
                    reasoning_effort_override="low",
                    max_runtime_seconds=max_runtime_seconds,
                ),
                compress_tool_results=True,
                stats=compression_stats,
            )
            return Agent(
                name="pro-mode-final-writer",
                model=model_builder(
                    reasoning_mode="fast",
                    reasoning_effort_override="low",
                    max_runtime_seconds=max_runtime_seconds,
                ),
                reasoning_model=model_builder(
                    reasoning_mode="deep",
                    reasoning_effort_override="high",
                    max_runtime_seconds=max_runtime_seconds,
                ),
                output_model=model_builder(
                    reasoning_mode="fast",
                    reasoning_effort_override="low",
                    max_runtime_seconds=max_runtime_seconds,
                ),
                output_model_prompt=(
                    "Refine the draft into lucid, elegant, scientist-facing prose. "
                    "Preserve every supported conclusion, number, caveat, and comparison. "
                    "Improve coherence, structure, and sentence flow. "
                    "Do not add facts. Do not imitate any named living writer. "
                    "Instead, use high-level prose qualities such as clarity, structure, concrete illustration, and controlled stylistic energy."
                ),
                instructions=[
                    "You are the final Pro Mode writer for scientifically grounded answers.",
                    "Lead with the answer or strongest supported takeaway.",
                    "Prefer connected prose with a crisp derivation or explanation when helpful.",
                    "Use short sections only when they genuinely improve readability.",
                    "Keep caveats brief and accurate.",
                    "Preserve all supported quantitative values, identities, and conditions from the draft.",
                    "If both perturbation-stability and detector confidence are mentioned, preserve the distinction: stability is robustness of the prediction set, while confidence is a per-detection score and not a calibrated probability.",
                    "Do not enumerate long raw confidence lists unless the user explicitly asked for them.",
                    "Do not invent facts, counts, coordinates, metadata, mechanisms, or citations.",
                    "Do not mention internal workflows, routes, councils, tools, blockers, or hidden reasoning.",
                    "Do not imitate any named living writer; aim instead for clear, elegant, idea-first prose.",
                    "Use disciplined explanatory nonfiction techniques rather than generic assistant prose.",
                    *(
                        [
                            "A structured scientific result surface with figures and tables will be shown separately in the UI for this turn.",
                            "Do not restate figure captions, metric tables, or methods sections in prose unless the user explicitly asked for a full written report.",
                            "For tool-backed scientific analysis, prefer one short takeaway paragraph and, if needed, one short caveat or next-step paragraph.",
                        ]
                        if scientific_result_surface_active
                        else []
                    ),
                    *[f"Technique: {item}" for item in PROSE_STYLE_GUIDELINES],
                    *[f"Student technique: {item}" for item in STUDENT_EXPLANATION_GUIDELINES],
                    *report_writer_instructions,
                    *math_writer_instructions,
                    *codeexec_writer_instructions,
                    *proof_writer_instructions,
                ],
                markdown=True,
                telemetry=False,
                retries=0,
                store_events=False,
                store_history_messages=False,
                session_state=dict(session_state or {}),
                add_session_state_to_context=True,
                compress_tool_results=True,
                compression_manager=compression_manager,
                reasoning=True,
                reasoning_min_steps=2,
                reasoning_max_steps=8,
                debug_mode=bool(debug),
            )

        prompt = build_pro_mode_final_writer_prompt(
            latest_user_text=latest_user_text,
            execution_regime=execution_regime,
            task_regime=task_regime,
            normalized_draft=normalized_draft,
            supporting_points=supporting_points,
            reservations=reservations,
            scientific_result_surface_active=scientific_result_surface_active,
            explicit_full_chat_report=explicit_full_chat_report,
            report_like_request=self._is_report_like_request(latest_user_text),
            math_explainer_request=math_explainer_request,
            code_execution_turn=code_execution_turn,
            code_execution_failed=code_execution_failed,
        )
        model_route = self._pro_mode_model_route_metadata(
            fallback_used=False,
            active_model=str(
                self._setting(self.settings, "resolved_pro_mode_model", self.model) or self.model
            ),
        )
        try:
            result, model_route = await asyncio.wait_for(
                self._arun_text_phase_with_optional_pro_mode_transport(
                    phase_name="pro_mode_final_writer",
                    prompt=prompt,
                    build_agent=_build_final_writer_agent,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    debug=debug,
                    reasoning_mode="fast",
                    reasoning_effort_override="low",
                    max_runtime_seconds=max_runtime_seconds,
                ),
                timeout=max(10.0, min(float(max_runtime_seconds), 45.0)),
            )
        except Exception:
            return "", {}
        if isinstance(session_state, dict):
            compression_state = dict(session_state.get("compression_stats") or {})
            compression_state["pro_mode_final_writer"] = {
                **dict(compression_stats),
                "model_route": dict(model_route),
            }
            session_state["compression_stats"] = compression_state
        response_text, _source = self._coerce_visible_output(result)
        normalized_response = str(response_text or "").strip()
        if not normalized_response:
            return "", {
                **dict(compression_stats),
                "model_route": dict(model_route),
            }
        if self._report_text_has_internal_orchestration(normalized_response):
            return "", {
                **dict(compression_stats),
                "model_route": dict(model_route),
            }
        if (
            re.search(r"\d", normalized_draft)
            and not re.search(r"\d", normalized_response)
            and not code_execution_failed
        ):
            return "", {
                **dict(compression_stats),
                "model_route": dict(model_route),
            }
        if self._classify_pro_mode_failure(normalized_response):
            return "", {
                **dict(compression_stats),
                "model_route": dict(model_route),
            }
        if scientific_result_surface_active and not explicit_full_chat_report:
            normalized_response = _compact_scientific_surface_response(normalized_response)
        return normalized_response, {
            **dict(compression_stats),
            "model_route": dict(model_route),
        }

    async def _execute_tool_program_actions(
        self,
        *,
        phase: str,
        phase_label: str,
        actions: list[ToolProgramAction],
        uploaded_files: list[str],
        latest_user_text: str,
        selection_context: dict[str, Any] | None,
        event_callback: Callable[[dict[str, Any]], None] | None,
        latest_result_refs_seed: dict[str, Any] | None = None,
        request_bisque_auth: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        normalized_actions = [
            action for action in list(actions or []) if str(action.tool_name or "").strip()
        ]
        if not normalized_actions:
            return []

        results: list[dict[str, Any]] = []
        latest_result_refs: dict[str, Any] = {
            str(key): value
            for key, value in dict(latest_result_refs_seed or {}).items()
            if value is not None
        }
        for action in normalized_actions:
            tool_name = str(action.tool_name or "").strip()
            args = self._hydrate_tool_program_action_args(
                tool_name=tool_name,
                args=dict(action.args or {}),
                latest_user_text=latest_user_text,
                uploaded_files=uploaded_files,
                selection_context=selection_context,
                latest_result_refs=latest_result_refs,
            )
            self._emit_event(
                event_callback,
                {
                    "kind": "graph",
                    "event_type": "pro_mode.tool_requested",
                    "phase": phase,
                    "status": "started",
                    "message": action.purpose,
                    "payload": {"tool": tool_name, "args": args},
                },
            )
            context_token = (
                set_request_bisque_auth(request_bisque_auth) if request_bisque_auth else None
            )
            try:
                raw_output = await asyncio.to_thread(
                    execute_tool_call,
                    tool_name,
                    args,
                    uploaded_files=list(uploaded_files or []),
                    user_text=latest_user_text,
                    latest_result_refs=dict(latest_result_refs),
                    selection_context=dict(selection_context or {}),
                )
                parsed_output, raw_text = self._json_result(raw_output)
                if isinstance(parsed_output, dict):
                    refs = parsed_output.get("latest_result_refs")
                    if isinstance(refs, dict):
                        for key, value in refs.items():
                            if value is None:
                                continue
                            latest_result_refs[str(key)] = value
                    if tool_name == "codegen_python_plan":
                        job_id = str(parsed_output.get("job_id") or "").strip()
                        if job_id:
                            latest_result_refs["latest_code_execution_job_id"] = job_id
                            latest_result_refs["codegen_python_plan.job_id"] = job_id
                invocation = {
                    "tool": tool_name,
                    "status": "completed",
                    "args": args,
                    "purpose": action.purpose,
                    "output_envelope": parsed_output if isinstance(parsed_output, dict) else {},
                    "output_summary": self._summarize_tool_output(
                        tool_name, parsed_output, raw_text
                    ),
                    "output_preview": raw_text[:4000],
                }
                event_status = "completed"
                event_message = f"{tool_name} completed for the {phase_label}."
                event_payload: dict[str, Any] = {
                    "tool": tool_name,
                    "summary": invocation["output_summary"],
                }
            except Exception as exc:
                invocation = {
                    "tool": tool_name,
                    "status": "failed",
                    "args": args,
                    "purpose": action.purpose,
                    "output_envelope": {},
                    "output_summary": {
                        "success": False,
                        "error": str(exc or exc.__class__.__name__),
                    },
                    "output_preview": str(exc or exc.__class__.__name__),
                }
                event_status = "failed"
                event_message = f"{tool_name} failed during the {phase_label}."
                event_payload = {"tool": tool_name, "error": str(exc or exc.__class__.__name__)}
            finally:
                if context_token is not None:
                    reset_request_bisque_auth(context_token)
            results.append(invocation)
            self._emit_event(
                event_callback,
                {
                    "kind": "graph",
                    "event_type": "pro_mode.tool_completed",
                    "phase": phase,
                    "status": event_status,
                    "message": event_message,
                    "payload": event_payload,
                },
            )
        return results

    @staticmethod
    def _codeexec_fallback_inputs(
        *,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        candidates: list[str] = []
        candidates.extend(str(path or "").strip() for path in list(uploaded_files or []))
        if isinstance(selection_context, dict):
            for key in (
                "selected_files",
                "selected_file_paths",
                "file_paths",
                "local_file_paths",
                "uploaded_files",
            ):
                value = selection_context.get(key)
                if isinstance(value, list):
                    candidates.extend(str(path or "").strip() for path in value)
        inputs: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_path in candidates:
            path_value = str(raw_path or "").strip()
            if not path_value or path_value in seen:
                continue
            source_path = Path(path_value).expanduser()
            if not source_path.exists():
                continue
            seen.add(path_value)
            inputs.append(
                {
                    "path": str(source_path),
                    "kind": ("directory" if source_path.is_dir() else "file"),
                    "description": "Selected input carried into deterministic code-execution fallback.",
                }
            )
        return inputs

    def _hydrate_tool_program_action_args(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        latest_user_text: str,
        uploaded_files: list[str],
        selection_context: dict[str, Any] | None,
        latest_result_refs: dict[str, Any],
    ) -> dict[str, Any]:
        hydrated = dict(args or {})
        if tool_name == "codegen_python_plan":
            if not str(hydrated.get("task_summary") or "").strip():
                hydrated["task_summary"] = str(latest_user_text or "").strip()
            if not isinstance(hydrated.get("inputs"), list):
                inputs = self._codeexec_fallback_inputs(
                    uploaded_files=uploaded_files,
                    selection_context=selection_context,
                )
                if inputs:
                    hydrated["inputs"] = inputs
        elif tool_name == "execute_python_job":
            if not str(hydrated.get("job_id") or "").strip():
                job_id = (
                    str(latest_result_refs.get("latest_code_execution_job_id") or "").strip()
                    or str(latest_result_refs.get("codegen_python_plan.job_id") or "").strip()
                )
                if job_id:
                    hydrated["job_id"] = job_id
        return hydrated

    async def _execute_research_program_actions(
        self,
        *,
        family: str,
        actions: list[ToolProgramAction],
        uploaded_files: list[str],
        latest_user_text: str,
        selection_context: dict[str, Any] | None,
        event_callback: Callable[[dict[str, Any]], None] | None,
    ) -> list[dict[str, Any]]:
        family_actions = [
            action
            for action in list(actions or [])
            if self._research_program_tool_family(action.tool_name) == family
        ]
        return await self._execute_tool_program_actions(
            phase="research_program",
            phase_label="iterative research program",
            actions=family_actions,
            uploaded_files=uploaded_files,
            latest_user_text=latest_user_text,
            selection_context=selection_context,
            event_callback=event_callback,
        )

    async def _run_pro_mode_research_program_workflow(
        self,
        *,
        messages: list[dict[str, Any]],
        latest_user_text: str,
        uploaded_files: list[str],
        max_runtime_seconds: int,
        reasoning_mode: str | None,
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
        event_callback: Callable[[dict[str, Any]], None] | None,
        selected_tool_names: list[str],
        selection_context: dict[str, Any] | None,
        conversation_state_seed: dict[str, Any] | None,
        debug: bool | None,
    ) -> dict[str, Any]:
        tool_names = [
            tool_name
            for tool_name in self._normalize_selected_tool_names(selected_tool_names)
            if tool_name in RESEARCH_PROGRAM_SAFE_TOOL_NAMES
        ]
        if not tool_names:
            tool_names = [
                tool_name
                for tool_name in self._research_program_tool_bundle(
                    user_text=latest_user_text,
                    uploaded_files=uploaded_files,
                    selection_context=selection_context,
                    prior_pro_mode_state=conversation_state_seed,
                )
                if tool_name in TOOL_SCHEMA_MAP
            ]
        phase_timings: dict[str, float] = {}
        tool_invocations: list[dict[str, Any]] = []
        local_state: dict[str, Any] = {}
        max_iterations = 4 if int(max_runtime_seconds) >= 1800 else 3
        step_cap = max(60, min(int(max_runtime_seconds // 3), 900))

        def state_from_input(step_input: StepInput | None = None) -> dict[str, Any]:
            if step_input is not None and step_input.workflow_session is not None:
                session_data = step_input.workflow_session.session_data or {}
                step_input.workflow_session.session_data = session_data
                state = session_data.setdefault("research_program_state", {})
                if isinstance(state, dict):
                    local_state.clear()
                    local_state.update(state)
                    return state
            return local_state

        async def initialize(step_input: StepInput) -> StepOutput:
            state = state_from_input(step_input)
            initial_state = self._initial_research_program_state(
                latest_user_text=latest_user_text,
                messages=messages,
                tool_names=tool_names,
                uploaded_files=uploaded_files,
                selection_context=selection_context,
                existing_state=dict(state),
                seed_state=conversation_state_seed,
            )
            state.clear()
            state.update(initial_state)
            self._emit_event(
                event_callback,
                {
                    "kind": "graph",
                    "event_type": "pro_mode.phase_started",
                    "phase": "research_program",
                    "status": "started",
                    "message": "Starting the iterative scientific evidence program.",
                    "payload": {
                        "selected_tool_names": tool_names,
                        "requirements": dict(state.get("requirements") or {}),
                        "seeded_handles": dict(state.get("handles") or {}),
                    },
                },
            )
            return StepOutput(content={"selected_tool_names": tool_names})

        async def plan_iteration(step_input: StepInput) -> StepOutput:
            state = state_from_input(step_input)
            state["iteration"] = int(state.get("iteration") or 0) + 1
            handles = dict(state.get("handles") or {})
            requirements = dict(state.get("requirements") or {})
            prompt = self._tool_program_plan_prompt(
                latest_user_text=latest_user_text,
                recent_context=str(state.get("recent_context") or ""),
                tool_catalog=list(state.get("tool_catalog") or []),
                evidence_summaries=list(state.get("evidence_summaries") or []),
                handles=handles,
                iteration=int(state.get("iteration") or 1),
            )
            fallback = self._fallback_tool_program_plan(
                latest_user_text=latest_user_text,
                handles=handles,
                available_tool_names=tool_names,
            )
            needs_new_uploaded_artifact_measurement = bool(
                int(state.get("iteration") or 0) == 1
                and list(handles.get("image_files") or [])
                and "vision"
                in {
                    str(item or "").strip()
                    for item in list(requirements.get("required_families") or [])
                    if str(item or "").strip()
                }
            )
            should_bootstrap_from_fallback = bool(
                int(state.get("iteration") or 0) == 1
                and (
                    needs_new_uploaded_artifact_measurement
                    or not list(state.get("evidence_summaries") or [])
                )
                and any(
                    bool(list(handles.get(key) or []))
                    for key in ("dataset_uris", "resource_uris", "image_files", "downloaded_files")
                )
                and not self._research_program_has_enough_evidence(
                    evidence_summaries=(
                        list(state.get("evidence_summaries") or [])
                        if needs_new_uploaded_artifact_measurement
                        else []
                    ),
                    executed_families=list(state.get("executed_families") or []),
                    requirements=requirements,
                )
            )
            if should_bootstrap_from_fallback:
                plan = fallback
                phase_timings[f"research_program_plan_{state['iteration']}"] = 0.0
            else:
                started = time.monotonic()
                plan = await self._run_tool_program_phase(
                    phase_name="research_program_plan",
                    schema=ToolProgramIterationPlan,
                    prompt=prompt,
                    fallback=fallback,
                    session_state=state,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    reasoning_mode="deep"
                    if str(reasoning_mode or "deep").strip().lower() != "fast"
                    else "auto",
                    max_runtime_seconds=step_cap,
                    debug=debug,
                )
                phase_timings[f"research_program_plan_{state['iteration']}"] = round(
                    time.monotonic() - started, 3
                )
            sanitized_actions: list[ToolProgramAction] = []
            for action in list(plan.actions or []):
                normalized_action = self._normalize_research_program_action(
                    action=action,
                    handles=handles,
                    session_state=state,
                    latest_user_text=latest_user_text,
                    available_tool_names=tool_names,
                )
                if normalized_action is not None:
                    sanitized_actions.append(normalized_action)
                if len(sanitized_actions) >= 3:
                    break
            sanitized_actions = self._stabilize_research_program_iteration_actions(
                sanitized_actions,
                session_state=state,
                handles=handles,
                latest_user_text=latest_user_text,
            )
            if not sanitized_actions and not plan.ready_to_answer:
                plan = fallback
                sanitized_actions = []
                for action in list(plan.actions or []):
                    normalized_action = self._normalize_research_program_action(
                        action=action,
                        handles=handles,
                        session_state=state,
                        latest_user_text=latest_user_text,
                        available_tool_names=tool_names,
                    )
                    if normalized_action is not None:
                        sanitized_actions.append(normalized_action)
                    if len(sanitized_actions) >= 3:
                        break
                sanitized_actions = self._stabilize_research_program_iteration_actions(
                    sanitized_actions,
                    session_state=state,
                    handles=handles,
                    latest_user_text=latest_user_text,
                )
            plan.actions = sanitized_actions
            state["current_plan"] = plan.model_dump(mode="json")
            return StepOutput(content=plan.model_dump(mode="json"))

        def _family_step(family: str) -> Step:
            async def _executor(step_input: StepInput) -> StepOutput:
                state = state_from_input(step_input)
                plan = ToolProgramIterationPlan.model_validate(
                    dict(state.get("current_plan") or {})
                )
                started = time.monotonic()
                results = await self._execute_research_program_actions(
                    family=family,
                    actions=list(plan.actions or []),
                    uploaded_files=uploaded_files,
                    latest_user_text=latest_user_text,
                    selection_context=selection_context,
                    event_callback=event_callback,
                )
                phase_timings[f"research_program_{family}_{state.get('iteration') or 0}"] = round(
                    time.monotonic() - started,
                    3,
                )
                return StepOutput(content={"family": family, "results": results})

            return Step(name=f"research_program_{family}", executor=_executor)

        async def collect_iteration(step_input: StepInput) -> StepOutput:
            state = state_from_input(step_input)
            new_results: list[dict[str, Any]] = []
            for family in ("catalog", "vision", "analysis", "code"):
                payload = step_input.get_step_content(f"research_program_{family}")
                if isinstance(payload, dict):
                    new_results.extend(list(payload.get("results") or []))
            if new_results:
                tool_invocations.extend(new_results)
                state["tool_invocations"] = list(tool_invocations)
            executed_families = list(state.get("executed_families") or [])
            processed_image_files = list(state.get("processed_image_files") or [])
            for invocation in new_results:
                output_summary = invocation.get("output_summary")
                if not isinstance(output_summary, dict) or not output_summary.get("success", True):
                    continue
                family = self._research_program_tool_family(
                    str(invocation.get("tool") or "").strip()
                )
                if family not in executed_families:
                    executed_families.append(family)
                tool_name = str(invocation.get("tool") or "").strip()
                args = dict(invocation.get("args") or {})
                if tool_name == "yolo_detect":
                    for path in list(args.get("file_paths") or []):
                        value = str(path or "").strip()
                        if value and value not in processed_image_files:
                            processed_image_files.append(value)
                elif tool_name == "bioio_load_image":
                    value = str(args.get("file_path") or "").strip()
                    if value and value not in processed_image_files:
                        processed_image_files.append(value)
            handles = self._tool_program_handles_from_invocations(
                tool_invocations,
                uploaded_files=uploaded_files,
                selection_context=selection_context,
            )
            introspection_results = self._collect_research_program_artifact_introspection_results(
                latest_user_text=latest_user_text,
                handles=handles,
                existing_invocations=tool_invocations,
            )
            if introspection_results:
                new_results.extend(introspection_results)
                tool_invocations.extend(introspection_results)
                handles = self._tool_program_handles_from_invocations(
                    tool_invocations,
                    uploaded_files=uploaded_files,
                    selection_context=selection_context,
                )
            evidence_summaries = list(state.get("evidence_summaries") or [])
            for invocation in new_results:
                summary = invocation.get("output_summary")
                if isinstance(summary, dict) and summary:
                    evidence_summaries.append(
                        f"{invocation.get('tool')}: {json.dumps(summary, ensure_ascii=False)}"
                    )
                else:
                    evidence_summaries.append(
                        f"{invocation.get('tool')}: completed with no structured summary."
                    )
            state["handles"] = handles
            state["executed_families"] = executed_families
            state["evidence_summaries"] = evidence_summaries[-30:]
            state["processed_image_files"] = processed_image_files[-5000:]
            return StepOutput(
                content={
                    "new_result_count": len(new_results),
                    "handles": handles,
                    "executed_families": executed_families,
                    "evidence_summaries": evidence_summaries[-10:],
                }
            )

        async def gate_iteration(step_input: StepInput) -> StepOutput:
            state = state_from_input(step_input)
            plan = ToolProgramIterationPlan.model_validate(dict(state.get("current_plan") or {}))
            collect_payload = dict(step_input.get_step_content("research_program_collect") or {})
            new_result_count = int(collect_payload.get("new_result_count") or 0)
            evidence_summaries = list(state.get("evidence_summaries") or [])
            executed_families = list(state.get("executed_families") or [])
            requirements = dict(state.get("requirements") or {})
            enough_evidence = self._research_program_has_enough_evidence(
                evidence_summaries=evidence_summaries,
                executed_families=executed_families,
                requirements=requirements,
            )
            done = bool(enough_evidence and int(state.get("iteration") or 0) >= 1)
            if not done:
                done = bool(plan.ready_to_answer and (enough_evidence or new_result_count > 0))
            if (
                not done
                and new_result_count == 0
                and (enough_evidence or int(state.get("iteration") or 0) >= 2)
            ):
                done = True
            state["program_done"] = done
            return StepOutput(
                content={
                    "done": done,
                    "iteration": int(state.get("iteration") or 0),
                    "requirements": requirements,
                    "executed_families": executed_families,
                }
            )

        def _loop_end_condition(outputs: list[StepOutput]) -> bool:
            if not outputs:
                return False
            last = outputs[-1]
            content = getattr(last, "content", None)
            return bool(isinstance(content, dict) and content.get("done"))

        def _build_deterministic_synthesis() -> ToolProgramSynthesis | None:
            summaries_by_tool: dict[str, list[dict[str, Any]]] = {}
            for invocation in list(tool_invocations):
                summary = invocation.get("output_summary")
                if not isinstance(summary, dict) or not summary.get("success", True):
                    continue
                tool_name = str(invocation.get("tool") or "").strip()
                if not tool_name:
                    continue
                summaries_by_tool.setdefault(tool_name, []).append(summary)

            dataset_summary = (summaries_by_tool.get("bisque_download_dataset") or [None])[-1]
            hdf5_summary = (summaries_by_tool.get("inspect_local_hdf5") or [None])[-1]
            metadata_summary = (summaries_by_tool.get("load_bisque_resource") or [None])[-1]
            image_summary = (summaries_by_tool.get("bioio_load_image") or [None])[-1]
            detection_summary = (summaries_by_tool.get("yolo_detect") or [None])[-1]
            quant_summary = (summaries_by_tool.get("quantify_objects") or [None])[-1]
            code_summary = (summaries_by_tool.get("execute_python_job") or [None])[-1]
            if not any(
                (
                    dataset_summary,
                    hdf5_summary,
                    metadata_summary,
                    image_summary,
                    detection_summary,
                    quant_summary,
                    code_summary,
                )
            ):
                return None

            lowered = str(latest_user_text or "").strip().lower()
            report_like = self._is_report_like_request(latest_user_text) or bool(
                re.search(r"\b(summary|analysis|ecological|summarize|compare)\b", lowered)
            )
            introspection_request = self._is_structured_artifact_introspection_request(
                latest_user_text
            )
            uploaded_names = self._image_like_uploaded_names(uploaded_files)
            response_parts: list[str] = []
            evidence_basis: list[str] = []
            unresolved_points: list[str] = []

            if isinstance(hdf5_summary, dict) and hdf5_summary.get("success", True):
                root_keys = [
                    str(item)
                    for item in list(hdf5_summary.get("root_keys") or [])
                    if str(item).strip()
                ]
                file_label = str(
                    hdf5_summary.get("file_name")
                    or Path(str(hdf5_summary.get("file_path") or "")).name
                ).strip()
                group_count = hdf5_summary.get("group_count")
                dataset_count = hdf5_summary.get("dataset_count")
                default_dataset_path = str(hdf5_summary.get("default_dataset_path") or "").strip()
                key_line = (
                    ", ".join(f"`{key}`" for key in root_keys)
                    if root_keys
                    else "no top-level keys were exposed"
                )
                response_parts.append(
                    f"The HDF5 file{f' `{file_label}`' if file_label else ''} has the following top-level keys: {key_line}."
                )
                detail_bits: list[str] = []
                if isinstance(group_count, int):
                    detail_bits.append(f"{group_count} groups")
                if isinstance(dataset_count, int):
                    detail_bits.append(f"{dataset_count} datasets")
                if default_dataset_path:
                    detail_bits.append(f"default dataset `{default_dataset_path}`")
                if detail_bits:
                    response_parts.append(
                        "The deterministic HDF5 inspection also found "
                        + ", ".join(detail_bits)
                        + "."
                    )
                evidence_basis.append("deterministic HDF5 structure inspection")
                if introspection_request and not report_like:
                    return ToolProgramSynthesis(
                        response_text="\n\n".join(response_parts),
                        evidence_basis=evidence_basis[:8],
                        unresolved_points=[],
                        confidence="high",
                    )

            if report_like and uploaded_names:
                response_parts.append(
                    "I analyzed the referenced BisQue dataset together with the attached aerial image and grounded the report below only in measurements produced during this run."
                )
            elif report_like:
                response_parts.append(
                    "I analyzed the referenced BisQue dataset and grounded the summary below only in measurements produced during this run."
                )

            dataset_bits: list[str] = []
            if isinstance(dataset_summary, dict):
                total_members = dataset_summary.get("total_members")
                downloaded = dataset_summary.get("downloaded")
                if total_members is not None:
                    dataset_bits.append(f"the dataset contains {int(total_members)} members")
                if downloaded is not None:
                    dataset_bits.append(
                        f"{int(downloaded)} files were downloaded for deterministic analysis"
                    )
                evidence_basis.append("dataset download metadata")
            if isinstance(metadata_summary, dict):
                tag_count = metadata_summary.get("tag_count")
                if tag_count is not None:
                    dataset_bits.append(f"the dataset metadata exposed {int(tag_count)} tags")
                    evidence_basis.append("dataset metadata inspection")
            if dataset_bits:
                response_parts.append("For the dataset, " + ", and ".join(dataset_bits) + ".")

            if isinstance(quant_summary, dict):
                quant_bits: list[str] = []
                total_objects = quant_summary.get("total_objects")
                if total_objects is not None:
                    quant_bits.append(f"{int(total_objects)} detected objects were quantified")
                counts_by_class = quant_summary.get("counts_by_class")
                if isinstance(counts_by_class, dict) and counts_by_class:
                    rendered_counts = ", ".join(
                        f"{key}: {int(value)}"
                        for key, value in list(counts_by_class.items())[:6]
                        if str(key).strip()
                    )
                    if rendered_counts:
                        quant_bits.append(f"class counts were {rendered_counts}")
                distribution_summary = quant_summary.get("distribution_summary")
                if isinstance(distribution_summary, dict):
                    bbox_area = distribution_summary.get("bbox_area_px")
                    if isinstance(bbox_area, dict):
                        mean_area = bbox_area.get("mean")
                        median_area = bbox_area.get("median")
                        if isinstance(mean_area, (int, float)):
                            text = f"mean box area was {float(mean_area):.1f} px^2"
                            if isinstance(median_area, (int, float)):
                                text += f" (median {float(median_area):.1f} px^2)"
                            quant_bits.append(text)
                if quant_bits:
                    response_parts.append(
                        "Object-level measurements show that " + ", and ".join(quant_bits) + "."
                    )
                    evidence_basis.append("object quantification")
            elif isinstance(detection_summary, dict):
                detect_bits: list[str] = []
                total_boxes = detection_summary.get("total_boxes")
                if total_boxes is not None:
                    detect_bits.append(f"{int(total_boxes)} detections were produced")
                counts_by_class = detection_summary.get("counts_by_class")
                if isinstance(counts_by_class, dict) and counts_by_class:
                    rendered_counts = ", ".join(
                        f"{key}: {int(value)}"
                        for key, value in list(counts_by_class.items())[:6]
                        if str(key).strip()
                    )
                    if rendered_counts:
                        detect_bits.append(f"class counts were {rendered_counts}")
                if detect_bits:
                    response_parts.append(
                        "The detector reported that " + ", and ".join(detect_bits) + "."
                    )
                    evidence_basis.append("object detection")

            image_bits: list[str] = []
            current_image_bits: list[str] = []
            if isinstance(image_summary, dict):
                dimensions = image_summary.get("dimensions")
                if isinstance(dimensions, dict):
                    x = dimensions.get("X")
                    y = dimensions.get("Y")
                    z = dimensions.get("Z")
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                        image_bits.append(f"the attached image spans {int(x)} x {int(y)} pixels")
                    if isinstance(z, (int, float)) and int(z) > 1:
                        image_bits.append(f"with {int(z)} axial slices")
                array_shape = image_summary.get("array_shape")
                if isinstance(array_shape, list) and array_shape and not image_bits:
                    image_bits.append(f"the loaded image array shape was {array_shape}")
                reader = str(image_summary.get("reader") or "").strip()
                if reader:
                    image_bits.append(f"it was ingested through the {reader} reader")
                evidence_basis.append("image loading and metadata")
            if isinstance(detection_summary, dict) and uploaded_names:
                current_records = [
                    record
                    for record in list(detection_summary.get("per_image") or [])
                    if isinstance(record, dict)
                    and os.path.basename(
                        str(record.get("source_name") or record.get("source_path") or "").strip()
                    )
                    in uploaded_names
                ]
                if current_records:
                    current_box_total = sum(
                        int(record.get("box_count") or 0) for record in current_records
                    )
                    aggregate_counts: dict[str, int] = {}
                    for record in current_records:
                        class_counts = record.get("class_counts")
                        if not isinstance(class_counts, dict):
                            continue
                        for key, value in class_counts.items():
                            label = str(key or "").strip()
                            if not label:
                                continue
                            aggregate_counts[label] = aggregate_counts.get(label, 0) + int(
                                value or 0
                            )
                    width = next(
                        (
                            int(record.get("image_width"))
                            for record in current_records
                            if isinstance(record.get("image_width"), (int, float))
                        ),
                        None,
                    )
                    height = next(
                        (
                            int(record.get("image_height"))
                            for record in current_records
                            if isinstance(record.get("image_height"), (int, float))
                        ),
                        None,
                    )
                    current_image_bits.append(
                        f"the newly attached aerial image contributed {current_box_total} detections"
                    )
                    if aggregate_counts:
                        rendered_counts = ", ".join(
                            f"{key}: {value}" for key, value in list(aggregate_counts.items())[:6]
                        )
                        current_image_bits.append(f"with class counts {rendered_counts}")
                    if width is not None and height is not None:
                        current_image_bits.append(f"across a {width} x {height} pixel frame")
                    evidence_basis.append("uploaded-image detection")
            if image_bits:
                response_parts.append("For the new image, " + ", and ".join(image_bits) + ".")
            if current_image_bits:
                response_parts.append(
                    "For the attached image specifically, "
                    + ", and ".join(current_image_bits)
                    + "."
                )

            if report_like:
                ecology_bits: list[str] = []
                if isinstance(quant_summary, dict) or isinstance(detection_summary, dict):
                    ecology_bits.append(
                        "the ecological interpretation can rely on measured object evidence rather than impression alone"
                    )
                    if isinstance(detection_summary, dict):
                        scientific_summary = detection_summary.get("scientific_summary")
                        if isinstance(scientific_summary, dict):
                            overall = scientific_summary.get("overall")
                            if isinstance(overall, dict):
                                prairie_count = overall.get("prairie_dog_count")
                                burrow_count = overall.get("burrow_count")
                                spacing = overall.get("nearest_burrow_distance_px_mean")
                                if isinstance(prairie_count, (int, float)) and isinstance(
                                    burrow_count, (int, float)
                                ):
                                    ecology_bits.append(
                                        f"the measured baseline includes {int(prairie_count)} prairie-dog detections and {int(burrow_count)} burrow detections"
                                    )
                                if isinstance(spacing, (int, float)):
                                    ecology_bits.append(
                                        f"mean nearest-burrow spacing in the measured detection set was {float(spacing):.1f} px"
                                    )
                elif isinstance(image_summary, dict):
                    ecology_bits.append(
                        "the ecological interpretation remains limited because this run established image structure but not validated object detections"
                    )
                if dataset_bits:
                    ecology_bits.append(
                        "the dataset context provides a measured baseline for comparing the newly acquired aerial frame"
                    )
                if ecology_bits:
                    response_parts.append(
                        "Ecological interpretation: " + ", and ".join(ecology_bits) + "."
                    )
                limitation_bits: list[str] = []
                if not isinstance(quant_summary, dict):
                    limitation_bits.append(
                        "annotation and bounding-box statistics were not fully established"
                    )
                if uploaded_names and not current_image_bits:
                    limitation_bits.append(
                        "the newly attached image was not isolated as a separate measured branch in this run"
                    )
                if limitation_bits:
                    response_parts.append("Limitations: " + ", and ".join(limitation_bits) + ".")

            if isinstance(code_summary, dict):
                key_measurements = code_summary.get("key_measurements")
                if isinstance(key_measurements, list) and key_measurements:
                    response_parts.append(
                        "Additional deterministic code execution produced the following measured outputs: "
                        + "; ".join(str(item) for item in key_measurements[:6])
                        + "."
                    )
                    evidence_basis.append("code-executed measurements")

            if not isinstance(quant_summary, dict):
                unresolved_points.append(
                    "Grounded annotation and bounding-box statistics were not fully established in this run."
                )
            if report_like and not isinstance(detection_summary, dict):
                unresolved_points.append(
                    "The uploaded image was not pushed through a substantive detection or segmentation measurement step."
                )
            if not response_parts:
                return None
            return ToolProgramSynthesis(
                response_text="\n\n".join(response_parts),
                evidence_basis=evidence_basis[:8],
                unresolved_points=unresolved_points[:4],
                confidence=("medium" if quant_summary or detection_summary else "low"),
            )

        async def synthesize(step_input: StepInput) -> StepOutput:
            state = state_from_input(step_input)
            evidence_summaries = list(state.get("evidence_summaries") or [])
            handles = dict(state.get("handles") or {})
            report_like = self._is_report_like_request(latest_user_text) or bool(
                re.search(
                    r"\b(summary|analysis|ecological|summarize|compare)\b",
                    str(latest_user_text or "").strip().lower(),
                )
            )
            prompt = "\n".join(
                [
                    "Write the final user-facing scientific answer from the measured evidence only.",
                    "Lead with the answer, then explain the strongest evidence succinctly.",
                    "Write like a frontier research assistant: open with a crisp bottom line, then use short paragraphs for interpretation, comparison, and implications.",
                    "Do not turn the answer into a repetitive checklist of measurements or caveats; synthesize what matters most.",
                    "Do not invent coordinates, acquisition metadata, species presence, or counts that were not measured.",
                    "If the evidence is partial, say exactly what was and was not established.",
                    "Use polished, scientist-facing prose.",
                    "",
                    f"User request: {latest_user_text}",
                    "",
                    "Compressed evidence:",
                    json.dumps(evidence_summaries[-20:], ensure_ascii=False, indent=2),
                    "",
                    "Available handles:",
                    json.dumps(handles, ensure_ascii=False, indent=2),
                ]
            )
            fallback = ToolProgramSynthesis(
                response_text=(
                    "I gathered some evidence, but not enough grounded measurements to write a reliable final answer."
                ),
                evidence_basis=evidence_summaries[-5:],
                unresolved_points=[
                    "More deterministic evidence is needed before a stronger conclusion is justified."
                ],
                confidence="low",
            )
            started = time.monotonic()
            synthesis = await self._run_tool_program_phase(
                phase_name="research_program_synthesis",
                schema=ToolProgramSynthesis,
                prompt=prompt,
                fallback=fallback,
                session_state=state,
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                reasoning_mode=reasoning_mode,
                max_runtime_seconds=step_cap,
                debug=debug,
            )
            deterministic_synthesis = _build_deterministic_synthesis()
            response_text = str(synthesis.response_text or "").strip()
            if deterministic_synthesis is not None and (
                not response_text
                or response_text == fallback.response_text
                or (
                    len(response_text) < 240
                    and (
                        "not enough grounded measurements" in response_text.lower()
                        or "reliable final answer" in response_text.lower()
                    )
                )
            ):
                synthesis = deterministic_synthesis
                response_text = str(synthesis.response_text or "").strip()
            if report_like and str(synthesis.response_text or "").strip():
                report_packet_fallback = self._fallback_report_packet(
                    synthesis=synthesis,
                    evidence_summaries=evidence_summaries,
                )
                report_packet_prompt = "\n".join(
                    [
                        "Convert the measured evidence into a structured scientific report packet.",
                        "Keep the packet domain-general: focus on measured findings, comparisons, interpretation, limitations, and next steps.",
                        "Prioritize the strongest distinctions and decision-relevant findings so the final response can stay concise but informative.",
                        "Do not invent counts, coordinates, metadata, or unsupported claims.",
                        "If multiple uploaded images or artifacts were analyzed, comparisons should name the most important differences.",
                        "",
                        f"User request: {latest_user_text}",
                        "",
                        "Current synthesis draft:",
                        str(synthesis.response_text or "").strip(),
                        "",
                        "Evidence basis:",
                        json.dumps(
                            list(synthesis.evidence_basis or []), ensure_ascii=False, indent=2
                        ),
                        "",
                        "Unresolved points:",
                        json.dumps(
                            list(synthesis.unresolved_points or []), ensure_ascii=False, indent=2
                        ),
                        "",
                        "Compressed evidence summaries:",
                        json.dumps(evidence_summaries[-20:], ensure_ascii=False, indent=2),
                        "",
                        "Handles:",
                        json.dumps(handles, ensure_ascii=False, indent=2),
                    ]
                )
                report_packet = await self._run_tool_program_phase(
                    phase_name="research_program_report_packet",
                    schema=ToolProgramReportPacket,
                    prompt=report_packet_prompt,
                    fallback=report_packet_fallback,
                    session_state=state,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    reasoning_mode="deep",
                    max_runtime_seconds=step_cap,
                    debug=debug,
                )
                report_text = await self._run_research_report_writer(
                    latest_user_text=latest_user_text,
                    report_packet=report_packet,
                    evidence_summaries=evidence_summaries,
                    handles=handles,
                    session_state=state,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    max_runtime_seconds=step_cap,
                    debug=debug,
                )
                if (
                    report_text
                    and (
                        len(report_text) >= max(len(response_text), 600)
                        or self._report_text_has_internal_orchestration(response_text)
                    )
                    and not self._report_text_has_internal_orchestration(report_text)
                ):
                    synthesis = ToolProgramSynthesis(
                        response_text=report_text,
                        evidence_basis=list(synthesis.evidence_basis or []),
                        unresolved_points=list(synthesis.unresolved_points or []),
                        confidence=synthesis.confidence,
                    )
            phase_timings["research_program_synthesis"] = round(time.monotonic() - started, 3)
            state["synthesis"] = synthesis.model_dump(mode="json")
            self._emit_event(
                event_callback,
                {
                    "kind": "graph",
                    "event_type": "pro_mode.phase_completed",
                    "phase": "research_program",
                    "status": "completed",
                    "message": "Iterative scientific evidence program completed.",
                    "payload": {
                        "iterations": int(state.get("iteration") or 0),
                        "tool_invocation_count": len(tool_invocations),
                    },
                },
            )
            return StepOutput(content=synthesis.model_dump(mode="json"))

        workflow = Workflow(
            name="pro_mode_research_program",
            description="Iterative tool-assisted scientific evidence program.",
            steps=[
                Step(name="research_program_initialize", executor=initialize),
                Loop(
                    name="research_program_loop",
                    max_iterations=max_iterations,
                    end_condition=_loop_end_condition,
                    steps=[
                        Step(name="research_program_plan", executor=plan_iteration),
                        Parallel(
                            "research_program_parallel",
                            _family_step("catalog"),
                            _family_step("vision"),
                            _family_step("analysis"),
                            _family_step("code"),
                        ),
                        Step(name="research_program_collect", executor=collect_iteration),
                        Step(name="research_program_gate", executor=gate_iteration),
                    ],
                ),
                Step(name="research_program_synthesize", executor=synthesize),
            ],
            stream=False,
            telemetry=False,
            store_events=False,
            debug_mode=bool(debug),
            session_state={"research_program_state": {}},
        )

        try:
            output = await workflow.arun(
                input={"messages": messages, "user_text": latest_user_text},
                user_id=user_id,
                run_id=f"{str(run_id or '').strip()}::research_program" if run_id else None,
                session_id=self._scope_session_id(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    run_id=f"{str(run_id or '').strip()}::research_program"
                    if run_id
                    else "research_program",
                ),
            )
            content = dict(getattr(output, "content", {}) or {})
            synthesis = ToolProgramSynthesis.model_validate(content or {})
            return {
                "response_text": str(synthesis.response_text or "").strip(),
                "tool_invocations": list(tool_invocations),
                "metadata": {
                    "research_program": {
                        "iterations": int(local_state.get("iteration") or 0),
                        "evidence_summaries": list(local_state.get("evidence_summaries") or []),
                        "handles": dict(local_state.get("handles") or {}),
                        "requirements": dict(local_state.get("requirements") or {}),
                        "executed_families": list(local_state.get("executed_families") or []),
                        "compression_stats": dict(local_state.get("compression_stats") or {}),
                        "phase_timings": phase_timings,
                    }
                },
                "model": self.model,
                "selected_domains": ["core"],
                "runtime_status": "completed",
                "runtime_error": None,
                "selected_tool_names": tool_names,
                "attempted_tool_sets": [tool_names],
            }
        except Exception as exc:
            return {
                "response_text": (
                    "I couldn't complete the iterative scientific evidence program cleanly."
                ),
                "tool_invocations": list(tool_invocations),
                "metadata": {
                    "research_program": {
                        "iterations": int(local_state.get("iteration") or 0),
                        "evidence_summaries": list(local_state.get("evidence_summaries") or []),
                        "handles": dict(local_state.get("handles") or {}),
                        "requirements": dict(local_state.get("requirements") or {}),
                        "executed_families": list(local_state.get("executed_families") or []),
                        "compression_stats": dict(local_state.get("compression_stats") or {}),
                        "phase_timings": phase_timings,
                    }
                },
                "model": self.model,
                "selected_domains": ["core"],
                "runtime_status": "error",
                "runtime_error": str(exc or exc.__class__.__name__),
                "selected_tool_names": tool_names,
                "attempted_tool_sets": [tool_names],
            }

    @staticmethod
    def _app_session_id(
        *,
        conversation_id: str | None,
        run_id: str | None,
        user_id: str | None,
    ) -> str | None:
        root = str(conversation_id or "").strip() or str(run_id or "").strip()
        if not root:
            return None
        owner = str(user_id or "").strip() or "anonymous"
        return f"{owner}::{root}"

    @staticmethod
    def _project_id_for_scope(
        *,
        knowledge_scope: ScientificKnowledgeScope,
        knowledge_context: dict[str, Any] | None,
    ) -> str | None:
        scoped = str(knowledge_scope.project_id or "").strip()
        if scoped:
            return scoped
        context_project_id = str((knowledge_context or {}).get("project_id") or "").strip()
        return context_project_id or None

    def _runtime_debug(
        self,
        *,
        path: str,
        route: AgnoRouteDecision,
        tool_names: list[str],
        reasoning_mode: str | None,
        prompt_profile: str | None = None,
        response_source: str | None = None,
    ) -> dict[str, Any]:
        debug: dict[str, Any] = {
            "path": path,
            "agent_mode": ("tool_enabled" if tool_names else "single_model"),
            "prompt_profile": str(prompt_profile or "").strip() or None,
            "selected_domains": list(route.selected_domains or ["core"]),
            "tool_names": list(tool_names),
            "route_reason": str(route.reason or "heuristic"),
            "reasoning_mode": str(reasoning_mode or "auto"),
            "reasoning_effort": self._reasoning_effort_for_mode(reasoning_mode),
            "completion": {
                "api": "chat_completions",
                "bounded": False,
                "max_tokens": None,
                "max_completion_tokens": None,
                "verbosity": self._verbosity_for_response(),
            },
        }
        if response_source:
            debug["response_source"] = response_source
        return debug

    @staticmethod
    def _serialize_requirement(requirement: RunRequirement) -> dict[str, Any]:
        return requirement.to_dict() if hasattr(requirement, "to_dict") else {}

    def _pending_hitl_payload(
        self,
        *,
        run_output: RunOutput,
        selected_tool_names: list[str],
        workflow_hint: dict[str, Any] | None,
    ) -> dict[str, Any]:
        interruptions: list[dict[str, Any]] = []
        for requirement in list(run_output.requirements or []):
            tool_execution = requirement.tool_execution
            if tool_execution is None:
                continue
            interruptions.append(
                {
                    "tool_name": str(tool_execution.tool_name or "").strip() or "tool",
                    "tool_args": dict(tool_execution.tool_args or {}),
                    "approval_id": str(tool_execution.approval_id or "").strip() or None,
                    "message": str(run_output.content or "").strip() or None,
                }
            )
        return {
            "kind": "agno_approval",
            "selected_tool_names": list(selected_tool_names),
            "workflow_hint": dict(workflow_hint or {}),
            "interruptions": interruptions,
            "requirements": [
                self._serialize_requirement(requirement)
                for requirement in list(run_output.requirements or [])
            ],
        }

    def _tool_invocations_from_run_output(
        self, run_output: RunOutput | None
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if run_output is None:
            return records
        for tool in list(run_output.tools or []):
            tool_name = str(tool.tool_name or "").strip()
            if not tool_name:
                continue
            parsed, raw_text = self._json_result(tool.result)
            status = "error" if bool(tool.tool_call_error) else "completed"
            records.append(
                {
                    "tool": tool_name,
                    "status": status,
                    "args": dict(tool.tool_args or {}),
                    "output_envelope": parsed if isinstance(parsed, dict) else {},
                    "output_summary": self._summarize_tool_output(tool_name, parsed, raw_text),
                    "output_preview": raw_text[:800],
                }
            )
        return records

    @staticmethod
    def _merge_tool_invocations(
        primary: list[dict[str, Any]],
        fallback: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in [*primary, *fallback]:
            tool_name = str(record.get("tool") or "").strip()
            status = str(record.get("status") or "").strip()
            args = json.dumps(record.get("args") or {}, sort_keys=True, default=str)
            key = f"{tool_name}:{status}:{args}"
            if not tool_name or key in seen:
                continue
            seen.add(key)
            merged.append(dict(record))
        return merged

    @staticmethod
    def _segmentation_model_label(tool_name: str, payload: dict[str, Any]) -> str:
        raw_label = str(
            payload.get("resolved_model_ref")
            or payload.get("model")
            or payload.get("model_id")
            or payload.get("backend")
            or tool_name
            or ""
        ).strip()
        lowered = raw_label.lower()
        if "medsam" in lowered or tool_name in {"segment_image_sam2", "sam2_prompt_image"}:
            return "MedSAM2"
        if "sam3" in lowered or tool_name == "segment_image_sam3":
            return "SAM3"
        if "megaseg" in lowered or "dynunet" in lowered or tool_name == "segment_image_megaseg":
            return "Megaseg DynUNet"
        return raw_label or "Segmentation"

    @classmethod
    def _segmentation_details_requested(cls, latest_user_text: str) -> bool:
        lowered = str(latest_user_text or "").strip().lower()
        if not lowered:
            return False
        return bool(
            re.search(
                r"\b("
                r"path|paths|where|saved|save|download|upload|bisque|artifact|artifacts|"
                r"python|code|script|json|npy|bbox|bounding box|coordinates?|quantif|measure|"
                r"statistics|compare|report|analysis|table|csv"
                r")\b",
                lowered,
            )
        )

    @classmethod
    def _prefers_compact_segmentation_summary(cls, latest_user_text: str) -> bool:
        return (
            cls._is_segmentation_request(latest_user_text)
            and not cls._is_report_like_request(latest_user_text)
            and not cls._segmentation_details_requested(latest_user_text)
        )

    @staticmethod
    def _segmentation_response_looks_unpolished(response_text: str) -> bool:
        text = str(response_text or "").strip()
        if not text:
            return False
        lowered = text.lower()
        path_like_hits = re.findall(
            r"(?:^|[\s`])(?:/[^`\s]+|(?:[A-Za-z0-9._-]+/){2,}[A-Za-z0-9._-]+\.(?:npy|png|jpg|jpeg|tif|tiff|json|csv))",
            text,
        )
        return bool(
            len(text) > 900
            or text.count("```") >= 2
            or len(path_like_hits) >= 2
            or ("| metric |" in lowered and "| value |" in lowered)
            or "processing metadata" in lowered
            or "supporting information" in lowered
            or "how to load the mask" in lowered
            or "saved as a numpy" in lowered
            or "how to use the mask" in lowered
            or "bounding-box coordinates" in lowered
            or "quantification step" in lowered
            or "no mask paths were supplied" in lowered
            or "bottom line:" in lowered
            or "successfully isolated" in lowered
        )

    @classmethod
    def _deterministic_segmentation_response_from_payload(
        cls,
        *,
        tool_name: str,
        payload: dict[str, Any],
        latest_user_text: str,
    ) -> str:
        if not isinstance(payload, dict):
            return ""
        success = payload.get("success")
        if success is False:
            return ""

        files_rows = payload.get("files_processed")
        if not isinstance(files_rows, list):
            files_rows = payload.get("files") if isinstance(payload.get("files"), list) else []

        processed = payload.get("processed")
        if not isinstance(processed, (int, float)):
            processed = payload.get("processed_files")
        processed_count = int(processed or 0)
        total_files_raw = payload.get("total_files")
        total_files = int(total_files_raw or 0)
        if total_files <= 0:
            total_files = max(processed_count, len(files_rows), 1 if payload.get("file") else 0)
        if processed_count <= 0:
            processed_count = max(len(files_rows), 1 if payload.get("file") else 0)

        total_masks_raw = (
            payload.get("total_masks_generated")
            or payload.get("instance_count_reported_total")
            or payload.get("instance_count_reported")
            or payload.get("instance_count_measured_total")
            or payload.get("instance_count_measured")
            or payload.get("total_masks")
            or 0
        )
        try:
            total_masks = int(total_masks_raw or 0)
        except Exception:
            total_masks = 0

        coverage_value = payload.get("coverage_percent_mean")
        if not isinstance(coverage_value, (int, float)):
            coverage_value = payload.get("coverage_percent")

        visualizations = payload.get("visualization_paths")
        visual_count = len(visualizations) if isinstance(visualizations, list) else 0
        if not visual_count and str(payload.get("visualization_path") or "").strip():
            visual_count = 1
        if not visual_count and isinstance(files_rows, list):
            visual_count = sum(
                1 for row in files_rows if isinstance(row, dict) and row.get("visualization_saved")
            )
        if not visual_count and isinstance(payload.get("ui_artifacts"), list):
            visual_count = sum(
                1
                for item in payload.get("ui_artifacts")
                if isinstance(item, dict) and str(item.get("type") or "").strip().lower() == "image"
            )

        model_label = cls._segmentation_model_label(tool_name, payload)
        compact_request = cls._prefers_compact_segmentation_summary(latest_user_text)
        response_parts: list[str] = []

        if total_masks > 0:
            if total_files <= 1:
                response_parts.append(
                    f"{model_label} produced {total_masks} mask{'s' if total_masks != 1 else ''} for the uploaded image."
                )
            else:
                response_parts.append(
                    f"{model_label} processed {processed_count} of {total_files} image{'s' if total_files != 1 else ''} and produced {total_masks} masks total."
                )
            if isinstance(coverage_value, (int, float)):
                if total_files <= 1:
                    response_parts.append(
                        f"The mask covers about {float(coverage_value):.1f}% of the image."
                    )
                else:
                    response_parts.append(
                        f"Mean image coverage was about {float(coverage_value):.1f}%."
                    )
            if visual_count > 0:
                response_parts.append(
                    "Overlay and mask preview artifacts are available for inspection."
                )
            if compact_request:
                response_parts.append(
                    "If you want, I can also extract a bounding box, measure the region, or upload the mask to BisQue."
                )
        else:
            target = (
                "the uploaded image"
                if total_files <= 1
                else f"{processed_count or total_files} image{'s' if (processed_count or total_files) != 1 else ''}"
            )
            response_parts.append(
                f"{model_label} ran on {target}, but it did not produce a confident mask."
            )
            if visual_count > 0:
                response_parts.append("Overlay artifacts are available to inspect what happened.")
            if compact_request:
                response_parts.append(
                    "If you want, I can retry with points, boxes, or a more specific prompt."
                )

        return " ".join(part.strip() for part in response_parts if str(part or "").strip()).strip()

    @classmethod
    def _deterministic_segmentation_response(
        cls,
        *,
        latest_user_text: str,
        tool_invocations: list[dict[str, Any]],
    ) -> str:
        for invocation in reversed(list(tool_invocations or [])):
            tool_name = str(invocation.get("tool") or "").strip()
            if tool_name not in {
                "segment_image_megaseg",
                "segment_image_sam2",
                "segment_image_sam3",
                "sam2_prompt_image",
            }:
                continue
            status = str(invocation.get("status") or "").strip().lower()
            if status and status not in {"completed", "success"}:
                continue
            envelope = invocation.get("output_envelope")
            summary = invocation.get("output_summary")
            payload: dict[str, Any] = {}
            if isinstance(envelope, dict):
                payload.update(envelope)
            if isinstance(summary, dict):
                for key, value in summary.items():
                    if (
                        payload.get(key) in (None, "", [], {}) and value not in (None, "", [], {})
                    ) or key not in payload:
                        payload[key] = value
            text = cls._deterministic_segmentation_response_from_payload(
                tool_name=tool_name,
                payload=payload,
                latest_user_text=latest_user_text,
            )
            if text:
                return text
        return ""

    @classmethod
    def _should_prefer_deterministic_segmentation_response(
        cls,
        *,
        latest_user_text: str,
        response_text: str,
        tool_invocations: list[dict[str, Any]],
    ) -> bool:
        if not cls._deterministic_segmentation_response(
            latest_user_text=latest_user_text,
            tool_invocations=tool_invocations,
        ):
            return False
        normalized = str(response_text or "").strip()
        if not normalized:
            return True
        if cls._prefers_compact_segmentation_summary(latest_user_text):
            return True
        if cls._segmentation_details_requested(latest_user_text):
            return False
        return cls._segmentation_response_looks_unpolished(normalized)

    def _emit_event(
        self, callback: Callable[[dict[str, Any]], None] | None, payload: dict[str, Any]
    ) -> None:
        if callable(callback):
            callback(payload)

    def _run_output_to_result(
        self,
        *,
        run_output: RunOutput | None,
        fallback_text: str,
        route: AgnoRouteDecision,
        tool_names: list[str],
        reasoning_mode: str | None,
        workflow_hint: dict[str, Any] | None,
        hitl_resume: dict[str, Any] | None,
        run_id: str | None,
        app_session_id: str | None,
        user_id: str | None,
        latest_user_text: str,
        memory_policy: ScientificMemoryPolicy,
        knowledge_scope: ScientificKnowledgeScope,
        knowledge_context: dict[str, Any] | None,
        memory_context: dict[str, Any],
        knowledge_result: dict[str, Any],
        fallback_tool_invocations: list[dict[str, Any]] | None = None,
        runtime_status: str | None = None,
        runtime_error: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
        debug_override: dict[str, Any] | None = None,
    ) -> AgnoChatRuntimeResult:
        text, output_source = self._coerce_visible_output(run_output)
        response_text = str(text or fallback_text or "").strip()
        if not response_text:
            memory_fallback = self.memory_service.fallback_response(
                latest_user_text=latest_user_text,
                memory_context=memory_context,
            )
            if memory_fallback:
                response_text = str(memory_fallback).strip()
                output_source = "memory_fallback"
        tool_invocations = self._merge_tool_invocations(
            self._tool_invocations_from_run_output(run_output),
            list(fallback_tool_invocations or []),
        )
        deterministic_segmentation_text = self._deterministic_segmentation_response(
            latest_user_text=latest_user_text,
            tool_invocations=tool_invocations,
        )
        if (
            deterministic_segmentation_text
            and self._should_prefer_deterministic_segmentation_response(
                latest_user_text=latest_user_text,
                response_text=response_text,
                tool_invocations=tool_invocations,
            )
        ):
            response_text = deterministic_segmentation_text
            output_source = "segmentation_summary"
        if not response_text and tool_invocations:
            response_text = self._tool_invocation_fallback_text(tool_invocations)
            if response_text:
                output_source = "tool_fallback"
        if not response_text and runtime_error:
            response_text = (
                "The run ended before a stable model answer was produced. "
                f"Runtime status: {runtime_status or 'error'}."
            )
            output_source = "runtime_fallback"
        project_id = self._project_id_for_scope(
            knowledge_scope=knowledge_scope,
            knowledge_context=knowledge_context,
        )
        interrupted = bool(run_output is not None and list(run_output.requirements or []))
        resume_rejected = str((hitl_resume or {}).get("decision") or "").strip().lower() == "reject"
        if interrupted or resume_rejected:
            memory_update = ScientificMemoryUpdate(
                skipped=["approval_pending" if interrupted else "approval_rejected"]
            )
        else:
            memory_update = self.memory_service.update_after_run(
                session_id=app_session_id,
                user_id=user_id,
                title=(latest_user_text[:120].strip() or "New scientist session"),
                latest_user_text=latest_user_text,
                response_text=response_text,
                policy=memory_policy,
                knowledge_scope=knowledge_scope.model_dump(mode="json"),
                run_id=run_id,
            )
        learning_result = self.learning_journal.evaluate_and_promote(
            user_id=user_id,
            session_id=app_session_id,
            project_id=project_id,
            run_id=run_id,
            query=latest_user_text,
            response_text=response_text,
            selected_domains=list(route.selected_domains or ["core"]),
            tool_invocations=tool_invocations,
            interrupted=interrupted or resume_rejected,
            error=None,
        )
        memory_metadata = {
            **memory_context,
            **memory_update.metadata(),
        }
        metadata: dict[str, Any] = {
            "runtime": "agno",
            "debug": self._runtime_debug(
                path=(
                    "approval_resume"
                    if hitl_resume
                    else ("tool_agent" if tool_names else "text_completion")
                ),
                route=route,
                tool_names=tool_names,
                reasoning_mode=reasoning_mode,
                response_source=output_source,
            ),
            "tool_invocations": tool_invocations,
            "memory": memory_metadata,
            "knowledge": dict(knowledge_result),
            "learning": learning_result.metadata(),
            "answer_summary": self._analysis_state_answer_summary(response_text),
        }
        if isinstance(debug_override, dict) and debug_override:
            metadata["debug"] = {
                **dict(metadata.get("debug") or {}),
                **debug_override,
            }
        if isinstance(extra_metadata, dict) and extra_metadata:
            for key, value in extra_metadata.items():
                if key == "debug" and isinstance(value, dict):
                    metadata["debug"] = {
                        **dict(metadata.get("debug") or {}),
                        **value,
                    }
                elif key == "tool_invocations":
                    metadata["tool_invocations"] = self._merge_tool_invocations(
                        metadata["tool_invocations"],
                        list(value or []),
                    )
                else:
                    metadata[key] = value
        existing_contract = (
            metadata.get("contract") if isinstance(metadata.get("contract"), dict) else None
        )
        metadata["contract"] = (
            self._coerce_research_presentation_contract_payload(
                existing_contract,
                fallback_result=response_text,
            )
            if existing_contract
            else self._fallback_research_presentation_contract(
                user_text=latest_user_text,
                response_text=response_text,
                metadata=metadata,
                tool_invocations=metadata.get("tool_invocations")
                if isinstance(metadata.get("tool_invocations"), list)
                else [],
            )
        )
        metadata["debug"]["runtime_status"] = str(runtime_status or "completed")
        if runtime_error:
            metadata["debug"]["runtime_error"] = runtime_error
        if run_output is not None and list(run_output.requirements or []):
            metadata["interrupted"] = True
            metadata["pending_hitl"] = self._pending_hitl_payload(
                run_output=run_output,
                selected_tool_names=tool_names,
                workflow_hint=workflow_hint,
            )
        if hitl_resume is not None:
            metadata["resume_decision"] = (
                str(hitl_resume.get("decision") or "").strip().lower() or None
            )
        return AgnoChatRuntimeResult(
            response_text=response_text,
            selected_domains=list(route.selected_domains or ["core"]),
            domain_outputs={str(route.primary_domain or "core"): response_text},
            tool_calls=len(metadata["tool_invocations"]),
            model=str(getattr(run_output, "model", None) or self.model),
            metadata=metadata,
        )

    def _tool_invocation_fallback_text(self, tool_invocations: list[dict[str, Any]]) -> str:
        if not tool_invocations:
            return ""
        lines: list[str] = []
        for invocation in tool_invocations[:3]:
            tool_name = str(invocation.get("tool") or "tool").strip()
            status = str(invocation.get("status") or "").strip().lower()
            summary = invocation.get("output_summary")
            envelope = invocation.get("output_envelope")
            if isinstance(summary, dict):
                rendered = ", ".join(
                    f"{key}={value}"
                    for key, value in summary.items()
                    if value not in (None, "", [], {})
                )
            else:
                rendered = ""
            if not rendered and isinstance(envelope, dict):
                rendered = ", ".join(
                    f"{key}={value}"
                    for key, value in envelope.items()
                    if value not in (None, "", [], {})
                )[:260]
            if not rendered:
                rendered = str(invocation.get("output_preview") or "").strip()[:260]
            if tool_name and rendered:
                prefix = f"{tool_name}: "
                if status and status != "completed":
                    prefix = f"{tool_name} ({status}): "
                lines.append(prefix + rendered)
        return "\n".join(lines).strip()

    @staticmethod
    def _latest_failed_code_execution_summary(
        tool_invocations: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        latest_failure: dict[str, Any] | None = None
        for invocation in list(tool_invocations or []):
            if str(invocation.get("tool") or "").strip() != "execute_python_job":
                continue
            status = str(invocation.get("status") or "").strip().lower()
            raw_summary = invocation.get("output_summary")
            summary = dict(raw_summary) if isinstance(raw_summary, dict) else {}
            if bool(summary.get("success")) or status == "completed":
                latest_failure = None
                continue
            if (
                status in {"failed", "error"}
                or summary.get("success") is False
                or str(summary.get("error_message") or "").strip()
                or str(summary.get("error_class") or "").strip()
            ):
                latest_failure = {
                    **summary,
                    "status": status or str(summary.get("status") or "").strip().lower(),
                    "output_preview": str(invocation.get("output_preview") or "").strip(),
                }
        return latest_failure

    def _code_execution_fail_closed_text(
        self,
        tool_invocations: list[dict[str, Any]],
    ) -> str | None:
        failure = self._latest_failed_code_execution_summary(tool_invocations)
        if not isinstance(failure, dict):
            return None
        error_class = str(failure.get("error_class") or "").strip()
        error_message = str(failure.get("error_message") or failure.get("output_preview") or "").strip()
        lines = [
            "The requested code execution did not complete successfully, so this turn does not include measured code-derived outputs."
        ]
        if error_class and error_message:
            lines.append(f"The latest execution failed with {error_class}: {error_message}")
        elif error_message:
            lines.append(f"The latest execution failed: {error_message}")
        elif error_class:
            lines.append(f"The latest execution failed with {error_class}.")
        lines.append(
            "I am not reporting expected, approximate, or inferred numeric results from the failed run."
        )
        return " ".join(line.rstrip(".") + "." for line in lines if line).strip()

    def _code_execution_fail_closed_reservations(
        self,
        tool_invocations: list[dict[str, Any]],
    ) -> list[str]:
        failure = self._latest_failed_code_execution_summary(tool_invocations)
        if not isinstance(failure, dict):
            return []
        notes = [
            "Code execution failed in this turn. Do not report expected, estimated, approximate, or visually inferred numeric outputs as measured results."
        ]
        error_message = str(failure.get("error_message") or failure.get("output_preview") or "").strip()
        if error_message:
            notes.append(f"Latest code execution failure: {error_message}")
        return notes

    def _resume_requirements(
        self,
        *,
        pending_hitl: dict[str, Any],
        decision: str,
        note: str | None = None,
    ) -> list[RunRequirement]:
        requirements: list[RunRequirement] = []
        for raw in list(pending_hitl.get("requirements") or []):
            if not isinstance(raw, dict):
                continue
            requirement = RunRequirement.from_dict(raw)
            if decision == "reject":
                requirement.reject(note=note)
            else:
                requirement.confirm()
            requirements.append(requirement)
        return requirements

    async def run(
        self,
        *,
        messages: list[dict[str, Any]],
        uploaded_files: list[str],
        max_tool_calls: int,
        max_runtime_seconds: int,
        reasoning_mode: str | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
        user_id: str | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
        selected_tool_names: list[str] | None = None,
        workflow_hint: dict[str, Any] | None = None,
        selection_context: dict[str, Any] | None = None,
        knowledge_context: dict[str, Any] | None = None,
        memory_policy: dict[str, Any] | None = None,
        knowledge_scope: dict[str, Any] | None = None,
        hitl_resume: dict[str, Any] | None = None,
        benchmark: dict[str, Any] | None = None,
        debug: bool | None = None,
    ) -> AgnoChatRuntimeResult:
        final_result: AgnoChatRuntimeResult | None = None
        async for event in self.stream(
            messages=messages,
            uploaded_files=uploaded_files,
            max_tool_calls=max_tool_calls,
            max_runtime_seconds=max_runtime_seconds,
            reasoning_mode=reasoning_mode,
            conversation_id=conversation_id,
            run_id=run_id,
            user_id=user_id,
            event_callback=event_callback,
            selected_tool_names=selected_tool_names,
            workflow_hint=workflow_hint,
            selection_context=selection_context,
            knowledge_context=knowledge_context,
            memory_policy=memory_policy,
            knowledge_scope=knowledge_scope,
            hitl_resume=hitl_resume,
            benchmark=benchmark,
            debug=debug,
        ):
            if not isinstance(event, dict):
                continue
            if str(event.get("event") or "").strip().lower() != "done":
                continue
            payload = event.get("data")
            if not isinstance(payload, dict):
                continue
            final_result = AgnoChatRuntimeResult(
                response_text=str(payload.get("response_text") or ""),
                selected_domains=list(payload.get("selected_domains") or []),
                domain_outputs=dict(payload.get("domain_outputs") or {}),
                tool_calls=int(payload.get("tool_calls") or 0),
                model=str(payload.get("model") or self.model),
                metadata=dict(payload.get("metadata") or {}),
            )
        if final_result is None:
            raise RuntimeError("Agno runtime completed without a final payload.")
        return final_result

    async def stream(
        self,
        *,
        messages: list[dict[str, Any]],
        uploaded_files: list[str],
        max_tool_calls: int,
        max_runtime_seconds: int,
        reasoning_mode: str | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
        user_id: str | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
        selected_tool_names: list[str] | None = None,
        workflow_hint: dict[str, Any] | None = None,
        selection_context: dict[str, Any] | None = None,
        knowledge_context: dict[str, Any] | None = None,
        memory_policy: dict[str, Any] | None = None,
        knowledge_scope: dict[str, Any] | None = None,
        hitl_resume: dict[str, Any] | None = None,
        benchmark: dict[str, Any] | None = None,
        debug: bool | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        effective_max_runtime_seconds = self._effective_runtime_budget_seconds(
            max_runtime_seconds=max_runtime_seconds,
            workflow_hint=workflow_hint,
        )
        latest_user_text = self._latest_user_text(messages)
        prior_analysis_state = self._load_analysis_conversation_state(
            conversation_id=conversation_id,
            user_id=user_id,
            run_id=run_id,
        )
        normalized_selection_context = self._normalize_selection_context_for_current_turn(
            user_text=latest_user_text,
            uploaded_files=uploaded_files,
            selection_context=selection_context,
            prior_state=prior_analysis_state,
        )
        effective_selection_context = self._merge_selection_context_with_analysis_state(
            user_text=latest_user_text,
            uploaded_files=uploaded_files,
            selection_context=normalized_selection_context,
            analysis_state=prior_analysis_state,
        )
        turn_intent = self._build_turn_intent(
            user_text=latest_user_text,
            uploaded_files=uploaded_files,
            selected_tool_names=selected_tool_names,
            workflow_hint=workflow_hint,
            selection_context=effective_selection_context,
            knowledge_context=knowledge_context,
        )
        route = self._select_route(turn_intent)
        resolved_memory_policy = self.memory_service.normalize_policy(memory_policy)
        resolved_knowledge_scope = self.knowledge_hub.normalize_scope(
            knowledge_scope,
            default_project_id=str((knowledge_context or {}).get("project_id") or "").strip()
            or None,
        )
        pending_hitl = (
            dict(hitl_resume.get("pending_hitl") or {}) if isinstance(hitl_resume, dict) else {}
        )
        explicit_tool_names = self._normalize_selected_tool_names(
            selected_tool_names or pending_hitl.get("selected_tool_names")
        )
        inferred_default_tool_names = self._infer_default_tool_names_for_turn(
            turn_intent=turn_intent,
            route=route,
            explicit_tool_names=explicit_tool_names,
        )
        effective_selected_tool_names = explicit_tool_names or inferred_default_tool_names
        tool_names = self._allowed_tools_for_route(
            route=route,
            selected_tool_names=effective_selected_tool_names,
        )
        agno_session_id = self._scope_session_id(
            conversation_id=conversation_id,
            user_id=user_id,
            run_id=run_id,
        )
        app_session_id = self._app_session_id(
            conversation_id=conversation_id,
            run_id=run_id,
            user_id=user_id,
        )
        workflow_id = str((workflow_hint or {}).get("id") or "").strip().lower()

        def _persist_turn_analysis_state(
            *,
            metadata: dict[str, Any] | None,
            task_regime_override: str | None = None,
        ) -> None:
            resolved_task_regime = str(
                task_regime_override or ""
            ).strip() or self._task_regime_for_turn(
                user_text=latest_user_text,
                uploaded_files=uploaded_files,
                selection_context=effective_selection_context,
                tool_plan=None,
            )
            self._persist_analysis_state(
                conversation_id=conversation_id,
                user_id=user_id,
                run_id=run_id,
                title=(latest_user_text[:120].strip() or "Scientific analysis state"),
                latest_user_text=latest_user_text,
                uploaded_files=uploaded_files,
                selection_context=effective_selection_context,
                task_regime=resolved_task_regime,
                metadata=metadata,
            )

        if workflow_id == "pro_mode":
            pro_mode_debug = (
                bool(debug)
                and str(self._setting(self.settings, "environment", "development") or "development")
                .strip()
                .lower()
                != "production"
            )
            prior_pro_mode_state = self._load_pro_mode_conversation_state(
                conversation_id=conversation_id,
                user_id=user_id,
                run_id=run_id,
            )
            effective_selection_context = self._merge_selection_context_with_analysis_state(
                user_text=latest_user_text,
                uploaded_files=uploaded_files,
                selection_context=effective_selection_context,
                analysis_state=prior_analysis_state,
            )
            effective_selection_context = self._merge_selection_context_with_analysis_state(
                user_text=latest_user_text,
                uploaded_files=uploaded_files,
                selection_context=effective_selection_context,
                analysis_state=prior_pro_mode_state,
            )
            intake_decision = await self.pro_mode.intake(
                messages=list(messages),
                latest_user_text=latest_user_text,
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                reasoning_mode=reasoning_mode,
                max_runtime_seconds=min(effective_max_runtime_seconds, 90),
                debug=pro_mode_debug,
            )
            intake_decision = self._stabilize_pro_mode_intake_decision(
                decision=intake_decision,
                messages=list(messages),
                latest_user_text=latest_user_text,
                uploaded_files=uploaded_files,
                selected_tool_names=selected_tool_names,
                selection_context=effective_selection_context,
                prior_pro_mode_state=prior_pro_mode_state,
            )
            forced_execution_regime = self._normalize_forced_execution_regime(benchmark)
            if forced_execution_regime is not None:
                intake_decision = self._force_pro_mode_execution_regime(
                    decision=intake_decision,
                    forced_execution_regime=forced_execution_regime,
                    latest_user_text=latest_user_text,
                    uploaded_files=uploaded_files,
                    selection_context=effective_selection_context,
                    prior_pro_mode_state=prior_pro_mode_state,
                )
            if intake_decision.execution_regime == "autonomous_cycle" and bool(
                dict(benchmark or {}).get("disable_autonomy_memory_knowledge")
            ):
                intake_decision.context_policy = intake_decision.context_policy.model_copy(
                    update={"load_memory": False, "load_knowledge": False}
                )
                intake_decision.load_memory = False
                intake_decision.load_knowledge = False
            context_policy = dict(intake_decision.context_policy.model_dump(mode="json"))
            self.memory_service.ensure_session(
                session_id=app_session_id,
                user_id=user_id,
                title=latest_user_text[:120].strip() or "New scientist session",
                memory_policy=resolved_memory_policy,
                knowledge_scope=resolved_knowledge_scope.model_dump(mode="json"),
            )
            memory_context = self._empty_memory_context(resolved_memory_policy)
            knowledge_result = self._empty_knowledge_context(resolved_knowledge_scope)
            if intake_decision.route == "tool_workflow" or intake_decision.execution_regime in {
                "focused_team",
                "reasoning_solver",
                "proof_workflow",
                "autonomous_cycle",
            }:
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "event_type": "pro_mode.phase_completed",
                        "phase": "context_policy",
                        "status": "completed",
                        "message": "Resolved the explicit Pro Mode context policy.",
                        "payload": context_policy,
                    },
                )
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "event_type": "pro_mode.phase_completed",
                        "phase": "execution_router",
                        "status": "completed",
                        "message": f"Selected the `{intake_decision.execution_regime}` Pro Mode regime.",
                        "payload": {
                            "route": intake_decision.route,
                            "execution_regime": intake_decision.execution_regime,
                            "task_regime": intake_decision.task_regime,
                        },
                    },
                )
            if intake_decision.route in {"deep_reasoning", "tool_workflow"} and context_policy.get(
                "load_memory"
            ):
                memory_context = self.memory_service.retrieve_context(
                    session_id=app_session_id,
                    user_id=user_id,
                    query=latest_user_text,
                    policy=resolved_memory_policy,
                    knowledge_scope=resolved_knowledge_scope.model_dump(mode="json"),
                )
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "event_type": "memory.retrieved",
                        "phase": "memory",
                        "status": "retrieved",
                        "message": (
                            f"Retrieved {memory_context.hit_count} memory item"
                            f"{'' if memory_context.hit_count == 1 else 's'}."
                            if memory_context.hit_count > 0
                            else "No prior memory matched this turn."
                        ),
                        **memory_context.metadata(),
                    },
                )
            if intake_decision.route in {"deep_reasoning", "tool_workflow"} and context_policy.get(
                "load_knowledge"
            ):
                knowledge_result = self.knowledge_hub.retrieve_context(
                    user_id=user_id,
                    session_id=app_session_id,
                    query=latest_user_text,
                    scope=resolved_knowledge_scope,
                    domain_id=route.primary_domain,
                    workflow_hint=workflow_hint,
                    knowledge_context=knowledge_context,
                    selection_context=effective_selection_context,
                    uploaded_files=uploaded_files,
                )
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "event_type": "knowledge.retrieved",
                        "phase": "knowledge",
                        "status": "retrieved",
                        "message": (
                            f"Loaded {len(knowledge_result.hits)} knowledge hit"
                            f"{'' if len(knowledge_result.hits) == 1 else 's'}."
                            if knowledge_result.hits
                            else "No project notebook or curated knowledge was needed."
                        ),
                        **knowledge_result.metadata(),
                    },
                )
            shared_context = self._pro_mode_shared_context_payload(
                memory_context=memory_context,
                knowledge_result=knowledge_result,
                analysis_state=prior_analysis_state,
                selection_context=effective_selection_context,
                uploaded_files=uploaded_files,
            )
            if intake_decision.route == "direct_response":
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "event_type": "pro_mode.phase_started",
                        "phase": "fast_dialogue",
                        "status": "started",
                        "message": "Delegating to the direct-response Pro Mode path.",
                        "payload": {"task_regime": intake_decision.task_regime},
                    },
                )
                fast_dialogue_result = await self._run_pro_mode_fast_dialogue(
                    latest_user_text=latest_user_text,
                    intake_direct_response=intake_decision.direct_response,
                    task_regime=intake_decision.task_regime,
                    shared_context=shared_context,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    max_runtime_seconds=min(effective_max_runtime_seconds, 90),
                    debug=pro_mode_debug,
                )
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "event_type": "pro_mode.phase_completed",
                        "phase": "fast_dialogue",
                        "status": "completed" if fast_dialogue_result.response_text else "failed",
                        "message": (
                            "Direct-response Pro Mode path completed."
                            if fast_dialogue_result.response_text
                            else "Direct-response Pro Mode path did not return a stable answer."
                        ),
                        "payload": {
                            "runtime_status": fast_dialogue_result.runtime_status,
                        },
                    },
                )
                final_metadata = dict(fast_dialogue_result.metadata or {})
                pro_mode_metadata = dict(final_metadata.get("pro_mode") or {})
                fast_dialogue_completed = (
                    str(fast_dialogue_result.runtime_status or "").strip().lower() == "completed"
                )
                pro_mode_metadata.update(
                    {
                        "route": intake_decision.route,
                        "execution_regime": intake_decision.execution_regime,
                        "task_regime": intake_decision.task_regime,
                        "context_policy": context_policy,
                        "intake": intake_decision.model_dump(mode="json"),
                        "active_roles": ["Front Door Triage", "Fast Dialogue"],
                        "phase_order": [
                            "intake",
                            "context_policy",
                            "execution_router",
                            "fast_dialogue",
                            "finalize",
                        ],
                        "phase_timings": {},
                        "round_count": 0,
                        "discussion_round_count": 0,
                        "model_call_count": 1,
                        "convergence": {
                            "per_role_vote": {},
                            "central_blockers": [],
                            "ready": bool(fast_dialogue_result.response_text)
                            and fast_dialogue_completed,
                            "consensus_level": "high"
                            if fast_dialogue_completed and fast_dialogue_result.response_text
                            else "low",
                        },
                        "role_stats": {},
                        "calculator": {"used": False, "call_count": 0, "results": []},
                        "verifier": {
                            "passed": bool(fast_dialogue_result.response_text)
                            and fast_dialogue_completed,
                            "issues": [],
                            "suggested_changes": [],
                            "confidence": "medium"
                            if fast_dialogue_completed and fast_dialogue_result.response_text
                            else "low",
                        },
                        "summary": (
                            "Answered directly through the dedicated Pro Mode reasoning model."
                            if fast_dialogue_completed
                            else "Returned the intake draft because the dedicated Pro Mode reasoning model path did not complete."
                        ),
                    }
                )
                final_metadata["pro_mode"] = pro_mode_metadata
                final_metadata.setdefault("debug", {})
                if isinstance(final_metadata["debug"], dict):
                    final_metadata["debug"].update(
                        {
                            "path": "pro_mode",
                            "agent_mode": "direct_response",
                            "prompt_profile": "pro_mode",
                            "selected_domains": [route.primary_domain],
                            "tool_names": [],
                            "route_reason": "pro_mode_direct_response",
                            "reasoning_mode": reasoning_mode,
                            "response_source": "direct_response",
                            "dev_trace_enabled": pro_mode_debug,
                            "intake_route": intake_decision.route,
                            "execution_regime": intake_decision.execution_regime,
                            "context_policy": context_policy,
                            "active_roles": pro_mode_metadata["active_roles"],
                            "model_call_count": pro_mode_metadata["model_call_count"],
                            "runtime_status": fast_dialogue_result.runtime_status,
                        }
                    )
                response_text = str(fast_dialogue_result.response_text or "").strip()
                writer_stats: dict[str, Any] = {}
                polished_response_text, writer_stats = await self._run_pro_mode_final_writer(
                    latest_user_text=latest_user_text,
                    draft_response_text=response_text,
                    execution_regime="direct_response",
                    task_regime=intake_decision.task_regime,
                    supporting_points=[
                        *[
                            str(item or "").strip()
                            for item in list(
                                (shared_context or {}).get("analysis_brief_lines") or []
                            )
                            if str(item or "").strip()
                        ],
                        *[
                            str(item or "").strip()
                            for item in list((shared_context or {}).get("knowledge_messages") or [])
                            if str(item or "").strip()
                        ][:2],
                    ],
                    reservations=[],
                    session_state={"pro_mode_context": dict(shared_context or {})},
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    max_runtime_seconds=min(effective_max_runtime_seconds, 45),
                    debug=pro_mode_debug,
                )
                if polished_response_text:
                    response_text = polished_response_text
                pro_mode_metadata["writer"] = {
                    "applied": bool(polished_response_text),
                    "kind": "final_writer",
                    "compression_stats": writer_stats,
                }
                pro_mode_metadata["model_call_count"] = 2 if polished_response_text else 1
                final_metadata["pro_mode"] = pro_mode_metadata
                if isinstance(final_metadata.get("debug"), dict):
                    final_metadata["debug"]["model_call_count"] = pro_mode_metadata[
                        "model_call_count"
                    ]
                final_metadata["contract"] = self._fallback_research_presentation_contract(
                    user_text=latest_user_text,
                    response_text=response_text,
                    metadata=final_metadata,
                    tool_invocations=[],
                )
                if response_text:
                    yield {"event": "token", "data": {"delta": response_text}}
                _persist_turn_analysis_state(
                    metadata=final_metadata,
                    task_regime_override=str(
                        pro_mode_metadata.get("task_regime") or intake_decision.task_regime or ""
                    ),
                )
                yield {
                    "event": "done",
                    "data": {
                        "response_text": response_text,
                        "selected_domains": [route.primary_domain],
                        "domain_outputs": {route.primary_domain: response_text},
                        "tool_calls": 0,
                        "model": str(
                            pro_mode_metadata.get("model_route", {}).get("active_model")
                            or self._setting(self.settings, "resolved_pro_mode_model", self.model)
                            or self.model
                        ),
                        "metadata": final_metadata,
                    },
                }
                return
            if intake_decision.route == "tool_workflow":
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "event_type": "pro_mode.phase_started",
                        "phase": "tool_workflow",
                        "status": "started",
                        "message": "Delegating to the tool-enabled workflow.",
                        "payload": {
                            "selected_tool_names": list(intake_decision.selected_tool_names or [])
                        },
                    },
                )
                tool_result = await self._run_pro_mode_tool_workflow(
                    messages=list(
                        messages[
                            -max(
                                1,
                                int(
                                    context_policy.get("history_window")
                                    or intake_decision.recent_history_turns
                                    or 1
                                )
                                * 2,
                            ) :
                        ]
                    ),
                    latest_user_text=latest_user_text,
                    uploaded_files=uploaded_files,
                    max_tool_calls=max_tool_calls,
                    max_runtime_seconds=effective_max_runtime_seconds,
                    reasoning_mode=reasoning_mode,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    event_callback=event_callback,
                    selected_tool_names=list(intake_decision.selected_tool_names or []),
                    tool_plan_category=intake_decision.tool_plan_category,
                    strict_tool_validation=bool(intake_decision.strict_tool_validation),
                    selection_context=effective_selection_context,
                    knowledge_context=knowledge_context,
                    shared_context=shared_context,
                    conversation_state_seed=prior_pro_mode_state,
                    debug=pro_mode_debug,
                    allow_research_program=intake_decision.execution_regime == "iterative_research",
                )
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "event_type": "pro_mode.phase_completed",
                        "phase": "tool_workflow",
                        "status": "completed",
                        "message": "Tool-enabled workflow completed.",
                        "payload": {
                            "tool_invocation_count": len(tool_result.get("tool_invocations") or []),
                            "runtime_status": tool_result.get("runtime_status") or "completed",
                        },
                    },
                )
                research_program_meta = dict(
                    tool_result.get("metadata", {}).get("research_program") or {}
                )
                used_research_program = bool(research_program_meta)
                pro_mode_metadata = {
                    "execution_path": "research_program"
                    if used_research_program
                    else "tool_workflow",
                    "route": intake_decision.route,
                    "execution_regime": (
                        "iterative_research"
                        if used_research_program
                        else intake_decision.execution_regime
                    ),
                    "task_regime": intake_decision.task_regime,
                    "context_policy": context_policy,
                    "intake": intake_decision.model_dump(mode="json"),
                    "active_roles": (
                        [
                            "Front Door Triage",
                            "Research Program Planner",
                            "Research Program Synthesizer",
                        ]
                        if used_research_program
                        else ["Front Door Triage", "Tool Workflow"]
                    ),
                    "phase_order": (
                        [
                            "intake",
                            "context_policy",
                            "execution_router",
                            "research_program",
                            "tool_workflow",
                            "finalize",
                        ]
                        if used_research_program
                        else [
                            "intake",
                            "context_policy",
                            "execution_router",
                            "tool_workflow",
                            "finalize",
                        ]
                    ),
                    "phase_timings": dict(research_program_meta.get("phase_timings") or {}),
                    "round_count": 0,
                    "discussion_round_count": int(research_program_meta.get("iterations") or 0),
                    "model_call_count": 1 + int(research_program_meta.get("iterations") or 0),
                    "convergence": {
                        "per_role_vote": {},
                        "central_blockers": [],
                        "ready": True,
                        "consensus_level": "medium",
                    },
                    "role_stats": {},
                    "calculator": {"used": False, "call_count": 0, "results": []},
                    "verifier": {
                        "passed": bool(tool_result.get("response_text")),
                        "issues": [],
                        "suggested_changes": [],
                        "confidence": "medium",
                    },
                    "tool_validation": {
                        "selected_tool_names": list(tool_result.get("selected_tool_names") or []),
                        "attempted_tool_sets": list(tool_result.get("attempted_tool_sets") or []),
                        "runtime_status": tool_result.get("runtime_status") or "completed",
                    },
                    "attempted_tool_sets": list(tool_result.get("attempted_tool_sets") or []),
                    "summary": (
                        "Delegated to an iterative scientific evidence program after intake triage."
                        if used_research_program
                        else "Delegated to a tool-enabled workflow after intake triage."
                    ),
                }
                if used_research_program:
                    pro_mode_metadata["research_program"] = {
                        "iterations": int(research_program_meta.get("iterations") or 0),
                        "evidence_summaries": list(
                            research_program_meta.get("evidence_summaries") or []
                        ),
                        "handles": dict(research_program_meta.get("handles") or {}),
                        "requirements": dict(research_program_meta.get("requirements") or {}),
                        "executed_families": list(
                            research_program_meta.get("executed_families") or []
                        ),
                        "compression_stats": dict(
                            research_program_meta.get("compression_stats") or {}
                        ),
                    }
                if pro_mode_debug:
                    pro_mode_metadata["dev_conversation"] = {
                        "messages": [],
                        "rounds": [],
                        "calculator_results": [],
                        "tool_selected": list(tool_result.get("selected_tool_names") or []),
                        "attempted_tool_sets": list(tool_result.get("attempted_tool_sets") or []),
                        "tool_invocations": list(tool_result.get("tool_invocations") or []),
                        "evidence_summaries": list(
                            research_program_meta.get("evidence_summaries") or []
                        ),
                        "handles": dict(research_program_meta.get("handles") or {}),
                        "requirements": dict(research_program_meta.get("requirements") or {}),
                        "compression_stats": dict(
                            research_program_meta.get("compression_stats") or {}
                        ),
                        "markdown": (
                            "## Pro Mode Internal Conversation\n\n"
                            f"### {'Research Program' if used_research_program else 'Tool Workflow'}\n"
                            f"- Selected tools: {', '.join(tool_result.get('selected_tool_names') or []) or 'none'}\n"
                            f"- Attempted tool sets: {json.dumps(tool_result.get('attempted_tool_sets') or [])}\n"
                            f"- Runtime status: `{tool_result.get('runtime_status') or 'completed'}`\n"
                            + (
                                f"- Iterations: {int(research_program_meta.get('iterations') or 0)}\n"
                                f"- Evidence summaries: {json.dumps(research_program_meta.get('evidence_summaries') or [])}\n"
                                f"- Requirements: {json.dumps(research_program_meta.get('requirements') or {})}\n"
                                if used_research_program
                                else ""
                            )
                        ),
                    }
                nested_metadata = dict(tool_result.get("metadata") or {})
                response_text = str(tool_result.get("response_text") or "").strip()
                writer_stats: dict[str, Any] = {}
                writer_applied = False
                metadata_specialist = MetadataSpecialistSummary()
                metadata_summary_requested = self._is_image_metadata_request(latest_user_text)
                if self._should_run_metadata_specialist(
                    latest_user_text=latest_user_text,
                    tool_invocations=list(tool_result.get("tool_invocations") or []),
                ):
                    metadata_specialist = await self._run_metadata_specialist(
                        latest_user_text=latest_user_text,
                        tool_invocations=list(tool_result.get("tool_invocations") or []),
                        session_state={
                            "pro_mode_context": dict(shared_context or {}),
                            "research_program": {
                                "requirements": dict(
                                    research_program_meta.get("requirements") or {}
                                ),
                                "handles": dict(research_program_meta.get("handles") or {}),
                            },
                        },
                        conversation_id=conversation_id,
                        run_id=run_id,
                        user_id=user_id,
                        max_runtime_seconds=min(effective_max_runtime_seconds, 60),
                        debug=pro_mode_debug,
                    )
                    if (
                        metadata_summary_requested
                        and str(metadata_specialist.direct_answer or "").strip()
                    ):
                        response_text = str(metadata_specialist.direct_answer or "").strip()
                if response_text:
                    fail_closed_text = self._code_execution_fail_closed_text(
                        list(tool_result.get("tool_invocations") or [])
                    )
                    if fail_closed_text:
                        response_text = fail_closed_text
                    supporting_points = [
                        str(item.get("output_summary") or "").strip()
                        for item in list(tool_result.get("tool_invocations") or [])
                        if str(item.get("output_summary") or "").strip()
                    ]
                    supporting_points.extend(
                        [
                            str(item or "").strip()
                            for item in list(research_program_meta.get("evidence_summaries") or [])
                            if str(item or "").strip()
                        ]
                    )
                    supporting_points.extend(
                        [
                            str(item or "").strip()
                            for item in [
                                *list(metadata_specialist.verified_findings or []),
                                *list(metadata_specialist.filename_inferences or []),
                                *list(metadata_specialist.scientific_context or []),
                                *list(metadata_specialist.equipment_context or []),
                                *list(metadata_specialist.derived_location_context or []),
                            ]
                            if str(item or "").strip()
                        ]
                    )
                    reservations = []
                    missing_families = list(
                        (research_program_meta.get("requirements") or {}).get("missing_families")
                        or []
                    )
                    if missing_families:
                        reservations.append(
                            "Missing evidence families: "
                            + ", ".join(str(item or "").strip() for item in missing_families)
                        )
                    reservations.extend(
                        [
                            str(item or "").strip()
                            for item in [
                                *list(metadata_specialist.missing_metadata or []),
                                *list(metadata_specialist.caveats or []),
                            ]
                            if str(item or "").strip()
                        ]
                    )
                    reservations.extend(
                        self._code_execution_fail_closed_reservations(
                            list(tool_result.get("tool_invocations") or [])
                        )
                    )
                    polished_response_text, writer_stats = await self._run_pro_mode_final_writer(
                        latest_user_text=latest_user_text,
                        draft_response_text=response_text,
                        execution_regime=str(
                            pro_mode_metadata["execution_regime"] or "tool_workflow"
                        ),
                        task_regime=intake_decision.task_regime,
                        supporting_points=supporting_points,
                        reservations=reservations,
                        session_state={
                            "pro_mode_context": dict(shared_context or {}),
                            "research_program": {
                                "requirements": dict(
                                    research_program_meta.get("requirements") or {}
                                ),
                                "handles": dict(research_program_meta.get("handles") or {}),
                            },
                        },
                        conversation_id=conversation_id,
                        run_id=run_id,
                        user_id=user_id,
                        max_runtime_seconds=min(effective_max_runtime_seconds, 90),
                        debug=pro_mode_debug,
                    )
                    if polished_response_text:
                        response_text = polished_response_text
                        writer_applied = True
                structured_phase_routes = self._structured_phase_model_routes(
                    research_program_meta.get("compression_stats")
                )
                if isinstance(writer_stats.get("model_route"), dict) and writer_stats.get(
                    "model_route"
                ):
                    structured_phase_routes["pro_mode_final_writer"] = dict(
                        writer_stats.get("model_route") or {}
                    )
                if structured_phase_routes:
                    pro_mode_metadata["model_routes"] = structured_phase_routes
                pro_mode_metadata["tool_runtime_model"] = str(
                    tool_result.get("model") or self.model
                )
                pro_mode_metadata["writer"] = {
                    "applied": writer_applied,
                    "kind": "final_writer",
                    "compression_stats": writer_stats,
                }
                if any(
                    [
                        str(metadata_specialist.direct_answer or "").strip(),
                        list(metadata_specialist.verified_findings or []),
                        list(metadata_specialist.filename_inferences or []),
                        list(metadata_specialist.scientific_context or []),
                        list(metadata_specialist.equipment_context or []),
                        list(metadata_specialist.derived_location_context or []),
                        list(metadata_specialist.missing_metadata or []),
                    ]
                ):
                    pro_mode_metadata["metadata_specialist"] = metadata_specialist.model_dump(
                        mode="json"
                    )
                self._save_pro_mode_conversation_state(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    run_id=run_id,
                    state=self._extract_pro_mode_conversation_state(
                        latest_user_text=latest_user_text,
                        uploaded_files=uploaded_files,
                        selection_context=effective_selection_context,
                        task_regime=intake_decision.task_regime,
                        tool_invocations=list(tool_result.get("tool_invocations") or []),
                        research_program_meta=research_program_meta,
                    ),
                    title=(latest_user_text[:120].strip() or "Pro Mode conversation state"),
                )
                final_metadata = dict(nested_metadata)
                final_metadata["tool_invocations"] = list(tool_result.get("tool_invocations") or [])
                final_metadata["pro_mode"] = pro_mode_metadata
                final_metadata.setdefault("debug", {})
                if isinstance(final_metadata["debug"], dict):
                    final_metadata["debug"].update(
                        {
                            "path": "pro_mode",
                            "agent_mode": "tool_workflow",
                            "prompt_profile": "pro_mode",
                            "selected_domains": list(
                                tool_result.get("selected_domains") or ["core"]
                            ),
                            "tool_names": list(tool_result.get("selected_tool_names") or []),
                            "attempted_tool_sets": list(
                                tool_result.get("attempted_tool_sets") or []
                            ),
                            "route_reason": (
                                "pro_mode_research_program"
                                if used_research_program
                                else "pro_mode_tool_workflow"
                            ),
                            "reasoning_mode": reasoning_mode,
                            "response_source": "research_program"
                            if used_research_program
                            else "tool_workflow",
                            "dev_trace_enabled": pro_mode_debug,
                            "intake_route": "tool_workflow",
                            "execution_regime": pro_mode_metadata["execution_regime"],
                            "context_policy": context_policy,
                            "active_roles": pro_mode_metadata["active_roles"],
                            "model_call_count": pro_mode_metadata["model_call_count"],
                            "runtime_status": tool_result.get("runtime_status") or "completed",
                        }
                    )
                final_metadata["contract"] = self._fallback_research_presentation_contract(
                    user_text=latest_user_text,
                    response_text=response_text,
                    metadata=final_metadata,
                    tool_invocations=list(tool_result.get("tool_invocations") or []),
                )
                if response_text:
                    yield {"event": "token", "data": {"delta": response_text}}
                _persist_turn_analysis_state(
                    metadata=final_metadata,
                    task_regime_override=str(
                        pro_mode_metadata.get("task_regime") or intake_decision.task_regime or ""
                    ),
                )
                yield {
                    "event": "done",
                    "data": {
                        "response_text": response_text,
                        "selected_domains": list(tool_result.get("selected_domains") or ["core"]),
                        "domain_outputs": {"core": response_text},
                        "tool_calls": len(tool_result.get("tool_invocations") or []),
                        "model": str(tool_result.get("model") or self.model),
                        "metadata": final_metadata,
                    },
                }
                return
            if intake_decision.execution_regime == "proof_workflow":
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "event_type": "pro_mode.phase_started",
                        "phase": "proof_workflow",
                        "status": "started",
                        "message": "Delegating to the iterative proof workflow.",
                        "payload": {"task_regime": intake_decision.task_regime},
                    },
                )
                proof_branch_started = time.monotonic()
                pro_mode_result = await self._run_pro_mode_proof_workflow(
                    latest_user_text=latest_user_text,
                    shared_context=shared_context,
                    proof_state_seed=prior_pro_mode_state,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    max_runtime_seconds=effective_max_runtime_seconds,
                    debug=pro_mode_debug,
                )
                proof_meta = (
                    dict(pro_mode_result.metadata.get("pro_mode", {}).get("proof_workflow") or {})
                    if isinstance(pro_mode_result.metadata, dict)
                    else {}
                )
                if not str(pro_mode_result.response_text or "").strip():
                    fallback_to_council = self._pro_mode_expert_council_enabled()
                    self._emit_event(
                        event_callback,
                        {
                            "kind": "graph",
                            "event_type": "pro_mode.phase_failed",
                            "phase": "proof_workflow",
                            "status": "failed",
                            "message": (
                                "Proof workflow returned no stable answer; falling back to the expert council."
                                if fallback_to_council
                                else "Proof workflow returned no stable answer and council fallback is disabled."
                            ),
                            "payload": {"runtime_error": pro_mode_result.runtime_error},
                        },
                    )
                    if fallback_to_council:
                        intake_decision.execution_regime = "expert_council"
                    else:
                        fallback_text = str(pro_mode_result.response_text or "").strip() or (
                            "I couldn't complete a stable proof answer within the current Pro Mode workflow."
                        )
                        failure_metadata = {
                            "execution_path": "proof_workflow",
                            "route": intake_decision.route,
                            "execution_regime": "proof_workflow",
                            "task_regime": intake_decision.task_regime,
                            "context_policy": context_policy,
                            "intake": intake_decision.model_dump(mode="json"),
                            "active_roles": [
                                "Front Door Triage",
                                "Proof Planner",
                                "Proof Constructor",
                                "Formal Gap Checker",
                                "Proof Skeptic",
                            ],
                            "phase_order": [
                                "intake",
                                "context_policy",
                                "execution_router",
                                "proof_workflow",
                                "finalize",
                            ],
                            "round_count": int(proof_meta.get("iterations") or 0),
                            "discussion_round_count": int(proof_meta.get("iterations") or 0),
                            "model_call_count": max(
                                2, 1 + int(proof_meta.get("iterations") or 0) * 5
                            ),
                            "convergence": {
                                "per_role_vote": {},
                                "central_blockers": list(
                                    dict(proof_meta.get("proof_state") or {}).get("blocker_gaps")
                                    or []
                                ),
                                "ready": False,
                                "consensus_level": "low",
                            },
                            "role_stats": {},
                            "calculator": {"used": False, "call_count": 0, "results": []},
                            "verifier": {
                                "passed": False,
                                "issues": list(
                                    dict(proof_meta.get("proof_state") or {}).get("blocker_gaps")
                                    or []
                                ),
                                "suggested_changes": list(
                                    dict(proof_meta.get("proof_state") or {}).get("attack_points")
                                    or []
                                ),
                                "confidence": str(
                                    dict(proof_meta.get("proof_state") or {}).get("confidence")
                                    or "medium"
                                ),
                            },
                            "proof_workflow": proof_meta,
                            "reasoning_effort": "high",
                            "summary": "Proof workflow did not converge and council fallback is disabled.",
                            "fallback_policy": "no_expert_council",
                        }
                        final_result = self._run_output_to_result(
                            run_output=None,
                            fallback_text=fallback_text,
                            route=route,
                            tool_names=[],
                            reasoning_mode=reasoning_mode,
                            workflow_hint=workflow_hint,
                            hitl_resume=hitl_resume,
                            run_id=run_id,
                            app_session_id=app_session_id,
                            user_id=user_id,
                            latest_user_text=latest_user_text,
                            memory_policy=resolved_memory_policy,
                            knowledge_scope=resolved_knowledge_scope,
                            knowledge_context=knowledge_context,
                            memory_context=memory_context.metadata(),
                            knowledge_result=knowledge_result.metadata(),
                            fallback_tool_invocations=[],
                            runtime_status=pro_mode_result.runtime_status,
                            runtime_error=pro_mode_result.runtime_error,
                            extra_metadata={
                                **dict(pro_mode_result.metadata or {}),
                                "pro_mode": failure_metadata,
                            },
                            debug_override={
                                "path": "pro_mode",
                                "agent_mode": "proof_workflow",
                                "prompt_profile": "pro_mode",
                                "route_reason": "pro_mode_proof_workflow_no_council_fallback",
                                "reasoning_mode": reasoning_mode,
                                "response_source": "proof_workflow_failure",
                                "dev_trace_enabled": pro_mode_debug,
                                "intake_route": intake_decision.route,
                                "execution_regime": "proof_workflow",
                                "context_policy": context_policy,
                                "active_roles": failure_metadata["active_roles"],
                                "model_call_count": failure_metadata["model_call_count"],
                                "runtime_status": pro_mode_result.runtime_status,
                            },
                        )
                        yield {"event": "token", "data": {"delta": final_result.response_text}}
                        _persist_turn_analysis_state(
                            metadata=final_result.metadata,
                            task_regime_override="rigorous_proof",
                        )
                        yield {
                            "event": "done",
                            "data": {
                                "response_text": final_result.response_text,
                                "selected_domains": final_result.selected_domains,
                                "domain_outputs": final_result.domain_outputs,
                                "tool_calls": final_result.tool_calls,
                                "model": final_result.model,
                                "metadata": final_result.metadata,
                            },
                        }
                        return
                else:
                    self._emit_event(
                        event_callback,
                        {
                            "kind": "graph",
                            "event_type": "pro_mode.phase_completed",
                            "phase": "proof_workflow",
                            "status": "completed",
                            "message": "Iterative proof workflow completed.",
                            "payload": {"runtime_status": pro_mode_result.runtime_status},
                        },
                    )
                    proof_metadata = {
                        "execution_path": "proof_workflow",
                        "route": intake_decision.route,
                        "execution_regime": "proof_workflow",
                        "task_regime": intake_decision.task_regime,
                        "context_policy": context_policy,
                        "intake": intake_decision.model_dump(mode="json"),
                        "active_roles": [
                            "Front Door Triage",
                            "Proof Planner",
                            "Proof Constructor",
                            "Formal Gap Checker",
                            "Proof Skeptic",
                        ],
                        "phase_order": [
                            "intake",
                            "context_policy",
                            "execution_router",
                            "proof_workflow",
                            "finalize",
                        ],
                        "round_count": int(proof_meta.get("iterations") or 0),
                        "discussion_round_count": int(proof_meta.get("iterations") or 0),
                        "model_call_count": max(2, 1 + int(proof_meta.get("iterations") or 0) * 5),
                        "convergence": {
                            "per_role_vote": {},
                            "central_blockers": list(
                                dict(proof_meta.get("proof_state") or {}).get("blocker_gaps") or []
                            ),
                            "ready": not bool(
                                dict(proof_meta.get("proof_state") or {}).get("blocker_gaps") or []
                            ),
                            "consensus_level": (
                                "high"
                                if not bool(
                                    dict(proof_meta.get("proof_state") or {}).get("blocker_gaps")
                                    or []
                                )
                                else "medium"
                            ),
                        },
                        "role_stats": {},
                        "calculator": {"used": False, "call_count": 0, "results": []},
                        "verifier": {
                            "passed": not bool(
                                dict(proof_meta.get("proof_state") or {}).get("blocker_gaps") or []
                            ),
                            "issues": list(
                                dict(proof_meta.get("proof_state") or {}).get("blocker_gaps") or []
                            ),
                            "suggested_changes": list(
                                dict(proof_meta.get("proof_state") or {}).get("attack_points") or []
                            ),
                            "confidence": str(
                                dict(proof_meta.get("proof_state") or {}).get("confidence")
                                or "medium"
                            ),
                        },
                        "proof_workflow": proof_meta,
                        "reasoning_effort": "high",
                        "summary": "Solved with the iterative proof workflow instead of the generic expert council.",
                    }
                    response_text = str(pro_mode_result.response_text or "").strip()
                    writer_stats: dict[str, Any] = {}
                    proof_elapsed = max(0.0, time.monotonic() - proof_branch_started)
                    remaining_writer_budget = max(
                        1,
                        min(
                            45,
                            int(max(1.0, float(effective_max_runtime_seconds) - proof_elapsed)),
                        ),
                    )
                    polished_response_text, writer_stats = await self._run_pro_mode_final_writer(
                        latest_user_text=latest_user_text,
                        draft_response_text=response_text,
                        execution_regime="proof_workflow",
                        task_regime=intake_decision.task_regime,
                        supporting_points=[
                            *(
                                [
                                    str(
                                        dict(proof_meta.get("proof_state") or {}).get(
                                            "canonical_reduction"
                                        )
                                        or ""
                                    ).strip()
                                ]
                                if str(
                                    dict(proof_meta.get("proof_state") or {}).get(
                                        "canonical_reduction"
                                    )
                                    or ""
                                ).strip()
                                else []
                            ),
                            *[
                                str(item or "").strip()
                                for item in list(
                                    dict(proof_meta.get("proof_state") or {}).get("verified_steps")
                                    or []
                                )
                                if str(item or "").strip()
                            ],
                            *[
                                str(item or "").strip()
                                for item in list(
                                    dict(proof_meta.get("proof_state") or {}).get(
                                        "resolved_obligations"
                                    )
                                    or []
                                )
                                if str(item or "").strip()
                            ],
                            *[
                                json.dumps(item, ensure_ascii=False)
                                for item in list(proof_meta.get("iteration_summaries") or [])[-4:]
                            ],
                        ],
                        reservations=[
                            str(item or "").strip()
                            for item in list(
                                dict(proof_meta.get("proof_state") or {}).get("blocker_gaps") or []
                            )
                            if str(item or "").strip()
                        ],
                        session_state={
                            "pro_mode_context": dict(shared_context or {}),
                            "proof_workflow": proof_meta,
                        },
                        conversation_id=conversation_id,
                        run_id=run_id,
                        user_id=user_id,
                        max_runtime_seconds=remaining_writer_budget,
                        debug=pro_mode_debug,
                    )
                    if polished_response_text:
                        response_text = polished_response_text
                    proof_metadata["writer"] = {
                        "applied": bool(polished_response_text),
                        "kind": "final_writer",
                        "compression_stats": writer_stats,
                    }
                    self._save_pro_mode_conversation_state(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        run_id=run_id,
                        state=self._extract_pro_mode_conversation_state(
                            latest_user_text=latest_user_text,
                            uploaded_files=uploaded_files,
                            selection_context=effective_selection_context,
                            task_regime=intake_decision.task_regime,
                            tool_invocations=[],
                            research_program_meta=None,
                            proof_workflow_meta=proof_meta,
                        ),
                        title=(latest_user_text[:120].strip() or "Pro Mode conversation state"),
                    )
                    final_result = self._run_output_to_result(
                        run_output=None,
                        fallback_text=response_text,
                        route=route,
                        tool_names=[],
                        reasoning_mode=reasoning_mode,
                        workflow_hint=workflow_hint,
                        hitl_resume=hitl_resume,
                        run_id=run_id,
                        app_session_id=app_session_id,
                        user_id=user_id,
                        latest_user_text=latest_user_text,
                        memory_policy=resolved_memory_policy,
                        knowledge_scope=resolved_knowledge_scope,
                        knowledge_context=knowledge_context,
                        memory_context=memory_context.metadata(),
                        knowledge_result=knowledge_result.metadata(),
                        fallback_tool_invocations=[],
                        runtime_status=pro_mode_result.runtime_status,
                        runtime_error=pro_mode_result.runtime_error,
                        extra_metadata={
                            **dict(pro_mode_result.metadata or {}),
                            "pro_mode": proof_metadata,
                        },
                        debug_override={
                            "path": "pro_mode",
                            "agent_mode": "proof_workflow",
                            "prompt_profile": "pro_mode",
                            "route_reason": "pro_mode_proof_workflow",
                            "reasoning_mode": reasoning_mode,
                            "response_source": "proof_workflow",
                            "dev_trace_enabled": pro_mode_debug,
                            "intake_route": intake_decision.route,
                            "execution_regime": "proof_workflow",
                            "context_policy": context_policy,
                            "active_roles": proof_metadata["active_roles"],
                            "model_call_count": proof_metadata["model_call_count"],
                            "runtime_status": pro_mode_result.runtime_status,
                        },
                    )
                    yield {"event": "token", "data": {"delta": response_text}}
                    _persist_turn_analysis_state(
                        metadata=final_result.metadata,
                        task_regime_override="rigorous_proof",
                    )
                    yield {
                        "event": "done",
                        "data": {
                            "response_text": final_result.response_text,
                            "selected_domains": final_result.selected_domains,
                            "domain_outputs": final_result.domain_outputs,
                            "tool_calls": final_result.tool_calls,
                            "model": final_result.model,
                            "metadata": final_result.metadata,
                        },
                    }
                    return
            if intake_decision.execution_regime == "autonomous_cycle":
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "event_type": "pro_mode.phase_started",
                        "phase": "autonomous_cycle",
                        "status": "started",
                        "message": "Delegating to the bounded autonomous-cycle controller.",
                        "payload": {"task_regime": intake_decision.task_regime},
                    },
                )
                pro_mode_result = await self._run_pro_mode_autonomous_cycle(
                    messages=list(messages),
                    latest_user_text=latest_user_text,
                    uploaded_files=uploaded_files,
                    shared_context=shared_context,
                    selection_context=effective_selection_context,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    max_tool_calls=max_tool_calls,
                    max_runtime_seconds=effective_max_runtime_seconds,
                    reasoning_mode=reasoning_mode,
                    autonomy_state_seed=prior_pro_mode_state,
                    event_callback=event_callback,
                    benchmark=benchmark,
                    debug=pro_mode_debug,
                )
                autonomy_metadata = (
                    dict(pro_mode_result.metadata.get("pro_mode") or {})
                    if isinstance(pro_mode_result.metadata, dict)
                    else {}
                )
                if not str(pro_mode_result.response_text or "").strip():
                    self._emit_event(
                        event_callback,
                        {
                            "kind": "graph",
                            "event_type": "pro_mode.phase_failed",
                            "phase": "autonomous_cycle",
                            "status": "failed",
                            "message": "Autonomous cycle returned no stable answer.",
                            "payload": {"runtime_error": pro_mode_result.runtime_error},
                        },
                    )
                    fallback_text = str(pro_mode_result.response_text or "").strip() or (
                        "I couldn't produce a stable answer within the bounded autonomous-cycle workflow."
                    )
                    autonomy_metadata = {
                        "execution_path": "autonomous_cycle",
                        "route": intake_decision.route,
                        "execution_regime": "autonomous_cycle",
                        "task_regime": intake_decision.task_regime,
                        "context_policy": context_policy,
                        "intake": intake_decision.model_dump(mode="json"),
                        "active_roles": ["Front Door Triage", "Autonomy Controller"],
                        "phase_order": [
                            "intake",
                            "context_policy",
                            "execution_router",
                            "autonomous_cycle",
                            "finalize",
                        ],
                        "round_count": 0,
                        "discussion_round_count": 0,
                        "model_call_count": 1,
                        "convergence": {
                            "per_role_vote": {},
                            "central_blockers": [],
                            "ready": False,
                            "consensus_level": "low",
                        },
                        "role_stats": {},
                        "calculator": {"used": False, "call_count": 0, "results": []},
                        "verifier": {
                            "passed": False,
                            "issues": [],
                            "suggested_changes": [],
                            "confidence": "low",
                        },
                        "summary": "Autonomous cycle did not converge to a stable answer.",
                    }
                    final_result = self._run_output_to_result(
                        run_output=None,
                        fallback_text=fallback_text,
                        route=route,
                        tool_names=[],
                        reasoning_mode=reasoning_mode,
                        workflow_hint=workflow_hint,
                        hitl_resume=hitl_resume,
                        run_id=run_id,
                        app_session_id=app_session_id,
                        user_id=user_id,
                        latest_user_text=latest_user_text,
                        memory_policy=resolved_memory_policy,
                        knowledge_scope=resolved_knowledge_scope,
                        knowledge_context=knowledge_context,
                        memory_context=memory_context.metadata(),
                        knowledge_result=knowledge_result.metadata(),
                        fallback_tool_invocations=[],
                        runtime_status=pro_mode_result.runtime_status,
                        runtime_error=pro_mode_result.runtime_error,
                        extra_metadata={
                            **dict(pro_mode_result.metadata or {}),
                            "pro_mode": autonomy_metadata,
                        },
                        debug_override={
                            "path": "pro_mode",
                            "agent_mode": "autonomous_cycle",
                            "prompt_profile": "pro_mode",
                            "route_reason": "pro_mode_autonomous_cycle_failure",
                            "reasoning_mode": reasoning_mode,
                            "response_source": "autonomous_cycle_failure",
                            "dev_trace_enabled": pro_mode_debug,
                            "intake_route": intake_decision.route,
                            "execution_regime": "autonomous_cycle",
                            "context_policy": context_policy,
                            "active_roles": autonomy_metadata["active_roles"],
                            "model_call_count": autonomy_metadata["model_call_count"],
                            "runtime_status": pro_mode_result.runtime_status,
                        },
                    )
                    yield {"event": "token", "data": {"delta": final_result.response_text}}
                    _persist_turn_analysis_state(
                        metadata=final_result.metadata,
                        task_regime_override=str(
                            intake_decision.task_regime or "conceptual_high_uncertainty"
                        ),
                    )
                    yield {
                        "event": "done",
                        "data": {
                            "response_text": final_result.response_text,
                            "selected_domains": final_result.selected_domains,
                            "domain_outputs": final_result.domain_outputs,
                            "tool_calls": final_result.tool_calls,
                            "model": final_result.model,
                            "metadata": final_result.metadata,
                        },
                    }
                    return
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "event_type": "pro_mode.phase_completed",
                        "phase": "autonomous_cycle",
                        "status": "completed",
                        "message": "Autonomous cycle completed.",
                        "payload": {"runtime_status": pro_mode_result.runtime_status},
                    },
                )
                response_text = str(pro_mode_result.response_text or "").strip()
                autonomy_state = dict(autonomy_metadata.get("autonomy_state") or {})
                cycle_metrics = dict(autonomy_metadata.get("cycle_metrics") or {})
                supporting_points = [
                    *[
                        str(item or "").strip()
                        for item in list(autonomy_state.get("evidence_ledger") or [])
                        if str(item or "").strip()
                    ],
                    *[
                        json.dumps(item, ensure_ascii=False)
                        for item in list(autonomy_metadata.get("reasoning_trace_summary") or [])[
                            -4:
                        ]
                    ],
                ]
                reservations = [
                    str(item or "").strip()
                    for item in list(autonomy_state.get("open_obligations") or [])
                    if str(item or "").strip()
                ]
                polished_response_text, writer_stats = await self._run_pro_mode_final_writer(
                    latest_user_text=latest_user_text,
                    draft_response_text=response_text,
                    execution_regime="autonomous_cycle",
                    task_regime=intake_decision.task_regime,
                    supporting_points=supporting_points,
                    reservations=reservations,
                    session_state={
                        "pro_mode_context": dict(shared_context or {}),
                        "pro_mode": autonomy_metadata,
                    },
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    max_runtime_seconds=min(effective_max_runtime_seconds, 90),
                    debug=pro_mode_debug,
                )
                if polished_response_text:
                    response_text = polished_response_text
                autonomy_metadata.update(
                    {
                        "route": intake_decision.route,
                        "task_regime": intake_decision.task_regime,
                        "context_policy": context_policy,
                        "intake": intake_decision.model_dump(mode="json"),
                        "active_roles": [
                            "Front Door Triage",
                            "Autonomy Controller",
                            *(
                                ["Focused Team Lead"]
                                if "focused_team"
                                in {
                                    str(
                                        item.get("action") or item.get("selected_action") or ""
                                    ).strip()
                                    for item in list(
                                        autonomy_metadata.get("reasoning_trace_summary") or []
                                    )
                                    if isinstance(item, dict)
                                }
                                else []
                            ),
                            *(
                                ["Reasoning Solver"]
                                if "reasoning_solver"
                                in {
                                    str(
                                        item.get("action") or item.get("selected_action") or ""
                                    ).strip()
                                    for item in list(
                                        autonomy_metadata.get("reasoning_trace_summary") or []
                                    )
                                    if isinstance(item, dict)
                                }
                                else []
                            ),
                            *(
                                ["Tool Workflow"]
                                if "tool_workflow"
                                in {
                                    str(
                                        item.get("action") or item.get("selected_action") or ""
                                    ).strip()
                                    for item in list(
                                        autonomy_metadata.get("reasoning_trace_summary") or []
                                    )
                                    if isinstance(item, dict)
                                }
                                else []
                            ),
                        ],
                        "phase_order": [
                            "intake",
                            "context_policy",
                            "execution_router",
                            "autonomous_cycle",
                            "finalize",
                        ],
                        "round_count": int(cycle_metrics.get("cycles_completed") or 0),
                        "discussion_round_count": int(cycle_metrics.get("cycles_completed") or 0),
                        "model_call_count": max(
                            1,
                            int(cycle_metrics.get("cycles_completed") or 0) * 3 + 1,
                        ),
                        "convergence": {
                            "per_role_vote": {},
                            "central_blockers": list(autonomy_state.get("open_obligations") or []),
                            "ready": bool(cycle_metrics.get("converged")),
                            "consensus_level": (
                                "high"
                                if bool(cycle_metrics.get("converged"))
                                else "medium"
                                if float(cycle_metrics.get("evidence_sufficiency_score") or 0.0)
                                >= 0.6
                                else "low"
                            ),
                        },
                        "role_stats": {},
                        "calculator": {"used": False, "call_count": 0, "results": []},
                        "verifier": {
                            "passed": bool(cycle_metrics.get("converged")),
                            "issues": list(autonomy_state.get("open_obligations") or []),
                            "suggested_changes": list(
                                autonomy_state.get("next_best_actions") or []
                            ),
                            "confidence": (
                                "high"
                                if bool(cycle_metrics.get("converged"))
                                else "medium"
                                if float(cycle_metrics.get("evidence_sufficiency_score") or 0.0)
                                >= 0.6
                                else "low"
                            ),
                        },
                        "summary": (
                            "Solved with a bounded long-cycle autonomous workflow."
                            if bool(cycle_metrics.get("converged"))
                            else "Used a bounded long-cycle autonomous workflow and stopped at a typed checkpoint."
                        ),
                        "writer": {
                            "applied": bool(polished_response_text),
                            "kind": "final_writer",
                            "compression_stats": writer_stats,
                        },
                    }
                )
                self._save_pro_mode_conversation_state(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    run_id=run_id,
                    state=self._extract_pro_mode_conversation_state(
                        latest_user_text=latest_user_text,
                        uploaded_files=uploaded_files,
                        selection_context=effective_selection_context,
                        task_regime=intake_decision.task_regime,
                        tool_invocations=list(pro_mode_result.tool_invocations or []),
                        research_program_meta=None,
                        autonomy_meta=autonomy_metadata,
                    ),
                    title=(latest_user_text[:120].strip() or "Pro Mode conversation state"),
                )
                final_result = self._run_output_to_result(
                    run_output=None,
                    fallback_text=response_text,
                    route=route,
                    tool_names=[],
                    reasoning_mode=reasoning_mode,
                    workflow_hint=workflow_hint,
                    hitl_resume=hitl_resume,
                    run_id=run_id,
                    app_session_id=app_session_id,
                    user_id=user_id,
                    latest_user_text=latest_user_text,
                    memory_policy=resolved_memory_policy,
                    knowledge_scope=resolved_knowledge_scope,
                    knowledge_context=knowledge_context,
                    memory_context=memory_context.metadata(),
                    knowledge_result=knowledge_result.metadata(),
                    fallback_tool_invocations=pro_mode_result.tool_invocations,
                    runtime_status=pro_mode_result.runtime_status,
                    runtime_error=pro_mode_result.runtime_error,
                    extra_metadata={
                        **dict(pro_mode_result.metadata or {}),
                        "pro_mode": autonomy_metadata,
                    },
                    debug_override={
                        "path": "pro_mode",
                        "agent_mode": "autonomous_cycle",
                        "prompt_profile": "pro_mode",
                        "route_reason": "pro_mode_autonomous_cycle",
                        "reasoning_mode": reasoning_mode,
                        "response_source": "autonomous_cycle",
                        "dev_trace_enabled": pro_mode_debug,
                        "intake_route": intake_decision.route,
                        "execution_regime": "autonomous_cycle",
                        "context_policy": context_policy,
                        "active_roles": list(autonomy_metadata.get("active_roles") or []),
                        "model_call_count": int(autonomy_metadata.get("model_call_count") or 0),
                        "runtime_status": pro_mode_result.runtime_status,
                    },
                )
                yield {"event": "token", "data": {"delta": response_text}}
                _persist_turn_analysis_state(
                    metadata=final_result.metadata,
                    task_regime_override=str(
                        intake_decision.task_regime or "conceptual_high_uncertainty"
                    ),
                )
                yield {
                    "event": "done",
                    "data": {
                        "response_text": final_result.response_text,
                        "selected_domains": final_result.selected_domains,
                        "domain_outputs": final_result.domain_outputs,
                        "tool_calls": final_result.tool_calls,
                        "model": final_result.model,
                        "metadata": final_result.metadata,
                    },
                }
                return
            if intake_decision.execution_regime == "focused_team":
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "event_type": "pro_mode.phase_started",
                        "phase": "focused_team",
                        "status": "started",
                        "message": "Delegating to the focused reasoning team.",
                        "payload": {"task_regime": intake_decision.task_regime},
                    },
                )
                pro_mode_result = await self._run_pro_mode_focused_team(
                    latest_user_text=latest_user_text,
                    shared_context=shared_context,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    max_runtime_seconds=effective_max_runtime_seconds,
                    debug=pro_mode_debug,
                )
                if not str(pro_mode_result.response_text or "").strip():
                    self._emit_event(
                        event_callback,
                        {
                            "kind": "graph",
                            "event_type": "pro_mode.phase_failed",
                            "phase": "focused_team",
                            "status": "failed",
                            "message": "Focused team returned no stable answer.",
                            "payload": {"runtime_error": pro_mode_result.runtime_error},
                        },
                    )
                    fallback_text = str(pro_mode_result.response_text or "").strip() or (
                        "I couldn't produce a stable focused-team answer within the current Pro Mode workflow."
                    )
                    focused_team_metadata = {
                        "execution_path": "focused_team",
                        "route": intake_decision.route,
                        "execution_regime": "focused_team",
                        "task_regime": intake_decision.task_regime,
                        "context_policy": context_policy,
                        "intake": intake_decision.model_dump(mode="json"),
                        "active_roles": ["Front Door Triage", "Focused Team Lead"],
                        "phase_order": [
                            "intake",
                            "context_policy",
                            "execution_router",
                            "focused_team",
                            "finalize",
                        ],
                        "round_count": 1,
                        "discussion_round_count": 1,
                        "model_call_count": 1,
                        "convergence": {
                            "per_role_vote": {},
                            "central_blockers": [],
                            "ready": False,
                            "consensus_level": "low",
                        },
                        "role_stats": {},
                        "calculator": {"used": False, "call_count": 0, "results": []},
                        "verifier": {
                            "passed": False,
                            "issues": [],
                            "suggested_changes": [],
                            "confidence": "low",
                        },
                        "summary": "Focused team did not converge to a stable answer.",
                    }
                    final_result = self._run_output_to_result(
                        run_output=None,
                        fallback_text=fallback_text,
                        route=route,
                        tool_names=[],
                        reasoning_mode=reasoning_mode,
                        workflow_hint=workflow_hint,
                        hitl_resume=hitl_resume,
                        run_id=run_id,
                        app_session_id=app_session_id,
                        user_id=user_id,
                        latest_user_text=latest_user_text,
                        memory_policy=resolved_memory_policy,
                        knowledge_scope=resolved_knowledge_scope,
                        knowledge_context=knowledge_context,
                        memory_context=memory_context.metadata(),
                        knowledge_result=knowledge_result.metadata(),
                        fallback_tool_invocations=[],
                        runtime_status=pro_mode_result.runtime_status,
                        runtime_error=pro_mode_result.runtime_error,
                        extra_metadata={
                            **dict(pro_mode_result.metadata or {}),
                            "pro_mode": focused_team_metadata,
                        },
                        debug_override={
                            "path": "pro_mode",
                            "agent_mode": "focused_team",
                            "prompt_profile": "pro_mode",
                            "route_reason": "pro_mode_focused_team_failure",
                            "reasoning_mode": reasoning_mode,
                            "response_source": "focused_team_failure",
                            "dev_trace_enabled": pro_mode_debug,
                            "intake_route": intake_decision.route,
                            "execution_regime": "focused_team",
                            "context_policy": context_policy,
                            "active_roles": focused_team_metadata["active_roles"],
                            "model_call_count": focused_team_metadata["model_call_count"],
                            "runtime_status": pro_mode_result.runtime_status,
                        },
                    )
                    yield {"event": "token", "data": {"delta": final_result.response_text}}
                    _persist_turn_analysis_state(
                        metadata=final_result.metadata,
                        task_regime_override=str(
                            intake_decision.task_regime or "conceptual_high_uncertainty"
                        ),
                    )
                    yield {
                        "event": "done",
                        "data": {
                            "response_text": final_result.response_text,
                            "selected_domains": final_result.selected_domains,
                            "domain_outputs": final_result.domain_outputs,
                            "tool_calls": final_result.tool_calls,
                            "model": final_result.model,
                            "metadata": final_result.metadata,
                        },
                    }
                    return
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "event_type": "pro_mode.phase_completed",
                        "phase": "focused_team",
                        "status": "completed",
                        "message": "Focused reasoning team completed.",
                        "payload": {"runtime_status": pro_mode_result.runtime_status},
                    },
                )
                focused_team_metadata = (
                    dict(pro_mode_result.metadata.get("pro_mode") or {})
                    if isinstance(pro_mode_result.metadata, dict)
                    else {}
                )
                response_text = str(pro_mode_result.response_text or "").strip()
                writer_stats: dict[str, Any] = {}
                focused_team_details = dict(focused_team_metadata.get("focused_team") or {})
                polished_response_text, writer_stats = await self._run_pro_mode_final_writer(
                    latest_user_text=latest_user_text,
                    draft_response_text=response_text,
                    execution_regime="focused_team",
                    task_regime=intake_decision.task_regime,
                    supporting_points=[
                        *[
                            str(item.get("headline") or "").strip()
                            for item in list(focused_team_details.get("member_notes") or [])
                            if isinstance(item, dict) and str(item.get("headline") or "").strip()
                        ],
                        *[
                            str(item or "").strip()
                            for item in list(
                                dict(focused_team_details.get("review") or {}).get("must_include")
                                or []
                            )
                            if str(item or "").strip()
                        ],
                    ],
                    reservations=[
                        str(item or "").strip()
                        for item in list(
                            dict(focused_team_metadata.get("verifier") or {}).get("issues") or []
                        )
                        if str(item or "").strip()
                    ],
                    session_state={
                        "pro_mode_context": dict(shared_context or {}),
                        "pro_mode": focused_team_metadata,
                    },
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    max_runtime_seconds=min(effective_max_runtime_seconds, 90),
                    debug=pro_mode_debug,
                )
                if polished_response_text:
                    response_text = polished_response_text
                focused_team_metadata["writer"] = {
                    "applied": bool(polished_response_text),
                    "kind": "final_writer",
                    "compression_stats": writer_stats,
                }
                final_result = self._run_output_to_result(
                    run_output=None,
                    fallback_text=response_text,
                    route=route,
                    tool_names=[],
                    reasoning_mode=reasoning_mode,
                    workflow_hint=workflow_hint,
                    hitl_resume=hitl_resume,
                    run_id=run_id,
                    app_session_id=app_session_id,
                    user_id=user_id,
                    latest_user_text=latest_user_text,
                    memory_policy=resolved_memory_policy,
                    knowledge_scope=resolved_knowledge_scope,
                    knowledge_context=knowledge_context,
                    memory_context=memory_context.metadata(),
                    knowledge_result=knowledge_result.metadata(),
                    fallback_tool_invocations=[],
                    runtime_status=pro_mode_result.runtime_status,
                    runtime_error=pro_mode_result.runtime_error,
                    extra_metadata={
                        **dict(pro_mode_result.metadata or {}),
                        "pro_mode": focused_team_metadata,
                    },
                    debug_override={
                        "path": "pro_mode",
                        "agent_mode": "focused_team",
                        "prompt_profile": "pro_mode",
                        "route_reason": "pro_mode_focused_team",
                        "reasoning_mode": reasoning_mode,
                        "response_source": "focused_team",
                        "dev_trace_enabled": pro_mode_debug,
                        "intake_route": intake_decision.route,
                        "execution_regime": "focused_team",
                        "context_policy": context_policy,
                        "active_roles": list(focused_team_metadata.get("active_roles") or []),
                        "model_call_count": int(focused_team_metadata.get("model_call_count") or 0),
                        "runtime_status": pro_mode_result.runtime_status,
                    },
                )
                yield {"event": "token", "data": {"delta": response_text}}
                _persist_turn_analysis_state(
                    metadata=final_result.metadata,
                    task_regime_override=str(
                        intake_decision.task_regime or "conceptual_high_uncertainty"
                    ),
                )
                yield {
                    "event": "done",
                    "data": {
                        "response_text": final_result.response_text,
                        "selected_domains": final_result.selected_domains,
                        "domain_outputs": final_result.domain_outputs,
                        "tool_calls": final_result.tool_calls,
                        "model": final_result.model,
                        "metadata": final_result.metadata,
                    },
                }
                return
            if intake_decision.execution_regime == "reasoning_solver":
                codeexec_reasoning_turn = (
                    str(intake_decision.tool_plan_category or "").strip() == "code_execution"
                    and {
                        tool_name
                        for tool_name in list(intake_decision.selected_tool_names or [])
                        if str(tool_name or "").strip()
                    }
                    >= {"codegen_python_plan", "execute_python_job"}
                )
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "event_type": "pro_mode.phase_started",
                        "phase": "reasoning_solver",
                        "status": "started",
                        "message": (
                            "Delegating to the dedicated code-execution reasoning agent."
                            if codeexec_reasoning_turn
                            else "Delegating to the self-contained reasoning solver."
                        ),
                        "payload": {
                            "task_regime": intake_decision.task_regime,
                            "selected_tool_names": list(intake_decision.selected_tool_names or []),
                        },
                    },
                )
                if codeexec_reasoning_turn:
                    pro_mode_result = await self._run_pro_mode_codeexec_reasoning_solver(
                        latest_user_text=latest_user_text,
                        task_regime=intake_decision.task_regime,
                        shared_context=shared_context,
                        uploaded_files=uploaded_files,
                        selection_context=effective_selection_context,
                        selected_tool_names=list(intake_decision.selected_tool_names or []),
                        conversation_id=conversation_id,
                        run_id=run_id,
                        user_id=user_id,
                        max_runtime_seconds=effective_max_runtime_seconds,
                        debug=pro_mode_debug,
                        event_callback=event_callback,
                    )
                else:
                    pro_mode_result = await self._run_pro_mode_reasoning_solver(
                        latest_user_text=latest_user_text,
                        task_regime=intake_decision.task_regime,
                        shared_context=shared_context,
                        conversation_id=conversation_id,
                        run_id=run_id,
                        user_id=user_id,
                        max_runtime_seconds=effective_max_runtime_seconds,
                        debug=pro_mode_debug,
                    )
                solver_tool_invocations = (
                    list(pro_mode_result.metadata.get("tool_invocations") or [])
                    if isinstance(pro_mode_result.metadata, dict)
                    else []
                )
                if not str(pro_mode_result.response_text or "").strip():
                    self._emit_event(
                        event_callback,
                        {
                            "kind": "graph",
                            "event_type": "pro_mode.phase_failed",
                            "phase": "reasoning_solver",
                            "status": "failed",
                            "message": "Reasoning solver returned no answer and no legacy council fallback was used.",
                            "payload": {"runtime_error": pro_mode_result.runtime_error},
                        },
                    )
                    fallback_text = str(pro_mode_result.response_text or "").strip() or (
                        "I couldn't produce a stable answer within the current Pro Mode reasoning workflow."
                    )
                    solver_metadata = {
                        "execution_path": "reasoning_solver",
                        "route": intake_decision.route,
                        "execution_regime": "reasoning_solver",
                        "task_regime": intake_decision.task_regime,
                        "context_policy": context_policy,
                        "intake": intake_decision.model_dump(mode="json"),
                        "active_roles": ["Front Door Triage", "Reasoning Solver"],
                        "phase_order": [
                            "intake",
                            "context_policy",
                            "execution_router",
                            "reasoning_solver",
                            "finalize",
                        ],
                        "round_count": 0,
                        "discussion_round_count": 0,
                        "model_call_count": 1,
                        "convergence": {
                            "per_role_vote": {},
                            "central_blockers": [],
                            "ready": False,
                            "consensus_level": "low",
                        },
                        "role_stats": {},
                        "calculator": {"used": False, "call_count": 0, "results": []},
                        "verifier": {
                            "passed": False,
                            "issues": [],
                            "suggested_changes": [],
                            "confidence": "low",
                        },
                        "summary": "Reasoning solver did not converge and no legacy council fallback was used.",
                        "fallback_policy": "reasoning_solver_only",
                    }
                    final_result = self._run_output_to_result(
                        run_output=None,
                        fallback_text=fallback_text,
                        route=route,
                        tool_names=[],
                        reasoning_mode=reasoning_mode,
                        workflow_hint=workflow_hint,
                        hitl_resume=hitl_resume,
                        run_id=run_id,
                        app_session_id=app_session_id,
                        user_id=user_id,
                        latest_user_text=latest_user_text,
                        memory_policy=resolved_memory_policy,
                        knowledge_scope=resolved_knowledge_scope,
                        knowledge_context=knowledge_context,
                        memory_context=memory_context.metadata(),
                        knowledge_result=knowledge_result.metadata(),
                        fallback_tool_invocations=solver_tool_invocations,
                        runtime_status=pro_mode_result.runtime_status,
                        runtime_error=pro_mode_result.runtime_error,
                        extra_metadata={
                            **dict(pro_mode_result.metadata or {}),
                            "pro_mode": solver_metadata,
                        },
                        debug_override={
                            "path": "pro_mode",
                            "agent_mode": "reasoning_solver",
                            "prompt_profile": "pro_mode",
                            "route_reason": "pro_mode_reasoning_solver_no_council_fallback",
                            "reasoning_mode": reasoning_mode,
                            "response_source": "reasoning_solver_failure",
                            "dev_trace_enabled": pro_mode_debug,
                            "intake_route": intake_decision.route,
                            "execution_regime": "reasoning_solver",
                            "context_policy": context_policy,
                            "active_roles": solver_metadata["active_roles"],
                            "model_call_count": solver_metadata["model_call_count"],
                            "runtime_status": pro_mode_result.runtime_status,
                        },
                    )
                    yield {"event": "token", "data": {"delta": final_result.response_text}}
                    _persist_turn_analysis_state(
                        metadata=final_result.metadata,
                        task_regime_override=str(
                            intake_decision.task_regime or "self_contained_reasoning"
                        ),
                    )
                    yield {
                        "event": "done",
                        "data": {
                            "response_text": final_result.response_text,
                            "selected_domains": final_result.selected_domains,
                            "domain_outputs": final_result.domain_outputs,
                            "tool_calls": final_result.tool_calls,
                            "model": final_result.model,
                            "metadata": final_result.metadata,
                        },
                    }
                    return
                else:
                    self._emit_event(
                        event_callback,
                        {
                            "kind": "graph",
                            "event_type": "pro_mode.phase_completed",
                            "phase": "reasoning_solver",
                            "status": "completed",
                            "message": "Self-contained reasoning solver completed.",
                            "payload": {"runtime_status": pro_mode_result.runtime_status},
                        },
                    )
                    solver_runtime_meta = (
                        dict(pro_mode_result.metadata.get("pro_mode") or {})
                        if isinstance(pro_mode_result.metadata, dict)
                        else {}
                    )
                    verifier_payload = dict(solver_runtime_meta.get("verifier") or {})
                    solver_metadata = {
                        "execution_path": (
                            "codeexec_reasoning_solver"
                            if codeexec_reasoning_turn
                            else "reasoning_solver"
                        ),
                        "route": intake_decision.route,
                        "execution_regime": "reasoning_solver",
                        "task_regime": intake_decision.task_regime,
                        "context_policy": context_policy,
                        "intake": intake_decision.model_dump(mode="json"),
                        "active_roles": (
                            ["Front Door Triage", "Code Execution Reasoner"]
                            if codeexec_reasoning_turn
                            else ["Front Door Triage", "Reasoning Solver"]
                        ),
                        "phase_order": [
                            "intake",
                            "context_policy",
                            "execution_router",
                            "reasoning_solver",
                            "finalize",
                        ],
                        "round_count": 0,
                        "discussion_round_count": 0,
                        "model_call_count": 2 + (1 if verifier_payload else 0),
                        "convergence": {
                            "per_role_vote": {},
                            "central_blockers": list(verifier_payload.get("issues") or []),
                            "ready": bool(verifier_payload.get("passed", True)),
                            "consensus_level": "high"
                            if bool(verifier_payload.get("passed", True))
                            else "medium",
                        },
                        "role_stats": {},
                        "calculator": {"used": False, "call_count": 0, "results": []},
                        "verifier": (
                            verifier_payload
                            if verifier_payload
                            else {
                                "passed": bool(pro_mode_result.response_text),
                                "issues": [],
                                "suggested_changes": [],
                                "confidence": "high",
                            }
                        ),
                        "summary": "Solved with the self-contained reasoning solver instead of the full expert council.",
                    }
                    if dict(solver_runtime_meta.get("model_route") or {}):
                        solver_metadata["model_route"] = dict(
                            solver_runtime_meta.get("model_route") or {}
                        )
                    deterministic_fallback_meta = dict(
                        solver_runtime_meta.get("deterministic_required_tool_fallback") or {}
                    )
                    if deterministic_fallback_meta:
                        solver_metadata["deterministic_required_tool_fallback"] = (
                            deterministic_fallback_meta
                        )
                        solver_metadata["summary"] = (
                            "Planned with the dedicated code-execution reasoner and completed "
                            "the required code tools through deterministic fallback."
                        )
                    response_text = str(pro_mode_result.response_text or "").strip()
                    writer_stats: dict[str, Any] = {}
                    supporting_points = []
                    if codeexec_reasoning_turn:
                        for invocation in list(solver_tool_invocations or []):
                            summary = invocation.get("output_summary")
                            if isinstance(summary, dict) and summary:
                                supporting_points.append(json.dumps(summary, ensure_ascii=False))
                            elif str(summary or "").strip():
                                supporting_points.append(str(summary or "").strip())
                    reservations = [
                        str(item or "").strip()
                        for item in list(
                            dict(solver_metadata.get("verifier") or {}).get("issues") or []
                        )
                        if str(item or "").strip()
                    ]
                    if codeexec_reasoning_turn:
                        reservations.extend(
                            self._code_execution_fail_closed_reservations(
                                solver_tool_invocations
                            )
                        )
                    polished_response_text, writer_stats = await self._run_pro_mode_final_writer(
                        latest_user_text=latest_user_text,
                        draft_response_text=response_text,
                        execution_regime="reasoning_solver",
                        task_regime=intake_decision.task_regime,
                        supporting_points=supporting_points,
                        reservations=reservations,
                        session_state={"pro_mode_context": dict(shared_context or {})},
                        conversation_id=conversation_id,
                        run_id=run_id,
                        user_id=user_id,
                        max_runtime_seconds=min(effective_max_runtime_seconds, 90),
                        debug=pro_mode_debug,
                    )
                    if polished_response_text:
                        response_text = polished_response_text
                    solver_metadata["writer"] = {
                        "applied": bool(polished_response_text),
                        "kind": "final_writer",
                        "compression_stats": writer_stats,
                    }
                    final_result = self._run_output_to_result(
                        run_output=None,
                        fallback_text=response_text,
                        route=route,
                        tool_names=[],
                        reasoning_mode=reasoning_mode,
                        workflow_hint=workflow_hint,
                        hitl_resume=hitl_resume,
                        run_id=run_id,
                        app_session_id=app_session_id,
                        user_id=user_id,
                        latest_user_text=latest_user_text,
                        memory_policy=resolved_memory_policy,
                        knowledge_scope=resolved_knowledge_scope,
                        knowledge_context=knowledge_context,
                        memory_context=memory_context.metadata(),
                        knowledge_result=knowledge_result.metadata(),
                        fallback_tool_invocations=solver_tool_invocations,
                        runtime_status=pro_mode_result.runtime_status,
                        runtime_error=pro_mode_result.runtime_error,
                        extra_metadata={
                            **dict(pro_mode_result.metadata or {}),
                            "pro_mode": solver_metadata,
                        },
                        debug_override={
                            "path": "pro_mode",
                            "agent_mode": "reasoning_solver",
                            "prompt_profile": "pro_mode",
                            "route_reason": "pro_mode_reasoning_solver",
                            "reasoning_mode": reasoning_mode,
                            "response_source": "reasoning_solver",
                            "dev_trace_enabled": pro_mode_debug,
                            "intake_route": intake_decision.route,
                            "execution_regime": "reasoning_solver",
                            "context_policy": context_policy,
                            "active_roles": solver_metadata["active_roles"],
                            "model_call_count": solver_metadata["model_call_count"],
                            "runtime_status": pro_mode_result.runtime_status,
                        },
                    )
                    yield {"event": "token", "data": {"delta": response_text}}
                    _persist_turn_analysis_state(
                        metadata=final_result.metadata,
                        task_regime_override=str(
                            intake_decision.task_regime or "self_contained_reasoning"
                        ),
                    )
                    yield {
                        "event": "done",
                        "data": {
                            "response_text": final_result.response_text,
                            "selected_domains": final_result.selected_domains,
                            "domain_outputs": final_result.domain_outputs,
                            "tool_calls": final_result.tool_calls,
                            "model": final_result.model,
                            "metadata": final_result.metadata,
                        },
                    }
                    return
            pro_mode_runner = self.pro_mode
            if (
                intake_decision.execution_regime == "expert_council"
                and not self._pro_mode_expert_council_enabled()
            ):
                pro_mode_runner = ProModeWorkflowRunner(
                    model_builder=self._build_pro_mode_model,
                    fallback_model_builder=self._build_model,
                    enable_expert_council=True,
                )
            pro_mode_result = await pro_mode_runner.execute(
                messages=list(messages),
                latest_user_text=latest_user_text,
                conversation_id=conversation_id,
                run_id=run_id,
                user_id=user_id,
                reasoning_mode=reasoning_mode,
                max_runtime_seconds=effective_max_runtime_seconds,
                event_callback=event_callback,
                debug=pro_mode_debug,
                intake_decision=intake_decision,
                shared_context=shared_context,
            )
            response_text = str(pro_mode_result.response_text or "").strip()
            pro_mode_metadata = (
                dict(pro_mode_result.metadata.get("pro_mode") or {})
                if isinstance(pro_mode_result.metadata, dict)
                else {}
            )
            execution_regime = str(
                pro_mode_metadata.get("execution_regime") or intake_decision.execution_regime or ""
            ).strip()
            writer_stats: dict[str, Any] = {}
            metadata_specialist = MetadataSpecialistSummary()
            metadata_summary_requested = self._is_image_metadata_request(latest_user_text)
            if self._should_run_metadata_specialist(
                latest_user_text=latest_user_text,
                tool_invocations=list(pro_mode_result.tool_invocations or []),
            ):
                metadata_specialist = await self._run_metadata_specialist(
                    latest_user_text=latest_user_text,
                    tool_invocations=list(pro_mode_result.tool_invocations or []),
                    session_state={
                        "pro_mode_context": dict(shared_context or {}),
                        "pro_mode": pro_mode_metadata,
                    },
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    max_runtime_seconds=min(effective_max_runtime_seconds, 60),
                    debug=pro_mode_debug,
                )
                if (
                    metadata_summary_requested
                    and str(metadata_specialist.direct_answer or "").strip()
                ):
                    response_text = str(metadata_specialist.direct_answer or "").strip()
            if response_text and execution_regime and execution_regime != "fast_dialogue":
                fail_closed_text = self._code_execution_fail_closed_text(
                    list(pro_mode_result.tool_invocations or [])
                )
                if fail_closed_text:
                    response_text = fail_closed_text
                supporting_points = [
                    str(item.get("output_summary") or "").strip()
                    for item in list(pro_mode_result.tool_invocations or [])
                    if str(item.get("output_summary") or "").strip()
                ]
                supporting_points.extend(
                    [
                        str(item or "").strip()
                        for item in list(
                            (pro_mode_metadata.get("calculator") or {}).get("results") or []
                        )
                        if str(item or "").strip()
                    ]
                )
                supporting_points.extend(
                    [
                        str(item or "").strip()
                        for item in [
                            *list(metadata_specialist.verified_findings or []),
                            *list(metadata_specialist.filename_inferences or []),
                            *list(metadata_specialist.scientific_context or []),
                            *list(metadata_specialist.equipment_context or []),
                            *list(metadata_specialist.derived_location_context or []),
                        ]
                        if str(item or "").strip()
                    ]
                )
                reservations = [
                    str(item or "").strip()
                    for item in list((pro_mode_metadata.get("verifier") or {}).get("issues") or [])
                    if str(item or "").strip()
                ]
                reservations.extend(
                    [
                        str(item or "").strip()
                        for item in [
                            *list(metadata_specialist.missing_metadata or []),
                            *list(metadata_specialist.caveats or []),
                        ]
                        if str(item or "").strip()
                    ]
                )
                reservations.extend(
                    self._code_execution_fail_closed_reservations(
                        list(pro_mode_result.tool_invocations or [])
                    )
                )
                polished_response_text, writer_stats = await self._run_pro_mode_final_writer(
                    latest_user_text=latest_user_text,
                    draft_response_text=response_text,
                    execution_regime=execution_regime,
                    task_regime=str(
                        pro_mode_metadata.get("task_regime") or intake_decision.task_regime or ""
                    ).strip()
                    or None,
                    supporting_points=supporting_points,
                    reservations=reservations,
                    session_state={
                        "pro_mode_context": dict(shared_context or {}),
                        "pro_mode": pro_mode_metadata,
                    },
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    max_runtime_seconds=min(effective_max_runtime_seconds, 90),
                    debug=pro_mode_debug,
                )
                if polished_response_text:
                    response_text = polished_response_text
                if isinstance(pro_mode_result.metadata, dict):
                    pro_mode_result.metadata.setdefault("pro_mode", {})
                    if isinstance(pro_mode_result.metadata["pro_mode"], dict):
                        pro_mode_result.metadata["pro_mode"]["writer"] = {
                            "applied": bool(polished_response_text),
                            "kind": "final_writer",
                            "compression_stats": writer_stats,
                        }
                        if any(
                            [
                                str(metadata_specialist.direct_answer or "").strip(),
                                list(metadata_specialist.verified_findings or []),
                                list(metadata_specialist.filename_inferences or []),
                                list(metadata_specialist.scientific_context or []),
                                list(metadata_specialist.equipment_context or []),
                                list(metadata_specialist.derived_location_context or []),
                                list(metadata_specialist.missing_metadata or []),
                            ]
                        ):
                            pro_mode_result.metadata["pro_mode"]["metadata_specialist"] = (
                                metadata_specialist.model_dump(mode="json")
                            )
            elif isinstance(pro_mode_result.metadata, dict):
                pro_mode_result.metadata.setdefault("pro_mode", {})
                if isinstance(pro_mode_result.metadata["pro_mode"], dict):
                    pro_mode_result.metadata["pro_mode"]["writer"] = {
                        "applied": False,
                        "kind": "final_writer",
                        "compression_stats": writer_stats,
                    }
            if response_text:
                yield {"event": "token", "data": {"delta": response_text}}
            final_result = self._run_output_to_result(
                run_output=None,
                fallback_text=response_text,
                route=route,
                tool_names=(["numpy_calculator"] if pro_mode_result.tool_invocations else []),
                reasoning_mode=reasoning_mode,
                workflow_hint=workflow_hint,
                hitl_resume=hitl_resume,
                run_id=run_id,
                app_session_id=app_session_id,
                user_id=user_id,
                latest_user_text=latest_user_text,
                memory_policy=resolved_memory_policy,
                knowledge_scope=resolved_knowledge_scope,
                knowledge_context=knowledge_context,
                memory_context=memory_context.metadata(),
                knowledge_result=knowledge_result.metadata(),
                fallback_tool_invocations=pro_mode_result.tool_invocations,
                runtime_status=pro_mode_result.runtime_status,
                runtime_error=pro_mode_result.runtime_error,
                extra_metadata=pro_mode_result.metadata,
                debug_override={
                    "path": "pro_mode",
                    "agent_mode": "workflow",
                    "prompt_profile": "pro_mode",
                    "calculator_scope": "controller_only",
                    "dev_trace_enabled": pro_mode_debug,
                    "intake_route": intake_decision.route,
                    "execution_regime": str(
                        (
                            dict(pro_mode_result.metadata.get("pro_mode") or {})
                            if isinstance(pro_mode_result.metadata, dict)
                            else {}
                        ).get("execution_regime")
                        or intake_decision.execution_regime
                    ),
                    "active_roles": list(
                        (
                            dict(pro_mode_result.metadata.get("pro_mode") or {})
                            if isinstance(pro_mode_result.metadata, dict)
                            else {}
                        ).get("active_roles")
                        or []
                    ),
                    "model_call_count": int(
                        (
                            dict(pro_mode_result.metadata.get("pro_mode") or {})
                            if isinstance(pro_mode_result.metadata, dict)
                            else {}
                        ).get("model_call_count")
                        or 0
                    ),
                },
            )
            memory_metadata = (
                dict(final_result.metadata.get("memory") or {})
                if isinstance(final_result.metadata.get("memory"), dict)
                else {}
            )
            learning_metadata = (
                dict(final_result.metadata.get("learning") or {})
                if isinstance(final_result.metadata.get("learning"), dict)
                else {}
            )
            self._emit_event(
                event_callback,
                {
                    "kind": "graph",
                    "event_type": "memory.updated",
                    "phase": "memory",
                    "status": "updated",
                    "message": (
                        "Updated session memory."
                        if bool(memory_metadata.get("summary_updated"))
                        or bool(memory_metadata.get("writes"))
                        else "No durable memory update was needed."
                    ),
                    **memory_metadata,
                },
            )
            learning_event_type = (
                "learning.promoted"
                if int(learning_metadata.get("promoted_count") or 0) > 0
                else "learning.skipped"
            )
            self._emit_event(
                event_callback,
                {
                    "kind": "graph",
                    "event_type": learning_event_type,
                    "phase": "learning",
                    "status": "completed"
                    if learning_event_type == "learning.promoted"
                    else "skipped",
                    "message": (
                        f"Promoted {int(learning_metadata.get('promoted_count') or 0)} reusable note"
                        f"{'' if int(learning_metadata.get('promoted_count') or 0) == 1 else 's'}."
                        if learning_event_type == "learning.promoted"
                        else "No artifact-backed learning note was promoted."
                    ),
                    **learning_metadata,
                },
            )
            _persist_turn_analysis_state(
                metadata=final_result.metadata,
                task_regime_override=str(
                    (
                        dict(pro_mode_result.metadata.get("pro_mode") or {})
                        if isinstance(pro_mode_result.metadata, dict)
                        else {}
                    ).get("task_regime")
                    or intake_decision.task_regime
                    or ""
                ),
            )
            yield {
                "event": "done",
                "data": {
                    "response_text": final_result.response_text,
                    "selected_domains": final_result.selected_domains,
                    "domain_outputs": final_result.domain_outputs,
                    "tool_calls": final_result.tool_calls,
                    "model": final_result.model,
                    "metadata": final_result.metadata,
                },
            }
            return
        self.memory_service.ensure_session(
            session_id=app_session_id,
            user_id=user_id,
            title=latest_user_text[:120].strip() or "New scientist session",
            memory_policy=resolved_memory_policy,
            knowledge_scope=resolved_knowledge_scope.model_dump(mode="json"),
        )
        memory_context = self.memory_service.retrieve_context(
            session_id=app_session_id,
            user_id=user_id,
            query=latest_user_text,
            policy=resolved_memory_policy,
            knowledge_scope=resolved_knowledge_scope.model_dump(mode="json"),
        )
        knowledge_result = self.knowledge_hub.retrieve_context(
            user_id=user_id,
            session_id=app_session_id,
            query=latest_user_text,
            scope=resolved_knowledge_scope,
            domain_id=route.primary_domain,
            workflow_hint=workflow_hint,
            knowledge_context=knowledge_context,
            selection_context=effective_selection_context,
            uploaded_files=uploaded_files,
        )
        prepared_messages = list(messages)
        if memory_context.system_messages:
            prepared_messages = [*memory_context.system_messages, *prepared_messages]
        self._emit_event(
            event_callback,
            {
                "kind": "graph",
                "event_type": "memory.retrieved",
                "phase": "memory",
                "status": "retrieved",
                "message": (
                    f"Retrieved {memory_context.hit_count} memory item"
                    f"{'' if memory_context.hit_count == 1 else 's'}."
                    if memory_context.hit_count > 0
                    else "No prior memory matched this turn."
                ),
                **memory_context.metadata(),
            },
        )
        if knowledge_result.system_messages:
            prepared_messages = [*knowledge_result.system_messages, *prepared_messages]
        self._emit_event(
            event_callback,
            {
                "kind": "graph",
                "event_type": "knowledge.retrieved",
                "phase": "knowledge",
                "status": "retrieved",
                "message": (
                    f"Loaded {len(knowledge_result.hits)} knowledge hit"
                    f"{'' if len(knowledge_result.hits) == 1 else 's'}."
                    if knowledge_result.hits
                    else "No project notebook or curated knowledge was needed."
                ),
                **knowledge_result.metadata(),
            },
        )
        agent = self._build_agent(
            domain_id=route.primary_domain,
            selected_tool_names=tool_names,
            memory_policy=resolved_memory_policy,
            history_message_count=len(prepared_messages),
            reasoning_mode=reasoning_mode,
            max_tool_calls=max_tool_calls,
            max_runtime_seconds=effective_max_runtime_seconds,
            uploaded_files=uploaded_files,
            user_text=latest_user_text,
            selection_context=effective_selection_context,
            analysis_state=prior_analysis_state,
            debug=bool(debug),
        )
        agno_messages = self._messages_to_agno(prepared_messages)
        response_chunks: list[str] = []
        run_output: RunOutput | None = None
        partial_tool_invocations: list[dict[str, Any]] = []
        runtime_status = "completed"
        runtime_error: str | None = None
        deadline = time.monotonic() + max(1.0, float(effective_max_runtime_seconds))
        post_tool_idle_timeout = min(
            MAX_POST_TOOL_IDLE_TIMEOUT_SECONDS,
            max(12.0, float(effective_max_runtime_seconds) * 0.1),
        )

        if hitl_resume:
            decision = str(hitl_resume.get("decision") or "").strip().lower() or "approve"
            note = str(hitl_resume.get("note") or "").strip() or None
            requirements = self._resume_requirements(
                pending_hitl=pending_hitl,
                decision=decision,
                note=note,
            )
            iterator = agent.acontinue_run(
                run_id=run_id,
                requirements=requirements,
                stream=True,
                stream_events=True,
                yield_run_output=True,
                user_id=user_id,
                session_id=agno_session_id,
                debug_mode=bool(debug),
            )
            self._emit_event(
                event_callback,
                {
                    "kind": "graph",
                    "phase": "approval",
                    "status": "started",
                    "message": f"Resuming paused run with decision={decision}.",
                },
            )
        else:
            iterator = agent.arun(
                agno_messages,
                stream=True,
                stream_events=True,
                yield_run_output=True,
                user_id=user_id,
                session_id=agno_session_id,
                run_id=run_id,
                debug_mode=bool(debug),
                metadata={"conversation_id": conversation_id, "runtime": "agno"},
            )

        iterator_handle = iterator.__aiter__()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                runtime_status = "timeout"
                runtime_error = (
                    f"Run exceeded runtime budget ({int(effective_max_runtime_seconds)}s)."
                )
                break
            step_timeout = remaining
            if partial_tool_invocations:
                step_timeout = min(step_timeout, post_tool_idle_timeout)
            try:
                item = await asyncio.wait_for(iterator_handle.__anext__(), timeout=step_timeout)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                runtime_status = "post_tool_timeout" if partial_tool_invocations else "timeout"
                runtime_error = (
                    "Tool execution completed, but the final model response stalled."
                    if partial_tool_invocations
                    else f"Run exceeded runtime budget ({int(effective_max_runtime_seconds)}s)."
                )
                break
            except (
                Exception
            ) as exc:  # pragma: no cover - live adapter failures are environment-specific
                runtime_status = "error"
                runtime_error = str(exc or exc.__class__.__name__)
                break

            if isinstance(item, RunOutput):
                run_output = item
                continue
            if not isinstance(item, RunOutputEvent):
                continue

            event_name = str(getattr(item, "event", "") or "")
            lowered = event_name.strip().lower()

            if lowered == "runstarted":
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "phase": "solve",
                        "status": "started",
                        "message": "Agno run started.",
                        "payload": {
                            "model": getattr(item, "model", None),
                            "model_provider": getattr(item, "model_provider", None),
                        },
                    },
                )
                continue

            if lowered == "modelrequeststarted":
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "phase": "model_request",
                        "status": "started",
                        "message": "Requesting model completion.",
                        "payload": {
                            "model": getattr(item, "model", None),
                            "model_provider": getattr(item, "model_provider", None),
                        },
                    },
                )
                continue

            if lowered == "modelrequestcompleted":
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "phase": "model_request",
                        "status": "completed",
                        "message": "Model completion received.",
                        "payload": {
                            "input_tokens": getattr(item, "input_tokens", None),
                            "output_tokens": getattr(item, "output_tokens", None),
                            "total_tokens": getattr(item, "total_tokens", None),
                            "reasoning_tokens": getattr(item, "reasoning_tokens", None),
                        },
                    },
                )
                continue

            if lowered == "reasoningstarted":
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "phase": "reasoning",
                        "status": "started",
                        "message": "Model is reasoning.",
                    },
                )
                continue

            if lowered == "reasoningstep":
                reasoning_content = str(getattr(item, "reasoning_content", "") or "").strip()
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "phase": "reasoning",
                        "status": "progress",
                        "message": reasoning_content[:400] or "Reasoning step.",
                    },
                )
                continue

            if lowered == "reasoningcompleted":
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "phase": "reasoning",
                        "status": "completed",
                        "message": "Reasoning finished.",
                    },
                )
                continue

            if lowered == "toolcallstarted":
                tool: ToolExecution | None = getattr(item, "tool", None)
                tool_name = str(getattr(tool, "tool_name", None) or "").strip() or "tool"
                self._emit_event(
                    event_callback,
                    {
                        "kind": "tool",
                        "phase": "tool",
                        "status": "started",
                        "tool_name": tool_name,
                        "message": f"Running {tool_name}.",
                        "payload": {
                            "tool": tool_name,
                            "args": dict(getattr(tool, "tool_args", None) or {}),
                        },
                    },
                )
                continue

            if lowered == "toolcallcompleted":
                tool = getattr(item, "tool", None)
                tool_name = str(getattr(tool, "tool_name", None) or "").strip() or "tool"
                parsed, raw_text = self._json_result(getattr(tool, "result", None))
                partial_tool_invocations.append(
                    {
                        "tool": tool_name,
                        "status": "completed",
                        "args": dict(getattr(tool, "tool_args", None) or {}),
                        "output_envelope": parsed if isinstance(parsed, dict) else {},
                        "output_summary": self._summarize_tool_output(tool_name, parsed, raw_text),
                        "output_preview": raw_text[:800],
                    }
                )
                self._emit_event(
                    event_callback,
                    {
                        "kind": "tool",
                        "phase": "tool",
                        "status": "completed",
                        "tool_name": tool_name,
                        "message": raw_text[:400] or f"{tool_name} completed.",
                        "payload": {
                            "tool": tool_name,
                            "args": dict(getattr(tool, "tool_args", None) or {}),
                            "result": parsed if isinstance(parsed, dict) else {},
                        },
                    },
                )
                continue

            if lowered == "toolcallerror":
                tool = getattr(item, "tool", None)
                tool_name = str(getattr(tool, "tool_name", None) or "").strip() or "tool"
                partial_tool_invocations.append(
                    {
                        "tool": tool_name,
                        "status": "error",
                        "args": dict(getattr(tool, "tool_args", None) or {}),
                        "output_envelope": {},
                        "output_summary": {},
                        "output_preview": str(getattr(item, "error", "") or "")[:800],
                    }
                )
                self._emit_event(
                    event_callback,
                    {
                        "kind": "tool",
                        "phase": "tool",
                        "status": "failed",
                        "tool_name": tool_name,
                        "message": str(getattr(item, "error", "") or f"{tool_name} failed."),
                    },
                )
                continue

            if lowered == "runcontent":
                delta = self._extract_text_from_content(getattr(item, "content", None))
                if delta:
                    response_chunks.append(delta)
                    yield {"event": "token", "data": {"delta": delta}}
                continue

            if lowered == "runpaused":
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "phase": "approval",
                        "status": "started",
                        "message": "Approval required before tool execution can continue.",
                    },
                )
                continue

            if lowered == "runcontinued":
                self._emit_event(
                    event_callback,
                    {
                        "kind": "graph",
                        "phase": "approval",
                        "status": "completed",
                        "message": "Approval resolved; resuming run.",
                    },
                )
                continue

        if runtime_status != "completed" and hasattr(iterator, "aclose"):
            with suppress(Exception):
                await iterator.aclose()
        if runtime_status != "completed":
            self._emit_event(
                event_callback,
                {
                    "kind": "graph",
                    "phase": "solve",
                    "status": "failed",
                    "message": runtime_error or "Run ended early.",
                    "payload": {
                        "runtime_status": runtime_status,
                        "tool_invocation_count": len(partial_tool_invocations),
                    },
                },
            )

        final_result = self._run_output_to_result(
            run_output=run_output,
            fallback_text="".join(response_chunks),
            route=route,
            tool_names=tool_names,
            reasoning_mode=reasoning_mode,
            workflow_hint=workflow_hint,
            hitl_resume=hitl_resume,
            run_id=run_id,
            app_session_id=app_session_id,
            user_id=user_id,
            latest_user_text=latest_user_text,
            memory_policy=resolved_memory_policy,
            knowledge_scope=resolved_knowledge_scope,
            knowledge_context=knowledge_context,
            memory_context=memory_context.metadata(),
            knowledge_result=knowledge_result.metadata(),
            fallback_tool_invocations=partial_tool_invocations,
            runtime_status=runtime_status,
            runtime_error=runtime_error,
        )
        _persist_turn_analysis_state(metadata=final_result.metadata)
        memory_metadata = (
            dict(final_result.metadata.get("memory") or {})
            if isinstance(final_result.metadata.get("memory"), dict)
            else {}
        )
        learning_metadata = (
            dict(final_result.metadata.get("learning") or {})
            if isinstance(final_result.metadata.get("learning"), dict)
            else {}
        )
        self._emit_event(
            event_callback,
            {
                "kind": "graph",
                "event_type": "memory.updated",
                "phase": "memory",
                "status": "updated",
                "message": (
                    "Updated session memory."
                    if bool(memory_metadata.get("summary_updated"))
                    or bool(memory_metadata.get("writes"))
                    else "No durable memory update was needed."
                ),
                **memory_metadata,
            },
        )
        learning_event_type = (
            "learning.promoted"
            if int(learning_metadata.get("promoted_count") or 0) > 0
            else "learning.skipped"
        )
        self._emit_event(
            event_callback,
            {
                "kind": "graph",
                "event_type": learning_event_type,
                "phase": "learning",
                "status": "completed" if learning_event_type == "learning.promoted" else "skipped",
                "message": (
                    f"Promoted {int(learning_metadata.get('promoted_count') or 0)} reusable note"
                    f"{'' if int(learning_metadata.get('promoted_count') or 0) == 1 else 's'}."
                    if learning_event_type == "learning.promoted"
                    else "No artifact-backed learning note was promoted."
                ),
                **learning_metadata,
            },
        )
        yield {
            "event": "done",
            "data": {
                "response_text": final_result.response_text,
                "selected_domains": final_result.selected_domains,
                "domain_outputs": final_result.domain_outputs,
                "tool_calls": final_result.tool_calls,
                "model": final_result.model,
                "metadata": final_result.metadata,
            },
        }

    async def synthesize_tool_endpoint_response(
        self,
        *,
        user_text: str,
        domain_id: str,
        tool_name: str,
        tool_result: dict[str, Any],
        default_response_text: str,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        tool_snapshot = json.dumps(tool_result, ensure_ascii=False, default=str)[:12000]
        segmentation_tool = tool_name in {
            "segment_image_megaseg",
            "segment_image_sam2",
            "segment_image_sam3",
            "sam2_prompt_image",
        }
        prompt = (
            "Summarize this scientific tool result for the user.\n\n"
            f"Domain: {domain_id}\n"
            f"Tool: {tool_name}\n"
            f"User request: {user_text}\n"
            f"Default response: {default_response_text}\n"
            f"Tool result JSON: {tool_snapshot}\n\n"
            + (
                "For segmentation results, keep the answer user-facing and compact: say what the model produced, report mask count and coverage if available, mention overlays or previews if they exist, avoid raw storage paths unless the user asked for them, avoid code snippets unless requested, and do not overclaim mask correctness beyond what was measured.\n\n"
                if segmentation_tool
                else ""
            )
            + "Write a concise but informative final answer."
        )
        agent = self._build_agent(
            domain_id=domain_id or "core",
            selected_tool_names=[],
            reasoning_mode="fast",
            max_tool_calls=1,
            max_runtime_seconds=120,
            user_text=user_text,
        )
        result = await agent.arun(
            prompt,
            stream=False,
            user_id=None,
            session_id=self._scope_session_id(
                conversation_id=conversation_id,
                user_id=None,
                run_id=None,
            ),
            debug_mode=False,
        )
        response_text, _source = self._coerce_visible_output(result)
        final_text = str(response_text or "").strip() or default_response_text
        deterministic_segmentation_text = ""
        if segmentation_tool:
            deterministic_segmentation_text = (
                self._deterministic_segmentation_response_from_payload(
                    tool_name=tool_name,
                    payload=dict(tool_result or {}),
                    latest_user_text=user_text,
                )
            )
        if deterministic_segmentation_text and (
            self._prefers_compact_segmentation_summary(user_text)
            or self._segmentation_response_looks_unpolished(final_text)
        ):
            final_text = deterministic_segmentation_text
        return {
            "response_text": final_text,
            "tool_insights": [],
            "synthesized": final_text != default_response_text,
            "metadata": {"runtime": "agno_tool_endpoint"},
        }

    _CORE_TOOL_NAMES: ClassVar[set[str]] = {
        "upload_to_bisque",
        "add_tags_to_resource",
        "bisque_add_to_dataset",
        "bisque_create_dataset",
        "bisque_add_gobjects",
        "delete_bisque_resource",
        "run_bisque_module",
        "bisque_find_assets",
        "search_bisque_resources",
        "bisque_advanced_search",
        "load_bisque_resource",
        "bisque_fetch_xml",
        "bisque_download_resource",
        "bisque_download_dataset",
        "bisque_ping",
        "structure_report",
        "compare_structures",
        "propose_reactive_sites",
        "formula_balance_check",
        "numpy_calculator",
    }
