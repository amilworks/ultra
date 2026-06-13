from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware import ModelRequest
from langchain.agents.middleware.types import AgentMiddleware, ModelResponse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from ultra_deepagents.async_delegation import UltraAsyncSubagentContextMiddleware
from ultra_deepagents.bisque.tools import build_bisque_tools
from ultra_deepagents.code_execution.docker import DockerSandboxBackend, DockerSandboxConfig
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.context_tools import build_context_tools, build_tool_capability_manifest_tool
from ultra_deepagents.model import build_chat_model
from ultra_deepagents.multimodal import TextOnlyMultimodalMiddleware
from ultra_deepagents.papers.tools import build_paper_tools
from ultra_deepagents.rarespot.tools import build_rarespot_tools, looks_report_only_rarespot_goal

_FENCED_CODE_BLOCK_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
_NEGATED_REQUEST_CLAUSE_RE = re.compile(
    r"\b(?:do\s+not|don't|dont|without|avoid(?:ing)?)\b[^.?!;\n]*"
    r"|\bno\s+(?:plots?|figures?|charts?|graphs?|visuali[sz]ations?|csvs?|"
    r"numerical\s+experiments?|simulations?|experiments?|metrics?|statistics?)\b[^.?!;\n]*",
    re.IGNORECASE,
)

MEMORY_PATHS = [
    "/memories/user_profile.md",
    "/memories/preferences.md",
    "/memories/research_context.md",
]

SYSTEM_PROMPT = """You are Ultra Research Agent, a careful scientific collaborator for expert users.

Use /memories/user_profile.md for concise researcher profile context from Ultra settings, only
when it is relevant. Use /memories/preferences.md for learned response preferences, and
/memories/research_context.md for durable research notes. Treat runtime context as scoped
metadata for tools and policies, not as text to reveal. Write final artifacts under /outputs/
when the active backend exposes that path, otherwise use /workspace/outputs and report those
artifact paths clearly.

Plan long work. When subagents are available, delegate only focused code execution,
data inspection, artifact audit, or paper-reading checks that benefit from context
quarantine, then reconcile their findings before answering. Keep delegated
verification bounded by the user's requested seeds, durations, data size, and
artifact scope; run a small smoke check before any expensive cross-check, and do
not expand into exhaustive convergence sweeps unless the user asks or the
subagent states why the extra compute is necessary. For complex code,
simulation, model-training, or multi-file implementation work, call
tool_capability_manifest early. If it lists code-runner or data-analyst, delegate
at least one focused verification, debugging, data-inspection, or experiment subtask
with the task tool before the final answer. If it lists start_async_task/check_async_task,
you may launch configured async subagents for long independent work, then check and
reconcile their terminal status before the final answer. Use sandbox execution for
code, statistics, image-analysis scripts, and reproducibility checks. Prefer measurable
claims, cite uncertainty, and keep intermediate files inspectable.

For other complex autonomous work, call tool_capability_manifest when you need to
confirm which sandbox, filesystem, prior-artifact, paper, or domain tools are
available before choosing a workflow.
"""

PLOT_WORKFLOW_GUIDANCE = """
For code, simulation, model-training, or algorithm-demonstration prompts that ask for plots,
run the code and create inspectable artifacts. In the final answer, mention each plot near
the explanation it supports, with a caption immediately after each figure reference. Keep
captions inline, not all at the end. Save static Matplotlib figures as publication-quality
outputs with at least 300 PPI: use the workspace matplotlibrc defaults
(`savefig.dpi: 300`) or call `fig.savefig(..., dpi=300, bbox_inches="tight")` /
`plt.savefig(..., dpi=300, bbox_inches="tight")` explicitly. For animated GIFs, also save a
static 300 PPI summary or key-frame figure when the user needs a durable publication-style
visual. For follow-up plot edits
such as "Change the plot to include error bars", edit
or rerun the existing plotting code and identify the updated artifact instead of starting an
unrelated analysis.
"""

SANDBOX_RUNTIME_GUIDANCE = """
For sandbox execution, do not wrap sandbox commands with shell timeout, gtimeout, or
execute timeout arguments. Long-running analysis is allowed. Prefer scientifically
meaningful convergence checks, smaller smoke-test data, checkpoints, and resumable
artifacts over arbitrary wall-clock caps.
"""

TEXT_ONLY_ARTIFACT_GUIDANCE = """
This deployment's active model is text-only. Do not call read_file on image, audio, video,
or PDF artifacts for visual inspection. Verify generated plots by checking source code,
input data, file existence, file size, and structured outputs such as result.json. Describe
figures from the computations that produced them, and keep artifact paths visible so the UI
can render the files inline.
"""

PRIOR_ARTIFACT_GUIDANCE = """
When this is a follow-up, prior durable artifacts may be listed in the run context. Use
artifact_manifest to inspect them. If code execution needs a prior file, call
stage_artifact_for_analysis first and use the returned /workspace/staged_artifacts path.
Do not say prior files are unavailable until artifact_manifest confirms none are present.
"""

UPLOADED_FILE_GUIDANCE = """
When the user selected or uploaded files for this run, their file IDs are listed in the run
context. Before writing code that reads those files, call stage_uploaded_files_for_analysis
and use the returned /workspace/staged_uploads paths. Do not guess upload paths or claim
uploaded files are unavailable until that tool reports they are missing.
"""

PAPER_REVIEW_GUIDANCE = """
For arXiv links and uploaded/local PDFs, use the paper tools instead of guessing from the
URL or filename. Ingest the paper first, search/read exact pages for claims, equations,
assumptions, methods, limitations, and figure references, and cite paper claims with page
grounding such as paper_id:p7. When a figure or equation page matters to the explanation,
call render_paper_page so the UI can display that page inline. For follow-up questions,
call paper_manifest first, then search_paper/read_paper_pages on the cached paper before
answering. Do not rely only on rendered page images or prior summaries for paper-specific
follow-up claims.
"""

RARESPOT_GUIDANCE = """
For prairie dog or burrow detection requests, use rarespot_ecology_inference as the production
RareSpot path. If uploaded file IDs are present, call rarespot_ecology_inference directly and
let the tool use the run context; do not stage uploaded files just to pass sandbox paths into
RareSpot. The default production RareSpot configuration uses 512 px tiles with 25% overlap.
After a successful RareSpot tool result, answer from its counts_by_class,
confidence_summary, configuration, key_artifacts, and artifact IDs. Do not search the
sandbox filesystem for RareSpot outputs, and do not rerun the same RareSpot configuration
unless the user asks for a second inference pass or a changed threshold/configuration.
Do not create stub or duplicate CSV, JSON, Markdown, or figure outputs for RareSpot results:
the nested RareSpot tool result and its key_artifacts are canonical. Only create a new
report, table, or comparison artifact when the user asks for derived synthesis across runs.
For report-only or synthesis-only follow-ups such as "write a combined report across all
RareSpot runs in this chat", do not call rarespot_ecology_inference again. Use
artifact_manifest and stage_artifact_for_analysis to inspect prior nested RareSpot reports,
CSV files, JSON predictions, and overlays. Call rarespot_ecology_inference again only when
the user explicitly asks for a new inference pass, changed threshold/configuration, or new
image/dataset.
"""

BISQUE_GUIDANCE = """
When BisQue resource or dataset URIs are present, use the BisQue tools through the Go
control plane. Use bisque_search_resources for account-scoped queries, bisque_download_resource
to materialize selected BisQue resources into local file_ids before code execution, and
bisque_upload_files to send local V2 upload file_ids back to the linked BisQue account.
For requests about "my" BisQue resources, pass scope="owner"; for newest/recent resources,
pass sort="recent"; for file-type questions, pass extensions such as ["png"] or
["nii", "nii.gz", "nifti"] instead of estimating from broad search results.
For dataset questions ("do I have any datasets?"), pass resource_type="dataset"; for counts
("how many images do I have?"), pass count_all=True and report the returned count. After
uploading multiple related outputs, group them with bisque_create_dataset using the
resource_uri values from the upload responses.
To push generated figures/results to BisQue, use bisque_upload_workspace_files with the
output path you produced (e.g. /workspace/outputs/figure.png or /outputs/figure.png). When the
user asks to push a figure that was generated in an EARLIER turn, that file is a prior durable
artifact, not a current-run file: pass its durable artifact path (e.g.
outputs/ct_scan_visualization.png), its basename, or its artifact_id from artifact_manifest —
bisque_upload_workspace_files resolves prior artifacts and pushes them by artifact_id. Call
artifact_manifest first if you are unsure of the exact artifact path or id.
When you report an uploaded, pushed, downloaded, or searched BisQue resource to the user, give
the client_view_url returned by the tool — that is the canonical link that opens the resource in
the BisQue web viewer (https://<bisque>/client_service/view?resource=<resource_uri>). Do NOT
report the bare data_service resource_uri (the API URL) as the place to view it, and never
hand-construct a BisQue URL yourself: always use the exact client_view_url the tool returned.
Keep resource_uri only for follow-up tool calls such as bisque_create_dataset.
For numeric metadata comparisons (age, slice count, dose, year, study size, etc.) use the
metadata_filters argument, never tag_query relational operators: BisQue compares numeric
tags lexically, so tag_query='age:>50' incorrectly returns age 7 and omits age 100.
metadata_filters takes [{"tag","op","value"}] with op in eq/ne/gt/gte/lt/lte/contains.
Example: "all CT scans above age 50" -> bisque_search_resources(resource_type="image",
tag_query="modality:CT", metadata_filters=[{"tag":"age","op":"gt","value":"50"}],
scope="owner", count_all=True). Keep string-equality tags (modality:CT) in tag_query for
server-side narrowing and put the numeric comparison in metadata_filters.
Never expose or ask for BisQue credentials in the answer; the control plane owns account auth.
"""


RESULTS_CONTRACT_GUIDANCE = """
Results contract (Pro intelligence — mandatory for the final chat answer, not only the report):
- Report every decision-relevant estimate as mean ± spread from at least 3 seeds or initial
  conditions and at least 2 observation durations, including clearly-resolved cases, and
  carry the spread column into any CSV/table artifact.
- State the decision rule verbatim and apply it per row: assign a definitive class only when
  |estimate| > 3× its spread AND at least one independent non-primary discriminator agrees;
  otherwise label the row marginal.
- Classification tables include the parameters that produced each row: initial conditions,
  duration, step size, estimate ± spread, discriminators agreeing, class, confidence.
- When a structural claim depends on a projection, wrapping, or binning choice (for example
  strobe clusters modulo 2π), report the count under both views or explain the aliasing.
- Probe at least 3 initial conditions at borderline parameter values and state whether the
  classification is basin- or IC-dependent.
- Name the canonical system when one matches and compare against commonly reported behavior;
  never fabricate citations — "commonly reported" phrasing with a confidence note is correct.
- Include a short Limitations paragraph in the chat answer itself.
- Durable /outputs hold only final code, verification scripts, data tables, figures, and the
  report; scratch or diagnostic scripts stay under /workspace (use /workspace/diagnostics/).
"""

SCOPED_DELEGATION_CONFIDENCE_LEVELS = ["high", "medium", "low", "unresolved"]

SCOPED_DELEGATION_RESPONSE_FORMAT = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "One concise paragraph with the subtask result.",
        },
        "key_findings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific numerical, textual, or methodological findings.",
        },
        "artifacts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace or durable output path, when one was produced or inspected.",
                    },
                    "description": {
                        "type": "string",
                        "description": "What the artifact contains and why it matters.",
                    },
                },
                "required": ["path", "description"],
                "additionalProperties": False,
            },
            "description": "Relevant files, reports, figures, or staged inputs.",
        },
        "failures": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Errors, missing inputs, or caveats. Use an empty array when none occurred.",
        },
        "confidence": {
            "type": "string",
            "enum": SCOPED_DELEGATION_CONFIDENCE_LEVELS,
            "description": (
                "Earned confidence level. Use high only when the decision rule passes "
                "(estimate magnitude exceeds 3x its spread, or equivalent); use "
                "unresolved when the evidence cannot distinguish the hypotheses."
            ),
        },
        "confidence_basis": {
            "type": "string",
            "description": "One sentence tying the confidence level to the decision rule or evidence.",
        },
    },
    "required": [
        "summary",
        "key_findings",
        "artifacts",
        "failures",
        "confidence",
        "confidence_basis",
    ],
    "additionalProperties": False,
}

BASE_SUBAGENTS = [
    {
        "name": "literature-reviewer",
        "description": (
            "Reads arXiv/PDF papers with narrow paper tools, searches exact pages, extracts "
            "equations/figures/methods, and returns page-grounded review notes."
        ),
        "response_format": SCOPED_DELEGATION_RESPONSE_FORMAT,
        "system_prompt": (
            "You are a page-grounded literature reviewer for research users. Use only the "
            "provided paper tools to ingest, search, read, and render papers. Never make a "
            "paper-specific claim unless it is grounded in an ingested paper page or chunk. "
            "Return concise notes with: papers reviewed, key claims, methods, equations, "
            "figures/pages worth rendering, limitations, relevance to the user's task, "
            "missing evidence, and confidence. Cite claims as paper_id:pN. If the user asks "
            "about a figure, equation, proof, or architecture diagram, call render_paper_page "
            "for the relevant page."
        ),
    },
]

SCOPED_DELEGATION_CONTEXT_TOOLS = {
    "artifact_manifest",
    "stage_artifact_for_analysis",
    "stage_uploaded_files_for_analysis",
}

SCOPED_DELEGATION_SUBAGENTS = [
    {
        "name": "code-runner",
        "description": (
            "Runs focused sandbox execution, debugging, reproducibility checks, plotting, "
            "and numerical experiments, then returns concise findings and artifact paths."
        ),
        "response_format": SCOPED_DELEGATION_RESPONSE_FORMAT,
        "system_prompt": (
            "You are Ultra's scoped code-runner subagent. Use built-in filesystem and "
            "sandbox execution tools for focused code, statistics, plotting, simulation, "
            "model-training, or reproducibility subtasks. Use the provided context tools to "
            "stage selected uploads and stage prior artifacts before code reads them; avoid "
            "guessing paths. Keep intermediate files under /workspace and durable outputs under "
            "/outputs when available. Preserve the user's requested compute scope: do not add "
            "longer durations, finer step sizes, more seeds, or broader convergence sweeps unless "
            "the subtask explicitly asks for them or a short smoke check reveals a material "
            "uncertainty. Return a concise "
            "final report with commands/scripts run, key numerical results, generated "
            "artifact paths, failures, and confidence. Set confidence=high only when "
            "your evidence passes a stated decision rule (for example estimate "
            "magnitude above 3x its spread); otherwise use medium, low, or unresolved, "
            "and explain the basis in confidence_basis. Do not perform broad literature "
            "review, BisQue account operations, RareSpot inference, or user-facing final "
            "synthesis; the coordinator reconciles your result."
        ),
    },
    {
        "name": "data-analyst",
        "description": (
            "Stages selected uploads or prior artifacts, inspects data/manifests, summarizes "
            "dataset structure, and returns analysis-ready evidence without broad synthesis."
        ),
        "response_format": SCOPED_DELEGATION_RESPONSE_FORMAT,
        "system_prompt": (
            "You are Ultra's scoped data-analyst subagent. Use only the provided context "
            "tools plus built-in filesystem tools to inspect selected uploads, prior "
            "artifacts, manifests, tables, and derived analysis files. Stage files before "
            "reading them from code, avoid guessing paths, and keep outputs concise. Return "
            "what data/artifacts were inspected, important schema/shape/metadata facts, "
            "quality concerns, recommended next analysis steps, and exact staged or durable "
            "paths. Set confidence=high only when the evidence is direct and complete; "
            "otherwise use medium, low, or unresolved, and explain the basis in "
            "confidence_basis. Do not run RareSpot inference, manage BisQue accounts, or "
            "write the final user answer."
        ),
    },
]


_ULTRA_HARNESS_PROFILE_REGISTERED = False


def ensure_ultra_harness_profile() -> None:
    """Keep long code/tool work in the coordinator unless we opt into a narrow subagent."""
    global _ULTRA_HARNESS_PROFILE_REGISTERED
    if _ULTRA_HARNESS_PROFILE_REGISTERED:
        return
    register_harness_profile(
        "openai",
        HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        ),
    )
    _ULTRA_HARNESS_PROFILE_REGISTERED = True


def build_subagents(
    paper_tools: Sequence[BaseTool | Any] | None = None,
    *,
    context: AgentRunContext | None = None,
    context_tools: Sequence[BaseTool | Any] | None = None,
    text_only_model: bool = True,
    skills_sources: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    subagents: list[dict[str, Any]] = []

    if paper_tools:
        literature = dict(BASE_SUBAGENTS[0])
        literature["response_format"] = deepcopy(literature["response_format"])
        literature["tools"] = list(paper_tools)
        subagents.append(literature)

    if _should_register_scoped_delegation_subagents(context):
        delegation_context_tools = _filter_tools_by_name(
            context_tools or (),
            SCOPED_DELEGATION_CONTEXT_TOOLS,
        )
        for template in SCOPED_DELEGATION_SUBAGENTS:
            subagent = dict(template)
            if "response_format" in subagent:
                subagent["response_format"] = deepcopy(subagent["response_format"])
            if subagent["name"] in {"code-runner", "data-analyst"}:
                subagent["tools"] = delegation_context_tools
            subagents.append(subagent)

    for subagent in subagents:
        if skills_sources:
            # Subagents share the parent backend, so the same /skills/ route
            # serves their rigor protocols during delegated verification work.
            subagent["skills"] = list(skills_sources)
        if text_only_model:
            subagent["middleware"] = [TextOnlyMultimodalMiddleware()]
    return subagents


def build_async_subagents(
    settings: RuntimeSettings,
    *,
    context: AgentRunContext | None = None,
) -> list[dict[str, Any]]:
    """Configured remote/background Deep Agents subagents.

    These are intentionally operator-configured rather than inferred from a
    prompt. Deep Agents routes specs with ``graph_id`` through
    AsyncSubAgentMiddleware, which exposes start/check/update/cancel/list tools.
    """
    if not _should_register_async_delegation_subagents(context):
        return []
    return [
        _normalize_configured_async_subagent(spec, index=index)
        for index, spec in enumerate(settings.async_subagents)
    ]


def _normalize_configured_async_subagent(spec: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError(f"RuntimeSettings.async_subagents[{index}] must be an object")
    prefix = f"RuntimeSettings.async_subagents[{index}]"
    normalized = dict(spec)
    normalized["name"] = _required_configured_async_subagent_string(
        spec,
        "name",
        prefix=prefix,
    )
    normalized["description"] = _required_configured_async_subagent_string(
        spec,
        "description",
        prefix=prefix,
    )
    normalized["graph_id"] = _required_configured_async_subagent_string(
        spec,
        "graph_id",
        prefix=prefix,
    )
    return normalized


def _required_configured_async_subagent_string(
    spec: dict[str, Any],
    field: str,
    *,
    prefix: str,
) -> str:
    if field not in spec or spec[field] is None:
        raise ValueError(f"{prefix}.{field} is required")
    value = spec[field]
    if not isinstance(value, str):
        raise ValueError(f"{prefix}.{field} must be a non-empty string")
    value = value.strip()
    if not value:
        raise ValueError(f"{prefix}.{field} is required")
    return value


def _filter_tools_by_name(
    tools: Sequence[BaseTool | Any],
    names: set[str],
) -> list[BaseTool | Any]:
    return [tool for tool in tools if getattr(tool, "name", "") in names]


def looks_scoped_delegation_goal(goal: str) -> bool:
    """True when a goal is computational-study shaped (and not a RareSpot run).

    Single source of truth for both subagent registration and the runner's
    rigor-contract enforcement gate, so the two can never drift apart.
    """
    goal = str(goal or "")
    if looks_report_only_rarespot_goal(goal):
        return False
    lowered = " ".join(goal.lower().split())
    if any(
        token in lowered
        for token in (
            "rarespot",
            "prairie dog",
            "prairie dogs",
            "burrow",
            "burrows",
        )
    ):
        return False
    return any(
        token in lowered
        for token in (
            "analy",
            "code",
            "debug",
            "experiment",
            "metric",
            "model",
            "plot",
            "reproduc",
            "script",
            "simulation",
            "statistics",
            "train",
            "workflow",
        )
    )


def looks_quantitative_rigor_goal(goal: str) -> bool:
    """True when a goal needs the Pro quantitative/scientific results contract.

    This is intentionally narrower than scoped delegation. Code review, debug,
    and workflow-analysis prompts can benefit from subagents, but should not be
    forced into irrelevant ``±`` / 3×-spread language.
    """
    goal = str(goal or "")
    if looks_report_only_rarespot_goal(goal):
        return False
    lowered = " ".join(goal.lower().split())
    if any(
        token in lowered
        for token in (
            "rarespot",
            "prairie dog",
            "prairie dogs",
            "burrow",
            "burrows",
        )
    ):
        return False
    request_text = request_classification_text(goal)
    lowered = " ".join(request_text.lower().split())
    return any(
        token in lowered
        for token in (
            "benchmark",
            "bifurcation",
            "classif",
            "convergence",
            "estimate",
            "experiment",
            "exponent",
            "fit",
            "lyapunov",
            "metric",
            "monte carlo",
            "numerical",
            "parameter sweep",
            "quantitative",
            "regime",
            "simulation",
            "statistic",
            "spectrum",
        )
    )


def request_classification_text(text: str) -> str:
    """Normalize request text before keyword-based contract classification.

    Guards should classify the user's request, not example/source code or
    explicitly negated work such as "do not create plots".
    """
    stripped = _FENCED_CODE_BLOCK_RE.sub(" ", str(text or ""))
    return _NEGATED_REQUEST_CLAUSE_RE.sub(" ", stripped)


def is_rigor_intelligence(context: AgentRunContext | None) -> bool:
    """True when the run's Intelligence selection maps to the rigor tier.

    UI vocabulary: Intelligence "High" (default) keeps standard behavior;
    Intelligence "Pro" arrives as ``workflow_hint.id == "pro_mode"`` on the
    run envelope and activates the enforced results contract.
    """
    if context is None:
        return False
    hint = context.workflow_hint if isinstance(context.workflow_hint, dict) else {}
    return str(hint.get("id") or "").strip().lower() == "pro_mode"


def _should_register_scoped_delegation_subagents(context: AgentRunContext | None) -> bool:
    if context is None:
        return False
    return looks_scoped_delegation_goal(context.goal)


def _should_register_async_delegation_subagents(context: AgentRunContext | None) -> bool:
    if context is None:
        return False
    goal = str(context.goal or "")
    if looks_report_only_rarespot_goal(goal):
        return False
    lowered = " ".join(goal.lower().split())
    if any(
        token in lowered
        for token in (
            "rarespot",
            "prairie dog",
            "prairie dogs",
            "burrow",
            "burrows",
        )
    ):
        return False
    return any(
        token in lowered
        for token in (
            "analy",
            "async",
            "background",
            "code",
            "debug",
            "delegate",
            "experiment",
            "long-running",
            "metric",
            "model",
            "plot",
            "reproduc",
            "script",
            "simulation",
            "statistics",
            "train",
            "workflow",
        )
    )


def build_system_prompt(settings: RuntimeSettings, context: AgentRunContext | None = None) -> str:
    sections = [
        SYSTEM_PROMPT.strip(),
        PLOT_WORKFLOW_GUIDANCE.strip(),
        SANDBOX_RUNTIME_GUIDANCE.strip(),
    ]
    if not settings.model_supports_multimodal:
        sections.append(TEXT_ONLY_ARTIFACT_GUIDANCE.strip())
    sections.append(PRIOR_ARTIFACT_GUIDANCE.strip())
    sections.append(UPLOADED_FILE_GUIDANCE.strip())
    sections.append(PAPER_REVIEW_GUIDANCE.strip())
    sections.append(RARESPOT_GUIDANCE.strip())
    sections.append(BISQUE_GUIDANCE.strip())
    if context is not None:
        brief = build_run_context_brief(context)
        if brief:
            sections.append(brief)
    return "\n\n".join(sections) + "\n"


def build_run_context_brief(context: AgentRunContext, *, max_artifacts: int = 8) -> str:
    lines = [
        "Active run context:",
        f"- run_id: {context.run_id}",
        f"- thread_id: {context.thread_id}",
    ]
    if context.goal.strip():
        lines.append(f"- goal: {context.goal.strip()}")
    if context.selected_file_ids:
        file_ids = ", ".join(context.selected_file_ids)
        lines.append(f"- selected uploaded file ids: {file_ids} | use stage_uploaded_files_for_analysis")
    if str(context.run_metadata.get("bisque_session_id") or "").strip():
        lines.append(
            "- linked BisQue account available: use bisque_search_resources, "
            "bisque_download_resource, bisque_upload_files, and bisque_create_dataset "
            "through the control plane; "
            "use scope='owner' for the user's own resources, sort='recent' for newest-first "
            "queries, extensions=['png'] or extensions=['nii','nii.gz','nifti'] for file-type "
            "searches, resource_type='dataset' for dataset questions, and count_all=True for totals"
        )
    if context.selected_resource_uris:
        resource_uris = ", ".join(context.selected_resource_uris[:8])
        lines.append(f"- selected BisQue resource URIs: {resource_uris} | use bisque_download_resource")
    if context.selected_dataset_uris:
        dataset_uris = ", ".join(context.selected_dataset_uris[:8])
        lines.append(f"- selected BisQue dataset URIs: {dataset_uris} | use BisQue tools before analysis")
    ingested_papers = context.knowledge_context.get("ingested_papers")
    if isinstance(ingested_papers, list) and ingested_papers:
        lines.append("- ingested papers available through paper_manifest/search_paper/read_paper_pages:")
        for paper in ingested_papers[:max_artifacts]:
            if not isinstance(paper, dict):
                continue
            paper_id = str(paper.get("paper_id") or "").strip()
            source = str(paper.get("source") or "").strip()
            page_count = paper.get("page_count")
            status = str(paper.get("extraction_status") or "unknown").strip()
            lines.append(
                f"  - {paper_id or '(no paper_id)'} | pages={page_count} | "
                f"status={status} | source={source or '(cached)'}"
            )
        if len(ingested_papers) > max_artifacts:
            lines.append(f"  - ... {len(ingested_papers) - max_artifacts} more; call paper_manifest")
    artifacts = [
        descriptor
        for descriptor in context.resource_descriptors
        if str(descriptor.get("type") or "artifact") == "artifact"
    ]
    if artifacts:
        lines.append("- prior durable artifacts available:")
        for descriptor in artifacts[:max_artifacts]:
            artifact_id = str(descriptor.get("artifact_id") or "").strip()
            path = str(descriptor.get("path") or descriptor.get("relative_path") or "").strip()
            title = str(descriptor.get("title") or path or artifact_id).strip()
            kind = str(descriptor.get("kind") or "artifact").strip()
            lines.append(
                f"  - {artifact_id or '(no artifact_id)'} | {kind} | {title} | "
                f"{path or '(no path)'} | use stage_artifact_for_analysis"
            )
        if len(artifacts) > max_artifacts:
            lines.append(f"  - ... {len(artifacts) - max_artifacts} more; call artifact_manifest")
    return "\n".join(lines)


def build_runtime_prompt_suffix(
    context: AgentRunContext | None,
    *,
    elapsed_seconds: float | None = None,
) -> str:
    """Request-dependent system-prompt suffix: run brief, wall-clock, contract.

    Pure helper so the middleware stays trivially testable. The wall-clock line
    exists because the model cannot see session time and otherwise mislabels
    inner sandbox compute as the study's wall-clock time.
    """
    if context is None:
        return ""
    sections: list[str] = []
    brief = build_run_context_brief(context)
    if brief:
        sections.append(brief)
    if elapsed_seconds is not None and elapsed_seconds >= 0:
        minutes, seconds = divmod(int(elapsed_seconds), 60)
        sections.append(
            f"Elapsed wall-clock for this run so far: {minutes}m{seconds:02d}s. "
            "When reporting runtimes, report this wall-clock time and any inner "
            "compute time as separate labeled numbers."
        )
    if is_rigor_intelligence(context) and looks_quantitative_rigor_goal(context.goal):
        sections.append(RESULTS_CONTRACT_GUIDANCE.strip())
    return "\n\n".join(sections)


class UltraRunContextPromptMiddleware(AgentMiddleware[Any, Any, Any]):
    """Append the per-run context brief to the assembled system message.

    This must append rather than re-render the whole prompt: user middleware
    runs inside the deepagents base stack, so a wholesale replacement (the
    previous ``@dynamic_prompt`` approach) silently erased sections appended
    by earlier middleware — the skills listing and the task-tool guidance.
    The static portion of the prompt is supplied once at ``create_deep_agent``
    time; only the run-context brief is request-dependent.

    Also appends elapsed wall-clock (the agent is built per run, so instance
    creation approximates run start; the value is set once in ``__init__`` and
    only read afterwards, keeping the instance concurrency-safe) and, for
    Intelligence Pro runs on computational-study goals, the results contract.
    """

    def __init__(self, settings: RuntimeSettings) -> None:
        super().__init__()
        self._settings = settings
        self._started_monotonic = time.monotonic()

    def _brief(self, request: ModelRequest) -> str:
        runtime_context = getattr(request.runtime, "context", None)
        context = runtime_context if isinstance(runtime_context, AgentRunContext) else None
        if context is None:
            return ""
        return build_runtime_prompt_suffix(
            context,
            elapsed_seconds=time.monotonic() - self._started_monotonic,
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        brief = self._brief(request)
        if not brief:
            return handler(request)
        return handler(
            request.override(
                system_message=append_to_system_message(request.system_message, brief)
            )
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        brief = self._brief(request)
        if not brief:
            return await handler(request)
        return await handler(
            request.override(
                system_message=append_to_system_message(request.system_message, brief)
            )
        )


def build_runtime_prompt_middleware(settings: RuntimeSettings) -> Any:
    return UltraRunContextPromptMiddleware(settings)


def build_sandbox_backend(
    settings: RuntimeSettings,
    *,
    workspace_dir: str | Path,
    outputs_dir: str | Path | None = None,
) -> DockerSandboxBackend:
    return DockerSandboxBackend(
        workspace_dir=workspace_dir,
        outputs_dir=outputs_dir,
        config=DockerSandboxConfig(
            image=settings.sandbox_image,
            network=settings.sandbox_network,
            cpus=settings.sandbox_cpus,
            memory=settings.sandbox_memory,
            pids_limit=settings.sandbox_pids_limit,
            timeout_seconds=settings.sandbox_timeout_seconds,
            output_limit_bytes=settings.sandbox_output_limit_bytes,
        ),
    )


def _memory_user_slug(user_id: str | None) -> str:
    """Filesystem-safe slug for a user id, so per-user memory directories never
    collide or escape the memory root."""
    raw = str(user_id or "").strip()
    if not raw:
        return ""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_.")
    return slug[:128]


def resolve_user_memory_root(
    settings: RuntimeSettings,
    user_id: str | None,
    *,
    thread_id: str | None = None,
) -> Path:
    """Per-user durable memory directory under the shared memory root.

    Scoping ``/memories/`` per user keeps one researcher's preferences and
    research context from leaking into another's runs. Falls back to the shared
    memory when no user is attached. The shared ``papers/`` cache lives directly
    under the memory root, so user and anonymous directories are nested to avoid
    collisions.
    """
    base = Path(settings.memory_root)
    slug = _memory_user_slug(user_id)
    if slug:
        return base / "users" / slug
    thread_slug = _memory_user_slug(thread_id)
    if thread_slug:
        return base / "anonymous_threads" / thread_slug
    return base / "anonymous" / "unscoped"


SKILLS_SOURCES = ["/skills/"]
"""Backend-relative skill sources for SkillsMiddleware (progressive disclosure)."""


def resolve_skills_root(settings: RuntimeSettings) -> Path | None:
    """Directory holding repo-shipped agent skills, or None when unavailable.

    An explicit ``settings.skills_root`` wins; otherwise fall back to the
    ``skills/`` directory shipped next to this package so the worker finds the
    protocols regardless of its working directory. Only a directory containing
    at least one ``*/SKILL.md`` counts, so deployments without skills register
    no middleware instead of advertising an empty skill list.
    """
    if settings.skills_root.strip():
        root = Path(settings.skills_root).expanduser()
    else:
        root = Path(__file__).resolve().parents[2] / "skills"
    if not root.is_dir():
        return None
    if not any(root.glob("*/SKILL.md")):
        return None
    return root


def build_agent_backend(
    settings: RuntimeSettings,
    *,
    workspace_dir: str | Path,
    artifact_dir: str | Path | None = None,
    user_id: str | None = None,
    thread_id: str | None = None,
) -> CompositeBackend:
    """Route sandbox execution separately from durable agent files."""
    memory_root = resolve_user_memory_root(settings, user_id, thread_id=thread_id)
    artifact_root = Path(artifact_dir) if artifact_dir is not None else Path(settings.artifact_root)
    memory_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    routes: dict[str, Any] = {
        "/memories/": FilesystemBackend(memory_root, virtual_mode=True),
        "/outputs/": FilesystemBackend(artifact_root, virtual_mode=True),
    }
    skills_root = resolve_skills_root(settings)
    if skills_root is not None:
        routes["/skills/"] = FilesystemBackend(skills_root, virtual_mode=True)

    return CompositeBackend(
        default=build_sandbox_backend(
            settings,
            workspace_dir=workspace_dir,
            outputs_dir=artifact_root,
        ),
        routes=routes,
        artifacts_root="/outputs/.deepagents",
    )


def build_research_agent(
    settings: RuntimeSettings,
    *,
    model: BaseChatModel | None = None,
    backend: Any | None = None,
    workspace_dir: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    tools: Sequence[BaseTool | Any] | None = None,
    context: AgentRunContext | None = None,
    checkpointer: Any | None = None,
) -> Any:
    ensure_ultra_harness_profile()
    resolved_backend = backend
    skills_sources: list[str] | None = None
    if resolved_backend is None and workspace_dir is not None:
        resolved_backend = build_agent_backend(
            settings,
            workspace_dir=workspace_dir,
            artifact_dir=artifact_dir,
            user_id=context.user_id if context is not None else None,
            thread_id=context.thread_id if context is not None else None,
        )
        # Skills ride the /skills/ route of the backend we just built; a
        # caller-supplied backend has no such route, so sources stay unset.
        if resolve_skills_root(settings) is not None:
            skills_sources = list(SKILLS_SOURCES)
    if resolved_backend is None:
        resolved_backend = StateBackend()

    middleware: list[Any] = []
    middleware.append(build_runtime_prompt_middleware(settings))
    if not settings.model_supports_multimodal:
        middleware.append(TextOnlyMultimodalMiddleware())

    resolved_tools = list(tools or [])
    context_tools = build_context_tools(upload_roots=settings.rarespot_upload_roots)
    paper_tools = (
        build_paper_tools(
            upload_roots=settings.rarespot_upload_roots,
            cache_root=Path(settings.memory_root) / "papers",
        )
        if _should_register_paper_tools(context)
        else []
    )
    resolved_tools.extend(context_tools)
    resolved_tools.extend(paper_tools)
    if _should_register_bisque_tools(context):
        resolved_tools.extend(build_bisque_tools(settings))
    if _should_register_rarespot_tools(context):
        resolved_tools.extend(build_rarespot_tools(settings))
    subagents = build_subagents(
        paper_tools,
        context=context,
        context_tools=context_tools,
        text_only_model=not settings.model_supports_multimodal,
        skills_sources=skills_sources,
    )
    async_subagents = build_async_subagents(settings, context=context)
    if async_subagents:
        middleware.append(UltraAsyncSubagentContextMiddleware(async_subagents))
    resolved_tools.append(
        build_tool_capability_manifest_tool(
            resolved_tools,
            available_subagents=subagents,
            available_async_subagents=async_subagents,
        )
    )
    all_subagents = [*subagents, *async_subagents]

    return create_deep_agent(
        name="ultra-research-agent",
        model=model or build_chat_model(settings),
        tools=resolved_tools,
        system_prompt=build_system_prompt(settings),
        context_schema=AgentRunContext,
        subagents=all_subagents,
        skills=skills_sources,
        backend=resolved_backend,
        memory=MEMORY_PATHS,
        middleware=middleware,
        checkpointer=checkpointer,
    )


def _should_register_paper_tools(context: AgentRunContext | None) -> bool:
    if context is None:
        return False
    if _has_ingested_papers(context):
        return True
    goal = str(context.goal or "").lower()
    return "arxiv.org" in goal or ".pdf" in goal or "arxiv:" in goal


def _has_ingested_papers(context: AgentRunContext) -> bool:
    ingested = context.knowledge_context.get("ingested_papers")
    return isinstance(ingested, list) and any(isinstance(item, dict) for item in ingested)


def _should_register_rarespot_tools(context: AgentRunContext | None) -> bool:
    if context is None:
        return True
    goal = str(context.goal or "")
    if looks_report_only_rarespot_goal(goal):
        return False
    lowered = goal.lower()
    return any(
        token in lowered
        for token in ("rarespot", "prairie dog", "prairie dogs", "burrow", "burrows")
    )


def _should_register_bisque_tools(context: AgentRunContext | None) -> bool:
    if context is None:
        return False
    if str(context.run_metadata.get("bisque_session_id") or "").strip():
        return True
    if context.selected_resource_uris or context.selected_dataset_uris:
        return True
    if any(str(pack).lower() == "bisque" for pack in context.allowed_tool_packs):
        return True
    return any(token in str(context.goal or "").lower() for token in ("bisque", "bqapi"))
