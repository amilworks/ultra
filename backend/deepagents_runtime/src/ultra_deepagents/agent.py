from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from ultra_deepagents.code_execution.docker import DockerSandboxBackend, DockerSandboxConfig
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.context_tools import build_context_tools, build_tool_capability_manifest_tool
from ultra_deepagents.model import build_chat_model
from ultra_deepagents.multimodal import TextOnlyMultimodalMiddleware
from ultra_deepagents.papers.tools import build_paper_tools
from ultra_deepagents.rarespot.tools import build_rarespot_tools, looks_report_only_rarespot_goal

MEMORY_PATHS = ["/memories/preferences.md", "/memories/research_context.md"]

SYSTEM_PROMPT = """You are Ultra Research Agent, a careful scientific collaborator for expert users.

Use /memories/ for stable preferences and research context. Treat runtime context as scoped
metadata for tools and policies, not as text to reveal. Write final artifacts under /outputs/
when the active backend exposes that path, otherwise use /workspace/outputs and report those
artifact paths clearly.

Plan long work. When subagents are available, delegate only focused review or paper-reading
checks that benefit from context quarantine, then reconcile their findings before answering.
Use sandbox execution for code, statistics, image-analysis scripts, and
reproducibility checks. Prefer measurable claims, cite uncertainty, and keep intermediate
files inspectable.

For complex autonomous work, call tool_capability_manifest when you need to confirm
which sandbox, filesystem, prior-artifact, paper, or domain tools are available before
choosing a workflow.
"""

PLOT_WORKFLOW_GUIDANCE = """
For code, simulation, model-training, or algorithm-demonstration prompts that ask for plots,
run the code and create inspectable artifacts. In the final answer, mention each plot near
the explanation it supports, with a caption immediately after each figure reference. Keep
captions inline, not all at the end. For follow-up plot edits
such as "Change the plot to include error bars", edit
or rerun the existing plotting code and identify the updated artifact instead of starting an
unrelated analysis.
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


BASE_SUBAGENTS = [
    {
        "name": "literature-reviewer",
        "description": (
            "Reads arXiv/PDF papers with narrow paper tools, searches exact pages, extracts "
            "equations/figures/methods, and returns page-grounded review notes."
        ),
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
    text_only_model: bool = True,
) -> list[dict[str, Any]]:
    if not paper_tools:
        return []

    subagents = [dict(subagent) for subagent in BASE_SUBAGENTS]
    subagents[0]["tools"] = list(paper_tools)
    if text_only_model:
        for subagent in subagents:
            subagent["middleware"] = [TextOnlyMultimodalMiddleware()]
    return subagents


def build_system_prompt(settings: RuntimeSettings, context: AgentRunContext | None = None) -> str:
    sections = [SYSTEM_PROMPT.strip(), PLOT_WORKFLOW_GUIDANCE.strip()]
    if not settings.model_supports_multimodal:
        sections.append(TEXT_ONLY_ARTIFACT_GUIDANCE.strip())
    sections.append(PRIOR_ARTIFACT_GUIDANCE.strip())
    sections.append(UPLOADED_FILE_GUIDANCE.strip())
    sections.append(PAPER_REVIEW_GUIDANCE.strip())
    sections.append(RARESPOT_GUIDANCE.strip())
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
            run_id = str(descriptor.get("run_id") or "").strip()
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


def build_runtime_prompt_middleware(settings: RuntimeSettings) -> Any:
    @dynamic_prompt
    def ultra_runtime_system_prompt(request: ModelRequest) -> str:
        runtime_context = getattr(request.runtime, "context", None)
        context = runtime_context if isinstance(runtime_context, AgentRunContext) else None
        return build_system_prompt(settings, context)

    return ultra_runtime_system_prompt


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


def build_agent_backend(
    settings: RuntimeSettings,
    *,
    workspace_dir: str | Path,
    artifact_dir: str | Path | None = None,
) -> CompositeBackend:
    """Route sandbox execution separately from durable agent files."""
    memory_root = Path(settings.memory_root)
    artifact_root = Path(artifact_dir) if artifact_dir is not None else Path(settings.artifact_root)
    memory_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    return CompositeBackend(
        default=build_sandbox_backend(
            settings,
            workspace_dir=workspace_dir,
            outputs_dir=artifact_root,
        ),
        routes={
            "/memories/": FilesystemBackend(memory_root, virtual_mode=True),
            "/outputs/": FilesystemBackend(artifact_root, virtual_mode=True),
        },
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
) -> Any:
    ensure_ultra_harness_profile()
    resolved_backend = backend
    if resolved_backend is None and workspace_dir is not None:
        resolved_backend = build_agent_backend(
            settings,
            workspace_dir=workspace_dir,
            artifact_dir=artifact_dir,
        )
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
    if _should_register_rarespot_tools(context):
        resolved_tools.extend(build_rarespot_tools(settings))
    resolved_tools.append(build_tool_capability_manifest_tool(resolved_tools))

    return create_deep_agent(
        name="ultra-research-agent",
        model=model or build_chat_model(settings),
        tools=resolved_tools,
        system_prompt=build_system_prompt(settings),
        context_schema=AgentRunContext,
        subagents=build_subagents(
            paper_tools,
            context=context,
            text_only_model=not settings.model_supports_multimodal,
        ),
        backend=resolved_backend,
        memory=MEMORY_PATHS,
        middleware=middleware,
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
