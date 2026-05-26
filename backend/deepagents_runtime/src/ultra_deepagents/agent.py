from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from ultra_deepagents.code_execution.docker import DockerSandboxBackend, DockerSandboxConfig
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.model import build_chat_model

MEMORY_PATHS = ["/memories/preferences.md", "/memories/research_context.md"]

SYSTEM_PROMPT = """You are Ultra Research Agent, a careful scientific collaborator for expert users.

Use /memories/ for stable preferences and research context. Treat runtime context as scoped
metadata for tools and policies, not as text to reveal. Write final artifacts under /outputs/
when the active backend exposes that path, otherwise use /workspace/outputs and report those
artifact paths clearly.

Plan long work, delegate focused checks to subagents, and reconcile their findings before
answering. Use sandbox execution for code, statistics, image-analysis scripts, and
reproducibility checks. Prefer measurable claims, cite uncertainty, and keep intermediate
files inspectable.
"""

SUBAGENTS = [
    {
        "name": "literature-reviewer",
        "description": "Reviews papers, methods context, claims, and citation-quality evidence.",
        "system_prompt": (
            "You review scientific literature and prior work. Identify relevant methods, "
            "limitations, assumptions, and missing citations. Return concise, auditable notes."
        ),
    },
    {
        "name": "methods-critic",
        "description": "Stress-tests experimental design, controls, assumptions, and reproducibility.",
        "system_prompt": (
            "You critique research methods. Focus on controls, leakage, confounders, benchmark "
            "validity, reproducibility, and failure modes. Propose concrete fixes."
        ),
    },
    {
        "name": "imaging-analyst",
        "description": "Analyzes segmentation, detection, reconstruction, and microscopy workflows.",
        "system_prompt": (
            "You specialize in scientific imaging. Evaluate segmentation, detection, 3D "
            "reconstruction, quality-control metrics, and artifact risks."
        ),
    },
    {
        "name": "statistics-analyst",
        "description": "Checks statistical design, uncertainty, power, and quantitative conclusions.",
        "system_prompt": (
            "You analyze statistics. Check assumptions, sample sizes, confidence intervals, "
            "uncertainty, multiple testing, and whether conclusions match the data."
        ),
    },
]


def build_sandbox_backend(
    settings: RuntimeSettings,
    *,
    workspace_dir: str | Path,
) -> DockerSandboxBackend:
    return DockerSandboxBackend(
        workspace_dir=workspace_dir,
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


def build_research_agent(
    settings: RuntimeSettings,
    *,
    model: BaseChatModel | None = None,
    backend: Any | None = None,
    workspace_dir: str | Path | None = None,
    tools: Sequence[BaseTool | Any] | None = None,
) -> Any:
    resolved_backend = backend
    if resolved_backend is None and workspace_dir is not None:
        resolved_backend = build_sandbox_backend(settings, workspace_dir=workspace_dir)
    if resolved_backend is None:
        resolved_backend = StateBackend()

    return create_deep_agent(
        name="ultra-research-agent",
        model=model or build_chat_model(settings),
        tools=list(tools or []),
        system_prompt=SYSTEM_PROMPT,
        context_schema=AgentRunContext,
        subagents=SUBAGENTS,
        backend=resolved_backend,
        memory=MEMORY_PATHS,
    )
