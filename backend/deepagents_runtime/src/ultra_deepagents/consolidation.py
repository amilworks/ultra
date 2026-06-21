"""Background memory consolidation scaffold.

A separate deep agent that, between conversations, reviews a user's recent runs
(via the episodic-search endpoint) and merges durable findings into that user's
``research_context/`` notes. This keeps the hot path lean while still building a
living lab notebook. It is intentionally *not* wired to an always-on cron here:
an operator should enable it only once usage justifies the cost. The lookback
window passed to ``search_past_research`` MUST match the cron cadence, or runs
are reprocessed (too-frequent) or dropped (too-rare).

The consolidation agent shares the main agent's per-user memory backend and
permissions, so its writes land in the same ``/memories/research_context/`` the
chat agent reads, and it cannot author app/org-owned files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel

from ultra_deepagents.agent import (
    MEMORY_PATHS,
    SKILLS_SOURCES,
    build_agent_backend,
    resolve_memory_permissions,
    resolve_skills_root,
)
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.episodic.tools import build_episodic_tools
from ultra_deepagents.model import build_chat_model

CONSOLIDATION_SYSTEM_PROMPT = """You are Ultra's memory consolidation agent. You run between a
researcher's conversations to keep their durable research notes current. You are not talking to
the user; you only read recent history and update memory files.

Procedure:
1. Call search_past_research (use since_days to match the consolidation window) to retrieve the
   researcher's recent completed sessions.
2. For each session that established something reusable — a dataset's characteristics, a chosen
   method and its parameters, a conclusion and the evidence for it, or a decision and why —
   ensure it is captured under /memories/research_context/<project-or-dataset-slug>.md as a
   dated, evidence-linked entry, and keep /memories/research_context/INDEX.md as an accurate,
   dated table of contents.
3. Merge rather than duplicate: if a finding is already recorded, refine it; do not create a
   second entry. Remove nothing that is unrelated to the sessions you reviewed.
4. Skip one-off scratch, trivial Q&A, and anything already faithfully recorded.

Keep entries concise and factual. Do not fabricate findings that are not in the retrieved
sessions. When done, write a one-line summary of what you changed.
"""

CONSOLIDATION_DEFAULT_PROMPT = (
    "Consolidate my recent research sessions into durable notes. Review the last "
    "{since_days} days and update /memories/research_context/ and its INDEX.md."
)


def build_consolidation_agent(
    settings: RuntimeSettings,
    *,
    context: AgentRunContext,
    workspace_dir: str | Path,
    model: BaseChatModel | None = None,
) -> Any:
    """Build the per-user consolidation deep agent.

    Shares the main agent's per-user memory backend + permissions, and exposes
    only the episodic-search tool (its sole input is the user's run history).
    """
    backend = build_agent_backend(
        settings,
        workspace_dir=workspace_dir,
        user_id=context.user_id,
        thread_id=context.thread_id,
        org_id=context.org_id,
    )
    skills_sources = list(SKILLS_SOURCES) if resolve_skills_root(settings) is not None else None
    return create_deep_agent(
        name="ultra-consolidation-agent",
        model=model or build_chat_model(settings),
        tools=list(build_episodic_tools(settings)),
        system_prompt=CONSOLIDATION_SYSTEM_PROMPT,
        context_schema=AgentRunContext,
        skills=skills_sources,
        backend=backend,
        memory=MEMORY_PATHS,
        permissions=resolve_memory_permissions(settings),
    )
