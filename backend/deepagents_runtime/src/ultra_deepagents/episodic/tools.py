"""Episodic memory: a tool to search the user's own past Ultra conversations.

Adapts the deep agents episodic-memory pattern to Ultra's self-hosted control
plane (we are not on LangGraph Platform, so the SDK ``threads.search`` is not
available). The tool calls the run-anchored ``POST /v2/runs/{run_id}/episodic-search``
endpoint: the worker passes only the run id + worker token, and the control plane
resolves the run owner server-side, so a user can only ever search their own
history. Mirrors the run-anchored auth headers used by the BisQue tools.
"""

from __future__ import annotations

import json
from typing import Any

from langchain.tools import ToolRuntime, tool

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext

_EPISODIC_TIMEOUT_SECONDS = 30.0
_MAX_HITS = 20


def _episodic_headers(context: AgentRunContext | None, settings: RuntimeSettings) -> dict[str, str]:
    headers: dict[str, str] = {}
    worker_token = str(getattr(settings, "control_worker_token", "") or "").strip()
    if worker_token:
        headers["X-Ultra-Worker-Token"] = worker_token
    if context is not None:
        run_id = str(context.run_id or "").strip()
        if run_id:
            headers["X-Ultra-Run-Id"] = run_id
        user_id = str(context.user_id or "").strip()
        if user_id:
            headers["X-Ultra-User-Id"] = user_id
        org_id = str(context.org_id or "").strip()
        if org_id:
            headers["X-Ultra-Org-Id"] = org_id
    return headers


def search_past_research_history(
    settings: RuntimeSettings,
    *,
    context: AgentRunContext,
    query: str,
    since_days: int = 0,
    limit: int = 8,
) -> dict[str, Any]:
    """Call the control plane episodic-search endpoint for this run's owner."""
    import httpx

    run_id = str(context.run_id or "").strip()
    if not run_id:
        return {"ok": False, "error": "no_run_id", "hits": []}
    safe_limit = max(1, min(int(limit or 8), _MAX_HITS))
    payload: dict[str, Any] = {"query": str(query or "").strip(), "limit": safe_limit}
    if since_days and int(since_days) > 0:
        payload["since_days"] = int(since_days)
    base = settings.control_base_url.rstrip("/")
    url = f"{base}/v2/runs/{run_id}/episodic-search"
    try:
        with httpx.Client(timeout=_EPISODIC_TIMEOUT_SECONDS) as client:
            response = client.post(url, json=payload, headers=_episodic_headers(context, settings))
            response.raise_for_status()
            data = response.json()
    except Exception as exc:  # noqa: BLE001 - best-effort; never fail the run on recall
        return {"ok": False, "error": str(exc), "hits": []}
    hits = data.get("hits") if isinstance(data, dict) else None
    return {"ok": True, "hits": hits if isinstance(hits, list) else []}


def build_episodic_tools(settings: RuntimeSettings) -> list[Any]:
    """Build the episodic-memory tool(s) for the research agent."""

    @tool
    def search_past_research(
        runtime: ToolRuntime[AgentRunContext],
        query: str = "",
        since_days: int = 0,
    ) -> str:
        """Search THIS researcher's own past Ultra conversations and runs for prior work.

        Use for follow-ups that depend on earlier sessions: "what did we conclude about this
        dataset last month?", "the parameters from my last RareSpot run", "compare my 2024
        and 2026 results". Matches the query against past run goals, final answers, and
        conversation titles, most recent first. Leave query empty to list the most recent
        runs. since_days optionally restricts to the last N days. Returns dated summaries with
        thread/run ids and a response snippet — cite recalled findings by date, and never
        invent a prior conclusion that is not in the results.
        """
        result = search_past_research_history(
            settings,
            context=runtime.context,
            query=query,
            since_days=since_days,
        )
        return json.dumps(result, indent=2, sort_keys=True, default=str)

    return [search_past_research]
