"""Catalog-resource discovery + staging tools for the research agent.

These give the agent the autonomy loop scientists need: DISCOVER their own
prior data (``search_resources``), then PULL it into the sandbox to analyze
(``stage_resource_for_analysis``) — e.g. "find the CT scans with 'norm' in the
name and plot the middle slice of each", or "run MegaSeg on the images I
uploaded last week". Both tools are run-anchored: the control plane resolves the
run OWNER server-side from the worker token and scopes every query to that owner's
READABLE catalog — the owner's own resources plus any shared with them (the same
set they see in their Resources page), never another user's private resources.
The agent acts as the owner, so it has exactly the owner's access and never more.
"""

from __future__ import annotations

import json
from typing import Any

from langchain.tools import ToolRuntime, tool

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.context_tools import _public_stage_result, stage_catalog_resources

_RESOURCE_TIMEOUT_SECONDS = 30.0
_MAX_RESULTS = 100


def _resource_headers(context: AgentRunContext | None, settings: RuntimeSettings) -> dict[str, str]:
    """Run-anchored worker auth: the control plane derives the run OWNER from the
    worker token + run id and scopes the catalog to that owner's readable set
    (own + shared-with-owner); the user/org headers never widen access across
    owners."""
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


def search_resources_catalog(
    settings: RuntimeSettings,
    *,
    context: AgentRunContext,
    query: str = "",
    kind: str = "",
    source: str = "",
    limit: int = 50,
) -> dict[str, Any]:
    """Search the run owner's Resources catalog via the control plane."""
    import httpx

    run_id = str(context.run_id or "").strip()
    if not run_id:
        return {"ok": False, "error": "no_run_id", "resources": []}
    safe_limit = max(1, min(int(limit or 50), _MAX_RESULTS))
    payload: dict[str, Any] = {"query": str(query or "").strip(), "limit": safe_limit}
    if str(kind or "").strip():
        payload["kind"] = str(kind).strip()
    if str(source or "").strip():
        payload["source"] = str(source).strip()
    base = settings.control_base_url.rstrip("/")
    url = f"{base}/v2/runs/{run_id}/resource-search"
    try:
        with httpx.Client(timeout=_RESOURCE_TIMEOUT_SECONDS) as client:
            response = client.post(url, json=payload, headers=_resource_headers(context, settings))
            response.raise_for_status()
            data = response.json()
    except Exception as exc:  # noqa: BLE001 - best-effort; never fail the run on discovery
        return {"ok": False, "error": str(exc), "resources": []}
    resources = data.get("resources") if isinstance(data, dict) else None
    total = data.get("total_count") if isinstance(data, dict) else None
    return {
        "ok": True,
        "resources": resources if isinstance(resources, list) else [],
        "total_count": int(total) if isinstance(total, int) else 0,
    }


def resolve_catalog_resources(
    settings: RuntimeSettings,
    *,
    context: AgentRunContext,
    resource_ids: list[str],
) -> dict[str, Any]:
    """Verify the given resource ids are owned by the run owner (control plane)."""
    import httpx

    run_id = str(context.run_id or "").strip()
    if not run_id:
        return {"ok": False, "error": "no_run_id", "resources": [], "missing": list(resource_ids)}
    base = settings.control_base_url.rstrip("/")
    url = f"{base}/v2/runs/{run_id}/resource-resolve"
    try:
        with httpx.Client(timeout=_RESOURCE_TIMEOUT_SECONDS) as client:
            response = client.post(
                url,
                json={"resource_ids": list(resource_ids)},
                headers=_resource_headers(context, settings),
            )
            response.raise_for_status()
            data = response.json()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "resources": [], "missing": list(resource_ids)}
    resources = data.get("resources") if isinstance(data, dict) else None
    missing = data.get("missing") if isinstance(data, dict) else None
    return {
        "ok": True,
        "resources": resources if isinstance(resources, list) else [],
        "missing": missing if isinstance(missing, list) else [],
    }


def _parse_resource_ids(resource_ids: list[str] | str | None) -> list[str]:
    if resource_ids is None:
        return []
    if isinstance(resource_ids, str):
        stripped = resource_ids.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        if "," in stripped:
            return [token.strip() for token in stripped.split(",") if token.strip()]
        return [stripped]
    return [str(item).strip() for item in resource_ids if str(item).strip()]


def build_resource_tools(settings: RuntimeSettings, *, upload_roots: tuple[str, ...] = ()) -> list[Any]:
    """Discover + stage tools over the run owner's Resources catalog."""
    resolved_upload_roots = tuple(upload_roots)

    @tool
    def search_resources(
        runtime: ToolRuntime[AgentRunContext],
        query: str = "",
        kind: str = "",
        source: str = "",
        limit: int = 50,
    ) -> str:
        """Search YOUR Resources library — the user's uploaded datasets, images, and prior results — by keyword. Use this to find data to analyze that was NOT attached to this chat, e.g. "the CT scans with 'norm' in the name", "the NPM1 image we looked at before", "my segmentation outputs". query matches resource names; kind filters by type (image, dataset, document, table, model); source filters origin (upload, bisque_import). Returns each match's resource_id, original_name, kind, size, and metadata. Then call stage_resource_for_analysis with the resource_id(s) to pull the file(s) into /workspace and analyze them with execute()."""
        result = search_resources_catalog(
            settings,
            context=runtime.context,
            query=query,
            kind=kind,
            source=source,
            limit=limit,
        )
        return json.dumps(result, indent=2, sort_keys=True, default=str)

    @tool
    def stage_resource_for_analysis(
        runtime: ToolRuntime[AgentRunContext],
        resource_ids: list[str] | str,
    ) -> str:
        """Copy one or more resources from YOUR library into /workspace/staged_resources/ so execute() code can read and analyze them. Pass resource_id values from search_resources (a single id, a comma-separated list, or a JSON array). Returns each staged file's sandbox_path (a /workspace path) to read in code. Use this to act on data you found with search_resources — plot it, run MegaSeg/RareSpot inference, compute PCA features, train a model, etc. Resources you cannot access come back under "missing"; resources with no locally stored file come back under "unavailable"."""
        ids = _parse_resource_ids(resource_ids)
        if not ids:
            return json.dumps({"ok": False, "error": "no_resource_ids"}, indent=2, sort_keys=True)
        resolved = resolve_catalog_resources(settings, context=runtime.context, resource_ids=ids)
        if not resolved.get("ok"):
            return json.dumps(
                {"ok": False, "error": resolved.get("error", "resolve_failed")},
                indent=2,
                sort_keys=True,
            )
        result = stage_catalog_resources(
            runtime.context,
            upload_roots=resolved_upload_roots,
            resources=resolved.get("resources", []),
        )
        # Resources the owner cannot read are reported separately from ones that
        # are owned but have no locally stageable file.
        public = _public_stage_result(result)
        public["missing"] = list(resolved.get("missing", []))
        return json.dumps(public, indent=2, sort_keys=True, default=str)

    return [search_resources, stage_resource_for_analysis]
