from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any

from langchain.tools import ToolRuntime, tool

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext


def control_post_json(
    settings: RuntimeSettings,
    path: str,
    payload: dict[str, Any],
    *,
    context: AgentRunContext | None = None,
) -> dict[str, Any]:
    """Post to the Go control plane without forwarding model/provider credentials."""
    import httpx

    url = f"{settings.control_base_url.rstrip('/')}/{path.lstrip('/')}"
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            url,
            json=payload,
            headers=_control_headers(context, include_principal=True),
        )
        response.raise_for_status()
        data = response.json()
    return dict(data) if isinstance(data, dict) else {"ok": True, "data": data}


def control_post_files(
    settings: RuntimeSettings,
    path: str,
    files: list[tuple[str, tuple[str, Any, str]]],
    *,
    context: AgentRunContext | None = None,
) -> dict[str, Any]:
    """Post multipart files to the Go control plane without forwarding provider credentials."""
    import httpx

    url = f"{settings.control_base_url.rstrip('/')}/{path.lstrip('/')}"
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            url,
            files=files,
            headers=_control_headers(context, include_principal=True),
        )
        response.raise_for_status()
        data = response.json()
    return dict(data) if isinstance(data, dict) else {"ok": True, "data": data}


def search_bisque_resources(
    settings: RuntimeSettings,
    *,
    resource_type: str = "image",
    tag_query: str = "",
    tag_order: str = "",
    query: str = "",
    name_contains: str = "",
    extensions: list[str] | str | None = None,
    scope: str = "",
    sort: str = "",
    limit: int = 25,
    offset: int = 0,
    count_all: bool = False,
    context: AgentRunContext | None = None,
) -> dict[str, Any]:
    safe_limit = max(1, min(int(limit or 25), 100))
    payload: dict[str, Any] = {
        "resource_type": str(resource_type or "image").strip() or "image",
        "tag_query": str(tag_query or "").strip(),
        "query": str(query or "").strip(),
        "limit": safe_limit,
    }
    if str(tag_order or "").strip():
        payload["tag_order"] = str(tag_order or "").strip()
    if str(name_contains or "").strip():
        payload["name_contains"] = str(name_contains or "").strip()
    cleaned_extensions = _string_list(extensions)
    if cleaned_extensions:
        payload["extensions"] = cleaned_extensions
    if str(scope or "").strip():
        payload["scope"] = str(scope or "").strip()
    if str(sort or "").strip():
        payload["sort"] = str(sort or "").strip()
    safe_offset = max(0, int(offset or 0))
    if safe_offset > 0:
        payload["offset"] = safe_offset
    if count_all:
        payload["count_all"] = True
    return control_post_json(
        settings,
        "/v2/bisque/search",
        payload,
        context=context,
    )


def download_bisque_resources(
    settings: RuntimeSettings,
    *,
    resources: list[str],
    context: AgentRunContext | None = None,
) -> dict[str, Any]:
    cleaned = [str(resource).strip() for resource in resources if str(resource).strip()]
    if not cleaned:
        return {
            "ok": False,
            "error": "no_resources",
            "file_count": 0,
            "uploaded": [],
            "imports": [],
            "results": [],
            "failed_count": 0,
        }

    uploaded: list[dict[str, Any]] = []
    imports: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    failed_count = 0
    for resource in cleaned:
        try:
            response = control_post_json(
                settings,
                "/v2/bisque/download",
                {"resources": [resource]},
                context=context,
            )
        except Exception as exc:
            failed_count += 1
            results.append(_control_failure_result(resource, exc))
            continue

        resource_uploads = [
            item
            for item in response.get("uploaded", [])
            if isinstance(item, dict)
        ] if isinstance(response.get("uploaded"), list) else []
        resource_imports = [
            item
            for item in response.get("imports", [])
            if isinstance(item, dict)
        ] if isinstance(response.get("imports"), list) else []
        uploaded.extend(resource_uploads)
        imports.extend(resource_imports)
        results.append(
            {
                "resource_uri": resource,
                "ok": bool(resource_uploads or resource_imports),
                "status": "imported" if resource_uploads or resource_imports else "empty",
                "file_ids": [
                    str(item.get("file_id") or "").strip()
                    for item in resource_uploads
                    if str(item.get("file_id") or "").strip()
                ],
                "response": response,
            }
        )

    return {
        "ok": bool(uploaded or imports),
        "file_count": len(uploaded),
        "uploaded": uploaded,
        "imports": imports,
        "results": results,
        "failed_count": failed_count,
    }


def upload_bisque_outputs(
    settings: RuntimeSettings,
    *,
    file_ids: list[str] | None = None,
    artifact_ids: list[str] | None = None,
    context: AgentRunContext | None = None,
) -> dict[str, Any]:
    cleaned = [
        str(file_id).strip()
        for file_id in (file_ids or [])
        if str(file_id).strip()
    ]
    cleaned_artifacts = [
        str(artifact_id).strip()
        for artifact_id in (artifact_ids or [])
        if str(artifact_id).strip()
    ]
    return control_post_json(
        settings,
        "/v2/bisque/upload",
        {"file_ids": cleaned, "artifact_ids": cleaned_artifacts},
        context=context,
    )


def upload_bisque_workspace_files(
    settings: RuntimeSettings,
    *,
    paths: list[str] | str | None = None,
    context: AgentRunContext | None = None,
) -> dict[str, Any]:
    if context is None:
        return {"ok": False, "error": "missing_run_context"}
    resolved = _resolve_run_file_paths(paths, context)
    if not resolved:
        return {"ok": False, "error": "no_paths", "uploaded_file_ids": []}

    handles = []
    files: list[tuple[str, tuple[str, Any, str]]] = []
    try:
        for path in resolved:
            handle = path.open("rb")
            handles.append(handle)
            mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            files.append(("files", (path.name, handle, mime_type)))
        upload_response = control_post_files(
            settings,
            "/v2/uploads",
            files,
            context=context,
        )
    finally:
        for handle in handles:
            try:
                handle.close()
            except OSError:
                pass

    uploaded = upload_response.get("uploaded")
    file_ids = [
        str(item.get("file_id") or "").strip()
        for item in uploaded
        if isinstance(item, dict) and str(item.get("file_id") or "").strip()
    ] if isinstance(uploaded, list) else []
    if not file_ids:
        return {
            "ok": False,
            "error": "control_upload_returned_no_file_ids",
            "upload_response": upload_response,
            "uploaded_file_ids": [],
        }
    bisque_upload = upload_bisque_outputs(
        settings,
        file_ids=file_ids,
        artifact_ids=[],
        context=context,
    )
    return {
        "ok": True,
        "uploaded_file_ids": file_ids,
        "control_upload": upload_response,
        "bisque_upload": bisque_upload,
    }


def upload_bisque_files(
    settings: RuntimeSettings,
    *,
    file_ids: list[str],
    context: AgentRunContext | None = None,
) -> dict[str, Any]:
    return upload_bisque_outputs(settings, file_ids=file_ids, artifact_ids=[], context=context)


def build_bisque_tools(settings: RuntimeSettings) -> list[Any]:
    @tool
    def bisque_search_resources(
        runtime: ToolRuntime[AgentRunContext],
        resource_type: str = "image",
        tag_query: str = "",
        tag_order: str = "",
        query: str = "",
        name_contains: str = "",
        extensions: list[str] | str | None = None,
        scope: str = "",
        sort: str = "",
        limit: int = 25,
        offset: int = 0,
        count_all: bool = False,
    ) -> str:
        """Search linked BisQue resources.

        Use scope="owner" for the user's own resources, sort="recent" for newest-first
        results, name_contains for filename fragments, and extensions such as ["png"]
        or ["nii", "nii.gz"] for file-type searches. Set count_all=True only when the
        user asks for an inventory count or total.
        """
        return _json_text(
            search_bisque_resources(
                settings,
                resource_type=resource_type,
                tag_query=tag_query,
                tag_order=tag_order,
                query=query,
                name_contains=name_contains,
                extensions=extensions,
                scope=scope,
                sort=sort,
                limit=limit,
                offset=offset,
                count_all=count_all,
                context=runtime.context,
            )
        )

    @tool
    def bisque_download_resource(
        runtime: ToolRuntime[AgentRunContext],
        resource_uri: str = "",
        resource_uris: list[str] | str | None = None,
    ) -> str:
        """Download BisQue resources into the local upload store and return file_ids for analysis."""
        resources = _resource_list(resource_uri, resource_uris)
        if not resources:
            resources = list(runtime.context.selected_resource_uris)
        return _json_text(download_bisque_resources(settings, resources=resources, context=runtime.context))

    @tool
    def bisque_upload_files(
        runtime: ToolRuntime[AgentRunContext],
        file_ids: list[str] | str | None = None,
        artifact_ids: list[str] | str | None = None,
    ) -> str:
        """Upload local V2 upload file_ids or durable artifact_ids to the linked BisQue account."""
        selected = _string_list(file_ids)
        if not selected:
            selected = list(runtime.context.selected_file_ids)
        selected_artifacts = _string_list(artifact_ids)
        return _json_text(
            upload_bisque_outputs(
                settings,
                file_ids=selected,
                artifact_ids=selected_artifacts,
                context=runtime.context,
            )
        )

    @tool
    def bisque_upload_workspace_files(
        runtime: ToolRuntime[AgentRunContext],
        paths: list[str] | str | None = None,
    ) -> str:
        """Upload generated /workspace or /outputs files to the linked BisQue account."""
        return _json_text(upload_bisque_workspace_files(settings, paths=paths, context=runtime.context))

    return [
        bisque_search_resources,
        bisque_download_resource,
        bisque_upload_files,
        bisque_upload_workspace_files,
    ]


def _control_headers(
    context: AgentRunContext | None,
    *,
    include_principal: bool = False,
) -> dict[str, str]:
    headers: dict[str, str] = {}
    if context is None:
        return headers
    run_id = str(context.run_id or "").strip()
    if run_id:
        headers["X-Ultra-Run-Id"] = run_id
    session_id = str(context.run_metadata.get("bisque_session_id") or "").strip()
    if session_id:
        headers["X-Ultra-Bisque-Session-Id"] = session_id
    if include_principal:
        user_id = str(context.user_id or "").strip()
        org_id = str(context.org_id or "").strip()
        if user_id:
            headers["X-Ultra-User-Id"] = user_id
        if org_id:
            headers["X-Ultra-Org-Id"] = org_id
    return headers


def _control_failure_result(resource: str, exc: Exception) -> dict[str, Any]:
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    payload: dict[str, Any] = {}
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            data = response.json()
            if isinstance(data, dict):
                payload = dict(data)
        except Exception:
            payload = {}
    error = str(payload.get("error") or payload.get("message") or exc).strip()
    if not error:
        error = exc.__class__.__name__
    result: dict[str, Any] = {
        "resource_uri": resource,
        "ok": False,
        "status": "failed",
        "error": error,
    }
    if isinstance(status_code, int):
        result["status_code"] = status_code
    return result


def _resolve_run_file_paths(paths: list[str] | str | None, context: AgentRunContext) -> list[Path]:
    raw_paths = _string_list(paths)
    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw in raw_paths:
        path = _resolve_run_file_path(raw, context)
        if path in seen:
            continue
        seen.add(path)
        resolved.append(path)
    return resolved


def _resolve_run_file_path(raw_path: str, context: AgentRunContext) -> Path:
    raw = str(raw_path or "").strip()
    if not raw:
        raise ValueError("BisQue workspace upload path is required")
    workspace_root = Path(context.workspace_root).expanduser().resolve()
    artifact_root = Path(context.artifact_root).expanduser().resolve()
    candidate: Path
    if raw.startswith("/workspace/"):
        candidate = workspace_root / raw.removeprefix("/workspace/")
    elif raw == "/workspace":
        candidate = workspace_root
    elif raw.startswith("/outputs/"):
        candidate = artifact_root / raw.removeprefix("/outputs/")
    elif raw == "/outputs":
        candidate = artifact_root
    else:
        path = Path(raw).expanduser()
        if path.is_absolute():
            candidate = path
        else:
            candidate = workspace_root / path
            if not candidate.exists():
                artifact_candidate = artifact_root / path
                if artifact_candidate.exists():
                    candidate = artifact_candidate
    candidate = candidate.resolve()
    if not (_path_is_under(candidate, workspace_root) or _path_is_under(candidate, artifact_root)):
        raise ValueError("BisQue workspace upload path is outside this run workspace")
    if not candidate.exists() or not candidate.is_file():
        raise ValueError(f"BisQue workspace upload path does not exist: {raw}")
    return candidate


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resource_list(resource_uri: str, resource_uris: list[str] | str | None) -> list[str]:
    resources = _string_list(resource_uris)
    if str(resource_uri or "").strip():
        resources.insert(0, str(resource_uri).strip())
    seen: set[str] = set()
    unique: list[str] = []
    for resource in resources:
        if resource not in seen:
            seen.add(resource)
            unique.append(resource)
    return unique


def _string_list(value: list[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
        return [text]
    return [str(item).strip() for item in value if str(item).strip()]


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
