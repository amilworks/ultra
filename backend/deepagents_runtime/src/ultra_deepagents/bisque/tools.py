from __future__ import annotations

import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.tools import ToolRuntime, tool

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.context_tools import artifact_source_path


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
            headers=_control_headers(context, settings=settings, include_principal=True),
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
            headers=_control_headers(context, settings=settings, include_principal=True),
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
    metadata_filters: list[dict[str, Any]] | str | None = None,
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
    cleaned_metadata_filters = _metadata_filter_list(metadata_filters)
    if cleaned_metadata_filters:
        payload["metadata_filters"] = cleaned_metadata_filters
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


def list_bisque_module_runs(
    settings: RuntimeSettings,
    *,
    scope: str = "owner",
    sort: str = "recent",
    limit: int = 25,
    status: str = "",
    count_all: bool = False,
    context: AgentRunContext | None = None,
) -> dict[str, Any]:
    """List recent BisQue module-execution (MEX) runs with status + module + time."""
    response = search_bisque_resources(
        settings,
        resource_type="mex",
        scope=scope or "owner",
        sort=sort or "recent",
        limit=limit,
        count_all=count_all,
        context=context,
    )
    results = response.get("results") if isinstance(response, dict) else None
    runs = [item for item in results if isinstance(item, dict)] if isinstance(results, list) else []
    wanted = str(status or "").strip().upper()
    if wanted:
        runs = [run for run in runs if str(run.get("status") or "").strip().upper() == wanted]
    summary = [
        {
            "mex_uri": str(run.get("resource_uri") or "").strip(),
            "mex_uniq": str(run.get("resource_uniq") or "").strip(),
            "module_name": str(run.get("module_name") or run.get("name") or "").strip(),
            "status": str(run.get("status") or "").strip(),
            "created_at": str(run.get("created_at") or "").strip(),
            "client_view_url": str(run.get("client_view_url") or "").strip(),
        }
        for run in runs
    ]
    return {
        "ok": True,
        "count": response.get("count", len(summary)) if isinstance(response, dict) else len(summary),
        "runs": summary,
    }


def get_bisque_module_run(
    settings: RuntimeSettings,
    *,
    mex_uri: str,
    context: AgentRunContext | None = None,
) -> dict[str, Any]:
    """Fetch one module run's structured inputs/outputs (download-ready result resources)."""
    cleaned = str(mex_uri or "").strip()
    if not cleaned:
        return {"ok": False, "error": "mex_uri_required"}
    payload = {"mex_uri": cleaned} if "/" in cleaned else {"mex_uniq": cleaned}
    response = control_post_json(settings, "/v2/bisque/module-run", payload, context=context)
    if isinstance(response, dict) and "ok" not in response:
        response["ok"] = bool(str(response.get("resource_uri") or "").strip())
    return response


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
    targets, missing = _resolve_bisque_upload_targets(paths, context)
    if not targets:
        return {
            "ok": False,
            "error": "no_resolvable_paths",
            "missing_paths": missing,
            "hint": (
                "Pass a generated output path (e.g. /workspace/outputs/figure.png) or, for a "
                "figure produced in an earlier turn, its durable artifact path or artifact_id "
                "from artifact_manifest."
            ),
            "pushed": [],
        }

    file_targets = [target for target in targets if target.kind == "file"]
    artifact_targets = [target for target in targets if target.kind == "artifact"]

    result: dict[str, Any] = {"pushed": [], "missing_paths": missing}
    errors: list[str] = []

    # Current-run files: stage into the V2 upload store, then link to BisQue.
    if file_targets:
        handles = []
        files: list[tuple[str, tuple[str, Any, str]]] = []
        try:
            for target in file_targets:
                assert target.path is not None
                handle = target.path.open("rb")
                handles.append(handle)
                mime_type = mimetypes.guess_type(target.path.name)[0] or "application/octet-stream"
                files.append(("files", (target.path.name, handle, mime_type)))
            upload_response = control_post_files(settings, "/v2/uploads", files, context=context)
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
        result["control_upload"] = upload_response
        result["uploaded_file_ids"] = file_ids
        if file_ids:
            bisque_upload = upload_bisque_outputs(settings, file_ids=file_ids, artifact_ids=[], context=context)
            result["bisque_upload_files"] = bisque_upload
            result["pushed"].extend(_collect_pushed(bisque_upload, "workspace_file"))
        else:
            errors.append("control_upload_returned_no_file_ids")

    # Prior durable artifacts: push straight from the artifact store by id (no
    # re-upload, ownership-checked on the control plane).
    if artifact_targets:
        artifact_ids = [target.artifact_id for target in artifact_targets]
        result["artifact_ids"] = artifact_ids
        bisque_upload = upload_bisque_outputs(settings, file_ids=[], artifact_ids=artifact_ids, context=context)
        result["bisque_upload_artifacts"] = bisque_upload
        result["pushed"].extend(_collect_pushed(bisque_upload, "durable_artifact"))

    if missing:
        errors.append("unresolved_paths")
    result["ok"] = bool(result["pushed"]) and not errors
    if errors:
        result["error"] = errors[0]
        result["errors"] = errors
    return result


def _collect_pushed(bisque_upload: dict[str, Any], via: str) -> list[dict[str, Any]]:
    uploads = bisque_upload.get("uploads") if isinstance(bisque_upload, dict) else None
    pushed: list[dict[str, Any]] = []
    if not isinstance(uploads, list):
        return pushed
    for item in uploads:
        if not isinstance(item, dict):
            continue
        resource_uri = str(item.get("resource_uri") or "").strip()
        if not resource_uri:
            continue
        pushed.append(
            {
                "via": via,
                # client_view_url is the canonical BisQue link that loads the
                # resource in the browser — report THIS to the user, not the bare
                # data_service resource_uri (which is the API URL).
                "client_view_url": str(item.get("client_view_url") or "").strip(),
                "resource_uri": resource_uri,
                "name": str(item.get("name") or "").strip(),
                "resource_uniq": str(item.get("resource_uniq") or "").strip(),
                "file_id": str(item.get("file_id") or "").strip(),
                "artifact_id": str(item.get("artifact_id") or "").strip(),
            }
        )
    return pushed


def upload_bisque_files(
    settings: RuntimeSettings,
    *,
    file_ids: list[str],
    context: AgentRunContext | None = None,
) -> dict[str, Any]:
    return upload_bisque_outputs(settings, file_ids=file_ids, artifact_ids=[], context=context)


def create_bisque_dataset(
    settings: RuntimeSettings,
    *,
    name: str,
    resource_uris: list[str] | str | None = None,
    context: AgentRunContext | None = None,
) -> dict[str, Any]:
    cleaned_name = str(name or "").strip()
    if not cleaned_name:
        return {"ok": False, "error": "dataset_name_required"}
    members = _string_list(resource_uris)
    if not members:
        return {"ok": False, "error": "no_member_resource_uris"}
    response = control_post_json(
        settings,
        "/v2/bisque/datasets",
        {"name": cleaned_name, "resource_uris": members},
        context=context,
    )
    if "ok" not in response:
        response["ok"] = bool(str(response.get("resource_uri") or "").strip())
    return response


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
        metadata_filters: list[dict[str, Any]] | str | None = None,
        scope: str = "",
        sort: str = "",
        limit: int = 25,
        offset: int = 0,
        count_all: bool = False,
    ) -> str:
        """Search or count linked BisQue resources (images, datasets, files, tables).

        Set resource_type to "image", "dataset", "file", or "table" — e.g. answer
        "do I have any datasets on BisQue?" with resource_type="dataset". Use
        scope="owner" for the user's own resources, sort="recent" for newest-first
        results, name_contains for filename fragments, and extensions such as ["png"]
        or ["nii", "nii.gz"] for file-type searches. Set count_all=True only when the
        user asks for an inventory count or total.

        For NUMERIC metadata tag comparisons (age, slice count, dose, year, etc.) use
        metadata_filters, NOT tag_query — BisQue compares numeric tags lexically, so
        tag_query='age:>50' wrongly returns age 7 and drops age 100. metadata_filters
        is a list of {"tag","op","value"} where op is one of eq, ne, gt, gte, lt, lte,
        contains; relational ops require a numeric value and are evaluated correctly.
        Example: "CT scans above age 50" ->
        tag_query='modality:CT', metadata_filters=[{"tag":"age","op":"gt","value":"50"}].
        Combine string-equality filters (e.g. modality:CT) in tag_query for server-side
        narrowing with metadata_filters for the numeric part. For string filename
        filtering, tag_query supports wildcards and boolean keywords on tag values, e.g.
        tag_query='filename:*.nii or filename:*.nii.gz'.
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
                metadata_filters=metadata_filters,
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
        """Upload local V2 upload file_ids or durable artifact_ids to the linked BisQue account.

        Each returned upload includes client_view_url — the canonical BisQue link that
        opens the resource in the web viewer. Report that URL to the user, not the bare
        resource_uri (the data_service API URL).
        """
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
        """Push generated output files to the linked BisQue account.

        Accepts current-run output paths (e.g. /workspace/outputs/figure.png or
        /outputs/figure.png) AND figures produced in an earlier turn: pass the
        durable artifact path (e.g. outputs/ct_scan_visualization.png), its basename,
        or its artifact_id from artifact_manifest. Files from a prior run live only as
        durable artifacts — this tool resolves them automatically and pushes them by
        artifact_id, so "push these resulting images to BisQue" works across turns.
        """
        return _json_text(upload_bisque_workspace_files(settings, paths=paths, context=runtime.context))

    @tool
    def bisque_create_dataset(
        runtime: ToolRuntime[AgentRunContext],
        name: str,
        resource_uris: list[str] | str | None = None,
    ) -> str:
        """Create a BisQue dataset that groups already-uploaded BisQue resources.

        Pass the BisQue resource_uri values returned by bisque_upload_files,
        bisque_upload_workspace_files, or bisque_search_resources. Use this after
        uploading multiple related outputs so they appear as one dataset in BisQue.
        """
        return _json_text(
            create_bisque_dataset(
                settings,
                name=name,
                resource_uris=resource_uris,
                context=runtime.context,
            )
        )

    @tool
    def bisque_module_runs(
        runtime: ToolRuntime[AgentRunContext],
        mex_uri: str = "",
        scope: str = "owner",
        sort: str = "recent",
        limit: int = 25,
        status: str = "",
    ) -> str:
        """List BisQue module-execution (MEX) runs, or inspect one run's results.

        Answer "what modules have I run recently on BisQue?" by calling with no
        mex_uri: returns each run's module_name, status (FINISHED/RUNNING/FAILED),
        created_at, and mex_uri. Pass status="FINISHED" to filter.

        To pull a result from a run (e.g. a segmentation mask), pass mex_uri (the run's
        resource_uri or resource_uniq): returns the run's structured inputs and outputs.
        Each resource-typed output carries resource_uri (feed to bisque_download_resource
        to materialize it into Ultra for analysis) and client_view_url (report to the user
        to view it on BisQue).
        """
        if str(mex_uri or "").strip():
            return _json_text(
                get_bisque_module_run(settings, mex_uri=mex_uri, context=runtime.context)
            )
        return _json_text(
            list_bisque_module_runs(
                settings,
                scope=scope,
                sort=sort,
                limit=limit,
                status=status,
                context=runtime.context,
            )
        )

    return [
        bisque_search_resources,
        bisque_download_resource,
        bisque_upload_files,
        bisque_upload_workspace_files,
        bisque_create_dataset,
        bisque_module_runs,
    ]


def _control_headers(
    context: AgentRunContext | None,
    *,
    settings: RuntimeSettings | None = None,
    include_principal: bool = False,
) -> dict[str, str]:
    headers: dict[str, str] = {}
    worker_token = str(getattr(settings, "control_worker_token", "") or "").strip()
    if worker_token:
        headers["X-Ultra-Worker-Token"] = worker_token
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


@dataclass(frozen=True)
class _BisqueUploadTarget:
    """A resolved upload source: either a live file on disk or a durable artifact_id."""

    raw: str
    kind: str  # "file" or "artifact"
    path: Path | None = None
    artifact_id: str = ""
    name: str = ""


def _resolve_bisque_upload_targets(
    paths: list[str] | str | None,
    context: AgentRunContext,
) -> tuple[list[_BisqueUploadTarget], list[str]]:
    """Resolve requested paths/names to upload targets.

    Each request resolves to either a current-run file on disk (uploaded via the
    multipart path) or a prior durable artifact (pushed by artifact_id). A figure
    referenced in a later turn — e.g. ``/outputs/ct_scan_visualization.png`` from an
    earlier run — is a prior durable artifact, not a current-run workspace file.
    """
    targets: list[_BisqueUploadTarget] = []
    missing: list[str] = []
    seen_files: set[str] = set()
    seen_artifacts: set[str] = set()
    for raw in _string_list(paths):
        target = _resolve_bisque_upload_target(raw, context)
        if target is None:
            missing.append(raw)
            continue
        if target.kind == "artifact":
            if target.artifact_id in seen_artifacts:
                continue
            seen_artifacts.add(target.artifact_id)
        else:
            key = str(target.path)
            if key in seen_files:
                continue
            seen_files.add(key)
        targets.append(target)
    return targets, missing


def _resolve_bisque_upload_target(
    raw_path: str,
    context: AgentRunContext,
) -> _BisqueUploadTarget | None:
    raw = str(raw_path or "").strip()
    if not raw:
        return None
    # Prefer a fresh file the current run actually produced; only then fall back to
    # prior durable artifacts surfaced through resource_descriptors.
    host_file = _resolve_current_run_host_file(raw, context)
    if host_file is not None:
        return _BisqueUploadTarget(raw=raw, kind="file", path=host_file, name=host_file.name)
    descriptor = _match_prior_artifact_descriptor(context, raw)
    if descriptor is not None:
        artifact_id = str(descriptor.get("artifact_id") or "").strip()
        name = _artifact_descriptor_name(descriptor)
        if artifact_id:
            return _BisqueUploadTarget(raw=raw, kind="artifact", artifact_id=artifact_id, name=name)
        source = artifact_source_path(context, descriptor)
        if source is not None and source.exists() and source.is_file():
            return _BisqueUploadTarget(raw=raw, kind="file", path=source, name=source.name)
    return None


def _resolve_current_run_host_file(raw: str, context: AgentRunContext) -> Path | None:
    workspace_root = Path(context.workspace_root).expanduser().resolve()
    artifact_root = Path(context.artifact_root).expanduser().resolve()
    if raw in ("/workspace", "/outputs"):
        return None
    candidates: list[Path] = []
    if raw.startswith("/workspace/"):
        rel = raw.removeprefix("/workspace/")
        candidates += [workspace_root / rel, workspace_root / "outputs" / rel]
    elif raw.startswith("/outputs/"):
        rel = raw.removeprefix("/outputs/")
        # /outputs maps to artifact_root, but figures written to /workspace/outputs
        # are harvested under artifact_root/outputs — check both, and the live
        # workspace copy that exists mid-run.
        candidates += [
            artifact_root / rel,
            artifact_root / "outputs" / rel,
            workspace_root / "outputs" / rel,
            workspace_root / rel,
        ]
    else:
        path = Path(raw).expanduser()
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates += [
                workspace_root / raw,
                workspace_root / "outputs" / raw,
                artifact_root / raw,
                artifact_root / "outputs" / raw,
            ]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not (_path_is_under(resolved, workspace_root) or _path_is_under(resolved, artifact_root)):
            continue
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def _match_prior_artifact_descriptor(
    context: AgentRunContext,
    raw: str,
) -> dict[str, Any] | None:
    forms = _artifact_match_forms(raw)
    basename = Path(raw).name.strip()
    basename_matches: list[dict[str, Any]] = []
    for descriptor in context.resource_descriptors:
        if str(descriptor.get("type") or "artifact") != "artifact":
            continue
        descriptor_keys = {
            str(descriptor.get("artifact_id") or "").strip(),
            str(descriptor.get("path") or "").strip(),
            str(descriptor.get("relative_path") or "").strip(),
            str(descriptor.get("source_path") or "").strip(),
            str(descriptor.get("storage_uri") or "").strip().removeprefix("file://"),
        }
        descriptor_keys.discard("")
        if forms & descriptor_keys:
            return dict(descriptor)
        if basename and basename in {Path(key).name for key in descriptor_keys}:
            basename_matches.append(dict(descriptor))
    if basename_matches:
        # Deterministic: prefer the most recently created artifact when several
        # prior figures share a basename.
        basename_matches.sort(
            key=lambda descriptor: str(descriptor.get("created_at") or descriptor.get("run_id") or ""),
            reverse=True,
        )
        return basename_matches[0]
    return None


def _artifact_match_forms(raw: str) -> set[str]:
    raw = raw.strip()
    forms = {raw, raw.lstrip("/")}
    if raw.startswith("/workspace/outputs/"):
        forms.add("outputs/" + raw.removeprefix("/workspace/outputs/"))
    if raw.startswith("/workspace/"):
        forms.add(raw.removeprefix("/workspace/"))
    if raw.startswith("/outputs/"):
        forms.add(raw.removeprefix("/outputs/"))
        forms.add("outputs/" + raw.removeprefix("/outputs/"))
    forms.discard("")
    return forms


def _artifact_descriptor_name(descriptor: dict[str, Any]) -> str:
    for key in ("path", "relative_path", "source_path", "title"):
        value = str(descriptor.get(key) or "").strip()
        if value:
            name = Path(value).name
            if name:
                return name
    return str(descriptor.get("artifact_id") or "artifact").strip() or "artifact"


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


def _metadata_filter_list(
    value: list[dict[str, Any]] | dict[str, Any] | str | None,
) -> list[dict[str, str]]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    filters: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag") or item.get("name") or "").strip()
        if not tag:
            continue
        op = str(item.get("op") or item.get("operator") or "eq").strip().lower() or "eq"
        filter_value = item.get("value")
        filters.append({"tag": tag, "op": op, "value": "" if filter_value is None else str(filter_value).strip()})
    return filters


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
