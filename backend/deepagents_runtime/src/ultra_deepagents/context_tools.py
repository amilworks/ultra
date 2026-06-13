from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from langchain.tools import ToolRuntime, tool

from ultra_deepagents.context import AgentRunContext


def build_prior_artifact_manifest(context: AgentRunContext) -> dict[str, Any]:
    artifacts = []
    for descriptor in context.resource_descriptors:
        if str(descriptor.get("type") or "artifact") != "artifact":
            continue
        entry = _public_descriptor(descriptor)
        entry["access"] = (
            "remote_storage_uri"
            if str(entry.get("remote_storage_uri") or "").strip()
            else "stage_artifact_for_analysis"
        )
        artifacts.append(entry)
    return {
        "run_id": context.run_id,
        "thread_id": context.thread_id,
        "workspace_root": "/workspace",
        "artifact_root": "/outputs",
        "prior_artifacts": artifacts,
    }


def artifact_manifest_text(context: AgentRunContext) -> str:
    """Return durable prior-artifact context as compact JSON for the model."""
    return json.dumps(build_prior_artifact_manifest(context), indent=2, sort_keys=True)


def stage_uploaded_files(
    context: AgentRunContext,
    *,
    upload_roots: Iterable[str | Path] = (),
    file_ids: Iterable[str] | str | None = None,
) -> dict[str, Any]:
    requested = _unique_upload_file_ids(file_ids if file_ids is not None else context.selected_file_ids)
    if not requested:
        return {
            "ok": False,
            "error": "no_file_ids",
            "staged_files": [],
            "missing_file_ids": [],
        }

    roots = _resolved_upload_roots(upload_roots)
    stage_root = Path(context.workspace_root).expanduser().resolve() / "staged_uploads"
    staged_files: list[dict[str, Any]] = []
    missing: list[str] = []
    for file_id in requested:
        source = _find_uploaded_file(file_id, roots)
        if source is None:
            missing.append(file_id)
            continue
        target_name = _uploaded_original_name(source.name, file_id)
        target = stage_root / _safe_path_token(file_id) / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        staged_files.append(
            {
                "file_id": file_id,
                "source_path": str(source),
                "staged_path": str(target),
                "sandbox_path": f"/workspace/staged_uploads/{_safe_path_token(file_id)}/{target_name}",
            }
        )

    return {
        "ok": len(staged_files) > 0 and not missing,
        "staged_files": staged_files,
        "missing_file_ids": missing,
    }


def stage_artifact(context: AgentRunContext, *, artifact_id: str = "", path: str = "") -> dict[str, Any]:
    descriptor = _find_artifact_descriptor(context, artifact_id=artifact_id, path=path)
    if descriptor is None:
        return {
            "ok": False,
            "error": "artifact_not_found",
            "artifact_id": artifact_id,
            "path": path,
        }

    source = _artifact_source_path(context, descriptor)
    if source is None:
        return {
            "ok": False,
            "error": "artifact_source_not_resolved",
            "artifact": _public_descriptor(descriptor),
        }
    if not source.exists() or not source.is_file():
        return {
            "ok": False,
            "error": "artifact_file_missing",
            "source_path": str(source),
            "artifact": _public_descriptor(descriptor),
        }

    stage_root = Path(context.workspace_root).expanduser().resolve() / "staged_artifacts"
    run_id = _safe_path_token(str(descriptor.get("run_id") or "prior_run"))
    target_name = _safe_path_token(source.name) or "artifact"
    target = stage_root / run_id / target_name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {
        "ok": True,
        "artifact": _public_descriptor(descriptor),
        "source_path": str(source),
        "staged_path": str(target),
        "sandbox_path": f"/workspace/staged_artifacts/{run_id}/{target_name}",
    }


def stage_artifact_for_analysis_text(
    context: AgentRunContext,
    *,
    artifact_id: str = "",
    path: str = "",
) -> str:
    """Return model-visible artifact staging output with sandbox paths only."""
    return json.dumps(
        _public_stage_result(stage_artifact(context, artifact_id=artifact_id, path=path)),
        indent=2,
        sort_keys=True,
    )


def stage_uploaded_files_for_analysis_text(
    context: AgentRunContext,
    *,
    upload_roots: Iterable[str | Path] = (),
    file_ids: Iterable[str] | str | None = None,
) -> str:
    """Return model-visible upload staging output with sandbox paths only."""
    return json.dumps(
        _public_stage_result(
            stage_uploaded_files(context, upload_roots=upload_roots, file_ids=file_ids)
        ),
        indent=2,
        sort_keys=True,
    )


@tool
def artifact_manifest(runtime: ToolRuntime[AgentRunContext]) -> str:
    """List prior durable artifacts available to this run with IDs, paths, types, and checksums."""
    return artifact_manifest_text(runtime.context)


@tool
def stage_artifact_for_analysis(
    runtime: ToolRuntime[AgentRunContext],
    artifact_id: str = "",
    path: str = "",
) -> str:
    """Copy a prior artifact into /workspace/staged_artifacts so execute() code can read or modify it."""
    return stage_artifact_for_analysis_text(runtime.context, artifact_id=artifact_id, path=path)


def build_context_tools(upload_roots: Iterable[str | Path] = ()) -> list[Any]:
    resolved_upload_roots = tuple(upload_roots)

    @tool
    def stage_uploaded_files_for_analysis(
        runtime: ToolRuntime[AgentRunContext],
        file_ids: list[str] | str | None = None,
    ) -> str:
        """Copy selected uploaded files into /workspace/staged_uploads so execute() code can analyze them."""
        return stage_uploaded_files_for_analysis_text(
            runtime.context,
            upload_roots=resolved_upload_roots,
            file_ids=file_ids,
        )

    return [artifact_manifest, stage_artifact_for_analysis, stage_uploaded_files_for_analysis]


def build_tool_capability_manifest(
    registered_tools: Iterable[Any],
    *,
    available_subagents: Iterable[dict[str, Any]] = (),
    available_async_subagents: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Build the model-visible manifest for the active Deep Agents tool surface."""
    registered_tool_names = sorted(
        {
            str(getattr(tool_obj, "name", "") or "").strip()
            for tool_obj in registered_tools
            if str(getattr(tool_obj, "name", "") or "").strip()
        }
    )
    subagent_descriptors = _public_subagent_descriptors(available_subagents)
    async_subagent_descriptors = _public_subagent_descriptors(available_async_subagents)
    builtin_tools = [
        {
            "name": "execute",
            "category": "sandbox",
            "purpose": "Run shell or Python commands in the per-run /workspace sandbox.",
        },
        {
            "name": "write_file",
            "category": "filesystem",
            "purpose": "Write source, reports, and other working files into the active backend.",
        },
        {
            "name": "read_file",
            "category": "filesystem",
            "purpose": "Read text files from /workspace, /outputs, /memories, or staged artifacts.",
        },
        {
            "name": "edit_file",
            "category": "filesystem",
            "purpose": "Patch existing text files without rewriting them from scratch.",
        },
        {
            "name": "ls",
            "category": "filesystem",
            "purpose": "Inspect directory contents under the active backend routes.",
        },
        {
            "name": "glob",
            "category": "filesystem",
            "purpose": "Find files by pattern under the active backend routes.",
        },
        {
            "name": "grep",
            "category": "filesystem",
            "purpose": "Search text files under the active backend routes.",
        },
        {
            "name": "write_todos",
            "category": "planning",
            "purpose": "Track multi-step task progress during autonomous work.",
        },
    ]
    if subagent_descriptors:
        builtin_tools.append(
            {
                "name": "task",
                "category": "delegation",
                "purpose": (
                    "Launch one of the available scoped subagents for isolated, "
                    "multi-step work and reconcile its final report."
                ),
            }
        )
    if async_subagent_descriptors:
        builtin_tools.extend(
            [
                {
                    "name": "start_async_task",
                    "category": "background_delegation",
                    "purpose": (
                        "Launch one configured remote async subagent as a background task "
                        "and return its task ID immediately."
                    ),
                },
                {
                    "name": "check_async_task",
                    "category": "background_delegation",
                    "purpose": "Check the latest status or result for a configured async subagent task.",
                },
                {
                    "name": "update_async_task",
                    "category": "background_delegation",
                    "purpose": "Send follow-up instructions to a running async subagent task.",
                },
                {
                    "name": "cancel_async_task",
                    "category": "background_delegation",
                    "purpose": "Cancel a running async subagent task.",
                },
                {
                    "name": "list_async_tasks",
                    "category": "background_delegation",
                    "purpose": "List tracked async subagent tasks with their latest known statuses.",
                },
            ]
        )
    return {
        "deepagents_builtin_tools": builtin_tools,
        "registered_tools": registered_tool_names,
        "available_subagents": subagent_descriptors,
        "available_async_subagents": async_subagent_descriptors,
        "selected_tool_names": (
            "The current run may include selected_tool_names in runtime context. "
            "Use domain tools only when they are relevant to the user goal."
        ),
        "storage": {
            "workspace": "/workspace",
            "outputs": "/outputs",
            "memories": "/memories",
            "staged_uploads": "/workspace/staged_uploads",
            "staged_artifacts": "/workspace/staged_artifacts",
        },
    }


def _public_subagent_descriptors(
    subagents: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    for subagent in subagents:
        name = str(subagent.get("name") or "").strip()
        description = str(subagent.get("description") or "").strip()
        if not name or not description:
            continue
        descriptor = {"name": name, "description": description}
        tool_names = sorted(
            {
                str(getattr(tool_obj, "name", "") or "").strip()
                for tool_obj in subagent.get("tools") or []
                if str(getattr(tool_obj, "name", "") or "").strip()
            }
        )
        if tool_names:
            descriptor["tool_names"] = tool_names
        response_format = _public_response_format_descriptor(
            subagent.get("response_format")
        )
        if response_format:
            descriptor["response_format"] = response_format
        descriptors.append(descriptor)
    return descriptors


def _public_response_format_descriptor(response_format: Any) -> dict[str, Any]:
    if not isinstance(response_format, dict):
        return {}
    schema_type = str(response_format.get("type") or "").strip()
    raw_required = response_format.get("required")
    required = [
        str(item).strip()
        for item in (raw_required if isinstance(raw_required, list | tuple) else [])
        if str(item).strip()
    ]
    properties = response_format.get("properties")
    property_names = (
        sorted(str(key).strip() for key in properties if str(key).strip())
        if isinstance(properties, dict)
        else []
    )
    descriptor: dict[str, Any] = {}
    if schema_type:
        descriptor["type"] = schema_type
    if required:
        descriptor["required"] = required
    if property_names:
        descriptor["properties"] = property_names
    return descriptor


def build_tool_capability_manifest_tool(
    registered_tools: Iterable[Any],
    *,
    available_subagents: Iterable[dict[str, Any]] = (),
    available_async_subagents: Iterable[dict[str, Any]] = (),
) -> Any:
    """Expose a compact, model-visible manifest of the active tool surface."""
    manifest = build_tool_capability_manifest(
        registered_tools,
        available_subagents=available_subagents,
        available_async_subagents=available_async_subagents,
    )

    @tool
    def tool_capability_manifest() -> str:
        """Return the active Deep Agents tool categories, storage paths, and registered app tools."""
        return json.dumps(manifest, indent=2, sort_keys=True)

    return tool_capability_manifest


def _find_artifact_descriptor(
    context: AgentRunContext,
    *,
    artifact_id: str = "",
    path: str = "",
) -> dict[str, Any] | None:
    target_artifact_id = artifact_id.strip()
    target_path = path.strip()
    for descriptor in context.resource_descriptors:
        if str(descriptor.get("type") or "artifact") != "artifact":
            continue
        if target_artifact_id and str(descriptor.get("artifact_id") or "").strip() == target_artifact_id:
            return dict(descriptor)
        if target_path:
            candidate_paths = {
                str(descriptor.get("path") or "").strip(),
                str(descriptor.get("relative_path") or "").strip(),
                str(descriptor.get("source_path") or "").strip(),
            }
            if target_path in candidate_paths:
                return dict(descriptor)
    return None


def artifact_source_path(context: AgentRunContext, descriptor: dict[str, Any]) -> Path | None:
    """Resolve a prior-artifact descriptor to its durable host path under the artifact store.

    Public entry point so other tool modules (e.g. BisQue upload) reuse the exact
    same resolution rules instead of re-deriving artifact-store paths.
    """
    return _artifact_source_path(context, descriptor)


def _artifact_source_path(context: AgentRunContext, descriptor: dict[str, Any]) -> Path | None:
    artifact_store_root = Path(context.artifact_root).expanduser().resolve().parent
    candidates = [
        _file_uri_path(str(descriptor.get("storage_uri") or "")),
        str(descriptor.get("source_path") or ""),
    ]
    run_id = str(descriptor.get("run_id") or "").strip()
    relative_path = str(descriptor.get("path") or descriptor.get("relative_path") or "").strip()
    if run_id and relative_path and not relative_path.startswith("/"):
        candidates.append(str(artifact_store_root / run_id / relative_path))
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if _is_under(path, artifact_store_root):
            return path
    return None


def _file_uri_path(uri: str) -> str:
    if not uri.startswith("file://"):
        return ""
    return uri.removeprefix("file://")


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _public_descriptor(descriptor: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "type",
        "artifact_id",
        "output_id",
        "run_id",
        "kind",
        "title",
        "path",
        "relative_path",
        "mime_type",
        "size_bytes",
        "sha256",
        "tool_name",
        "deepagents_path",
        "remote_storage_uri",
    }
    return {key: value for key, value in descriptor.items() if key in allowed}


def _public_stage_result(result: dict[str, Any]) -> dict[str, Any]:
    public: dict[str, Any] = {}
    for key, value in result.items():
        if key in {"source_path", "storage_uri"}:
            continue
        if key == "artifact" and isinstance(value, dict):
            public[key] = _public_descriptor(value)
            continue
        if key == "staged_files" and isinstance(value, list):
            public[key] = [
                _public_staged_file(item)
                for item in value
                if isinstance(item, dict)
            ]
            continue
        if key == "staged_path":
            public[key] = _sandbox_path_or_value(result, value)
            continue
        public[key] = value
    return public


def _public_staged_file(item: dict[str, Any]) -> dict[str, Any]:
    public: dict[str, Any] = {}
    for key, value in item.items():
        if key in {"source_path", "storage_uri"}:
            continue
        if key == "staged_path":
            public[key] = _sandbox_path_or_value(item, value)
            continue
        public[key] = value
    return public


def _sandbox_path_or_value(payload: dict[str, Any], fallback: Any) -> Any:
    sandbox_path = str(payload.get("sandbox_path") or "").strip()
    if sandbox_path.startswith("/workspace/") or sandbox_path.startswith("/outputs/"):
        return sandbox_path
    return fallback


def _safe_path_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")


def _unique_upload_file_ids(file_ids: Iterable[str] | str) -> list[str]:
    if isinstance(file_ids, str):
        file_ids = _parse_upload_file_ids(file_ids)
    seen: set[str] = set()
    unique: list[str] = []
    for raw in file_ids:
        file_id = str(raw or "").strip()
        if not file_id or file_id in seen or not _safe_upload_file_id(file_id):
            continue
        seen.add(file_id)
        unique.append(file_id)
    return unique


def _safe_upload_file_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_.:-]+", value))


def _parse_upload_file_ids(value: str) -> list[str]:
    stripped = value.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item is not None]
    if "," in stripped:
        return [token.strip() for token in stripped.split(",") if token.strip()]
    return [stripped]


def _resolved_upload_roots(upload_roots: Iterable[str | Path]) -> tuple[Path, ...]:
    roots: list[Path] = []
    repo_root = Path(__file__).resolve().parents[4]
    for raw in upload_roots:
        value = str(raw or "").strip()
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        resolved = path.resolve()
        if resolved not in roots:
            roots.append(resolved)
    return tuple(roots)


def _find_uploaded_file(file_id: str, roots: tuple[Path, ...]) -> Path | None:
    for root in roots:
        if not root.exists():
            continue
        for pattern in (file_id, f"{file_id}__*", f"{file_id}.*"):
            for candidate in sorted(root.glob(pattern)):
                path = candidate.expanduser().resolve()
                if path.is_file() and _is_under(path, root):
                    return path
    return None


def _uploaded_original_name(filename: str, file_id: str) -> str:
    prefix = f"{file_id}__"
    if filename.startswith(prefix):
        return _safe_path_token(filename[len(prefix):]) or "upload"
    return _safe_path_token(filename) or "upload"
