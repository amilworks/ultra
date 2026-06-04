from __future__ import annotations

import json
import hashlib
import re
import time
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext


class RareSpotEcologyInferenceArgs(BaseModel):
    local_paths: list[str] = Field(
        default_factory=list,
        description="Local image file or directory paths to process. Directories are scanned for image files.",
    )
    resource_uris: list[str] = Field(
        default_factory=list,
        description="BisQue resource URIs to stage and process.",
    )
    dataset_uris: list[str] = Field(
        default_factory=list,
        description="BisQue dataset URIs whose image resources should be staged and processed.",
    )
    wait_for_completion: bool = Field(
        default=True,
        description="When true, poll the queued RareSpot run and return artifact handles after completion.",
    )
    timeout_seconds: int = Field(
        default=3600,
        ge=1,
        le=86400,
        description="Maximum time to wait when wait_for_completion is true.",
    )
    confidence_threshold: float | None = Field(
        default=None,
        ge=0.001,
        le=1.0,
        description="Optional YOLO confidence threshold for this inference run.",
    )
    iou_threshold: float | None = Field(
        default=None,
        ge=0.001,
        le=1.0,
        description="Optional YOLO NMS IoU threshold for this inference run.",
    )
    tile_overlap: float | None = Field(
        default=None,
        ge=0.0,
        le=0.9,
        description="Optional tile overlap fraction for this inference run.",
    )
    image_size: int | None = Field(
        default=None,
        ge=128,
        le=4096,
        description="Optional YOLO image size/tile size for this inference run.",
    )
    spectral: bool | None = Field(
        default=None,
        description="Optional override for spectral instability scoring.",
    )
    thread_id: str | None = Field(
        default=None,
        description="Optional thread id. Defaults to the active Deep Agents runtime context thread.",
    )
    user_id: str | None = Field(
        default=None,
        description="Optional user id. Defaults to the active Deep Agents runtime context user.",
    )


TOOL_DESCRIPTION = """Submit a queued RareSpot ecology inference job for prairie dog and burrow detection.

Use this tool for production YOLOv5 inference, not ad hoc sandbox model serving. It routes to the
single-concurrency RareSpot worker queue, uses 512 px tiles with 25% overlap by default, enables
spectral instability scoring by default, and returns compact run/artifact handles. For follow-up analysis,
you may set optional confidence_threshold, iou_threshold, tile_overlap, image_size, and spectral
arguments to run controlled comparison passes. If explicit local_paths, resource_uris, or
dataset_uris are omitted, the tool reads the Deep Agents runtime context for selected files,
BisQue resources, and datasets when available.

The completed nested RareSpot run returns canonical key_artifacts with artifact IDs and
download URLs. Do not create stub or duplicate local CSV/JSON/Markdown outputs to mirror
those results; create new outputs only for true derived analysis across multiple runs.
"""


def build_rarespot_tools(settings: RuntimeSettings) -> list[StructuredTool]:
    if not settings.rarespot_tool_enabled:
        return []

    def rarespot_ecology_inference(
        local_paths: list[str] | None = None,
        resource_uris: list[str] | None = None,
        dataset_uris: list[str] | None = None,
        wait_for_completion: bool = True,
        timeout_seconds: int = 3600,
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
        tile_overlap: float | None = None,
        image_size: int | None = None,
        spectral: bool | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
    ) -> str:
        context = active_context()
        selected_file_ids: list[str] = []
        if context is not None:
            thread_id = thread_id or context.thread_id
            user_id = user_id or context.user_id
            selected_file_ids = list(context.selected_file_ids)
            if not resource_uris:
                resource_uris = list(context.selected_resource_uris)
            if not dataset_uris:
                dataset_uris = list(context.selected_dataset_uris)
        if not thread_id:
            raise ValueError("rarespot_ecology_inference requires a thread_id or active runtime context.")
        sanitized_local_paths = host_readable_local_paths(local_paths or [], settings=settings)
        config_overrides = compact_config_overrides(
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            tile_overlap=tile_overlap,
            image_size=image_size,
            spectral=spectral,
        )
        if should_skip_report_only_call(
            context=context,
            config_overrides=config_overrides,
            local_paths=sanitized_local_paths,
            resource_uris=list(resource_uris or []),
            dataset_uris=list(dataset_uris or []),
        ):
            return json.dumps(
                {
                    "status": "skipped",
                    "reason": "report_only_context",
                    "final_answer_hint": (
                        "This turn asks for RareSpot report/synthesis over existing runs, "
                        "not a new inference pass. Use artifact_manifest and "
                        "stage_artifact_for_analysis to inspect prior RareSpot artifacts. "
                        "Only call rarespot_ecology_inference again when the user explicitly "
                        "asks for a new threshold, new configuration, or new image inference."
                    ),
                }
            )
        idempotency_key = build_rarespot_idempotency_key(
            parent_run_id=context.run_id if context is not None else "",
            selected_file_ids=selected_file_ids,
            local_paths=sanitized_local_paths,
            resource_uris=list(resource_uris or []),
            dataset_uris=list(dataset_uris or []),
            config_overrides=config_overrides,
        )
        body = {
            "user_id": user_id or "researcher",
            "goal": "Run RareSpot ecology inference.",
            "messages": [{"role": "tool", "content": "RareSpot ecology inference requested."}],
            "file_ids": selected_file_ids,
            "resource_uris": list(resource_uris or []),
            "dataset_uris": list(dataset_uris or []),
            "selected_tool_names": ["rarespot_ecology_inference"],
            "workflow_hint": {"id": "rarespot_ecology"},
            "idempotency_key": idempotency_key,
            "metadata": {
                "idempotency_key": idempotency_key,
                "file_ids": selected_file_ids,
                "local_paths": sanitized_local_paths,
                "tool_name": "rarespot_ecology_inference",
                "tile_overlap": 0.25,
                "spectral": True,
                "rarespot_config": config_overrides,
            },
        }
        run = create_rarespot_run(settings, thread_id=thread_id, body=body)
        if not wait_for_completion:
            return json.dumps({"run_id": run["run_id"], "status": run.get("status"), "artifact_ids": []})
        return json.dumps(wait_for_rarespot_run(settings, run_id=run["run_id"], timeout_seconds=timeout_seconds))

    return [
        StructuredTool.from_function(
            func=rarespot_ecology_inference,
            name="rarespot_ecology_inference",
            description=TOOL_DESCRIPTION,
            args_schema=RareSpotEcologyInferenceArgs,
        )
    ]


def host_readable_local_paths(paths: list[str], *, settings: RuntimeSettings | None = None) -> list[str]:
    readable: list[str] = []
    workspace_roots = workspace_host_roots(settings)
    for raw_path in paths:
        path = str(raw_path or "").strip()
        if not path:
            continue
        if path.startswith("/workspace/"):
            continue
        resolved = Path(path).expanduser().resolve(strict=False)
        if any(_is_under(resolved, root) for root in workspace_roots):
            continue
        readable.append(path)
    return readable


def workspace_host_roots(settings: RuntimeSettings | None) -> list[Path]:
    if settings is None:
        return []
    roots: list[Path] = []
    raw_root = str(settings.workspace_root or "").strip()
    if raw_root:
        roots.append(Path(raw_root).expanduser().resolve(strict=False))
    return roots


def _is_under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def compact_config_overrides(**values: Any) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def should_skip_report_only_call(
    *,
    context: AgentRunContext | None,
    config_overrides: dict[str, Any],
    local_paths: list[str],
    resource_uris: list[str],
    dataset_uris: list[str],
) -> bool:
    if context is None:
        return False
    if config_overrides or local_paths or resource_uris or dataset_uris:
        return False
    return looks_report_only_rarespot_goal(context.goal)


def looks_report_only_rarespot_goal(goal: str) -> bool:
    text = " ".join(str(goal or "").lower().split())
    if "rarespot" not in text:
        return False
    if not re.search(r"\b(write|compile|combine|combined|synthes|summari[sz]e|report)\b", text):
        return False
    explicit_patterns = (
        r"\b(run|rerun|execute|perform|launch)\b.{0,80}\b(rarespot|inference|detect|detection|pass)\b",
        r"\b(stricter|permissive|new|another)\b.{0,80}\b(threshold|confidence|inference|pass)\b",
    )
    explicit_new_run = any(
        not _is_negated_rarespot_run_directive(text, match.start())
        for pattern in explicit_patterns
        for match in re.finditer(pattern, text)
    )
    return not explicit_new_run


def _is_negated_rarespot_run_directive(text: str, start: int) -> bool:
    prefix = text[max(0, start - 32):start]
    return bool(
        re.search(r"\b(?:do\s+not|don't|dont|never|without)\s+$", prefix)
        or re.search(r"\bno\s+(?:need\s+to\s+|new\s+)?$", prefix)
    )


def build_rarespot_idempotency_key(
    *,
    parent_run_id: str,
    selected_file_ids: list[str],
    local_paths: list[str],
    resource_uris: list[str],
    dataset_uris: list[str],
    config_overrides: dict[str, Any],
) -> str:
    payload = {
        "parent_run_id": parent_run_id or "manual",
        "selected_file_ids": sorted(str(item) for item in selected_file_ids),
        "local_paths": sorted(str(item) for item in local_paths),
        "resource_uris": sorted(str(item) for item in resource_uris),
        "dataset_uris": sorted(str(item) for item in dataset_uris),
        "config_overrides": config_overrides,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"rarespot-{payload['parent_run_id']}-{digest[:24]}"


def active_context() -> AgentRunContext | None:
    try:
        from langgraph.runtime import get_runtime

        runtime = get_runtime(AgentRunContext)
        return runtime.context
    except Exception:
        return None


def create_rarespot_run(settings: RuntimeSettings, *, thread_id: str, body: dict[str, Any]) -> dict[str, Any]:
    import httpx

    url = f"{settings.rarespot_control_base_url.rstrip('/')}/v2/threads/{thread_id}/runs"
    idempotency_key = str(body.get("idempotency_key") or (body.get("metadata") or {}).get("idempotency_key") or "").strip()
    headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, json=body, headers=headers)
        response.raise_for_status()
        return dict(response.json())


def wait_for_rarespot_run(
    settings: RuntimeSettings,
    *,
    run_id: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    import httpx

    base_url = settings.rarespot_control_base_url.rstrip("/")
    deadline = time.monotonic() + float(timeout_seconds)
    last_run: dict[str, Any] = {}
    with httpx.Client(timeout=30.0) as client:
        while time.monotonic() < deadline:
            run_response = client.get(f"{base_url}/v2/runs/{run_id}")
            run_response.raise_for_status()
            last_run = dict(run_response.json())
            status = str(last_run.get("status") or "")
            if status in {"succeeded", "failed", "canceled"}:
                artifact_response = client.get(f"{base_url}/v2/runs/{run_id}/artifacts")
                artifact_response.raise_for_status()
                artifacts_payload = artifact_response.json()
                artifacts = list(artifacts_payload.get("artifacts") or [])
                summary = parse_response_summary(last_run.get("response_text"))
                terminal_summary = fetch_terminal_event_summary(
                    client,
                    base_url=base_url,
                    run_id=run_id,
                )
                if terminal_summary:
                    summary = {**summary, **terminal_summary}
                artifact_descriptors = normalize_artifact_descriptors(artifacts)
                if status != "succeeded":
                    error = rarespot_terminal_error(
                        status=status,
                        run=last_run,
                        terminal_summary=terminal_summary,
                    )
                    return {
                        "run_id": run_id,
                        "status": status,
                        "error": error,
                        "configuration": summary.get("configuration", {}),
                        "counts_by_class": {},
                        "confidence_summary": {},
                        "top_spectral_review_candidates": [],
                        "artifact_ids": [artifact.get("artifact_id") for artifact in artifact_descriptors],
                        "artifacts": artifact_descriptors,
                        "key_artifacts": select_key_artifacts(artifact_descriptors),
                        "final_answer_hint": rarespot_failure_final_answer_hint(status=status, error=error),
                    }
                return {
                    "run_id": run_id,
                    "status": status,
                    "configuration": summary.get("configuration", {}),
                    "counts_by_class": summary.get("counts_by_class", {}),
                    "confidence_summary": summary.get("confidence_summary", {}),
                    "top_spectral_review_candidates": summary.get("top_spectral_review_candidates", []),
                    "artifact_ids": [artifact.get("artifact_id") for artifact in artifact_descriptors],
                    "artifacts": artifact_descriptors,
                    "key_artifacts": select_key_artifacts(artifact_descriptors),
                    "final_answer_hint": (
                        "RareSpot inference is complete. Use counts_by_class, confidence_summary, "
                        "configuration, and key_artifacts directly in the final answer. Do not search "
                        "the sandbox filesystem or rerun the same configuration unless the user asks "
                        "for another inference pass. Do not create stub or duplicate RareSpot CSV, "
                        "JSON, Markdown, or figure outputs; key_artifacts are the canonical "
                        "downloadable outputs. Create new outputs only for derived comparisons or "
                        "synthesis requested by the user."
                    ),
                }
            time.sleep(2.0)
    return {
        "run_id": run_id,
        "status": last_run.get("status", "timeout"),
        "artifact_ids": [],
        "final_answer_hint": "RareSpot inference did not reach a terminal status before the tool timeout.",
    }


def rarespot_terminal_error(
    *,
    status: str,
    run: dict[str, Any],
    terminal_summary: dict[str, Any],
) -> str:
    for source in (terminal_summary, run):
        for key in ("error", "reason", "message"):
            value = str(source.get(key) or "").strip()
            if value:
                return value
    return f"RareSpot run {status}."


def rarespot_failure_final_answer_hint(*, status: str, error: str) -> str:
    if status == "canceled":
        prefix = "RareSpot inference was canceled"
    else:
        prefix = "RareSpot inference failed"
    suffix = f": {error}" if error else "."
    return (
        f"{prefix}{suffix} "
        "Do not claim production detections, counts, overlays, CSVs, or reports from this run. "
        "Do not run sandbox heuristic replacement or ad hoc CV fallback as if it were RareSpot. "
        "Report the production inference failure honestly and ask the operator to fix the "
        "RareSpot runtime, weights, or input configuration before retrying."
    )


def normalize_artifact_descriptors(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    for artifact in artifacts:
        artifact_id = str(artifact.get("artifact_id") or "").strip()
        if not artifact_id:
            continue
        descriptors.append(
            {
                "artifact_id": artifact_id,
                "kind": artifact.get("kind"),
                "title": artifact.get("title"),
                "category": artifact.get("category"),
                "download_url": f"/v2/artifacts/{artifact_id}/download",
            }
        )
    return descriptors


def select_key_artifacts(artifacts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    keys: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        category = str(artifact.get("category") or "").lower()
        kind = str(artifact.get("kind") or "").lower()
        title = str(artifact.get("title") or "").lower()
        if "annotated_overlay" not in keys and category == "overlay" and kind == "image":
            keys["annotated_overlay"] = artifact
        if "detections_csv" not in keys and kind == "csv" and (
            category == "prediction" or "detection" in title
        ):
            keys["detections_csv"] = artifact
        if "report" not in keys and (kind == "report" or category == "report"):
            keys["report"] = artifact
        if "predictions_json" not in keys and kind == "json" and (
            category == "prediction" or "prediction" in title
        ):
            keys["predictions_json"] = artifact
    return keys


def fetch_terminal_event_summary(client: Any, *, base_url: str, run_id: str) -> dict[str, Any]:
    try:
        event_response = client.get(f"{base_url}/v2/runs/{run_id}/events?limit=50&after_sequence=0")
        event_response.raise_for_status()
        payload = event_response.json()
    except Exception:
        return {}
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        return {}
    terminal_payload: dict[str, Any] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("event_kind") not in {"run.completed", "run.failed", "run.canceled"}:
            continue
        payload = event.get("payload")
        if isinstance(payload, dict):
            terminal_payload = payload
            terminal_payload.setdefault("terminal_event_kind", event.get("event_kind"))
    return terminal_payload


def parse_response_summary(response_text: Any) -> dict[str, Any]:
    if not response_text:
        return {}
    try:
        parsed = json.loads(str(response_text))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}
