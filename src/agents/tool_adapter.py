"""Adapters from internal tool schemas to lightweight internal tool handles."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import inspect
import json
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from src.logger import logger
from src.tools import execute_tool_call
from src.tooling.domains import (
    ANALYSIS_TOOL_SCHEMAS,
    BISQUE_TOOL_SCHEMAS,
    CHEMISTRY_TOOL_SCHEMAS,
    CODE_EXECUTION_TOOL_SCHEMAS,
    VISION_TOOL_SCHEMAS,
)
from src.tooling.engine import _progress_summary_from_result
from .contracts import ArtifactBinding, EvidenceArtifact, GraphEventRecord, ToolResultEnvelope

FunctionTool = dict[str, Any]
ensure_strict_json_schema = None


ToolSchema = dict[str, Any]
CURRENT_AGENT_DOMAIN: ContextVar[str] = ContextVar("CURRENT_AGENT_DOMAIN", default="")
GraphEventCallback = Callable[[GraphEventRecord], None]
ToolApprovalResolver = Callable[[str, dict[str, Any], str], Any]


def _truncate_text(value: Any, *, limit: int = 320) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= int(limit):
        return text
    return text[: max(0, int(limit) - 3)] + "..."


def _safe_json_payload(raw_result: Any) -> dict[str, Any] | None:
    if isinstance(raw_result, dict):
        return raw_result
    text = str(raw_result or "").strip()
    if not text:
        return None
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _derive_bisque_client_view_url(resource_uri: str | None, existing_url: str | None = None) -> str | None:
    candidate = str(existing_url or "").strip()
    if candidate:
        return candidate
    normalized_resource_uri = str(resource_uri or "").strip()
    if not normalized_resource_uri:
        return None
    try:
        parsed = urlparse(normalized_resource_uri)
    except Exception:
        return None
    if not parsed.scheme or not parsed.netloc:
        return None
    normalized = normalized_resource_uri
    if "/image_service/" in normalized:
        normalized = normalized.replace("/image_service/", "/data_service/", 1)
    elif "/data_service/" not in normalized:
        return None
    return f"{parsed.scheme}://{parsed.netloc}/client_service/view?resource={normalized}"


def _summarize_tool_output(tool_name: str, raw_result: Any) -> tuple[dict[str, Any] | None, str | None]:
    payload = _safe_json_payload(raw_result)
    preview = _truncate_text(raw_result)
    if not isinstance(payload, dict):
        return None, (preview or None)

    message_preview = _truncate_text(payload.get("message"))
    if tool_name == "upload_to_bisque":
        summary = {
            "success": bool(payload.get("success")),
            "kind": "bisque_upload",
            "uploaded": payload.get("uploaded"),
            "total": payload.get("total"),
            "dataset_action": payload.get("dataset_action"),
            "dataset_success": payload.get("dataset_success"),
            "dataset_name": payload.get("dataset_name"),
            "dataset_uri": payload.get("dataset_uri"),
            "dataset_members_added": payload.get("dataset_members_added"),
            "dataset_client_view_url": str(payload.get("dataset_client_view_url") or "").strip() or None,
            "rows": (
                _progress_summary_from_result(tool_name, {**payload, "success": True}) or {}
            ).get("rows", []),
        }
        error_text = str(payload.get("error") or "").strip()
        if error_text:
            summary["error"] = error_text
        return (
            {key: value for key, value in summary.items() if value is not None},
            message_preview or preview or None,
        )

    if tool_name == "delete_bisque_resource":
        resource_uri = str(payload.get("deleted_uri") or payload.get("resource_uri") or "").strip()
        client_view_url = _derive_bisque_client_view_url(
            resource_uri,
            str(payload.get("deleted_client_view_url") or payload.get("client_view_url") or "").strip() or None,
        )
        summary = {
            "success": bool(payload.get("success")),
            "kind": "bisque_delete",
            "resource_name": payload.get("resource_name"),
            "resource_uri": resource_uri or None,
            "client_view_url": client_view_url,
            "deletion_verified": payload.get("deletion_verified"),
            "deletion_verification_attempts": payload.get("deletion_verification_attempts"),
        }
        if resource_uri:
            summary["rows"] = [
                {
                    "name": str(payload.get("resource_name") or "").strip() or resource_uri,
                    "uri": client_view_url or resource_uri,
                    "resource_uri": resource_uri,
                    "client_view_url": client_view_url,
                }
            ]
        error_text = str(payload.get("error") or "").strip()
        if error_text:
            summary["error"] = error_text
        return (
            {key: value for key, value in summary.items() if value is not None},
            message_preview or preview or None,
        )

    if tool_name == "add_tags_to_resource":
        resource_uri = str(payload.get("resource_uri") or "").strip()
        client_view_url = _derive_bisque_client_view_url(
            resource_uri,
            str(payload.get("client_view_url") or "").strip() or None,
        )
        summary = {
            "success": bool(payload.get("success")),
            "kind": "bisque_tags",
            "resource_uri": resource_uri or None,
            "client_view_url": client_view_url,
            "tag_count": payload.get("total_tags"),
            "tags": payload.get("tags_added"),
            "verification_attempts": payload.get("verification_attempts"),
        }
        if resource_uri:
            summary["rows"] = [
                {
                    "name": "Tagged resource",
                    "uri": client_view_url or resource_uri,
                    "resource_uri": resource_uri,
                    "client_view_url": client_view_url,
                }
            ]
        error_text = str(payload.get("error") or "").strip()
        if error_text:
            summary["error"] = error_text
        return (
            {key: value for key, value in summary.items() if value is not None},
            message_preview or preview or None,
        )

    if tool_name == "bisque_add_gobjects":
        resource_uri = str(payload.get("resource_uri") or "").strip()
        client_view_url = _derive_bisque_client_view_url(
            resource_uri,
            str(payload.get("client_view_url") or "").strip() or None,
        )
        summary = {
            "success": bool(payload.get("success")),
            "kind": "bisque_annotations",
            "resource_uri": resource_uri or None,
            "client_view_url": client_view_url,
            "added_total": payload.get("added_total"),
            "counts_by_type": payload.get("counts_by_type"),
            "verification_attempts": payload.get("verification_attempts"),
        }
        if resource_uri:
            summary["rows"] = [
                {
                    "name": "Annotated resource",
                    "uri": client_view_url or resource_uri,
                    "resource_uri": resource_uri,
                    "client_view_url": client_view_url,
                }
            ]
        error_text = str(payload.get("error") or "").strip()
        if error_text:
            summary["error"] = error_text
        return (
            {key: value for key, value in summary.items() if value is not None},
            message_preview or preview or None,
        )

    engine_summary = _progress_summary_from_result(tool_name, payload)
    if isinstance(engine_summary, dict) and engine_summary:
        return engine_summary, (message_preview or preview or None)
    if tool_name == "segment_image_sam3":
        summary: dict[str, Any] = {
            "success": bool(payload.get("success")),
            "processed": payload.get("processed"),
            "total_files": payload.get("total_files"),
            "total_masks_generated": payload.get("total_masks_generated"),
            "concept_prompt": payload.get("concept_prompt"),
            "concept_prompt_source": payload.get("concept_prompt_source"),
            "coverage_scope": payload.get("coverage_scope"),
            "instance_count_reported_total": payload.get("instance_count_reported_total"),
            "instance_count_measured_total": payload.get("instance_count_measured_total"),
            "instance_count_mismatch_files": payload.get("instance_count_mismatch_files"),
            "instance_count_scope": payload.get("instance_count_scope"),
            "coverage_percent_mean": payload.get("coverage_percent_mean"),
            "coverage_percent_min": payload.get("coverage_percent_min"),
            "coverage_percent_max": payload.get("coverage_percent_max"),
            "instance_coverage_percent_mean": payload.get("instance_coverage_percent_mean"),
            "instance_coverage_percent_min": payload.get("instance_coverage_percent_min"),
            "instance_coverage_percent_max": payload.get("instance_coverage_percent_max"),
            "instance_area_voxels_mean": payload.get("instance_area_voxels_mean"),
            "instance_area_voxels_min": payload.get("instance_area_voxels_min"),
            "instance_area_voxels_max": payload.get("instance_area_voxels_max"),
            "auto_zero_mask_fallback_files": payload.get("auto_zero_mask_fallback_files"),
            "mode": payload.get("mode"),
            "preset": payload.get("preset"),
        }
        return (
            {k: v for k, v in summary.items() if v is not None},
            message_preview or preview or None,
        )

    if tool_name == "yolo_detect":
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        counts_by_class = (
            payload.get("counts_by_class")
            if isinstance(payload.get("counts_by_class"), dict)
            else {}
        )
        stability_audit = (
            payload.get("prediction_stability_audit")
            if isinstance(payload.get("prediction_stability_audit"), dict)
            else {}
        )
        normalized_counts: list[tuple[str, int]] = []
        for name, count in counts_by_class.items():
            try:
                normalized_counts.append((str(name), int(count)))
            except Exception:
                continue
        top_classes = [
            {"class": name, "count": count}
            for name, count in sorted(
                normalized_counts,
                key=lambda row: (-int(row[1]), str(row[0])),
            )[:5]
        ]
        summary = {
            "success": bool(payload.get("success")),
            "message": payload.get("message"),
            "total_boxes": metrics.get("total_boxes"),
            "avg_confidence": metrics.get("avg_confidence"),
            "finetune_recommended": metrics.get("finetune_recommended"),
            "top_classes": top_classes,
            "analysis_summary": payload.get("analysis_summary"),
            "prediction_images": payload.get("prediction_images"),
            "prediction_images_raw": payload.get("prediction_images_raw"),
            "prediction_image_records": payload.get("prediction_image_records"),
            "scientific_summary": payload.get("scientific_summary"),
            "ecology_context": payload.get("ecology_context"),
            "spatial_analysis": payload.get("spatial_analysis"),
            "inference_configuration": payload.get("inference_configuration"),
            "counts_by_class": payload.get("counts_by_class"),
            "prediction_stability_audit": {
                "summary": stability_audit.get("summary"),
                "review_candidates": list(stability_audit.get("review_candidates") or [])[:5],
                "backend_method": stability_audit.get("backend_method"),
            }
            if stability_audit
            else {},
        }
        return (
            {k: v for k, v in summary.items() if v is not None},
            message_preview or preview or None,
        )

    if tool_name in {"score_spectral_instability", "analyze_prediction_stability"}:
        summary_block = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        top_ranked = summary_block.get("top_ranked") if isinstance(summary_block.get("top_ranked"), list) else []
        summary = {
            "success": bool(payload.get("success")),
            "message": payload.get("message"),
            "analysis_kind": payload.get("analysis_kind"),
            "method": payload.get("method"),
            "backend_method": payload.get("backend_method"),
            "method_version": payload.get("method_version"),
            "model_name": payload.get("model_name"),
            "scores_json": payload.get("scores_json"),
            "image_count": summary_block.get("image_count"),
            "nonzero_score_count": summary_block.get("nonzero_score_count"),
            "max_score": summary_block.get("max_score"),
            "mean_score": summary_block.get("mean_score"),
            "median_score": summary_block.get("median_score"),
            "top_ranked": list(top_ranked)[:5],
            "review_candidates": list(payload.get("review_candidates") or [])[:5],
            "active_learning_note": payload.get("active_learning_note"),
        }
        return (
            {k: v for k, v in summary.items() if v is not None},
            message_preview or preview or None,
        )

    generic_summary: dict[str, Any] = {}
    if "success" in payload:
        generic_summary["success"] = bool(payload.get("success"))
    if "message" in payload:
        generic_summary["message"] = _truncate_text(payload.get("message"), limit=220)
    return (generic_summary or None), (message_preview or preview or None)


def _normalize_artifact_title(path_value: str) -> str:
    name = Path(str(path_value or "")).name.strip()
    return name or "artifact"


def _as_evidence_artifact(
    *,
    path_value: str,
    source_tool: str,
    kind: str,
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvidenceArtifact | None:
    normalized_path = str(path_value or "").strip()
    if not normalized_path:
        return None
    suffix = Path(normalized_path).suffix.lower()
    mime_type: str | None = None
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".tif", ".tiff"}:
        mime_type = "image/*"
    elif suffix in {".nii", ".nii.gz", ".nrrd", ".mha", ".mhd"}:
        mime_type = "application/octet-stream"
    elif suffix in {".h5", ".hdf5"}:
        mime_type = "application/x-hdf5"
    elif suffix in {".csv"}:
        mime_type = "text/csv"
    elif suffix in {".json"}:
        mime_type = "application/json"
    return EvidenceArtifact(
        kind=kind,
        title=str(title or "").strip() or _normalize_artifact_title(normalized_path),
        path=normalized_path,
        mime_type=mime_type,
        source_tool=source_tool,
        metadata=dict(metadata or {}),
    )


def _collect_path_candidates(payload: dict[str, Any]) -> list[EvidenceArtifact]:
    artifacts: list[EvidenceArtifact] = []
    seen: set[tuple[str, str]] = set()

    def _maybe_add(
        path_value: str,
        *,
        source_tool: str,
        kind: str,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        artifact = _as_evidence_artifact(
            path_value=path_value,
            source_tool=source_tool,
            kind=kind,
            title=title,
            metadata=metadata,
        )
        if artifact is None:
            return
        dedupe_key = (str(artifact.kind), str(artifact.path))
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        artifacts.append(artifact)

    for key in ("visualization_paths", "prediction_images", "output_files", "preferred_upload_paths"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value[:32]:
                if isinstance(item, dict):
                    _maybe_add(
                        str(item.get("path") or ""),
                        source_tool=str(payload.get("tool_name") or ""),
                        kind="ui_artifact",
                        title=str(item.get("title") or "") or None,
                    )
                else:
                    _maybe_add(
                        str(item or ""),
                        source_tool=str(payload.get("tool_name") or ""),
                        kind="ui_artifact",
                    )

    records = payload.get("prediction_image_records")
    if str(payload.get("tool_name") or "").strip() != "yolo_detect" and isinstance(records, list):
        for item in records[:32]:
            if not isinstance(item, dict):
                continue
            source_path = str(item.get("source_path") or "").strip()
            if not source_path:
                continue
            _maybe_add(
                source_path,
                source_tool=str(payload.get("tool_name") or ""),
                kind="ui_artifact",
                title=str(item.get("source_name") or "") or None,
                metadata={
                    "preview_path": str(item.get("preview_path") or "").strip() or None,
                    "preview_name": str(item.get("preview_name") or "").strip() or None,
                    "preview_kind": str(item.get("preview_kind") or "").strip() or None,
                    "box_count": item.get("box_count"),
                    "class_counts": item.get("class_counts"),
                },
            )

    artifacts_field = payload.get("artifacts")
    if isinstance(artifacts_field, list):
        for item in artifacts_field[:32]:
            if not isinstance(item, dict):
                continue
            _maybe_add(
                str(item.get("path") or ""),
                source_tool=str(payload.get("tool_name") or ""),
                kind="download_artifact",
                title=str(item.get("title") or "") or None,
            )
    return artifacts


def _build_tool_result_envelope(tool_name: str, raw_result: Any) -> ToolResultEnvelope:
    payload = _safe_json_payload(raw_result) or {}
    output_summary, output_preview = _summarize_tool_output(tool_name, raw_result)
    normalized_payload = dict(payload)
    normalized_payload["tool_name"] = str(tool_name)

    measurements: list[dict[str, Any]] = []
    summary_block = output_summary or {}
    for key, value in summary_block.items():
        if isinstance(value, (int, float, str)) and key not in {"message", "success"}:
            measurements.append({"name": str(key), "value": value})

    evidence: list[dict[str, Any]] = []
    message = str(payload.get("message") or output_preview or "").strip()
    if message:
        evidence.append({"source": f"tool:{tool_name}", "summary": message})

    artifacts = _collect_path_candidates(normalized_payload)
    bindings: list[ArtifactBinding] = []
    refs = payload.get("latest_result_refs")
    if isinstance(refs, dict):
        for key, value in refs.items():
            token = str(key or "").strip()
            if not token:
                continue
            bindings.append(ArtifactBinding(key=token, value=value))

    next_actions: list[str] = []
    if tool_name == "yolo_detect":
        ecology_context = (
            payload.get("ecology_context")
            if isinstance(payload.get("ecology_context"), dict)
            else {}
        )
        if not ecology_context and isinstance(summary_block.get("scientific_summary"), dict):
            scientific_summary = dict(summary_block.get("scientific_summary") or {})
            ecology_context = (
                scientific_summary.get("ecology_context")
                if isinstance(scientific_summary.get("ecology_context"), dict)
                else {}
            )
        review_flags = (
            ecology_context.get("review_flags")
            if isinstance(ecology_context.get("review_flags"), dict)
            else {}
        )
        manual_review_recommended = bool(review_flags.get("manual_review_recommended"))
        if ecology_context and manual_review_recommended:
            next_actions.append(
                "Review the tile and nearby survey context before drawing stronger occupancy or habitat conclusions."
            )
        elif bool(summary_block.get("finetune_recommended")):
            next_actions.append("Consider a finetuned detector if the target class is domain-specific.")
    if tool_name == "segment_image_sam3" and int(summary_block.get("processed") or 0) <= 0:
        next_actions.append("Refine segmentation prompts or inspect the input image before retrying.")

    return ToolResultEnvelope(
        success=bool(payload.get("success", True)),
        summary=message or str(output_preview or "").strip(),
        measurements=measurements[:24],
        evidence=evidence[:12],
        ui_artifacts=[artifact for artifact in artifacts if artifact.kind == "ui_artifact"][:24],
        download_artifacts=[artifact for artifact in artifacts if artifact.kind == "download_artifact"][:24],
        structured_outputs=(summary_block or {}),
        next_actions=next_actions,
        artifact_bindings=bindings[:24],
    )


def _merge_latest_result_refs(state: "ToolRunState", tool_name: str, raw_result: Any) -> None:
    payload = _safe_json_payload(raw_result)
    if not isinstance(payload, dict):
        return
    refs = payload.get("latest_result_refs")
    if isinstance(refs, dict):
        state.latest_result_refs.update(refs)
    if tool_name == "segment_image_sam3":
        mask_paths: list[str] = []
        preferred = payload.get("preferred_upload_paths")
        if isinstance(preferred, list):
            for item in preferred:
                token = str(item or "").strip()
                if token:
                    mask_paths.append(token)
        files_processed = payload.get("files_processed")
        if isinstance(files_processed, list):
            for row in files_processed:
                if not isinstance(row, dict):
                    continue
                token = str(row.get("preferred_upload_path") or "").strip()
                if token:
                    mask_paths.append(token)
        deduped: list[str] = []
        seen: set[str] = set()
        for item in mask_paths:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        if deduped:
            state.latest_result_refs["segment_image_sam3.mask_paths"] = deduped[:24]
            state.latest_result_refs["segment_image_sam3.preferred_upload_paths"] = deduped[:24]
            state.latest_result_refs["segment_image_sam3.latest_mask_path"] = deduped[0]
            state.latest_result_refs["latest_mask_path"] = deduped[0]


async def invoke_internal_tool_with_state(
    *,
    tool_name: str,
    arguments: str | dict[str, Any],
    state: "ToolRunState" | None,
    domain_id: str | None = None,
) -> str:
    """Execute one internal tool call with the same state/budget tracking used by SDK tools."""

    args_json = (
        str(arguments)
        if isinstance(arguments, str)
        else json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    )
    invocation_id: int | None = None
    uploaded_files: list[str] = []
    if isinstance(state, ToolRunState):
        invocation_id = await state.begin_tool_invocation(
            tool_name,
            args_json,
            domain_id=(str(domain_id or "").strip() or None),
        )
        uploaded_files = list(state.uploaded_files)
    try:
        raw_result = execute_tool_call(
            tool_name,
            arguments,
            uploaded_files=uploaded_files,
            user_text=(state.latest_user_text if isinstance(state, ToolRunState) else ""),
            latest_result_refs=(state.latest_result_refs if isinstance(state, ToolRunState) else None),
        )
        if isinstance(state, ToolRunState) and invocation_id is not None:
            output_summary: dict[str, Any] | None = None
            output_preview: str | None = None
            output_envelope: ToolResultEnvelope | None = None
            try:
                _merge_latest_result_refs(state, tool_name, raw_result)
                output_summary, output_preview = _summarize_tool_output(tool_name, raw_result)
                output_envelope = _build_tool_result_envelope(tool_name, raw_result)
            except Exception as summary_exc:
                logger.warning(
                    "Tool summary extraction failed for %s: %s",
                    tool_name,
                    summary_exc,
                )
            await state.finish_tool_invocation(
                invocation_id,
                status="success",
                output_summary=output_summary,
                output_preview=output_preview,
                output_envelope=output_envelope,
            )
        return str(raw_result)
    except Exception as exc:
        logger.exception("Tool invocation failed: %s", tool_name)
        if isinstance(state, ToolRunState) and invocation_id is not None:
            await state.finish_tool_invocation(
                invocation_id,
                status="error",
                error=str(exc),
            )
        return json.dumps(
            {
                "success": False,
                "error": f"Tool {tool_name} failed: {exc}",
            },
            ensure_ascii=False,
        )


def list_internal_tool_schemas() -> list[ToolSchema]:
    """Return all internal function tool schemas."""

    all_schemas: list[ToolSchema] = []
    for group in (
        BISQUE_TOOL_SCHEMAS,
        VISION_TOOL_SCHEMAS,
        ANALYSIS_TOOL_SCHEMAS,
        CODE_EXECUTION_TOOL_SCHEMAS,
        CHEMISTRY_TOOL_SCHEMAS,
    ):
        for item in group:
            if isinstance(item, dict):
                all_schemas.append(item)
    return all_schemas


def map_schemas_by_name(schemas: list[ToolSchema]) -> dict[str, ToolSchema]:
    """Create a name -> schema map for tool schemas."""

    out: dict[str, ToolSchema] = {}
    for schema in schemas:
        fn = schema.get("function")
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "").strip()
        if name:
            out[name] = schema
    return out


def _prepare_agent_params_json_schema(params: Any) -> tuple[dict[str, Any], bool]:
    if not isinstance(params, dict):
        return {"type": "object", "properties": {}}, True
    if ensure_strict_json_schema is None:
        return dict(params), False
    strict_params = deepcopy(params)
    try:
        ensure_strict_json_schema(strict_params)
    except Exception:
        return dict(params), False
    return strict_params, True


@dataclass
class _ToolRunLedger:
    started_monotonic: float = field(default_factory=time.monotonic)
    tool_calls: int = 0
    tool_invocations: list[dict[str, Any]] = field(default_factory=list)
    invocation_seq: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class ToolRunState:
    """Scoped mutable state for tool invocation budgets and artifact refs."""

    uploaded_files: list[str]
    max_tool_calls: int
    max_runtime_seconds: int
    latest_user_text: str = ""
    scope_id: str = "root"
    latest_result_refs: dict[str, Any] = field(default_factory=dict)
    event_callback: GraphEventCallback | None = None
    _ledger: _ToolRunLedger = field(default_factory=_ToolRunLedger)

    @property
    def started_monotonic(self) -> float:
        return float(self._ledger.started_monotonic)

    @property
    def tool_calls(self) -> int:
        return int(self._ledger.tool_calls)

    @property
    def tool_invocations(self) -> list[dict[str, Any]]:
        return self._ledger.tool_invocations

    def spawn_scope(
        self,
        scope_id: str,
        *,
        initial_refs: dict[str, Any] | None = None,
    ) -> "ToolRunState":
        next_refs = dict(self.latest_result_refs if self.scope_id == "root" else {})
        if isinstance(initial_refs, dict):
            next_refs.update(initial_refs)
        return ToolRunState(
            uploaded_files=list(self.uploaded_files),
            max_tool_calls=int(self.max_tool_calls),
            max_runtime_seconds=int(self.max_runtime_seconds),
            latest_user_text=self.latest_user_text,
            scope_id=str(scope_id or "").strip() or "scope",
            latest_result_refs=next_refs,
            event_callback=self.event_callback,
            _ledger=self._ledger,
        )

    def _emit_event(self, record: GraphEventRecord) -> None:
        callback = self.event_callback
        if callback is None:
            return
        try:
            callback(record)
        except Exception as exc:
            logger.warning("Tool graph event callback failed for scope %s: %s", self.scope_id, exc)

    async def reserve_tool_call(self, tool_name: str) -> None:
        """Reserve a tool call slot and enforce runtime budgets."""

        elapsed = time.monotonic() - self.started_monotonic
        if elapsed > float(self.max_runtime_seconds):
            raise RuntimeError(
                f"chat runtime exceeded budget ({self.max_runtime_seconds}s) before calling {tool_name}"
            )
        async with self._ledger.lock:
            self._ledger.tool_calls += 1
            if self._ledger.tool_calls > int(self.max_tool_calls):
                raise RuntimeError(f"tool call budget exceeded ({self.max_tool_calls})")

    @staticmethod
    def _args_fingerprint(args_json: str) -> str:
        token = str(args_json or "")
        if not token:
            return ""
        return hashlib.sha256(token.encode("utf-8", errors="ignore")).hexdigest()[:16]

    async def begin_tool_invocation(
        self,
        tool_name: str,
        args_json: str,
        *,
        domain_id: str | None = None,
    ) -> int:
        """Reserve budget and register the start of one tool invocation."""

        elapsed = time.monotonic() - self.started_monotonic
        if elapsed > float(self.max_runtime_seconds):
            raise RuntimeError(
                f"chat runtime exceeded budget ({self.max_runtime_seconds}s) before calling {tool_name}"
            )
        async with self._ledger.lock:
            self._ledger.tool_calls += 1
            if self._ledger.tool_calls > int(self.max_tool_calls):
                raise RuntimeError(f"tool call budget exceeded ({self.max_tool_calls})")
            self._ledger.invocation_seq += 1
            invocation_id = int(self._ledger.invocation_seq)
            entry: dict[str, Any] = {
                "invocation_id": invocation_id,
                "tool": str(tool_name),
                "status": "running",
                "started_monotonic": time.monotonic(),
                "args_fingerprint": self._args_fingerprint(args_json),
                "scope_id": self.scope_id,
            }
            token = str(domain_id or "").strip()
            if token:
                entry["domain_id"] = token
            self._ledger.tool_invocations.append(entry)
            self._emit_event(
                GraphEventRecord(
                    kind="tool",
                    workflow_kind="tool_execution",
                    phase="tool",
                    status="started",
                    agent_role="domain_specialist",
                    node="tool",
                    domain_id=token or None,
                    scope_id=self.scope_id,
                    message=f"Calling {tool_name}",
                    payload={
                        "tool": str(tool_name),
                        "invocation_id": invocation_id,
                        "scope_id": self.scope_id,
                    },
                )
            )
            return invocation_id

    async def finish_tool_invocation(
        self,
        invocation_id: int,
        *,
        status: str,
        error: str | None = None,
        output_summary: dict[str, Any] | None = None,
        output_preview: str | None = None,
        output_envelope: ToolResultEnvelope | None = None,
    ) -> None:
        """Mark one invocation as completed/failed with duration metadata."""

        target = int(invocation_id)
        async with self._ledger.lock:
            target_entry: dict[str, Any] | None = None
            for entry in reversed(self._ledger.tool_invocations):
                if int(entry.get("invocation_id") or -1) != target:
                    continue
                finished = time.monotonic()
                started = float(entry.get("started_monotonic") or finished)
                entry["finished_monotonic"] = finished
                entry["duration_seconds"] = round(max(0.0, finished - started), 6)
                entry["status"] = str(status or "unknown")
                if error:
                    entry["error"] = str(error)
                if isinstance(output_summary, dict) and output_summary:
                    entry["output_summary"] = output_summary
                preview = str(output_preview or "").strip()
                if preview:
                    entry["output_preview"] = _truncate_text(preview, limit=500)
                if output_envelope is not None:
                    entry["output_envelope"] = output_envelope.model_dump(mode="json")
                elif isinstance(output_summary, dict) or preview or error:
                    envelope = _build_tool_result_envelope(
                        str(entry.get("tool") or ""),
                        {
                            "success": status == "success",
                            "message": preview or error or "",
                            "latest_result_refs": dict(self.latest_result_refs),
                            **(output_summary or {}),
                        },
                    )
                    entry["output_envelope"] = envelope.model_dump(mode="json")
                target_entry = entry
                break
        if target_entry is not None:
            self._emit_event(
                GraphEventRecord(
                    kind="tool",
                    workflow_kind="tool_execution",
                    phase="tool",
                    status=("completed" if status == "success" else "failed"),
                    agent_role="domain_specialist",
                    node="tool",
                    domain_id=str(target_entry.get("domain_id") or "").strip() or None,
                    scope_id=str(target_entry.get("scope_id") or self.scope_id),
                    message=str(target_entry.get("output_preview") or error or "").strip() or None,
                    payload={
                        "tool": str(target_entry.get("tool") or ""),
                        "invocation_id": int(target_entry.get("invocation_id") or 0),
                        "status": str(status or ""),
                        "output_summary": (
                            dict(target_entry.get("output_summary"))
                            if isinstance(target_entry.get("output_summary"), dict)
                            else {}
                        ),
                        "output_envelope": (
                            dict(target_entry.get("output_envelope"))
                            if isinstance(target_entry.get("output_envelope"), dict)
                            else {}
                        ),
                        "error": str(error or "").strip() or None,
                    },
                )
            )


def serialize_tool_run_state(state: Any) -> dict[str, Any]:
    if isinstance(state, dict):
        return dict(state)
    if not isinstance(state, ToolRunState):
        raise TypeError(f"Unsupported tool-run state for serialization: {type(state)!r}")
    return {
        "__type__": "ToolRunState",
        "uploaded_files": list(state.uploaded_files),
        "max_tool_calls": int(state.max_tool_calls),
        "max_runtime_seconds": int(state.max_runtime_seconds),
        "latest_user_text": str(state.latest_user_text or ""),
        "scope_id": str(state.scope_id or "").strip() or "root",
        "latest_result_refs": deepcopy(dict(state.latest_result_refs or {})),
        "tool_calls": int(state.tool_calls),
        "tool_invocations": deepcopy(list(state.tool_invocations or [])),
        "invocation_seq": int(getattr(state._ledger, "invocation_seq", 0) or 0),
    }


def deserialize_tool_run_state(
    payload: dict[str, Any],
    *,
    event_callback: GraphEventCallback | None = None,
) -> ToolRunState:
    state = ToolRunState(
        uploaded_files=[
            str(item).strip()
            for item in list(payload.get("uploaded_files") or [])
            if str(item or "").strip()
        ],
        max_tool_calls=max(1, int(payload.get("max_tool_calls") or 1)),
        max_runtime_seconds=max(1, int(payload.get("max_runtime_seconds") or 1)),
        latest_user_text=str(payload.get("latest_user_text") or ""),
        scope_id=str(payload.get("scope_id") or "").strip() or "root",
        latest_result_refs=deepcopy(
            dict(payload.get("latest_result_refs") or {})
            if isinstance(payload.get("latest_result_refs"), dict)
            else {}
        ),
        event_callback=event_callback,
    )
    invocations = payload.get("tool_invocations")
    if isinstance(invocations, list):
        state._ledger.tool_invocations = deepcopy(invocations)
    explicit_tool_calls = payload.get("tool_calls")
    if explicit_tool_calls is not None:
        try:
            state._ledger.tool_calls = max(0, int(explicit_tool_calls))
        except Exception:
            state._ledger.tool_calls = len(state._ledger.tool_invocations)
    else:
        state._ledger.tool_calls = len(state._ledger.tool_invocations)
    explicit_invocation_seq = payload.get("invocation_seq")
    if explicit_invocation_seq is not None:
        try:
            state._ledger.invocation_seq = max(0, int(explicit_invocation_seq))
        except Exception:
            state._ledger.invocation_seq = max(
                [int(item.get("invocation_id") or 0) for item in state._ledger.tool_invocations if isinstance(item, dict)]
                or [0]
            )
    else:
        state._ledger.invocation_seq = max(
            [int(item.get("invocation_id") or 0) for item in state._ledger.tool_invocations if isinstance(item, dict)]
            or [0]
        )
    state._ledger.started_monotonic = time.monotonic()
    return state


def build_function_tools_for_allowlist(
    *,
    allowed_tool_names: set[str],
    schema_map: dict[str, ToolSchema],
    needs_approval_resolver: ToolApprovalResolver | None = None,
) -> list[FunctionTool]:
    """Convert allowed internal schemas into lightweight internal tool handles."""

    tools: list[FunctionTool] = []
    for tool_name in sorted(allowed_tool_names):
        schema = schema_map.get(tool_name)
        if not isinstance(schema, dict):
            continue
        fn = schema.get("function")
        if not isinstance(fn, dict):
            continue
        description = str(fn.get("description") or "").strip() or f"Call {tool_name}"
        params = fn.get("parameters")
        params_json_schema, strict_json_schema = _prepare_agent_params_json_schema(params)
        if not strict_json_schema:
            logger.info(
                "Agent tool schema for %s is not strict-compatible; using non-strict validation.",
                tool_name,
            )

        async def _invoke(context: Any, args_json: str, _tool_name: str = tool_name) -> str:
            state = getattr(context, "context", None)
            domain_id = str(CURRENT_AGENT_DOMAIN.get("") or "").strip()
            return await invoke_internal_tool_with_state(
                tool_name=_tool_name,
                arguments=args_json,
                state=(state if isinstance(state, ToolRunState) else None),
                domain_id=(domain_id or None),
            )

        async def _needs_approval(context: Any, params: dict[str, Any], call_id: str, _tool_name: str = tool_name) -> bool:
            if needs_approval_resolver is None:
                return False
            decision = needs_approval_resolver(
                _tool_name,
                dict(params or {}),
                str(call_id or "").strip(),
            )
            if inspect.isawaitable(decision):
                decision = await decision
            return bool(decision)

        tools.append(
            {
                "name": tool_name,
                "description": description,
                "params_json_schema": params_json_schema,
                "on_invoke_tool": _invoke,
                "strict_json_schema": strict_json_schema,
                "needs_approval": (
                    _needs_approval if needs_approval_resolver is not None else False
                ),
            }
        )
    return tools
