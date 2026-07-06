"""Batch model-inference processor for the Data-Agent worker.

One processor handles analysis.* job types. For analysis.megaseg it streams each input
image to the stateless GPU /v1/infer, writes the produced mask + metrics into the shared
upload root, and registers them with the control plane (grouped into the job's results
collection). Per-image failures are isolated; the batch resumes after a worker restart by
skipping items already recorded done in the persisted output_summary; cancellation is
honored between images.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..data_agent.worker import DataAgentJobEnvelope, DefaultDataAgentProcessor
from ..rarespot.config import RareSpotConfig
from ..rarespot.inference import run_rarespot_inference
from ..rarespot.uploads import SAFE_FILE_ID_RE, unique_file_ids
from .client import fetch_job, register_outputs, run_megaseg_infer
from .config import AnalysisSettings

logger = logging.getLogger("ultra.analysis.worker")

_CANCEL_POLL_EVERY = 5
# Derived viewer artifacts live alongside the source in the upload store; the model needs
# the ORIGINAL volume, never the pyramid.
_DERIVED_MARKERS = ("__pyramid", "__thumbnail")


class _AnalysisJobCanceledError(RuntimeError):
    pass


def _resolve_source_path(file_id: str, upload_roots: tuple[Path, ...]) -> Path | None:
    """Resolve a resource_id to its ORIGINAL uploaded file at <root>/<file_id>__<name>.

    Deliberately a TOP-LEVEL (non-recursive) match: the original upload lands directly in the
    upload root, while derived artifacts (e.g. <id>__pyramid.tif) live in a derived/ subdir.
    A recursive scan would sort derived/ before the original and feed the model a pyramid,
    which it cannot read — so we match top-level only and skip derived markers.
    """
    if not SAFE_FILE_ID_RE.fullmatch(file_id):
        return None
    for root in upload_roots:
        base = Path(root).expanduser()
        if not base.exists():
            continue
        matches = sorted(
            candidate
            for candidate in base.glob(f"{file_id}__*")
            if candidate.is_file() and not any(marker in candidate.name for marker in _DERIVED_MARKERS)
        )
        if matches:
            return matches[0]
        exact = base / file_id
        if exact.is_file():
            return exact
    return None


def _copy_with_sha(src: Path, dst: Path) -> tuple[int, str]:
    hasher = hashlib.sha256()
    size = 0
    with src.open("rb") as fin, dst.open("wb") as fout:
        while True:
            chunk = fin.read(1024 * 1024)
            if not chunk:
                break
            fout.write(chunk)
            hasher.update(chunk)
            size += len(chunk)
    return size, hasher.hexdigest()


def _megaseg_params(job: DataAgentJobEnvelope) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    selector = job.input_selector or {}
    metadata = job.metadata or {}
    if isinstance(selector.get("params"), dict):
        raw.update(selector["params"])
    elif isinstance(metadata.get("params"), dict):
        raw.update(metadata["params"])
    # Batch defaults favor speed: the per-image overlay/report are skipped (the batch report
    # is aggregated separately); callers can re-enable via params.
    params: dict[str, Any] = {"save_visualizations": False, "generate_report": False}
    params.update(raw)
    return params


def _job_params(job: DataAgentJobEnvelope) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    metadata = job.metadata or {}
    selector = job.input_selector or {}
    if isinstance(metadata.get("params"), dict):
        raw.update(metadata["params"])
    if isinstance(selector.get("params"), dict):
        raw.update(selector["params"])
    return raw


def _bool_param(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _int_param(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_param(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _id_part(value: Any) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value or ""))
    return cleaned.strip("_") or "artifact"


def _path_part(value: Any) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {".", "-", "_"} else "_" for ch in str(value or ""))
    return cleaned.strip("._") or "artifact"


def _safe_job_path_part(value: Any, label: str) -> str:
    part = str(value or "").strip()
    if not part or part in {".", ".."} or "/" in part or "\\" in part or "\x00" in part:
        raise ValueError(f"unsafe {label} path segment: {part or '<empty>'}")
    return part


class AnalysisProcessor:
    """Data-Agent processor for batch model inference. Unknown job types fall through to the
    default processor so this can be the node's sole data-agent worker without breaking
    caption/tag/snapshot jobs."""

    def __init__(self, settings: AnalysisSettings | None = None) -> None:
        self.settings = settings or AnalysisSettings.from_env()
        self._fallback = DefaultDataAgentProcessor()

    async def __call__(self, job: DataAgentJobEnvelope, progress: Any) -> dict[str, Any]:
        job_type = (job.job_type or "").strip().lower()
        if job_type == "analysis.megaseg":
            return await self._process_megaseg(job, progress)
        if job_type == "analysis.rarespot":
            return await self._process_rarespot(job, progress)
        return await self._fallback(job, progress)

    async def _process_megaseg(self, job: DataAgentJobEnvelope, progress: Any) -> dict[str, Any]:
        settings = self.settings
        if not settings.megaseg_service_url or not settings.megaseg_service_api_key:
            raise RuntimeError("MEGASEG_SERVICE_URL and MEGASEG_SERVICE_API_KEY are required for analysis.megaseg")
        principal = job.principal_headers(settings)
        params = _megaseg_params(job)
        # The envelope carries the create-time metadata, so we always have the results
        # collection here even if the job's stored metadata changes during processing.
        collection_id = str((job.metadata or {}).get("results_collection_id") or "")
        requested = unique_file_ids(job.resource_ids)
        total = len(requested)

        items: dict[str, dict[str, Any]] = await asyncio.to_thread(self._prior_items, job, principal)

        path_by_id: dict[str, Path] = {}
        for rid in requested:
            source = _resolve_source_path(rid, settings.upload_roots)
            if source is not None:
                path_by_id[rid] = source

        canceled = False
        for index, rid in enumerate(requested):
            if index % _CANCEL_POLL_EVERY == 0 and await self._is_canceled(job, principal):
                canceled = True
                break
            if items.get(rid, {}).get("status") == "done":
                await self._report(progress, items, total, job, f"Skipped {rid} (already done).")
                continue
            path = path_by_id.get(rid)
            if path is None:
                items[rid] = {"status": "failed", "error": "input file not found"}
                await self._report(progress, items, total, job, f"Input {rid} not found.")
                continue
            try:
                outputs = await asyncio.to_thread(self._infer_and_stage_megaseg, job, rid, path, params)
                if outputs:
                    await asyncio.to_thread(
                        register_outputs,
                        control_base_url=settings.control_base_url,
                        job_id=job.job_id,
                        principal_headers=principal,
                        outputs=outputs,
                        timeout=settings.control_status_timeout_seconds,
                        collection_id=collection_id,
                    )
                items[rid] = {"status": "done", "outputs": len(outputs)}
            except Exception as exc:  # noqa: BLE001 - isolate one image's failure
                logger.exception("megaseg batch: resource %s failed", rid)
                items[rid] = {"status": "failed", "error": str(exc)[:500]}
            await self._report(progress, items, total, job, f"Processed {rid}.")

        summary = self._summary(items, total, job)
        summary["canceled"] = canceled
        return summary

    async def _process_rarespot(self, job: DataAgentJobEnvelope, progress: Any) -> dict[str, Any]:
        settings = self.settings
        principal = job.principal_headers(settings)
        collection_id = str((job.metadata or {}).get("results_collection_id") or "")
        requested = unique_file_ids(job.resource_ids)
        total = len(requested)
        items: dict[str, dict[str, Any]] = await asyncio.to_thread(self._prior_items, job, principal)

        path_by_id: dict[str, Path] = {}
        for rid in requested:
            if items.get(rid, {}).get("status") == "done":
                continue
            source = _resolve_source_path(rid, settings.upload_roots)
            if source is not None:
                path_by_id[rid] = source
                continue
            items[rid] = {"status": "failed", "error": "input file not found"}

        canceled = await self._is_canceled(job, principal)
        if canceled:
            summary = self._rarespot_summary(items, total, job)
            summary["canceled"] = True
            summary["terminal_status"] = "canceled"
            await self._report(progress, items, total, job, "RareSpot job canceled.", model="rarespot")
            return summary

        runnable_ids = [rid for rid in requested if rid in path_by_id]
        if not runnable_ids:
            summary = self._rarespot_summary(items, total, job)
            summary["canceled"] = False
            self._apply_rarespot_terminal_status(summary)
            await self._report(progress, items, total, job, "RareSpot found no runnable inputs.", model="rarespot")
            return summary

        try:
            outputs, source_rows, counts_by_class, detections_count = await asyncio.to_thread(
                self._infer_and_stage_rarespot,
                job,
                runnable_ids,
                path_by_id,
                principal,
            )
            if await self._is_canceled(job, principal):
                raise _AnalysisJobCanceledError("RareSpot job was canceled.")
            if outputs:
                await asyncio.to_thread(
                    register_outputs,
                    control_base_url=settings.control_base_url,
                    job_id=job.job_id,
                    principal_headers=principal,
                    outputs=outputs,
                    timeout=settings.control_status_timeout_seconds,
                    collection_id=collection_id,
                )
            if await self._is_canceled(job, principal):
                raise _AnalysisJobCanceledError("RareSpot job was canceled.")
            output_count_by_source: dict[str, int] = {}
            for output in outputs:
                rid = str(output.get("source_resource_id") or "")
                if rid:
                    output_count_by_source[rid] = output_count_by_source.get(rid, 0) + 1
            for rid in runnable_ids:
                row = source_rows.get(rid, {})
                items[rid] = {
                    "status": "done",
                    "outputs": output_count_by_source.get(rid, 0),
                    "detections": int(row.get("detections") or 0),
                    "counts_by_class": dict(row.get("counts_by_class") or {}),
                }
            await self._report(
                progress,
                items,
                total,
                job,
                "RareSpot inference completed.",
                model="rarespot",
                counts_by_class=counts_by_class,
                detections_count=detections_count,
            )
        except _AnalysisJobCanceledError:
            canceled = True
        except Exception as exc:  # noqa: BLE001 - isolate the batch failure into per-item rows
            logger.exception("rarespot batch failed")
            for rid in runnable_ids:
                items[rid] = {"status": "failed", "error": str(exc)[:500]}

        summary = self._rarespot_summary(items, total, job)
        summary["canceled"] = canceled
        self._apply_rarespot_terminal_status(summary)
        return summary

    def _infer_and_stage_megaseg(
        self,
        job: DataAgentJobEnvelope,
        rid: str,
        path: Path,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        settings = self.settings
        safe_job_id = _safe_job_path_part(job.job_id, "job_id")
        safe_rid = _safe_job_path_part(rid, "resource_id")
        dest = Path(tempfile.mkdtemp(prefix="megaseg-extract-"))
        try:
            result = run_megaseg_infer(
                service_url=settings.megaseg_service_url,
                api_key=settings.megaseg_service_api_key,
                image_path=path,
                params=params,
                timeout=settings.megaseg_timeout_seconds,
                dest_dir=dest,
            )
            files = result.get("files") or []
            first = files[0] if isinstance(files, list) and files else {}
            model_version = Path(str(result.get("checkpoint_path") or "")).stem
            out_dir = settings.upload_root / settings.output_prefix / safe_job_id / safe_rid
            out_dir.mkdir(parents=True, exist_ok=True)

            outputs: list[dict[str, Any]] = []
            mask_rel = first.get("mask_path")
            if isinstance(mask_rel, str) and mask_rel:
                staged = self._stage(job, rid, dest, mask_rel, out_dir, "mask", "image/tiff", model_version)
                if staged is not None:
                    staged["metadata"]["segmentation"] = first.get("segmentation")
                    staged["metadata"]["intensity_context"] = first.get("intensity_context")
                    outputs.append(staged)
            summary_rel = first.get("summary_json_path")
            if isinstance(summary_rel, str) and summary_rel:
                staged = self._stage(job, rid, dest, summary_rel, out_dir, "metrics", "application/json", model_version)
                if staged is not None:
                    outputs.append(staged)
            return outputs
        finally:
            shutil.rmtree(dest, ignore_errors=True)

    def _infer_and_stage_rarespot(
        self,
        job: DataAgentJobEnvelope,
        runnable_ids: list[str],
        path_by_id: dict[str, Path],
        principal: dict[str, str],
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, int], int]:
        safe_job_id = _safe_job_path_part(job.job_id, "job_id")
        run_dir = Path(tempfile.mkdtemp(prefix="rarespot-analysis-"))
        try:
            progress_events = 0

            def progress_callback(payload: dict[str, Any]) -> None:
                nonlocal progress_events
                progress_events += 1
                if self._is_canceled_sync(job, principal):
                    raise _AnalysisJobCanceledError("RareSpot job was canceled.")

            image_paths = [path_by_id[rid] for rid in runnable_ids]
            result = run_rarespot_inference(
                image_paths=image_paths,
                run_id=job.job_id,
                thread_id=str((job.metadata or {}).get("thread_id") or job.job_id),
                output_dir=run_dir,
                config=self._rarespot_config(job),
                progress_callback=progress_callback,
            )
            self._raise_if_canceled_sync(job, principal)
            source_rows, artifact_source_ids, artifact_index_source_ids = self._rarespot_source_maps(
                result,
                path_by_id,
                runnable_ids,
            )
            stage_dir = self.settings.upload_root / self.settings.output_prefix / safe_job_id / "rarespot"
            outputs: list[dict[str, Any]] = []
            requested = set(runnable_ids)
            for index, artifact in enumerate(result.get("artifacts") or []):
                if not isinstance(artifact, dict):
                    continue
                source_rid = self._rarespot_artifact_source_id(
                    artifact,
                    run_dir,
                    artifact_source_ids,
                    artifact_index_source_ids,
                    requested,
                )
                staged = self._stage_rarespot_artifact(
                    job,
                    run_dir,
                    stage_dir,
                    artifact,
                    index,
                    source_rid,
                )
                if staged is not None:
                    outputs.append(staged)
            counts_by_class = {str(k): int(v) for k, v in (result.get("counts_by_class") or {}).items()}
            detections_count = sum(counts_by_class.values())
            return outputs, source_rows, counts_by_class, detections_count
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def _rarespot_config(self, job: DataAgentJobEnvelope) -> RareSpotConfig:
        config = RareSpotConfig.from_env()
        params = _job_params(job)
        updates: dict[str, Any] = {}
        if "tile_size" in params or "imgsz" in params:
            updates["tile_size"] = _int_param(params.get("tile_size", params.get("imgsz")), config.tile_size)
        if "tile_overlap" in params:
            updates["tile_overlap"] = _float_param(params.get("tile_overlap"), config.tile_overlap)
        if "conf" in params or "conf_threshold" in params:
            updates["conf"] = _float_param(params.get("conf", params.get("conf_threshold")), config.conf)
        if "iou" in params or "iou_threshold" in params:
            updates["iou"] = _float_param(params.get("iou", params.get("iou_threshold")), config.iou)
        if "spectral" in params:
            updates["spectral"] = _bool_param(params.get("spectral"), config.spectral)
        if "stability" in params:
            updates["stability"] = _bool_param(params.get("stability"), config.stability)
        if "stability_match_iou" in params:
            updates["stability_match_iou"] = _float_param(
                params.get("stability_match_iou"),
                config.stability_match_iou,
            )
        return replace(config, **updates) if updates else config

    def _rarespot_source_maps(
        self,
        result: dict[str, Any],
        path_by_id: dict[str, Path],
        runnable_ids: list[str],
    ) -> tuple[dict[str, dict[str, Any]], dict[Path, str], dict[int, str]]:
        source_by_path = {
            path.resolve(strict=False): rid
            for rid, path in path_by_id.items()
        }
        rows_by_id: dict[str, dict[str, Any]] = {}
        artifact_source_ids: dict[Path, str] = {}
        artifact_index_source_ids = {index: rid for index, rid in enumerate(runnable_ids)}
        for index, prediction in enumerate(result.get("predictions") or []):
            if not isinstance(prediction, dict):
                continue
            input_path = Path(str(prediction.get("input_path") or "")).expanduser().resolve(strict=False)
            rid = source_by_path.get(input_path)
            if not rid:
                continue
            artifact_index_source_ids[index] = rid
            counts = {
                str(k): int(v)
                for k, v in (prediction.get("class_counts") or {}).items()
            }
            detections = sum(counts.values())
            if not counts:
                detections = len(list(prediction.get("boxes") or []))
            rows_by_id[rid] = {
                "counts_by_class": counts,
                "detections": detections,
            }
            xml_path = str(prediction.get("prediction_xml_path") or "")
            if xml_path:
                artifact_source_ids[Path(xml_path).expanduser().resolve(strict=False)] = rid
        return rows_by_id, artifact_source_ids, artifact_index_source_ids

    def _rarespot_artifact_source_id(
        self,
        artifact: dict[str, Any],
        run_dir: Path,
        artifact_source_ids: dict[Path, str],
        artifact_index_source_ids: dict[int, str],
        requested: set[str],
    ) -> str:
        explicit = str(artifact.get("source_resource_id") or "").strip()
        if explicit in requested:
            return explicit
        source = self._rarespot_artifact_source_path(run_dir, artifact)
        if source is None:
            return ""
        exact = artifact_source_ids.get(source.resolve(strict=False), "")
        if exact:
            return exact
        return self._rarespot_artifact_index_source_id(
            source,
            run_dir,
            artifact_index_source_ids,
        )

    def _rarespot_artifact_index_source_id(
        self,
        source: Path,
        run_dir: Path,
        artifact_index_source_ids: dict[int, str],
    ) -> str:
        try:
            rel = source.resolve(strict=False).relative_to(run_dir.resolve())
        except (OSError, ValueError):
            return ""
        if len(rel.parts) < 2 or rel.parts[0] != "overlays":
            return ""
        prefix, separator, _rest = rel.name.partition("-")
        if not separator or not prefix.isdigit():
            return ""
        return artifact_index_source_ids.get(int(prefix), "")

    def _stage_rarespot_artifact(
        self,
        job: DataAgentJobEnvelope,
        run_dir: Path,
        stage_dir: Path,
        artifact: dict[str, Any],
        index: int,
        source_rid: str,
    ) -> dict[str, Any] | None:
        src = self._rarespot_artifact_source_path(run_dir, artifact)
        if src is None or not src.is_file():
            return None
        run_resolved = run_dir.resolve()
        try:
            rel = src.resolve().relative_to(run_resolved)
        except ValueError:
            return None
        dst = stage_dir.joinpath(*(_path_part(part) for part in rel.parts))
        dst.parent.mkdir(parents=True, exist_ok=True)
        size, sha = _copy_with_sha(src, dst)
        storage_path = dst.relative_to(self.settings.upload_root).as_posix()
        kind = str(artifact.get("kind") or artifact.get("category") or "artifact").strip() or "artifact"
        content_type = str(
            artifact.get("mime_type")
            or artifact.get("content_type")
            or "application/octet-stream"
        )
        metadata = {
            "title": str(artifact.get("title") or src.name),
            "category": str(artifact.get("category") or ""),
            "kind": kind,
            "mime_type": content_type,
            "content_type": content_type,
            "size_bytes": size,
            "sha256": sha,
        }
        output = {
            "resource_id": f"file_an_{_id_part(job.job_id)}_rarespot_{index:04d}_{_id_part(kind)}",
            "storage_path": storage_path,
            "original_name": src.name,
            "content_type": content_type,
            "size_bytes": size,
            "sha256": sha,
            "artifact_kind": kind,
            "add_to_collection": True,
            "metadata": metadata,
        }
        if source_rid:
            output["source_resource_id"] = source_rid
        return output

    def _rarespot_artifact_source_path(self, run_dir: Path, artifact: dict[str, Any]) -> Path | None:
        raw = str(artifact.get("path") or artifact.get("source_path") or "").strip()
        if not raw:
            return None
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = run_dir / candidate
        try:
            resolved = candidate.resolve()
            resolved.relative_to(run_dir.resolve())
        except (OSError, ValueError):
            return None
        return resolved

    def _stage(
        self,
        job: DataAgentJobEnvelope,
        rid: str,
        dest: Path,
        rel_path: str,
        out_dir: Path,
        artifact_kind: str,
        content_type: str,
        model_version: str,
    ) -> dict[str, Any] | None:
        src = (dest / rel_path).resolve()
        if not str(src).startswith(f"{dest.resolve()}/") or not src.is_file():
            return None
        base = Path(rel_path).name
        storage_filename = f"{artifact_kind}__{base}"
        dst = out_dir / storage_filename
        size, sha = _copy_with_sha(src, dst)
        storage_path = dst.relative_to(self.settings.upload_root).as_posix()
        return {
            "resource_id": f"file_an_{job.job_id}_{rid}_{artifact_kind}",
            "storage_path": storage_path,
            "original_name": base,
            "content_type": content_type,
            "size_bytes": size,
            "sha256": sha,
            "source_resource_id": rid,
            "artifact_kind": artifact_kind,
            "add_to_collection": True,
            "metadata": {"model_version": model_version},
        }

    def _prior_items(self, job: DataAgentJobEnvelope, principal: dict[str, str]) -> dict[str, dict[str, Any]]:
        record = fetch_job(
            control_base_url=self.settings.control_base_url,
            job_id=job.job_id,
            principal_headers=principal,
            timeout=self.settings.control_status_timeout_seconds,
        )
        summary = record.get("output_summary") if isinstance(record, dict) else None
        items = summary.get("items") if isinstance(summary, dict) else None
        if isinstance(items, dict):
            return {str(k): dict(v) for k, v in items.items() if isinstance(v, dict)}
        return {}

    async def _is_canceled(self, job: DataAgentJobEnvelope, principal: dict[str, str]) -> bool:
        record = await asyncio.to_thread(
            fetch_job,
            control_base_url=self.settings.control_base_url,
            job_id=job.job_id,
            principal_headers=principal,
            timeout=self.settings.control_status_timeout_seconds,
        )
        return str(record.get("status") or "").lower() == "canceled"

    def _is_canceled_sync(self, job: DataAgentJobEnvelope, principal: dict[str, str]) -> bool:
        record = fetch_job(
            control_base_url=self.settings.control_base_url,
            job_id=job.job_id,
            principal_headers=principal,
            timeout=self.settings.control_status_timeout_seconds,
        )
        return str(record.get("status") or "").lower() == "canceled"

    def _raise_if_canceled_sync(self, job: DataAgentJobEnvelope, principal: dict[str, str]) -> None:
        if self._is_canceled_sync(job, principal):
            raise _AnalysisJobCanceledError("RareSpot job was canceled.")

    async def _report(
        self,
        progress: Any,
        items: dict[str, dict[str, Any]],
        total: int,
        job: DataAgentJobEnvelope,
        message: str,
        *,
        model: str = "megaseg",
        counts_by_class: dict[str, int] | None = None,
        detections_count: int | None = None,
    ) -> None:
        completed = sum(1 for value in items.values() if value.get("status") == "done")
        await progress(
            progress_completed=completed,
            progress_total=total,
            message=message,
            output_summary=self._summary(
                items,
                total,
                job,
                model=model,
                counts_by_class=counts_by_class,
                detections_count=detections_count,
            ),
        )

    def _summary(
        self,
        items: dict[str, dict[str, Any]],
        total: int,
        job: DataAgentJobEnvelope,
        *,
        model: str = "megaseg",
        counts_by_class: dict[str, int] | None = None,
        detections_count: int | None = None,
    ) -> dict[str, Any]:
        done = sum(1 for value in items.values() if value.get("status") == "done")
        failed = sum(1 for value in items.values() if value.get("status") == "failed")
        display = "RareSpot" if model == "rarespot" else "MegaSeg"
        summary = {
            "summary_kind": "batch_inference",
            "model": model,
            "summary": f"{display} processed {done}/{total} images ({failed} failed).",
            "processed": done,
            "failed": failed,
            "total": total,
            "results_collection_id": (job.metadata or {}).get("results_collection_id"),
            "items": items,
        }
        if model == "rarespot":
            resolved_counts = counts_by_class if counts_by_class is not None else self._counts_by_class(items)
            resolved_detections = detections_count if detections_count is not None else sum(resolved_counts.values())
            summary["counts_by_class"] = resolved_counts
            summary["detections_count"] = int(resolved_detections)
        return summary

    def _rarespot_summary(
        self,
        items: dict[str, dict[str, Any]],
        total: int,
        job: DataAgentJobEnvelope,
    ) -> dict[str, Any]:
        return self._summary(items, total, job, model="rarespot")

    def _apply_rarespot_terminal_status(self, summary: dict[str, Any]) -> None:
        if summary.get("canceled") is True:
            summary["terminal_status"] = "canceled"
            return
        processed = int(summary.get("processed") or 0)
        failed = int(summary.get("failed") or 0)
        if processed == 0 and failed > 0:
            summary["terminal_status"] = "failed"

    def _counts_by_class(self, items: dict[str, dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in items.values():
            for class_name, count in (item.get("counts_by_class") or {}).items():
                counts[str(class_name)] = counts.get(str(class_name), 0) + int(count)
        return counts
