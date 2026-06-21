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
from pathlib import Path
from typing import Any

from ..data_agent.worker import DataAgentJobEnvelope, DefaultDataAgentProcessor
from ..rarespot.uploads import SAFE_FILE_ID_RE, unique_file_ids
from .client import fetch_job, register_outputs, run_megaseg_infer
from .config import AnalysisSettings

logger = logging.getLogger("ultra.analysis.worker")

_CANCEL_POLL_EVERY = 5
# Derived viewer artifacts live alongside the source in the upload store; the model needs
# the ORIGINAL volume, never the pyramid.
_DERIVED_MARKERS = ("__pyramid", "__thumbnail")


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
            raise NotImplementedError("analysis.rarespot is not yet enabled on this worker")
        return await self._fallback(job, progress)

    async def _process_megaseg(self, job: DataAgentJobEnvelope, progress: Any) -> dict[str, Any]:
        settings = self.settings
        if not settings.megaseg_service_url or not settings.megaseg_service_api_key:
            raise RuntimeError("MEGASEG_SERVICE_URL and MEGASEG_SERVICE_API_KEY are required for analysis.megaseg")
        principal = job.principal_headers()
        params = _megaseg_params(job)
        # The envelope carries the create-time metadata, so we always have the results
        # collection here even if the job's stored metadata changes during processing.
        collection_id = str((job.metadata or {}).get("results_collection_id") or "")
        requested = unique_file_ids(job.resource_ids)
        total = len(requested)

        items: dict[str, dict[str, Any]] = self._prior_items(job, principal)

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

    def _infer_and_stage_megaseg(
        self,
        job: DataAgentJobEnvelope,
        rid: str,
        path: Path,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        settings = self.settings
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
            out_dir = settings.upload_root / settings.output_prefix / job.job_id / rid
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

    async def _report(self, progress: Any, items: dict[str, dict[str, Any]], total: int, job: DataAgentJobEnvelope, message: str) -> None:
        completed = sum(1 for value in items.values() if value.get("status") == "done")
        await progress(
            progress_completed=completed,
            progress_total=total,
            message=message,
            output_summary=self._summary(items, total, job),
        )

    def _summary(self, items: dict[str, dict[str, Any]], total: int, job: DataAgentJobEnvelope) -> dict[str, Any]:
        done = sum(1 for value in items.values() if value.get("status") == "done")
        failed = sum(1 for value in items.values() if value.get("status") == "failed")
        return {
            "summary_kind": "batch_inference",
            "model": "megaseg",
            "summary": f"MegaSeg processed {done}/{total} images ({failed} failed).",
            "processed": done,
            "failed": failed,
            "total": total,
            "results_collection_id": (job.metadata or {}).get("results_collection_id"),
            "items": items,
        }
