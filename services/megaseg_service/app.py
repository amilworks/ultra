"""Private FastAPI service for Megaseg GPU inference."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tarfile
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("megaseg_service")


def build_megaseg_model(*args: Any, **kwargs: Any) -> Any:
    from src.science.megaseg_core import build_megaseg_model as _build_megaseg_model

    return _build_megaseg_model(*args, **kwargs)


def resolve_megaseg_device(*args: Any, **kwargs: Any) -> Any:
    requested = str((args[0] if args else kwargs.get("requested")) or "").strip().lower()
    if requested and requested not in {"auto", "default"}:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_megaseg_batch(*args: Any, **kwargs: Any) -> Any:
    from src.science.megaseg_core import run_megaseg_batch as _run_megaseg_batch

    return _run_megaseg_batch(*args, **kwargs)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_event(event: str, **payload: Any) -> None:
    safe_payload = {"event": event, **payload}
    logger.info(json.dumps(safe_payload, sort_keys=True, default=str))


@dataclass(slots=True)
class ServiceSettings:
    checkpoint_path: Path
    device: str
    job_root: Path
    artifact_root: Path
    api_key: str
    max_concurrent_jobs: int = 1
    amp_enabled: bool = False

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        checkpoint_raw = str(os.getenv("MEGASEG_CHECKPOINT_PATH") or "").strip()
        api_key = str(os.getenv("MEGASEG_API_KEY") or "").strip()
        if not checkpoint_raw:
            raise RuntimeError("MEGASEG_CHECKPOINT_PATH is required for the Megaseg service.")
        if not api_key:
            raise RuntimeError("MEGASEG_API_KEY is required for the Megaseg service.")
        job_root_raw = str(os.getenv("MEGASEG_JOB_ROOT") or "/srv/ultra/megaseg-service/jobs").strip()
        artifact_root_raw = str(
            os.getenv("MEGASEG_ARTIFACT_ROOT") or "/srv/ultra/megaseg-service/artifacts"
        ).strip()
        device = str(os.getenv("MEGASEG_DEVICE") or "cuda").strip() or "cuda"
        max_concurrent_jobs_raw = str(os.getenv("MEGASEG_MAX_CONCURRENT_JOBS") or "1").strip()
        amp_raw = str(os.getenv("MEGASEG_ENABLE_AMP") or "").strip().lower()
        try:
            max_concurrent_jobs = max(1, int(max_concurrent_jobs_raw))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("MEGASEG_MAX_CONCURRENT_JOBS must be an integer.") from exc
        return cls(
            checkpoint_path=Path(checkpoint_raw).expanduser().resolve(),
            device=device,
            job_root=Path(job_root_raw).expanduser().resolve(),
            artifact_root=Path(artifact_root_raw).expanduser().resolve(),
            api_key=api_key,
            max_concurrent_jobs=max_concurrent_jobs,
            amp_enabled=amp_raw in {"1", "true", "yes", "on"},
        )


class SourceSpec(BaseModel):
    uri: str


class JobRequest(BaseModel):
    sources: list[SourceSpec] = Field(default_factory=list)
    structure_channel: int = 4
    nucleus_channel: int | None = 6
    channel_index_base: int = 1
    mask_threshold: float = 0.5
    save_visualizations: bool = True
    generate_report: bool = True
    device: str | None = None
    checkpoint_path: str | None = None
    structure_name: str | None = None
    amp_enabled: bool | None = None


class ArtifactEntry(BaseModel):
    name: str
    size_bytes: int


class JobRecord(BaseModel):
    job_id: str
    status: str
    created_at: str
    updated_at: str
    request: JobRequest
    resolved_sources: list[str] = Field(default_factory=list)
    uploaded_files: list[str] = Field(default_factory=list)
    artifact_manifest: list[ArtifactEntry] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class ServiceRuntime:
    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.record_lock = threading.Lock()
        self.model: Any | None = None
        self.model_device: str | None = None
        self.model_checkpoint_path: Path | None = None
        self.health_error: str | None = None
        self.worker_task: asyncio.Task[Any] | None = None


def _job_record_path(settings: ServiceSettings, job_id: str) -> Path:
    return settings.job_root / f"{job_id}.json"


def _job_root_dir(settings: ServiceSettings, job_id: str) -> Path:
    return settings.artifact_root / job_id


def _job_inputs_dir(settings: ServiceSettings, job_id: str) -> Path:
    return _job_root_dir(settings, job_id) / "inputs"


def _job_results_dir(settings: ServiceSettings, job_id: str) -> Path:
    return _job_root_dir(settings, job_id) / "results"


def _load_job_record(runtime: ServiceRuntime, job_id: str) -> JobRecord:
    path = _job_record_path(runtime.settings, job_id)
    if not path.exists():
        raise FileNotFoundError(job_id)
    with runtime.record_lock:
        return JobRecord.model_validate_json(path.read_text(encoding="utf-8"))


def _save_job_record(runtime: ServiceRuntime, record: JobRecord) -> JobRecord:
    record.updated_at = _utc_now()
    path = _job_record_path(runtime.settings, record.job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with runtime.record_lock:
        path.write_text(
            json.dumps(record.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return record


def _safe_upload_name(name: str, existing: set[str]) -> str:
    candidate = Path(str(name or "upload.bin")).name or "upload.bin"
    if candidate not in existing:
        existing.add(candidate)
        return candidate
    stem = Path(candidate).stem or "upload"
    suffix = Path(candidate).suffix
    index = 2
    while True:
        updated = f"{stem}_{index}{suffix}"
        if updated not in existing:
            existing.add(updated)
            return updated
        index += 1


def _collect_artifact_manifest(results_dir: Path) -> list[ArtifactEntry]:
    manifest: list[ArtifactEntry] = []
    if not results_dir.exists():
        return manifest
    for path in sorted(results_dir.rglob("*")):
        if not path.is_file():
            continue
        manifest.append(
            ArtifactEntry(
                name=path.relative_to(results_dir).as_posix(),
                size_bytes=int(path.stat().st_size),
            )
        )
    return manifest


def _extract_uploaded_archive(upload_path: Path, inputs_dir: Path) -> Path:
    archive_name = upload_path.name
    original_name = archive_name[:-7] if archive_name.endswith(".tar.gz") else upload_path.stem
    extracted_root = inputs_dir / original_name
    with tarfile.open(upload_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            member_path = (inputs_dir / member.name).resolve()
            if not str(member_path).startswith(f"{inputs_dir.resolve()}{os.sep}") and member_path != inputs_dir.resolve():
                raise RuntimeError(f"Refusing to extract unsafe archive member: {member.name}")
        archive.extractall(inputs_dir)
    if not extracted_root.exists():
        raise RuntimeError(f"Uploaded archive did not contain expected root directory: {original_name}")
    return extracted_root


def _materialize_uploaded_source(upload_path: Path, inputs_dir: Path) -> Path:
    name = upload_path.name.lower()
    if name.endswith(".ome.zarr.tar.gz") or name.endswith(".zarr.tar.gz"):
        return _extract_uploaded_archive(upload_path, inputs_dir)
    return upload_path


def _mark_stale_jobs(runtime: ServiceRuntime) -> list[str]:
    queued: list[str] = []
    runtime.settings.job_root.mkdir(parents=True, exist_ok=True)
    for record_path in sorted(runtime.settings.job_root.glob("*.json")):
        try:
            record = JobRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            _log_event("job_record_invalid", path=str(record_path), error=str(exc))
            continue
        if record.status == "queued":
            queued.append(record.job_id)
            continue
        if record.status != "running":
            continue
        record.status = "failed"
        record.finished_at = _utc_now()
        record.error = "Megaseg service restarted while the job was running."
        _save_job_record(runtime, record)
    return queued


def _resolve_request_checkpoint(runtime: ServiceRuntime, request: JobRequest) -> Path:
    raw = str(request.checkpoint_path or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return runtime.settings.checkpoint_path


def _resolve_request_device(runtime: ServiceRuntime, request: JobRequest) -> str:
    return str(request.device or runtime.settings.device or "cuda").strip() or "cuda"


def _get_model_for_request(runtime: ServiceRuntime, request: JobRequest) -> tuple[Any, Path, str]:
    checkpoint_path = _resolve_request_checkpoint(runtime, request)
    device_name = _resolve_request_device(runtime, request)
    if (
        runtime.model is not None
        and runtime.model_checkpoint_path == checkpoint_path
        and str(runtime.model_device or "") == str(resolve_megaseg_device(device_name))
    ):
        return runtime.model, checkpoint_path, device_name
    device = resolve_megaseg_device(device_name)
    if device_name.lower().startswith("cuda") and device.type != "cuda":
        raise RuntimeError("Megaseg service is configured for CUDA but no CUDA device is available.")
    model = build_megaseg_model(checkpoint_path=checkpoint_path, device=device)
    return model, checkpoint_path, device_name


def _run_job_sync(runtime: ServiceRuntime, job_id: str) -> None:
    record = _load_job_record(runtime, job_id)
    if record.status != "queued":
        return
    record.status = "running"
    record.started_at = _utc_now()
    record.error = None
    _save_job_record(runtime, record)
    _log_event("job_started", job_id=job_id, source_count=len(record.resolved_sources))

    try:
        if runtime.health_error:
            raise RuntimeError(runtime.health_error)
        model, checkpoint_path, device_name = _get_model_for_request(runtime, record.request)
        result = run_megaseg_batch(
            file_paths=list(record.resolved_sources),
            output_dir=_job_results_dir(runtime.settings, job_id),
            checkpoint_path=checkpoint_path,
            structure_channel=int(record.request.structure_channel),
            nucleus_channel=(
                int(record.request.nucleus_channel)
                if record.request.nucleus_channel is not None
                else None
            ),
            channel_index_base=int(record.request.channel_index_base),
            mask_threshold=float(record.request.mask_threshold),
            save_visualizations=bool(record.request.save_visualizations),
            generate_report=bool(record.request.generate_report),
            device=device_name,
            structure_name=str(record.request.structure_name or "structure"),
            model=model,
            amp_enabled=(
                bool(record.request.amp_enabled)
                if record.request.amp_enabled is not None
                else bool(runtime.settings.amp_enabled)
            ),
        )
        record.result = result
        record.artifact_manifest = _collect_artifact_manifest(_job_results_dir(runtime.settings, job_id))
        record.status = "succeeded" if bool(result.get("success")) else "failed"
        record.error = None if record.status == "succeeded" else str(
            result.get("error") or "Megaseg inference completed without successful outputs."
        )
        record.finished_at = _utc_now()
        _save_job_record(runtime, record)
        _log_event(
            "job_finished",
            job_id=job_id,
            status=record.status,
            artifacts=len(record.artifact_manifest),
        )
    except Exception as exc:  # noqa: BLE001
        record.status = "failed"
        record.finished_at = _utc_now()
        record.error = str(exc)
        record.result = {"success": False, "error": str(exc)}
        record.artifact_manifest = _collect_artifact_manifest(_job_results_dir(runtime.settings, job_id))
        _save_job_record(runtime, record)
        logger.exception("Megaseg job %s failed", job_id)


async def _worker_loop(app: FastAPI) -> None:
    runtime: ServiceRuntime = app.state.megaseg_runtime
    while True:
        job_id = await runtime.queue.get()
        try:
            await asyncio.to_thread(_run_job_sync, runtime, job_id)
        finally:
            runtime.queue.task_done()


def _require_auth(request: Request, runtime: ServiceRuntime) -> None:
    header = str(request.headers.get("Authorization") or "").strip()
    expected = f"Bearer {runtime.settings.api_key}"
    if not header or header != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Megaseg service bearer token.",
        )


async def _parse_job_request(runtime: ServiceRuntime, request: Request) -> JobRecord:
    content_type = str(request.headers.get("content-type") or "").lower()
    created_at = _utc_now()
    job_id = uuid4().hex
    uploaded_files: list[str] = []
    resolved_sources: list[str] = []

    if "multipart/form-data" in content_type:
        form = await request.form()
        request_json = form.get("request_json")
        if request_json is None:
            raise HTTPException(status_code=400, detail="request_json is required for multipart jobs.")
        try:
            payload = JobRequest.model_validate_json(str(request_json))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Invalid request_json: {exc}") from exc
        inputs_dir = _job_inputs_dir(runtime.settings, job_id)
        inputs_dir.mkdir(parents=True, exist_ok=True)
        existing_names: set[str] = set()
        for _, value in form.multi_items():
            filename = getattr(value, "filename", None)
            file_handle = getattr(value, "file", None)
            if not filename or file_handle is None:
                continue
            safe_name = _safe_upload_name(str(filename), existing_names)
            destination = inputs_dir / safe_name
            with destination.open("wb") as handle:
                shutil.copyfileobj(file_handle, handle)
            materialized_source = _materialize_uploaded_source(destination, inputs_dir)
            uploaded_files.append(str(materialized_source))
            resolved_sources.append(str(materialized_source))
    else:
        try:
            payload = JobRequest.model_validate(await request.json())
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc

    for source in list(payload.sources or []):
        uri = str(source.uri or "").strip()
        if uri:
            resolved_sources.append(uri)

    if not resolved_sources:
        raise HTTPException(status_code=400, detail="At least one source URI or uploaded file is required.")

    record = JobRecord(
        job_id=job_id,
        status="queued",
        created_at=created_at,
        updated_at=created_at,
        request=payload,
        resolved_sources=resolved_sources,
        uploaded_files=uploaded_files,
    )
    return _save_job_record(runtime, record)


def create_app(settings: ServiceSettings | None = None) -> FastAPI:
    resolved_settings = settings or ServiceSettings.from_env()
    runtime = ServiceRuntime(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        runtime.settings.job_root.mkdir(parents=True, exist_ok=True)
        runtime.settings.artifact_root.mkdir(parents=True, exist_ok=True)
        stale_queued = _mark_stale_jobs(runtime)
        try:
            device = resolve_megaseg_device(runtime.settings.device)
            if runtime.settings.device.lower().startswith("cuda") and device.type != "cuda":
                raise RuntimeError("Megaseg service is configured for CUDA but no CUDA device is available.")
            runtime.model = build_megaseg_model(
                checkpoint_path=runtime.settings.checkpoint_path,
                device=device,
            )
            runtime.model_device = str(device)
            runtime.model_checkpoint_path = runtime.settings.checkpoint_path
            runtime.health_error = None
            _log_event(
                "model_ready",
                checkpoint_path=str(runtime.settings.checkpoint_path),
                device=str(device),
            )
        except Exception as exc:  # noqa: BLE001
            runtime.health_error = str(exc)
            logger.exception("Megaseg service startup failed")
        runtime.worker_task = asyncio.create_task(_worker_loop(app))
        for job_id in stale_queued:
            await runtime.queue.put(job_id)
        try:
            yield
        finally:
            if runtime.worker_task is not None:
                runtime.worker_task.cancel()
                try:
                    await runtime.worker_task
                except asyncio.CancelledError:
                    pass

    app = FastAPI(title="Megaseg Service", version="0.1.0", lifespan=lifespan)
    app.state.megaseg_runtime = runtime

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        _require_auth(request, runtime)
        payload = {
            "ok": runtime.health_error is None,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": runtime.model_device or runtime.settings.device,
            "checkpoint_path": str(runtime.settings.checkpoint_path),
            "model_ready": runtime.model is not None and runtime.health_error is None,
            "queue_depth": int(runtime.queue.qsize()),
            "max_concurrent_jobs": int(runtime.settings.max_concurrent_jobs),
        }
        if runtime.health_error:
            payload["error"] = runtime.health_error
            return JSONResponse(status_code=503, content=payload)
        return JSONResponse(status_code=200, content=payload)

    @app.post("/v1/jobs", status_code=202)
    async def create_job(request: Request) -> dict[str, Any]:
        _require_auth(request, runtime)
        record = await _parse_job_request(runtime, request)
        await runtime.queue.put(record.job_id)
        _log_event("job_queued", job_id=record.job_id, source_count=len(record.resolved_sources))
        return {
            "job_id": record.job_id,
            "status": record.status,
            "created_at": record.created_at,
        }

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str, request: Request) -> dict[str, Any]:
        _require_auth(request, runtime)
        try:
            record = _load_job_record(runtime, job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Megaseg job not found.") from exc
        return record.model_dump(mode="json")

    @app.get("/v1/jobs/{job_id}/artifacts/{artifact_name:path}")
    async def download_artifact(job_id: str, artifact_name: str, request: Request) -> FileResponse:
        _require_auth(request, runtime)
        try:
            _load_job_record(runtime, job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Megaseg job not found.") from exc
        results_dir = _job_results_dir(runtime.settings, job_id).resolve()
        candidate = (results_dir / artifact_name).resolve()
        if not str(candidate).startswith(f"{results_dir}{os.sep}") and candidate != results_dir:
            raise HTTPException(status_code=400, detail="Invalid artifact path.")
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(status_code=404, detail="Megaseg artifact not found.")
        return FileResponse(candidate)

    return app


try:
    app = create_app()
except Exception as exc:  # noqa: BLE001
    logger.exception("Megaseg service failed to initialize default app")
    app = FastAPI(title="Megaseg Service", version="0.1.0")

    @app.get("/health")
    async def health_fallback() -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "model_ready": False,
                "error": str(exc),
            },
        )
