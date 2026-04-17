"""Private FastAPI service for asynchronous code execution."""

from __future__ import annotations

import asyncio
import json
import shutil
import tarfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse

from services.codeexec_service.models import ServiceSettings
from services.codeexec_service.runner import run_codeexec_attempt
from src.tooling.code_execution_contract import (
    CodeExecutionArtifact,
    CodeExecutionAttemptResult,
    CodeExecutionJobRecord,
    CodeExecutionJobRequest,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_app(settings: ServiceSettings) -> FastAPI:
    queue: asyncio.Queue[str] = asyncio.Queue()
    records: dict[str, CodeExecutionJobRecord] = {}

    async def _worker() -> None:
        while True:
            job_id = await queue.get()
            record = records[job_id]
            record.status = "running"
            record.started_at = _utc_now()
            record.updated_at = _utc_now()
            work_dir = settings.artifact_root / job_id / "workdir"
            try:
                result_payload = run_codeexec_attempt(
                    job_id=job_id,
                    work_dir=work_dir,
                    request=record.request.model_dump(mode="json"),
                    worker_image=settings.worker_image,
                    docker_network=settings.docker_network,
                )
                result = CodeExecutionAttemptResult.model_validate(result_payload)
                record.result = result
                record.status = "succeeded" if result.success else "failed"
                record.artifact_manifest = [
                    CodeExecutionArtifact(
                        name=str(item.get("relative_path") or item.get("title") or item.get("path")),
                        relative_path=str(item.get("relative_path") or item.get("path") or ""),
                        size_bytes=int(item.get("size_bytes") or 0),
                        kind="file",
                    )
                    for item in list(result.artifacts or [])
                ]
                record.error = result.error_message
            except Exception as exc:  # noqa: BLE001
                record.status = "failed"
                record.error = str(exc)
            record.finished_at = _utc_now()
            record.updated_at = _utc_now()
            queue.task_done()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        worker_task = asyncio.create_task(_worker())
        try:
            yield
        finally:
            worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker_task

    import contextlib

    app = FastAPI(lifespan=lifespan)

    def _require_auth(request: Request) -> None:
        auth = str(request.headers.get("Authorization") or "").strip()
        if auth != f"Bearer {settings.api_key}":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    @app.get("/health")
    async def health(request: Request) -> dict[str, object]:
        _require_auth(request)
        return {"ok": True, "queue_depth": queue.qsize(), "worker_image": settings.worker_image}

    @app.post("/v1/jobs", status_code=status.HTTP_202_ACCEPTED)
    async def submit_job(
        request: Request,
        request_json: str = Form(),
        bundle: UploadFile = File(),
    ) -> dict[str, str]:
        _require_auth(request)
        request_payload = CodeExecutionJobRequest.model_validate_json(request_json)
        requested_job_id = str(request_payload.job_id or "").strip() or f"job_{uuid4().hex[:12]}"
        job_root = settings.artifact_root / requested_job_id
        work_dir = job_root / "workdir"
        bundle_path = job_root / "job.tar.gz"

        shutil.rmtree(job_root, ignore_errors=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        bundle_bytes = await bundle.read()
        bundle_path.write_bytes(bundle_bytes)
        with tarfile.open(bundle_path, "r:gz") as archive:
            archive.extractall(work_dir)

        record = CodeExecutionJobRecord(
            job_id=requested_job_id,
            status="queued",
            created_at=_utc_now(),
            updated_at=_utc_now(),
            request=request_payload.model_copy(update={"job_id": requested_job_id}),
            artifact_manifest=[],
            result=None,
            error=None,
            started_at=None,
            finished_at=None,
        )
        records[requested_job_id] = record
        await queue.put(requested_job_id)
        return {"job_id": requested_job_id}

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str, request: Request) -> dict[str, Any]:
        _require_auth(request)
        record = records.get(job_id)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown job_id")
        return record.model_dump(mode="json")

    @app.get("/v1/jobs/{job_id}/artifacts/{artifact_path:path}")
    async def get_artifact(job_id: str, artifact_path: str, request: Request):
        _require_auth(request)
        path = settings.artifact_root / job_id / "workdir" / artifact_path
        if not path.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
        return FileResponse(path)

    return app

