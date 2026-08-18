"""Private FastAPI GPU compute service — model-agnostic.

A single, small framework: Bearer auth, a job manager (async submit / progress /
cancel / restart-recovery), and an executor registry. Every model-specific thing
(training, evaluation, inference) is an :class:`Executor`; adding one is one file
plus a registry line, never a change here. Mirrors the megaseg service's operational
shape (import-safe module-level ``app``, health-gated lifespan, per-GPU semaphore,
work off the event loop) so it deploys with the same systemd/Docker machinery.

Endpoints (all Bearer-authed):
    GET  /health                    liveness + loaded executors + load errors
    GET  /v1/executors              the registered model_key.capability manifests
    POST /v1/jobs                   {executor, spec, idempotency_key?} -> {job_id}
    GET  /v1/jobs/{id}              full record: status, progress, result, error
    POST /v1/jobs/{id}/cancel       idempotent; kills the GPU process group
    GET  /v1/jobs                   recent job records (debug/ops)
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import executors as executor_registry
from .jobs import (
    IDEMPOTENCY_CONFLICT_DETAIL,
    INVALID_JOB_SPEC_DETAIL,
    IdempotencyConflictError,
    InvalidJobSpecError,
    JobManager,
)
from .settings import ServiceSettings

logger = logging.getLogger("ultra.compute")


class SubmitJobRequest(BaseModel):
    executor: str = Field(..., description="model_key.capability, e.g. rarespot.train")
    spec: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = Field(default=None, min_length=1)


class ComputeRuntime:
    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self.manager = JobManager(
            job_root=settings.job_root,
            workspace_root=settings.workspace_root,
            max_concurrent=settings.max_concurrent,
            history_limit=settings.history_limit,
        )
        self.startup_error: str | None = None


def _require_auth(request: Request, runtime: ComputeRuntime) -> None:
    header = str(request.headers.get("Authorization") or "").strip()
    expected = f"Bearer {runtime.settings.api_key}"
    if not header or not hmac.compare_digest(header, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid compute service bearer token.",
        )


def create_app(settings: ServiceSettings | None = None) -> FastAPI:
    resolved = settings or ServiceSettings.from_env()
    runtime = ComputeRuntime(resolved)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
        try:
            executor_registry.load_builtin_executors(only=resolved.executors)
            if not executor_registry.list_manifests():
                raise RuntimeError(
                    "no executors loaded; load errors: "
                    + str(executor_registry.load_errors())
                )
            runtime.manager.start(asyncio.get_running_loop())
            runtime.startup_error = None
            logger.info(
                "compute service ready: executors=%s errors=%s",
                [m.key for m in executor_registry.list_manifests()],
                executor_registry.load_errors(),
            )
        except Exception as exc:  # noqa: BLE001
            runtime.startup_error = str(exc)
            logger.exception("compute service startup failed")
        try:
            yield
        finally:
            await runtime.manager.shutdown()

    app = FastAPI(title="Ultra Compute Service", version="0.1.0", lifespan=lifespan)
    app.state.compute_runtime = runtime

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        _require_auth(request, runtime)
        manifests = executor_registry.list_manifests()
        load_errors = executor_registry.load_errors()
        ready = runtime.startup_error is None and bool(manifests)
        payload = {
            "ok": ready,
            "executors": [m.key for m in manifests],
            "load_errors": load_errors,
            "max_concurrent": runtime.settings.max_concurrent,
        }
        if runtime.startup_error:
            payload["error"] = runtime.startup_error
        return JSONResponse(status_code=200 if ready else 503, content=payload)

    @app.get("/v1/executors")
    async def list_executors(request: Request) -> dict[str, Any]:
        _require_auth(request, runtime)
        return {
            "executors": [
                {
                    "key": m.key,
                    "model_key": m.model_key,
                    "capability": m.capability,
                    "description": m.description,
                    "gpu": m.gpu,
                    "resource": m.resource,
                }
                for m in executor_registry.list_manifests()
            ],
            "load_errors": executor_registry.load_errors(),
        }

    @app.post(
        "/v1/jobs",
        status_code=202,
        responses={
            409: {"description": "Idempotency key conflicts with retained request identity."}
        },
    )
    async def submit_job(request: Request, body: SubmitJobRequest) -> dict[str, Any]:
        _require_auth(request, runtime)
        if runtime.startup_error is not None:
            raise HTTPException(status_code=503, detail=f"service not ready: {runtime.startup_error}")
        if executor_registry.get_executor(body.executor) is None:
            raise HTTPException(
                status_code=400,
                detail=f"unknown executor {body.executor!r}; known: "
                + ", ".join(m.key for m in executor_registry.list_manifests()),
            )
        try:
            record = await asyncio.to_thread(
                runtime.manager.submit,
                executor_key=body.executor,
                spec=body.spec,
                idempotency_key=body.idempotency_key,
            )
        except IdempotencyConflictError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=IDEMPOTENCY_CONFLICT_DETAIL,
            ) from None
        except InvalidJobSpecError:
            raise HTTPException(status_code=422, detail=INVALID_JOB_SPEC_DETAIL) from None
        return {"job_id": record.job_id, "status": record.status, "created_at": record.created_at}

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str, request: Request) -> dict[str, Any]:
        _require_auth(request, runtime)
        record = runtime.manager.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="job not found")
        return record.model_dump(mode="json")

    @app.post("/v1/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str, request: Request) -> dict[str, Any]:
        _require_auth(request, runtime)
        record = await asyncio.to_thread(runtime.manager.cancel, job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="job not found")
        return {"job_id": record.job_id, "status": record.status}

    @app.get("/v1/jobs")
    async def list_jobs(request: Request, limit: int = 50) -> dict[str, Any]:
        _require_auth(request, runtime)
        return {"jobs": [r.model_dump(mode="json") for r in runtime.manager.list_jobs(limit=limit)]}

    return app


try:
    app = create_app()
except Exception as exc:  # noqa: BLE001 - import-safe: a bad env must still serve 503 /health
    logger.exception("compute service failed to initialize default app")
    _init_error = str(exc)

    app = FastAPI(title="Ultra Compute Service", version="0.1.0")

    @app.get("/health")
    async def health_fallback() -> JSONResponse:
        # This degraded endpoint is unauthenticated (no settings loaded), so it must
        # NOT echo the raw init error (it can name env vars / paths). The detail is
        # in the logs above.
        return JSONResponse(
            status_code=503,
            content={"ok": False, "error": "compute service failed to initialize; see logs"},
        )
