"""Worker-side client for the GPU compute service (services/compute_service).

The training worker runs on a CPU node and offloads GPU work (finetune, evaluate)
to the compute service. This is the thin HTTP client the RareSpot adapter uses in
place of the old ssh+rsync+docker recipe.

Two properties matter for the user's "launch back to back to back with no errors":
- **Idempotency**: ``submit_and_wait`` keys the job on ``idempotency_key`` (the
  training job id + phase), so a NATS redelivery after a worker restart RESUMES the
  already-running service job instead of launching a duplicate — the service returns
  the existing record and we keep polling it to completion.
- **Cancellation**: if the worker's run is cancelled mid-poll, we POST /cancel so the
  service kills the GPU process group (the ssh path orphaned it).

Configuration (worker env; deliberately UNSET on the service node so the adapter's
routing does not recurse):
    ULTRA_COMPUTE_SERVICE_URL     base url, e.g. http://tesla.internal:8030
    ULTRA_COMPUTE_SERVICE_TOKEN   bearer token (matches the service's ULTRA_COMPUTE_API_KEY)
    ULTRA_COMPUTE_POLL_INTERVAL   seconds between status polls (default 10)
    ULTRA_COMPUTE_REQUEST_TIMEOUT seconds per HTTP call (default 30)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


class ComputeServiceError(RuntimeError):
    """A compute-service job failed, or the service was unreachable/unauthorized."""


class ComputeJobCancelled(RuntimeError):
    """The caller asked to cancel; the service job was cancelled."""


@dataclass(frozen=True)
class ComputeServiceConfig:
    base_url: str
    token: str
    poll_interval: float = 10.0
    request_timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "ComputeServiceConfig | None":
        base_url = str(os.getenv("ULTRA_COMPUTE_SERVICE_URL") or "").strip().rstrip("/")
        if not base_url:
            return None  # not configured => adapter uses its legacy/local path
        token = str(os.getenv("ULTRA_COMPUTE_SERVICE_TOKEN") or "").strip()

        def _float(name: str, default: float) -> float:
            raw = str(os.getenv(name) or "").strip()
            try:
                return float(raw) if raw else default
            except ValueError:
                return default

        return cls(
            base_url=base_url,
            token=token,
            poll_interval=max(1.0, _float("ULTRA_COMPUTE_POLL_INTERVAL", 10.0)),
            request_timeout=max(5.0, _float("ULTRA_COMPUTE_REQUEST_TIMEOUT", 30.0)),
        )


class ComputeServiceClient:
    def __init__(self, config: ComputeServiceConfig) -> None:
        self._config = config
        self._headers = {"Authorization": f"Bearer {config.token}"} if config.token else {}

    def _url(self, path: str) -> str:
        return f"{self._config.base_url}{path}"

    def submit(self, *, executor: str, spec: dict[str, Any], idempotency_key: str | None = None) -> str:
        body: dict[str, Any] = {"executor": executor, "spec": spec}
        if idempotency_key:
            body["idempotency_key"] = idempotency_key
        try:
            resp = httpx.post(
                self._url("/v1/jobs"),
                json=body,
                headers=self._headers,
                timeout=self._config.request_timeout,
            )
        except httpx.HTTPError as exc:
            raise ComputeServiceError(f"compute service unreachable: {exc}") from exc
        if resp.status_code != 202:
            raise ComputeServiceError(
                f"compute service rejected {executor} job ({resp.status_code}): {resp.text[:400]}"
            )
        return str(resp.json()["job_id"])

    def get(self, job_id: str) -> dict[str, Any]:
        try:
            resp = httpx.get(
                self._url(f"/v1/jobs/{job_id}"),
                headers=self._headers,
                timeout=self._config.request_timeout,
            )
        except httpx.HTTPError as exc:
            raise ComputeServiceError(f"compute service get({job_id}) unreachable: {exc}") from exc
        if resp.status_code != 200:
            raise ComputeServiceError(f"compute service get({job_id}) failed ({resp.status_code}): {resp.text[:200]}")
        return resp.json()

    def cancel(self, job_id: str) -> None:
        try:
            httpx.post(self._url(f"/v1/jobs/{job_id}/cancel"), headers=self._headers, timeout=self._config.request_timeout)
        except httpx.HTTPError as exc:  # best-effort; the run is ending anyway
            logger.warning("compute cancel(%s) failed: %s", job_id, exc)

    def submit_and_wait(
        self,
        *,
        executor: str,
        spec: dict[str, Any],
        idempotency_key: str | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Submit (idempotently), poll to terminal, forward progress, honour cancel.

        Returns the job's ``result`` dict on success. Raises
        :class:`ComputeServiceError` on failure and :class:`ComputeJobCancelled` if
        cancelled. Transient poll errors are tolerated a few times so a blip in the
        network does not fail a multi-hour training job.
        """
        job_id = self.submit(executor=executor, spec=spec, idempotency_key=idempotency_key)
        logger.info("compute job %s submitted (executor=%s)", job_id, executor)
        last_progress_sig: tuple[Any, ...] | None = None
        transient_failures = 0
        while True:
            if should_cancel is not None and should_cancel():
                self.cancel(job_id)
                raise ComputeJobCancelled(f"compute job {job_id} cancelled by caller")
            time.sleep(self._config.poll_interval)
            try:
                record = self.get(job_id)
                transient_failures = 0
            except ComputeServiceError as exc:
                transient_failures += 1
                if transient_failures >= 6:
                    raise
                logger.warning("compute poll %s transient failure %d/6: %s", job_id, transient_failures, exc)
                continue

            progress = record.get("progress") or {}
            if on_progress is not None:
                sig = (progress.get("completed"), progress.get("total"), progress.get("message"))
                if sig != last_progress_sig:
                    last_progress_sig = sig
                    try:
                        on_progress(progress)
                    except Exception:  # noqa: BLE001 - progress reporting must never fail the job
                        logger.debug("on_progress callback raised", exc_info=True)

            status = str(record.get("status") or "")
            if status == "succeeded":
                return dict(record.get("result") or {})
            if status == "failed":
                raise ComputeServiceError(f"compute job {job_id} ({executor}) failed: {record.get('error')}")
            if status == "cancelled":
                raise ComputeJobCancelled(f"compute job {job_id} ({executor}) was cancelled")
