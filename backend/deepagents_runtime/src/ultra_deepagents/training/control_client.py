"""Thin synchronous control-plane client for the training worker.

Synchronous on purpose (urllib, no event loop): the OS-thread lease keepalive
(keepalive.py) must renew from a wall-clock thread that is immune to event-loop
stalls, and every worker call site offloads through ``asyncio.to_thread``. The
header/auth shape mirrors the data-agent worker's ``_request_control_plane_json``
(worker-token + principal headers).

Endpoint contract (the Go control plane is being built against this exactly):

- ``POST   /v2/training/jobs/{job_id}/lease``   {worker_id, ttl_seconds}
- ``PATCH  /v2/training/jobs/{job_id}/lease``   {lease_token, ttl_seconds}
- ``DELETE /v2/training/jobs/{job_id}/lease``   {lease_token}
- ``POST   /v2/training/jobs/{job_id}/events``  {event_id, event_type, message, metadata}
- ``PATCH  /v2/training/jobs/{job_id}/status``  {status, error, output_summary}
- ``POST   /v2/training/models/{model_key}/versions``      (worker result writer)
- ``POST   /v2/training/jobs/{job_id}/benchmark-result``   (worker result writer)
- ``POST   /v2/training/jobs/{job_id}/gold-result``        (worker result writer)
- ``POST   /v2/training/jobs/{job_id}/status-result``      (worker result writer)

Lease POST/PATCH answer 409 when another worker holds the lease — the one kill
signal a worker trusts (TrainingLeaseConflict).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from ultra_deepagents.config import RuntimeSettings

logger = logging.getLogger(__name__)


class TrainingLeaseConflict(RuntimeError):  # noqa: N818 - mirrors DataAgentLeaseConflict
    """Authoritative 409: another worker owns (or took) the job lease."""


class TrainingLeaseUnavailable(RuntimeError):  # noqa: N818 - mirrors DataAgentLeaseUnavailable
    """The control plane could not grant/renew the lease (non-409 failure)."""


@dataclass(frozen=True)
class TrainingJobLease:
    job_id: str
    worker_id: str
    lease_token: str
    lease_expires_at: str = ""


class TrainingControlClient:
    def __init__(
        self,
        settings: RuntimeSettings,
        *,
        worker_id: str,
        principal_headers: dict[str, str] | None = None,
    ) -> None:
        self._settings = settings
        self._base_url = str(getattr(settings, "control_base_url", "") or "").rstrip("/")
        self._worker_id = worker_id
        self._principal_headers = dict(principal_headers or {})
        self._timeout = max(0.1, float(getattr(settings, "control_status_timeout_seconds", 2.0)))
        # Result writers can carry thousands of gold items; give them headroom.
        self._result_timeout = max(self._timeout, 30.0)

    @property
    def configured(self) -> bool:
        return bool(self._base_url)

    # ------------------------------------------------------------------ lease

    def acquire_lease(self, job_id: str, *, ttl_seconds: float) -> TrainingJobLease | None:
        if not self._base_url:
            return None
        payload = {"worker_id": self._worker_id, "ttl_seconds": ttl_seconds}
        data = self._lease_request(job_id, "POST", payload)
        return self._lease_from(job_id, data)

    def renew_lease(self, lease: TrainingJobLease, *, ttl_seconds: float) -> TrainingJobLease:
        if not self._base_url:
            return lease
        payload = {"lease_token": lease.lease_token, "ttl_seconds": ttl_seconds}
        data = self._lease_request(lease.job_id, "PATCH", payload)
        return self._lease_from(lease.job_id, data) or lease

    def release_lease(self, lease: TrainingJobLease) -> None:
        if not self._base_url:
            return
        try:
            self._request(
                self._job_url(lease.job_id, "/lease"), "DELETE", {"lease_token": lease.lease_token}
            )
        except Exception:
            logger.warning(
                "Training job lease release failed; the lease will expire on its own.",
                extra={"job_id": lease.job_id},
                exc_info=True,
            )

    def _lease_request(
        self, job_id: str, method: str, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        try:
            return self._request(self._job_url(job_id, "/lease"), method, payload)
        except urllib_error.HTTPError as exc:
            if exc.code == 409:
                raise TrainingLeaseConflict(
                    f"training job lease for {job_id} is held by another worker"
                ) from exc
            raise TrainingLeaseUnavailable(
                f"training job lease request failed with HTTP {exc.code}"
            ) from exc
        except Exception as exc:
            raise TrainingLeaseUnavailable("training job lease request failed") from exc

    def _lease_from(self, job_id: str, data: dict[str, Any] | None) -> TrainingJobLease | None:
        if not isinstance(data, dict):
            return None
        return TrainingJobLease(
            job_id=str(data.get("job_id") or job_id),
            worker_id=str(data.get("worker_id") or self._worker_id),
            lease_token=str(data.get("lease_token") or ""),
            lease_expires_at=str(data.get("lease_expires_at") or ""),
        )

    # ---------------------------------------------------------- job lifecycle

    def append_event(
        self,
        job_id: str,
        *,
        event_type: str,
        message: str = "",
        event_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self._base_url:
            return None
        payload = {
            "event_id": event_id,
            "event_type": event_type,
            "message": message,
            "metadata": metadata or {},
        }
        return self._request(self._job_url(job_id, "/events"), "POST", payload)

    def update_job_status(
        self,
        job_id: str,
        *,
        status: str,
        error: str = "",
        output_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self._base_url:
            return None
        payload = {
            "status": status,
            "error": error,
            "output_summary": output_summary or {},
        }
        return self._request(self._job_url(job_id, "/status"), "PATCH", payload)

    # ---------------------------------------------------------- result writers

    def register_model_version(
        self, model_key: str, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        if not self._base_url:
            return None
        quoted = urllib_parse.quote(str(model_key), safe="")
        url = f"{self._base_url}/v2/training/models/{quoted}/versions"
        return self._request(url, "POST", payload, timeout=self._result_timeout)

    def insert_benchmark_run(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self._base_url:
            return None
        return self._request(
            self._job_url(job_id, "/benchmark-result"),
            "POST",
            payload,
            timeout=self._result_timeout,
        )

    def finalize_gold_set(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """POST the freeze verdict: {status: frozen|failed, content_hash, item_count,
        label_stats, strata_summary, provenance, split_manifest_uri, failure_reasons,
        items:[...]}. Go finalizes the gold_sets row transactionally from this."""
        if not self._base_url:
            return None
        return self._request(
            self._job_url(job_id, "/gold-result"), "POST", payload, timeout=self._result_timeout
        )

    def upsert_model_status(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self._base_url:
            return None
        return self._request(
            self._job_url(job_id, "/status-result"), "POST", payload, timeout=self._result_timeout
        )

    # ------------------------------------------------------------------ wire

    def _job_url(self, job_id: str, suffix: str) -> str:
        quoted = urllib_parse.quote(str(job_id), safe="")
        return f"{self._base_url}/v2/training/jobs/{quoted}{suffix}"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        headers.update(self._principal_headers)
        token = str(getattr(self._settings, "control_worker_token", "") or "").strip()
        if token:
            headers["X-Ultra-Worker-Token"] = token
        return headers

    def _request(
        self,
        url: str,
        method: str,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        body = json.dumps(payload).encode("utf-8")
        request = urllib_request.Request(url, data=body, method=method, headers=self._headers())
        with urllib_request.urlopen(request, timeout=timeout or self._timeout) as response:
            raw = response.read()
        if not raw:
            return None
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else None
