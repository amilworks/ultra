"""HTTP client for the private Megaseg inference service."""

from __future__ import annotations

import json
import mimetypes
import tarfile
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import httpx


class MegasegServiceError(RuntimeError):
    """Raised when the Megaseg service returns an invalid or failed response."""


class MegasegServiceTimeoutError(MegasegServiceError):
    """Raised when waiting for a Megaseg service job times out."""


class MegasegServiceClient:
    """Small sync client used by the backend Megaseg tool path."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._base_url = str(base_url or "").strip().rstrip("/")
        self._api_key = str(api_key or "").strip() or None
        self._timeout = max(1.0, float(timeout_seconds or 60.0))

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        url = f"{self._base_url}{path}"
        headers = dict(self._headers())
        headers.update(kwargs.pop("headers", {}) or {})
        with httpx.Client(timeout=self._timeout) as client:
            response = client.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()
        return response

    def submit_job(
        self,
        *,
        request_payload: dict[str, Any],
        local_upload_paths: list[Path] | None = None,
    ) -> dict[str, Any]:
        upload_paths = [Path(path).expanduser() for path in list(local_upload_paths or [])]
        if upload_paths:
            with ExitStack() as stack:
                files: list[tuple[str, tuple[str, Any, str]]] = []
                for path in upload_paths:
                    if path.is_dir():
                        archive_name = f"{path.name}.tar.gz"
                        temp_file = stack.enter_context(
                            tempfile.NamedTemporaryFile(mode="w+b", suffix=".tar.gz")
                        )
                        with tarfile.open(fileobj=temp_file, mode="w:gz") as archive:
                            archive.add(path, arcname=path.name)
                        temp_file.flush()
                        temp_file.seek(0)
                        files.append(("files", (archive_name, temp_file, "application/gzip")))
                        continue

                    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                    handle = stack.enter_context(path.open("rb"))
                    files.append(("files", (path.name, handle, content_type)))
                response = self._request(
                    "POST",
                    "/v1/jobs",
                    data={"request_json": json.dumps(request_payload, ensure_ascii=False)},
                    files=files,
                )
        else:
            response = self._request("POST", "/v1/jobs", json=request_payload)
        payload = response.json()
        if not isinstance(payload, dict) or not payload.get("job_id"):
            raise MegasegServiceError("Megaseg service did not return a job_id.")
        return payload

    def get_job(self, job_id: str) -> dict[str, Any]:
        response = self._request("GET", f"/v1/jobs/{job_id}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise MegasegServiceError("Megaseg service returned a non-object job payload.")
        return payload

    def wait_for_job(
        self,
        *,
        job_id: str,
        poll_interval_seconds: float = 2.0,
        wait_timeout_seconds: float = 7200.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + max(1.0, float(wait_timeout_seconds or 7200.0))
        interval = max(0.2, float(poll_interval_seconds or 2.0))
        while True:
            payload = self.get_job(job_id)
            status = str(payload.get("status") or "").strip().lower()
            if status in {"succeeded", "failed"}:
                return payload
            if time.monotonic() >= deadline:
                raise MegasegServiceTimeoutError(
                    f"Megaseg service job {job_id} did not finish before the wait timeout."
                )
            time.sleep(interval)

    def download_artifact(self, *, job_id: str, artifact_name: str, destination: Path) -> Path:
        response = self._request(
            "GET",
            f"/v1/jobs/{job_id}/artifacts/{artifact_name}",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
        return destination
