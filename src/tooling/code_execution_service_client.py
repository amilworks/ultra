"""HTTP client for the private code execution service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx


class CodeExecutionServiceClient:
    """Small sync client used by the backend code execution path."""

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
        with httpx.Client(timeout=self._timeout) as client:
            response = client.request(
                method,
                f"{self._base_url}{path}",
                headers=self._headers(),
                **kwargs,
            )
        response.raise_for_status()
        return response

    def submit_job(
        self,
        *,
        request_payload: dict[str, Any],
        bundle_path: Path,
        local_input_paths: list[Path],
    ) -> dict[str, Any]:
        with bundle_path.open("rb") as bundle_handle:
            files: list[tuple[str, tuple[str, Any, str]]] = [
                ("bundle", (bundle_path.name, bundle_handle, "application/gzip"))
            ]
            open_handles: list[Any] = []
            try:
                for path in local_input_paths:
                    handle = path.open("rb")
                    open_handles.append(handle)
                    files.append(("files", (path.name, handle, "application/octet-stream")))
                response = self._request(
                    "POST",
                    "/v1/jobs",
                    data={"request_json": json.dumps(request_payload, ensure_ascii=False)},
                    files=files,
                )
            finally:
                for handle in open_handles:
                    handle.close()
        return response.json()
