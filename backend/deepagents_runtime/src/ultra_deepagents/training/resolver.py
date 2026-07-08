"""Shared serving-weights resolver client — the ONE canary/active read (§8.1).

``resolve(model_key, run_id)`` asks the control plane which weights to serve:
the lineage active pointer, or the newest canary for the deterministic 10% of
run_ids (``GET /v2/training/models/{model_key}/resolve``). Serving must NEVER
break on resolution — every failure path (no base URL, HTTP error, malformed
payload) returns ``None`` so callers fail open to their baked weights, and the
observation write swallows+logs errors because telemetry loss is acceptable
where a failed detection pass is not. Wire conventions (sync urllib, worker
token header, timeout) mirror control_client.py.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib import parse as urllib_parse
from urllib import request as urllib_request

logger = logging.getLogger(__name__)


class ServingWeightsResolver:
    def __init__(
        self,
        settings: Any = None,
        *,
        base_url: str = "",
        ttl_seconds: float = 30.0,
    ) -> None:
        if settings is None and not base_url:
            from ultra_deepagents.config import RuntimeSettings

            settings = RuntimeSettings.from_env()
        self._settings = settings
        self._base_url = (
            base_url or str(getattr(settings, "control_base_url", "") or "")
        ).rstrip("/")
        self._ttl_seconds = max(0.0, float(ttl_seconds))
        self._timeout = max(
            0.1, float(getattr(settings, "control_status_timeout_seconds", 2.0) or 2.0)
        )
        # The cache key includes run_id: the canary split is per-run, so an
        # answer cached for one run must never be served for another.
        self._cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}

    @property
    def configured(self) -> bool:
        return bool(self._base_url)

    def resolve(self, model_key: str, run_id: str) -> dict[str, Any] | None:
        """Never raises: any error returns ``None`` (fail open to baked weights).

        Success payload: ``{model_key, weights_uri, version_id, is_canary}``,
        cached per (model_key, run_id) for ``ttl_seconds``.
        """
        try:
            if not self._base_url:
                return None
            key = (str(model_key), str(run_id or ""))
            now = time.monotonic()
            cached = self._cache.get(key)
            if cached is not None and cached[0] > now:
                return dict(cached[1])
            quoted = urllib_parse.quote(str(model_key), safe="")
            query = urllib_parse.urlencode({"run_id": str(run_id or "")})
            url = f"{self._base_url}/v2/training/models/{quoted}/resolve?{query}"
            data = self._request(url, "GET", None)
            if not isinstance(data, dict) or not str(data.get("weights_uri") or "").strip():
                return None
            self._cache[key] = (now + self._ttl_seconds, dict(data))
            return dict(data)
        except Exception:
            logger.warning(
                "Serving-weights resolution failed; failing open to baked weights.",
                extra={"model_key": model_key, "run_id": run_id},
                exc_info=True,
            )
            return None

    def record_canary_observation(
        self, model_key: str, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Telemetry only: swallows+logs every failure — serving never breaks on it."""
        try:
            if not self._base_url:
                return None
            quoted = urllib_parse.quote(str(model_key), safe="")
            url = f"{self._base_url}/v2/training/models/{quoted}/canary-observations"
            return self._request(url, "POST", dict(payload or {}))
        except Exception:
            logger.warning(
                "Canary observation write failed; continuing.",
                extra={"model_key": model_key},
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------ wire

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        token = str(getattr(self._settings, "control_worker_token", "") or "").strip()
        if token:
            headers["X-Ultra-Worker-Token"] = token
        return headers

    def _request(
        self, url: str, method: str, payload: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib_request.Request(url, data=body, method=method, headers=self._headers())
        with urllib_request.urlopen(request, timeout=self._timeout) as response:
            raw = response.read()
        if not raw:
            return None
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else None
