"""HTTP client for the Bisque Ultra orchestrator API."""

from __future__ import annotations

import io
from typing import Any

import httpx

from src.config import get_settings


class OrchestratorClient:
    """Small API client used by local tooling and future external callers."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int | None = None,
        api_key: str | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.orchestrator_api_url).rstrip("/")
        self.timeout = timeout if timeout is not None else int(settings.orchestrator_api_timeout)
        self.api_key = (api_key if api_key is not None else settings.orchestrator_api_key) or None

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        uploaded_files: list[str] | None = None,
        file_ids: list[str] | None = None,
        resource_uris: list[str] | None = None,
        dataset_uris: list[str] | None = None,
        conversation_id: str | None = None,
        goal: str | None = None,
        reasoning_mode: str = "deep",
        scratchpad: bool = True,
        max_tool_calls: int = 12,
        max_runtime_seconds: int = 900,
    ) -> dict[str, Any]:
        """Submit a chat turn with optional file references and execution budgets.

        Parameters
        ----------
        messages : list[dict[str, str]]
            Conversation messages for the turn.
        uploaded_files : list[str] or None, default=None
            Legacy file path references accepted by the backend.
        file_ids : list[str] or None, default=None
            Opaque upload identifiers from ``/v1/uploads``.
        conversation_id : str or None, default=None
            Existing conversation identifier for continuity.
        goal : str or None, default=None
            Optional explicit goal override for planning.
        scratchpad : bool, default=True
            Deprecated compatibility argument. Ignored by backend schema.
        max_tool_calls : int, default=12
            Tool-call execution budget.
        max_runtime_seconds : int, default=900
            Maximum runtime budget in seconds.

        Returns
        -------
        dict[str, Any]
            Parsed JSON response from ``/v1/chat``.

        Notes
        -----
        Prefer ``file_ids`` over raw paths for browser upload flows.
        """
        payload = {
            "messages": messages,
            "uploaded_files": uploaded_files or [],
            "file_ids": file_ids or [],
            "resource_uris": resource_uris or [],
            "dataset_uris": dataset_uris or [],
            "conversation_id": str(conversation_id or "").strip() or None,
            "goal": goal,
            "reasoning_mode": str(reasoning_mode or "auto").strip().lower() or "auto",
            "budgets": {
                "max_tool_calls": int(max_tool_calls),
                "max_runtime_seconds": int(max_runtime_seconds),
            },
        }
        _ = scratchpad
        return self._post("/v1/chat", payload)

    def create_run(
        self,
        *,
        goal: str,
        plan: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Create a durable workflow run from a plan payload.

        Parameters
        ----------
        goal : str
            Run goal description.
        plan : dict[str, Any]
            Workflow plan payload compatible with backend orchestration schema.
        idempotency_key : str or None, default=None
            Optional key used to deduplicate retried create requests.

        Returns
        -------
        dict[str, Any]
            Parsed JSON response from ``/v1/runs``.
        """
        payload = {"goal": goal, "plan": plan}
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = str(idempotency_key)
        return self._post("/v1/runs", payload, headers=headers)

    def get_run(self, run_id: str) -> dict[str, Any]:
        """Fetch a single run record by ``run_id``.

        Parameters
        ----------
        run_id : str
            Workflow run identifier.

        Returns
        -------
        dict[str, Any]
            Run payload from ``/v1/runs/{run_id}``.
        """
        return self._get(f"/v1/runs/{run_id}")

    def get_run_events(self, run_id: str, *, limit: int = 200) -> dict[str, Any]:
        """Fetch run events in chronological order.

        Parameters
        ----------
        run_id : str
            Workflow run identifier.
        limit : int, default=200
            Maximum number of events to return.

        Returns
        -------
        dict[str, Any]
            Event list payload from ``/v1/runs/{run_id}/events``.
        """
        return self._get(f"/v1/runs/{run_id}/events", params={"limit": int(limit)})

    def list_artifacts(self, run_id: str, *, limit: int = 500) -> dict[str, Any]:
        """List persisted artifacts for a run.

        Parameters
        ----------
        run_id : str
            Workflow run identifier.
        limit : int, default=500
            Maximum artifact records to return.

        Returns
        -------
        dict[str, Any]
            Artifact list payload.
        """
        return self._get(f"/v1/artifacts/{run_id}", params={"limit": int(limit)})

    def get_artifact_manifest(self, run_id: str) -> dict[str, Any]:
        """Fetch the manifest describing run artifact provenance.

        Parameters
        ----------
        run_id : str
            Workflow run identifier.

        Returns
        -------
        dict[str, Any]
            Manifest payload for the run.
        """
        return self._get(f"/v1/artifacts/{run_id}/manifest")

    def evaluate_contract_health(
        self,
        *,
        run_ids: list[str] | None = None,
        limit: int = 25,
    ) -> dict[str, Any]:
        """Run contract-quality audits for recent or specific runs.

        Parameters
        ----------
        run_ids : list[str] or None, default=None
            Optional explicit run IDs to audit.
        limit : int, default=25
            Maximum number of runs audited when ``run_ids`` is not provided.

        Returns
        -------
        dict[str, Any]
            Contract-health audit payload.
        """
        payload = {
            "run_ids": run_ids or [],
            "limit": int(limit),
        }
        return self._post("/v1/evals/contracts", payload)

    def list_stat_tools(self) -> dict[str, Any]:
        """Return curated statistical tools available in the backend runtime.

        Returns
        -------
        dict[str, Any]
            Tool catalog payload.
        """
        return self._get("/v1/stats/tools")

    def run_stat_tool(self, *, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute a curated statistical tool by name.

        Parameters
        ----------
        tool_name : str
            Curated tool identifier.
        payload : dict[str, Any]
            Tool-specific input payload.

        Returns
        -------
        dict[str, Any]
            Tool execution result payload.
        """
        return self._post("/v1/stats/run", {"tool_name": tool_name, "payload": payload})

    def create_repro_report(
        self,
        *,
        run_id: str | None = None,
        title: str | None = None,
        result_summary: str | None = None,
        measurements: list[dict[str, Any]] | None = None,
        statistical_analysis: list[dict[str, Any]] | dict[str, Any] | None = None,
        qc_warnings: list[str] | None = None,
        limitations: list[str] | None = None,
        provenance: dict[str, Any] | None = None,
        next_steps: list[dict[str, Any] | str] | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """Generate a reproducibility report artifact from structured analysis outputs.

        Parameters
        ----------
        run_id : str or None, default=None
            Associated run identifier.
        title : str or None, default=None
            Report title override.
        result_summary : str or None, default=None
            Narrative summary text.
        measurements : list[dict[str, Any]] or None, default=None
            Structured measurement rows.
        statistical_analysis : list[dict[str, Any]] or dict[str, Any] or None, default=None
            Statistical analysis payload(s).
        qc_warnings : list[str] or None, default=None
            QC warning lines.
        limitations : list[str] or None, default=None
            Limitation statements.
        provenance : dict[str, Any] or None, default=None
            Provenance metadata.
        next_steps : list[dict[str, Any] | str] or None, default=None
            Recommended next actions.
        output_dir : str or None, default=None
            Optional explicit output directory.

        Returns
        -------
        dict[str, Any]
            Report creation response payload.
        """
        payload = {
            "run_id": run_id,
            "title": title,
            "result_summary": result_summary,
            "measurements": measurements or [],
            "statistical_analysis": statistical_analysis,
            "qc_warnings": qc_warnings or [],
            "limitations": limitations or [],
            "provenance": provenance or {},
            "next_steps": next_steps or [],
            "output_dir": output_dir,
        }
        return self._post("/v1/workflows/repro-report", payload)

    def load_scientific_image(
        self,
        *,
        file_path: str,
        scene: int | str | None = None,
        use_aicspylibczi: bool = False,
        array_mode: str = "plane",
        t_index: int | None = None,
        c_index: int | None = None,
        z_index: int | None = None,
        save_array: bool = True,
        include_array: bool = False,
        max_inline_elements: int = 16384,
    ) -> dict[str, Any]:
        """Load a scientific image and return normalized metadata plus preview payloads.

        Parameters
        ----------
        file_path : str
            Path to the source image file.
        scene : int or str or None, default=None
            Optional scene selector.
        use_aicspylibczi : bool, default=False
            Whether to force AICS CZI loading path.
        array_mode : str, default="plane"
            Array extraction mode requested by backend.
        t_index : int or None, default=None
            Time index for slice extraction.
        c_index : int or None, default=None
            Channel index for slice extraction.
        z_index : int or None, default=None
            Depth index for slice extraction.
        save_array : bool, default=True
            Whether backend should persist array artifacts.
        include_array : bool, default=False
            Whether to inline array data into response payload.
        max_inline_elements : int, default=16384
            Maximum array elements allowed for inline payloads.

        Returns
        -------
        dict[str, Any]
            Image-load response payload.

        Notes
        -----
        Prefer explicit index selection for deterministic outputs.
        """
        payload = {
            "file_path": file_path,
            "scene": scene,
            "use_aicspylibczi": bool(use_aicspylibczi),
            "array_mode": array_mode,
            "t_index": t_index,
            "c_index": c_index,
            "z_index": z_index,
            "save_array": bool(save_array),
            "include_array": bool(include_array),
            "max_inline_elements": int(max_inline_elements),
        }
        return self._post("/v1/data/load-image", payload)

    def upload_files(self, files: list[Any]) -> dict[str, Any]:
        """Upload one or more files and return opaque file IDs.

        Parameters
        ----------
        files : list[Any]
            File-like objects or tuple payloads accepted by ``httpx`` multipart
            upload.

        Returns
        -------
        dict[str, Any]
            Upload response payload with created file IDs.
        """
        multipart: list[tuple[str, tuple[str, Any, str]]] = []
        for item in files:
            if item is None:
                continue
            filename = getattr(item, "name", None)
            content_type = getattr(item, "type", None) or "application/octet-stream"
            if filename and hasattr(item, "read"):
                try:
                    if hasattr(item, "seek"):
                        item.seek(0)
                except Exception:
                    pass
                multipart.append(("files", (str(filename), item, str(content_type))))
                continue

            if isinstance(item, tuple) and len(item) >= 2:
                name = str(item[0])
                payload = item[1]
                ctype = str(item[2]) if len(item) >= 3 and item[2] else "application/octet-stream"
                if isinstance(payload, (bytes, bytearray)):
                    payload = io.BytesIO(payload)
                multipart.append(("files", (name, payload, ctype)))
                continue

            raise TypeError(f"Unsupported upload file payload type: {type(item)!r}")

        if not multipart:
            return {"file_count": 0, "uploaded": []}
        return self._post_multipart("/v1/uploads", files=multipart)

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"X-API-Key": str(self.api_key)}

    def _post(
        self, path: str, payload: dict[str, Any], headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        all_headers = self._headers()
        if headers:
            all_headers.update(headers)
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}{path}", json=payload, headers=all_headers)
            response.raise_for_status()
            return dict(response.json())

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = self._headers()
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}{path}", params=params, headers=headers)
            response.raise_for_status()
            return dict(response.json())

    def _post_multipart(
        self,
        path: str,
        *,
        files: list[tuple[str, tuple[str, Any, str]]],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        all_headers = self._headers()
        if headers:
            all_headers.update(headers)
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}{path}",
                files=files,
                headers=all_headers,
            )
            response.raise_for_status()
            return dict(response.json())
