from __future__ import annotations

import argparse
import hashlib
import http.cookiejar
import json
import math
import os
import posixpath
import re
import stat
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, cast
from urllib import parse, request

from ultra_deepagents.materials.trace_binding import (
    MATERIALS_TYPED_SCIENTIFIC_TOOL_OPERATIONS,
)
from ultra_deepagents.materials.validation import (
    canonical_record_json,
    parse_assessment_record,
)

RARESPOT_SKILL_NAME = "prairie-dog-detection"
MEDICAL_VOLUME_SKILL_NAME = "medical-volume-slices"
RARESPOT_SCRIPT_PATH = "/opt/rarespot/rarespot_detect.py"
MEDICAL_VOLUME_SCRIPT_PATH = "/opt/medvol/volume_slices.py"
SKILL_SCRIPT_PATHS = {
    RARESPOT_SCRIPT_PATH: RARESPOT_SKILL_NAME,
    MEDICAL_VOLUME_SCRIPT_PATH: MEDICAL_VOLUME_SKILL_NAME,
}
SKILL_READ_PATTERN = re.compile(r"/skills/([A-Za-z0-9_.-]+)/SKILL\.md\b")
MATERIALS_SKILL_NAMES = {
    "computational-materials",
    "materials-characterization-advanced",
    "materials-characterization",
    "materials-crystal-plasticity",
    "materials-mechanics-degradation",
    "materials-processing-kinetics",
    "materials-sensor-data",
    "materials-structure-thermo",
}
# First-party, closed-schema materials operations are execution evidence in their own
# right. Requiring a generic sandbox ``execute`` call for these paths would both
# mis-score the preferred integration and encourage the model to rediscover stable
# scientific APIs in a code runner. Inventory/inspection-only and static support tools
# are deliberately absent.
MATERIALS_TYPED_SCIENTIFIC_TOOL_NAMES = set(MATERIALS_TYPED_SCIENTIFIC_TOOL_OPERATIONS)
MATERIALS_TYPED_SCIENTIFIC_OPERATIONS = frozenset(
    operation
    for operations in MATERIALS_TYPED_SCIENTIFIC_TOOL_OPERATIONS.values()
    for operation in operations
)
MATERIALS_VALIDATION_FILENAME = "materials_validation.json"
MATERIALS_VALIDATION_MAX_BYTES = 1_000_000
CALPHAD_UPLOAD_MANIFEST_MAX_BYTES = 1_000_000
CALPHAD_UPLOAD_MANIFEST_SCHEMA = "1"
IMMUTABLE_RUNTIME_IMAGE_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
TYPED_MATERIALS_EVIDENCE_ID_PATTERN = re.compile(r"^typed-([a-z0-9][a-z0-9_.-]*):([0-9a-f]{64})$")
TYPED_MATERIALS_EVIDENCE_PATH_PATTERNS = (
    re.compile(r"^outputs/calphad/(equilibrium|scheil)/([0-9a-f]{64})\.json$"),
    re.compile(
        r"^outputs/kinetics/"
        r"(transport_coefficients|single_phase_diffusion_1d|binary_precipitation_kwn)/"
        r"([0-9a-f]{64})\.json$"
    ),
)


class ControlPlaneClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 10.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.extra_headers = _clean_extra_headers(extra_headers or {})
        self.cookie_jar = http.cookiejar.CookieJar()
        self.opener = request.build_opener(request.HTTPCookieProcessor(self.cookie_jar))

    def create_thread(self, *, title: str) -> dict[str, Any]:
        payload = self._request("POST", "/v2/threads", {"title": title, "messages": []})
        return _unwrap(payload, "thread")

    def login_bisque_account(self, *, username: str, password: str) -> dict[str, Any]:
        payload = self._request(
            "POST",
            "/v2/auth/login",
            {"username": username, "password": password},
        )
        return dict(payload) if isinstance(payload, dict) else {"authenticated": False}

    def get_thread(self, thread_id: str) -> dict[str, Any]:
        payload = self._request("GET", f"/v2/threads/{parse.quote(thread_id, safe='')}")
        return _unwrap(payload, "thread")

    def create_run(
        self,
        *,
        thread_id: str,
        goal: str,
        messages: list[dict[str, str]],
        idempotency_key: str,
        file_ids: list[str] | None = None,
        workflow_hint: dict[str, Any] | None = None,
        selection_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "goal": goal,
            "messages": messages,
            "idempotency_key": idempotency_key,
            "file_ids": file_ids or [],
        }
        if workflow_hint:
            body["workflow_hint"] = workflow_hint
        if selection_context:
            body["selection_context"] = selection_context
        payload = self._request(
            "POST",
            f"/v2/threads/{parse.quote(thread_id, safe='')}/runs",
            body,
            headers={"Idempotency-Key": idempotency_key},
        )
        return _unwrap(payload, "run")

    def upload_files(self, paths: list[str | Path]) -> list[str]:
        files = [Path(path).expanduser().resolve() for path in paths]
        if not files:
            return []
        payload = self._request_multipart("/v2/uploads", files)
        uploaded = payload.get("uploaded", [])
        if not isinstance(uploaded, list):
            return []
        file_ids: list[str] = []
        for item in uploaded:
            if not isinstance(item, dict):
                continue
            file_id = str(item.get("file_id") or "").strip()
            if file_id:
                file_ids.append(file_id)
        return file_ids

    def patch_resource_metadata(
        self,
        file_id: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        payload = self._request(
            "PATCH",
            f"/v2/resources/{parse.quote(file_id, safe='')}",
            {"metadata": metadata},
        )
        return _unwrap(payload, "resource")

    def get_run(self, run_id: str) -> dict[str, Any]:
        return _unwrap(self._request("GET", f"/v2/runs/{parse.quote(run_id, safe='')}"), "run")

    def list_thread_messages(self, thread_id: str) -> list[dict[str, Any]]:
        payload = self._request(
            "GET",
            f"/v2/threads/{parse.quote(thread_id, safe='')}/messages",
        )
        messages = payload.get("messages", [])
        return messages if isinstance(messages, list) else []

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        payload = self._request("GET", f"/v2/artifacts/{parse.quote(artifact_id, safe='')}")
        return _unwrap(payload, "artifact")

    def list_run_events(self, run_id: str, *, limit: int = 2000) -> list[dict[str, Any]]:
        requested_limit = max(1, int(limit))
        events: list[dict[str, Any]] = []
        after_sequence = 0
        while True:
            payload = self._request(
                "GET",
                f"/v2/runs/{parse.quote(run_id, safe='')}/events"
                f"?limit={requested_limit}&after_sequence={after_sequence}",
            )
            page = payload.get("events", [])
            if not isinstance(page, list):
                break
            records = [event for event in page if isinstance(event, dict)]
            events.extend(records)
            next_after_sequence = max(
                [after_sequence, *(_event_sequence(event) for event in records)]
            )
            if len(records) < requested_limit or next_after_sequence <= after_sequence:
                break
            after_sequence = next_after_sequence
        return events

    def list_run_artifacts(self, run_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        payload = self._request(
            "GET",
            f"/v2/runs/{parse.quote(run_id, safe='')}/artifacts?limit={limit}",
        )
        artifacts = payload.get("artifacts", [])
        return artifacts if isinstance(artifacts, list) else []

    def download_artifact(self, artifact_id: str) -> bytes:
        path = f"/v2/artifacts/{parse.quote(artifact_id, safe='')}/download"
        return self._request_bytes("GET", path)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = request.Request(f"{self.base_url}{path}", data=data, method=method)
        req.add_header("Accept", "application/json")
        req.add_header("Content-Type", "application/json")
        req.add_header("X-Ultra-User-Id", "local-user")
        req.add_header("X-Ultra-Org-Id", "local-org")
        req.add_header("X-Ultra-Role", "researcher")
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        self._add_extra_headers(req)
        with self.opener.open(req, timeout=self.timeout_seconds) as response:
            body = response.read().decode("utf-8")
        return json.loads(body) if body else {}

    def _request_bytes(self, method: str, path: str) -> bytes:
        req = request.Request(f"{self.base_url}{path}", method=method)
        req.add_header("X-Ultra-User-Id", "local-user")
        req.add_header("X-Ultra-Org-Id", "local-org")
        req.add_header("X-Ultra-Role", "researcher")
        self._add_extra_headers(req)
        with self.opener.open(req, timeout=self.timeout_seconds) as response:
            return cast(bytes, response.read())

    def _request_multipart(self, path: str, files: list[Path]) -> dict[str, Any]:
        boundary = f"----ultra-trace-{uuid.uuid4().hex}"
        chunks: list[bytes] = []
        for file_path in files:
            data = file_path.read_bytes()
            filename = file_path.name
            chunks.extend(
                [
                    f"--{boundary}\r\n".encode(),
                    (
                        'Content-Disposition: form-data; name="files"; '
                        f'filename="{_escape_multipart_filename(filename)}"\r\n'
                    ).encode(),
                    f"Content-Type: {_content_type_for_path(file_path)}\r\n\r\n".encode(),
                    data,
                    b"\r\n",
                ]
            )
        chunks.append(f"--{boundary}--\r\n".encode())
        body = b"".join(chunks)
        req = request.Request(f"{self.base_url}{path}", data=body, method="POST")
        req.add_header("Accept", "application/json")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        req.add_header("X-Ultra-User-Id", "local-user")
        req.add_header("X-Ultra-Org-Id", "local-org")
        req.add_header("X-Ultra-Role", "researcher")
        self._add_extra_headers(req)
        with self.opener.open(req, timeout=self.timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
        return json.loads(response_body) if response_body else {}

    def _add_extra_headers(self, req: request.Request) -> None:
        for key, value in self.extra_headers.items():
            req.add_header(key, value)


def append_followup_messages(
    transcript: list[dict[str, str]],
    *,
    response_text: str,
    followup: str,
) -> list[dict[str, str]]:
    return [
        *transcript,
        {"role": "assistant", "content": response_text},
        {"role": "user", "content": followup},
    ]


def summarize_run_trace(
    *,
    run: dict[str, Any],
    events: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    accepted_ms: float | None = None,
    first_event_ms: float | None = None,
    first_backend_event_ms: float | None = None,
    first_delta_ms: float | None = None,
    first_tool_ms: float | None = None,
    first_artifact_ms: float | None = None,
    first_recovery_ms: float | None = None,
    terminal_ms: float | None = None,
    download_checks: dict[str, bool] | None = None,
    linked_artifacts: list[dict[str, Any]] | None = None,
    materials_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checks = download_checks or {}
    event_counts = Counter(str(event.get("event_kind") or "") for event in events)
    event_counts.pop("", None)
    response_text = str(run.get("response_text") or "")
    tool_names = _tool_names(events)
    terminal_event = _terminal_event(events)
    idle_recoveries = _idle_recoveries(events)
    rarespot_configurations = _rarespot_configurations(events)
    tool_capability_manifests = _tool_capability_manifests(events)
    skill_reads = _skill_reads(events)
    skill_scripts = _skill_scripts(events)
    available_skills = _available_skills(
        tool_capability_manifests,
        skill_reads,
        skill_scripts,
    )
    context_tool_hygiene = _context_tool_output_hygiene(events)
    delegation = _delegation_evidence(events)
    async_delegation = _async_delegation_evidence(events)
    tool_lifecycle = _tool_lifecycle_evidence(events)
    token_usage = _token_usage_evidence(events)
    mutation_intents, mutation_scope_valid = _run_remote_mutation_intents(run)
    return {
        "run_id": run.get("run_id"),
        "thread_id": run.get("thread_id"),
        "goal": run.get("goal"),
        "status": run.get("status"),
        "response_len": len(response_text),
        "response_preview": _response_preview(response_text),
        "response_tail": _response_tail(response_text),
        "tool_names": tool_names,
        "rarespot_configurations": rarespot_configurations,
        "tool_capability_manifests": tool_capability_manifests,
        "available_skills": available_skills,
        "skill_reads": skill_reads,
        "skill_scripts": skill_scripts,
        "context_tool_hygiene": context_tool_hygiene,
        "delegation": delegation,
        "async_delegation": async_delegation,
        "tool_lifecycle": tool_lifecycle,
        "token_usage": token_usage,
        "remote_mutation_intents": mutation_intents,
        "remote_mutation_scope_valid": mutation_scope_valid,
        "event_counts": dict(sorted(event_counts.items())),
        "accepted_ms": _round_ms(accepted_ms),
        "first_event_ms": _round_ms(first_event_ms),
        "first_backend_event_ms": _round_ms(first_backend_event_ms),
        "first_delta_ms": _round_ms(first_delta_ms),
        "first_tool_ms": _round_ms(first_tool_ms),
        "first_artifact_ms": _round_ms(first_artifact_ms),
        "first_recovery_ms": _round_ms(first_recovery_ms),
        "terminal_ms": _round_ms(terminal_ms),
        "terminal_event_kind": terminal_event.get("event_kind") if terminal_event else None,
        "terminal_event_sequence": _event_sequence(terminal_event) if terminal_event else None,
        "idle_recovery_count": len(idle_recoveries),
        "idle_recoveries": idle_recoveries,
        "artifact_count": len(artifacts),
        "artifacts": [_summarize_artifact(artifact, checks) for artifact in artifacts],
        "linked_artifacts": linked_artifacts or [],
        "materials_validation": materials_validation,
    }


def trace_prompt(
    client: ControlPlaneClient,
    *,
    thread_id: str,
    prompt: str,
    messages: list[dict[str, str]] | None = None,
    idempotency_key: str | None = None,
    file_ids: list[str] | None = None,
    timeout_seconds: float = 300.0,
    poll_interval_seconds: float = 1.0,
    verify_downloads: bool = False,
    materials_evidence_dir: Path | None = None,
    workflow_hint: dict[str, Any] | None = None,
    selection_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    key = idempotency_key or f"trace-{uuid.uuid4().hex}"
    accepted_started = time.monotonic()
    run = client.create_run(
        thread_id=thread_id,
        goal=prompt,
        messages=messages or [{"role": "user", "content": prompt}],
        idempotency_key=key,
        file_ids=file_ids,
        workflow_hint=workflow_hint,
        selection_context=selection_context,
    )
    accepted_ms = (time.monotonic() - accepted_started) * 1000
    run_id = str(run["run_id"])
    started = time.monotonic()
    first_event_ms: float | None = None
    first_backend_event_ms: float | None = None
    first_delta_ms: float | None = None
    first_tool_ms: float | None = None
    first_artifact_ms: float | None = None
    first_recovery_ms: float | None = None
    terminal_ms: float | None = None
    events: list[dict[str, Any]] = []
    while True:
        elapsed = time.monotonic() - started
        if elapsed > timeout_seconds:
            raise TimeoutError(f"run {run_id} did not finish within {timeout_seconds:.1f}s")
        run = client.get_run(run_id)
        events = client.list_run_events(run_id)
        elapsed_ms = elapsed * 1000
        if first_event_ms is None and events:
            first_event_ms = elapsed_ms
        if first_backend_event_ms is None and any(
            event.get("event_kind") not in {"run.accepted", ""} for event in events
        ):
            first_backend_event_ms = elapsed_ms
        if first_delta_ms is None and any(
            event.get("event_kind") == "message.delta" for event in events
        ):
            first_delta_ms = elapsed_ms
        if first_tool_ms is None and any(
            str(event.get("event_kind") or "").startswith("tool_call.") for event in events
        ):
            first_tool_ms = elapsed_ms
        if first_artifact_ms is None and any(
            event.get("event_kind") == "artifact.created" for event in events
        ):
            first_artifact_ms = elapsed_ms
        if first_recovery_ms is None and _idle_recoveries(events):
            first_recovery_ms = elapsed_ms
        if run.get("status") in {"succeeded", "failed", "canceled"}:
            terminal_ms = elapsed_ms
            break
        time.sleep(poll_interval_seconds)
    if terminal_ms is not None:
        run = client.get_run(run_id)
        final_events = client.list_run_events(run_id)
        if final_events:
            events = final_events
    artifacts = client.list_run_artifacts(run_id)
    downloads = _verify_downloads(client, artifacts) if verify_downloads else {}
    materials_validation = _inspect_materials_validation_artifact(
        client,
        artifacts,
        evidence_dir=materials_evidence_dir,
    )
    linked_artifacts = (
        _summarize_linked_artifacts(client, str(run.get("response_text") or ""))
        if verify_downloads
        else []
    )
    summary = summarize_run_trace(
        run=run,
        events=events,
        artifacts=artifacts,
        accepted_ms=accepted_ms,
        first_event_ms=first_event_ms,
        first_backend_event_ms=first_backend_event_ms,
        first_delta_ms=first_delta_ms,
        first_tool_ms=first_tool_ms,
        first_artifact_ms=first_artifact_ms,
        first_recovery_ms=first_recovery_ms,
        terminal_ms=terminal_ms,
        download_checks=downloads,
        linked_artifacts=linked_artifacts,
        materials_validation=materials_validation,
    )
    return run, summary


def _read_bounded_regular_file(path: Path, *, maximum: int, label: str) -> bytes:
    resolved = path.expanduser()
    try:
        before = resolved.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is missing or unreadable") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_size <= 0
        or before.st_size > maximum
    ):
        raise ValueError(f"{label} must be a bounded regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise ValueError(f"{label} could not be opened securely") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or opened.st_size != before.st_size
        ):
            raise ValueError(f"{label} changed while it was opened")
        remaining = opened.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"{label} was truncated while it was read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"{label} grew while it was read")
        after = os.fstat(descriptor)
        if (after.st_size, after.st_mtime_ns) != (opened.st_size, opened.st_mtime_ns):
            raise ValueError(f"{label} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_unique_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not finite UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _bounded_manifest_text(record: dict[str, Any], key: str, maximum: int) -> str:
    value = str(record.get(key) or "").strip()
    if not value or len(value) > maximum or any(ord(character) < 32 for character in value):
        raise ValueError(f"CALPHAD upload manifest has invalid {key}")
    return value


def _calphad_upload_metadata_from_manifest(
    upload_path: str | Path,
    manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    upload = Path(upload_path).expanduser().absolute()
    upload_payload = _read_bounded_regular_file(
        upload,
        maximum=64 * 1024 * 1024,
        label="CALPHAD upload",
    )
    manifest_file = Path(manifest_path).expanduser().absolute()
    manifest_payload = _read_bounded_regular_file(
        manifest_file,
        maximum=CALPHAD_UPLOAD_MANIFEST_MAX_BYTES,
        label="CALPHAD upload manifest",
    )
    manifest = _load_unique_json_object(manifest_payload, label="CALPHAD upload manifest")
    databases = manifest.get("databases")
    if manifest.get("schema_version") != CALPHAD_UPLOAD_MANIFEST_SCHEMA or not isinstance(
        databases, list
    ):
        raise ValueError("CALPHAD upload manifest schema is invalid")
    matching = [
        record
        for record in databases
        if isinstance(record, dict) and record.get("filename") == upload.name
    ]
    if len(matching) != 1:
        raise ValueError("CALPHAD upload manifest must bind the upload filename exactly once")
    record = matching[0]
    expected_format = {".tdb": "tdb", ".dat": "dat"}.get(upload.suffix.casefold())
    if expected_format is None:
        raise ValueError("CALPHAD upload must use a readable .tdb or .dat suffix")
    if str(record.get("format") or "").strip().casefold() != expected_format:
        raise ValueError("CALPHAD upload manifest format does not match the upload suffix")
    upload_sha256 = hashlib.sha256(upload_payload).hexdigest()
    if record.get("sha256") != upload_sha256 or record.get("size_bytes") != len(upload_payload):
        raise ValueError("CALPHAD upload bytes do not match the reviewed manifest")
    database_id = _bounded_manifest_text(record, "database_id", 512)
    source = _bounded_manifest_text(record, "source_uri", 2048)
    source_uri = parse.urlsplit(source)
    if source_uri.scheme != "https" or not source_uri.hostname or source_uri.username:
        raise ValueError("CALPHAD upload manifest source_uri must be a public HTTPS URI")
    license_id = _bounded_manifest_text(record, "license_id", 256)
    assessment_scope = _bounded_manifest_text(record, "assessment_scope", 4096)
    reference_state = _bounded_manifest_text(record, "reference_state", 1024)
    raw_limits = record.get("tdb_temperature_limits_K")
    if not isinstance(raw_limits, list) or len(raw_limits) != 2:
        raise ValueError("CALPHAD upload manifest must declare two temperature limits")
    limits: list[float] = []
    for value in raw_limits:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("CALPHAD upload temperature limits must be finite numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("CALPHAD upload temperature limits must be finite numbers")
        limits.append(number)
    if limits[0] <= 0 or limits[0] >= limits[1] or limits[1] > 10_000:
        raise ValueError("CALPHAD upload temperature limits are invalid")
    raw_pressure_limits = record.get("assessment_pressure_limits_Pa")
    if not isinstance(raw_pressure_limits, list) or len(raw_pressure_limits) != 2:
        raise ValueError("CALPHAD upload manifest must declare two pressure limits")
    pressure_limits: list[float] = []
    for value in raw_pressure_limits:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("CALPHAD upload pressure limits must be finite numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("CALPHAD upload pressure limits must be finite numbers")
        pressure_limits.append(number)
    if (
        pressure_limits[0] < 1e-9
        or pressure_limits[0] > pressure_limits[1]
        or pressure_limits[1] > 1e12
    ):
        raise ValueError("CALPHAD upload pressure limits are invalid")
    metadata = {
        "calphad": {
            "database_id": database_id,
            "source": source,
            "license_id": license_id,
            "assessment_scope": assessment_scope,
            "reference_state": reference_state,
            "tdb_temperature_limits_K": limits,
            "assessment_pressure_limits_Pa": pressure_limits,
            "declaration_authority": "resource_owner",
        }
    }
    evidence = {
        "file_name": upload.name,
        "file_sha256": upload_sha256,
        "file_size_bytes": len(upload_payload),
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "database_id": database_id,
        "database_format": expected_format,
        "assessment_pressure_limits_Pa": pressure_limits,
        "declaration_authority": "resource_owner",
    }
    return metadata, evidence


def _apply_upload_calphad_manifests(
    client: ControlPlaneClient,
    *,
    upload_paths: list[str | Path],
    file_ids: list[str],
    manifest_paths: list[str | Path],
) -> list[dict[str, Any]]:
    if not manifest_paths:
        return []
    if len(manifest_paths) != len(upload_paths):
        raise ValueError("each CALPHAD upload manifest must correspond to one --upload-file")
    if len(file_ids) != len(upload_paths):
        raise ValueError("the control plane did not return one file_id per CALPHAD upload")
    applied: list[dict[str, Any]] = []
    for upload_path, file_id, manifest_path in zip(
        upload_paths, file_ids, manifest_paths, strict=True
    ):
        metadata, evidence = _calphad_upload_metadata_from_manifest(
            upload_path,
            manifest_path,
        )
        resource = client.patch_resource_metadata(file_id, metadata)
        resource_metadata = resource.get("metadata")
        actual_calphad = (
            resource_metadata.get("calphad") if isinstance(resource_metadata, dict) else None
        )
        expected_calphad = metadata["calphad"]
        if not isinstance(actual_calphad, dict) or any(
            actual_calphad.get(key) != value for key, value in expected_calphad.items()
        ):
            raise ValueError("control plane did not persist exact CALPHAD owner declarations")
        applied.append({**evidence, "file_id": file_id})
    return applied


def run_trace(args: argparse.Namespace) -> dict[str, Any]:
    client_kwargs: dict[str, Any] = {"timeout_seconds": args.http_timeout}
    auth_headers = _auth_headers_from_args(args)
    if auth_headers:
        client_kwargs["extra_headers"] = auth_headers
    client = ControlPlaneClient(args.base_url, **client_kwargs)
    bisque_session = _login_bisque_from_args(client, args)
    file_ids = client.upload_files(args.upload_file) if args.upload_file else []
    calphad_upload_provenance = _apply_upload_calphad_manifests(
        client,
        upload_paths=args.upload_file,
        file_ids=file_ids,
        manifest_paths=getattr(args, "upload_calphad_manifest", []),
    )
    thread = client.create_thread(title=args.title)
    thread_id = str(thread["thread_id"])
    transcript = [{"role": "user", "content": args.prompt}]
    workflow_hint = _workflow_hint_from_args(args)
    selection_context = _selection_context_from_args(args)
    first_run, first_summary = trace_prompt(
        client,
        thread_id=thread_id,
        prompt=args.prompt,
        messages=transcript,
        file_ids=file_ids,
        timeout_seconds=args.timeout,
        poll_interval_seconds=args.poll_interval,
        verify_downloads=args.verify_downloads,
        materials_evidence_dir=getattr(args, "materials_evidence_dir", None),
        workflow_hint=workflow_hint,
        selection_context=selection_context,
    )
    result: dict[str, Any] = {
        "thread_id": thread_id,
        "prompt": first_summary,
    }
    if file_ids:
        result["uploaded_file_ids"] = file_ids
    if calphad_upload_provenance:
        result["upload_calphad_provenance"] = calphad_upload_provenance
    if bisque_session:
        result["bisque_session"] = bisque_session
    followups = _followup_prompts(args)
    followup_summaries: list[dict[str, Any]] = []
    previous_run = first_run
    for followup in followups:
        transcript = append_followup_messages(
            transcript,
            response_text=str(previous_run.get("response_text") or ""),
            followup=followup,
        )
        previous_run, followup_summary = trace_prompt(
            client,
            thread_id=thread_id,
            prompt=followup,
            messages=transcript,
            file_ids=file_ids,
            timeout_seconds=args.timeout,
            poll_interval_seconds=args.poll_interval,
            verify_downloads=args.verify_downloads,
            materials_evidence_dir=getattr(args, "materials_evidence_dir", None),
            workflow_hint=workflow_hint,
            selection_context=selection_context,
        )
        followup_summaries.append(followup_summary)
    if followup_summaries:
        result["followups"] = followup_summaries
        result["followup"] = followup_summaries[-1]
    try:
        thread_record = client.get_thread(thread_id)
        result["thread"] = summarize_thread_record(thread_record)
        result["thread_messages"] = summarize_thread_messages(
            client.list_thread_messages(thread_id)
        )
    except Exception as exc:
        result["thread_error"] = str(exc)
    return result


def evaluate_trace_requirements(
    result: dict[str, Any],
    *,
    min_artifacts: int = 0,
    min_response_chars: int = 0,
    require_downloads: bool = False,
) -> list[str]:
    issues: list[str] = []
    for label, summary in _labeled_trace_turn_summaries(result):
        run_id = str(summary.get("run_id") or "<unknown>")
        status = summary.get("status")
        if status != "succeeded":
            issues.append(f"{label} run {run_id} status={status}, want succeeded")
        response_len = int(summary.get("response_len") or 0)
        if response_len < min_response_chars:
            issues.append(
                f"{label} run {run_id} response_len={response_len} "
                f"below min_response_chars={min_response_chars}"
            )
        artifacts = _summary_artifacts(summary)
        artifact_count = len(artifacts)
        if artifact_count < min_artifacts:
            issues.append(
                f"{label} run {run_id} artifact_count={artifact_count} "
                f"below min_artifacts={min_artifacts}"
            )
        if require_downloads:
            for artifact in artifacts:
                if artifact.get("download_ok") is not False:
                    continue
                artifact_id = artifact.get("artifact_id") or "<unknown>"
                issues.append(
                    f"{label} run {run_id} artifact {artifact_id} download verification failed"
                )
    return issues


def _typed_completions_match_validation(
    completions: list[dict[str, Any]],
    validation: Any,
) -> bool:
    """Bind each typed completion to this turn's exact validation evidence."""

    if not completions:
        return True
    if not isinstance(validation, dict) or validation.get("valid") is not True:
        return False
    raw_bindings = validation.get("typed_result_bindings")
    if not isinstance(raw_bindings, list):
        return False
    validation_bindings = {
        (
            str(binding.get("operation") or "").strip().lower(),
            str(binding.get("result_artifact_sha256") or "").strip().lower(),
        )
        for binding in raw_bindings
        if isinstance(binding, dict)
    }
    record_sha256 = str(validation.get("record_sha256") or "").strip().lower()
    for completion in completions:
        tool_name = str(completion.get("tool_name") or "").strip()
        tool_call_id = str(completion.get("tool_call_id") or "").strip()
        operation = str(completion.get("scientific_operation") or "").strip().lower()
        result_sha256 = str(completion.get("result_artifact_sha256") or "").strip().lower()
        allowed_operations = MATERIALS_TYPED_SCIENTIFIC_TOOL_OPERATIONS.get(tool_name, ())
        if (
            not tool_call_id
            or completion.get("scientific_result_ok") is not True
            or operation not in allowed_operations
            or (operation, result_sha256) not in validation_bindings
        ):
            return False
        emitted_validation_sha256 = (
            str(completion.get("materials_validation_artifact_sha256") or "").strip().lower()
        )
        if emitted_validation_sha256 and emitted_validation_sha256 != record_sha256:
            return False
    return True


def evaluate_materials_trace_quality(
    result: dict[str, Any],
    *,
    min_score: float = 9.0,
) -> dict[str, Any]:
    """Evaluate whether materials traces contain auditable scientific evidence.

    This checks trace completeness and the internal consistency of the generated
    validation record. It does not replace an independent MatTools verifier.
    """

    turns = _trace_turn_summaries(result)
    issues: list[str] = []
    if not turns:
        return {
            "score": 0.0,
            "passed": False,
            "issues": ["no trace turns found"],
            "turn_count": 0,
        }

    terminal_ok = all(turn.get("status") == "succeeded" for turn in turns)
    lifecycle_records: list[dict[str, Any]] = []
    for turn in turns:
        lifecycle = turn.get("tool_lifecycle")
        lifecycle_records.append(
            cast(dict[str, Any], lifecycle) if isinstance(lifecycle, dict) else {}
        )
    matched_skill_read_completion = all(
        any(
            str(record.get("skill") or "") in MATERIALS_SKILL_NAMES
            for record in (lifecycle.get("matched_skill_read_completions") or [])
            if isinstance(record, dict)
        )
        for lifecycle in lifecycle_records
    )
    execute_completion_records = [
        [
            record
            for record in (lifecycle.get("matched_execute_completions") or [])
            if isinstance(record, dict)
        ]
        for lifecycle in lifecycle_records
    ]
    typed_scientific_completion_records = [
        [
            record
            for record in (lifecycle.get("matched_typed_scientific_completions") or [])
            if isinstance(record, dict)
        ]
        for lifecycle in lifecycle_records
    ]
    matched_execute_completion = all(bool(records) for records in execute_completion_records)
    matched_typed_scientific_completion = all(
        bool(records) for records in typed_scientific_completion_records
    )
    validations = [turn.get("materials_validation") for turn in turns]
    typed_validation_binding_by_turn = [
        _typed_completions_match_validation(records, validation)
        for records, validation in zip(
            typed_scientific_completion_records,
            validations,
            strict=True,
        )
    ]
    typed_validation_binding_ok = all(typed_validation_binding_by_turn)
    scientific_execution_present = all(
        bool(execute_records or typed_records)
        for execute_records, typed_records in zip(
            execute_completion_records,
            typed_scientific_completion_records,
            strict=True,
        )
    )
    # A typed scientific operation does not need a redundant generic execute call.
    # If an execute call is present, however, it must still carry an immutable image
    # identity and a successful exit; a typed success must not mask a bad side path.
    execute_runtime_image_attested = all(
        all(record.get("runtime_image_digest_matched") is True for record in records)
        for records in execute_completion_records
    )
    execute_exit_code_ok = all(
        all(record.get("exit_code_ok") is True for record in records)
        for records in execute_completion_records
    )
    no_failed_tool_terminal = all(
        int(lifecycle.get("failed_terminal_count") or 0) == 0 for lifecycle in lifecycle_records
    )
    execution_ok = (
        scientific_execution_present
        and typed_validation_binding_ok
        and execute_runtime_image_attested
        and execute_exit_code_ok
        and no_failed_tool_terminal
    )
    validation_present = all(isinstance(record, dict) for record in validations)
    validation_valid = validation_present and all(
        bool(record.get("valid")) for record in validations if isinstance(record, dict)
    )
    first_party_scientific_record_valid = (
        validation_valid
        and typed_validation_binding_ok
        and all(
            record.get("scientific_status") == "verified"
            for record in validations
            if isinstance(record, dict)
        )
    )
    no_silent_success = validation_valid and all(
        record.get("silent_success") is False for record in validations if isinstance(record, dict)
    )
    hashed_evidence = validation_valid and all(
        int(record.get("evidence_count") or 0) > 0 and record.get("evidence_verified") is True
        for record in validations
        if isinstance(record, dict)
    )
    durable_validation_artifact = all(
        any(
            _durable_artifact_output_path(artifact) == f"outputs/{MATERIALS_VALIDATION_FILENAME}"
            for artifact in _summary_artifacts(turn)
        )
        for turn in turns
    )
    remote_mutation_scope_valid = all(
        turn.get("remote_mutation_scope_valid") is True for turn in turns
    )
    remote_mutation_aligned = remote_mutation_scope_valid and all(
        _remote_mutations_match_scope(turn) for turn in turns
    )

    if not scientific_execution_present:
        execution_issue = (
            "materials result has no matched successful execute completion or "
            "typed scientific operation"
        )
    elif not execute_runtime_image_attested:
        execution_issue = "completed execute call lacks a matched immutable runtime image digest"
    elif not execute_exit_code_ok:
        execution_issue = "completed execute call did not report exit code 0"
    elif not no_failed_tool_terminal:
        execution_issue = "trace contains a failed tool terminal"
    elif not typed_validation_binding_ok:
        execution_issue = (
            "typed scientific completion is not bound to compatible validation evidence by "
            "tool-call ID, operation, and result-artifact SHA-256"
        )
    else:
        execution_issue = "materials result has no successful code-execution evidence"

    score = 0.0
    signals = (
        (terminal_ok, 1.0, "one or more orchestration turns did not succeed"),
        (
            matched_skill_read_completion,
            1.5,
            "a focused materials skill read has no matched successful completion",
        ),
        (execution_ok, 1.0, execution_issue),
        (validation_present, 1.5, "materials_validation.json was not inspected"),
        (validation_valid, 1.5, "materials validation record is invalid or self-inconsistent"),
        (
            first_party_scientific_record_valid,
            1.5,
            "first-party scientific record does not self-report verified status",
        ),
        (no_silent_success, 1.0, "a succeeded run contains a silent scientific failure"),
        (hashed_evidence, 0.75, "validation checks have no content-addressed evidence"),
        (durable_validation_artifact, 0.75, "validation record is not a durable run artifact"),
        (
            remote_mutation_aligned,
            0.5,
            "a remote BisQue mutation exceeded the immutable run capability",
        ),
    )
    for passed, weight, issue in signals:
        if passed:
            score += weight
        else:
            issues.append(issue)
    maximum_score = sum(weight for _, weight, _ in signals)
    rounded_score = round(min(score / maximum_score * 10.0, 10.0), 1)
    return {
        "score": rounded_score,
        "passed": rounded_score >= min_score and not issues,
        "issues": issues,
        "turn_count": len(turns),
        "quality_scope": "trace_and_first_party_validation_record",
        "independent_scientific_verification": False,
        "scientific_conclusion_verified": False,
        "signals": {
            "terminal_ok": terminal_ok,
            "materials_skill": matched_skill_read_completion,
            "matched_skill_read_completion": matched_skill_read_completion,
            "code_execution": execution_ok,
            "first_party_scientific_execution": execution_ok,
            "scientific_execution_present": scientific_execution_present,
            "matched_execute_completion": matched_execute_completion,
            "matched_typed_scientific_completion": matched_typed_scientific_completion,
            "typed_validation_binding_ok": typed_validation_binding_ok,
            "typed_validation_binding_by_turn": typed_validation_binding_by_turn,
            "execute_runtime_image_attested": execute_runtime_image_attested,
            "execute_exit_code_ok": execute_exit_code_ok,
            "no_failed_tool_terminal": no_failed_tool_terminal,
            "validation_present": validation_present,
            "validation_valid": validation_valid,
            "first_party_scientific_record_valid": first_party_scientific_record_valid,
            "no_silent_success": no_silent_success,
            "hashed_evidence": hashed_evidence,
            "durable_validation_artifact": durable_validation_artifact,
            "remote_mutation_aligned": remote_mutation_aligned,
            "remote_mutation_scope_valid": remote_mutation_scope_valid,
        },
    }


def evaluate_materials_trace_performance(
    result: dict[str, Any],
    *,
    max_model_calls_per_turn: int = 8,
    max_input_tokens_per_turn: int = 250_000,
    max_tool_calls_per_turn: int = 12,
    max_input_amplification_vs_peak: float = 6.0,
    max_terminal_ms_per_turn: float | None = None,
) -> dict[str, Any]:
    """Apply explicit live-run tripwires to bounded materials workflows.

    These are orchestration-regression limits, not scientific tolerances or global
    platform quotas. They intentionally use unique usage-event IDs so transport
    redelivery cannot create a false regression.
    """

    if max_model_calls_per_turn < 1:
        raise ValueError("max_model_calls_per_turn must be >= 1")
    if max_input_tokens_per_turn < 1:
        raise ValueError("max_input_tokens_per_turn must be >= 1")
    if max_tool_calls_per_turn < 1:
        raise ValueError("max_tool_calls_per_turn must be >= 1")
    if not math.isfinite(max_input_amplification_vs_peak) or (
        max_input_amplification_vs_peak < 1.0
    ):
        raise ValueError("max_input_amplification_vs_peak must be finite and >= 1")
    if max_terminal_ms_per_turn is not None and (
        not math.isfinite(max_terminal_ms_per_turn) or max_terminal_ms_per_turn <= 0
    ):
        raise ValueError("max_terminal_ms_per_turn must be finite and > 0 when set")

    turns = _trace_turn_summaries(result)
    if not turns:
        return {
            "passed": False,
            "issues": ["no trace turns found"],
            "turn_count": 0,
            "limits": {
                "max_model_calls_per_turn": max_model_calls_per_turn,
                "max_input_tokens_per_turn": max_input_tokens_per_turn,
                "max_tool_calls_per_turn": max_tool_calls_per_turn,
                "max_input_amplification_vs_peak": max_input_amplification_vs_peak,
                "max_terminal_ms_per_turn": max_terminal_ms_per_turn,
            },
            "turns": [],
        }

    issues: list[str] = []
    observations: list[dict[str, Any]] = []
    for index, turn in enumerate(turns, start=1):
        usage = turn.get("token_usage")
        lifecycle = turn.get("tool_lifecycle")
        usage_record = usage if isinstance(usage, dict) else {}
        lifecycle_record = lifecycle if isinstance(lifecycle, dict) else {}
        model_calls = int(usage_record.get("model_call_count") or 0)
        input_tokens = int(usage_record.get("input_tokens") or 0)
        amplification = float(usage_record.get("input_amplification_vs_peak") or 0.0)
        invalid_usage = int(usage_record.get("invalid_event_count") or 0)
        conflicting_usage = int(usage_record.get("conflicting_duplicate_count") or 0)
        tool_calls = int(lifecycle_record.get("started_call_count") or 0)
        failed_tools = int(lifecycle_record.get("failed_terminal_count") or 0)
        terminal_ms_value = turn.get("terminal_ms")
        terminal_ms = (
            float(terminal_ms_value)
            if isinstance(terminal_ms_value, int | float)
            and not isinstance(terminal_ms_value, bool)
            else None
        )
        label = f"turn[{index}]"
        if model_calls == 0:
            issues.append(f"{label} has no valid token-usage evidence")
        if invalid_usage:
            issues.append(f"{label} has {invalid_usage} malformed token-usage event(s)")
        if conflicting_usage:
            issues.append(
                f"{label} has {conflicting_usage} conflicting duplicate token-usage event(s)"
            )
        if model_calls > max_model_calls_per_turn:
            issues.append(
                f"{label} model_call_count={model_calls} exceeds {max_model_calls_per_turn}"
            )
        if input_tokens > max_input_tokens_per_turn:
            issues.append(
                f"{label} input_tokens={input_tokens} exceeds {max_input_tokens_per_turn}"
            )
        if tool_calls > max_tool_calls_per_turn:
            issues.append(f"{label} tool_call_count={tool_calls} exceeds {max_tool_calls_per_turn}")
        if amplification > max_input_amplification_vs_peak:
            issues.append(
                f"{label} input_amplification_vs_peak={amplification} exceeds "
                f"{max_input_amplification_vs_peak}"
            )
        if failed_tools:
            issues.append(f"{label} has {failed_tools} failed tool terminal(s)")
        if (
            max_terminal_ms_per_turn is not None
            and terminal_ms is not None
            and terminal_ms > max_terminal_ms_per_turn
        ):
            issues.append(f"{label} terminal_ms={terminal_ms} exceeds {max_terminal_ms_per_turn}")
        observations.append(
            {
                "run_id": turn.get("run_id"),
                "model_call_count": model_calls,
                "input_tokens": input_tokens,
                "output_tokens": int(usage_record.get("output_tokens") or 0),
                "peak_input_tokens": int(usage_record.get("peak_input_tokens") or 0),
                "input_amplification_vs_peak": amplification,
                "tool_call_count": tool_calls,
                "failed_tool_terminal_count": failed_tools,
                "terminal_ms": terminal_ms,
            }
        )

    return {
        "passed": not issues,
        "issues": issues,
        "turn_count": len(turns),
        "scope": "bounded_materials_orchestration_regression",
        "limits": {
            "max_model_calls_per_turn": max_model_calls_per_turn,
            "max_input_tokens_per_turn": max_input_tokens_per_turn,
            "max_tool_calls_per_turn": max_tool_calls_per_turn,
            "max_input_amplification_vs_peak": max_input_amplification_vs_peak,
            "max_terminal_ms_per_turn": max_terminal_ms_per_turn,
        },
        "turns": observations,
    }


def _remote_mutations_match_scope(turn: dict[str, Any]) -> bool:
    tool_names = {str(name) for name in (turn.get("tool_names") or [])}
    upload_called = bool(
        tool_names.intersection({"bisque_upload_files", "bisque_upload_workspace_files"})
    )
    dataset_called = "bisque_create_dataset" in tool_names
    if not upload_called and not dataset_called:
        return True
    allowed = {str(item) for item in (turn.get("remote_mutation_intents") or [])}
    return (not upload_called or "bisque.upload" in allowed) and (
        not dataset_called or "bisque.create_dataset" in allowed
    )


def _run_remote_mutation_intents(run: dict[str, Any]) -> tuple[list[str], bool]:
    metadata = run.get("metadata")
    if not isinstance(metadata, dict):
        return [], True
    raw = metadata.get("remote_mutation_intents")
    if raw is None:
        return [], True
    if not isinstance(raw, list):
        return [], False
    allowed_order = ("bisque.upload", "bisque.create_dataset")
    if any(not isinstance(item, str) or item not in allowed_order for item in raw):
        return [], False
    requested = set(raw)
    return [item for item in allowed_order if item in requested], len(requested) == len(raw)


def _labeled_trace_turn_summaries(result: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    turns: list[tuple[str, dict[str, Any]]] = []
    prompt = result.get("prompt")
    if isinstance(prompt, dict):
        turns.append(("prompt", prompt))

    followups = result.get("followups")
    if isinstance(followups, list):
        turns.extend(
            (f"followup[{index}]", followup)
            for index, followup in enumerate(followups, start=1)
            if isinstance(followup, dict)
        )
        return turns

    followup = result.get("followup")
    if isinstance(followup, dict):
        turns.append(("followup", followup))
    return turns


def evaluate_paper_trace_quality(
    result: dict[str, Any],
    *,
    min_score: float = 8.5,
) -> dict[str, Any]:
    turns = _trace_turn_summaries(result)
    issues: list[str] = []
    if not turns:
        return {
            "score": 0.0,
            "passed": False,
            "issues": ["no trace turns found"],
            "turn_count": 0,
        }

    terminal_ok = all(turn.get("status") == "succeeded" for turn in turns)
    response_ok = all(int(turn.get("response_len") or 0) >= 800 for turn in turns)
    citations_ok = _turns_have_page_citations(turns)
    tools_ok = _turns_have_paper_tools(turns)
    figures_ok = _trace_has_figure_evidence(turns)
    technical_ok = _trace_has_technical_content(turns)
    followup_context_ok = len(turns) <= 1 or all(_has_any_paper_tool(turn) for turn in turns[1:])
    no_recoveries = all(int(turn.get("idle_recovery_count") or 0) == 0 for turn in turns)

    score = 0.0
    if terminal_ok:
        score += 1.5
    else:
        issues.append("one or more turns did not succeed")
    if response_ok:
        score += 1.0
    else:
        issues.append("one or more responses are too short for paper review")
    if citations_ok:
        score += 2.0
    else:
        issues.append("missing page-grounded citations")
    if tools_ok:
        score += 2.0
    else:
        issues.append("missing paper-tool evidence")
    if figures_ok:
        score += 1.0
    else:
        issues.append("missing rendered/downloadable figure evidence")
    if technical_ok:
        score += 1.5
    else:
        issues.append("missing technical/equation content")
    if followup_context_ok:
        score += 0.75
    else:
        issues.append("follow-ups did not use cached paper context")
    if no_recoveries:
        score += 0.25
    else:
        issues.append("run had idle recoveries")

    rounded_score = round(min(score, 10.0), 1)
    return {
        "score": rounded_score,
        "passed": rounded_score >= min_score and not issues,
        "issues": issues,
        "turn_count": len(turns),
        "signals": {
            "terminal_ok": terminal_ok,
            "response_ok": response_ok,
            "page_citations": citations_ok,
            "paper_tools": tools_ok,
            "figures": figures_ok,
            "technical_content": technical_ok,
            "followup_context": followup_context_ok,
            "no_idle_recoveries": no_recoveries,
        },
    }


def evaluate_rarespot_trace_quality(
    result: dict[str, Any],
    *,
    min_score: float = 8.5,
    expected_tile_size: int = 512,
    expected_tile_overlap: float = 0.25,
    expected_stride: int = 384,
) -> dict[str, Any]:
    turns = _trace_turn_summaries(result)
    issues: list[str] = []
    if not turns:
        return {
            "score": 0.0,
            "passed": False,
            "issues": ["no trace turns found"],
            "turn_count": 0,
        }

    terminal_ok = all(turn.get("status") == "succeeded" for turn in turns)
    response_ok = all(int(turn.get("response_len") or 0) >= 500 for turn in turns)
    rare_tools_ok = _trace_has_rarespot_tool(turns)
    detections_ok = _trace_mentions_detection_metrics(turns)
    artifact_ok = _trace_has_rarespot_artifacts(turns)
    comparative_ok = len(turns) < 2 or _trace_has_comparative_language(turns)
    report_ok = len(turns) < 3 or _trace_has_report_language(turns[-1])
    no_recoveries = all(int(turn.get("idle_recovery_count") or 0) == 0 for turn in turns)
    no_stub_outputs = not _trace_has_stub_or_duplicate_rarespot_outputs(turns)
    report_turns_avoid_inference = _report_only_rarespot_turns_avoid_inference(turns)
    no_irrelevant_paper_tools = not _trace_has_paper_tools(turns)
    expected_config_ok = _trace_rarespot_config_matches(
        turns,
        expected_tile_size=expected_tile_size,
        expected_tile_overlap=expected_tile_overlap,
        expected_stride=expected_stride,
    )

    score = 0.0
    if terminal_ok:
        score += 1.5
    else:
        issues.append("one or more turns did not succeed")
    if response_ok:
        score += 1.0
    else:
        issues.append("one or more responses are too short for RareSpot analysis")
    if rare_tools_ok:
        score += 2.0
    else:
        issues.append("no RareSpot inference tool call was observed")
    if detections_ok:
        score += 1.5
    else:
        issues.append("missing quantitative detection metrics")
    if artifact_ok:
        score += 1.5
    else:
        issues.append("missing downloadable RareSpot image/csv/report artifacts")
    if comparative_ok:
        score += 1.0
    else:
        issues.append("follow-up did not compare inference runs")
    if report_ok:
        score += 1.0
    else:
        issues.append("final follow-up did not synthesize a combined report")
    if no_recoveries:
        score += 0.5
    else:
        issues.append("run had idle recoveries")
    if not no_stub_outputs:
        issues.append("response mentions stub or duplicate RareSpot outputs")
    if not report_turns_avoid_inference:
        issues.append("report-only RareSpot turn reran inference instead of using prior artifacts")
    if not no_irrelevant_paper_tools:
        issues.append("RareSpot trace used irrelevant paper tools")
    if not expected_config_ok:
        issues.append(
            "RareSpot configuration did not match expected "
            f"tile_size={expected_tile_size}, tile_overlap={expected_tile_overlap}, stride={expected_stride}"
        )

    rounded_score = round(min(score, 10.0), 1)
    return {
        "score": rounded_score,
        "passed": rounded_score >= min_score and not issues,
        "issues": issues,
        "turn_count": len(turns),
        "signals": {
            "terminal_ok": terminal_ok,
            "response_ok": response_ok,
            "rarespot_tool": rare_tools_ok,
            "detection_metrics": detections_ok,
            "artifacts": artifact_ok,
            "comparative_followup": comparative_ok,
            "combined_report": report_ok,
            "no_idle_recoveries": no_recoveries,
            "no_stub_outputs": no_stub_outputs,
            "report_turns_avoid_inference": report_turns_avoid_inference,
            "no_irrelevant_paper_tools": no_irrelevant_paper_tools,
            "expected_configuration": expected_config_ok,
        },
    }


def evaluate_tool_autonomy_trace_quality(
    result: dict[str, Any],
    *,
    min_score: float = 8.5,
) -> dict[str, Any]:
    turns = _trace_turn_summaries(result)
    issues: list[str] = []
    if not turns:
        return {
            "score": 0.0,
            "passed": False,
            "issues": ["no trace turns found"],
            "turn_count": 0,
        }

    terminal_ok = all(turn.get("status") == "succeeded" for turn in turns)
    response_ok = all(int(turn.get("response_len") or 0) >= 500 for turn in turns)
    code_execution_ok = _trace_has_tool(turns, {"execute"})
    filesystem_ok = _trace_has_tool(
        turns,
        {
            "read_file",
            "write_file",
            "edit_file",
            "ls",
            "glob",
            "grep",
            "artifact_manifest",
            "stage_artifact_for_analysis",
            "stage_uploaded_files_for_analysis",
        },
    )
    durable_outputs_ok = _trace_has_code_and_analysis_artifacts(turns)
    downloads_ok = _trace_artifact_downloads_verified(turns)
    followup_context_ok = _tool_autonomy_followups_use_prior_context(turns)
    substantive_ok = _trace_has_autonomy_analysis_content(turns)
    no_recoveries = all(int(turn.get("idle_recovery_count") or 0) == 0 for turn in turns)

    score = 0.0
    if terminal_ok:
        score += 1.5
    else:
        issues.append("one or more turns did not succeed")
    if response_ok:
        score += 1.0
    else:
        issues.append("one or more responses are too short for autonomous analysis")
    if code_execution_ok:
        score += 1.5
    else:
        issues.append("missing execute tool evidence")
    if filesystem_ok:
        score += 1.0
    else:
        issues.append("missing filesystem/context tool evidence")
    if durable_outputs_ok:
        score += 2.0
    else:
        issues.append("missing durable code plus analysis artifacts")
    if downloads_ok:
        score += 1.0
    else:
        issues.append("artifact download verification failed or was not captured")
    if followup_context_ok:
        score += 1.0
    else:
        issues.append("follow-ups did not inspect or stage prior context")
    if substantive_ok:
        score += 1.0
    else:
        issues.append("missing quantitative/reproducibility analysis content")
    if no_recoveries:
        score += 1.0
    else:
        issues.append("run had idle recoveries")

    rounded_score = round(min(score, 10.0), 1)
    return {
        "score": rounded_score,
        "passed": rounded_score >= min_score and not issues,
        "issues": issues,
        "turn_count": len(turns),
        "signals": {
            "terminal_ok": terminal_ok,
            "response_ok": response_ok,
            "code_execution": code_execution_ok,
            "filesystem_tools": filesystem_ok,
            "durable_outputs": durable_outputs_ok,
            "downloaded_outputs": downloads_ok,
            "followup_context": followup_context_ok,
            "substantive_analysis": substantive_ok,
            "no_idle_recoveries": no_recoveries,
        },
    }


def evaluate_tool_capability_trace_quality(
    result: dict[str, Any],
    *,
    min_score: float = 8.5,
) -> dict[str, Any]:
    turns = _trace_turn_summaries(result)
    issues: list[str] = []
    if not turns:
        return {
            "score": 0.0,
            "passed": False,
            "issues": ["no trace turns found"],
            "turn_count": 0,
        }

    manifests = _trace_tool_capability_manifests(turns)
    builtin_names = _manifest_builtin_tool_names(manifests)
    registered_names = _manifest_registered_tool_names(manifests)
    known_builtin_names = {
        "execute",
        "write_file",
        "read_file",
        "edit_file",
        "ls",
        "glob",
        "grep",
        "task",
    }
    declared_names = builtin_names | registered_names
    allowed_without_manifest = known_builtin_names | TOOL_CAPABILITY_TOOL_NAMES
    used_tool_names = {
        str(tool_name)
        for turn in turns
        for tool_name in (turn.get("tool_names") or [])
        if str(tool_name).strip()
    }
    missing_used_tools = sorted(used_tool_names - declared_names - allowed_without_manifest)

    terminal_ok = all(turn.get("status") == "succeeded" for turn in turns)
    manifest_called = bool(manifests)
    execute_declared = "execute" in builtin_names
    filesystem_declared = {"write_file", "read_file", "ls"}.issubset(builtin_names)
    context_tools_declared = bool(
        {
            "artifact_manifest",
            "stage_artifact_for_analysis",
            "stage_uploaded_files_for_analysis",
        }
        & registered_names
    )
    storage_declared = any(
        isinstance(manifest.get("storage"), dict)
        and manifest["storage"].get("workspace") == "/workspace"
        and manifest["storage"].get("outputs") == "/outputs"
        for manifest in manifests
    )
    used_tools_declared = not missing_used_tools
    context_tool_hygiene = _trace_context_tool_hygiene(turns)
    context_tool_outputs_safe = int(context_tool_hygiene.get("host_path_leak_count") or 0) == 0

    score = 0.0
    if terminal_ok:
        score += 1.0
    else:
        issues.append("one or more turns did not succeed")
    if manifest_called:
        score += 2.0
    else:
        issues.append("tool capability manifest was not called or captured")
    if execute_declared:
        score += 1.5
    else:
        issues.append("execute was not declared by any captured tool capability manifest")
    if filesystem_declared:
        score += 1.0
    else:
        issues.append("filesystem tools were not declared by any captured manifest")
    if context_tools_declared:
        score += 1.0
    else:
        issues.append("context/artifact staging tools were not declared by any captured manifest")
    if storage_declared:
        score += 1.5
    else:
        issues.append("workspace/output storage routes were not declared by any captured manifest")
    if used_tools_declared:
        score += 2.0
    else:
        issues.append(
            "used tools were not declared in any captured manifest: "
            + ", ".join(missing_used_tools)
        )
    if not context_tool_outputs_safe:
        issues.append("context tool output leaked host paths")

    rounded_score = round(min(score, 10.0), 1)
    return {
        "score": rounded_score,
        "passed": rounded_score >= min_score and not issues,
        "issues": issues,
        "turn_count": len(turns),
        "signals": {
            "terminal_ok": terminal_ok,
            "manifest_called": manifest_called,
            "execute_declared": execute_declared,
            "filesystem_declared": filesystem_declared,
            "context_tools_declared": context_tools_declared,
            "storage_declared": storage_declared,
            "used_tools_declared": used_tools_declared,
            "context_tool_outputs_safe": context_tool_outputs_safe,
        },
        "context_tool_hygiene": context_tool_hygiene,
    }


def evaluate_delegation_trace_quality(
    result: dict[str, Any],
    *,
    min_score: float = 8.5,
) -> dict[str, Any]:
    turns = _trace_turn_summaries(result)
    issues: list[str] = []
    if not turns:
        return {
            "score": 0.0,
            "passed": False,
            "issues": ["no trace turns found"],
            "turn_count": 0,
        }

    manifests = _trace_tool_capability_manifests(turns)
    builtin_names = _manifest_builtin_tool_names(manifests)
    available_subagents = _manifest_available_subagent_names(manifests)
    scoped_subagent_context_tools = _manifest_scoped_subagent_context_tool_names(manifests)
    delegation = _trace_delegation_evidence(turns)

    terminal_ok = all(turn.get("status") == "succeeded" for turn in turns)
    response_ok = all(int(turn.get("response_len") or 0) >= 500 for turn in turns)
    task_declared = "task" in builtin_names
    subagents_declared = bool(available_subagents)
    scoped_subagent_context_tools_declared = bool(scoped_subagent_context_tools) and all(
        _context_staging_tool_names().issubset(set(tool_names))
        for tool_names in scoped_subagent_context_tools.values()
    )
    task_used = int(delegation.get("task_call_count") or 0) > 0
    subagent_observed = bool(delegation.get("subagent_names"))
    subagent_deltas = int(delegation.get("subagent_message_delta_count") or 0) > 0
    scoped_tool_events = int(delegation.get("scoped_tool_event_count") or 0) > 0
    scoped_usage_events = int(delegation.get("scoped_usage_event_count") or 0) > 0
    subagent_tools = bool(delegation.get("subagent_tool_names"))
    no_recoveries = all(int(turn.get("idle_recovery_count") or 0) == 0 for turn in turns)
    context_tool_hygiene = _trace_context_tool_hygiene(turns)
    context_tool_outputs_safe = int(context_tool_hygiene.get("host_path_leak_count") or 0) == 0

    score = 0.0
    if terminal_ok:
        score += 1.0
    else:
        issues.append("one or more turns did not succeed")
    if response_ok:
        score += 1.0
    else:
        issues.append("one or more responses are too short for delegation trace")
    if task_declared:
        score += 1.5
    else:
        issues.append("task was not declared by any captured tool capability manifest")
    if subagents_declared:
        score += 1.5
    else:
        issues.append("no available subagents were declared by any captured manifest")
    if not scoped_subagent_context_tools_declared:
        issues.append("scoped subagent context tools were not declared by any captured manifest")
    if task_used:
        score += 1.5
    else:
        issues.append("task tool was not used")
    if subagent_observed:
        score += 1.0
    else:
        issues.append("no subagent names were observed in trace events")
    if subagent_deltas:
        score += 1.0
    else:
        issues.append("no subagent message deltas were observed")
    if scoped_tool_events:
        score += 1.0
    else:
        issues.append("no scoped subagent tool events were observed")
    if scoped_usage_events:
        score += 0.75
    else:
        issues.append("no scoped subagent token usage events were observed")
    if subagent_tools:
        score += 0.75
    else:
        issues.append("no subagent tool names were observed")
    if no_recoveries:
        score += 0.75
    else:
        issues.append("run had idle recoveries")
    if not context_tool_outputs_safe:
        issues.append("context tool output leaked host paths")

    rounded_score = round(min(score, 10.0), 1)
    return {
        "score": rounded_score,
        "passed": rounded_score >= min_score and not issues,
        "issues": issues,
        "turn_count": len(turns),
        "signals": {
            "terminal_ok": terminal_ok,
            "response_ok": response_ok,
            "task_declared": task_declared,
            "subagents_declared": subagents_declared,
            "scoped_subagent_context_tools_declared": scoped_subagent_context_tools_declared,
            "task_used": task_used,
            "subagent_observed": subagent_observed,
            "subagent_message_deltas": subagent_deltas,
            "subagent_tools_scoped": scoped_tool_events,
            "subagent_usage_scoped": scoped_usage_events,
            "subagent_tools": subagent_tools,
            "no_idle_recoveries": no_recoveries,
            "context_tool_outputs_safe": context_tool_outputs_safe,
        },
        "delegation": delegation,
        "context_tool_hygiene": context_tool_hygiene,
        "available_subagents": sorted(available_subagents),
        "scoped_subagent_context_tools": scoped_subagent_context_tools,
    }


def evaluate_async_delegation_trace_quality(
    result: dict[str, Any],
    *,
    min_score: float = 8.5,
) -> dict[str, Any]:
    turns = _trace_turn_summaries(result)
    issues: list[str] = []
    if not turns:
        return {
            "score": 0.0,
            "passed": False,
            "issues": ["no trace turns found"],
            "turn_count": 0,
        }

    manifests = _trace_tool_capability_manifests(turns)
    builtin_names = _manifest_builtin_tool_names(manifests)
    available_async_subagents = _manifest_available_async_subagent_names(manifests)
    async_delegation = _trace_async_delegation_evidence(turns)

    terminal_ok = all(turn.get("status") == "succeeded" for turn in turns)
    response_ok = all(int(turn.get("response_len") or 0) >= 50 for turn in turns)
    async_tools_declared = ASYNC_TASK_TOOL_NAMES.issubset(builtin_names)
    async_subagents_declared = bool(available_async_subagents)
    start_used = int(async_delegation.get("start_call_count") or 0) > 0
    task_id_captured = bool(async_delegation.get("task_ids"))
    monitor_used = int(async_delegation.get("monitor_call_count") or 0) > 0
    statuses_observed = bool(async_delegation.get("statuses"))
    subagent_observed = bool(async_delegation.get("subagent_names"))
    failure_statuses = sorted(
        {
            str(status)
            for status in (async_delegation.get("statuses") or [])
            if str(status).lower() in ASYNC_FAILURE_STATUSES
        }
    )
    no_async_failures = (
        int(async_delegation.get("failure_count") or 0) == 0 and not failure_statuses
    )
    no_recoveries = all(int(turn.get("idle_recovery_count") or 0) == 0 for turn in turns)

    score = 0.0
    if terminal_ok:
        score += 1.0
    else:
        issues.append("one or more turns did not succeed")
    if response_ok:
        score += 0.75
    else:
        issues.append("one or more responses are too short for async delegation trace")
    if async_tools_declared:
        score += 1.25
    else:
        issues.append("async subagent tools were not declared by any captured manifest")
    if async_subagents_declared:
        score += 1.25
    else:
        issues.append("no available async subagents were declared by any captured manifest")
    if start_used:
        score += 1.5
    else:
        issues.append("start_async_task was not used")
    if task_id_captured:
        score += 1.0
    else:
        issues.append("no async task_id was captured")
    if monitor_used:
        score += 1.0
    else:
        issues.append("no async task monitor tool was used")
    if statuses_observed:
        score += 0.75
    else:
        issues.append("no async task statuses were observed")
    if subagent_observed:
        score += 0.75
    else:
        issues.append("no async subagent names were observed")
    if not no_async_failures:
        if int(async_delegation.get("failure_count") or 0) > 0:
            issues.append("async subagent tool failure observed")
        for status in failure_statuses:
            issues.append(f"async task reported terminal failure status: {status}")
        for error in async_delegation.get("errors") or []:
            issues.append(f"async subagent error: {error}")
    if no_recoveries:
        score += 0.75
    else:
        issues.append("run had idle recoveries")

    rounded_score = round(min(score, 10.0), 1)
    return {
        "score": rounded_score,
        "passed": rounded_score >= min_score and not issues,
        "issues": issues,
        "turn_count": len(turns),
        "signals": {
            "terminal_ok": terminal_ok,
            "response_ok": response_ok,
            "async_tools_declared": async_tools_declared,
            "async_subagents_declared": async_subagents_declared,
            "start_used": start_used,
            "task_id_captured": task_id_captured,
            "monitor_used": monitor_used,
            "statuses_observed": statuses_observed,
            "subagent_observed": subagent_observed,
            "no_async_failures": no_async_failures,
            "no_idle_recoveries": no_recoveries,
        },
        "async_delegation": async_delegation,
        "available_async_subagents": sorted(available_async_subagents),
    }


def evaluate_bisque_trace_quality(
    result: dict[str, Any],
    *,
    min_score: float = 8.5,
) -> dict[str, Any]:
    turns = _trace_turn_summaries(result)
    issues: list[str] = []
    if not turns:
        return {
            "score": 0.0,
            "passed": False,
            "issues": ["no trace turns found"],
            "turn_count": 0,
        }

    terminal_ok = all(turn.get("status") == "succeeded" for turn in turns)
    response_ok = all(int(turn.get("response_len") or 0) >= 800 for turn in turns)
    search_ok = _trace_has_tool(turns, {"bisque_search_resources"})
    download_ok = _trace_has_tool(turns, {"bisque_download_resource"})
    upload_ok = _trace_has_tool(
        turns,
        {"bisque_upload_workspace_files", "bisque_upload_files"},
    )
    analysis_ok = _trace_has_tool(
        turns,
        {
            "execute",
            "stage_uploaded_files_for_analysis",
            "stage_artifact_for_analysis",
            "read_file",
            "write_file",
        },
    )
    artifacts_ok = _trace_has_bisque_analysis_artifacts(turns)
    downloads_verified = _trace_artifact_downloads_verified(turns)
    followup_context_ok = len(turns) <= 1 or all(
        _turn_uses_bisque_followup_context(turn) for turn in turns[1:]
    )
    response_mentions_upload = _trace_mentions_bisque_upload(turns)

    score = 0.0
    if terminal_ok:
        score += 1.25
    else:
        issues.append("one or more turns did not succeed")
    if response_ok:
        score += 1.0
    else:
        issues.append("one or more responses are too short for BisQue analysis")
    if search_ok:
        score += 1.25
    else:
        issues.append("missing BisQue search tool evidence")
    if download_ok:
        score += 1.5
    else:
        issues.append("missing BisQue download/import tool evidence")
    if analysis_ok:
        score += 1.25
    else:
        issues.append("missing code execution or staged analysis evidence")
    if upload_ok:
        score += 1.5
    else:
        issues.append("missing BisQue upload tool evidence")
    if artifacts_ok:
        score += 1.0
    else:
        issues.append("missing durable image/report/code analysis artifacts")
    if downloads_verified:
        score += 0.75
    else:
        issues.append("artifact download verification failed or was not captured")
    if followup_context_ok:
        score += 1.0
    else:
        issues.append("follow-ups did not reuse prior BisQue/image/artifact context")
    if response_mentions_upload:
        score += 0.5
    else:
        issues.append("response does not mention uploaded BisQue resources")

    rounded_score = round(min(score, 10.0), 1)
    return {
        "score": rounded_score,
        "passed": rounded_score >= min_score and not issues,
        "issues": issues,
        "turn_count": len(turns),
        "signals": {
            "terminal_ok": terminal_ok,
            "response_ok": response_ok,
            "bisque_search": search_ok,
            "bisque_download": download_ok,
            "analysis_tools": analysis_ok,
            "bisque_upload": upload_ok,
            "analysis_artifacts": artifacts_ok,
            "downloaded_artifacts": downloads_verified,
            "followup_context": followup_context_ok,
            "response_mentions_upload": response_mentions_upload,
        },
    }


def build_tool_capability_matrix(result: dict[str, Any]) -> dict[str, Any]:
    turns = _trace_turn_summaries(result)
    manifests = _trace_tool_capability_manifests(turns)
    builtin_names = _manifest_builtin_tool_names(manifests)
    registered_names = _manifest_registered_tool_names(manifests)
    available_skills = _trace_available_skills(turns, manifests)
    skill_reads = _trace_skill_records(turns, "skill_reads")
    skill_scripts = _trace_skill_records(turns, "skill_scripts")
    available_subagents = _manifest_available_subagent_names(manifests)
    available_async_subagents = _manifest_available_async_subagent_names(manifests)
    scoped_subagent_context_tools = _manifest_scoped_subagent_context_tool_names(manifests)
    scoped_subagent_context_tools_declared = bool(scoped_subagent_context_tools) and all(
        _context_staging_tool_names().issubset(set(tool_names))
        for tool_names in scoped_subagent_context_tools.values()
    )
    declared_names = builtin_names | registered_names | TOOL_CAPABILITY_TOOL_NAMES
    delegation = _trace_delegation_evidence(turns)
    async_delegation = _trace_async_delegation_evidence(turns)
    all_used_tools = {
        str(tool_name)
        for turn in turns
        for tool_name in (turn.get("tool_names") or [])
        if str(tool_name).strip()
    }
    known_builtin_names = {
        "execute",
        "write_file",
        "read_file",
        "edit_file",
        "ls",
        "glob",
        "grep",
        "task",
    } | ASYNC_TASK_TOOL_NAMES
    storage_routes = [
        cast(dict[str, Any], storage)
        for manifest in manifests
        if isinstance((storage := manifest.get("storage")), dict)
    ]
    outputs_declared = any(storage.get("outputs") == "/outputs" for storage in storage_routes)
    skills_used = bool(skill_reads or skill_scripts)
    rarespot_skill_available = RARESPOT_SKILL_NAME in set(available_skills)
    rarespot_skill_used = _skill_records_have_skill(skill_scripts, RARESPOT_SKILL_NAME)

    capabilities = {
        "sandbox_execution": {
            "available": "execute" in builtin_names,
            "used": "execute" in all_used_tools,
        },
        "filesystem": {
            "available": bool(
                {"write_file", "read_file", "edit_file", "ls", "glob", "grep"} & builtin_names
            ),
            "used": bool(
                {"write_file", "read_file", "edit_file", "ls", "glob", "grep"} & all_used_tools
            ),
        },
        "context_staging": {
            "available": bool(_context_staging_tool_names() & registered_names),
            "used": bool(_context_staging_tool_names() & all_used_tools),
        },
        "durable_outputs": {
            "available": outputs_declared,
            "used": bool(_trace_artifacts(turns)),
        },
        "skills": {
            "available": bool(available_skills),
            "used": skills_used,
        },
        "delegation": {
            "available": "task" in builtin_names and bool(available_subagents),
            "used": bool(delegation["subagent_names"]),
            "available_subagents": sorted(available_subagents),
            "scoped_subagent_context_tools_declared": scoped_subagent_context_tools_declared,
            "scoped_subagent_context_tools": scoped_subagent_context_tools,
        },
        "background_delegation": {
            "available": ASYNC_TASK_TOOL_NAMES.issubset(builtin_names)
            and bool(available_async_subagents),
            "used": int(async_delegation.get("start_call_count") or 0) > 0,
            "available_async_subagents": sorted(available_async_subagents),
            "start_call_count": int(async_delegation.get("start_call_count") or 0),
            "monitor_call_count": int(async_delegation.get("monitor_call_count") or 0),
            "failure_count": int(async_delegation.get("failure_count") or 0),
            "statuses": list(async_delegation.get("statuses") or []),
            "errors": list(async_delegation.get("errors") or []),
        },
        "paper_review": {
            "available": bool(PAPER_TOOL_NAMES & registered_names),
            "used": bool(PAPER_TOOL_NAMES & all_used_tools),
        },
        "rarespot_detection": {
            "available": rarespot_skill_available,
            "used": rarespot_skill_used,
        },
    }
    turn_summaries: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        turn_manifests = [
            raw for raw in (turn.get("tool_capability_manifests") or []) if isinstance(raw, dict)
        ]
        turn_available_subagents = _manifest_available_subagent_names(turn_manifests)
        turn_available_async_subagents = _manifest_available_async_subagent_names(turn_manifests)
        turn_scoped_subagent_context_tools = _manifest_scoped_subagent_context_tool_names(
            turn_manifests
        )
        turn_tools = {
            str(tool_name) for tool_name in (turn.get("tool_names") or []) if str(tool_name).strip()
        }
        turn_skills_used = bool(turn.get("skill_reads") or turn.get("skill_scripts"))
        turn_summaries.append(
            {
                "run_id": turn.get("run_id"),
                "status": turn.get("status"),
                "tool_names": sorted(turn_tools),
                "available_subagents": sorted(turn_available_subagents),
                "available_async_subagents": sorted(turn_available_async_subagents),
                "scoped_subagent_context_tools": turn_scoped_subagent_context_tools,
                "artifact_count": len(
                    [
                        raw
                        for raw in [
                            *(turn.get("artifacts") or []),
                            *(turn.get("linked_artifacts") or []),
                        ]
                        if isinstance(raw, dict)
                    ]
                ),
                "capabilities": {
                    "sandbox_execution": "execute" in turn_tools,
                    "filesystem": bool(
                        {"write_file", "read_file", "edit_file", "ls", "glob", "grep"} & turn_tools
                    ),
                    "context_staging": bool(_context_staging_tool_names() & turn_tools),
                    "skills": turn_skills_used,
                    "followup_context_reuse": index > 0
                    and bool(
                        {
                            "artifact_manifest",
                            "stage_artifact_for_analysis",
                            "read_file",
                            "grep",
                            "glob",
                            "ls",
                        }
                        & turn_tools
                    ),
                    "delegation": "task" in turn_tools
                    or bool(
                        (
                            turn.get("delegation", {})
                            if isinstance(turn.get("delegation"), dict)
                            else {}
                        ).get("subagent_names")
                    ),
                    "background_delegation": bool(ASYNC_TASK_TOOL_NAMES & turn_tools)
                    or int(
                        (
                            turn.get("async_delegation", {})
                            if isinstance(turn.get("async_delegation"), dict)
                            else {}
                        ).get("start_call_count")
                        or 0
                    )
                    > 0,
                    "paper_review": bool(PAPER_TOOL_NAMES & turn_tools),
                    "rarespot_detection": _turn_has_rarespot_skill_script(turn),
                },
            }
        )

    return {
        "turn_count": len(turns),
        "manifest_count": len(manifests),
        "declared_builtin_tools": sorted(builtin_names),
        "declared_registered_tools": sorted(registered_names),
        "available_skills": available_skills,
        "skill_reads": skill_reads,
        "skill_scripts": skill_scripts,
        "used_tools": sorted(all_used_tools),
        "unknown_tools": sorted(all_used_tools - declared_names - known_builtin_names),
        "capabilities": capabilities,
        "turns": turn_summaries,
    }


def evaluate_thread_trace_quality(
    result: dict[str, Any],
    *,
    min_score: float = 8.5,
) -> dict[str, Any]:
    turns = _trace_turn_summaries(result)
    messages = result.get("thread_messages")
    thread = result.get("thread")
    issues: list[str] = []
    if not turns:
        return {
            "score": 0.0,
            "passed": False,
            "issues": ["no trace turns found"],
            "turn_count": 0,
        }
    if not isinstance(messages, dict):
        return {
            "score": 0.0,
            "passed": False,
            "issues": ["thread messages were not captured"],
            "turn_count": len(turns),
        }

    expected_roles = [role for _turn in turns for role in ("user", "assistant")]
    roles = [str(role) for role in messages.get("roles") or []]
    run_ids = [str(run_id) for run_id in messages.get("run_ids") or []]
    content_lengths = [int(length or 0) for length in messages.get("content_lengths") or []]
    turn_run_ids = [str(turn.get("run_id") or "") for turn in turns]

    score = 0.0
    if roles == expected_roles:
        score += 2.0
    else:
        issues.append("thread transcript does not contain one user/assistant pair per turn")

    if not any(role in {"tool", "system", "developer"} for role in roles):
        score += 1.5
    else:
        issues.append("thread transcript contains tool/internal messages")

    pair_run_ids_ok = True
    missing_assistant_runs: list[str] = []
    for index, run_id in enumerate(turn_run_ids):
        user_index = 2 * index
        assistant_index = user_index + 1
        if (
            assistant_index >= len(run_ids)
            or assistant_index >= len(roles)
            or roles[assistant_index] != "assistant"
            or run_ids[assistant_index] != run_id
        ):
            pair_run_ids_ok = False
            missing_assistant_runs.append(run_id or "<unknown>")
            continue
        if user_index >= len(run_ids) or run_ids[user_index] != run_id:
            pair_run_ids_ok = False
    if pair_run_ids_ok:
        score += 2.0
    else:
        for run_id in missing_assistant_runs:
            issues.append(f"assistant message missing for run {run_id}")
        if not missing_assistant_runs:
            issues.append("thread message run IDs do not match trace turn run IDs")

    assistant_lengths = [
        content_lengths[index]
        for index, role in enumerate(roles)
        if role == "assistant" and index < len(content_lengths)
    ]
    if len(assistant_lengths) == len(turns) and all(length > 0 for length in assistant_lengths):
        score += 1.5
    else:
        issues.append("one or more assistant messages are empty")

    terminal_ok = all(turn.get("status") == "succeeded" for turn in turns)
    if terminal_ok:
        score += 1.0
    else:
        issues.append("one or more trace turns did not succeed")

    latest_run_id = str(thread.get("latest_run_id") or "") if isinstance(thread, dict) else ""
    final_run_id = turn_run_ids[-1] if turn_run_ids else ""
    if latest_run_id == final_run_id:
        score += 2.0
    else:
        issues.append("thread latest_run_id does not match final user-facing run")

    rounded_score = round(min(score, 10.0), 1)
    return {
        "score": rounded_score,
        "passed": rounded_score >= min_score and not issues,
        "issues": issues,
        "turn_count": len(turns),
        "signals": {
            "roles_alternate": roles == expected_roles,
            "no_tool_messages": not any(role in {"tool", "system", "developer"} for role in roles),
            "run_ids_match": pair_run_ids_ok,
            "assistant_messages_non_empty": len(assistant_lengths) == len(turns)
            and all(length > 0 for length in assistant_lengths),
            "terminal_ok": terminal_ok,
            "latest_run_matches_final": latest_run_id == final_run_id,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trace a live Go/NATS/Deep Agents run.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--title", default="Deep Agents live trace")
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--followup",
        action="append",
        default=[],
        help="Follow-up prompt to run after the initial prompt. Repeat for multi-turn traces.",
    )
    parser.add_argument(
        "--upload-file",
        action="append",
        default=[],
        help="Path to a local file to upload through /v2/uploads before creating the run.",
    )
    parser.add_argument(
        "--upload-calphad-manifest",
        action="append",
        default=[],
        help=(
            "Reviewed CALPHAD registry manifest corresponding positionally to one "
            "--upload-file. The trace verifies exact bytes and persists bounded owner "
            "source/license/assessment/reference-state/temperature declarations before "
            "creating the run."
        ),
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--http-timeout", type=float, default=10.0)
    parser.add_argument(
        "--auth-cookie-env",
        default="ULTRA_LIVE_TRACE_COOKIE",
        help=(
            "Environment variable containing a Cookie header for authenticated "
            "WorkOS/dev live traces. Set to an empty string to disable."
        ),
    )
    parser.add_argument(
        "--authorization-env",
        default="ULTRA_LIVE_TRACE_AUTHORIZATION",
        help=(
            "Environment variable containing an Authorization header for authenticated "
            "live traces. Set to an empty string to disable."
        ),
    )
    parser.add_argument(
        "--workflow-hint-id",
        default="",
        help=(
            "Optional workflow hint id sent on run creation (e.g. pro_mode to "
            "exercise Intelligence Pro rigor enforcement)."
        ),
    )
    parser.add_argument(
        "--suggested-domain",
        default="",
        help=(
            "Optional selection_context.suggested_domain sent on every run "
            "(for example materials to exercise domain routing explicitly)."
        ),
    )
    parser.add_argument("--verify-downloads", action="store_true")
    parser.add_argument("--require-downloads", action="store_true")
    parser.add_argument("--min-artifacts", type=int, default=0)
    parser.add_argument("--min-response-chars", type=int, default=0)
    parser.add_argument("--paper-quality", action="store_true")
    parser.add_argument("--require-paper-quality", action="store_true")
    parser.add_argument("--min-paper-quality-score", type=float, default=8.5)
    parser.add_argument("--rarespot-quality", action="store_true")
    parser.add_argument("--require-rarespot-quality", action="store_true")
    parser.add_argument("--min-rarespot-quality-score", type=float, default=8.5)
    parser.add_argument("--materials-quality", action="store_true")
    parser.add_argument("--require-materials-quality", action="store_true")
    parser.add_argument("--min-materials-quality-score", type=float, default=9.0)
    parser.add_argument("--materials-performance", action="store_true")
    parser.add_argument("--require-materials-performance", action="store_true")
    parser.add_argument("--materials-max-model-calls", type=int, default=8)
    parser.add_argument("--materials-max-input-tokens", type=int, default=250_000)
    parser.add_argument("--materials-max-tool-calls", type=int, default=12)
    parser.add_argument("--materials-max-input-amplification", type=float, default=6.0)
    parser.add_argument(
        "--materials-max-terminal-seconds",
        type=float,
        default=0.0,
        help="Optional per-turn terminal latency tripwire; 0 disables the latency limit.",
    )
    parser.add_argument(
        "--materials-evidence-dir",
        type=Path,
        help=(
            "Retain the exact downloaded materials_validation.json bytes in this "
            "content-addressed directory for independent promotion revalidation."
        ),
    )
    parser.add_argument("--tool-autonomy-quality", action="store_true")
    parser.add_argument("--require-tool-autonomy-quality", action="store_true")
    parser.add_argument("--min-tool-autonomy-quality-score", type=float, default=8.5)
    parser.add_argument("--tool-capability-quality", action="store_true")
    parser.add_argument("--require-tool-capability-quality", action="store_true")
    parser.add_argument("--min-tool-capability-quality-score", type=float, default=8.5)
    parser.add_argument("--delegation-quality", action="store_true")
    parser.add_argument("--require-delegation-quality", action="store_true")
    parser.add_argument("--min-delegation-quality-score", type=float, default=8.5)
    parser.add_argument("--async-delegation-quality", action="store_true")
    parser.add_argument("--require-async-delegation-quality", action="store_true")
    parser.add_argument("--min-async-delegation-quality-score", type=float, default=8.5)
    parser.add_argument("--bisque-quality", action="store_true")
    parser.add_argument("--require-bisque-quality", action="store_true")
    parser.add_argument("--min-bisque-quality-score", type=float, default=8.5)
    parser.add_argument(
        "--bisque-username-env",
        default="",
        help="Environment variable containing a BisQue username for linked-account traces.",
    )
    parser.add_argument(
        "--bisque-password-env",
        default="",
        help="Environment variable containing a BisQue password for linked-account traces.",
    )
    parser.add_argument("--capability-matrix", action="store_true")
    parser.add_argument("--expected-rarespot-tile-size", type=int, default=512)
    parser.add_argument("--expected-rarespot-tile-overlap", type=float, default=0.25)
    parser.add_argument("--expected-rarespot-stride", type=int, default=384)
    parser.add_argument("--thread-quality", action="store_true")
    parser.add_argument("--require-thread-quality", action="store_true")
    parser.add_argument("--min-thread-quality-score", type=float, default=8.5)
    args = parser.parse_args(argv)
    try:
        result = run_trace(args)
        issues = evaluate_trace_requirements(
            result,
            min_artifacts=args.min_artifacts,
            min_response_chars=args.min_response_chars,
            require_downloads=args.require_downloads,
        )
        if args.paper_quality or args.require_paper_quality:
            paper_quality = evaluate_paper_trace_quality(
                result,
                min_score=args.min_paper_quality_score,
            )
            result["paper_quality"] = paper_quality
            if args.require_paper_quality and not paper_quality["passed"]:
                issues.extend(f"paper quality: {issue}" for issue in paper_quality["issues"])
        if args.rarespot_quality or args.require_rarespot_quality:
            rarespot_quality = evaluate_rarespot_trace_quality(
                result,
                min_score=args.min_rarespot_quality_score,
                expected_tile_size=args.expected_rarespot_tile_size,
                expected_tile_overlap=args.expected_rarespot_tile_overlap,
                expected_stride=args.expected_rarespot_stride,
            )
            result["rarespot_quality"] = rarespot_quality
            if args.require_rarespot_quality and not rarespot_quality["passed"]:
                issues.extend(f"rarespot quality: {issue}" for issue in rarespot_quality["issues"])
        if args.materials_quality or args.require_materials_quality:
            materials_quality = evaluate_materials_trace_quality(
                result,
                min_score=args.min_materials_quality_score,
            )
            result["materials_quality"] = materials_quality
            if args.require_materials_quality and not materials_quality["passed"]:
                issues.extend(
                    f"materials quality: {issue}" for issue in materials_quality["issues"]
                )
        if args.materials_performance or args.require_materials_performance:
            max_terminal_ms = (
                args.materials_max_terminal_seconds * 1000.0
                if args.materials_max_terminal_seconds > 0
                else None
            )
            materials_performance = evaluate_materials_trace_performance(
                result,
                max_model_calls_per_turn=args.materials_max_model_calls,
                max_input_tokens_per_turn=args.materials_max_input_tokens,
                max_tool_calls_per_turn=args.materials_max_tool_calls,
                max_input_amplification_vs_peak=(args.materials_max_input_amplification),
                max_terminal_ms_per_turn=max_terminal_ms,
            )
            result["materials_performance"] = materials_performance
            if args.require_materials_performance and not materials_performance["passed"]:
                issues.extend(
                    f"materials performance: {issue}" for issue in materials_performance["issues"]
                )
        if args.tool_autonomy_quality or args.require_tool_autonomy_quality:
            tool_autonomy_quality = evaluate_tool_autonomy_trace_quality(
                result,
                min_score=args.min_tool_autonomy_quality_score,
            )
            result["tool_autonomy_quality"] = tool_autonomy_quality
            if args.require_tool_autonomy_quality and not tool_autonomy_quality["passed"]:
                issues.extend(
                    f"tool autonomy quality: {issue}" for issue in tool_autonomy_quality["issues"]
                )
        if args.tool_capability_quality or args.require_tool_capability_quality:
            tool_capability_quality = evaluate_tool_capability_trace_quality(
                result,
                min_score=args.min_tool_capability_quality_score,
            )
            result["tool_capability_quality"] = tool_capability_quality
            if args.require_tool_capability_quality and not tool_capability_quality["passed"]:
                issues.extend(
                    f"tool capability quality: {issue}"
                    for issue in tool_capability_quality["issues"]
                )
        if args.delegation_quality or args.require_delegation_quality:
            delegation_quality = evaluate_delegation_trace_quality(
                result,
                min_score=args.min_delegation_quality_score,
            )
            result["delegation_quality"] = delegation_quality
            if args.require_delegation_quality and not delegation_quality["passed"]:
                issues.extend(
                    f"delegation quality: {issue}" for issue in delegation_quality["issues"]
                )
        if args.async_delegation_quality or args.require_async_delegation_quality:
            async_delegation_quality = evaluate_async_delegation_trace_quality(
                result,
                min_score=args.min_async_delegation_quality_score,
            )
            result["async_delegation_quality"] = async_delegation_quality
            if args.require_async_delegation_quality and not async_delegation_quality["passed"]:
                issues.extend(
                    f"async delegation quality: {issue}"
                    for issue in async_delegation_quality["issues"]
                )
        if args.capability_matrix:
            result["capability_matrix"] = build_tool_capability_matrix(result)
        if args.bisque_quality or args.require_bisque_quality:
            bisque_quality = evaluate_bisque_trace_quality(
                result,
                min_score=args.min_bisque_quality_score,
            )
            result["bisque_quality"] = bisque_quality
            if args.require_bisque_quality and not bisque_quality["passed"]:
                issues.extend(f"bisque quality: {issue}" for issue in bisque_quality["issues"])
        if args.thread_quality or args.require_thread_quality:
            thread_quality = evaluate_thread_trace_quality(
                result,
                min_score=args.min_thread_quality_score,
            )
            result["thread_quality"] = thread_quality
            if args.require_thread_quality and not thread_quality["passed"]:
                issues.extend(f"thread quality: {issue}" for issue in thread_quality["issues"])
        if issues:
            result["issues"] = issues
        print(json.dumps(result, indent=2, sort_keys=True))
        if issues:
            return 2
    except Exception as exc:
        print(json.dumps({"error": str(exc), "error_type": type(exc).__name__}, indent=2))
        return 1
    return 0


def _unwrap(payload: dict[str, Any], key: str) -> dict[str, Any]:
    nested = payload.get(key)
    return nested if isinstance(nested, dict) else payload


def _followup_prompts(args: argparse.Namespace) -> list[str]:
    raw = getattr(args, "followup", [])
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, list | tuple):
        return [str(item) for item in raw if str(item).strip()]
    return [str(raw)] if str(raw).strip() else []


def _workflow_hint_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    hint_id = str(getattr(args, "workflow_hint_id", "") or "").strip()
    if not hint_id:
        return None
    return {"id": hint_id, "source": "live_trace"}


def _selection_context_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    suggested_domain = str(getattr(args, "suggested_domain", "") or "").strip().lower()
    if not suggested_domain:
        return None
    return {"suggested_domain": suggested_domain, "source": "live_trace"}


def _auth_headers_from_args(args: argparse.Namespace) -> dict[str, str]:
    headers: dict[str, str] = {}
    cookie_env = str(getattr(args, "auth_cookie_env", "ULTRA_LIVE_TRACE_COOKIE") or "").strip()
    authorization_env = str(
        getattr(args, "authorization_env", "ULTRA_LIVE_TRACE_AUTHORIZATION") or ""
    ).strip()
    if cookie_env:
        cookie = str(os.environ.get(cookie_env) or "").strip()
        if cookie:
            headers["Cookie"] = cookie
    if authorization_env:
        authorization = str(os.environ.get(authorization_env) or "").strip()
        if authorization:
            headers["Authorization"] = authorization
    return headers


def _clean_extra_headers(headers: dict[str, str]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, value in headers.items():
        normalized_key = str(key or "").strip()
        normalized_value = str(value or "").strip()
        if normalized_key and normalized_value:
            cleaned[normalized_key] = normalized_value
    return cleaned


def _login_bisque_from_args(
    client: ControlPlaneClient,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    username_env = str(getattr(args, "bisque_username_env", "") or "").strip()
    password_env = str(getattr(args, "bisque_password_env", "") or "").strip()
    if not username_env and not password_env:
        return None
    if not username_env or not password_env:
        raise ValueError("both --bisque-username-env and --bisque-password-env are required")
    username = str(os.environ.get(username_env) or "").strip()
    password = str(os.environ.get(password_env) or "")
    if not username:
        raise ValueError(f"BisQue username env var {username_env} is empty")
    if not password:
        raise ValueError(f"BisQue password env var {password_env} is empty")
    session = client.login_bisque_account(username=username, password=password)
    user = session.get("user")
    username_from_session = ""
    if isinstance(user, dict):
        username_from_session = str(
            user.get("username") or user.get("name") or user.get("email") or ""
        ).strip()
    return {
        "authenticated": bool(session.get("authenticated")),
        "mode": str(session.get("mode") or ""),
        "username": username_from_session or username,
    }


def _event_sequence(event: dict[str, Any]) -> int:
    try:
        return int(event.get("sequence") or 0)
    except (TypeError, ValueError):
        return 0


def _response_preview(response_text: str, max_chars: int = 4000) -> str:
    normalized = response_text.strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "…"


def _response_tail(response_text: str, max_chars: int = 4000) -> str:
    normalized = response_text.strip()
    if len(normalized) <= max_chars:
        return ""
    return "…" + normalized[-(max_chars - 1) :].lstrip()


def _tool_names(events: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for event in events:
        if event.get("event_kind") != "tool_call.started":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        tool_name = str(
            payload.get("tool_name") or payload.get("name") or payload.get("tool") or ""
        ).strip()
        if tool_name:
            names.append(tool_name)
    return names


def _token_usage_evidence(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize unique per-model-call usage without trusting redeliveries.

    Worker events carry a stable ``usage_event_id``. NATS/control-plane replay may
    expose the same event more than once, so raw event count is not a model-call
    count. Conflicting duplicate IDs and malformed/non-additive totals are surfaced
    explicitly and excluded from the scientific performance totals.
    """

    records_by_id: dict[str, tuple[int, int, int]] = {}
    duplicate_event_count = 0
    conflicting_duplicate_count = 0
    invalid_event_count = 0
    anonymous_index = 0
    ordered_records: list[tuple[int, int, int]] = []

    for event in events:
        if event.get("event_kind") != "run.token_usage":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            invalid_event_count += 1
            continue
        input_tokens = _strict_nonnegative_usage_count(payload.get("input_tokens"))
        output_tokens = _strict_nonnegative_usage_count(payload.get("output_tokens"))
        total_tokens = _strict_nonnegative_usage_count(payload.get("total_tokens"))
        if (
            input_tokens is None
            or output_tokens is None
            or total_tokens is None
            or total_tokens != input_tokens + output_tokens
        ):
            invalid_event_count += 1
            continue
        record = (input_tokens, output_tokens, total_tokens)
        usage_event_id = str(payload.get("usage_event_id") or "").strip()
        if not usage_event_id:
            anonymous_index += 1
            usage_event_id = f"<anonymous:{anonymous_index}>"
        existing = records_by_id.get(usage_event_id)
        if existing is not None:
            duplicate_event_count += 1
            if existing != record:
                conflicting_duplicate_count += 1
            continue
        records_by_id[usage_event_id] = record
        ordered_records.append(record)

    input_counts = [record[0] for record in ordered_records]
    output_counts = [record[1] for record in ordered_records]
    total_counts = [record[2] for record in ordered_records]
    cumulative_input = sum(input_counts)
    peak_input = max(input_counts, default=0)
    return {
        "model_call_count": len(ordered_records),
        "input_tokens": cumulative_input,
        "output_tokens": sum(output_counts),
        "total_tokens": sum(total_counts),
        "first_input_tokens": input_counts[0] if input_counts else 0,
        "last_input_tokens": input_counts[-1] if input_counts else 0,
        "peak_input_tokens": peak_input,
        "input_amplification_vs_peak": (
            round(cumulative_input / peak_input, 6) if peak_input else 0.0
        ),
        "duplicate_event_count": duplicate_event_count,
        "conflicting_duplicate_count": conflicting_duplicate_count,
        "invalid_event_count": invalid_event_count,
    }


def _strict_nonnegative_usage_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        integer = int(value)
        return integer if integer >= 0 else None
    return None


def _tool_lifecycle_evidence(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Pair tool starts and terminals without trusting tool-name-only evidence."""

    starts: dict[tuple[str, str], dict[str, Any]] = {}
    terminal_keys: set[tuple[str, str]] = set()
    matched_skill_reads: set[tuple[str, str]] = set()
    execute_completions: list[dict[str, Any]] = []
    typed_scientific_completions: list[dict[str, Any]] = []
    failed_terminal_count = 0
    orphan_terminal_count = 0
    duplicate_start_count = 0

    for event in events:
        event_kind = str(event.get("event_kind") or "")
        if event_kind not in {
            "tool_call.started",
            "tool_call.completed",
            "tool_call.failed",
        }:
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            if event_kind in {"tool_call.completed", "tool_call.failed"}:
                orphan_terminal_count += 1
                if event_kind == "tool_call.failed":
                    failed_terminal_count += 1
            continue
        tool_name = str(
            payload.get("tool_name") or payload.get("name") or payload.get("tool") or ""
        ).strip()
        tool_call_id = str(
            payload.get("tool_call_id") or payload.get("call_id") or payload.get("id") or ""
        ).strip()
        if not tool_name or not tool_call_id:
            if event_kind in {"tool_call.completed", "tool_call.failed"}:
                orphan_terminal_count += 1
                if event_kind == "tool_call.failed":
                    failed_terminal_count += 1
            continue
        key = (tool_call_id, tool_name)
        if event_kind == "tool_call.started":
            if key in starts:
                duplicate_start_count += 1
            else:
                starts[key] = payload
            continue

        if event_kind == "tool_call.failed":
            failed_terminal_count += 1
        started_payload = starts.get(key)
        if started_payload is None:
            orphan_terminal_count += 1
            continue
        terminal_keys.add(key)
        if event_kind != "tool_call.completed":
            continue

        if tool_name == "read_file":
            for text in _payload_strings(started_payload):
                for match in SKILL_READ_PATTERN.finditer(text):
                    skill = match.group(1).strip()
                    if skill:
                        matched_skill_reads.add((skill, tool_call_id))
        elif tool_name == "execute":
            start_digest = _immutable_runtime_image_digest(
                started_payload.get("runtime_image_digest")
            )
            terminal_digest = _immutable_runtime_image_digest(payload.get("runtime_image_digest"))
            exit_code_observed = "exit_code" in payload
            exit_code = _strict_tool_exit_code(payload.get("exit_code"))
            digest_matched = bool(start_digest and start_digest == terminal_digest)
            exit_code_ok = not exit_code_observed or exit_code == 0
            execute_completions.append(
                {
                    "tool_call_id": tool_call_id,
                    "runtime_image_digest": terminal_digest,
                    "runtime_image_digest_matched": digest_matched,
                    "exit_code_observed": exit_code_observed,
                    "exit_code": exit_code,
                    "exit_code_ok": exit_code_ok,
                    "successful": digest_matched and exit_code_ok,
                }
            )
        elif tool_name in MATERIALS_TYPED_SCIENTIFIC_TOOL_NAMES:
            record: dict[str, Any] = {
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
            }
            operation = str(payload.get("scientific_operation") or "").strip().lower()
            result_sha256 = str(payload.get("result_artifact_sha256") or "").strip().lower()
            allowed_operations = MATERIALS_TYPED_SCIENTIFIC_TOOL_OPERATIONS[tool_name]
            if (
                payload.get("scientific_result_ok") is True
                and operation in allowed_operations
                and re.fullmatch(r"[0-9a-f]{64}", result_sha256)
            ):
                record.update(
                    {
                        "scientific_operation": operation,
                        "result_artifact_sha256": result_sha256,
                        "scientific_result_ok": True,
                    }
                )
                validation_sha256 = (
                    str(payload.get("materials_validation_artifact_sha256") or "").strip().lower()
                )
                if re.fullmatch(r"[0-9a-f]{64}", validation_sha256):
                    record["materials_validation_artifact_sha256"] = validation_sha256
            typed_scientific_completions.append(record)

    return {
        "started_call_count": len(starts),
        "matched_terminal_count": len(terminal_keys),
        "started_only_count": len(set(starts).difference(terminal_keys)),
        "orphan_terminal_count": orphan_terminal_count,
        "duplicate_start_count": duplicate_start_count,
        "failed_terminal_count": failed_terminal_count,
        "matched_skill_read_completions": [
            {"skill": skill, "tool_call_id": tool_call_id}
            for skill, tool_call_id in sorted(matched_skill_reads)
        ],
        "matched_execute_completions": execute_completions,
        "matched_typed_scientific_completions": typed_scientific_completions,
    }


def _immutable_runtime_image_digest(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if IMMUTABLE_RUNTIME_IMAGE_PATTERN.fullmatch(normalized) else ""


def _strict_tool_exit_code(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value or "").strip()
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return None


def _skill_reads(events: list[dict[str, Any]]) -> list[dict[str, str]]:
    records: set[tuple[str, str]] = set()
    for event in events:
        if event.get("event_kind") != "tool_call.started":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        tool_name = str(
            payload.get("tool_name") or payload.get("name") or payload.get("tool") or ""
        )
        if tool_name != "read_file":
            continue
        for text in _payload_strings(payload):
            for match in SKILL_READ_PATTERN.finditer(text):
                skill = match.group(1).strip()
                if skill:
                    records.add((skill, f"/skills/{skill}/SKILL.md"))
    return [{"skill": skill, "path": path} for skill, path in sorted(records)]


def _skill_scripts(events: list[dict[str, Any]]) -> list[dict[str, str]]:
    records: set[tuple[str, str]] = set()
    for event in events:
        if event.get("event_kind") != "tool_call.started":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        tool_name = str(
            payload.get("tool_name") or payload.get("name") or payload.get("tool") or ""
        )
        if tool_name != "execute":
            continue
        text = "\n".join(_payload_strings(payload))
        for script_path, skill_name in SKILL_SCRIPT_PATHS.items():
            if script_path in text:
                records.add((skill_name, script_path))
    return [{"skill": skill, "path": path} for skill, path in sorted(records)]


def _available_skills(
    manifests: list[dict[str, Any]],
    skill_reads: list[dict[str, str]],
    skill_scripts: list[dict[str, str]],
) -> list[str]:
    names = _manifest_available_skill_names(manifests)
    for record in (*skill_reads, *skill_scripts):
        skill = str(record.get("skill") or "").strip()
        if skill:
            names.add(skill)
    return sorted(names)


def _payload_strings(value: Any) -> list[str]:
    strings: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            strings.extend(_payload_strings(item))
        return strings
    if isinstance(value, list | tuple | set):
        for item in value:
            strings.extend(_payload_strings(item))
        return strings
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            strings.append(stripped)
    return strings


def _rarespot_configurations(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    configurations: list[dict[str, Any]] = []
    for event in events:
        if event.get("event_kind") != "tool_call.completed":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        output_text = str(payload.get("output_preview") or payload.get("output") or "")
        parsed = _json_object_from_text(output_text)
        if not parsed:
            continue
        if "configuration" not in parsed:
            continue
        config = parsed.get("configuration")
        if not isinstance(config, dict):
            continue
        normalized = _normalize_rarespot_configuration(config)
        if normalized:
            configurations.append(normalized)
    return configurations


def _tool_capability_manifests(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for event in events:
        if event.get("event_kind") != "tool_call.completed":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        tool_name = str(
            payload.get("tool_name") or payload.get("name") or payload.get("tool") or ""
        )
        if tool_name not in TOOL_CAPABILITY_TOOL_NAMES:
            continue
        output_text = str(payload.get("output_preview") or payload.get("output") or "")
        parsed = _json_object_from_text(output_text)
        if not parsed:
            continue
        if "deepagents_builtin_tools" not in parsed or "registered_tools" not in parsed:
            continue
        manifests.append(parsed)
    return manifests


def _context_tool_output_hygiene(events: list[dict[str, Any]]) -> dict[str, Any]:
    checked_output_count = 0
    leaks: list[str] = []
    for event in events:
        if event.get("event_kind") != "tool_call.completed":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        tool_name = str(
            payload.get("tool_name") or payload.get("name") or payload.get("tool") or ""
        )
        if tool_name not in _context_staging_tool_names():
            continue
        checked_output_count += 1
        output_text = str(payload.get("output_preview") or payload.get("output") or "")
        parsed = _json_object_from_text(output_text)
        if parsed is None:
            leaks.extend(_raw_context_tool_host_path_leaks(tool_name, output_text))
            continue
        leaks.extend(_context_tool_host_path_leaks(tool_name, parsed))
    return {
        "checked_output_count": checked_output_count,
        "host_path_leak_count": len(leaks),
        "safe": not leaks,
        "leaks": sorted(leaks),
    }


def _context_tool_host_path_leaks(
    tool_name: str,
    value: Any,
    *,
    location: str = "$",
) -> list[str]:
    leaks: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            child_location = f"{location}.{key_text}" if location else key_text
            if key_text in _CONTEXT_TOOL_PRIVATE_PATH_KEYS:
                leaks.append(f"{tool_name}: {child_location}={_short_leak_value(item)}")
                continue
            leaks.extend(
                _context_tool_host_path_leaks(
                    tool_name,
                    item,
                    location=child_location,
                )
            )
        return leaks
    if isinstance(value, list):
        for index, item in enumerate(value):
            leaks.extend(
                _context_tool_host_path_leaks(
                    tool_name,
                    item,
                    location=f"{location}[{index}]",
                )
            )
        return leaks
    if isinstance(value, str) and _looks_like_host_path_leak(value):
        leaks.append(f"{tool_name}: {location}={_short_leak_value(value)}")
    return leaks


def _raw_context_tool_host_path_leaks(tool_name: str, output_text: str) -> list[str]:
    leaks: list[str] = []
    for match in _HOST_PATH_TEXT_PATTERN.finditer(output_text):
        leaks.append(f"{tool_name}: raw={_short_leak_value(match.group(0))}")
    if "file://" in output_text:
        leaks.append(f"{tool_name}: raw=file://...")
    return sorted(set(leaks))


def _looks_like_host_path_leak(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if text.startswith("file://"):
        return True
    if re.match(r"^[A-Za-z]:[\\/]", text):
        return True
    return any(text.startswith(prefix) for prefix in _HOST_PATH_PREFIXES)


def _short_leak_value(value: Any, *, limit: int = 120) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _delegation_evidence(events: list[dict[str, Any]]) -> dict[str, Any]:
    task_call_count = 0
    subagent_names: set[str] = set()
    subagent_tool_names: set[str] = set()
    subagent_message_delta_count = 0
    scoped_tool_event_count = 0
    scoped_usage_event_count = 0

    for event in events:
        event_kind = str(event.get("event_kind") or "")
        agent_role = str(event.get("agent_role") or "").strip()
        payload = event.get("payload")
        payload_dict = payload if isinstance(payload, dict) else {}

        if event_kind == "tool_call.started":
            tool_name = str(
                payload_dict.get("tool_name")
                or payload_dict.get("name")
                or payload_dict.get("tool")
                or ""
            ).strip()
            if tool_name == "task":
                task_call_count += 1
                subagent_type = str(payload_dict.get("subagent_type") or "").strip()
                if subagent_type:
                    subagent_names.add(subagent_type)
            subagent_name = str(payload_dict.get("subagent_name") or "").strip()
            namespace = payload_dict.get("namespace")
            namespace_name = ""
            if isinstance(namespace, list) and namespace:
                namespace_name = str(namespace[-1] or "").strip()
            scoped_subagent_name = subagent_name or namespace_name
            if scoped_subagent_name:
                scoped_tool_event_count += 1
                subagent_names.add(scoped_subagent_name)
                if tool_name:
                    subagent_tool_names.add(tool_name)
            elif agent_role and agent_role not in {"tool", "assistant"}:
                scoped_tool_event_count += 1
                subagent_names.add(agent_role)
                if tool_name:
                    subagent_tool_names.add(tool_name)

        if event_kind == "run.token_usage":
            subagent_name = str(payload_dict.get("subagent_name") or "").strip()
            namespace = payload_dict.get("namespace")
            namespace_name = ""
            if isinstance(namespace, list) and namespace:
                namespace_name = str(namespace[-1] or "").strip()
            scoped_subagent_name = subagent_name or namespace_name
            if scoped_subagent_name:
                scoped_usage_event_count += 1
                subagent_names.add(scoped_subagent_name)
            elif agent_role and agent_role not in {"assistant", "coordinator", "tool"}:
                scoped_usage_event_count += 1
                subagent_names.add(agent_role)

        if event_kind == "subagent.message.delta":
            subagent_message_delta_count += 1
            subagent_name = str(payload_dict.get("subagent_name") or "").strip()
            if subagent_name:
                subagent_names.add(subagent_name)
            elif agent_role:
                subagent_names.add(agent_role)

    return {
        "task_call_count": task_call_count,
        "subagent_names": sorted(subagent_names),
        "subagent_tool_names": sorted(subagent_tool_names),
        "subagent_message_delta_count": subagent_message_delta_count,
        "scoped_tool_event_count": scoped_tool_event_count,
        "scoped_usage_event_count": scoped_usage_event_count,
    }


def _async_delegation_evidence(events: list[dict[str, Any]]) -> dict[str, Any]:
    start_call_count = 0
    check_call_count = 0
    update_call_count = 0
    cancel_call_count = 0
    list_call_count = 0
    completed_result_count = 0
    task_ids: set[str] = set()
    subagent_names: set[str] = set()
    statuses: set[str] = set()
    tool_names: set[str] = set()
    failure_count = 0
    errors: set[str] = set()

    for event in events:
        event_kind = str(event.get("event_kind") or "")
        payload = event.get("payload")
        payload_dict = payload if isinstance(payload, dict) else {}
        tool_name = str(
            payload_dict.get("tool_name")
            or payload_dict.get("name")
            or payload_dict.get("tool")
            or ""
        ).strip()
        if tool_name not in ASYNC_TASK_TOOL_NAMES:
            continue
        tool_names.add(tool_name)

        event_failed = False
        for task_id in _async_payload_values(payload_dict, "async_task_id", "async_task_ids"):
            task_ids.add(task_id)
        for agent_name in _async_payload_values(
            payload_dict,
            "async_subagent_name",
            "async_subagent_names",
        ):
            subagent_names.add(agent_name)
        for status in _async_payload_values(payload_dict, "async_status", "async_statuses"):
            statuses.add(status)
            if status.lower() in ASYNC_FAILURE_STATUSES:
                event_failed = True
        for error in _async_payload_values(payload_dict, "async_error", "async_errors"):
            event_failed = True
            errors.add(error)
        if payload_dict.get("async_failure") is True:
            event_failed = True

        if event_kind == "tool_call.started":
            if tool_name == "start_async_task":
                start_call_count += 1
                subagent_type = str(payload_dict.get("subagent_type") or "").strip()
                if subagent_type:
                    subagent_names.add(subagent_type)
            elif tool_name == "check_async_task":
                check_call_count += 1
            elif tool_name == "update_async_task":
                update_call_count += 1
            elif tool_name == "cancel_async_task":
                cancel_call_count += 1
            elif tool_name == "list_async_tasks":
                list_call_count += 1

            task_id = str(payload_dict.get("task_id") or "").strip()
            if task_id:
                task_ids.add(task_id)
            continue

        if event_kind != "tool_call.completed":
            continue

        output_text = str(payload_dict.get("output_preview") or payload_dict.get("output") or "")
        for task_id in _async_task_ids_from_text(output_text):
            task_ids.add(task_id)
        for agent_name in _async_agent_names_from_text(output_text):
            subagent_names.add(agent_name)
        for status in _async_statuses_from_text(output_text):
            statuses.add(status)
            if status.lower() in ASYNC_FAILURE_STATUSES:
                event_failed = True
        failure_text = _async_failure_text(output_text)
        if failure_text:
            event_failed = True
            errors.add(failure_text)
        else:
            for error in _async_errors_from_text(output_text):
                event_failed = True
                errors.add(error)
        parsed = _json_object_from_text(output_text)
        if parsed:
            task_id = str(parsed.get("thread_id") or parsed.get("task_id") or "").strip()
            if task_id:
                task_ids.add(task_id)
            status = str(parsed.get("status") or "").strip()
            if status:
                statuses.add(status)
                if status.lower() in ASYNC_FAILURE_STATUSES:
                    event_failed = True
            error = str(parsed.get("error") or "").strip()
            if error:
                event_failed = True
                errors.add(error)
            result = str(parsed.get("result") or "").strip()
            if status == "success" and result:
                completed_result_count += 1
        if event_failed:
            failure_count += 1

    monitor_call_count = check_call_count + list_call_count
    return {
        "start_call_count": start_call_count,
        "check_call_count": check_call_count,
        "update_call_count": update_call_count,
        "cancel_call_count": cancel_call_count,
        "list_call_count": list_call_count,
        "monitor_call_count": monitor_call_count,
        "task_ids": sorted(task_ids),
        "subagent_names": sorted(subagent_names),
        "statuses": sorted(statuses),
        "completed_result_count": completed_result_count,
        "failure_count": failure_count,
        "errors": sorted(errors),
        "tool_names": sorted(tool_names),
    }


def _async_payload_values(payload: dict[str, Any], *keys: str) -> list[str]:
    values: list[str] = []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list | tuple | set):
            for item in value:
                text = str(item or "").strip()
                if text:
                    values.append(text)
            continue
        text = str(value or "").strip()
        if text:
            values.append(text)
    return values


def _async_task_ids_from_text(text: str) -> list[str]:
    ids: list[str] = []
    for match in re.finditer(r"\btask_id:\s*([^\s,;]+)", text):
        task_id = match.group(1).strip()
        if task_id:
            ids.append(task_id)
    for match in re.finditer(r"\bCancelled async subagent task:\s*([^\s,;]+)", text):
        task_id = match.group(1).strip()
        if task_id:
            ids.append(task_id)
    return ids


def _async_agent_names_from_text(text: str) -> list[str]:
    names: list[str] = []
    for match in re.finditer(r"\bagent:\s*([^\s,;]+)", text):
        name = match.group(1).strip()
        if name:
            names.append(name)
    return names


def _async_statuses_from_text(text: str) -> list[str]:
    statuses: list[str] = []
    for match in re.finditer(r"\bstatus:\s*([A-Za-z_:-]+)", text):
        status = match.group(1).strip()
        if status:
            statuses.append(status)
    if re.search(r"\bCancelled async subagent task:\s*[^\s,;]+", text):
        statuses.append("cancelled")
    return statuses


def _async_errors_from_text(text: str) -> list[str]:
    errors: list[str] = []
    for line in text.splitlines():
        for match in re.finditer(
            r"\berror:\s*(.+?)(?=\s+\b(?:task_id|agent|status|error):|$)",
            line,
        ):
            error = " ".join(match.group(1).strip().split())
            if error:
                errors.append(error)
    return errors


def _async_failure_text(text: str) -> str:
    stripped = " ".join(text.strip().split())
    if not stripped:
        return ""
    lowered = stripped.lower()
    failure_prefixes = (
        "failed to launch async subagent",
        "failed to get run status",
        "failed to update async subagent",
        "failed to cancel run",
        "unknown async subagent type",
        "no async subagents are configured",
        "no tracked task found",
        "unknown async subagent task",
        "async subagent task is missing required state",
        "task_id is required",
        "start_async_task description is required",
        "update_async_task message is required",
        "agentruncontext is required",
    )
    if any(lowered.startswith(prefix) for prefix in failure_prefixes):
        return stripped[:500]
    if "requires async invocation" in lowered:
        return stripped[:500]
    return ""


def _json_object_from_text(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    candidates = [stripped]
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        candidates.append(stripped[first : last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_rarespot_configuration(config: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    tile_size = _optional_int(config.get("tile_size"))
    if tile_size is not None:
        normalized["tile_size"] = tile_size
    tile_overlap = _optional_float(config.get("tile_overlap"))
    if tile_overlap is not None:
        normalized["tile_overlap"] = tile_overlap
    stride = _optional_int(config.get("stride"))
    if stride is not None:
        normalized["stride"] = stride
    conf = _optional_float(config.get("conf"))
    if conf is not None:
        normalized["conf"] = conf
    iou = _optional_float(config.get("iou"))
    if iou is not None:
        normalized["iou"] = iou
    return normalized


def _round_ms(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


def _terminal_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    terminal_events = [
        event
        for event in events
        if event.get("event_kind") in {"run.completed", "run.failed", "run.canceled"}
    ]
    if not terminal_events:
        return None
    return max(terminal_events, key=_event_sequence)


def _idle_recoveries(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recoveries: list[dict[str, Any]] = []
    for event in events:
        if event.get("event_kind") != "trace.message.delta":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        if payload.get("reason") != "model_stream_idle":
            continue
        recoveries.append(
            {
                "recovery_index": _optional_int(payload.get("recovery_index")),
                "timeout_seconds": _optional_float(payload.get("timeout_seconds")),
                "sequence": _event_sequence(event),
            }
        )
    return recoveries


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summarize_artifact(
    artifact: dict[str, Any],
    download_checks: dict[str, bool],
) -> dict[str, Any]:
    artifact_id = str(artifact.get("artifact_id") or "")
    summary = {
        "artifact_id": artifact_id,
        "kind": artifact.get("kind"),
        "path": artifact.get("path"),
        "mime_type": artifact.get("mime_type"),
        "size_bytes": artifact.get("size_bytes"),
        "download_ok": download_checks.get(artifact_id),
    }
    if artifact.get("run_id"):
        summary["run_id"] = artifact.get("run_id")
    if artifact.get("tool_name"):
        summary["tool_name"] = artifact.get("tool_name")
    if artifact.get("sha256"):
        summary["sha256"] = artifact.get("sha256")
    return summary


def _retain_materials_validation_bytes(
    evidence_dir: Path,
    payload: bytes,
    digest: str,
) -> Path:
    """Write exact downloaded validation bytes to a content-addressed file."""

    root_input = evidence_dir.expanduser()
    if root_input.is_symlink():
        raise ValueError("materials evidence directory cannot be a symlink")
    root_input.mkdir(parents=True, exist_ok=True, mode=0o700)
    root = root_input.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("materials evidence path is not a directory")
    target = root / f"materials-validation-{digest}.json"
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise ValueError("retained materials validation path conflicts with existing bytes")
        return target

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("could not finish materials validation evidence write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return target


def _typed_evidence_binding(evidence: Any) -> tuple[str, str] | None:
    evidence_sha256 = str(evidence.sha256 or "").strip().lower()
    artifact_id = str(evidence.artifact_id or "").strip().lower()
    id_match = TYPED_MATERIALS_EVIDENCE_ID_PATTERN.fullmatch(artifact_id)
    if id_match is not None:
        operation, identity_sha256 = id_match.groups()
        if (
            operation in MATERIALS_TYPED_SCIENTIFIC_OPERATIONS
            and identity_sha256 == evidence_sha256
        ):
            return operation, identity_sha256
        return None
    if artifact_id.startswith("typed-"):
        return None

    durable_path = _normalized_durable_output_path(evidence.path)
    if durable_path is None:
        return None
    for pattern in TYPED_MATERIALS_EVIDENCE_PATH_PATTERNS:
        path_match = pattern.fullmatch(durable_path)
        if path_match is None:
            continue
        operation, identity_sha256 = path_match.groups()
        if identity_sha256 == evidence_sha256:
            return operation, identity_sha256
        return None
    return None


def _typed_result_bindings(assessment: Any) -> list[dict[str, Any]]:
    """Recover exact operation/result identities from closed typed evidence."""

    validators_by_binding: dict[tuple[str, str], set[str]] = {}
    for check in assessment.checks:
        for evidence in check.evidence:
            binding = _typed_evidence_binding(evidence)
            if binding is None:
                continue
            validators_by_binding.setdefault(binding, set()).add(check.validator_id)
    return [
        {
            "operation": operation,
            "result_artifact_sha256": result_sha256,
            "validator_ids": sorted(validators_by_binding[(operation, result_sha256)]),
        }
        for operation, result_sha256 in sorted(validators_by_binding)
    ]


def _inspect_materials_validation_artifact(
    client: ControlPlaneClient,
    artifacts: list[dict[str, Any]],
    *,
    evidence_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Download, hash, parse, and recompute a materials validation record."""

    candidates = [
        artifact
        for artifact in artifacts
        if _durable_artifact_output_path(artifact) == f"outputs/{MATERIALS_VALIDATION_FILENAME}"
    ]
    if not candidates:
        return None
    if len(candidates) != 1:
        return {
            "valid": False,
            "error": f"expected one {MATERIALS_VALIDATION_FILENAME}, found {len(candidates)}",
            "candidate_count": len(candidates),
            "independent": False,
        }
    artifact = candidates[0]
    artifact_id = str(artifact.get("artifact_id") or "").strip()
    base = {
        "artifact_id": artifact_id,
        "path": artifact.get("path"),
        "durable_path": _durable_artifact_output_path(artifact),
        "independent": False,
    }
    if not artifact_id:
        return {**base, "valid": False, "error": "validation artifact has no artifact_id"}
    declared_size = _optional_int(artifact.get("size_bytes"))
    if declared_size is None:
        return {
            **base,
            "valid": False,
            "error": "validation artifact has no declared size_bytes",
        }
    if declared_size > MATERIALS_VALIDATION_MAX_BYTES:
        return {
            **base,
            "valid": False,
            "error": (f"validation artifact exceeds {MATERIALS_VALIDATION_MAX_BYTES} byte limit"),
        }
    try:
        raw = client.download_artifact(artifact_id)
        if len(raw) > MATERIALS_VALIDATION_MAX_BYTES:
            raise ValueError(
                f"validation artifact exceeds {MATERIALS_VALIDATION_MAX_BYTES} byte limit"
            )
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("materials validation record must be a JSON object")
        assessment = parse_assessment_record(payload)
        raw_sha256 = hashlib.sha256(raw).hexdigest()
        declared_sha256 = str(artifact.get("sha256") or "").strip().lower()
        if declared_sha256 and declared_sha256 != raw_sha256:
            raise ValueError("validation artifact SHA-256 differs from control-plane metadata")
    except Exception as exc:
        return {
            **base,
            "valid": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        canonical_sha256 = hashlib.sha256(
            canonical_record_json(assessment).encode("utf-8")
        ).hexdigest()
    except Exception as exc:
        return {
            **base,
            "valid": False,
            "error": f"canonical validation record failed: {type(exc).__name__}: {exc}",
        }

    retained: dict[str, Any] = {}
    if evidence_dir is not None:
        try:
            retained_path = _retain_materials_validation_bytes(
                evidence_dir,
                raw,
                raw_sha256,
            )
        except Exception as exc:
            return {
                **base,
                "valid": False,
                "error": f"validation evidence retention failed: {type(exc).__name__}: {exc}",
            }
        retained = {
            "retained_path": str(retained_path),
            "retained_sha256": raw_sha256,
            "retained_size_bytes": len(raw),
        }

    outcome_counts = Counter(check.outcome.value for check in assessment.checks)
    evidence_errors, verified_evidence = _verify_materials_evidence_artifacts(
        assessment=assessment,
        artifacts=artifacts,
    )
    return {
        **base,
        **retained,
        "valid": True,
        "schema_version": "1",
        "scientific_status": assessment.scientific_status.value,
        "run_status": assessment.run_status,
        "verified": assessment.verified,
        "silent_success": assessment.silent_success,
        "capability_supported": assessment.capability_supported,
        "contradiction_failures": list(assessment.contradiction_failures),
        "required_validator_ids": list(assessment.required_validator_ids),
        "record_sha256": raw_sha256,
        "canonical_sha256": canonical_sha256,
        "size_bytes": len(raw),
        "check_count": len(assessment.checks),
        "required_count": len(assessment.required_validator_ids),
        "evidence_count": sum(len(check.evidence) for check in assessment.checks),
        "evidence_verified_count": verified_evidence,
        "evidence_verified": not evidence_errors,
        "evidence_errors": evidence_errors,
        "typed_result_bindings": _typed_result_bindings(assessment),
        "outcomes": dict(sorted(outcome_counts.items())),
        "critical_failures": list(assessment.critical_failures),
        "missing_validator_ids": list(assessment.missing_validator_ids),
        "reasons": list(assessment.reasons),
    }


def _normalized_durable_output_path(value: Any) -> str | None:
    """Normalize only paths rooted in the durable ``/outputs`` namespace."""

    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return None
    if raw.startswith("/outputs/"):
        raw = raw[1:]
    elif not raw.startswith("outputs/"):
        return None
    normalized = posixpath.normpath(raw)
    if normalized == "outputs" or not normalized.startswith("outputs/"):
        return None
    return normalized


def _durable_artifact_output_path(artifact: dict[str, Any]) -> str | None:
    """Resolve the control plane's basename form only for collected outputs."""

    normalized = _normalized_durable_output_path(artifact.get("path"))
    if normalized:
        return normalized
    raw = str(artifact.get("path") or "").strip().replace("\\", "/")
    if (
        raw
        and "/" not in raw
        and raw not in {".", ".."}
        and str(artifact.get("tool_name") or "").strip() == "outputs_collector"
        and str(artifact.get("artifact_id") or "").strip()
        and str(artifact.get("run_id") or "").strip()
    ):
        return f"outputs/{raw}"
    return None


def _verify_materials_evidence_artifacts(
    *,
    assessment: Any,
    artifacts: list[dict[str, Any]],
) -> tuple[list[str], int]:
    """Match every evidence reference to a content-addressed run artifact."""

    by_id: dict[str, list[dict[str, Any]]] = {}
    by_path: dict[str, list[dict[str, Any]]] = {}
    for artifact in artifacts:
        artifact_id = str(artifact.get("artifact_id") or "").strip()
        durable_path = _durable_artifact_output_path(artifact)
        if artifact_id:
            by_id.setdefault(artifact_id, []).append(artifact)
        if durable_path:
            by_path.setdefault(durable_path, []).append(artifact)

    errors: list[str] = []
    verified = 0
    for check in assessment.checks:
        for evidence in check.evidence:
            label = f"{check.validator_id}:{evidence.name}"
            matches: list[dict[str, Any]] = []
            if evidence.artifact_id:
                matches = by_id.get(evidence.artifact_id, [])
                if len(matches) != 1:
                    errors.append(f"{label} artifact_id resolved to {len(matches)} run artifacts")
                    continue
            if evidence.path:
                normalized_path = _normalized_durable_output_path(evidence.path)
                if normalized_path is None:
                    errors.append(f"{label} path is outside durable /outputs")
                    continue
                path_matches = by_path.get(normalized_path, [])
                if len(path_matches) != 1:
                    errors.append(f"{label} path resolved to {len(path_matches)} run artifacts")
                    continue
                if matches and matches[0] is not path_matches[0]:
                    errors.append(f"{label} artifact_id and path refer to different artifacts")
                    continue
                matches = path_matches
            if len(matches) != 1:
                errors.append(f"{label} has no resolvable artifact_id or durable path")
                continue

            matched = matches[0]
            matched_path = _durable_artifact_output_path(matched)
            if matched_path is None:
                errors.append(f"{label} matched artifact is outside durable /outputs")
                continue
            actual_sha256 = str(matched.get("sha256") or "").strip().lower()
            if actual_sha256 != evidence.sha256:
                errors.append(f"{label} SHA-256 does not match the control-plane artifact")
                continue
            actual_size = _optional_int(matched.get("size_bytes"))
            if evidence.size_bytes is not None and actual_size != evidence.size_bytes:
                errors.append(f"{label} size_bytes does not match the control-plane artifact")
                continue
            verified += 1
    return errors, verified


def _summary_artifacts(summary: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for key in ("artifacts", "linked_artifacts"):
        raw_artifacts = summary.get(key) or []
        if not isinstance(raw_artifacts, list):
            continue
        artifacts.extend(artifact for artifact in raw_artifacts if isinstance(artifact, dict))
    return artifacts


def summarize_thread_record(thread: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "thread_id": thread.get("thread_id"),
        "latest_run_id": thread.get("latest_run_id"),
        "title": thread.get("title"),
    }
    return {key: value for key, value in summary.items() if value is not None}


def summarize_thread_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    roles: list[str] = []
    run_ids: list[str] = []
    content_lengths: list[int] = []
    content_previews: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = str(message.get("content") or "")
        roles.append(str(message.get("role") or ""))
        run_ids.append(str(message.get("run_id") or ""))
        content_lengths.append(len(content))
        content_previews.append(_response_preview(content, max_chars=240))
    return {
        "count": len(roles),
        "roles": roles,
        "run_ids": run_ids,
        "content_lengths": content_lengths,
        "content_previews": content_previews,
    }


def _trace_turn_summaries(result: dict[str, Any]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    prompt = result.get("prompt")
    if isinstance(prompt, dict):
        turns.append(prompt)
    followups = result.get("followups")
    if isinstance(followups, list):
        turns.extend(turn for turn in followups if isinstance(turn, dict))
    else:
        followup = result.get("followup")
        if isinstance(followup, dict):
            turns.append(followup)
    return turns


def _turn_text(turn: dict[str, Any]) -> str:
    preview = str(turn.get("response_preview") or "")
    tail = str(turn.get("response_tail") or "")
    if tail and tail != preview:
        return f"{preview}\n{tail}"
    return preview


def _trace_has_tool(turns: list[dict[str, Any]], names: set[str]) -> bool:
    return any(
        isinstance(turn.get("tool_names"), list)
        and any(str(tool_name) in names for tool_name in turn.get("tool_names") or [])
        for turn in turns
    )


def _trace_has_code_and_analysis_artifacts(turns: list[dict[str, Any]]) -> bool:
    saw_code = False
    saw_analysis_artifact = False
    for artifact in _trace_artifacts(turns):
        kind = str(artifact.get("kind") or "").lower()
        path = str(artifact.get("path") or "").lower()
        mime_type = str(artifact.get("mime_type") or "").lower()
        if kind == "code" or path.endswith((".py", ".ipynb", ".r", ".jl")):
            saw_code = True
            continue
        if (
            kind in {"figure", "image", "table", "report", "model", "dataset"}
            or mime_type.startswith("image/")
            or mime_type in {"text/csv", "application/json", "text/markdown"}
            or path.endswith(
                (".png", ".jpg", ".jpeg", ".gif", ".csv", ".json", ".md", ".pth", ".pt")
            )
        ):
            saw_analysis_artifact = True
    return saw_code and saw_analysis_artifact


def _trace_has_bisque_analysis_artifacts(turns: list[dict[str, Any]]) -> bool:
    saw_visual_or_report = False
    saw_code = False
    for artifact in _trace_artifacts(turns):
        kind = str(artifact.get("kind") or "").lower()
        path = str(artifact.get("path") or "").lower()
        mime_type = str(artifact.get("mime_type") or "").lower()
        if kind == "code" or path.endswith((".py", ".ipynb", ".r", ".jl")):
            saw_code = True
            continue
        if (
            kind in {"figure", "image", "report", "table", "dataset"}
            or mime_type.startswith("image/")
            or mime_type in {"text/markdown", "text/csv", "application/json"}
            or path.endswith((".png", ".jpg", ".jpeg", ".gif", ".csv", ".json", ".md"))
        ):
            saw_visual_or_report = True
    return saw_visual_or_report and saw_code


def _trace_artifact_downloads_verified(turns: list[dict[str, Any]]) -> bool:
    artifacts = _trace_artifacts(turns)
    return bool(artifacts) and all(artifact.get("download_ok") is True for artifact in artifacts)


def _tool_autonomy_followups_use_prior_context(turns: list[dict[str, Any]]) -> bool:
    if len(turns) <= 1:
        return True
    context_tools = {
        "artifact_manifest",
        "stage_artifact_for_analysis",
        "read_file",
        "grep",
        "glob",
        "ls",
    }
    for turn in turns[1:]:
        tool_names = turn.get("tool_names") or []
        if not isinstance(tool_names, list):
            return False
        if not any(str(tool_name) in context_tools for tool_name in tool_names):
            return False
    return True


def _turn_uses_bisque_followup_context(turn: dict[str, Any]) -> bool:
    tool_names = turn.get("tool_names") or []
    if not isinstance(tool_names, list):
        return False
    context_tools = {
        "artifact_manifest",
        "stage_artifact_for_analysis",
        "stage_uploaded_files_for_analysis",
        "read_file",
        "grep",
        "glob",
        "ls",
    }
    return any(str(tool_name) in context_tools for tool_name in tool_names)


def _trace_mentions_bisque_upload(turns: list[dict[str, Any]]) -> bool:
    combined = "\n".join(_turn_text(turn) for turn in turns).lower()
    return bool(
        re.search(r"\bbisque\b", combined)
        and re.search(r"\bupload(?:ed|s|ing)?\b|/data_service/", combined)
    )


def _trace_has_autonomy_analysis_content(turns: list[dict[str, Any]]) -> bool:
    combined = "\n".join(_turn_text(turn) for turn in turns).lower()
    patterns = [
        r"\bmetric|auc|accuracy|precision|recall|f1|loss\b",
        r"\bplot|figure|curve|matrix|visuali[sz]",
        r"\btable|csv|json|artifact|saved|download",
        r"\breproduc|seed|environment|rerun",
        r"\blimitation|caveat|uncertainty",
        r"\brecommend|next experiment|next step",
        r"\bcompare|comparison|tradeoff|versus|vs\.",
    ]
    if sum(1 for pattern in patterns if re.search(pattern, combined)) >= 5:
        return True

    quantitative_math = bool(
        re.search(
            r"(?:\b\d+(?:\.\d+)?\b|\\\(.*?\\\)|\by\s*=|\bx\s*=|\bx\^|\by\^|"
            r"\bdifference|range|endpoint|monotonic|quadratic|cubic|polynomial|growth)",
            combined,
        )
    )
    visual_explanation = bool(re.search(r"\bplot|figure|curve|visuali[sz]|chart|image", combined))
    durable_work = bool(
        re.search(r"\bartifact|saved|download|code|source|script|outputs?/", combined)
    )
    interpretation = bool(
        re.search(
            r"\bdemonstrates?|shows?|observations?|comparison|changed|faster|"
            r"slower|versus|tradeoff|limitation|caveat|reproduc|rerun",
            combined,
        )
    )
    return quantitative_math and visual_explanation and durable_work and interpretation


def _trace_artifacts(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for turn in turns:
        for raw in [*(turn.get("artifacts") or []), *(turn.get("linked_artifacts") or [])]:
            if isinstance(raw, dict):
                artifacts.append(raw)
    return artifacts


def _trace_tool_capability_manifests(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for turn in turns:
        for raw in turn.get("tool_capability_manifests") or []:
            if isinstance(raw, dict):
                manifests.append(raw)
    return manifests


def _trace_context_tool_hygiene(turns: list[dict[str, Any]]) -> dict[str, Any]:
    checked_output_count = 0
    host_path_leak_count = 0
    leaks: set[str] = set()
    for turn in turns:
        raw = turn.get("context_tool_hygiene")
        if not isinstance(raw, dict):
            continue
        checked_output_count += int(raw.get("checked_output_count") or 0)
        host_path_leak_count += int(raw.get("host_path_leak_count") or 0)
        for leak in raw.get("leaks") or []:
            text = str(leak or "").strip()
            if text:
                leaks.add(text)
    return {
        "checked_output_count": checked_output_count,
        "host_path_leak_count": host_path_leak_count,
        "safe": host_path_leak_count == 0,
        "leaks": sorted(leaks),
    }


def _trace_available_skills(
    turns: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
) -> list[str]:
    names = _manifest_available_skill_names(manifests)
    for turn in turns:
        for name in turn.get("available_skills") or []:
            normalized = str(name or "").strip()
            if normalized:
                names.add(normalized)
        records = [
            *(turn.get("skill_reads") or []),
            *(turn.get("skill_scripts") or []),
        ]
        for record in records:
            if not isinstance(record, dict):
                continue
            skill = str(record.get("skill") or "").strip()
            if skill:
                names.add(skill)
    return sorted(names)


def _trace_skill_records(turns: list[dict[str, Any]], key: str) -> list[dict[str, str]]:
    records: set[tuple[str, str]] = set()
    for turn in turns:
        for raw in turn.get(key) or []:
            if not isinstance(raw, dict):
                continue
            skill = str(raw.get("skill") or "").strip()
            path = str(raw.get("path") or "").strip()
            if skill and path:
                records.add((skill, path))
    return [{"skill": skill, "path": path} for skill, path in sorted(records)]


def _skill_records_have_skill(records: list[dict[str, str]], skill_name: str) -> bool:
    return any(str(record.get("skill") or "") == skill_name for record in records)


def _manifest_builtin_tool_names(manifests: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for manifest in manifests:
        for raw in manifest.get("deepagents_builtin_tools") or []:
            if isinstance(raw, dict):
                name = str(raw.get("name") or "").strip()
            else:
                name = str(raw or "").strip()
            if name:
                names.add(name)
    return names


def _manifest_registered_tool_names(manifests: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for manifest in manifests:
        for raw in manifest.get("registered_tools") or []:
            name = str(raw or "").strip()
            if name:
                names.add(name)
    return names


def _manifest_available_skill_names(manifests: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for manifest in manifests:
        for raw in manifest.get("available_skills") or []:
            if isinstance(raw, dict):
                name = str(raw.get("name") or "").strip()
            else:
                name = str(raw or "").strip()
            if name:
                names.add(name)
    return names


def _manifest_available_subagent_names(manifests: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for manifest in manifests:
        for raw in manifest.get("available_subagents") or []:
            if isinstance(raw, dict):
                name = str(raw.get("name") or "").strip()
            else:
                name = str(raw or "").strip()
            if name:
                names.add(name)
    return names


def _manifest_scoped_subagent_context_tool_names(
    manifests: list[dict[str, Any]],
) -> dict[str, list[str]]:
    tools_by_subagent: dict[str, set[str]] = {}
    for manifest in manifests:
        for raw in manifest.get("available_subagents") or []:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip()
            if name not in SCOPED_DELEGATION_SUBAGENT_NAMES:
                continue
            tools = {
                str(tool_name or "").strip()
                for tool_name in (raw.get("tool_names") or [])
                if str(tool_name or "").strip()
            }
            tools_by_subagent.setdefault(name, set()).update(tools)
    return {name: sorted(tool_names) for name, tool_names in sorted(tools_by_subagent.items())}


def _manifest_available_async_subagent_names(manifests: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for manifest in manifests:
        for raw in manifest.get("available_async_subagents") or []:
            if isinstance(raw, dict):
                name = str(raw.get("name") or "").strip()
            else:
                name = str(raw or "").strip()
            if name:
                names.add(name)
    return names


def _trace_delegation_evidence(turns: list[dict[str, Any]]) -> dict[str, Any]:
    task_call_count = 0
    subagent_names: set[str] = set()
    subagent_tool_names: set[str] = set()
    subagent_message_delta_count = 0
    scoped_tool_event_count = 0
    scoped_usage_event_count = 0

    for turn in turns:
        raw = turn.get("delegation")
        if not isinstance(raw, dict):
            continue
        task_call_count += int(raw.get("task_call_count") or 0)
        subagent_message_delta_count += int(raw.get("subagent_message_delta_count") or 0)
        scoped_tool_event_count += int(raw.get("scoped_tool_event_count") or 0)
        scoped_usage_event_count += int(raw.get("scoped_usage_event_count") or 0)
        for name in raw.get("subagent_names") or []:
            normalized = str(name or "").strip()
            if normalized:
                subagent_names.add(normalized)
        for name in raw.get("subagent_tool_names") or []:
            normalized = str(name or "").strip()
            if normalized:
                subagent_tool_names.add(normalized)

    return {
        "task_call_count": task_call_count,
        "subagent_names": sorted(subagent_names),
        "subagent_tool_names": sorted(subagent_tool_names),
        "subagent_message_delta_count": subagent_message_delta_count,
        "scoped_tool_event_count": scoped_tool_event_count,
        "scoped_usage_event_count": scoped_usage_event_count,
    }


def _trace_async_delegation_evidence(turns: list[dict[str, Any]]) -> dict[str, Any]:
    start_call_count = 0
    check_call_count = 0
    update_call_count = 0
    cancel_call_count = 0
    list_call_count = 0
    monitor_call_count = 0
    completed_result_count = 0
    task_ids: set[str] = set()
    subagent_names: set[str] = set()
    statuses: set[str] = set()
    tool_names: set[str] = set()
    failure_count = 0
    errors: set[str] = set()

    for turn in turns:
        raw = turn.get("async_delegation")
        if not isinstance(raw, dict):
            continue
        start_call_count += int(raw.get("start_call_count") or 0)
        check_call_count += int(raw.get("check_call_count") or 0)
        update_call_count += int(raw.get("update_call_count") or 0)
        cancel_call_count += int(raw.get("cancel_call_count") or 0)
        list_call_count += int(raw.get("list_call_count") or 0)
        monitor_call_count += int(raw.get("monitor_call_count") or 0)
        completed_result_count += int(raw.get("completed_result_count") or 0)
        failure_count += int(raw.get("failure_count") or 0)
        for name in raw.get("task_ids") or []:
            normalized = str(name or "").strip()
            if normalized:
                task_ids.add(normalized)
        for name in raw.get("subagent_names") or []:
            normalized = str(name or "").strip()
            if normalized:
                subagent_names.add(normalized)
        for status in raw.get("statuses") or []:
            normalized = str(status or "").strip()
            if normalized:
                statuses.add(normalized)
        for name in raw.get("tool_names") or []:
            normalized = str(name or "").strip()
            if normalized:
                tool_names.add(normalized)
        for error in raw.get("errors") or []:
            normalized = str(error or "").strip()
            if normalized:
                errors.add(normalized)

    return {
        "start_call_count": start_call_count,
        "check_call_count": check_call_count,
        "update_call_count": update_call_count,
        "cancel_call_count": cancel_call_count,
        "list_call_count": list_call_count,
        "monitor_call_count": monitor_call_count,
        "task_ids": sorted(task_ids),
        "subagent_names": sorted(subagent_names),
        "statuses": sorted(statuses),
        "completed_result_count": completed_result_count,
        "failure_count": failure_count,
        "errors": sorted(errors),
        "tool_names": sorted(tool_names),
    }


def _context_staging_tool_names() -> set[str]:
    return {
        "artifact_manifest",
        "stage_artifact_for_analysis",
        "stage_uploaded_files_for_analysis",
    }


SCOPED_DELEGATION_SUBAGENT_NAMES = {"code-runner"}
_CONTEXT_TOOL_PRIVATE_PATH_KEYS = {"source_path", "storage_uri"}
_MACOS_USER_PREFIX = "/" + "Users/"
_HOST_PATH_PREFIXES = (
    _MACOS_USER_PREFIX,
    "/home/",
    "/tmp/",
    "/private/",
    "/var/",
    "/Volumes/",
    "/srv/",
)
_HOST_PATH_TEXT_PATTERN = re.compile(
    r"(?:file://)?(?:"
    + "|".join(re.escape(prefix) for prefix in _HOST_PATH_PREFIXES)
    + r")[^\s\"'}\]]+"
)


def _turns_have_page_citations(turns: list[dict[str, Any]]) -> bool:
    citation_pattern = re.compile(
        r"(?:[A-Za-z0-9_.-]+:p\d+\b|\bp(?:age)?\.?\s*\d+\b)",
        re.IGNORECASE,
    )
    return all(citation_pattern.search(_turn_text(turn)) for turn in turns)


PAPER_TOOL_NAMES = {
    "bind_paper_text_literal",
    "extract_paper_table_evidence",
    "ingest_arxiv_paper",
    "ingest_pdf_resource",
    "paper_manifest",
    "search_paper",
    "read_paper_pages",
    "read_paper_section",
    "render_paper_page",
}


def _has_any_paper_tool(turn: dict[str, Any]) -> bool:
    tool_names = turn.get("tool_names") or []
    if not isinstance(tool_names, list):
        return False
    return any(str(name) in PAPER_TOOL_NAMES for name in tool_names)


def _turns_have_paper_tools(turns: list[dict[str, Any]]) -> bool:
    return all(_has_any_paper_tool(turn) for turn in turns)


def _trace_has_paper_tools(turns: list[dict[str, Any]]) -> bool:
    return any(_has_any_paper_tool(turn) for turn in turns)


def _trace_has_figure_evidence(turns: list[dict[str, Any]]) -> bool:
    for turn in turns:
        artifacts = turn.get("artifacts") or []
        if isinstance(artifacts, list):
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    continue
                kind = str(artifact.get("kind") or "").lower()
                mime_type = str(artifact.get("mime_type") or "").lower()
                if (
                    kind in {"figure", "image"}
                    or mime_type.startswith("image/")
                    or str(artifact.get("path") or "")
                    .lower()
                    .endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))
                ):
                    if artifact.get("download_ok") is not False:
                        return True
        tool_names = turn.get("tool_names") or []
        if isinstance(tool_names, list) and "render_paper_page" in tool_names:
            return True
    return False


def _trace_has_technical_content(turns: list[dict[str, Any]]) -> bool:
    combined = "\n".join(_turn_text(turn) for turn in turns).lower()
    technical_patterns = [
        r"\\sqrt|sqrt|√",
        r"\bd[_-]?k\b",
        r"\bq\b.*\bk\b.*\bv\b|\bquery\b.*\bkey\b.*\bvalue\b",
        r"softmax",
        r"variance",
        r"projection|concatenat|multi-head",
        r"limitation|assumption",
    ]
    matches = sum(1 for pattern in technical_patterns if re.search(pattern, combined))
    return matches >= 4


TOOL_CAPABILITY_TOOL_NAMES = {"tool_capability_manifest"}
ASYNC_TASK_TOOL_NAMES = {
    "start_async_task",
    "check_async_task",
    "update_async_task",
    "cancel_async_task",
    "list_async_tasks",
}
ASYNC_FAILURE_STATUSES = {"error", "failed", "failure", "timeout", "interrupted"}


def _turn_has_rarespot_skill_script(turn: dict[str, Any]) -> bool:
    scripts = turn.get("skill_scripts") or []
    if not isinstance(scripts, list):
        return False
    return any(
        isinstance(script, dict)
        and str(script.get("skill") or "") == RARESPOT_SKILL_NAME
        and str(script.get("path") or "") == RARESPOT_SCRIPT_PATH
        for script in scripts
    )


def _turn_has_rarespot_inference(turn: dict[str, Any]) -> bool:
    return _turn_has_rarespot_skill_script(turn)


def _trace_has_rarespot_tool(turns: list[dict[str, Any]]) -> bool:
    return any(_turn_has_rarespot_inference(turn) for turn in turns)


def _trace_mentions_detection_metrics(turns: list[dict[str, Any]]) -> bool:
    combined = "\n".join(_turn_text(turn) for turn in turns).lower()
    patterns = [
        r"prairie[_ -]?dog",
        r"burrow",
        r"detect",
        r"confidence|conf(?:idence)? threshold",
        r"\bcount|counts\b",
    ]
    return sum(1 for pattern in patterns if re.search(pattern, combined)) >= 4


def _trace_has_rarespot_artifacts(turns: list[dict[str, Any]]) -> bool:
    kinds_seen: set[str] = set()
    for turn in turns:
        artifacts = [
            *(turn.get("artifacts") or []),
            *(turn.get("linked_artifacts") or []),
        ]
        if not isinstance(artifacts, list):
            continue
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            if artifact.get("download_ok") is False:
                continue
            kind = str(artifact.get("kind") or "").lower()
            path = str(artifact.get("path") or "").lower()
            mime_type = str(artifact.get("mime_type") or "").lower()
            if kind:
                kinds_seen.add(kind)
            if path.endswith(".csv") or mime_type == "text/csv":
                kinds_seen.add("csv")
            if mime_type.startswith("image/") or path.endswith((".png", ".jpg", ".jpeg")):
                kinds_seen.add("image")
            if path.endswith(".md") or kind == "report":
                kinds_seen.add("report")
    return "image" in kinds_seen and "csv" in kinds_seen


def _trace_has_stub_or_duplicate_rarespot_outputs(turns: list[dict[str, Any]]) -> bool:
    combined = "\n".join(_turn_text(turn) for turn in turns).lower()
    for sentence in re.split(r"[\n\r.!?]+", combined):
        if not re.search(
            r"\bstub\b|local copies? of .*rarespot|"
            r"\bduplicate\b[^.\n\r!?]{0,80}\b(output|artifact|file|copy|rarespot)\b",
            sentence,
        ):
            continue
        if re.search(
            r"\b(no|not|never|without|avoid(?:ed|s)?|prevent(?:ed|s)?)\b"
            r"[^.\n\r!?]{0,80}\b(stub|duplicate|local copies?)\b",
            sentence,
        ):
            continue
        return True
    duplicate_path_patterns = (
        "outputs/rarespot_summary.json",
        "outputs/rarespot_detections.csv",
    )
    for turn in turns:
        artifacts = turn.get("artifacts") or []
        if not isinstance(artifacts, list):
            continue
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            path = str(artifact.get("path") or "").lower()
            if any(pattern in path for pattern in duplicate_path_patterns):
                return True
    return False


def _trace_has_comparative_language(turns: list[dict[str, Any]]) -> bool:
    combined = "\n".join(_turn_text(turn) for turn in turns[1:]).lower()
    return bool(re.search(r"compare|compared|difference|stricter|threshold|robust", combined))


def _trace_has_report_language(turn: dict[str, Any]) -> bool:
    text = _turn_text(turn).lower()
    return bool(re.search(r"report|synthes|summary|limitation|next steps|recommend", text))


def _report_only_rarespot_turns_avoid_inference(turns: list[dict[str, Any]]) -> bool:
    for turn in turns:
        goal = str(turn.get("goal") or "")
        if not _looks_report_only_rarespot_goal(goal):
            continue
        if _turn_has_rarespot_inference(turn):
            return False
    return True


def _trace_rarespot_config_matches(
    turns: list[dict[str, Any]],
    *,
    expected_tile_size: int,
    expected_tile_overlap: float,
    expected_stride: int,
) -> bool:
    saw_rarespot_turn = False
    for turn in turns:
        if not _turn_has_rarespot_inference(turn):
            continue
        saw_rarespot_turn = True
        configurations = turn.get("rarespot_configurations") or []
        if not isinstance(configurations, list) or not configurations:
            return False
        for config in configurations:
            if not isinstance(config, dict):
                return False
            tile_size = _optional_int(config.get("tile_size"))
            tile_overlap = _optional_float(config.get("tile_overlap"))
            stride = _optional_int(config.get("stride"))
            if tile_size != int(expected_tile_size):
                return False
            if tile_overlap is None or abs(tile_overlap - float(expected_tile_overlap)) > 1e-9:
                return False
            if stride != int(expected_stride):
                return False
    return saw_rarespot_turn


def _looks_report_only_rarespot_goal(goal: str) -> bool:
    text = " ".join(goal.lower().split())
    if "rarespot" not in text:
        return False
    if not re.search(r"\b(write|compile|combine|combined|synthes|summari[sz]e|report)\b", text):
        return False
    explicit_new_run = re.search(
        r"\b(run|rerun|execute|perform|launch)\b.{0,80}\b(rarespot|inference|detect|detection|pass)\b",
        text,
    ) or re.search(
        r"\b(stricter|permissive|new|another)\b.{0,80}\b(threshold|confidence|inference|pass)\b",
        text,
    )
    return not bool(explicit_new_run)


def _verify_downloads(
    client: ControlPlaneClient,
    artifacts: list[dict[str, Any]],
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for artifact in artifacts:
        artifact_id = str(artifact.get("artifact_id") or "")
        if not artifact_id:
            continue
        try:
            checks[artifact_id] = len(client.download_artifact(artifact_id)) > 0
        except Exception:
            checks[artifact_id] = False
    return checks


def _summarize_linked_artifacts(
    client: ControlPlaneClient,
    response_text: str,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for artifact_id in _artifact_ids_from_text(response_text):
        artifact: dict[str, Any] = {"artifact_id": artifact_id}
        try:
            artifact = {**artifact, **client.get_artifact(artifact_id)}
        except Exception:
            pass
        try:
            download_ok = len(client.download_artifact(artifact_id)) > 0
        except Exception:
            download_ok = False
        artifact["download_ok"] = download_ok
        summaries.append(_summarize_artifact(artifact, {artifact_id: download_ok}))
    return summaries


def _artifact_ids_from_text(text: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    hrefs = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    for searchable in (text, *hrefs):
        for match in re.finditer(r"/v2/artifacts/([^/\]`)>\s]+)/download", searchable):
            artifact_id = parse.unquote(match.group(1))
            if "..." in artifact_id or "…" in artifact_id:
                continue
            if artifact_id and artifact_id not in seen:
                seen.add(artifact_id)
                ids.append(artifact_id)
    return ids


def _content_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".gif":
        return "image/gif"
    if suffix == ".csv":
        return "text/csv"
    if suffix in {".txt", ".md"}:
        return "text/plain; charset=utf-8"
    return "application/octet-stream"


def _escape_multipart_filename(filename: str) -> str:
    return filename.replace("\\", "\\\\").replace('"', '\\"')


if __name__ == "__main__":
    raise SystemExit(main())
