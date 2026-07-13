#!/usr/bin/env python3
"""Qualify typed pycalphad evidence through the real Go HTTP/Postgres path.

The promotable mode executes the fixed typed CLI in an already-built immutable
materials runtime image with networking disabled.  A host-pinned mode exists
only to diagnose scientific/runtime failures; its synthetic runtime identity is
never accepted as production-live qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

SCHEMA_VERSION = "ultra.calphad.cross-language-gate.v1"
BACKEND_EVIDENCE_SCHEMA = "ultra.calphad.cross-language-qualification.v1"
TYPED_EVIDENCE_SCHEMA = "ultra.calphad.tool-evidence.v3"
TYPED_REQUEST_SCHEMA = "ultra.calphad.typed-request.v2"
PYCALPHAD_VERSION = "0.11.2"
RESULT_MARKER = "ULTRA_CALPHAD_TOOL_RESULT="
BACKEND_MARKER = "CALPHAD_CROSS_LANGUAGE_EVIDENCE "
GO_TEST = "TestCalphadTypedCLIHTTPPostgresQualification"
GO_PACKAGE = "github.com/amilworks/bisque-ultra/backend/controlplane/integration"
GO_COMMAND = (
    "go",
    "test",
    "-json",
    "-count=1",
    "./integration",
    "-run",
    f"^{GO_TEST}$",
)
IMAGE_TITLES = {
    "Ultra Deep Agents scientific sandbox": "/opt/ultra-runtime",
    "Ultra deterministic materials domain gate": "/opt/ultra/src",
}
DATABASE_TOKENS = {"ci", "qualification", "qual", "test", "tests"}
FORBIDDEN_DATABASE_TOKENS = {"prod", "production", "primary", "live"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DATABASE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,63}$")
MAX_COMMAND_OUTPUT = 32 * 1024 * 1024
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024

REFERENCE_COMPONENTS = ("AL", "CO", "W", "VA")
REFERENCE_PHASES = (
    "AL12W",
    "AL2W",
    "AL3CO",
    "AL4W",
    "AL5CO2",
    "AL5W",
    "AL9CO2",
    "BCC_A2",
    "BCC_B2",
    "CO3W",
    "FCC_A1",
    "HCP_A3",
    "L12_FCC",
    "LIQUID",
    "MAL13CO4",
    "MU",
    "OAL13CO4",
    "YAL13CO4",
)
REFERENCE_STABLE_PHASES = ("AL4W", "AL5CO2", "BCC_B2")
REFERENCE_TEMPERATURE_K = 1173.0
REFERENCE_PRESSURE_PA = 101325.0
REFERENCE_INDEPENDENT_COMPOSITIONS = {"CO": [0.26], "W": [0.065]}
REFERENCE_DEPENDENT_COMPONENT = "AL"
REFERENCE_GM_J_PER_MOL = -85970.06746
REFERENCE_GM_ABSOLUTE_TOLERANCE_J_PER_MOL = 1e-4

SOURCE_PATHS = (
    Path(".github/workflows/materials-domain-gate.yml"),
    Path("Makefile"),
    Path("backend/controlplane/integration/calphad_cross_language_http_test.go"),
    Path("backend/controlplane/integration/calphad_cross_language_test.go"),
    Path("backend/controlplane/internal/domain/calphad.go"),
    Path("backend/controlplane/internal/httpapi/calphad_evidence.go"),
    Path("backend/controlplane/internal/httpapi/calphad_scientific_evidence.go"),
    Path("backend/controlplane/internal/httpapi/handlers_calphad.go"),
    Path("backend/controlplane/internal/store/calphad_ledger.go"),
    Path("backend/controlplane/internal/store/schema.sql"),
    Path("backend/controlplane/internal/store/schema_apply.go"),
    Path("backend/controlplane/internal/store/schema_check.go"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/materials/calphad.py"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/materials/calphad_cli.py"),
    Path("backend/deepagents_runtime/materials_data/calphad/manifest.json"),
    Path("backend/deepagents_runtime/materials_data/calphad/alcow_CALPHAD-2017-Wang.tdb"),
    Path("deploy/docker/deepagents-sandbox.Dockerfile"),
    Path("deploy/docker/materials-requirements.txt"),
    Path("scripts/calphad_cross_language_gate.py"),
    Path("tests/test_calphad_cross_language_gate.py"),
)


class QualificationError(RuntimeError):
    """A fail-closed cross-language qualification error."""


@dataclass(frozen=True)
class RuntimeAttestation:
    mode: str
    runtime_image_id: str
    image_ref: str = ""
    image_title: str = ""
    image_revision: str = ""
    pythonpath: str = ""
    image_inspected: bool = False
    image_inspection_payload: bytes = b""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_bounded(path: Path, *, maximum: int, label: str) -> bytes:
    try:
        info = path.lstat()
    except OSError as exc:
        raise QualificationError(f"{label} is missing or unreadable") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_size <= 0
        or info.st_size > maximum
    ):
        raise QualificationError(f"{label} must be a bounded regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QualificationError(f"{label} could not be opened securely") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino)
            or opened.st_size != info.st_size
        ):
            raise QualificationError(f"{label} changed while it was opened")
        remaining = opened.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise QualificationError(f"{label} was truncated while it was read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise QualificationError(f"{label} grew while it was read")
        after = os.fstat(descriptor)
        if (after.st_size, after.st_mtime_ns) != (opened.st_size, opened.st_mtime_ns):
            raise QualificationError(f"{label} changed while it was read")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def load_unique_json(payload: bytes, *, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise QualificationError(f"{label} contains non-finite JSON: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise QualificationError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            payload,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationError(f"{label} is not finite UTF-8 JSON") from exc


def require_exact_keys(value: Any, keys: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise QualificationError(f"{label} does not have the exact required schema")
    return value


def qualification_database_identity(dsn: str) -> dict[str, Any]:
    text = str(dsn or "").strip()
    if not text:
        raise QualificationError("ULTRA_CONTROL_TEST_DATABASE_URL is required")
    parsed = urlsplit(text)
    database = parsed.path.lstrip("/").split("/", 1)[0]
    try:
        port = parsed.port or 5432
    except ValueError as exc:
        raise QualificationError("qualification database URL has an invalid port") from exc
    tokens = {token.lower() for token in re.split(r"[_-]+", database) if token}
    if (
        parsed.scheme not in {"postgres", "postgresql"}
        or not parsed.hostname
        or DATABASE_NAME_RE.fullmatch(database) is None
        or not (tokens & DATABASE_TOKENS)
        or bool(tokens & FORBIDDEN_DATABASE_TOKENS)
    ):
        raise QualificationError(
            "refusing cross-language qualification against a non-disposable PostgreSQL target"
        )
    role = unquote(parsed.username or "").strip()
    if not role:
        raise QualificationError("qualification database URL must name a PostgreSQL role")
    return {
        "scheme": "postgresql",
        "host": parsed.hostname,
        "port": port,
        "database": database,
        "role": role,
        "credentials_recorded": False,
    }


def qualification_database_pair(serving_dsn: str, migration_dsn: str) -> dict[str, Any]:
    serving = qualification_database_identity(serving_dsn)
    migration = qualification_database_identity(migration_dsn)
    for field in ("scheme", "host", "port", "database"):
        if serving[field] != migration[field]:
            raise QualificationError(
                "serving and migration URLs must target the same disposable PostgreSQL database"
            )
    if serving["role"] == migration["role"]:
        raise QualificationError("serving and migration PostgreSQL roles must be distinct")
    return {
        **serving,
        "serving_role": serving["role"],
        "migration_role": migration["role"],
    }


def run_bounded(
    command: tuple[str, ...],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int,
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise QualificationError(f"command could not complete: {command[0]}") from exc
    if len(completed.stdout) > MAX_COMMAND_OUTPUT or len(completed.stderr) > MAX_COMMAND_OUTPUT:
        raise QualificationError(f"command output exceeded its fixed bound: {command[0]}")
    return completed


def inspect_repository(root: Path, expected_git_sha: str, *, require_clean: bool) -> dict[str, Any]:
    expected = expected_git_sha.strip().lower()
    if GIT_SHA_RE.fullmatch(expected) is None:
        raise QualificationError("expected Git SHA must be 40 lowercase hexadecimal characters")
    revision = run_bounded(("git", "rev-parse", "HEAD"), cwd=root, timeout=30)
    status = run_bounded(
        ("git", "status", "--porcelain", "--untracked-files=all"), cwd=root, timeout=30
    )
    if revision.returncode != 0 or status.returncode != 0:
        raise QualificationError("repository provenance could not be inspected")
    head = revision.stdout.decode("utf-8", "strict").strip().lower()
    if head != expected:
        raise QualificationError(f"repository HEAD {head!r} does not match expected Git SHA")
    clean = status.stdout == b""
    if require_clean and not clean:
        raise QualificationError("promotable image qualification requires a clean repository")
    return {"head_sha": head, "clean": clean}


def build_source_manifest(root: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for relative in SOURCE_PATHS:
        payload = read_bounded(root / relative, maximum=64 * 1024 * 1024, label=str(relative))
        manifest.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
    return manifest


def inspect_image_payload(
    payload: bytes,
    *,
    image_ref: str,
    expected_image_id: str,
    expected_title: str,
    expected_git_sha: str,
) -> RuntimeAttestation:
    decoded = load_unique_json(payload, label="docker image inspection")
    if not isinstance(decoded, list) or len(decoded) != 1 or not isinstance(decoded[0], dict):
        raise QualificationError("docker image inspection did not return one image")
    record = decoded[0]
    image_id = str(record.get("Id") or "").lower()
    expected_id = expected_image_id.strip().lower()
    config = record.get("Config")
    if not isinstance(config, dict):
        raise QualificationError("docker image inspection omits Config")
    labels = config.get("Labels") or {}
    environment = config.get("Env") or []
    if not isinstance(labels, dict) or not isinstance(environment, list):
        raise QualificationError("docker image configuration has invalid labels or environment")
    env_map: dict[str, str] = {}
    for entry in environment:
        if isinstance(entry, str) and "=" in entry:
            key, value = entry.split("=", 1)
            env_map[key] = value
    title = str(labels.get("org.opencontainers.image.title") or "")
    revision = str(labels.get("org.opencontainers.image.revision") or "").lower()
    pythonpath = env_map.get("PYTHONPATH", "")
    required_pythonpath = IMAGE_TITLES.get(expected_title)
    if required_pythonpath is None:
        raise QualificationError("expected image title is not an approved materials runtime")
    if IMAGE_ID_RE.fullmatch(expected_id) is None or image_id != expected_id:
        raise QualificationError(
            "inspected Docker image ID does not match the expected immutable ID"
        )
    if title != expected_title or revision != expected_git_sha or pythonpath != required_pythonpath:
        raise QualificationError(
            "Docker image title, source revision, or PYTHONPATH is not production-bound"
        )
    return RuntimeAttestation(
        mode="pinned_image",
        runtime_image_id=image_id,
        image_ref=image_ref,
        image_title=title,
        image_revision=revision,
        pythonpath=pythonpath,
        image_inspected=True,
        image_inspection_payload=payload,
    )


def inspect_runtime_image(
    root: Path,
    *,
    image_ref: str,
    expected_image_id: str,
    expected_title: str,
    expected_git_sha: str,
) -> RuntimeAttestation:
    if not image_ref.strip():
        raise QualificationError("--image is required in pinned-image mode")
    completed = run_bounded(("docker", "image", "inspect", image_ref), cwd=root, timeout=60)
    if completed.returncode != 0:
        raise QualificationError("docker image inspect failed for the requested materials runtime")
    return inspect_image_payload(
        completed.stdout,
        image_ref=image_ref,
        expected_image_id=expected_image_id,
        expected_title=expected_title,
        expected_git_sha=expected_git_sha,
    )


def load_reference_database(root: Path) -> tuple[dict[str, Any], bytes]:
    data_root = root / "backend/deepagents_runtime/materials_data/calphad"
    manifest_payload = read_bounded(
        data_root / "manifest.json", maximum=1024 * 1024, label="CALPHAD manifest"
    )
    manifest = load_unique_json(manifest_payload, label="CALPHAD manifest")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "1":
        raise QualificationError("CALPHAD reference manifest schema is invalid")
    records = manifest.get("databases")
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        raise QualificationError("CALPHAD reference manifest must contain one qualified database")
    record = records[0]
    required = {
        "database_id",
        "filename",
        "sha256",
        "size_bytes",
        "source_uri",
        "license_id",
        "assessment_scope",
        "reference_state",
        "tdb_temperature_limits_K",
        "assessment_pressure_limits_Pa",
        "format",
        "components",
        "phases",
    }
    if not required.issubset(record):
        raise QualificationError("CALPHAD reference manifest omits required provenance")
    database_format = str(record.get("format") or "").casefold()
    filename = str(record.get("filename") or "")
    if database_format not in {"tdb", "dat"} or Path(filename).suffix.casefold() != (
        f".{database_format}"
    ):
        raise QualificationError("CALPHAD reference manifest has an invalid database format")
    if record.get("assessment_pressure_limits_Pa") != [
        REFERENCE_PRESSURE_PA,
        REFERENCE_PRESSURE_PA,
    ]:
        raise QualificationError(
            "CALPHAD reference manifest must bind the fixed qualification pressure"
        )
    database = read_bounded(
        data_root / str(record["filename"]),
        maximum=64 * 1024 * 1024,
        label="CALPHAD reference database",
    )
    if len(database) != record["size_bytes"] or sha256_bytes(database) != record["sha256"]:
        raise QualificationError("CALPHAD reference database does not match its manifest")
    return record, database


def database_binding(record: dict[str, Any], *, resource_id: str, path: str) -> dict[str, Any]:
    return {
        "kind": "resource",
        "database_id": record["database_id"],
        "path": path,
        "resource_id": resource_id,
        "database_format": record["format"],
        "sha256": record["sha256"],
        "size_bytes": record["size_bytes"],
        "source": record["source_uri"],
        "license_id": record["license_id"],
        "assessment_scope": record["assessment_scope"],
        "reference_state": record["reference_state"],
        "temperature_limits_K": record["tdb_temperature_limits_K"],
        "assessment_pressure_limits_Pa": record["assessment_pressure_limits_Pa"],
        "binding_schema": "ultra.selected_resource.v1",
        "binding_authority": "control_resource_catalog",
        "declaration_authority": "resource_owner",
    }


def staged_database_filename(record: dict[str, Any]) -> str:
    suffix = Path(str(record.get("filename") or "")).suffix.lower()
    digest = str(record.get("sha256") or "").lower()
    if suffix not in {".tdb", ".dat"} or SHA256_RE.fullmatch(digest) is None:
        raise QualificationError("CALPHAD reference cannot form a content-addressed staged name")
    return digest + suffix


def typed_request(
    *,
    operation: str,
    runtime_image_id: str,
    binding: dict[str, Any],
    inspection_sha256: str = "",
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "schema_version": TYPED_REQUEST_SCHEMA,
        "operation": operation,
        "runtime_image_id": runtime_image_id,
        "database": binding,
        "selection": {"components": None, "phases": None},
    }
    if operation == "equilibrium":
        request["selection"] = {
            "components": list(REFERENCE_COMPONENTS),
            "phases": list(REFERENCE_PHASES),
        }
        request["inspection_artifact_sha256"] = inspection_sha256
        request["conditions"] = {
            "temperatures_K": [REFERENCE_TEMPERATURE_K],
            "pressures_Pa": [REFERENCE_PRESSURE_PA],
            "independent_compositions": REFERENCE_INDEPENDENT_COMPOSITIONS,
        }
    return request


def validate_reference_equilibrium_checkpoint(evidence: dict[str, Any]) -> dict[str, Any]:
    request = evidence.get("request")
    response = evidence.get("result")
    if not isinstance(request, dict) or not isinstance(response, dict):
        raise QualificationError("reference equilibrium evidence omits request or response")
    selection = request.get("selection")
    conditions = request.get("conditions")
    if (
        not isinstance(selection, dict)
        or set(selection.get("components") or ()) != set(REFERENCE_COMPONENTS)
        or tuple(selection.get("phases") or ()) != REFERENCE_PHASES
        or not isinstance(conditions, dict)
        or conditions.get("temperatures_K") != [REFERENCE_TEMPERATURE_K]
        or conditions.get("pressures_Pa") != [REFERENCE_PRESSURE_PA]
        or conditions.get("independent_compositions") != REFERENCE_INDEPENDENT_COMPOSITIONS
    ):
        raise QualificationError(
            "equilibrium evidence does not use the exact published global-phase checkpoint"
        )
    runtime_request = response.get("request")
    expected_runtime_compositions = {
        component: {"values": values, "units": "mole_fraction"}
        for component, values in REFERENCE_INDEPENDENT_COMPOSITIONS.items()
    }
    if (
        not isinstance(runtime_request, dict)
        or runtime_request.get("dependent_component") != REFERENCE_DEPENDENT_COMPONENT
        or not isinstance(runtime_request.get("conditions"), dict)
        or runtime_request["conditions"].get("independent_compositions")
        != expected_runtime_compositions
    ):
        raise QualificationError(
            "reference equilibrium did not retain the canonical AL-dependent parameterization"
        )
    result = response.get("result")
    points = result.get("points") if isinstance(result, dict) else None
    if not isinstance(points, list) or len(points) != 1 or not isinstance(points[0], dict):
        raise QualificationError("reference equilibrium evidence must contain exactly one point")
    point = points[0]
    phases = point.get("stable_phases")
    vertices = point.get("stable_phase_vertices")
    if not isinstance(phases, list) or not isinstance(vertices, list):
        raise QualificationError("reference equilibrium omits stable phases or vertices")
    observed_phases = tuple(
        sorted(str(record.get("name") or "") for record in phases if isinstance(record, dict))
    )
    observed_vertex_phases = tuple(
        sorted({str(record.get("phase") or "") for record in vertices if isinstance(record, dict)})
    )
    if (
        observed_phases != REFERENCE_STABLE_PHASES
        or observed_vertex_phases != REFERENCE_STABLE_PHASES
    ):
        raise QualificationError(
            "reference equilibrium did not reproduce the published three-phase field"
        )
    gm_value = point.get("GM_J_per_mol")
    if (
        isinstance(gm_value, bool)
        or not isinstance(gm_value, int | float)
        or not math.isfinite(gm_value)
        or not math.isclose(
            float(gm_value),
            REFERENCE_GM_J_PER_MOL,
            rel_tol=0.0,
            abs_tol=REFERENCE_GM_ABSOLUTE_TOLERANCE_J_PER_MOL,
        )
    ):
        raise QualificationError(
            "reference equilibrium did not reproduce the reviewed checkpoint Gibbs energy"
        )
    return {
        "checkpoint_id": "wang2017_fig12a_alco_al4w_al5co2_three_phase",
        "evidence_scope": "same-assessment cross-engine phase-field reproduction",
        "temperature_K": REFERENCE_TEMPERATURE_K,
        "pressure_Pa": REFERENCE_PRESSURE_PA,
        "bulk_composition_mole_fraction": {"AL": 0.675, "CO": 0.26, "W": 0.065},
        "global_phase_count": len(REFERENCE_PHASES),
        "expected_stable_phases": list(REFERENCE_STABLE_PHASES),
        "observed_stable_phases": list(observed_phases),
    }


def write_request(request_root: Path, request: dict[str, Any]) -> Path:
    payload = canonical_json(request)
    digest = sha256_bytes(payload)
    request_root.mkdir(parents=True, exist_ok=True)
    path = request_root / f"{digest}.json"
    path.write_bytes(payload)
    return path


def retain_exact_artifact(path: Path, payload: bytes, *, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError:
        existing = read_bounded(path, maximum=len(payload), label=f"existing {label}")
        if existing != payload:
            raise QualificationError(f"content-addressed {label} collision")
        return
    except OSError as exc:
        raise QualificationError(f"could not retain {label}") from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise QualificationError(f"could not finish retaining {label}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def replace_regular_file(path: Path, payload: bytes, *, label: str) -> None:
    """Replace one stable bundle pointer without following a final symlink."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_TRUNC
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o644)
    except OSError as exc:
        raise QualificationError(f"could not write {label}") from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise QualificationError(f"could not finish writing {label}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def parse_cli_result(stdout: bytes, *, operation: str) -> dict[str, Any]:
    markers: list[dict[str, Any]] = []
    for raw_line in stdout.splitlines():
        try:
            line = raw_line.decode("utf-8", "strict")
        except UnicodeDecodeError as exc:
            raise QualificationError(f"{operation} typed CLI emitted non-UTF-8 output") from exc
        if line.startswith(RESULT_MARKER):
            value = load_unique_json(
                line[len(RESULT_MARKER) :].encode("utf-8"),
                label=f"{operation} typed CLI result",
            )
            if not isinstance(value, dict):
                raise QualificationError(f"{operation} typed CLI result is not an object")
            markers.append(value)
    if (
        len(markers) != 1
        or markers[0].get("ok") is not True
        or markers[0].get("operation") != operation
    ):
        raise QualificationError(f"{operation} typed CLI did not emit one successful result")
    return markers[0]


HOST_DRIVER = r"""
import os
from pathlib import Path
import ultra_deepagents.materials.calphad_cli as cli
workspace = Path(os.environ["ULTRA_CALPHAD_HOST_WORKSPACE"]).resolve()
outputs = Path(os.environ["ULTRA_CALPHAD_HOST_OUTPUTS"]).resolve()
cli.WORKSPACE_ROOT = workspace
cli.REQUEST_ROOT = workspace / ".ultra/calphad/requests"
cli.STAGED_DATABASE_ROOTS = (workspace / ".ultra/calphad/staged",)
cli.OUTPUT_ROOT = outputs / "calphad"
request = str(Path(os.environ["ULTRA_CALPHAD_HOST_REQUEST"]).resolve())
raise SystemExit(cli.main(["--request", request]))
""".strip()


def docker_typed_cli_command(
    *,
    runtime_image_id: str,
    trusted_runtime_root: str,
    workspace: Path,
    outputs: Path,
    request_path: Path,
) -> tuple[str, ...]:
    if trusted_runtime_root not in {"/opt/ultra-runtime", "/opt/ultra/src"}:
        raise QualificationError("typed CLI runtime root is not an approved baked path")
    container_request = "/workspace/" + request_path.relative_to(workspace).as_posix()
    bootstrap = (
        "import sys;"
        f"sys.path.insert(0,{trusted_runtime_root!r});"
        "from ultra_deepagents.materials.calphad_cli import main;"
        "raise SystemExit(main(['--request',sys.argv[1]]))"
    )
    return (
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=1g",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "4096",
        "--cpus",
        "8",
        "--memory",
        "32g",
        "--mount",
        f"type=bind,src={workspace},dst=/workspace,readonly",
        "--mount",
        f"type=bind,src={outputs},dst=/outputs",
        "--entrypoint",
        "python3",
        runtime_image_id,
        "-I",
        "-c",
        bootstrap,
        container_request,
    )


def execute_typed_cli(
    root: Path,
    *,
    runtime: RuntimeAttestation,
    workspace: Path,
    outputs: Path,
    request_path: Path,
    operation: str,
) -> dict[str, Any]:
    if runtime.mode == "pinned_image":
        command = docker_typed_cli_command(
            runtime_image_id=runtime.runtime_image_id,
            trusted_runtime_root=runtime.pythonpath,
            workspace=workspace,
            outputs=outputs,
            request_path=request_path,
        )
        environment = None
    else:
        command = (
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--python",
            "3.11",
            "--with",
            f"pycalphad=={PYCALPHAD_VERSION}",
            "python",
            "-c",
            HOST_DRIVER,
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root / "backend/deepagents_runtime/src")
        environment["PYTHONHASHSEED"] = "0"
        environment["ULTRA_CALPHAD_HOST_WORKSPACE"] = str(workspace)
        environment["ULTRA_CALPHAD_HOST_OUTPUTS"] = str(outputs)
        environment["ULTRA_CALPHAD_HOST_REQUEST"] = str(request_path)
    completed = run_bounded(command, cwd=root, env=environment, timeout=120)
    if completed.returncode != 0:
        diagnostic = (
            completed.stdout.decode("utf-8", "replace")
            + "\n"
            + completed.stderr.decode("utf-8", "replace")
        ).strip()
        raise QualificationError(
            f"{operation} typed CLI failed with exit {completed.returncode}: " + diagnostic[-2000:]
        )
    return parse_cli_result(completed.stdout, operation=operation)


def validate_artifact(
    path: Path,
    *,
    operation: str,
    runtime_image_id: str,
    resource_id: str,
    database_sha256: str,
    database_size_bytes: int,
    inspection_sha256: str = "",
    require_canonical_staged_path: bool = False,
) -> dict[str, Any]:
    payload = read_bounded(path, maximum=MAX_ARTIFACT_BYTES, label=f"{operation} artifact")
    digest = sha256_bytes(payload)
    if path.name != f"{digest}.json":
        raise QualificationError(f"{operation} artifact is not content-addressed")
    evidence = load_unique_json(payload, label=f"{operation} artifact")
    if not isinstance(evidence, dict):
        raise QualificationError(f"{operation} artifact is not an object")
    if (
        evidence.get("schema_version") != TYPED_EVIDENCE_SCHEMA
        or evidence.get("operation") != operation
    ):
        raise QualificationError(
            f"{operation} artifact has the wrong typed-evidence schema: "
            f"schema={evidence.get('schema_version')!r} operation={evidence.get('operation')!r}"
        )
    binding = evidence.get("database_binding")
    request = evidence.get("request")
    result = evidence.get("result")
    execution = evidence.get("execution_contract")
    if (
        not isinstance(binding, dict)
        or not isinstance(request, dict)
        or not isinstance(result, dict)
        or not isinstance(execution, dict)
    ):
        raise QualificationError(
            f"{operation} artifact omits binding, request, result, or execution contract"
        )
    if (
        binding.get("kind") != "resource"
        or binding.get("resource_id") != resource_id
        or binding.get("database_format") not in {"tdb", "dat"}
        or binding.get("sha256") != database_sha256
        or binding.get("size_bytes") != database_size_bytes
        or binding.get("assessment_pressure_limits_Pa")
        != [REFERENCE_PRESSURE_PA, REFERENCE_PRESSURE_PA]
        or binding.get("binding_schema") != "ultra.selected_resource.v1"
        or binding.get("binding_authority") != "control_resource_catalog"
        or binding.get("declaration_authority") != "resource_owner"
        or request.get("runtime_image_id") != runtime_image_id
        or request.get("operation") != operation
    ):
        raise QualificationError(
            f"{operation} artifact is not bound to the selected runtime/resource"
        )
    execution_keys = {
        "interface",
        "caller_code_accepted",
        "caller_models_or_solver_options_accepted",
        "network",
        "no_new_privileges",
        "read_only_root_filesystem",
        "cap_drop_all",
        "cpus_at_most",
        "memory_bytes_at_most",
        "pids_at_most",
        "runtime_image_id",
        "max_components",
        "max_phases",
        "max_axis_values",
        "max_grid_points",
        "wall_time_seconds",
        "max_result_bytes",
    }
    if (
        set(execution) != execution_keys
        or execution.get("interface") != "fixed ultra_deepagents.materials.calphad public surface"
        or execution.get("caller_code_accepted") is not False
        or execution.get("caller_models_or_solver_options_accepted") is not False
        or execution.get("network") != "none"
        or execution.get("no_new_privileges") is not True
        or execution.get("read_only_root_filesystem") is not True
        or execution.get("cap_drop_all") is not True
        or execution.get("cpus_at_most") != 8
        or execution.get("memory_bytes_at_most") != 32 * 1024**3
        or execution.get("pids_at_most") != 4096
        or execution.get("runtime_image_id") != runtime_image_id
        or execution.get("max_components") != 32
        or execution.get("max_phases") != 128
        or execution.get("max_axis_values") != 64
        or execution.get("max_grid_points") != 256
        or execution.get("wall_time_seconds") != 30
        or execution.get("max_result_bytes") != 16 * 1024 * 1024
    ):
        raise QualificationError(f"{operation} artifact execution/isolation contract is invalid")
    if operation == "inspect":
        result_database = result
        version = result_database.get("pycalphad_version")
    else:
        result_database = result.get("database")
        version = (
            result_database.get("pycalphad_version") if isinstance(result_database, dict) else None
        )
        if request.get("inspection_artifact_sha256") != inspection_sha256:
            raise QualificationError(
                "equilibrium artifact does not bind the exact inspection artifact"
            )
    if not isinstance(result_database, dict):
        raise QualificationError(f"{operation} artifact omits its database manifest")
    database_format = binding["database_format"]
    staged_name = str(result_database.get("name") or "")
    staged_path = str(result_database.get("path") or "")
    if (
        result_database.get("format") != database_format
        or staged_name != f"{database_sha256}.{database_format}"
        or Path(staged_path).suffix.casefold() != f".{database_format}"
    ):
        raise QualificationError(f"{operation} artifact database name is not content-addressed")
    if require_canonical_staged_path and staged_path != (
        "/workspace/.ultra/calphad/staged/" + staged_name
    ):
        raise QualificationError(
            f"{operation} artifact database path is not the canonical callback path"
        )
    if version != PYCALPHAD_VERSION:
        raise QualificationError(
            f"{operation} artifact was not generated by pycalphad {PYCALPHAD_VERSION}"
        )
    return {
        "path": path,
        "sha256": digest,
        "size_bytes": len(payload),
        "payload": payload,
        "evidence": evidence,
    }


def resolve_cli_artifact(outputs: Path, result: dict[str, Any], *, operation: str) -> Path:
    artifact = result.get("artifact")
    if not isinstance(artifact, dict):
        raise QualificationError(f"{operation} typed CLI omitted artifact metadata")
    declared_path = artifact.get("path")
    declared_sha = artifact.get("sha256")
    declared_size = artifact.get("size_bytes")
    directory = "inspection" if operation == "inspect" else "equilibrium"
    if (
        not isinstance(declared_sha, str)
        or SHA256_RE.fullmatch(declared_sha) is None
        or declared_path != f"/outputs/calphad/{directory}/{declared_sha}.json"
        or not isinstance(declared_size, int)
        or declared_size <= 0
    ):
        raise QualificationError(f"{operation} typed CLI returned invalid artifact metadata")
    path = outputs / "calphad" / directory / f"{declared_sha}.json"
    payload = read_bounded(path, maximum=MAX_ARTIFACT_BYTES, label=f"{operation} artifact")
    if sha256_bytes(payload) != declared_sha or len(payload) != declared_size:
        raise QualificationError(f"{operation} typed CLI artifact disagrees with its result marker")
    return path


def generate_artifacts(
    root: Path,
    *,
    runtime: RuntimeAttestation,
    expected_git_sha: str,
    retained_artifact_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    record, database = load_reference_database(root)
    resource_id = "calphad-cross-language-" + expected_git_sha[:20]
    with tempfile.TemporaryDirectory(prefix="ultra-calphad-cross-language-") as temporary:
        temp_root = Path(temporary).resolve()
        workspace = temp_root / "workspace"
        outputs = temp_root / "outputs"
        staged = workspace / ".ultra/calphad/staged"
        requests = workspace / ".ultra/calphad/requests"
        staged.mkdir(parents=True)
        # Pre-create bind-mounted output directories as the host user. The
        # sandbox runs as root in current images; if it creates these
        # directories itself, TemporaryDirectory cleanup cannot remove the
        # read-only content-addressed files on Linux CI.
        (outputs / "calphad/inspection").mkdir(parents=True)
        (outputs / "calphad/equilibrium").mkdir(parents=True)
        staged_name = staged_database_filename(record)
        staged_path = staged / staged_name
        staged_path.write_bytes(database)
        if runtime.mode == "pinned_image":
            request_database_path = "/workspace/.ultra/calphad/staged/" + staged_name
        else:
            request_database_path = str(staged_path)
        binding = database_binding(record, resource_id=resource_id, path=request_database_path)

        inspect_request = write_request(
            requests,
            typed_request(
                operation="inspect",
                runtime_image_id=runtime.runtime_image_id,
                binding=binding,
            ),
        )
        inspect_result = execute_typed_cli(
            root,
            runtime=runtime,
            workspace=workspace,
            outputs=outputs,
            request_path=inspect_request,
            operation="inspect",
        )
        inspect_path = resolve_cli_artifact(outputs, inspect_result, operation="inspect")
        inspect_artifact = validate_artifact(
            inspect_path,
            operation="inspect",
            runtime_image_id=runtime.runtime_image_id,
            resource_id=resource_id,
            database_sha256=str(record["sha256"]),
            database_size_bytes=int(record["size_bytes"]),
            require_canonical_staged_path=runtime.image_inspected,
        )

        equilibrium_request = write_request(
            requests,
            typed_request(
                operation="equilibrium",
                runtime_image_id=runtime.runtime_image_id,
                binding=binding,
                inspection_sha256=inspect_artifact["sha256"],
            ),
        )
        equilibrium_result = execute_typed_cli(
            root,
            runtime=runtime,
            workspace=workspace,
            outputs=outputs,
            request_path=equilibrium_request,
            operation="equilibrium",
        )
        equilibrium_path = resolve_cli_artifact(
            outputs, equilibrium_result, operation="equilibrium"
        )
        equilibrium_artifact = validate_artifact(
            equilibrium_path,
            operation="equilibrium",
            runtime_image_id=runtime.runtime_image_id,
            resource_id=resource_id,
            database_sha256=str(record["sha256"]),
            database_size_bytes=int(record["size_bytes"]),
            inspection_sha256=inspect_artifact["sha256"],
            require_canonical_staged_path=runtime.image_inspected,
        )
        equilibrium_artifact["scientific_checkpoint"] = validate_reference_equilibrium_checkpoint(
            equilibrium_artifact["evidence"]
        )

        retained_artifact_root.mkdir(parents=True, exist_ok=True)
        retained_database = retained_artifact_root / "database" / staged_name
        retained_inspect = retained_artifact_root / "inspection" / inspect_path.name
        retained_equilibrium = retained_artifact_root / "equilibrium" / equilibrium_path.name
        retain_exact_artifact(
            retained_database,
            database,
            label="CALPHAD database input",
        )
        retain_exact_artifact(
            retained_inspect,
            inspect_artifact["payload"],
            label="inspection artifact",
        )
        retain_exact_artifact(
            retained_equilibrium,
            equilibrium_artifact["payload"],
            label="equilibrium artifact",
        )

    inspect_artifact["path"] = retained_inspect
    equilibrium_artifact["path"] = retained_equilibrium
    return (
        inspect_artifact,
        equilibrium_artifact,
        {
            "resource_id": resource_id,
            "database_id": record["database_id"],
            "database_sha256": record["sha256"],
            "database_size_bytes": record["size_bytes"],
            "database_format": record["format"],
            "assessment_pressure_limits_Pa": record["assessment_pressure_limits_Pa"],
            "license_id": record["license_id"],
            "source": record["source_uri"],
        },
        {
            "path": retained_database,
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
            "format": record["format"],
        },
    )


def parse_go_test_json(payload: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    terminal_actions: list[str] = []
    package_terminal_actions: list[str] = []
    test_output: list[str] = []
    package = ""
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        value = load_unique_json(raw_line, label=f"go test JSON line {line_number}")
        if not isinstance(value, dict):
            raise QualificationError("go test JSON event is not an object")
        if value.get("Test") == GO_TEST and value.get("Action") in {"pass", "fail", "skip"}:
            terminal_actions.append(str(value["Action"]))
            package = str(value.get("Package") or "")
        if (
            not value.get("Test")
            and value.get("Package") == GO_PACKAGE
            and value.get("Action") in {"pass", "fail", "skip"}
        ):
            package_terminal_actions.append(str(value["Action"]))
        output = value.get("Output")
        if value.get("Test") == GO_TEST and isinstance(output, str):
            # `go test -json` may split one long test log line into multiple
            # Output events. Reconstruct only this exact test's ordered stream
            # before locating and decoding the evidence marker.
            test_output.append(output)
    if (
        terminal_actions != ["pass"]
        or package != GO_PACKAGE
        or package_terminal_actions != ["pass"]
    ):
        raise QualificationError(
            "cross-language Go test/package did not pass exactly once: "
            f"test_actions={terminal_actions!r} package_actions={package_terminal_actions!r} "
            f"package={package!r}"
        )
    combined_output = "".join(test_output)
    if combined_output.count(BACKEND_MARKER) != 1:
        raise QualificationError(
            "cross-language Go test did not emit exactly one backend evidence marker"
        )
    encoded = combined_output.split(BACKEND_MARKER, 1)[1].splitlines()[0].strip()
    marker = load_unique_json(encoded.encode("utf-8"), label="backend evidence marker")
    if not isinstance(marker, dict):
        raise QualificationError("backend evidence marker is not an object")
    return marker, {"name": GO_TEST, "package": package, "action": "pass"}


def validate_backend_marker(
    marker: dict[str, Any],
    *,
    database: dict[str, Any],
    runtime: RuntimeAttestation,
    resource: dict[str, Any],
    inspection: dict[str, Any],
    equilibrium: dict[str, Any],
) -> dict[str, bool]:
    required_top = {
        "schema_version",
        "live_http_callback",
        "live_postgres",
        "database",
        "resource_id",
        "revision_id",
        "run_id",
        "runtime_image_id",
        "pycalphad_version",
        "database_sha256",
        "database_size_bytes",
        "database_format",
        "assessment_pressure_limits_Pa",
        "database_inventory_sha256",
        "inspect",
        "equilibrium",
    }
    require_exact_keys(marker, required_top, label="backend evidence marker")
    backend_database = marker.get("database")
    inspect = marker.get("inspect")
    equilibrium_marker = marker.get("equilibrium")
    if (
        not isinstance(backend_database, dict)
        or not isinstance(inspect, dict)
        or not isinstance(equilibrium_marker, dict)
    ):
        raise QualificationError("backend marker nested evidence is invalid")
    require_exact_keys(
        backend_database,
        {
            "name",
            "server_address",
            "server_port",
            "connection_target_host",
            "connection_target_port",
            "transaction_read_only",
            "serving_role",
            "migration_role",
            "serving_role_superuser",
            "serving_role_create_role",
            "serving_role_create_database",
            "serving_role_replication",
            "serving_role_bypass_rls",
            "serving_role_owned_tables",
            "serving_role_owned_functions",
            "calphad_owner_roles",
            "calphad_reachable_roles",
            "calphad_owner_role_reachable",
            "public_schema_owner",
            "public_owner_role_reachable",
            "can_create_public_schema",
            "serving_role_select_all",
            "serving_role_insert_all",
            "serving_role_insert_any",
            "serving_role_execute_create_revision",
            "serving_role_execute_append_validation",
            "serving_writer_functions_exact",
            "serving_execute_unexpected_writer",
            "serving_role_execute_internal",
            "serving_role_public_execute",
            "serving_unexpected_table_acl_grantees",
            "serving_unexpected_function_acl_grantees",
            "serving_role_mutation_privilege",
        },
        label="backend PostgreSQL identity",
    )
    if (
        marker.get("schema_version") != BACKEND_EVIDENCE_SCHEMA
        or marker.get("live_http_callback") is not True
        or marker.get("live_postgres") is not True
        or marker.get("runtime_image_id") != runtime.runtime_image_id
        or marker.get("pycalphad_version") != PYCALPHAD_VERSION
        or marker.get("resource_id") != resource["resource_id"]
        or marker.get("database_sha256") != resource["database_sha256"]
        or marker.get("database_size_bytes") != resource["database_size_bytes"]
        or marker.get("database_format") != resource["database_format"]
        or marker.get("assessment_pressure_limits_Pa") != resource["assessment_pressure_limits_Pa"]
        or not isinstance(marker.get("revision_id"), str)
        or not marker["revision_id"]
        or not isinstance(marker.get("run_id"), str)
        or not marker["run_id"]
        or SHA256_RE.fullmatch(str(marker.get("database_inventory_sha256") or "")) is None
    ):
        raise QualificationError("backend marker does not bind live HTTP/Postgres evidence")
    if (
        backend_database.get("name") != database["database"]
        or backend_database.get("serving_role") != database["serving_role"]
        or backend_database.get("migration_role") != database["migration_role"]
        or backend_database.get("connection_target_host") != database["host"]
        or backend_database.get("connection_target_port") != database["port"]
        or backend_database.get("transaction_read_only") != "off"
        or backend_database.get("serving_role_superuser") is not False
        or backend_database.get("serving_role_create_role") is not False
        or backend_database.get("serving_role_create_database") is not False
        or backend_database.get("serving_role_replication") is not False
        or backend_database.get("serving_role_bypass_rls") is not False
        or backend_database.get("serving_role_owned_tables") != []
        or backend_database.get("serving_role_owned_functions") != []
        or not isinstance(backend_database.get("calphad_owner_roles"), list)
        or not backend_database["calphad_owner_roles"]
        or not all(
            isinstance(role, str) and role for role in backend_database["calphad_owner_roles"]
        )
        or backend_database.get("calphad_reachable_roles") != []
        or backend_database.get("calphad_owner_role_reachable") is not False
        or not isinstance(backend_database.get("public_schema_owner"), str)
        or not backend_database["public_schema_owner"].strip()
        or backend_database.get("public_owner_role_reachable") is not False
        or backend_database.get("can_create_public_schema") is not False
        or backend_database.get("serving_role_select_all") is not True
        or backend_database.get("serving_role_insert_all") is not False
        or backend_database.get("serving_role_insert_any") is not False
        or backend_database.get("serving_role_execute_create_revision") is not True
        or backend_database.get("serving_role_execute_append_validation") is not True
        or backend_database.get("serving_writer_functions_exact") is not True
        or backend_database.get("serving_execute_unexpected_writer") is not False
        or backend_database.get("serving_role_execute_internal") is not False
        or backend_database.get("serving_role_public_execute") is not False
        or backend_database.get("serving_unexpected_table_acl_grantees") != []
        or backend_database.get("serving_unexpected_function_acl_grantees") != []
        or backend_database.get("serving_role_mutation_privilege") is not False
    ):
        raise QualificationError("backend PostgreSQL identity or serving-role separation is unsafe")
    for label, record, artifact in (
        ("inspect", inspect, inspection),
        ("equilibrium", equilibrium_marker, equilibrium),
    ):
        if (
            record.get("evidence_sha256") != artifact["sha256"]
            or record.get("evidence_size_bytes") != artifact["size_bytes"]
            or SHA256_RE.fullmatch(str(record.get("request_sha256") or "")) is None
            or record.get("evidence_retention") != "retained"
            or record.get("promotable") is not True
            or record.get("postgres_bytes_exact") is not True
        ):
            raise QualificationError(
                f"backend {label} evidence is not exact, retained, and promotable"
            )
    if equilibrium_marker.get("inspection_evidence_sha256") != inspection["sha256"] or inspect.get(
        "request_sha256"
    ) == equilibrium_marker.get("request_sha256"):
        raise QualificationError("backend inspection/request lineage is invalid")
    return {
        "actual_typed_cli_artifacts": True,
        "pycalphad_0_11_2": True,
        "live_go_http_callback": True,
        "live_postgres": True,
        "role_separated_postgres": True,
        "exact_retained_evidence_bytes": True,
        "database_inventory_lineage": True,
        "inspection_equilibrium_lineage": True,
        "distinct_typed_request_hashes": True,
    }


def run_backend_qualification(
    root: Path,
    *,
    serving_dsn: str,
    migration_dsn: str,
    runtime: RuntimeAttestation,
    database_input_path: Path,
    inspection_path: Path,
    equilibrium_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    environment = os.environ.copy()
    environment.update(
        {
            "ULTRA_CONTROL_TEST_DATABASE_URL": serving_dsn,
            "ULTRA_CONTROL_TEST_MIGRATION_DATABASE_URL": migration_dsn,
            "ULTRA_CALPHAD_CROSS_LANGUAGE_QUALIFICATION": "1",
            "ULTRA_CALPHAD_DATABASE_INPUT_ARTIFACT": str(database_input_path),
            "ULTRA_CALPHAD_INSPECTION_ARTIFACT": str(inspection_path),
            "ULTRA_CALPHAD_EQUILIBRIUM_ARTIFACT": str(equilibrium_path),
            "ULTRA_CALPHAD_RUNTIME_IMAGE_ID": runtime.runtime_image_id,
        }
    )
    completed = run_bounded(
        GO_COMMAND,
        cwd=root / "backend/controlplane",
        env=environment,
        timeout=300,
    )
    if completed.returncode != 0:
        stdout_tail = completed.stdout.decode("utf-8", "replace")[-4000:]
        stderr_tail = completed.stderr.decode("utf-8", "replace")[-2000:]
        raise QualificationError(
            f"cross-language Go qualification exited {completed.returncode}; "
            f"stdout_tail={stdout_tail!r}; stderr_tail={stderr_tail!r}"
        )
    marker, test = parse_go_test_json(completed.stdout)
    return marker, test, completed.stdout


def write_content_addressed(
    output_dir: Path,
    name: str,
    payload: bytes,
    *,
    suffix: str = ".json",
) -> dict[str, Any]:
    if suffix not in {".json", ".jsonl"}:
        raise QualificationError("content-addressed output suffix is unsupported")
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = sha256_bytes(payload)
    path = output_dir / f"{name}-{digest}{suffix}"
    if path.exists():
        existing = read_bounded(
            path,
            maximum=len(payload),
            label=f"existing content-addressed {name} output",
        )
        if existing != payload:
            raise QualificationError(f"content-addressed output collision: {path.name}")
    else:
        retain_exact_artifact(path, payload, label=f"{name} output")
    return {"path": str(path), "sha256": digest, "size_bytes": len(payload)}


def relative_evidence_reference(reference: dict[str, Any], *, root: Path) -> dict[str, Any]:
    try:
        relative = Path(str(reference["path"])).resolve().relative_to(root.resolve())
    except (KeyError, OSError, ValueError) as exc:
        raise QualificationError("evidence path is outside its report bundle") from exc
    return {**reference, "path": relative.as_posix()}


def is_production_live_qualified(
    *,
    mode: str,
    repository: dict[str, Any],
    runtime: RuntimeAttestation,
    checks: dict[str, bool],
) -> bool:
    """Return the only condition that may be represented as a production pass."""

    return (
        mode == "pinned-image"
        and repository.get("clean") is True
        and runtime.image_inspected
        and bool(runtime.image_inspection_payload)
        and runtime.image_title == "Ultra Deep Agents scientific sandbox"
        and runtime.pythonpath == "/opt/ultra-runtime"
        and bool(checks)
        and all(value is True for value in checks.values())
    )


def run_gate(
    *,
    repository_root: Path,
    output_dir: Path,
    expected_git_sha: str,
    serving_dsn: str,
    migration_dsn: str,
    qualification_database_confirmed: bool,
    mode: str,
    image_ref: str = "",
    expected_image_id: str = "",
    expected_image_title: str = "Ultra Deep Agents scientific sandbox",
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = repository_root.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    if not qualification_database_confirmed:
        raise QualificationError("--qualification-database-confirmed is required")
    database = qualification_database_pair(serving_dsn, migration_dsn)
    repository = inspect_repository(root, expected_git_sha, require_clean=mode == "pinned-image")
    source_manifest = build_source_manifest(root)
    if mode == "pinned-image":
        runtime = inspect_runtime_image(
            root,
            image_ref=image_ref,
            expected_image_id=expected_image_id,
            expected_title=expected_image_title,
            expected_git_sha=expected_git_sha,
        )
    elif mode == "host-fallback":
        synthetic = sha256_bytes(
            f"non-oci-host-fallback:{expected_git_sha}:pycalphad-{PYCALPHAD_VERSION}".encode()
        )
        runtime = RuntimeAttestation(
            mode="host_fallback_non_oci",
            runtime_image_id="sha256:" + synthetic,
            image_title="non-OCI host fallback; not production evidence",
            image_revision=expected_git_sha,
            image_inspected=False,
        )
    else:
        raise QualificationError("mode must be pinned-image or host-fallback")

    image_inspection_evidence: dict[str, Any] | None = None
    if runtime.image_inspected:
        retained_inspection = write_content_addressed(
            output,
            "docker-image-inspect",
            runtime.image_inspection_payload,
        )
        image_inspection_evidence = relative_evidence_reference(
            retained_inspection,
            root=output,
        )

    artifact_root = output / "artifacts"
    inspection, equilibrium, resource, database_input = generate_artifacts(
        root,
        runtime=runtime,
        expected_git_sha=expected_git_sha,
        retained_artifact_root=artifact_root,
    )
    if runtime.image_inspected:
        backend_marker, go_test, go_log = run_backend_qualification(
            root,
            serving_dsn=serving_dsn,
            migration_dsn=migration_dsn,
            runtime=runtime,
            database_input_path=Path(database_input["path"]),
            inspection_path=inspection["path"],
            equilibrium_path=equilibrium["path"],
        )
        checks = validate_backend_marker(
            backend_marker,
            database=database,
            runtime=runtime,
            resource=resource,
            inspection=inspection,
            equilibrium=equilibrium,
        )
    else:
        # Host generation proves only that pycalphad and the typed producer can
        # calculate the fixed case. Its database result path is a host path, not
        # the canonical /workspace callback identity, and its v2 sandbox claims
        # were not enforced by Docker. Never submit or promote these artifacts.
        backend_marker = {
            "schema_version": BACKEND_EVIDENCE_SCHEMA,
            "live_http_callback": False,
            "live_postgres": False,
            "database": {
                "name": database["database"],
                "serving_role": database["serving_role"],
                "migration_role": database["migration_role"],
                "identity_verified_live": False,
            },
            "resource_id": resource["resource_id"],
            "revision_id": "",
            "run_id": "",
            "runtime_image_id": runtime.runtime_image_id,
            "pycalphad_version": PYCALPHAD_VERSION,
            "database_sha256": resource["database_sha256"],
            "database_size_bytes": resource["database_size_bytes"],
            "database_inventory_sha256": "",
            "inspect": {
                "evidence_sha256": inspection["sha256"],
                "evidence_size_bytes": inspection["size_bytes"],
                "request_sha256": "",
                "evidence_retention": "not_submitted",
                "promotable": False,
                "postgres_bytes_exact": False,
            },
            "equilibrium": {
                "evidence_sha256": equilibrium["sha256"],
                "evidence_size_bytes": equilibrium["size_bytes"],
                "request_sha256": "",
                "inspection_evidence_sha256": inspection["sha256"],
                "evidence_retention": "not_submitted",
                "promotable": False,
                "postgres_bytes_exact": False,
            },
        }
        go_test = {"name": GO_TEST, "package": GO_PACKAGE, "action": "not_run"}
        go_log = (
            canonical_json(
                {
                    "diagnostic": "host fallback artifacts are intentionally not submitted",
                    "production_live_qualified": False,
                }
            )
            + b"\n"
        )
        checks = {
            "actual_typed_cli_artifacts": True,
            "pycalphad_0_11_2": True,
            "live_go_http_callback": False,
            "live_postgres": False,
            "role_separated_postgres": False,
            "exact_retained_evidence_bytes": False,
            "database_inventory_lineage": False,
            "inspection_equilibrium_lineage": True,
            "distinct_typed_request_hashes": False,
        }
    checks["immutable_runtime_image_inspected"] = runtime.image_inspected
    checks["docker_image_inspection_retained"] = image_inspection_evidence is not None
    checks["pinned_sandbox_policy_enforced"] = runtime.image_inspected
    checks["clean_repository"] = repository["clean"]
    checks["image_revision_matches_git"] = (
        runtime.image_inspected and runtime.image_revision == expected_git_sha
    )
    production_live_qualified = is_production_live_qualified(
        mode=mode,
        repository=repository,
        runtime=runtime,
        checks=checks,
    )

    log_evidence = write_content_addressed(output, "go-test", go_log, suffix=".jsonl")
    bundled_log_evidence = relative_evidence_reference(log_evidence, root=output)
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate": "calphad-typed-cli-http-postgres-cross-language",
        "generated_at_utc": utc_now(),
        "expected_git_sha": expected_git_sha,
        "repository": repository,
        "source_manifest": source_manifest,
        "generation": {
            "mode": runtime.mode,
            "runtime_identity_kind": "immutable_oci_image"
            if runtime.image_inspected
            else "synthetic_non_oci",
            "image_ref": runtime.image_ref,
            "runtime_image_id": runtime.runtime_image_id,
            "image_title": runtime.image_title,
            "image_revision": runtime.image_revision,
            "pythonpath": runtime.pythonpath,
            "image_inspected": runtime.image_inspected,
            "docker_image_inspect": image_inspection_evidence,
            "pycalphad_version": PYCALPHAD_VERSION,
            "sandbox_policy": {
                "enforced_by_gate": runtime.image_inspected,
                "network": "none",
                "read_only_root_filesystem": True,
                "no_new_privileges": True,
                "cap_drop_all": True,
                "cpus_at_most": 8,
                "memory_bytes_at_most": 32 * 1024**3,
                "pids_at_most": 4096,
            },
        },
        "resource": resource,
        "typed_cli_artifacts": {
            "database_input": {
                "path": Path(database_input["path"]).resolve().relative_to(output).as_posix(),
                "sha256": database_input["sha256"],
                "size_bytes": database_input["size_bytes"],
                "format": database_input["format"],
            },
            "inspect": {
                "path": Path(inspection["path"]).resolve().relative_to(output).as_posix(),
                "sha256": inspection["sha256"],
                "size_bytes": inspection["size_bytes"],
            },
            "equilibrium": {
                "path": Path(equilibrium["path"]).resolve().relative_to(output).as_posix(),
                "sha256": equilibrium["sha256"],
                "size_bytes": equilibrium["size_bytes"],
                "inspection_artifact_sha256": inspection["sha256"],
                "scientific_checkpoint": equilibrium["scientific_checkpoint"],
            },
        },
        "backend": {
            "schema_version": backend_marker["schema_version"],
            "command": list(GO_COMMAND) if runtime.image_inspected else [],
            "test": go_test,
            "go_test_log": bundled_log_evidence,
            "live_http_callback": backend_marker["live_http_callback"],
            "live_postgres": backend_marker["live_postgres"],
            "database": backend_marker["database"],
            "resource_id": backend_marker["resource_id"],
            "revision_id": backend_marker["revision_id"],
            "run_id": backend_marker["run_id"],
            "runtime_image_id": backend_marker["runtime_image_id"],
            "pycalphad_version": backend_marker["pycalphad_version"],
            "database_sha256": backend_marker["database_sha256"],
            "database_size_bytes": backend_marker["database_size_bytes"],
            "database_format": backend_marker["database_format"],
            "assessment_pressure_limits_Pa": backend_marker["assessment_pressure_limits_Pa"],
            "database_inventory_sha256": backend_marker["database_inventory_sha256"],
            "inspect": backend_marker["inspect"],
            "equilibrium": backend_marker["equilibrium"],
        },
        "checks": checks,
        "production_live_qualified": production_live_qualified,
        "promotable": production_live_qualified,
        "status": "qualified" if production_live_qualified else "non_promotable_diagnostic",
    }
    report_payload = canonical_json(report) + b"\n"
    report_evidence = write_content_addressed(
        output, "calphad-cross-language-qualification", report_payload
    )
    manifest = {
        "schema_version": "ultra.calphad.cross-language-report-manifest.v1",
        "report": relative_evidence_reference(report_evidence, root=output),
        "production_live_qualified": production_live_qualified,
        "runtime_image_id": runtime.runtime_image_id,
        "expected_git_sha": expected_git_sha,
    }
    replace_regular_file(
        output / "report_manifest.json",
        canonical_json(manifest) + b"\n",
        label="report manifest",
    )
    return report, report_evidence


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--qualification-database-confirmed", action="store_true")
    parser.add_argument("--mode", choices=("pinned-image", "host-fallback"), default="pinned-image")
    parser.add_argument("--image", default="")
    parser.add_argument("--expected-image-id", default="")
    parser.add_argument(
        "--expected-image-title",
        choices=tuple(sorted(IMAGE_TITLES)),
        default="Ultra Deep Agents scientific sandbox",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report, report_evidence = run_gate(
            repository_root=args.repository_root,
            output_dir=args.output_dir,
            expected_git_sha=args.expected_git_sha.strip().lower(),
            serving_dsn=os.environ.get("ULTRA_CONTROL_TEST_DATABASE_URL", ""),
            migration_dsn=os.environ.get("ULTRA_CONTROL_TEST_MIGRATION_DATABASE_URL", ""),
            qualification_database_confirmed=args.qualification_database_confirmed,
            mode=args.mode,
            image_ref=args.image,
            expected_image_id=args.expected_image_id,
            expected_image_title=args.expected_image_title,
        )
    except QualificationError as exc:
        print(f"CALPHAD cross-language qualification failed: {exc}", file=sys.stderr)
        return 1
    print(f"CALPHAD cross-language report: {report_evidence['path']}")
    print(f"production_live_qualified={str(report['production_live_qualified']).lower()}")
    return 0 if report["production_live_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
