"""Docker-backed execution runner for the code execution service."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

_DEFAULT_ENTRYPOINT = "main.py"
_DEFAULT_COMMAND = "python main.py"
_MAX_STDIO_CHARS = 12000


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_output_token(value: str | None) -> str:
    token = str(value or "").strip().replace("\\", "/")
    while token.startswith("./"):
        token = token[2:]
    return token


def _normalize_expected_output_token(value: str | None) -> str:
    token = _normalize_output_token(value)
    if not token:
        return ""
    if len(token) > 180:
        return ""
    if any(char in token for char in ("\n", "\r", "\t")):
        return ""
    if " " in token:
        return ""
    if token.endswith(":") or token.endswith("..."):
        return ""
    if re.search(r"[<>{}|`]", token):
        return ""
    filename = Path(token).name
    if "." not in filename:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9._/\-+]+", token):
        return ""
    return token


def _classify_execution_error(*, timed_out: bool, exit_code: int, stderr_text: str) -> str | None:
    stderr = str(stderr_text or "")
    if timed_out:
        return "timeout"
    if exit_code == 0:
        return None
    if "SyntaxError" in stderr:
        return "syntax_error"
    if "ModuleNotFoundError" in stderr or "ImportError" in stderr:
        return "dependency_error"
    if "MemoryError" in stderr or "Cannot allocate memory" in stderr:
        return "resource_error"
    if "PermissionError" in stderr:
        return "policy_error"
    return "runtime_error"


def _repair_hint(error_class: str | None) -> str:
    mapping = {
        "syntax_error": "Fix Python syntax issues and regenerate executable code.",
        "dependency_error": "Use only preinstalled/allowlisted dependencies and correct import names.",
        "resource_error": "Reduce memory/compute load, process smaller data chunks, and avoid large in-memory copies.",
        "timeout": "Optimize runtime complexity or reduce workload size to complete within timeout.",
        "policy_error": "Avoid restricted file-system/network operations and keep all paths relative to workspace.",
        "runtime_error": "Inspect stderr traceback and handle edge cases (empty data, shape mismatches, missing columns).",
    }
    return mapping.get(error_class or "", "Inspect stderr traceback and update the generated code.")


def _collect_artifacts(source_dir: Path, known_sources: dict[str, str]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for file_path in sorted(source_dir.rglob("*")):
        if not file_path.is_file():
            continue
        abs_path = str(file_path.resolve())
        prior_sha = known_sources.get(abs_path)
        current_sha = _sha256_file(file_path)
        if prior_sha and prior_sha == current_sha:
            continue
        relative_path = str(file_path.relative_to(source_dir))
        artifacts.append(
            {
                "path": abs_path,
                "relative_path": relative_path,
                "size_bytes": int(file_path.stat().st_size),
                "sha256": current_sha,
                "title": relative_path,
                "kind": "file",
            }
        )
        if len(artifacts) >= 200:
            break
    return artifacts


def _expected_output_summary(
    *,
    artifacts: list[dict[str, Any]],
    expected_outputs: list[str],
) -> tuple[list[str], list[str], list[str]]:
    output_files = [
        token
        for token in (
            _normalize_output_token(str(item.get("relative_path") or item.get("path") or ""))
            for item in artifacts
        )
        if token
    ]
    output_files = list(dict.fromkeys(output_files))
    normalized_expected = [
        token for token in (_normalize_expected_output_token(item) for item in expected_outputs) if token
    ]
    normalized_expected = list(dict.fromkeys(normalized_expected))
    output_set = set(output_files)
    output_basenames = {Path(item).name for item in output_files if item}
    missing_expected = [
        expected
        for expected in normalized_expected
        if expected not in output_set and Path(expected).name not in output_basenames
    ]
    return output_files, normalized_expected, missing_expected


def run_codeexec_attempt(
    *,
    job_id: str,
    work_dir: Path,
    request: dict[str, Any],
    worker_image: str,
    docker_network: str,
) -> dict[str, Any]:
    """Run one execution attempt inside the curated worker image."""

    spec_path = work_dir / "job_spec.json"
    source_dir = (work_dir / "source").resolve()
    inputs_dir = (work_dir / "inputs").resolve()
    if not spec_path.exists():
        return {
            "success": False,
            "job_id": str(job_id),
            "execution_backend": "service",
            "error_class": "invalid_spec",
            "error_message": f"Missing job_spec.json for job_id={job_id}",
        }
    if not source_dir.exists():
        return {
            "success": False,
            "job_id": str(job_id),
            "execution_backend": "service",
            "error_class": "invalid_spec",
            "error_message": f"Missing source directory for job_id={job_id}",
        }

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    remote_input_uris = [
        str(item.get("uri") or "").strip()
        for item in list(request.get("inputs") or [])
        if isinstance(item, dict) and str(item.get("uri") or "").strip()
    ]
    if remote_input_uris:
        return {
            "success": False,
            "job_id": str(job_id),
            "execution_backend": "service",
            "error_class": "unsupported_remote_inputs",
            "error_message": (
                "The code execution service currently requires staged local inputs for execution. "
                "Remote URIs must be downloaded by the caller before submission."
            ),
        }

    known_sources: dict[str, str] = {}
    for file_info in list(spec.get("files") or []):
        if not isinstance(file_info, dict):
            continue
        relative_path = str(file_info.get("path") or "").strip()
        if not relative_path:
            continue
        source_file = (source_dir / relative_path).resolve()
        if source_file.exists() and source_file.is_file():
            known_sources[str(source_file)] = _sha256_file(source_file)

    command = str(spec.get("command") or _DEFAULT_COMMAND).strip() or _DEFAULT_COMMAND
    entrypoint = str(spec.get("entrypoint") or _DEFAULT_ENTRYPOINT).strip() or _DEFAULT_ENTRYPOINT
    container_entrypoint = str(Path(entrypoint).as_posix())
    if not re.search(r"\bpython(?:3)?\b", command) and not command.endswith(container_entrypoint):
        command = f"python {container_entrypoint}"

    wrapped_command = "\n".join(
        [
            "export HOME=/tmp",
            "export XDG_CONFIG_HOME=/tmp/xdg",
            "export XDG_CACHE_HOME=/tmp/xdg-cache",
            "export MPLCONFIGDIR=/tmp/matplotlib",
            "export MPLBACKEND=Agg",
            'mkdir -p "$HOME" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"',
            command,
        ]
    )

    timeout_seconds = int(request.get("timeout_seconds") or 900)
    cpu_limit = request.get("cpu_limit") or 2.0
    memory_mb = int(request.get("memory_mb") or 4096)
    run_cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        str(docker_network or "none"),
        "--cpus",
        str(cpu_limit),
        "--memory",
        f"{memory_mb}m",
        "--pids-limit",
        "512",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,size=1024m",
        "-v",
        f"{source_dir}:/workspace",
        "-w",
        "/workspace",
        "-e",
        "HOME=/tmp",
        "-e",
        "XDG_CONFIG_HOME=/tmp/xdg",
        "-e",
        "XDG_CACHE_HOME=/tmp/xdg-cache",
        "-e",
        "MPLCONFIGDIR=/tmp/matplotlib",
        "-e",
        "MPLBACKEND=Agg",
    ]
    if inputs_dir.exists():
        run_cmd.extend(["-v", f"{inputs_dir}:/inputs:ro"])
    try:
        run_cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
    except Exception:
        pass
    run_cmd.extend([worker_image, "bash", "-lc", wrapped_command])

    started = time.time()
    timed_out = False
    stdout_text = ""
    stderr_text = ""
    exit_code = 1
    try:
        completed = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        exit_code = int(completed.returncode)
        stdout_text = str(completed.stdout or "")
        stderr_text = str(completed.stderr or "")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout_text = str(exc.stdout or "")
        stderr_text = str(exc.stderr or "")
        exit_code = 124
    except Exception as exc:  # noqa: BLE001
        stderr_text = f"Failed to execute docker run: {exc}"

    runtime_seconds = round(time.time() - started, 3)
    artifacts = _collect_artifacts(source_dir, known_sources)
    output_files, normalized_expected, missing_expected = _expected_output_summary(
        artifacts=artifacts,
        expected_outputs=list(request.get("expected_outputs") or list(spec.get("expected_outputs") or [])),
    )
    error_class = _classify_execution_error(
        timed_out=timed_out,
        exit_code=exit_code,
        stderr_text=stderr_text,
    )
    success = (exit_code == 0) and not timed_out
    return {
        "success": success,
        "job_id": str(job_id),
        "execution_backend": "service",
        "exit_code": exit_code,
        "runtime_seconds": runtime_seconds,
        "stdout_tail": stdout_text[-_MAX_STDIO_CHARS:],
        "stderr_tail": stderr_text[-_MAX_STDIO_CHARS:],
        "error_class": error_class,
        "error_message": (
            "" if success else (stderr_text[-_MAX_STDIO_CHARS:] or "Python execution failed.")
        ),
        "repair_hint": None if success else _repair_hint(error_class),
        "attempt_id": f"svc_exec_{uuid4().hex[:12]}",
        "artifacts": artifacts,
        "output_files": output_files,
        "expected_outputs": normalized_expected,
        "missing_expected_outputs": missing_expected,
        "analysis_outputs": [],
        "measurement_candidates": [],
        "key_measurements": [],
        "metrics": {
            "artifacts_count": len(artifacts),
            "output_files_count": len(output_files),
            "missing_expected_outputs_count": len(missing_expected),
        },
        "ui_artifacts": [
            {
                "path": item["path"],
                "title": str(item.get("title") or item.get("relative_path") or "artifact"),
                "type": "file",
                "kind": str(item.get("kind") or "file"),
            }
            for item in artifacts[:80]
        ],
        "execution_command": command,
        "worker_image": worker_image,
    }
