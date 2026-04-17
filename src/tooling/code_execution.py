"""Helpers for LLM-authored Python code generation and sandbox execution."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from openai import OpenAI
from src.config import get_settings
from src.logger import logger
from src.tooling.code_execution_jobs import build_service_submission_bundle
from src.tooling.code_execution_service_client import CodeExecutionServiceClient

_ALLOWED_DEPENDENCIES = {
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "sklearn",
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "matplotlib",
    "seaborn",
    "statsmodels",
    "scikit-image",
    "imageio",
    "tifffile",
    "pillow",
    "joblib",
    "networkx",
    "numba",
}

_DEFAULT_ENTRYPOINT = "main.py"
_DEFAULT_COMMAND = "python main.py"
_MAX_FILE_BYTES = 2_000_000
_MAX_FILES = 25
_MAX_INPUTS = 24
_MAX_STDIO_CHARS = 12000
_MAX_JSON_PARSE_BYTES = 1_000_000
_MAX_MEASUREMENT_CANDIDATES = 64
_MAX_ANALYSIS_OUTPUTS = 12
_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_REMOTE_INPUT_PREFIXES = ("s3://", "http://", "https://")


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _jobs_root() -> Path:
    settings = get_settings()
    root = Path(str(getattr(settings, "artifact_root", "data/artifacts"))).resolve()
    jobs_root = root / "code_jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    return jobs_root


def _job_dir(job_id: str) -> Path:
    token = str(job_id or "").strip()
    if not token:
        raise ValueError("job_id is required")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{6,128}", token):
        raise ValueError("job_id contains invalid characters")
    path = (_jobs_root() / token).resolve()
    jobs_root = _jobs_root().resolve()
    if jobs_root not in path.parents:
        raise ValueError("Invalid job_id path")
    return path


def _sanitize_relpath(path_value: str) -> str:
    value = str(path_value or "").strip().replace("\\", "/")
    if not value:
        raise ValueError("Empty file path")
    normalized = Path(value)
    if normalized.is_absolute():
        raise ValueError("Absolute file paths are not allowed")
    safe = str(Path(*[part for part in normalized.parts if part not in {"", ".", ".."}]))
    if not safe or safe.startswith("."):
        raise ValueError("Invalid file path")
    return safe


def _safe_input_mount_name(value: str, *, index: int) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._-")
    base = base[:80] or "input"
    return f"input_{index:02d}_{base}"


def _normalize_job_inputs(inputs: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(list(inputs or [])[:_MAX_INPUTS], start=1):
        if not isinstance(item, dict):
            continue
        raw_path = str(item.get("path") or "").strip()
        if not raw_path:
            continue
        source_path = Path(raw_path).expanduser()
        description = str(item.get("description") or "").strip()
        declared_kind = str(item.get("kind") or "").strip().lower()
        exists = source_path.exists()
        inferred_kind = "directory" if exists and source_path.is_dir() else "file"
        kind = declared_kind if declared_kind in {"file", "directory"} else inferred_kind
        mount_name = _safe_input_mount_name(source_path.name or inferred_kind, index=index)
        sandbox_path = f"/inputs/{mount_name}"
        normalized_item: dict[str, Any] = {
            "path": str(source_path),
            "sandbox_path": sandbox_path,
            "kind": kind,
            "description": description or None,
            "exists": exists,
            "name": source_path.name or mount_name,
        }
        if exists:
            try:
                normalized_item["size_bytes"] = int(source_path.stat().st_size)
            except Exception:
                pass
        normalized.append(
            {key: value for key, value in normalized_item.items() if value not in (None, "")}
        )
    return normalized


def _docker_input_mounts(inputs: list[dict[str, Any]] | None) -> list[tuple[str, str]]:
    mounts: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in list(inputs or []):
        if not isinstance(item, dict):
            continue
        host_path = str(item.get("path") or "").strip()
        sandbox_path = str(item.get("sandbox_path") or "").strip()
        if not host_path or not sandbox_path.startswith("/inputs/"):
            continue
        source_path = Path(host_path).expanduser()
        if not source_path.exists():
            continue
        mount = (str(source_path.resolve()), sandbox_path)
        if mount in seen:
            continue
        seen.add(mount)
        mounts.append(mount)
    return mounts


def _normalize_dependencies(dependencies: Any) -> tuple[list[str], list[str]]:
    deps_in = dependencies if isinstance(dependencies, list) else []
    allowed: list[str] = []
    dropped: list[str] = []
    for raw in deps_in[:64]:
        token = str(raw or "").strip()
        if not token:
            continue
        base = re.split(r"[<>=!~\s\[]+", token, maxsplit=1)[0].strip().lower()
        if base in _ALLOWED_DEPENDENCIES:
            allowed.append(token)
        else:
            dropped.append(token)
    dedup_allowed = list(dict.fromkeys(allowed))
    dedup_dropped = list(dict.fromkeys(dropped))
    return dedup_allowed, dedup_dropped


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        snippet = raw[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


def _is_valid_python_source(source: str) -> bool:
    candidate = str(source or "").strip()
    if not candidate:
        return False
    try:
        compile(candidate, "<codegen>", "exec")
    except Exception:
        return False
    return True


def _extract_python_source_from_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""

    # Prefer fenced python snippets when present.
    fenced_blocks = [str(block or "").strip() for block in _CODE_FENCE_RE.findall(raw)]
    for block in fenced_blocks:
        if _is_valid_python_source(block):
            return block

    if _is_valid_python_source(raw):
        return raw

    # Best-effort salvage: trim leading prose until the remainder compiles.
    lines = raw.splitlines()
    for idx in range(len(lines)):
        candidate = "\n".join(lines[idx:]).strip()
        if _is_valid_python_source(candidate):
            return candidate
    return ""


def _fallback_spec(
    task_summary: str,
    *,
    previous_failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    note = (
        str(previous_failure.get("error_message") or "").strip()
        if isinstance(previous_failure, dict)
        else ""
    )
    body = [
        "import json",
        "from pathlib import Path",
        "",
        f"TASK = {json.dumps(str(task_summary or '').strip())}",
        f"PREVIOUS_FAILURE = {json.dumps(note)}",
        "",
        "def main() -> None:",
        "    output = {",
        '        "task": TASK,',
        '        "status": "generated_fallback_script",',
        '        "previous_failure": PREVIOUS_FAILURE,',
        '        "message": "Replace this fallback with model-authored code."',
        "    }",
        '    Path("result.json").write_text(json.dumps(output, indent=2), encoding="utf-8")',
        '    print("Wrote result.json")',
        "",
        'if __name__ == "__main__":',
        "    main()",
    ]
    return {
        "entrypoint": _DEFAULT_ENTRYPOINT,
        "command": _DEFAULT_COMMAND,
        "files": [{"path": _DEFAULT_ENTRYPOINT, "content": "\n".join(body) + "\n"}],
        "dependencies": [],
        "expected_outputs": ["result.json"],
        "reasoning_summary": "Fallback script generated because model output was unavailable or invalid JSON.",
    }


def _resolved_codegen_client() -> tuple[OpenAI, str]:
    settings = get_settings()
    provider = str(getattr(settings, "resolved_codegen_provider", "openai")).strip().lower()
    base_url = str(getattr(settings, "resolved_codegen_base_url", "") or "").strip()
    api_key = getattr(settings, "resolved_codegen_api_key", None)
    if not api_key and provider in {"vllm", "ollama"}:
        api_key = "EMPTY"
    timeout = int(
        getattr(settings, "codegen_timeout_seconds", 0) or getattr(settings, "openai_timeout", 60)
    )
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=int(getattr(settings, "openai_max_retries", 2)),
    )
    model = str(getattr(settings, "resolved_codegen_model", "") or "").strip()
    logger.info(
        "Codegen client initialized: provider=%s base_url=%s model=%s",
        provider,
        base_url,
        model,
    )
    return client, model


def generate_python_job_spec(
    *,
    task_summary: str,
    inputs: list[dict[str, Any]] | None = None,
    constraints: dict[str, Any] | None = None,
    attempt_index: int = 1,
    previous_failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    if bool(getattr(settings, "llm_mock_mode", False)):
        return _fallback_spec(task_summary, previous_failure=previous_failure)

    normalized_inputs = _normalize_job_inputs(inputs)

    system_prompt = (
        "You generate Python job specifications for a sandboxed scientific execution platform.\n"
        "Return strict JSON only, with keys:\n"
        "entrypoint, command, files, dependencies, expected_outputs, reasoning_summary.\n"
        "Rules:\n"
        "- Python only.\n"
        "- files is a list of objects {path, content}.\n"
        "- Use relative paths only.\n"
        "- Keep code deterministic and robust to errors.\n"
        "- Prefer common scientific libraries (numpy, pandas, scipy, sklearn, opencv, scikit-image).\n"
        "- Never access network or absolute host paths.\n"
        "- If inputs are provided, read them from their sandbox_path values mounted read-only inside the container.\n"
        "- Never reference the original host path for an input file.\n"
    )
    user_payload = {
        "task_summary": str(task_summary or "").strip(),
        "inputs": normalized_inputs,
        "constraints": constraints or {},
        "attempt_index": int(max(1, attempt_index)),
        "previous_failure": previous_failure or {},
    }
    client, model = _resolved_codegen_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _stable_json(user_payload)},
            ],
            stream=False,
            temperature=0.1,
            max_tokens=4000,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Codegen model call failed; using fallback spec: %s", exc)
        return _fallback_spec(task_summary, previous_failure=previous_failure)
    content = ""
    try:
        if response.choices and response.choices[0].message:
            content = str(response.choices[0].message.content or "").strip()
    except Exception:
        content = ""
    parsed = _extract_first_json_object(content)
    if not isinstance(parsed, dict):
        raw_python = _extract_python_source_from_text(content)
        if raw_python:
            return {
                "entrypoint": _DEFAULT_ENTRYPOINT,
                "command": _DEFAULT_COMMAND,
                "files": [{"path": _DEFAULT_ENTRYPOINT, "content": raw_python.rstrip() + "\n"}],
                "dependencies": [],
                "expected_outputs": ["result.json"],
                "reasoning_summary": (
                    "Used raw Python source from non-JSON model output; "
                    "wrapped into a single-file job spec."
                ),
            }
        return _fallback_spec(task_summary, previous_failure=previous_failure)
    return parsed


def persist_python_job_spec(
    *,
    job_id: str | None,
    generated_spec: dict[str, Any],
    task_summary: str,
    inputs: list[dict[str, Any]] | None = None,
    constraints: dict[str, Any] | None = None,
    attempt_index: int = 1,
    previous_failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_job_id = str(job_id or "").strip() or f"codejob_{uuid4().hex[:16]}"
    job_path = _job_dir(normalized_job_id)
    source_dir = job_path / "source"
    versions_dir = job_path / "versions"
    source_dir.mkdir(parents=True, exist_ok=True)
    versions_dir.mkdir(parents=True, exist_ok=True)

    raw_files = generated_spec.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        generated_spec = _fallback_spec(task_summary, previous_failure=previous_failure)
        raw_files = generated_spec["files"]
    if len(raw_files) > _MAX_FILES:
        raw_files = raw_files[:_MAX_FILES]

    persisted_files: list[dict[str, Any]] = []
    source_manifest: dict[str, str] = {}
    for item in raw_files:
        if not isinstance(item, dict):
            continue
        rel_path = _sanitize_relpath(str(item.get("path") or ""))
        content = str(item.get("content") or "")
        if len(content.encode("utf-8")) > _MAX_FILE_BYTES:
            raise ValueError(f"Generated file too large: {rel_path}")
        abs_path = (source_dir / rel_path).resolve()
        if source_dir.resolve() not in abs_path.parents and abs_path != source_dir.resolve():
            raise ValueError(f"Rejected unsafe path: {rel_path}")
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")
        persisted_files.append({"path": rel_path, "size_bytes": abs_path.stat().st_size})
        source_manifest[rel_path] = _sha256_file(abs_path)

    persisted_file_paths = {str(item.get("path") or "").strip() for item in persisted_files}
    try:
        entrypoint = _sanitize_relpath(str(generated_spec.get("entrypoint") or _DEFAULT_ENTRYPOINT))
    except Exception:
        entrypoint = _DEFAULT_ENTRYPOINT
    if not entrypoint.endswith(".py"):
        entrypoint = _DEFAULT_ENTRYPOINT
    if entrypoint not in persisted_file_paths:
        if _DEFAULT_ENTRYPOINT in persisted_file_paths:
            entrypoint = _DEFAULT_ENTRYPOINT
        else:
            py_files = sorted([path for path in persisted_file_paths if path.endswith(".py")])
            entrypoint = py_files[0] if py_files else _DEFAULT_ENTRYPOINT
            if entrypoint not in persisted_file_paths:
                fallback_path = source_dir / _DEFAULT_ENTRYPOINT
                fallback_path.write_text(
                    "print('No valid Python entrypoint generated.')\n",
                    encoding="utf-8",
                )
                persisted_files.append(
                    {"path": _DEFAULT_ENTRYPOINT, "size_bytes": fallback_path.stat().st_size}
                )
                source_manifest[_DEFAULT_ENTRYPOINT] = _sha256_file(fallback_path)
                entrypoint = _DEFAULT_ENTRYPOINT

    raw_command = generated_spec.get("command")
    command = str(raw_command).strip() if isinstance(raw_command, str) else _DEFAULT_COMMAND
    if not command:
        command = _DEFAULT_COMMAND
    if command.strip() in {"python", "python3"}:
        command = f"{command.strip()} {entrypoint}"
    deps, dropped = _normalize_dependencies(generated_spec.get("dependencies"))
    expected_outputs = generated_spec.get("expected_outputs")
    if not isinstance(expected_outputs, list):
        expected_outputs = []
    normalized_expected_outputs = []
    for output in expected_outputs[:64]:
        token = _normalize_expected_output_token(str(output or ""))
        if token:
            normalized_expected_outputs.append(token)
    reasoning_summary = str(generated_spec.get("reasoning_summary") or "").strip()
    normalized_inputs = _normalize_job_inputs(inputs)

    spec = {
        "job_id": normalized_job_id,
        "task_summary": str(task_summary or "").strip(),
        "entrypoint": entrypoint,
        "command": command,
        "files": persisted_files,
        "dependencies": deps,
        "dropped_dependencies": dropped,
        "expected_outputs": normalized_expected_outputs,
        "reasoning_summary": reasoning_summary,
        "attempt_index": int(max(1, attempt_index)),
        "previous_failure": previous_failure or {},
        "inputs": normalized_inputs,
        "constraints": constraints or {},
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "source_manifest": source_manifest,
    }
    spec_path = job_path / "job_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    version_path = versions_dir / f"{int(time.time() * 1000)}.json"
    version_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    return {
        "success": True,
        "job_id": normalized_job_id,
        "job_dir": str(job_path),
        "spec_path": str(spec_path),
        "entrypoint": entrypoint,
        "command": command,
        "dependencies": deps,
        "dropped_dependencies": dropped,
        "expected_outputs": normalized_expected_outputs,
        "reasoning_summary": reasoning_summary,
        "attempt_index": spec["attempt_index"],
        "ui_artifacts": [
            {"path": str(spec_path), "title": "Python job spec", "type": "file"},
        ],
    }


def load_python_job_spec(job_id: str) -> dict[str, Any]:
    path = _job_dir(job_id) / "job_spec.json"
    if not path.exists():
        raise FileNotFoundError(f"Unknown job_id: {job_id}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid job spec payload: {job_id}")
    return data


def _docker_available() -> bool:
    try:
        subprocess.run(
            ["docker", "version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return True
    except Exception:
        return False


def _service_execution_enabled(settings: Any) -> bool:
    return bool(str(getattr(settings, "code_execution_service_url", "") or "").strip())


def _build_service_request_inputs(
    inputs: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[Path]]:
    normalized_inputs: list[dict[str, Any]] = []
    local_input_paths: list[Path] = []
    for item in list(inputs or []):
        if not isinstance(item, dict):
            continue
        sandbox_path = str(item.get("sandbox_path") or "").strip()
        if not sandbox_path.startswith("/inputs/"):
            continue
        raw_path = str(item.get("path") or "").strip()
        raw_kind = str(item.get("kind") or "file").strip().lower()
        kind = raw_kind if raw_kind in {"file", "directory"} else "file"
        name = str(item.get("name") or "").strip() or Path(raw_path or sandbox_path).name
        payload: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "sandbox_path": sandbox_path,
        }
        description = str(item.get("description") or "").strip()
        if description:
            payload["description"] = description
        if raw_path.startswith(_REMOTE_INPUT_PREFIXES):
            payload["uri"] = raw_path
        elif raw_path:
            local_path = Path(raw_path).expanduser()
            if local_path.exists():
                local_input_paths.append(local_path)
        normalized_inputs.append(payload)
    return normalized_inputs, local_input_paths


def _download_service_artifacts(
    *,
    client: CodeExecutionServiceClient,
    service_job_id: str,
    source_dir: Path,
    artifact_manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    downloaded_artifacts: list[dict[str, Any]] = []
    source_root = source_dir.resolve()
    seen_relative_paths: set[str] = set()
    for item in artifact_manifest:
        if not isinstance(item, dict):
            continue
        relative_path = _normalize_output_token(
            str(item.get("relative_path") or item.get("name") or item.get("path") or "")
        )
        artifact_name = str(item.get("name") or relative_path).strip()
        if not relative_path or not artifact_name or relative_path in seen_relative_paths:
            continue
        destination = (source_root / relative_path).resolve()
        if source_root not in destination.parents and destination != source_root:
            raise ValueError(f"Rejected service artifact path: {relative_path}")
        client.download_artifact(
            job_id=service_job_id,
            artifact_name=artifact_name,
            destination=destination,
        )
        downloaded_artifacts.append(
            {
                "path": str(destination),
                "relative_path": relative_path,
                "size_bytes": int(destination.stat().st_size),
                "kind": str(item.get("kind") or "file"),
                "title": str(item.get("title") or Path(relative_path).name or relative_path),
            }
        )
        seen_relative_paths.add(relative_path)
    return downloaded_artifacts


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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


def _collect_measurement_candidates(
    value: Any,
    *,
    prefix: str,
    sink: list[dict[str, Any]],
    max_items: int,
) -> None:
    if len(sink) >= max_items:
        return

    if _is_numeric_scalar(value):
        metric_name = prefix or "value"
        sink.append({"name": metric_name, "value": float(value)})
        return

    if isinstance(value, dict):
        for key, item in list(value.items())[:24]:
            if len(sink) >= max_items:
                break
            token = re.sub(r"[^a-zA-Z0-9_]+", "_", str(key or "").strip()).strip("_")
            token = token or "field"
            next_prefix = f"{prefix}.{token}" if prefix else token
            _collect_measurement_candidates(
                item,
                prefix=next_prefix,
                sink=sink,
                max_items=max_items,
            )
        return

    if isinstance(value, list):
        numeric_values = [float(item) for item in value if _is_numeric_scalar(item)]
        if numeric_values:
            metric_name = prefix or "values"
            count = len(numeric_values)
            mean_value = sum(numeric_values) / float(count)
            sink.append({"name": f"{metric_name}.count", "value": count})
            if len(sink) < max_items:
                sink.append({"name": f"{metric_name}.mean", "value": mean_value})
            if len(sink) < max_items:
                sink.append({"name": f"{metric_name}.min", "value": min(numeric_values)})
            if len(sink) < max_items:
                sink.append({"name": f"{metric_name}.max", "value": max(numeric_values)})
            if len(numeric_values) <= 4:
                for index, item in enumerate(numeric_values):
                    if len(sink) >= max_items:
                        break
                    sink.append({"name": f"{metric_name}[{index}]", "value": float(item)})
            return

        for index, item in enumerate(value[:3]):
            if len(sink) >= max_items:
                break
            next_prefix = f"{prefix}[{index}]" if prefix else f"item[{index}]"
            _collect_measurement_candidates(
                item,
                prefix=next_prefix,
                sink=sink,
                max_items=max_items,
            )


def _measurement_priority(name: str) -> tuple[int, str]:
    key = str(name or "").lower()
    high = (
        "iou",
        "dice",
        "jaccard",
        "accuracy",
        "auc",
        "f1",
        "precision",
        "recall",
        "specificity",
        "sensitivity",
        "explained_variance",
        "variance",
        "cell_count",
        "object_count",
        "area",
        "volume",
        "perimeter",
        "eccentricity",
        "r2",
        "rmse",
        "mae",
        "mse",
    )
    for rank, token in enumerate(high):
        if token in key:
            return (rank, key)
    return (len(high) + 1, key)


def _extract_json_artifact_summary(path: Path, *, relative_path: str) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    suffix = str(path.suffix or "").strip().lower()
    if suffix != ".json":
        return None
    if path.stat().st_size > _MAX_JSON_PARSE_BYTES:
        return {
            "path": relative_path,
            "kind": "json",
            "parse_status": "skipped_large_file",
            "size_bytes": int(path.stat().st_size),
            "measurement_candidates": [],
            "top_level_keys": [],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "path": relative_path,
            "kind": "json",
            "parse_status": "parse_error",
            "error": str(exc),
            "measurement_candidates": [],
            "top_level_keys": [],
        }

    measurements: list[dict[str, Any]] = []
    _collect_measurement_candidates(
        payload,
        prefix="",
        sink=measurements,
        max_items=_MAX_MEASUREMENT_CANDIDATES,
    )
    top_level_keys: list[str] = []
    if isinstance(payload, dict):
        top_level_keys = [str(key) for key in list(payload.keys())[:20]]
    return {
        "path": relative_path,
        "kind": "json",
        "parse_status": "ok",
        "top_level_keys": top_level_keys,
        "measurement_candidates": measurements,
    }


def _extract_csv_artifact_summary(path: Path, *, relative_path: str) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    suffix = str(path.suffix or "").strip().lower()
    if suffix != ".csv":
        return None
    if path.stat().st_size > _MAX_JSON_PARSE_BYTES:
        return {
            "path": relative_path,
            "kind": "csv",
            "parse_status": "skipped_large_file",
            "size_bytes": int(path.stat().st_size),
            "measurement_candidates": [],
            "top_level_keys": [],
        }

    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = [
                str(item or "").strip()
                for item in (reader.fieldnames or [])
                if str(item or "").strip()
            ]
            numeric_stats: dict[str, dict[str, float]] = {}
            row_count = 0
            for row in reader:
                if not isinstance(row, dict):
                    continue
                row_count += 1
                if row_count > 5000:
                    break
                for key, value in row.items():
                    if key is None:
                        continue
                    token = str(value or "").strip()
                    if not token:
                        continue
                    try:
                        numeric_value = float(token)
                    except Exception:
                        continue
                    stats = numeric_stats.setdefault(
                        str(key).strip(),
                        {
                            "count": 0.0,
                            "sum": 0.0,
                            "min": numeric_value,
                            "max": numeric_value,
                        },
                    )
                    stats["count"] += 1.0
                    stats["sum"] += numeric_value
                    stats["min"] = min(float(stats["min"]), numeric_value)
                    stats["max"] = max(float(stats["max"]), numeric_value)
    except Exception as exc:
        return {
            "path": relative_path,
            "kind": "csv",
            "parse_status": "parse_error",
            "error": str(exc),
            "measurement_candidates": [],
            "top_level_keys": [],
        }

    measurements: list[dict[str, Any]] = [{"name": "row_count", "value": float(row_count)}]
    stem = Path(relative_path).stem or "csv"
    for field in sorted(numeric_stats.keys())[:12]:
        stats = numeric_stats.get(field) or {}
        count = float(stats.get("count") or 0.0)
        if count <= 0:
            continue
        sum_value = float(stats.get("sum") or 0.0)
        measurements.append({"name": f"{stem}.{field}.count", "value": count})
        if len(measurements) >= _MAX_MEASUREMENT_CANDIDATES:
            break
        measurements.append({"name": f"{stem}.{field}.mean", "value": sum_value / count})
        if len(measurements) >= _MAX_MEASUREMENT_CANDIDATES:
            break
        measurements.append(
            {"name": f"{stem}.{field}.min", "value": float(stats.get("min") or 0.0)}
        )
        if len(measurements) >= _MAX_MEASUREMENT_CANDIDATES:
            break
        measurements.append(
            {"name": f"{stem}.{field}.max", "value": float(stats.get("max") or 0.0)}
        )
        if len(measurements) >= _MAX_MEASUREMENT_CANDIDATES:
            break

    return {
        "path": relative_path,
        "kind": "csv",
        "parse_status": "ok",
        "top_level_keys": fieldnames[:20],
        "measurement_candidates": measurements[:_MAX_MEASUREMENT_CANDIDATES],
    }


def _extract_execution_insights(
    *,
    source_dir: Path,
    artifacts: list[dict[str, Any]],
    expected_outputs: list[str],
) -> dict[str, Any]:
    normalized_output_files: list[str] = []
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        relative = _normalize_output_token(str(item.get("relative_path") or item.get("path") or ""))
        if relative:
            normalized_output_files.append(relative)
    normalized_output_files = list(dict.fromkeys(normalized_output_files))

    normalized_expected: list[str] = []
    for token in expected_outputs:
        normalized = _normalize_expected_output_token(token)
        if normalized:
            normalized_expected.append(normalized)
    normalized_expected = list(dict.fromkeys(normalized_expected))

    output_set = set(normalized_output_files)
    output_basenames = {Path(item).name for item in normalized_output_files if item}
    missing_expected: list[str] = []
    for expected in normalized_expected:
        if expected in output_set:
            continue
        if Path(expected).name in output_basenames:
            continue
        missing_expected.append(expected)

    analysis_outputs: list[dict[str, Any]] = []
    measurement_candidates: list[dict[str, Any]] = []
    seen_measurements: set[str] = set()
    for item in artifacts:
        if len(analysis_outputs) >= _MAX_ANALYSIS_OUTPUTS:
            break
        if not isinstance(item, dict):
            continue
        relative_path = _normalize_output_token(str(item.get("relative_path") or ""))
        if not relative_path:
            continue
        artifact_path = (source_dir / relative_path).resolve()
        summary = _extract_json_artifact_summary(artifact_path, relative_path=relative_path)
        if not isinstance(summary, dict):
            summary = _extract_csv_artifact_summary(artifact_path, relative_path=relative_path)
        if not isinstance(summary, dict):
            continue
        analysis_outputs.append(
            {
                "path": summary.get("path"),
                "kind": summary.get("kind"),
                "parse_status": summary.get("parse_status"),
                "top_level_keys": summary.get("top_level_keys"),
                "error": summary.get("error"),
                "size_bytes": summary.get("size_bytes"),
                "measurement_count": len(summary.get("measurement_candidates") or []),
            }
        )
        for metric in summary.get("measurement_candidates") or []:
            if not isinstance(metric, dict):
                continue
            name = str(metric.get("name") or "").strip()
            if not name or name in seen_measurements:
                continue
            value = metric.get("value")
            if not _is_numeric_scalar(value):
                continue
            measurement_candidates.append({"name": name, "value": float(value)})
            seen_measurements.add(name)
            if len(measurement_candidates) >= _MAX_MEASUREMENT_CANDIDATES:
                break
        if len(measurement_candidates) >= _MAX_MEASUREMENT_CANDIDATES:
            break

    key_measurements = sorted(
        measurement_candidates,
        key=lambda item: _measurement_priority(str(item.get("name") or "")),
    )[:12]

    return {
        "output_files": normalized_output_files,
        "expected_outputs": normalized_expected,
        "missing_expected_outputs": missing_expected,
        "analysis_outputs": analysis_outputs,
        "measurement_candidates": measurement_candidates[:_MAX_MEASUREMENT_CANDIDATES],
        "key_measurements": key_measurements,
    }


def _extract_stdout_json_summary(stdout_text: str) -> dict[str, Any] | None:
    payload = _extract_first_json_object(stdout_text)
    if not isinstance(payload, dict):
        return None
    measurements: list[dict[str, Any]] = []
    _collect_measurement_candidates(
        payload,
        prefix="stdout",
        sink=measurements,
        max_items=24,
    )
    return {
        "kind": "stdout_json",
        "parse_status": "ok",
        "top_level_keys": [str(key) for key in list(payload.keys())[:20]],
        "measurement_candidates": measurements,
    }


def _repair_hint(error_class: str | None) -> str:
    mapping = {
        "syntax_error": "Fix Python syntax issues and regenerate executable code.",
        "dependency_error": "Use only preinstalled/allowlisted dependencies and correct import names.",
        "resource_error": "Reduce memory/compute load, process smaller data chunks, and avoid large in-memory copies.",
        "timeout": "Optimize runtime complexity or reduce workload size to complete within timeout.",
        "policy_error": "Avoid restricted file-system/network operations and keep all paths relative to workspace.",
        "runtime_error": "Inspect traceback and handle edge cases (empty data, shape mismatches, missing columns).",
    }
    return mapping.get(error_class or "", "Inspect stderr traceback and update the generated code.")


def _execute_python_job_attempt_via_service(
    *,
    job_id: str,
    spec: dict[str, Any],
    source_dir: Path,
    effective_timeout: int,
    effective_cpu: float | str,
    effective_memory_mb: int,
) -> dict[str, Any]:
    settings = get_settings()
    service_url = str(getattr(settings, "code_execution_service_url", "") or "").strip()
    if not service_url:
        return {
            "success": False,
            "job_id": str(job_id),
            "execution_backend": "service",
            "error_class": "backend_unavailable",
            "error_message": (
                "Code execution service is not configured. Set CODE_EXECUTION_SERVICE_URL "
                "or use the local docker backend."
            ),
        }

    expected_output_tokens: list[str] = []
    for item in list(spec.get("expected_outputs") or []):
        token = _normalize_expected_output_token(str(item or ""))
        if token:
            expected_output_tokens.append(token)

    request_inputs, local_input_paths = _build_service_request_inputs(
        spec.get("inputs") if isinstance(spec.get("inputs"), list) else []
    )
    request_payload = {
        "job_id": str(job_id),
        "timeout_seconds": int(effective_timeout),
        "cpu_limit": float(effective_cpu),
        "memory_mb": int(effective_memory_mb),
        "expected_outputs": expected_output_tokens,
        "inputs": request_inputs,
    }
    client = CodeExecutionServiceClient(
        base_url=service_url,
        api_key=getattr(settings, "code_execution_service_api_key", None),
        timeout_seconds=float(getattr(settings, "code_execution_service_timeout_seconds", 60) or 60),
    )
    wait_timeout = int(
        max(
            effective_timeout + 120,
            int(getattr(settings, "code_execution_service_wait_timeout_seconds", 7200) or 7200),
        )
    )
    poll_interval = float(
        getattr(settings, "code_execution_service_poll_interval_seconds", 1.5) or 1.5
    )
    bundle_path, _remote_inputs, _local_inputs = build_service_submission_bundle(_job_dir(job_id))
    try:
        submitted = client.submit_job(
            request_payload=request_payload,
            bundle_path=bundle_path,
            local_input_paths=local_input_paths,
        )
        service_job_id = str(submitted.get("job_id") or "").strip() or str(job_id)
        terminal = client.wait_for_job(
            job_id=service_job_id,
            poll_interval_seconds=poll_interval,
            wait_timeout_seconds=wait_timeout,
        )
    finally:
        bundle_path.unlink(missing_ok=True)

    status = str(terminal.get("status") or "").strip().lower()
    raw_result = terminal.get("result")
    result_payload = dict(raw_result) if isinstance(raw_result, dict) else {}
    raw_manifest = terminal.get("artifact_manifest")
    artifact_manifest = list(raw_manifest) if isinstance(raw_manifest, list) else []
    downloaded_artifacts = _download_service_artifacts(
        client=client,
        service_job_id=service_job_id,
        source_dir=source_dir,
        artifact_manifest=artifact_manifest,
    )

    insights = _extract_execution_insights(
        source_dir=source_dir,
        artifacts=downloaded_artifacts,
        expected_outputs=expected_output_tokens,
    )
    measurement_candidates = list(
        result_payload.get("measurement_candidates") or insights.get("measurement_candidates") or []
    )
    key_measurements = list(result_payload.get("key_measurements") or [])
    if not key_measurements:
        key_measurements = sorted(
            [
                item
                for item in measurement_candidates
                if isinstance(item, dict) and str(item.get("name") or "").strip()
            ],
            key=lambda item: _measurement_priority(str(item.get("name") or "")),
        )[:12]

    success = bool(result_payload.get("success"))
    if not result_payload and status == "succeeded":
        success = True
    return {
        "success": success,
        "job_id": str(job_id),
        "service_job_id": service_job_id,
        "execution_backend": "service",
        "exit_code": result_payload.get("exit_code"),
        "runtime_seconds": result_payload.get("runtime_seconds"),
        "timeout_seconds": effective_timeout,
        "stdout_tail": str(result_payload.get("stdout_tail") or ""),
        "stderr_tail": str(result_payload.get("stderr_tail") or ""),
        "error_class": (
            str(result_payload.get("error_class") or "").strip()
            or ("service_execution_failed" if not success else None)
        ),
        "error_message": (
            str(result_payload.get("error_message") or "").strip()
            or str(terminal.get("error") or "").strip()
            or ("" if success else "Code execution service failed.")
        ),
        "repair_hint": (
            str(result_payload.get("repair_hint") or "").strip()
            or (None if success else _repair_hint(str(result_payload.get("error_class") or "")))
        ),
        "attempt_id": str(result_payload.get("attempt_id") or f"svc_{service_job_id}"),
        "artifacts": downloaded_artifacts,
        "output_files": list(result_payload.get("output_files") or insights.get("output_files") or []),
        "expected_outputs": list(
            result_payload.get("expected_outputs") or insights.get("expected_outputs") or []
        ),
        "missing_expected_outputs": list(
            result_payload.get("missing_expected_outputs")
            or insights.get("missing_expected_outputs")
            or []
        ),
        "analysis_outputs": list(
            result_payload.get("analysis_outputs") or insights.get("analysis_outputs") or []
        ),
        "measurement_candidates": measurement_candidates[:_MAX_MEASUREMENT_CANDIDATES],
        "key_measurements": key_measurements,
        "metrics": result_payload.get("metrics")
        or {
            "artifacts_count": len(downloaded_artifacts),
            "output_files_count": len(insights.get("output_files") or []),
            "missing_expected_outputs_count": len(insights.get("missing_expected_outputs") or []),
            "measurement_candidates_count": len(measurement_candidates),
        },
        "ui_artifacts": [
            {
                "path": item["path"],
                "title": str(item.get("title") or item.get("relative_path") or "artifact"),
                "type": "file",
                "kind": str(item.get("kind") or "file"),
            }
            for item in downloaded_artifacts[:80]
        ],
        "execution_command": str(spec.get("command") or _DEFAULT_COMMAND).strip() or _DEFAULT_COMMAND,
        "service_url": service_url,
    }


def _execute_python_job_attempt(
    *,
    job_id: str,
    spec: dict[str, Any],
    source_dir: Path,
    image: str,
    network: str,
    effective_timeout: int,
    effective_cpu: float | str,
    effective_memory_mb: int,
) -> dict[str, Any]:
    known_sources: dict[str, str] = {}
    for file_info in spec.get("files", []):
        if not isinstance(file_info, dict):
            continue
        rel = str(file_info.get("path") or "").strip()
        if not rel:
            continue
        source_file = (source_dir / rel).resolve()
        if source_file.exists() and source_file.is_file():
            known_sources[str(source_file)] = _sha256_file(source_file)

    command = str(spec.get("command") or _DEFAULT_COMMAND).strip() or _DEFAULT_COMMAND
    entrypoint = str(spec.get("entrypoint") or _DEFAULT_ENTRYPOINT).strip() or _DEFAULT_ENTRYPOINT
    container_entry = str(Path(entrypoint).as_posix())
    if not re.search(r"\bpython(?:3)?\b", command) and not command.endswith(container_entry):
        command = f"python {container_entry}"

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

    run_cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        network,
        "--cpus",
        str(effective_cpu),
        "--memory",
        f"{effective_memory_mb}m",
        "--pids-limit",
        "512",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,size=1024m",
        "--tmpfs",
        "/inputs:rw,size=64m",
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
    try:
        run_cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
    except Exception:
        pass
    for host_path, sandbox_path in _docker_input_mounts(
        spec.get("inputs") if isinstance(spec.get("inputs"), list) else []
    ):
        run_cmd.extend(["-v", f"{host_path}:{sandbox_path}:ro"])
    run_cmd.extend([image, "bash", "-lc", wrapped_command])

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
            timeout=effective_timeout,
        )
        exit_code = int(completed.returncode)
        stdout_text = str(completed.stdout or "")
        stderr_text = str(completed.stderr or "")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout_text = str(exc.stdout or "")
        stderr_text = str(exc.stderr or "")
        exit_code = 124
    except Exception as exc:
        stderr_text = f"Failed to execute docker run: {exc}"
        exit_code = 1

    runtime_seconds = round(time.time() - started, 3)
    artifacts: list[dict[str, Any]] = []
    for file_path in sorted(source_dir.rglob("*")):
        if not file_path.is_file():
            continue
        abs_path = str(file_path.resolve())
        prior_sha = known_sources.get(abs_path)
        current_sha = _sha256_file(file_path)
        if prior_sha and prior_sha == current_sha:
            continue
        rel = str(file_path.relative_to(source_dir))
        artifacts.append(
            {
                "path": abs_path,
                "relative_path": rel,
                "size_bytes": file_path.stat().st_size,
                "sha256": current_sha,
                "title": rel,
                "kind": "file",
            }
        )
        if len(artifacts) >= 200:
            break

    expected_outputs = spec.get("expected_outputs")
    expected_output_tokens: list[str] = []
    if isinstance(expected_outputs, list):
        for item in expected_outputs:
            token = _normalize_expected_output_token(str(item or ""))
            if token:
                expected_output_tokens.append(token)
    insights = _extract_execution_insights(
        source_dir=source_dir,
        artifacts=artifacts,
        expected_outputs=expected_output_tokens,
    )
    analysis_outputs = list(insights.get("analysis_outputs") or [])
    measurement_candidates = list(insights.get("measurement_candidates") or [])
    output_files = list(insights.get("output_files") or [])
    normalized_expected_outputs = list(insights.get("expected_outputs") or [])
    missing_expected_outputs = list(insights.get("missing_expected_outputs") or [])

    stdout_summary = _extract_stdout_json_summary(stdout_text)
    if isinstance(stdout_summary, dict):
        if len(analysis_outputs) < _MAX_ANALYSIS_OUTPUTS:
            analysis_outputs.append(
                {
                    "path": "stdout",
                    "kind": str(stdout_summary.get("kind") or "stdout_json"),
                    "parse_status": str(stdout_summary.get("parse_status") or "ok"),
                    "top_level_keys": stdout_summary.get("top_level_keys") or [],
                    "error": stdout_summary.get("error"),
                    "size_bytes": None,
                    "measurement_count": len(stdout_summary.get("measurement_candidates") or []),
                }
            )
        seen_measurements = {
            str(item.get("name") or "").strip()
            for item in measurement_candidates
            if isinstance(item, dict)
        }
        for metric in stdout_summary.get("measurement_candidates") or []:
            if not isinstance(metric, dict):
                continue
            name = str(metric.get("name") or "").strip()
            value = metric.get("value")
            if not name or name in seen_measurements:
                continue
            if not _is_numeric_scalar(value):
                continue
            measurement_candidates.append({"name": name, "value": float(value)})
            seen_measurements.add(name)
            if len(measurement_candidates) >= _MAX_MEASUREMENT_CANDIDATES:
                break

    key_measurements = sorted(
        [
            item
            for item in measurement_candidates
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        ],
        key=lambda item: _measurement_priority(str(item.get("name") or "")),
    )[:12]

    error_class = _classify_execution_error(
        timed_out=timed_out,
        exit_code=exit_code,
        stderr_text=stderr_text,
    )
    failure_signature = ""
    if error_class:
        import hashlib

        digest = hashlib.sha256(
            f"{error_class}|{stderr_text[-2000:]}".encode("utf-8", errors="ignore")
        ).hexdigest()
        failure_signature = digest
    success = (exit_code == 0) and not timed_out
    return {
        "success": success,
        "job_id": str(job_id),
        "execution_backend": "docker",
        "exit_code": exit_code,
        "runtime_seconds": runtime_seconds,
        "timeout_seconds": effective_timeout,
        "stdout_tail": stdout_text[-_MAX_STDIO_CHARS:],
        "stderr_tail": stderr_text[-_MAX_STDIO_CHARS:],
        "error_class": error_class,
        "error_message": (
            "" if success else (stderr_text[-_MAX_STDIO_CHARS:] or "Python execution failed.")
        ),
        "failure_signature": failure_signature or None,
        "repair_hint": None if success else _repair_hint(error_class),
        "attempt_id": f"exec_{uuid4().hex[:12]}",
        "artifacts": artifacts[:200],
        "output_files": output_files,
        "expected_outputs": normalized_expected_outputs,
        "missing_expected_outputs": missing_expected_outputs,
        "analysis_outputs": analysis_outputs,
        "measurement_candidates": measurement_candidates[:_MAX_MEASUREMENT_CANDIDATES],
        "key_measurements": key_measurements,
        "metrics": {
            "artifacts_count": len(artifacts),
            "output_files_count": len(output_files),
            "missing_expected_outputs_count": len(missing_expected_outputs),
            "measurement_candidates_count": len(measurement_candidates),
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
        "docker_image": image,
    }


def execute_python_job_once(
    *,
    job_id: str,
    timeout_seconds: int | None = None,
    cpu_limit: float | str | None = None,
    memory_mb: int | None = None,
    auto_repair: bool = True,
    max_repair_cycles: int | None = None,
    execution_backend: str | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    backend = (
        str(execution_backend or getattr(settings, "code_execution_default_backend", "docker"))
        .strip()
        .lower()
    )
    service_enabled = _service_execution_enabled(settings)
    if backend not in {"docker", "service"}:
        return {
            "success": False,
            "job_id": str(job_id),
            "error_class": "unsupported_backend",
            "error_message": (
                f"Unsupported execution_backend={backend}. "
                "v1 supports docker and the dedicated service backend."
            ),
        }
    if backend == "service" and not service_enabled:
        return {
            "success": False,
            "job_id": str(job_id),
            "error_class": "backend_unavailable",
            "error_message": (
                "Code execution service is not configured. Set CODE_EXECUTION_SERVICE_URL "
                "or use the docker backend."
            ),
        }
    if not service_enabled and not _docker_available():
        return {
            "success": False,
            "job_id": str(job_id),
            "error_class": "backend_unavailable",
            "error_message": "Docker is not available. Build/start Docker and retry execute_python_job.",
        }

    image = ""
    network = ""
    if not service_enabled:
        image = str(
            getattr(settings, "code_execution_docker_image", "bisque-ultra-codeexec:py311")
        ).strip()
        network = str(getattr(settings, "code_execution_docker_network", "none")).strip() or "none"
    timeout_default = int(getattr(settings, "code_execution_default_timeout_seconds", 900))
    timeout_cap = int(getattr(settings, "code_execution_max_timeout_seconds", 3600))
    effective_timeout = int(timeout_seconds or timeout_default)
    effective_timeout = max(1, min(effective_timeout, timeout_cap))

    default_cpu_limit = float(getattr(settings, "code_execution_default_cpu_limit", 2.0))
    default_memory_mb = int(getattr(settings, "code_execution_default_memory_mb", 4096))
    effective_cpu = cpu_limit if cpu_limit is not None else default_cpu_limit
    effective_memory_mb = int(memory_mb or default_memory_mb)
    max_cycles = int(max_repair_cycles or getattr(settings, "code_execution_max_repair_cycles", 5))
    max_cycles = max(0, max_cycles)

    job_path = _job_dir(str(job_id))
    source_dir = (job_path / "source").resolve()
    if not source_dir.exists():
        return {
            "success": False,
            "job_id": str(job_id),
            "error_class": "invalid_spec",
            "error_message": f"source directory is missing for job_id={job_id}",
        }

    repairable_error_classes = {
        "syntax_error",
        "dependency_error",
        "runtime_error",
        "timeout",
        "resource_error",
    }
    stop_on_error_classes = {
        "policy_error",
    }
    attempt_history: list[dict[str, Any]] = []
    seen_failure_signatures: set[str] = set()
    cycles_used = 0
    stop_reason: str | None = None
    result: dict[str, Any] | None = None

    while True:
        spec = load_python_job_spec(job_id)
        attempt_index = int(max(1, int(spec.get("attempt_index") or 1)))
        if service_enabled:
            result = _execute_python_job_attempt_via_service(
                job_id=str(job_id),
                spec=spec,
                source_dir=source_dir,
                effective_timeout=effective_timeout,
                effective_cpu=effective_cpu,
                effective_memory_mb=effective_memory_mb,
            )
        else:
            result = _execute_python_job_attempt(
                job_id=str(job_id),
                spec=spec,
                source_dir=source_dir,
                image=image,
                network=network,
                effective_timeout=effective_timeout,
                effective_cpu=effective_cpu,
                effective_memory_mb=effective_memory_mb,
            )
        result["attempt_index"] = attempt_index
        attempt_history.append(
            {
                "attempt_index": attempt_index,
                "attempt_id": result.get("attempt_id"),
                "success": bool(result.get("success")),
                "error_class": result.get("error_class"),
                "exit_code": result.get("exit_code"),
                "runtime_seconds": result.get("runtime_seconds"),
                "failure_signature": result.get("failure_signature"),
            }
        )

        if result.get("success"):
            break

        error_class = str(result.get("error_class") or "").strip()
        signature = str(result.get("failure_signature") or "").strip()
        if not bool(auto_repair):
            stop_reason = "auto_repair_disabled"
            break
        if error_class in stop_on_error_classes:
            stop_reason = "non_repairable_error_class"
            break
        if error_class and error_class not in repairable_error_classes:
            stop_reason = "unsupported_error_class"
            break
        if signature and signature in seen_failure_signatures:
            stop_reason = "repeated_failure_signature"
            break
        if signature:
            seen_failure_signatures.add(signature)
        if cycles_used >= max_cycles:
            stop_reason = "max_repair_cycles_exhausted"
            break

        previous_failure = {
            "error_class": result.get("error_class"),
            "error_message": result.get("error_message"),
            "stderr_tail": result.get("stderr_tail"),
            "stdout_tail": result.get("stdout_tail"),
            "failure_signature": result.get("failure_signature"),
            "exit_code": result.get("exit_code"),
            "attempt_id": result.get("attempt_id"),
            "attempt_index": attempt_index,
        }
        try:
            prepare_python_job(
                task_summary=str(spec.get("task_summary") or ""),
                job_id=str(job_id),
                inputs=spec.get("inputs") if isinstance(spec.get("inputs"), list) else [],
                constraints=spec.get("constraints")
                if isinstance(spec.get("constraints"), dict)
                else {},
                attempt_index=attempt_index + 1,
                previous_failure=previous_failure,
            )
        except Exception as exc:  # noqa: BLE001
            stop_reason = "repair_codegen_failed"
            result["error_message"] = (
                f"{result.get('error_message')}\n\nRepair generation failed: {exc}"
            ).strip()
            break
        cycles_used += 1

    if result is None:
        result = {
            "success": False,
            "job_id": str(job_id),
            "execution_backend": "service" if service_enabled else "docker",
            "error_class": "unknown_execution_error",
            "error_message": "Execution ended before producing a result.",
        }
    result["repair_cycles_used"] = cycles_used
    result["max_repair_cycles"] = max_cycles
    result["attempt_history"] = attempt_history[:20]
    if stop_reason:
        result["repair_stop_reason"] = stop_reason
    if result.get("success"):
        if cycles_used > 0:
            result["message"] = f"Python job succeeded after {cycles_used} repair cycle(s)."
    else:
        result["repair_exhausted"] = stop_reason in {
            "max_repair_cycles_exhausted",
            "repeated_failure_signature",
            "repair_codegen_failed",
        }
    latest_result_path = job_path / "latest_execution.json"
    latest_result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def prepare_python_job(
    *,
    task_summary: str,
    job_id: str | None = None,
    inputs: list[dict[str, Any]] | None = None,
    constraints: dict[str, Any] | None = None,
    attempt_index: int = 1,
    previous_failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    generated_spec = generate_python_job_spec(
        task_summary=task_summary,
        inputs=inputs,
        constraints=constraints,
        attempt_index=attempt_index,
        previous_failure=previous_failure,
    )
    return persist_python_job_spec(
        job_id=job_id,
        generated_spec=generated_spec,
        task_summary=task_summary,
        inputs=inputs,
        constraints=constraints,
        attempt_index=attempt_index,
        previous_failure=previous_failure,
    )


def reset_python_job(job_id: str) -> None:
    target = _job_dir(job_id)
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
