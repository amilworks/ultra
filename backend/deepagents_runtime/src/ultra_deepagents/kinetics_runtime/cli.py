"""Typed JSON CLI for the network-disabled isolated kinetics image."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .contract import MAX_REQUEST_BYTES, safe_existing_file
from .errors import (
    KineticsError,
    KineticsExecutionError,
    KineticsInputError,
    KineticsTimeoutError,
    KineticsUnsupportedError,
)
from .runner import _canonical_json_bytes, execute_request, runtime_support


def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise KineticsInputError(f"request JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise KineticsInputError(f"request JSON contains non-finite number {value}")


def _load_request(path: Path) -> Any:
    payload = path.read_bytes()
    if len(payload) > MAX_REQUEST_BYTES:
        raise KineticsInputError(f"request exceeds {MAX_REQUEST_BYTES} bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise KineticsInputError("request must be UTF-8 JSON") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_closed_object,
            parse_constant=_reject_constant,
        )
    except KineticsInputError:
        raise
    except json.JSONDecodeError as exc:
        raise KineticsInputError("request is not valid JSON") from exc


def _safe_output_path(raw: str, *, output_root: Path) -> Path:
    root = output_root.resolve(strict=True)
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        parent = candidate.parent.resolve(strict=True)
        parent.relative_to(root)
    except FileNotFoundError as exc:
        raise KineticsInputError("output parent directory does not exist") from exc
    except ValueError as exc:
        raise KineticsInputError("output path escapes output_root") from exc
    if candidate.exists() and not candidate.is_file():
        raise KineticsInputError("output path must be a regular file")
    if candidate.is_symlink():
        raise KineticsInputError("output path must not be a symbolic link")
    return candidate


def _write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--request")
    parser.add_argument("--workspace-root", default="/workspace")
    parser.add_argument("--output")
    parser.add_argument("--output-root", default="/outputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.self_check:
            if args.request or args.output:
                raise KineticsInputError(
                    "--self-check cannot be combined with --request or --output"
                )
            sys.stdout.buffer.write(_canonical_json_bytes(runtime_support()) + b"\n")
            return 0
        if not args.request:
            raise KineticsInputError("--request is required")
        workspace_root = Path(args.workspace_root).resolve(strict=True)
        request_path = safe_existing_file(
            args.request,
            workspace_root=workspace_root,
            field="request path",
        )
        result = execute_request(_load_request(request_path), workspace_root=workspace_root)
        payload = _canonical_json_bytes(result) + b"\n"
        if args.output:
            output_path = _safe_output_path(args.output, output_root=Path(args.output_root))
            _write_atomic(output_path, payload)
        else:
            sys.stdout.buffer.write(payload)
        return 0
    except KineticsError as exc:
        error = {
            "schema_version": "ultra.materials.kinetics-error.v1",
            "error": {"code": exc.code, "message": str(exc)},
        }
        sys.stderr.buffer.write(_canonical_json_bytes(error) + b"\n")
        if isinstance(exc, KineticsTimeoutError):
            return 5
        if isinstance(exc, KineticsUnsupportedError):
            return 3
        if isinstance(exc, KineticsExecutionError):
            return 4
        return 2
    except (FileNotFoundError, OSError) as exc:
        error = {
            "schema_version": "ultra.materials.kinetics-error.v1",
            "error": {"code": "io_failure", "message": str(exc)},
        }
        sys.stderr.buffer.write(_canonical_json_bytes(error) + b"\n")
        return 6


if __name__ == "__main__":
    raise SystemExit(main())
