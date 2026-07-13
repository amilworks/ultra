#!/usr/bin/env python3
"""Run the exact pinned MatTools scorer with a synthetic safe ``utils`` module."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
import sys
import types
from pathlib import Path

from mattools_safe_parser import SafeComplexDictParser, parse_complex_string

OFFICIAL_RUNNER_SHA256 = "4004d0a9d7b103a0a29ada96d7ac7b7977f7cb6fdd73d2cee774b8fd62cc4d70"
OFFICIAL_UNSAFE_UTILS_SHA256 = "ee41f88d71f11997d8160294fb5ad200b294de823968a75d1ac1cc1352d9ec29"


class RunnerWrapperError(RuntimeError):
    """Raised when the reviewed scorer boundary is not exact."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    absolute = path.expanduser().absolute()
    if absolute.is_symlink() or not absolute.is_file():
        raise RunnerWrapperError(f"{label} must be a regular non-symlink file")
    return absolute.resolve()


def install_safe_utils(
    snapshot_src: Path,
    *,
    expected_runner_sha256: str = OFFICIAL_RUNNER_SHA256,
    expected_utils_sha256: str = OFFICIAL_UNSAFE_UTILS_SHA256,
) -> tuple[Path, Path]:
    snapshot = snapshot_src.expanduser().absolute()
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise RunnerWrapperError("snapshot source must be a regular directory")
    snapshot = snapshot.resolve()
    runner = _regular_file(snapshot / "result_analysis.py", "official runner")
    unsafe_utils = _regular_file(snapshot / "utils.py", "pinned upstream utils")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_runner_sha256):
        raise RunnerWrapperError("expected runner SHA-256 is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_utils_sha256):
        raise RunnerWrapperError("expected utils SHA-256 is invalid")
    if _sha256(runner) != expected_runner_sha256:
        raise RunnerWrapperError("official runner hash differs from the reviewed snapshot")
    if _sha256(unsafe_utils) != expected_utils_sha256:
        raise RunnerWrapperError("upstream utils hash differs from the reviewed snapshot")
    if "utils" in sys.modules:
        raise RunnerWrapperError("refusing to replace an already-imported utils module")
    safe_utils = types.ModuleType("utils")
    safe_utils.__file__ = str(Path(__file__).resolve()) + "#synthetic-safe-utils"
    safe_utils.ComplexDictParser = SafeComplexDictParser
    safe_utils.parse_complex_string = parse_complex_string
    safe_utils.ULTRA_SAFE_PARSER = True
    sys.modules["utils"] = safe_utils
    return snapshot, runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-src", required=True, type=Path)
    parser.add_argument("--generated-function-path", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-utils-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot, runner = install_safe_utils(
        args.snapshot_src,
        expected_runner_sha256=args.expected_runner_sha256,
        expected_utils_sha256=args.expected_utils_sha256,
    )
    sys.path.insert(0, str(snapshot))
    if args.preflight:
        if args.generated_function_path is not None:
            raise RunnerWrapperError("preflight does not accept generated candidate input")
        namespace = runpy.run_path(str(runner), run_name="ultra_mattools_safe_preflight")
        imported = namespace.get("ComplexDictParser")
        if imported is not SafeComplexDictParser:
            raise RunnerWrapperError("official runner did not bind the synthetic safe parser")
        print(
            json.dumps(
                {
                    "official_runner_sha256": _sha256(runner),
                    "safe_parser_bound": True,
                    "snapshot_utils_imported": False,
                    "task_execution_performed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.generated_function_path is None:
        raise RunnerWrapperError("--generated-function-path is required outside preflight")
    generated = args.generated_function_path.expanduser().absolute()
    if generated.is_symlink() or not generated.is_dir():
        raise RunnerWrapperError("generated-function path must be a regular directory")
    sys.argv = [str(runner), "--generated_function_path", str(generated.resolve())]
    runpy.run_path(str(runner), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
