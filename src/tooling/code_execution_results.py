"""Reusable result summarization helpers for code execution outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.tooling.code_execution import _extract_execution_insights


def build_execution_summary(
    *,
    source_dir: Path,
    artifacts: list[dict[str, Any]],
    expected_outputs: list[str],
) -> dict[str, Any]:
    """Build the normalized output summary used by execution callers."""

    return _extract_execution_insights(
        source_dir=source_dir,
        artifacts=artifacts,
        expected_outputs=expected_outputs,
    )


__all__ = ["build_execution_summary"]
