"""Reproducible report generation utilities."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import get_settings


def _markdown_table(rows: list[dict[str, Any]], max_rows: int = 50) -> str:
    if not rows:
        return "_No rows available._"
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(str(key))
    head = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join(["---"] * len(keys)) + " |"
    lines = [head, sep]
    for row in rows[:max_rows]:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    if len(rows) > max_rows:
        lines.append(f"\n_Only first {max_rows} rows shown ({len(rows)} total)._")
    return "\n".join(lines)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_repro_report(
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
    artifact_root: str | None = None,
) -> dict[str, Any]:
    """Generate deterministic markdown/JSON reproducibility report artifacts.

    Parameters
    ----------
    run_id : str or None, default=None
        Optional run identifier linked to this report.
    title : str or None, default=None
        Report title override.
    result_summary : str or None, default=None
        Executive summary text.
    measurements : list[dict[str, Any]] or None, default=None
        Structured measurement rows.
    statistical_analysis : list[dict[str, Any]] or dict[str, Any] or None, default=None
        Statistical reasoning payload(s).
    qc_warnings : list[str] or None, default=None
        Quality-control warnings.
    limitations : list[str] or None, default=None
        Study limitations.
    provenance : dict[str, Any] or None, default=None
        Provenance metadata.
    next_steps : list[dict[str, Any] | str] or None, default=None
        Suggested follow-up actions.
    output_dir : str or None, default=None
        Optional explicit output directory.
    artifact_root : str or None, default=None
        Optional artifact root override used when `output_dir` is omitted.

    Returns
    -------
    dict[str, Any]
        Report generation envelope containing paths, hashes, and UI artifact
        metadata.
    """
    settings = get_settings()
    artifact_base = Path(artifact_root or settings.artifact_root)

    if output_dir:
        report_dir = Path(output_dir)
    elif run_id:
        report_dir = artifact_base / str(run_id) / "reports"
    else:
        report_dir = Path("data") / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    report_stem = f"repro_report_{stamp}"
    md_path = report_dir / f"{report_stem}.md"
    json_path = report_dir / f"{report_stem}.json"

    title_text = title or "Scientific Reproducibility Report"
    summary_text = (result_summary or "").strip() or "No summary provided."
    measurements = measurements or []
    qc_warnings = [str(x) for x in (qc_warnings or []) if str(x).strip()]
    limitations = [str(x) for x in (limitations or []) if str(x).strip()]
    provenance = provenance or {}
    next_steps = next_steps or []

    if isinstance(statistical_analysis, dict):
        stats_block = [statistical_analysis]
    elif isinstance(statistical_analysis, list):
        stats_block = [x for x in statistical_analysis if isinstance(x, dict)]
    else:
        stats_block = []

    if run_id and "run_id" not in provenance:
        provenance["run_id"] = run_id
    provenance.setdefault("generated_at_utc", datetime.utcnow().isoformat() + "Z")

    md_lines: list[str] = [
        f"# {title_text}",
        "",
        f"- Generated: {provenance['generated_at_utc']}",
    ]
    if run_id:
        md_lines.append(f"- Run ID: {run_id}")
    md_lines.extend(
        [
            "",
            "## Executive Summary",
            summary_text,
            "",
            "## Measurements",
            _markdown_table(measurements),
            "",
            "## Statistical Reasoning",
        ]
    )
    if stats_block:
        md_lines.append(_markdown_table(stats_block))
    else:
        md_lines.append("_No statistical analysis provided._")

    md_lines.extend(["", "## QC Warnings"])
    if qc_warnings:
        md_lines.extend([f"- {item}" for item in qc_warnings])
    else:
        md_lines.append("- None")

    md_lines.extend(["", "## Limitations"])
    if limitations:
        md_lines.extend([f"- {item}" for item in limitations])
    else:
        md_lines.append("- None")

    md_lines.extend(["", "## Provenance"])
    for key in sorted(provenance.keys()):
        md_lines.append(f"- {key}: {provenance[key]}")

    md_lines.extend(["", "## Next Steps"])
    if next_steps:
        for step in next_steps:
            if isinstance(step, dict):
                action = step.get("action") or step.get("label") or step
                md_lines.append(f"- {action}")
            else:
                md_lines.append(f"- {step}")
    else:
        md_lines.append("- None")

    md_text = "\n".join(md_lines) + "\n"
    md_path.write_text(md_text, encoding="utf-8")

    payload = {
        "run_id": run_id,
        "title": title_text,
        "result_summary": summary_text,
        "measurements": measurements,
        "statistical_analysis": stats_block,
        "qc_warnings": qc_warnings,
        "limitations": limitations,
        "provenance": provenance,
        "next_steps": next_steps,
        "report_markdown_path": str(md_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return {
        "success": True,
        "run_id": run_id,
        "report_markdown_path": str(md_path),
        "report_json_path": str(json_path),
        "report_sha256": _sha256_file(md_path),
        "report_bundle_sha256": _sha256_file(json_path),
        "ui_artifacts": [
            {
                "type": "summary",
                "title": "Reproducibility report generated",
                "payload": (
                    f"Report saved to {md_path.name}. "
                    "Includes methods, measurements, statistical reasoning, QC, and provenance."
                ),
            },
            {
                "type": "table",
                "title": "Report files",
                "payload": [
                    {"file": md_path.name, "path": str(md_path)},
                    {"file": json_path.name, "path": str(json_path)},
                ],
            },
        ],
    }


__all__ = ["generate_repro_report"]
