"""Render-proof completion gate: an .html deliverable without passing headless
render evidence is an incomplete deliverable (two live incidents shipped
"verified" pages that threw on load / required internet; static checks cannot
see either, one render catches both)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.runner import _missing_render_proof_kinds

_BASE_SETTINGS = dict(
    openai_base_url="http://example.test/v1",
    openai_model="test-model",
)


def _settings(**overrides) -> RuntimeSettings:
    return RuntimeSettings(**_BASE_SETTINGS, **overrides)


def _html_event(path: str) -> dict:
    return {"payload": {"path": path, "kind": "report"}}


def _write_page(workspace: Path, name: str = "page.html") -> Path:
    page = workspace / name
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text("<html><body>hi</body></html>")
    return page


def _write_proof(
    workspace: Path,
    *,
    console_errors: list | None = None,
    page_errors: list | None = None,
    name: str = "page.console.json",
    subdir: str = "diagnostics/report_preview",
) -> Path:
    proof = workspace / subdir / name
    proof.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if console_errors is not None:
        payload["console_errors"] = console_errors
    if page_errors is not None:
        payload["page_errors"] = page_errors
    proof.write_text(json.dumps(payload))
    return proof


def test_no_html_artifacts_demands_nothing(tmp_path: Path):
    missing = _missing_render_proof_kinds(
        _settings(),
        [{"payload": {"path": "figure.png", "kind": "figure"}}],
        workspace_dir=tmp_path / "ws",
        artifact_dir=tmp_path / "art",
    )
    assert missing == []


def test_html_without_proof_is_missing_render_proof(tmp_path: Path):
    workspace = tmp_path / "ws"
    _write_page(workspace)
    missing = _missing_render_proof_kinds(
        _settings(),
        [_html_event("page.html")],
        workspace_dir=workspace,
        artifact_dir=tmp_path / "art",
    )
    assert missing == ["render_proof"]


def test_fresh_passing_proof_satisfies_the_gate(tmp_path: Path):
    workspace = tmp_path / "ws"
    _write_page(workspace)
    _write_proof(workspace, console_errors=[], page_errors=[])
    missing = _missing_render_proof_kinds(
        _settings(),
        [_html_event("page.html")],
        workspace_dir=workspace,
        artifact_dir=tmp_path / "art",
    )
    assert missing == []


def test_proof_with_errors_does_not_satisfy(tmp_path: Path):
    workspace = tmp_path / "ws"
    _write_page(workspace)
    _write_proof(workspace, console_errors=["TypeError: boom"], page_errors=[])
    missing = _missing_render_proof_kinds(
        _settings(),
        [_html_event("page.html")],
        workspace_dir=workspace,
        artifact_dir=tmp_path / "art",
    )
    assert missing == ["render_proof"]


def test_proof_missing_error_keys_does_not_satisfy(tmp_path: Path):
    workspace = tmp_path / "ws"
    _write_page(workspace)
    _write_proof(workspace)  # empty JSON object: keys absent, contract unmet
    missing = _missing_render_proof_kinds(
        _settings(),
        [_html_event("page.html")],
        workspace_dir=workspace,
        artifact_dir=tmp_path / "art",
    )
    assert missing == ["render_proof"]


def test_stale_proof_predating_the_page_does_not_satisfy(tmp_path: Path):
    """A proof from a previous build must not certify the current page —
    otherwise one passing check whitelists every later (possibly broken) edit."""
    workspace = tmp_path / "ws"
    proof = _write_proof(workspace, console_errors=[], page_errors=[])
    old = time.time() - 600
    os.utime(proof, (old, old))
    _write_page(workspace)  # page written AFTER the proof
    missing = _missing_render_proof_kinds(
        _settings(),
        [_html_event("page.html")],
        workspace_dir=workspace,
        artifact_dir=tmp_path / "art",
    )
    assert missing == ["render_proof"]


def test_browser_verify_variant_in_outputs_is_accepted(tmp_path: Path):
    workspace = tmp_path / "ws"
    artifact_dir = tmp_path / "art"
    _write_page(workspace)
    proof = artifact_dir / "torus_browser_verify.json"
    proof.parent.mkdir(parents=True, exist_ok=True)
    proof.write_text(json.dumps({"console_errors": [], "page_errors": []}))
    missing = _missing_render_proof_kinds(
        _settings(),
        [_html_event("page.html")],
        workspace_dir=workspace,
        artifact_dir=artifact_dir,
    )
    assert missing == []


def test_gate_disabled_by_settings_demands_nothing(tmp_path: Path):
    workspace = tmp_path / "ws"
    _write_page(workspace)
    missing = _missing_render_proof_kinds(
        _settings(render_proof_required=False),
        [_html_event("page.html")],
        workspace_dir=workspace,
        artifact_dir=tmp_path / "art",
    )
    assert missing == []
