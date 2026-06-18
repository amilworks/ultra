"""Catalog resource search + staging tool tests."""

from __future__ import annotations

import json
from pathlib import Path

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.context_tools import _public_stage_result, stage_catalog_resources
from ultra_deepagents.resources.tools import (
    resolve_catalog_resources,
    search_resources_catalog,
)


def _settings() -> RuntimeSettings:
    return RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        control_worker_token="trace-worker-secret",
    )


def _context(tmp: Path, *, user_id: str = "user-1", run_id: str = "run-now") -> AgentRunContext:
    ws = tmp / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    return AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="org-1",
        user_id=user_id,
        project_id="proj",
        thread_id="thread-now",
        run_id=run_id,
        goal="plot the middle slice of the norm CT scans",
        workspace_root=str(ws),
    )


def _fake_httpx(monkeypatch, captured: dict, payload: dict) -> None:
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    class FakeClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json, headers=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers or {}
            return FakeResponse()

    monkeypatch.setattr("httpx.Client", FakeClient)


def test_search_resources_posts_run_anchored_request(monkeypatch, tmp_path):
    captured: dict = {}
    _fake_httpx(
        monkeypatch,
        captured,
        {
            "resources": [
                {"resource_id": "file_norm_ct", "original_name": "norm_ct_1.tiff", "resource_kind": "image"}
            ],
            "total_count": 1,
        },
    )
    result = search_resources_catalog(
        _settings(), context=_context(tmp_path), query="norm", kind="image", limit=20
    )
    assert captured["url"] == "http://control.test/v2/runs/run-now/resource-search"
    assert captured["headers"]["X-Ultra-Worker-Token"] == "trace-worker-secret"
    assert captured["headers"]["X-Ultra-Run-Id"] == "run-now"
    assert captured["headers"]["X-Ultra-User-Id"] == "user-1"
    assert captured["json"] == {"query": "norm", "limit": 20, "kind": "image"}
    assert result["ok"] is True
    assert result["resources"][0]["resource_id"] == "file_norm_ct"
    assert result["total_count"] == 1


def test_search_resources_handles_errors_gracefully(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise RuntimeError("control plane down")

    monkeypatch.setattr("httpx.Client", boom)
    result = search_resources_catalog(_settings(), context=_context(tmp_path), query="x")
    assert result["ok"] is False
    assert result["resources"] == []


def test_resolve_resources_posts_resource_ids(monkeypatch, tmp_path):
    captured: dict = {}
    _fake_httpx(
        monkeypatch,
        captured,
        {"resources": [{"resource_id": "file_a", "original_name": "a.tiff"}], "missing": ["file_b"]},
    )
    result = resolve_catalog_resources(
        _settings(), context=_context(tmp_path), resource_ids=["file_a", "file_b"]
    )
    assert captured["url"] == "http://control.test/v2/runs/run-now/resource-resolve"
    assert captured["json"] == {"resource_ids": ["file_a", "file_b"]}
    assert result["resources"][0]["resource_id"] == "file_a"
    assert result["missing"] == ["file_b"]


def test_stage_catalog_resources_copies_owned_file_into_workspace(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    # Canonical upload storage: {resource_id}__{original_name}
    (uploads / "file_npm1__NPM1_13054_IM.tiff").write_bytes(b"TIFFDATA")
    context = _context(tmp_path)

    result = stage_catalog_resources(
        context,
        upload_roots=(str(uploads),),
        resources=[
            {"resource_id": "file_npm1", "original_name": "NPM1_13054_IM.tiff", "source_type": "upload"},
            {"resource_id": "file_gone", "original_name": "gone.tiff", "source_type": "upload"},
        ],
    )
    assert result["ok"] is True
    assert len(result["staged_resources"]) == 1
    staged = result["staged_resources"][0]
    assert staged["resource_id"] == "file_npm1"
    assert staged["sandbox_path"] == "/workspace/staged_resources/file_npm1/NPM1_13054_IM.tiff"
    # The file is physically copied into the run workspace.
    copied = Path(context.workspace_root) / "staged_resources" / "file_npm1" / "NPM1_13054_IM.tiff"
    assert copied.read_bytes() == b"TIFFDATA"
    # A resource with no local file is reported, not silently dropped.
    assert result["unavailable"][0]["resource_id"] == "file_gone"

    # Model-visible projection redacts host paths to sandbox paths only.
    public = _public_stage_result(result)
    payload = json.dumps(public)
    assert str(uploads) not in payload
    assert str(tmp_path) not in payload
    assert public["staged_resources"][0]["staged_path"] == staged["sandbox_path"]
    assert "source_path" not in public["staged_resources"][0]


def test_stage_catalog_resources_rejects_unsafe_resource_id_before_glob(tmp_path):
    # Defense-in-depth: the worker enforces the safe-id charset itself, so a
    # malformed/traversal id never reaches the filesystem glob even if the control
    # plane were ever tricked into returning one.
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "secret__passwd").write_bytes(b"SECRET")
    context = _context(tmp_path)
    result = stage_catalog_resources(
        context,
        upload_roots=(str(uploads),),
        resources=[
            {"resource_id": "../../etc/secret", "original_name": "x", "source_type": "upload"},
            {"resource_id": "a/b", "original_name": "x", "source_type": "upload"},
            {"resource_id": "*", "original_name": "x", "source_type": "upload"},
        ],
    )
    # Every unsafe id is dropped; nothing staged, nothing globbed.
    assert result["staged_resources"] == []
    assert not (Path(context.workspace_root) / "staged_resources").exists() or not any(
        (Path(context.workspace_root) / "staged_resources").iterdir()
    )


def test_resource_tools_registered_only_for_authenticated_runs(tmp_path):
    import ultra_deepagents.agent as agent_module

    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:9/v1",
        openai_model="deepseek_v4",
        workspace_root=str(tmp_path / "ws"),
        memory_root=str(tmp_path / "mem"),
        artifact_root=str(tmp_path / "art"),
    )

    def names_for(user_id: str) -> set[str]:
        captured: dict = {}
        orig = agent_module.create_deep_agent
        agent_module.create_deep_agent = lambda **k: captured.update(k) or "x"
        try:
            agent_module.build_research_agent(
                settings,
                model=object(),
                workspace_dir=tmp_path / "ws" / "r",
                context=AgentRunContext(
                    assistant_id="a", org_id="o", user_id=user_id, project_id="p",
                    thread_id="t", run_id="r", goal="analyze my data",
                ),
            )
        finally:
            agent_module.create_deep_agent = orig
        return {getattr(t, "name", "") for t in captured["tools"]}

    authed = names_for("researcher-1")
    assert "search_resources" in authed
    assert "stage_resource_for_analysis" in authed

    anon = names_for("")
    assert "search_resources" not in anon
    assert "stage_resource_for_analysis" not in anon
