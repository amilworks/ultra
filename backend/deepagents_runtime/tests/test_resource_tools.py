"""Catalog resource search + staging tool tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.context_tools import (
    _public_stage_result,
    public_selected_resource_descriptor,
    stage_catalog_resources,
    stage_uploaded_files,
)
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
                {
                    "resource_id": "file_norm_ct",
                    "original_name": "norm_ct_1.tiff",
                    "resource_kind": "image",
                }
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
        {
            "resources": [{"resource_id": "file_a", "original_name": "a.tiff"}],
            "missing": ["file_b"],
        },
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
    (uploads / "file_npm1__scan.tiff").write_bytes(b"TIFFDATA")
    context = _context(tmp_path)

    result = stage_catalog_resources(
        context,
        upload_roots=(str(uploads),),
        resources=[
            {
                "resource_id": "file_npm1",
                "original_name": "scan.tiff",
                "content_type": "image/tiff",
                "resource_kind": "image",
                "source_type": "upload",
                "sha256": "d" * 64,
                "size_bytes": 8,
                "metadata": {
                    "source": "upload_store",
                    "caption": "Owner-declared caption",
                    "credentials": {"token": "catalog-secret"},
                    "license_text": "private vendor license prose",
                },
            },
            {"resource_id": "file_gone", "original_name": "gone.tiff", "source_type": "upload"},
        ],
    )
    assert result["ok"] is True
    assert len(result["staged_resources"]) == 1
    staged = result["staged_resources"][0]
    assert staged["resource_id"] == "file_npm1"
    assert staged["binding_schema"] == "ultra.catalog_resource.v1"
    assert staged["binding_authority"] == "control_resource_catalog"
    assert staged["sha256"] == "d" * 64
    assert staged["size_bytes"] == 8
    assert staged["content_type"] == "image/tiff"
    assert staged["sandbox_path"] == "/workspace/staged_resources/file_npm1/scan.tiff"
    # The file is physically copied into the run workspace.
    copied = Path(context.workspace_root) / "staged_resources" / "file_npm1" / "scan.tiff"
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
    assert "catalog-secret" not in payload
    assert "private vendor license prose" not in payload
    assert "f" * 64 not in payload


def test_stage_catalog_resources_copies_ome_zarr_bundle_directory(tmp_path):
    # OME-Zarr (and any folder-format upload) is committed as a DIRECTORY bundle under
    # {root}/bundles/{id}/{name}/, not a single blob. Staging must copytree the whole tree
    # into the sandbox so downstream code (otsu, etc.) can read the zarr. Regression for the
    # run that failed "file_not_in_upload_store" even though the bundle was present on disk.
    uploads = tmp_path / "uploads"
    bundle = uploads / "bundles" / "file_zarr1" / "scan.ome.zarr"
    (bundle / "0").mkdir(parents=True)
    (bundle / ".zgroup").write_text('{"zarr_format": 2}')
    (bundle / ".zattrs").write_text('{"multiscales": []}')
    (bundle / "0" / ".zarray").write_text('{"shape": [4, 4]}')
    (bundle / "0" / "0.0").write_bytes(b"CHUNK")
    context = _context(tmp_path)

    result = stage_catalog_resources(
        context,
        upload_roots=(str(uploads),),
        resources=[
            {
                "resource_id": "file_zarr1",
                "original_name": "scan.ome.zarr",
                "source_type": "upload",
                "sha256": "a" * 64,
                "size_bytes": 123,
                "tree_identity": {
                    "schema": "ultra.resource-tree-identity.v1",
                    "authority": "control_resource_catalog",
                    "canonical_json_schema": "ultra.canonical-json.v1",
                    "tree_manifest_schema": "ultra.tree-manifest.v1",
                    "tree_manifest_path": ".ultra/tree-manifest.json",
                    "tree_manifest_sha256": "a" * 64,
                    "scope": "all_regular_files_except_tree_manifest",
                },
            },
        ],
    )
    assert result["ok"] is True
    staged = result["staged_resources"][0]
    assert staged["kind"] == "directory"
    assert staged["sandbox_path"] == "/workspace/staged_resources/file_zarr1/scan.ome.zarr"
    assert staged["tree_identity"]["tree_manifest_sha256"] == "a" * 64
    # The whole zarr tree is physically copied into the run workspace.
    copied = Path(context.workspace_root) / "staged_resources" / "file_zarr1" / "scan.ome.zarr"
    assert copied.is_dir()
    assert (copied / ".zgroup").read_text() == '{"zarr_format": 2}'
    assert (copied / "0" / "0.0").read_bytes() == b"CHUNK"
    # Host paths stay redacted in the model-visible projection.
    public = _public_stage_result(result)
    assert str(uploads) not in json.dumps(public)
    assert public["staged_resources"][0]["kind"] == "directory"


def test_selected_bundle_tree_identity_survives_delegation_and_staging(tmp_path):
    uploads = tmp_path / "uploads"
    bundle = uploads / "bundles" / "file_zarr2" / "signals.zarr"
    bundle.mkdir(parents=True)
    (bundle / ".zattrs").write_text("{}")
    identity = {
        "schema": "ultra.resource-tree-identity.v1",
        "authority": "control_resource_catalog",
        "canonical_json_schema": "ultra.canonical-json.v1",
        "tree_manifest_schema": "ultra.tree-manifest.v1",
        "tree_manifest_path": ".ultra/tree-manifest.json",
        "tree_manifest_sha256": "b" * 64,
        "scope": "all_regular_files_except_tree_manifest",
    }
    descriptor = {
        "type": "selected_resource",
        "binding_schema": "ultra.selected_resource.v1",
        "authority": "control_resource_catalog",
        "resource_id": "file_zarr2",
        "file_id": "file_zarr2",
        "original_name": "signals.zarr",
        "sha256": "b" * 64,
        "size_bytes": 2,
        "tree_identity": identity,
    }
    delegated = public_selected_resource_descriptor(descriptor)
    assert delegated["tree_identity"] == identity

    context = replace(
        _context(tmp_path),
        selected_file_ids=("file_zarr2",),
        resource_descriptors=(descriptor,),
    )
    result = stage_uploaded_files(context, upload_roots=(uploads,))
    assert result["ok"] is True
    assert result["staged_files"][0]["tree_identity"] == identity

    # A nested digest that is not the catalog's top-level digest is denied.
    forged = {**descriptor, "tree_identity": {**identity, "tree_manifest_sha256": "c" * 64}}
    assert "tree_identity" not in public_selected_resource_descriptor(forged)


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
                    assistant_id="a",
                    org_id="o",
                    user_id=user_id,
                    project_id="p",
                    thread_id="t",
                    run_id="r",
                    goal="analyze my data",
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
