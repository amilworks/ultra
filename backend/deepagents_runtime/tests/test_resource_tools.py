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
    CATALOG_REFUSED,
    CATALOG_UNAVAILABLE,
    build_resource_tools,
    lens_deep_link,
    normalize_resource_hits,
    resolve_catalog_resources,
    search_resources_catalog,
    with_lens_url,
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


def _fake_httpx_status_error(monkeypatch, status: int) -> None:
    """Every POST answers `status` and raise_for_status raises the real
    httpx.HTTPStatusError, exactly as httpx would — so _catalog_failure's
    isinstance classification is exercised, not a lookalike."""
    import httpx

    request = httpx.Request("POST", "http://control.test/v2/runs/run-now/resource-search")

    class FakeResponse:
        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                f"server returned {status}",
                request=request,
                response=httpx.Response(status, request=request),
            )

    class FakeClient:
        def __init__(self, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json, headers=None):
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


# ---------------------------------------------------------------------------
# Lens deep links: the model cites the control plane's lens_url verbatim; when an
# older control plane omits it the worker synthesizes the relative form, but only
# for ids that pass the same safe-id charset the staging path enforces.
# ---------------------------------------------------------------------------


def test_with_lens_url_keeps_control_plane_link_verbatim():
    hit = {
        "resource_id": "file_norm_ct",
        "original_name": "norm_ct_1.tiff",
        "lens_url": "https://ultra.example.edu/?view=lens&resource=file_norm_ct",
    }
    assert with_lens_url(hit) is hit
    # A relative control-plane link is equally authoritative — no rewriting.
    relative = {"resource_id": "file_a", "lens_url": "/?view=lens&resource=file_a"}
    assert with_lens_url(relative)["lens_url"] == "/?view=lens&resource=file_a"


def test_with_lens_url_synthesizes_relative_link_for_safe_id():
    assert with_lens_url({"resource_id": "file_norm.ct:1-2_x"}) == {
        "resource_id": "file_norm.ct:1-2_x",
        "lens_url": "/?view=lens&resource=file_norm.ct%3A1-2_x",
    }
    # Surrounding whitespace is not part of the id; the record itself is untouched.
    padded = with_lens_url({"resource_id": "  file_b  ", "original_name": "b.tiff"})
    assert padded["lens_url"] == "/?view=lens&resource=file_b"
    assert padded["resource_id"] == "  file_b  "
    assert padded["original_name"] == "b.tiff"


def test_with_lens_url_never_synthesizes_for_unsafe_or_absent_ids():
    for bad in ("../../etc/passwd", "a/b", "*", "id with space", "x&y=1", "id#frag", "", "   "):
        hit = with_lens_url({"resource_id": bad, "original_name": "x"})
        assert "lens_url" not in hit, bad
    assert "lens_url" not in with_lens_url({"original_name": "no-id.tiff"})
    assert "lens_url" not in with_lens_url({"resource_id": 42})
    assert "lens_url" not in with_lens_url({"resource_id": ["file_a"]})


def test_with_lens_url_replaces_blank_or_non_string_lens_url():
    # A non-string lens_url cannot be emitted to the model as a link; it is dropped
    # and re-synthesized for a safe id, or dropped outright for an unsafe one.
    assert with_lens_url({"resource_id": "file_a", "lens_url": ""})["lens_url"] == (
        "/?view=lens&resource=file_a"
    )
    assert with_lens_url({"resource_id": "file_a", "lens_url": None})["lens_url"] == (
        "/?view=lens&resource=file_a"
    )
    assert with_lens_url({"resource_id": "file_a", "lens_url": {"href": "x"}})["lens_url"] == (
        "/?view=lens&resource=file_a"
    )
    assert "lens_url" not in with_lens_url({"resource_id": "a/b", "lens_url": 7})


def test_lens_deep_link_percent_encodes_every_reserved_character():
    assert lens_deep_link("file_a") == "/?view=lens&resource=file_a"
    assert lens_deep_link("a&b=c#d/e f") == "/?view=lens&resource=a%26b%3Dc%23d%2Fe%20f"


def test_normalize_resource_hits_drops_non_dict_entries():
    assert normalize_resource_hits(None) == []
    assert normalize_resource_hits("file_a") == []
    assert normalize_resource_hits([None, "x", 3, {"resource_id": "file_a"}]) == [
        {"resource_id": "file_a", "lens_url": "/?view=lens&resource=file_a"}
    ]


def test_search_resources_passes_lens_url_through_and_synthesizes_missing(monkeypatch, tmp_path):
    captured: dict = {}
    _fake_httpx(
        monkeypatch,
        captured,
        {
            "resources": [
                {
                    "resource_id": "file_abs",
                    "original_name": "abs.tiff",
                    "lens_url": "https://ultra.example.edu/?view=lens&resource=file_abs",
                },
                {"resource_id": "file_rel", "original_name": "rel.tiff"},
                {"resource_id": "../evil", "original_name": "evil"},
                "not-a-hit",
            ],
            "total_count": 3,
        },
    )
    result = search_resources_catalog(_settings(), context=_context(tmp_path), query="x")
    assert result["ok"] is True
    by_id = {hit["resource_id"]: hit for hit in result["resources"]}
    assert by_id["file_abs"]["lens_url"] == (
        "https://ultra.example.edu/?view=lens&resource=file_abs"
    )
    assert by_id["file_rel"]["lens_url"] == "/?view=lens&resource=file_rel"
    assert "lens_url" not in by_id["../evil"]
    assert len(result["resources"]) == 3
    assert result["total_count"] == 3


def test_search_resources_total_count_ignores_bool_and_non_int(monkeypatch, tmp_path):
    for total in (True, "7", 3.5, None):
        _fake_httpx(monkeypatch, {}, {"resources": [], "total_count": total})
        result = search_resources_catalog(_settings(), context=_context(tmp_path), query="x")
        assert result["total_count"] == 0, total


def test_search_resources_failure_carries_machine_readable_reason_without_host_details(
    monkeypatch, tmp_path
):
    class CatalogDownError(RuntimeError):
        pass

    def boom(*a, **k):
        raise CatalogDownError("POST http://control.test/v2/runs/run-now/resource-search refused")

    monkeypatch.setattr("httpx.Client", boom)
    result = search_resources_catalog(_settings(), context=_context(tmp_path), query="x")
    assert result == {
        "ok": False,
        "error": CATALOG_UNAVAILABLE,
        "reason": "CatalogDownError",
        "resources": [],
    }
    # The control-plane URL is an operator fact and never reaches the model.
    assert "control.test" not in json.dumps(result)


def test_resolve_resources_failure_carries_machine_readable_reason(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise TimeoutError("http://control.test timed out")

    monkeypatch.setattr("httpx.Client", boom)
    result = resolve_catalog_resources(
        _settings(), context=_context(tmp_path), resource_ids=["file_a", "file_b"]
    )
    assert result["ok"] is False
    assert result["error"] == CATALOG_UNAVAILABLE
    assert result["reason"] == "TimeoutError"
    assert result["missing"] == ["file_a", "file_b"]
    assert "control.test" not in json.dumps(result)


def test_catalog_answer_statuses_split_refused_from_unavailable(monkeypatch, tmp_path):
    # A status the catalog ANSWERED with is a refusal when it cannot change on
    # retry (4xx auth/permissions/config, and 501 for an unserved endpoint) and
    # an outage when it can (other 5xx). The status reaches the model as the
    # reason; the host never does.
    cases = [
        (401, CATALOG_REFUSED),
        (404, CATALOG_REFUSED),
        (501, CATALOG_REFUSED),
        (500, CATALOG_UNAVAILABLE),
        (503, CATALOG_UNAVAILABLE),
    ]
    for status, expected_error in cases:
        _fake_httpx_status_error(monkeypatch, status)
        result = search_resources_catalog(_settings(), context=_context(tmp_path), query="x")
        assert result == {
            "ok": False,
            "error": expected_error,
            "reason": f"http_{status}",
            "resources": [],
        }, status
        assert "control.test" not in json.dumps(result)


def test_resolve_resources_refused_status_carries_http_reason(monkeypatch, tmp_path):
    _fake_httpx_status_error(monkeypatch, 401)
    result = resolve_catalog_resources(
        _settings(), context=_context(tmp_path), resource_ids=["file_a"]
    )
    assert result["ok"] is False
    assert result["error"] == CATALOG_REFUSED
    assert result["reason"] == "http_401"
    assert result["missing"] == ["file_a"]
    assert "control.test" not in json.dumps(result)


def test_resolve_resources_normalizes_lens_url_on_hits(monkeypatch, tmp_path):
    _fake_httpx(
        monkeypatch,
        {},
        {
            "resources": [
                {"resource_id": "file_a", "original_name": "a.tiff"},
                {
                    "resource_id": "file_b",
                    "original_name": "b.tiff",
                    "lens_url": "https://ultra.example.edu/?view=lens&resource=file_b",
                },
            ],
            "missing": [],
        },
    )
    result = resolve_catalog_resources(
        _settings(), context=_context(tmp_path), resource_ids=["file_a", "file_b"]
    )
    links = {hit["resource_id"]: hit["lens_url"] for hit in result["resources"]}
    assert links == {
        "file_a": "/?view=lens&resource=file_a",
        "file_b": "https://ultra.example.edu/?view=lens&resource=file_b",
    }


def _tool_runtime(context: AgentRunContext):
    from langchain.tools import ToolRuntime

    return ToolRuntime(
        state={},
        context=context,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="call-1",
        store=None,
        tools=[],
    )


def test_stage_tool_stamps_lens_url_on_staged_records_without_using_it_for_io(
    monkeypatch, tmp_path
):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "file_a__a.tiff").write_bytes(b"TIFF")
    # A hostile lens_url must never influence where bytes come from: the staged
    # record still resolves through the upload store by id and keeps the link only
    # as presentation metadata.
    captured: dict = {}
    _fake_httpx(
        monkeypatch,
        captured,
        {
            "resources": [
                {
                    "resource_id": "file_a",
                    "original_name": "a.tiff",
                    "source_type": "upload",
                    "lens_url": "https://ultra.example.edu/?view=lens&resource=file_a",
                },
                {"resource_id": "file_gone", "original_name": "gone.tiff", "source_type": "upload"},
            ],
            "missing": ["file_other"],
        },
    )
    tools = {t.name: t for t in build_resource_tools(_settings(), upload_roots=(str(uploads),))}
    context = _context(tmp_path)
    raw = tools["stage_resource_for_analysis"].invoke(
        {"resource_ids": "file_a, file_gone", "runtime": _tool_runtime(context)}
    )
    public = json.loads(raw)
    assert captured["json"] == {"resource_ids": ["file_a", "file_gone"]}
    assert public["ok"] is True
    assert public["missing"] == ["file_other"]
    (staged,) = public["staged_resources"]
    assert staged["resource_id"] == "file_a"
    assert staged["sandbox_path"] == "/workspace/staged_resources/file_a/a.tiff"
    assert staged["lens_url"] == "https://ultra.example.edu/?view=lens&resource=file_a"
    assert (
        Path(context.workspace_root) / "staged_resources" / "file_a" / "a.tiff"
    ).read_bytes() == (b"TIFF")
    # Unavailable records carry no link: there is nothing to open.
    (unavailable,) = public["unavailable"]
    assert unavailable["resource_id"] == "file_gone"
    assert "lens_url" not in unavailable
    assert str(uploads) not in raw


def test_stage_tool_forwards_catalog_unavailable_error_and_reason(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise ConnectionError("http://control.test refused")

    monkeypatch.setattr("httpx.Client", boom)
    tools = {t.name: t for t in build_resource_tools(_settings())}
    raw = tools["stage_resource_for_analysis"].invoke(
        {"resource_ids": ["file_a"], "runtime": _tool_runtime(_context(tmp_path))}
    )
    assert json.loads(raw) == {
        "ok": False,
        "error": CATALOG_UNAVAILABLE,
        "reason": "ConnectionError",
    }
    assert "control.test" not in raw


def test_stage_tool_forwards_catalog_refused_error_and_reason(monkeypatch, tmp_path):
    _fake_httpx_status_error(monkeypatch, 401)
    tools = {t.name: t for t in build_resource_tools(_settings())}
    raw = tools["stage_resource_for_analysis"].invoke(
        {"resource_ids": ["file_a"], "runtime": _tool_runtime(_context(tmp_path))}
    )
    assert json.loads(raw) == {"ok": False, "error": CATALOG_REFUSED, "reason": "http_401"}
    assert "control.test" not in raw


def test_search_tool_output_carries_lens_url_and_describes_it(monkeypatch, tmp_path):
    _fake_httpx(
        monkeypatch,
        {},
        {"resources": [{"resource_id": "file_a", "original_name": "a.tiff"}], "total_count": 1},
    )
    tools = {t.name: t for t in build_resource_tools(_settings())}
    search = tools["search_resources"]
    description = str(search.description).lower()
    assert "lens_url" in description
    assert "the resources tab" in description
    assert "catalog_unavailable" in description
    raw = search.invoke({"query": "a", "runtime": _tool_runtime(_context(tmp_path))})
    assert json.loads(raw)["resources"][0]["lens_url"] == "/?view=lens&resource=file_a"
