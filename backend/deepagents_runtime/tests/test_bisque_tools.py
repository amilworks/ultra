from ultra_deepagents.agent import build_research_agent, build_run_context_brief
from ultra_deepagents.bisque.tools import (
    _bisque_mutation_authorized,
    bisque_dataset_annotation_summary,
    bisque_dataset_member_count,
    bisque_image_annotation_count,
    bisque_mutation_goal_authorized,
    control_post_json,
    create_bisque_dataset,
    download_bisque_resources,
    get_bisque_module_run,
    list_bisque_module_runs,
    search_bisque_resources,
    upload_bisque_outputs,
    upload_bisque_workspace_files,
)
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext


def _mutation_context(goal: str, *intents: str) -> AgentRunContext:
    return AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-bisque-intent",
        goal=goal,
        remote_mutation_intents=tuple(intents),
    )


def test_bisque_mutations_require_exact_server_authored_scope():
    denied_upload_goals = (
        "Compute the XRD pattern and save durable outputs.",
        "Send me the report.",
        "Copy the CIF into /outputs.",
        "Create a local dataset from the results.",
        "Upload the resulting plot for me.",
        "Search BisQue, then send me the report.",
        "Copy the CIF into /outputs, then inspect BisQue.",
        "Do not upload or publish these results to BisQue.",
        "Never copy the CIF to my linked BisQue account.",
        "Analyze it without sending anything to BisQue.",
    )
    for goal in denied_upload_goals:
        assert _bisque_mutation_authorized(_mutation_context(goal), "upload") is False

    allowed_upload_goals = (
        "Upload the resulting plot to BisQue.",
        "Push the outputs to my linked BisQue account.",
        "Send this report to the BisQue server.",
        "Copy the CIF into my BisQue account.",
    )
    for goal in allowed_upload_goals:
        assert (
            _bisque_mutation_authorized(_mutation_context(goal, "bisque.upload"), "upload") is True
        )

    denied_dataset_goals = (
        "Create a local dataset from the results.",
        "Create a local BisQue dataset for offline use.",
        "Create a dataset from these files.",
        "Do not create a BisQue dataset.",
        "Never make a dataset in my linked BisQue account.",
    )
    for goal in denied_dataset_goals:
        assert _bisque_mutation_authorized(_mutation_context(goal), "create_dataset") is False

    allowed_dataset_goals = (
        "Create a BisQue dataset grouping the uploaded results.",
        "Make a dataset in my linked BisQue account.",
    )
    for goal in allowed_dataset_goals:
        assert (
            _bisque_mutation_authorized(
                _mutation_context(goal, "bisque.create_dataset"), "create_dataset"
            )
            is True
        )

    assert (
        _bisque_mutation_authorized(
            _mutation_context("Upload it", "bisque.create_dataset"), "upload"
        )
        is False
    )


def test_goal_heuristic_rejects_explanatory_and_hypothetical_non_consent():
    for goal in (
        "Explain how users upload files to BisQue.",
        "Can Ultra upload files to BisQue?",
        "If I asked you to upload a plot to BisQue, what would happen?",
        "I didn't ask you to upload the plot to BisQue.",
        "Explain how to create a dataset in BisQue.",
    ):
        assert bisque_mutation_goal_authorized(goal, "upload") is False
        assert bisque_mutation_goal_authorized(goal, "create_dataset") is False


def test_research_agent_registers_bisque_tools_for_selected_resource_context(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("ultra_deepagents.agent.create_deep_agent", fake_create_deep_agent)
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-bisque",
        goal="Analyze this BisQue image and save any derived outputs back to BisQue.",
        remote_mutation_intents=("bisque.upload", "bisque.create_dataset"),
        selected_resource_uris=("https://bisque.example.org/data_service/image/abc",),
    )

    build_research_agent(settings, model=object(), backend=object(), context=context)

    tool_names = {getattr(tool, "name", "") for tool in captured["tools"]}
    assert "bisque_search_resources" in tool_names
    assert "bisque_download_resource" in tool_names
    assert "bisque_upload_files" in tool_names
    assert "bisque_create_dataset" in tool_names
    assert "bisque_module_runs" in tool_names
    manifest_tool = next(
        tool
        for tool in captured["tools"]
        if getattr(tool, "name", "") == "tool_capability_manifest"
    )
    manifest = manifest_tool.invoke({})
    assert "bisque_search_resources" in manifest
    assert "bisque_download_resource" in manifest
    assert "bisque_upload_files" in manifest


def test_research_agent_omits_bisque_mutation_tools_without_exact_run_scope(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("ultra_deepagents.agent.create_deep_agent", fake_create_deep_agent)
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-bisque-read-only",
        goal="Inspect my linked BisQue account.",
        run_metadata={"bisque_session_id": "bisque_session_opaque"},
    )

    build_research_agent(settings, model=object(), backend=object(), context=context)

    tool_names = {getattr(tool, "name", "") for tool in captured["tools"]}
    assert "bisque_search_resources" in tool_names
    assert "bisque_download_resource" in tool_names
    assert "bisque_module_runs" in tool_names
    assert "bisque_upload_files" not in tool_names
    assert "bisque_upload_workspace_files" not in tool_names
    assert "bisque_create_dataset" not in tool_names


def test_research_agent_registers_bisque_tools_for_linked_account_followup(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("ultra_deepagents.agent.create_deep_agent", fake_create_deep_agent)
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-linked-followup",
        goal="Upload the updated overlay and report back to the account.",
        remote_mutation_intents=("bisque.upload",),
        run_metadata={"bisque_session_id": "bisque_session_opaque"},
    )

    build_research_agent(settings, model=object(), backend=object(), context=context)

    tool_names = {getattr(tool, "name", "") for tool in captured["tools"]}
    assert "bisque_search_resources" in tool_names
    assert "bisque_download_resource" in tool_names
    assert "bisque_upload_files" in tool_names


def test_run_context_brief_mentions_linked_bisque_account_without_session_id():
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-linked-followup",
        goal="Upload the updated overlay and report back to the account.",
        remote_mutation_intents=("bisque.upload",),
        run_metadata={"bisque_session_id": "bisque_session_secret"},
    )

    brief = build_run_context_brief(context)

    assert "linked BisQue account" in brief
    assert "bisque_search_resources" in brief
    assert "bisque_download_resource" in brief
    assert "bisque_upload_files" in brief
    assert "bisque_session_secret" not in brief
    # scope='owner' is scoped to the user's BisQue resources; "my files" routes to
    # the Ultra catalog, which the brief advertises on its own line.
    assert "the user's own BisQue resources" in brief
    assert "the user's own resources" not in brief
    assert "search_resources; every hit carries a lens_url" in brief


def test_bisque_search_calls_control_plane_without_credentials(monkeypatch):
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        openai_api_key="should-not-leak",
    )
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "count": 1,
                "results": [
                    {
                        "resource_uri": "https://bisque.example.org/data_service/image/abc",
                        "name": "prairie.jpg",
                    }
                ],
            }

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

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

    result = search_bisque_resources(
        settings,
        resource_type="image",
        tag_query="species:prairie_dog",
        limit=5,
    )

    assert captured["url"] == "http://control.test/v2/bisque/search"
    assert captured["json"] == {
        "resource_type": "image",
        "tag_query": "species:prairie_dog",
        "query": "",
        "limit": 5,
    }
    assert "Authorization" not in captured["headers"]
    assert "should-not-leak" not in repr(captured)
    assert result["count"] == 1
    assert "credentials" not in repr(result).lower()


def test_bisque_search_supports_owner_recent_extension_filters(monkeypatch):
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
    )
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "count": 1,
                "results": [
                    {
                        "resource_uri": "https://bisque.example.org/data_service/image/abc",
                        "name": "EnrNE_recent.PNG",
                    }
                ],
            }

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

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

    result = search_bisque_resources(
        settings,
        resource_type="image",
        scope="owner",
        sort="recent",
        name_contains="EnrNE_",
        extensions=["png"],
        limit=10,
        count_all=True,
    )

    assert captured["url"] == "http://control.test/v2/bisque/search"
    assert captured["json"] == {
        "resource_type": "image",
        "tag_query": "",
        "query": "",
        "limit": 10,
        "scope": "owner",
        "sort": "recent",
        "name_contains": "EnrNE_",
        "extensions": ["png"],
        "count_all": True,
    }
    assert "Authorization" not in captured["headers"]
    assert result["results"][0]["name"] == "EnrNE_recent.PNG"


def test_bisque_search_forwards_numeric_metadata_filters(monkeypatch):
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
    )
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"count": 3, "results": []}

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json, headers=None):
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr("httpx.Client", FakeClient)

    # Numeric value passed as an int and op upper-cased — both normalized.
    search_bisque_resources(
        settings,
        resource_type="image",
        tag_query="modality:CT",
        metadata_filters=[{"tag": "age", "op": "GT", "value": 50}],
        scope="owner",
        count_all=True,
    )

    assert captured["json"]["tag_query"] == "modality:CT"
    assert captured["json"]["metadata_filters"] == [{"tag": "age", "op": "gt", "value": "50"}]
    # The relational comparison must NOT be smuggled into tag_query, where BisQue
    # would compare it lexically.
    assert ">" not in captured["json"]["tag_query"]


def test_bisque_metadata_filter_list_parses_json_string_and_single_dict():
    from ultra_deepagents.bisque.tools import _metadata_filter_list

    assert _metadata_filter_list('[{"tag":"age","op":"gte","value":"50"}]') == [
        {"tag": "age", "op": "gte", "value": "50"}
    ]
    assert _metadata_filter_list({"tag": "modality", "value": "CT"}) == [
        {"tag": "modality", "op": "eq", "value": "CT"}
    ]
    assert _metadata_filter_list([{"op": "gt", "value": "1"}]) == []
    assert _metadata_filter_list(None) == []


def test_bisque_control_call_sends_run_scoped_session_and_principal_reference(monkeypatch):
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        openai_api_key="should-not-leak",
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-bisque",
        goal="Use bqapi",
        run_metadata={"bisque_session_id": "bisque_session_opaque"},
        run_lease_worker_id="worker-a",
        run_lease_token="lease-secret",
    )
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

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

    result = control_post_json(
        settings, "/v2/bisque/search", {"resource_type": "image"}, context=context
    )

    assert result == {"ok": True}
    assert captured["headers"] == {
        "X-Ultra-Run-Id": "run-bisque",
        "X-Ultra-Bisque-Session-Id": "bisque_session_opaque",
        "X-Ultra-Worker-Id": "worker-a",
        "X-Ultra-Run-Lease-Token": "lease-secret",
        "X-Ultra-User-Id": "user-1",
        "X-Ultra-Org-Id": "local-org",
    }
    assert "Authorization" not in captured["headers"]
    assert "should-not-leak" not in repr(captured)


def test_bisque_upload_accepts_artifact_ids_without_credentials(monkeypatch):
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        openai_api_key="should-not-leak",
    )
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "count": 1,
                "uploads": [
                    {
                        "artifact_id": "artifact-report",
                        "resource_uri": "https://bisque.example.org/data_service/file/report",
                    }
                ],
            }

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

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

    result = upload_bisque_outputs(
        settings,
        artifact_ids=["artifact-report"],
        file_ids=[],
        context=_mutation_context("Upload the report to BisQue", "bisque.upload"),
    )

    assert captured["url"] == "http://control.test/v2/bisque/upload"
    assert captured["json"] == {"file_ids": [], "artifact_ids": ["artifact-report"]}
    assert "Authorization" not in captured["headers"]
    assert "should-not-leak" not in repr(captured)
    assert result["uploads"][0]["artifact_id"] == "artifact-report"


def test_bisque_control_call_sends_worker_token_when_configured(monkeypatch):
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        control_worker_token="worker-secret",
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-bisque",
        goal="Use bqapi",
        run_metadata={"bisque_session_id": "bisque_session_opaque"},
        run_lease_worker_id="worker-a",
        run_lease_token="lease-secret",
    )
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json, headers=None):
            captured["headers"] = headers or {}
            return FakeResponse()

    monkeypatch.setattr("httpx.Client", FakeClient)

    control_post_json(settings, "/v2/bisque/search", {"resource_type": "image"}, context=context)

    assert captured["headers"]["X-Ultra-Worker-Token"] == "worker-secret"
    assert captured["headers"]["X-Ultra-Run-Id"] == "run-bisque"
    assert captured["headers"]["X-Ultra-Bisque-Session-Id"] == "bisque_session_opaque"
    assert captured["headers"]["X-Ultra-Worker-Id"] == "worker-a"
    assert captured["headers"]["X-Ultra-Run-Lease-Token"] == "lease-secret"


def test_bisque_create_dataset_posts_members_to_control_plane(monkeypatch):
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        openai_api_key="should-not-leak",
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-bisque",
        goal="Group uploads into a dataset",
        remote_mutation_intents=("bisque.create_dataset",),
        run_metadata={"bisque_session_id": "bisque_session_opaque"},
    )
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "name": "analysis outputs",
                "resource_uri": "https://bisque.example.org/data_service/00-DS",
                "resource_uniq": "00-DS",
                "member_count": 2,
                "client_view_url": "https://bisque.example.org/client_service/view?resource=00-DS",
            }

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

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

    result = create_bisque_dataset(
        settings,
        name="analysis outputs",
        resource_uris=[
            "https://bisque.example.org/data_service/00-A",
            "https://bisque.example.org/data_service/00-B",
        ],
        context=context,
    )

    assert captured["url"] == "http://control.test/v2/bisque/datasets"
    assert captured["json"] == {
        "name": "analysis outputs",
        "resource_uris": [
            "https://bisque.example.org/data_service/00-A",
            "https://bisque.example.org/data_service/00-B",
        ],
    }
    assert captured["headers"]["X-Ultra-Bisque-Session-Id"] == "bisque_session_opaque"
    assert "Authorization" not in captured["headers"]
    assert result["ok"] is True
    assert result["resource_uniq"] == "00-DS"
    assert result["member_count"] == 2


def test_bisque_list_module_runs_summarizes_status_and_module(monkeypatch):
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
    )
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "count": 3,
                "results": [
                    {
                        "resource_uri": "https://b/data_service/00-A",
                        "resource_uniq": "00-A",
                        "module_name": "nph_prediction_V2",
                        "status": "FINISHED",
                        "created_at": "2026-06-12T01:07:16",
                        "client_view_url": "https://b/client_service/view?resource=https://b/data_service/00-A",
                    },
                    {
                        "resource_uri": "https://b/data_service/00-B",
                        "resource_uniq": "00-B",
                        "module_name": "CellSegment3DUnet",
                        "status": "FAILED",
                        "created_at": "2023-03-01T03:37:50",
                    },
                ],
            }

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json, headers=None):
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr("httpx.Client", FakeClient)

    result = list_bisque_module_runs(settings, status="FINISHED")
    assert captured["url"] == "http://control.test/v2/bisque/search"
    assert captured["json"]["resource_type"] == "mex"
    # status filter keeps only the FINISHED run
    assert [run["module_name"] for run in result["runs"]] == ["nph_prediction_V2"]
    assert result["runs"][0]["status"] == "FINISHED"
    assert result["runs"][0]["mex_uri"] == "https://b/data_service/00-A"


def test_bisque_get_module_run_returns_outputs_for_download(monkeypatch):
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
    )
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "resource_uri": "https://b/data_service/00-MEX",
                "resource_type": "mex",
                "status": "FINISHED",
                "module_name": "nph_prediction_V2",
                "outputs": [
                    {
                        "name": "Segmentation",
                        "type": "image",
                        "resource_uri": "https://b/data_service/00-SEG",
                        "client_view_url": "https://b/client_service/view?resource=https://b/data_service/00-SEG",
                    }
                ],
            }

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json, headers=None):
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr("httpx.Client", FakeClient)

    # Bare resource_uniq routes to mex_uniq; a full URI routes to mex_uri.
    result = get_bisque_module_run(settings, mex_uri="00-MEX")
    assert captured["url"] == "http://control.test/v2/bisque/module-run"
    assert captured["json"] == {"mex_uniq": "00-MEX"}
    assert result["ok"] is True
    assert result["outputs"][0]["name"] == "Segmentation"
    # The output resource_uri is what gets fed to bisque_download_resource.
    assert result["outputs"][0]["resource_uri"] == "https://b/data_service/00-SEG"


def test_bisque_get_module_run_requires_mex_uri():
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
    )
    assert get_bisque_module_run(settings, mex_uri="  ") == {
        "ok": False,
        "error": "mex_uri_required",
    }


def test_bisque_create_dataset_requires_name_and_members():
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
    )

    missing_name = create_bisque_dataset(settings, name="  ", resource_uris=["https://x"])
    assert missing_name == {"ok": False, "error": "dataset_name_required"}

    missing_members = create_bisque_dataset(settings, name="dataset", resource_uris=[])
    assert missing_members == {"ok": False, "error": "no_member_resource_uris"}


def test_bisque_download_returns_structured_failure_without_tool_exception(monkeypatch):
    import httpx

    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        openai_api_key="should-not-leak",
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-bisque",
        goal="Download a BisQue image",
        run_metadata={"bisque_session_id": "bisque_session_opaque"},
    )
    captured: dict = {}

    class FakeResponse:
        status_code = 502

        def __init__(self):
            self.request = httpx.Request("POST", "http://control.test/v2/bisque/download")

        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "bad gateway",
                request=self.request,
                response=self,
            )

        def json(self):
            return {"error": "BisQue upstream returned 500 Internal Server Error"}

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

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

    result = download_bisque_resources(
        settings,
        resources=["https://bisque.example.org/data_service/00-bad"],
        context=context,
    )

    assert result["ok"] is False
    assert result["file_count"] == 0
    assert result["failed_count"] == 1
    assert result["results"][0] == {
        "resource_uri": "https://bisque.example.org/data_service/00-bad",
        "ok": False,
        "status": "failed",
        "status_code": 502,
        "error": "BisQue upstream returned 500 Internal Server Error",
    }
    assert captured["headers"] == {
        "X-Ultra-Run-Id": "run-bisque",
        "X-Ultra-Bisque-Session-Id": "bisque_session_opaque",
        "X-Ultra-User-Id": "user-1",
        "X-Ultra-Org-Id": "local-org",
    }
    assert "Authorization" not in captured["headers"]
    assert "should-not-leak" not in repr(captured)


def test_bisque_upload_returns_structured_failure_without_tool_exception(monkeypatch):
    # Regression for the OME-Zarr otsu run: bisque_upload_files hit a control-plane error and the
    # bare raise_for_status() propagated, failing the WHOLE run. The upload must degrade into a
    # structured result (like its download sibling) so the model can read the error and route
    # around it — here the new clean 4xx for a directory bundle that can't be uploaded as a file.
    import httpx

    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        openai_api_key="should-not-leak",
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-bisque",
        goal="Upload an OME-Zarr to BisQue",
        remote_mutation_intents=("bisque.upload",),
    )
    captured: dict = {}

    class FakeResponse:
        status_code = 400

        def __init__(self):
            self.request = httpx.Request("POST", "http://control.test/v2/bisque/upload")

        def raise_for_status(self):
            raise httpx.HTTPStatusError("bad request", request=self.request, response=self)

        def json(self):
            return {"error": "resource is a directory bundle, not a single file"}

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json, headers=None):
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr("httpx.Client", FakeClient)

    result = upload_bisque_outputs(settings, file_ids=["file_zarr1"], context=context)

    assert result["ok"] is False
    assert result["status_code"] == 400
    assert "directory bundle" in result["error"]
    assert result["file_ids"] == ["file_zarr1"]
    assert "should-not-leak" not in repr(captured)


def test_bisque_download_continues_after_failed_candidate(monkeypatch):
    import httpx

    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
    )
    calls: list[dict] = []

    class FakeResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self.payload = payload
            self.request = httpx.Request("POST", "http://control.test/v2/bisque/download")

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    "bad gateway",
                    request=self.request,
                    response=self,
                )

        def json(self):
            return self.payload

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json, headers=None):
            calls.append({"url": url, "json": json, "headers": headers or {}})
            resource = json["resources"][0]
            if resource.endswith("bad"):
                return FakeResponse(502, {"error": "BisQue upstream returned 500"})
            return FakeResponse(
                200,
                {
                    "file_count": 1,
                    "uploaded": [{"file_id": "file-good"}],
                    "imports": [
                        {
                            "status": "imported",
                            "resource_uri": resource,
                            "uploaded": {"file_id": "file-good"},
                        }
                    ],
                },
            )

    monkeypatch.setattr("httpx.Client", FakeClient)

    result = download_bisque_resources(
        settings,
        resources=[
            "https://bisque.example.org/data_service/00-bad",
            "https://bisque.example.org/data_service/00-good",
        ],
    )

    assert [call["json"] for call in calls] == [
        {"resources": ["https://bisque.example.org/data_service/00-bad"]},
        {"resources": ["https://bisque.example.org/data_service/00-good"]},
    ]
    assert result["ok"] is True
    assert result["file_count"] == 1
    assert result["failed_count"] == 1
    assert result["uploaded"] == [{"file_id": "file-good"}]
    assert result["results"][0]["ok"] is False
    assert result["results"][1]["ok"] is True


def test_bisque_workspace_upload_stages_local_file_then_uploads_to_bisque(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    output = workspace / "outputs" / "overlay.png"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        openai_api_key="should-not-leak",
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-bisque",
        goal="Upload generated image",
        remote_mutation_intents=("bisque.upload",),
        workspace_root=str(workspace),
        artifact_root=str(tmp_path / "artifacts"),
        run_metadata={"bisque_session_id": "bisque_session_opaque"},
    )
    calls: list[dict] = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None, headers=None, files=None):
            calls.append({"url": url, "json": json, "headers": headers or {}, "files": files})
            if url == "http://control.test/v2/uploads":
                assert json is None
                assert headers == {
                    "X-Ultra-Run-Id": "run-bisque",
                    "X-Ultra-Bisque-Session-Id": "bisque_session_opaque",
                    "X-Ultra-User-Id": "user-1",
                    "X-Ultra-Org-Id": "local-org",
                }
                assert files and files[0][0] == "files"
                _, uploaded_file = files[0]
                assert uploaded_file[0] == "overlay.png"
                assert uploaded_file[1].read() == b"\x89PNG\r\n\x1a\nfake"
                uploaded_file[1].seek(0)
                return FakeResponse(
                    {
                        "file_count": 1,
                        "uploaded": [{"file_id": "file-overlay", "original_name": "overlay.png"}],
                    }
                )
            if url == "http://control.test/v2/bisque/upload":
                assert json == {"file_ids": ["file-overlay"], "artifact_ids": []}
                assert headers == {
                    "X-Ultra-Run-Id": "run-bisque",
                    "X-Ultra-Bisque-Session-Id": "bisque_session_opaque",
                    "X-Ultra-User-Id": "user-1",
                    "X-Ultra-Org-Id": "local-org",
                }
                return FakeResponse(
                    {
                        "count": 1,
                        "uploads": [
                            {
                                "file_id": "file-overlay",
                                "resource_uri": "https://bisque.example.org/data_service/image/uploaded",
                                "client_view_url": "https://bisque.example.org/client_service/view?resource=https://bisque.example.org/data_service/image/uploaded",
                            }
                        ],
                    }
                )
            raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr("httpx.Client", FakeClient)

    result = upload_bisque_workspace_files(
        settings,
        paths=["/workspace/outputs/overlay.png"],
        context=context,
    )

    assert [call["url"] for call in calls] == [
        "http://control.test/v2/uploads",
        "http://control.test/v2/bisque/upload",
    ]
    assert result["ok"] is True
    assert result["uploaded_file_ids"] == ["file-overlay"]
    assert result["bisque_upload_files"]["uploads"][0]["resource_uri"].endswith("/uploaded")
    assert result["pushed"][0]["via"] == "workspace_file"
    assert result["pushed"][0]["resource_uri"].endswith("/uploaded")
    # The viewable BisQue link is surfaced so the model reports it instead of the
    # bare data_service resource_uri.
    assert (
        result["pushed"][0]["client_view_url"]
        == "https://bisque.example.org/client_service/view?resource=https://bisque.example.org/data_service/image/uploaded"
    )
    assert "%3A" not in result["pushed"][0]["client_view_url"]
    assert "should-not-leak" not in repr(calls)


def test_bisque_push_prior_artifact_resolves_across_turns(monkeypatch, tmp_path):
    # Reproduces the reported failure: a figure generated in an EARLIER run is asked
    # to be pushed in a later turn. The later run has a fresh empty workspace; the
    # figure exists only as a prior durable artifact under the artifact store.
    artifact_store = tmp_path / "artifacts"
    prior_run_dir = artifact_store / "run-prior"
    prior_figure = prior_run_dir / "outputs" / "ct_scan_visualization.png"
    prior_figure.parent.mkdir(parents=True)
    prior_figure.write_bytes(b"\x89PNG\r\n\x1a\nct-figure")

    current_workspace = tmp_path / "workspaces" / "run-now"
    current_artifacts = artifact_store / "run-now"
    current_workspace.mkdir(parents=True)
    current_artifacts.mkdir(parents=True)

    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
    )
    artifact_descriptor = {
        "type": "artifact",
        "artifact_id": "artifact_run_prior_abc123",
        "run_id": "run-prior",
        "kind": "figure",
        "path": "outputs/ct_scan_visualization.png",
        "source_path": str(prior_figure.resolve()),
        "storage_uri": f"file://{prior_figure.resolve()}",
    }
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-now",
        goal="Push these resulting images to bisque",
        remote_mutation_intents=("bisque.upload",),
        workspace_root=str(current_workspace),
        artifact_root=str(current_artifacts),
        resource_descriptors=(artifact_descriptor,),
        run_metadata={"bisque_session_id": "bisque_session_opaque"},
    )
    calls: list[dict] = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "count": 1,
                "uploads": [
                    {
                        "artifact_id": "artifact_run_prior_abc123",
                        "resource_uri": "https://bisque.example.org/data_service/00-CT",
                        "name": "ct_scan_visualization.png",
                        "resource_uniq": "00-CT",
                        "client_view_url": "https://bisque.example.org/client_service/view?resource=https://bisque.example.org/data_service/00-CT",
                    }
                ],
            }

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None, headers=None, files=None):
            calls.append({"url": url, "json": json, "files": files})
            return FakeResponse()

    monkeypatch.setattr("httpx.Client", FakeClient)

    # The agent passes the path from its earlier message, exactly as in the bug report.
    result = upload_bisque_workspace_files(
        settings,
        paths=["/outputs/ct_scan_visualization.png"],
        context=context,
    )

    # No multipart re-upload: the durable artifact is pushed straight by artifact_id.
    assert [call["url"] for call in calls] == ["http://control.test/v2/bisque/upload"]
    assert calls[0]["json"] == {"file_ids": [], "artifact_ids": ["artifact_run_prior_abc123"]}
    assert result["ok"] is True
    assert result["artifact_ids"] == ["artifact_run_prior_abc123"]
    assert result["pushed"][0]["via"] == "durable_artifact"
    assert result["pushed"][0]["resource_uri"].endswith("/00-CT")
    assert result["pushed"][0]["client_view_url"].endswith(
        "/client_service/view?resource=https://bisque.example.org/data_service/00-CT"
    )


def test_bisque_push_prior_artifact_resolves_by_basename(monkeypatch, tmp_path):
    artifact_store = tmp_path / "artifacts"
    prior_figure = artifact_store / "run-prior" / "outputs" / "ct_scan_visualization.png"
    prior_figure.parent.mkdir(parents=True)
    prior_figure.write_bytes(b"fig")
    current_artifacts = artifact_store / "run-now"
    current_artifacts.mkdir(parents=True)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-now",
        goal="push it",
        remote_mutation_intents=("bisque.upload",),
        workspace_root=str(workspace),
        artifact_root=str(current_artifacts),
        resource_descriptors=(
            {
                "type": "artifact",
                "artifact_id": "artifact_run_prior_def456",
                "run_id": "run-prior",
                "path": "outputs/ct_scan_visualization.png",
                "source_path": str(prior_figure.resolve()),
            },
        ),
    )
    calls: list[dict] = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "count": 1,
                "uploads": [
                    {"artifact_id": "artifact_run_prior_def456", "resource_uri": "https://b/00-X"}
                ],
            }

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None, headers=None, files=None):
            calls.append({"url": url, "json": json})
            return FakeResponse()

    monkeypatch.setattr("httpx.Client", FakeClient)

    result = upload_bisque_workspace_files(
        settings, paths=["ct_scan_visualization.png"], context=context
    )
    assert calls[0]["json"]["artifact_ids"] == ["artifact_run_prior_def456"]
    assert result["ok"] is True


def test_bisque_push_reports_unresolved_path_without_uploading(monkeypatch, tmp_path):
    workspace = tmp_path / "ws"
    artifacts = tmp_path / "art" / "run-now"
    workspace.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-now",
        goal="push missing",
        remote_mutation_intents=("bisque.upload",),
        workspace_root=str(workspace),
        artifact_root=str(artifacts),
    )

    def fail_client(*args, **kwargs):
        raise AssertionError("must not call control plane for unresolved paths")

    monkeypatch.setattr("httpx.Client", fail_client)

    result = upload_bisque_workspace_files(settings, paths=["/outputs/nope.png"], context=context)
    assert result["ok"] is False
    assert result["error"] == "no_resolvable_paths"
    assert result["missing_paths"] == ["/outputs/nope.png"]


class _CaptureClient:
    """Minimal httpx.Client stand-in that records the request + response payload."""

    _captured: dict = {}
    _response: dict = {}

    def __init__(self, timeout):
        _CaptureClient._captured["timeout"] = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json, headers=None):
        _CaptureClient._captured["url"] = url
        _CaptureClient._captured["json"] = json
        _CaptureClient._captured["headers"] = headers or {}
        payload = _CaptureClient._response

        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return payload

        return _Resp()


def _settings():
    return RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        openai_api_key="should-not-leak",
    )


def test_bisque_dataset_member_count_calls_control_plane(monkeypatch):
    _CaptureClient._captured = {}
    _CaptureClient._response = {"dataset_uniq": "00-ds", "member_count": 2752, "members": []}
    monkeypatch.setattr("httpx.Client", _CaptureClient)

    result = bisque_dataset_member_count(_settings(), dataset_uniq="00-ds", limit=50, offset=100)

    assert _CaptureClient._captured["url"] == "http://control.test/v2/bisque/dataset-members"
    assert _CaptureClient._captured["json"] == {
        "dataset_uniq": "00-ds",
        "limit": 50,
        "offset": 100,
    }
    assert result["ok"] is True
    assert result["member_count"] == 2752


def test_bisque_image_annotation_count_calls_control_plane(monkeypatch):
    _CaptureClient._captured = {}
    _CaptureClient._response = {
        "image_uniq": "00-ax",
        "annotation_count": 3,
        "gobject_types": ["Impala", "Springbok", "For Review"],
    }
    monkeypatch.setattr("httpx.Client", _CaptureClient)

    result = bisque_image_annotation_count(_settings(), image_uniq="00-ax")

    assert _CaptureClient._captured["url"] == "http://control.test/v2/bisque/image-annotations"
    assert _CaptureClient._captured["json"] == {"image_uniq": "00-ax"}
    assert result["ok"] is True
    assert result["annotation_count"] == 3


def test_bisque_dataset_annotation_summary_uses_long_timeout(monkeypatch):
    _CaptureClient._captured = {}
    _CaptureClient._response = {
        "dataset_uniq": "00-ds",
        "member_count": 2752,
        "images_with_annotations": 7,
        "total_annotations": 15,
        "annotated_images": [],
    }
    monkeypatch.setattr("httpx.Client", _CaptureClient)

    result = bisque_dataset_annotation_summary(_settings(), dataset_uniq="00-ds")

    assert _CaptureClient._captured["url"] == "http://control.test/v2/bisque/dataset-annotations"
    assert _CaptureClient._captured["json"] == {"dataset_uniq": "00-ds"}
    # A dataset-wide scan must allow more than the default 60s client timeout.
    assert _CaptureClient._captured["timeout"] >= 300.0
    assert result["ok"] is True
    assert result["images_with_annotations"] == 7


def test_bisque_count_tools_require_a_reference():
    assert bisque_dataset_member_count(_settings())["error"] == "dataset_uniq_required"
    assert bisque_image_annotation_count(_settings())["error"] == "image_uniq_required"
    assert bisque_dataset_annotation_summary(_settings())["error"] == "dataset_uniq_required"


def test_bisque_tool_wrappers_never_shadow_their_implementations():
    """The annotation-summary wrapper shared its raw function's exact name, so
    inside build_bisque_tools the bare name was a closure cell holding the
    DECORATED StructuredTool — the wrapper called the tool object instead of the
    implementation and raised TypeError instantly on every production call
    (12/12 failures across 2 users, 2026-07-22 → 2026-08-18), with agents
    silently degrading to per-image iteration. Any tool whose body closes over
    its own name is the same bug waiting; this guard bans the class."""
    from ultra_deepagents.bisque.tools import build_bisque_tools

    tools = build_bisque_tools(_settings(), context=_mutation_context("audit", "bisque.upload"))
    checked = 0
    for tool_obj in tools:
        func = getattr(tool_obj, "func", None)
        code = getattr(func, "__code__", None)
        if code is None:
            continue
        checked += 1
        assert tool_obj.name not in code.co_freevars, (
            f"{tool_obj.name} closes over its own name — its body would call the "
            "decorated tool object instead of the implementation"
        )
    assert checked >= 5, "expected to audit the registered bisque tool wrappers"


def test_annotation_summary_wrapper_executes_the_implementation(monkeypatch):
    """Invoke the REGISTERED wrapper (not the raw function) the way a run does:
    the request must actually reach the control plane. The raw-function test
    above stayed green for a month while every wrapper call failed."""
    import json as _json
    from types import SimpleNamespace

    from ultra_deepagents.bisque.tools import build_bisque_tools

    _CaptureClient._captured = {}
    _CaptureClient._response = {
        "dataset_uniq": "00-ds",
        "member_count": 106,
        "images_with_annotations": 7,
        "total_annotations": 15,
        "annotated_images": [],
    }
    monkeypatch.setattr("httpx.Client", _CaptureClient)

    context = _mutation_context("summarize annotations", "bisque.upload")
    tools = build_bisque_tools(_settings(), context=context)
    summary = next(t for t in tools if t.name == "bisque_dataset_annotation_summary")

    raw = summary.func(runtime=SimpleNamespace(context=context), dataset_uniq="00-ds")
    payload = _json.loads(raw)

    assert _CaptureClient._captured["url"] == "http://control.test/v2/bisque/dataset-annotations"
    assert _CaptureClient._captured["json"] == {"dataset_uniq": "00-ds"}
    assert payload["ok"] is True
    assert payload["images_with_annotations"] == 7
