import json

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.rarespot import tools as rarespot_tools
from ultra_deepagents.rarespot.inference import summarize_detection_confidences
from ultra_deepagents.rarespot.tools import build_rarespot_tools, wait_for_rarespot_run


def test_rarespot_tool_submits_selected_upload_file_ids_from_context(monkeypatch):
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
    )
    context = AgentRunContext(
        assistant_id="ultra-research",
        org_id="org-1",
        user_id="user-1",
        project_id="project-1",
        thread_id="thread-1",
        run_id="agent-run-1",
        selected_file_ids=("file-1", "file-2"),
        selected_resource_uris=("bisque://resource/1",),
        selected_dataset_uris=("bisque://dataset/2",),
    )
    captured: dict = {}

    def fake_create_run(settings, *, thread_id, body):
        captured["thread_id"] = thread_id
        captured["body"] = body
        return {"run_id": "run-1", "status": "queued"}

    monkeypatch.setattr(rarespot_tools, "active_context", lambda: context)
    monkeypatch.setattr(rarespot_tools, "create_rarespot_run", fake_create_run)

    [tool] = build_rarespot_tools(settings)
    result = json.loads(tool.invoke({"wait_for_completion": False}))

    assert result == {"run_id": "run-1", "status": "queued", "artifact_ids": []}
    assert captured["thread_id"] == "thread-1"
    assert captured["body"]["file_ids"] == ["file-1", "file-2"]
    assert captured["body"]["resource_uris"] == ["bisque://resource/1"]
    assert captured["body"]["dataset_uris"] == ["bisque://dataset/2"]
    assert captured["body"]["selected_tool_names"] == ["rarespot_ecology_inference"]
    assert captured["body"]["workflow_hint"] == {"id": "rarespot_ecology"}
    assert captured["body"]["metadata"]["file_ids"] == ["file-1", "file-2"]


def test_rarespot_tool_does_not_forward_sandbox_staged_upload_paths(monkeypatch):
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
    )
    context = AgentRunContext(
        assistant_id="ultra-research",
        org_id="org-1",
        user_id="user-1",
        project_id="project-1",
        thread_id="thread-1",
        run_id="agent-run-1",
        selected_file_ids=("file-1",),
    )
    captured: dict = {}

    def fake_create_run(settings, *, thread_id, body):
        captured["thread_id"] = thread_id
        captured["body"] = body
        return {"run_id": "run-1", "status": "queued"}

    monkeypatch.setattr(rarespot_tools, "active_context", lambda: context)
    monkeypatch.setattr(rarespot_tools, "create_rarespot_run", fake_create_run)

    [tool] = build_rarespot_tools(settings)
    result = json.loads(
        tool.invoke(
            {
                "local_paths": ["/workspace/staged_uploads/file-1/prairie.jpg"],
                "wait_for_completion": False,
            }
        )
    )

    assert result["run_id"] == "run-1"
    assert captured["body"]["file_ids"] == ["file-1"]
    assert captured["body"]["metadata"]["local_paths"] == []


def test_rarespot_tool_forwards_optional_inference_settings(monkeypatch):
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
    )
    context = AgentRunContext(
        assistant_id="ultra-research",
        org_id="org-1",
        user_id="user-1",
        project_id="project-1",
        thread_id="thread-1",
        run_id="agent-run-1",
        selected_file_ids=("file-1",),
    )
    captured: dict = {}

    def fake_create_run(settings, *, thread_id, body):
        captured["thread_id"] = thread_id
        captured["body"] = body
        return {"run_id": "run-1", "status": "queued"}

    monkeypatch.setattr(rarespot_tools, "active_context", lambda: context)
    monkeypatch.setattr(rarespot_tools, "create_rarespot_run", fake_create_run)

    [tool] = build_rarespot_tools(settings)
    result = json.loads(
        tool.invoke(
            {
                "confidence_threshold": 0.55,
                "iou_threshold": 0.4,
                "tile_overlap": 0.1,
                "image_size": 960,
                "spectral": False,
                "wait_for_completion": False,
            }
        )
    )

    assert result["run_id"] == "run-1"
    assert captured["body"]["metadata"]["rarespot_config"] == {
        "confidence_threshold": 0.55,
        "iou_threshold": 0.4,
        "tile_overlap": 0.1,
        "image_size": 960,
        "spectral": False,
    }
    assert captured["body"]["idempotency_key"]
    assert captured["body"]["metadata"]["idempotency_key"] == captured["body"]["idempotency_key"]


def test_rarespot_tool_filters_host_workspace_paths_and_reuses_selected_file_ids(monkeypatch, tmp_path):
    workspace_root = tmp_path / "workspaces"
    staged_path = workspace_root / "parent-run" / "staged_uploads" / "file-1" / "prairie.jpg"
    staged_path.parent.mkdir(parents=True)
    staged_path.write_bytes(b"fake")
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        workspace_root=str(workspace_root),
    )
    context = AgentRunContext(
        assistant_id="ultra-research",
        org_id="org-1",
        user_id="user-1",
        project_id="project-1",
        thread_id="thread-1",
        run_id="parent-run",
        selected_file_ids=("file-1",),
    )
    captured: dict = {}

    def fake_create_run(settings, *, thread_id, body):
        captured["thread_id"] = thread_id
        captured["body"] = body
        return {"run_id": "run-1", "status": "queued"}

    monkeypatch.setattr(rarespot_tools, "active_context", lambda: context)
    monkeypatch.setattr(rarespot_tools, "create_rarespot_run", fake_create_run)

    [tool] = build_rarespot_tools(settings)
    result = json.loads(
        tool.invoke(
            {
                "local_paths": [str(staged_path)],
                "wait_for_completion": False,
            }
        )
    )

    assert result["run_id"] == "run-1"
    assert captured["body"]["file_ids"] == ["file-1"]
    assert captured["body"]["metadata"]["local_paths"] == []


def test_rarespot_tool_idempotency_key_is_stable_for_same_parent_run_and_config(monkeypatch):
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
    )
    context = AgentRunContext(
        assistant_id="ultra-research",
        org_id="org-1",
        user_id="user-1",
        project_id="project-1",
        thread_id="thread-1",
        run_id="parent-run",
        selected_file_ids=("file-1",),
    )
    keys: list[str] = []

    def fake_create_run(settings, *, thread_id, body):
        keys.append(body["idempotency_key"])
        return {"run_id": f"run-{len(keys)}", "status": "queued"}

    monkeypatch.setattr(rarespot_tools, "active_context", lambda: context)
    monkeypatch.setattr(rarespot_tools, "create_rarespot_run", fake_create_run)

    [tool] = build_rarespot_tools(settings)
    tool.invoke({"confidence_threshold": 0.55, "wait_for_completion": False})
    tool.invoke({"confidence_threshold": 0.55, "wait_for_completion": False})
    tool.invoke({"confidence_threshold": 0.7, "wait_for_completion": False})

    assert keys[0] == keys[1]
    assert keys[2] != keys[0]


def test_rarespot_tool_skips_report_only_context_without_new_config(monkeypatch):
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
    )
    context = AgentRunContext(
        assistant_id="ultra-research",
        org_id="org-1",
        user_id="user-1",
        project_id="project-1",
        thread_id="thread-1",
        run_id="parent-run",
        goal=(
            "Write a combined quantitative report across all RareSpot runs in this chat. "
            "Include thresholds, counts by class, artifact links, and limitations."
        ),
        selected_file_ids=("file-1",),
    )

    def fail_create_run(settings, *, thread_id, body):
        raise AssertionError("report-only RareSpot synthesis should not enqueue inference")

    monkeypatch.setattr(rarespot_tools, "active_context", lambda: context)
    monkeypatch.setattr(rarespot_tools, "create_rarespot_run", fail_create_run)

    [tool] = build_rarespot_tools(settings)
    result = json.loads(tool.invoke({"wait_for_completion": False}))

    assert result["status"] == "skipped"
    assert result["reason"] == "report_only_context"
    assert "artifact_manifest" in result["final_answer_hint"]


def test_rarespot_tool_skips_report_only_context_with_negated_rerun(monkeypatch):
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
    )
    context = AgentRunContext(
        assistant_id="ultra-research",
        org_id="org-1",
        user_id="user-1",
        project_id="project-1",
        thread_id="thread-1",
        run_id="parent-run",
        goal=(
            "Write a combined quantitative report across the RareSpot runs in this chat. "
            "Include thresholds, counts by class, confidence summaries, artifact links, "
            "limitations, and recommended next analyses. Do not rerun inference; use "
            "the prior RareSpot outputs."
        ),
        selected_file_ids=("file-1",),
    )

    def fail_create_run(settings, *, thread_id, body):
        raise AssertionError("negated rerun phrasing should not enqueue inference")

    monkeypatch.setattr(rarespot_tools, "active_context", lambda: context)
    monkeypatch.setattr(rarespot_tools, "create_rarespot_run", fail_create_run)

    [tool] = build_rarespot_tools(settings)
    result = json.loads(tool.invoke({"wait_for_completion": False}))

    assert result["status"] == "skipped"
    assert result["reason"] == "report_only_context"


def test_rarespot_tool_allows_report_context_with_explicit_new_threshold(monkeypatch):
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
    )
    context = AgentRunContext(
        assistant_id="ultra-research",
        org_id="org-1",
        user_id="user-1",
        project_id="project-1",
        thread_id="thread-1",
        run_id="parent-run",
        goal="Write a combined report, then run another RareSpot pass at confidence threshold 0.15.",
        selected_file_ids=("file-1",),
    )
    captured: dict = {}

    def fake_create_run(settings, *, thread_id, body):
        captured["body"] = body
        return {"run_id": "run-1", "status": "queued"}

    monkeypatch.setattr(rarespot_tools, "active_context", lambda: context)
    monkeypatch.setattr(rarespot_tools, "create_rarespot_run", fake_create_run)

    [tool] = build_rarespot_tools(settings)
    result = json.loads(
        tool.invoke({"confidence_threshold": 0.15, "wait_for_completion": False})
    )

    assert result["run_id"] == "run-1"
    assert captured["body"]["metadata"]["rarespot_config"] == {"confidence_threshold": 0.15}


def test_summarize_detection_confidences_reports_classwise_stats():
    summary = summarize_detection_confidences(
        [
            {
                "boxes": [
                    {"class_name": "prairie_dog", "confidence": 0.5},
                    {"class_name": "prairie_dog", "confidence": 0.7},
                    {"class_name": "burrow", "confidence": 0.9},
                ]
            },
            {"boxes": [{"class_name": "prairie_dog", "confidence": 0.9}]},
        ]
    )

    assert summary == {
        "burrow": {"count": 1, "min": 0.9, "mean": 0.9, "max": 0.9},
        "prairie_dog": {"count": 3, "min": 0.5, "mean": 0.7, "max": 0.9},
    }


def test_wait_for_rarespot_run_uses_terminal_event_payload(monkeypatch):
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        rarespot_control_base_url="http://control.test",
    )
    calls: list[str] = []

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

        def get(self, url):
            calls.append(url)
            if url.endswith("/v2/runs/run-1"):
                return FakeResponse(
                    {
                        "run_id": "run-1",
                        "status": "succeeded",
                        "response_text": "RareSpot ecology inference completed.",
                    }
                )
            if url.endswith("/v2/runs/run-1/artifacts"):
                return FakeResponse(
                    {
                        "artifacts": [
                            {
                                "artifact_id": "artifact-overlay",
                                "kind": "image",
                                "title": "Overlay: prairie.png",
                                "category": "overlay",
                            },
                            {
                                "artifact_id": "artifact-csv",
                                "kind": "csv",
                                "title": "RareSpot detections",
                                "category": "prediction",
                            },
                            {
                                "artifact_id": "artifact-report",
                                "kind": "report",
                                "title": "RareSpot ecology inference report",
                                "category": "report",
                            },
                        ]
                    }
                )
            if url.endswith("/v2/runs/run-1/events?limit=50&after_sequence=0"):
                return FakeResponse(
                    {
                        "events": [
                            {
                                "event_kind": "run.completed",
                                "payload": {
                                    "configuration": {"conf": 0.55},
                                    "counts_by_class": {"prairie_dog": 2},
                                    "confidence_summary": {
                                        "prairie_dog": {"count": 2, "mean": 0.7, "min": 0.6, "max": 0.8}
                                    },
                                    "top_spectral_review_candidates": [{"file_name": "tile.jpg", "score": 1.2}],
                                },
                            }
                        ]
                    }
                )
            raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr("httpx.Client", FakeClient)

    result = wait_for_rarespot_run(settings, run_id="run-1", timeout_seconds=5)

    assert result["configuration"] == {"conf": 0.55}
    assert result["counts_by_class"] == {"prairie_dog": 2}
    assert result["confidence_summary"]["prairie_dog"]["mean"] == 0.7
    assert result["top_spectral_review_candidates"] == [{"file_name": "tile.jpg", "score": 1.2}]
    assert result["artifact_ids"] == ["artifact-overlay", "artifact-csv", "artifact-report"]
    assert result["key_artifacts"]["annotated_overlay"]["artifact_id"] == "artifact-overlay"
    assert result["key_artifacts"]["detections_csv"]["download_url"] == "/v2/artifacts/artifact-csv/download"
    assert result["key_artifacts"]["report"]["artifact_id"] == "artifact-report"
    assert "Do not search the sandbox filesystem" in result["final_answer_hint"]
    assert "Do not create stub" in result["final_answer_hint"]
    assert "canonical downloadable outputs" in result["final_answer_hint"]


def test_create_rarespot_run_sends_idempotency_header(monkeypatch):
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        rarespot_control_base_url="http://control.test",
    )
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"run_id": "run-1", "status": "queued"}

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

    result = rarespot_tools.create_rarespot_run(
        settings,
        thread_id="thread-1",
        body={"idempotency_key": "rarespot-parent-config", "metadata": {}},
    )

    assert result == {"run_id": "run-1", "status": "queued"}
    assert captured["headers"]["Idempotency-Key"] == "rarespot-parent-config"
