from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.runner import run_job
from ultra_deepagents.schemas import RunJobEnvelope


class FakeRareSpotSynthesisAgent:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def astream_events(self, payload, *, context=None, version=None):
        self.messages = list(payload["messages"])
        assert version == "v3"
        combined = "\n".join(str(message.get("content") or "") for message in self.messages)
        assert "RareSpot ecology inference has already completed" in combined
        assert "artifact_run-nested_overlay" in combined
        assert "Use relative download_url values exactly as provided" in combined
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        *self.messages,
                        {
                            "role": "assistant",
                            "content": (
                                "RareSpot detected 2 burrows and 0 prairie dogs using 512 px "
                                "tiles with 25% overlap, producing the overlay, CSV, and report."
                            ),
                        },
                    ],
                },
            },
        }


class FakeReportOnlyAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        combined = "\n".join(str(message.get("content") or "") for message in payload["messages"])
        assert "RareSpot ecology inference has already completed" not in combined
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        *payload["messages"],
                        {
                            "role": "assistant",
                            "content": "Combined report synthesized from prior RareSpot outputs.",
                        },
                    ],
                },
            },
        }


def test_run_job_preloads_rarespot_for_uploaded_detection_requests(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, Any]] = []

    def fake_create_rarespot_run(_settings, *, thread_id, body):
        calls.append(("create", thread_id, body))
        return {"run_id": "run-nested", "status": "queued"}

    def fake_wait_for_rarespot_run(_settings, *, run_id, timeout_seconds):
        calls.append(("wait", run_id, timeout_seconds))
        return {
            "run_id": run_id,
            "status": "succeeded",
            "response_text": "RareSpot ecology inference completed. Detection counts by class: burrow: 2.",
            "configuration": {
                "tile_size": 512,
                "tile_overlap": 0.25,
                "stride": 384,
                "confidence_threshold": 0.25,
            },
            "counts_by_class": {"burrow": 2, "prairie_dog": 0},
            "confidence_summary": {"burrow": {"count": 2, "mean": 0.31}},
            "key_artifacts": {
                "overlay": {
                    "artifact_id": "artifact_run-nested_overlay",
                    "download_url": "/v2/artifacts/artifact_run-nested_overlay/download",
                    "kind": "figure",
                },
                "detections_csv": {
                    "artifact_id": "artifact_run-nested_csv",
                    "download_url": "/v2/artifacts/artifact_run-nested_csv/download",
                    "kind": "table",
                },
                "report": {
                    "artifact_id": "artifact_run-nested_report",
                    "download_url": "/v2/artifacts/artifact_run-nested_report/download",
                    "kind": "report",
                },
            },
        }

    monkeypatch.setattr(
        "ultra_deepagents.runner.create_rarespot_run",
        fake_create_rarespot_run,
        raising=False,
    )
    monkeypatch.setattr(
        "ultra_deepagents.runner.wait_for_rarespot_run",
        fake_wait_for_rarespot_run,
        raising=False,
    )

    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        workspace_root=str(tmp_path / "workspaces"),
        artifact_root=str(tmp_path / "artifacts"),
        completion_max_continuations=0,
    )
    job = RunJobEnvelope(
        run_id="run-outer",
        thread_id="thread-1",
        user_id="researcher-1",
        goal="Run RareSpot prairie dog detection on the uploaded image and provide quantitative analysis.",
        messages=[
            {
                "role": "user",
                "content": (
                    "Run RareSpot prairie dog detection on the uploaded image. "
                    "Return counts, confidence summary, overlay, CSV, and report."
                ),
            }
        ],
        file_ids=["file-1"],
    )
    published: list[dict[str, Any]] = []
    agent = FakeRareSpotSynthesisAgent()

    async def publish(event: dict[str, Any]) -> None:
        published.append(event)

    result = asyncio.run(
        run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: agent,
        )
    )

    assert result.startswith("RareSpot detected 2 burrows")
    assert calls[0][0] == "create"
    assert calls[0][2]["file_ids"] == ["file-1"]
    assert calls[0][2]["selected_tool_names"] == ["rarespot_ecology_inference"]
    assert calls[0][2]["workflow_hint"] == {"id": "rarespot_ecology"}
    assert calls[1] == ("wait", "run-nested", 3600)
    assert any(
        event["event_kind"] == "tool_call.started"
        and event["payload"]["tool_name"] == "rarespot_ecology_inference"
        for event in published
    )
    completed = [
        event
        for event in published
        if event["event_kind"] == "tool_call.completed"
        and event["payload"]["tool_name"] == "rarespot_ecology_inference"
    ]
    assert completed
    parsed = json.loads(completed[0]["payload"]["output"])
    assert parsed["configuration"]["tile_size"] == 512


def test_run_job_failed_rarespot_preload_fails_parent_without_synthesizing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_create_rarespot_run(_settings, *, thread_id, body):
        return {"run_id": "run-nested-failed", "status": "queued"}

    def fake_wait_for_rarespot_run(_settings, *, run_id, timeout_seconds):
        return {
            "run_id": run_id,
            "status": "failed",
            "error": "RareSpot weights not found",
            "configuration": {},
            "counts_by_class": {},
            "confidence_summary": {},
            "artifact_ids": [],
            "key_artifacts": {},
            "final_answer_hint": (
                "RareSpot inference failed. Do not claim production detections."
            ),
        }

    def fail_agent_factory(*_args, **_kwargs):
        raise AssertionError("failed RareSpot preload must not invoke model synthesis")

    monkeypatch.setattr(
        "ultra_deepagents.runner.create_rarespot_run",
        fake_create_rarespot_run,
        raising=False,
    )
    monkeypatch.setattr(
        "ultra_deepagents.runner.wait_for_rarespot_run",
        fake_wait_for_rarespot_run,
        raising=False,
    )

    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        workspace_root=str(tmp_path / "workspaces"),
        artifact_root=str(tmp_path / "artifacts"),
        completion_max_continuations=0,
    )
    job = RunJobEnvelope(
        run_id="run-outer",
        thread_id="thread-1",
        user_id="researcher-1",
        goal="Run RareSpot prairie dog detection on the uploaded image.",
        messages=[{"role": "user", "content": "Run prairie dog detection on this image."}],
        file_ids=["file-1"],
    )
    published: list[dict[str, Any]] = []

    async def publish(event: dict[str, Any]) -> None:
        published.append(event)

    try:
        asyncio.run(
            run_job(
                job,
                settings,
                publish_event=publish,
                agent_factory=fail_agent_factory,
            )
        )
    except RuntimeError as exc:
        assert "RareSpot inference failed" in str(exc)
    else:
        raise AssertionError("failed RareSpot preload should fail the parent run")

    assert any(
        event["event_kind"] == "tool_call.started"
        and event["payload"].get("source") == "deterministic_preload"
        for event in published
    )
    assert any(
        event["event_kind"] == "tool_call.failed"
        and event["payload"].get("source") == "deterministic_preload"
        and "RareSpot weights not found" in event["payload"].get("error", "")
        for event in published
    )
    assert not any(
        event["event_kind"] == "tool_call.completed"
        and event["payload"].get("source") == "deterministic_preload"
        for event in published
    )


def test_run_job_does_not_preload_rarespot_for_negated_report_only_request(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fail_create_rarespot_run(_settings, *, thread_id, body):
        raise AssertionError("report-only followups must not enqueue RareSpot inference")

    monkeypatch.setattr(
        "ultra_deepagents.runner.create_rarespot_run",
        fail_create_rarespot_run,
        raising=False,
    )

    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        workspace_root=str(tmp_path / "workspaces"),
        artifact_root=str(tmp_path / "artifacts"),
        completion_max_continuations=0,
    )
    job = RunJobEnvelope(
        run_id="run-report",
        thread_id="thread-1",
        user_id="researcher-1",
        goal=(
            "Write a combined quantitative report across the RareSpot runs in this chat. "
            "Include thresholds, counts by class, confidence summaries, artifact links, "
            "limitations, and recommended next analyses. Do not rerun inference; use "
            "the prior RareSpot outputs."
        ),
        messages=[
            {"role": "user", "content": "Run RareSpot prairie dog detection on this image."},
            {"role": "assistant", "content": "RareSpot inference completed with one burrow."},
            {
                "role": "user",
                "content": (
                    "Write a combined quantitative report across the RareSpot runs in this chat. "
                    "Do not rerun inference; use the prior RareSpot outputs."
                ),
            },
        ],
        file_ids=["file-1"],
    )
    published: list[dict[str, Any]] = []

    async def publish(event: dict[str, Any]) -> None:
        published.append(event)

    result = asyncio.run(
        run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeReportOnlyAgent(),
        )
    )

    assert result == "Combined report synthesized from prior RareSpot outputs."
    assert not any(
        event["event_kind"].startswith("tool_call")
        and event["payload"].get("source") == "deterministic_preload"
        for event in published
    )
