from __future__ import annotations

from ultra_deepagents.live_trace import (
    ControlPlaneClient,
    append_followup_messages,
    artifact_download_urls,
    build_tool_capability_matrix,
    build_followup_messages,
    evaluate_bisque_trace_quality,
    evaluate_paper_trace_quality,
    evaluate_rarespot_trace_quality,
    evaluate_thread_trace_quality,
    evaluate_tool_capability_trace_quality,
    evaluate_tool_autonomy_trace_quality,
    evaluate_trace_requirements,
    _artifact_ids_from_text,
    summarize_thread_messages,
    summarize_run_trace,
)


def _rarespot_default_config() -> dict[str, float | int]:
    return {"tile_size": 512, "tile_overlap": 0.25, "stride": 384, "conf": 0.25, "iou": 0.45}


def test_build_followup_messages_preserves_prior_transcript():
    messages = build_followup_messages(
        prompt="Create the first plot.",
        response_text="Saved plot.png.",
        followup="Make it green.",
    )

    assert messages == [
        {"role": "user", "content": "Create the first plot."},
        {"role": "assistant", "content": "Saved plot.png."},
        {"role": "user", "content": "Make it green."},
    ]


def test_append_followup_messages_preserves_growing_transcript():
    messages = append_followup_messages(
        [
            {"role": "user", "content": "Read the paper."},
            {"role": "assistant", "content": "The paper is cached."},
            {"role": "user", "content": "Explain the theorem."},
        ],
        response_text="The theorem follows from the bound.",
        followup="Now inspect Figure 2.",
    )

    assert messages == [
        {"role": "user", "content": "Read the paper."},
        {"role": "assistant", "content": "The paper is cached."},
        {"role": "user", "content": "Explain the theorem."},
        {"role": "assistant", "content": "The theorem follows from the bound."},
        {"role": "user", "content": "Now inspect Figure 2."},
    ]


def test_run_trace_handles_multiple_followups_with_shared_files(monkeypatch):
    import argparse

    from ultra_deepagents import live_trace

    calls: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, base_url: str, *, timeout_seconds: float = 10.0) -> None:
            self.base_url = base_url
            self.timeout_seconds = timeout_seconds

        def upload_files(self, paths):
            assert paths == ["paper.pdf"]
            return ["file-paper"]

        def create_thread(self, *, title: str):
            assert title == "paper stress"
            return {"thread_id": "thread-paper"}

    def fake_trace_prompt(client, **kwargs):
        calls.append(kwargs)
        run_number = len(calls)
        return (
            {"response_text": f"assistant response {run_number}"},
            {
                "run_id": f"run-{run_number}",
                "status": "succeeded",
                "response_len": 1000,
                "artifact_count": 0,
                "artifacts": [],
            },
        )

    monkeypatch.setattr(live_trace, "ControlPlaneClient", FakeClient)
    monkeypatch.setattr(live_trace, "trace_prompt", fake_trace_prompt)

    result = live_trace.run_trace(
        argparse.Namespace(
            base_url="http://control.test",
            http_timeout=3,
            upload_file=["paper.pdf"],
            title="paper stress",
            prompt="Read the paper.",
            followup=["Explain theorem.", "Inspect Figure 2.", "List limitations."],
            timeout=300,
            poll_interval=1,
            verify_downloads=True,
        )
    )

    assert result["uploaded_file_ids"] == ["file-paper"]
    assert result["prompt"]["run_id"] == "run-1"
    assert [summary["run_id"] for summary in result["followups"]] == [
        "run-2",
        "run-3",
        "run-4",
    ]
    assert result["followup"]["run_id"] == "run-4"
    assert [call["file_ids"] for call in calls] == [
        ["file-paper"],
        ["file-paper"],
        ["file-paper"],
        ["file-paper"],
    ]
    assert calls[1]["messages"] == [
        {"role": "user", "content": "Read the paper."},
        {"role": "assistant", "content": "assistant response 1"},
        {"role": "user", "content": "Explain theorem."},
    ]
    assert calls[3]["messages"] == [
        {"role": "user", "content": "Read the paper."},
        {"role": "assistant", "content": "assistant response 1"},
        {"role": "user", "content": "Explain theorem."},
        {"role": "assistant", "content": "assistant response 2"},
        {"role": "user", "content": "Inspect Figure 2."},
        {"role": "assistant", "content": "assistant response 3"},
        {"role": "user", "content": "List limitations."},
    ]


def test_run_trace_records_visible_thread_transcript_after_completion(monkeypatch):
    import argparse

    from ultra_deepagents import live_trace

    class FakeClient:
        def __init__(self, base_url: str, *, timeout_seconds: float = 10.0) -> None:
            self.base_url = base_url
            self.timeout_seconds = timeout_seconds

        def upload_files(self, paths):
            return []

        def create_thread(self, *, title: str):
            return {"thread_id": "thread-1", "latest_run_id": None}

        def get_thread(self, thread_id: str):
            assert thread_id == "thread-1"
            return {"thread_id": "thread-1", "latest_run_id": "run-2"}

        def list_thread_messages(self, thread_id: str):
            assert thread_id == "thread-1"
            return [
                {"role": "user", "content": "First prompt.", "run_id": "run-1"},
                {"role": "assistant", "content": "First answer.", "run_id": "run-1"},
                {"role": "user", "content": "Follow up.", "run_id": "run-2"},
                {"role": "assistant", "content": "Followup answer.", "run_id": "run-2"},
            ]

    def fake_trace_prompt(client, **kwargs):
        run_number = 1 if kwargs["prompt"] == "First prompt." else 2
        return (
            {"run_id": f"run-{run_number}", "response_text": f"answer {run_number}"},
            {
                "run_id": f"run-{run_number}",
                "status": "succeeded",
                "response_len": 1000,
                "artifact_count": 0,
                "artifacts": [],
            },
        )

    monkeypatch.setattr(live_trace, "ControlPlaneClient", FakeClient)
    monkeypatch.setattr(live_trace, "trace_prompt", fake_trace_prompt)

    result = live_trace.run_trace(
        argparse.Namespace(
            base_url="http://control.test",
            http_timeout=3,
            upload_file=[],
            title="thread quality",
            prompt="First prompt.",
            followup=["Follow up."],
            timeout=300,
            poll_interval=1,
            verify_downloads=False,
        )
    )

    assert result["thread"] == {"thread_id": "thread-1", "latest_run_id": "run-2"}
    assert result["thread_messages"]["roles"] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert result["thread_messages"]["run_ids"] == ["run-1", "run-1", "run-2", "run-2"]


def test_trace_prompt_refreshes_events_after_terminal_status():
    class FakeClient:
        def __init__(self) -> None:
            self.event_calls = 0

        def create_run(self, **kwargs):
            return {"run_id": "run-terminal"}

        def get_run(self, run_id: str):
            assert run_id == "run-terminal"
            return {
                "run_id": run_id,
                "thread_id": "thread-1",
                "goal": "terminal event",
                "status": "succeeded",
                "response_text": "complete",
            }

        def list_run_events(self, run_id: str):
            self.event_calls += 1
            if self.event_calls == 1:
                return [
                    {
                        "event_kind": "run.started",
                        "sequence": 1,
                        "payload": {},
                    }
                ]
            return [
                {
                    "event_kind": "run.started",
                    "sequence": 1,
                    "payload": {},
                },
                {
                    "event_kind": "run.completed",
                    "sequence": 2,
                    "payload": {"response_text": "complete"},
                },
            ]

        def list_run_artifacts(self, run_id: str):
            return []

    from ultra_deepagents.live_trace import trace_prompt

    _, summary = trace_prompt(
        FakeClient(),
        thread_id="thread-1",
        prompt="terminal event",
        timeout_seconds=1,
        poll_interval_seconds=0,
    )

    assert summary["event_counts"]["run.completed"] == 1
    assert summary["terminal_event_kind"] == "run.completed"


def test_run_trace_logs_into_bisque_with_env_credentials(monkeypatch):
    import argparse

    from ultra_deepagents import live_trace

    monkeypatch.setenv("ULTRA_TEST_BISQUE_USERNAME", "linked-user")
    monkeypatch.setenv("ULTRA_TEST_BISQUE_PASSWORD", "secret-password")
    calls: list[tuple[str, object]] = []

    class FakeClient:
        def __init__(self, base_url: str, *, timeout_seconds: float = 10.0) -> None:
            self.base_url = base_url
            self.timeout_seconds = timeout_seconds

        def login_bisque_account(self, *, username: str, password: str):
            calls.append(("login", {"username": username, "password": password}))
            return {
                "authenticated": True,
                "mode": "bisque",
                "user": {"username": username},
            }

        def upload_files(self, paths):
            calls.append(("upload_files", paths))
            return []

        def create_thread(self, *, title: str):
            calls.append(("create_thread", title))
            return {"thread_id": "thread-bisque"}

        def get_thread(self, thread_id: str):
            return {"thread_id": thread_id, "latest_run_id": "run-1"}

        def list_thread_messages(self, thread_id: str):
            return [
                {"role": "user", "content": "Use BisQue.", "run_id": "run-1"},
                {"role": "assistant", "content": "Uploaded to BisQue.", "run_id": "run-1"},
            ]

    def fake_trace_prompt(client, **kwargs):
        calls.append(("trace_prompt", kwargs["prompt"]))
        return (
            {"run_id": "run-1", "response_text": "Uploaded to BisQue."},
            {
                "run_id": "run-1",
                "status": "succeeded",
                "response_len": 1000,
                "artifact_count": 1,
                "artifacts": [{"artifact_id": "artifact-1", "download_ok": True}],
            },
        )

    monkeypatch.setattr(live_trace, "ControlPlaneClient", FakeClient)
    monkeypatch.setattr(live_trace, "trace_prompt", fake_trace_prompt)

    result = live_trace.run_trace(
        argparse.Namespace(
            base_url="http://control.test",
            http_timeout=3,
            upload_file=[],
            title="bisque trace",
            prompt="Use BisQue.",
            followup=[],
            timeout=300,
            poll_interval=1,
            verify_downloads=True,
            bisque_username_env="ULTRA_TEST_BISQUE_USERNAME",
            bisque_password_env="ULTRA_TEST_BISQUE_PASSWORD",
        )
    )

    assert calls[0] == (
        "login",
        {"username": "linked-user", "password": "secret-password"},
    )
    assert result["bisque_session"] == {
        "authenticated": True,
        "mode": "bisque",
        "username": "linked-user",
    }


def test_evaluate_thread_trace_quality_scores_clean_multiturn_transcript():
    result = {
        "thread": {"thread_id": "thread-1", "latest_run_id": "run-2"},
        "thread_messages": {
            "count": 4,
            "roles": ["user", "assistant", "user", "assistant"],
            "run_ids": ["run-1", "run-1", "run-2", "run-2"],
            "content_lengths": [12, 900, 13, 850],
        },
        "prompt": {"run_id": "run-1", "status": "succeeded", "response_len": 900},
        "followups": [
            {"run_id": "run-2", "status": "succeeded", "response_len": 850},
        ],
    }

    quality = evaluate_thread_trace_quality(result)

    assert quality["passed"] is True
    assert quality["score"] >= 8.5
    assert quality["issues"] == []


def test_evaluate_bisque_trace_quality_scores_multiturn_download_analyze_upload():
    result = {
        "prompt": {
            "run_id": "run-1",
            "status": "succeeded",
            "response_len": 2200,
            "response_preview": (
                "Searched the linked BisQue account, recovered after failed image "
                "candidates, imported EnrNE_Day2_Run1__0210.JPG, computed RGB "
                "channel means, saved a histogram figure and markdown report, "
                "then uploaded generated outputs back to BisQue data_service URIs."
            ),
            "tool_names": [
                "bisque_search_resources",
                "bisque_download_resource",
                "stage_uploaded_files_for_analysis",
                "execute",
                "bisque_upload_workspace_files",
            ],
            "artifacts": [
                {
                    "artifact_id": "artifact-figure",
                    "kind": "figure",
                    "mime_type": "image/png",
                    "download_ok": True,
                },
                {
                    "artifact_id": "artifact-report",
                    "kind": "report",
                    "mime_type": "text/markdown",
                    "download_ok": True,
                },
                {
                    "artifact_id": "artifact-code",
                    "kind": "code",
                    "path": "analyze_bisque_image.py",
                    "download_ok": True,
                },
            ],
        },
        "followups": [
            {
                "run_id": "run-2",
                "status": "succeeded",
                "response_len": 1600,
                "response_preview": (
                    "Reused the same BisQue image and prior artifacts, produced an "
                    "additional RGB histogram comparison figure, and uploaded the "
                    "new figure to a BisQue data_service resource."
                ),
                "tool_names": [
                    "artifact_manifest",
                    "stage_artifact_for_analysis",
                    "stage_uploaded_files_for_analysis",
                    "read_file",
                    "execute",
                    "bisque_upload_workspace_files",
                ],
                "artifacts": [
                    {
                        "artifact_id": "artifact-followup-figure",
                        "kind": "figure",
                        "mime_type": "image/png",
                        "download_ok": True,
                    },
                    {
                        "artifact_id": "artifact-followup-code",
                        "kind": "code",
                        "path": "compare_rgb_histograms.py",
                        "download_ok": True,
                    },
                ],
            }
        ],
    }

    quality = evaluate_bisque_trace_quality(result)

    assert quality["passed"] is True
    assert quality["score"] >= 8.5
    assert quality["issues"] == []


def test_evaluate_thread_trace_quality_reports_tool_pollution_and_missing_assistant():
    result = {
        "thread": {"thread_id": "thread-1", "latest_run_id": "run-internal"},
        "thread_messages": {
            "count": 3,
            "roles": ["user", "tool", "user"],
            "run_ids": ["run-1", "run-internal", "run-2"],
            "content_lengths": [12, 50, 13],
        },
        "prompt": {"run_id": "run-1", "status": "succeeded", "response_len": 900},
        "followups": [
            {"run_id": "run-2", "status": "succeeded", "response_len": 850},
        ],
    }

    quality = evaluate_thread_trace_quality(result)

    assert quality["passed"] is False
    assert "thread transcript does not contain one user/assistant pair per turn" in quality["issues"]
    assert "thread transcript contains tool/internal messages" in quality["issues"]
    assert "assistant message missing for run run-1" in quality["issues"]
    assert "thread latest_run_id does not match final user-facing run" in quality["issues"]


def test_evaluate_tool_autonomy_trace_quality_scores_multiturn_coding_work():
    result = {
        "prompt": {
            "run_id": "run-1",
            "status": "succeeded",
            "response_len": 2800,
            "response_preview": (
                "Built a reproducible sklearn pipeline with metrics, ROC curves, "
                "calibration, confusion matrix, feature importance, and saved artifacts."
            ),
            "tool_names": ["write_file", "execute", "execute"],
            "idle_recovery_count": 0,
            "artifacts": [
                {"kind": "code", "path": "outputs/analysis.py", "download_ok": True},
                {"kind": "figure", "path": "outputs/roc.png", "download_ok": True},
                {"kind": "table", "path": "outputs/metrics.csv", "download_ok": True},
            ],
        },
        "followups": [
            {
                "run_id": "run-2",
                "status": "succeeded",
                "response_len": 1900,
                "response_preview": (
                    "Reused the prior artifacts, added PCA and error analysis, and "
                    "saved updated plots and report."
                ),
                "tool_names": [
                    "artifact_manifest",
                    "stage_artifact_for_analysis",
                    "read_file",
                    "write_file",
                    "execute",
                ],
                "idle_recovery_count": 0,
                "artifacts": [
                    {"kind": "figure", "path": "outputs/pca_errors.png", "download_ok": True},
                    {"kind": "report", "path": "outputs/report.md", "download_ok": True},
                ],
            },
            {
                "run_id": "run-3",
                "status": "succeeded",
                "response_len": 1400,
                "response_preview": (
                    "Final technical comparison report includes quantitative metrics, "
                    "model tradeoffs, limitations, reproducibility notes, and recommended "
                    "next experiments."
                ),
                "tool_names": [
                    "artifact_manifest",
                    "stage_artifact_for_analysis",
                    "read_file",
                    "write_file",
                ],
                "idle_recovery_count": 0,
                "artifacts": [
                    {"kind": "report", "path": "outputs/final_report.md", "download_ok": True},
                ],
            },
        ],
    }

    quality = evaluate_tool_autonomy_trace_quality(result)

    assert quality["passed"] is True
    assert quality["score"] >= 8.5
    assert quality["issues"] == []
    assert quality["signals"]["code_execution"] is True
    assert quality["signals"]["followup_context"] is True
    assert quality["signals"]["durable_outputs"] is True


def test_evaluate_tool_autonomy_trace_quality_uses_tail_for_conclusions():
    long_intro = (
        "Created a Python script and generated the requested plot artifact. "
        "The figure is saved and downloadable. "
        * 80
    )
    response_tail = (
        "Quantitative check: y=x^2 grows from 0 to 25 with successive "
        "increments 1, 3, 5, 7, 9. The follow-up comparison against y=x^3 "
        "shows y=125 at x=5, which is 5x the quadratic endpoint. "
        "Reproducibility: the code is saved, has no random seed requirement, "
        "and can be rerun in the same environment. Limitation: this is a "
        "small deterministic demonstration, not a statistical experiment."
    )
    result = {
        "prompt": {
            "run_id": "run-1",
            "status": "succeeded",
            "response_len": len(long_intro) + len(response_tail),
            "response_preview": long_intro[:4000],
            "response_tail": response_tail,
            "tool_names": ["tool_capability_manifest", "write_file", "execute"],
            "idle_recovery_count": 0,
            "artifacts": [
                {"kind": "code", "path": "outputs/plot_squared.py", "download_ok": True},
                {"kind": "figure", "path": "outputs/plot_squared.png", "download_ok": True},
            ],
        },
        "followups": [
            {
                "run_id": "run-2",
                "status": "succeeded",
                "response_len": 1200,
                "response_preview": (
                    "Used artifact_manifest and read_file to stage the prior script, "
                    "then wrote and executed a cubed comparison plot."
                ),
                "response_tail": (
                    "The comparison plot, saved CSV table, and report make the "
                    "tradeoff between quadratic and cubic growth explicit."
                ),
                "tool_names": [
                    "artifact_manifest",
                    "stage_artifact_for_analysis",
                    "read_file",
                    "write_file",
                    "execute",
                ],
                "idle_recovery_count": 0,
                "artifacts": [
                    {"kind": "code", "path": "outputs/plot_cubed.py", "download_ok": True},
                    {"kind": "figure", "path": "outputs/plot_cubed.png", "download_ok": True},
                ],
            },
        ],
    }

    quality = evaluate_tool_autonomy_trace_quality(result)

    assert quality["passed"] is True
    assert quality["signals"]["substantive_analysis"] is True


def test_evaluate_tool_autonomy_trace_quality_accepts_mathematical_plot_analysis():
    result = {
        "prompt": {
            "run_id": "run-1",
            "status": "succeeded",
            "response_len": 1296,
            "response_preview": (
                "The figure shows the quadratic function \\(y = x^2\\) evaluated "
                "at integer points \\(x = 0, 1, 2, 3, 4, 5\\). The saved plot "
                "demonstrates non-linear growth: successive differences are "
                "1, 3, 5, 7, 9, and the range rises from 0 to 25. The code "
                "and figure are durable artifacts."
            ),
            "tool_names": ["tool_capability_manifest", "write_file", "execute", "ls"],
            "idle_recovery_count": 0,
            "artifacts": [
                {"kind": "code", "path": "plot_xsquared.py", "download_ok": True},
                {"kind": "figure", "path": "outputs/plot_xsquared.png", "download_ok": True},
            ],
        },
        "followups": [
            {
                "run_id": "run-2",
                "status": "succeeded",
                "response_len": 1920,
                "response_preview": (
                    "Using artifact_manifest, stage_artifact_for_analysis, and "
                    "read_file, I reused the prior code and modified the analysis "
                    "to \\(y = x^3\\). The comparison shows 0-125 versus 0-25, "
                    "with cubic successive differences 1, 7, 19, 37, 61. "
                    "The updated source code and plot are saved for future follow-ups."
                ),
                "tool_names": [
                    "tool_capability_manifest",
                    "artifact_manifest",
                    "stage_artifact_for_analysis",
                    "read_file",
                    "write_file",
                    "execute",
                ],
                "idle_recovery_count": 0,
                "artifacts": [
                    {"kind": "code", "path": "outputs/plot_xcubed.py", "download_ok": True},
                    {"kind": "figure", "path": "outputs/plot_xcubed.png", "download_ok": True},
                ],
            },
        ],
    }

    quality = evaluate_tool_autonomy_trace_quality(result)

    assert quality["passed"] is True
    assert quality["signals"]["substantive_analysis"] is True


def test_evaluate_tool_autonomy_trace_quality_reports_missing_core_signals():
    result = {
        "prompt": {
            "run_id": "run-1",
            "status": "failed",
            "response_len": 120,
            "response_preview": "I can do this.",
            "tool_names": ["write_file"],
            "idle_recovery_count": 1,
            "artifacts": [],
        },
        "followups": [
            {
                "run_id": "run-2",
                "status": "succeeded",
                "response_len": 200,
                "response_preview": "Follow-up done.",
                "tool_names": [],
                "idle_recovery_count": 0,
                "artifacts": [
                    {"kind": "code", "path": "outputs/script.py", "download_ok": False},
                ],
            },
        ],
    }

    quality = evaluate_tool_autonomy_trace_quality(result)

    assert quality["passed"] is False
    assert quality["score"] < 8.5
    assert "one or more turns did not succeed" in quality["issues"]
    assert "missing execute tool evidence" in quality["issues"]
    assert "missing durable code plus analysis artifacts" in quality["issues"]
    assert "artifact download verification failed or was not captured" in quality["issues"]
    assert "follow-ups did not inspect or stage prior context" in quality["issues"]
    assert "one or more responses are too short for autonomous analysis" in quality["issues"]
    assert "run had idle recoveries" in quality["issues"]


def test_summarize_thread_messages_keeps_shape_without_large_content():
    summary = summarize_thread_messages(
        [
            {"role": "user", "content": "A" * 10, "run_id": "run-1"},
            {"role": "assistant", "content": "B" * 500, "run_id": "run-1"},
        ]
    )

    assert summary["count"] == 2
    assert summary["roles"] == ["user", "assistant"]
    assert summary["run_ids"] == ["run-1", "run-1"]
    assert summary["content_lengths"] == [10, 500]
    assert summary["content_previews"][1].endswith("…")


def test_summarize_run_trace_reports_events_artifacts_and_downloads():
    summary = summarize_run_trace(
        run={
            "run_id": "run-1",
            "status": "succeeded",
            "response_text": "Created a plot from paper_id:p4.",
        },
        events=[
            {"event_kind": "run.accepted"},
            {"event_kind": "run.started"},
            {"event_kind": "tool_call.started", "payload": {"tool_name": "search_paper"}},
            {"event_kind": "message.delta"},
            {"event_kind": "artifact.created"},
            {"event_kind": "run.completed"},
        ],
        artifacts=[
            {
                "artifact_id": "artifact-1",
                "kind": "figure",
                "path": "plot.png",
                "mime_type": "image/png",
                "size_bytes": 123,
            }
        ],
        first_delta_ms=412.4,
        terminal_ms=1234.8,
        accepted_ms=51.2,
        first_event_ms=100.0,
        first_backend_event_ms=150.0,
        first_tool_ms=250.0,
        first_artifact_ms=900.0,
        first_recovery_ms=None,
        download_checks={"artifact-1": True},
    )

    assert summary["status"] == "succeeded"
    assert summary["response_len"] == len("Created a plot from paper_id:p4.")
    assert summary["response_preview"] == "Created a plot from paper_id:p4."
    assert summary["tool_names"] == ["search_paper"]
    assert summary["event_counts"] == {
        "run.accepted": 1,
        "run.started": 1,
        "tool_call.started": 1,
        "message.delta": 1,
        "artifact.created": 1,
        "run.completed": 1,
    }
    assert summary["accepted_ms"] == 51.2
    assert summary["artifact_count"] == 1
    assert summary["artifacts"] == [
        {
            "artifact_id": "artifact-1",
            "kind": "figure",
            "path": "plot.png",
            "mime_type": "image/png",
            "size_bytes": 123,
            "download_ok": True,
        }
    ]


def test_summarize_run_trace_extracts_rarespot_configuration_from_tool_output():
    summary = summarize_run_trace(
        run={
            "run_id": "run-1",
            "status": "succeeded",
            "response_text": "RareSpot completed.",
        },
        events=[
            {"event_kind": "tool_call.started", "payload": {"tool_name": "rarespot_ecology_inference"}},
            {
                "event_kind": "tool_call.completed",
                "payload": {
                    "tool_name": "rarespot_ecology_inference",
                    "output_preview": (
                        '{"run_id":"nested-1","status":"succeeded",'
                        '"configuration":{"tile_size":512,"tile_overlap":0.25,'
                        '"stride":384,"conf":0.25,"iou":0.45}}'
                    ),
                },
            },
            {"event_kind": "run.completed"},
        ],
        artifacts=[],
    )

    assert summary["rarespot_configurations"] == [
        {"tile_size": 512, "tile_overlap": 0.25, "stride": 384, "conf": 0.25, "iou": 0.45}
    ]


def test_summarize_run_trace_extracts_tool_capability_manifest():
    summary = summarize_run_trace(
        run={
            "run_id": "run-1",
            "status": "succeeded",
            "response_text": "I inspected available tools and ran the analysis.",
        },
        events=[
            {"event_kind": "tool_call.started", "payload": {"tool_name": "tool_capability_manifest"}},
            {
                "event_kind": "tool_call.completed",
                "payload": {
                    "tool_name": "tool_capability_manifest",
                    "output_preview": (
                        '{"deepagents_builtin_tools":[{"name":"execute"},{"name":"write_file"},'
                        '{"name":"read_file"}],"registered_tools":["artifact_manifest",'
                        '"stage_uploaded_files_for_analysis"],"storage":{"workspace":"/workspace",'
                        '"outputs":"/outputs","memories":"/memories"}}'
                    ),
                },
            },
            {"event_kind": "run.completed"},
        ],
        artifacts=[],
    )

    assert summary["tool_capability_manifests"] == [
        {
            "deepagents_builtin_tools": [
                {"name": "execute"},
                {"name": "write_file"},
                {"name": "read_file"},
            ],
            "registered_tools": [
                "artifact_manifest",
                "stage_uploaded_files_for_analysis",
            ],
            "storage": {
                "workspace": "/workspace",
                "outputs": "/outputs",
                "memories": "/memories",
            },
        }
    ]


def test_evaluate_tool_capability_trace_quality_scores_known_tool_surface():
    result = {
        "prompt": {
            "run_id": "run-1",
            "status": "succeeded",
            "response_len": 1200,
            "tool_names": [
                "tool_capability_manifest",
                "execute",
                "artifact_manifest",
                "stage_uploaded_files_for_analysis",
            ],
            "tool_capability_manifests": [
                {
                    "deepagents_builtin_tools": [
                        {"name": "execute"},
                        {"name": "write_file"},
                        {"name": "read_file"},
                        {"name": "edit_file"},
                        {"name": "ls"},
                        {"name": "glob"},
                        {"name": "grep"},
                        {"name": "write_todos"},
                    ],
                    "registered_tools": [
                        "artifact_manifest",
                        "stage_artifact_for_analysis",
                        "stage_uploaded_files_for_analysis",
                        "tool_capability_manifest",
                    ],
                    "storage": {
                        "workspace": "/workspace",
                        "outputs": "/outputs",
                        "memories": "/memories",
                        "staged_uploads": "/workspace/staged_uploads",
                        "staged_artifacts": "/workspace/staged_artifacts",
                    },
                }
            ],
        }
    }

    quality = evaluate_tool_capability_trace_quality(result)

    assert quality["passed"] is True
    assert quality["score"] >= 8.5
    assert quality["issues"] == []
    assert quality["signals"]["manifest_called"] is True
    assert quality["signals"]["execute_declared"] is True
    assert quality["signals"]["used_tools_declared"] is True


def test_evaluate_tool_capability_trace_quality_reports_missing_manifest_and_unknown_tools():
    result = {
        "prompt": {
            "run_id": "run-1",
            "status": "succeeded",
            "response_len": 1200,
            "tool_names": ["execute", "mystery_tool"],
            "tool_capability_manifests": [],
        }
    }

    quality = evaluate_tool_capability_trace_quality(result)

    assert quality["passed"] is False
    assert quality["score"] < 8.5
    assert "tool capability manifest was not called or captured" in quality["issues"]
    assert "used tools were not declared in any captured manifest: mystery_tool" in quality["issues"]


def test_build_tool_capability_matrix_summarizes_available_and_used_categories():
    result = {
        "prompt": {
            "run_id": "run-1",
            "status": "succeeded",
            "tool_names": [
                "tool_capability_manifest",
                "execute",
                "write_file",
                "artifact_manifest",
                "stage_artifact_for_analysis",
                "mystery_tool",
            ],
            "tool_capability_manifests": [
                {
                    "deepagents_builtin_tools": [
                        {"name": "execute"},
                        {"name": "write_file"},
                        {"name": "read_file"},
                        {"name": "edit_file"},
                        {"name": "ls"},
                        {"name": "glob"},
                        {"name": "grep"},
                        {"name": "write_todos"},
                    ],
                    "registered_tools": [
                        "artifact_manifest",
                        "stage_artifact_for_analysis",
                        "stage_uploaded_files_for_analysis",
                        "rarespot_ecology_inference",
                    ],
                    "storage": {
                        "workspace": "/workspace",
                        "outputs": "/outputs",
                        "memories": "/memories",
                        "staged_artifacts": "/workspace/staged_artifacts",
                    },
                }
            ],
            "artifacts": [
                {"kind": "code", "path": "outputs/analysis.py", "download_ok": True},
                {"kind": "figure", "path": "outputs/plot.png", "download_ok": True},
            ],
        },
        "followups": [
            {
                "run_id": "run-2",
                "status": "succeeded",
                "tool_names": [
                    "artifact_manifest",
                    "stage_artifact_for_analysis",
                    "read_file",
                    "execute",
                ],
                "tool_capability_manifests": [],
                "artifacts": [
                    {"kind": "report", "path": "outputs/report.md", "download_ok": True}
                ],
            }
        ],
    }

    matrix = build_tool_capability_matrix(result)

    assert matrix["turn_count"] == 2
    assert matrix["unknown_tools"] == ["mystery_tool"]
    assert matrix["capabilities"]["sandbox_execution"] == {"available": True, "used": True}
    assert matrix["capabilities"]["filesystem"] == {"available": True, "used": True}
    assert matrix["capabilities"]["context_staging"] == {"available": True, "used": True}
    assert matrix["capabilities"]["durable_outputs"] == {"available": True, "used": True}
    assert matrix["capabilities"]["rarespot_detection"] == {"available": True, "used": False}
    assert matrix["capabilities"]["paper_review"] == {"available": False, "used": False}
    assert matrix["turns"][0]["capabilities"]["context_staging"] is True
    assert matrix["turns"][1]["capabilities"]["followup_context_reuse"] is True


def test_evaluate_paper_trace_quality_scores_page_grounded_multiturn_response():
    result = {
        "prompt": {
            "run_id": "run-1",
            "status": "succeeded",
            "response_len": 4000,
            "response_preview": (
                "Scaled dot-product attention computes softmax(QK^T / sqrt(d_k))V "
                "with page-grounded evidence from arxiv_1706.03762:p4. "
                "Figure 2 shows the attention block."
            ),
            "artifact_count": 1,
            "artifacts": [{"kind": "figure", "download_ok": True}],
            "tool_names": ["ingest_arxiv_paper", "search_paper", "read_paper_pages", "render_paper_page"],
            "idle_recovery_count": 0,
        },
        "followups": [
            {
                "run_id": "run-2",
                "status": "succeeded",
                "response_len": 2200,
                "response_preview": (
                    "The variance of q dot k is d_k sigma^4, so dividing by sqrt(d_k) "
                    "keeps the logits stable; see arxiv_1706.03762:p4."
                ),
                "artifact_count": 0,
                "artifacts": [],
                "tool_names": ["paper_manifest", "search_paper", "read_paper_pages"],
                "idle_recovery_count": 0,
            },
            {
                "run_id": "run-3",
                "status": "succeeded",
                "response_len": 2600,
                "response_preview": (
                    "The Q, K, and V projections are concatenated across heads and "
                    "projected back; see arxiv_1706.03762:p5. Limitations include "
                    "sequence length cost and assumptions about tokenized inputs."
                ),
                "artifact_count": 1,
                "artifacts": [{"kind": "figure", "download_ok": True}],
                "tool_names": ["search_paper", "read_paper_section", "render_paper_page"],
                "idle_recovery_count": 0,
            },
        ],
    }

    quality = evaluate_paper_trace_quality(result)

    assert quality["passed"] is True
    assert quality["score"] >= 8.5
    assert quality["issues"] == []
    assert quality["turn_count"] == 3


def test_evaluate_paper_trace_quality_reports_missing_research_grade_signals():
    quality = evaluate_paper_trace_quality(
        {
            "prompt": {
                "run_id": "run-1",
                "status": "succeeded",
                "response_len": 100,
                "response_preview": "This paper is interesting.",
                "artifact_count": 0,
                "artifacts": [],
                "tool_names": [],
                "idle_recovery_count": 1,
            }
        },
        min_score=8.5,
    )

    assert quality["passed"] is False
    assert quality["score"] < 8.5
    assert "missing page-grounded citations" in quality["issues"]
    assert "missing paper-tool evidence" in quality["issues"]
    assert "missing rendered/downloadable figure evidence" in quality["issues"]
    assert "missing technical/equation content" in quality["issues"]
    assert "run had idle recoveries" in quality["issues"]


def test_evaluate_rarespot_trace_quality_scores_multirun_analysis():
    result = {
        "prompt": {
            "run_id": "run-1",
            "status": "succeeded",
            "response_len": 2200,
            "response_preview": (
                "RareSpot inference completed with 4 prairie_dog detections and 2 burrow "
                "detections. Configuration: confidence threshold 0.25, tile overlap 0.25. "
                "The overlay and detection CSV are saved for download."
            ),
            "artifact_count": 3,
            "artifacts": [
                {"kind": "image", "download_ok": True},
                {"kind": "csv", "download_ok": True},
                {"kind": "report", "download_ok": True},
            ],
            "tool_names": ["rarespot_ecology_inference"],
            "rarespot_configurations": [_rarespot_default_config()],
            "idle_recovery_count": 0,
        },
        "followups": [
            {
                "run_id": "run-2",
                "status": "succeeded",
                "response_len": 1800,
                "response_preview": (
                    "Second RareSpot run used confidence threshold 0.55 and produced "
                    "3 prairie_dog detections and 1 burrow detection. Compared with run-1, "
                    "the stricter pass removed lower-confidence detections."
                ),
                "artifact_count": 3,
                "artifacts": [{"kind": "image", "download_ok": True}, {"kind": "csv", "download_ok": True}],
                "tool_names": ["rarespot_ecology_inference"],
                "rarespot_configurations": [_rarespot_default_config()],
                "idle_recovery_count": 0,
            },
            {
                "run_id": "run-3",
                "status": "succeeded",
                "response_len": 2400,
                "response_preview": (
                    "Combined report: run-1 and run-2 show robust prairie dog detections. "
                    "Artifact IDs are artifact-a and artifact-b. Confidence mean and spectral "
                    "review metrics support manual review of uncertain boxes. Limitations include "
                    "single-image sampling and threshold sensitivity."
                ),
                "artifact_count": 1,
                "artifacts": [{"kind": "report", "download_ok": True}],
                "tool_names": ["stage_artifact_for_analysis"],
                "idle_recovery_count": 0,
            },
        ],
    }

    quality = evaluate_rarespot_trace_quality(result)

    assert quality["passed"] is True
    assert quality["score"] >= 8.5
    assert quality["issues"] == []


def test_evaluate_rarespot_trace_quality_rejects_stub_or_duplicate_outputs():
    result = {
        "prompt": {
            "run_id": "run-1",
            "status": "succeeded",
            "response_len": 2200,
            "response_preview": (
                "RareSpot inference completed with 2 burrow detections and 0 prairie dog "
                "detections. Local copies of the summary and a stub CSV were saved."
            ),
            "artifact_count": 2,
            "artifacts": [
                {"kind": "artifact", "path": "outputs/rarespot_summary.json", "download_ok": True},
                {"kind": "table", "path": "outputs/rarespot_detections.csv", "download_ok": True},
            ],
            "linked_artifacts": [
                {"kind": "image", "path": "overlay.png", "download_ok": True},
                {"kind": "csv", "path": "detections.csv", "download_ok": True},
                {"kind": "report", "path": "report.md", "download_ok": True},
            ],
            "tool_names": ["rarespot_ecology_inference"],
            "rarespot_configurations": [_rarespot_default_config()],
            "idle_recovery_count": 0,
        },
    }

    quality = evaluate_rarespot_trace_quality(result)

    assert quality["passed"] is False
    assert "response mentions stub or duplicate RareSpot outputs" in quality["issues"]


def test_evaluate_rarespot_trace_quality_accepts_negated_stub_or_duplicate_text():
    result = {
        "prompt": {
            "run_id": "run-1",
            "status": "succeeded",
            "response_len": 2200,
            "response_preview": (
                "RareSpot inference completed with 2 burrow detections and 0 prairie dog "
                "detections. Count by class: burrow=2, prairie dog=0. Confidence mean "
                "is 0.71 with threshold 0.25. NMS duplicate suppression used IoU 0.45. "
                "These are the canonical artifacts; no duplicate or stub outputs have "
                "been created."
            ),
            "artifact_count": 0,
            "artifacts": [],
            "linked_artifacts": [
                {"kind": "image", "path": "overlay.png", "download_ok": True},
                {"kind": "csv", "path": "detections.csv", "download_ok": True},
                {"kind": "report", "path": "report.md", "download_ok": True},
            ],
            "tool_names": ["rarespot_ecology_inference"],
            "rarespot_configurations": [_rarespot_default_config()],
            "idle_recovery_count": 0,
        },
    }

    quality = evaluate_rarespot_trace_quality(result)

    assert quality["passed"] is True
    assert quality["signals"]["no_stub_outputs"] is True


def test_evaluate_rarespot_trace_quality_accepts_verified_linked_nested_artifacts():
    result = {
        "prompt": {
            "run_id": "run-1",
            "status": "succeeded",
            "response_len": 2200,
            "response_preview": (
                "RareSpot inference completed with 2 burrow detections and 0 prairie dog detections. "
                "Download annotated overlay and detections CSV from /v2/artifacts/.../download."
            ),
            "artifact_count": 0,
            "artifacts": [],
            "linked_artifacts": [
                {"kind": "image", "path": "overlay.png", "download_ok": True},
                {"kind": "csv", "path": "detections.csv", "download_ok": True},
                {"kind": "report", "path": "report.md", "download_ok": True},
            ],
            "tool_names": ["rarespot_ecology_inference"],
            "rarespot_configurations": [_rarespot_default_config()],
            "idle_recovery_count": 0,
        },
        "followups": [
            {
                "run_id": "run-2",
                "status": "succeeded",
                "response_len": 1600,
                "response_preview": (
                    "Compared the stricter confidence threshold run against run-1. "
                    "The robust detection remained, the low-confidence burrow disappeared."
                ),
                "artifact_count": 0,
                "artifacts": [],
                "linked_artifacts": [],
                "tool_names": ["rarespot_ecology_inference"],
                "rarespot_configurations": [_rarespot_default_config()],
                "idle_recovery_count": 0,
            },
            {
                "run_id": "run-3",
                "status": "succeeded",
                "response_len": 1800,
                "response_preview": (
                    "Combined report synthesizes configuration values, confidence summaries, "
                    "spectral review metrics, limitations, and recommended next steps."
                ),
                "artifact_count": 0,
                "artifacts": [],
                "linked_artifacts": [],
                "tool_names": ["artifact_manifest"],
                "idle_recovery_count": 0,
            },
        ],
    }

    quality = evaluate_rarespot_trace_quality(result)

    assert quality["passed"] is True
    assert quality["signals"]["artifacts"] is True
    assert quality["signals"]["expected_configuration"] is True


def test_evaluate_rarespot_trace_quality_rejects_unexpected_tile_configuration():
    result = {
        "prompt": {
            "run_id": "run-1",
            "status": "succeeded",
            "response_len": 2200,
            "response_preview": (
                "RareSpot inference completed with 2 burrow detections and 0 prairie dog detections. "
                "Download annotated overlay and detections CSV from /v2/artifacts/.../download."
            ),
            "artifact_count": 0,
            "artifacts": [],
            "linked_artifacts": [
                {"kind": "image", "path": "overlay.png", "download_ok": True},
                {"kind": "csv", "path": "detections.csv", "download_ok": True},
            ],
            "tool_names": ["rarespot_ecology_inference"],
            "rarespot_configurations": [
                {"tile_size": 1280, "tile_overlap": 0.25, "stride": 960, "conf": 0.25, "iou": 0.45}
            ],
            "idle_recovery_count": 0,
        },
    }

    quality = evaluate_rarespot_trace_quality(result)

    assert quality["passed"] is False
    assert quality["signals"]["expected_configuration"] is False
    assert any("tile_size=512" in issue for issue in quality["issues"])


def test_evaluate_rarespot_trace_quality_rejects_report_only_rerun():
    result = {
        "prompt": {
            "run_id": "run-1",
            "goal": "Run RareSpot prairie dog detection on the uploaded image.",
            "status": "succeeded",
            "response_len": 2200,
            "response_preview": (
                "RareSpot inference completed with 2 burrow detections and 0 prairie dog "
                "detections. Download the annotated overlay and detections CSV."
            ),
            "artifact_count": 0,
            "artifacts": [],
            "linked_artifacts": [
                {"kind": "image", "path": "overlay.png", "download_ok": True},
                {"kind": "csv", "path": "detections.csv", "download_ok": True},
            ],
            "tool_names": ["rarespot_ecology_inference"],
            "rarespot_configurations": [_rarespot_default_config()],
            "idle_recovery_count": 0,
        },
        "followups": [
            {
                "run_id": "run-2",
                "goal": (
                    "Write a combined quantitative report across all RareSpot runs in "
                    "this chat. Include thresholds, counts, artifact links, and limitations."
                ),
                "status": "succeeded",
                "response_len": 1800,
                "response_preview": (
                    "Combined report compares threshold, count, confidence, limitations, "
                    "recommendations, and artifact IDs."
                ),
                "artifact_count": 0,
                "artifacts": [],
                "linked_artifacts": [],
                "tool_names": ["artifact_manifest", "rarespot_ecology_inference"],
                "idle_recovery_count": 0,
            },
        ],
    }

    quality = evaluate_rarespot_trace_quality(result)

    assert quality["passed"] is False
    assert "report-only RareSpot turn reran inference instead of using prior artifacts" in quality["issues"]
    assert quality["signals"]["report_turns_avoid_inference"] is False


def test_evaluate_rarespot_trace_quality_rejects_irrelevant_paper_tools():
    result = {
        "prompt": {
            "run_id": "run-1",
            "goal": "Run RareSpot prairie dog detection on the uploaded image.",
            "status": "succeeded",
            "response_len": 2200,
            "response_preview": (
                "RareSpot inference completed with 2 burrow detections and 0 prairie dog "
                "detections. Download the annotated overlay and detections CSV."
            ),
            "artifact_count": 0,
            "artifacts": [],
            "linked_artifacts": [
                {"kind": "image", "path": "overlay.png", "download_ok": True},
                {"kind": "csv", "path": "detections.csv", "download_ok": True},
            ],
            "tool_names": ["rarespot_ecology_inference", "read_paper_pages"],
            "rarespot_configurations": [_rarespot_default_config()],
            "idle_recovery_count": 0,
        }
    }

    quality = evaluate_rarespot_trace_quality(result)

    assert quality["passed"] is False
    assert "RareSpot trace used irrelevant paper tools" in quality["issues"]
    assert quality["signals"]["no_irrelevant_paper_tools"] is False


def test_summarize_run_trace_reports_autonomy_recovery_and_timing_metrics():
    summary = summarize_run_trace(
        run={
            "run_id": "run-1",
            "status": "succeeded",
            "response_text": "Recovered and completed.",
        },
        events=[
            {"event_kind": "run.accepted", "sequence": 1},
            {"event_kind": "run.started", "sequence": 2},
            {"event_kind": "tool_call.started", "sequence": 3},
            {
                "event_kind": "trace.message.delta",
                "sequence": 4,
                "payload": {
                    "reason": "model_stream_idle",
                    "recovery_index": 1,
                    "timeout_seconds": 240,
                },
            },
            {"event_kind": "artifact.created", "sequence": 5},
            {"event_kind": "run.completed", "sequence": 6},
        ],
        artifacts=[],
        accepted_ms=75.555,
        first_event_ms=10.04,
        first_backend_event_ms=52.34,
        first_delta_ms=None,
        first_tool_ms=612.91,
        first_artifact_ms=2030.81,
        first_recovery_ms=1800.12,
        terminal_ms=2400.44,
        download_checks={},
    )

    assert summary["accepted_ms"] == 75.6
    assert summary["first_event_ms"] == 10.0
    assert summary["first_backend_event_ms"] == 52.3
    assert summary["first_delta_ms"] is None
    assert summary["first_tool_ms"] == 612.9
    assert summary["first_artifact_ms"] == 2030.8
    assert summary["first_recovery_ms"] == 1800.1
    assert summary["idle_recovery_count"] == 1
    assert summary["idle_recoveries"] == [
        {"recovery_index": 1, "timeout_seconds": 240.0, "sequence": 4}
    ]
    assert summary["terminal_event_kind"] == "run.completed"
    assert summary["terminal_event_sequence"] == 6


def test_artifact_download_urls_quote_artifact_ids():
    assert artifact_download_urls("http://control.test/", ["artifact/one", "artifact two"]) == {
        "artifact/one": "http://control.test/v2/artifacts/artifact%2Fone/download",
        "artifact two": "http://control.test/v2/artifacts/artifact%20two/download",
    }


def test_artifact_ids_from_text_extracts_download_links_without_markdown_trailing_chars():
    assert _artifact_ids_from_text(
        "Overlay: [`download`](/v2/artifacts/artifact_run_123_rarespot_000004/download) "
        "CSV: /v2/artifacts/artifact_run_123_rarespot_000002/download]"
    ) == [
        "artifact_run_123_rarespot_000004",
        "artifact_run_123_rarespot_000002",
    ]


def test_artifact_ids_from_text_ignores_shortened_markdown_link_labels():
    assert _artifact_ids_from_text(
        "Overlay: "
        "[`/v2/artifacts/artifact_run_...rarespot_000004/download`]"
        "(/v2/artifacts/artifact_run_1234567890abcdef_rarespot_000004/download) "
        "Report: "
        "[`/v2/artifacts/artifact_run_…rarespot_000005/download`]"
        "(/v2/artifacts/artifact_run_1234567890abcdef_rarespot_000005/download)"
    ) == [
        "artifact_run_1234567890abcdef_rarespot_000004",
        "artifact_run_1234567890abcdef_rarespot_000005",
    ]


def test_artifact_ids_from_text_extracts_markdown_labels_with_download_href():
    assert _artifact_ids_from_text(
        "Overlay: "
        "[`/v2/artifacts/artifact_run_1234567890abcdef_rarespot_000004/download`](download) "
        "CSV: "
        "[`/v2/artifacts/artifact_run_1234567890abcdef_rarespot_000002/download`](download)"
    ) == [
        "artifact_run_1234567890abcdef_rarespot_000004",
        "artifact_run_1234567890abcdef_rarespot_000002",
    ]


def test_control_plane_client_paginates_run_events_with_cursor():
    client = ControlPlaneClient("http://control.test")
    calls: list[str] = []
    pages = {
        "/v2/runs/run-1/events?limit=2&after_sequence=0": {
            "events": [
                {"event_kind": "run.started", "sequence": 1},
                {"event_kind": "message.delta", "sequence": 2},
            ],
        },
        "/v2/runs/run-1/events?limit=2&after_sequence=2": {
            "events": [
                {"event_kind": "message.delta", "sequence": 3},
                {"event_kind": "run.completed", "sequence": 4},
            ],
        },
        "/v2/runs/run-1/events?limit=2&after_sequence=4": {
            "events": [],
        },
    }

    def fake_request(method, path, payload=None, *, headers=None):
        assert method == "GET"
        assert payload is None
        assert headers is None
        calls.append(path)
        return pages[path]

    client._request = fake_request  # type: ignore[method-assign]

    assert client.list_run_events("run-1", limit=2) == [
        {"event_kind": "run.started", "sequence": 1},
        {"event_kind": "message.delta", "sequence": 2},
        {"event_kind": "message.delta", "sequence": 3},
        {"event_kind": "run.completed", "sequence": 4},
    ]
    assert calls == [
        "/v2/runs/run-1/events?limit=2&after_sequence=0",
        "/v2/runs/run-1/events?limit=2&after_sequence=2",
        "/v2/runs/run-1/events?limit=2&after_sequence=4",
    ]


def test_control_plane_client_uploads_files_and_returns_file_ids(tmp_path):
    upload = tmp_path / "paper.pdf"
    upload.write_bytes(b"%PDF-1.4\nfixture")
    client = ControlPlaneClient("http://control.test")

    def fake_multipart(path, files):
        assert path == "/v2/uploads"
        assert files == [upload]
        return {"uploaded": [{"file_id": "file-paper"}]}

    client._request_multipart = fake_multipart  # type: ignore[method-assign]

    assert client.upload_files([upload]) == ["file-paper"]


def test_control_plane_client_create_run_sends_uploaded_file_ids():
    client = ControlPlaneClient("http://control.test")
    captured = {}

    def fake_request(method, path, payload=None, *, headers=None):
        captured.update(
            {
                "method": method,
                "path": path,
                "payload": payload,
                "headers": headers,
            }
        )
        return {"run_id": "run-1"}

    client._request = fake_request  # type: ignore[method-assign]

    run = client.create_run(
        thread_id="thread-1",
        goal="Read the uploaded paper.",
        messages=[{"role": "user", "content": "Read the uploaded paper."}],
        idempotency_key="key-1",
        file_ids=["file-paper"],
    )

    assert run["run_id"] == "run-1"
    assert captured["path"] == "/v2/threads/thread-1/runs"
    assert captured["payload"]["file_ids"] == ["file-paper"]
    assert captured["headers"] == {"Idempotency-Key": "key-1"}


def test_evaluate_trace_requirements_reports_actionable_failures():
    issues = evaluate_trace_requirements(
        {
            "prompt": {
                "run_id": "run-1",
                "status": "succeeded",
                "response_len": 12,
                "artifact_count": 1,
                "artifacts": [{"artifact_id": "artifact-1", "download_ok": False}],
            },
            "followup": {
                "run_id": "run-2",
                "status": "failed",
                "response_len": 0,
                "artifact_count": 0,
                "artifacts": [],
            },
        },
        min_artifacts=2,
        min_response_chars=50,
        require_downloads=True,
    )

    assert issues == [
        "prompt run run-1 response_len=12 below min_response_chars=50",
        "prompt run run-1 artifact_count=1 below min_artifacts=2",
        "prompt run run-1 artifact artifact-1 download verification failed",
        "followup run run-2 status=failed, want succeeded",
        "followup run run-2 response_len=0 below min_response_chars=50",
        "followup run run-2 artifact_count=0 below min_artifacts=2",
    ]


def test_evaluate_trace_requirements_counts_linked_artifacts():
    issues = evaluate_trace_requirements(
        {
            "prompt": {
                "run_id": "run-1",
                "status": "succeeded",
                "response_len": 200,
                "artifact_count": 0,
                "artifacts": [],
                "linked_artifacts": [
                    {"artifact_id": "overlay", "kind": "image", "download_ok": True},
                    {"artifact_id": "csv", "kind": "csv", "download_ok": True},
                ],
            },
        },
        min_artifacts=2,
        min_response_chars=50,
        require_downloads=True,
    )

    assert issues == []


def test_evaluate_trace_requirements_checks_every_followup_turn():
    issues = evaluate_trace_requirements(
        {
            "prompt": {
                "run_id": "run-1",
                "status": "succeeded",
                "response_len": 200,
                "artifact_count": 1,
                "artifacts": [{"artifact_id": "artifact-1", "download_ok": True}],
            },
            "followups": [
                {
                    "run_id": "run-2",
                    "status": "failed",
                    "response_len": 0,
                    "artifact_count": 0,
                    "artifacts": [],
                },
                {
                    "run_id": "run-3",
                    "status": "succeeded",
                    "response_len": 200,
                    "artifact_count": 1,
                    "artifacts": [{"artifact_id": "artifact-3", "download_ok": True}],
                },
            ],
            "followup": {
                "run_id": "run-3",
                "status": "succeeded",
                "response_len": 200,
                "artifact_count": 1,
                "artifacts": [{"artifact_id": "artifact-3", "download_ok": True}],
            },
        },
        min_artifacts=1,
        min_response_chars=50,
        require_downloads=True,
    )

    assert issues == [
        "followup[1] run run-2 status=failed, want succeeded",
        "followup[1] run run-2 response_len=0 below min_response_chars=50",
        "followup[1] run run-2 artifact_count=0 below min_artifacts=1",
    ]
