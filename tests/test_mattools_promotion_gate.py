from __future__ import annotations

import dataclasses
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mattools_promotion_gate.py"
SPEC = importlib.util.spec_from_file_location("mattools_promotion_gate", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def _write_synthetic_snapshot(root: Path) -> gate.BenchmarkSnapshot:
    root.mkdir(parents=True, exist_ok=True)
    (root / "LICENSE").write_text(
        "Apache License\nVersion 2.0, January 2004\n",
        encoding="utf-8",
    )
    src = root / "src"
    questions = src / "question_segments" / "pymatgen_analysis_defects"
    questions.mkdir(parents=True)
    (src / "result_analysis.py").write_text(
        "total_tasks_number = 2\ntotal_sub_tasks = 3\nref_sub_tasks_list = [1, 2]\n",
        encoding="utf-8",
    )
    for task_id, expected in (("public_alpha", 1), ("public_beta", 2)):
        task = questions / task_id
        task.mkdir()
        (task / "question.txt").write_text(
            f"PUBLIC QUESTION {task_id}\n",
            encoding="utf-8",
        )
        properties = {
            "JSON_File_Name": task_id,
            "properties": {
                f"SECRET_EXPECTED_{index}": {
                    "description": "must stay isolated",
                    "value": str(index),
                }
                for index in range(expected)
            },
        }
        (task / "properties.json").write_text(
            json.dumps(properties),
            encoding="utf-8",
        )
        (task / "new_unit_test.py").write_text(
            "SECRET_VERIFIER_SENTINEL = True\n",
            encoding="utf-8",
        )
    return gate.load_benchmark_snapshot(
        root,
        strict_official=False,
        expected_parent_count=2,
        expected_subtask_count=3,
        expected_task_order=("public_alpha", "public_beta"),
        expected_manifest_sha256=None,
    )


def _campaign_config() -> dict[str, object]:
    return {
        "campaign_id": "synthetic-campaign",
        "trial_count": 1,
        "selected_ordinals": [1, 2],
        "evaluation_mode": "diagnostic_full_trial",
        "validator_replays": 2,
        "control_plane_base_url": "http://127.0.0.1:8000",
        "model_id": "test-model",
        "provider_id": "test-provider",
        "runtime_image_digest": "sha256:" + "1" * 64,
        "runtime_pymatgen_version": "2026.5.4",
        "runtime_defects_version": "2025.1.18",
        "reasoning_mode": "deep",
        "budgets": {"max_runtime_seconds": 60, "max_tool_calls": 10},
        "license_attestation": {
            "accepted": True,
            "use_basis": "noncommercial",
            "use_purpose": "noncommercial unit-test fixture",
            "repository_license": "Apache-2.0",
            "dataset_card_license": "CC-BY-NC-4.0",
            "separate_license_evidence_sha256": None,
        },
        "ultra": {
            "commit": "a" * 40,
            "dirty": False,
            "skills_sha256": "b" * 64,
        },
    }


def _add_terminal_attempt(
    state: dict[str, object],
    output: Path,
    task: gate.BenchmarkTask,
    *,
    include_execute: bool,
) -> None:
    attempt_dir = output / "attempts" / f"{task.ordinal:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = attempt_dir / "prompt.txt"
    prompt_path.write_text(gate.build_ultra_prompt(task.question_text), encoding="utf-8")
    code_path = attempt_dir / "submission.py"
    source = "def solve_materials_task():\n    return {'value': 1}\n"
    code_path.write_text(source, encoding="utf-8")
    response_path = attempt_dir / "response.txt"
    response_path.write_text("done\n", encoding="utf-8")
    run_id = f"run_{task.ordinal}"
    thread_id = f"thread_{task.ordinal}"
    run_path = attempt_dir / "run.json"
    run_path.write_text(
        json.dumps({"run_id": run_id, "thread_id": thread_id, "status": "succeeded"}),
        encoding="utf-8",
    )
    events: list[dict[str, object]] = [
        {
            "event_kind": "run.token_usage",
            "payload": {
                "model": "test-model",
                "provider": "test-provider",
                "total_tokens": 10,
            },
        }
    ]
    if include_execute:
        events.extend(
            [
                {
                    "event_kind": "tool_call.started",
                    "payload": {"tool_name": "execute", "tool_call_id": "exec-1"},
                },
                {
                    "event_kind": "tool_call.completed",
                    "payload": {
                        "tool_name": "execute",
                        "tool_call_id": "exec-1",
                        "runtime_image_digest": "sha256:" + "1" * 64,
                    },
                },
            ]
        )
    events_path = attempt_dir / "events.json"
    events_path.write_text(json.dumps(events), encoding="utf-8")
    artifact_id = f"artifact_{task.ordinal}"
    artifact_records = [
        {
            "artifact_id": artifact_id,
            "path": f"/outputs/{gate.SOLUTION_FILENAME}",
            "sha256": gate.sha256_file(code_path),
        }
    ]
    artifacts_path = attempt_dir / "artifacts.json"
    artifacts_path.write_text(json.dumps(artifact_records), encoding="utf-8")
    trace = gate._trace_summary(events)
    config = state["config"]
    assert isinstance(config, dict)
    attempts = state["attempts"]
    assert isinstance(attempts, dict)
    attempts[gate.attempt_key(1, task.task_id)] = {
        "trial": 1,
        "task_id": task.task_id,
        "ordinal": task.ordinal,
        "subtask_count": task.subtask_count,
        "submission_status": "captured",
        "run_id": run_id,
        "thread_id": thread_id,
        "run_status": "succeeded",
        "prompt_path": str(prompt_path),
        "prompt_sha256": gate.sha256_file(prompt_path),
        "code_path": str(code_path),
        "code_sha256": gate.sha256_file(code_path),
        "function_name": "solve_materials_task",
        "source_kind": "artifact",
        "solution_artifact_id": artifact_id,
        "response_path": str(response_path),
        "response_sha256": gate.sha256_file(response_path),
        "run_record_path": str(run_path),
        "run_record_sha256": gate.sha256_file(run_path),
        "events_record_path": str(events_path),
        "events_record_sha256": gate.sha256_file(events_path),
        "artifacts_record_path": str(artifacts_path),
        "artifacts_record_sha256": gate.sha256_file(artifacts_path),
        "trace_summary": trace,
        "actual_runtime_provenance": gate._actual_runtime_provenance(
            trace,
            declared_model_id=str(config["model_id"]),
            declared_provider_id=str(config["provider_id"]),
        ),
    }


def test_snapshot_parser_checks_denominators_order_and_hashes(tmp_path: Path) -> None:
    snapshot = _write_synthetic_snapshot(tmp_path / "benchmark")

    assert [task.task_id for task in snapshot.tasks] == ["public_alpha", "public_beta"]
    assert [task.subtask_count for task in snapshot.tasks] == [1, 2]
    assert snapshot.runner_parent_count == 2
    assert snapshot.runner_subtask_count == 3
    assert len(snapshot.manifest_sha256) == 64
    assert snapshot.tasks[0].question_sha256 == gate.sha256_file(snapshot.tasks[0].question_path)


def test_snapshot_parser_rejects_reordered_runner_vector(tmp_path: Path) -> None:
    root = tmp_path / "benchmark"
    _write_synthetic_snapshot(root)
    (root / "src" / "result_analysis.py").write_text(
        "total_tasks_number = 2\ntotal_sub_tasks = 3\nref_sub_tasks_list = [2, 1]\n",
        encoding="utf-8",
    )

    with pytest.raises(gate.GateError, match="properties but the runner assigns"):
        gate.load_benchmark_snapshot(
            root,
            strict_official=False,
            expected_parent_count=2,
            expected_subtask_count=3,
            expected_task_order=("public_alpha", "public_beta"),
            expected_manifest_sha256=None,
        )


def test_strict_snapshot_file_enumerator_rejects_dirty_git_checkout(tmp_path: Path) -> None:
    root = tmp_path / "benchmark"
    _write_synthetic_snapshot(root)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.test"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)

    tracked = gate._clean_tracked_snapshot_files(root)
    assert "src/result_analysis.py" in tracked

    (root / "untracked-answer.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(gate.GateError, match="checkout is dirty"):
        gate._clean_tracked_snapshot_files(root)


def test_prompt_exposes_question_but_not_expected_values_or_verifier(tmp_path: Path) -> None:
    snapshot = _write_synthetic_snapshot(tmp_path / "benchmark")
    task = snapshot.tasks[0]

    prompt = gate.build_ultra_prompt(task.question_text)

    assert "PUBLIC QUESTION public_alpha" in prompt
    assert "SECRET_EXPECTED" not in prompt
    assert "SECRET_VERIFIER_SENTINEL" not in prompt
    assert task.task_id not in prompt.replace("PUBLIC QUESTION public_alpha", "")
    assert gate.SOLUTION_FILENAME in prompt
    assert gate.SOLUTION_FUNCTION_NAME in prompt
    assert "mattools" not in prompt.lower()
    assert "mattools" not in gate.SOLUTION_FILENAME.lower()
    assert "mattools" not in gate.SIDECAR_FILENAME.lower()
    assert "mattools" not in gate.SOLUTION_FUNCTION_NAME.lower()


def test_control_plane_payload_has_materials_selection_context_and_no_task_id() -> None:
    client = gate.ControlPlaneClient(
        "http://127.0.0.1:8000",
        headers={"X-Ultra-User-Id": "test"},
        timeout=1,
    )
    captured: dict[str, object] = {}

    def fake_request(
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        captured.update({"method": method, "path": path, "payload": payload, "headers": headers})
        return {"run": {"run_id": "run_test", "status": "queued"}}

    client._request = fake_request  # type: ignore[method-assign]
    client.create_run(
        thread_id="thread_test",
        prompt="ONLY PUBLIC QUESTION",
        idempotency_key="opaque-key",
        reasoning_mode="deep",
        budgets={"max_runtime_seconds": 60, "max_tool_calls": 10},
    )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["selection_context"] == {
        "suggested_domain": "materials",
    }
    assert payload["evaluation_profile"] == "materials_cleanroom_v1"
    assert "task_id" not in json.dumps(payload)
    model_visible = json.dumps(payload).lower()
    assert "mattools" not in model_visible
    assert "ordinal" not in model_visible
    assert "trial" not in model_visible
    assert "revision" not in model_visible
    assert "workflow_hint" not in payload

    title = gate.model_visible_thread_title("opaque-key")
    assert "mattools" not in title.lower()
    assert "trial" not in title.lower()
    assert not any(character.isdigit() for character in title.split()[:-1])


def test_official_scoring_semantics_and_thresholds() -> None:
    success = gate.classify_official_result("ok", 3)
    partial = gate.classify_official_result(["wrong value", 1, 3], 3)
    failure = gate.classify_official_result("FunctionError", 3)

    assert success["classification"] == "success"
    assert success["runnable"] is True
    assert success["scientific_pass"] == 3
    assert success["strict_scientific_pass"] == 0
    assert success["strict_verifiable_from_official_log"] is False
    assert partial["runnable"] is True
    assert partial["scientific_pass"] == 2
    assert failure["runnable"] is False
    assert failure["scientific_pass"] == 0
    assert gate.threshold_counts() == {
        "runnable_minimum": 118,
        "scientific_minimum": 249,
        "per_trial_runnable_minimum": 40,
        "per_trial_scientific_minimum": 83,
    }


def test_scoring_reproduces_upstream_loose_ok_membership() -> None:
    # The pinned upstream code checks `if "ok" in evaluation_result` rather
    # than exact equality. Preserve that behavior so our count matches its log.
    classified = gate.classify_official_result("broken", 2)
    assert classified["classification"] == "success"
    assert classified["scientific_pass"] == 2
    assert classified["strict_scientific_pass"] == 0
    assert classified["strict_verifiable_from_official_log"] is False


def test_partial_result_with_wrong_total_fails_closed() -> None:
    with pytest.raises(gate.GateError, match="expected a total of 3"):
        gate.classify_official_result(["wrong", 1, 99], 3)


def test_strict_shadow_rejects_pre_normalization_broken_string(tmp_path: Path) -> None:
    snapshot = _write_synthetic_snapshot(tmp_path / "benchmark")
    submission_sha = "a" * 64
    broken = json.dumps("broken")
    exact_ok = json.dumps("ok")
    code_stdout = "{'value': 1}"
    shadow_path = tmp_path / "strict-shadow.json"
    shadow_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "purpose": "pre-normalization strict shadow; not the published MatTools score",
                "submission_sha256": submission_sha,
                "result_count": 2,
                "results": [
                    {
                        "ordinal": 1,
                        "question_file_path": "public_alpha",
                        "runnable": True,
                        "code_stdout": code_stdout,
                        "code_stdout_sha256": gate.sha256_bytes(code_stdout.encode()),
                        "code_stdout_truncated": False,
                        "raw_verifier_output": broken,
                        "raw_verifier_output_sha256": gate.sha256_bytes(broken.encode()),
                        "raw_verifier_output_truncated": False,
                    },
                    {
                        "ordinal": 2,
                        "question_file_path": "public_beta",
                        "runnable": True,
                        "code_stdout": code_stdout,
                        "code_stdout_sha256": gate.sha256_bytes(code_stdout.encode()),
                        "code_stdout_truncated": False,
                        "raw_verifier_output": exact_ok,
                        "raw_verifier_output_sha256": gate.sha256_bytes(exact_ok.encode()),
                        "raw_verifier_output_truncated": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    parsed = gate.parse_strict_shadow_report(
        shadow_path,
        snapshot.tasks,
        expected_submission_sha256=submission_sha,
    )

    assert parsed["results"][0]["classification"] == "strict_failure"
    assert parsed["results"][0]["scientific_pass"] == 0
    assert parsed["results"][1]["classification"] == "strict_success"
    assert parsed["results"][1]["scientific_pass"] == 2
    assert parsed["runnable"] == 2
    assert parsed["scientific_pass"] == 2


def test_official_log_parser_separates_runnable_and_scientific_success(
    tmp_path: Path,
) -> None:
    snapshot = _write_synthetic_snapshot(tmp_path / "benchmark")
    log = tmp_path / "evaluation.log"
    log.write_text(
        "2026-01-01 - INFO - Evaluation result for function | solve_a | ok\n"
        "2026-01-01 - INFO - Evaluation result for function | solve_b | "
        "['wrong beta', 1, 2]\n"
        "2026-01-01 - INFO - Total tasks: 2, Correct: 1, Partially Correct: 1, "
        "Incorrect: 0, Function Errors: 0, Result Errors: 1, Successes: 1, "
        "Accuracy: 50.00%, Function Runnable Rate: 100.00%\n"
        "2026-01-01 - INFO - Total sub-tasks: 3, Correct: 2, Incorrect: 1, "
        "Accuracy: 66.67%\n",
        encoding="utf-8",
    )

    parsed = gate.parse_official_evaluation_log(log, snapshot.tasks)

    assert parsed["runnable"] == 2
    assert parsed["runnable_denominator"] == 2
    assert parsed["scientific_pass"] == 2
    assert parsed["scientific_denominator"] == 3
    assert [item["classification"] for item in parsed["results"]] == [
        "success",
        "partial",
    ]


def test_checkpoint_audit_recomputes_forged_evaluation_from_raw_logs(
    tmp_path: Path,
) -> None:
    snapshot = _write_synthetic_snapshot(tmp_path / "benchmark")
    output = tmp_path / "campaign"
    state: dict[str, object] = {
        "schema_version": gate.SCHEMA_VERSION,
        "campaign_id": "synthetic",
        "benchmark": snapshot.provenance_record(),
        "config": _campaign_config(),
        "attempts": {},
        "evaluations": {},
    }
    for task in snapshot.tasks:
        _add_terminal_attempt(state, output, task, include_execute=True)
    input_path = output / "evaluation" / "function_generation_results.jsonl"
    input_path.parent.mkdir(parents=True)
    input_path.write_text(
        gate.official_jsonl_content(snapshot=snapshot, checkpoint_data=state, trial=1),
        encoding="utf-8",
    )
    log_path = input_path.parent / "evaluation_logs_test.log"
    log_path.write_text(
        "2026-01-01 - INFO - Evaluation result for function | solve_a | ok\n"
        "2026-01-01 - INFO - Evaluation result for function | solve_b | "
        "['wrong beta', 1, 2]\n"
        "2026-01-01 - INFO - Total tasks: 2, Correct: 1, Partially Correct: 1, "
        "Incorrect: 0, Function Errors: 0, Result Errors: 1, Successes: 1, "
        "Accuracy: 50.00%, Function Runnable Rate: 100.00%\n"
        "2026-01-01 - INFO - Total sub-tasks: 3, Correct: 2, Incorrect: 1, "
        "Accuracy: 66.67%\n",
        encoding="utf-8",
    )
    stdout_path = input_path.parent / "runner.stdout.log"
    stderr_path = input_path.parent / "runner.stderr.log"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    shadow_path = input_path.parent / "strict-shadow-results.json"
    shadow_results = []
    for task in snapshot.tasks:
        raw = json.dumps("ok")
        code_stdout = "{'value': 1}"
        shadow_results.append(
            {
                "ordinal": task.ordinal,
                "question_file_path": task.task_id,
                "runnable": True,
                "code_stdout": code_stdout,
                "code_stdout_sha256": gate.sha256_bytes(code_stdout.encode()),
                "code_stdout_truncated": False,
                "raw_verifier_output": raw,
                "raw_verifier_output_sha256": gate.sha256_bytes(raw.encode()),
                "raw_verifier_output_truncated": False,
            }
        )
    shadow_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "purpose": "pre-normalization strict shadow; not the published MatTools score",
                "submission_sha256": gate.sha256_file(input_path),
                "result_count": len(shadow_results),
                "results": shadow_results,
            }
        ),
        encoding="utf-8",
    )
    shadow_stdout = input_path.parent / "strict-shadow.stdout.log"
    shadow_stderr = input_path.parent / "strict-shadow.stderr.log"
    shadow_stdout.write_text("", encoding="utf-8")
    shadow_stderr.write_text("", encoding="utf-8")
    evaluations = state["evaluations"]
    assert isinstance(evaluations, dict)
    replay = {
        "replay": 1,
        "evaluator_image_id_before": None,
        "evaluator_image_id_after": None,
        "runnable": 0,
        "scientific_pass": 0,
        "full_question_success": 0,
        "results": [],
        "input_jsonl_path": str(input_path),
        "input_jsonl_sha256": gate.sha256_file(input_path),
        "log_path": str(log_path),
        "log_sha256": gate.sha256_file(log_path),
        "runner_stdout_path": str(stdout_path),
        "runner_stdout_sha256": gate.sha256_file(stdout_path),
        "runner_stderr_path": str(stderr_path),
        "runner_stderr_sha256": gate.sha256_file(stderr_path),
        "strict_shadow": {
            "path": str(shadow_path),
            "sha256": gate.sha256_file(shadow_path),
            "scientific_pass": 999,
            "results": [],
        },
        "strict_shadow_stdout_path": str(shadow_stdout),
        "strict_shadow_stdout_sha256": gate.sha256_file(shadow_stdout),
        "strict_shadow_stderr_path": str(shadow_stderr),
        "strict_shadow_stderr_sha256": gate.sha256_file(shadow_stderr),
    }
    checkpoint = gate.CampaignCheckpoint(output / "state.json", state)
    replay = gate.seal_terminal_replay(checkpoint, "trial-01", replay)
    evaluations["trial-01"] = {
        "status": "complete",
        "evaluator_environment": {},
        "runner": {},
        "sandbox_policy_attestation": {},
        "replays": [replay],
    }

    audited, audit = gate.revalidate_checkpoint_evidence(
        snapshot,
        state,
        campaign_root=output,
    )
    replay = audited["evaluations"]["trial-01"]["replays"][0]

    assert replay["runnable"] == 2
    assert replay["scientific_pass"] == 2
    assert replay["strict_shadow"]["scientific_pass"] == 3
    assert audit["valid"] is False
    assert any("forged or stale scientific_pass" in issue for issue in audit["issues"])
    assert any("forged or stale strict shadow" in issue for issue in audit["issues"])


def test_replay_comparison_uses_scientific_classification() -> None:
    first = {
        "results": [
            {
                "task_id": "a",
                "classification": "partial",
                "runnable": True,
                "scientific_pass": 1,
            }
        ]
    }
    same = json.loads(json.dumps(first))
    changed = json.loads(json.dumps(first))
    changed["results"][0]["scientific_pass"] = 0

    assert gate.replay_classifications_match([first, same]) is True
    assert gate.replay_classifications_match([first, changed]) is False
    assert gate.replay_classifications_match([first]) is False


def test_attempt_scoring_keeps_semantic_runnable_separate_from_published_runner(
    tmp_path: Path,
) -> None:
    snapshot = _write_synthetic_snapshot(tmp_path / "benchmark")

    def replay(number: int) -> dict[str, object]:
        return {
            "replay": number,
            "terminal_record_sha256": f"{number}" * 64,
            "results": [
                {
                    "task_id": "public_alpha",
                    "classification": "function_error",
                    "runnable": False,
                    "scientific_pass": 0,
                    "scientific_fail": 1,
                },
                {
                    "task_id": "public_beta",
                    "classification": "success",
                    "runnable": True,
                    "scientific_pass": 2,
                    "scientific_fail": 0,
                },
            ],
            "strict_shadow": {
                "status": "complete",
                "results": [
                    {
                        "task_id": "public_alpha",
                        "runnable": True,
                        "classification": "strict_success",
                        "scientific_pass": 1,
                        "scientific_fail": 0,
                        "exact_ok": True,
                        "raw_verifier_output_sha256": "a" * 64,
                    },
                    {
                        "task_id": "public_beta",
                        "runnable": True,
                        "classification": "strict_success",
                        "scientific_pass": 2,
                        "scientific_fail": 0,
                        "exact_ok": True,
                        "raw_verifier_output_sha256": "b" * 64,
                    },
                ],
            },
        }

    cleanroom_binding = {
        "valid": True,
        "user_identity_independently_bound": True,
    }
    state = {
        "attempts": {
            gate.attempt_key(1, "public_alpha"): {
                "cleanroom_binding": cleanroom_binding,
            }
        },
        "evaluations": {
            "trial-01": {
                "status": "complete",
                "reproducible": True,
                "replays": [replay(1), replay(2)],
            }
        },
    }

    trial = gate._trial_report(1, snapshot, state)

    assert trial["scoring_evidence_complete"] is True
    assert trial["runnable"] == 2
    assert trial["published_runner_runnable"] == 1
    assert trial["attempts"][0]["cleanroom_binding"] == cleanroom_binding
    assert trial["attempts"][1]["cleanroom_binding"] is None
    alpha = trial["attempts"][0]["scoring_evidence"]
    assert alpha["primary"]["strict_shadow"] == {
        "semantic_runnable": True,
        "strict_scientific_classification": "strict_success",
        "strict_scientific_pass": 1,
        "strict_scientific_fail": 0,
        "strict_exact_ok": True,
        "raw_verifier_output_sha256": "a" * 64,
    }
    assert alpha["primary"]["published_upstream"]["runnable"] is False


def test_terminal_replay_seal_rejects_changed_replay_and_changed_seal(tmp_path: Path) -> None:
    snapshot = _write_synthetic_snapshot(tmp_path / "benchmark")
    checkpoint = gate.CampaignCheckpoint.open_or_create(
        tmp_path / "campaign" / "state.json",
        snapshot=snapshot,
        config=_campaign_config(),
    )
    replay = gate.seal_terminal_replay(
        checkpoint,
        "trial-01",
        {
            "replay": 1,
            "input_jsonl_sha256": "a" * 64,
            "strict_shadow": {"sha256": "b" * 64, "scientific_pass": 2},
        },
    )
    gate.verify_terminal_replay_seal(checkpoint.path.parent, "trial-01", replay)

    changed_replay = json.loads(json.dumps(replay))
    changed_replay["strict_shadow"]["scientific_pass"] = 999
    with pytest.raises(gate.GateError, match="differs from its terminal seal"):
        gate.verify_terminal_replay_seal(
            checkpoint.path.parent,
            "trial-01",
            changed_replay,
        )

    seal_path = checkpoint.path.parent / replay["terminal_record_path"]
    seal_path.write_text('{"forged":true}\n', encoding="utf-8")
    with pytest.raises(gate.GateError, match="SHA-256 mismatch"):
        gate.verify_terminal_replay_seal(checkpoint.path.parent, "trial-01", replay)


def test_failed_replay_checkpoint_cannot_cherry_pick_a_later_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _write_synthetic_snapshot(tmp_path / "benchmark")
    checkpoint = gate.CampaignCheckpoint.open_or_create(
        tmp_path / "campaign" / "state.json",
        snapshot=snapshot,
        config=_campaign_config(),
    )
    failed = gate.seal_terminal_replay(
        checkpoint,
        "trial-01",
        {
            "replay": 1,
            "status": "timeout",
            "reason": "outer timeout",
            "failure_artifacts": [],
            "failure_artifact_manifest_sha256": gate.sha256_bytes(gate.canonical_json_bytes([])),
        },
    )
    checkpoint.set_evaluation(
        "trial-01",
        {"replays": [], "failed_replays": [failed]},
    )
    image_inspected = False

    def inspect_image(**_kwargs: object) -> dict[str, object]:
        nonlocal image_inspected
        image_inspected = True
        return {}

    monkeypatch.setattr(gate, "inspect_evaluator_image", inspect_image)

    with pytest.raises(gate.GateError, match="start a fresh campaign"):
        gate.run_official_trial_evaluation(
            snapshot=snapshot,
            checkpoint=checkpoint,
            output_dir=checkpoint.path.parent,
            trial=1,
            validator_command=("python",),
            replay_count=2,
            evaluator_timeout=1,
            runtime_image_digest="sha256:" + "1" * 64,
            expected_image_id="sha256:" + "2" * 64,
            evaluator_environment_lock={},
            sandbox_attestation_path=None,
            sandbox_attestation_signature_path=None,
            sandbox_attestation_public_key_path=None,
        )

    assert image_inspected is False


def test_checkpoint_resume_preserves_run_and_terminal_failure(tmp_path: Path) -> None:
    snapshot = _write_synthetic_snapshot(tmp_path / "benchmark")
    state_path = tmp_path / "campaign" / "state.json"
    config = _campaign_config()
    checkpoint = gate.CampaignCheckpoint.open_or_create(
        state_path,
        snapshot=snapshot,
        config=config,
    )
    checkpoint.update_attempt(
        "trial-01/public_alpha",
        {"run_id": "run_original", "submission_status": "terminal_failure"},
    )

    resumed = gate.CampaignCheckpoint.open_or_create(
        state_path,
        snapshot=snapshot,
        config=config,
    )
    terminal_attempt = resumed.get_attempt("trial-01/public_alpha")
    assert terminal_attempt["run_id"] == "run_original"
    terminal_path = state_path.parent / terminal_attempt["terminal_record_path"]
    assert terminal_path.is_file()
    assert gate.sha256_file(terminal_path) == terminal_attempt["terminal_record_sha256"]
    terminal_payload = json.loads(terminal_path.read_text(encoding="utf-8"))
    assert terminal_payload["attempt_key"] == "trial-01/public_alpha"
    assert terminal_payload["attempt"]["run_id"] == "run_original"
    with pytest.raises(gate.GateError, match="write-once evidence"):
        gate.write_once_bytes(terminal_path, b"different")
    with pytest.raises(gate.GateError, match="refusing to replace terminal attempt"):
        resumed.update_attempt(
            "trial-01/public_alpha",
            {"run_id": "run_replacement", "submission_status": "captured"},
        )


def test_checkpoint_resume_rejects_benchmark_drift(tmp_path: Path) -> None:
    snapshot = _write_synthetic_snapshot(tmp_path / "benchmark")
    state_path = tmp_path / "campaign" / "state.json"
    config = _campaign_config()
    gate.CampaignCheckpoint.open_or_create(
        state_path,
        snapshot=snapshot,
        config=config,
    )
    drifted = dataclasses.replace(snapshot, manifest_sha256="f" * 64)

    with pytest.raises(gate.GateError, match="benchmark digest differs"):
        gate.CampaignCheckpoint.open_or_create(
            state_path,
            snapshot=drifted,
            config=config,
        )


def test_checkpoint_resume_rejects_validator_replay_count_drift(tmp_path: Path) -> None:
    snapshot = _write_synthetic_snapshot(tmp_path / "benchmark")
    state_path = tmp_path / "campaign" / "state.json"
    state_path.parent.mkdir(parents=True)
    config = _campaign_config()
    gate.CampaignCheckpoint.open_or_create(state_path, snapshot=snapshot, config=config)
    changed = dict(config)
    changed["validator_replays"] = 3

    with pytest.raises(gate.GateError, match="validator_replays"):
        gate.CampaignCheckpoint.open_or_create(
            state_path,
            snapshot=snapshot,
            config=changed,
        )


def test_jsonl_uses_official_order_and_exact_captured_code(tmp_path: Path) -> None:
    snapshot = _write_synthetic_snapshot(tmp_path / "benchmark")
    checkpoint_data: dict[str, object] = {"attempts": {}}
    attempts = checkpoint_data["attempts"]
    assert isinstance(attempts, dict)
    for task in reversed(snapshot.tasks):
        source = f"def solve_{task.ordinal}():\n    return {{'value': {task.ordinal}}}\n"
        code_path = tmp_path / f"submission-{task.ordinal}.py"
        code_path.write_text(source, encoding="utf-8")
        attempts[gate.attempt_key(1, task.task_id)] = {
            "submission_status": "captured",
            "code_path": str(code_path),
            "code_sha256": gate.sha256_file(code_path),
            "function_name": f"solve_{task.ordinal}",
        }
    destination = tmp_path / "function_generation_results.jsonl"

    gate.prepare_official_jsonl(
        snapshot=snapshot,
        checkpoint_data=checkpoint_data,
        trial=1,
        destination=destination,
    )

    records = [json.loads(line) for line in destination.read_text().splitlines()]
    assert [record["question_file_path"] for record in records] == [
        "public_alpha",
        "public_beta",
    ]
    assert records[0]["function"] == "def solve_1():\n    return {'value': 1}\n"


def test_incomplete_report_has_no_comparable_aggregate_score(tmp_path: Path) -> None:
    snapshot = _write_synthetic_snapshot(tmp_path / "benchmark")
    state = {
        "campaign_id": "synthetic",
        "config": _campaign_config(),
        "attempts": {},
        "evaluations": {},
    }

    report = gate.build_report(snapshot, state)

    assert report["counts"]["runnable_denominator"] == 147
    assert report["counts"]["scientific_denominator"] == 414
    assert report["counts"]["runnable"] is None
    assert report["counts"]["scientific_pass"] is None
    assert report["rates"] == {
        "function_runnable": None,
        "published_runner_function_runnable": None,
        "task_success": None,
        "strict_task_success": None,
    }
    assert report["hard_gates"]["actual_model_provider_provenance"] is False
    assert report["promotion"]["passed"] is False


def test_report_writer_emits_json_markdown_and_hash_manifest(tmp_path: Path) -> None:
    snapshot = _write_synthetic_snapshot(tmp_path / "benchmark")
    output = tmp_path / "campaign"
    checkpoint = gate.CampaignCheckpoint.open_or_create(
        output / "state.json",
        snapshot=snapshot,
        config=_campaign_config(),
    )

    report = gate.write_reports(output, snapshot, checkpoint)

    assert report["rates"]["function_runnable"] is None
    assert (output / "results.json").is_file()
    assert "Comparable aggregate FRR: **not available**" in (output / "results.md").read_text(
        encoding="utf-8"
    )
    manifest = json.loads((output / "report_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == gate.REPORT_MANIFEST_SCHEMA_VERSION
    assert manifest["manifest_kind"] == gate.REPORT_MANIFEST_KIND
    assert manifest["results_json"]["sha256"] == gate.sha256_file(output / "results.json")
    assert manifest["results_markdown"]["sha256"] == gate.sha256_file(output / "results.md")
    first_bundle = {
        name: (output / name).read_bytes()
        for name in ("results.json", "results.md", "report_manifest.json")
    }
    gate.write_reports(output, snapshot, checkpoint)
    assert first_bundle == {
        name: (output / name).read_bytes()
        for name in ("results.json", "results.md", "report_manifest.json")
    }
    verification = gate.revalidate_report_bundle(snapshot, output / "report_manifest.json")
    assert verification["revalidation_kind"] == gate.REPORT_REVALIDATION_KIND
    assert verification["bundle_exact"] is True
    assert verification["checkpoint_exact"] is True
    assert verification["task_execution_performed"] is False
    assert verification["valid"] is False  # the intentionally empty campaign is not evidence-valid


def test_report_revalidator_rejects_forged_aggregate_even_when_manifest_is_rehashed(
    tmp_path: Path,
) -> None:
    snapshot = _write_synthetic_snapshot(tmp_path / "benchmark")
    output = tmp_path / "campaign"
    checkpoint = gate.CampaignCheckpoint.open_or_create(
        output / "state.json",
        snapshot=snapshot,
        config=_campaign_config(),
    )
    gate.write_reports(output, snapshot, checkpoint)
    results_path = output / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results["counts"]["strict_scientific_pass"] = 999
    results["counts"]["strict_scientific_pass_observed_in_completed_trials"] = 999
    results["rates"]["strict_task_success"] = 1.0
    gate.atomic_write_json(results_path, results)
    manifest_path = output / "report_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["results_json"]["sha256"] = gate.sha256_file(results_path)
    gate.atomic_write_json(manifest_path, manifest)

    verification = gate.revalidate_report_bundle(snapshot, manifest_path)

    assert verification["manifest_integrity_valid"] is True
    assert verification["results_json_exact"] is False
    assert verification["manifest_exact"] is False
    assert verification["valid"] is False


def test_report_revalidator_rejects_arbitrary_markdown_with_matching_manifest_hash(
    tmp_path: Path,
) -> None:
    snapshot = _write_synthetic_snapshot(tmp_path / "benchmark")
    output = tmp_path / "campaign"
    checkpoint = gate.CampaignCheckpoint.open_or_create(
        output / "state.json",
        snapshot=snapshot,
        config=_campaign_config(),
    )
    gate.write_reports(output, snapshot, checkpoint)
    markdown_path = output / "results.md"
    markdown_path.write_text("# PASS\n\nEverything passed.\n", encoding="utf-8")
    manifest_path = output / "report_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["results_markdown"]["sha256"] = gate.sha256_file(markdown_path)
    gate.atomic_write_json(manifest_path, manifest)

    verification = gate.revalidate_report_bundle(snapshot, manifest_path)

    assert verification["manifest_integrity_valid"] is True
    assert verification["results_markdown_exact"] is False
    assert verification["manifest_exact"] is False
    assert verification["valid"] is False


def test_report_revalidator_rejects_dummy_checkpoint_with_matching_manifest_hash(
    tmp_path: Path,
) -> None:
    snapshot = _write_synthetic_snapshot(tmp_path / "benchmark")
    output = tmp_path / "campaign"
    checkpoint = gate.CampaignCheckpoint.open_or_create(
        output / "state.json",
        snapshot=snapshot,
        config=_campaign_config(),
    )
    gate.write_reports(output, snapshot, checkpoint)
    checkpoint.path.write_text('{"schema_version":"1"}\n', encoding="utf-8")
    manifest_path = output / "report_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint"]["sha256"] = gate.sha256_file(checkpoint.path)
    gate.atomic_write_json(manifest_path, manifest)

    verification = gate.revalidate_report_bundle(snapshot, manifest_path)

    assert verification["manifest_integrity_valid"] is True
    assert verification["checkpoint_evidence_valid"] is False
    assert verification["bundle_exact"] is False
    assert verification["valid"] is False


def test_report_recomputes_trace_booleans_and_rejects_tampered_code(tmp_path: Path) -> None:
    snapshot = _write_synthetic_snapshot(tmp_path / "benchmark")
    output = tmp_path / "campaign"
    state: dict[str, object] = {
        "schema_version": gate.SCHEMA_VERSION,
        "campaign_id": "synthetic",
        "benchmark": snapshot.provenance_record(),
        "config": _campaign_config(),
        "attempts": {},
        "evaluations": {},
    }
    _add_terminal_attempt(state, output, snapshot.tasks[0], include_execute=False)
    _add_terminal_attempt(state, output, snapshot.tasks[1], include_execute=True)
    attempts = state["attempts"]
    assert isinstance(attempts, dict)
    first = attempts[gate.attempt_key(1, snapshot.tasks[0].task_id)]
    assert isinstance(first, dict)
    claimed_trace = first["trace_summary"]
    assert isinstance(claimed_trace, dict)
    claimed_trace["production_execute_tool_evidence"] = True

    report = gate.build_report(snapshot, state, campaign_root=output)

    assert report["hard_gates"]["production_execute_tool_evidence"] is False
    assert report["hard_gates"]["checkpoint_evidence_integrity"] is False
    assert any(
        "forged or stale trace summary" in issue
        for issue in report["checkpoint_evidence_audit"]["issues"]
    )

    code_path = Path(str(first["code_path"]))
    code_path.write_text("def forged():\n    return {}\n", encoding="utf-8")
    tampered = gate.build_report(snapshot, state, campaign_root=output)
    assert any(
        "code: SHA-256 mismatch" in issue
        for issue in tampered["checkpoint_evidence_audit"]["issues"]
    )


def test_code_capture_helpers_require_unambiguous_function() -> None:
    source = "def helper(x):\n    return x\n\ndef solve_materials_task():\n    return {'x': 1}\n"
    assert gate.select_submission_function(source) == "solve_materials_task"
    assert gate.extract_fenced_python(f"```python\n{source}```") == source
    assert gate.extract_fenced_python(f"```python\n{source}```\n```py\n{source}```") is None


def test_base_url_rejects_embedded_credentials() -> None:
    with pytest.raises(gate.GateError, match="credentials are forbidden"):
        gate.validate_base_url("https://user:secret@example.test")


def test_evaluator_refuses_production_runtime_image(monkeypatch: pytest.MonkeyPatch) -> None:
    digest = "sha256:" + "a" * 64
    inspect_result = subprocess.CompletedProcess(
        args=["docker"],
        returncode=0,
        stdout=json.dumps([{"Id": digest, "RepoDigests": []}]),
        stderr="",
    )
    monkeypatch.setattr(gate, "_run_capture", lambda *args, **kwargs: inspect_result)

    with pytest.raises(gate.GateError, match="refusing to score in the production"):
        gate.inspect_evaluator_image(runtime_image_digest=digest)


def _reviewed_evaluator_lock() -> dict[str, object]:
    lock_path = (
        Path(__file__).resolve().parents[1]
        / "deploy/docker/mattools-evaluator-linux-arm64-lock.json"
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock.update(
        {
            "approved_from_git_head": True,
            "path": str(gate.EVALUATOR_DEFAULT_LOCK),
            "sha256": gate.sha256_file(lock_path),
        }
    )
    return lock


def _reviewed_evaluator_labels(lock: dict[str, object]) -> dict[str, str]:
    build = lock["build"]
    upstream = lock["upstream"]
    platform_record = lock["platform"]
    assert isinstance(build, dict)
    assert isinstance(upstream, dict)
    assert isinstance(platform_record, dict)
    return {
        "io.ultra.mattools.adapted-requirements-sha256": str(build["adapted_requirements_sha256"]),
        "io.ultra.mattools.base-image": str(build["base_image"]),
        "io.ultra.mattools.environment-kind": str(lock["environment_kind"]),
        "io.ultra.mattools.official-artifact": "false",
        "io.ultra.mattools.snapshot-manifest-sha256": str(upstream["manifest_sha256"]),
        "io.ultra.mattools.safe-parser-sha256": str(build["safe_parser_sha256"]),
        "io.ultra.mattools.runner-wrapper-sha256": str(build["runner_wrapper_sha256"]),
        "io.ultra.mattools.semantic-repairs-sha256": str(build["semantic_repairs_sha256"]),
        "io.ultra.mattools.strict-shadow-sha256": str(build["strict_shadow_sha256"]),
        "io.ultra.mattools.supplemental-requirements-sha256": str(
            build["supplemental_requirements_sha256"]
        ),
        "io.ultra.mattools.target-platform": str(platform_record["docker"]),
        "io.ultra.mattools.tool-source-manifest-sha256": str(build["tool_source_manifest_sha256"]),
        "io.ultra.mattools.candidate-fixture-file-count": str(
            build["candidate_fixture_file_count"]
        ),
        "io.ultra.mattools.candidate-fixture-manifest-sha256": str(
            build["candidate_fixture_manifest_sha256"]
        ),
        "io.ultra.mattools.candidate-visible-source-policy": str(
            build["candidate_visible_source_policy"]
        ),
        "io.ultra.mattools.upstream-requirements-sha256": str(upstream["requirements_sha256"]),
        "org.opencontainers.image.revision": str(upstream["revision"]),
    }


def _reviewed_evaluator_probe(lock: dict[str, object]) -> dict[str, object]:
    build = lock["build"]
    upstream = lock["upstream"]
    assert isinstance(build, dict)
    assert isinstance(upstream, dict)
    return {
        "python": lock["python_version"],
        "platform": lock["platform"],
        "packages": gate.OFFICIAL_PACKAGE_VERSIONS,
        "resolved_packages": lock["packages"],
        "candidate_fixture_file_count": build["candidate_fixture_file_count"],
        "candidate_fixture_manifest_sha256": build["candidate_fixture_manifest_sha256"],
        "candidate_visible_non_fixture_paths": [],
        "candidate_visible_executable_source_paths": [],
        "candidate_visible_dependency_test_paths": {
            "pymatgen": [],
            "pymatgen-analysis-defects": [],
        },
        "upstream_requirements_sha256": upstream["requirements_sha256"],
        "adapted_requirements_sha256": build["adapted_requirements_sha256"],
        "supplemental_requirements_sha256": build["supplemental_requirements_sha256"],
        "task_execution_performed": False,
    }


def test_reviewed_environment_lock_binds_tracked_build_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = (
        Path(__file__).resolve().parents[1]
        / "deploy/docker/mattools-evaluator-linux-arm64-lock.json"
    )
    monkeypatch.setattr(gate, "_is_git_tracked_unchanged", lambda *args: True)

    lock = gate.load_approved_evaluator_environment_lock(lock_path)

    assert lock["environment_kind"] == "reviewed-reconstruction-variant"
    assert lock["official_artifact"] is False
    assert lock["approved_from_git_head"] is True
    assert len(lock["packages"]) == 290
    assert lock["build"]["tool_source_file_count"] == 2756
    assert lock["build"]["candidate_fixture_file_count"] == 141
    assert lock["build"]["candidate_visible_source_policy"] == "input-fixtures-only"
    assert lock["build"]["builder_path"] == gate.EVALUATOR_BUILDER.as_posix()
    assert lock["build"]["builder_sha256"] == gate.sha256_file(
        Path(__file__).resolve().parents[1] / gate.EVALUATOR_BUILDER
    )
    assert lock["package_map_sha256"] == gate.sha256_bytes(
        gate.canonical_json_bytes(lock["packages"])
    )


def test_evaluator_requires_exact_upstream_scientific_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator_digest = "sha256:" + "a" * 64
    runtime_digest = "sha256:" + "b" * 64
    responses = iter(
        (
            subprocess.CompletedProcess(
                args=["docker", "inspect"],
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            "Id": evaluator_digest,
                            "RepoDigests": [],
                            "Architecture": "arm64",
                            "Os": "linux",
                            "Config": {"Labels": {}},
                        }
                    ]
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["docker", "run"],
                returncode=0,
                stdout=json.dumps(
                    {
                        "python": "3.11.8",
                        "packages": {
                            "pymatgen": "2026.5.4",
                            "pymatgen-analysis-defects": "2025.1.18",
                        },
                    }
                ),
                stderr="",
            ),
        )
    )
    monkeypatch.setattr(gate, "_run_capture", lambda *args, **kwargs: next(responses))

    with pytest.raises(gate.GateError, match="differ from the official MatTools stack"):
        gate.inspect_evaluator_image(runtime_image_digest=runtime_digest)


def test_evaluator_records_exact_independent_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator_digest = "sha256:" + "a" * 64
    runtime_digest = "sha256:" + "b" * 64
    lock = _reviewed_evaluator_lock()
    responses = iter(
        (
            subprocess.CompletedProcess(
                args=["docker", "inspect"],
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            "Id": evaluator_digest,
                            "RepoDigests": [],
                            "Architecture": "arm64",
                            "Os": "linux",
                            "Config": {"Labels": _reviewed_evaluator_labels(lock)},
                        }
                    ]
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["docker", "run"],
                returncode=0,
                stdout=json.dumps(_reviewed_evaluator_probe(lock)),
                stderr="",
            ),
        )
    )
    monkeypatch.setattr(gate, "_run_capture", lambda *args, **kwargs: next(responses))

    environment = gate.inspect_evaluator_image(
        runtime_image_digest=runtime_digest,
        environment_lock=lock,
    )

    assert environment["comparable"] is True
    assert environment["official_artifact"] is False
    assert environment["environment_kind"] == "reviewed-reconstruction-variant"
    assert environment["labels_match_approved_lock"] is True
    assert environment["embedded_inputs_match_approved_lock"] is True
    assert environment["independent_from_production_runtime"] is True
    assert environment["image_id"] == evaluator_digest
    assert environment["production_runtime_image_digest"] == runtime_digest


def test_evaluator_rejects_missing_reconstruction_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator_digest = "sha256:" + "a" * 64
    runtime_digest = "sha256:" + "b" * 64
    lock = _reviewed_evaluator_lock()
    labels = _reviewed_evaluator_labels(lock)
    labels.pop("io.ultra.mattools.base-image")
    responses = iter(
        (
            subprocess.CompletedProcess(
                args=["docker", "inspect"],
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            "Id": evaluator_digest,
                            "RepoDigests": [],
                            "Architecture": "arm64",
                            "Os": "linux",
                            "Config": {"Labels": labels},
                        }
                    ]
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["docker", "run"],
                returncode=0,
                stdout=json.dumps(_reviewed_evaluator_probe(lock)),
                stderr="",
            ),
        )
    )
    monkeypatch.setattr(gate, "_run_capture", lambda *args, **kwargs: next(responses))

    with pytest.raises(gate.GateError, match="image, source, platform, labels"):
        gate.inspect_evaluator_image(
            runtime_image_digest=runtime_digest,
            environment_lock=lock,
        )


def test_runtime_model_provider_are_observed_not_trusted_from_cli() -> None:
    trace = gate._trace_summary(
        [
            {
                "event_kind": "run.token_usage",
                "payload": {
                    "model": "deepseek_v4",
                    "provider": "openai-compatible-vllm",
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            }
        ]
    )

    valid = gate._actual_runtime_provenance(
        trace,
        declared_model_id="deepseek_v4",
        declared_provider_id="openai-compatible-vllm",
    )
    wrong = gate._actual_runtime_provenance(
        trace,
        declared_model_id="claimed-model",
        declared_provider_id="claimed-provider",
    )

    assert valid["validated"] is True
    assert valid["observed_model_ids"] == ["deepseek_v4"]
    assert valid["observed_provider_ids"] == ["openai-compatible-vllm"]
    assert wrong["validated"] is False


def test_missing_runtime_provider_blocks_provenance() -> None:
    trace = gate._trace_summary(
        [{"event_kind": "run.token_usage", "payload": {"model": "deepseek_v4"}}]
    )
    provenance = gate._actual_runtime_provenance(
        trace,
        declared_model_id="deepseek_v4",
        declared_provider_id="operator-claim",
    )

    assert provenance["model_matches_declaration"] is True
    assert provenance["provider_observable"] is False
    assert provenance["validated"] is False


def _worker_cleanroom_attestation(
    *, run_id: str, thread_id: str, user_id: str, goal: str
) -> dict[str, object]:
    run_sha = gate.sha256_bytes(run_id.encode())
    payload: dict[str, object] = {
        "schema_version": "1",
        "attestation_kind": "worker_evaluation_profile",
        "worker_owned": True,
        "evaluation_profile": gate.MATERIALS_CLEANROOM_PROFILE,
        "profile_source": "typed_job_envelope",
        "trusted_envelope_field": "evaluation_profile",
        "namespace_id": f"{gate.MATERIALS_CLEANROOM_PROFILE}-{run_sha}",
        "run_id_sha256": run_sha,
        "thread_id_sha256": gate.sha256_bytes(thread_id.encode()),
        "user_id_sha256": gate.sha256_bytes(user_id.encode()),
        "goal_sha256": gate.sha256_bytes(goal.encode()),
        "input_policy": "goal_only",
        "provided_message_count": 3,
        "effective_message_count": 1,
        "prior_thread_context_discarded": True,
        "same_run_retry_state_allowed": True,
        "run_scoped_workspace": True,
        "run_scoped_memory": True,
        "disabled_capabilities": list(gate.WORKER_CLEANROOM_DISABLED_CAPABILITIES),
    }
    payload["attestation_sha256"] = gate.sha256_bytes(gate.canonical_json_bytes(payload))
    return payload


def test_worker_cleanroom_attestation_is_exact_bound_and_revalidatable() -> None:
    run_id = "run-clean"
    thread_id = "thread-clean"
    user_id = "user-clean"
    goal = "PUBLIC GOAL"
    event = {
        "run_id": run_id,
        "event_kind": gate.WORKER_EVALUATION_ATTESTATION_EVENT,
        "payload": _worker_cleanroom_attestation(
            run_id=run_id,
            thread_id=thread_id,
            user_id=user_id,
            goal=goal,
        ),
    }
    accepted = {
        "run_id": run_id,
        "event_kind": "run.accepted",
        "payload": {"evaluation_profile": gate.MATERIALS_CLEANROOM_PROFILE},
    }

    live_trace = gate._trace_summary([accepted, event])
    retained_trace = gate._trace_summary(
        [gate._event_record(accepted), gate._event_record(event)]
    )
    binding = gate._worker_cleanroom_binding(
        retained_trace,
        run_id=run_id,
        thread_id=thread_id,
        goal_sha256=gate.sha256_bytes(goal.encode()),
        user_id_sha256=gate.sha256_bytes(user_id.encode()),
    )

    assert live_trace == retained_trace
    assert retained_trace["worker_cleanroom_profile_attested"] is True
    assert retained_trace["server_cleanroom_profile_attested"] is True
    assert binding["valid"] is True
    assert binding["user_identity_independently_bound"] is True


def test_worker_cleanroom_attestation_rejects_extra_fields_and_tampering() -> None:
    payload = _worker_cleanroom_attestation(
        run_id="run-clean",
        thread_id="thread-clean",
        user_id="user-clean",
        goal="PUBLIC GOAL",
    )
    payload["unexpected"] = "not accepted"
    trace = gate._trace_summary(
        [
            {
                "run_id": "run-clean",
                "event_kind": gate.WORKER_EVALUATION_ATTESTATION_EVENT,
                "payload": payload,
            }
        ]
    )

    assert trace["worker_cleanroom_profile_attested"] is False


def test_trace_requires_completed_production_execute_and_observed_image_digest() -> None:
    digest = "sha256:" + "c" * 64
    trace = gate._trace_summary(
        [
            {
                "event_kind": "tool_call.started",
                "payload": {"tool_name": "execute", "tool_call_id": "exec-1"},
            },
            {
                "event_kind": "tool_call.completed",
                "payload": {
                    "tool_name": "execute",
                    "tool_call_id": "exec-1",
                    "runtime_image_digest": digest,
                },
            },
        ]
    )

    assert trace["production_execute_tool_evidence"] is True
    assert trace["observed_execute_image_digests"] == [digest]

    missing_terminal = gate._trace_summary(
        [
            {
                "event_kind": "tool_call.started",
                "payload": {"tool_name": "execute", "tool_call_id": "exec-1"},
            }
        ]
    )
    assert missing_terminal["production_execute_tool_evidence"] is False

    missing_start = gate._trace_summary(
        [
            {
                "event_kind": "tool_call.completed",
                "payload": {
                    "tool_name": "execute",
                    "tool_call_id": "exec-1",
                    "runtime_image_digest": digest,
                },
            }
        ]
    )
    assert missing_start["production_execute_tool_evidence"] is False

    mismatched_calls = gate._trace_summary(
        [
            {
                "event_kind": "tool_call.started",
                "payload": {"tool_name": "execute", "tool_call_id": "exec-a"},
            },
            {
                "event_kind": "tool_call.failed",
                "payload": {"tool_name": "execute", "tool_call_id": "exec-a"},
            },
            {
                "event_kind": "tool_call.completed",
                "payload": {
                    "tool_name": "execute",
                    "tool_call_id": "exec-b",
                    "runtime_image_digest": digest,
                },
            },
        ]
    )
    assert mismatched_calls["production_execute_tool_evidence"] is False


def test_evaluation_requires_explicit_immutable_image_id(tmp_path: Path) -> None:
    snapshot = _write_synthetic_snapshot(tmp_path / "benchmark")
    checkpoint = gate.CampaignCheckpoint.open_or_create(
        tmp_path / "campaign" / "state.json",
        snapshot=snapshot,
        config=_campaign_config(),
    )

    with pytest.raises(gate.GateError, match="expected-evaluator-image-id"):
        gate.evaluate_campaign(
            snapshot=snapshot,
            checkpoint=checkpoint,
            output_dir=tmp_path / "campaign",
            validator_command=("python",),
            replay_count=2,
            evaluator_timeout=1,
            expected_image_id=None,
            evaluator_environment_lock={},
            sandbox_attestation_path=None,
            sandbox_attestation_signature_path=None,
            sandbox_attestation_public_key_path=None,
        )


@pytest.mark.parametrize(
    ("submit_only", "diagnostic", "trials", "task_limit", "expected"),
    [
        (True, False, 1, 3, "submission_only"),
        (False, True, 1, None, "diagnostic_full_trial"),
        (False, True, 1, gate.PARENT_TASKS_PER_TRIAL, "diagnostic_full_trial"),
        (False, False, gate.PROMOTION_TRIALS, None, "promotion"),
    ],
)
def test_evaluation_mode_is_explicit_and_non_promotable_diagnostics_are_full_trial(
    submit_only: bool,
    diagnostic: bool,
    trials: int,
    task_limit: int | None,
    expected: str,
) -> None:
    args = gate.argparse.Namespace(
        submit_only=submit_only,
        diagnostic_evaluate=diagnostic,
        trials=trials,
        task_limit=task_limit,
    )

    assert gate.resolve_evaluation_mode(args) == expected


@pytest.mark.parametrize(
    ("submit_only", "diagnostic", "trials", "task_limit", "message"),
    [
        (True, True, 1, None, "mutually exclusive"),
        (False, True, 1, 3, "complete 49-task trial"),
        (False, True, 3, None, "complete 49-task trial"),
        (False, False, 1, None, "three complete 49-task trials"),
    ],
)
def test_evaluation_mode_rejects_ambiguous_or_incomplete_scoring_shapes(
    submit_only: bool,
    diagnostic: bool,
    trials: int,
    task_limit: int | None,
    message: str,
) -> None:
    args = gate.argparse.Namespace(
        submit_only=submit_only,
        diagnostic_evaluate=diagnostic,
        trials=trials,
        task_limit=task_limit,
    )

    with pytest.raises(gate.GateError, match=message):
        gate.resolve_evaluation_mode(args)


def test_pinned_validator_command_is_project_independent_and_uses_full_lock() -> None:
    command = gate.pinned_validator_command()

    assert "--no-project" in command
    requirements_index = command.index("--with-requirements") + 1
    assert Path(command[requirements_index]).name == "mattools-validator-requirements.lock.txt"
    assert gate.HOST_VALIDATOR_REQUIREMENTS.is_file()
    assert "pymatgen==2024.8.9" in gate.HOST_VALIDATOR_REQUIREMENTS.read_text(encoding="utf-8")


def test_pinned_host_validator_executes_real_no_task_import_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="incompatible-parent"\nversion="0"\nrequires-python=">=3.13"\n',
        encoding="utf-8",
    )
    unsafe_utils = tmp_path / "utils.py"
    unsafe_utils.write_text("raise AssertionError('unsafe snapshot utils imported')\n", encoding="utf-8")
    (tmp_path / "docker_sandbox.py").write_text("class DockerSandbox:\n    pass\n")
    runner = tmp_path / "result_analysis.py"
    runner.write_text(
        "import docker\nimport openpyxl\nimport pandas\nfrom utils import ComplexDictParser\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gate, "OFFICIAL_RUNNER_SHA256", gate.sha256_file(runner))
    monkeypatch.setattr(gate, "OFFICIAL_UNSAFE_UTILS_SHA256", gate.sha256_file(unsafe_utils))

    observed = gate.inspect_host_validator_environment(
        tmp_path,
        gate.pinned_validator_command(),
    )

    assert observed["task_execution_performed"] is False
    assert observed["required_packages"] == gate.HOST_VALIDATOR_REQUIRED_VERSIONS
    assert observed["python_implementation"] == "CPython"
    assert len(observed["python_executable_sha256"]) == 64
    assert observed["platform"]
    assert len(observed["resolved_packages_sha256"]) == 64
    assert observed["safe_parser_preflight"]["safe_parser_bound"] is True


@pytest.mark.parametrize(
    ("basis", "purpose", "evidence", "expected"),
    [
        ("noncommercial", "noncommercial internal qualification", None, True),
        ("noncommercial", "placeholder", None, False),
        ("separately_licensed", "licensed release qualification", None, False),
        (
            "separately_licensed",
            "licensed release qualification",
            "sha256:" + "c" * 64,
            True,
        ),
    ],
)
def test_license_attestation_requires_concrete_purpose_and_legal_basis(
    basis: str,
    purpose: str,
    evidence: str | None,
    expected: bool,
) -> None:
    assert (
        gate.license_attestation_valid(
            {
                "accepted": True,
                "use_basis": basis,
                "use_purpose": purpose,
                "repository_license": "Apache-2.0",
                "dataset_card_license": "CC-BY-NC-4.0",
                "separate_license_evidence_sha256": evidence,
            }
        )
        is expected
    )


def test_diagnostic_completion_uses_trial_local_evaluator_gates() -> None:
    shared = {
        name: True
        for name in (
            "official_snapshot",
            "license_attested",
            "checkpoint_evidence_integrity",
            "actual_ultra_control_plane_path",
            "actual_model_provider_provenance",
            "production_execute_tool_evidence",
            "production_execute_runtime_image_attestation",
            "required_solution_artifacts",
            "expected_values_and_verifiers_isolated",
            "provenance_complete",
        )
    }
    shared.update(
        {
            "three_trial_completeness": False,
            "official_evaluator_environment_exact": False,
            "immediate_replay_reproducible": False,
            "external_sandbox_isolation_evidence": False,
        }
    )
    report = {
        "hard_gates": shared,
        "trials": [
            {
                "status": "complete",
                "reproducible": True,
                "evaluator_environment": {
                    "comparable": True,
                    "independent_from_production_runtime": True,
                },
                "sandbox_policy_attestation": {"valid": True},
            }
        ],
    }

    assert gate.diagnostic_evaluation_completed(report) is True


def test_evaluator_refuses_candidate_execution_before_invalid_sandbox_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _write_synthetic_snapshot(tmp_path / "benchmark")
    checkpoint = gate.CampaignCheckpoint.open_or_create(
        tmp_path / "campaign" / "state.json",
        snapshot=snapshot,
        config=_campaign_config(),
    )
    evaluator_id = "sha256:" + "a" * 64
    monkeypatch.setattr(
        gate,
        "inspect_evaluator_image",
        lambda **_kwargs: {"image_id": evaluator_id},
    )
    monkeypatch.setattr(
        gate,
        "validate_sandbox_attestation",
        lambda *_args, **_kwargs: {"valid": False},
    )

    with pytest.raises(gate.GateError, match="refusing to execute MatTools candidate code"):
        gate.run_official_trial_evaluation(
            snapshot=snapshot,
            checkpoint=checkpoint,
            output_dir=tmp_path / "campaign",
            trial=1,
            validator_command=("python",),
            replay_count=2,
            evaluator_timeout=1,
            runtime_image_digest="sha256:" + "b" * 64,
            expected_image_id=evaluator_id,
            evaluator_environment_lock={},
            sandbox_attestation_path=None,
            sandbox_attestation_signature_path=None,
            sandbox_attestation_public_key_path=None,
        )

    assert not (tmp_path / "campaign" / "evaluations").exists()


def test_self_asserted_sandbox_policy_cannot_pass(tmp_path: Path) -> None:
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "evaluator_image_id": "sha256:" + "a" * 64,
                "network_egress_denied": True,
                "host_access_denied": True,
            }
        ),
        encoding="utf-8",
    )

    result = gate.validate_sandbox_attestation(
        policy,
        image_id="sha256:" + "a" * 64,
        signature_path=None,
        public_key_path=None,
    )

    assert result["valid"] is False
    assert result["harness_enforces_isolation"] is False


def test_signed_external_sandbox_evidence_can_satisfy_integrity_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_id = "sha256:" + "a" * 64
    evidence = tmp_path / "isolation-evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "evaluator_image_id": image_id,
                "observed_at": "2026-07-09T00:00:00Z",
                "observed_container_id": "container-probe-1",
                "network_egress_probe": {"attempted": True, "result": "blocked"},
                "host_access_probe": {
                    "host_mount_count": 0,
                    "docker_socket_mounted": False,
                },
                "resource_limits": {
                    "memory_bytes": 1_073_741_824,
                    "pids_limit": 256,
                    "nano_cpus": 1_000_000_000,
                },
            }
        ),
        encoding="utf-8",
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "attestation_kind": "external_sandbox_isolation",
                "evaluator_image_id": image_id,
                "network_egress_denied": True,
                "host_access_denied": True,
                "resource_limits_enforced": True,
                "external_enforcement": True,
                "enforcement_mechanism": "qualification Docker-daemon policy v1",
                "isolation_evidence_path": evidence.name,
                "isolation_evidence_sha256": "sha256:" + gate.sha256_file(evidence),
                "signed_by": "release-operator@example.test",
                "signed_at": "2026-07-09T00:00:00Z",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    signature = tmp_path / "policy.sig"
    signature.write_bytes(b"detached-signature")
    public_key = tmp_path / "operator-public.pem"
    public_key.write_text("PUBLIC KEY PLACEHOLDER\n", encoding="utf-8")
    monkeypatch.setattr(gate.shutil, "which", lambda name: "/usr/bin/openssl")
    monkeypatch.setattr(gate, "_is_git_tracked_unchanged", lambda *args: True)
    monkeypatch.setattr(
        gate,
        "_run_capture",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["openssl"], returncode=0, stdout="Verified OK\n", stderr=""
        ),
    )

    result = gate.validate_sandbox_attestation(
        policy,
        image_id=image_id,
        signature_path=signature,
        public_key_path=public_key,
    )

    assert result["valid"] is True
    assert result["operator_signature_verified"] is True
    assert result["isolation_evidence_sha256"] == "sha256:" + gate.sha256_file(evidence)
    assert result["harness_enforces_isolation"] is False
