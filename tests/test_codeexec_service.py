from __future__ import annotations

import io
import json
import tarfile
import time
from pathlib import Path

import scripts.eval_codeexec_reasoning_suite as eval_suite
from fastapi.testclient import TestClient
from services.codeexec_service import app as codeexec_service_app


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer secret-token"}


def test_codeexec_service_job_lifecycle_and_artifact_download(
    monkeypatch, tmp_path: Path
) -> None:
    settings = codeexec_service_app.ServiceSettings(
        job_root=tmp_path / "jobs",
        artifact_root=tmp_path / "artifacts",
        api_key="secret-token",
        worker_image="ultra-codeexec-job:current",
        docker_network="none",
        max_concurrent_jobs=1,
    )

    def fake_run_attempt(*, job_id: str, work_dir: Path, **_kwargs):
        outputs = work_dir / "source" / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        (outputs / "metrics.json").write_text('{"accuracy": 0.94}', encoding="utf-8")
        return {
            "success": True,
            "job_id": job_id,
            "execution_backend": "service",
            "exit_code": 0,
            "runtime_seconds": 3.2,
            "stdout_tail": "done",
            "stderr_tail": "",
            "artifacts": [
                {
                    "path": str(outputs / "metrics.json"),
                    "relative_path": "outputs/metrics.json",
                    "size_bytes": 18,
                    "kind": "file",
                    "title": "metrics.json",
                }
            ],
            "output_files": ["outputs/metrics.json"],
            "expected_outputs": ["metrics.json"],
            "missing_expected_outputs": [],
            "analysis_outputs": [],
            "measurement_candidates": [{"name": "accuracy", "value": 0.94}],
            "key_measurements": [{"name": "accuracy", "value": 0.94}],
            "metrics": {"artifacts_count": 1},
        }

    monkeypatch.setattr(codeexec_service_app, "run_codeexec_attempt", fake_run_attempt)

    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        job_spec = json.dumps(
            {
                "job_id": "codejob_abc123",
                "entrypoint": "main.py",
                "command": "python main.py",
                "expected_outputs": ["metrics.json"],
                "inputs": [],
            }
        ).encode("utf-8")
        main_py = b"print('hello')\n"
        spec_info = tarfile.TarInfo(name="job_spec.json")
        spec_info.size = len(job_spec)
        archive.addfile(spec_info, io.BytesIO(job_spec))
        source_info = tarfile.TarInfo(name="source/main.py")
        source_info.size = len(main_py)
        archive.addfile(source_info, io.BytesIO(main_py))
    archive_buffer.seek(0)

    app = codeexec_service_app.create_app(settings)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 401

        response = client.post(
            "/v1/jobs",
            headers=_auth_headers(),
            data={
                "request_json": json.dumps(
                    {
                        "job_id": "codejob_abc123",
                        "timeout_seconds": 600,
                        "cpu_limit": 2.0,
                        "memory_mb": 4096,
                        "expected_outputs": ["metrics.json"],
                        "inputs": [],
                    }
                )
            },
            files=[("bundle", ("job.tar.gz", archive_buffer.getvalue(), "application/gzip"))],
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]

        terminal = None
        for _ in range(50):
            terminal = client.get(f"/v1/jobs/{job_id}", headers=_auth_headers()).json()
            if terminal["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.05)

        assert terminal is not None
        assert terminal["status"] == "succeeded"
        artifact_response = client.get(
            f"/v1/jobs/{job_id}/artifacts/outputs/metrics.json",
            headers=_auth_headers(),
        )
        assert artifact_response.status_code == 200
        assert artifact_response.json()["accuracy"] == 0.94


def test_codeexec_service_health_stays_responsive_during_running_job(
    monkeypatch, tmp_path: Path
) -> None:
    settings = codeexec_service_app.ServiceSettings(
        job_root=tmp_path / "jobs",
        artifact_root=tmp_path / "artifacts",
        api_key="secret-token",
        worker_image="ultra-codeexec-job:current",
        docker_network="none",
        max_concurrent_jobs=1,
    )

    def fake_run_attempt(*, job_id: str, work_dir: Path, **_kwargs):
        time.sleep(0.5)
        output = work_dir / "source" / "result.json"
        output.write_text('{"ok": true}', encoding="utf-8")
        return {
            "success": True,
            "job_id": job_id,
            "execution_backend": "service",
            "exit_code": 0,
            "runtime_seconds": 0.5,
            "stdout_tail": "done",
            "stderr_tail": "",
            "artifacts": [
                {
                    "path": str(output),
                    "relative_path": "result.json",
                    "size_bytes": 12,
                    "kind": "file",
                    "title": "result.json",
                }
            ],
            "output_files": ["result.json"],
            "expected_outputs": ["result.json"],
            "missing_expected_outputs": [],
            "analysis_outputs": [],
            "measurement_candidates": [],
            "key_measurements": [],
            "metrics": {"artifacts_count": 1},
        }

    monkeypatch.setattr(codeexec_service_app, "run_codeexec_attempt", fake_run_attempt)

    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        job_spec = json.dumps(
            {
                "job_id": "slowjob_abc123",
                "entrypoint": "main.py",
                "command": "python main.py",
                "expected_outputs": ["result.json"],
                "inputs": [],
            }
        ).encode("utf-8")
        main_py = b"print('hello')\n"
        spec_info = tarfile.TarInfo(name="job_spec.json")
        spec_info.size = len(job_spec)
        archive.addfile(spec_info, io.BytesIO(job_spec))
        source_info = tarfile.TarInfo(name="source/main.py")
        source_info.size = len(main_py)
        archive.addfile(source_info, io.BytesIO(main_py))
    archive_buffer.seek(0)

    app = codeexec_service_app.create_app(settings)
    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs",
            headers=_auth_headers(),
            data={
                "request_json": json.dumps(
                    {
                        "job_id": "slowjob_abc123",
                        "timeout_seconds": 600,
                        "cpu_limit": 2.0,
                        "memory_mb": 4096,
                        "expected_outputs": ["result.json"],
                        "inputs": [],
                    }
                )
            },
            files=[("bundle", ("job.tar.gz", archive_buffer.getvalue(), "application/gzip"))],
        )
        assert response.status_code == 202
        time.sleep(0.05)
        started = time.perf_counter()
        health_response = client.get("/health", headers=_auth_headers())
        elapsed = time.perf_counter() - started
        assert health_response.status_code == 200
        assert elapsed < 0.2

        terminal = None
        for _ in range(50):
            terminal = client.get("/v1/jobs/slowjob_abc123", headers=_auth_headers()).json()
            if terminal["status"] == "succeeded":
                break
            time.sleep(0.05)

        assert terminal is not None
        assert terminal["status"] == "succeeded"


def test_codeexec_reasoning_fixture_loader_reads_expected_problem_ids() -> None:
    problems = eval_suite.load_problems()

    problem_ids = {str(item.get("id") or "") for item in problems}
    assert "nonlinear_model_selection" in problem_ids
    assert "microscopy_feature_qc" in problem_ids


def test_codeexec_reasoning_score_run_rewards_fail_closed_responses() -> None:
    score = eval_suite.score_run(
        {
            "success": False,
            "response_text": "The requested code execution did not complete successfully, so no measured outputs are reported.",
            "missing_expected_outputs": ["result.json"],
            "measurement_count": 0,
        }
    )

    assert score["artifact_completion"] == 0.0
    assert score["failure_honesty"] == 1.0
