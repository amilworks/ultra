from __future__ import annotations

import json
from pathlib import Path

from src.config import Settings
from src.tooling.code_execution import execute_python_job_once


def test_execute_python_job_once_uses_service_when_configured(
    monkeypatch, tmp_path: Path
) -> None:
    job_dir = tmp_path / "artifacts" / "code_jobs" / "codejob_123"
    source_dir = job_dir / "source"
    source_dir.mkdir(parents=True)
    (source_dir / "main.py").write_text("print('hello')\n", encoding="utf-8")
    (job_dir / "job_spec.json").write_text(
        json.dumps(
            {
                "job_id": "codejob_123",
                "entrypoint": "main.py",
                "command": "python main.py",
                "expected_outputs": ["metrics.json"],
                "inputs": [],
                "attempt_index": 1,
            }
        ),
        encoding="utf-8",
    )

    settings = Settings(
        _env_file=None,
        artifact_root=str(tmp_path / "artifacts"),
        code_execution_enabled=True,
        code_execution_service_url="http://codeexec.internal:8020",
        code_execution_service_api_key="secret-token",
    )
    monkeypatch.setattr("src.tooling.code_execution.get_settings", lambda: settings)

    class DummyClient:
        def __init__(self, **_kwargs):
            pass

        def submit_job(self, **_kwargs):
            return {"job_id": "svc-job-1", "status": "queued"}

        def wait_for_job(self, **_kwargs):
            return {
                "job_id": "svc-job-1",
                "status": "succeeded",
                "result": {
                    "success": True,
                    "job_id": "codejob_123",
                    "execution_backend": "service",
                    "exit_code": 0,
                    "runtime_seconds": 4.2,
                    "stdout_tail": "done",
                    "stderr_tail": "",
                    "artifacts": [
                        {
                            "path": str(tmp_path / "downloaded" / "metrics.json"),
                            "relative_path": "outputs/metrics.json",
                            "size_bytes": 20,
                            "kind": "file",
                            "title": "metrics.json",
                        }
                    ],
                    "output_files": ["outputs/metrics.json"],
                    "expected_outputs": ["metrics.json"],
                    "missing_expected_outputs": [],
                    "analysis_outputs": [],
                    "measurement_candidates": [{"name": "accuracy", "value": 0.95}],
                    "key_measurements": [{"name": "accuracy", "value": 0.95}],
                    "metrics": {"artifacts_count": 1},
                },
                "artifact_manifest": [
                    {
                        "name": "outputs/metrics.json",
                        "relative_path": "outputs/metrics.json",
                        "size_bytes": 20,
                        "kind": "file",
                    }
                ],
            }

        def download_artifact(self, **kwargs):
            destination = kwargs["destination"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text('{"accuracy": 0.95}', encoding="utf-8")
            return destination

    monkeypatch.setattr("src.tooling.code_execution.CodeExecutionServiceClient", DummyClient)

    result = execute_python_job_once(job_id="codejob_123")

    assert result["success"] is True
    assert result["execution_backend"] == "service"
    assert result["key_measurements"][0]["name"] == "accuracy"
    assert result["service_job_id"] == "svc-job-1"
    assert (
        source_dir / "outputs" / "metrics.json"
    ).read_text(encoding="utf-8") == '{"accuracy": 0.95}'
