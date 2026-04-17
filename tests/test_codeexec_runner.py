from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

from services.codeexec_service.runner import run_codeexec_attempt


def test_run_codeexec_attempt_collects_outputs_from_worker(monkeypatch, tmp_path: Path) -> None:
    work_dir = tmp_path / "workdir"
    source_dir = work_dir / "source"
    source_dir.mkdir(parents=True)
    main_py = source_dir / "main.py"
    main_py.write_text("print('hello')\n", encoding="utf-8")
    (work_dir / "job_spec.json").write_text(
        json.dumps(
            {
                "job_id": "job-123",
                "entrypoint": "main.py",
                "command": "python main.py",
                "expected_outputs": ["metrics.json"],
                "files": [{"path": "main.py"}],
            }
        ),
        encoding="utf-8",
    )
    captured_command: list[str] = []

    def fake_run(command: list[str], **_kwargs):
        captured_command.extend(command)
        outputs = source_dir / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        (outputs / "metrics.json").write_text('{"accuracy": 0.95}', encoding="utf-8")
        return CompletedProcess(command, 0, stdout="done\n", stderr="")

    monkeypatch.setattr("services.codeexec_service.runner.subprocess.run", fake_run)

    result = run_codeexec_attempt(
        job_id="job-123",
        work_dir=work_dir,
        request={
            "timeout_seconds": 600,
            "cpu_limit": 2.0,
            "memory_mb": 4096,
            "expected_outputs": ["metrics.json"],
            "inputs": [],
        },
        worker_image="ultra-codeexec-job:current",
        docker_network="none",
    )

    assert result["success"] is True
    assert result["execution_backend"] == "service"
    assert result["output_files"] == ["outputs/metrics.json"]
    assert result["missing_expected_outputs"] == []
    assert result["artifacts"][0]["relative_path"] == "outputs/metrics.json"
    assert "ultra-codeexec-job:current" in captured_command
