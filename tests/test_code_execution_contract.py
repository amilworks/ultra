from __future__ import annotations

from src.tooling.code_execution_contract import (
    CodeExecutionArtifact,
    CodeExecutionAttemptResult,
    CodeExecutionInput,
    CodeExecutionJobRecord,
    CodeExecutionJobRequest,
)


def test_code_execution_job_request_normalizes_inputs() -> None:
    payload = CodeExecutionJobRequest.model_validate(
        {
            "job_id": "codejob_12345678",
            "timeout_seconds": 600,
            "cpu_limit": 2.0,
            "memory_mb": 4096,
            "inputs": [
                {
                    "name": "training.csv",
                    "kind": "file",
                    "sandbox_path": "/inputs/training.csv",
                    "local_path": "/tmp/training.csv",
                },
                {
                    "name": "public.ome.tiff",
                    "kind": "file",
                    "sandbox_path": "/inputs/public.ome.tiff",
                    "uri": "s3://allencell/example/public.ome.tiff",
                },
            ],
        }
    )

    assert payload.job_id == "codejob_12345678"
    assert payload.inputs[0].local_path == "/tmp/training.csv"
    assert payload.inputs[1].uri == "s3://allencell/example/public.ome.tiff"


def test_code_execution_job_record_round_trips_artifact_manifest() -> None:
    record = CodeExecutionJobRecord(
        job_id="job-1",
        status="succeeded",
        created_at="2026-04-17T00:00:00Z",
        updated_at="2026-04-17T00:05:00Z",
        request=CodeExecutionJobRequest(job_id="job-1"),
        artifact_manifest=[
            CodeExecutionArtifact(
                name="metrics.json",
                relative_path="outputs/metrics.json",
                size_bytes=128,
                kind="file",
            )
        ],
        result=CodeExecutionAttemptResult(
            success=True,
            job_id="job-1",
            execution_backend="service",
            exit_code=0,
            runtime_seconds=12.5,
            stdout_tail="done",
            stderr_tail="",
            artifacts=[],
            output_files=["outputs/metrics.json"],
            expected_outputs=["metrics.json"],
            missing_expected_outputs=[],
            analysis_outputs=[],
            measurement_candidates=[],
            key_measurements=[],
            metrics={"artifacts_count": 1},
        ),
    )

    restored = CodeExecutionJobRecord.model_validate(record.model_dump(mode="json"))
    assert restored.job_id == "job-1"
    assert restored.artifact_manifest[0].relative_path == "outputs/metrics.json"
    assert restored.result is not None
    assert restored.result.execution_backend == "service"
