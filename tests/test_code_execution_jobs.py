from __future__ import annotations

import json
import tarfile
from pathlib import Path

from src.tooling.code_execution_jobs import build_service_submission_bundle, load_job_spec_inputs
from src.tooling.code_execution_results import build_execution_summary


def test_build_service_submission_bundle_contains_job_spec_and_source(tmp_path: Path) -> None:
    job_dir = tmp_path / "codejob_abc123"
    source_dir = job_dir / "source"
    source_dir.mkdir(parents=True)
    (source_dir / "main.py").write_text("print('hello')\n", encoding="utf-8")
    (job_dir / "job_spec.json").write_text(
        json.dumps(
            {
                "job_id": "codejob_abc123",
                "entrypoint": "main.py",
                "command": "python main.py",
                "inputs": [{"path": str(tmp_path / "data.csv"), "sandbox_path": "/inputs/data.csv"}],
                "expected_outputs": ["metrics.json"],
            }
        ),
        encoding="utf-8",
    )

    bundle_path, remote_inputs, local_inputs = build_service_submission_bundle(job_dir)

    assert bundle_path.exists()
    assert remote_inputs == []
    assert local_inputs[0]["sandbox_path"] == "/inputs/data.csv"
    with tarfile.open(bundle_path, "r:gz") as archive:
        assert "job_spec.json" in archive.getnames()
        assert "source/main.py" in archive.getnames()


def test_load_job_spec_inputs_returns_spec_inputs(tmp_path: Path) -> None:
    job_dir = tmp_path / "codejob_inputs"
    job_dir.mkdir(parents=True)
    (job_dir / "job_spec.json").write_text(
        json.dumps(
            {
                "job_id": "codejob_inputs",
                "inputs": [
                    {"path": "/tmp/a.csv", "sandbox_path": "/inputs/a.csv"},
                    {
                        "path": "s3://allencell/example/public.ome.tiff",
                        "sandbox_path": "/inputs/public.ome.tiff",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    inputs = load_job_spec_inputs(job_dir)

    assert inputs[0]["sandbox_path"] == "/inputs/a.csv"
    assert inputs[1]["path"] == "s3://allencell/example/public.ome.tiff"


def test_build_execution_summary_extracts_csv_and_json_metrics(tmp_path: Path) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "metrics.json").write_text(
        json.dumps({"accuracy": 0.93, "roc_auc": 0.97}),
        encoding="utf-8",
    )
    (workdir / "feature_importance.csv").write_text(
        "feature,importance\nA,0.7\nB,0.3\n",
        encoding="utf-8",
    )

    summary = build_execution_summary(
        source_dir=workdir,
        artifacts=[
            {
                "path": str(workdir / "metrics.json"),
                "relative_path": "metrics.json",
                "size_bytes": 32,
            },
            {
                "path": str(workdir / "feature_importance.csv"),
                "relative_path": "feature_importance.csv",
                "size_bytes": 48,
            },
        ],
        expected_outputs=["metrics.json", "feature_importance.csv"],
    )

    metric_names = {item["name"] for item in summary["key_measurements"]}
    assert "accuracy" in metric_names
    assert "feature_importance.importance.mean" in metric_names
