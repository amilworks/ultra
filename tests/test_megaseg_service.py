import json
import tarfile
import time
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from services.megaseg_service import app as megaseg_service_app
from src.science.megaseg_service_client import MegasegServiceClient


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer secret-token"}


def test_megaseg_service_job_lifecycle_and_artifact_download(monkeypatch, tmp_path):
    checkpoint = tmp_path / "epoch_650.ckpt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    settings = megaseg_service_app.ServiceSettings(
        checkpoint_path=checkpoint,
        device="cpu",
        job_root=tmp_path / "jobs",
        artifact_root=tmp_path / "artifacts",
        api_key="secret-token",
        max_concurrent_jobs=1,
        amp_enabled=False,
    )

    monkeypatch.setattr(megaseg_service_app, "build_megaseg_model", lambda **_kwargs: object())

    def fake_run_megaseg_batch(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        sample_dir = output_dir / "sample"
        sample_dir.mkdir(parents=True, exist_ok=True)
        mask_path = sample_dir / "sample__megaseg_mask.tiff"
        probability_path = sample_dir / "sample__megaseg_probability.tiff"
        summary_json_path = sample_dir / "sample__megaseg_summary.json"
        summary_csv_path = output_dir / "megaseg_summary.csv"
        report_path = output_dir / "megaseg_report.md"
        mask_path.write_text("mask", encoding="utf-8")
        probability_path.write_text("probability", encoding="utf-8")
        summary_json_path.write_text("{}", encoding="utf-8")
        summary_csv_path.write_text("file,success\nsample,True\n", encoding="utf-8")
        report_path.write_text("# Megaseg Inference Report\n", encoding="utf-8")
        return {
            "success": True,
            "device": "cpu",
            "checkpoint_path": str(checkpoint),
            "output_directory": str(output_dir),
            "summary_csv_path": str(summary_csv_path),
            "report_path": str(report_path),
            "warnings": [],
            "aggregate": {
                "processed_files": 1,
                "total_files": 1,
                "mean_coverage_percent": 1.0,
                "median_coverage_percent": 1.0,
                "mean_object_count": 2.0,
                "median_object_count": 2.0,
            },
            "files": [
                {
                    "file": "sample",
                    "success": True,
                    "mask_path": str(mask_path),
                    "probability_path": str(probability_path),
                    "summary_json_path": str(summary_json_path),
                    "visualizations": [],
                    "segmentation": {
                        "coverage_percent": 1.0,
                        "object_count": 2,
                        "active_slice_count": 1,
                        "largest_component_voxels": 12,
                    },
                    "intensity_context": {},
                    "technical_summary": "ok",
                }
            ],
        }

    monkeypatch.setattr(megaseg_service_app, "run_megaseg_batch", fake_run_megaseg_batch)

    app = megaseg_service_app.create_app(settings)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 401

        health_response = client.get("/health", headers=_auth_headers())
        assert health_response.status_code == 200
        assert health_response.json()["ok"] is True

        response = client.post(
            "/v1/jobs",
            headers=_auth_headers(),
            data={
                "request_json": json.dumps(
                    {
                        "structure_channel": 1,
                        "nucleus_channel": None,
                        "channel_index_base": 1,
                        "mask_threshold": 0.5,
                        "save_visualizations": False,
                        "generate_report": True,
                        "sources": [{"uri": "s3://allencell/aics/example.ome.zarr/"}],
                    }
                )
            },
            files=[("files", ("sample.ome.tiff", b"raw-bytes", "application/octet-stream"))],
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]

        terminal = None
        for _ in range(50):
            job_response = client.get(f"/v1/jobs/{job_id}", headers=_auth_headers())
            assert job_response.status_code == 200
            terminal = job_response.json()
            if terminal["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.05)

        assert terminal is not None
        assert terminal["status"] == "succeeded"
        assert terminal["artifact_manifest"]

        artifact_response = client.get(
            f"/v1/jobs/{job_id}/artifacts/megaseg_report.md",
            headers=_auth_headers(),
        )
        assert artifact_response.status_code == 200
        assert "# Megaseg Inference Report" in artifact_response.text


def test_megaseg_service_marks_stale_running_jobs_failed(monkeypatch, tmp_path):
    checkpoint = tmp_path / "epoch_650.ckpt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    settings = megaseg_service_app.ServiceSettings(
        checkpoint_path=checkpoint,
        device="cpu",
        job_root=tmp_path / "jobs",
        artifact_root=tmp_path / "artifacts",
        api_key="secret-token",
        max_concurrent_jobs=1,
        amp_enabled=False,
    )
    settings.job_root.mkdir(parents=True, exist_ok=True)
    stale_record_path = settings.job_root / "stale-job.json"
    stale_record_path.write_text(
        json.dumps(
            {
                "job_id": "stale-job",
                "status": "running",
                "created_at": "2026-04-16T00:00:00+00:00",
                "updated_at": "2026-04-16T00:00:00+00:00",
                "request": {
                    "sources": [{"uri": "s3://allencell/aics/example.ome.zarr/"}],
                    "structure_channel": 1,
                    "nucleus_channel": None,
                    "channel_index_base": 1,
                    "mask_threshold": 0.5,
                    "save_visualizations": False,
                    "generate_report": True,
                    "device": None,
                    "checkpoint_path": None,
                    "structure_name": None,
                    "amp_enabled": None,
                },
                "resolved_sources": ["s3://allencell/aics/example.ome.zarr/"],
                "uploaded_files": [],
                "artifact_manifest": [],
                "result": None,
                "error": None,
                "started_at": "2026-04-16T00:00:00+00:00",
                "finished_at": None,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(megaseg_service_app, "build_megaseg_model", lambda **_kwargs: object())
    monkeypatch.setattr(
        megaseg_service_app, "run_megaseg_batch", lambda **_kwargs: {"success": True, "files": []}
    )

    app = megaseg_service_app.create_app(settings)
    with TestClient(app) as client:
        response = client.get("/v1/jobs/stale-job", headers=_auth_headers())
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "failed"
        assert "restarted" in payload["error"].lower()


def test_megaseg_service_extracts_uploaded_zarr_archive(monkeypatch, tmp_path):
    checkpoint = tmp_path / "epoch_650.ckpt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    settings = megaseg_service_app.ServiceSettings(
        checkpoint_path=checkpoint,
        device="cpu",
        job_root=tmp_path / "jobs",
        artifact_root=tmp_path / "artifacts",
        api_key="secret-token",
        max_concurrent_jobs=1,
        amp_enabled=False,
    )

    monkeypatch.setattr(megaseg_service_app, "build_megaseg_model", lambda **_kwargs: object())

    captured_sources: list[str] = []

    def fake_run_megaseg_batch(**kwargs):
        captured_sources.extend([str(item) for item in kwargs["file_paths"]])
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "success": True,
            "device": "cpu",
            "checkpoint_path": str(checkpoint),
            "output_directory": str(output_dir),
            "files": [],
        }

    monkeypatch.setattr(megaseg_service_app, "run_megaseg_batch", fake_run_megaseg_batch)

    archive_buffer = BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        payload = b'{"zarr_format": 2}'
        info = tarfile.TarInfo(name="sample.ome.zarr/.zgroup")
        info.size = len(payload)
        archive.addfile(info, BytesIO(payload))
    archive_buffer.seek(0)

    app = megaseg_service_app.create_app(settings)
    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs",
            headers=_auth_headers(),
            data={
                "request_json": json.dumps(
                    {
                        "structure_channel": 1,
                        "nucleus_channel": None,
                        "channel_index_base": 1,
                        "mask_threshold": 0.5,
                        "save_visualizations": False,
                        "generate_report": False,
                        "sources": [],
                    }
                )
            },
            files=[
                ("files", ("sample.ome.zarr.tar.gz", archive_buffer.getvalue(), "application/gzip"))
            ],
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]

        terminal = None
        for _ in range(50):
            job_response = client.get(f"/v1/jobs/{job_id}", headers=_auth_headers())
            assert job_response.status_code == 200
            terminal = job_response.json()
            if terminal["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.05)

        assert terminal is not None
        assert terminal["status"] == "succeeded"

    assert captured_sources
    extracted_path = Path(captured_sources[0])
    assert extracted_path.is_dir()
    assert extracted_path.name == "sample.ome.zarr"
    assert (extracted_path / ".zgroup").exists()


def test_megaseg_service_client_archives_directory_upload(monkeypatch, tmp_path):
    source_dir = tmp_path / "sample.ome.zarr"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / ".zgroup").write_text('{"zarr_format": 2}', encoding="utf-8")

    captured: dict[str, object] = {}

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"job_id": "job-123"}

    def fake_request(self, method, path, **kwargs):
        captured["method"] = method
        captured["path"] = path
        files = kwargs.get("files") or []
        assert len(files) == 1
        field_name, file_tuple = files[0]
        assert field_name == "files"
        filename, handle, content_type = file_tuple
        captured["filename"] = filename
        captured["content_type"] = content_type
        captured["archive_bytes"] = handle.read()
        return _Response()

    monkeypatch.setattr(MegasegServiceClient, "_request", fake_request)

    client = MegasegServiceClient(base_url="http://example.invalid", api_key="secret-token")
    payload = {"sources": [], "structure_channel": 1, "nucleus_channel": None}
    result = client.submit_job(request_payload=payload, local_upload_paths=[source_dir])

    assert result["job_id"] == "job-123"
    assert captured["filename"] == "sample.ome.zarr.tar.gz"
    assert captured["content_type"] == "application/gzip"

    archive_buffer = BytesIO(captured["archive_bytes"])  # type: ignore[arg-type]
    with tarfile.open(fileobj=archive_buffer, mode="r:gz") as archive:
        assert "sample.ome.zarr/.zgroup" in archive.getnames()
