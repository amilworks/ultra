"""Tests for the batch MegaSeg inference processor: per-image staging + registration,
failure isolation, and resume-skip — with the GPU and control-plane clients mocked."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import ultra_deepagents.analysis.processor as proc
from ultra_deepagents.analysis.config import AnalysisSettings
from ultra_deepagents.analysis.processor import AnalysisProcessor
from ultra_deepagents.data_agent.worker import DataAgentJobEnvelope
from ultra_deepagents.rarespot.uploads import UploadedFileResolution


def _settings(tmp_path: Path) -> AnalysisSettings:
    upload_root = tmp_path / "uploads"
    upload_root.mkdir(parents=True, exist_ok=True)
    return AnalysisSettings(
        control_base_url="http://control",
        control_status_timeout_seconds=5.0,
        upload_root=upload_root,
        upload_roots=(upload_root,),
        upload_database_url="",
        megaseg_service_url="http://gpu",
        megaseg_service_api_key="key",
        megaseg_timeout_seconds=60.0,
    )


def _megaseg_job() -> DataAgentJobEnvelope:
    return DataAgentJobEnvelope(
        job_id="job1",
        job_type="analysis.megaseg",
        resource_ids=("file_a", "file_b", "file_missing"),
        resource_count=3,
        input_selector={"params": {"structure_channel": 4}},
        metadata={"results_collection_id": "col1", "model": "megaseg"},
    )


def test_megaseg_batch_registers_outputs_and_isolates_failures(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    img_a = settings.upload_root / "file_a__a.tif"
    img_a.write_bytes(b"A")
    img_b = settings.upload_root / "file_b__b.tif"
    img_b.write_bytes(b"B")

    async def fake_resolve(requested, *, upload_roots, database_url=""):
        mapping = {"file_a": img_a, "file_b": img_b}
        found = [r for r in requested if r in mapping]
        return UploadedFileResolution(
            image_paths=[mapping[r] for r in found],
            missing_file_ids=[r for r in requested if r not in mapping],
        )

    def fake_infer(*, service_url, api_key, image_path, params, timeout, dest_dir):
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        if Path(image_path).name.startswith("file_b"):
            raise RuntimeError("inference boom")
        (dest / "mask.tif").write_bytes(b"MASK")
        (dest / "summary.json").write_text(json.dumps({"coverage_percent": 1.0}), encoding="utf-8")
        return {
            "checkpoint_path": "/models/epoch_650.ckpt",
            "files": [
                {
                    "mask_path": "mask.tif",
                    "summary_json_path": "summary.json",
                    "segmentation": {"object_count": 2},
                    "intensity_context": {},
                }
            ],
        }

    registered: list[tuple[str, list[dict], str]] = []

    def fake_register(*, control_base_url, job_id, principal_headers, outputs, timeout, collection_id=""):
        registered.append((job_id, outputs, collection_id))
        return {"count": len(outputs)}

    monkeypatch.setattr(proc, "resolve_uploaded_file_ids", fake_resolve)
    monkeypatch.setattr(proc, "run_megaseg_infer", fake_infer)
    monkeypatch.setattr(proc, "register_outputs", fake_register)
    monkeypatch.setattr(proc, "fetch_job", lambda **kwargs: {})

    progress_calls: list[dict] = []

    async def progress(**kwargs):
        progress_calls.append(kwargs)

    summary = asyncio.run(AnalysisProcessor(settings)(_megaseg_job(), progress))

    assert summary["processed"] == 1
    assert summary["failed"] == 2  # file_b inference error + file_missing not found
    assert summary["total"] == 3
    assert summary["results_collection_id"] == "col1"

    assert len(registered) == 1
    job_id, outputs, collection_id = registered[0]
    assert job_id == "job1"
    assert collection_id == "col1"  # passed from the envelope metadata
    assert sorted(o["artifact_kind"] for o in outputs) == ["mask", "metrics"]

    mask = next(o for o in outputs if o["artifact_kind"] == "mask")
    assert mask["storage_path"] == "analysis/job1/file_a/mask__mask.tif"
    assert mask["source_resource_id"] == "file_a"
    assert mask["size_bytes"] == 4 and mask["sha256"]
    assert mask["resource_id"] == "file_an_job1_file_a_mask"
    assert mask["metadata"]["model_version"] == "epoch_650"
    assert mask["metadata"]["segmentation"] == {"object_count": 2}
    assert (settings.upload_root / mask["storage_path"]).read_bytes() == b"MASK"

    assert progress_calls and progress_calls[-1]["progress_completed"] == 1


def test_megaseg_batch_resumes_already_done_items(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    img_a = settings.upload_root / "file_a__a.tif"
    img_a.write_bytes(b"A")

    async def fake_resolve(requested, *, upload_roots, database_url=""):
        return UploadedFileResolution(image_paths=[img_a], missing_file_ids=[])

    infer_calls: list[str] = []

    def fake_infer(*, service_url, api_key, image_path, params, timeout, dest_dir):
        infer_calls.append(str(image_path))
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "mask.tif").write_bytes(b"M")
        return {"checkpoint_path": "/m/e.ckpt", "files": [{"mask_path": "mask.tif"}]}

    monkeypatch.setattr(proc, "resolve_uploaded_file_ids", fake_resolve)
    monkeypatch.setattr(proc, "run_megaseg_infer", fake_infer)
    monkeypatch.setattr(proc, "register_outputs", lambda **kwargs: {})
    # Prior run already completed file_a — the processor must skip it (no second inference).
    monkeypatch.setattr(
        proc,
        "fetch_job",
        lambda **kwargs: {"output_summary": {"items": {"file_a": {"status": "done"}}}},
    )

    job = DataAgentJobEnvelope(
        job_id="job2",
        job_type="analysis.megaseg",
        resource_ids=("file_a",),
        resource_count=1,
        metadata={"results_collection_id": "col2"},
    )

    async def progress(**kwargs):
        return None

    summary = asyncio.run(AnalysisProcessor(settings)(job, progress))
    assert summary["processed"] == 1
    assert infer_calls == []  # resumed: already-done item was not re-inferred
