"""Tests for the batch MegaSeg inference processor: per-image staging + registration,
failure isolation, and resume-skip — with the GPU and control-plane clients mocked."""

from __future__ import annotations

import asyncio
import io
import json
import tarfile
from pathlib import Path

import pytest
import ultra_deepagents.analysis.client as analysis_client
import ultra_deepagents.analysis.processor as proc
from ultra_deepagents.analysis.client import _safe_extract
from ultra_deepagents.analysis.config import AnalysisSettings
from ultra_deepagents.analysis.processor import AnalysisProcessor
from ultra_deepagents.data_agent.worker import DataAgentJobEnvelope


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

    def fake_resolve(file_id, upload_roots):
        return {"file_a": img_a, "file_b": img_b}.get(file_id)

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

    monkeypatch.setattr(proc, "_resolve_source_path", fake_resolve)
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

    def fake_resolve(file_id, upload_roots):
        return img_a if file_id == "file_a" else None

    infer_calls: list[str] = []

    def fake_infer(*, service_url, api_key, image_path, params, timeout, dest_dir):
        infer_calls.append(str(image_path))
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "mask.tif").write_bytes(b"M")
        return {"checkpoint_path": "/m/e.ckpt", "files": [{"mask_path": "mask.tif"}]}

    monkeypatch.setattr(proc, "_resolve_source_path", fake_resolve)
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


def test_analysis_settings_loads_control_worker_token_from_runtime_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ULTRA_CONTROL_WORKER_TOKEN", "analysis-worker-secret")
    monkeypatch.setenv("ULTRA_CONTROL_UPLOAD_ROOT", str(tmp_path / "uploads"))

    settings = AnalysisSettings.from_env()

    assert settings.control_worker_token == "analysis-worker-secret"


def test_megaseg_control_plane_calls_include_worker_token(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    settings.control_worker_token = "analysis-worker-secret"
    img_a = settings.upload_root / "file_a__a.tif"
    img_a.write_bytes(b"A")

    def fake_infer(*, service_url, api_key, image_path, params, timeout, dest_dir):
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "mask.tif").write_bytes(b"M")
        return {"checkpoint_path": "/m/e.ckpt", "files": [{"mask_path": "mask.tif"}]}

    seen_fetch_headers: list[dict[str, str]] = []
    seen_register_headers: list[dict[str, str]] = []

    def fake_fetch(*, principal_headers, **_kwargs):
        seen_fetch_headers.append(dict(principal_headers))
        return {}

    def fake_register(*, principal_headers, **_kwargs):
        seen_register_headers.append(dict(principal_headers))
        return {"count": 1}

    monkeypatch.setattr(proc, "_resolve_source_path", lambda file_id, upload_roots: img_a)
    monkeypatch.setattr(proc, "run_megaseg_infer", fake_infer)
    monkeypatch.setattr(proc, "fetch_job", fake_fetch)
    monkeypatch.setattr(proc, "register_outputs", fake_register)

    async def progress(**kwargs):
        return None

    job = DataAgentJobEnvelope(
        job_id="job-token",
        job_type="analysis.megaseg",
        owner_user_id="owner-a",
        owner_org_id="org-a",
        resource_ids=("file_a",),
        resource_count=1,
    )

    asyncio.run(AnalysisProcessor(settings)(job, progress))

    assert seen_fetch_headers
    assert seen_register_headers
    assert all(headers["X-Ultra-Worker-Token"] == "analysis-worker-secret" for headers in seen_fetch_headers)
    assert all(headers["X-Ultra-Worker-Token"] == "analysis-worker-secret" for headers in seen_register_headers)


def test_rarespot_batch_registers_outputs_and_isolates_missing_inputs(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    img_a = settings.upload_root / "file_a__survey-a.jpg"
    img_a.write_bytes(b"A")

    def fake_resolve(file_id, upload_roots):
        return img_a if file_id == "file_a" else None

    inference_calls: list[dict[str, object]] = []

    def fake_infer(*, image_paths, run_id, thread_id, output_dir, config, progress_callback):
        inference_calls.append(
            {
                "image_paths": image_paths,
                "run_id": run_id,
                "thread_id": thread_id,
                "config": config,
            }
        )
        out = Path(output_dir)
        (out / "prediction_xml").mkdir(parents=True, exist_ok=True)
        (out / "predictions.json").write_text("{}", encoding="utf-8")
        (out / "prediction_xml" / "survey-a.xml").write_text("<xml />", encoding="utf-8")
        (out / "overlay-a.png").write_bytes(b"PNG")
        progress_callback({"event": "inference_completed", "detections": 3})
        return {
            "counts_by_class": {"burrow": 1, "prairie_dog": 2},
            "predictions": [
                {
                    "input_path": str(img_a),
                    "prediction_xml_path": str(out / "prediction_xml" / "survey-a.xml"),
                    "class_counts": {"burrow": 1, "prairie_dog": 2},
                    "boxes": [{"class_name": "prairie_dog"}, {"class_name": "prairie_dog"}, {"class_name": "burrow"}],
                }
            ],
            "artifacts": [
                {
                    "path": "predictions.json",
                    "kind": "json",
                    "title": "RareSpot predictions",
                    "mime_type": "application/json",
                    "category": "prediction",
                    "source_path": str(out / "predictions.json"),
                    "storage_uri": (out / "predictions.json").as_uri(),
                },
                {
                    "path": "prediction_xml/survey-a.xml",
                    "kind": "xml",
                    "title": "Prediction XML: survey-a.jpg",
                    "mime_type": "application/xml",
                    "category": "prediction",
                },
                {
                    "path": "overlay-a.png",
                    "kind": "image",
                    "title": "Overlay: survey-a.jpg",
                    "mime_type": "image/png",
                    "category": "overlay",
                    "source_resource_id": "file_a",
                },
            ],
        }

    registered: list[tuple[str, list[dict], str]] = []

    def fake_register(*, control_base_url, job_id, principal_headers, outputs, timeout, collection_id=""):
        registered.append((job_id, outputs, collection_id))
        return {"count": len(outputs)}

    monkeypatch.setattr(proc, "_resolve_source_path", fake_resolve)
    monkeypatch.setattr(proc, "run_rarespot_inference", fake_infer, raising=False)
    monkeypatch.setattr(proc, "register_outputs", fake_register)
    monkeypatch.setattr(proc, "fetch_job", lambda **kwargs: {})

    progress_calls: list[dict] = []

    async def progress(**kwargs):
        progress_calls.append(kwargs)

    job = DataAgentJobEnvelope(
        job_id="jobR",
        job_type="analysis.rarespot",
        resource_ids=("file_a", "file_missing"),
        resource_count=2,
        input_selector={"params": {"imgsz": 640, "conf_threshold": 0.4, "stability": False}},
        metadata={
            "results_collection_id": "colR",
            "thread_id": "threadR",
            "params": {"tile_overlap": 0.5, "iou_threshold": 0.3, "spectral": False},
        },
    )

    summary = asyncio.run(AnalysisProcessor(settings)(job, progress))

    assert len(inference_calls) == 1
    call = inference_calls[0]
    assert call["image_paths"] == [img_a]
    assert call["run_id"] == "jobR"
    assert call["thread_id"] == "threadR"
    config = call["config"]
    assert config.tile_size == 640
    assert config.tile_overlap == 0.5
    assert config.conf == 0.4
    assert config.iou == 0.3
    assert config.spectral is False
    assert config.stability is False

    assert summary["summary_kind"] == "batch_inference"
    assert summary["model"] == "rarespot"
    assert summary["processed"] == 1
    assert summary["failed"] == 1
    assert summary["total"] == 2
    assert summary["results_collection_id"] == "colR"
    assert summary["counts_by_class"] == {"burrow": 1, "prairie_dog": 2}
    assert summary["detections_count"] == 3
    assert summary["items"]["file_a"]["status"] == "done"
    assert summary["items"]["file_a"]["detections"] == 3
    assert summary["items"]["file_missing"]["status"] == "failed"

    assert len(registered) == 1
    job_id, outputs, collection_id = registered[0]
    assert job_id == "jobR"
    assert collection_id == "colR"
    assert len(outputs) == 3
    assert {output["artifact_kind"] for output in outputs} == {"json", "xml", "image"}
    assert all(output["storage_path"].startswith("analysis/jobR/rarespot/") for output in outputs)
    assert all((settings.upload_root / output["storage_path"]).exists() for output in outputs)
    aggregate = next(output for output in outputs if output["artifact_kind"] == "json")
    xml_output = next(output for output in outputs if output["artifact_kind"] == "xml")
    overlay = next(output for output in outputs if output["artifact_kind"] == "image")
    assert "source_resource_id" not in aggregate
    assert xml_output["source_resource_id"] == "file_a"
    assert overlay["source_resource_id"] == "file_a"
    assert overlay["metadata"]["title"] == "Overlay: survey-a.jpg"
    assert overlay["metadata"]["category"] == "overlay"
    assert overlay["metadata"]["kind"] == "image"
    assert overlay["metadata"]["mime_type"] == "image/png"
    assert overlay["metadata"]["size_bytes"] == 3
    assert overlay["metadata"]["sha256"] == overlay["sha256"]
    assert all("source_path" not in output["metadata"] for output in outputs)
    assert all("storage_uri" not in output["metadata"] for output in outputs)
    assert progress_calls and progress_calls[-1]["output_summary"]["model"] == "rarespot"


def test_rarespot_batch_maps_real_overlay_artifacts_by_prediction_index(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    img_a = settings.upload_root / "file_a__survey-a.jpg"
    img_b = settings.upload_root / "file_b__survey-b.jpg"
    img_a.write_bytes(b"A")
    img_b.write_bytes(b"B")

    def fake_resolve(file_id, upload_roots):
        return {"file_a": img_a, "file_b": img_b}.get(file_id)

    def fake_infer(*, image_paths, run_id, thread_id, output_dir, config, progress_callback):
        out = Path(output_dir)
        (out / "prediction_xml").mkdir(parents=True, exist_ok=True)
        (out / "overlays").mkdir(parents=True, exist_ok=True)
        (out / "predictions.json").write_text("{}", encoding="utf-8")
        (out / "prediction_xml" / "survey-a.xml").write_text("<xml>a</xml>", encoding="utf-8")
        (out / "prediction_xml" / "survey-b.xml").write_text("<xml>b</xml>", encoding="utf-8")
        for name in (
            "0000-survey-a.png",
            "0000-survey-a-stability.png",
            "0001-survey-b.png",
            "0001-survey-b-stability.png",
        ):
            (out / "overlays" / name).write_bytes(name.encode("utf-8"))
        progress_callback({"event": "inference_completed", "detections": 4})
        return {
            "counts_by_class": {"burrow": 1, "prairie_dog": 3},
            "predictions": [
                {
                    "input_path": str(img_a),
                    "prediction_xml_path": str(out / "prediction_xml" / "survey-a.xml"),
                    "class_counts": {"prairie_dog": 2},
                },
                {
                    "input_path": str(img_b),
                    "prediction_xml_path": str(out / "prediction_xml" / "survey-b.xml"),
                    "class_counts": {"burrow": 1, "prairie_dog": 1},
                },
            ],
            "artifacts": [
                {
                    "path": "predictions.json",
                    "kind": "json",
                    "title": "RareSpot predictions",
                    "mime_type": "application/json",
                    "category": "prediction",
                },
                {
                    "path": "prediction_xml/survey-a.xml",
                    "kind": "xml",
                    "title": "Prediction XML: survey-a.jpg",
                    "mime_type": "application/xml",
                    "category": "prediction",
                },
                {
                    "path": "prediction_xml/survey-b.xml",
                    "kind": "xml",
                    "title": "Prediction XML: survey-b.jpg",
                    "mime_type": "application/xml",
                    "category": "prediction",
                },
                *[
                    {
                        "path": f"overlays/{name}",
                        "kind": "image",
                        "title": f"Overlay: {name}",
                        "mime_type": "image/png",
                        "category": "overlay",
                    }
                    for name in (
                        "0000-survey-a.png",
                        "0000-survey-a-stability.png",
                        "0001-survey-b.png",
                        "0001-survey-b-stability.png",
                    )
                ],
            ],
        }

    registered: list[tuple[str, list[dict], str]] = []

    def fake_register(*, control_base_url, job_id, principal_headers, outputs, timeout, collection_id=""):
        registered.append((job_id, outputs, collection_id))
        return {"count": len(outputs)}

    monkeypatch.setattr(proc, "_resolve_source_path", fake_resolve)
    monkeypatch.setattr(proc, "run_rarespot_inference", fake_infer, raising=False)
    monkeypatch.setattr(proc, "register_outputs", fake_register)
    monkeypatch.setattr(proc, "fetch_job", lambda **kwargs: {})

    async def progress(**kwargs):
        return None

    job = DataAgentJobEnvelope(
        job_id="jobR-overlays",
        job_type="analysis.rarespot",
        resource_ids=("file_a", "file_b"),
        resource_count=2,
        metadata={"results_collection_id": "colR"},
    )

    summary = asyncio.run(AnalysisProcessor(settings)(job, progress))

    assert summary["processed"] == 2
    assert summary["items"]["file_a"]["outputs"] == 3
    assert summary["items"]["file_b"]["outputs"] == 3
    assert len(registered) == 1
    outputs = registered[0][1]
    assert len(outputs) == 7
    aggregate = next(output for output in outputs if output["artifact_kind"] == "json")
    assert "source_resource_id" not in aggregate
    overlay_sources = {
        output["original_name"]: output.get("source_resource_id")
        for output in outputs
        if output["artifact_kind"] == "image"
    }
    assert overlay_sources == {
        "0000-survey-a.png": "file_a",
        "0000-survey-a-stability.png": "file_a",
        "0001-survey-b.png": "file_b",
        "0001-survey-b-stability.png": "file_b",
    }


def test_rarespot_batch_returns_failed_summary_without_inference_when_inputs_missing(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    inference_called = False
    registered: list[dict] = []

    def fake_infer(**_kwargs):
        nonlocal inference_called
        inference_called = True
        raise AssertionError("inference should not run without resolved inputs")

    monkeypatch.setattr(proc, "_resolve_source_path", lambda file_id, upload_roots: None)
    monkeypatch.setattr(proc, "run_rarespot_inference", fake_infer, raising=False)
    monkeypatch.setattr(proc, "register_outputs", lambda **kwargs: registered.append(kwargs))
    monkeypatch.setattr(proc, "fetch_job", lambda **kwargs: {})

    async def progress(**kwargs):
        return None

    job = DataAgentJobEnvelope(
        job_id="jobR-missing",
        job_type="analysis.rarespot",
        resource_ids=("missing_a", "missing_b"),
        resource_count=2,
        metadata={"results_collection_id": "colR"},
    )

    summary = asyncio.run(AnalysisProcessor(settings)(job, progress))

    assert not inference_called
    assert registered == []
    assert summary["model"] == "rarespot"
    assert summary["processed"] == 0
    assert summary["failed"] == 2
    assert summary["total"] == 2
    assert summary["terminal_status"] == "failed"
    assert summary["items"]["missing_a"]["status"] == "failed"
    assert summary["items"]["missing_b"]["status"] == "failed"


def test_rarespot_control_plane_calls_include_worker_token(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    settings.control_worker_token = "analysis-worker-secret"
    img_a = settings.upload_root / "file_a__survey-a.jpg"
    img_a.write_bytes(b"A")

    def fake_infer(*, image_paths, run_id, thread_id, output_dir, config, progress_callback):
        out = Path(output_dir)
        (out / "predictions.json").write_text("{}", encoding="utf-8")
        progress_callback({"event": "done"})
        return {
            "counts_by_class": {},
            "predictions": [{"input_path": str(img_a), "class_counts": {}}],
            "artifacts": [
                {
                    "path": "predictions.json",
                    "kind": "json",
                    "mime_type": "application/json",
                }
            ],
        }

    seen_fetch_headers: list[dict[str, str]] = []
    seen_register_headers: list[dict[str, str]] = []

    def fake_fetch(*, principal_headers, **_kwargs):
        seen_fetch_headers.append(dict(principal_headers))
        return {}

    def fake_register(*, principal_headers, **_kwargs):
        seen_register_headers.append(dict(principal_headers))
        return {"count": 1}

    monkeypatch.setattr(proc, "_resolve_source_path", lambda file_id, upload_roots: img_a)
    monkeypatch.setattr(proc, "run_rarespot_inference", fake_infer, raising=False)
    monkeypatch.setattr(proc, "fetch_job", fake_fetch)
    monkeypatch.setattr(proc, "register_outputs", fake_register)

    async def progress(**kwargs):
        return None

    job = DataAgentJobEnvelope(
        job_id="jobR-token",
        job_type="analysis.rarespot",
        owner_user_id="owner-a",
        owner_org_id="org-a",
        resource_ids=("file_a",),
        resource_count=1,
    )

    asyncio.run(AnalysisProcessor(settings)(job, progress))

    assert seen_fetch_headers
    assert seen_register_headers
    assert all(headers["X-Ultra-Worker-Token"] == "analysis-worker-secret" for headers in seen_fetch_headers)
    assert all(headers["X-Ultra-Worker-Token"] == "analysis-worker-secret" for headers in seen_register_headers)


def test_rarespot_canceled_after_progress_callback_skips_staging_and_registration(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    img_a = settings.upload_root / "file_a__survey-a.jpg"
    img_a.write_bytes(b"A")
    progress_seen = False
    registered: list[dict[str, object]] = []

    def fake_fetch(**_kwargs):
        return {"status": "canceled"} if progress_seen else {}

    def fake_infer(*, image_paths, run_id, thread_id, output_dir, config, progress_callback):
        nonlocal progress_seen
        out = Path(output_dir)
        progress_callback({"event": "tile_completed"})
        progress_seen = True
        (out / "predictions.json").write_text("{}", encoding="utf-8")
        return {
            "counts_by_class": {},
            "predictions": [{"input_path": str(img_a), "class_counts": {}}],
            "artifacts": [
                {
                    "path": "predictions.json",
                    "kind": "json",
                    "mime_type": "application/json",
                }
            ],
        }

    monkeypatch.setattr(proc, "_resolve_source_path", lambda file_id, upload_roots: img_a)
    monkeypatch.setattr(proc, "fetch_job", fake_fetch)
    monkeypatch.setattr(proc, "run_rarespot_inference", fake_infer, raising=False)
    monkeypatch.setattr(proc, "register_outputs", lambda **kwargs: registered.append(kwargs))

    async def progress(**kwargs):
        return None

    job = DataAgentJobEnvelope(
        job_id="jobR-cancel-after-progress",
        job_type="analysis.rarespot",
        resource_ids=("file_a",),
        resource_count=1,
    )

    summary = asyncio.run(AnalysisProcessor(settings)(job, progress))

    assert summary["canceled"] is True
    assert summary["terminal_status"] == "canceled"
    assert registered == []
    assert not (settings.upload_root / "analysis" / "jobR-cancel-after-progress" / "rarespot").exists()


def test_rarespot_checks_cancellation_on_every_progress_callback(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    img_a = settings.upload_root / "file_a__survey-a.jpg"
    img_a.write_bytes(b"A")
    progress_callbacks = 0
    registered: list[dict[str, object]] = []

    def fake_fetch(**_kwargs):
        return {"status": "canceled"} if progress_callbacks >= 1 else {}

    def fake_infer(*, image_paths, run_id, thread_id, output_dir, config, progress_callback):
        nonlocal progress_callbacks
        progress_callback({"event": "first"})
        progress_callbacks += 1
        progress_callback({"event": "second"})
        (Path(output_dir) / "predictions.json").write_text("{}", encoding="utf-8")
        return {"counts_by_class": {}, "predictions": [], "artifacts": []}

    monkeypatch.setattr(proc, "_resolve_source_path", lambda file_id, upload_roots: img_a)
    monkeypatch.setattr(proc, "fetch_job", fake_fetch)
    monkeypatch.setattr(proc, "run_rarespot_inference", fake_infer, raising=False)
    monkeypatch.setattr(proc, "register_outputs", lambda **kwargs: registered.append(kwargs))

    async def progress(**kwargs):
        return None

    job = DataAgentJobEnvelope(
        job_id="jobR-cancel-every-callback",
        job_type="analysis.rarespot",
        resource_ids=("file_a",),
        resource_count=1,
    )

    summary = asyncio.run(AnalysisProcessor(settings)(job, progress))

    assert summary["canceled"] is True
    assert summary["terminal_status"] == "canceled"
    assert registered == []
    assert not (settings.upload_root / "analysis" / "jobR-cancel-every-callback" / "rarespot").exists()


def test_rarespot_inference_failure_without_prior_success_is_terminal_failed(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    img_a = settings.upload_root / "file_a__survey-a.jpg"
    img_a.write_bytes(b"A")

    def fake_infer(**_kwargs):
        raise RuntimeError("rarespot crashed")

    monkeypatch.setattr(proc, "_resolve_source_path", lambda file_id, upload_roots: img_a)
    monkeypatch.setattr(proc, "fetch_job", lambda **kwargs: {})
    monkeypatch.setattr(proc, "run_rarespot_inference", fake_infer, raising=False)

    async def progress(**kwargs):
        return None

    job = DataAgentJobEnvelope(
        job_id="jobR-fatal",
        job_type="analysis.rarespot",
        resource_ids=("file_a",),
        resource_count=1,
    )

    summary = asyncio.run(AnalysisProcessor(settings)(job, progress))

    assert summary["processed"] == 0
    assert summary["failed"] == 1
    assert summary["terminal_status"] == "failed"
    assert summary["items"]["file_a"]["status"] == "failed"


def test_rarespot_registration_failure_without_prior_success_is_terminal_failed(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    img_a = settings.upload_root / "file_a__survey-a.jpg"
    img_a.write_bytes(b"A")

    def fake_infer(*, image_paths, run_id, thread_id, output_dir, config, progress_callback):
        out = Path(output_dir)
        (out / "predictions.json").write_text("{}", encoding="utf-8")
        return {
            "counts_by_class": {},
            "predictions": [{"input_path": str(img_a), "class_counts": {}}],
            "artifacts": [
                {
                    "path": "predictions.json",
                    "kind": "json",
                    "mime_type": "application/json",
                }
            ],
        }

    monkeypatch.setattr(proc, "_resolve_source_path", lambda file_id, upload_roots: img_a)
    monkeypatch.setattr(proc, "fetch_job", lambda **kwargs: {})
    monkeypatch.setattr(proc, "run_rarespot_inference", fake_infer, raising=False)
    monkeypatch.setattr(proc, "register_outputs", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("register down")))

    async def progress(**kwargs):
        return None

    job = DataAgentJobEnvelope(
        job_id="jobR-register-fatal",
        job_type="analysis.rarespot",
        resource_ids=("file_a",),
        resource_count=1,
    )

    summary = asyncio.run(AnalysisProcessor(settings)(job, progress))

    assert summary["processed"] == 0
    assert summary["failed"] == 1
    assert summary["terminal_status"] == "failed"
    assert summary["items"]["file_a"]["status"] == "failed"


def test_megaseg_rejects_unsafe_job_id_before_staging_outside_upload_root(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    img_a = settings.upload_root / "file_a__a.tif"
    img_a.write_bytes(b"A")
    registered: list[dict[str, object]] = []

    def fake_infer(*, service_url, api_key, image_path, params, timeout, dest_dir):
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "mask.tif").write_bytes(b"M")
        return {"checkpoint_path": "/m/e.ckpt", "files": [{"mask_path": "mask.tif"}]}

    monkeypatch.setattr(proc, "_resolve_source_path", lambda file_id, upload_roots: img_a)
    monkeypatch.setattr(proc, "run_megaseg_infer", fake_infer)
    monkeypatch.setattr(proc, "fetch_job", lambda **kwargs: {})
    monkeypatch.setattr(proc, "register_outputs", lambda **kwargs: registered.append(kwargs))

    async def progress(**kwargs):
        return None

    job = DataAgentJobEnvelope(
        job_id="../../escape",
        job_type="analysis.megaseg",
        resource_ids=("file_a",),
        resource_count=1,
    )

    summary = asyncio.run(AnalysisProcessor(settings)(job, progress))

    assert summary["failed"] == 1
    assert registered == []
    assert not (tmp_path / "escape").exists()


def test_rarespot_rejects_unsafe_job_id_before_staging_outside_upload_root(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    img_a = settings.upload_root / "file_a__survey-a.jpg"
    img_a.write_bytes(b"A")
    registered: list[dict[str, object]] = []

    def fake_infer(*, image_paths, run_id, thread_id, output_dir, config, progress_callback):
        out = Path(output_dir)
        (out / "predictions.json").write_text("{}", encoding="utf-8")
        return {
            "counts_by_class": {},
            "predictions": [{"input_path": str(img_a), "class_counts": {}}],
            "artifacts": [
                {
                    "path": "predictions.json",
                    "kind": "json",
                    "mime_type": "application/json",
                }
            ],
        }

    monkeypatch.setattr(proc, "_resolve_source_path", lambda file_id, upload_roots: img_a)
    monkeypatch.setattr(proc, "fetch_job", lambda **kwargs: {})
    monkeypatch.setattr(proc, "run_rarespot_inference", fake_infer, raising=False)
    monkeypatch.setattr(proc, "register_outputs", lambda **kwargs: registered.append(kwargs))

    async def progress(**kwargs):
        return None

    job = DataAgentJobEnvelope(
        job_id="../../escape",
        job_type="analysis.rarespot",
        resource_ids=("file_a",),
        resource_count=1,
    )

    summary = asyncio.run(AnalysisProcessor(settings)(job, progress))

    assert summary["failed"] == 1
    assert summary["terminal_status"] == "failed"
    assert registered == []
    assert not (tmp_path / "escape").exists()


def test_safe_extract_rejects_too_many_members(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analysis_client, "MAX_EXTRACTED_TAR_MEMBERS", 1)
    archive_path = tmp_path / "result.tar.gz"
    dest_dir = tmp_path / "dest"
    payload = b"x"

    with tarfile.open(archive_path, "w:gz") as archive:
        root = tarfile.TarInfo("safe")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        member = tarfile.TarInfo("safe/a.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with tarfile.open(archive_path, "r:gz") as archive:
        monkeypatch.setattr(
            archive,
            "getmembers",
            lambda: (_ for _ in ()).throw(AssertionError("getmembers should not be used")),
        )
        with pytest.raises(RuntimeError, match="too many"):
            _safe_extract(archive, dest_dir)

    assert not (dest_dir / "safe" / "a.txt").exists()


def test_safe_extract_rejects_oversize_uncompressed_members(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analysis_client, "MAX_EXTRACTED_TAR_UNCOMPRESSED_BYTES", 3)
    archive_path = tmp_path / "result.tar.gz"
    dest_dir = tmp_path / "dest"
    payload = b"abcd"

    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("safe/a.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with tarfile.open(archive_path, "r:gz") as archive:
        with pytest.raises(RuntimeError, match="uncompressed"):
            _safe_extract(archive, dest_dir)

    assert not (dest_dir / "safe" / "a.txt").exists()


@pytest.mark.parametrize(
    ("link_type", "link_name"),
    [
        (tarfile.SYMTYPE, "safe/link"),
        (tarfile.LNKTYPE, "safe/hardlink"),
    ],
)
def test_safe_extract_rejects_link_entries_that_escape(
    tmp_path: Path,
    link_type: bytes,
    link_name: str,
) -> None:
    archive_path = tmp_path / "result.tar.gz"
    dest_dir = tmp_path / "dest"
    outside_dir = tmp_path / "outside"
    dest_dir.mkdir()
    outside_dir.mkdir()
    (outside_dir / "target.txt").write_bytes(b"outside")

    with tarfile.open(archive_path, "w:gz") as archive:
        root = tarfile.TarInfo("safe")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)

        link = tarfile.TarInfo(link_name)
        link.type = link_type
        link.linkname = "../outside/target.txt"
        archive.addfile(link)

        if link_type == tarfile.SYMTYPE:
            payload = b"escaped"
            nested = tarfile.TarInfo("safe/link/escape.txt")
            nested.size = len(payload)
            archive.addfile(nested, io.BytesIO(payload))

    with tarfile.open(archive_path, "r:gz") as archive:
        with pytest.raises(RuntimeError):
            _safe_extract(archive, dest_dir)

    assert not (outside_dir / "escape.txt").exists()
