import asyncio
from pathlib import Path

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.rarespot import worker
from ultra_deepagents.rarespot.uploads import resolve_uploaded_file_ids
from ultra_deepagents.rarespot.worker import DefaultRareSpotJobRunner


def test_resolve_uploaded_file_ids_from_upload_root(tmp_path: Path):
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    uploaded_image = upload_root / "file-1__abc123__EnrNE_Day2_Run1__0242.JPG"
    uploaded_image.write_bytes(b"jpeg bytes")

    resolution = asyncio.run(
        resolve_uploaded_file_ids(
            ["file-1"],
            upload_roots=(upload_root,),
            allowed_roots=(tmp_path,),
        )
    )

    assert resolution.image_paths == [uploaded_image.resolve()]
    assert resolution.missing_file_ids == []


def test_default_runner_stages_uploaded_file_ids(monkeypatch, tmp_path: Path):
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    uploaded_image = upload_root / "file-1__abc123__prairie.jpg"
    uploaded_image.write_bytes(b"jpeg bytes")
    artifact_root = tmp_path / "artifacts"
    captured: dict[str, list[str]] = {}

    async def fake_stage_inputs(*, local_paths, resource_uris, dataset_uris, stage_dir, config):
        captured["local_paths"] = list(local_paths)
        return {
            "image_paths": [Path(local_paths[0])],
            "local_image_count": 1,
            "downloaded_image_count": 0,
        }

    def fake_run_rarespot_inference(*, image_paths, run_id, thread_id, output_dir, config):
        return {
            "counts_by_class": {"prairie_dog": 0, "burrow": 0},
            "top_spectral_review_candidates": [],
            "artifacts": [],
        }

    monkeypatch.setattr(worker, "stage_inputs", fake_stage_inputs)
    monkeypatch.setattr(worker, "run_rarespot_inference", fake_run_rarespot_inference)
    settings = RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        rarespot_artifact_root=str(artifact_root),
        rarespot_allowed_input_roots=(str(tmp_path),),
        rarespot_upload_roots=(str(upload_root),),
    )

    result = asyncio.run(
        DefaultRareSpotJobRunner(settings).run(
            {
                "run_id": "run-1",
                "thread_id": "thread-1",
                "artifact_root": str(artifact_root),
                "metadata": {"file_ids": ["file-1"]},
            }
        )
    )

    assert captured["local_paths"] == [str(uploaded_image.resolve())]
    assert result["counts_by_class"] == {"prairie_dog": 0, "burrow": 0}
