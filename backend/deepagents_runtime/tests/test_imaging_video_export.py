from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from ultra_deepagents.imaging import video_export
from ultra_deepagents.imaging.video_export import run_video_export_job
from ultra_deepagents.imaging.worker import _extract_supported_payload


def _recipe(resource_id: str, source_sha256: str, source_size: int) -> dict[str, Any]:
    return {
        "schema": "ultra.image-video-export-recipe.v1",
        "renderer_revision": "ultra.slice-video-renderer.v1",
        "resource_id": resource_id,
        "source": {"sha256": source_sha256, "size_bytes": source_size},
        "axes": {"T": 2, "C": 3, "Z": 4},
        "mode": "z_sweep",
        "profile": "complete",
        "fps": 24,
        "source_frame_count": 4,
        "frame_indices": [0, 1, 2, 3],
        "fixed_z": 1,
        "fixed_t": 1,
        "strict_scalar_slice": False,
        "channels": [0, 2],
        "channel_colors": ["#ff0000", "#00ffff"],
        "enhancement": "",
        "negative": False,
        "scalar_render_mode": "intensity",
        "scalar_threshold_value": None,
        "scalar_threshold_foreground": "",
        "full_resolution": False,
        "max_frame_edge": 1024,
    }


def _job(tmp_path: Path) -> dict[str, Any]:
    source = tmp_path / "source.ome.tiff"
    source.write_bytes(b"immutable-source")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    recipe = _recipe("file_stack", source_sha256, source.stat().st_size)
    render_id = hashlib.sha256(
        json.dumps(recipe, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    derived = tmp_path / "derived"
    base = f"file_stack__video.{render_id}"
    return {
        "resource_id": "file_stack",
        "src_path": str(source),
        "source_sha256": source_sha256,
        "source_size_bytes": source.stat().st_size,
        "render_id": render_id,
        "recipe": recipe,
        "output_path": str(derived / f"{base}.mp4"),
        "manifest_path": str(derived / f"{base}.manifest.json"),
        "queued_path": str(derived / f"{base}.queued.json"),
        "progress_path": str(derived / f"{base}.progress.json"),
        "failed_path": str(derived / f"{base}.failed.json"),
        "owner_role": "researcher",
        "owner_user_id": "user-a",
        "owner_org_id": "org-a",
        "force_id": "vidjob_1",
    }


def test_worker_routes_typed_video_job_and_preserves_owner() -> None:
    envelope = {
        "job_id": "vidjob_1",
        "job_type": "image.render_video",
        "owner_user_id": "user-a",
        "owner_org_id": "org-a",
        "metadata": {"src_path": "/data/source", "owner_role": "researcher"},
    }
    extracted = _extract_supported_payload(envelope)
    assert extracted is not None
    job_type, payload = extracted
    assert job_type == "image.render_video"
    assert payload["owner_user_id"] == "user-a"
    assert payload["owner_org_id"] == "org-a"
    assert payload["force_id"] == "vidjob_1"


def test_video_recipe_identity_rejects_mutated_frame_schedule(tmp_path: Path) -> None:
    job = _job(tmp_path)
    # Keep the recipe semantically valid while changing one source selector, so
    # this exercises the immutable identity check rather than schema validation.
    job["recipe"]["fixed_t"] = 0
    with pytest.raises(video_export.DeterministicDerivativeError) as raised:
        video_export._validate_job(job)
    assert raised.value.code == "video_recipe_identity_mismatch"


def test_frame_url_freezes_selected_channels_time_and_depth() -> None:
    recipe = _recipe("file_stack", "a" * 64, 10)
    url = video_export._frame_url(
        "http://control-plane:8000", "file_stack", recipe, 3, "b" * 64
    )
    assert url.startswith(
        "http://control-plane:8000/v2/internal/uploads/file_stack/render-frame?"
    )
    assert "z=3" in url
    assert "t=1" in url
    assert "channels=0%2C2" in url
    assert "channel_colors=%23ff0000%2C%2300ffff" in url
    assert "full_resolution=false" in url


class _Sink(io.BytesIO):
    def close(self) -> None:
        # The fake encoder does not need the bytes after close; keep the sink writable
        # so cleanup can safely call close more than once.
        pass


class _FakeEncoder:
    def __init__(self, command: list[str], **_: Any) -> None:
        self.output = Path(command[-1])
        self.stdin = _Sink()

    def wait(self, timeout: int) -> int:
        assert timeout == 300
        self.output.write_bytes(b"fake-mp4")
        return 0

    def kill(self) -> None:
        return None


def test_video_job_publishes_manifest_last_with_exact_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job(tmp_path)
    # Minimal PNG header carrying a stable 7x5 IHDR geometry. The encoder is mocked;
    # frame bytes need only exercise the transport/geometry contract here.
    frame = (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + (7).to_bytes(4, "big")
        + (5).to_bytes(4, "big")
        + b"\x08\x06\x00\x00\x00"
    )
    monkeypatch.setenv("ULTRA_CONTROL_BASE_URL", "http://control-plane:8000")
    monkeypatch.setenv("ULTRA_CONTROL_WORKER_TOKEN", "worker-secret")
    monkeypatch.setattr(video_export, "_fetch_frame", lambda *_args, **_kwargs: frame)
    monkeypatch.setattr(video_export.subprocess, "Popen", _FakeEncoder)

    result = run_video_export_job(job)
    assert result["status"] == "ready"
    manifest = json.loads(Path(job["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["schema"] == "ultra.image-video-export-manifest.v1"
    assert manifest["render_id"] == job["render_id"]
    assert manifest["recipe"] == job["recipe"]
    assert manifest["artifact"]["width"] == 8
    assert manifest["artifact"]["height"] == 6
    assert manifest["artifact"]["frame_count"] == 4
    assert Path(job["output_path"]).read_bytes() == b"fake-mp4"
    assert not Path(job["progress_path"]).exists()
