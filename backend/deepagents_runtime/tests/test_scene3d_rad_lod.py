from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from ultra_deepagents.imaging.derivative_manifest import TransientDerivativeError
from ultra_deepagents.scene3d import rad_lod


def test_build_paged_rad_uses_fixed_names_without_mutating_source_generation(tmp_path, monkeypatch):
    source = tmp_path / "private-upload-name.ply"
    source.write_bytes(b"ply\n")
    source_before = source.stat()
    destination = tmp_path / "derived"
    destination.mkdir()
    monkeypatch.setattr(rad_lod, "_resolved_executable", lambda: sys.executable)

    def fake_run(command, **kwargs):
        assert command == [
            sys.executable,
            "--quality",
            "--rad-chunked",
            "--max-sh=2",
            "scene.ply",
        ]
        assert kwargs["cwd"] == str(destination)
        staged = destination / "scene.ply"
        assert staged.read_bytes() == source.read_bytes()
        assert os.stat(source).st_ino != os.stat(staged).st_ino
        (destination / "scene-lod.rad").write_bytes(b"RAD header")
        (destination / "scene-lod-1.radc").write_bytes(b"second")
        (destination / "scene-lod-0.radc").write_bytes(b"first")
        return SimpleNamespace(returncode=0, stdout="ok")

    result = rad_lod.build_paged_rad(
        str(source), str(destination), retained_sh_degree=2, run_fn=fake_run
    )

    assert result.header == rad_lod.RadArtifact("scene-lod.rad", 10)
    assert [artifact.name for artifact in result.chunks] == [
        "scene-lod-0.radc",
        "scene-lod-1.radc",
    ]
    assert result.method == "bhatt-lod-quality"
    assert not (destination / "scene.ply").exists()
    source_after = source.stat()
    assert source_after.st_ino == source_before.st_ino
    assert source_after.st_ctime_ns == source_before.st_ctime_ns


def test_build_paged_rad_fails_closed_when_converter_publishes_no_pages(tmp_path, monkeypatch):
    source = tmp_path / "scene.ply"
    source.write_bytes(b"ply\n")
    destination = tmp_path / "derived"
    destination.mkdir()
    monkeypatch.setattr(rad_lod, "_resolved_executable", lambda: sys.executable)

    def fake_run(_command, **kwargs):
        Path(kwargs["cwd"], "scene-lod.rad").write_bytes(b"RAD header")
        return SimpleNamespace(returncode=0, stdout="incomplete")

    with pytest.raises(TransientDerivativeError, match="scene_lod_output_missing"):
        rad_lod.build_paged_rad(
            str(source), str(destination), retained_sh_degree=0, run_fn=fake_run
        )
    assert not (destination / "scene.ply").exists()


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("scene-lod.rad", True),
        ("scene-lod-0.radc", True),
        ("scene-lod-42.radc", True),
        ("scene-lod-00.radc", False),
        ("../scene-lod.rad", False),
        ("scene-lod-1.bin", False),
    ],
)
def test_rad_artifact_names_are_canonical(name, expected):
    assert rad_lod.is_rad_artifact_name(name) is expected
