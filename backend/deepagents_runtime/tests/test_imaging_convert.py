"""Unit tests for the pyramid-conversion command builders (no engine binary needed)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from ultra_deepagents.imaging.convert import (
    ConversionDependencyError,
    ConversionInputError,
    ConversionProcessError,
    ConversionResourceError,
    PyramidSpec,
    convert_command,
    pyramid_options,
)


def test_pyramid_options_default():
    assert pyramid_options() == "compression lzw tiles 512 pyramid topdirs"


def test_pyramid_options_custom():
    assert pyramid_options(256, "zip", "topdirs") == "compression zip tiles 256 pyramid topdirs"


def test_convert_command_default_spec():
    assert convert_command("/data/a.czi", "/derived/a.tif", imgcnv_bin="imgcnv") == [
        "imgcnv",
        "-i",
        "/data/a.czi",
        "-o",
        "/derived/a.tif",
        "-t",
        "bigtiff",
        "-options",
        "compression lzw tiles 512 pyramid topdirs",
    ]


def test_convert_command_custom_spec():
    spec = PyramidSpec(tile_size=1024, compression="zip", layout="topdirs", fmt="bigtiff")
    cmd = convert_command("a.svs", "a.tif", spec=spec, imgcnv_bin="/usr/local/bin/imgcnv")
    assert cmd[:6] == ["/usr/local/bin/imgcnv", "-i", "a.svs", "-o", "a.tif", "-t"]
    assert cmd[6] == "bigtiff"
    assert cmd[-1] == "compression zip tiles 1024 pyramid topdirs"


def test_derive_pyramid_creates_destination_directory(tmp_path):
    # imgcnv won't mkdir the output dir; derive_pyramid must, or it fails with
    # "Cannot write into" on a fresh derived/ (regression: first-upload in a clean
    # docker volume). Use `true` as a stand-in engine so no native lib is needed.
    src = tmp_path / "src.tif"
    src.write_bytes(b"x")
    dst = tmp_path / "derived" / "out.tif"
    from ultra_deepagents.imaging.convert import derive_pyramid

    result = derive_pyramid(str(src), str(dst), imgcnv_bin="true")
    assert result.ok
    assert dst.parent.is_dir()


def test_derive_pyramid_classifies_missing_dependency(tmp_path):
    from ultra_deepagents.imaging.convert import derive_pyramid

    with pytest.raises(ConversionDependencyError):
        derive_pyramid("source.tif", tmp_path / "out.tif", imgcnv_bin="definitely-not-imgcnv")


def test_derive_pyramid_classifies_killed_process_as_resource_failure(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from ultra_deepagents.imaging import convert

    monkeypatch.setattr(
        convert.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=137,
            stdout="",
            stderr="killed",
        ),
    )
    with pytest.raises(ConversionResourceError):
        convert.derive_pyramid("source.tif", tmp_path / "out.tif")


def test_derive_pyramid_classifies_stable_imgcnv_input_diagnostic_and_bounds_details(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from ultra_deepagents.imaging import convert

    monkeypatch.setattr(
        convert.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="Input format is not supported\n" + ("x" * 10_000),
            stderr="details\n" + ("y" * 10_000),
        ),
    )

    with pytest.raises(ConversionInputError) as excinfo:
        convert.derive_pyramid("source.invalid", tmp_path / "out.tif")

    assert excinfo.value.returncode == 1
    assert len(excinfo.value.stdout) <= 4096
    assert len(excinfo.value.stderr) <= 4096
    assert "Input format is not supported" in excinfo.value.stdout


def test_derive_pyramid_keeps_unknown_nonzero_diagnostic_retryable(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from ultra_deepagents.imaging import convert

    monkeypatch.setattr(
        convert.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="unsupported codec handshake failed",
        ),
    )

    with pytest.raises(ConversionProcessError):
        convert.derive_pyramid("source.tif", tmp_path / "out.tif")


def test_pyramidspec_options_roundtrip():
    assert PyramidSpec(tile_size=128).options() == "compression lzw tiles 128 pyramid topdirs"


@pytest.mark.parametrize(
    "call",
    [
        lambda: pyramid_options(0),
        lambda: pyramid_options(-512),
        lambda: pyramid_options(512, "bogus"),
        lambda: pyramid_options(512, "lzw", "sideways"),
    ],
)
def test_invalid_options_raise(call):
    with pytest.raises(ValueError):
        call()
