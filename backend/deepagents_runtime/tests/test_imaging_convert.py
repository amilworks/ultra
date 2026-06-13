"""Unit tests for the pyramid-conversion command builders (no engine binary needed)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from ultra_deepagents.imaging.convert import (
    PyramidSpec,
    convert_command,
    meta_command,
    pyramid_options,
)


def test_pyramid_options_default():
    assert pyramid_options() == "compression lzw tiles 512 pyramid topdirs"


def test_pyramid_options_custom():
    assert pyramid_options(256, "zip", "topdirs") == "compression zip tiles 256 pyramid topdirs"


def test_convert_command_default_spec():
    assert convert_command("/data/a.czi", "/derived/a.tif", imgcnv_bin="imgcnv") == [
        "imgcnv", "-i", "/data/a.czi", "-o", "/derived/a.tif",
        "-t", "bigtiff", "-options", "compression lzw tiles 512 pyramid topdirs",
    ]


def test_convert_command_custom_spec():
    spec = PyramidSpec(tile_size=1024, compression="zip", layout="topdirs", fmt="bigtiff")
    cmd = convert_command("a.svs", "a.tif", spec=spec, imgcnv_bin="/usr/local/bin/imgcnv")
    assert cmd[:6] == ["/usr/local/bin/imgcnv", "-i", "a.svs", "-o", "a.tif", "-t"]
    assert cmd[6] == "bigtiff"
    assert cmd[-1] == "compression zip tiles 1024 pyramid topdirs"


def test_meta_command():
    assert meta_command("a.nd2", imgcnv_bin="imgcnv") == ["imgcnv", "-i", "a.nd2", "-meta", "-verbose", "0"]


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
