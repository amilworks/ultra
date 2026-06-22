"""Unit tests for the bioio transcode fallback (no bioio/native engine needed)."""

from __future__ import annotations

import builtins
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from ultra_deepagents.imaging.transcode import TranscodeError, TranscodeResult, transcode_to_ome_tiff


def test_transcode_raises_when_bioio_missing(monkeypatch, tmp_path):
    # When bioio isn't installed (or fails to import), transcode must raise a clean
    # TranscodeError so the worker records a permanent-failure marker and the viewer
    # degrades to the "preview unavailable" card — not crash the convert loop.
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "bioio" or name.startswith("bioio."):
            raise ImportError("bioio not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(TranscodeError):
        transcode_to_ome_tiff(str(tmp_path / "x.lif"), str(tmp_path / "out.ome.tif"))


def test_transcode_result_multichannel_or_volume_flag():
    # Drives the fmt choice: any multichannel OR z-stack series must derive to
    # OME-BigTIFF (plain BigTIFF would flatten channels/planes).
    flat = TranscodeResult("p", 1, 0, "s", num_c=1, num_z=1, dtype="uint8", series_names=["s"])
    multichannel = TranscodeResult("p", 1, 0, "s", num_c=2, num_z=1, dtype="uint8", series_names=["s"])
    volume = TranscodeResult("p", 1, 0, "s", num_c=1, num_z=40, dtype="uint8", series_names=["s"])
    timelapse = TranscodeResult("p", 1, 0, "s", num_c=1, num_z=1, dtype="uint8", series_names=["s"], num_t=61)
    assert flat.is_multichannel_or_volume is False
    assert multichannel.is_multichannel_or_volume is True
    assert volume.is_multichannel_or_volume is True
    # A time-lapse (t>1) must also stay OME-BigTIFF or its frames collapse to one plane.
    assert timelapse.is_multichannel_or_volume is True
