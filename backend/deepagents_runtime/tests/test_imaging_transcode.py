"""Unit tests for the bioio transcode fallback (no bioio/native engine needed)."""

from __future__ import annotations

import builtins
import os
import sys
from types import ModuleType, SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from ultra_deepagents.imaging.transcode import (
    TranscodeDependencyError,
    TranscodeError,
    TranscodeOperationalError,
    TranscodeResourceError,
    TranscodeResult,
    transcode_to_ome_tiff,
)


def test_transcode_raises_when_bioio_missing(monkeypatch, tmp_path):
    # Missing runtime dependencies are operational and retryable; they must not
    # manufacture a permanent unsupported-source marker for a valid image.
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "bioio" or name.startswith("bioio."):
            raise ImportError("bioio not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(TranscodeDependencyError):
        transcode_to_ome_tiff(str(tmp_path / "x.lif"), str(tmp_path / "out.ome.tif"))


def test_transcode_result_multichannel_or_volume_flag():
    # Drives the fmt choice: any multichannel OR z-stack series must derive to
    # OME-BigTIFF (plain BigTIFF would flatten channels/planes).
    flat = TranscodeResult("p", 1, 0, "s", num_c=1, num_z=1, dtype="uint8", series_names=["s"])
    multichannel = TranscodeResult(
        "p", 1, 0, "s", num_c=2, num_z=1, dtype="uint8", series_names=["s"]
    )
    volume = TranscodeResult("p", 1, 0, "s", num_c=1, num_z=40, dtype="uint8", series_names=["s"])
    timelapse = TranscodeResult(
        "p", 1, 0, "s", num_c=1, num_z=1, dtype="uint8", series_names=["s"], num_t=61
    )
    assert flat.is_multichannel_or_volume is False
    assert multichannel.is_multichannel_or_volume is True
    assert volume.is_multichannel_or_volume is True
    # A time-lapse (t>1) must also stay OME-BigTIFF or its frames collapse to one plane.
    assert timelapse.is_multichannel_or_volume is True


def _install_fake_bioio(monkeypatch, bio_image):
    bioio_module = ModuleType("bioio")
    bioio_module.BioImage = bio_image
    tifffile_module = ModuleType("tifffile")
    tifffile_module.imwrite = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "bioio", bioio_module)
    monkeypatch.setitem(sys.modules, "tifffile", tifffile_module)


def test_structured_transcode_failures_share_the_base_contract():
    assert issubclass(TranscodeDependencyError, TranscodeError)
    assert issubclass(TranscodeOperationalError, TranscodeError)
    assert issubclass(TranscodeResourceError, TranscodeError)


def test_bioio_unsupported_format_exception_is_retryable_plugin_ambiguity(monkeypatch, tmp_path):
    exceptions_module = ModuleType("bioio_base.exceptions")

    class UnsupportedFileFormatError(Exception):
        pass

    exceptions_module.UnsupportedFileFormatError = UnsupportedFileFormatError
    bioio_base_module = ModuleType("bioio_base")
    bioio_base_module.exceptions = exceptions_module
    monkeypatch.setitem(sys.modules, "bioio_base", bioio_base_module)
    monkeypatch.setitem(sys.modules, "bioio_base.exceptions", exceptions_module)
    _install_fake_bioio(
        monkeypatch,
        lambda _path: (_ for _ in ()).throw(UnsupportedFileFormatError("plugin not installed")),
    )

    with pytest.raises(TranscodeDependencyError):
        transcode_to_ome_tiff(str(tmp_path / "source.czi"), str(tmp_path / "out.ome.tif"))


def test_generic_bioio_constructor_failure_is_retryable_operational(monkeypatch, tmp_path):
    _install_fake_bioio(
        monkeypatch,
        lambda _path: (_ for _ in ()).throw(RuntimeError("opaque reader failure")),
    )

    with pytest.raises(TranscodeOperationalError):
        transcode_to_ome_tiff(str(tmp_path / "source.czi"), str(tmp_path / "out.ome.tif"))


def test_empty_decoded_array_is_explicit_deterministic_input_failure(monkeypatch, tmp_path):
    class EmptyImage:
        scenes = ["Scene:0"]
        current_scene = "Scene:0"

        def set_scene(self, _index):
            return None

        def get_image_data(self, _dims):
            return SimpleNamespace(size=0)

    _install_fake_bioio(monkeypatch, lambda _path: EmptyImage())

    with pytest.raises(TranscodeError) as excinfo:
        transcode_to_ome_tiff(str(tmp_path / "source.czi"), str(tmp_path / "out.ome.tif"))

    assert excinfo.value.__class__.__name__ == "TranscodeInputError"
