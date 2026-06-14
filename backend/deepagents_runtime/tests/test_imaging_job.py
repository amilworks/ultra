"""Unit tests for the image.derive_pyramid job runner (no engine binary needed)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from ultra_deepagents.imaging.convert import ConvertResult
from ultra_deepagents.imaging.job import DerivePyramidJob, run_derive_pyramid_job
from ultra_deepagents.imaging.worker import extract_derive_pyramid_payload


def test_job_from_dict_defaults_and_spec():
    job = DerivePyramidJob.from_dict({"resource_id": "r1", "src_path": "/a.czi", "dst_path": "/d.tif"})
    assert job.tile_size == 512 and job.layout == "topdirs" and job.fmt == "bigtiff"
    assert job.spec().options() == "compression lzw tiles 512 pyramid topdirs"


def test_runner_converts_and_reports_metadata():
    seen: dict = {}

    def fake_convert(src, dst, *, spec):
        seen["src"], seen["dst"], seen["spec"] = src, dst, spec
        return ConvertResult(src, dst, 0, "", "")

    def fake_meta(path):
        if path == "/a.czi":
            return {}  # source: not a native tiled pyramid -> proceed to convert
        assert path == "/d.tif"
        return {
            "image_num_resolution_levels": 5,
            "image_res_l_scales": "1,0.5,0.25,0.125,0.0625",
            "image_num_x": 2048,
            "image_num_y": 2048,
        }

    out = run_derive_pyramid_job(
        {"resource_id": "r1", "src_path": "/a.czi", "dst_path": "/d.tif", "tile_size": 256},
        convert_fn=fake_convert,
        meta_fn=fake_meta,
    )
    assert out["status"] == "succeeded"
    assert out["resource_id"] == "r1"
    assert out["derived_path"] == "/d.tif"
    assert out["levels"] == 5
    assert out["num_x"] == 2048
    assert seen["src"] == "/a.czi"
    assert seen["spec"].tile_size == 256


def test_auto_fmt_volume_derives_ome_bigtiff():
    # A z-stack source must derive to OME-BigTIFF so its Z planes survive (plain
    # BigTIFF flattens a multichannel OME hyperstack to a single plane).
    seen = {}

    def fake_convert(src, dst, *, spec):
        seen["fmt"] = spec.fmt
        return ConvertResult(src, dst, 0, "", "")

    out = run_derive_pyramid_job(
        {"resource_id": "r", "src_path": "/v.ome.tiff", "dst_path": "/d.tif", "fmt": "auto"},
        convert_fn=fake_convert,
        meta_fn=lambda _p: {"image_num_z": 80, "image_num_c": 7},
    )
    assert seen["fmt"] == "ome-bigtiff" and out["fmt"] == "ome-bigtiff"


def test_auto_fmt_flat_2d_derives_bigtiff():
    # A flat 2D slide stays BigTIFF (tile-addressable; OME wrapper breaks -tile).
    seen = {}

    def fake_convert(src, dst, *, spec):
        seen["fmt"] = spec.fmt
        return ConvertResult(src, dst, 0, "", "")

    run_derive_pyramid_job(
        {"resource_id": "r", "src_path": "/big.svs", "dst_path": "/d.tif", "fmt": "auto"},
        convert_fn=fake_convert,
        meta_fn=lambda _p: {"image_num_z": 1, "image_num_c": 3},
    )
    assert seen["fmt"] == "bigtiff"


def test_auto_fmt_paged_zstack_derives_ome_bigtiff():
    seen = {}

    def fake_convert(src, dst, *, spec):
        seen["fmt"] = spec.fmt
        return ConvertResult(src, dst, 0, "", "")

    run_derive_pyramid_job(
        {"resource_id": "r", "src_path": "/pages.tif", "dst_path": "/d.tif", "fmt": "auto"},
        convert_fn=fake_convert,
        meta_fn=lambda _p: {"image_num_z": 1, "image_num_c": 1, "image_num_p": 40},
    )
    assert seen["fmt"] == "ome-bigtiff"


def test_native_tiled_pyramid_source_skips_convert():
    # A source already exposing a tiled multi-resolution pyramid (e.g. a COG/orthomosaic)
    # is served tile-by-tile directly, so the potentially huge convert is skipped.
    called = {"convert": False}

    def fake_convert(src, dst, *, spec):
        called["convert"] = True
        return ConvertResult(src, dst, 0, "", "")

    out = run_derive_pyramid_job(
        {"resource_id": "r", "src_path": "/ortho.tif", "dst_path": "/d.tif", "fmt": "auto"},
        convert_fn=fake_convert,
        meta_fn=lambda _p: {"image_num_resolution_levels": 8, "tile_num_x": 256},
    )
    assert out["status"] == "skipped_native_pyramid"
    assert out["derived_path"] is None
    assert called["convert"] is False


def test_pyramidal_but_untiled_source_still_converts():
    # Multi-resolution but NOT tiled (no tile grid) -> -tile needs a derived tiled pyramid.
    called = {"convert": False}

    def fake_convert(src, dst, *, spec):
        called["convert"] = True
        return ConvertResult(src, dst, 0, "", "")

    out = run_derive_pyramid_job(
        {"resource_id": "r", "src_path": "/striped.tif", "dst_path": "/d.tif", "fmt": "auto"},
        convert_fn=fake_convert,
        meta_fn=lambda _p: {"image_num_resolution_levels": 6},  # no tile_num_x
    )
    assert called["convert"] is True and out["status"] == "succeeded"


def test_auto_fmt_without_engine_falls_back_to_bigtiff():
    # No meta_fn (native engine absent) keeps the tile-serving default.
    seen = {}

    def fake_convert(src, dst, *, spec):
        seen["fmt"] = spec.fmt
        return ConvertResult(src, dst, 0, "", "")

    run_derive_pyramid_job(
        {"resource_id": "r", "src_path": "/x.tif", "dst_path": "/d.tif", "fmt": "auto"},
        convert_fn=fake_convert,
        meta_fn=None,
    )
    assert seen["fmt"] == "bigtiff"


def test_runner_propagates_convert_failure():
    def failing(src, dst, *, spec):
        raise RuntimeError("imgcnv conversion failed")

    with pytest.raises(RuntimeError, match="imgcnv conversion failed"):
        run_derive_pyramid_job(
            {"resource_id": "r", "src_path": "/a", "dst_path": "/d"}, convert_fn=failing
        )


def test_extract_payload_from_data_agent_envelope():
    env = {
        "job_type": "image.derive_pyramid",
        "metadata": {"resource_id": "r", "src_path": "/a.czi", "dst_path": "/d.tif", "tile_size": 256},
    }
    job = extract_derive_pyramid_payload(env)
    assert job is not None and job["src_path"] == "/a.czi" and job["tile_size"] == 256


def test_extract_payload_skips_other_job_types():
    assert extract_derive_pyramid_payload({"job_type": "caption.generate", "metadata": {"src_path": "/x"}}) is None


def test_extract_payload_accepts_direct_job_dict():
    direct = {"resource_id": "r", "src_path": "/a", "dst_path": "/d"}
    assert extract_derive_pyramid_payload(direct) == direct


def test_runner_metadata_is_best_effort():
    def ok_convert(src, dst, *, spec):
        return ConvertResult(src, dst, 0, "", "")

    def bad_meta(path):
        raise ValueError("cannot read meta")

    out = run_derive_pyramid_job(
        {"resource_id": "r", "src_path": "/a", "dst_path": "/d"},
        convert_fn=ok_convert,
        meta_fn=bad_meta,
    )
    assert out["status"] == "succeeded"
    assert "meta_warning" in out
