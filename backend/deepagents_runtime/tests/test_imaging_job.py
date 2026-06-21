"""Unit tests for the image.derive_pyramid job runner (no engine binary needed)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from ultra_deepagents.imaging.convert import ConvertResult
from ultra_deepagents.imaging.job import DerivePyramidJob, run_derive_pyramid_job
from ultra_deepagents.imaging.transcode import TranscodeResult
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
        if path == "/a.lsm":
            # source: decodable, not a native tiled pyramid -> proceed to convert
            return {"image_num_x": 2048, "image_num_y": 2048}
        assert path == "/d.tif"
        return {
            "image_num_resolution_levels": 5,
            "image_res_l_scales": "1,0.5,0.25,0.125,0.0625",
            "image_num_x": 2048,
            "image_num_y": 2048,
        }

    out = run_derive_pyramid_job(
        {"resource_id": "r1", "src_path": "/a.lsm", "dst_path": "/d.tif", "tile_size": 256},
        convert_fn=fake_convert,
        meta_fn=fake_meta,
    )
    assert out["status"] == "succeeded"
    assert out["resource_id"] == "r1"
    assert out["derived_path"] == "/d.tif"
    assert out["levels"] == 5
    assert out["num_x"] == 2048
    assert seen["src"] == "/a.lsm"
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
        meta_fn=lambda _p: {"image_num_x": 4096, "image_num_y": 4096, "image_num_z": 80, "image_num_c": 7},
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
        meta_fn=lambda _p: {"image_num_x": 50000, "image_num_y": 40000, "image_num_z": 1, "image_num_c": 3},
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
        meta_fn=lambda _p: {"image_num_x": 1024, "image_num_y": 1024, "image_num_z": 1, "image_num_c": 1, "image_num_p": 40},
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
        meta_fn=lambda _p: {"image_num_x": 8000, "image_num_y": 8000, "image_num_resolution_levels": 6},  # no tile_num_x
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


def test_undecodable_source_transcodes_via_bioio(tmp_path):
    # libbioimage can't decode the source (empty meta) -> bioio transcodes it to an
    # intermediate OME-TIFF, and the pyramid is built from THAT (not the original).
    src = str(tmp_path / "scan.lif")
    dst = str(tmp_path / "d.tif")
    intermediate = dst + ".transcode.ome.tif"
    seen: dict = {}

    def fake_meta(path):
        if path.endswith(".transcode.ome.tif"):
            return {"image_num_x": 3000, "image_num_y": 3000, "image_num_z": 2, "image_num_c": 2}
        return {}  # source: undecodable by libbioimage

    def fake_transcode(s, d, **_kw):
        open(d, "w").close()  # create the intermediate so the cleanup has a real file
        return TranscodeResult(
            path=d, series_count=16, series_index=4, series_name="Series005",
            num_c=2, num_z=2, dtype="uint8", series_names=[f"S{i}" for i in range(16)],
        )

    def fake_convert(s, d, *, spec):
        seen["convert_src"] = s
        seen["fmt"] = spec.fmt
        seen["intermediate_present_at_convert"] = os.path.exists(intermediate)
        return ConvertResult(s, d, 0, "", "")

    out = run_derive_pyramid_job(
        {"resource_id": "r", "src_path": src, "dst_path": dst, "fmt": "auto"},
        convert_fn=fake_convert, meta_fn=fake_meta, transcode_fn=fake_transcode,
    )
    assert out["status"] == "succeeded"
    assert out["transcoded"] is True and out["source_reader"] == "bioio"
    assert out["series_count"] == 16 and out["series_index"] == 4 and out["series_name"] == "Series005"
    # The pyramid is derived from the transcoded intermediate, not the .lif.
    assert seen["convert_src"] == intermediate
    # Multichannel/volume series must stay OME-BigTIFF (preserve channels + z planes).
    assert seen["fmt"] == "ome-bigtiff"
    assert seen["intermediate_present_at_convert"] is True
    # The redundant intermediate is reclaimed after the pyramid exists.
    assert not os.path.exists(intermediate)


def test_decodable_source_does_not_transcode():
    # A source libbioimage CAN read must never invoke the bioio fallback.
    called = {"transcode": False}

    def fake_transcode(s, d, **_kw):
        called["transcode"] = True
        raise AssertionError("transcode must not run for a decodable source")

    out = run_derive_pyramid_job(
        {"resource_id": "r", "src_path": "/a.tif", "dst_path": "/d.tif", "fmt": "auto"},
        convert_fn=lambda s, d, *, spec: ConvertResult(s, d, 0, "", ""),
        meta_fn=lambda _p: {"image_num_x": 2048, "image_num_y": 2048, "image_num_z": 1, "image_num_c": 1},
        transcode_fn=fake_transcode,
    )
    assert called["transcode"] is False
    assert out.get("transcoded") is None


def test_prefer_bioio_extension_forces_transcode(tmp_path):
    # .czi is in PREFER_BIOIO_EXTENSIONS -> route through bioio even though libbioimage
    # COULD decode it; the pyramid is built from the bioio transcode (Zeiss mosaics read
    # correctly there). meta_fn(source) is not even consulted for the routing decision.
    src = str(tmp_path / "scene.czi")
    dst = str(tmp_path / "d.tif")
    seen: dict = {}

    def fake_meta(path):
        # libbioimage CAN read the czi (real geometry), but prefer-bioio overrides it.
        if path.endswith(".transcode.ome.tif"):
            return {"image_num_x": 5913, "image_num_y": 5679, "image_num_z": 1, "image_num_c": 2}
        return {"image_num_x": 5913, "image_num_y": 5679, "image_num_z": 1, "image_num_c": 2}

    def fake_transcode(s, d, **_kw):
        open(d, "w").close()
        return TranscodeResult(
            path=d, series_count=1, series_index=0, series_name="Scene0",
            num_c=2, num_z=1, dtype="uint16", series_names=["Scene0"],
        )

    def fake_convert(s, d, *, spec):
        seen["convert_src"] = s
        seen["fmt"] = spec.fmt
        return ConvertResult(s, d, 0, "", "")

    out = run_derive_pyramid_job(
        {"resource_id": "r", "src_path": src, "dst_path": dst, "fmt": "auto"},
        convert_fn=fake_convert, meta_fn=fake_meta, transcode_fn=fake_transcode,
    )
    assert out["transcoded"] is True and out["source_reader"] == "bioio"
    assert seen["convert_src"].endswith(".transcode.ome.tif")  # built from bioio, not the czi
    assert seen["fmt"] == "ome-bigtiff"  # 2 channels -> keep them


def test_prefer_bioio_soft_falls_back_to_libbioimage(tmp_path):
    # If bioio cannot read a prefer-bioio source, fall back to a normal libbioimage
    # convert of the source — don't discard a working native render.
    from ultra_deepagents.imaging.transcode import TranscodeError

    src = str(tmp_path / "scene.czi")
    dst = str(tmp_path / "d.tif")
    seen: dict = {}

    def fake_meta(_path):
        return {"image_num_x": 4000, "image_num_y": 4000, "image_num_z": 1, "image_num_c": 1}

    def failing_transcode(s, d, **_kw):
        raise TranscodeError("bioio cannot read this czi variant")

    def fake_convert(s, d, *, spec):
        seen["convert_src"] = s
        seen["fmt"] = spec.fmt
        return ConvertResult(s, d, 0, "", "")

    out = run_derive_pyramid_job(
        {"resource_id": "r", "src_path": src, "dst_path": dst, "fmt": "auto"},
        convert_fn=fake_convert, meta_fn=fake_meta, transcode_fn=failing_transcode,
    )
    assert out["status"] == "succeeded"
    assert out.get("transcoded") is None      # bioio failed -> libbioimage path taken
    assert seen["convert_src"] == src          # converted the original czi, no intermediate
    assert seen["fmt"] == "bigtiff"            # flat single-channel 2D -> bigtiff
    assert not os.path.exists(dst + ".transcode.ome.tif")  # intermediate cleaned up


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
        "metadata": {"resource_id": "r", "src_path": "/a.lsm", "dst_path": "/d.tif", "tile_size": 256},
    }
    job = extract_derive_pyramid_payload(env)
    assert job is not None and job["src_path"] == "/a.lsm" and job["tile_size"] == 256


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
