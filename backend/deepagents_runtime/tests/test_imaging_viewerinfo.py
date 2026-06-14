"""Unit tests for the libbioimage meta -> viewer-info mapping (pure)."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ultra_deepagents.imaging.viewerinfo import (
    atlas_layout,
    build_atlas_scheme,
    build_tile_scheme,
    build_viewer_info,
    paged_depth,
)


@pytest.mark.parametrize(
    "meta",
    [
        {"image_num_x": 0, "image_num_y": 0, "format": "TIFF"},  # both zero
        {"image_num_x": 512, "image_num_y": 0, "format": "TIFF"},  # one zero
        {"format": "TIFF", "image_pixel_depth": 8},  # dims missing entirely
    ],
)
def test_viewer_info_fails_closed_on_degenerate_geometry(meta):
    # A file libbioimage opens but can't characterize (0-sized canvas) must fail
    # closed with a decode error (service.py maps the "cannot decode" marker -> 422,
    # a clean "preview unavailable") rather than emit a 0x0 viewer that the frontend
    # renders as a blank canvas with no message.
    with pytest.raises(ValueError, match="cannot decode"):
        build_viewer_info(meta)


def test_paged_multipage_tiff_is_a_scrubbable_z_stack():
    # A plain multi-page TIFF reports planes as pages (image_num_p) with z=1.
    meta = {
        "image_num_x": 128, "image_num_y": 128, "image_num_z": 1, "image_num_t": 1,
        "image_num_c": 1, "image_num_p": 8, "format": "TIFF", "image_pixel_depth": 8,
    }
    assert paged_depth(meta) == 8
    vi = build_viewer_info(meta)
    assert vi["axis_sizes"]["Z"] == 8 and vi["is_volume"] is True
    assert vi["viewer"]["volume_mode"] == "slice_stack"


def test_paged_depth_ignores_multichannel_or_time_paged_files():
    # Pages that interleave channels/time are NOT pure z depth; leave to -slice.
    assert paged_depth({"image_num_p": 8, "image_num_c": 2, "image_num_z": 1}) == 0
    assert paged_depth({"image_num_p": 8, "image_num_t": 4, "image_num_z": 1}) == 0
    assert paged_depth({"image_num_p": 70, "image_num_z": 70}) == 0  # already a real z-stack
    assert paged_depth({"image_num_p": 1}) == 0


def test_atlas_layout_near_square_grid_and_bounded_cell():
    # 924x624 x 80 planes, cap 256: downsample 4 -> 231x156 cell, ceil(sqrt(80))=9 cols.
    lay = atlas_layout(924, 624, 80, cell_cap=256)
    assert lay == {
        "downsample": 4, "cell_w": 231, "cell_h": 156,
        "columns": 9, "rows": 9, "slice_count": 80,
    }
    # A small stack already under the cap is not upsampled (downsample 1).
    assert atlas_layout(200, 120, 8, cell_cap=256)["downsample"] == 1


def test_build_atlas_scheme_matches_frontend_decode_contract():
    # The atlas image dims MUST equal cols*cell_w x rows*cell_h (what the engine
    # assembles and the frontend atlasToVolumeTexture decodes).
    scheme = build_atlas_scheme(
        {"image_num_x": 924, "image_num_y": 624, "image_num_z": 80}, depth=80
    )
    assert scheme["columns"] == 9 and scheme["rows"] == 9
    assert scheme["slice_width"] == 231 and scheme["slice_height"] == 156
    assert scheme["atlas_width"] == 231 * 9 and scheme["atlas_height"] == 156 * 9
    assert scheme["slice_count"] == 80 and scheme["downsample"] == 4


def test_build_atlas_scheme_none_for_non_volume():
    assert build_atlas_scheme({"image_num_x": 512, "image_num_y": 512, "image_num_z": 1}) is None


def test_viewer_info_slice_stack_emits_atlas_scheme():
    # A multichannel z-stack (slice_stack volume) carries the atlas layout the 3D
    # viewer needs; a non-volume image does not.
    vi = build_viewer_info({
        "image_num_x": 924, "image_num_y": 624, "image_num_z": 80, "image_num_c": 7,
        "format": "OME-TIFF", "image_pixel_depth": 16,
    })
    assert vi["viewer"]["volume_mode"] == "slice_stack"
    scheme = vi["viewer"]["atlas_scheme"]
    assert scheme["slice_count"] == 80 and scheme["columns"] == 9
    assert scheme["atlas_width"] == scheme["slice_width"] * scheme["columns"]
    assert "atlas_scheme" not in build_viewer_info(
        {"image_num_x": 640, "image_num_y": 480, "format": "JPEG", "image_pixel_depth": 8}
    )["viewer"]


def test_viewer_info_rgba_orthomosaic_is_a_photo_not_microscopy():
    # A geospatial RGBA orthomosaic (image_mode RGBA, Red/Green/Blue/Alpha bands,
    # 8-bit) is color data, not fluorescence: it must render as a plain zoomable
    # photo (display path), NOT the composite channel-pills + window/level UI.
    vi = build_viewer_info({
        "image_num_x": 95174, "image_num_y": 91416, "image_num_z": 1, "image_num_c": 4,
        "format": "BigTIFF", "image_pixel_depth": 8, "image_pixel_format": "unsigned integer",
        "image_mode": "RGBA", "image_num_resolution_levels": 8,
        "image_resolution_level_scales": "1.0,0.5,0.25,0.125,0.0625,0.031,0.0156,0.0078",
        "channels/channel:0/name": "Red", "channels/channel:1/name": "Green",
        "channels/channel:2/name": "Blue", "channels/channel:3/name": "Alpha",
    })
    assert vi["modality"] == "image"
    assert vi["viewer"]["render_policy"] == "display"
    assert vi["viewer"]["channel_mode"] == "single"
    assert vi["viewer"]["backend_mode"] == "pyramid"  # natively pyramidal => DeepZoom tiles


def test_viewer_info_rgb_photo_by_name_is_image():
    vi = build_viewer_info({
        "image_num_x": 4000, "image_num_y": 3000, "image_num_c": 3, "format": "TIFF",
        "image_pixel_depth": 8, "image_pixel_format": "unsigned integer",
        "channels/channel:0/name": "Red", "channels/channel:1/name": "Green",
        "channels/channel:2/name": "Blue",
    })
    assert vi["modality"] == "image" and vi["viewer"]["channel_mode"] == "single"


def test_viewer_info_fluorescence_not_misread_as_photo():
    # 16-bit 2-channel DAPI/FITC, a 5-channel stack, and 16-bit RGBA must all stay
    # microscopy/composite (the photo_like guard requires 8-bit unsigned + RGB names/mode).
    dapi = build_viewer_info({
        "image_num_x": 512, "image_num_y": 512, "image_num_c": 2, "format": "OME-TIFF",
        "image_pixel_depth": 16, "image_pixel_format": "unsigned integer",
        "channels/channel:0/name": "DAPI", "channels/channel:1/name": "FITC",
    })
    assert dapi["modality"] == "microscopy" and dapi["viewer"]["channel_mode"] == "composite"
    five = build_viewer_info({
        "image_num_x": 512, "image_num_y": 512, "image_num_c": 5, "format": "CZI",
        "image_pixel_depth": 8, "image_pixel_format": "unsigned integer", "image_mode": "RGB",
    })
    assert five["modality"] == "microscopy"  # c=5 not a photo even if mode claims RGB
    rgba16 = build_viewer_info({
        "image_num_x": 512, "image_num_y": 512, "image_num_c": 4, "format": "TIFF",
        "image_pixel_depth": 16, "image_pixel_format": "unsigned integer", "image_mode": "RGBA",
    })
    assert rgba16["modality"] == "microscopy"  # 16-bit RGBA is not an 8-bit photo


def test_viewer_info_microscopy_zstack():
    meta = {
        "image_num_x": 2048, "image_num_y": 2048, "image_num_z": 20, "image_num_c": 2, "image_num_t": 1,
        "image_pixel_depth": 16, "image_pixel_format": "unsigned integer", "format": "OME-TIFF",
        "channels/channel:0/name": "DAPI", "channels/channel:0/color": "0,0,1",
        "channels/channel:1/name": "FITC", "channels/channel:1/color": "0,1,0",
        "pixel_resolution_x": 0.325, "pixel_resolution_y": 0.325, "pixel_resolution_z": 1.0,
    }
    vi = build_viewer_info(meta)
    assert vi["axis_sizes"] == {"T": 1, "C": 2, "Z": 20, "Y": 2048, "X": 2048}
    assert vi["dtype"] == "uint16" and vi["pixel_format"] == "u"
    assert vi["modality"] == "microscopy"
    assert vi["is_volume"] and vi["is_multichannel"] and not vi["is_timeseries"]
    assert vi["channel_names"] == ["DAPI", "FITC"]
    assert vi["channel_colors"][0]["rgb"] == [0, 0, 255]
    assert vi["channel_colors"][0]["hex"] == "#0000ff"
    assert vi["physical_spacing"] == {"x": 0.325, "y": 0.325, "z": 1.0}
    assert vi["tile_scheme"] is None  # no resolution levels in source meta
    # nested frontend contract
    assert vi["kind"] == "image"
    assert vi["viewer"]["volume_mode"] == "slice_stack"
    assert vi["viewer"]["delivery_mode"] == "direct"  # no pyramid -> direct
    assert "tile_scheme" not in vi["viewer"]
    assert vi["phys"]["channel_names"] == ["DAPI", "FITC"]
    assert vi["phys"]["ch"] == 2 and vi["phys"]["z"] == 20
    assert vi["display_defaults"]["z_index"] == 10
    assert vi["metadata"]["reader"] == "libbioimage"
    assert vi["metadata"]["microscopy"]["channel_names"] == ["DAPI", "FITC"]


def test_viewer_info_pyramid_delivers_multiscale():
    meta = {
        "image_num_x": 8192, "image_num_y": 8192, "image_num_z": 1, "image_num_c": 3,
        "image_pixel_depth": 8, "image_pixel_format": "unsigned integer", "format": "OME-TIFF",
        "image_num_resolution_levels": 4, "image_resolution_level_scales": "1.0,0.5,0.25,0.125",
        "tile_size_x": "256,256,256,256",
    }
    vi = build_viewer_info(meta)
    assert vi["backend_mode"] == "pyramid"
    assert vi["viewer"]["delivery_mode"] == "deferred_multiscale"
    assert vi["viewer"]["tile_scheme"]["levels"][0]["columns"] == 32
    assert vi["tile_scheme"] is not None


def test_tile_scheme_from_pyramid_levels():
    meta = {
        "image_num_x": 8192, "image_num_y": 8192,
        "image_num_resolution_levels": 4, "image_resolution_level_scales": "1.0,0.5,0.25,0.125",
        "tile_size_x": "256,256,256,256",
    }
    ts = build_tile_scheme(meta)
    assert ts["tile_size"] == 256 and ts["format"] == "png"
    assert len(ts["levels"]) == 4
    assert ts["levels"][0] == {"level": 0, "width": 8192, "height": 8192, "columns": 32, "rows": 32, "downsample": 1}
    assert ts["levels"][1]["width"] == 4096 and ts["levels"][1]["columns"] == 16 and ts["levels"][1]["downsample"] == 2
    assert ts["levels"][3]["downsample"] == 8


def test_no_tile_scheme_single_level():
    assert build_tile_scheme({"image_num_x": 512, "image_num_y": 512, "image_num_resolution_levels": 1, "image_res_l_scales": "1.0"}) is None
    assert build_tile_scheme({"image_num_x": 0, "image_num_y": 0}) is None


def test_viewer_info_float_and_medical():
    vi = build_viewer_info({"image_num_x": 256, "image_num_y": 256, "image_num_z": 128, "format": "NIfTI", "image_pixel_depth": 32, "image_pixel_format": "floating point"})
    assert vi["dtype"] == "float32" and vi["pixel_format"] == "f"
    assert vi["modality"] == "medical"
    assert vi["is_volume"]


def test_viewer_info_plain_image():
    vi = build_viewer_info({"image_num_x": 640, "image_num_y": 480, "format": "JPEG", "image_pixel_depth": 8})
    assert vi["modality"] == "image"
    assert vi["axis_sizes"]["Z"] == 1 and not vi["is_volume"]
    assert vi["tile_scheme"] is None
