"""Unit tests for the libbioimage meta -> viewer-info mapping (pure)."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ultra_deepagents.imaging.viewerinfo import (
    atlas_layout,
    build_acquisition,
    build_atlas_scheme,
    build_mosaic,
    build_raw_header,
    build_tile_scheme,
    build_viewer_info,
    default_visible_channels,
    paged_depth,
)


def test_default_visible_channels_dedupes_color_and_skips_brightfield():
    # A 7-channel stack with duplicate-color fluorophores + a brightfield channel
    # should default to ONE channel per distinct color (so cells/nuclei show), not
    # a single channel (which renders as one diffuse 3D blob) and not the
    # brightfield. Mirrors the real CMDRP/EGFP/H3342/Bright_100X file.
    # Real near-duplicate LUT colors (not exact: #ff0000 vs #ff0014) must still
    # collapse by dominant primary, and the white brightfield must be excluded, so
    # the blue nuclei channel survives.
    names = ["CMDRP_1", "CMDRP", "EGFP_1", "EGFP", "H3342_1", "H3342", "Bright_100X"]
    colors = [
        {"index": 0, "hex": "#ff0000", "rgb": [255, 0, 0]},
        {"index": 1, "hex": "#ff0014", "rgb": [255, 0, 20]},
        {"index": 2, "hex": "#00ff00", "rgb": [0, 255, 0]},
        {"index": 3, "hex": "#00ff5c", "rgb": [0, 255, 92]},
        {"index": 4, "hex": "#0000ff", "rgb": [0, 0, 255]},
        {"index": 5, "hex": "#0000ff", "rgb": [0, 0, 255]},
        {"index": 6, "hex": "#ffffff", "rgb": [255, 255, 255]},
    ]
    assert default_visible_channels(names, colors, 7, "composite") == [0, 2, 4]
    # Single-channel / non-composite stays on channel 0.
    assert default_visible_channels(["DAPI"], [{"index": 0, "hex": "#0000ff", "rgb": [0, 0, 255]}], 1, "single") == [0]


def test_build_acquisition_surfaces_tiff_xmp_provenance():
    # Pix4D-style BigTIFF: provenance from TIFF/XMP tags + colorspace + pyramid.
    meta = {
        "format": "BigTIFF", "image_mode": "RGBA", "image_num_resolution_levels": 8,
        "TIFF/Software": "Pix4Dfields 2.1.0 (4b5eefb83)",
        "TIFF/DateTime": "2022:07:17 13:49:38",
        "ColorProfile/color_space": "RGBA",
    }
    acq = build_acquisition(meta)
    assert acq["software"] == "Pix4Dfields 2.1.0 (4b5eefb83)"
    assert acq["acquired"] == "2022:07:17 13:49:38"
    assert acq["color_space"] == "RGBA"
    assert acq["pyramid_levels"] == 8


def test_build_acquisition_surfaces_ome_instrument():
    # OME-TIFF: the OME-XML-derived instrument context libbioimage parses.
    meta = {
        "format": "OME-TIFF",
        "OME/Image/AcquisitionDate": "2017-03-27T20:44:57.625",
        "OME/Image/Pixels/Channel/AcquisitionMode": "SpinningDiskConfocal",
        "OME/Image/ObjectiveSettings/Medium": "Water",
        "OME/Image/ObjectiveSettings/RefractiveIndex": "1.33",
        "OME/Image/Pixels/Channel/DetectorSettings/Binning": "2x2",
        "OME/Experimenter/UserName": "svc_aicspipeline",
        "objectives/objective:0/name": "C-Apochromat 100x/1.25 W",
    }
    acq = build_acquisition(meta)
    assert acq["acquired"].startswith("2017-03-27")
    assert acq["acquisition_mode"] == "SpinningDiskConfocal"
    assert acq["objective_medium"] == "Water"
    assert acq["refractive_index"] == "1.33"
    assert acq["detector_binning"] == "2x2"
    assert acq["experimenter"] == "svc_aicspipeline"
    assert acq["objective"].startswith("C-Apochromat")
    # A bare file with no provenance yields an empty dict (the UI hides the group).
    assert build_acquisition({"format": "TIFF", "image_num_x": 10}) == {}


def test_build_raw_header_filters_and_bounds():
    meta = {f"OME/Image/Plane:{i}/TheZ": str(i) for i in range(200)}  # per-plane explosion
    meta.update({
        "TIFF/Software": "X", "OME/Creator": "Bio-Formats", "format": "OME-TIFF",
        "image_num_x": 10, "image_num_c": 4, "channels/channel:0/name": "Red",
        "metadata_version": "3.4.1", "tile_size_x": "256",
    })
    header = build_raw_header(meta)
    assert "TIFF/Software" in header and "OME/Creator" in header
    assert all("/Plane" not in k for k in header)  # per-plane noise excluded
    assert not any(
        k.startswith(("image_num", "tile_", "channels/")) or k in ("format", "metadata_version")
        for k in header
    )
    assert len(header) <= 80


def test_build_viewer_info_emits_format_acquisition_header():
    vi = build_viewer_info({
        "image_num_x": 95174, "image_num_y": 91416, "image_num_z": 1, "image_num_c": 4,
        "image_pixel_depth": 8, "image_pixel_format": "unsigned integer",
        "format": "BigTIFF", "image_mode": "RGBA", "image_num_resolution_levels": 8,
        "TIFF/Software": "Pix4Dfields 2.1.0", "ColorProfile/color_space": "RGBA",
        "channels/channel:0/name": "Red", "channels/channel:1/name": "Green",
        "channels/channel:2/name": "Blue", "channels/channel:3/name": "Alpha",
    })
    md = vi["metadata"]
    assert md["format"] == "BigTIFF"  # the real container format, not the reader
    assert md["reader"] == "libbioimage"
    assert md["acquisition"]["software"] == "Pix4Dfields 2.1.0"
    assert "TIFF/Software" in md["header"]
    # channel names flow through phys for every multichannel image (RGBA bands here)
    assert vi["phys"]["channel_names"] == ["Red", "Green", "Blue", "Alpha"]


def test_default_visible_channels_prefers_signal_over_noise():
    # The real AICS file: the "_1" variant of each color pair (indices 0,2,4) is pure
    # NOISE; the real imagery is in 1,3,5. With per-channel structure scores the
    # default must pick the REAL channel within each color group and drop the noise
    # ones — otherwise the volume renders a noise blob. (Measured on real data: noise
    # autocorrelation ~0.02, signal ~0.95.)
    names = ["CMDRP_1", "CMDRP", "EGFP_1", "EGFP", "H3342_1", "H3342", "Bright_100X"]
    colors = [
        {"rgb": [255, 0, 0]}, {"rgb": [255, 0, 20]},
        {"rgb": [0, 255, 0]}, {"rgb": [0, 255, 92]},
        {"rgb": [0, 0, 255]}, {"rgb": [0, 0, 255]},
        {"rgb": [255, 255, 255]},
    ]
    scores = [0.02, 0.95, 0.03, 0.96, 0.02, 0.93, 0.97]
    assert default_visible_channels(names, colors, 7, "composite", scores) == [1, 3, 5]
    # No scores -> backward-compatible first-per-color behavior.
    assert default_visible_channels(names, colors, 7, "composite", None) == [0, 2, 4]
    # Degenerate: if EVERY channel scores as noise, ignore scores rather than show
    # nothing (fall back to first-per-color).
    assert default_visible_channels(names, colors, 7, "composite", [0.0] * 7) == [0, 2, 4]


def test_viewer_info_multichannel_zstack_defaults_to_composite_channels():
    vi = build_viewer_info({
        "image_num_x": 924, "image_num_y": 624, "image_num_z": 80, "image_num_c": 3,
        "format": "OME-TIFF", "image_pixel_depth": 16,
        "channels/channel:0/name": "CMDRP", "channels/channel:0/color": "255,0,0",
        "channels/channel:1/name": "EGFP", "channels/channel:1/color": "0,255,0",
        "channels/channel:2/name": "H3342", "channels/channel:2/color": "0,0,255",
    })
    assert vi["display_defaults"]["channels"] == [0, 1, 2]
    assert vi["display_defaults"]["channel_mode"] == "composite"


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
    # A non-medical (microscopy) z-stack is 2D-only: volume_mode stays "slice_stack"
    # (so the 2D Z scrub knows it is a stack) but the 3D surfaces are withheld.
    assert vi["viewer"]["available_surfaces"] == ["2d", "metadata"]
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


def test_tile_scheme_prefers_actual_czi_pyramid_levels_over_synthetic_levels():
    # libbioimage reports both a synthetic power-of-two pyramid and the actual
    # CZI/Bio-Formats levels. The viewer must advertise only the real levels,
    # otherwise it can request nonexistent/coarsened levels and first-fit looks
    # blocky compared with Fiji.
    meta = {
        "format": "CZI",
        "image_num_x": 5913,
        "image_num_y": 5679,
        "image_num_resolution_levels": 10,
        "image_resolution_level_scales": "1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625,0.007812,0.003906,0.001953",
        "image_num_resolution_levels_actual": 4,
        "image_resolution_level_scales_actual": "1.0,0.5,0.25,0.125",
        "tile_size_x": "256,256,256,256",
    }

    ts = build_tile_scheme(meta)

    assert ts is not None
    assert len(ts["levels"]) == 4
    assert [level["downsample"] for level in ts["levels"]] == [1, 2, 4, 8]
    assert ts["levels"][-1]["level"] == 3
    assert ts["levels"][-1]["width"] == 739
    assert ts["levels"][-1]["height"] == 710


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
    assert vi["viewer"]["available_surfaces"] == ["2d", "metadata"]  # flat 2D: no 3D surfaces


def test_viewer_info_3d_surfaces_are_medical_only():
    # The 3D surfaces (reslice + volume) are offered ONLY for medical (clinical)
    # volumes. A non-medical (microscopy) z-stack is 2D-only — its multichannel 3D
    # render is not shippable yet, so we lead with reliable 2D + Z/T scrubbing. A
    # flat 2D image gets neither. (volume_mode is unaffected; see the slice_stack
    # assertions elsewhere.)
    microscopy = build_viewer_info({
        "image_num_x": 924, "image_num_y": 624, "image_num_z": 80, "image_num_c": 7,
        "format": "OME-TIFF", "image_pixel_depth": 16,
    })
    assert microscopy["modality"] == "microscopy"
    assert microscopy["viewer"]["available_surfaces"] == ["2d", "metadata"]
    medical = build_viewer_info({
        "image_num_x": 256, "image_num_y": 256, "image_num_z": 128,
        "format": "NIfTI", "image_pixel_depth": 32, "image_pixel_format": "floating point",
    })
    assert medical["modality"] == "medical"
    assert medical["viewer"]["available_surfaces"] == ["2d", "metadata", "mpr", "volume"]


def test_build_mosaic_detects_unstitched_czi_mosaic():
    # Real CZI mosaic keys (full nested path); build_mosaic matches by key SUFFIX.
    meta = {
        "CZI/ImageDocument/Metadata/Information/Image/SizeM": 66,
        "CZI/ImageDocument/Metadata/Experiment/ExperimentBlocks/AcquisitionBlock/"
        "SubDimensionSetups/RegionsSetup/SampleHolder/IsOnlineStitchingEnabled": "false",
        "CZI/ImageDocument/Metadata/Experiment/ExperimentBlocks/AcquisitionBlock/"
        "SubDimensionSetups/RegionsSetup/SampleHolder/Overlap": 0.1,
    }
    assert build_mosaic(meta) == {"tiles": 66, "stitched": False, "overlap": 0.1}


def test_build_mosaic_stitched_flag_and_none_cases():
    assert build_mosaic({"x/Information/Image/SizeM": 4, "y/SampleHolder/IsOnlineStitchingEnabled": "true"}) == {
        "tiles": 4,
        "stitched": True,
    }
    # Single field (SizeM<=1) or no mosaic dimension -> not a mosaic.
    assert build_mosaic({"x/Information/Image/SizeM": 1}) is None
    assert build_mosaic({}) is None
    # Tiles only, no stitch/overlap metadata -> just the count.
    assert build_mosaic({"a/SizeM": 9}) == {"tiles": 9}


def test_build_viewer_info_emits_mosaic_for_unstitched_scan():
    info = build_viewer_info({
        "image_num_x": 5913, "image_num_y": 5679, "format": "CZI",
        "image_pixel_depth": 16, "image_pixel_format": "unsigned integer",
        "CZI/ImageDocument/Metadata/Information/Image/SizeM": 66,
        "a/SampleHolder/IsOnlineStitchingEnabled": "false",
        "a/SampleHolder/Overlap": 0.1,
    })
    assert info["metadata"]["mosaic"] == {"tiles": 66, "stitched": False, "overlap": 0.1}
    # A normal single-field image carries no mosaic block.
    plain = build_viewer_info({"image_num_x": 512, "image_num_y": 512, "format": "TIFF"})
    assert "mosaic" not in plain["metadata"]
