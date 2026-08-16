"""Tests for the image engine backends.

The StubEngine path requires only Pillow; the real LibBioImageEngine is exercised
only for its unavailable-without-native-lib behavior here (full engine tests run
where ``libimgcnv.so`` is installed).
"""

from __future__ import annotations

import hashlib
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from ultra_deepagents.imaging.convert import ConvertResult
from ultra_deepagents.imaging.engine import (
    EngineUnavailable,
    LibBioImageEngine,
    StubEngine,
)
from ultra_deepagents.imaging.job import run_derive_pyramid_job

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture()
def stub():
    pytest.importorskip("PIL")
    return StubEngine()


def test_stub_tile_is_png(stub):
    out = stub.tile("a.czi", level=0, col=1, row=2, tile_size=64)
    assert out.startswith(_PNG_MAGIC)


def test_stub_is_deterministic(stub):
    a = stub.tile("a.czi", level=0, col=1, row=2, tile_size=64)
    b = stub.tile("a.czi", level=0, col=1, row=2, tile_size=64)
    assert a == b


def test_stub_varies_by_request(stub):
    a = stub.tile("a.czi", level=0, col=1, row=2, tile_size=64)
    b = stub.tile("a.czi", level=0, col=3, row=2, tile_size=64)
    c = stub.tile("b.czi", level=0, col=1, row=2, tile_size=64)
    assert a != b and a != c


def test_stub_all_image_ops_return_png(stub):
    assert stub.region("a.czi", x1=0, y1=0, x2=64, y2=64).startswith(_PNG_MAGIC)
    assert stub.slice_plane("a.czi", z=3).startswith(_PNG_MAGIC)
    assert stub.thumbnail("a.czi", max_size=32, z=1).startswith(_PNG_MAGIC)
    assert stub.atlas("a.czi", grid=(2, 2)).startswith(_PNG_MAGIC)


def test_stub_meta_and_formats(stub):
    meta = stub.meta("a.czi")
    assert meta["image_num_z"] >= 1
    assert meta["image_num_c"] >= 1
    assert "czi" in stub.formats()


def test_stub_histogram_shape(stub):
    hist = stub.histogram("a.czi", bins=16)
    assert hist["bins"] == 16
    assert len(hist["channels"][0]["counts"]) == 16


def test_libbioimage_engine_unavailable_without_native_lib():
    try:
        import libbioimage.libbioimage  # noqa: F401
    except Exception:
        pass
    else:
        pytest.skip("libbioimage is installed; the unavailable path does not apply")
    with pytest.raises(EngineUnavailable):
        LibBioImageEngine()


def test_libbioimage_scalar_plan_rejects_unbounded_native_source_before_plane_decode():
    class FakeBim:
        def __init__(self) -> None:
            self.reads = 0

        def meta(self, path, cache):
            return {
                "image_num_x": 16_384,
                "image_num_y": 16_384,
                "image_num_z": 1,
                "image_num_t": 1,
                "image_num_c": 1,
                "image_num_p": 1,
            }

        def read(self, path, pipeline, cache):
            self.reads += 1
            raise AssertionError("oversize plan must not decode scalar planes")

    engine = object.__new__(LibBioImageEngine)
    engine._bim = FakeBim()
    engine._cache = object()

    with pytest.raises(ValueError, match="source plane input"):
        engine.scalar_plan("oversize.nii")
    assert engine._bim.reads == 0


def test_libbioimage_scalar_plan_keeps_bounded_nonintegral_source_eligible():
    class FakeBim:
        def meta(self, path, cache):
            return {
                "image_num_x": 924,
                "image_num_y": 624,
                "image_num_z": 80,
                "image_num_t": 2,
                "image_num_c": 2,
                "image_num_p": 0,
            }

    engine = object.__new__(LibBioImageEngine)
    engine._bim = FakeBim()
    engine._cache = object()

    plan = engine.scalar_plan("bounded.czi", channel=1, t=1)
    assert (plan["width"], plan["height"], plan["depth"]) == (462, 312, 80)
    assert (plan["channel"], plan["t"]) == (1, 1)


def test_libbioimage_keeps_ultra_derived_pyramids_on_the_native_axis_path(
    monkeypatch,
):
    class FakeBim:
        def meta(self, path, cache):
            assert path == "/cache/derived/sample__pyramid.tif"
            return {
                "image_num_x": 32,
                "image_num_y": 24,
                "image_num_z": 7,
                "image_num_t": 2,
                "image_num_c": 3,
                "image_num_p": 0,
            }

    class ForbiddenSemanticEngine:
        def __init__(self):
            raise AssertionError("Ultra-owned pyramids must stay on libbioimage")

    from ultra_deepagents.imaging import bioio_engine

    monkeypatch.setattr(bioio_engine, "BioioEngine", ForbiddenSemanticEngine)
    engine = object.__new__(LibBioImageEngine)
    engine._bim = FakeBim()
    engine._cache = object()
    engine._semantic_tiff_engine = None

    plan = engine.scalar_plan(
        "/cache/derived/sample__pyramid.tif",
        channel=2,
        t=1,
    )

    assert (plan["source_depth"], plan["channel"], plan["t"]) == (7, 2, 1)


def test_localized_owned_pyramid_retains_native_libbioimage_routing(
    tmp_path,
    monkeypatch,
):
    from ultra_deepagents.imaging import service as service_module

    derived = tmp_path / "uploads" / "derived"
    derived.mkdir(parents=True)
    source = derived / "sample__pyramid.tif"
    source.write_bytes(b"owned pyramid")
    monkeypatch.setattr(service_module, "_PYRAMID_CACHE_ENABLED", True)
    monkeypatch.setattr(
        service_module,
        "_PYRAMID_CACHE_DIR",
        str(tmp_path / "local-cache"),
    )
    localized = service_module.localize_pyramid(str(source))
    engine = object.__new__(LibBioImageEngine)
    engine._semantic_tiff_engine = object()

    assert localized != str(source)
    assert engine._tiff_scalar_engine(localized) is None


def test_strict_digest_pyramid_localizes_and_retains_native_libbioimage_routing(
    tmp_path,
    monkeypatch,
):
    from ultra_deepagents.imaging import service as service_module

    source = tmp_path / "source.ome.tif"
    source_bytes = b"strict source generation"
    source.write_bytes(source_bytes)
    destination = tmp_path / "derived" / "sample__pyramid.tif"
    viewer_info = {
        "dims_order": "TCZYX",
        "axis_sizes": {"T": 1, "C": 1, "Z": 1, "Y": 8, "X": 16},
        "dtype": "uint16",
        "channel_names": ["Intensity"],
        "physical_spacing": {"x": 1.0, "y": 1.0, "z": 1.0},
        "metadata": {"spacing_units": {"x": "pixel", "y": "pixel", "z": "voxel"}},
        "viewer": {"tile_scheme": {"levels": [{"level": 0, "width": 16, "height": 8}]}},
        "tile_scheme": {"levels": [{"level": 0, "width": 16, "height": 8}]},
    }

    def write_artifact(src, dst, *, spec):
        with open(dst, "wb") as artifact:
            artifact.write(b"digest pyramid")
        return ConvertResult(src, dst, 0, "", "")

    result = run_derive_pyramid_job(
        {
            "resource_id": "sample",
            "src_path": str(source),
            "dst_path": str(destination),
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "source_size_bytes": len(source_bytes),
        },
        convert_fn=write_artifact,
        meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
        viewer_info_fn=lambda _path: viewer_info,
    )
    artifact_path = result["derived_path"]
    assert "__pyramid.sha256-" in artifact_path

    monkeypatch.setattr(service_module, "_PYRAMID_CACHE_ENABLED", True)
    monkeypatch.setattr(service_module, "_PYRAMID_CACHE_DIR", str(tmp_path / "local-cache"))
    localized = service_module.localize_pyramid(artifact_path)
    engine = object.__new__(LibBioImageEngine)
    engine._semantic_tiff_engine = object()

    assert localized != artifact_path
    assert engine._tiff_scalar_engine(localized) is None


@pytest.mark.parametrize(
    "path",
    [
        "/cache/source/sample.ome.tiff",
        "/cache/not-derived/sample__pyramid.tif",
        "/cache/derived/sample.tif",
        "/cache/derived/nested/sample__pyramid.tif",
    ],
)
def test_libbioimage_routes_every_other_tiff_to_the_semantic_decoder(path):
    semantic = object()
    engine = object.__new__(LibBioImageEngine)
    engine._semantic_tiff_engine = semantic

    assert engine._tiff_scalar_engine(path) is semantic


def test_libbioimage_scalar_plan_rejects_native_float_nearest_before_meta_or_read():
    class FakeBim:
        def __init__(self) -> None:
            self.meta_reads = 0
            self.pixel_reads = 0

        def meta(self, path, cache):
            self.meta_reads += 1
            raise AssertionError("native non-exact nearest must reject before metadata")

        def read(self, path, pipeline, cache):
            self.pixel_reads += 1
            raise AssertionError("native non-exact nearest must reject before pixels")

    engine = object.__new__(LibBioImageEngine)
    engine._bim = FakeBim()
    engine._cache = object()
    engine._tiff_scalar_engine = lambda _path: None

    with pytest.raises(ValueError, match="nearest.*exact Mask"):
        engine.scalar_plan("float.nii", sampling="nearest")
    assert engine._bim.meta_reads == 0
    assert engine._bim.pixel_reads == 0


def test_libbioimage_z_downsample_is_iterative_without_retaining_factor_planes(monkeypatch):
    np = pytest.importorskip("numpy")

    class FakeBim:
        def meta(self, path, cache):
            return {
                "image_num_x": 8,
                "image_num_y": 8,
                "image_num_z": 513,
                "image_num_t": 1,
                "image_num_c": 1,
                "image_num_p": 0,
            }

        def read(self, path, pipeline, cache):
            z = int(pipeline.split("-slice z:", 1)[1].split()[0])
            return np.full((8, 8), z, dtype="float32")

    engine = object.__new__(LibBioImageEngine)
    engine._bim = FakeBim()
    engine._cache = object()
    engine._np = np
    monkeypatch.setattr(
        np,
        "mean",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("z reduction must not retain a list of source planes")
        ),
    )

    plan = engine.scalar_plan("deep.czi")
    assert (plan["source_depth"], plan["depth"], plan["downsample_z"]) == (513, 257, 2)
    planes = engine.scalar_planes("deep.czi", zs=[0, 256], channel=0, t=0, pages=plan["pages"])
    assert float(planes[0][0, 0]) == pytest.approx(0.5)
    assert float(planes[1][0, 0]) == pytest.approx(512.0)


def test_libbioimage_thumbnail_auto_selects_bounded_level_for_large_pyramid():
    meta = {
        "image_num_x": 95174,
        "image_num_y": 91416,
        "image_num_resolution_levels": 11,
        "image_resolution_level_scales": (
            "1.000000,0.500000,0.249995,0.124992,0.062496,0.031248,"
            "0.015624,0.007807,0.003898,0.001944,0.000967"
        ),
    }
    seen: dict[str, str] = {}

    class FakeBim:
        def meta(self, path, cache):
            assert path == "/huge.tif"
            return meta

    engine = object.__new__(LibBioImageEngine)
    engine._bim = FakeBim()
    engine._cache = object()

    def fake_render(self, path, pipeline):
        seen["path"] = path
        seen["pipeline"] = pipeline
        return _PNG_MAGIC

    engine._render = types.MethodType(fake_render, engine)

    assert engine.thumbnail("/huge.tif", max_size=512) == _PNG_MAGIC
    assert seen == {
        "path": "/huge.tif",
        "pipeline": "-res-level 7 -resize 512,512,BC,MX -depth 8,D,U",
    }


# A thumbnail of a valid image always has content, so an empty (0,0,0) region from the
# engine is a transient hiccup under concurrent mixed load (reproduced live on the 95k-px
# EnrNE orthomosaic), never a real edge tile. thumbnail() must retry instead of 500ing and
# leaving a broken thumbnail in the Resources grid.
_HUGE_META = {
    "image_num_x": 95174,
    "image_num_y": 91416,
    "image_num_resolution_levels": 11,
    "image_resolution_level_scales": (
        "1.000000,0.500000,0.249995,0.124992,0.062496,0.031248,"
        "0.015624,0.007807,0.003898,0.001944,0.000967"
    ),
}
_EMPTY_REGION = "engine returned an empty region (shape (0, 0, 0))"


def _engine_with_render(meta, render_fn):
    class FakeBim:
        def meta(self, path, cache):
            return meta

    engine = object.__new__(LibBioImageEngine)
    engine._bim = FakeBim()
    engine._cache = object()
    engine._render = types.MethodType(render_fn, engine)
    return engine


def test_slice_plane_scrub_caps_long_edge_without_pyramid():
    # A flat (non-pyramidal) 2048-px plane: a transient z-scrub frame
    # (full_resolution=False) can't drop a -res-level (there is no pyramid), so the
    # long-edge cap is what keeps scrub fast (measured 0.47s/4.2MB -> 0.13s/0.8MB).
    # The settled view (full_resolution=True) stays native for pixel-accurate readouts.
    meta = {"image_num_x": 2048, "image_num_y": 2048, "image_num_z": 20}
    seen: dict[str, str] = {}

    def render(self, path, pipeline):
        seen["pipeline"] = pipeline
        return _PNG_MAGIC

    engine = _engine_with_render(meta, render)
    engine.slice_plane("/big.tif", z=10, full_resolution=False)
    assert "-resize 1024,1024,BC,MX" in seen["pipeline"]
    engine.slice_plane("/big.tif", z=10, full_resolution=True)
    assert "resize" not in seen["pipeline"]


def test_thumbnail_retries_transient_empty_region():
    calls = {"n": 0}

    def render(self, path, pipeline):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError(_EMPTY_REGION)
        return _PNG_MAGIC

    engine = _engine_with_render(_HUGE_META, render)
    assert engine.thumbnail("/huge.tif", max_size=512) == _PNG_MAGIC
    assert calls["n"] == 2  # one transient empty, recovered on retry


def test_thumbnail_drops_level_on_final_attempt():
    pipelines_seen = []

    def render(self, path, pipeline):
        pipelines_seen.append(pipeline)
        if len(pipelines_seen) < 3:  # both computed-level attempts come back empty
            raise ValueError(_EMPTY_REGION)
        return _PNG_MAGIC

    engine = _engine_with_render(_HUGE_META, render)
    assert engine.thumbnail("/huge.tif", max_size=512) == _PNG_MAGIC
    assert len(pipelines_seen) == 3
    assert "-res-level 7" in pipelines_seen[0]  # computed bounded level
    assert "-res-level" not in pipelines_seen[2]  # level dropped on the last attempt


def test_thumbnail_raises_when_region_stays_empty():
    calls = {"n": 0}

    def render(self, path, pipeline):
        calls["n"] += 1
        raise ValueError(_EMPTY_REGION)

    engine = _engine_with_render(_HUGE_META, render)
    with pytest.raises(ValueError, match="empty region"):
        engine.thumbnail("/huge.tif", max_size=512)
    assert calls["n"] == 3  # exhausts [level, level, None] then surfaces


def test_thumbnail_does_not_retry_unrelated_value_error():
    calls = {"n": 0}

    def render(self, path, pipeline):
        calls["n"] += 1
        raise ValueError("cannot encode array with ndim=5")

    engine = _engine_with_render(_HUGE_META, render)
    with pytest.raises(ValueError, match="ndim=5"):
        engine.thumbnail("/huge.tif", max_size=512)
    assert calls["n"] == 1  # a real error surfaces immediately, never retried


def test_thumbnail_prefers_actual_czi_pyramid_level_count():
    czi_meta = {
        "image_num_x": 5913,
        "image_num_y": 5679,
        "image_pixel_depth": 16,
        "image_pixel_format": "unsigned integer",
        "image_num_c": 1,
        "image_num_resolution_levels": 10,
        "image_resolution_level_scales": "1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625,0.007812,0.003906,0.001953",
        "image_num_resolution_levels_actual": 4,
        "image_resolution_level_scales_actual": "1.0,0.5,0.25,0.125",
    }
    seen: dict[str, str] = {}

    def render(self, path, pipeline):
        seen["path"] = path
        seen["pipeline"] = pipeline
        return _PNG_MAGIC

    engine = _engine_with_render(czi_meta, render)

    assert engine.thumbnail("/scene.czi", max_size=256) == _PNG_MAGIC
    assert seen == {
        "path": "/scene.czi",
        "pipeline": "-res-level 3 -resize 256,256,BC,MX -depth 8,D,U",
    }


def test_libbioimage_slice_scrub_reads_bounded_level_but_settled_reads_native():
    # A large-plane z-stack: scrub frames (full_resolution=False) read a bounded
    # pyramid level; the settled view (full_resolution=True) reads the native plane
    # so pixel measurements stay exact. Small planes stay native even when scrubbing.
    big = {
        "image_num_x": 4096,
        "image_num_y": 4096,
        "image_num_z": 40,
        "image_num_resolution_levels": 4,
        "image_resolution_level_scales": "1.000000,0.500000,0.250000,0.125000",
    }

    def make(meta):
        class FakeBim:
            def meta(self, path, cache):
                return meta

        engine = object.__new__(LibBioImageEngine)
        engine._bim = FakeBim()
        engine._cache = object()
        captured: dict[str, str] = {}

        def fake_render(self, path, pipeline):
            captured["pipeline"] = pipeline
            return _PNG_MAGIC

        engine._render = types.MethodType(fake_render, engine)
        return engine, captured

    # scrub: 4096px at cap 1024 -> level 2 (1024px)
    engine, cap = make(big)
    assert engine.slice_plane("/big.tif", z=5, full_resolution=False) == _PNG_MAGIC
    assert "-res-level 2" in cap["pipeline"]

    # settled: native, no downscale
    engine, cap = make(big)
    engine.slice_plane("/big.tif", z=5, full_resolution=True)
    assert "-res-level" not in cap["pipeline"]

    # an explicit level always wins, even for a scrub frame
    engine, cap = make(big)
    engine.slice_plane("/big.tif", z=5, level=1, full_resolution=False)
    assert "-res-level 1" in cap["pipeline"]

    # a sub-cap plane stays native while scrubbing (no quality loss)
    small = dict(big, image_num_x=800, image_num_y=800)
    engine, cap = make(small)
    engine.slice_plane("/small.tif", z=5, full_resolution=False)
    assert "-res-level" not in cap["pipeline"]


def test_atlas_plan_tolerates_none_colors():
    # _parse_fusion_request can yield None for a selected channel with no LUT color;
    # atlas_plan must preserve None (composite_channels skips it), not tuple(None).
    import numpy as np

    meta = {
        "image_num_x": 100,
        "image_num_y": 80,
        "image_num_z": 10,
        "image_num_resolution_levels": 1,
    }

    class FakeBim:
        def meta(self, path, cache):
            return meta

    engine = object.__new__(LibBioImageEngine)
    engine._bim = FakeBim()
    engine._cache = object()
    engine._np = np

    plan = engine.atlas_plan(
        "/v.tif", channels=[1, 3, 5], colors=[(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), None]
    )
    assert plan["read_channels"] == [1, 3, 5]
    assert plan["cell_colors"] == [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), None]
    assert plan["depth"] == 10 and plan["columns"] >= 1


def test_libbioimage_histogram_preserves_exact_non_tiff_channel_and_time_identity():
    import numpy as np

    meta = {
        "image_num_x": 4,
        "image_num_y": 3,
        "image_num_z": 2,
        "image_num_t": 3,
        "image_num_c": 3,
        "image_num_p": 0,
        "image_num_scenes": 1,
    }
    seen: list[str] = []

    class FakeBim:
        def meta(self, path, cache):
            assert path == "/exact.czi"
            return meta

        def read(self, path, pipeline, cache):
            seen.append(pipeline)
            z_index = int(pipeline.split("-slice z:", 1)[1].split(",", 1)[0])
            time_index = int(pipeline.split(",t:", 1)[1].split()[0])
            one_based_channel = int(pipeline.split("-remap ", 1)[1].split()[0])
            value = time_index * 100 + one_based_channel * 10 + z_index
            return np.full((3, 4), value, dtype="float32")

    engine = object.__new__(LibBioImageEngine)
    engine._np = np
    engine._bim = FakeBim()
    engine._cache = object()
    engine._semantic_tiff_engine = None

    hist = engine.histogram("/exact.czi", bins=4, channels=[2], t=2)

    assert hist["bins"] == 4
    assert hist["channel"] == 1
    assert hist["t"] == 2
    assert hist["scope"] == "volume"
    assert [entry["index"] for entry in hist["channels"]] == [1]
    assert hist["channels"][0]["min"] == pytest.approx(220)
    assert hist["channels"][0]["max"] == pytest.approx(221)
    assert sum(hist["channels"][0]["counts"]) == 24
    assert hist["sample_count"] == 24
    assert hist["threshold"]["channel"] == 1
    assert hist["threshold"]["t"] == 2
    assert hist["data_semantics"]["kind"] == "intensity"
    assert hist["data_semantics"]["strength"] == "unknown"
    assert hist["data_semantics"]["recommended_view"] == "intensity"
    assert len(seen) == 4
    assert all("-remap 2" in pipeline and ",t:2" in pipeline for pipeline in seen)

    seen.clear()
    display = engine.histogram("/exact.czi", bins=4, channels=[1, 2], t=2, scope="display")
    assert display["scope"] == "display"
    assert [entry["index"] for entry in display["channels"]] == [0, 1]
    assert display["channels"][0]["edges"] == display["channels"][1]["edges"]
    assert len(seen) == 2
    assert all(",t:2" in pipeline for pipeline in seen)


def _engine_with_meta(meta):
    class FakeBim:
        def meta(self, path, cache):
            return meta

    engine = object.__new__(LibBioImageEngine)
    engine._bim = FakeBim()
    engine._cache = object()
    return engine


def test_display_out_depth_full_range_for_rgba_photo():
    # An 8-bit RGBA orthomosaic/slide is a display photo: full-range ("8,F,U") so true
    # colors show AND the (constant) alpha survives — data-range would zero a fully-
    # opaque alpha and render native tiles blank.
    photo = {
        "image_pixel_format": "unsigned integer",
        "image_pixel_depth": 8,
        "image_num_c": 4,
        "image_mode": "RGBA",
    }
    assert _engine_with_meta(photo)._display_out_depth("/x.tif") == "8,F,U"

    # RGB photo recognized by channel names too (no explicit mode).
    rgb_named = {
        "image_pixel_format": "unsigned integer",
        "image_pixel_depth": 8,
        "image_num_c": 3,
        "channels/channel:0/name": "Red",
        "channels/channel:1/name": "Green",
        "channels/channel:2/name": "Blue",
    }
    assert _engine_with_meta(rgb_named)._display_out_depth("/x.tif") == "8,F,U"


def test_display_out_depth_data_range_for_scientific():
    # 16-bit microscopy, float, and single-channel 8-bit keep data-range ("8,D,U") so
    # their values map into the 8-bit display (full-range would render them near-black).
    for meta in (
        {"image_pixel_format": "unsigned integer", "image_pixel_depth": 16, "image_num_c": 2},
        {"image_pixel_format": "float", "image_pixel_depth": 32, "image_num_c": 1},
        {"image_pixel_format": "unsigned integer", "image_pixel_depth": 8, "image_num_c": 1},
    ):
        assert _engine_with_meta(meta)._display_out_depth("/x.tif") == "8,D,U"


def test_tile_uses_full_range_depth_for_photo():
    # The deep-zoom fix: a photo's tile read carries the full-range depth into the pipeline.
    photo = {
        "image_pixel_format": "unsigned integer",
        "image_pixel_depth": 8,
        "image_num_c": 4,
        "image_mode": "RGBA",
    }
    engine = _engine_with_meta(photo)
    seen = {}

    def fake_render(self, path, pipeline):
        seen["pipeline"] = pipeline
        return b"\x89PNG"

    engine._render = types.MethodType(fake_render, engine)
    engine.tile("/x.tif", level=0, col=93, row=89, tile_size=512)
    assert "8,F,U" in seen["pipeline"] and "-tile 512,93,89,0" in seen["pipeline"]


# --- Level-0 base fallback for planar multi-page SubIFD pyramids --------------------
# On that layout (LIF -> bioio -> OME-TIFF -> imgcnv) the engine's embedded -tile AND
# -tile-roi readers return an empty (0,0,0) region for the BASE level while sub-levels
# and native full-plane reads work (verified on the real engine against the affected
# prod pyramids). tile() must serve level 0 by decoding the native plane and cropping —
# and must NOT engage for genuine out-of-grid tiles, other levels, or huge planes.


def _fallback_engine(*, width, height, plane, is_photo=True, windows=None):
    np = pytest.importorskip("numpy")
    from PIL import Image

    meta = {"image_num_x": width, "image_num_y": height}

    class FakeBim:
        def meta(self, path, cache):
            return meta

        def read(self, path, pipeline, cache):
            if "-tile " in pipeline:
                return np.empty((0, 0, 0), dtype="uint8")  # the broken embedded-tile read
            return plane  # the native full-plane read (slice_plane pipeline)

    engine = object.__new__(LibBioImageEngine)
    engine._bim = FakeBim()
    engine._cache = object()
    engine._np = np
    engine._Image = Image
    engine._is_display_photo = types.MethodType(lambda self, p: is_photo, engine)
    engine._display_out_depth = types.MethodType(lambda self, p: "8,F,U", engine)
    if windows is not None:
        engine._display_global_windows = types.MethodType(
            lambda self, p, channels=None: windows, engine
        )
    return engine


def test_tile_level0_empty_falls_back_to_native_plane_crop():
    np = pytest.importorskip("numpy")
    import io as _io

    from PIL import Image

    # 1000x1200 gradient plane; tile (col=1,row=1) -> crop x 512:1000, y 512:1024
    plane = (np.arange(1200 * 1000, dtype="uint32").reshape(1200, 1000) % 251).astype("uint8")
    engine = _fallback_engine(width=1000, height=1200, plane=plane)

    png = engine.tile("/planar.tif", level=0, col=1, row=1, tile_size=512)

    img = Image.open(_io.BytesIO(png))
    assert img.size == (488, 512)  # (width, height): x-edge tile is partial
    assert (np.asarray(img) == plane[512:1024, 512:1000]).all()


def test_tile_level0_fallback_windowed_scalar_branch():
    np = pytest.importorskip("numpy")
    import io as _io

    from PIL import Image

    plane = np.linspace(0.0, 1000.0, 600 * 600, dtype="float32").reshape(600, 600)
    engine = _fallback_engine(
        width=600, height=600, plane=plane, is_photo=False, windows=[(0.0, 1000.0)]
    )

    png = engine.tile("/scalar.tif", level=0, col=0, row=0, tile_size=512)

    img = Image.open(_io.BytesIO(png))
    assert img.size == (512, 512)
    # global window (0..1000) applied AFTER crop == applied before crop: spot-check corners
    arr = np.asarray(img)
    assert arr[0, 0] == 0
    assert arr.max() > 0


def test_tile_level0_out_of_grid_keeps_empty_region_error():
    np = pytest.importorskip("numpy")
    plane = np.zeros((600, 600), dtype="uint8")
    engine = _fallback_engine(width=600, height=600, plane=plane)
    with pytest.raises(ValueError, match="empty region"):
        engine.tile("/planar.tif", level=0, col=2, row=0, tile_size=512)  # x1=1024 >= 600


def test_tile_nonzero_level_empty_region_is_not_fallback():
    np = pytest.importorskip("numpy")
    plane = np.zeros((600, 600), dtype="uint8")
    engine = _fallback_engine(width=600, height=600, plane=plane)
    with pytest.raises(ValueError, match="empty region"):
        engine.tile("/planar.tif", level=1, col=0, row=0, tile_size=512)


def test_tile_level0_fallback_respects_pixel_budget():
    np = pytest.importorskip("numpy")
    plane = np.zeros((4, 4), dtype="uint8")  # would "work", but the plane claims gigapixel
    engine = _fallback_engine(width=100_000, height=100_000, plane=plane)
    with pytest.raises(ValueError, match="empty region"):
        engine.tile("/giga.tif", level=0, col=0, row=0, tile_size=512)


def test_tile_level0_fallback_fused_branch_composites_crop():
    np = pytest.importorskip("numpy")
    import io as _io

    from PIL import Image

    # (C,H,W) float plane: 2 channels; fused branch must crop then composite
    plane = np.stack(
        [
            np.full((600, 600), 100.0, dtype="float32"),
            np.full((600, 600), 200.0, dtype="float32"),
        ]
    )
    engine = _fallback_engine(width=600, height=600, plane=plane)

    png = engine.tile(
        "/planar.tif",
        level=0,
        col=0,
        row=0,
        tile_size=512,
        channels=[1, 2],
        colors=[(255, 0, 0), (0, 255, 0)],
    )

    img = Image.open(_io.BytesIO(png))
    assert img.size == (512, 512)
    assert img.mode == "RGB"
