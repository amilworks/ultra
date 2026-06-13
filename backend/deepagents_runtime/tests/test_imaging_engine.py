"""Tests for the image engine backends.

The StubEngine path requires only Pillow; the real LibBioImageEngine is exercised
only for its unavailable-without-native-lib behavior here (full engine tests run
where ``libimgcnv.so`` is installed).
"""

from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from ultra_deepagents.imaging.engine import (
    EngineUnavailable,
    LibBioImageEngine,
    StubEngine,
)

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
    assert "-res-level 7" in pipelines_seen[0]       # computed bounded level
    assert "-res-level" not in pipelines_seen[2]     # level dropped on the last attempt


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


def test_libbioimage_slice_scrub_reads_bounded_level_but_settled_reads_native():
    # A large-plane z-stack: scrub frames (full_resolution=False) read a bounded
    # pyramid level; the settled view (full_resolution=True) reads the native plane
    # so pixel measurements stay exact. Small planes stay native even when scrubbing.
    big = {
        "image_num_x": 4096, "image_num_y": 4096, "image_num_z": 40,
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

    meta = {"image_num_x": 100, "image_num_y": 80, "image_num_z": 10, "image_num_resolution_levels": 1}

    class FakeBim:
        def meta(self, path, cache):
            return meta

    engine = object.__new__(LibBioImageEngine)
    engine._bim = FakeBim()
    engine._cache = object()
    engine._np = np

    plan = engine.atlas_plan("/v.tif", channels=[1, 3, 5], colors=[(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), None])
    assert plan["read_channels"] == [1, 3, 5]
    assert plan["cell_colors"] == [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), None]
    assert plan["depth"] == 10 and plan["columns"] >= 1


def test_libbioimage_histogram_auto_selects_bounded_level_for_large_pyramid():
    import numpy as np

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

        def read(self, path, pipeline, cache):
            seen["path"] = path
            seen["pipeline"] = pipeline
            return np.zeros((4, 8, 8), dtype=np.uint8)

    engine = object.__new__(LibBioImageEngine)
    engine._np = np
    engine._bim = FakeBim()
    engine._cache = object()

    hist = engine.histogram("/huge.tif", bins=16)

    assert hist["bins"] == 16
    assert len(hist["channels"]) == 4
    assert seen == {"path": "/huge.tif", "pipeline": "-res-level 6"}
