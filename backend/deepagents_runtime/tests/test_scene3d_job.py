"""End-to-end derive: manifest schema, chunk bytes, poster, failure taxonomy.

The slow test at the bottom runs the real 14,469,103-splat file. It is skipped unless
``ULTRA_SCENE3D_REAL_PLY`` names it, because it reads 3.4 GB and takes minutes.
"""

from __future__ import annotations

import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from PIL import Image
from test_scene3d_ply import (
    POSTSHOT_SPLAT_PROPS,
    splat_rows,
    write_colmap_points,
    write_ply,
    write_postshot_splats,
)
from ultra_deepagents.imaging.derivative_manifest import (
    DeterministicDerivativeError,
    TransientDerivativeError,
)
from ultra_deepagents.scene3d import ply, spark_encode
from ultra_deepagents.scene3d.job import (
    MANIFEST_NAME,
    POSTER_NAME,
    Scene3dDeriveJob,
    failure_marker_path,
    run_scene3d_derive_job,
)

REAL_SPLAT_PLY = os.environ.get("ULTRA_SCENE3D_REAL_PLY")
REAL_SPLAT_COUNT = 14_469_103
REAL_SPLAT_STRIDE = 236


def _derive(tmp_path, src, **options):
    dst = tmp_path / "derived"
    result = run_scene3d_derive_job(
        {"resource_id": "file-1", "src_path": str(src), "dst_dir": str(dst), **options}
    )
    document = json.loads((dst / MANIFEST_NAME).read_text())
    return result, document, dst


def _chunk_header(blob):
    magic = blob[0:4]
    version, flags = struct.unpack_from("<HH", blob, 4)
    count, sh_degree = struct.unpack_from("<II", blob, 8)
    bbox_min = np.frombuffer(blob[16:28], "<f4")
    bbox_max = np.frombuffer(blob[28:40], "<f4")
    origin = np.frombuffer(blob[40:52], "<f4")
    return magic, version, flags, count, sh_degree, bbox_min, bbox_max, origin


def test_splat_derive_emits_the_section_6_manifest(tmp_path):
    src = write_postshot_splats(tmp_path / "scene.ply", count=5000)

    result, document, dst = _derive(tmp_path, src, max_splats_per_chunk=800, tier_count=3)

    assert result["status"] == "succeeded"
    assert document["schema"] == "ultra.scene3d.v1"
    assert document["scene_kind"] == "splat"
    assert document["source"] == {
        "format": "ply",
        "writer": "postshot",
        "vertex_count": 5000,
        "bytes": os.path.getsize(src),
        "declared_sh_degree": 3,
        "measured_sh_degree": 0,
        "stride_bytes": 236,
    }
    assert document["world"]["frame"] == "source"
    assert document["world"]["units"] == "arbitrary"
    assert len(document["world"]["bbox"]) == 6
    assert document["service_urls"] == {
        "chunk": "/v2/uploads/file-1/scene3d/chunk/{index}",
        "download": "/v2/resources/file-1/download",
    }
    layer = document["layers"][0]
    assert layer["type"] == "splats"
    assert layer["encoding"] == "usx-v1"
    assert layer["total"] == 5000
    assert layer["activation_domain"] == "post"
    assert layer["source_frame"] == "source"
    assert set(layer["quantization"]) == {
        "center",
        "scale",
        "rotation",
        "color",
        "clamped_color_fraction",
    }
    assert sum(chunk["count"] for chunk in layer["chunks"]) == 5000
    assert sorted(index for tier in layer["tiers"] for index in tier) == list(
        range(len(layer["chunks"]))
    )
    assert (dst / POSTER_NAME).exists()


def test_chunk_bytes_match_their_manifest_entry_and_reconstruct_world_positions(tmp_path):
    rows = splat_rows(2000)
    src = write_ply(tmp_path / "scene.ply", props=POSTSHOT_SPLAT_PROPS, rows=rows)

    _result, document, dst = _derive(tmp_path, src, max_splats_per_chunk=300, tier_count=2)

    world = np.stack([rows["x"], rows["y"], rows["z"]], axis=1)
    seen = []
    for entry in document["layers"][0]["chunks"]:
        blob = (dst / f"chunk_{entry['index']:05d}.bin").read_bytes()
        magic, version, _flags, count, sh_degree, bbox_min, bbox_max, origin = _chunk_header(blob)
        assert magic == b"USX1"
        assert version == 1
        assert sh_degree == 0  # measured, and this file's f_rest is all zero
        assert count == entry["count"]
        assert len(blob) == entry["bytes"] == 64 + count * 32
        assert np.array_equal(origin, np.asarray(entry["origin"], dtype=np.float32))
        ext_a = np.frombuffer(blob[64 : 64 + count * 16], "<u4").reshape(count, 4)
        local = ext_a[:, 0:3].copy().view(np.float32)
        assert np.all(local >= bbox_min) and np.all(local <= bbox_max)
        seen.append(local + origin)
    # Every source position comes back, exactly, from chunk-local + origin.
    recovered = np.concatenate(seen)
    assert recovered.shape == world.shape
    assert np.array_equal(recovered[np.lexsort(recovered.T)], world[np.lexsort(world.T)])


def test_point_cloud_derive_emits_upc1_with_source_srgb_colors(tmp_path):
    src = write_colmap_points(tmp_path / "points.ply", count=3000)
    header = ply.read_header(src)
    colors = np.stack(
        [
            np.concatenate([block[name] for block in ply.iter_chunks(src, header, names=(name,))])
            for name in ("red", "green", "blue")
        ],
        axis=1,
    )

    _result, document, dst = _derive(tmp_path, src, max_splats_per_chunk=400, tier_count=2)

    assert document["scene_kind"] == "pointcloud"
    layer = document["layers"][0]
    assert layer["type"] == "points"
    assert layer["encoding"] == "upc-v1"
    assert layer["quantization"] == {"center": "f32-exact", "color": "u8-srgb"}
    seen = []
    for entry in layer["chunks"]:
        blob = (dst / f"chunk_{entry['index']:05d}.bin").read_bytes()
        magic, _version, flags, count, sh_degree, _lo, _hi, _origin = _chunk_header(blob)
        assert magic == b"UPC1"
        assert sh_degree == 0
        assert flags == 0  # no alpha property in the source
        assert len(blob) == entry["bytes"] == 64 + count * 12 + count * 4
        rgba = np.frombuffer(blob[64 + count * 12 :], np.uint8).reshape(count, 4)
        assert np.all(rgba[:, 3] == 255)
        seen.append(rgba[:, 0:3])
    recovered = np.concatenate(seen)
    # Bytes preserved exactly: no linearization on the point path.
    assert np.array_equal(np.sort(recovered, axis=0), np.sort(colors, axis=0))


def test_limitations_state_the_measured_versus_declared_sh_degree(tmp_path):
    src = write_postshot_splats(tmp_path / "zeros.ply", count=800)

    _result, document, _dst = _derive(tmp_path, src)

    joined = " ".join(document["limitations"])
    assert "declares spherical-harmonic degree 3" in joined
    assert "measured degree 0" in joined
    assert "display-referred" in joined
    assert "clamped" in joined
    assert "not the WebGL render" in joined
    assert "Tier 0" in joined
    assert "view hint" in joined


def test_limitations_state_dropped_bands_when_the_source_really_has_them(tmp_path):
    f_rest = np.zeros((600, 45), np.float32)
    f_rest[:, 8] = 0.5  # band 3
    src = write_postshot_splats(tmp_path / "sh3.ply", count=600, f_rest=f_rest)

    result, document, _dst = _derive(tmp_path, src)

    assert result["measured_sh_degree"] == 3
    assert document["source"]["measured_sh_degree"] == 3
    joined = " ".join(document["limitations"])
    assert "view-dependent spherical-harmonic band(s)" in joined
    assert "declares spherical-harmonic degree" not in joined  # nothing was over-declared


def test_clamped_color_fraction_is_reported(tmp_path):
    rows = splat_rows(400)
    for i in range(3):
        rows[f"f_dc_{i}"] = np.full(400, 6.0, dtype=np.float32)  # 0.5 + C0*6 = 2.19

    src = write_ply(tmp_path / "hot.ply", props=POSTSHOT_SPLAT_PROPS, rows=rows)
    _result, document, _dst = _derive(tmp_path, src)

    assert document["layers"][0]["quantization"]["clamped_color_fraction"] == 1.0
    assert "100.00%" in " ".join(document["limitations"])


def test_poster_is_a_small_deterministic_png(tmp_path):
    src = write_postshot_splats(tmp_path / "scene.ply", count=2000)

    _result, _document, dst = _derive(tmp_path, src)
    first = (dst / POSTER_NAME).read_bytes()
    (dst / POSTER_NAME).unlink()
    _derive(tmp_path, src)
    second = (dst / POSTER_NAME).read_bytes()

    with Image.open(dst / POSTER_NAME) as image:
        assert image.mode == "RGBA"
        assert max(image.size) <= 512
        assert np.asarray(image)[:, :, 3].any()  # something was actually drawn
    assert first == second  # byte-identical across runs


def test_manifest_is_written_last_so_a_reader_never_sees_a_partial_derive(tmp_path):
    src = write_postshot_splats(tmp_path / "scene.ply", count=500)
    dst = tmp_path / "derived"
    order: list[str] = []

    def spy(path, **kwargs):
        order.append(os.path.basename(str(path)))
        from ultra_deepagents.scene3d import poster

        return poster.render_poster(path, **kwargs)

    run_scene3d_derive_job(
        Scene3dDeriveJob(resource_id="f", src_path=str(src), dst_dir=str(dst)), poster_fn=spy
    )

    written = sorted(os.listdir(dst))
    assert order == [POSTER_NAME]
    assert MANIFEST_NAME in written
    assert os.path.getmtime(dst / MANIFEST_NAME) >= os.path.getmtime(dst / POSTER_NAME)


def test_missing_source_is_deterministic_and_writes_a_failure_marker(tmp_path):
    dst = tmp_path / "derived"

    with pytest.raises(DeterministicDerivativeError) as caught:
        run_scene3d_derive_job(str(tmp_path / "absent.ply"), str(dst), resource_id="f")

    assert caught.value.code == "scene_source_missing"
    marker = json.loads(open(failure_marker_path(str(dst))).read())
    assert marker["code"] == "scene_source_missing"
    assert marker["schema"] == "ultra.scene3d-derive-failure.v1"


def test_ascii_source_is_deterministic_not_transient(tmp_path):
    src = tmp_path / "ascii.ply"
    src.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\n"
        "property float z\nend_header\n0 0 0\n1 1 1\n"
    )

    with pytest.raises(DeterministicDerivativeError) as caught:
        run_scene3d_derive_job(str(src), str(tmp_path / "derived"), resource_id="f")

    assert caught.value.code == "unsupported_scene_encoding"


def test_truncated_source_is_deterministic(tmp_path):
    src = write_postshot_splats(tmp_path / "cut.ply", count=200)
    payload = src.read_bytes()
    src.write_bytes(payload[: len(payload) - 236 * 20])  # 20 records short of the header

    with pytest.raises(DeterministicDerivativeError) as caught:
        run_scene3d_derive_job(str(src), str(tmp_path / "derived"), resource_id="f")

    assert caught.value.code == "truncated_scene_source"


def test_unreadable_source_is_transient(tmp_path):
    src = write_postshot_splats(tmp_path / "scene.ply", count=32)

    def unavailable(path):
        raise OSError(5, "device is busy")

    with pytest.raises(TransientDerivativeError) as caught:
        run_scene3d_derive_job(
            str(src), str(tmp_path / "derived"), resource_id="f", read_header_fn=unavailable
        )

    assert caught.value.code == "scene_source_unavailable"
    assert not os.path.exists(failure_marker_path(str(tmp_path / "derived")))


def test_a_successful_derive_clears_a_previous_failure_marker(tmp_path):
    dst = tmp_path / "derived"
    dst.mkdir()
    marker = failure_marker_path(str(dst))
    with open(marker, "w") as stream:
        stream.write("{}")
    src = write_postshot_splats(tmp_path / "scene.ply", count=64)

    run_scene3d_derive_job(str(src), str(dst), resource_id="f")

    assert not os.path.exists(marker)


def test_the_runner_is_injectable_without_touching_the_filesystem_reader(tmp_path):
    """The point of the injection seams: exercise the job with a stub reader."""
    src = write_postshot_splats(tmp_path / "scene.ply", count=128)
    calls: list[int] = []

    def measure(path, header, sample):
        calls.append(sample)
        return 0

    result = run_scene3d_derive_job(
        Scene3dDeriveJob(
            resource_id="f", src_path=str(src), dst_dir=str(tmp_path / "d"), sh_sample=77
        ),
        measure_sh_fn=measure,
    )

    assert calls == [77]
    assert result["total"] == 128


@pytest.mark.slow
@pytest.mark.skipif(
    not REAL_SPLAT_PLY or not os.path.exists(REAL_SPLAT_PLY or ""),
    reason="set ULTRA_SCENE3D_REAL_PLY to the 14.5M-splat source to run the full derive",
)
def test_real_splat_file_derives_with_exact_counts(tmp_path):
    """The whole 3.4 GB file: header, measured SH degree, and an exact element count."""
    header = ply.read_header(REAL_SPLAT_PLY)
    assert header.count == REAL_SPLAT_COUNT
    assert header.stride == REAL_SPLAT_STRIDE
    assert header.data_offset == 1512
    assert ply.detect_scene_kind(header) == "splat"
    assert ply.declared_sh_degree(header) == 3
    assert ply.measured_sh_degree(REAL_SPLAT_PLY, header) == 0  # 45 declared, all zero

    dst = tmp_path / "derived"
    result = run_scene3d_derive_job(
        {"resource_id": "willa", "src_path": REAL_SPLAT_PLY, "dst_dir": str(dst)}
    )

    assert result["total"] == REAL_SPLAT_COUNT
    document = json.loads((dst / MANIFEST_NAME).read_text())
    assert document["source"]["vertex_count"] == REAL_SPLAT_COUNT
    assert document["source"]["stride_bytes"] == REAL_SPLAT_STRIDE
    assert document["source"]["writer"] == "postshot"
    assert document["source"]["declared_sh_degree"] == 3
    assert document["source"]["measured_sh_degree"] == 0
    layer = document["layers"][0]
    assert layer["total"] == REAL_SPLAT_COUNT
    assert sum(chunk["count"] for chunk in layer["chunks"]) == REAL_SPLAT_COUNT
    assert (
        sum(layer["chunks"][index]["count"] for tier in layer["tiers"] for index in tier)
        == REAL_SPLAT_COUNT
    )
    assert all(
        chunk["bytes"] == spark_encode.CHUNK_HEADER_BYTES + chunk["count"] * 32
        for chunk in layer["chunks"]
    )
