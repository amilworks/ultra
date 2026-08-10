"""End-to-end derive: manifest schema, chunk bytes, poster, failure taxonomy.

The slow test at the bottom runs the real 14,469,103-splat file. It is skipped unless
``ULTRA_SCENE3D_REAL_PLY`` names it, because it reads 3.4 GB and takes minutes.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import zipfile

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
    StaleDerivativeJobError,
    TransientDerivativeError,
)
from ultra_deepagents.scene3d import job as scene_job
from ultra_deepagents.scene3d import ply, rad_lod, spark_encode
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
    payload = src.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    dst = tmp_path / "derived" / f"file-1__scene3d.v5.sha256-{digest}"
    result = run_scene3d_derive_job(
        {
            "resource_id": "file-1",
            "src_path": str(src),
            "dst_dir": str(dst),
            "source_sha256": digest,
            "source_size_bytes": len(payload),
            **options,
        }
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
        "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "property_provenance": {
            "preserved": [
                "x",
                "y",
                "z",
                "f_dc_0",
                "f_dc_1",
                "f_dc_2",
                "opacity",
                "scale_0",
                "scale_1",
                "scale_2",
                "rot_0",
                "rot_1",
                "rot_2",
                "rot_3",
            ],
            "synthesized": [],
            "omitted": [f"f_rest_{index}" for index in range(45)],
            "omitted_elements": [],
        },
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
        "out_of_range_color_fraction",
    }
    assert sum(chunk["count"] for chunk in layer["chunks"]) == 5000
    assert sorted(index for tier in layer["tiers"] for index in tier) == list(
        range(len(layer["chunks"]))
    )
    assert (dst / POSTER_NAME).exists()


def test_colmap_zip_with_multiple_models_fails_before_model_selection(tmp_path):
    src = tmp_path / "ambiguous.zip"
    with zipfile.ZipFile(src, "w") as archive:
        for model in ("sparse/0", "sparse/1"):
            archive.writestr(f"{model}/cameras.bin", b"")
            archive.writestr(f"{model}/images.bin", b"")

    with pytest.raises(DeterministicDerivativeError, match="ambiguous_colmap_models"):
        run_scene3d_derive_job(
            str(src),
            str(tmp_path / "derived"),
            resource_id="file-ambiguous",
        )


def test_colmap_zip_discovery_is_case_insensitive(tmp_path):
    src = tmp_path / "mixed-case.zip"
    with zipfile.ZipFile(src, "w") as archive:
        archive.writestr("Sparse/0/CAMERAS.BIN", b"")
        archive.writestr("Sparse/0/IMAGES.BIN", b"")
        archive.writestr("Sparse/0/POINTS3D.BIN", b"")

    destination = tmp_path / "extracted"
    destination.mkdir()

    assert scene_job._extract_colmap_zip(str(src), str(destination)) == str(
        destination / "Sparse" / "0"
    )


def test_production_splat_delivery_publishes_paged_quality_rad(tmp_path, monkeypatch):
    src = write_postshot_splats(tmp_path / "scene.ply", count=5000)

    def fake_build(_source, directory, *, retained_sh_degree):
        # The fixture declares degree 3 but its full source contains only zero-valued
        # higher bands, so the exact scan safely strips those empty texture planes.
        assert retained_sh_degree == 0
        root = os.fspath(directory)
        with open(os.path.join(root, "scene-lod.rad"), "wb") as stream:
            stream.write(b"RAD header")
        with open(os.path.join(root, "scene-lod-0.radc"), "wb") as stream:
            stream.write(b"RAD page")
        return rad_lod.RadLodResult(
            header=rad_lod.RadArtifact("scene-lod.rad", 10),
            chunks=[rad_lod.RadArtifact("scene-lod-0.radc", 8)],
            method="bhatt-lod-quality",
            builder_revision=rad_lod.RAD_BUILDER_REVISION,
        )

    monkeypatch.setattr(scene_job.rad_lod, "build_paged_rad", fake_build)
    result, document, dst = _derive(
        tmp_path,
        src,
        max_splats_per_chunk=800,
        tier_count=3,
        splat_delivery="spark-rad-v1",
    )

    assert result["chunk_count"] == 2
    assert result["tier_count"] == 0
    assert document["generator_revision"] == "scene3d-rad-v5"
    assert document["source"]["property_provenance"]["synthesized"] == []
    assert document["source"]["property_provenance"]["omitted"] == [
        f"f_rest_{index}" for index in range(45)
    ]
    layer = document["layers"][0]
    assert layer["encoding"] == "spark-rad-v1"
    assert layer["chunks"] == []
    assert layer["tiers"] == []
    assert layer["lod"]["method"] == "bhatt-lod-quality"
    assert layer["lod"]["max_sh_degree"] == 0
    assert layer["lod"]["header"] == {"name": "scene-lod.rad", "bytes": 10}
    assert document["service_urls"]["lod"].endswith("/scene3d/lod/scene-lod.rad")
    assert (dst / "scene-lod-0.radc").read_bytes() == b"RAD page"

    def unexpected_rad_rederive(*_args, **_kwargs):
        raise AssertionError("a committed RAD generation was derived twice")

    monkeypatch.setattr(scene_job.rad_lod, "build_paged_rad", unexpected_rad_rederive)
    payload = src.read_bytes()
    reused = run_scene3d_derive_job(
        {
            "resource_id": "file-1",
            "src_path": str(src),
            "dst_dir": str(dst),
            "source_sha256": hashlib.sha256(payload).hexdigest(),
            "source_size_bytes": len(payload),
            "splat_delivery": "spark-rad-v1",
        }
    )
    assert reused["reused"] is True
    assert reused["chunk_count"] == 2


def test_production_splat_delivery_retains_a_band_populated_only_in_the_final_row(
    tmp_path, monkeypatch
):
    count = 512
    f_rest = np.zeros((count, 45), np.float32)
    f_rest[-1, 8] = 0.25  # first coefficient in degree 3, outside a tiny sample
    src = write_postshot_splats(tmp_path / "late-sh.ply", count=count, f_rest=f_rest)

    def fake_build(_source, directory, *, retained_sh_degree):
        assert retained_sh_degree == 3
        root = os.fspath(directory)
        with open(os.path.join(root, "scene-lod.rad"), "wb") as stream:
            stream.write(b"RAD header")
        with open(os.path.join(root, "scene-lod-0.radc"), "wb") as stream:
            stream.write(b"RAD page")
        return rad_lod.RadLodResult(
            header=rad_lod.RadArtifact("scene-lod.rad", 10),
            chunks=[rad_lod.RadArtifact("scene-lod-0.radc", 8)],
            method="bhatt-lod-quality",
            builder_revision=rad_lod.RAD_BUILDER_REVISION,
        )

    monkeypatch.setattr(scene_job.rad_lod, "build_paged_rad", fake_build)
    result, document, _dst = _derive(
        tmp_path,
        src,
        sh_sample=1,
        splat_delivery="spark-rad-v1",
    )

    assert result["measured_sh_degree"] == 3
    assert document["source"]["measured_sh_degree"] == 3
    assert document["layers"][0]["lod"]["max_sh_degree"] == 3


def test_production_splat_delivery_rejects_nonfinite_coordinates_before_build(
    tmp_path, monkeypatch
):
    rows = splat_rows(32)
    rows["x"][7] = np.nan
    src = write_ply(tmp_path / "invalid-scene.ply", props=POSTSHOT_SPLAT_PROPS, rows=rows)

    def unexpected_build(*_args, **_kwargs):
        raise AssertionError("the RAD builder must not receive non-finite geometry")

    monkeypatch.setattr(scene_job.rad_lod, "build_paged_rad", unexpected_build)
    with pytest.raises(DeterministicDerivativeError, match="nonfinite_scene_coordinates"):
        _derive(tmp_path, src, splat_delivery="spark-rad-v1")


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
    assert document["source"]["property_provenance"] == {
        "preserved": ["x", "y", "z", "red", "green", "blue"],
        "synthesized": ["alpha=255"],
        "omitted": ["nx", "ny", "nz"],
        "omitted_elements": [],
    }
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


def test_colorless_point_cloud_reports_synthesized_white_display_color(tmp_path):
    rows = {
        "x": np.asarray([0.0, 1.0], dtype=np.float32),
        "y": np.asarray([0.0, 1.0], dtype=np.float32),
        "z": np.asarray([0.0, 1.0], dtype=np.float32),
        "confidence": np.asarray([0.9, 0.8], dtype=np.float32),
    }
    src = write_ply(tmp_path / "colorless.ply", props=list(rows), rows=rows)

    _result, document, dst = _derive(tmp_path, src, max_splats_per_chunk=256)

    assert document["source"]["property_provenance"] == {
        "preserved": ["x", "y", "z"],
        "synthesized": ["red=255", "green=255", "blue=255", "alpha=255"],
        "omitted": ["confidence"],
        "omitted_elements": [],
    }
    layer = document["layers"][0]
    blob = (dst / f"chunk_{layer['chunks'][0]['index']:05d}.bin").read_bytes()
    rgba = np.frombuffer(blob[64 + 2 * 12 :], np.uint8).reshape(2, 4)
    assert np.all(rgba == 255)
    assert "no RGB colour" in " ".join(document["limitations"])


def test_mesh_ply_discloses_omitted_face_topology(tmp_path):
    src = tmp_path / "mesh.ply"
    header = (
        b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"element face 1\nproperty list uchar int vertex_indices\nend_header\n"
    )
    src.write_bytes(
        header
        + struct.pack("<ffffff", 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        + b"\x03"
        + struct.pack("<iii", 0, 1, 0)
    )

    _result, document, _dst = _derive(tmp_path, src, max_splats_per_chunk=256)

    assert document["scene_kind"] == "pointcloud"
    assert document["source"]["property_provenance"]["omitted_elements"] == [
        {"name": "face", "count": 1}
    ]
    assert "face topology" in " ".join(document["limitations"])


def test_ply_derivation_does_not_build_a_whole_scene_octree_or_column_table(tmp_path, monkeypatch):
    """A large scene must stay bounded by its read/chunk size, not vertex count.

    The original implementation allocated full x/y/z columns and a global octree plan.
    That peaks near 425 MB for the 53 MB real point fixture and exceeds a worker's memory
    budget for the 3.2 GB splat fixture. The streaming path must not call either helper.
    """

    src = write_colmap_points(tmp_path / "large-points.ply", count=50_000)

    def whole_scene_allocation(*_args, **_kwargs):
        raise AssertionError("whole-scene allocation path was used")

    monkeypatch.setattr(scene_job, "_read_columns", whole_scene_allocation)
    monkeypatch.setattr(scene_job.chunker, "build_chunk_plan", whole_scene_allocation)

    result, document, _dst = _derive(
        tmp_path,
        src,
        max_splats_per_chunk=1_000,
        tier_count=4,
        preview_points=5_000,
    )

    layer = document["layers"][0]
    assert result["total"] == 50_000
    assert sum(chunk["count"] for chunk in layer["chunks"]) == 50_000
    tier_zero_count = sum(layer["chunks"][index]["count"] for index in layer["tiers"][0])
    assert 0 < tier_zero_count < 50_000


def test_limitations_state_the_measured_versus_declared_sh_degree(tmp_path):
    src = write_postshot_splats(tmp_path / "zeros.ply", count=800)

    _result, document, _dst = _derive(tmp_path, src)

    joined = " ".join(document["limitations"])
    assert "declares spherical-harmonic degree 3" in joined
    assert "measured degree 0" in joined
    assert "display-referred" in joined
    assert "outside [0,1]" in joined
    assert "remain preserved" in joined
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


def test_out_of_range_color_fraction_is_reported_without_altering_splat_values(tmp_path):
    rows = splat_rows(400)
    for i in range(3):
        rows[f"f_dc_{i}"] = np.full(400, 6.0, dtype=np.float32)  # 0.5 + C0*6 = 2.19

    src = write_ply(tmp_path / "hot.ply", props=POSTSHOT_SPLAT_PROPS, rows=rows)
    _result, document, _dst = _derive(tmp_path, src)

    assert document["layers"][0]["quantization"]["out_of_range_color_fraction"] == 1.0
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


def test_strict_job_rejects_a_catalog_digest_mismatch_without_publishing(tmp_path):
    src = write_postshot_splats(tmp_path / "scene.ply", count=64)
    wrong_digest = "0" * 64
    dst = tmp_path / "derived" / f"file-1__scene3d.v5.sha256-{wrong_digest}"

    with pytest.raises(StaleDerivativeJobError) as caught:
        run_scene3d_derive_job(
            {
                "resource_id": "file-1",
                "src_path": str(src),
                "dst_dir": str(dst),
                "source_sha256": wrong_digest,
                "source_size_bytes": src.stat().st_size,
            }
        )

    assert caught.value.code == "catalog_source_digest_mismatch"
    assert not dst.exists()
    assert not os.path.exists(failure_marker_path(str(dst)))


@pytest.mark.parametrize("legacy_revision", ("", ".v2"))
def test_strict_job_rejects_a_legacy_generation_name(tmp_path, legacy_revision):
    src = write_postshot_splats(tmp_path / "scene.ply", count=64)
    payload = src.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    legacy_dst = tmp_path / "derived" / f"file-1__scene3d{legacy_revision}.sha256-{digest}"

    with pytest.raises(StaleDerivativeJobError) as caught:
        run_scene3d_derive_job(
            {
                "resource_id": "file-1",
                "src_path": str(src),
                "dst_dir": str(legacy_dst),
                "source_sha256": digest,
                "source_size_bytes": len(payload),
                "splat_delivery": "spark-rad-v1",
            }
        )

    assert caught.value.code == "scene_destination_identity_mismatch"
    assert not legacy_dst.exists()


def test_strict_deterministic_failure_is_atomic_and_does_not_leak_source_path(tmp_path):
    src = write_postshot_splats(tmp_path / "private-scene-name.ply", count=80)
    payload = src.read_bytes()[:-236]
    src.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    dst = tmp_path / "derived" / f"file-1__scene3d.v5.sha256-{digest}"

    with pytest.raises(DeterministicDerivativeError) as caught:
        run_scene3d_derive_job(
            {
                "resource_id": "file-1",
                "src_path": str(src),
                "dst_dir": str(dst),
                "source_sha256": digest,
                "source_size_bytes": len(payload),
            }
        )

    assert caught.value.code == "truncated_scene_source"
    marker_path = failure_marker_path(str(dst))
    marker_text = open(marker_path, encoding="utf-8").read()
    marker = json.loads(marker_text)
    assert marker["source_sha256"] == digest
    assert marker["code"] == "truncated_scene_source"
    assert "src_path" not in marker_text
    assert str(src) not in marker_text
    assert not dst.exists()
    assert not list(dst.parent.glob(f".{dst.name}.tmp-*"))


def test_strict_redelivery_reuses_the_committed_generation_without_rederiving(
    tmp_path, monkeypatch
):
    src = write_postshot_splats(tmp_path / "scene.ply", count=128)
    first, _document, dst = _derive(tmp_path, src)

    def unexpected_rederive(*_args, **_kwargs):
        raise AssertionError("a committed immutable generation was derived twice")

    monkeypatch.setattr(scene_job.streaming, "derive_ply", unexpected_rederive)
    payload = src.read_bytes()
    second = run_scene3d_derive_job(
        {
            "resource_id": "file-1",
            "src_path": str(src),
            "dst_dir": str(dst),
            "source_sha256": hashlib.sha256(payload).hexdigest(),
            "source_size_bytes": len(payload),
        }
    )

    assert first["status"] == second["status"] == "succeeded"
    assert second["reused"] is True


def test_reconstruction_commit_requires_every_advertised_camera_preview(tmp_path):
    payload = b"immutable reconstruction archive"
    digest = hashlib.sha256(payload).hexdigest()
    source = tmp_path / "scene.zip"
    source.write_bytes(payload)
    destination = tmp_path / "derived" / f"file-1__scene3d.v5.sha256-{digest}"
    destination.mkdir(parents=True)
    (destination / POSTER_NAME).write_bytes(b"poster")
    (destination / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schema": "ultra.scene3d.v1",
                "generator_revision": "scene3d-rad-v5",
                "scene_kind": "reconstruction",
                "source": {"sha256": digest, "bytes": len(payload)},
                "layers": [],
                "reconstruction": {"preview_images": 1},
            }
        )
    )
    job = Scene3dDeriveJob.from_dict(
        {
            "resource_id": "file-1",
            "src_path": str(source),
            "dst_dir": str(destination),
            "source_sha256": digest,
            "source_size_bytes": len(payload),
        }
    )

    assert scene_job._published_scene_matches(destination, job) is False
    (destination / "camera-image_00000.jpg").write_bytes(b"jpeg")
    assert scene_job._published_scene_matches(destination, job) is True


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("max_splats_per_chunk", 255),
        ("max_splats_per_chunk", 250_001),
        ("tier_count", 9),
        ("sh_sample", 500_001),
        ("poster_sample", 500_001),
        ("preview_splats", 100_001),
        ("preview_points", 280_001),
    ],
)
def test_queue_job_bounds_every_allocation_multiplier(tmp_path, option, value):
    src = write_postshot_splats(tmp_path / "scene.ply", count=32)
    payload = src.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    dst = tmp_path / "derived" / f"file-1__scene3d.v5.sha256-{digest}"

    with pytest.raises(DeterministicDerivativeError) as caught:
        run_scene3d_derive_job(
            {
                "resource_id": "file-1",
                "src_path": str(src),
                "dst_dir": str(dst),
                "source_sha256": digest,
                "source_size_bytes": len(payload),
                option: value,
            }
        )

    assert caught.value.code == "invalid_scene_job"
    assert not dst.exists()
    assert json.loads(open(failure_marker_path(str(dst)), encoding="utf-8").read())["code"] == (
        "invalid_scene_job"
    )


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
