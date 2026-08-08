"""Header-driven PLY reading: both real strides, and the measured-vs-declared SH degree.

The synthetic builder here is the fixture the rest of the scene3d suite imports, so it
has to produce files that are byte-identical in structure to the two measured sources:
Postshot's 236 B LF-terminated splat record and the 27 B CRLF-terminated point record.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from ultra_deepagents.scene3d import ply

# Property layouts of the two measured files (Appendix A of the contract).
POSTSHOT_SPLAT_PROPS = (
    ["x", "y", "z"]
    + [f"f_dc_{i}" for i in range(3)]
    + [f"f_rest_{i}" for i in range(45)]
    + ["opacity"]
    + [f"scale_{i}" for i in range(3)]
    + [f"rot_{i}" for i in range(4)]
)
INRIA_SPLAT_PROPS = (
    ["x", "y", "z", "nx", "ny", "nz"]
    + [f"f_dc_{i}" for i in range(3)]
    + [f"f_rest_{i}" for i in range(45)]
    + ["opacity"]
    + [f"scale_{i}" for i in range(3)]
    + [f"rot_{i}" for i in range(4)]
)


def write_ply(
    path,
    *,
    props,
    rows,
    types=None,
    comments=(),
    newline="\n",
    fmt="binary_little_endian",
):
    """Write a synthetic binary PLY. ``rows`` maps property name -> 1-D array."""
    types = types or {}
    count = len(next(iter(rows.values())))
    header = [f"ply{newline}", f"format {fmt} 1.0{newline}", f"element vertex {count}{newline}"]
    header.extend(f"comment {comment}{newline}" for comment in comments)
    header.extend(f"property {types.get(name, 'float')} {name}{newline}" for name in props)
    header.append(f"end_header{newline}")
    dtype = np.dtype(
        [
            (
                name,
                {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4"}[
                    types.get(name, "float")
                ],
            )
            for name in props
        ]
    )
    record = np.zeros(count, dtype=dtype)
    for name, values in rows.items():
        record[name] = values
    with open(path, "wb") as stream:
        stream.write("".join(header).encode("ascii"))
        stream.write(record.tobytes())
    return path


def splat_rows(count, *, rng=None, f_rest=None):
    """A plausible splat table in the measured activation domains."""
    rng = rng or np.random.default_rng(11)
    rows = {
        "x": rng.uniform(-50, 50, count).astype(np.float32),
        "y": rng.uniform(-12, 12, count).astype(np.float32),
        "z": rng.uniform(-50, 50, count).astype(np.float32),
        "opacity": rng.uniform(-4.15, 13.2, count).astype(np.float32),
    }
    for i in range(3):
        rows[f"f_dc_{i}"] = rng.uniform(-1.8, 7.8, count).astype(np.float32)
        rows[f"scale_{i}"] = rng.uniform(-10.6, 0.59, count).astype(np.float32)
    quat = rng.normal(size=(count, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    for i in range(4):
        rows[f"rot_{i}"] = quat[:, i].astype(np.float32)
    for i in range(45):
        rows[f"f_rest_{i}"] = (
            np.zeros(count, np.float32) if f_rest is None else f_rest[:, i].astype(np.float32)
        )
    return rows


def point_rows(count, *, rng=None):
    rng = rng or np.random.default_rng(13)
    return {
        "x": rng.uniform(0, 449.7, count).astype(np.float32),
        "y": rng.uniform(0, 112.6, count).astype(np.float32),
        "z": rng.uniform(0, 1119.2, count).astype(np.float32),
        "nx": np.zeros(count, np.float32),
        "ny": np.zeros(count, np.float32),
        "nz": np.zeros(count, np.float32),
        "red": rng.integers(0, 256, count).astype(np.uint8),
        "green": rng.integers(0, 256, count).astype(np.uint8),
        "blue": rng.integers(0, 256, count).astype(np.uint8),
    }


def write_postshot_splats(path, count=64, **kwargs):
    return write_ply(
        path,
        props=POSTSHOT_SPLAT_PROPS,
        rows=splat_rows(count, **kwargs),
        comments=("postshot.anti_aliasing=1",),
    )


def write_colmap_points(path, count=64, **kwargs):
    return write_ply(
        path,
        props=["x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"],
        rows=point_rows(count, **kwargs),
        types={"red": "uchar", "green": "uchar", "blue": "uchar"},
        newline="\r\n",
    )


def test_postshot_stride_236_is_derived_from_the_header(tmp_path):
    header = ply.read_header(write_postshot_splats(tmp_path / "splats.ply", count=8))

    assert header.stride == 236  # 59 float properties, no normals
    assert header.count == 8
    assert header.props[0].offset == 0
    assert header.prop("opacity").offset == 4 * (3 + 3 + 45)
    assert header.prop("rot_3").offset == 232
    assert ply.detect_scene_kind(header) == "splat"
    assert ply.source_writer(header) == "postshot"


def test_inria_stride_248_shifts_every_offset_by_the_normals(tmp_path):
    rows = splat_rows(8)
    rows.update({name: np.zeros(8, np.float32) for name in ("nx", "ny", "nz")})
    path = write_ply(tmp_path / "inria.ply", props=INRIA_SPLAT_PROPS, rows=rows)
    header = ply.read_header(path)

    assert header.stride == 248
    # Every splat field sits 12 bytes later than in the Postshot layout: this is exactly
    # the misread a hardcoded stride produces.
    assert header.prop("opacity").offset == 4 * (3 + 3 + 3 + 45)
    assert header.prop("rot_3").offset == 244
    assert ply.detect_scene_kind(header) == "splat"


def test_point_cloud_header_handles_crlf_and_mixed_property_widths(tmp_path):
    path = write_colmap_points(tmp_path / "points.ply", count=5)
    header = ply.read_header(path)

    assert header.stride == 27  # 6 floats + 3 bytes
    assert header.data_offset == os.path.getsize(path) - 5 * 27
    assert header.prop("red").offset == 24
    assert ply.detect_scene_kind(header) == "pointcloud"
    assert ply.declared_sh_degree(header) == 0


def test_iter_chunks_round_trips_every_record_in_file_order(tmp_path):
    rows = splat_rows(1000)
    path = write_ply(tmp_path / "splats.ply", props=POSTSHOT_SPLAT_PROPS, rows=rows)
    header = ply.read_header(path)

    blocks = list(ply.iter_chunks(path, header, 128, names=("x", "opacity")))

    assert sum(int(block.shape[0]) for block in blocks) == 1000
    assert np.array_equal(np.concatenate([block["x"] for block in blocks]), rows["x"])
    assert np.array_equal(np.concatenate([block["opacity"] for block in blocks]), rows["opacity"])


def test_ascii_is_detected_and_refused_with_a_clear_message(tmp_path):
    path = tmp_path / "ascii.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n1.0\n")
    with pytest.raises(ply.PlyFormatError, match="ASCII PLY.*binary"):
        ply.read_header(path)


def test_list_property_is_refused_rather_than_misread(tmp_path):
    path = tmp_path / "mesh.ply"
    path.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        b"property list uchar int vertex_index\nend_header\n"
    )
    with pytest.raises(ply.PlyFormatError, match="stride-addressable"):
        ply.read_header(path)


def test_non_vertex_xyz_element_is_not_admitted_as_a_point_cloud(tmp_path):
    path = tmp_path / "samples.ply"
    path.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement samples 1\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n" + bytes(12)
    )

    with pytest.raises(ply.PlyFormatError, match="no vertex element"):
        ply.read_header(path)


def test_header_inventory_records_omitted_face_topology(tmp_path):
    path = tmp_path / "mesh.ply"
    header_bytes = (
        b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"element face 1\nproperty list uchar int vertex_indices\nend_header\n"
    )
    path.write_bytes(header_bytes + bytes(24) + b"\x03" + bytes(12))

    header = ply.read_header(path)

    assert header.data_offset == len(header_bytes)
    assert [(item.name, item.count, item.has_list) for item in header.elements] == [
        ("vertex", 2, False),
        ("face", 1, True),
    ]


def test_measured_sh_degree_is_zero_when_45_declared_coefficients_are_all_zero(tmp_path):
    path = write_postshot_splats(tmp_path / "zeros.ply", count=4096)
    header = ply.read_header(path)

    assert ply.declared_sh_degree(header) == 3  # the header claims the full layout
    assert ply.measured_sh_degree(path, header, 1000) == 0
    assert ply.measured_sh_degree(path, header, 100000) == 0  # full-scan path too


@pytest.mark.parametrize(
    ("populated_slots", "expected"),
    [
        ((0, 15, 30), 1),  # band 1: coefficient 0 of each channel
        ((7,), 2),  # band 2 tops out at coefficient 7
        ((8,), 3),  # coefficient 8 is the first of band 3
        ((3, 40), 3),  # highest populated band wins, not the count of them
    ],
)
def test_measured_sh_degree_reports_the_highest_populated_band(tmp_path, populated_slots, expected):
    count = 512
    f_rest = np.zeros((count, 45), np.float32)
    for slot in populated_slots:
        f_rest[:, slot] = 0.25
    path = write_postshot_splats(tmp_path / f"sh{expected}.ply", count=count, f_rest=f_rest)
    header = ply.read_header(path)

    assert ply.measured_sh_degree(path, header, 200) == expected


def test_measured_sh_degree_samples_randomly_not_the_prefix(tmp_path):
    """A file whose higher bands only appear late must not measure as degree 0."""
    count = 20000
    f_rest = np.zeros((count, 45), np.float32)
    f_rest[count - 200 :, 8] = 1.0  # band 3, only in the tail
    path = write_postshot_splats(tmp_path / "tail.ply", count=count, f_rest=f_rest)
    header = ply.read_header(path)

    assert ply.measured_sh_degree(path, header, 8000) == 3


def test_scene_kind_requires_the_whole_splat_parameter_set(tmp_path):
    rows = {name: np.zeros(4, np.float32) for name in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2")}
    path = write_ply(tmp_path / "half.ply", props=list(rows), rows=rows)

    # Any splat-specific field makes this an attempted splat schema. Silently
    # downgrading it to points would omit declared appearance fields while claiming
    # the source was rendered faithfully.
    with pytest.raises(ply.PlyFormatError, match="incomplete Gaussian-splat schema"):
        ply.detect_scene_kind(ply.read_header(path))
