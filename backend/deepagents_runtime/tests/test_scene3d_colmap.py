"""COLMAP sparse-model reading, and the cameras layer it feeds into the derive.

Every fixture is synthesised here, by the writers at the top of the file, so the suite
carries no external data and the *exact* byte layout under test is visible beside the
assertions. The writers emit the real COLMAP layouts: a 24-byte camera head plus
model-dependent ``f64`` params, a 64-byte image head followed by a NUL-terminated name of
arbitrary length and a 24-byte-per-entry observation block, and a 51-byte point head
followed by an 8-byte-per-entry track.

The two things most worth breaking on are asserted explicitly:

- the observation block is **seeked over, never read** (a counting stream proves it);
- ``qvec``/``tvec`` reach the wire **verbatim** — world-to-camera, ``(w, x, y, z)`` — and
  the derive performs neither the inversion nor the handedness flip that the renderer
  owns (contract §2).
"""

from __future__ import annotations

import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from ultra_deepagents.imaging.derivative_manifest import DeterministicDerivativeError
from ultra_deepagents.scene3d import colmap
from ultra_deepagents.scene3d.job import MANIFEST_NAME, POSTER_NAME, run_scene3d_derive_job

# ---------------------------------------------------------------------------
# fixture writers — the real binary and text layouts, nothing derived from memory
# ---------------------------------------------------------------------------

OBSERVATION = np.dtype([("x", "<f8"), ("y", "<f8"), ("point3d_id", "<i8")])

# (camera_id, model_id, width, height, params)
PINHOLE_CAMERA = (1, 1, 1920, 1080, (1000.0, 1001.5, 960.0, 540.25))
# OPENCV with real distortion: the frusta cannot apply it, and the manifest must say so.
OPENCV_CAMERA = (2, 4, 800, 600, (500.0, 501.0, 400.0, 300.0, -0.21, 0.03, 0.001, -0.002))
# SIMPLE_RADIAL with k = 0 is exactly a SIMPLE_PINHOLE; it must NOT count as distorted.
UNDISTORTED_RADIAL = (3, 2, 640, 480, (450.0, 320.0, 240.0, 0.0))


def camera_record(camera_id, model_id, width, height, params):
    head = struct.pack("<IiQQ", camera_id, model_id, width, height)
    return head + struct.pack(f"<{len(params)}d", *params)


def write_cameras_bin(path, cameras):
    body = b"".join(camera_record(*camera) for camera in cameras)
    path.write_bytes(struct.pack("<Q", len(cameras)) + body)
    return path


def write_cameras_txt(path, cameras):
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(cameras)}",
    ]
    for camera_id, model_id, width, height, params in cameras:
        fields = [str(camera_id), colmap.CAMERA_MODELS[model_id].name, str(width), str(height)]
        fields.extend(repr(float(value)) for value in params)
        lines.append(" ".join(fields))
    path.write_text("\n".join(lines) + "\n")
    return path


def observation_block(count, seed=0):
    rng = np.random.default_rng(seed)
    block = np.empty(count, dtype=OBSERVATION)
    block["x"] = rng.random(count) * 1920.0
    block["y"] = rng.random(count) * 1080.0
    block["point3d_id"] = rng.integers(-1, 4096, count)
    return block


def image_record(image, seed=0):
    """(image_id, qvec_wxyz, tvec, camera_id, name, num_points2D) -> bytes."""
    image_id, qvec, tvec, camera_id, name, num_points2d = image
    blob = struct.pack("<I7dI", image_id, *qvec, *tvec, camera_id)
    blob += name.encode("utf-8") + b"\x00"
    blob += struct.pack("<Q", num_points2d)
    blob += observation_block(num_points2d, seed).tobytes()
    return blob


def write_images_bin(path, images):
    body = b"".join(image_record(image, seed=index) for index, image in enumerate(images))
    path.write_bytes(struct.pack("<Q", len(images)) + body)
    return path


def write_images_txt(path, images):
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    for index, (image_id, qvec, tvec, camera_id, name, num_points2d) in enumerate(images):
        fields = [str(image_id)]
        fields.extend(repr(float(value)) for value in (*qvec, *tvec))
        fields.extend([str(camera_id), name])
        lines.append(" ".join(fields))
        block = observation_block(num_points2d, seed=index)
        lines.append(
            " ".join(f"{row['x']!r} {row['y']!r} {int(row['point3d_id'])}" for row in block)
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def point_record(point):
    """(point_id, xyz, rgb, error, track) -> bytes."""
    point_id, xyz, rgb, error, track = point
    blob = struct.pack("<Q3d3BdQ", point_id, *xyz, *rgb, error, len(track))
    for image_id, point2d_idx in track:
        blob += struct.pack("<ii", image_id, point2d_idx)
    return blob


def write_points3d_bin(path, points):
    body = b"".join(point_record(point) for point in points)
    path.write_bytes(struct.pack("<Q", len(points)) + body)
    return path


def write_points3d_txt(path, points):
    lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    for point_id, xyz, rgb, error, track in points:
        fields = [str(point_id)]
        fields.extend(repr(float(value)) for value in xyz)
        fields.extend(str(int(value)) for value in rgb)
        fields.append(repr(float(error)))
        for image_id, point2d_idx in track:
            fields.extend([str(image_id), str(point2d_idx)])
        lines.append(" ".join(fields))
    path.write_text("\n".join(lines) + "\n")
    return path


def write_model(directory, *, cameras=None, images=None, points=None, text=False):
    """Write a model triple; omit a component by passing ``None`` for it."""
    directory.mkdir(parents=True, exist_ok=True)
    suffix = ".txt" if text else ".bin"
    if cameras is not None:
        writer = write_cameras_txt if text else write_cameras_bin
        writer(directory / f"cameras{suffix}", cameras)
    if images is not None:
        writer = write_images_txt if text else write_images_bin
        writer(directory / f"images{suffix}", images)
    if points is not None:
        writer = write_points3d_txt if text else write_points3d_bin
        writer(directory / f"points3D{suffix}", points)
    return directory


def sample_images(count=4, camera_id=1):
    """Distinct, deliberately asymmetric poses — any re-ordering or sign flip shows."""
    rng = np.random.default_rng(11)
    images = []
    for index in range(count):
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        images.append(
            (
                index + 1,
                tuple(float(value) for value in quat),
                (float(index) * 1.5, -0.5 * index, 2.0 + index),
                camera_id,
                f"IMG_{count - index:04d}.JPG",  # written in reverse name order
                3 * index,
            )
        )
    return images


def sample_points(count=400, seed=5):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-10.0, 10.0, size=(count, 3))
    rgb = rng.integers(0, 256, size=(count, 3))
    return [
        (
            index + 1,
            tuple(float(value) for value in xyz[index]),
            tuple(int(value) for value in rgb[index]),
            0.5 + index * 0.001,
            [(1, index), (2, index + 1)][: index % 3],  # 0, 1 or 2 track entries
        )
        for index in range(count)
    ]


class CountingStream:
    """A file wrapper that counts bytes actually read, so a `seek` skip is provable."""

    def __init__(self, stream):
        self._stream = stream
        self.bytes_read = 0

    def read(self, size=-1):
        blob = self._stream.read(size)
        self.bytes_read += len(blob)
        return blob

    def seek(self, offset, whence=0):
        return self._stream.seek(offset, whence)

    def tell(self):
        return self._stream.tell()


# ---------------------------------------------------------------------------
# cameras
# ---------------------------------------------------------------------------


def test_every_camera_model_round_trips_with_its_declared_param_count(tmp_path):
    """All eleven model ids, each with the exact params its layout declares."""
    cameras = [
        (
            model_id + 10,
            model_id,
            640 + model_id,
            480 + model_id,
            tuple(float(model_id * 100 + slot) for slot in range(model.param_count)),
        )
        for model_id, model in sorted(colmap.CAMERA_MODELS.items())
    ]
    model_dir = write_model(tmp_path / "sparse", cameras=cameras, images=[], points=[])

    read = colmap.read_cameras(model_dir)

    assert len(read) == 11
    for camera_id, model_id, width, height, params in cameras:
        camera = read[camera_id]
        assert camera.model == colmap.CAMERA_MODELS[model_id].name
        assert (camera.width, camera.height) == (width, height)
        assert camera.params == params
        assert len(camera.params) == colmap.CAMERA_MODELS[model_id].param_count
    # The param counts the contract spells out, verified against the table.
    assert [colmap.CAMERA_MODELS[i].param_count for i in range(11)] == [
        3,
        4,
        4,
        5,
        8,
        8,
        12,
        5,
        4,
        5,
        12,
    ]


def test_an_unknown_camera_model_id_is_a_format_error(tmp_path):
    path = tmp_path / "cameras.bin"
    # Three params, so the record clears the size guard and the *model id* is what fails.
    path.write_bytes(struct.pack("<Q", 1) + camera_record(1, 99, 640, 480, (1.0, 2.0, 3.0)))

    with pytest.raises(colmap.ColmapFormatError) as caught:
        colmap.read_cameras(tmp_path)

    assert "99" in str(caught.value)


def test_has_distortion_is_about_the_numbers_not_the_model_name():
    distorted = colmap.ColmapCamera(2, "OPENCV", 800, 600, OPENCV_CAMERA[4])
    zeroed = colmap.ColmapCamera(3, "SIMPLE_RADIAL", 640, 480, UNDISTORTED_RADIAL[4])
    pinhole = colmap.ColmapCamera(1, "PINHOLE", 1920, 1080, PINHOLE_CAMERA[4])

    assert colmap.has_distortion(distorted) is True
    assert colmap.has_distortion(zeroed) is False
    assert colmap.has_distortion(pinhole) is False


# ---------------------------------------------------------------------------
# images — variable-length names and the observation block
# ---------------------------------------------------------------------------


def test_variable_length_names_parse_and_the_observation_block_is_skipped(tmp_path):
    """A 5000-observation first image must not hide the second one.

    The block is 5000 x 24 = 120,000 bytes. The parser has to step over exactly that
    many, land on the next record, and never materialise a single observation.
    """
    images = [
        (7, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1, "a.jpg", 5000),
        (
            8,
            (0.5, 0.5, -0.5, 0.5),
            (1.0, 2.0, 3.0),
            2,
            "expedition/2024-08-06/subdir with spaces/IMG_0002 (copy).JPG",
            3,
        ),
        (9, (0.0, 1.0, 0.0, 0.0), (4.0, 5.0, 6.0), 1, "z.png", 0),
    ]
    path = write_images_bin(tmp_path / "images.bin", images)
    payload = path.read_bytes()
    assert len(payload) > 120_000  # the observations dominate the file

    with open(path, "rb") as raw:
        stream = CountingStream(raw)
        read = colmap._read_images_bin(stream, len(payload))
        # Offset advanced over every record, landing exactly at EOF...
        assert stream.tell() == len(payload)
        # ...while almost nothing was actually read: the 120,000-byte block was seeked.
        assert stream.bytes_read < 2_000

    assert [image.image_id for image in read] == [7, 8, 9]
    assert [image.name for image in read] == [image[4] for image in images]
    assert read[1].qvec_wxyz == (0.5, 0.5, -0.5, 0.5)
    assert read[1].tvec == (1.0, 2.0, 3.0)
    assert read[1].camera_id == 2


def test_a_name_longer_than_one_read_block_is_reassembled(tmp_path):
    long_name = "nested/" * 40 + "IMG_9999.JPG"
    assert len(long_name) > 128  # crosses the reader's block boundary
    images = [(1, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1, long_name, 2)]
    write_images_bin(tmp_path / "images.bin", images)

    read = colmap.read_images(tmp_path)

    assert [image.name for image in read] == [long_name]


def test_a_declared_count_the_file_cannot_hold_is_rejected_before_allocating(tmp_path):
    path = tmp_path / "points3D.bin"
    # 2**40 points would be a 51 TB file and a 24 TB allocation.
    path.write_bytes(
        struct.pack("<Q", 2**40) + point_record((1, (0.0, 0.0, 0.0), (1, 2, 3), 0.1, []))
    )

    with pytest.raises(colmap.ColmapFormatError) as caught:
        colmap.read_points3d(tmp_path)

    assert "declares" in str(caught.value)


def test_a_truncated_images_file_is_a_format_error(tmp_path):
    path = write_images_bin(tmp_path / "images.bin", sample_images(3))
    payload = path.read_bytes()
    path.write_bytes(payload[: len(payload) - 40])

    with pytest.raises(colmap.ColmapFormatError):
        colmap.read_images(tmp_path)


# ---------------------------------------------------------------------------
# points
# ---------------------------------------------------------------------------


def test_points_stream_into_arrays_with_variable_length_tracks(tmp_path):
    points = sample_points(300)
    write_model(tmp_path / "sparse", points=points)

    xyz, rgb = colmap.read_points3d(tmp_path / "sparse")

    assert xyz.shape == (300, 3) and xyz.dtype == np.float64
    assert rgb.shape == (300, 3) and rgb.dtype == np.uint8
    assert np.array_equal(xyz, np.asarray([point[1] for point in points], dtype=np.float64))
    assert np.array_equal(rgb, np.asarray([point[2] for point in points], dtype=np.uint8))


def test_points_cross_the_flush_block_boundary_intact(tmp_path, monkeypatch):
    """The bounded unpack buffer must not lose or reorder a point when it flushes."""
    monkeypatch.setattr(colmap, "_POINT_BLOCK", 7)
    points = sample_points(50, seed=9)
    write_model(tmp_path / "sparse", points=points)

    xyz, rgb = colmap.read_points3d(tmp_path / "sparse")

    assert np.array_equal(xyz, np.asarray([point[1] for point in points], dtype=np.float64))
    assert np.array_equal(rgb, np.asarray([point[2] for point in points], dtype=np.uint8))


# ---------------------------------------------------------------------------
# .txt parity
# ---------------------------------------------------------------------------


def test_txt_variants_parse_identically_to_bin(tmp_path):
    cameras = [PINHOLE_CAMERA, OPENCV_CAMERA, UNDISTORTED_RADIAL]
    images = sample_images(4)
    points = sample_points(64, seed=3)
    binary = write_model(tmp_path / "bin", cameras=cameras, images=images, points=points)
    text = write_model(tmp_path / "txt", cameras=cameras, images=images, points=points, text=True)

    assert colmap.read_cameras(text) == colmap.read_cameras(binary)
    assert colmap.read_images(text) == colmap.read_images(binary)
    text_xyz, text_rgb = colmap.read_points3d(text)
    bin_xyz, bin_rgb = colmap.read_points3d(binary)
    assert np.array_equal(text_xyz, bin_xyz)
    assert np.array_equal(text_rgb, bin_rgb)
    assert colmap.camera_layer_json(
        colmap.read_cameras(text), colmap.read_images(text)
    ) == colmap.camera_layer_json(colmap.read_cameras(binary), colmap.read_images(binary))


def test_an_image_with_no_observations_keeps_its_empty_second_line(tmp_path):
    """COLMAP writes an empty points2D line; skipping blanks would eat the next pose."""
    images = [
        (1, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1, "first.jpg", 0),
        (2, (0.0, 0.0, 1.0, 0.0), (1.0, 1.0, 1.0), 1, "second.jpg", 0),
    ]
    write_model(tmp_path / "txt", cameras=[PINHOLE_CAMERA], images=images, text=True)

    read = colmap.read_images(tmp_path / "txt")

    assert [image.image_id for image in read] == [1, 2]
    assert [image.name for image in read] == ["first.jpg", "second.jpg"]


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def test_detect_model_dir_finds_sparse_0_from_a_parent(tmp_path):
    root = tmp_path / "reconstruction"
    (root / "images").mkdir(parents=True)
    (root / "images" / "IMG_0001.JPG").write_bytes(b"not a model")
    write_model(
        root / "sparse" / "0",
        cameras=[PINHOLE_CAMERA],
        images=sample_images(2),
        points=sample_points(4),
    )

    assert colmap.detect_model_dir(root) == str(root / "sparse" / "0")
    assert colmap.detect_model_dir(root / "sparse" / "0") == str(root / "sparse" / "0")


def test_detect_model_dir_accepts_sparse_and_dense_sparse(tmp_path):
    flat = tmp_path / "flat"
    write_model(flat / "sparse", points=sample_points(4))
    dense = tmp_path / "dense-run"
    write_model(dense / "dense" / "sparse", points=sample_points(4))

    assert colmap.detect_model_dir(flat) == str(flat / "sparse")
    assert colmap.detect_model_dir(dense) == str(dense / "dense" / "sparse")


def test_detect_model_dir_prefers_a_binary_model_over_a_text_one(tmp_path):
    root = tmp_path / "both"
    write_model(root / "sparse" / "0", points=sample_points(4), text=True)
    write_model(root / "dense" / "sparse", points=sample_points(4))

    # sparse/0 is searched first, but it is text-only; the .bin model wins.
    assert colmap.detect_model_dir(root) == str(root / "dense" / "sparse")


def test_model_files_prefers_bin_when_both_flavours_sit_together(tmp_path):
    model_dir = tmp_path / "sparse"
    write_model(model_dir, cameras=[PINHOLE_CAMERA], images=sample_images(2), points=[])
    write_model(model_dir, cameras=[PINHOLE_CAMERA], images=sample_images(2), points=[], text=True)

    files = colmap.model_files(model_dir)

    assert files.cameras.endswith("cameras.bin")
    assert files.images.endswith("images.bin")
    assert files.is_binary is True


def test_detect_model_dir_returns_none_for_anything_that_is_not_a_model(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "notes.txt").write_text("hello")
    (plain / "scene.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\n")
    calibration_only = tmp_path / "calibration"
    write_model(calibration_only, cameras=[PINHOLE_CAMERA])

    assert colmap.detect_model_dir(plain) is None
    assert colmap.detect_model_dir(plain / "scene.ply") is None
    assert colmap.detect_model_dir(tmp_path / "absent") is None
    # cameras alone is a calibration, not a scene.
    assert colmap.detect_model_dir(calibration_only) is None


def test_detect_model_dir_accepts_a_path_to_one_of_the_model_files(tmp_path):
    model_dir = write_model(tmp_path / "sparse" / "0", points=sample_points(4))

    assert colmap.detect_model_dir(model_dir / "points3D.bin") == str(model_dir)


def test_a_ply_named_after_a_colmap_table_is_not_redirected_at_the_model(tmp_path):
    """`points3D.ply` beside a model is still a PLY; deriving the model instead would
    hand the caller a scene it did not ask for."""
    model_dir = write_model(tmp_path / "sparse", points=sample_points(4))
    decoy = model_dir / "points3D.ply"
    decoy.write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n")

    assert colmap.detect_model_dir(decoy) is None


def test_a_model_carrying_rigs_and_frames_still_parses(tmp_path):
    """Modern COLMAP writes rigs.bin/frames.bin; a legacy model has neither."""
    model_dir = write_model(
        tmp_path / "sparse" / "0",
        cameras=[PINHOLE_CAMERA],
        images=sample_images(3),
        points=sample_points(16),
    )
    (model_dir / "rigs.bin").write_bytes(struct.pack("<Q", 1) + b"\x00" * 32)
    (model_dir / "frames.bin").write_bytes(struct.pack("<Q", 3) + b"\x00" * 64)

    files = colmap.model_files(model_dir)

    assert colmap.detect_model_dir(tmp_path) == str(model_dir)
    assert files.has_rig_metadata is True
    assert len(colmap.read_images(model_dir)) == 3
    assert len(colmap.read_cameras(model_dir)) == 1
    assert colmap.read_points3d(model_dir)[0].shape == (16, 3)
    # A legacy model has neither file and is not treated as deficient.
    legacy = write_model(tmp_path / "legacy", points=sample_points(4))
    assert colmap.model_files(legacy).has_rig_metadata is False


# ---------------------------------------------------------------------------
# the cameras layer
# ---------------------------------------------------------------------------


def test_camera_layer_json_matches_the_contract_shape_and_sorts_by_name(tmp_path):
    cameras = [PINHOLE_CAMERA, OPENCV_CAMERA]
    images = [
        (1, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1, "IMG_0002.JPG", 4),
        (2, (0.0, 1.0, 0.0, 0.0), (1.0, 2.0, 3.0), 2, "IMG_0001.JPG", 0),
        (3, (0.0, 0.0, 1.0, 0.0), (4.0, 5.0, 6.0), 1, "IMG_0003.JPG", 7),
    ]
    model_dir = write_model(tmp_path / "sparse", cameras=cameras, images=images)

    payload = colmap.camera_layer_json(
        colmap.read_cameras(model_dir), colmap.read_images(model_dir)
    )

    assert list(payload) == ["cameras"]
    assert [row["name"] for row in payload["cameras"]] == [
        "IMG_0001.JPG",
        "IMG_0002.JPG",
        "IMG_0003.JPG",
    ]
    assert payload["cameras"][0] == {
        "qvec": [0.0, 1.0, 0.0, 0.0],
        "tvec": [1.0, 2.0, 3.0],
        "name": "IMG_0001.JPG",
        "camera": {
            "model": "OPENCV",
            "width": 800,
            "height": 600,
            "params": [500.0, 501.0, 400.0, 300.0, -0.21, 0.03, 0.001, -0.002],
        },
    }
    assert list(payload["cameras"][0]["camera"]) == ["model", "width", "height", "params"]
    # Serialisable as-is, and stable across runs.
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_an_image_whose_camera_is_not_calibrated_is_dropped_not_guessed(tmp_path):
    images = [
        (1, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1, "known.jpg", 0),
        (2, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 42, "orphan.jpg", 0),
    ]
    model_dir = write_model(tmp_path / "sparse", cameras=[PINHOLE_CAMERA], images=images)

    payload = colmap.camera_layer_json(
        colmap.read_cameras(model_dir), colmap.read_images(model_dir)
    )

    assert [row["name"] for row in payload["cameras"]] == ["known.jpg"]


def test_qvec_and_tvec_are_emitted_verbatim_and_never_inverted():
    """The derive does NOT convert the frame. That is the renderer's one job.

    ``qvec`` stays world-to-camera in (w, x, y, z) order and ``tvec`` stays a translation
    — not a camera centre. Each wrong-but-plausible transform is asserted *against*, so a
    future "helpful" conversion here fails loudly instead of silently mirroring a scene.
    """
    qvec = (0.5, 0.5, -0.5, 0.5)
    tvec = (1.0, -2.0, 3.0)
    images = [colmap.ColmapImage(1, qvec, tvec, 9, "IMG_0001.JPG")]
    cameras = {9: colmap.ColmapCamera(9, "PINHOLE", 1920, 1080, (1000.0, 1001.0, 960.0, 540.0))}

    row = colmap.camera_layer_json(cameras, images)["cameras"][0]

    assert row["qvec"] == [0.5, 0.5, -0.5, 0.5]
    assert row["tvec"] == [1.0, -2.0, 3.0]
    assert row["qvec"] != [0.5, -0.5, 0.5, 0.5]  # not wxyz -> xyzw
    assert row["qvec"] != [0.5, -0.5, 0.5, -0.5]  # not the camera-to-world conjugate
    assert row["tvec"] != [1.0, 2.0, -3.0]  # not the diag(1, -1, -1) flip
    centre = colmap.camera_centers(images)[0]
    assert not np.allclose(row["tvec"], centre)  # tvec is not -R^T t


def test_camera_centers_invert_the_pose_the_way_the_contract_states():
    """-R^T t, never t. Used only for bounds and the poster; never written to the wire."""
    identity = colmap.ColmapImage(1, (1.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0), 1, "a")
    # +90 degrees about z: R maps world x to camera y.
    root = float(np.sqrt(0.5))
    yawed = colmap.ColmapImage(2, (root, 0.0, 0.0, root), (1.0, 0.0, 0.0), 1, "b")

    centers = colmap.camera_centers([identity, yawed])

    assert np.allclose(centers[0], [-1.0, -2.0, -3.0])
    assert np.allclose(centers[1], [0.0, 1.0, 0.0])
    assert colmap.camera_centers([]).shape == (0, 3)


# ---------------------------------------------------------------------------
# the derive
# ---------------------------------------------------------------------------


def _derive(tmp_path, src, **options):
    dst = tmp_path / "derived"
    result = run_scene3d_derive_job(
        {"resource_id": "file-9", "src_path": str(src), "dst_dir": str(dst), **options}
    )
    document = json.loads((dst / MANIFEST_NAME).read_text())
    return result, document, dst


def _layer(document, layer_type):
    return next(layer for layer in document["layers"] if layer["type"] == layer_type)


def test_a_colmap_model_derives_points_plus_a_cameras_layer(tmp_path):
    points = sample_points(400)
    images = sample_images(5)
    root = tmp_path / "scan"
    write_model(
        root / "sparse" / "0",
        cameras=[PINHOLE_CAMERA, OPENCV_CAMERA],
        images=[*images, (99, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 2, "B_opencv.JPG", 12)],
        points=points,
    )

    result, document, dst = _derive(tmp_path, root, max_splats_per_chunk=64, tier_count=2)

    assert result["status"] == "succeeded"
    assert result["scene_kind"] == "colmap"
    assert result["total"] == 400
    assert result["camera_count"] == 6
    assert document["schema"] == "ultra.scene3d.v1"
    assert document["scene_kind"] == "colmap"
    assert document["source"]["format"] == "colmap"
    assert document["source"]["vertex_count"] == 400
    assert document["source"]["stride_bytes"] == 0  # COLMAP records are variable-width
    assert document["world"]["frame"] == "source"
    assert [layer["type"] for layer in document["layers"]] == ["points", "cameras"]

    point_layer = _layer(document, "points")
    assert point_layer["encoding"] == "upc-v1"
    assert point_layer["source_frame"] == "source"
    assert sum(chunk["count"] for chunk in point_layer["chunks"]) == 400
    assert len(point_layer["chunks"]) > 1  # actually chunked, not one blob

    camera_layer = _layer(document, "cameras")
    assert camera_layer["encoding"] == "json"
    assert camera_layer["total"] == 6
    assert camera_layer["activation_domain"] == "post"
    assert camera_layer["source_frame"] == "rdf"
    assert len(camera_layer["chunks"]) == 1
    # The provenance panel renders these by name; the "centre" line must deny, not imply.
    assert camera_layer["quantization"]["rotation"].startswith("f64-exact quaternion")
    assert "not transmitted" in camera_layer["quantization"]["center"]
    assert (dst / POSTER_NAME).exists()

    # The point positions survive the round trip through UPC1.
    recovered = []
    for entry in point_layer["chunks"]:
        blob = (dst / f"chunk_{entry['index']:05d}.bin").read_bytes()
        assert blob[0:4] == b"UPC1"
        count = struct.unpack_from("<I", blob, 8)[0]
        local = np.frombuffer(blob[64 : 64 + count * 12], "<f4").reshape(count, 3)
        recovered.append(local + np.asarray(entry["origin"], dtype=np.float32))
    recovered = np.concatenate(recovered)
    world = np.asarray([point[1] for point in points], dtype=np.float32)
    assert np.array_equal(recovered[np.lexsort(recovered.T)], world[np.lexsort(world.T)])


def test_the_cameras_chunk_comes_after_every_point_chunk_and_decodes_as_json(tmp_path):
    """Point chunk indices must not move because a model happens to have cameras."""
    points = sample_points(300)
    images = sample_images(4)
    model_dir = write_model(
        tmp_path / "sparse", cameras=[PINHOLE_CAMERA], images=images, points=points
    )

    _result, document, dst = _derive(tmp_path, model_dir, max_splats_per_chunk=64, tier_count=2)

    point_indices = [chunk["index"] for chunk in _layer(document, "points")["chunks"]]
    camera_chunk = _layer(document, "cameras")["chunks"][0]
    assert point_indices == list(range(len(point_indices)))
    assert camera_chunk["index"] == len(point_indices)
    assert _layer(document, "cameras")["tiers"] == [[camera_chunk["index"]]]

    blob = (dst / f"chunk_{camera_chunk['index']:05d}.bin").read_bytes()
    assert len(blob) == camera_chunk["bytes"]
    payload = json.loads(blob.decode("utf-8"))
    assert [row["name"] for row in payload["cameras"]] == sorted(image[4] for image in images)
    assert camera_chunk["count"] == len(payload["cameras"])
    assert camera_chunk["origin"] == [0.0, 0.0, 0.0]


def test_the_derived_camera_chunk_holds_the_source_pose_verbatim(tmp_path):
    """End-to-end: the numbers on the wire are the numbers in images.bin."""
    images = sample_images(3)
    model_dir = write_model(
        tmp_path / "sparse", cameras=[PINHOLE_CAMERA], images=images, points=sample_points(32)
    )

    _result, document, dst = _derive(tmp_path, model_dir)

    camera_chunk = _layer(document, "cameras")["chunks"][0]
    payload = json.loads((dst / f"chunk_{camera_chunk['index']:05d}.bin").read_bytes().decode())
    by_name = {row["name"]: row for row in payload["cameras"]}
    for _image_id, qvec, tvec, _camera_id, name, _observations in images:
        assert by_name[name]["qvec"] == list(qvec)
        assert by_name[name]["tvec"] == list(tvec)
        assert by_name[name]["camera"]["model"] == "PINHOLE"
        assert by_name[name]["camera"]["params"] == list(PINHOLE_CAMERA[4])


def test_a_model_with_images_but_no_points_still_derives(tmp_path):
    """A cameras-only scene: no point layer, bounds and poster from the camera centres."""
    images = sample_images(6)
    model_dir = write_model(tmp_path / "sparse", cameras=[PINHOLE_CAMERA], images=images)

    result, document, dst = _derive(tmp_path, model_dir)

    assert result["status"] == "succeeded"
    assert result["total"] == 0
    assert result["camera_count"] == 6
    assert [layer["type"] for layer in document["layers"]] == ["cameras"]
    assert document["source"]["vertex_count"] == 0
    assert _layer(document, "cameras")["chunks"][0]["index"] == 0
    assert (dst / "chunk_00000.bin").exists()
    assert (dst / POSTER_NAME).exists()
    centres = colmap.camera_centers(colmap.read_images(model_dir))
    assert np.allclose(document["world"]["bbox"][0:3], centres.min(axis=0))
    assert np.allclose(document["world"]["bbox"][3:6], centres.max(axis=0))
    joined = " ".join(document["limitations"])
    assert "no 3D points" in joined
    assert "bounds come from the camera centres" in joined


def test_limitations_state_the_frame_the_distortion_and_the_dropped_images(tmp_path):
    images = [
        (1, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1, "pinhole.JPG", 0),
        (2, (0.0, 1.0, 0.0, 0.0), (1.0, 1.0, 1.0), 2, "opencv.JPG", 0),
        (3, (0.0, 0.0, 1.0, 0.0), (2.0, 0.0, 1.0), 3, "undistorted_radial.JPG", 0),
        (4, (0.0, 0.0, 0.0, 1.0), (3.0, 1.0, 2.0), 77, "orphan.JPG", 0),
    ]
    model_dir = write_model(
        tmp_path / "sparse",
        cameras=[PINHOLE_CAMERA, OPENCV_CAMERA, UNDISTORTED_RADIAL],
        images=images,
        points=sample_points(64),
    )
    (model_dir / "rigs.bin").write_bytes(struct.pack("<Q", 0))

    _result, document, _dst = _derive(tmp_path, model_dir)

    joined = " ".join(document["limitations"])
    assert "world-to-camera" in joined
    assert "(w, x, y, z)" in joined
    assert "right-down-forward" in joined
    assert "the renderer performs the single inversion" in joined
    # Exactly one of the three drawn frusta carries effective distortion: the OPENCV one.
    # SIMPLE_RADIAL with k = 0 is a pinhole and must not be counted.
    assert "1 of 3 camera frusta" in joined
    assert "OPENCV" in joined
    assert "1 of 4 registered image(s)" in joined
    assert "rig/frame metadata" in joined
    assert "2D feature observations" in joined
    assert _layer(document, "cameras")["total"] == 4
    assert _layer(document, "cameras")["chunks"][0]["count"] == 3


def test_points_with_non_finite_coordinates_are_dropped_and_declared(tmp_path):
    """A NaN coordinate is not merely useless — it never terminates the octree.

    The subdivision splits on midpoints, and every comparison against NaN is False, so a
    NaN point stays in the same child cell forever while the cell is still too large.
    The derive drops those rows before planning, counts them, and says so.
    """
    points = sample_points(64, seed=8)
    broken = [
        (900, (float("nan"), 0.0, 0.0), (255, 0, 0), 0.1, []),
        (901, (0.0, float("inf"), 0.0), (0, 255, 0), 0.1, []),
    ]
    model_dir = write_model(
        tmp_path / "sparse",
        cameras=[PINHOLE_CAMERA],
        images=sample_images(2),
        points=[*points, *broken],
    )

    result, document, _dst = _derive(tmp_path, model_dir, max_splats_per_chunk=16)

    assert result["total"] == 64  # the 64 finite points, nothing else
    assert document["source"]["vertex_count"] == 66  # what the file declared
    assert _layer(document, "points")["total"] == 64
    joined = " ".join(document["limitations"])
    assert "2 of 66 points carried a non-finite coordinate" in joined
    assert "Every other point was kept." in joined


def test_an_empty_model_fails_deterministically(tmp_path):
    model_dir = write_model(tmp_path / "sparse", cameras=[PINHOLE_CAMERA], images=[], points=[])

    with pytest.raises(DeterministicDerivativeError) as caught:
        run_scene3d_derive_job(str(model_dir), str(tmp_path / "derived"), resource_id="f")

    assert caught.value.code == "empty_scene_source"


def test_a_corrupt_model_fails_deterministically_not_transiently(tmp_path):
    model_dir = write_model(tmp_path / "sparse", points=sample_points(8))
    (model_dir / "points3D.bin").write_bytes(struct.pack("<Q", 4) + b"\x01" * 9)

    with pytest.raises(DeterministicDerivativeError) as caught:
        run_scene3d_derive_job(str(model_dir), str(tmp_path / "derived"), resource_id="f")

    assert caught.value.code == "unsupported_scene_source"


def test_a_text_model_derives_the_same_manifest_as_its_binary_twin(tmp_path):
    cameras = [PINHOLE_CAMERA]
    images = sample_images(3)
    points = sample_points(96, seed=2)
    binary = write_model(tmp_path / "bin", cameras=cameras, images=images, points=points)
    text = write_model(tmp_path / "txt", cameras=cameras, images=images, points=points, text=True)

    _one, binary_doc, binary_dst = _derive(tmp_path / "a", binary, max_splats_per_chunk=32)
    _two, text_doc, text_dst = _derive(tmp_path / "b", text, max_splats_per_chunk=32)

    assert binary_doc["layers"] == text_doc["layers"]
    assert binary_doc["world"] == text_doc["world"]
    assert binary_doc["limitations"] == text_doc["limitations"]
    for entry in _layer(binary_doc, "points")["chunks"]:
        name = f"chunk_{entry['index']:05d}.bin"
        assert (binary_dst / name).read_bytes() == (text_dst / name).read_bytes()
