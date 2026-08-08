"""The synthetic COLMAP fixture, and the camera maths it exists to prove.

``scripts/make_synthetic_colmap.py`` places cameras in the world and *derives* COLMAP's
world-to-camera ``qvec``/``tvec`` from that placement. This file walks the derivation
back and demands the world positions again.

Two rules keep the exercise honest:

1. **The parser here is independent of the writer.** ``_MinimalReader`` is written
   straight from the frozen binary layout, not by importing the generator's writers, so a
   shared struct-format mistake cannot cancel out.
2. **Every geometric assertion has a proof that it can fail.** For each "the reader gets
   the right answer" test there is a companion asserting that the *wrong* convention —
   ``t`` instead of ``−Rᵀt``, ``Rᵀ`` instead of ``R`` — produces a detectably different
   answer. A tolerance test that would also pass on wrong data proves nothing.

The final section cross-checks ``ultra_deepagents.scene3d.colmap`` — the production
reader, written concurrently — against the same fixture: two parsers built from one spec,
sharing no code, must agree exactly. Those tests skip with a stated reason if the module
is ever absent, because the generator tests above are what actually prove the fixture.
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import make_synthetic_colmap as gen  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

# --------------------------------------------------------------------------------------
# An independent reader, transcribed from the frozen COLMAP layout.
#
# Deliberately not sharing a line of code with the generator's writers. Each reader
# asserts it consumed the file exactly — a layout that is wrong by even one byte leaves a
# tail or overruns, and `assert cursor == len(blob)` catches both.
# --------------------------------------------------------------------------------------

_OBS_DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("point3d_id", "<i8")])
_TRACK_DTYPE = np.dtype([("image_id", "<i4"), ("point2d_idx", "<i4")])

# model_id -> (name, number of float64 parameters)
_MODEL_TABLE: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


class _MinimalReader:
    """Reads a COLMAP model the way the contract says it must be read."""

    @staticmethod
    def cameras_bin(path: Path) -> dict[int, dict[str, Any]]:
        blob = path.read_bytes()
        (count,) = struct.unpack_from("<Q", blob, 0)
        cursor = 8
        out: dict[int, dict[str, Any]] = {}
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack_from("<IiQQ", blob, cursor)
            cursor += 24
            name, n_params = _MODEL_TABLE[model_id]
            params = struct.unpack_from(f"<{n_params}d", blob, cursor)
            cursor += 8 * n_params
            out[camera_id] = {
                "camera_id": camera_id,
                "model_id": model_id,
                "model": name,
                "width": width,
                "height": height,
                "params": list(params),
            }
        assert cursor == len(blob), f"cameras.bin: {len(blob) - cursor} trailing bytes"
        return out

    @staticmethod
    def images_bin(path: Path, *, with_points2d: bool = False) -> list[dict[str, Any]]:
        blob = path.read_bytes()
        (count,) = struct.unpack_from("<Q", blob, 0)
        cursor = 8
        out: list[dict[str, Any]] = []
        for _ in range(count):
            fields = struct.unpack_from("<I4d3dI", blob, cursor)
            cursor += 64
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = fields
            # The name is NUL-terminated with no fixed width, so there is no stride and
            # no way to seek to image N without having parsed 0..N-1.
            end = blob.index(b"\x00", cursor)
            name = blob[cursor:end].decode("utf-8")
            cursor = end + 1
            (n_obs,) = struct.unpack_from("<Q", blob, cursor)
            cursor += 8
            record: dict[str, Any] = {
                "image_id": image_id,
                "qvec": np.array([qw, qx, qy, qz], dtype=np.float64),
                "tvec": np.array([tx, ty, tz], dtype=np.float64),
                "camera_id": camera_id,
                "name": name,
                "num_points2D": n_obs,
            }
            if with_points2d:
                block = np.frombuffer(blob, dtype=_OBS_DTYPE, count=n_obs, offset=cursor)
                record["xys"] = np.column_stack([block["x"], block["y"]]).astype(np.float64)
                record["point3d_ids"] = block["point3d_id"].astype(np.int64)
            # 24 B per observation, skipped rather than materialised (contract: 1000
            # images x 8000 keypoints is ~192 MB of data we never want).
            cursor += 24 * n_obs
            out.append(record)
        assert cursor == len(blob), f"images.bin: {len(blob) - cursor} trailing bytes"
        return out

    @staticmethod
    def points3d_bin(path: Path) -> list[dict[str, Any]]:
        blob = path.read_bytes()
        (count,) = struct.unpack_from("<Q", blob, 0)
        cursor = 8
        out: list[dict[str, Any]] = []
        for _ in range(count):
            # 51 B: u64 id, 3xf64 xyz, 3xu8 rgb, f64 error, u64 track_length. The part
            # before track_length is 43 B and is NOT 8-byte aligned, so this is read
            # field by field and never as a typed-array view.
            fields = struct.unpack_from("<Q3d3BdQ", blob, cursor)
            cursor += 51
            point_id, x, y, z, r, g, b, error, track_length = fields
            track = np.frombuffer(blob, dtype=_TRACK_DTYPE, count=track_length, offset=cursor)
            cursor += 8 * track_length
            out.append(
                {
                    "point3d_id": point_id,
                    "xyz": np.array([x, y, z], dtype=np.float64),
                    "rgb": (r, g, b),
                    "error": error,
                    "track": [
                        (int(entry["image_id"]), int(entry["point2d_idx"])) for entry in track
                    ],
                }
            )
        assert cursor == len(blob), f"points3D.bin: {len(blob) - cursor} trailing bytes"
        return out

    @staticmethod
    def _rows(path: Path) -> list[str]:
        """Lines with '#' comments dropped, blank lines preserved.

        ``images.txt`` uses two lines per image and the second is empty when an image has
        no observations, so blank lines are structural and must survive.
        """
        lines = path.read_text(encoding="utf-8").split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        return [line for line in lines if not line.lstrip().startswith("#")]

    @staticmethod
    def cameras_txt(path: Path) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        for line in _MinimalReader._rows(path):
            if not line.strip():
                continue
            parts = line.split()
            model_id = next(mid for mid, (name, _n) in _MODEL_TABLE.items() if name == parts[1])
            out[int(parts[0])] = {
                "camera_id": int(parts[0]),
                "model_id": model_id,
                "model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": [float(value) for value in parts[4:]],
            }
        return out

    @staticmethod
    def images_txt(path: Path, *, with_points2d: bool = False) -> list[dict[str, Any]]:
        rows = _MinimalReader._rows(path)
        assert len(rows) % 2 == 0, "images.txt must hold exactly two lines per image"
        out: list[dict[str, Any]] = []
        for index in range(0, len(rows), 2):
            head = rows[index].split()
            obs = rows[index + 1].split()
            assert len(obs) % 3 == 0, "images.txt POINTS2D line must be triples"
            record: dict[str, Any] = {
                "image_id": int(head[0]),
                "qvec": np.array([float(value) for value in head[1:5]], dtype=np.float64),
                "tvec": np.array([float(value) for value in head[5:8]], dtype=np.float64),
                "camera_id": int(head[8]),
                "name": head[9],
                "num_points2D": len(obs) // 3,
            }
            if with_points2d:
                record["xys"] = np.array(
                    [[float(obs[i]), float(obs[i + 1])] for i in range(0, len(obs), 3)],
                    dtype=np.float64,
                ).reshape(-1, 2)
                record["point3d_ids"] = np.array(
                    [int(obs[i + 2]) for i in range(0, len(obs), 3)], dtype=np.int64
                )
            out.append(record)
        return out

    @staticmethod
    def points3d_txt(path: Path) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for line in _MinimalReader._rows(path):
            if not line.strip():
                continue
            parts = line.split()
            track = [(int(parts[i]), int(parts[i + 1])) for i in range(8, len(parts), 2)]
            out.append(
                {
                    "point3d_id": int(parts[0]),
                    "xyz": np.array([float(value) for value in parts[1:4]], dtype=np.float64),
                    "rgb": tuple(int(value) for value in parts[4:7]),
                    "error": float(parts[7]),
                    "track": track,
                }
            )
        return out


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _generate(out: Path, *extra: str) -> dict[str, Any]:
    """Run the generator's CLI and return its ground truth."""
    assert gen.main(["--out", str(out), "--quiet", *extra]) == 0
    return json.loads((out / "ground_truth.json").read_text(encoding="utf-8"))


def _small(out: Path, *extra: str) -> dict[str, Any]:
    """A model small enough to parse record-by-record in a unit test."""
    return _generate(out, "--cameras", "9", "--points", "1500", "--seed", "11", *extra)


def _centre_from_pose(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """The one inversion everything depends on: ``C = -Rᵀ t``."""
    rotation = gen.qvec_to_rotmat(qvec)
    return -rotation.T @ tvec


def _project(rotation: np.ndarray, tvec: np.ndarray, point: np.ndarray, camera: dict[str, Any]):
    """Pinhole projection through a COLMAP pose. Returns ``(u, v, depth)``."""
    params = camera["params"]
    if camera["model"] in (
        "PINHOLE",
        "OPENCV",
        "OPENCV_FISHEYE",
        "FULL_OPENCV",
        "FOV",
        "THIN_PRISM_FISHEYE",
    ):
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    else:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    cam = rotation @ np.asarray(point, dtype=np.float64) + tvec
    depth = float(cam[2])
    return fx * cam[0] / depth + cx, fy * cam[1] / depth + cy, depth


def _colmap_module():
    """``ultra_deepagents.scene3d.colmap``, or a loud skip if it is not there yet."""
    try:
        from ultra_deepagents.scene3d import colmap  # noqa: PLC0415
    except ImportError:  # pragma: no cover - only on a branch without the reader
        pytest.skip(
            "ultra_deepagents.scene3d.colmap is not importable. The generator tests above "
            "are what prove the fixture; these cross-check the reader against it."
        )
    return colmap


# --------------------------------------------------------------------------------------
# The pose round trip — the thing the fixture exists for
# --------------------------------------------------------------------------------------


def test_camera_centres_round_trip_to_the_world_positions_they_came_from(tmp_path: Path):
    truth = _small(tmp_path)
    images = _MinimalReader.images_bin(tmp_path / "sparse" / "0" / "images.bin")
    assert len(images) == len(truth["images"])

    worst = 0.0
    for record, expected in zip(images, truth["images"], strict=True):
        assert record["image_id"] == expected["image_id"]
        assert record["name"] == expected["name"]
        centre = _centre_from_pose(record["qvec"], record["tvec"])
        worst = max(worst, float(np.abs(centre - np.array(expected["centre_world"])).max()))
    assert worst < 1e-9, f"-R^T t missed the generating world centre by {worst:.3e}"


def test_using_tvec_as_the_camera_centre_is_detectably_wrong(tmp_path: Path):
    """The companion proof: the test above would fail on the classic COLMAP bug.

    ``t`` is the world origin expressed in camera coordinates. Reading it as the camera
    centre is the single most common COLMAP mistake, and on a turntable rig it still
    yields a tidy-looking ring — just the wrong one.
    """
    truth = _small(tmp_path)
    images = _MinimalReader.images_bin(tmp_path / "sparse" / "0" / "images.bin")
    radius = truth["ring"]["radius"]

    for sign, label in ((1.0, "+t"), (-1.0, "-t")):
        errors = [
            float(np.linalg.norm(sign * record["tvec"] - np.array(expected["centre_world"])))
            for record, expected in zip(images, truth["images"], strict=True)
        ]
        assert min(errors) > 0.5 * radius, (
            f"using {label} as the camera centre landed within {min(errors):.3f} of the "
            f"true centre; this fixture cannot distinguish the conventions"
        )


def test_the_missing_transpose_hides_only_at_a_180_degree_pose(tmp_path: Path):
    """``-R t`` vs ``-Rᵀ t``: a forgotten transpose, and where it is *invisible*.

    The two agree exactly when ``(R - Rᵀ) t = 0``, i.e. when ``t`` lies along R's own
    rotation axis. On this ring that happens at azimuth 90°, where the look-at basis is
    forced symmetric and R is therefore a 180° rotation — so a 36-camera model contains
    one pose that cannot tell the two apart, and a 9-camera model contains none.

    This is a property of the pose, not a hole in the fixture. It is asserted rather than
    tuned away so that nobody later "fixes" a flaky threshold by widening it: any camera
    that fails to discriminate must be provably degenerate.
    """
    truth = _generate(tmp_path, "--cameras", "36", "--points", "200", "--seed", "11")
    images = _MinimalReader.images_bin(tmp_path / "sparse" / "0" / "images.bin")
    radius = truth["ring"]["radius"]

    blind = 0
    for record, expected in zip(images, truth["images"], strict=True):
        rotation = gen.qvec_to_rotmat(record["qvec"])
        gap = float(np.linalg.norm(-rotation @ record["tvec"] - np.array(expected["centre_world"])))
        if gap < 0.1 * radius:
            blind += 1
            # Degenerate only because R is symmetric: a 180° rotation is its own inverse.
            assert np.allclose(rotation, rotation.T, atol=1e-12), (
                "a non-symmetric pose failed to distinguish -R t from -R^T t"
            )
            assert float(np.linalg.norm(record["tvec"])) > 0.0
    assert blind == 1, f"expected exactly the azimuth-90 pose to be blind, got {blind}"
    # And the correct inversion is exact even there, which is the whole point.
    centre = _centre_from_pose(images[9]["qvec"], images[9]["tvec"])
    assert np.allclose(centre, truth["images"][9]["centre_world"], atol=1e-12)


def test_camera_centres_lie_on_the_generating_ring(tmp_path: Path):
    """An invariant the COLMAP file never states, recoverable only if the pose is right."""
    truth = _small(tmp_path)
    ring = truth["ring"]
    images = _MinimalReader.images_bin(tmp_path / "sparse" / "0" / "images.bin")

    centres = []
    for record in images:
        centre = _centre_from_pose(record["qvec"], record["tvec"])
        centres.append(centre)
        assert math.hypot(centre[0], centre[1]) == pytest.approx(ring["radius"], abs=1e-9)
        low = ring["elevation"] * (1.0 - ring["elevation_wobble"])
        high = ring["elevation"] * (1.0 + ring["elevation_wobble"])
        assert low - 1e-9 <= centre[2] <= high + 1e-9

    # The ring is deliberately not planar. A constant elevation would leave every pose
    # differing only in azimuth, and a bug confined to the z axis could hide behind that.
    heights = np.array([centre[2] for centre in centres])
    assert float(heights.std()) > 1e-6, "the camera ring is planar; z-axis bugs can hide"
    quats = np.array([record["qvec"] for record in images])
    assert len(np.unique(np.round(quats, 12), axis=0)) == len(images), "duplicate poses"


def test_bin_and_txt_models_parse_to_identical_values(tmp_path: Path):
    _small(tmp_path, "--outliers", "3")
    binary_dir = tmp_path / "sparse" / "0"
    text_dir = tmp_path / "sparse_txt"

    cameras_bin = _MinimalReader.cameras_bin(binary_dir / "cameras.bin")
    cameras_txt = _MinimalReader.cameras_txt(text_dir / "cameras.txt")
    assert cameras_bin == cameras_txt

    images_bin = _MinimalReader.images_bin(binary_dir / "images.bin", with_points2d=True)
    images_txt = _MinimalReader.images_txt(text_dir / "images.txt", with_points2d=True)
    assert len(images_bin) == len(images_txt) > 0
    for left, right in zip(images_bin, images_txt, strict=True):
        assert left["image_id"] == right["image_id"]
        assert left["camera_id"] == right["camera_id"]
        assert left["name"] == right["name"]
        assert left["num_points2D"] == right["num_points2D"]
        # Exact, not approximate: the generator writes repr(float), which round-trips.
        assert np.array_equal(left["qvec"], right["qvec"])
        assert np.array_equal(left["tvec"], right["tvec"])
        assert np.array_equal(left["xys"], right["xys"])
        assert np.array_equal(left["point3d_ids"], right["point3d_ids"])

    points_bin = _MinimalReader.points3d_bin(binary_dir / "points3D.bin")
    points_txt = _MinimalReader.points3d_txt(text_dir / "points3D.txt")
    assert len(points_bin) == len(points_txt) > 0
    for left, right in zip(points_bin, points_txt, strict=True):
        assert left["point3d_id"] == right["point3d_id"]
        assert left["rgb"] == right["rgb"]
        assert left["track"] == right["track"]
        assert np.array_equal(left["xyz"], right["xyz"])
        assert left["error"] == right["error"]


# --------------------------------------------------------------------------------------
# Intrinsics: the sensor is non-square and off-centre so those fields cannot be ignored
# --------------------------------------------------------------------------------------


def test_the_look_at_target_projects_to_the_principal_point(tmp_path: Path):
    """Every camera aims at the same world point, so it must land exactly at (cx, cy).

    The target is deliberately **not** the world origin: projecting the origin reduces to
    ``t`` alone and would pass with a transposed rotation. This exercises R, t, fx, fy,
    cx and cy at once.
    """
    truth = _small(tmp_path)
    cameras = _MinimalReader.cameras_bin(tmp_path / "sparse" / "0" / "cameras.bin")
    images = _MinimalReader.images_bin(tmp_path / "sparse" / "0" / "images.bin")
    target = np.array(truth["look_at_target"], dtype=np.float64)
    assert float(np.linalg.norm(target)) > 0.0, "a target at the origin would not test R"

    for record, expected in zip(images, truth["images"], strict=True):
        camera = cameras[record["camera_id"]]
        rotation = gen.qvec_to_rotmat(record["qvec"])
        u, v, depth = _project(rotation, record["tvec"], target, camera)
        assert u == pytest.approx(camera["params"][2], abs=1e-6)
        assert v == pytest.approx(camera["params"][3], abs=1e-6)
        # The target sits straight down the optical axis, so its depth is the distance.
        centre = np.array(expected["centre_world"])
        assert depth == pytest.approx(float(np.linalg.norm(target - centre)), rel=1e-12)


def test_a_transposed_rotation_never_projects_the_target_to_the_centre(tmp_path: Path):
    """Companion proof for the test above: R and Rᵀ are distinguishable here."""
    truth = _small(tmp_path)
    cameras = _MinimalReader.cameras_bin(tmp_path / "sparse" / "0" / "cameras.bin")
    images = _MinimalReader.images_bin(tmp_path / "sparse" / "0" / "images.bin")
    target = np.array(truth["look_at_target"], dtype=np.float64)

    near_centre = 0
    for record in images:
        camera = cameras[record["camera_id"]]
        rotation = gen.qvec_to_rotmat(record["qvec"]).T  # the wrong one, on purpose
        cam = rotation @ target + record["tvec"]
        if cam[2] <= 0.0:
            continue  # behind the camera: wrong in the most obvious way there is
        u = camera["params"][0] * cam[0] / cam[2] + camera["params"][2]
        v = camera["params"][1] * cam[1] / cam[2] + camera["params"][3]
        if abs(u - camera["params"][2]) < 1.0 and abs(v - camera["params"][3]) < 1.0:
            near_centre += 1
    assert near_centre == 0, f"{near_centre} transposed poses still hit the principal point"


def test_the_two_focal_lengths_are_separately_observable(tmp_path: Path):
    """A viewer that reads fx and applies it to both axes gets caught here.

    Offsetting the look-at target by the same world distance along the camera's own right
    and down axes moves the projection by ``fx*d/Z`` and ``fy*d/Z`` respectively, so the
    ratio of the two pixel offsets is exactly ``fx/fy`` — and the fixture's fx/fy is not 1.
    """
    truth = _small(tmp_path)
    cameras = _MinimalReader.cameras_bin(tmp_path / "sparse" / "0" / "cameras.bin")
    images = _MinimalReader.images_bin(tmp_path / "sparse" / "0" / "images.bin")
    target = np.array(truth["look_at_target"], dtype=np.float64)
    offset = 0.1 * truth["ring"]["radius"]

    camera = cameras[images[0]["camera_id"]]
    fx, fy, cx, cy = camera["params"][:4]
    assert fx != fy, "a square sensor cannot catch a dropped fy"
    assert cx != camera["width"] / 2 and cy != camera["height"] / 2, (
        "a centred principal point cannot catch a viewer that assumes symmetry"
    )

    for record in images:
        camera = cameras[record["camera_id"]]
        fx, fy, cx, cy = camera["params"][:4]
        rotation = gen.qvec_to_rotmat(record["qvec"])
        right_world, down_world = rotation[0], rotation[1]

        u_r, v_r, _ = _project(rotation, record["tvec"], target + offset * right_world, camera)
        u_d, v_d, _ = _project(rotation, record["tvec"], target + offset * down_world, camera)

        assert v_r == pytest.approx(cy, abs=1e-6), "a right-axis offset must not move v"
        assert u_d == pytest.approx(cx, abs=1e-6), "a down-axis offset must not move u"
        assert (u_r - cx) / (v_d - cy) == pytest.approx(fx / fy, rel=1e-9)
        assert (u_r - cx) > 0.0, "the camera's own +right axis must project to +u"
        assert (v_d - cy) > 0.0, "the camera's own +down axis must project to +v"


def test_world_up_projects_upward_in_the_image(tmp_path: Path):
    """+z in a z-up world must land *above* the principal point, because camera y is down."""
    truth = _small(tmp_path)
    cameras = _MinimalReader.cameras_bin(tmp_path / "sparse" / "0" / "cameras.bin")
    images = _MinimalReader.images_bin(tmp_path / "sparse" / "0" / "images.bin")
    target = np.array(truth["look_at_target"], dtype=np.float64)
    up = np.array(truth["world_up"], dtype=np.float64)

    for record in images:
        camera = cameras[record["camera_id"]]
        rotation = gen.qvec_to_rotmat(record["qvec"])
        _u, v, _depth = _project(rotation, record["tvec"], target + 0.25 * up, camera)
        assert v < camera["params"][3], "world +z rendered below the principal point"


# --------------------------------------------------------------------------------------
# Layout hazards the contract calls out by name
# --------------------------------------------------------------------------------------


def test_image_names_are_variable_length_and_survive_the_round_trip(tmp_path: Path):
    """``images.bin`` has no stride: a long name shifts every following record."""
    long_tail = "a" * 150
    _small(tmp_path, "--name-template", "shoot/" + long_tail + "/{index:04d}_az{az:g}.png")
    images = _MinimalReader.images_bin(tmp_path / "sparse" / "0" / "images.bin")
    lengths = {len(record["name"]) for record in images}

    assert max(lengths) > 150, "the fixture must stress a name longer than any fixed buffer"
    assert len(lengths) > 1, "names must vary in length, or the stride bug never shows"
    assert all(record["name"].endswith(".png") for record in images)
    # Parsing reached the end of the file, so every NUL boundary was found correctly.
    assert len(images) == 9


def test_ids_are_not_indices(tmp_path: Path):
    """Camera, image and point ids are non-contiguous and do not start at zero."""
    _small(tmp_path)
    binary_dir = tmp_path / "sparse" / "0"
    cameras = _MinimalReader.cameras_bin(binary_dir / "cameras.bin")
    images = _MinimalReader.images_bin(binary_dir / "images.bin")
    points = _MinimalReader.points3d_bin(binary_dir / "points3D.bin")

    image_ids = [record["image_id"] for record in images]
    point_ids = [record["point3d_id"] for record in points]

    # Asserted as properties, not against the generator's own constants: comparing to
    # `gen.POINT_ID_STRIDE` would move with the very change it is supposed to catch.
    assert min(cameras) > 0, "camera ids must not start at 0"
    assert min(image_ids) > 0, "image ids must not start at 0"
    assert min(point_ids) > 0, "point3D ids must not start at 0"
    assert image_ids != list(range(len(image_ids)))
    for label, ids in (("image", image_ids), ("point3D", point_ids)):
        strides = set(np.diff(ids))
        assert len(strides) == 1, f"{label} ids should step uniformly"
        assert strides.pop() > 1, f"{label} ids are contiguous; `id == index` would pass"


def test_tracks_index_real_observations(tmp_path: Path):
    """A mis-skipped points2D block corrupts every record after it; this catches that.

    Each ``points3D`` track entry names an image and an index into *that image's*
    observation array. Following the reference and finding the same point id back is only
    possible if both files were walked correctly.
    """
    _small(tmp_path)
    binary_dir = tmp_path / "sparse" / "0"
    images = {
        record["image_id"]: record
        for record in _MinimalReader.images_bin(binary_dir / "images.bin", with_points2d=True)
    }
    points = _MinimalReader.points3d_bin(binary_dir / "points3D.bin")

    followed = 0
    for point in points:
        for image_id, index in point["track"]:
            image = images[image_id]
            assert 0 <= index < image["num_points2D"]
            assert int(image["point3d_ids"][index]) == point["point3d_id"]
            followed += 1
    assert followed > 0, "the fixture produced no tracks at all"
    assert any(record["track"] for record in points)

    # Real images carry keypoints that were never triangulated. −1 is a sentinel, not an id.
    unregistered = sum(int((record["point3d_ids"] < 0).sum()) for record in images.values())
    assert unregistered > 0, "no point3D_id = -1 observations; the sentinel path is untested"


def test_a_model_with_no_observations_still_round_trips(tmp_path: Path):
    """``--obs-per-image 0``: empty keypoint blocks and empty tracks, both containers.

    ``images.txt`` still writes its second line per image, blank, so a text parser that
    drops empty lines desynchronises and reads every following pose off by one.
    """
    truth = _small(tmp_path, "--obs-per-image", "0", "--unregistered-fraction", "0")
    binary_dir = tmp_path / "sparse" / "0"

    images = _MinimalReader.images_bin(binary_dir / "images.bin", with_points2d=True)
    assert [record["num_points2D"] for record in images] == [0] * 9

    text = _MinimalReader.images_txt(tmp_path / "sparse_txt" / "images.txt", with_points2d=True)
    assert len(text) == len(images)
    for left, right in zip(images, text, strict=True):
        assert left["image_id"] == right["image_id"]
        assert left["name"] == right["name"]
        assert right["num_points2D"] == 0
        assert np.array_equal(left["qvec"], right["qvec"])
        assert np.array_equal(left["tvec"], right["tvec"])

    points = _MinimalReader.points3d_bin(binary_dir / "points3D.bin")
    assert all(record["track"] == [] for record in points)
    assert truth["observations"]["total"] == 0

    # The poses are still exact — this is the cameras-only shape the derive must survive.
    for record, expected in zip(images, truth["images"], strict=True):
        centre = _centre_from_pose(record["qvec"], record["tvec"])
        assert np.allclose(centre, expected["centre_world"], atol=1e-9)


def test_observation_pixels_are_the_true_projection(tmp_path: Path):
    """The 2D observations are not decoration — they are the exact pinhole projection."""
    _small(tmp_path)
    binary_dir = tmp_path / "sparse" / "0"
    cameras = _MinimalReader.cameras_bin(binary_dir / "cameras.bin")
    images = {
        record["image_id"]: record
        for record in _MinimalReader.images_bin(binary_dir / "images.bin", with_points2d=True)
    }
    points = _MinimalReader.points3d_bin(binary_dir / "points3D.bin")

    checked = 0
    for point in points[:200]:
        for image_id, index in point["track"]:
            image = images[image_id]
            camera = cameras[image["camera_id"]]
            rotation = gen.qvec_to_rotmat(image["qvec"])
            u, v, depth = _project(rotation, image["tvec"], point["xyz"], camera)
            assert depth > 0.0
            assert u == pytest.approx(float(image["xys"][index][0]), abs=1e-9)
            assert v == pytest.approx(float(image["xys"][index][1]), abs=1e-9)
            checked += 1
    assert checked > 0


def test_every_camera_model_writes_its_declared_parameter_count(tmp_path: Path):
    """``--all-models`` covers all 11 model ids; the param table is the striding hazard."""
    _small(tmp_path, "--all-models", "--distortion", "1.0")
    cameras = _MinimalReader.cameras_bin(tmp_path / "sparse" / "0" / "cameras.bin")
    assert len(cameras) == len(_MODEL_TABLE) == 11

    by_model = {record["model_id"]: record for record in cameras.values()}
    assert sorted(by_model) == sorted(_MODEL_TABLE)
    for model_id, record in by_model.items():
        name, n_params = _MODEL_TABLE[model_id]
        assert record["model"] == name
        assert len(record["params"]) == n_params
        assert all(math.isfinite(value) for value in record["params"])
    # The distortion coefficients are genuinely non-zero, so a reader that silently drops
    # trailing params reads a different camera rather than the same one.
    assert any(value != 0.0 for value in by_model[4]["params"][4:])

    # Images cycle through the cameras, so no image is left pointing at a missing id.
    images = _MinimalReader.images_bin(tmp_path / "sparse" / "0" / "images.bin")
    assert {record["camera_id"] for record in images} <= set(cameras)


def test_rig_and_frame_sidecars_do_not_disturb_the_legacy_files(tmp_path: Path):
    """Modern models carry rigs.bin/frames.bin; a legacy model has neither."""
    legacy = tmp_path / "legacy"
    modern = tmp_path / "modern"
    _small(legacy)
    _small(modern, "--with-rig-stubs")

    assert not (legacy / "sparse" / "0" / "rigs.bin").exists()
    assert (modern / "sparse" / "0" / "rigs.bin").exists()
    assert (modern / "sparse" / "0" / "frames.bin").exists()

    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        assert (legacy / "sparse" / "0" / name).read_bytes() == (
            modern / "sparse" / "0" / name
        ).read_bytes(), f"{name} changed when the sidecars were added"


# --------------------------------------------------------------------------------------
# Generator internals
# --------------------------------------------------------------------------------------


def test_quaternion_round_trip_holds_across_the_whole_rotation_group():
    """Including near 180°, where the naive ``sqrt(1+trace)`` branch loses all precision."""
    rng = np.random.default_rng(3)
    quats = rng.normal(size=(4000, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    # Half the sample is forced close to a 180° rotation (w ~ 0), the failure case.
    quats[::2, 0] = rng.normal(scale=1e-7, size=quats[::2].shape[0])
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)

    worst = 0.0
    for quat in quats:
        rotation = gen.qvec_to_rotmat(quat)
        recovered = gen.rotmat_to_qvec(rotation)
        worst = max(worst, float(np.abs(gen.qvec_to_rotmat(recovered) - rotation).max()))
        assert recovered[0] >= 0.0, "quaternion sign is not canonicalised"
    assert worst < 1e-12, f"quaternion round trip drifted by {worst:.3e}"


def test_look_at_basis_is_right_handed_rdf():
    centre = np.array([6.0, 0.0, 3.6])
    target = np.array([0.0, 0.0, 1.44])
    rotation = gen.look_at_world_to_camera(centre, target)
    right, down, forward = rotation

    assert np.allclose(forward, (target - centre) / np.linalg.norm(target - centre), atol=1e-12)
    assert np.allclose(np.cross(right, down), forward, atol=1e-12)  # RDF is right-handed
    assert float(np.dot(down, gen.WORLD_UP)) < 0.0, "camera +y must point down, not up"
    assert float(np.linalg.det(rotation)) == pytest.approx(1.0, abs=1e-12)


def test_look_at_refuses_a_degenerate_basis():
    with pytest.raises(ValueError, match="parallel to the up hint"):
        gen.look_at_world_to_camera(np.array([0.0, 0.0, 5.0]), np.zeros(3))
    with pytest.raises(ValueError, match="coincides"):
        gen.look_at_world_to_camera(np.zeros(3), np.zeros(3))


def test_names_that_cannot_survive_images_txt_are_refused(tmp_path: Path):
    """A name with a space round-trips through bin and not through txt."""
    with pytest.raises(ValueError, match="whitespace"):
        gen.build_model(
            cameras=2, points=10, radius=6.0, seed=1, name_template="a name {index}.jpg"
        )


def test_the_same_seed_writes_the_same_bytes(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _small(first)
    _small(second)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        assert (first / "sparse" / "0" / name).read_bytes() == (
            second / "sparse" / "0" / name
        ).read_bytes()
    assert (first / "sparse_txt" / "points3D.txt").read_text() == (
        second / "sparse_txt" / "points3D.txt"
    ).read_text()


def test_the_scene_is_a_recognisable_z_up_shape(tmp_path: Path):
    """The demo half of the fixture: a human must be able to name what they are seeing."""
    truth = _generate(tmp_path, "--cameras", "6", "--points", "20000", "--seed", "5")
    points = _MinimalReader.points3d_bin(tmp_path / "sparse" / "0" / "points3D.bin")
    xyz = np.array([record["xyz"] for record in points])
    rgb = {record["rgb"] for record in points}

    extents = xyz.max(axis=0) - xyz.min(axis=0)
    assert extents[2] < 0.5 * min(extents[0], extents[1]), "z must be the thin axis (z-up)"
    assert xyz[:, 2].min() >= 0.0, "the ground plane is at z = 0"
    # Ground, path, four walls, two roof slopes, gable — distinct colours, by design.
    assert len(rgb) >= 8
    assert gen.FACE_COLORS["wall_east"] in rgb and gen.FACE_COLORS["wall_west"] in rgb
    assert truth["points"]["count"] == len(points) == 20000


def test_from_ply_uses_the_real_positions_and_colours(tmp_path: Path):
    """``--from-ply`` swaps the synthetic shape for a real cloud, cameras and all."""
    source = tmp_path / "cloud.ply"
    rng = np.random.default_rng(2)
    count = 400
    xyz = rng.uniform(-3.0, 3.0, size=(count, 3)).astype("<f4")
    rgb = rng.integers(0, 256, size=(count, 3)).astype("u1")
    record = np.zeros(
        count,
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        ),
    )
    for axis, name in enumerate(("x", "y", "z")):
        record[name] = xyz[:, axis]
    for axis, name in enumerate(("red", "green", "blue")):
        record[name] = rgb[:, axis]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    source.write_bytes(header.encode("ascii") + record.tobytes())

    truth = _generate(
        tmp_path / "model", "--cameras", "5", "--points", "400", "--from-ply", str(source)
    )
    points = _MinimalReader.points3d_bin(tmp_path / "model" / "sparse" / "0" / "points3D.bin")
    assert len(points) == count
    assert truth["source_ply"]["ply_color_source"] == "red/green/blue"

    got_xyz = np.array(sorted(tuple(record["xyz"]) for record in points))
    want_xyz = np.array(sorted(tuple(row) for row in xyz.astype(np.float64)))
    assert np.allclose(got_xyz, want_xyz, atol=0.0, rtol=0.0)
    assert {record["rgb"] for record in points} == {tuple(int(v) for v in row) for row in rgb}

    # The ring is fitted to the imported cloud, not left at the synthetic default.
    images = _MinimalReader.images_bin(tmp_path / "model" / "sparse" / "0" / "images.bin")
    for record, expected in zip(images, truth["images"], strict=True):
        centre = _centre_from_pose(record["qvec"], record["tvec"])
        assert np.allclose(centre, expected["centre_world"], atol=1e-9)


# --------------------------------------------------------------------------------------
# Cross-check against the real reader, once it exists
# --------------------------------------------------------------------------------------


def test_reader_module_recovers_the_generator_world_centres(tmp_path: Path):
    """The whole point: ``camera_centers`` must land on the world positions we placed."""
    module = _colmap_module()
    truth = _small(tmp_path)
    images = module.read_images(tmp_path / "sparse" / "0")
    assert len(images) == len(truth["images"])

    by_id = {image.image_id: image for image in images}
    for expected in truth["images"]:
        image = by_id[expected["image_id"]]
        # The pose is carried verbatim, in COLMAP's own convention (contract §2).
        assert list(image.qvec_wxyz) == expected["qvec_wxyz"]
        assert list(image.tvec) == expected["tvec"]
        assert image.name == expected["name"]
        assert image.camera_id == expected["camera_id"]

    ordered = [by_id[expected["image_id"]] for expected in truth["images"]]
    centres = module.camera_centers(ordered)
    wanted = np.array([expected["centre_world"] for expected in truth["images"]])
    worst = float(np.abs(centres - wanted).max())
    assert worst < 1e-9, f"camera_centers missed the generating world centres by {worst:.3e}"

    # Companion proofs, so the assertion above cannot pass on a wrong convention. Both
    # are inversions a reader plausibly writes by accident.
    radius = truth["ring"]["radius"]
    tvecs = np.array([image.tvec for image in ordered])
    assert float(np.linalg.norm(tvecs - wanted, axis=1).min()) > 0.5 * radius, "t == C here"

    # `-R t` (forgetting the transpose) is a *population* claim, not a per-camera one:
    # it collapses onto `-Rᵀ t` whenever t lies along R's rotation axis. See
    # `test_the_missing_transpose_hides_only_at_a_180_degree_pose` for why that is
    # mathematics rather than a gap in the fixture.
    rotated = np.array(
        [-gen.qvec_to_rotmat(np.array(image.qvec_wxyz)) @ np.array(image.tvec) for image in ordered]
    )
    gap = np.linalg.norm(rotated - wanted, axis=1)
    assert float(np.median(gap)) > 0.5 * radius, "-R t and -R^T t are not separated here"
    assert int((gap > 0.1 * radius).sum()) >= len(ordered) - 1


def test_reader_module_agrees_between_bin_and_txt(tmp_path: Path):
    module = _colmap_module()
    _small(tmp_path)
    binary_dir = tmp_path / "sparse" / "0"
    text_dir = tmp_path / "sparse_txt"

    assert module.read_cameras(binary_dir) == module.read_cameras(text_dir)
    assert module.read_images(binary_dir) == module.read_images(text_dir)

    xyz_bin, rgb_bin = module.read_points3d(binary_dir)
    xyz_txt, rgb_txt = module.read_points3d(text_dir)
    assert np.array_equal(xyz_bin, xyz_txt)
    assert np.array_equal(rgb_bin, rgb_txt)


def test_reader_module_matches_the_independent_parser(tmp_path: Path):
    """Two parsers written from the same spec, sharing no code, must agree exactly."""
    module = _colmap_module()
    _small(tmp_path, "--outliers", "2")
    binary_dir = tmp_path / "sparse" / "0"

    mine = _MinimalReader.points3d_bin(binary_dir / "points3D.bin")
    xyz, rgb = module.read_points3d(binary_dir)
    assert np.array_equal(xyz, np.array([record["xyz"] for record in mine]))
    assert np.array_equal(rgb, np.array([record["rgb"] for record in mine], dtype=np.uint8))

    theirs = module.read_cameras(binary_dir)
    for camera_id, record in _MinimalReader.cameras_bin(binary_dir / "cameras.bin").items():
        assert theirs[camera_id].model == record["model"]
        assert theirs[camera_id].width == record["width"]
        assert theirs[camera_id].height == record["height"]
        assert list(theirs[camera_id].params) == record["params"]


def test_reader_module_reads_every_camera_model(tmp_path: Path):
    module = _colmap_module()
    _small(tmp_path, "--all-models", "--distortion", "1.0")
    cameras = module.read_cameras(tmp_path / "sparse" / "0")
    assert len(cameras) == 11

    by_name = {camera.model: camera for camera in cameras.values()}
    assert set(by_name) == {name for name, _n in _MODEL_TABLE.values()}
    for name, n_params in _MODEL_TABLE.values():
        assert len(by_name[name].params) == n_params
    assert not module.has_distortion(by_name["PINHOLE"])
    assert module.has_distortion(by_name["OPENCV"])

    # A pinhole-family model with zero coefficients must not raise a false caveat.
    _small(tmp_path / "clean", "--all-models")
    clean = module.read_cameras(tmp_path / "clean" / "sparse" / "0")
    assert not module.has_distortion({c.model: c for c in clean.values()}["OPENCV"])


def test_reader_module_tolerates_rig_and_frame_sidecars(tmp_path: Path):
    module = _colmap_module()
    _small(tmp_path, "--with-rig-stubs")
    binary_dir = tmp_path / "sparse" / "0"

    files = module.model_files(binary_dir)
    assert files.is_model and files.is_binary
    assert files.has_rig_metadata, "rigs.bin/frames.bin were not noticed"
    assert len(module.read_images(binary_dir)) == 9

    legacy = tmp_path / "legacy"
    _small(legacy)
    assert not module.model_files(legacy / "sparse" / "0").has_rig_metadata


def test_reader_module_discovers_the_generated_layout(tmp_path: Path):
    """``sparse/0`` is what COLMAP's mapper writes and what this generator emits."""
    module = _colmap_module()
    _small(tmp_path)
    found = module.detect_model_dir(tmp_path)
    assert found is not None
    assert Path(found) == tmp_path / "sparse" / "0"
    assert module.detect_model_dir(tmp_path / "sparse_txt") == str(tmp_path / "sparse_txt")
    assert module.model_bytes(found) > 0


def test_reader_module_emits_the_pose_verbatim(tmp_path: Path):
    """Contract §2: the derive converts no frames. The renderer owns the inversion."""
    module = _colmap_module()
    truth = _small(tmp_path)
    binary_dir = tmp_path / "sparse" / "0"
    layer = module.camera_layer_json(
        module.read_cameras(binary_dir), module.read_images(binary_dir)
    )

    by_name = {row["name"]: row for row in layer["cameras"]}
    assert len(by_name) == len(truth["images"])
    for expected in truth["images"]:
        row = by_name[expected["name"]]
        assert row["qvec"] == expected["qvec_wxyz"], "qvec was re-ordered or inverted"
        assert row["tvec"] == expected["tvec"], "tvec was converted; it must be verbatim"
        # Emphatically NOT the camera centre: that inversion belongs to sceneFrame.ts.
        assert row["tvec"] != expected["centre_world"]
