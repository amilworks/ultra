#!/usr/bin/env python3
"""Generate a synthetic — but geometrically exact — COLMAP sparse model.

This is two things at once: the fixture that proves the ``scene3d`` camera maths, and
demo data a human can actually look at in the Lens.

Because it is the fixture, it is written **forward from first principles**. Cameras are
placed in the world by their world-space centre and a look-at target; the world-to-camera
rotation is built from that basis; and only then is COLMAP's ``qvec``/``tvec`` derived
from it. Nothing here is derived from the reader. If the generator and the reader shared
a mistake — both inverting the pose the same wrong way, say — the round trip would pass
while the render was wrong, so the ground truth this writes (``ground_truth.json``) is
the *world-space* data that existed before any COLMAP convention was applied.

Conventions, spelled out because they are the easy ones to get wrong (contract §2):

* COLMAP stores a **world-to-camera** pose: ``x_cam = R @ x_world + t``.
* ``qvec`` is ``(w, x, y, z)`` — w first, unlike three.js/glTF's ``(x, y, z, w)``.
* The camera centre in world coordinates is ``C = -Rᵀ t``. It is **not** ``t`` and not
  ``-t``; those coincide with ``C`` only when ``R`` is the identity.
* The camera frame is RDF: +x right, +y **down**, +z **forward** (into the scene).
* This generator emits the COLMAP convention verbatim. It performs no RDF→RUB flip and
  no frame conversion; that belongs to the renderer alone (``sceneFrame.ts``).

Deliberately awkward properties, each of which catches a specific class of reader bug:

* a **non-square** sensor (``fx != fy``) and an **off-centre** principal point
  (``cx != width/2``, ``cy != height/2``) — a viewer that ignores ``fy`` or ``cx/cy``
  produces frusta that look plausible and are wrong;
* the look-at target is **not the world origin** (see ``--target-z``): projecting the
  origin exercises ``t`` alone and would pass even with a transposed ``R``;
* per-camera elevation wobble, so no two poses share a rotation;
* **non-contiguous** camera/image/point3D ids that do not start at 0, so a reader that
  assumes ``id == index`` fails;
* image names of **varying length** containing a subdirectory, because the name field in
  ``images.bin`` is NUL-terminated with no fixed stride;
* real 2D observations, so the reader must *skip* them by seeking rather than by
  assuming a fixed record size — and the tracks in ``points3D`` index into them, so a
  mis-skip corrupts everything that follows.

Usage::

    PYTHONPATH=src python scripts/make_synthetic_colmap.py --out /tmp/scene
    PYTHONPATH=src python scripts/make_synthetic_colmap.py --out /tmp/scene \\
        --from-ply /data/fused.ply --points 500000

Output layout::

    <out>/sparse/0/cameras.bin  images.bin  points3D.bin   [rigs.bin frames.bin stubs]
    <out>/sparse_txt/cameras.txt  images.txt  points3D.txt  (the same model, verbatim)
    <out>/ground_truth.json                                 (world-space truth)

The two model directories are written from the same in-memory model with full float
precision (``repr(float)`` round-trips exactly), so ``sparse/0`` and ``sparse_txt`` parse
to bit-identical values. Real COLMAP text exports are lossy; ours are not, on purpose,
so a bin-vs-txt test can assert equality rather than a tolerance.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

__all__ = [
    "CAMERA_MODELS",
    "Camera",
    "Image",
    "Model",
    "build_model",
    "look_at_world_to_camera",
    "main",
    "qvec_to_rotmat",
    "rotmat_to_qvec",
    "write_model",
]

# COLMAP camera models: id -> (name, parameter names in file order). The parameter count
# is the only thing standing between a reader and silently mis-striding cameras.bin, so
# it is a table, never a guess.
CAMERA_MODELS: dict[int, tuple[str, tuple[str, ...]]] = {
    0: ("SIMPLE_PINHOLE", ("f", "cx", "cy")),
    1: ("PINHOLE", ("fx", "fy", "cx", "cy")),
    2: ("SIMPLE_RADIAL", ("f", "cx", "cy", "k")),
    3: ("RADIAL", ("f", "cx", "cy", "k1", "k2")),
    4: ("OPENCV", ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")),
    5: ("OPENCV_FISHEYE", ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")),
    6: ("FULL_OPENCV", ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")),
    7: ("FOV", ("fx", "fy", "cx", "cy", "omega")),
    8: ("SIMPLE_RADIAL_FISHEYE", ("f", "cx", "cy", "k")),
    9: ("RADIAL_FISHEYE", ("f", "cx", "cy", "k1", "k2")),
    10: (
        "THIN_PRISM_FISHEYE",
        ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"),
    ),
}
MODEL_ID_BY_NAME: dict[str, int] = {name: mid for mid, (name, _p) in CAMERA_MODELS.items()}

# The default sensor. Deliberately non-square with an off-centre principal point: a
# viewer that drops fy renders every frustum with the wrong vertical field of view, and
# one that assumes cx == width/2 draws a symmetric frustum for an asymmetric camera.
DEFAULT_WIDTH = 1600
DEFAULT_HEIGHT = 1200
DEFAULT_FX = 1400.0
DEFAULT_FY = 1350.0
DEFAULT_CX = 815.0  # width/2 would be 800
DEFAULT_CY = 590.0  # height/2 would be 600
DEFAULT_RADIUS = 6.0

# World up. The scene is built z-up (a ground plane at z=0), which is what the manifest's
# `infer_up_axis` heuristic should then report as "z".
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# Ids start away from 0 and step by more than 1 so `id == index` readers break loudly.
CAMERA_ID_BASE, CAMERA_ID_STRIDE = 1, 3
IMAGE_ID_BASE, IMAGE_ID_STRIDE = 1, 2
POINT_ID_BASE, POINT_ID_STRIDE = 1000, 7

DEFAULT_NAME_TEMPLATE = "images/ring/{index:04d}_az{az:g}.jpg"

# Struct layouts, all little-endian with '<' so struct uses standard sizes and inserts
# no alignment padding. The points3D fixed part is 43 B before the track length and is
# therefore *not* 8-byte aligned — field-by-field is the only safe way to read it.
_CAMERA_HEAD = struct.Struct("<IiQQ")  # camera_id, model_id, width, height  (24 B)
_IMAGE_HEAD = struct.Struct("<I4d3dI")  # image_id, qvec wxyz, tvec, camera_id  (64 B)
_U64 = struct.Struct("<Q")
_POINT_HEAD = struct.Struct("<Q3d3BdQ")  # id, xyz, rgb, error, track_length  (51 B)
_OBS_DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("point3d_id", "<i8")])  # 24 B
_TRACK_DTYPE = np.dtype([("image_id", "<i4"), ("point2d_idx", "<i4")])  # 8 B

# Face colours. Every surface gets a distinct hue so a human can name the bug from a
# screenshot: if the orange and blue walls swap, an axis is mirrored; if the red roof is
# on the south side, the y axis is flipped.
FACE_COLORS: dict[str, tuple[int, int, int]] = {
    "ground": (122, 138, 108),  # sage
    "path": (72, 70, 68),  # dark grey, runs toward +x
    "wall_east": (214, 96, 45),  # orange   (+x)
    "wall_west": (60, 110, 180),  # blue     (-x)
    "wall_north": (238, 200, 70),  # yellow   (+y)
    "wall_south": (150, 80, 160),  # purple   (-y)
    "roof_north": (200, 60, 60),  # red      (+y slope)
    "roof_south": (90, 170, 150),  # teal     (-y slope)
    "gable": (238, 238, 238),  # near-white (the ±x triangles)
    "outlier": (255, 0, 255),  # magenta, far field
}


# --------------------------------------------------------------------------------------
# Pose algebra — built forward, then checked against its own inverse.
# --------------------------------------------------------------------------------------


def look_at_world_to_camera(
    centre: np.ndarray, target: np.ndarray, up: np.ndarray = WORLD_UP
) -> np.ndarray:
    """World-to-camera rotation ``R`` for a camera at ``centre`` looking at ``target``.

    The rows of the returned matrix are the camera's **right, down, forward** axes
    expressed in world coordinates — that is exactly what "world to camera" means for a
    rotation whose columns would be those axes.

    RDF is a right-handed triad with ``right x down = forward``, so::

        forward = normalize(target - centre)
        right   = normalize(forward x up)
        down    = forward x right

    ``forward x up`` (not ``up x forward``) is what makes ``right`` point to the viewer's
    right: facing −y with +z up, ``(0,−1,0) x (0,0,1) = (−1,0,0)`` — west, which is what
    your right hand points at when you face south.
    """
    centre = np.asarray(centre, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)

    forward = target - centre
    distance = float(np.linalg.norm(forward))
    if distance <= 0.0:
        raise ValueError("camera centre coincides with its look-at target")
    forward /= distance

    right = np.cross(forward, up)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-9:
        raise ValueError(
            "look direction is parallel to the up hint; the camera basis is undefined "
            "(lower --elevation, or raise --radius)"
        )
    right /= right_norm
    down = np.cross(forward, right)

    rotation = np.stack([right, down, forward], axis=0)
    # Cheap proof the triad is orthonormal and right-handed. A left-handed basis would
    # produce a "rotation" with determinant −1, whose quaternion does not exist.
    residual = float(np.abs(rotation @ rotation.T - np.eye(3)).max())
    if residual > 1e-12:
        raise AssertionError(f"look-at basis is not orthonormal (residual {residual:.3e})")
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > 1e-12:
        raise AssertionError(f"look-at basis is not right-handed (det {determinant:.12f})")
    return rotation


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """``(w, x, y, z)`` unit quaternion → rotation matrix.

    Identical to COLMAP's ``qvec2rotmat`` and to the frontend's ``quatToMat3`` (which
    takes xyzw). Used here only to check the generator against itself.
    """
    q = np.asarray(qvec, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        raise ValueError("quaternion has zero norm")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat_to_qvec(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix → ``(w, x, y, z)`` unit quaternion.

    Shepperd's method: pick the branch whose divisor is largest so the square root is
    never taken of a near-zero number. The naive ``w = sqrt(1+trace)/2`` branch alone
    loses all precision at 180°, which is exactly the pose of the camera on the far side
    of the ring.

    The sign is canonicalised to ``w >= 0``. ``q`` and ``−q`` are the same rotation, but
    a stable choice keeps the fixture byte-reproducible.
    """
    m = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w, x, y, z = (
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x, y, z = (
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        )
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x, y, z = (
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        )
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x, y, z = (
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        )
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= float(np.linalg.norm(q))
    if q[0] < 0.0:
        q = -q
    return q


# --------------------------------------------------------------------------------------
# Model records
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Camera:
    """One ``cameras.bin`` record."""

    camera_id: int
    model_id: int
    width: int
    height: int
    params: np.ndarray  # float64, len == len(CAMERA_MODELS[model_id][1])

    @property
    def model_name(self) -> str:
        return CAMERA_MODELS[self.model_id][0]

    @property
    def pinhole(self) -> tuple[float, float, float, float]:
        """``(fx, fy, cx, cy)`` — the undistorted part every COLMAP model carries."""
        names = CAMERA_MODELS[self.model_id][1]
        table = dict(zip(names, (float(v) for v in self.params), strict=True))
        if "fx" in table:
            return table["fx"], table["fy"], table["cx"], table["cy"]
        return table["f"], table["f"], table["cx"], table["cy"]


@dataclass
class Image:
    """One ``images.bin`` record, plus the world-space truth it was derived from."""

    image_id: int
    qvec: np.ndarray  # (4,) float64, COLMAP order (w, x, y, z), world-to-camera
    tvec: np.ndarray  # (3,) float64, world-to-camera translation
    camera_id: int
    name: str
    xys: np.ndarray  # (n, 2) float64 observation pixel coordinates
    point3d_ids: np.ndarray  # (n,) int64, −1 where the keypoint is not triangulated
    # Ground truth, never written to the COLMAP files:
    centre: np.ndarray  # (3,) world-space camera centre C
    rotation: np.ndarray  # (3, 3) world-to-camera R, rows = right/down/forward


@dataclass
class Model:
    """A complete sparse model, held in memory so bin and txt come from one source."""

    cameras: list[Camera]
    images: list[Image]
    point_ids: np.ndarray  # (n,) int64
    xyz: np.ndarray  # (n, 3) float64
    rgb: np.ndarray  # (n, 3) uint8
    errors: np.ndarray  # (n,) float64
    track_offsets: np.ndarray  # (n + 1,) int64, slice bounds into `tracks`
    tracks: np.ndarray  # (m,) _TRACK_DTYPE
    target: np.ndarray  # (3,) look-at target, ground truth
    meta: dict[str, Any]


# --------------------------------------------------------------------------------------
# Scene geometry
# --------------------------------------------------------------------------------------


def _sample_rect(
    rng: np.random.Generator, count: int, origin: np.ndarray, u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """``count`` uniform samples on the parallelogram ``origin + a*u + b*v``."""
    a = rng.random((count, 1))
    b = rng.random((count, 1))
    return origin[None, :] + a * u[None, :] + b * v[None, :]


def _sample_triangle(
    rng: np.random.Generator, count: int, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """``count`` uniform samples on triangle ``abc`` (fold the unit square in half)."""
    s = rng.random((count, 1))
    t = rng.random((count, 1))
    over = (s + t) > 1.0
    s = np.where(over, 1.0 - s, s)
    t = np.where(over, 1.0 - t, t)
    return a[None, :] + s * (b - a)[None, :] + t * (c - a)[None, :]


def _allocate(counts_by_area: list[float], total: int) -> list[int]:
    """Split ``total`` across surfaces in proportion to area, losing nothing."""
    area = float(sum(counts_by_area))
    if area <= 0.0 or total <= 0:
        return [0] * len(counts_by_area)
    raw = [total * value / area for value in counts_by_area]
    counts = [int(math.floor(value)) for value in raw]
    remainder = total - sum(counts)
    # Hand the leftovers to the surfaces with the largest fractional part, largest first.
    order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in range(remainder):
        counts[order[i % len(order)]] += 1
    return counts


def build_house_points(
    rng: np.random.Generator, count: int, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """A ground plane, a path, and a gable-roofed box — coloured by face.

    Recognisable on purpose. Every rendering bug this fixture exists to catch shows up as
    a *nameable* wrongness: swapped wall colours mean a mirrored axis, a roof below the
    walls means an inverted up axis, a squashed footprint means a dropped ``fy``.

    Dimensions scale with ``radius`` so the scene stays framed for any ring size.
    """
    ground_half = 1.35 * radius
    hx = 0.3333 * radius  # house half-extent along x (ridge runs along x)
    hy = 0.25 * radius  # house half-extent along y
    wall_z = 0.3333 * radius
    ridge_z = 0.5333 * radius
    path_half = 0.12 * radius

    slope = math.hypot(hy, ridge_z - wall_z)
    surfaces: list[tuple[str, float]] = [
        (
            "ground",
            (2 * ground_half) ** 2 - (2 * hx) * (2 * hy) - (ground_half - hx) * 2 * path_half,
        ),
        ("path", (ground_half - hx) * 2 * path_half),
        ("wall_east", 2 * hy * wall_z),
        ("wall_west", 2 * hy * wall_z),
        ("wall_north", 2 * hx * wall_z),
        ("wall_south", 2 * hx * wall_z),
        ("roof_north", 2 * hx * slope),
        ("roof_south", 2 * hx * slope),
        ("gable_east", 0.5 * 2 * hy * (ridge_z - wall_z)),
        ("gable_west", 0.5 * 2 * hy * (ridge_z - wall_z)),
    ]
    counts = _allocate([area for _name, area in surfaces], count)

    chunks: list[np.ndarray] = []
    colors: list[np.ndarray] = []

    def emit(points: np.ndarray, face: str) -> None:
        if points.size == 0:
            return
        chunks.append(points)
        colors.append(np.tile(np.array(FACE_COLORS[face], dtype=np.uint8), (len(points), 1)))

    for (name, _area), n in zip(surfaces, counts, strict=True):
        if n <= 0:
            continue
        if name == "ground":
            # Rejection-sample so no green points land inside the house or on the path.
            kept: list[np.ndarray] = []
            got = 0
            while got < n:
                batch = rng.uniform(-ground_half, ground_half, size=(max(n - got, 64) * 2, 2))
                inside_house = (np.abs(batch[:, 0]) <= hx) & (np.abs(batch[:, 1]) <= hy)
                on_path = (batch[:, 0] >= hx) & (np.abs(batch[:, 1]) <= path_half)
                batch = batch[~(inside_house | on_path)]
                if batch.size:
                    kept.append(batch)
                    got += len(batch)
            flat = np.concatenate(kept)[:n]
            emit(np.column_stack([flat, np.zeros(len(flat))]), "ground")
        elif name == "path":
            emit(
                _sample_rect(
                    rng,
                    n,
                    np.array([hx, -path_half, 0.0]),
                    np.array([ground_half - hx, 0.0, 0.0]),
                    np.array([0.0, 2 * path_half, 0.0]),
                ),
                "path",
            )
        elif name in ("wall_east", "wall_west"):
            sign = 1.0 if name == "wall_east" else -1.0
            emit(
                _sample_rect(
                    rng,
                    n,
                    np.array([sign * hx, -hy, 0.0]),
                    np.array([0.0, 2 * hy, 0.0]),
                    np.array([0.0, 0.0, wall_z]),
                ),
                name,
            )
        elif name in ("wall_north", "wall_south"):
            sign = 1.0 if name == "wall_north" else -1.0
            emit(
                _sample_rect(
                    rng,
                    n,
                    np.array([-hx, sign * hy, 0.0]),
                    np.array([2 * hx, 0.0, 0.0]),
                    np.array([0.0, 0.0, wall_z]),
                ),
                name,
            )
        elif name in ("roof_north", "roof_south"):
            sign = 1.0 if name == "roof_north" else -1.0
            emit(
                _sample_rect(
                    rng,
                    n,
                    np.array([-hx, sign * hy, wall_z]),
                    np.array([2 * hx, 0.0, 0.0]),
                    np.array([0.0, -sign * hy, ridge_z - wall_z]),
                ),
                name,
            )
        else:  # gable_east / gable_west — the triangles under the ridge
            sign = 1.0 if name == "gable_east" else -1.0
            emit(
                _sample_triangle(
                    rng,
                    n,
                    np.array([sign * hx, -hy, wall_z]),
                    np.array([sign * hx, hy, wall_z]),
                    np.array([sign * hx, 0.0, ridge_z]),
                ),
                "gable",
            )

    xyz = np.concatenate(chunks).astype(np.float64)
    rgb = np.concatenate(colors).astype(np.uint8)
    # COLMAP point order is arbitrary; shuffling stops any downstream chunker from
    # accidentally benefitting from surface-major ordering.
    order = rng.permutation(len(xyz))
    return xyz[order], rgb[order]


def load_ply_points(
    path: Path, count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """xyz/rgb from a real PLY, subsampled to at most ``count`` vertices.

    Colour source, in order: ``red/green/blue`` if present, else the degree-0 spherical
    harmonic ``0.5 + C0*f_dc`` clamped to [0,1] (contract §4.2 — display-referred, not
    linearised), else mid grey.
    """
    from ultra_deepagents.scene3d import ply as ply_module

    header = ply_module.read_header(path)
    if not header.has("x", "y", "z"):
        raise SystemExit(f"{path}: PLY element {header.element!r} has no x,y,z properties")

    has_rgb = header.has("red", "green", "blue")
    has_dc = header.has("f_dc_0", "f_dc_1", "f_dc_2")
    names: tuple[str, ...] = ("x", "y", "z")
    if has_rgb:
        names += ("red", "green", "blue")
    elif has_dc:
        names += ("f_dc_0", "f_dc_1", "f_dc_2")

    total = header.count
    take = min(count, total)
    wanted = np.sort(rng.choice(total, size=take, replace=False)) if take < total else None

    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    seen = 0
    for block in ply_module.iter_chunks(path, header, names=names):
        size = len(block)
        if wanted is None:
            local = np.arange(size)
        else:
            lo = int(np.searchsorted(wanted, seen, side="left"))
            hi = int(np.searchsorted(wanted, seen + size, side="left"))
            local = wanted[lo:hi] - seen
        seen += size
        if local.size == 0:
            continue
        picked = block[local]
        xyz_parts.append(
            np.column_stack(
                [
                    picked["x"].astype(np.float64),
                    picked["y"].astype(np.float64),
                    picked["z"].astype(np.float64),
                ]
            )
        )
        if has_rgb:
            rgb_parts.append(
                np.column_stack(
                    [
                        picked["red"].astype(np.uint8),
                        picked["green"].astype(np.uint8),
                        picked["blue"].astype(np.uint8),
                    ]
                )
            )
        elif has_dc:
            c0 = 0.28209479177387814
            dc = np.column_stack(
                [
                    picked["f_dc_0"].astype(np.float64),
                    picked["f_dc_1"].astype(np.float64),
                    picked["f_dc_2"].astype(np.float64),
                ]
            )
            rgb_parts.append(np.clip(np.round((0.5 + c0 * dc) * 255.0), 0, 255).astype(np.uint8))

    if not xyz_parts:
        raise SystemExit(f"{path}: read no vertices")
    xyz = np.concatenate(xyz_parts)
    if rgb_parts:
        rgb = np.concatenate(rgb_parts)
    else:
        rgb = np.full((len(xyz), 3), 160, dtype=np.uint8)
    provenance = {
        "from_ply": str(path),
        "ply_element": header.element,
        "ply_vertex_count": int(header.count),
        "ply_stride_bytes": int(header.stride),
        "ply_color_source": "red/green/blue"
        if has_rgb
        else ("f_dc (0.5+C0*dc)" if has_dc else "none"),
        "sampled": int(len(xyz)),
    }
    return xyz, rgb, provenance


# --------------------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------------------


def _model_params(
    model_id: int, fx: float, fy: float, cx: float, cy: float, distortion: float
) -> np.ndarray:
    """Plausible parameters for ``model_id`` at the given pinhole intrinsics.

    ``distortion`` scales every non-pinhole coefficient; at 0 (the default) every model
    is exactly a pinhole with the right *parameter count*, which is the thing a reader
    actually has to get right. Single-focal models take ``f = fx``; there is nowhere to
    put ``fy``, so the non-square sensor is only expressible by PINHOLE-family models.
    """
    table = {
        "f": fx,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "k": -0.12 * distortion,
        "k1": -0.12 * distortion,
        "k2": 0.03 * distortion,
        "k3": -0.004 * distortion,
        "k4": 0.0007 * distortion,
        "k5": 0.0,
        "k6": 0.0,
        "p1": 0.001 * distortion,
        "p2": -0.0005 * distortion,
        "sx1": 0.0002 * distortion,
        "sy1": -0.0001 * distortion,
        # The FOV model divides by omega, so it is never written as exactly zero.
        "omega": 0.9 * distortion if distortion > 0.0 else 1e-6,
    }
    names = CAMERA_MODELS[model_id][1]
    return np.array([table[name] for name in names], dtype=np.float64)


def _validate_name(name: str) -> str:
    """Image names must survive both containers.

    ``images.bin`` stores the name NUL-terminated, and ``images.txt`` stores it as the
    last whitespace-delimited field on the line. Whitespace would round-trip through one
    and not the other, so it is refused rather than silently mangled.
    """
    if not name:
        raise ValueError("image name is empty")
    if "\x00" in name:
        raise ValueError(f"image name contains NUL: {name!r}")
    if any(char.isspace() for char in name):
        raise ValueError(
            f"image name contains whitespace, which images.txt cannot encode: {name!r}"
        )
    return name


def build_model(
    *,
    cameras: int,
    points: int,
    radius: float | None,
    seed: int,
    elevation: float | None = None,
    target_z: float | None = None,
    wobble: float = 0.15,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fx: float = DEFAULT_FX,
    fy: float = DEFAULT_FY,
    cx: float = DEFAULT_CX,
    cy: float = DEFAULT_CY,
    camera_model: str = "PINHOLE",
    all_models: bool = False,
    distortion: float = 0.0,
    obs_per_image: int | None = None,
    track_target: float = 2.0,
    unregistered_fraction: float = 0.15,
    outliers: int = 0,
    name_template: str = DEFAULT_NAME_TEMPLATE,
    from_ply: Path | None = None,
) -> Model:
    """Place the cameras in the world, then derive COLMAP's pose from that placement."""
    if cameras < 1:
        raise ValueError("--cameras must be >= 1")
    if points < 1:
        raise ValueError("--points must be >= 1")
    if radius is not None and radius <= 0.0:
        raise ValueError("--radius must be > 0")

    rng = np.random.default_rng(seed)

    # ---- points -----------------------------------------------------------------
    if from_ply is not None:
        xyz, rgb, ply_meta = load_ply_points(Path(from_ply), points, rng)
        scene_kind = "from-ply"
    else:
        radius = DEFAULT_RADIUS if radius is None else radius
        xyz, rgb = build_house_points(rng, points, radius)
        ply_meta = {}
        scene_kind = "ground plane + gable-roofed box"

    if outliers > 0:
        # Far-field junk, exactly the thing `bbox_robust` exists to survive (contract §6).
        direction = rng.normal(size=(outliers, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        distance = rng.uniform(40.0, 120.0, size=(outliers, 1)) * radius
        xyz = np.concatenate([xyz, direction * distance])
        rgb = np.concatenate(
            [rgb, np.tile(np.array(FACE_COLORS["outlier"], dtype=np.uint8), (outliers, 1))]
        )

    n_points = len(xyz)
    centroid = xyz.mean(axis=0)
    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)

    # ---- ring geometry ----------------------------------------------------------
    if from_ply is not None:
        # Orbit the imported cloud rather than the origin, otherwise the cameras sit
        # somewhere unrelated to the geometry and the demo shows an empty view.
        ring_centre = centroid.copy()
        span = float(np.linalg.norm(bbox_max - bbox_min))
        # Fit the ring to the imported cloud unless the caller pinned a radius: a cloud
        # measured in millimetres would otherwise put every camera inside a single wall.
        radius = max(span * 0.75, 1e-6) if radius is None else radius
        default_elev = float(bbox_max[2] - centroid[2]) + 0.5 * radius
        default_target_z = float(centroid[2])
    else:
        ring_centre = np.zeros(3)
        default_elev = 0.6 * radius
        default_target_z = 0.24 * radius  # ~45% of the ridge height

    elev = default_elev if elevation is None else float(elevation)
    tz = default_target_z if target_z is None else float(target_z)
    target = np.array([ring_centre[0], ring_centre[1], tz], dtype=np.float64)

    # ---- cameras ----------------------------------------------------------------
    if all_models:
        model_ids = sorted(CAMERA_MODELS)
    else:
        if camera_model not in MODEL_ID_BY_NAME:
            raise ValueError(
                f"unknown camera model {camera_model!r}; known: {sorted(MODEL_ID_BY_NAME)}"
            )
        model_ids = [MODEL_ID_BY_NAME[camera_model]]
    camera_records = [
        Camera(
            camera_id=CAMERA_ID_BASE + k * CAMERA_ID_STRIDE,
            model_id=model_id,
            width=width,
            height=height,
            params=_model_params(model_id, fx, fy, cx, cy, distortion),
        )
        for k, model_id in enumerate(model_ids)
    ]

    # ---- poses, forward ---------------------------------------------------------
    image_records: list[Image] = []
    for i in range(cameras):
        theta = 2.0 * math.pi * i / cameras
        height_i = elev * (1.0 + wobble * math.sin(3.0 * theta))
        centre = np.array(
            [
                ring_centre[0] + radius * math.cos(theta),
                ring_centre[1] + radius * math.sin(theta),
                ring_centre[2] + height_i,
            ],
            dtype=np.float64,
        )
        rotation = look_at_world_to_camera(centre, target)
        qvec = rotmat_to_qvec(rotation)
        tvec = -rotation @ centre

        # Self-check: reconstruct the pose the way a reader must, and demand the world
        # centre back. This is the one invariant the whole fixture rests on, so it is
        # asserted at generation time and not merely in a test.
        reconstructed = qvec_to_rotmat(qvec)
        rot_err = float(np.abs(reconstructed - rotation).max())
        centre_err = float(np.abs(-reconstructed.T @ tvec - centre).max())
        if rot_err > 1e-12 or centre_err > 1e-9:
            raise AssertionError(
                f"pose round trip failed for camera {i}: rotation {rot_err:.3e}, "
                f"centre {centre_err:.3e}"
            )

        camera = camera_records[i % len(camera_records)]
        name = _validate_name(
            name_template.format(
                index=i,
                az=round(math.degrees(theta), 6),
                image_id=IMAGE_ID_BASE + i * IMAGE_ID_STRIDE,
            )
        )
        image_records.append(
            Image(
                image_id=IMAGE_ID_BASE + i * IMAGE_ID_STRIDE,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera.camera_id,
                name=name,
                xys=np.zeros((0, 2), dtype=np.float64),
                point3d_ids=np.zeros(0, dtype=np.int64),
                centre=centre,
                rotation=rotation,
            )
        )

    # ---- observations, and the tracks that index into them ----------------------
    point_ids = POINT_ID_BASE + np.arange(n_points, dtype=np.int64) * POINT_ID_STRIDE
    if obs_per_image is None:
        obs_per_image = max(1, int(math.ceil(n_points * track_target / max(cameras, 1))))

    camera_by_id = {camera.camera_id: camera for camera in camera_records}
    track_point_index: list[np.ndarray] = []
    track_image_id: list[np.ndarray] = []
    track_obs_index: list[np.ndarray] = []

    for image in image_records:
        camera = camera_by_id[image.camera_id]
        f_x, f_y, c_x, c_y = camera.pinhole
        cam = (image.rotation @ (xyz - image.centre).T).T  # == R @ x_world + t
        depth = cam[:, 2]
        in_front = depth > 1e-9
        u = np.full(n_points, np.nan)
        v = np.full(n_points, np.nan)
        u[in_front] = f_x * cam[in_front, 0] / depth[in_front] + c_x
        v[in_front] = f_y * cam[in_front, 1] / depth[in_front] + c_y
        visible = in_front & (u >= 0.0) & (u < camera.width) & (v >= 0.0) & (v < camera.height)
        candidates = np.flatnonzero(visible)
        take = min(obs_per_image, candidates.size)
        chosen = rng.choice(candidates, size=take, replace=False) if take else candidates[:0]

        # Real images carry far more keypoints than triangulated points; the untracked
        # ones are written with point3D_id = −1, which a reader must not treat as an id.
        extra = int(round(take * unregistered_fraction))
        obs_u = np.concatenate([u[chosen], rng.uniform(0.0, camera.width, size=extra)])
        obs_v = np.concatenate([v[chosen], rng.uniform(0.0, camera.height, size=extra)])
        obs_pid = np.concatenate([point_ids[chosen], np.full(extra, -1, dtype=np.int64)])
        obs_point_index = np.concatenate([chosen, np.full(extra, -1, dtype=np.int64)])

        order = rng.permutation(len(obs_u))
        image.xys = np.column_stack([obs_u[order], obs_v[order]]).astype(np.float64)
        image.point3d_ids = obs_pid[order].astype(np.int64)
        shuffled_index = obs_point_index[order]

        tracked = np.flatnonzero(shuffled_index >= 0)
        if tracked.size:
            track_point_index.append(shuffled_index[tracked])
            track_image_id.append(np.full(tracked.size, image.image_id, dtype=np.int64))
            track_obs_index.append(tracked.astype(np.int64))

    if track_point_index:
        flat_point = np.concatenate(track_point_index)
        flat_image = np.concatenate(track_image_id)
        flat_obs = np.concatenate(track_obs_index)
        order = np.argsort(flat_point, kind="stable")
        flat_point, flat_image, flat_obs = flat_point[order], flat_image[order], flat_obs[order]
    else:
        flat_point = np.zeros(0, dtype=np.int64)
        flat_image = np.zeros(0, dtype=np.int64)
        flat_obs = np.zeros(0, dtype=np.int64)

    counts = np.bincount(flat_point, minlength=n_points).astype(np.int64)
    track_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    tracks = np.zeros(len(flat_point), dtype=_TRACK_DTYPE)
    tracks["image_id"] = flat_image.astype(np.int32)
    tracks["point2d_idx"] = flat_obs.astype(np.int32)

    # Reprojection error: a plausible sub-pixel figure for triangulated points, 0 for the
    # ones no image saw (COLMAP writes those as −1 in some versions; 0 is unambiguous).
    errors = np.where(counts > 0, rng.uniform(0.2, 1.6, size=n_points), 0.0)

    meta: dict[str, Any] = {
        "scene": scene_kind,
        "seed": int(seed),
        "ring": {
            "centre": [float(value) for value in ring_centre],
            "radius": float(radius),
            "elevation": float(elev),
            "elevation_wobble": float(wobble),
            "cameras": int(cameras),
        },
        "look_at_target": [float(value) for value in target],
        "world_up": [float(value) for value in WORLD_UP],
        "intrinsics": {
            "width": int(width),
            "height": int(height),
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "distortion_scale": float(distortion),
            "models": [camera.model_name for camera in camera_records],
        },
        "points": {
            "count": int(n_points),
            "outliers": int(outliers),
            "bbox": [float(value) for value in (*bbox_min, *bbox_max)],
            "centroid": [float(value) for value in centroid],
        },
        "observations": {
            "per_image_target": int(obs_per_image),
            "unregistered_fraction": float(unregistered_fraction),
            "total": int(sum(len(image.xys) for image in image_records)),
            "tracked": int(len(flat_point)),
            "mean_track_length": float(counts.mean()) if n_points else 0.0,
            "points_with_empty_track": int(np.count_nonzero(counts == 0)),
        },
        **({"source_ply": ply_meta} if ply_meta else {}),
    }
    if all_models or distortion > 0.0:
        meta["caveat_observations"] = (
            "2D observations are the undistorted pinhole projection (fx, fy, cx, cy). "
            "With a distorted camera model they are therefore NOT the true projection; "
            "the pose and intrinsics are still exact."
        )

    return Model(
        cameras=camera_records,
        images=image_records,
        point_ids=point_ids,
        xyz=xyz,
        rgb=rgb,
        errors=errors,
        track_offsets=track_offsets,
        tracks=tracks,
        target=target,
        meta=meta,
    )


# --------------------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------------------


def _f(value: float) -> str:
    """Shortest text form that parses back to the identical float64.

    ``repr`` is round-trip exact in CPython, so the ``.txt`` model is not a lossy export
    of the ``.bin`` model — the two parse to bit-identical values and a test can assert
    equality rather than a tolerance.
    """
    return repr(float(value))


def write_cameras_bin(path: Path, cameras: list[Camera]) -> None:
    with open(path, "wb") as stream:
        stream.write(_U64.pack(len(cameras)))
        for camera in cameras:
            expected = len(CAMERA_MODELS[camera.model_id][1])
            if len(camera.params) != expected:
                raise AssertionError(
                    f"camera {camera.camera_id}: model {camera.model_name} needs {expected} "
                    f"params, got {len(camera.params)}"
                )
            stream.write(
                _CAMERA_HEAD.pack(camera.camera_id, camera.model_id, camera.width, camera.height)
            )
            stream.write(np.asarray(camera.params, dtype="<f8").tobytes())


def write_cameras_txt(path: Path, cameras: list[Camera]) -> None:
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(cameras)}",
    ]
    for camera in cameras:
        params = " ".join(_f(value) for value in camera.params)
        lines.append(
            f"{camera.camera_id} {camera.model_name} {camera.width} {camera.height} {params}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_images_bin(path: Path, images: list[Image]) -> None:
    with open(path, "wb") as stream:
        stream.write(_U64.pack(len(images)))
        for image in images:
            stream.write(
                _IMAGE_HEAD.pack(
                    image.image_id,
                    float(image.qvec[0]),
                    float(image.qvec[1]),
                    float(image.qvec[2]),
                    float(image.qvec[3]),
                    float(image.tvec[0]),
                    float(image.tvec[1]),
                    float(image.tvec[2]),
                    image.camera_id,
                )
            )
            stream.write(image.name.encode("utf-8"))
            stream.write(b"\x00")
            stream.write(_U64.pack(len(image.xys)))
            if len(image.xys):
                block = np.zeros(len(image.xys), dtype=_OBS_DTYPE)
                block["x"] = image.xys[:, 0]
                block["y"] = image.xys[:, 1]
                block["point3d_id"] = image.point3d_ids
                stream.write(block.tobytes())


def write_images_txt(path: Path, images: list[Image]) -> None:
    total_obs = sum(len(image.xys) for image in images)
    mean_obs = total_obs / len(images) if images else 0.0
    out: list[str] = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(images)}, mean observations per image: {mean_obs}",
    ]
    for image in images:
        pose = " ".join(_f(value) for value in (*image.qvec, *image.tvec))
        out.append(f"{image.image_id} {pose} {image.camera_id} {image.name}")
        parts: list[str] = []
        for (x, y), pid in zip(image.xys, image.point3d_ids, strict=True):
            parts.append(f"{_f(x)} {_f(y)} {int(pid)}")
        out.append(" ".join(parts))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_points3d_bin(path: Path, model: Model) -> None:
    with open(path, "wb") as stream:
        stream.write(_U64.pack(len(model.point_ids)))
        buffer = bytearray()
        for i in range(len(model.point_ids)):
            lo = int(model.track_offsets[i])
            hi = int(model.track_offsets[i + 1])
            buffer += _POINT_HEAD.pack(
                int(model.point_ids[i]),
                float(model.xyz[i, 0]),
                float(model.xyz[i, 1]),
                float(model.xyz[i, 2]),
                int(model.rgb[i, 0]),
                int(model.rgb[i, 1]),
                int(model.rgb[i, 2]),
                float(model.errors[i]),
                hi - lo,
            )
            if hi > lo:
                buffer += model.tracks[lo:hi].tobytes()
            if len(buffer) >= 1 << 20:
                stream.write(buffer)
                buffer = bytearray()
        if buffer:
            stream.write(buffer)


def write_points3d_txt(path: Path, model: Model) -> None:
    count = len(model.point_ids)
    mean_track = (len(model.tracks) / count) if count else 0.0
    out: list[str] = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
        f"# Number of points: {count}, mean track length: {mean_track}",
    ]
    for i in range(count):
        lo = int(model.track_offsets[i])
        hi = int(model.track_offsets[i + 1])
        track = model.tracks[lo:hi]
        pieces = [
            str(int(model.point_ids[i])),
            _f(model.xyz[i, 0]),
            _f(model.xyz[i, 1]),
            _f(model.xyz[i, 2]),
            str(int(model.rgb[i, 0])),
            str(int(model.rgb[i, 1])),
            str(int(model.rgb[i, 2])),
            _f(model.errors[i]),
        ]
        for entry in track:
            pieces.append(str(int(entry["image_id"])))
            pieces.append(str(int(entry["point2d_idx"])))
        out.append(" ".join(pieces))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# `rigs.bin` / `frames.bin` exist in modern COLMAP models and are absent from legacy ones.
# A reader must tolerate their presence, which is all these stubs are for.
#
# THESE ARE NOT REAL COLMAP RIG RECORDS. Their binary layout is not reproduced here — it
# was not part of the frozen contract, and inventing one would be worse than useless: a
# reader tested against a fabricated layout would be tested against nothing. They exist
# solely to prove that an unknown sidecar file in the model directory is ignored.
_RIG_STUB_NOTE = (
    b"ultra-synthetic-colmap: placeholder sidecar, NOT a real COLMAP rig/frame record.\n"
    b"Present only so a reader can be tested for tolerating unknown files in a model dir.\n"
)


def write_model(out_dir: Path, model: Model, *, rig_stubs: bool = False) -> dict[str, list[Path]]:
    """Write ``sparse/0/*.bin`` and the sibling ``sparse_txt/*.txt``, plus ground truth."""
    binary_dir = out_dir / "sparse" / "0"
    text_dir = out_dir / "sparse_txt"
    binary_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    write_cameras_bin(binary_dir / "cameras.bin", model.cameras)
    write_images_bin(binary_dir / "images.bin", model.images)
    write_points3d_bin(binary_dir / "points3D.bin", model)

    write_cameras_txt(text_dir / "cameras.txt", model.cameras)
    write_images_txt(text_dir / "images.txt", model.images)
    write_points3d_txt(text_dir / "points3D.txt", model)

    written: dict[str, list[Path]] = {
        "bin": [binary_dir / name for name in ("cameras.bin", "images.bin", "points3D.bin")],
        "txt": [text_dir / name for name in ("cameras.txt", "images.txt", "points3D.txt")],
        "stubs": [],
    }
    if rig_stubs:
        for name in ("rigs.bin", "frames.bin"):
            (binary_dir / name).write_bytes(_RIG_STUB_NOTE)
            written["stubs"].append(binary_dir / name)

    truth = {
        "generator": "scripts/make_synthetic_colmap.py",
        "schema": "ultra.scene3d.synthetic_colmap.v1",
        "convention": {
            "pose": "world-to-camera: x_cam = R @ x_world + t",
            "qvec_order": "wxyz",
            "camera_centre": "C = -R^T t  (NOT t, NOT -t)",
            "camera_axes": "RDF: +x right, +y down, +z forward",
            "frame_conversion": "none applied here; the renderer owns the RDF->RUB flip",
        },
        **model.meta,
        "cameras": [
            {
                "camera_id": camera.camera_id,
                "model_id": camera.model_id,
                "model": camera.model_name,
                "width": camera.width,
                "height": camera.height,
                "params": [float(value) for value in camera.params],
            }
            for camera in model.cameras
        ],
        "images": [
            {
                "image_id": image.image_id,
                "name": image.name,
                "camera_id": image.camera_id,
                # The world-space truth, written BEFORE the COLMAP convention was applied.
                "centre_world": [float(value) for value in image.centre],
                "right_world": [float(value) for value in image.rotation[0]],
                "down_world": [float(value) for value in image.rotation[1]],
                "forward_world": [float(value) for value in image.rotation[2]],
                "qvec_wxyz": [float(value) for value in image.qvec],
                "tvec": [float(value) for value in image.tvec],
                "num_points2D": int(len(image.xys)),
            }
            for image in model.images
        ],
    }
    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")
    written["truth"] = [out_dir / "ground_truth.json"]
    return written


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _summarise(out_dir: Path, model: Model, written: dict[str, list[Path]]) -> str:
    lines: list[str] = []
    meta = model.meta
    ring = meta["ring"]
    intr = meta["intrinsics"]
    obs = meta["observations"]
    bbox = meta["points"]["bbox"]

    lines.append(f"wrote a synthetic COLMAP model to {out_dir}")
    lines.append("")
    lines.append(f"  scene            {meta['scene']}")
    lines.append(
        f"  cameras          {ring['cameras']} on a ring, radius {ring['radius']:g}, "
        f"elevation {ring['elevation']:g} (+/-{ring['elevation_wobble'] * 100:g}% wobble)"
    )
    lines.append(
        f"  look-at target   ({', '.join(f'{v:g}' for v in meta['look_at_target'])})  "
        "-- deliberately NOT the origin"
    )
    lines.append(
        f"  sensor           {intr['width']}x{intr['height']}  fx={intr['fx']:g} fy={intr['fy']:g} "
        f"cx={intr['cx']:g} cy={intr['cy']:g}  "
        f"(square sensor would be fx==fy; centred would be cx={intr['width'] / 2:g})"
    )
    lines.append(f"  camera models    {', '.join(intr['models'])}")
    lines.append(
        f"  points3D         {meta['points']['count']:,} ({meta['points']['outliers']} far-field outliers)"
    )
    lines.append(
        f"  bbox             [{bbox[0]:.3f} {bbox[1]:.3f} {bbox[2]:.3f}] .. "
        f"[{bbox[3]:.3f} {bbox[4]:.3f} {bbox[5]:.3f}]"
    )
    lines.append(
        f"  observations     {obs['total']:,} total, {obs['tracked']:,} tracked, "
        f"mean track length {obs['mean_track_length']:.2f}, "
        f"{obs['points_with_empty_track']:,} points with an empty track"
    )
    if "caveat_observations" in meta:
        lines.append(f"  caveat           {meta['caveat_observations']}")
    if "source_ply" in meta:
        source = meta["source_ply"]
        lines.append(
            f"  from-ply         {source['from_ply']} "
            f"({source['ply_vertex_count']:,} vertices, colour from {source['ply_color_source']})"
        )
    lines.append("")
    for group in ("bin", "txt", "stubs", "truth"):
        for path in written.get(group, []):
            size = path.stat().st_size
            lines.append(f"  {size:>12,} B  {path.relative_to(out_dir)}")
    if written.get("stubs"):
        lines.append(
            "                   (rigs.bin/frames.bin are PLACEHOLDERS, not real COLMAP "
            "records -- they only prove a reader tolerates unknown sidecar files)"
        )
    lines.append("")
    lines.append("  conventions      qvec is (w,x,y,z) world-to-camera; camera centre is -R^T t.")
    lines.append("                   Camera axes are RDF. No frame conversion was applied.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic, geometrically exact COLMAP sparse model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--cameras", type=int, default=36, help="cameras on the ring")
    parser.add_argument(
        "--points", type=int, default=200_000, help="points3D count (cap when --from-ply)"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help=f"ring radius in world units (default {DEFAULT_RADIUS:g}; with --from-ply, auto-fit "
        "to the imported cloud unless given)",
    )
    parser.add_argument("--seed", type=int, default=7, help="RNG seed")
    parser.add_argument(
        "--from-ply",
        type=Path,
        default=None,
        help="use this PLY's xyz/rgb as points3D instead of synthesising a scene",
    )
    parser.add_argument(
        "--elevation", type=float, default=None, help="ring height (default 0.6*radius)"
    )
    parser.add_argument(
        "--target-z",
        type=float,
        default=None,
        help="height of the look-at target (default 0.24*radius; 0 aims at the world origin)",
    )
    parser.add_argument(
        "--camera-model",
        default="PINHOLE",
        choices=sorted(MODEL_ID_BY_NAME),
        help="COLMAP camera model for every image",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="write one camera per COLMAP model id (11 in all) and assign images round-robin",
    )
    parser.add_argument(
        "--distortion",
        type=float,
        default=0.0,
        help="scale for non-pinhole coefficients; 0 keeps every model exactly pinhole",
    )
    parser.add_argument(
        "--obs-per-image",
        type=int,
        default=None,
        help="2D observations per image (default: enough for the mean track length below)",
    )
    parser.add_argument("--track-length", type=float, default=2.0, help="target mean track length")
    parser.add_argument(
        "--unregistered-fraction",
        type=float,
        default=0.15,
        help="extra keypoints per image written with point3D_id = -1",
    )
    parser.add_argument("--outliers", type=int, default=0, help="far-field junk points to append")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fx", type=float, default=DEFAULT_FX)
    parser.add_argument("--fy", type=float, default=DEFAULT_FY)
    parser.add_argument("--cx", type=float, default=DEFAULT_CX)
    parser.add_argument("--cy", type=float, default=DEFAULT_CY)
    parser.add_argument(
        "--name-template",
        default=DEFAULT_NAME_TEMPLATE,
        help="image name; {index}, {az}, {image_id} are substituted. No whitespace.",
    )
    parser.add_argument(
        "--with-rig-stubs",
        action="store_true",
        help="also write placeholder rigs.bin/frames.bin (contents are NOT real COLMAP records)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress the summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model = build_model(
        cameras=args.cameras,
        points=args.points,
        radius=args.radius,
        seed=args.seed,
        elevation=args.elevation,
        target_z=args.target_z,
        width=args.width,
        height=args.height,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        camera_model=args.camera_model,
        all_models=args.all_models,
        distortion=args.distortion,
        obs_per_image=args.obs_per_image,
        track_target=args.track_length,
        unregistered_fraction=args.unregistered_fraction,
        outliers=args.outliers,
        name_template=args.name_template,
        from_ply=args.from_ply,
    )
    written = write_model(args.out, model, rig_stubs=args.with_rig_stubs)
    if not args.quiet:
        print(_summarise(args.out, model, written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
