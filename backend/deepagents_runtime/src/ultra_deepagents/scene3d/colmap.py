"""COLMAP sparse-model reader for the ``scene3d`` derive (contract §1, §2).

A COLMAP model is three files — ``cameras``, ``images``, ``points3D`` — in either a
binary or a text flavour, and the binary flavour is the one that matters: a 20M-point
``points3D.bin`` is the only realistic way a dense reconstruction arrives.

Three properties of the binary layout drive every decision here:

1. **Nothing has a fixed stride.** An image record ends with a NUL-terminated name of
   arbitrary length followed by ``num_points2D`` observations; a point record ends with
   a track of arbitrary length. There is no way to seek to record *N* without parsing
   ``0..N-1``, so every reader is a forward stream.
2. **The 2D observations are useless to us and enormous.** 1000 images x 8000 keypoints
   is ~192 MB of ``(x, y, point3D_id)`` triples that no part of the viewer consumes.
   :func:`read_images` steps over them with ``seek``; it never reads them.
3. **The point record's fixed part is 43 bytes and is not 8-byte aligned** (``u64`` id,
   3x ``f64`` xyz, 3x ``u8`` rgb, ``f64`` error). A typed-array view over the file is
   therefore wrong at every record; fields are unpacked individually and accumulated in
   bounded blocks, so a 20M-point model costs its two output arrays and nothing else.

**The frame convention is the thing that must not be got wrong** (contract §2).
``qvec``/``tvec`` are **world-to-camera** and ``qvec`` is ``(w, x, y, z)``. The camera
centre is ``-R^T t``, never ``t``. This module reads those numbers and hands them on
**verbatim**: :func:`camera_layer_json` performs no inversion, no re-ordering, and no
handedness flip, and the layer it feeds declares ``source_frame: "rdf"``. The renderer
owns the inversion and the ``diag(1, -1, -1)`` flip, in exactly one place
(``frontend/src/components/viewer/scene3d/sceneFrame.ts``). Doing it twice is exactly as
broken as never doing it.

:func:`camera_centers` is the one function that *does* invert a pose, and it exists only
so the derive can compute a bounding box and a poster for a scene whose geometry is
cameras. Its output is never written to the wire.
"""

from __future__ import annotations

import os
import re
import struct
from array import array
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, BinaryIO

import numpy as np

__all__ = [
    "CAMERA_MODELS",
    "CAMERA_MODELS_BY_NAME",
    "ColmapCamera",
    "ColmapCameraModel",
    "ColmapFormatError",
    "ColmapImage",
    "ColmapModelFiles",
    "camera_centers",
    "camera_layer_json",
    "detect_model_dir",
    "detect_model_dirs",
    "has_distortion",
    "model_bytes",
    "model_files",
    "read_cameras",
    "read_images",
    "read_points3d",
]


class ColmapFormatError(ValueError):
    """The model is not a COLMAP model we can read record-by-record.

    A ``ValueError`` subclass so the derive job classifies it as a *deterministic*
    failure: an immutable source that fails to parse once fails identically forever.
    """


@dataclass(frozen=True)
class ColmapCameraModel:
    """One entry of COLMAP's camera-model table (``src/colmap/sensor/models.h``).

    ``shared_focal`` means the model stores a single ``f`` for both axes, so the params
    run ``f, cx, cy, ...`` rather than ``fx, fy, cx, cy, ...``. ``param_count`` is
    checked on every read, because a params array of the wrong length means the record
    was misparsed and every byte after it is garbage.
    """

    model_id: int
    name: str
    shared_focal: bool
    distortion: int

    @property
    def param_count(self) -> int:
        return (3 if self.shared_focal else 4) + self.distortion


def _model(model_id: int, name: str, shared_focal: bool, distortion: int) -> ColmapCameraModel:
    return ColmapCameraModel(
        model_id=model_id, name=name, shared_focal=shared_focal, distortion=distortion
    )


#: model id -> model. Ids are wire values and never change; the names match the strings
#: the renderer's ``sceneIntrinsics.ts`` dispatches on.
CAMERA_MODELS: Mapping[int, ColmapCameraModel] = {
    0: _model(0, "SIMPLE_PINHOLE", True, 0),  # f, cx, cy
    1: _model(1, "PINHOLE", False, 0),  # fx, fy, cx, cy
    2: _model(2, "SIMPLE_RADIAL", True, 1),  # f, cx, cy, k
    3: _model(3, "RADIAL", True, 2),  # f, cx, cy, k1, k2
    4: _model(4, "OPENCV", False, 4),  # fx, fy, cx, cy, k1, k2, p1, p2
    5: _model(5, "OPENCV_FISHEYE", False, 4),  # fx, fy, cx, cy, k1, k2, k3, k4
    6: _model(6, "FULL_OPENCV", False, 8),  # + p1, p2, k3, k4, k5, k6
    7: _model(7, "FOV", False, 1),  # fx, fy, cx, cy, omega
    8: _model(8, "SIMPLE_RADIAL_FISHEYE", True, 1),  # f, cx, cy, k
    9: _model(9, "RADIAL_FISHEYE", True, 2),  # f, cx, cy, k1, k2
    10: _model(10, "THIN_PRISM_FISHEYE", False, 8),  # + p1, p2, k3, k4, sx1, sy1
}

CAMERA_MODELS_BY_NAME: Mapping[str, ColmapCameraModel] = {
    model.name: model for model in CAMERA_MODELS.values()
}


@dataclass(frozen=True)
class ColmapCamera:
    """One intrinsic calibration. ``params`` is positional and model-dependent."""

    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class ColmapImage:
    """One registered image: its **world-to-camera** pose and which camera took it.

    ``qvec_wxyz`` is COLMAP's ``(w, x, y, z)`` order, stored exactly as the file holds
    it. Nothing in this module re-orders or inverts it.
    """

    image_id: int
    qvec_wxyz: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    name: str


@dataclass(frozen=True)
class ColmapModelFiles:
    """Which of the three model files exist in a directory, ``.bin`` preferred."""

    directory: str
    cameras: str | None
    images: str | None
    points3d: str | None
    rigs: str | None = None
    frames: str | None = None

    @property
    def is_model(self) -> bool:
        """A model is the paired camera calibration and registered-image tables.

        ``points3D`` is optional. A points table alone may be useful data, but it does not
        prove a complete COLMAP reconstruction and the control-plane gate deliberately
        applies the same rule.
        """
        return bool(self.cameras and self.images)

    @property
    def is_binary(self) -> bool:
        present = [self.cameras, self.images, self.points3d]
        return any(path and path.endswith(".bin") for path in present)

    @property
    def has_rig_metadata(self) -> bool:
        """Modern COLMAP writes ``rigs.bin``/``frames.bin`` beside the legacy triple.

        Their presence is tolerated and reported; nothing here reads them.
        """
        return bool(self.rigs or self.frames)

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(
            path for path in (self.cameras, self.images, self.points3d) if path is not None
        )


_STEMS = ("cameras", "images", "points3d")
_SUFFIXES = (".bin", ".txt")
# Candidate subdirectories of a parent that holds the model rather than being it. COLMAP
# writes `sparse/0` from `mapper`, `sparse` from some GUI runs, and `dense/sparse` after
# `image_undistorter`.
_SUBDIR_PATTERNS = ("sparse", os.path.join("dense", "sparse"))
_NUMERIC = re.compile(r"^\d+$")

# Binary record geometry. Every constant here is a wire fact, not a guess.
_U64 = struct.Struct("<Q")
_CAMERA_HEAD = struct.Struct("<IiQQ")  # camera_id, model_id, width, height
_IMAGE_HEAD = struct.Struct("<I7dI")  # image_id, qvec[4], tvec[3], camera_id -> 64 B
_POINT_HEAD = struct.Struct("<Q3d3BdQ")  # id, xyz, rgb, error, track_length -> 51 B
_OBSERVATION_BYTES = 24  # float64 x, float64 y, int64 point3D_id
_TRACK_ENTRY_BYTES = 8  # int32 image_id, int32 point2D_idx
# Smallest possible record, used only to reject a declared count that the file cannot
# possibly hold before allocating anything for it.
_CAMERA_MIN_BYTES = _CAMERA_HEAD.size + 3 * 8
_IMAGE_MIN_BYTES = _IMAGE_HEAD.size + 1 + _U64.size
_POINT_MIN_BYTES = _POINT_HEAD.size
# Names are file paths; anything past this is a misparse, not a filename.
_MAX_NAME_BYTES = 4096
_NAME_BLOCK = 128
# Points flushed from the unpack buffer into the output arrays at a time. Bounded, so a
# 20M-point model never materialises a Python list of 20M tuples.
_POINT_BLOCK = 65_536
_READ_BUFFER = 1 << 20


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def _listing(directory: str) -> dict[str, str]:
    """Lowercased filename -> real filename, for one directory.

    A directory listing rather than :func:`os.path.exists` probes: COLMAP writes
    ``points3D.bin`` with a capital D, some tooling writes ``points3d.bin``, and a
    case-insensitive probe that works on macOS silently fails on the Linux worker.
    """
    entries: dict[str, str] = {}
    try:
        with os.scandir(directory) as scan:
            for entry in scan:
                if entry.is_file():
                    entries[entry.name.lower()] = entry.name
    except (NotADirectoryError, FileNotFoundError):
        return {}
    return entries


def model_files(directory: str | os.PathLike[str]) -> ColmapModelFiles:
    """Which model files ``directory`` holds. ``.bin`` wins when both flavours exist."""
    root = os.fspath(directory)
    listing = _listing(root)

    def pick(stem: str) -> str | None:
        for suffix in _SUFFIXES:
            found = listing.get(stem + suffix)
            if found is not None:
                return os.path.join(root, found)
        return None

    return ColmapModelFiles(
        directory=root,
        cameras=pick("cameras"),
        images=pick("images"),
        points3d=pick("points3d"),
        rigs=pick("rigs"),
        frames=pick("frames"),
    )


def _candidate_dirs(root: str) -> list[str]:
    """``root`` first, then the conventional model subdirectories, most specific first."""
    candidates = [root]
    for pattern in _SUBDIR_PATTERNS:
        base = os.path.join(root, pattern)
        if not os.path.isdir(base):
            continue
        # `sparse/0`, `sparse/1`, ... are separate reconstructions. Preserve every
        # candidate so the caller can refuse ambiguity rather than silently taking 0.
        try:
            numbered = sorted((name for name in os.listdir(base) if _NUMERIC.match(name)), key=int)
        except OSError:
            numbered = []
        candidates.extend(os.path.join(base, name) for name in numbered)
        candidates.append(base)
    return candidates


def detect_model_dirs(path: str | os.PathLike[str]) -> tuple[str, ...]:
    """Every directory that independently holds a COLMAP model.

    Accepts the model directory itself, a parent containing ``sparse/0``, ``sparse/`` or
    ``dense/sparse/``, or a path to one of the model files. The result is stable and
    preserves the candidate order, but it never assigns scientific priority to one
    reconstruction over another.

    Returns an empty tuple for anything that is not a COLMAP model, so the caller can fall
    through to the PLY path without catching an exception.
    """
    root = os.fspath(path)
    if os.path.isfile(root):
        stem, suffix = os.path.splitext(os.path.basename(root))
        # Both halves must match: a file called `points3D.ply` is a PLY that happens to
        # be named after a COLMAP table, and redirecting it at a model that shares its
        # directory would derive something the caller did not ask for.
        if stem.lower() not in _STEMS or suffix.lower() not in _SUFFIXES:
            return ()
        root = os.path.dirname(root) or "."
    if not os.path.isdir(root):
        return ()
    return tuple(
        candidate for candidate in _candidate_dirs(root) if model_files(candidate).is_model
    )


def detect_model_dir(path: str | os.PathLike[str]) -> str | None:
    """Return a model directory only when discovery is unambiguous.

    Multiple submodels commonly mean distinct reconstructions. Returning ``None``
    forces the derive boundary to handle that state explicitly instead of choosing
    the shallowest or lowest-numbered model on the scientist's behalf.
    """
    models = detect_model_dirs(path)
    return models[0] if len(models) == 1 else None


def model_bytes(directory: str | os.PathLike[str]) -> int:
    """Total size of the model files we actually read."""
    total = 0
    for path in model_files(directory).paths:
        try:
            total += os.path.getsize(path)
        except OSError:  # pragma: no cover - raced deletion; the reader reports it
            continue
    return total


def _require(directory: str | os.PathLike[str], stem: str) -> str:
    files = model_files(directory)
    path = {"cameras": files.cameras, "images": files.images, "points3d": files.points3d}[stem]
    if path is None:
        raise ColmapFormatError(f"COLMAP model {os.fspath(directory)!r} has no {stem} file")
    return path


# ---------------------------------------------------------------------------
# binary primitives
# ---------------------------------------------------------------------------


def _read_exact(stream: BinaryIO, size: int, what: str) -> bytes:
    blob = stream.read(size)
    if len(blob) != size:
        raise ColmapFormatError(
            f"truncated COLMAP {what}: wanted {size} bytes at offset "
            f"{stream.tell() - len(blob)}, got {len(blob)}"
        )
    return blob


def _read_count(stream: BinaryIO, limit: int, record_bytes: int, what: str) -> int:
    """The leading ``u64`` record count, rejected if the file cannot possibly hold it.

    The guard is not decoration: the count is attacker- (or corruption-) controlled, and
    without it a bogus ``2**63`` would drive a loop for the rest of the process's life or
    an allocation for the rest of the machine's memory.
    """
    (count,) = _U64.unpack(_read_exact(stream, _U64.size, f"{what} count"))
    remaining = max(0, limit - stream.tell())
    if count * record_bytes > remaining:
        raise ColmapFormatError(
            f"COLMAP {what} declares {count} records, which needs at least "
            f"{count * record_bytes} bytes but only {remaining} remain"
        )
    return int(count)


def _read_name(stream: BinaryIO) -> str:
    """The NUL-terminated image name. Block reads, then seek back over the overshoot.

    Byte-at-a-time would be correct too, but this runs once per image and names are
    short; reading a block and rewinding keeps it to one call for essentially every
    real name. Decoding is ``replace``-tolerant because the name lands in JSON, and one
    mojibake character is a better outcome than a whole model failing to derive.
    """
    parts = bytearray()
    while True:
        block = stream.read(_NAME_BLOCK)
        if not block:
            raise ColmapFormatError("truncated COLMAP images: image name is not NUL-terminated")
        index = block.find(b"\0")
        if index >= 0:
            parts += block[:index]
            stream.seek(index + 1 - len(block), os.SEEK_CUR)
            break
        parts += block
        if len(parts) > _MAX_NAME_BYTES:
            raise ColmapFormatError(
                f"COLMAP image name exceeds {_MAX_NAME_BYTES} bytes; the record is misparsed"
            )
    return parts.decode("utf-8", errors="replace")


def _skip(stream: BinaryIO, size: int, limit: int, what: str) -> None:
    """Step over a block without reading it.

    This is what keeps ``images.bin`` cheap: the 2D observations are 24 bytes each and
    nothing downstream consumes one of them. Seeking past EOF succeeds silently on a
    file object, so the bound is checked here rather than discovered as a confusing
    truncation error three records later.
    """
    if size < 0 or stream.tell() + size > limit:
        raise ColmapFormatError(f"truncated COLMAP {what}: block of {size} bytes runs past EOF")
    if size:
        stream.seek(size, os.SEEK_CUR)


# ---------------------------------------------------------------------------
# cameras
# ---------------------------------------------------------------------------


def read_cameras(model_dir: str | os.PathLike[str]) -> dict[int, ColmapCamera]:
    """``camera_id -> ColmapCamera`` for the model's ``cameras.bin``/``cameras.txt``."""
    path = _require(model_dir, "cameras")
    if path.endswith(".txt"):
        return _read_cameras_txt(path)
    limit = os.path.getsize(path)
    with open(path, "rb", buffering=_READ_BUFFER) as stream:
        return _read_cameras_bin(stream, limit)


def _camera_model(model_id: int) -> ColmapCameraModel:
    model = CAMERA_MODELS.get(int(model_id))
    if model is None:
        raise ColmapFormatError(f"unknown COLMAP camera model id {model_id}")
    return model


def _read_cameras_bin(stream: BinaryIO, limit: int) -> dict[int, ColmapCamera]:
    count = _read_count(stream, limit, _CAMERA_MIN_BYTES, "cameras")
    cameras: dict[int, ColmapCamera] = {}
    for _ in range(count):
        camera_id, model_id, width, height = _CAMERA_HEAD.unpack(
            _read_exact(stream, _CAMERA_HEAD.size, "cameras")
        )
        model = _camera_model(model_id)
        size = model.param_count
        params = struct.unpack(
            f"<{size}d", _read_exact(stream, size * 8, f"cameras ({model.name} params)")
        )
        cameras[int(camera_id)] = ColmapCamera(
            camera_id=int(camera_id),
            model=model.name,
            width=int(width),
            height=int(height),
            params=tuple(float(value) for value in params),
        )
    return cameras


def _text_lines(path: str) -> Iterable[str]:
    """Non-empty, non-comment lines. COLMAP's text exports lead with a ``#`` banner."""
    with open(path, encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            line = raw.strip()
            if line and not line.startswith("#"):
                yield line


def _read_cameras_txt(path: str) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    for line in _text_lines(path):
        parts = line.split()
        if len(parts) < 4:
            raise ColmapFormatError(f"malformed COLMAP cameras.txt line: {line!r}")
        name = parts[1].upper()
        model = CAMERA_MODELS_BY_NAME.get(name)
        if model is None:
            raise ColmapFormatError(f"unknown COLMAP camera model {parts[1]!r}")
        params = parts[4:]
        if len(params) != model.param_count:
            raise ColmapFormatError(
                f"COLMAP camera model {model.name} expects {model.param_count} params, "
                f"got {len(params)}"
            )
        try:
            camera = ColmapCamera(
                camera_id=int(parts[0]),
                model=model.name,
                width=int(parts[2]),
                height=int(parts[3]),
                params=tuple(float(value) for value in params),
            )
        except ValueError as exc:
            raise ColmapFormatError(f"malformed COLMAP cameras.txt line: {line!r}") from exc
        cameras[camera.camera_id] = camera
    return cameras


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------


def read_images(model_dir: str | os.PathLike[str]) -> list[ColmapImage]:
    """Every registered image's pose, in file order.

    The per-image 2D observation block is **skipped with a seek**, never read: it is 24
    bytes per keypoint (~192 MB for 1000 images at 8000 keypoints each) and no part of
    the viewer consumes it.
    """
    path = _require(model_dir, "images")
    if path.endswith(".txt"):
        return _read_images_txt(path)
    limit = os.path.getsize(path)
    with open(path, "rb", buffering=_READ_BUFFER) as stream:
        return _read_images_bin(stream, limit)


def _read_images_bin(stream: BinaryIO, limit: int) -> list[ColmapImage]:
    count = _read_count(stream, limit, _IMAGE_MIN_BYTES, "images")
    images: list[ColmapImage] = []
    for _ in range(count):
        fields = _IMAGE_HEAD.unpack(_read_exact(stream, _IMAGE_HEAD.size, "images"))
        name = _read_name(stream)
        (num_points2d,) = _U64.unpack(_read_exact(stream, _U64.size, "images (num_points2D)"))
        _skip(stream, int(num_points2d) * _OBSERVATION_BYTES, limit, "images (points2D)")
        images.append(
            ColmapImage(
                image_id=int(fields[0]),
                # Verbatim, in COLMAP's (w, x, y, z) world-to-camera order.
                qvec_wxyz=(
                    float(fields[1]),
                    float(fields[2]),
                    float(fields[3]),
                    float(fields[4]),
                ),
                tvec=(float(fields[5]), float(fields[6]), float(fields[7])),
                camera_id=int(fields[8]),
                name=name,
            )
        )
    return images


def _read_images_txt(path: str) -> list[ColmapImage]:
    """Two lines per image: the pose, then a points2D line that is discarded.

    The second line is consumed unconditionally — including when it is empty, which is
    what COLMAP writes for an image with no observations. Skipping blanks here instead
    would swallow the *next* image's pose line.
    """
    images: list[ColmapImage] = []
    with open(path, encoding="utf-8", errors="replace") as stream:
        expect_observations = False
        for raw in stream:
            if expect_observations:
                expect_observations = False
                continue
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME — the name is the rest of the
            # line, because COLMAP image names routinely contain spaces.
            parts = line.split(maxsplit=9)
            if len(parts) < 10:
                raise ColmapFormatError(f"malformed COLMAP images.txt line: {line!r}")
            try:
                images.append(
                    ColmapImage(
                        image_id=int(parts[0]),
                        qvec_wxyz=(
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                            float(parts[4]),
                        ),
                        tvec=(float(parts[5]), float(parts[6]), float(parts[7])),
                        camera_id=int(parts[8]),
                        name=parts[9].strip(),
                    )
                )
            except ValueError as exc:
                raise ColmapFormatError(f"malformed COLMAP images.txt line: {line!r}") from exc
            expect_observations = True
    return images


# ---------------------------------------------------------------------------
# points
# ---------------------------------------------------------------------------


def read_points3d(model_dir: str | os.PathLike[str]) -> tuple[np.ndarray, np.ndarray]:
    """``(xyz float64 (n, 3), rgb uint8 (n, 3))`` for the model's 3D points.

    Streamed. The binary reader preallocates both outputs from the declared count and
    fills them from a bounded unpack buffer, so a 20M-point model costs its two arrays
    (480 MB + 60 MB) and a 64k-point staging block — not a 20M-entry Python list.
    Per-point ``error`` and the observation tracks are read past and dropped.
    """
    path = _require(model_dir, "points3d")
    if path.endswith(".txt"):
        return _read_points3d_txt(path)
    limit = os.path.getsize(path)
    with open(path, "rb", buffering=_READ_BUFFER) as stream:
        return _read_points3d_bin(stream, limit)


def _read_points3d_bin(stream: BinaryIO, limit: int) -> tuple[np.ndarray, np.ndarray]:
    count = _read_count(stream, limit, _POINT_MIN_BYTES, "points3D")
    xyz = np.empty((count, 3), dtype=np.float64)
    rgb = np.empty((count, 3), dtype=np.uint8)
    block_xyz = array("d")
    block_rgb = array("B")
    filled = 0

    def flush() -> None:
        nonlocal filled
        rows = len(block_rgb) // 3
        if rows == 0:
            return
        xyz[filled : filled + rows] = np.frombuffer(block_xyz, dtype=np.float64).reshape(rows, 3)
        rgb[filled : filled + rows] = np.frombuffer(block_rgb, dtype=np.uint8).reshape(rows, 3)
        filled += rows
        del block_xyz[:]
        del block_rgb[:]

    for _ in range(count):
        fields = _POINT_HEAD.unpack(_read_exact(stream, _POINT_HEAD.size, "points3D"))
        block_xyz.extend(fields[1:4])
        block_rgb.extend(fields[4:7])
        _skip(stream, int(fields[8]) * _TRACK_ENTRY_BYTES, limit, "points3D (track)")
        if len(block_rgb) >= _POINT_BLOCK * 3:
            flush()
    flush()
    if filled != count:  # pragma: no cover - guarded by _read_exact above
        raise ColmapFormatError(f"COLMAP points3D declared {count} points, read {filled}")
    return xyz, rgb


def _byte(values: Sequence[str]) -> tuple[int, ...]:
    """Colour channels as bytes, clamped into range."""
    return tuple(min(255, max(0, int(value))) for value in values)


def _read_points3d_txt(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Text points have no count header, so the arrays grow instead of preallocating."""
    block_xyz = array("d")
    block_rgb = array("B")
    for line in _text_lines(path):
        # POINT3D_ID X Y Z R G B ERROR TRACK[] — everything past ERROR is the track.
        parts = line.split(maxsplit=8)
        if len(parts) < 8:
            raise ColmapFormatError(f"malformed COLMAP points3D.txt line: {line!r}")
        try:
            block_xyz.extend((float(parts[1]), float(parts[2]), float(parts[3])))
            # Clamped rather than masked: a channel outside [0,255] is a broken export,
            # and wrapping it would turn 300 into 44 — a plausible-looking wrong colour.
            block_rgb.extend(_byte(parts[4:7]))
        except ValueError as exc:
            raise ColmapFormatError(f"malformed COLMAP points3D.txt line: {line!r}") from exc
    rows = len(block_rgb) // 3
    if rows == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    xyz = np.frombuffer(block_xyz, dtype=np.float64).reshape(rows, 3).copy()
    rgb = np.frombuffer(block_rgb, dtype=np.uint8).reshape(rows, 3).copy()
    return xyz, rgb


# ---------------------------------------------------------------------------
# the cameras layer
# ---------------------------------------------------------------------------


def has_distortion(camera: ColmapCamera) -> bool:
    """Whether this camera carries **effective** distortion.

    The model name alone is not the answer: a ``SIMPLE_RADIAL`` with ``k = 0`` is exactly
    a ``SIMPLE_PINHOLE``, and COLMAP emits that constantly for already-undistorted
    images. The caveat in the provenance panel only earns its place when the numbers say
    the pinhole frustum is an approximation. Mirrors ``hasDistortion`` in
    ``sceneIntrinsics.ts``.
    """
    model = CAMERA_MODELS_BY_NAME.get(camera.model)
    if model is None or model.distortion == 0:
        return False
    first = 3 if model.shared_focal else 4
    return any(
        value != 0.0 and np.isfinite(value) for value in camera.params[first : model.param_count]
    )


def camera_layer_json(
    cameras: Mapping[int, ColmapCamera], images: Sequence[ColmapImage]
) -> dict[str, Any]:
    """The ``cameras`` layer payload, exactly as the renderer parses it.

    ``{"cameras": [{"qvec": [w,x,y,z], "tvec": [x,y,z], "name": ..., "camera": {...}}]}``,
    sorted by image name so two runs over the same model produce byte-identical output.

    **``qvec`` and ``tvec` are emitted verbatim.** They stay world-to-camera, ``qvec``
    stays ``(w, x, y, z)``, and no handedness flip is applied. The layer declares
    ``source_frame: "rdf"`` and the renderer performs the single inversion. An image
    whose ``camera_id`` is not in ``cameras`` is dropped rather than guessed at — the
    caller reports the drop in ``limitations``.
    """
    rows: list[dict[str, Any]] = []
    for image in sorted(images, key=lambda item: (item.name, item.image_id)):
        camera = cameras.get(image.camera_id)
        if camera is None:
            continue
        rows.append(
            {
                "qvec": [float(value) for value in image.qvec_wxyz],
                "tvec": [float(value) for value in image.tvec],
                "name": image.name,
                "camera": {
                    "model": camera.model,
                    "width": int(camera.width),
                    "height": int(camera.height),
                    "params": [float(value) for value in camera.params],
                },
            }
        )
    return {"cameras": rows}


def camera_centers(images: Sequence[ColmapImage]) -> np.ndarray:
    """World-space camera centres, ``-R^T t``, as ``(n, 3)`` float64.

    **This is not a wire format.** It exists so the derive can bound and frame a scene
    whose only geometry is its cameras, and so a cameras-only model gets a poster. The
    poses themselves are still emitted verbatim by :func:`camera_layer_json`; the
    renderer performs its own inversion, from the unconverted numbers, exactly once.
    """
    if not images:
        empty: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        return empty
    quat = np.asarray([image.qvec_wxyz for image in images], dtype=np.float64)
    tvec = np.asarray([image.tvec for image in images], dtype=np.float64)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    # A zero quaternion is not a rotation; identity keeps the centre finite so one broken
    # record cannot poison the whole bounding box with NaN.
    quat = np.divide(
        quat, norm, out=np.tile([1.0, 0.0, 0.0, 0.0], (quat.shape[0], 1)), where=norm > 0
    )
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # World-to-camera rotation R; the centre is -R^T t (never t, contract §2).
    rows = np.empty((quat.shape[0], 3, 3), dtype=np.float64)
    rows[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rows[:, 0, 1] = 2 * (x * y - z * w)
    rows[:, 0, 2] = 2 * (x * z + y * w)
    rows[:, 1, 0] = 2 * (x * y + z * w)
    rows[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rows[:, 1, 2] = 2 * (y * z - x * w)
    rows[:, 2, 0] = 2 * (x * z - y * w)
    rows[:, 2, 1] = 2 * (y * z + x * w)
    rows[:, 2, 2] = 1 - 2 * (x * x + y * y)
    centers: np.ndarray = -np.einsum("nji,nj->ni", rows, tvec)
    return centers
