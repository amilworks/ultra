"""Streaming, header-driven PLY reader for the ``scene3d`` derive.

The header is the only authority on layout. Real 3DGS writers disagree about which
properties they emit — INRIA writes ``nx,ny,nz`` (248 B/vertex for the canonical
degree-3 layout), Postshot omits them (236 B) — so every property offset here is
*derived* from the declared property order. A hardcoded stride silently misreads
every field of one of those two families, and the failure looks like a plausible
scene rather than an error.

The reader is deliberately dumb about semantics: it yields numpy structured views
over the raw records and leaves activation domains (``log`` scale, ``logit``
opacity) to :mod:`~ultra_deepagents.scene3d.spark_encode`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_BATCH",
    "DEFAULT_SH_SAMPLE",
    "DEFAULT_SH_SEED",
    "PlyFormatError",
    "PlyHeader",
    "PlyProperty",
    "declared_sh_degree",
    "detect_scene_kind",
    "f_rest_names",
    "iter_chunks",
    "measured_sh_degree",
    "read_header",
    "source_writer",
]

# PLY 1.0 scalar type names -> numpy base kinds. Both the canonical names and the
# explicit-width aliases (which several exporters emit) are accepted.
_SCALAR_KINDS: dict[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}
_BYTE_ORDER = {
    "binary_little_endian": "<",
    "binary_big_endian": ">",
    "ascii": "|",
}
# A header larger than this is malformed or hostile; the real degree-3 splat header
# is 1512 B and the point-cloud header 248 B.
MAX_HEADER_BYTES = 1 << 20
# 64k records per batch: ~15 MB for the 236 B splat record, small enough to keep the
# working set in cache and large enough that per-batch numpy overhead disappears.
DEFAULT_BATCH = 65536
# The sample size and seed the contract's Appendix A measurement used; reproducing
# them here means `measured_sh_degree` can be checked against that measurement.
DEFAULT_SH_SAMPLE = 200_000
DEFAULT_SH_SEED = 7
_F_REST_RE = re.compile(r"^f_rest_(\d+)$")
# Spherical-harmonic coefficient counts (all channels) for degrees 0..3. A band `l`
# contributes 2l+1 coefficients per channel; the DC band lives in f_dc, so f_rest
# holds ((l+1)^2 - 1) * 3.
_SH_REST_COUNTS = {0: 0, 9: 1, 24: 2, 45: 3}
# Properties that, together, prove the element is a 3D Gaussian and not a point.
_SPLAT_REQUIRED = ("f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "rot_0")
_POINT_REQUIRED = ("x", "y", "z")


class PlyFormatError(ValueError):
    """The file is not a PLY we can address record-by-record."""


@dataclass(frozen=True)
class PlyProperty:
    """One fixed-width scalar property and where it sits inside a record."""

    name: str
    dtype: str  # numpy dtype string including byte order, e.g. "<f4"
    offset: int  # byte offset within the element record
    size: int  # width in bytes


@dataclass(frozen=True)
class PlyHeader:
    """Everything needed to address the records of one PLY element."""

    format: str  # "binary_little_endian" | "binary_big_endian" | "ascii"
    element: str  # element name, normally "vertex"
    count: int  # declared record count
    stride: int  # bytes per record (meaningless for ascii)
    data_offset: int  # byte offset of this element's first record
    props: tuple[PlyProperty, ...]
    comments: tuple[str, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(prop.name for prop in self.props)

    def has(self, *names: str) -> bool:
        available = set(self.names)
        return all(name in available for name in names)

    def prop(self, name: str) -> PlyProperty:
        for candidate in self.props:
            if candidate.name == name:
                return candidate
        raise KeyError(name)

    def dtype(self, names: tuple[str, ...] | None = None) -> np.dtype:
        """A structured dtype over the record, optionally over a subset of fields.

        The itemsize is pinned to the stride, so a subset view still walks the record
        grid correctly and unread properties cost nothing.
        """
        selected = self.props if names is None else tuple(self.prop(name) for name in names)
        return np.dtype(
            {
                "names": [prop.name for prop in selected],
                "formats": [prop.dtype for prop in selected],
                "offsets": [prop.offset for prop in selected],
                "itemsize": self.stride,
            }
        )


def _split_header(blob: bytes) -> tuple[str, int]:
    """Return the header text and the byte offset of the first data record.

    Line endings are not normalized in the file: the measured point cloud uses CRLF
    and the measured splat file uses LF, and the difference lands directly in
    ``data_offset``.
    """
    if not blob.startswith(b"ply"):
        raise PlyFormatError("not a PLY file: missing the 'ply' magic")
    marker = blob.find(b"end_header")
    if marker < 0:
        raise PlyFormatError("PLY header is missing 'end_header' (or exceeds the size cap)")
    newline = blob.find(b"\n", marker)
    if newline < 0:
        raise PlyFormatError("PLY header 'end_header' is not terminated by a newline")
    try:
        text = blob[:marker].decode("ascii")
    except UnicodeDecodeError as exc:
        raise PlyFormatError("PLY header is not ASCII") from exc
    return text, newline + 1


def read_header(path: str | os.PathLike[str]) -> PlyHeader:
    """Parse a PLY header into an ordered property table.

    The element addressed is ``vertex`` when present, else the first element. Any
    element *before* it must be fixed-width, otherwise its size — and therefore our
    element's data offset — is not computable without reading the file.
    """
    with open(path, "rb") as stream:
        blob = stream.read(MAX_HEADER_BYTES)
    text, data_offset = _split_header(blob)

    fmt: str | None = None
    comments: list[str] = []
    # (name, count, props, stride, has_list)
    elements: list[tuple[str, int, list[PlyProperty], int, bool]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line == "ply":
            continue
        parts = line.split()
        keyword = parts[0]
        if keyword == "format":
            if len(parts) < 2 or parts[1] not in _BYTE_ORDER:
                raise PlyFormatError(f"unsupported PLY format line: {line!r}")
            fmt = parts[1]
        elif keyword in ("comment", "obj_info"):
            comments.append(line.split(" ", 1)[1] if " " in line else "")
        elif keyword == "element":
            if len(parts) < 3:
                raise PlyFormatError(f"malformed PLY element line: {line!r}")
            try:
                count = int(parts[2])
            except ValueError as exc:
                raise PlyFormatError(f"malformed PLY element count: {line!r}") from exc
            if count < 0:
                raise PlyFormatError(f"negative PLY element count: {line!r}")
            elements.append((parts[1], count, [], 0, False))
        elif keyword == "property":
            if not elements:
                raise PlyFormatError("PLY property declared before any element")
            name, size, props, stride, has_list = elements[-1]
            if parts[1] == "list":
                # A list property makes the record variable-width. We never index into
                # such an element, but we must know it exists to refuse it cleanly.
                elements[-1] = (name, size, props, stride, True)
                continue
            if len(parts) < 3:
                raise PlyFormatError(f"malformed PLY property line: {line!r}")
            kind = _SCALAR_KINDS.get(parts[1])
            if kind is None:
                raise PlyFormatError(f"unsupported PLY property type: {parts[1]!r}")
            width = int(kind[1:])
            order = _BYTE_ORDER[fmt] if fmt is not None else "<"
            props.append(
                PlyProperty(name=parts[2], dtype=f"{order}{kind}", offset=stride, size=width)
            )
            elements[-1] = (name, size, props, stride + width, has_list)
    if fmt is None:
        raise PlyFormatError("PLY header has no format line")
    if not elements:
        raise PlyFormatError("PLY header declares no elements")

    target = next((i for i, element in enumerate(elements) if element[0] == "vertex"), 0)
    for name, count, _props, stride, has_list in elements[:target]:
        if has_list:
            raise PlyFormatError(
                f"cannot address the vertex element: preceding element {name!r} is variable-width"
            )
        data_offset += count * stride
    name, count, props, stride, has_list = elements[target]
    if has_list:
        raise PlyFormatError(f"element {name!r} has a list property and is not stride-addressable")
    if not props:
        raise PlyFormatError(f"element {name!r} declares no properties")
    return PlyHeader(
        format=fmt,
        element=name,
        count=count,
        stride=stride,
        data_offset=data_offset,
        props=tuple(props),
        comments=tuple(comments),
    )


def iter_chunks(
    path: str | os.PathLike[str],
    header: PlyHeader,
    batch: int = DEFAULT_BATCH,
    *,
    names: tuple[str, ...] | None = None,
) -> Iterator[np.ndarray]:
    """Yield the element's records as numpy structured arrays, in file order.

    Stops early on a short read: the caller reconciles the yielded total against
    ``header.count`` and decides whether a truncated source is a failure. Nothing is
    silently padded or dropped here.
    """
    if header.format == "ascii":
        raise NotImplementedError(
            "ascii PLY is recognized but not decoded; re-export as binary_little_endian"
        )
    if batch < 1:
        raise ValueError("batch must be >= 1")
    record = header.dtype(names)
    with open(path, "rb") as stream:
        stream.seek(header.data_offset)
        remaining = header.count
        while remaining > 0:
            want = min(batch, remaining)
            blob = stream.read(want * header.stride)
            got = len(blob) // header.stride
            if got < 1:
                return
            yield np.frombuffer(blob, dtype=record, count=got)
            remaining -= got
            if got < want:
                return


def detect_scene_kind(header: PlyHeader) -> str:
    """``"splat"`` or ``"pointcloud"``.

    A 3D Gaussian is proven by the full parameter set — DC colour, logit opacity, log
    scale and a rotation. Anything else carrying x,y,z is a point map; ``f_dc`` alone
    is not enough, because a converter that dropped the scales would otherwise be read
    as a splat scene and render as invisible zero-extent Gaussians.
    """
    if header.has(*_SPLAT_REQUIRED):
        return "splat"
    if header.has(*_POINT_REQUIRED):
        return "pointcloud"
    raise PlyFormatError(
        f"element {header.element!r} has neither splat parameters nor x,y,z positions"
    )


def f_rest_names(header: PlyHeader) -> tuple[str, ...]:
    """``f_rest_*`` property names ordered by their numeric suffix, not header order."""
    found: list[tuple[int, str]] = []
    for name in header.names:
        match = _F_REST_RE.match(name)
        if match is not None:
            found.append((int(match.group(1)), name))
    found.sort()
    return tuple(name for _index, name in found)


def declared_sh_degree(header: PlyHeader) -> int:
    """The SH degree the header *claims*, from the ``f_rest_*`` count.

    A count that is not one of 0/9/24/45 is not a recognizable SH layout; we report 0
    rather than guessing a partial band, and the derive states so in its limitations.
    """
    return _SH_REST_COUNTS.get(len(f_rest_names(header)), 0)


def _rest_bands(count: int) -> np.ndarray:
    """Band index (1..3) for each ``f_rest`` slot in INRIA's channel-major layout.

    INRIA writes ``features_rest`` as (channel, coefficient) flattened, so slot ``i``
    is channel ``i // k``, coefficient ``c = i % k``, and coefficient ``c`` belongs to
    band ``floor(sqrt(c + 1))`` (band 1 -> c 0..2, band 2 -> c 3..7, band 3 -> c 8..14).
    """
    per_channel = count // 3
    coefficient = np.arange(count, dtype=np.int64) % per_channel
    return np.asarray(np.floor(np.sqrt(coefficient + 1)), dtype=np.int64)


def _sample_offsets(
    path: str | os.PathLike[str], header: PlyHeader, index: np.ndarray
) -> Iterator[np.ndarray]:
    """Yield records at scattered indices as structured arrays, in blocks."""
    block = 8192
    with open(path, "rb") as stream:
        descriptor = stream.fileno()
        for start in range(0, index.size, block):
            wanted = index[start : start + block]
            buffer = bytearray(wanted.size * header.stride)
            view = memoryview(buffer)
            kept = 0
            for position in wanted:
                offset = header.data_offset + int(position) * header.stride
                raw = os.pread(descriptor, header.stride, offset)
                if len(raw) < header.stride:
                    continue  # truncated tail; the derive reconciles totals separately
                view[kept * header.stride : (kept + 1) * header.stride] = raw
                kept += 1
            if kept:
                yield np.frombuffer(bytes(view[: kept * header.stride]), dtype=header.dtype())


def measured_sh_degree(
    path: str | os.PathLike[str],
    header: PlyHeader,
    sample: int = DEFAULT_SH_SAMPLE,
    *,
    seed: int = DEFAULT_SH_SEED,
) -> int:
    """The highest SH band that actually carries a non-zero coefficient.

    Files routinely *declare* degree 3 and store 45 zeroes — the measured drone scene
    allocates 180 of its 236 bytes per splat to bands that are exactly 0.0 — so the
    declared degree is not usable for anything. Returns 0 when every sampled
    coefficient is zero, whatever the header claims.

    Sampling is random (uniform over the element, fixed seed) rather than a prefix: a
    prefix would miss a scene whose higher bands are only populated in a later region.
    """
    names = f_rest_names(header)
    declared = declared_sh_degree(header)
    if not names or declared == 0 or header.count < 1:
        return 0
    if header.format == "ascii":
        raise NotImplementedError(
            "ascii PLY is recognized but not decoded; re-export as binary_little_endian"
        )
    bands = _rest_bands(len(names))
    nonzero = np.zeros(len(names), dtype=bool)

    if sample >= header.count:
        blocks: Iterator[np.ndarray] = iter_chunks(path, header, names=names)
    else:
        rng = np.random.default_rng(seed)
        # unique() both de-duplicates and sorts, turning scattered reads into a
        # forward-only walk of the file.
        index = np.unique(rng.integers(0, header.count, size=sample, dtype=np.int64))
        blocks = _sample_offsets(path, header, index)

    for block in blocks:
        for slot, name in enumerate(names):
            if nonzero[slot]:
                continue
            nonzero[slot] = bool(np.any(block[name] != 0.0))
        if nonzero[bands == declared].any():
            break  # the top band is populated; nothing lower can raise the answer
    if not nonzero.any():
        return 0
    return int(min(declared, bands[nonzero].max()))


def source_writer(header: PlyHeader) -> str | None:
    """Best-effort producer name from the header comments.

    Postshot stamps ``comment postshot.<key>=<value>``; other writers stamp a free
    sentence. Returns ``None`` rather than guessing when nothing is recognizable.
    """
    for comment in header.comments:
        lowered = comment.lower()
        for writer in ("postshot", "nerfstudio", "opensplat", "brush", "colmap", "polycam"):
            if writer in lowered:
                return writer
        if lowered.startswith("generated by "):
            return comment[len("generated by ") :].strip() or None
    return None
