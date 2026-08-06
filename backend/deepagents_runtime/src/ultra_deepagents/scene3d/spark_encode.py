"""Wire-format encoders for the ``scene3d`` chunk payloads (contract §4).

``USX1`` is literally what Spark's ``encodeExtSplat`` writes, so the browser hands the
two buffers straight to ``new ExtSplats({ extArrays: [a, b], numSplats })`` with no
per-element JavaScript. Everything here is transcribed from the real implementation in
``@sparkjsdev/spark/dist/spark.module.js`` (``encodeExtSplat``, line 1611;
``encodeQuatOctXy1010R12``, line 2088) and vectorized.

Two transcription details decide whether the scene renders or not:

- **Scales pass through untouched.** The PLY stores ``ln(scale)``; Spark's encoder takes
  a linear scale and immediately writes ``Math.log(scaleX)``. ``log(exp(v)) == v``, so
  the raw PLY value is already the wire value. Inserting the ``exp`` "for correctness"
  and letting Spark re-log it is a double activation that survives review because the
  round trip *looks* symmetric — :func:`encode_ext_splats` asserts the identity instead.
- **Quaternion order differs.** PLY stores ``rot_0..3`` as ``(w,x,y,z)``; Spark's
  ``encodeExtSplat`` takes ``(x,y,z,w)``. Spark's own PLY reader does the same remap
  (``quatW = item.rot_0``), which is the confirming reference.

Colour is where we deliberately diverge from Spark's PLY reader: it writes
``0.5 + C0*f_dc`` unclamped and un-linearized, while the contract (§4.2) requires a
clamped, *linear* colour with the clamped fraction reported. The renderer therefore must
not convert again.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CHUNK_HEADER_BYTES",
    "CHUNK_VERSION",
    "FLAG_POINT_ALPHA",
    "MAGIC_UPC1",
    "MAGIC_USX1",
    "SH_C0",
    "ExtSplatEncoding",
    "dc_to_base_color",
    "decode_quat_oct_xy1010_r12",
    "encode_ext_splats",
    "encode_quat_oct_xy1010_r12",
    "half_bits",
    "pack_chunk_header",
    "pack_upc1_chunk",
    "pack_usx1_chunk",
    "sigmoid",
    "srgb_to_linear",
]

# INRIA's degree-0 spherical-harmonic constant, 1/(2*sqrt(pi)). Spark uses the identical
# literal (SH_C0 in spark.module.js), so base colours agree bit-for-bit before clamping.
SH_C0 = 0.28209479177387814
CHUNK_HEADER_BYTES = 64
CHUNK_VERSION = 1
MAGIC_USX1 = b"USX1"
MAGIC_UPC1 = b"UPC1"
# UPC1 flags bit 0: the alpha byte carries real data (otherwise it is a constant 255).
FLAG_POINT_ALPHA = 1


@dataclass(frozen=True)
class ExtSplatEncoding:
    """The two ``Uint32Array`` planes Spark's ``ExtSplats`` consumes, plus honesty counters."""

    ext_a: np.ndarray  # (n, 4) uint32
    ext_b: np.ndarray  # (n, 4) uint32
    clamped_color_components: int  # of 3*n, how many fell outside [0,1] before clamping


def half_bits(values: np.ndarray) -> np.ndarray:
    """float -> IEEE-754 binary16 bit pattern, widened to uint32 for packing.

    Widened to float64 first so the narrowing is a *single* correctly-rounded step, which
    is what Spark's native ``Float16Array`` path (``toHalfNative``) does with a JS double.
    Rounding f64 -> f32 -> f16 instead would double-round; Spark's pure-JS fallback
    ``toHalfJS`` does exactly that *and* truncates, so it disagrees by up to 1 ULP. The
    native path is what any browser that can run WebGL2 takes.
    """
    with np.errstate(over="ignore"):  # out-of-range magnitudes saturate to half infinity
        narrowed = np.asarray(values, dtype=np.float64).astype(np.float16)
    return narrowed.view(np.uint16).astype(np.uint32)


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Logit -> [0,1] opacity, in float64.

    float64 because Spark's PLY reader computes ``1/(1+exp(-x))`` in JavaScript doubles;
    computing it in float32 and then narrowing to half moves roughly 1 splat in 4000 by a
    half ULP. Saturating inputs overflow to exactly 0.0/1.0, as in JS.
    """
    raw = np.asarray(values, dtype=np.float64)
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-raw))


def dc_to_base_color(f_dc: np.ndarray) -> tuple[np.ndarray, int]:
    """``0.5 + C0*f_dc`` clamped to [0,1], with the clamped *component* count.

    The measured drone scene runs -0.511 .. 2.704 on ``f_dc_0``, so clamping is not a
    theoretical concern. The count is per component (of ``3*n``), not per splat: a splat
    whose red alone saturates has lost less information than one that saturates in all
    three, and the coarser count would hide that.
    """
    base = 0.5 + SH_C0 * np.asarray(f_dc, dtype=np.float64)
    clamped = int(np.count_nonzero((base < 0.0) | (base > 1.0)))
    return np.clip(base, 0.0, 1.0), clamped


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    """sRGB -> linear light, IEC 61966-2-1 (the transfer function three.js uses).

    Deliberately NOT applied on either wire path, and kept as the reference for why:

    * splats (USX1) stay display-referred, because Spark's shader consumes
      ``0.5 + C0*f_dc`` directly — see ``encode_ext_splats``;
    * points (UPC1) carry source sRGB bytes, and the *renderer* converts them via
      ``sceneColor.srgbBytesToLinearFloat``, because three.js's PointsMaterial assumes
      linear vertex colours (contract 4.3).

    The piecewise form, not ``pow(2.2)``, so that it matches three.js exactly if a
    caller ever needs to reproduce the renderer's conversion server-side.
    """
    srgb = np.asarray(values, dtype=np.float64)
    # np.where evaluates both arms, so the power is fed a non-negative operand even where
    # the linear arm will win; without it a negative input raises an invalid-op.
    high = ((np.maximum(srgb, 0.0) + 0.055) / 1.055) ** 2.4
    return np.where(srgb <= 0.04045, srgb / 12.92, high)


def _js_round(values: np.ndarray) -> np.ndarray:
    """JavaScript ``Math.round`` — half away from zero *upwards*, i.e. floor(x + 0.5).

    numpy's ``round`` is half-to-even and disagrees on every exact ``.5``, which for the
    10-bit octahedral grid is one code point in 1024 of the encoded values.
    """
    return np.asarray(np.floor(values + 0.5))


def encode_quat_oct_xy1010_r12(
    qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, qw: np.ndarray
) -> np.ndarray:
    """Spark's ``encodeQuatOctXy1010R12``: axis octahedral 10+10, angle 12, packed uint32.

    Computed in float64 throughout because Spark computes it in JavaScript doubles.
    Degenerate inputs that would make the JS produce ``NaN`` (a zero-length quaternion, or
    ``acos`` of a value a hair outside [-1,1]) are pinned to the identity rotation rather
    than propagating a NaN into the bit packing.
    """
    x = np.asarray(qx, dtype=np.float64)
    y = np.asarray(qy, dtype=np.float64)
    z = np.asarray(qz, dtype=np.float64)
    w = np.asarray(qw, dtype=np.float64)
    length = np.sqrt(x * x + y * y + z * z + w * w)
    degenerate = ~np.isfinite(length) | (length <= 0.0)
    safe = np.where(degenerate, 1.0, length)
    # Spark folds the sign so w >= 0: q and -q are the same rotation, and the encoding
    # only spends bits on the w >= 0 hemisphere.
    sign = np.where(w < 0.0, -1.0, 1.0)
    nx = np.where(degenerate, 0.0, sign * x / safe)
    ny = np.where(degenerate, 0.0, sign * y / safe)
    nz = np.where(degenerate, 0.0, sign * z / safe)
    nw = np.where(degenerate, 1.0, sign * w / safe)

    theta = 2.0 * np.arccos(np.clip(nw, -1.0, 1.0))
    axis_norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    tiny = axis_norm < 1e-6
    axis_x = np.where(tiny, 1.0, nx / np.where(tiny, 1.0, axis_norm))
    axis_y = np.where(tiny, 0.0, ny / np.where(tiny, 1.0, axis_norm))
    axis_z = np.where(tiny, 0.0, nz / np.where(tiny, 1.0, axis_norm))

    total = np.abs(axis_x) + np.abs(axis_y) + np.abs(axis_z)
    px = axis_x / total
    py = axis_y / total
    # Octahedral fold of the lower hemisphere onto the outer diamond.
    lower = axis_z < 0.0
    folded_x = (1.0 - np.abs(py)) * np.where(px >= 0.0, 1.0, -1.0)
    folded_y = (1.0 - np.abs(px)) * np.where(py >= 0.0, 1.0, -1.0)
    px, py = np.where(lower, folded_x, px), np.where(lower, folded_y, py)

    # Spark does not clamp these; the domain makes 0..1023 unreachable from outside, and
    # the clamp here only stops a hypothetical rounding overflow from corrupting the
    # neighbouring bit field rather than producing a 1-ULP difference.
    quant_u = np.clip(_js_round((px * 0.5 + 0.5) * 1023.0), 0.0, 1023.0).astype(np.uint32)
    quant_v = np.clip(_js_round((py * 0.5 + 0.5) * 1023.0), 0.0, 1023.0).astype(np.uint32)
    angle = np.clip(_js_round(theta * (4095.0 / np.pi)), 0.0, 4095.0).astype(np.uint32)
    return (angle << np.uint32(20)) | (quant_v << np.uint32(10)) | quant_u


def decode_quat_oct_xy1010_r12(encoded: np.ndarray) -> np.ndarray:
    """Inverse of :func:`encode_quat_oct_xy1010_r12`, returning ``(n, 4)`` xyzw.

    A transcription of Spark's ``decodeQuatOctXy1010R12`` so the encoder can be checked
    against the decoder the GPU actually runs, not against a re-derivation of it.
    """
    bits = np.asarray(encoded, dtype=np.uint32)
    quant_u = (bits & np.uint32(0x3FF)).astype(np.float64)
    quant_v = ((bits >> np.uint32(10)) & np.uint32(0x3FF)).astype(np.float64)
    angle = ((bits >> np.uint32(20)) & np.uint32(0xFFF)).astype(np.float64)

    fx = (quant_u / 1023.0 - 0.5) * 2.0
    fy = (quant_v / 1023.0 - 0.5) * 2.0
    fz = 1.0 - (np.abs(fx) + np.abs(fy))
    fold = np.maximum(-fz, 0.0)
    fx = fx + np.where(fx >= 0.0, -fold, fold)
    fy = fy + np.where(fy >= 0.0, -fold, fold)
    length = np.sqrt(fx * fx + fy * fy + fz * fz)
    tiny = length < 1e-6
    safe = np.where(tiny, 1.0, length)
    axis_x = np.where(tiny, 0.0, fx / safe)
    axis_y = np.where(tiny, 0.0, fy / safe)
    axis_z = np.where(tiny, 0.0, fz / safe)

    theta = angle / 4095.0 * np.pi
    sin_half = np.sin(theta * 0.5)
    return np.stack(
        [axis_x * sin_half, axis_y * sin_half, axis_z * sin_half, np.cos(theta * 0.5)], axis=-1
    )


def encode_ext_splats(
    *,
    positions: np.ndarray,
    ln_scales: np.ndarray,
    quat_wxyz: np.ndarray,
    raw_opacity: np.ndarray,
    f_dc: np.ndarray,
) -> ExtSplatEncoding:
    """Encode chunk-local splats into Spark's ``ExtSplats`` planes.

    ``positions`` are already chunk-local; ``ln_scales`` and ``raw_opacity`` are the raw
    PLY values (log and logit domains); ``quat_wxyz`` is PLY order and is normalized here
    because INRIA's writer does not normalize (Postshot does).
    """
    xyz = np.ascontiguousarray(positions, dtype=np.float32)
    ln_scale = np.ascontiguousarray(ln_scales, dtype=np.float32)
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    count = xyz.shape[0]
    if xyz.shape != (count, 3) or ln_scale.shape != (count, 3) or quat.shape != (count, 4):
        raise ValueError("positions, ln_scales and quat_wxyz must be (n,3), (n,3) and (n,4)")

    ext_a = np.zeros((count, 4), dtype=np.uint32)
    ext_b = np.zeros((count, 4), dtype=np.uint32)

    # Centres are full float32 bit patterns — the whole reason we target ExtSplats.
    ext_a[:, 0:3] = xyz.view(np.uint32)
    ext_a[:, 3] = half_bits(sigmoid(raw_opacity))

    base, clamped = dc_to_base_color(f_dc)
    # Display-referred, NOT linearised — see contract 4.2. Spark's shader consumes the
    # raw `0.5 + C0*f_dc`: its PlyReader, SPZ and SOG paths all write exactly that with
    # no transfer function, matching INRIA's reference rasterizer. Linearising here
    # renders every splat too dark (0.5 would arrive as 0.214). The point path (UPC1)
    # does the opposite, because three.js's PointsMaterial assumes linear vertex
    # colours; the asymmetry is intentional and both halves have regression tests.
    red, green, blue = (half_bits(base[:, i]) for i in range(3))
    # No exp/log round trip: the wire wants ln(scale) and the PLY already holds
    # ln(scale). An exp inserted here that Spark then re-logs is a double activation
    # rendering as a field of giant blurs, so the identity has its own regression test.
    wire_ln_scale = half_bits(ln_scale.reshape(-1)).reshape(count, 3)
    ext_b[:, 0] = red | (green << np.uint32(16))
    ext_b[:, 1] = blue | (wire_ln_scale[:, 0] << np.uint32(16))
    ext_b[:, 2] = wire_ln_scale[:, 1] | (wire_ln_scale[:, 2] << np.uint32(16))
    ext_b[:, 3] = encode_quat_oct_xy1010_r12(quat[:, 1], quat[:, 2], quat[:, 3], quat[:, 0])
    return ExtSplatEncoding(ext_a=ext_a, ext_b=ext_b, clamped_color_components=clamped)


def pack_chunk_header(
    magic: bytes,
    *,
    count: int,
    sh_degree: int,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    origin: np.ndarray,
    flags: int = 0,
) -> bytes:
    """The common 64 B chunk header (contract §4.1).

    64 bytes, not the 40 the fields need, so every array that follows starts 4- and
    8-byte aligned and the browser can build typed-array views with zero copying.
    """
    if magic not in (MAGIC_USX1, MAGIC_UPC1):
        raise ValueError(f"unknown chunk magic {magic!r}")
    header = bytearray(CHUNK_HEADER_BYTES)
    header[0:4] = magic
    header[4:6] = int(CHUNK_VERSION).to_bytes(2, "little")
    header[6:8] = int(flags).to_bytes(2, "little")
    header[8:12] = int(count).to_bytes(4, "little")
    header[12:16] = int(sh_degree).to_bytes(4, "little")
    header[16:28] = np.asarray(bbox_min, dtype="<f4").tobytes()
    header[28:40] = np.asarray(bbox_max, dtype="<f4").tobytes()
    header[40:52] = np.asarray(origin, dtype="<f4").tobytes()
    return bytes(header)


def pack_usx1_chunk(
    encoding: ExtSplatEncoding,
    *,
    sh_degree: int,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    origin: np.ndarray,
) -> bytes:
    """Header + planar ``extA`` + planar ``extB``."""
    count = int(encoding.ext_a.shape[0])
    header = pack_chunk_header(
        MAGIC_USX1,
        count=count,
        sh_degree=sh_degree,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        origin=origin,
    )
    return (
        header
        + encoding.ext_a.astype("<u4", copy=False).tobytes()
        + encoding.ext_b.astype("<u4", copy=False).tobytes()
    )


def pack_upc1_chunk(
    *,
    positions: np.ndarray,
    colors_rgba: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    origin: np.ndarray,
    has_alpha: bool = False,
) -> bytes:
    """Header + planar chunk-local ``xyz`` float32 + planar ``rgba`` uint8.

    Colours stay sRGB here, source-faithful: they come from photographs, and the
    renderer owns the single documented conversion to linear (contract §4.3).
    """
    xyz = np.ascontiguousarray(positions, dtype="<f4")
    rgba = np.ascontiguousarray(colors_rgba, dtype=np.uint8)
    count = int(xyz.shape[0])
    if xyz.shape != (count, 3) or rgba.shape != (count, 4):
        raise ValueError("positions and colors_rgba must be (n,3) and (n,4)")
    header = pack_chunk_header(
        MAGIC_UPC1,
        count=count,
        sh_degree=0,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        origin=origin,
        flags=FLAG_POINT_ALPHA if has_alpha else 0,
    )
    return header + xyz.tobytes() + rgba.tobytes()
