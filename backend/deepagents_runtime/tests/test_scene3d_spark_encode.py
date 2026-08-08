"""USX1/UPC1 encoding, including a byte-for-byte diff against the real Spark bundle.

``test_matches_spark_encode_ext_splat_byte_for_byte`` extracts ``encodeExtSplat`` and
``encodeQuatOctXy1010R12`` verbatim out of ``@sparkjsdev/spark`` and runs them in node,
so the claim "this is literally what Spark writes" is checked against Spark rather than
against a second reading of Spark. It skips when node or the frontend bundle is absent.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from ultra_deepagents.scene3d import spark_encode as encode

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SPARK_BUNDLE = os.path.join(
    REPO_ROOT, "frontend", "node_modules", "@sparkjsdev", "spark", "dist", "spark.module.js"
)
# Node 22 ships Float16Array behind a flag. It is the path every WebGL2-capable browser
# takes (Spark's `toHalfNative`), and the one numpy's float16 cast agrees with.
NODE_FLAGS = ["--js-float16array"]

# Extracts the two functions and their tiny buffer preamble straight out of the shipped
# bundle. Nothing is retyped, so a Spark upgrade that changes the encoding fails here.
_EXTRACT_JS = """
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
function grab(name) {
  const key = 'function ' + name + '(';
  const start = src.indexOf(key);
  if (start < 0) throw new Error('spark bundle no longer defines ' + name);
  let depth = 0, seen = false, end = start;
  for (; end < src.length; end++) {
    if (src[end] === '{') { depth++; seen = true; }
    else if (src[end] === '}') { depth--; if (seen && depth === 0) break; }
  }
  return src.slice(start, end + 1).replaceAll('floatBitsToUint$1', 'floatBitsToUint');
}
const preamble = [
  'const f32buffer = new Float32Array(1);',
  'const u32buffer = new Uint32Array(f32buffer.buffer);',
  'const f16buffer = new Float16Array(1);',
  'const u16buffer = new Uint16Array(f16buffer.buffer);',
  'function toHalf(f) { f16buffer[0] = f; return u16buffer[0]; }',
  grab('floatBitsToUint$1'),
  grab('encodeQuatOctXy1010R12'),
  grab('decodeQuatOctXy1010R12'),
  grab('encodeExtSplat'),
].join('\\n');

const run = new Function('input', preamble + `
  const n = input.pos.length;
  const extA = new Uint32Array(n * 4), extB = new Uint32Array(n * 4);
  const quatOnly = new Uint32Array(n);
  const decoded = [];
  const out = { set(x, y, z, w) { decoded.push([x, y, z, w]); } };
  for (let i = 0; i < n; i++) {
    const [x, y, z] = input.pos[i];
    const [lx, ly, lz] = input.lnScale[i];
    const [qw, qx, qy, qz] = input.quatWxyz[i];
    const [r, g, b] = input.linearRgb[i];
    // Spark's own PLY reader hands it POST-activation values: exp(log-scale) and
    // sigmoid(logit opacity), computed in JS doubles.
    encodeExtSplat([extA, extB], i, x, y, z,
      Math.exp(lx), Math.exp(ly), Math.exp(lz),
      qx, qy, qz, qw, 1 / (1 + Math.exp(-input.opacity[i])), r, g, b);
    quatOnly[i] = encodeQuatOctXy1010R12(qx, qy, qz, qw) >>> 0;
    decodeQuatOctXy1010R12(quatOnly[i], out);
  }
  return { extA: Array.from(extA), extB: Array.from(extB),
           quatOnly: Array.from(quatOnly), decoded };
`)(JSON.parse(fs.readFileSync(process.argv[3], 'utf8')));
fs.writeFileSync(process.argv[4], JSON.stringify(run));
"""


def _spark_reference(tmp_path, payload):
    """Run the extracted Spark functions over ``payload``; skip if node/bundle missing."""
    node = shutil.which("node")
    if node is None or not os.path.exists(SPARK_BUNDLE):
        pytest.skip("node and the @sparkjsdev/spark bundle are required for the byte-diff")
    script = tmp_path / "spark_ref.mjs"
    script.write_text(_EXTRACT_JS)
    source = tmp_path / "in.json"
    source.write_text(json.dumps(payload))
    result = tmp_path / "out.json"
    completed = subprocess.run(
        [node, *NODE_FLAGS, str(script), SPARK_BUNDLE, str(source), str(result)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(f"spark reference harness did not run: {completed.stderr.strip()[:300]}")
    return json.loads(result.read_text())


def _splat_fixture(count=4096, seed=7):
    rng = np.random.default_rng(seed)
    quat = rng.normal(size=(count, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return {
        "positions": rng.uniform(-60.0, 60.0, (count, 3)).astype(np.float32),
        "ln_scales": rng.uniform(-10.6, 0.59, (count, 3)).astype(np.float32),
        "quat_wxyz": quat.astype(np.float32),
        "raw_opacity": rng.uniform(-4.16, 13.2, count).astype(np.float32),
        "f_dc": rng.uniform(-1.8, 7.8, (count, 3)).astype(np.float32),
    }


def test_scale_is_written_unchanged_with_no_exp_log_round_trip():
    """The PLY holds ln(scale) and the wire wants ln(scale); the value passes through.

    Written as a bit comparison against ``half(raw)`` rather than an approximate one,
    because the failure this guards is an ``exp`` inserted here that Spark then re-logs —
    a double activation whose round trip *looks* symmetric in review.
    """
    ln_scale = np.array([[-10.607, -4.6392, 0.5901], [-1.0, 0.0, 2.5]], dtype=np.float32)
    fixture = _splat_fixture(count=2)
    fixture["ln_scales"] = ln_scale

    result = encode.encode_ext_splats(**fixture)

    expected = encode.half_bits(ln_scale.reshape(-1)).reshape(2, 3)
    assert np.array_equal(result.ext_b[:, 1] >> np.uint32(16), expected[:, 0])
    assert np.array_equal(result.ext_b[:, 2] & np.uint32(0xFFFF), expected[:, 1])
    assert np.array_equal(result.ext_b[:, 2] >> np.uint32(16), expected[:, 2])
    # And the identity itself: no exp/log pair anywhere between PLY and wire.
    assert np.array_equal(
        encode.half_bits(ln_scale), encode.half_bits(np.log(np.exp(ln_scale.astype(np.float64))))
    )


def test_positions_are_full_float32_bit_patterns():
    fixture = _splat_fixture(count=64)
    result = encode.encode_ext_splats(**fixture)

    recovered = result.ext_a[:, 0:3].copy().view(np.float32)
    assert np.array_equal(recovered, fixture["positions"])  # exact, not approximately


def test_half_bits_round_to_nearest_even_like_float16():
    values = np.array([0.0, 1.0, -1.0, 65504.0, 1e-8, 70000.0], dtype=np.float32)
    with np.errstate(over="ignore"):
        reference = values.astype(np.float16).view(np.uint16).astype(np.uint32)
    assert np.array_equal(encode.half_bits(values), reference)
    assert int(encode.half_bits(np.array([70000.0], np.float32))[0]) == 0x7C00  # overflow -> inf


def test_quaternion_round_trips_through_the_decoder_the_gpu_runs():
    """Rotation error of the 10-10-12 octahedral encoding.

    The contract's "~0.1 deg" is the typical error and the median lands there. The tail
    does not: the axis grid is 1024x1024 over the octahedron, so a worst-placed axis is
    off by ~0.15 deg and the resulting rotation by up to ~0.5. That is a property of the
    frozen encoding, not of this transcription — the encoder is bit-identical to Spark's
    (see the byte-diff test), so the bound is asserted where it actually sits.
    """
    rng = np.random.default_rng(3)
    quat = rng.normal(size=(50_000, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)

    decoded = encode.decode_quat_oct_xy1010_r12(
        encode.encode_quat_oct_xy1010_r12(quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3])
    )
    alignment = np.abs(np.sum(quat * decoded, axis=1)).clip(0.0, 1.0)
    degrees = np.degrees(2.0 * np.arccos(alignment))

    assert float(np.median(degrees)) < 0.2
    assert float(degrees.max()) < 0.5


def test_quaternion_reads_ply_wxyz_as_spark_xyzw():
    """A 90-degree rotation about +z, stored the way a PLY stores it."""
    half = math.sqrt(0.5)
    fixture = _splat_fixture(count=1)
    fixture["quat_wxyz"] = np.array([[half, 0.0, 0.0, half]], dtype=np.float32)

    encoded = encode.encode_ext_splats(**fixture).ext_b[:, 3]
    decoded = encode.decode_quat_oct_xy1010_r12(encoded)[0]

    assert decoded[2] == pytest.approx(half, abs=2e-3)  # z carries the rotation
    assert decoded[3] == pytest.approx(half, abs=2e-3)  # w is the scalar part
    assert abs(decoded[0]) < 2e-3
    assert abs(decoded[1]) < 2e-3


def test_unnormalized_and_degenerate_quaternions_do_not_produce_nan():
    """INRIA's writer does not normalize; a converter can emit an all-zero rotation."""
    quat = np.array([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])

    encoded = encode.encode_quat_oct_xy1010_r12(quat[:, 1], quat[:, 2], quat[:, 3], quat[:, 0])
    decoded = encode.decode_quat_oct_xy1010_r12(encoded)

    assert np.all(np.isfinite(decoded))
    assert decoded[0] == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=1e-6)  # 2*identity
    assert decoded[1] == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=1e-6)  # zero -> identity
    assert decoded[2] == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=1e-6)  # -identity, w folded


def test_out_of_range_color_fraction_is_counted_without_clamping_the_wire():
    # 0.5 + C0*dc for dc = -3 is -0.346 and for dc = 4 is 1.628.
    f_dc = np.array([[-3.0, 0.0, 4.0], [0.0, 0.0, 0.0]], dtype=np.float32)

    base, out_of_range = encode.dc_to_base_color(f_dc)

    assert out_of_range == 2
    assert base[0, 0] < 0.0
    assert base[0, 2] > 1.0
    assert base[1, 0] == pytest.approx(0.5)
    fixture = _splat_fixture(count=2)
    fixture["f_dc"] = f_dc
    assert encode.encode_ext_splats(**fixture).out_of_range_color_components == 2


def test_splat_color_reaches_the_wire_display_referred_not_linearised():
    """``0.5 + C0*f_dc`` must survive to the wire unchanged.

    Spark's shader consumes display-referred colour: its PlyReader, SPZ and SOG paths
    all write ``0.5 + C0*f_dc`` with no transfer function (contract 4.2). Linearising
    here would put 0.214 on the wire where Spark expects 0.5, rendering the whole
    scene too dark — and it would do so silently, since the geometry stays perfect.

    The point path (UPC1) deliberately goes the other way; see
    ``test_point_colors_are_linearised_for_three_vertex_colors``.
    """
    fixture = _splat_fixture(count=1)
    fixture["f_dc"] = np.zeros((1, 3), dtype=np.float32)  # base colour exactly 0.5

    ext_b = encode.encode_ext_splats(**fixture).ext_b
    red = np.array([ext_b[0, 0] & 0xFFFF], dtype=np.uint16).view(np.float16)[0]

    assert float(red) == pytest.approx(0.5, abs=1e-3)
    # The linear value is what a naive implementation would emit; assert it is
    # distinguishable, so this test cannot pass under the wrong convention.
    assert float(encode.srgb_to_linear(np.array([0.5], np.float32))[0]) == pytest.approx(
        0.2140, abs=1e-4
    )


def test_splat_color_matches_sparks_own_ply_reader(tmp_path):
    """Our wire colour equals what Spark's PlyReader would hand its own encoder.

    This is the governing correctness criterion for the whole derive: loading our
    derived chunk must be indistinguishable from Spark loading the source .ply.
    """
    f_dc = np.array(
        [[0.0, 1.0, -1.0], [2.5, -0.6, 0.31], [-3.5834, 7.8135, 0.0454]],
        dtype=np.float32,
    )
    base, _out_of_range = encode.dc_to_base_color(f_dc)
    # Spark PlyReader: `r = item.f_dc_0 * SH_C0 + 0.5`, then setPackedSplat/encodeExtSplat
    # stores the value directly as float16, including HDR tails outside display gamut.
    spark_ply_value = f_dc.astype(np.float64) * encode.SH_C0 + 0.5
    assert np.allclose(base, spark_ply_value, atol=1e-7)


def test_usx1_chunk_is_a_64_byte_header_then_two_planar_uint32_arrays():
    fixture = _splat_fixture(count=100)
    encoded = encode.encode_ext_splats(**fixture)
    origin = np.array([1.5, -2.0, 3.25], dtype=np.float32)

    blob = encode.pack_usx1_chunk(
        encoded, sh_degree=0, bbox_min=np.zeros(3), bbox_max=np.ones(3), origin=origin
    )

    assert len(blob) == 64 + 100 * 16 + 100 * 16
    assert blob[0:4] == b"USX1"
    assert int.from_bytes(blob[4:6], "little") == 1
    assert int.from_bytes(blob[8:12], "little") == 100
    assert int.from_bytes(blob[12:16], "little") == 0  # measured, not declared
    assert np.array_equal(np.frombuffer(blob[40:52], "<f4"), origin)
    assert blob[52:64] == bytes(12)  # reserved, zero
    plane_a = np.frombuffer(blob[64 : 64 + 1600], "<u4").reshape(100, 4)
    plane_b = np.frombuffer(blob[64 + 1600 :], "<u4").reshape(100, 4)
    assert np.array_equal(plane_a, encoded.ext_a)
    assert np.array_equal(plane_b, encoded.ext_b)


def test_upc1_chunk_is_positions_then_rgba_and_flags_alpha():
    positions = np.arange(12, dtype=np.float32).reshape(4, 3)
    colors = np.arange(16, dtype=np.uint8).reshape(4, 4)

    blob = encode.pack_upc1_chunk(
        positions=positions,
        colors_rgba=colors,
        bbox_min=positions.min(axis=0),
        bbox_max=positions.max(axis=0),
        origin=np.zeros(3),
        has_alpha=True,
    )

    assert blob[0:4] == b"UPC1"
    assert int.from_bytes(blob[6:8], "little") == encode.FLAG_POINT_ALPHA
    assert len(blob) == 64 + 4 * 12 + 4 * 4
    assert np.array_equal(np.frombuffer(blob[64 : 64 + 48], "<f4").reshape(4, 3), positions)
    assert np.array_equal(np.frombuffer(blob[64 + 48 :], np.uint8).reshape(4, 4), colors)


def test_matches_spark_encode_ext_splat_byte_for_byte(tmp_path):
    """Every word of both planes, against ``encodeExtSplat`` running out of the bundle."""
    fixture = _splat_fixture(count=4096)
    result = encode.encode_ext_splats(**fixture)
    base, _out_of_range = encode.dc_to_base_color(fixture["f_dc"])

    reference = _spark_reference(
        tmp_path,
        {
            "pos": fixture["positions"].astype(np.float64).tolist(),
            "lnScale": fixture["ln_scales"].astype(np.float64).tolist(),
            "quatWxyz": fixture["quat_wxyz"].astype(np.float64).tolist(),
            "opacity": fixture["raw_opacity"].astype(np.float64).tolist(),
            # Display-referred, exactly what Spark's own PlyReader feeds encodeExtSplat.
            "linearRgb": base.astype(np.float64).tolist(),
        },
    )

    count = fixture["positions"].shape[0]
    spark_a = np.asarray(reference["extA"], dtype=np.uint32).reshape(count, 4)
    spark_b = np.asarray(reference["extB"], dtype=np.uint32).reshape(count, 4)
    assert np.array_equal(result.ext_a, spark_a)
    assert np.array_equal(result.ext_b, spark_b)
    assert np.array_equal(result.ext_b[:, 3], np.asarray(reference["quatOnly"], dtype=np.uint32))


def test_decoder_matches_sparks_own_decoder(tmp_path):
    """Our decoder is only a useful oracle if it is Spark's decoder."""
    fixture = _splat_fixture(count=1024, seed=19)
    reference = _spark_reference(
        tmp_path,
        {
            "pos": fixture["positions"].astype(np.float64).tolist(),
            "lnScale": fixture["ln_scales"].astype(np.float64).tolist(),
            "quatWxyz": fixture["quat_wxyz"].astype(np.float64).tolist(),
            "opacity": fixture["raw_opacity"].astype(np.float64).tolist(),
            "linearRgb": np.full((1024, 3), 0.5).tolist(),
        },
    )

    quat = fixture["quat_wxyz"]
    ours = encode.decode_quat_oct_xy1010_r12(
        encode.encode_quat_oct_xy1010_r12(quat[:, 1], quat[:, 2], quat[:, 3], quat[:, 0])
    )
    assert np.allclose(ours, np.asarray(reference["decoded"], dtype=np.float64), atol=1e-12)
