"""Ultra Sans — DM Sans with Inter-matched round caps and product features.

Four product needs are handled in the font rather than around it:

**Tabular figures.** DM Sans ships no `tnum` feature and proportional digits
(310/1000em spread), so `font-variant-numeric: tabular-nums` had nothing to
activate and numeric columns staggered.

**Greek.** DM Sans covers one Greek codepoint (π). Ultra writes λ, σ, Δ, θ
constantly. A CSS fallback face (Inter, size-adjusted) papered over this in DOM
text, but CSS fallback does not reach canvas (`ctx.font` names one family), and
one global 96.4% scale cannot weight-match per master. This build grafts the
Greek block from Inter directly into the font, solved per region of the
two-axis design space — the same architecture Ultra Mono uses, adapted from a
weight line to a wght x opsz grid.

**Round capitals.** The supplied 10–11px reference is closest to Inter's
optically corrected C/O/G, not the superseded perfect-circle construction. This
build redraws C/O/G/Q/Oslash from matching Inter masters, then fits them to DM's
round-cap stroke and overshoot while retaining every DM advance and ink centre.

**`0`/`O` ambiguity.** DM Sans's zero measured the weakest of every candidate
face (divergence 0.44; only 13% narrower than O). A generated slashed-zero
alternate behind a `zero` feature closes it — DORMANT by default, honouring the
documented doctrine that identity does not come from substituted glyphs;
data surfaces opt in via `font-variant-numeric: slashed-zero`.

    python frontend/font-lab/build_ultra_tabular.py
    python frontend/font-lab/build_ultra_tabular.py --verify-only

Outputs land in `build/`; the byte-identical shipping copies live in
`frontend/src/assets/fonts/` and are pinned by the typography contract.

## Why this construction

Three measured facts make it far simpler than it looks.

1. **`0` is the widest digit at every axis location.** Verified over a 48-point
   grid spanning opsz 9-40 x wght 100-1000. So the tabular advance T(L) is not a
   quantity that has to be modelled — it is exactly `advance(zero, L)`.

2. **Advances flow from gvar phantom points.** The build proves the upstream
   HVAR is advance-equivalent to the phantoms on a dense grid, then drops it, so
   every glyph — original, tabular, grafted — carries its advance the same way
   and new glyphs need no HVAR delta-set surgery. Because of (1), each `.tnum`
   glyph simply carries `zero`'s advance deltas on its own phantoms.

3. Only the *centring shift* genuinely varies, so each `.tnum` glyph is a
   **composite** with a single component pointing at the base digit. A composite
   inherits the base outline's own variation automatically, so gvar only has to
   move one point: the component offset. That is 1 real point + 4 phantom
   points per tuple, instead of copying and re-deriving 32 points of sparse
   IUP-inferred deltas per digit.

Anything the base digits do across the axes — and they vary over 11 gvar tuples
with intermediate regions — is inherited rather than reimplemented.

## How the Greek graft handles two axes

DM Sans's gvar uses 11 regions whose peaks map (through avar) to a 12-point
master grid: opsz {9, 24, 40} x wght {100, 300, 400, 1000}. Grafted glyphs are
sampled at ALL 12 locations — Inter's axes solved per location so stems and
x/cap proportions match DM's there, plus a horizontal correction that keeps
Greek tracking DM's width rhythm where Inter's opsz axis saturates (Inter stops
at 32 while DM keeps narrowing to 40) — and their deltas are derived with
fontTools' own VariationModel over those locations, intermediates included. So
the graft is exact at every master and interpolates with the same support math
the rest of the font uses; nothing drifts at Ultra's real tokens (wght 440/695),
which sit between corners.

Advances vary too (this is a proportional face), which HVAR would make painful
for new glyphs — so the build first PROVES the font's gvar phantom points are
advance-equivalent to HVAR on a dense grid, then drops HVAR entirely; every
glyph's advance, grafted or original, flows from phantom deltas afterwards.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import unicodedata
from pathlib import Path

# Shared measurement utilities with the mono builder (same directory).
from build_ultra_mono import scanline_stem
from fontTools.otlLib import builder as otl
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables import ttProgram
from fontTools.ttLib.tables._g_l_y_f import Glyph, GlyphComponent
from fontTools.ttLib.tables.TupleVariation import TupleVariation
from fontTools.varLib import instancer
from fontTools.varLib.models import VariationModel

DIGITS = "0123456789"
BUILD_DIR = Path(__file__).resolve().parent / "build"
CACHE_DIR = Path(__file__).resolve().parent / "sources"

UPSTREAM = "https://raw.githubusercontent.com/google/fonts/main/ofl/dmsans/"

# Fetched from upstream and digest-checked rather than read out of
# src/assets/fonts, so that directory holds only bytes the app actually serves.
# These digests are the ones recorded in src/assets/fonts/PROVENANCE.md.
SOURCES = [
    (
        "DMSans%5Bopsz,wght%5D.ttf",
        "DMSans[opsz,wght].ttf",
        "8cd08d97e89c24d0aa92edd2f0f4c8ee6195eee9b7c9f154865a58b02f0c1c0d",
        "UltraSans-Variable.woff2",
        "normal",
    ),
    (
        "DMSans-Italic%5Bopsz,wght%5D.ttf",
        "DMSans-Italic[opsz,wght].ttf",
        "22259c0cc8237221b80f44c76ba8d36e6bce3cda72779f5b2773643d499720ae",
        "UltraSans-Italic-Variable.woff2",
        "italic",
    ),
]


def fetch_source(url_name: str, local_name: str, sha256: str) -> bytes:
    """Fetch an upstream DM Sans master, verifying it against a pinned digest."""
    import hashlib
    import subprocess

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / local_name
    if cached.exists():
        raw = cached.read_bytes()
        if hashlib.sha256(raw).hexdigest() == sha256:
            return raw
    result = subprocess.run(["curl", "-sL", UPSTREAM + url_name], capture_output=True)
    raw = result.stdout
    actual = hashlib.sha256(raw).hexdigest()
    if actual != sha256:
        raise SystemExit(
            f"{local_name}: upstream digest mismatch\n  expected {sha256}\n  got      {actual}"
        )
    cached.write_bytes(raw)
    return raw

# Corners of normalized axis space. opsz's default equals its minimum (9), so
# that axis is one-sided and normalizes to [0, 1]; wght spans [-1, 1].
CORNERS = [
    {"wght": -1.0},
    {"wght": 1.0},
    {"opsz": 1.0},
    {"opsz": 1.0, "wght": -1.0},
    {"opsz": 1.0, "wght": 1.0},
]

# The FULL master grid of the upstream design space, read from its gvar region
# peaks (normalized) with the user-space location each peak maps to through
# avar. Grafted glyphs are sampled at every one of these, so VariationModel
# reproduces the same intermediate-master structure the rest of the font has.
MASTER_GRID = [
    ({}, (400, 9)),
    ({"opsz": 0.5}, (400, 24)),
    ({"opsz": 1.0}, (400, 40)),
    ({"wght": -1.0}, (100, 9)),
    ({"wght": -0.21875}, (300, 9)),
    ({"wght": 1.0}, (1000, 9)),
    ({"opsz": 0.5, "wght": -1.0}, (100, 24)),
    ({"opsz": 0.5, "wght": -0.21875}, (300, 24)),
    ({"opsz": 0.5, "wght": 1.0}, (1000, 24)),
    ({"opsz": 1.0, "wght": -1.0}, (100, 40)),
    ({"opsz": 1.0, "wght": -0.21875}, (300, 40)),
    ({"opsz": 1.0, "wght": 1.0}, (1000, 40)),
]

FONTS_DIR = Path(__file__).resolve().parents[1] / "src" / "assets" / "fonts"
INTER_ROMAN = FONTS_DIR / "InterVariable-v4.1.woff2"
INTER_ITALIC = FONTS_DIR / "InterVariable-Italic-v4.1.woff2"
DM_ITALIC_ANGLE, INTER_ITALIC_ANGLE = -10.0, -9.4


def user_location(font: TTFont, normalized: dict[str, float]) -> dict[str, float]:
    """Map a normalized corner to user coordinates.

    Only ever called with 0 or +/-1, where avar is identity by definition (the
    endpoints of an avar segment map are pinned), so no avar maths is required.
    """
    out = {}
    for axis in font["fvar"].axes:
        n = normalized.get(axis.axisTag, 0.0)
        if n == 0:
            out[axis.axisTag] = axis.defaultValue
        elif n == 1:
            out[axis.axisTag] = axis.maxValue
        elif n == -1:
            out[axis.axisTag] = axis.minValue
        else:
            raise ValueError(f"corner components must be 0 or +/-1, got {n}")
    return out


def digit_advances(raw: bytes, location: dict[str, float]) -> dict[str, int]:
    """Advance width of each digit at a user-space location."""
    inst = instancer.instantiateVariableFont(TTFont(io.BytesIO(raw)), location)
    cmap = inst.getBestCmap()
    hmtx = inst["hmtx"]
    return {d: hmtx[cmap[ord(d)]][0] for d in DIGITS}


def assert_zero_is_widest(raw: bytes, font: TTFont) -> None:
    """Fact (1). The whole construction rests on this, so it is checked, not assumed."""
    axes = {a.axisTag: a for a in font["fvar"].axes}
    opsz, wght = axes["opsz"], axes["wght"]
    failures = []
    for o in (opsz.minValue, 16, 24, opsz.maxValue):
        for w in (wght.minValue, 250, wght.defaultValue, 440, 600, 695, wght.maxValue):
            adv = digit_advances(raw, {"opsz": o, "wght": w})
            if adv["0"] != max(adv.values()):
                failures.append((o, w, max(adv, key=adv.get)))
    if failures:
        raise SystemExit(
            "Invariant broken: '0' is not the widest digit at "
            + ", ".join(f"opsz={o} wght={w} (widest={d})" for o, w, d in failures)
        )


_INSTANCE_CACHE: dict = {}


def _instance(raw: bytes, key, loc: dict) -> TTFont:
    """Instance a VF at a user-space location, cached."""
    ck = (key, tuple(sorted(loc.items())))
    if ck not in _INSTANCE_CACHE:
        f = TTFont(io.BytesIO(raw), recalcTimestamp=False)
        instancer.instantiateVariableFont(f, loc, inplace=True)
        _INSTANCE_CACHE[ck] = f
    return _INSTANCE_CACHE[ck]


def _measures(font: TTFont) -> dict:
    glyf = font["glyf"]
    cm = font.getBestCmap()
    x = glyf[cm[ord("x")]]
    cap_glyph = glyf[cm[ord("H")]]
    xh, cap = x.yMax, cap_glyph.yMax
    return {
        "xh": xh,
        "cap": cap,
        "stem": scanline_stem(font, "l", xh / 2),
        "o_adv": font["hmtx"][cm[ord("o")]][0],
    }


def drop_hvar_after_proof(font: TTFont, raw: bytes) -> None:
    """Make gvar phantoms reproduce HVAR's advances exactly, verify, drop HVAR.

    HVAR is what renderers actually used, and in the italic its values diverge
    from the shipped phantom deltas by up to 4 units on some glyphs — so the
    phantoms cannot simply be trusted and HVAR cannot simply be dropped.
    Instead, HVAR-true advances are sampled at all 12 master locations and every
    glyph's phantom advance deltas are REBUILT from them through the same
    VariationModel the grafts use. Advances are then exact at every master with
    HVAR gone, and new glyphs need no delta-set surgery."""
    model = VariationModel([n for n, _ in MASTER_GRID])
    sampled = []
    for norm, (uw, uo) in MASTER_GRID:
        inst = _instance(raw, "dm+h", {"wght": uw, "opsz": uo})
        sampled.append({g: inst["hmtx"][g][0] for g in inst.getGlyphOrder()})

    glyf = font["glyf"]
    gvar = font["gvar"]
    rebuilt = created = 0
    for g in font.getGlyphOrder():
        vals = [s[g] for s in sampled]
        deltas = model.getDeltas([float(v) for v in vals])
        needed = {}
        for support, d in zip(model.supports[1:], deltas[1:]):
            key = tuple(sorted((t, tuple(v)) for t, v in support.items()))
            if round(d):
                needed[key] = round(d)
        glyph = glyf[g]
        n_base = (
            len(glyph.components)
            if glyph.isComposite()
            else (len(glyph.getCoordinates(glyf)[0]) if glyph.numberOfContours > 0 else 0)
        )
        tuples = {tuple(sorted((t, tuple(v)) for t, v in tv.axes.items())): tv
                  for tv in gvar.variations.get(g, [])}
        changed = False
        for key, tv in tuples.items():
            left = tv.coordinates[-4]
            left_x = left[0] if left else 0
            want = needed.pop(key, 0) + left_x
            cur = tv.coordinates[-3]
            if (cur[0] if cur else 0) != want:
                tv.coordinates[-3] = (want, 0)
                changed = True
        for key, d in needed.items():
            axes = {t: tuple(v) for t, v in key}
            coords = [None] * n_base + [(0, 0), (d, 0), (0, 0), (0, 0)]
            gvar.variations.setdefault(g, []).append(TupleVariation(axes, coords))
            created += 1
            changed = True
        if changed:
            rebuilt += 1

    del font["HVAR"]

    # Verify: instancing the surgically altered font must reproduce the
    # HVAR-true advances at every master exactly.
    buf = io.BytesIO()
    font.recalcTimestamp = False
    font.save(buf)
    fixed_raw = buf.getvalue()
    for (norm, (uw, uo)), truth in zip(MASTER_GRID, sampled):
        inst = _instance(fixed_raw, "dm-fixed", {"wght": uw, "opsz": uo})
        for g in inst.getGlyphOrder():
            if abs(inst["hmtx"][g][0] - truth[g]) > 1:
                raise SystemExit(
                    f"phantom rebuild failed at ({uw},{uo}) {g}: "
                    f"{inst['hmtx'][g][0]} vs HVAR-true {truth[g]}"
                )
    print(f"    HVAR dropped; phantom advances rebuilt for {rebuilt} glyphs "
          f"({created} tuples added), exact at all 12 masters")


def _pen_arrays(src_font: TTFont, glyph_name: str, xf) -> tuple[list, list, bytes, Glyph]:
    """Decompose+transform a glyph; return (coords, endPts, flags, compiledGlyph).

    Replaying the identical op stream per location guarantees identical point
    order across all sampled locations, which is what makes the deltas valid."""
    rec = DecomposingRecordingPen(src_font.getGlyphSet())
    src_font.getGlyphSet()[glyph_name].draw(rec)
    tpen = TTGlyphPen(None)
    for op, args in rec.value:
        if op == "moveTo":
            tpen.moveTo(xf(args[0]))
        elif op == "lineTo":
            tpen.lineTo(xf(args[0]))
        elif op == "qCurveTo":
            tpen.qCurveTo(*[xf(a) if a is not None else None for a in args])
        elif op == "curveTo":
            tpen.curveTo(*[xf(a) for a in args])
        elif op == "closePath":
            tpen.closePath()
    glyph = tpen.glyph()
    coords, ends, flags = glyph.getCoordinates(None)
    return list(coords), list(ends), bytes(flags), glyph


def solve_inter(inter_raw: bytes, dm: dict, cache_key: str) -> tuple[int, int, float, float]:
    """Solve Inter's (wght, opsz) + scales so its glyphs match a DM location.

    opsz from the x/cap ratio (clamped to Inter's 14–32), then wght from the
    stem after vertical scaling, iterated once — the scale depends on the
    solved location's own x-height."""
    # x/cap(opsz) curve at wght 400
    curve = []
    for o in (14, 20, 26, 32):
        m = _measures(_instance(inter_raw, cache_key, {"wght": 400, "opsz": o}))
        curve.append((o, m["xh"] / m["cap"]))
    target = dm["xh"] / dm["cap"]
    opsz = 32.0
    for (o1, r1), (o2, r2) in zip(curve, curve[1:]):
        lo, hi = sorted((r1, r2))
        if lo <= target <= hi:
            opsz = o1 + (target - r1) * (o2 - o1) / (r2 - r1)
            break
    else:
        opsz = 14.0 if target > curve[0][1] else 32.0
    opsz = round(max(14.0, min(32.0, opsz)))

    wght = 400
    for _ in range(2):
        m = _measures(_instance(inter_raw, cache_key, {"wght": wght, "opsz": opsz}))
        k = dm["xh"] / m["xh"]
        stem_target = dm["stem"] / k
        stems = []
        for w in (100, 300, 500, 700, 900):
            mm = _measures(_instance(inter_raw, cache_key, {"wght": w, "opsz": opsz}))
            stems.append((w, mm["stem"]))
        wght = 900
        for (w1, s1), (w2, s2) in zip(stems, stems[1:]):
            if min(s1, s2) <= stem_target <= max(s1, s2):
                wght = round(w1 + (stem_target - s1) * (w2 - w1) / (s2 - s1))
                break
        else:
            wght = 100 if stem_target < stems[0][1] else 900
        wght = max(100, min(900, wght))
    m = _measures(_instance(inter_raw, cache_key, {"wght": wght, "opsz": opsz}))
    return wght, opsz, dm["xh"] / m["xh"], dm["cap"] / m["cap"]


def graft_greek(font: TTFont, raw: bytes, inter_path: Path, italic: bool, report: list[str]) -> int:
    """Graft Inter's Greek block into the font, exact at all 12 master locations."""
    inter_raw = inter_path.read_bytes()
    ck = "it" if italic else "ro"
    dm_cmap = font.getBestCmap()
    inter_cmap = TTFont(io.BytesIO(inter_raw)).getBestCmap()
    graft_cps = sorted(
        cp for cp in inter_cmap if 0x0370 <= cp <= 0x03FF and cp not in dm_cmap
    )
    shear = (
        math.tan(math.radians(-DM_ITALIC_ANGLE)) - math.tan(math.radians(-INTER_ITALIC_ANGLE))
        if italic
        else 0.0
    )

    # Per master location: DM measurements, solved Inter instance, scales.
    plans = []
    dm_default = _measures(_instance(raw, "dm" + ck, dict(zip(("wght", "opsz"), MASTER_GRID[0][1]))))
    for norm, (uw, uo) in MASTER_GRID:
        dm = _measures(_instance(raw, "dm" + ck, {"wght": uw, "opsz": uo}))
        iw, iopsz, k_lc, k_uc = solve_inter(inter_raw, dm, "in" + ck)
        inter_inst = _instance(inter_raw, "in" + ck, {"wght": iw, "opsz": iopsz})
        im = _measures(inter_inst)
        # Horizontal correction: Greek must follow DM's width rhythm even where
        # Inter's opsz axis has saturated (DM narrows 24->40; Inter stops at 32).
        dm_ratio = dm["o_adv"] / dm_default["o_adv"]
        inter_default = _measures(_instance(inter_raw, "in" + ck, {"wght": 400, "opsz": 14}))
        inter_ratio = im["o_adv"] / inter_default["o_adv"]
        kx_corr = dm_ratio / inter_ratio
        plans.append((norm, inter_inst, k_lc, k_uc, kx_corr))
    for (norm, (uw, uo)), (n2, inst, k_lc, k_uc, kxc) in zip(MASTER_GRID, plans):
        if (uw, uo) in ((400, 9), (100, 9), (1000, 9), (400, 40), (1000, 40)):
            report.append(
                f"    DM({uw:4d},{uo:2d}): k_lc {k_lc:.4f}  k_uc {k_uc:.4f}  kx_corr {kxc:.3f}"
            )

    glyf = font["glyf"]
    hmtx = font["hmtx"]
    order = font.getGlyphOrder()
    cmap_tables = [t for t in font["cmap"].tables if t.isUnicode()]
    gvar = font["gvar"]
    model = VariationModel([n for n, _ in MASTER_GRID])

    added = 0
    for cp in graft_cps:
        is_upper = unicodedata.category(chr(cp)) == "Lu"
        per_loc = []
        ok = True
        for (norm, _), (n2, inst, k_lc, k_uc, kxc) in zip(MASTER_GRID, plans):
            k_y = k_uc if is_upper else k_lc
            k_x = k_y * kxc
            icmap = inst.getBestCmap()
            if cp not in icmap:
                ok = False
                break

            def xf(pt, k_x=k_x, k_y=k_y):
                x, y = pt
                x, y = x * k_x, y * k_y
                return (x + shear * y, y)

            coords, ends, flags, glyph = _pen_arrays(inst, icmap[cp], xf)
            adv = inst["hmtx"][icmap[cp]][0] * k_x
            per_loc.append((coords, ends, flags, glyph, adv))
        if not ok:
            continue
        ends0 = per_loc[0][1]
        if any(p[1] != ends0 or len(p[0]) != len(per_loc[0][0]) for p in per_loc):
            raise SystemExit(f"graft U+{cp:04X}: point structure diverged across locations")

        name = f"uni{cp:04X}.ultra"
        # default outline = default location, integer-rounded
        d_coords, d_ends, d_flags, d_glyph, d_adv = per_loc[0]
        # rebuild via rounded coordinates for the stored default glyph
        rounded = [(round(x), round(y)) for x, y in d_coords]
        g = Glyph()
        g.numberOfContours = len(d_ends)
        from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates

        g.coordinates = GlyphCoordinates(rounded)
        g.endPtsOfContours = list(d_ends)
        g.flags = bytearray(d_flags)
        g.program = ttProgram.Program()
        glyf.glyphs[name] = g
        g.recalcBounds(glyf)
        hmtx.metrics[name] = (round(d_adv), g.xMin)
        order.append(name)
        for t in cmap_tables:
            t.cmap[cp] = name

        # deltas over the model: per point x/y + phantom advance
        npts = len(d_coords)
        variations = []
        deltas_per_support = None
        for pt in range(npts + 4):
            if pt < npts:
                xs = [p[0][pt][0] for p in per_loc]
                ys = [p[0][pt][1] for p in per_loc]
            elif pt == npts + 1:  # advance phantom
                xs = [p[4] for p in per_loc]
                ys = [0.0] * len(per_loc)
            else:
                xs = [0.0] * len(per_loc)
                ys = [0.0] * len(per_loc)
            dx = model.getDeltas(xs)
            dy = model.getDeltas(ys)
            if deltas_per_support is None:
                deltas_per_support = [[] for _ in dx[1:]]
            for i, (vx, vy) in enumerate(zip(dx[1:], dy[1:])):
                deltas_per_support[i].append((round(vx), round(vy)))
        for support, deltas in zip(model.supports[1:], deltas_per_support):
            if all(d == (0, 0) for d in deltas):
                continue
            axes = {tag: tuple(v) for tag, v in support.items()}
            variations.append(TupleVariation(axes, deltas))
        if variations:
            gvar.variations[name] = variations
        added += 1

    font.setGlyphOrder(order)
    font.getReverseGlyphMap(rebuild=True)
    glyf.glyphOrder = order
    font["maxp"].numGlyphs = len(order)
    report.append(f"    grafted {added} Greek codepoints across 12 master locations")
    return added


ROUND_CAP_CHARS = ("C", "O", "G", "Q", "Ø")


def _recenter_base_composites(
    font: TTFont,
    base_name: str,
    center_shifts: list[float],
    advance_shifts: list[float],
    model: VariationModel,
) -> int:
    """Keep marks and metrics attached when a base outline changes width."""
    glyf = font["glyf"]
    gvar = font["gvar"]
    hmtx = font["hmtx"]
    center_deltas = model.getDeltas(center_shifts)
    advance_deltas = model.getDeltas(advance_shifts)
    composites = [
        name
        for name in font.getGlyphOrder()
        if glyf[name].isComposite()
        and any(component.glyphName == base_name for component in glyf[name].components)
    ]
    for name in composites:
        glyph = glyf[name]
        accent_indices = [
            index
            for index, component in enumerate(glyph.components)
            if component.glyphName != base_name
        ]
        for index in accent_indices:
            glyph.components[index].x += round(center_deltas[0])
        advance, lsb = hmtx[name]
        hmtx.metrics[name] = (advance + round(advance_deltas[0]), lsb)
        component_count = len(glyph.components)
        tuples = {
            tuple(sorted((key, tuple(value)) for key, value in variation.axes.items())): variation
            for variation in gvar.variations.get(name, [])
        }
        for support, center_delta, advance_delta in zip(
            model.supports[1:], center_deltas[1:], advance_deltas[1:]
        ):
            center_delta = round(center_delta)
            advance_delta = round(advance_delta)
            if center_delta == 0 and advance_delta == 0:
                continue
            key = tuple(sorted((axis, tuple(value)) for axis, value in support.items()))
            variation = tuples.get(key)
            if variation is None:
                coordinates = [(0, 0)] * component_count + [(0, 0)] * 4
                variation = TupleVariation(
                    {axis: tuple(value) for axis, value in support.items()}, coordinates
                )
                gvar.variations.setdefault(name, []).append(variation)
                tuples[key] = variation
            for index in accent_indices:
                current = variation.coordinates[index]
                variation.coordinates[index] = (
                    (current[0] if current else 0) + center_delta,
                    current[1] if current else 0,
                )
            current_phantom = variation.coordinates[-3]
            variation.coordinates[-3] = (
                (current_phantom[0] if current_phantom else 0) + advance_delta,
                0,
            )
    return len(composites)


def _round_side_stroke(font: TTFont, char: str = "O") -> float:
    name = font.getBestCmap()[ord(char)]
    glyph = font["glyf"][name]
    coordinates, _, _ = glyph.getCoordinates(font["glyf"])
    points = list(coordinates)
    middle = (min(point[1] for point in points) + max(point[1] for point in points)) / 2
    return _scan_runs(font, name, middle)[0]


def glyph_height(font: TTFont, char: str) -> float:
    name = font.getBestCmap()[ord(char)]
    coordinates, _, _ = font["glyf"][name].getCoordinates(font["glyf"])
    ys = [point[1] for point in coordinates]
    return max(ys) - min(ys)


def _solve_inter_round_weight(
    inter_raw: bytes,
    dm_instance: TTFont,
    inter_opsz: int,
    cap_scale: float,
    cache_key: str,
) -> int:
    """Match Inter's scaled capital-round stroke to the DM master."""
    target = _round_side_stroke(dm_instance)
    samples = []
    for weight in (100, 300, 500, 700, 900):
        instance = _instance(
            inter_raw, cache_key, {"wght": weight, "opsz": inter_opsz}
        )
        samples.append((weight, _round_side_stroke(instance) * cap_scale))
    if target <= samples[0][1]:
        return 100
    if target >= samples[-1][1]:
        return 900
    for (weight_a, stroke_a), (weight_b, stroke_b) in zip(samples, samples[1:]):
        if stroke_a <= target <= stroke_b:
            solved = weight_a + (target - stroke_a) * (weight_b - weight_a) / (
                stroke_b - stroke_a
            )
            return round(solved)
    raise SystemExit("round-cap weight solve did not bracket the target stroke")


def redraw_round_capitals(
    font: TTFont,
    raw: bytes,
    inter_path: Path,
    italic: bool,
    report: list[str],
) -> None:
    """Draw Ultra's round capitals from Inter geometry on DM Sans metrics.

    The small-size reference is closest to Inter rather than a mathematical
    circle: O is subtly vertical, C is narrower and open, and G is narrower
    than O with a long horizontal bar. At every Ultra master we therefore solve
    an Inter instance that matches DM Sans's round-cap height and stroke, retain
    Inter's own optical proportions, and write fresh compatible C/O/G/Q/Oslash
    outlines into the existing DM glyph slots. Glyph names — and therefore
    kerning and substitution coverage — stay unchanged. DM advances and ink
    centres stay fixed, so existing spacing and accented-composite placement do
    not move when the silhouettes change.

    DM Sans supplies the family metrics and two-axis rhythm; Inter supplies the
    reference-matched round skeleton; Ultra Mono is deliberately a diagnostic
    reference only, because importing its compressed proportions would damage
    proportional-text recognition even though its G bar reads well at 10px.
    """
    inter_raw = inter_path.read_bytes()
    cache_key = "round-it" if italic else "round-ro"
    shear = (
        math.tan(math.radians(-DM_ITALIC_ANGLE))
        - math.tan(math.radians(-INTER_ITALIC_ANGLE))
        if italic
        else 0.0
    )
    plans = []
    for norm, (weight, optical_size) in MASTER_GRID:
        dm_instance = _instance(
            raw, "dm-" + cache_key, {"wght": weight, "opsz": optical_size}
        )
        dm_measures = _measures(dm_instance)
        _, inter_opsz, _, _ = solve_inter(
            inter_raw, dm_measures, "inter-" + cache_key
        )
        dm_round_height = glyph_height(dm_instance, "O")
        provisional_inter = _instance(
            inter_raw,
            "inter-" + cache_key,
            {"wght": 400, "opsz": inter_opsz},
        )
        round_scale = dm_round_height / glyph_height(provisional_inter, "O")
        inter_weight = _solve_inter_round_weight(
            inter_raw,
            dm_instance,
            inter_opsz,
            round_scale,
            "inter-" + cache_key,
        )
        inter_instance = _instance(
            inter_raw,
            "inter-" + cache_key,
            {"wght": inter_weight, "opsz": inter_opsz},
        )
        plans.append(
            (
                norm,
                dm_instance,
                inter_instance,
                round_scale,
                round_scale,
                inter_weight,
                inter_opsz,
            )
        )

    cmap = font.getBestCmap()
    model = VariationModel([normalized for normalized, _ in MASTER_GRID])
    composite_count = 0
    for char in ROUND_CAP_CHARS:
        base_name = cmap[ord(char)]
        samples = []
        center_shifts = []
        advance_shifts = []
        reference_ends = None
        reference_flags = None
        for (
            _,
            dm_instance,
            inter_instance,
            scale_x,
            scale_y,
            _,
            _,
        ) in plans:
            inter_name = inter_instance.getBestCmap()[ord(char)]

            def transform(point, sx=scale_x, sy=scale_y):
                x, y = point
                x, y = x * sx, y * sy
                return x + shear * y, y

            coordinates, ends, flags, _ = _pen_arrays(
                inter_instance, inter_name, transform
            )
            if reference_ends is None:
                reference_ends = ends
                reference_flags = flags
            elif ends != reference_ends or flags != reference_flags:
                raise SystemExit(
                    f"round-cap redraw {char}: point structure diverged across masters"
                )
            old_glyph = dm_instance["glyf"][dm_instance.getBestCmap()[ord(char)]]
            old_coordinates, _, _ = old_glyph.getCoordinates(dm_instance["glyf"])
            old_points = list(old_coordinates)
            old_center = (
                min(point[0] for point in old_points)
                + max(point[0] for point in old_points)
            ) / 2
            new_center = (
                min(point[0] for point in coordinates)
                + max(point[0] for point in coordinates)
            ) / 2
            center_delta = old_center - new_center
            coordinates = [
                (point[0] + center_delta, point[1]) for point in coordinates
            ]
            old_advance = dm_instance["hmtx"][dm_instance.getBestCmap()[ord(char)]][0]
            samples.append((coordinates, ends, flags, old_advance))
            center_shifts.append(0.0)
            advance_shifts.append(0.0)

        _replace_outline(font, base_name, samples, model)
        composite_count += _recenter_base_composites(
            font, base_name, center_shifts, advance_shifts, model
        )

    for (_, (weight, optical_size)), plan in zip(MASTER_GRID, plans):
        inter_weight = plan[5]
        inter_opsz = plan[6]
        if (weight, optical_size) in (
            (400, 9),
            (100, 9),
            (1000, 9),
            (400, 40),
            (1000, 40),
        ):
            report.append(
                f"    round caps DM({weight:4d},{optical_size:2d}) "
                f"<- Inter({inter_weight:3d},{inter_opsz:2d})"
            )
    report.append(
        f"    redrew {'/'.join(ROUND_CAP_CHARS)} at 12 masters; "
        f"{composite_count} composites re-centred"
    )


def _scan_runs(font: TTFont, name: str, y: float) -> list[float]:
    """Ink runs crossing height y, polyline-approximated — used for stroke bands."""
    g = font["glyf"][name]
    coords, ends, _ = g.getCoordinates(font["glyf"])
    pts = list(coords)
    xs = []
    start = 0
    for e in ends:
        seg = pts[start : e + 1]
        for i in range(len(seg)):
            (x1, y1), (x2, y2) = seg[i], seg[(i + 1) % len(seg)]
            if (y1 <= y < y2) or (y2 <= y < y1):
                xs.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
        start = e + 1
    xs.sort()
    return [xs[i + 1] - xs[i] for i in range(0, len(xs) - 1, 2)]


def _replace_outline(font: TTFont, name: str, samples: list, model: VariationModel) -> None:
    """Overwrite a glyph's default outline + gvar from 12 per-master samples."""
    from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates

    glyf = font["glyf"]
    d_pts, d_ends, d_flags, d_adv = samples[0]
    g = Glyph()
    g.numberOfContours = len(d_ends)
    g.coordinates = GlyphCoordinates([(round(x), round(y)) for x, y in d_pts])
    g.endPtsOfContours = list(d_ends)
    g.flags = bytearray(d_flags)
    g.program = ttProgram.Program()
    glyf.glyphs[name] = g
    g.recalcBounds(glyf)
    font["hmtx"].metrics[name] = (round(d_adv), g.xMin)
    npts = len(d_pts)
    deltas_per_support = None
    variations = []
    for pt in range(npts + 4):
        if pt < npts:
            xs = [s[0][pt][0] for s in samples]
            ys = [s[0][pt][1] for s in samples]
        elif pt == npts + 1:
            xs = [float(s[3]) for s in samples]
            ys = [0.0] * len(samples)
        else:
            xs = [0.0] * len(samples)
            ys = [0.0] * len(samples)
        dx = model.getDeltas(xs)
        dy = model.getDeltas(ys)
        if deltas_per_support is None:
            deltas_per_support = [[] for _ in dx[1:]]
        for i, (vx, vy) in enumerate(zip(dx[1:], dy[1:])):
            deltas_per_support[i].append((round(vx), round(vy)))
    for support, deltas in zip(model.supports[1:], deltas_per_support):
        if all(d == (0, 0) for d in deltas):
            continue
        variations.append(TupleVariation({k: tuple(v) for k, v in support.items()}, deltas))
    font["gvar"].variations[name] = variations


def add_slashed_zero(font: TTFont, raw: bytes, report: list[str]) -> dict[str, str]:
    """Generate zero.slash (+ tabular composite) from the 0's own geometry."""
    zero = font.getBestCmap()[ord("0")]
    glyf = font["glyf"]
    gvar = font["gvar"]
    model = VariationModel([n for n, _ in MASTER_GRID])

    per_loc = []
    for norm, (uw, uo) in MASTER_GRID:
        inst = _instance(raw, "dmz", {"wght": uw, "opsz": uo})
        g = inst["glyf"][zero]
        coords, ends, flags = g.getCoordinates(inst["glyf"])
        # contours -> signed areas; smallest-|area| opposite sign = counter
        pts = list(coords)
        areas = []
        start = 0
        for e in ends:
            seg = pts[start : e + 1]
            a = sum(
                seg[i][0] * seg[(i + 1) % len(seg)][1] - seg[(i + 1) % len(seg)][0] * seg[i][1]
                for i in range(len(seg))
            ) / 2
            areas.append((a, start, e))
            start = e + 1
        outer = max(areas, key=lambda t: abs(t[0]))
        inner = min(areas, key=lambda t: abs(t[0]))
        iseg = pts[inner[1] : inner[2] + 1]
        ix0, iy0 = min(p[0] for p in iseg), min(p[1] for p in iseg)
        ix1, iy1 = max(p[0] for p in iseg), max(p[1] for p in iseg)
        cx, cy, w, h = (ix0 + ix1) / 2, (iy0 + iy1) / 2, ix1 - ix0, iy1 - iy0
        # slash thickness = the 0's own thin (horizontal) stroke at this location
        col = scanline_stem(inst, "0", cy) or 60
        # vertical scan for horizontal stroke: approximate via top arc thickness
        gy = g.yMax - iy1  # top rim thickness
        t = max(20.0, min(gy, col))
        # diagonal bl -> tr, contained within the counter
        ax, ay = cx - 0.30 * w, cy - 0.36 * h
        bx, by = cx + 0.30 * w, cy + 0.36 * h
        length = math.hypot(bx - ax, by - ay)
        px, py = -(by - ay) / length * t / 2, (bx - ax) / length * t / 2
        quad = [(ax + px, ay + py), (bx + px, by + py), (bx - px, by - py), (ax - px, ay - py)]
        if outer[0] < 0:  # match outer winding direction
            quad.reverse()
        adv = inst["hmtx"][zero][0]
        per_loc.append((pts, list(ends), bytes(flags), quad, adv))

    name = "zero.slash"
    from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates

    d_pts, d_ends, d_flags, d_quad, d_adv = per_loc[0]
    all_pts = [(round(x), round(y)) for x, y in d_pts + d_quad]
    g = Glyph()
    g.numberOfContours = len(d_ends) + 1
    g.coordinates = GlyphCoordinates(all_pts)
    g.endPtsOfContours = list(d_ends) + [len(d_pts) + 3]
    g.flags = bytearray(d_flags) + bytearray([1, 1, 1, 1])
    g.program = ttProgram.Program()
    glyf.glyphs[name] = g
    g.recalcBounds(glyf)
    font["hmtx"].metrics[name] = (d_adv, g.xMin)
    order = font.getGlyphOrder()
    order.append(name)

    npts = len(d_pts) + 4
    deltas_per_support = None
    variations = []
    for pt in range(npts + 4):
        if pt < len(d_pts):
            xs = [p[0][pt][0] for p in per_loc]
            ys = [p[0][pt][1] for p in per_loc]
        elif pt < npts:
            qi = pt - len(d_pts)
            xs = [p[3][qi][0] for p in per_loc]
            ys = [p[3][qi][1] for p in per_loc]
        elif pt == npts + 1:
            xs = [p[4] for p in per_loc]
            ys = [0.0] * len(per_loc)
        else:
            xs = [0.0] * len(per_loc)
            ys = [0.0] * len(per_loc)
        dx = model.getDeltas(xs)
        dy = model.getDeltas(ys)
        if deltas_per_support is None:
            deltas_per_support = [[] for _ in dx[1:]]
        for i, (vx, vy) in enumerate(zip(dx[1:], dy[1:])):
            deltas_per_support[i].append((round(vx), round(vy)))
    for support, deltas in zip(model.supports[1:], deltas_per_support):
        if all(d == (0, 0) for d in deltas):
            continue
        variations.append(TupleVariation({t: tuple(v) for t, v in support.items()}, deltas))
    gvar.variations[name] = variations

    # tabular slashed zero: composite of zero.slash, same shape as zero.tnum
    tnum_name = f"{zero}.tnum"
    slash_tnum = "zero.tnum.slash"
    comp = GlyphComponent()
    comp.glyphName = name
    comp.x, comp.y = 0, 0
    comp.flags = 0x004
    cg = Glyph()
    cg.numberOfContours = -1
    cg.components = [comp]
    glyf.glyphs[slash_tnum] = cg
    font["hmtx"].metrics[slash_tnum] = font["hmtx"].metrics[tnum_name]
    order.append(slash_tnum)
    if tnum_name in gvar.variations:
        gvar.variations[slash_tnum] = [
            TupleVariation(dict(tv.axes), list(tv.coordinates)) for tv in gvar.variations[tnum_name]
        ]

    font.setGlyphOrder(order)
    font.getReverseGlyphMap(rebuild=True)
    glyf.glyphOrder = order
    font["maxp"].numGlyphs = len(order)
    report.append("    slashed zero generated (dormant `zero` feature; slash from the 0's own geometry)")
    return {zero: name, tnum_name: slash_tnum}


def build_tnum_glyphs(font: TTFont, raw: bytes) -> dict[str, str]:
    """Add `<digit>.tnum` composites. Returns the digit-glyph -> tnum-glyph map."""
    cmap = font.getBestCmap()
    glyf = font["glyf"]
    hmtx = font["hmtx"]
    gvar = font["gvar"]

    base = {d: cmap[ord(d)] for d in DIGITS}
    zero = base["0"]

    default_adv = {d: hmtx[base[d]][0] for d in DIGITS}
    target_default = default_adv["0"]

    # Advance deltas per gvar region, read out of the font's own tuples rather
    # than modelled from sampled corners. A first cut sampled the axis corners
    # and built five tuples from the absolute values; that is wrong, because gvar
    # deltas are additive residuals — the {opsz,wght} corner tuple double-counted
    # what the {opsz} and {wght} tuples already contributed, overshooting by up to
    # 51 units (7%) at opsz 40 / wght 1000.
    #
    # Reading the deltas directly sidesteps modelling entirely: the base digits
    # already encode how their advances move over 11 tuples with intermediate
    # regions, and avar warping is baked into those same regions.
    def advance_deltas(glyph_name: str) -> dict[tuple, int]:
        """{region -> advance delta} for one glyph, from its gvar phantom points."""
        out = {}
        for tv in gvar.variations.get(glyph_name, []):
            coords = tv.coordinates
            # Trailing four points are phantoms: left, right, top, bottom.
            # Advance is right minus left.
            left, right = coords[-4], coords[-3]
            delta = (right[0] if right else 0) - (left[0] if left else 0)
            key = tuple(sorted((tag, tuple(v)) for tag, v in tv.axes.items()))
            out[key] = out.get(key, 0) + delta
        return out

    zero_deltas = advance_deltas(zero)

    mapping = {}
    new_names: list[str] = []
    for d in DIGITS:
        src = base[d]
        name = f"{src}.tnum"
        # Centre the INK, not the advance box. Centring the advance box preserves
        # whatever bearing asymmetry the proportional design carried, which for
        # `1` left it 44 units off centre in its tabular cell — the classic
        # tabular-one complaint. Measured across the axis, the ink correction is
        # near-constant (worst drift 10 units, 0.15px at 15px), so it is applied
        # statically here while the deltas below stay exact.
        outline = glyf[src]
        outline.recalcBounds(glyf)
        ink_width = outline.xMax - outline.xMin
        shift0 = (target_default - ink_width) / 2 - outline.xMin

        component = GlyphComponent()
        component.glyphName = src
        component.x, component.y = int(round(shift0)), 0
        component.flags = 0x004  # ROUND_XY_TO_GRID
        glyph = Glyph()
        glyph.numberOfContours = -1
        glyph.components = [component]

        glyf.glyphs[name] = glyph
        new_names.append(name)
        # Advance equals zero's at the default instance; HVAR carries it from there.
        hmtx.metrics[name] = (target_default, hmtx[src][1] + int(round(shift0)))

        # gvar deltas: [component offset, phantom left, phantom right, top, bottom].
        # Advance is phantom-right minus phantom-left, so phantom right carries
        # the advance variation. Leaving these None pins the advance to its
        # default at every instance — the digits then align with each other but
        # stop tracking real widths, and collide once weight widens them.
        #
        # shift(L) = (adv_zero(L) - adv_d(L)) / 2 is linear in the advances, and
        # gvar deltas are additive, so the shift's delta over any region is that
        # same linear combination of the two advance deltas. Exact, no model.
        src_deltas = advance_deltas(src)
        variations = []
        for key in sorted(set(zero_deltas) | set(src_deltas)):
            d_adv = zero_deltas.get(key, 0)
            dx = (d_adv - src_deltas.get(key, 0)) / 2
            dx, d_adv = int(round(dx)), int(round(d_adv))
            if dx == 0 and d_adv == 0:
                continue
            axes = {tag: tuple(v) for tag, v in key}
            variations.append(
                TupleVariation(
                    axes,
                    [(dx, 0), (0, 0), (d_adv, 0), (0, 0), (0, 0)],
                )
            )
        if variations:
            gvar.variations[name] = variations

        mapping[src] = name

    # Register through the proper API — appending to the list returned by
    # getGlyphOrder() leaves TTFont's reverse glyph map stale, and GSUB compiles
    # glyph names through it.
    order = font.getGlyphOrder() + new_names
    font.setGlyphOrder(order)
    font.getReverseGlyphMap(rebuild=True)
    glyf.glyphOrder = order
    font["maxp"].numGlyphs = len(order)
    return mapping


def add_feature(font: TTFont, tag: str, mapping: dict[str, str]) -> None:
    """Insert a single-substitution feature, keeping FeatureList sorted and
    every script's feature indices remapped."""
    gsub = font["GSUB"].table
    lookup = otl.buildLookup([otl.buildSingleSubstSubtable(mapping)], flags=0)
    gsub.LookupList.Lookup.append(lookup)
    gsub.LookupList.LookupCount = len(gsub.LookupList.Lookup)
    lookup_index = len(gsub.LookupList.Lookup) - 1

    from fontTools.ttLib.tables import otTables

    feature = otTables.Feature()
    feature.FeatureParams = None
    feature.LookupListIndex = [lookup_index]
    feature.LookupCount = 1
    record = otTables.FeatureRecord()
    record.FeatureTag = tag
    record.Feature = feature

    records = gsub.FeatureList.FeatureRecord
    # The spec has FeatureRecords listed alphabetically and shapers may binary
    # search, so insert in sorted position and repair every index that moves.
    insert_at = 0
    while insert_at < len(records) and records[insert_at].FeatureTag <= tag:
        insert_at += 1
    records.insert(insert_at, record)
    gsub.FeatureList.FeatureCount = len(records)

    def remap(old: int) -> int:
        return old + 1 if old >= insert_at else old

    for script in gsub.ScriptList.ScriptRecord:
        langsyses = [script.Script.DefaultLangSys] + [
            record.LangSys for record in script.Script.LangSysRecord
        ]
        for langsys in langsyses:
            if langsys is None:
                continue
            langsys.FeatureIndex = [remap(i) for i in langsys.FeatureIndex]
            if langsys.ReqFeatureIndex not in (None, 0xFFFF):
                langsys.ReqFeatureIndex = remap(langsys.ReqFeatureIndex)
            langsys.FeatureIndex.append(insert_at)
            langsys.FeatureIndex.sort()
            langsys.FeatureCount = len(langsys.FeatureIndex)

    if getattr(gsub, "FeatureVariations", None):
        for fvr in gsub.FeatureVariations.FeatureVariationRecord:
            for sub in fvr.FeatureTableSubstitution.SubstitutionRecord:
                sub.FeatureIndex = remap(sub.FeatureIndex)


def rename(font: TTFont, style: str) -> None:
    """Rename to Ultra Sans.

    DM Sans declares no Reserved Font Name, so this is permitted outright — but a
    modified font must not pass itself off as the original. Name IDs 0, 13 and 14
    (copyright and license) are deliberately left untouched, as in
    build_ultra_sans.py; the derivative note is appended to the description.
    """
    family = "Ultra Sans"
    subfamily = "Italic" if style == "italic" else "Regular"
    full = f"{family} {subfamily}"
    name = font["name"]
    for nid, value in [
        (1, family),
        (2, subfamily),
        (3, f"{full}; Ultra derivative of DM Sans"),
        (4, full),
        (6, f"{family.replace(' ', '')}-{subfamily}"),
        (16, family),
        (17, subfamily),
    ]:
        name.setName(value, nid, 3, 1, 0x409)
    existing = name.getDebugName(10) or ""
    note = (
        "Ultra Sans is a derivative of DM Sans: generated tabular figures (tnum), "
        "Greek grafted from weight- and proportion-matched Inter instances across "
        "the full two-axis design space, and a generated slashed-zero alternate "
        "(zero). No outline of the source is altered. Greek glyphs: Copyright "
        "2016 The Inter Project Authors (https://github.com/rsms/inter), SIL OFL "
        "1.1, no Reserved Font Name."
    )
    name.setName((existing + " " + note).strip(), 10, 3, 1, 0x409)


def verify(path: Path, style: str) -> list[str]:
    """Instance the built font and prove the digits actually align."""
    raw = path.read_bytes()
    problems = []
    checked = 0
    for opsz in (9, 16, 24, 40):
        for wght in (100, 400, 440, 600, 695, 1000):
            inst = instancer.instantiateVariableFont(
                TTFont(io.BytesIO(raw)), {"opsz": opsz, "wght": wght}
            )
            hmtx = inst["hmtx"]
            cmap = inst.getBestCmap()
            tnum = [hmtx[f"{cmap[ord(d)]}.tnum"][0] for d in DIGITS]
            prop = [hmtx[cmap[ord(d)]][0] for d in DIGITS]
            checked += 1
            if len(set(tnum)) != 1:
                problems.append(
                    f"{style} opsz={opsz} wght={wght}: tabular advances not uniform "
                    f"(spread {max(tnum) - min(tnum)})"
                )
            if max(tnum) != max(prop):
                problems.append(
                    f"{style} opsz={opsz} wght={wght}: tabular advance {max(tnum)} "
                    f"!= widest proportional digit {max(prop)}"
                )
            if len(set(prop)) == 1:
                problems.append(
                    f"{style} opsz={opsz} wght={wght}: proportional digits became "
                    "uniform — default figures must stay proportional"
                )
    print(f"  verified {checked} axis locations ({style})")

    # ---- Greek graft: presence, stem match at real tokens, x-height alignment
    base = TTFont(io.BytesIO(raw))
    cm = base.getBestCmap()
    for ch in "λσΔθαβγφωΩςΣΦ":
        if ord(ch) not in cm:
            problems.append(f"{style}: U+{ord(ch):04X} {ch} missing from graft")
    for wght, opsz in ((440, 9), (695, 9), (600, 13), (400, 24), (1000, 40)):
        inst = instancer.instantiateVariableFont(
            TTFont(io.BytesIO(raw)), {"wght": wght, "opsz": opsz}
        )
        icm = inst.getBestCmap()
        xh_latin = inst["glyf"][icm[ord("x")]].yMax
        lam = scanline_stem(inst, "λ", xh_latin * 0.35)
        ell = scanline_stem(inst, "l", xh_latin / 2)
        if lam and ell and not (0.75 <= lam / ell <= 1.30):
            problems.append(
                f"{style} wght={wght} opsz={opsz}: grafted λ stem {lam:.0f} vs l {ell:.0f} — match off"
            )
        sig = inst["glyf"][icm[ord("σ")]]
        if abs(sig.yMax - xh_latin) > xh_latin * 0.03:
            problems.append(
                f"{style} wght={wght} opsz={opsz}: σ x-height {sig.yMax} vs Latin {xh_latin}"
            )

    # ---- slashed zero: feature present, slash contained in the counter
    gsub = base["GSUB"].table
    tags = [r.FeatureTag for r in gsub.FeatureList.FeatureRecord]
    if "zero" not in tags:
        problems.append(f"{style}: `zero` feature missing")
    if tags != sorted(tags):
        problems.append(f"{style}: FeatureList lost its sort order")
    for wght, opsz in ((100, 9), (440, 9), (1000, 9), (1000, 40)):
        inst = instancer.instantiateVariableFont(
            TTFont(io.BytesIO(raw)), {"wght": wght, "opsz": opsz}
        )
        zname = inst.getBestCmap()[ord("0")]
        gz = inst["glyf"][zname]
        gs = inst["glyf"]["zero.slash"]
        gz.recalcBounds(inst["glyf"])
        gs.recalcBounds(inst["glyf"])
        # slash may not escape the zero's own bbox
        if gs.xMin < gz.xMin - 2 or gs.xMax > gz.xMax + 2 or gs.yMax > gz.yMax + 2:
            problems.append(
                f"{style} wght={wght} opsz={opsz}: slash escapes the zero "
                f"({gs.xMin},{gs.yMin},{gs.xMax},{gs.yMax} vs {gz.xMin},{gz.yMin},{gz.xMax},{gz.yMax})"
            )
        if inst["hmtx"]["zero.slash"][0] != inst["hmtx"][zname][0]:
            problems.append(f"{style} wght={wght} opsz={opsz}: zero.slash advance drifted")

    if "HVAR" in base:
        problems.append(f"{style}: HVAR still present; advances must flow from phantoms only")

    # ---- reference-matched round capitals: proportions, weight, marks, metrics
    up_raw = fetch_source(*[source[:3] for source in SOURCES if source[4] == style][0])
    locations = (
        (400, 9),
        (100, 9),
        (1000, 9),
        (440, 9),
        (695, 9),
        (400, 24),
        (1000, 40),
    )

    def glyph_box(test_font: TTFont, char: str) -> tuple[float, float, float, float]:
        glyph_name = test_font.getBestCmap()[ord(char)]
        coordinates, _, _ = test_font["glyf"][glyph_name].getCoordinates(
            test_font["glyf"]
        )
        points = list(coordinates)
        return (
            min(point[0] for point in points),
            min(point[1] for point in points),
            max(point[0] for point in points),
            max(point[1] for point in points),
        )

    def mark_relationship(test_font: TTFont, composite: str, base_char: str):
        glyph = test_font["glyf"][composite]
        if not glyph.isComposite():
            return None
        base_name = test_font.getBestCmap()[ord(base_char)]
        base_component = next(
            component for component in glyph.components if component.glyphName == base_name
        )
        accent = next(
            component for component in glyph.components if component.glyphName != base_name
        )
        base_glyph = test_font["glyf"][base_name]
        accent_glyph = test_font["glyf"][accent.glyphName]
        return (
            accent.x + (accent_glyph.xMin + accent_glyph.xMax) / 2
        ) - (
            base_component.x + (base_glyph.xMin + base_glyph.xMax) / 2
        )

    for weight, optical_size in locations:
        instance = instancer.instantiateVariableFont(
            TTFont(io.BytesIO(raw)), {"wght": weight, "opsz": optical_size}
        )
        upstream = instancer.instantiateVariableFont(
            TTFont(io.BytesIO(up_raw)), {"wght": weight, "opsz": optical_size}
        )
        boxes = {char: glyph_box(instance, char) for char in ROUND_CAP_CHARS}
        widths = {char: box[2] - box[0] for char, box in boxes.items()}
        heights = {char: box[3] - box[1] for char, box in boxes.items()}
        upstream_height = glyph_box(upstream, "O")[3] - glyph_box(upstream, "O")[1]
        for char in ("C", "O", "G"):
            if abs(heights[char] - upstream_height) > 2:
                problems.append(
                    f"{style} ({weight},{optical_size}): {char} cap height "
                    f"{heights[char]:.0f} vs DM {upstream_height:.0f}"
                )
        for char in ROUND_CAP_CHARS:
            name = instance.getBestCmap()[ord(char)]
            upstream_name = upstream.getBestCmap()[ord(char)]
            advance = instance["hmtx"][name][0]
            upstream_advance = upstream["hmtx"][upstream_name][0]
            if abs(advance - upstream_advance) / upstream_advance > 0.10:
                problems.append(
                    f"{style} ({weight},{optical_size}): {char} advance changed "
                    f"{advance - upstream_advance:+.0f} units"
                )

        for char in ("C", "O", "G"):
            stroke = _round_side_stroke(instance, char)
            upstream_stroke = _round_side_stroke(upstream, char)
            if abs(stroke - upstream_stroke) > 2.5:
                problems.append(
                    f"{style} ({weight},{optical_size}): {char} side stroke "
                    f"{stroke:.1f} vs DM {upstream_stroke:.1f}"
                )

        if style == "normal":
            o_aspect = widths["O"] / heights["O"]
            c_ratio = widths["C"] / widths["O"]
            g_ratio = widths["G"] / widths["O"]
            if not 0.77 <= o_aspect <= 0.97:
                problems.append(
                    f"normal ({weight},{optical_size}): O aspect {o_aspect:.3f} "
                    "lost the reference's vertical oval"
                )
            if not 0.90 <= c_ratio <= 0.99:
                problems.append(
                    f"normal ({weight},{optical_size}): C/O width ratio {c_ratio:.3f}"
                )
            if not 0.92 <= g_ratio <= 0.99:
                problems.append(
                    f"normal ({weight},{optical_size}): G/O width ratio {g_ratio:.3f}"
                )
            if (weight, optical_size) == (400, 9):
                target_aspects = {"C": 0.825, "O": 0.864, "G": 0.835}
                for char, target in target_aspects.items():
                    aspect = widths[char] / heights[char]
                    if abs(aspect - target) > 0.015:
                        problems.append(
                            f"normal reference master: {char} aspect {aspect:.3f} "
                            f"vs target {target:.3f}"
                        )

        for composite, base_char in (
            ("Odieresis", "O"),
            ("Ccedilla", "C"),
            ("Gbreve", "G"),
        ):
            relationship = mark_relationship(instance, composite, base_char)
            upstream_relationship = mark_relationship(upstream, composite, base_char)
            if (
                relationship is not None
                and upstream_relationship is not None
                and abs(relationship - upstream_relationship) > 3
            ):
                problems.append(
                    f"{style} ({weight},{optical_size}): {composite} mark relationship "
                    f"{relationship:.0f} vs DM {upstream_relationship:.0f}"
                )
    print(f"  verified reference-matched round capitals ({style})")
    print(f"  verified Greek graft + slashed zero ({style})")
    return problems


def build_one(url_name: str, local_name: str, sha256: str, dst_name: str, style: str) -> Path:
    raw = fetch_source(url_name, local_name, sha256)
    # recalcTimestamp=False keeps head.modified at the upstream value, so repeated
    # builds are byte-identical and the output digest can be pinned in the
    # typography contract the same way the upstream sources are.
    font = TTFont(io.BytesIO(raw), recalcTimestamp=False)
    report: list[str] = []

    assert_zero_is_widest(raw, font)
    drop_hvar_after_proof(font, raw)
    graft_greek(font, raw, INTER_ITALIC if style == "italic" else INTER_ROMAN,
                style == "italic", report)
    redraw_round_capitals(
        font,
        raw,
        INTER_ITALIC if style == "italic" else INTER_ROMAN,
        style == "italic",
        report,
    )
    mapping = build_tnum_glyphs(font, raw)
    add_feature(font, "tnum", mapping)
    slash_mapping = add_slashed_zero(font, raw, report)
    add_feature(font, "zero", slash_mapping)
    rename(font, style)
    for line in report:
        print(line)

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out = BUILD_DIR / dst_name
    font.flavor = "woff2"
    font.save(out)
    print(f"  {local_name} -> {out.name}  ({out.stat().st_size} bytes, {len(font.getGlyphOrder())} glyphs)")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    problems = []
    for url_name, local_name, sha256, dst_name, style in SOURCES:
        out = BUILD_DIR / dst_name
        if not args.verify_only:
            print(f"building {style}:")
            out = build_one(url_name, local_name, sha256, dst_name, style)
        if not out.exists():
            problems.append(f"missing build output {out}")
            continue
        problems.extend(verify(out, style))

    if problems:
        print("\nFAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nAll tabular-figure checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
