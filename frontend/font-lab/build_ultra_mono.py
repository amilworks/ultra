"""Ultra Mono — a variable mono for Ultra, derived from DM Mono + JetBrains Mono.

    python frontend/font-lab/build_ultra_mono.py            # build + verify
    python frontend/font-lab/build_ultra_mono.py --verify-only

Outputs land in `build/`. Wiring them into the app is a separate, deliberate
change gated by the typography contract.

## Why this font exists

DM Mono is the designed monospace counterpart of DM Sans — the family Ultra Sans
derives from — so it is the mono that makes Ultra's type read as one system. But
as shipped it cannot do Ultra's job:

- **Three static weights (300/400/500), no 600.** Six rules in styles.css set
  mono at 600 (viewer orientation/axis/slice cues); with `font-synthesis: none`
  those silently snap to 500, erasing a deliberate step.
- **No Greek.** 1 codepoint in the Greek block (π). λ, σ, Δ, θ, α, β all fall
  out of the font mid-line in code blocks and data viewers.
- **Static files** in a system where every other face is variable.

## Why the construction is sound

**Masters, not shipped statics.** Google Fonts' DM Mono statics are NOT
point-compatible (62 roman / 76 italic glyphs mismatch) because upstream built
them per-instance with overlap removal. This pipeline instead compiles the
upstream Glyphs sources (two drawn masters: Light stem 70, Medium stem 106) into
a variable font with fontmake, then instances masters out of that VF — instances
of one VF are point-compatible by construction.

**The 600 is the family's own vector, extended.** DM Mono has exactly two drawn
masters, so its weight axis is linear by construction: user 300→500 maps to stem
70→106 (the shipped Regular instance sits slightly light of midpoint, at ~84).
The synthesized 600 master continues the masters' line to stem ~124 — the
designers' own weight vector, not an invented direction. Extrapolation is one
drawn-range half-step past Medium, and every glyph is checked afterwards for
counter collapse (a contour whose signed area shrinks toward zero or flips
sign); the tightest counter in the family keeps 69% of its area at 600.

**Greek is grafted per-master from weight-matched JetBrains instances.** Both
fonts share the same 600/1000em cell and both are OFL with no Reserved Font
Name. For each Ultra Mono master, JetBrains' variable weight axis is solved so
that after scaling, the grafted stems match that master's stems — lowercase and
capitals solved separately, because the two fonts' x-height/cap ratios differ
(DM 496/700, JB 550/730). Lowercase scales by 496/550, capitals by 700/730, both
about the cell's own center so JetBrains' optical centering survives. Italic
grafts are additionally sheared by the 1° slant difference (DM −10°, JB −9°).
Grafted glyphs are decomposed to simple outlines, so they are compatible across
masters (all are instances of the same JetBrains VF).

## Licensing

DM Mono: (c) 2020 The DM Mono Project Authors, OFL 1.1, no Reserved Font Name.
JetBrains Mono: (c) 2020 The JetBrains Mono Project Authors, OFL 1.1, no
Reserved Font Name. Both copyright notices travel in name ID 0 of every built
binary; the OFL text is carried in ID 13/14 and re-emitted next to the build.
The derivative is itself OFL 1.1 and may not be sold on its own.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import subprocess
import sys
import unicodedata
from pathlib import Path

from fontTools import varLib
from fontTools.designspaceLib import (
    AxisDescriptor,
    DesignSpaceDocument,
    InstanceDescriptor,
    SourceDescriptor,
)
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

LAB = Path(__file__).resolve().parent
BUILD = LAB / "build"
SOURCES = LAB / "sources"

UPSTREAM = {
    # file -> (url, sha256)
    "DMMono-MASTER.glyphs": (
        "https://raw.githubusercontent.com/googlefonts/dm-mono/main/source/DMMono-MASTER.glyphs",
        "7e73628b3cd9f3a164eaf3109145a59e15a633f3a9d12a2509c2bb027fc25314",
    ),
    "DMMono-Italics-MASTER.glyphs": (
        "https://raw.githubusercontent.com/googlefonts/dm-mono/main/source/DMMono-Italics-MASTER.glyphs",
        "a3ecd457114537a29921caca4d1a5eea926b031fa581d2d8363fb645fa77d4d5",
    ),
    "JetBrainsMono[wght].ttf": (
        "https://raw.githubusercontent.com/google/fonts/main/ofl/jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf",
        None,  # filled by --print-digests bootstrap; pinned below once known
    ),
    "JetBrainsMono-Italic[wght].ttf": (
        "https://raw.githubusercontent.com/google/fonts/main/ofl/jetbrainsmono/JetBrainsMono-Italic%5Bwght%5D.ttf",
        None,
    ),
}
# Pinned after first verified fetch (kept out of the dict literal so the two
# stages read clearly). These are google/fonts main as of 2026-08-17.
JB_DIGESTS = {
    "JetBrainsMono[wght].ttf": "48715a42ec242c21e9f02692891e147d022299a52e48d5e413e1a942193ffeda",
    "JetBrainsMono-Italic[wght].ttf": "85ae2a5cd3f56baf1ce1c21a851322c58e3d8fbe8e8ad4a4d090a820dd7fe558",
}

MASTER_WEIGHTS = [300, 400, 500]
SYNTH_WEIGHT = 600
ALL_WEIGHTS = MASTER_WEIGHTS + [SYNTH_WEIGHT]
CELL = 600  # the mono advance, both fonts, all weights
UPM = 1000

# Vertical fit of the graft (measured from the sources/VFs):
DM_XH, DM_CAP = 496, 700
JB_XH, JB_CAP = 550, 730
SCALE_LC = DM_XH / JB_XH  # 0.9018…
SCALE_UC = DM_CAP / JB_CAP  # 0.9589…
DM_SLANT, JB_SLANT = -10.0, -9.0  # degrees; graft shear = difference

# Fixed head dates so repeated builds are byte-identical (fontTools Long
# datetime, seconds since 1904-01-01; value = 2020-05-01 00:00:00, DM Mono era).
FIXED_DATE = 3670963200


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch(name: str) -> Path:
    SOURCES.mkdir(parents=True, exist_ok=True)
    url, pinned = UPSTREAM[name]
    pinned = JB_DIGESTS.get(name, pinned) if name in JB_DIGESTS else pinned
    dest = SOURCES / name
    if dest.exists() and (pinned in (None,) or sha256(dest.read_bytes()) == pinned):
        return dest
    raw = subprocess.run(["curl", "-sL", url], capture_output=True).stdout
    actual = sha256(raw)
    if pinned and not pinned.startswith("REPLACE") and actual != pinned:
        raise SystemExit(f"{name}: digest mismatch\n  expected {pinned}\n  got      {actual}")
    dest.write_bytes(raw)
    if pinned and pinned.startswith("REPLACE"):
        print(f"  [pin me] {name}: {actual}")
    return dest


def compile_upstream_vf(glyphs_file: Path, out: Path) -> Path:
    """fontmake the .glyphs masters into the upstream variable font."""
    if out.exists():
        return out
    result = subprocess.run(
        [sys.executable, "-m", "fontmake", "-g", str(glyphs_file), "-o", "variable",
         "--output-path", str(out)],
        capture_output=True, text=True, cwd=str(SOURCES),
    )
    if result.returncode != 0:
        raise SystemExit(f"fontmake failed for {glyphs_file.name}:\n{result.stderr[-2000:]}")
    return out


# ---------------------------------------------------------------- outline maths

def glyph_coords(font: TTFont, name: str):
    """(coords, endPts, flags) of a simple glyph."""
    g = font["glyf"][name]
    coords, ends, flags = g.getCoordinates(font["glyf"])
    return list(coords), list(ends), bytes(flags)


def set_glyph_coords(font: TTFont, name: str, coords):
    glyf = font["glyf"]
    g = glyf[name]
    cur = g.getCoordinates(glyf)[0]
    for i, pt in enumerate(coords):
        cur[i] = (round(pt[0]), round(pt[1]))
    g.recalcBounds(glyf)
    adv, _ = font["hmtx"][name]
    font["hmtx"][name] = (adv, g.xMin if g.numberOfContours else 0)


def contour_signed_areas(font: TTFont, name: str) -> list[float]:
    """Signed area per contour (shoelace over the point polygon — adequate as a
    topology probe: a counter collapsing to nothing crosses zero long before the
    polygon approximation matters)."""
    g = font["glyf"][name]
    if g.numberOfContours <= 0:
        return []
    coords, ends, _ = g.getCoordinates(font["glyf"])
    areas, start = [], 0
    for e in ends:
        pts = coords[start : e + 1]
        a = 0.0
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            a += x1 * y2 - x2 * y1
        areas.append(a / 2)
        start = e + 1
    return areas


def scanline_stem(font: TTFont, ch: str, y: float) -> float | None:
    """Width of the ink run(s) crossing y — flattens quadratics to segments."""
    cmap = font.getBestCmap()
    if ord(ch) not in cmap:
        return None
    name = cmap[ord(ch)]
    pen = DecomposingRecordingPen(font.getGlyphSet())
    font.getGlyphSet()[name].draw(pen)
    segs = []

    def flat_q(p0, p1, p2, n=16):
        pts = []
        for i in range(1, n + 1):
            t = i / n
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
            yy = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
            pts.append((x, yy))
        return pts

    cur = start = None
    for op, args in pen.value:
        if op == "moveTo":
            cur = start = args[0]
        elif op == "lineTo":
            segs.append((cur, args[0])); cur = args[0]
        elif op == "qCurveTo":
            pts = list(args)
            if pts[-1] is None:  # TrueType all-offcurve contour
                pts[-1] = start
            prev = cur
            # expand implied on-curves
            oncurve_chain = []
            for i in range(len(pts) - 1):
                c = pts[i]
                nxt = pts[i + 1]
                mid = ((c[0] + nxt[0]) / 2, (c[1] + nxt[1]) / 2) if i < len(pts) - 2 else nxt
                oncurve_chain.append((c, mid))
            for c, end in oncurve_chain:
                flat = flat_q(prev, c, end)
                p = prev
                for q in flat:
                    segs.append((p, q)); p = q
                prev = end
            cur = pts[-1]
        elif op == "curveTo":
            # cubic (shouldn't appear in TTF, but flatten anyway)
            p0, (c1, c2, p3) = cur, (args[0], args[1], args[2])
            prev = p0
            for i in range(1, 17):
                t = i / 16
                mt = 1 - t
                x = mt**3 * p0[0] + 3 * mt**2 * t * c1[0] + 3 * mt * t**2 * c2[0] + t**3 * p3[0]
                yy = mt**3 * p0[1] + 3 * mt**2 * t * c1[1] + 3 * mt * t**2 * c2[1] + t**3 * p3[1]
                segs.append((prev, (x, yy))); prev = (x, yy)
            cur = p3
        elif op == "closePath":
            if cur is not None and start is not None and cur != start:
                segs.append((cur, start))
            cur = start
    xs = []
    for (x1, y1), (x2, y2) in segs:
        if (y1 <= y < y2) or (y2 <= y < y1):
            t = (y - y1) / (y2 - y1)
            xs.append(x1 + t * (x2 - x1))
    xs.sort()
    runs = [xs[i + 1] - xs[i] for i in range(0, len(xs) - 1, 2)]
    return min(runs) if runs else None  # the stem, not the whole letter


# ------------------------------------------------------------------- pipeline

def instance_master(vf_path: Path, wght: int, pin: dict) -> TTFont:
    font = TTFont(vf_path)
    loc = {"wght": wght, **pin}
    instantiateVariableFont(font, loc, inplace=True, updateFontNames=False)
    return font


def synthesize_600(m500: TTFont, m300: TTFont) -> TTFont:
    """P600 = P500 + (P500 − P300)/2 — the family's own linear axis, continued."""
    out = TTFont(io.BytesIO(_bytes(m500)))
    glyf5, glyf3 = m500["glyf"], m300["glyf"]
    for name in m500.getGlyphOrder():
        g5, g3 = glyf5[name], glyf3[name]
        if g5.isComposite():
            comps5, comps3 = g5.components, g3.components
            gout = out["glyf"][name]
            for co, c5, c3 in zip(gout.components, comps5, comps3):
                co.x = round(c5.x + (c5.x - c3.x) / 2)
                co.y = round(c5.y + (c5.y - c3.y) / 2)
            continue
        if g5.numberOfContours <= 0:
            continue
        c5 = g5.getCoordinates(glyf5)[0]
        c3 = g3.getCoordinates(glyf3)[0]
        new = [(x5 + (x5 - x3) / 2, y5 + (y5 - y3) / 2) for (x5, y5), (x3, y3) in zip(c5, c3)]
        set_glyph_coords(out, name, new)
    return out


def _bytes(font: TTFont) -> bytes:
    buf = io.BytesIO()
    font.save(buf)
    return buf.getvalue()


def jb_stem_curve(jb_path: Path, ch: str, y: float) -> list[tuple[int, float]]:
    out = []
    for w in range(100, 801, 100):
        f = instance_master(jb_path, w, {})
        s = scanline_stem(f, ch, y)
        if s:
            out.append((w, s))
    return out


def solve_weight(curve: list[tuple[int, float]], target: float) -> int:
    best = min(range(len(curve) - 1),
               key=lambda i: 0 if curve[i][1] <= target <= curve[i + 1][1] else
               min(abs(curve[i][1] - target), abs(curve[i + 1][1] - target)))
    (w1, s1), (w2, s2) = curve[best], curve[best + 1]
    if s2 == s1:
        return w1
    w = w1 + (target - s1) * (w2 - w1) / (s2 - s1)
    return int(round(max(100, min(800, w))))


def graft_glyphs(master: TTFont, jb_path: Path, dm_stem: float, italic: bool,
                 graft_cps: list[int], report: list[str]) -> None:
    """Add Greek from JetBrains, weight-matched and rescaled, to one master."""
    # Probe heights matter: 'H' at half cap-height IS the crossbar, so a scan
    # there returns the crossbar run and the solver slams into the axis floor.
    # Scan the upper stems instead; 'l' has no such trap at half x-height.
    curve_lc = jb_stem_curve(jb_path, "l", JB_XH / 2)
    curve_uc = jb_stem_curve(jb_path, "H", JB_CAP * 0.82)
    w_lc = solve_weight(curve_lc, dm_stem / SCALE_LC)
    w_uc = solve_weight(curve_uc, dm_stem / SCALE_UC)
    inst_lc = instance_master(jb_path, w_lc, {})
    inst_uc = instance_master(jb_path, w_uc, {})
    report.append(f"    graft weights: lowercase JB@{w_lc}, capitals JB@{w_uc} (target stem {dm_stem:.0f})")

    shear = math.tan(math.radians(-DM_SLANT)) - math.tan(math.radians(-JB_SLANT)) if italic else 0.0

    glyf = master["glyf"]
    hmtx = master["hmtx"]
    cmap_tables = [t for t in master["cmap"].tables if t.isUnicode()]
    order = master.getGlyphOrder()
    existing = set(order)

    for cp in graft_cps:
        is_upper = unicodedata.category(chr(cp)) == "Lu"
        src_font = inst_uc if is_upper else inst_lc
        scale = SCALE_UC if is_upper else SCALE_LC
        src_cmap = src_font.getBestCmap()
        if cp not in src_cmap:
            continue
        src_name = src_cmap[cp]
        pen = DecomposingRecordingPen(src_font.getGlyphSet())
        src_font.getGlyphSet()[src_name].draw(pen)

        tpen = TTGlyphPen(None)
        # scale about the cell center horizontally and the baseline vertically,
        # then shear for the italic slant difference
        def xf(pt):
            x, y = pt
            x = CELL / 2 + (x - CELL / 2) * scale
            y = y * scale
            x += shear * y
            return (round(x), round(y))
        for op, args in pen.value:
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
        new_name = f"uni{cp:04X}.ultra"
        glyph = tpen.glyph()
        glyf.glyphs[new_name] = glyph
        glyph.recalcBounds(glyf)
        hmtx.metrics[new_name] = (CELL, glyph.xMin if glyph.numberOfContours else 0)
        if new_name not in existing:
            order.append(new_name)
            existing.add(new_name)
        for t in cmap_tables:
            t.cmap[cp] = new_name

    master.setGlyphOrder(order)
    master.getReverseGlyphMap(rebuild=True)
    glyf.glyphOrder = order
    master["maxp"].numGlyphs = len(order)


def build_family(glyphs_src: str, jb_src: str, out_name: str, italic: bool,
                 report: list[str]) -> Path:
    style = "italic" if italic else "roman"
    report.append(f"  {style}:")
    upstream_vf = compile_upstream_vf(SOURCES / glyphs_src, SOURCES / f"upstream-{style}-VF.ttf")
    jb_path = fetch(jb_src)

    pin = {"ital": 100} if italic else {"ital": 0}
    masters = {w: instance_master(upstream_vf, w, pin) for w in MASTER_WEIGHTS}
    masters[SYNTH_WEIGHT] = synthesize_600(masters[500], masters[300])

    # graft list: Greek block glyphs JetBrains has and DM Mono lacks
    jb_cmap = TTFont(jb_path).getBestCmap()
    dm_cmap = masters[400].getBestCmap()
    graft_cps = sorted(cp for cp in jb_cmap
                       if 0x0370 <= cp <= 0x03FF and cp not in dm_cmap)
    report.append(f"    graft list: {len(graft_cps)} Greek codepoints")

    stems = {}
    for w, master in masters.items():
        stem = scanline_stem(master, "l", DM_XH / 2)
        stems[w] = stem
        graft_glyphs(master, jb_path, stem, italic, graft_cps, report)
    report.append("    DM stems by weight: "
                  + ", ".join(f"{w}→{stems[w]:.0f}" for w in ALL_WEIGHTS))

    # ------------------------------------------------------------- merge to VF
    BUILD.mkdir(parents=True, exist_ok=True)
    ds = DesignSpaceDocument()
    axis = AxisDescriptor()
    axis.tag, axis.name = "wght", "Weight"
    axis.minimum, axis.default, axis.maximum = 300, 400, 600
    ds.addAxis(axis)
    master_paths = {}
    for w, master in masters.items():
        p = BUILD / f"_master-{style}-{w}.ttf"
        master["head"].created = FIXED_DATE
        master["head"].modified = FIXED_DATE
        # save() re-stamps head.modified unless told not to — the default master's
        # date flows into the merged VF, so nondeterminism here breaks the digest.
        master.recalcTimestamp = False
        master.save(p)
        master_paths[w] = p
        src = SourceDescriptor()
        src.path = str(p)
        src.location = {"Weight": w}
        ds.addSource(src)
    for w, nm in [(300, "Light"), (400, "Regular"), (500, "Medium"), (600, "SemiBold")]:
        inst = InstanceDescriptor()
        inst.styleName = f"{nm} Italic".strip() if italic and nm != "Regular" else ("Italic" if italic else nm)
        if italic and nm == "Regular":
            inst.styleName = "Italic"
        elif italic:
            inst.styleName = f"{nm} Italic"
        inst.location = {"Weight": w}
        ds.addInstance(inst)

    vf, _, _ = varLib.build(ds, optimize=True)

    # names + licensing: both upstream copyrights travel; derivative notes added.
    family = "Ultra Mono"
    subfamily = "Italic" if italic else "Regular"
    name = vf["name"]
    dm_c = "Copyright 2020 The DM Mono Project Authors (https://www.github.com/googlefonts/dm-mono)"
    jb_c = "Copyright 2020 The JetBrains Mono Project Authors (https://github.com/JetBrains/JetBrainsMono)"
    ofl = (SOURCES / ".." / ".." / "src" / "assets" / "fonts" / "OFL-1.1-DMSans.txt")
    for nid, value in [
        (0, f"{dm_c}. Greek glyphs: {jb_c}."),
        (1, family), (2, subfamily),
        (3, f"{family} {subfamily}; Ultra derivative of DM Mono with JetBrains Mono Greek"),
        (4, f"{family} {subfamily}"),
        (6, f"UltraMono-{subfamily}"),
        (10, "Ultra Mono is a derivative of DM Mono (Colophon Foundry for Google Fonts), "
             "extended with a generated SemiBold master continuing the family's own weight "
             "axis, and Greek glyphs grafted from weight-matched JetBrains Mono instances. "
             "Both sources are SIL OFL 1.1 with no Reserved Font Name."),
        (13, "This Font Software is licensed under the SIL Open Font License, Version 1.1."),
        (14, "https://openfontlicense.org"),
        (16, family), (17, subfamily),
    ]:
        name.setName(value, nid, 3, 1, 0x409)
    # scrub master-era name records that varLib may carry at other IDs
    name.names = [r for r in name.names if r.nameID <= 17 or r.nameID >= 256]

    if italic:
        vf["post"].italicAngle = DM_SLANT
        vf["OS/2"].fsSelection = (vf["OS/2"].fsSelection & ~0x40) | 0x01
        vf["head"].macStyle |= 0x2

    vf["head"].created = FIXED_DATE
    vf["head"].modified = FIXED_DATE

    out = BUILD / out_name
    vf.flavor = "woff2"
    vf.recalcTimestamp = False
    vf.save(out)
    for p in master_paths.values():
        p.unlink()
    report.append(f"    -> {out.name}  {out.stat().st_size} bytes, {vf['maxp'].numGlyphs} glyphs")
    return out


# ----------------------------------------------------------------- verification

def verify(path: Path, italic: bool) -> list[str]:
    problems = []
    raw = path.read_bytes()
    base = TTFont(io.BytesIO(raw))
    axes = {a.axisTag: (a.minValue, a.defaultValue, a.maxValue) for a in base["fvar"].axes}
    if axes.get("wght") != (300.0, 400.0, 600.0):
        problems.append(f"{path.name}: wght axis is {axes.get('wght')}, expected (300,400,600)")

    # 1. mono grid: every glyph, every instance, advance == CELL
    for w in (300, 340, 400, 500, 545, 600):
        inst = TTFont(io.BytesIO(raw))
        instantiateVariableFont(inst, {"wght": w}, inplace=True)
        bad = [g for g in inst.getGlyphOrder()
               if inst["hmtx"][g][0] not in (0, CELL)]
        if bad:
            problems.append(f"{path.name} wght={w}: {len(bad)} glyphs off the {CELL}-unit grid, e.g. {bad[:4]}")

    # 2. counter topology: 600 vs 500 — no contour may flip sign or collapse
    i500 = TTFont(io.BytesIO(raw)); instantiateVariableFont(i500, {"wght": 500}, inplace=True)
    i600 = TTFont(io.BytesIO(raw)); instantiateVariableFont(i600, {"wght": 600}, inplace=True)
    worst = (None, 1.0)
    for gname in i500.getGlyphOrder():
        a5 = contour_signed_areas(i500, gname)
        a6 = contour_signed_areas(i600, gname)
        if len(a5) != len(a6):
            problems.append(f"{path.name}: {gname} contour count changed 500→600")
            continue
        if not a5:
            continue
        # Winding is a convention, not an assumption: the largest-|area| contour
        # is the outer; anything of the OPPOSITE sign is a counter. Counters are
        # what extrapolated weight can crush.
        outer_sign = 1 if a5[max(range(len(a5)), key=lambda k: abs(a5[k]))] > 0 else -1
        for j, (x5, x6) in enumerate(zip(a5, a6)):
            if abs(x5) < 100:  # degenerate speck either way
                continue
            if (x5 > 0) != (x6 > 0):
                problems.append(f"{path.name}: {gname} contour {j} flipped sign at 600 (counter collapsed)")
            elif (1 if x5 > 0 else -1) != outer_sign and abs(x6) / abs(x5) < worst[1]:
                worst = (f"{gname}[{j}]", abs(x6) / abs(x5))
    if worst[0]:
        print(f"    smallest surviving counter at 600: {worst[0]} keeps {worst[1]*100:.0f}% of its 500 area")
        if worst[1] < 0.45:
            problems.append(f"{path.name}: counter {worst[0]} shrinks to {worst[1]*100:.0f}% at 600 — inspect visually")

    # 3. stems continue the family line (70/88/106/124 within tolerance)
    for w, expect in [(300, 70), (400, 88), (500, 106), (600, 124)]:
        inst = TTFont(io.BytesIO(raw)); instantiateVariableFont(inst, {"wght": w}, inplace=True)
        s = scanline_stem(inst, "l", DM_XH / 2)
        if s is None or abs(s - expect) > 6:
            problems.append(f"{path.name} wght={w}: 'l' stem {s} vs expected ~{expect}")

    # 4. Greek present, gridded, and plausibly weight-matched
    cm = base.getBestCmap()
    for ch in "λσΔθαβγφωΩςπ":
        if ord(ch) not in cm:
            problems.append(f"{path.name}: U+{ord(ch):04X} {ch} missing")
    inst = TTFont(io.BytesIO(raw)); instantiateVariableFont(inst, {"wght": 400}, inplace=True)
    lam = scanline_stem(inst, "λ", DM_XH * 0.35)
    ell = scanline_stem(inst, "l", DM_XH / 2)
    if lam and ell and not (0.80 <= lam / ell <= 1.25):
        problems.append(f"{path.name}: grafted λ stem {lam:.0f} vs l stem {ell:.0f} — weight match off")

    # 5. italic slant
    if italic and abs(base["post"].italicAngle - DM_SLANT) > 0.01:
        problems.append(f"{path.name}: italicAngle {base['post'].italicAngle} != {DM_SLANT}")

    print(f"    verified {path.name}: grid, counters, stems, Greek, slant")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    jobs = [
        ("DMMono-MASTER.glyphs", "JetBrainsMono[wght].ttf", "UltraMono-Variable.woff2", False),
        ("DMMono-Italics-MASTER.glyphs", "JetBrainsMono-Italic[wght].ttf",
         "UltraMono-Italic-Variable.woff2", True),
    ]
    report: list[str] = []
    problems: list[str] = []
    for glyphs_src, jb_src, out_name, italic in jobs:
        out = BUILD / out_name
        if not args.verify_only:
            fetch(glyphs_src)
            print(f"building {out_name}:")
            out = build_family(glyphs_src, jb_src, out_name, italic, report)
            for line in report:
                print(line)
            report.clear()
        if not out.exists():
            problems.append(f"missing {out}")
            continue
        problems.extend(verify(out, italic))

    if problems:
        print("\nFAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nAll Ultra Mono checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
