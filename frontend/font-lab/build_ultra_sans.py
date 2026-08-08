#!/usr/bin/env python3
"""Build Ultra Sans — a derivative typeface from the vendored Inter Variable.

Why transplant instead of freezing features
-------------------------------------------
The usual way to bake an OpenType feature into a font permanently is a "feature
freeze": rewrite `cmap` so U+0061 points at `a.1` instead of `a`. That is wrong
for Inter. Two measured facts kill it:

  1. Inter's letter alternates are NOT metric-compatible. `a.1` is 104 units
     wider than `a`; `I.1` is 353 wider. Fine either way — both approaches can
     carry the advance across.
  2. Inter's GPOS coverage does not include the letter alternates. `a.1`,
     `u.1`, `G.1`, `t.1`, `f.1`, `I.1`, `l.ss02` appear in NO kern pair or kern
     class. (The digit alternates do.) A cmap freeze therefore silently drops
     every kern pair touching `a` — one of the most-kerned glyphs in Latin.

So we go the other way: copy the alternate's outline, advance, and variation
deltas INTO the default glyph slot, keeping the glyph *name*. Every GPOS
coverage table, kern class, mark anchor, and GSUB context still resolves,
because as far as the rest of the font is concerned `a` is still `a` — it just
looks different now.

This is safe here because the substitution maps have zero overlap between keys
and values (verified at build time, see `_assert_disjoint`): no alternate is
itself the base of another substitution, so transplants never cascade and the
order they run in does not matter.

Usage
-----
    python build_ultra_sans.py --recipe geometric
    python build_ultra_sans.py --all          # build every recipe + specimen
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from fontTools.ttLib import TTFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
UPSTREAM = REPO / "frontend" / "src" / "assets" / "fonts"
BUILD = HERE / "build"

FACES = [
    ("InterVariable-v4.1.woff2", "Roman", "normal"),
    ("InterVariable-Italic-v4.1.woff2", "Italic", "italic"),
]

# Name IDs we rewrite. 0 (copyright), 13/14 (license) are deliberately absent —
# OFL 1.1 §2 requires the upstream notice travel with every derivative.
RENAMED_IDS = {
    1: "family",       # Font Family
    3: "unique",       # Unique identifier
    4: "full",         # Full font name
    6: "postscript",   # PostScript name
    16: "family",      # Typographic Family
    25: "psprefix",    # Variations PostScript Name Prefix
}

# Human-readable gloss for the build report. Keys are Inter feature tags.
FEATURE_GLOSS = {
    "cv01": "1 — flat-footed one",
    "cv02": "4 — open four",
    "cv03": "6 — straight-tailed six",
    "cv04": "9 — straight-tailed nine",
    "cv05": "l — tailed lowercase L",
    "cv06": "u — spurless u",
    "cv07": "ß — alternate eszett",
    "cv08": "I — serifed uppercase i",
    "cv09": "3 — flat-topped three",
    "cv10": "G — spurred G",
    "cv11": "a — single-storey a",
    "cv12": "f — tailless f",
    "cv13": "t — straight-bottomed t",
    "cv14": "ẞ — alternate capital eszett",
    "ss01": "3 4 6 9 — open digits",
    "ss02": "disambiguation set",
    "ss03": "comma accents",
    "ss04": "disambiguation, no serifed l",
    "ss07": "square dots on accents",
    "ss08": "straight quotes",
    "zero": "0 — slashed zero",
}


@dataclass
class BuildReport:
    recipe: str
    face: str
    features: list[str]
    transplanted: int = 0
    skipped: list[str] = field(default_factory=list)
    out: Path | None = None


# --------------------------------------------------------------------------
# GSUB introspection
# --------------------------------------------------------------------------

def single_substitution_map(font: TTFont, tag: str) -> dict[str, str]:
    """Flatten every SingleSubst/AlternateSubst lookup reachable from `tag`.

    Contextual and ligature lookups are ignored: they have no single "default
    shape" to transplant, so a feature built only from them is a no-op here and
    is reported as skipped rather than silently dropped.
    """
    gsub = font["GSUB"].table
    lookups = gsub.LookupList.Lookup
    indices: set[int] = set()
    for record in gsub.FeatureList.FeatureRecord:
        if record.FeatureTag == tag:
            indices.update(record.Feature.LookupListIndex)

    mapping: dict[str, str] = {}
    for index in sorted(indices):
        for subtable in lookups[index].SubTable:
            kind = subtable.__class__.__name__
            if kind == "SingleSubst":
                mapping.update(subtable.mapping)
            elif kind == "AlternateSubst":
                # First alternate is the feature's primary form.
                for glyph, alts in subtable.alternates.items():
                    if alts:
                        mapping[glyph] = alts[0]
    return mapping


def _assert_disjoint(mapping: dict[str, str]) -> None:
    """Transplants must not cascade — no alternate may itself be a base."""
    overlap = set(mapping) & set(mapping.values())
    if overlap:
        raise SystemExit(
            f"refusing to build: {len(overlap)} glyph(s) are both a "
            f"substitution source and target, so transplant order would "
            f"change the result: {sorted(overlap)[:8]}"
        )


# --------------------------------------------------------------------------
# The transplant
# --------------------------------------------------------------------------

def transplant(font: TTFont, mapping: dict[str, str], report: BuildReport) -> None:
    """Move each alternate's outline + metrics + deltas into its base slot."""
    glyf = font["glyf"]
    hmtx = font["hmtx"].metrics
    gvar = font["gvar"].variations if "gvar" in font else {}
    order = set(font.getGlyphOrder())

    advance_map = None
    if "HVAR" in font and font["HVAR"].table.AdvWidthMap is not None:
        advance_map = font["HVAR"].table.AdvWidthMap.mapping

    for base, alt in sorted(mapping.items()):
        if base not in order or alt not in order:
            report.skipped.append(f"{base}->{alt} (missing glyph)")
            continue

        # Outline. deepcopy so the alternate keeps its own independent copy —
        # it stays reachable through its feature tag and must not alias.
        glyf.glyphs[base] = copy.deepcopy(glyf[alt])

        # Advance width + left side bearing.
        hmtx[base] = hmtx[alt]

        # Variation deltas travel with the outline they describe: point counts
        # (or component counts, for composites) must agree with the new shape.
        if alt in gvar:
            gvar[base] = copy.deepcopy(gvar[alt])
        elif base in gvar:
            del gvar[base]

        # Advance-width variation index, so the width tracks wght/opsz.
        if advance_map is not None and alt in advance_map:
            advance_map[base] = advance_map[alt]

        report.transplanted += 1


# --------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------

def rename(font: TTFont, cfg: dict, recipe: str, style: str) -> None:
    """Rewrite identity name records; retain copyright and license verbatim."""
    name = font["name"]
    family = cfg["family"]
    if recipe not in ("calm", "default"):
        family = f"{family} {recipe.capitalize()}"

    ps_family = cfg["postscript"] + ("" if recipe in ("calm", "default") else recipe.capitalize())
    subfamily = "Italic" if style == "Italic" else "Regular"
    ps_style = "Italic" if style == "Italic" else "Regular"
    version = cfg["version"]

    values = {
        "family": family,
        "unique": f"{version};{cfg['vendor']};{ps_family}-{ps_style}",
        "full": f"{family} {subfamily}",
        "postscript": f"{ps_family}-{ps_style}",
        "psprefix": ps_family,
    }

    # Rewrite in place, preserving each record's platform/encoding/language so
    # we do not lose the Mac or Windows variants.
    for record in list(name.names):
        key = RENAMED_IDS.get(record.nameID)
        if key:
            name.setName(values[key], record.nameID, record.platformID,
                         record.platEncID, record.langID)

    # Version string (ID 5) keeps a pointer back to the Inter release it came
    # from — provenance you can read out of the binary itself.
    for record in list(name.names):
        if record.nameID == 5:
            name.setName(
                f"Version {version}.000; derived from Inter 4.001",
                5, record.platformID, record.platEncID, record.langID,
            )
        elif record.nameID == 0:
            upstream = record.toUnicode()
            if cfg["derivative_note"] not in upstream:
                name.setName(
                    f"{upstream}. {cfg['derivative_note']}",
                    0, record.platformID, record.platEncID, record.langID,
                )


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def build_face(cfg: dict, recipe: str, features: list[str],
               source: Path, style: str) -> BuildReport:
    report = BuildReport(recipe=recipe, face=style, features=list(features))
    font = TTFont(source)

    combined: dict[str, str] = {}
    for tag in features:
        mapping = single_substitution_map(font, tag)
        if not mapping:
            report.skipped.append(f"{tag} (no single substitutions)")
            continue
        combined.update(mapping)

    _assert_disjoint(combined)
    transplant(font, combined, report)
    rename(font, cfg, recipe, style)

    BUILD.mkdir(parents=True, exist_ok=True)
    suffix = "-Italic" if style == "Italic" else ""
    out = BUILD / f"{cfg['postscript']}-{recipe}{suffix}.woff2"
    font.flavor = "woff2"
    font.save(out)
    font.close()
    report.out = out
    return report


def verify(report: BuildReport, mapping_probe: list[tuple[str, str]]) -> list[str]:
    """Re-open the built font and confirm the transplant actually landed."""
    problems = []
    font = TTFont(report.out)
    glyf, hmtx = font["glyf"], font["hmtx"].metrics
    for base, alt in mapping_probe:
        if base not in glyf.glyphs or alt not in glyf.glyphs:
            continue
        if hmtx[base] != hmtx[alt]:
            problems.append(f"{base}: advance {hmtx[base][0]} != {alt} {hmtx[alt][0]}")
        a, b = glyf[base], glyf[alt]
        if a.numberOfContours != b.numberOfContours:
            problems.append(f"{base}: contour count differs from {alt}")
    font.close()
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", default="calm")
    parser.add_argument("--all", action="store_true", help="build every recipe")
    args = parser.parse_args()

    spec = tomllib.loads((HERE / "recipe.toml").read_text())
    cfg = spec["font"]
    recipes = spec["recipes"]

    wanted = list(recipes) if args.all else [args.recipe]
    unknown = [r for r in wanted if r not in recipes]
    if unknown:
        print(f"unknown recipe(s): {unknown}; have {list(recipes)}", file=sys.stderr)
        return 2

    if BUILD.exists():
        shutil.rmtree(BUILD)

    for recipe in wanted:
        features = recipes[recipe]["variants"]
        print(f"\n\033[1m{cfg['family']} — {recipe}\033[0m")
        print(f"  {recipes[recipe]['description']}")
        if features:
            for tag in features:
                print(f"    {tag}  {FEATURE_GLOSS.get(tag, '')}")
        else:
            print("    (no glyph changes — rename only)")

        for filename, style, _ in FACES:
            report = build_face(cfg, recipe, features, UPSTREAM / filename, style)
            size = report.out.stat().st_size
            print(f"  {style:<7} {report.transplanted:>4} glyphs  "
                  f"{size:>7,} B  {report.out.name}")
            for note in report.skipped:
                print(f"          skipped: {note}")

            probe = [("a", "a.1"), ("l", "l.ss02"), ("I", "I.1"), ("zero", "zero.slash")]
            problems = verify(report, [p for p in probe if _in_features(p, features)])
            for problem in problems:
                print(f"          \033[31mVERIFY {problem}\033[0m")

    # OFL 1.1 §2: the notice travels with every copy.
    shutil.copy(UPSTREAM / "OFL-1.1.txt", BUILD / "OFL-1.1.txt")
    print(f"\nwrote {BUILD.relative_to(REPO)}/  (OFL-1.1.txt included)")
    return 0


_PROBE_FEATURE = {"a": "cv11", "l": "cv05", "I": "cv08", "zero": "zero"}


def _in_features(pair: tuple[str, str], features: list[str]) -> bool:
    return _PROBE_FEATURE.get(pair[0]) in features


if __name__ == "__main__":
    raise SystemExit(main())
