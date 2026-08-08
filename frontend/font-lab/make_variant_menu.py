#!/usr/bin/env python3
"""Render every one of Inter's alternate-glyph features against one pangram.

This is the letterform-identification tool. Given a mockup you want to match,
you find the row whose glyphs match it, and that row names the feature tag to
put in `recipe.toml`.

It reads the affected characters straight out of Inter's GSUB rather than a
hand-written list, so the menu cannot drift from what the font actually does.
No extra font binaries are built: the page applies each feature live with
`font-feature-settings` over the stock face, which renders the identical
outlines the transplant would bake in.

    python make_variant_menu.py && open build/variants.html
"""

from __future__ import annotations

import html
import unicodedata
from pathlib import Path

from fontTools.ttLib import TTFont

from build_ultra_sans import (
    BUILD,
    FEATURE_GLOSS,
    UPSTREAM,
    single_substitution_map,
)

PANGRAM = "The quick brown fox jumped over the lazy dog."

# Features worth eyeballing: the ones that change a letter or digit you would
# actually type. Excludes positional/technical sets (numr, dnom, sups, subs,
# frac, ordn, case, calt, ccmp, locl) and the enclosed-glyph sets ss05/ss06.
MENU = [
    "cv01", "cv02", "cv03", "cv04", "cv05", "cv06", "cv07",
    "cv08", "cv09", "cv10", "cv11", "cv12", "cv13", "cv14",
    "ss01", "ss02", "ss03", "ss04", "ss07", "ss08", "zero", "salt",
]


def affected_characters(font: TTFont, tag: str) -> str:
    """The typeable characters a feature changes, in codepoint order."""
    reverse: dict[str, int] = {}
    for codepoint, glyph in font.getBestCmap().items():
        reverse.setdefault(glyph, codepoint)

    seen: set[int] = set()
    for base in single_substitution_map(font, tag):
        codepoint = reverse.get(base)
        if codepoint is None or codepoint < 0x21:
            continue
        # Skip the positional variants (superscript/subscript forms etc.) and
        # anything outside Latin + punctuation — noise in a specimen.
        if codepoint > 0x24F and not (0x2018 <= codepoint <= 0x201F):
            continue
        category = unicodedata.category(chr(codepoint))
        if category.startswith(("L", "N", "P")):
            seen.add(codepoint)
    return "".join(chr(c) for c in sorted(seen))


def main() -> int:
    font = TTFont(UPSTREAM / "InterVariable-v4.1.woff2")
    BUILD.mkdir(parents=True, exist_ok=True)

    rows = []
    for tag in MENU:
        chars = affected_characters(font, tag)
        if not chars:
            continue
        gloss = FEATURE_GLOSS.get(tag, "")
        # Cap the glyph strip: `salt` and `ss07` touch hundreds of accented
        # forms and would run off the page.
        strip = chars if len(chars) <= 46 else chars[:46] + "…"
        rows.append(f"""
    <div class="v" style="--f:'{tag}'">
      <div class="meta"><span class="tag">{tag}</span><span class="gloss">{html.escape(gloss)}</span>
        <span class="count">{len(chars)} glyph{'s' if len(chars) != 1 else ''}</span></div>
      <div class="strip off">{html.escape(strip)}</div>
      <div class="strip on"  style="font-feature-settings:'{tag}' 1">{html.escape(strip)}</div>
      <div class="pan off">{html.escape(PANGRAM)}</div>
      <div class="pan on"   style="font-feature-settings:'{tag}' 1">{html.escape(PANGRAM)}</div>
    </div>""")
    font.close()

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Inter variant menu — match your mockup</title>
<style>
  @font-face {{ font-family:"US"; src:url("./UltraSans-stock.woff2") format("woff2"); font-weight:100 900; }}
  :root {{ --ink:#171b1d; --ink2:#5b6266; --ink3:#8b9296; --rule:#e4e7e9; --bg:#fff; --hit:#c2410c;
          font-optical-sizing:auto; font-synthesis:none; }}
  @media (prefers-color-scheme:dark) {{
    :root {{ --ink:#dce3ea; --ink2:#a5abb0; --ink3:#6f767b; --rule:#2a2f33; --bg:#141719; --hit:#fb923c; }} }}
  *{{box-sizing:border-box}}
  body {{ margin:0; padding:44px 40px 90px; background:var(--bg); color:var(--ink);
         font-family:"US",system-ui,sans-serif; -webkit-font-smoothing:antialiased; }}
  .wrap {{ max-width:1180px; margin:0 auto; }}
  h1 {{ font-size:15px; font-weight:600; margin:0 0 5px; letter-spacing:-.01em; }}
  .sub {{ font-size:13px; color:var(--ink2); line-height:1.6; margin:0 0 34px; max-width:70ch; }}
  .v {{ border-top:1px solid var(--rule); padding:20px 0 22px;
        display:grid; grid-template-columns:1fr 1fr; gap:5px 28px; align-items:baseline; }}
  .meta {{ grid-column:1/-1; display:flex; align-items:baseline; gap:11px; margin-bottom:8px; }}
  .tag {{ font-size:11px; font-weight:600; letter-spacing:.05em; text-transform:uppercase; color:var(--hit); }}
  .gloss {{ font-size:13px; color:var(--ink); }}
  .count {{ font-size:11px; color:var(--ink3); margin-left:auto; }}
  .strip {{ font-size:31px; letter-spacing:.045em; word-break:break-all; }}
  .pan {{ font-size:17px; letter-spacing:-.011em; padding-top:5px; }}
  .off {{ color:var(--ink3); }}
  .on  {{ color:var(--ink); }}
  .lbl {{ position:sticky; top:0; background:var(--bg); display:grid;
          grid-template-columns:1fr 1fr; gap:28px; padding:9px 0 8px; z-index:2;
          font-size:10px; font-weight:600; letter-spacing:.09em; text-transform:uppercase; color:var(--ink3);
          border-bottom:1px solid var(--rule); }}
</style></head><body><div class="wrap">
  <h1>Inter variant menu — 22 features, one pangram</h1>
  <p class="sub">Left column is stock Inter; right column has the feature on. Find the row whose
  right-hand glyphs match your mockup — the orange tag is the feature to add to
  <code>recipe.toml</code>. Rendered live over the stock face, so these are the exact outlines
  the transplant bakes in.</p>
  <div class="lbl"><span>Stock Inter</span><span>Feature applied</span></div>
  {''.join(rows)}
</div></body></html>"""

    out = BUILD / "variants.html"
    out.write_text(page)
    print(f"wrote {out}  ({len(rows)} features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
