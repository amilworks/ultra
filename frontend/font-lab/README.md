# Ultra Sans — font lab

A build pipeline that produces **Ultra Sans**, a derivative typeface, from the
Inter Variable v4.1 binaries Ultra already vendors. No font editor required, no
new outlines drawn: it reassigns which of Inter's *existing* 2,937 glyphs are
the default ones.

```bash
python frontend/font-lab/build_ultra_sans.py --all
python -m http.server 5310 --directory frontend/font-lab   # then open /specimen.html
```

Outputs land in `build/` and are **not** wired into the app. Swapping the
product font is a separate, deliberate change — see *Adopting it* below.

## What it does

Inter ships alternate letterforms behind 22 OpenType feature tags (`cv01`–`cv14`,
`ss01`–`ss08`). Normally you opt into them per-element with
`font-feature-settings`. This pipeline makes a chosen set **permanent and
default**, producing a genuinely different font file rather than a CSS setting.

Three recipes are defined in [`recipe.toml`](recipe.toml):

| Recipe | Glyphs moved | Character |
| --- | ---: | --- |
| `stock` | 0 | Renamed Inter. The control. |
| `calm` | 89 | Tailed `l`, serifed `I`, slashed `0`, open `3 4 6 9`. Reads as Inter; stops `l`/`I`/`1` and `O`/`0` being ambiguous. |
| `geometric` | 193 | Adds single-storey `a`, spurless `u`, straight `t`, tailless `f`, spurred `G`. Pulls toward Futura/Avenir. |

## Why transplant, not "feature freeze"

The standard trick for baking in a feature is to rewrite `cmap` so U+0061 points
at `a.1`. **That is wrong for Inter**, and the font itself says so — two facts
measured from the binary:

1. The letter alternates are not metric-compatible: `a.1` is 104 units wider
   than `a`, `I.1` is 353 wider.
2. **Inter's GPOS coverage does not contain the letter alternates.** `a.1`,
   `u.1`, `G.1`, `t.1`, `f.1`, `I.1` and `l.ss02` appear in no kern pair and no
   kern class. (The *digit* alternates do.)

A cmap freeze would therefore silently drop every kern pair touching `a` — one
of the most-kerned glyphs in Latin text — and you would not see it until the
font was in production looking subtly wrong.

So the pipeline goes the other way. It copies the alternate's outline, advance
width, `gvar` deltas and `HVAR` advance index **into the default glyph's slot**,
keeping the glyph *name*. Every GPOS coverage table, kern class, mark anchor and
GSUB context still resolves, because to the rest of the font `a` is still `a`.

This is safe because the substitution maps have **zero overlap between keys and
values** — no alternate is itself the base of another substitution — so
transplants never cascade and order does not matter. The builder asserts this
(`_assert_disjoint`) and refuses to build if it ever stops being true.

Verified on the built binaries: each transplanted outline is identical to
Inter's corresponding alternate; both variable axes (`wght` 100–900,
`opsz` 14–32) survive; all 2,937 glyphs and all 10 GPOS lookups are retained.

## The reference: TWK Lausanne, and the `opsz` finding

The design reference is [TWK Lausanne](https://weltkern.com/typefaces/019497c9-4357-7256-87c4-0ae3fb803854)
(Nizar Kazan / Weltkern) — a neo-grotesque in the Folio/Helvetica line, which
puts it in the same family as Inter. It is a commercial typeface: we take
direction from it, we do not copy its outlines.

Measuring both with the same canvas ink method turned up something more useful
than any glyph swap. **The Lausanne webfont is static** — no `opsz` axis, no
`wght` axis, identical proportions at every size. Inter is variable, and its
optical-size axis narrows the round lowercase by ~8.5% between `opsz` 14 and 32.

Advance widths, em-normalised:

| Glyph | Inter `opsz` 32 | Inter `opsz` 14 | Lausanne | Gap at `opsz` 14 |
| --- | ---: | ---: | ---: | ---: |
| `o` | 0.549 | 0.600 | 0.598 | −0.3% |
| `e` | 0.536 | 0.583 | 0.580 | −0.5% |
| `a` | 0.518 | 0.562 | 0.555 | −1.2% |
| `n` | 0.547 | 0.591 | 0.584 | −1.2% |
| `g` | 0.565 | 0.613 | 0.621 | +1.3% |
| `c` | 0.523 | 0.571 | 0.581 | +1.8% |
| `H` | 0.708 | 0.743 | 0.714 | −3.9% |
| `t` | 0.311 | 0.327 | 0.347 | +6.1% |
| `O` | 0.749 | 0.765 | 0.817 | +6.8% |

Vertical proportions are near-identical: x-height 0.516 vs 0.517, descender
0.204 vs 0.200. Lausanne's caps are 1.9% shorter (0.714 vs 0.728).

So **Inter's text optical size already has Lausanne's lowercase proportions.**
The apparent gap is entirely Inter's display narrowing — and `typography.css`
sets `font-optical-sizing: auto` at `:root`, so every heading in Ultra gets the
narrowed cut. Pinning `opsz` low at display sizes recovers the generous, round,
Lausanne-like feel for the cost of one CSS declaration:

```css
.display-heading { font-optical-sizing: none; font-variation-settings: "opsz" 14; }
```

See [`optical.html`](optical.html) for the side-by-side. At body size the two
are indistinguishable — the axis has barely moved by then — so this is a
display-only change.

**What this does not reach.** Lausanne's round capitals are wider (`O` +6.8%)
while its straight capitals are narrower (`H` −3.9%) and shorter. That is more
contrast between round and square capitals than Inter has, and it is a property
of the outlines. No axis setting or feature swap produces it; it needs drawing.

## Three references, and what they agree on

`references.html` compares Inter against **Graphik** (Commercial Type),
**Söhne** (Klim) and **TWK Lausanne** (Weltkern). Graphik renders straight from
the macOS system font asset — nothing vendored. All three are proprietary and
appear as on-screen references only; Ultra Sans derives from Inter (OFL) and
from nothing else.

| Reference | Best-fit Inter `opsz` (width) | RMS width error | x/cap | `opsz` matching that x/cap |
| --- | ---: | ---: | ---: | ---: |
| Graphik | 17.04 | 3.46% | 0.7315 | 22.1 |
| Söhne Buch | 27.00 | 1.86% | 0.7284 | 23.5 |
| TWK Lausanne | 16.65 | 2.27% | 0.7240 | 25.4 |
| Inter `opsz` 14 | — | — | **0.7503** | — |
| Inter `opsz` 32 | — | — | **0.7087** | — |

**Graphik fits worst** — not because it sits far away on average, but because it
has a width signature a single axis cannot reproduce: narrow `s` (−7.5% against
its own best fit), wide `t` (+9.5%), wide round capitals (`O` +6.4%). Scaling
`opsz` moves every glyph together; it cannot make `s` narrow and `t` wide at once.

**What all three agree on.** Their x-height relative to cap height lands within
1% of each other (0.724–0.732) — and all three sit well below Inter at text
sizes (0.750). That one number is most of why Inter reads as a UI typeface and
these read as brand typefaces: Inter's x-height is deliberately taller so small
text holds up on screen. Inter's own axis crosses the references' consensus at
about `opsz` 24. Everything else in these tables is width; this is proportion.

## Weight: why Söhne looks darker, and the Inter setting that matches it

Söhne reads darker than Inter at the same size because its strokes are thicker
**relative to its letters**, not in absolute terms. Inter's absolute stem is only
2.7% thinner — but Inter's x-height is larger, so the same stroke covers
proportionally less:

| | stem (em) | x-height | stem / x-height | rendered ink density | vs Söhne |
| --- | ---: | ---: | ---: | ---: | ---: |
| Söhne Buch | 0.0900 | 0.5230 | 0.1721 | 112.3 | — |
| Graphik Regular | 0.0840 | 0.5230 | 0.1606 | 99.8 | −11.2% |
| Inter 400 | 0.0876 | 0.5444 | 0.1609 | 107.8 | −4.0% |
| **Inter 430** | **0.0937** | 0.5444 | **0.1721** | **112.4** | **+0.1%** |

Two independent methods agree on **`wght` 430**: solving stem/x-height from the
outlines lands at 431, and rasterising a real sentence then integrating ink over
the x-height band lands at 430. The solve is stable across the optical range
(429–431 for `opsz` 14–27) because Inter's x-height does not move with weight.

`weight.html` renders the ladder. Note that **Graphik is the lightest of the
three by a wide margin** — 11% below Söhne — which is why it reads as airy.

**A metric that misleads.** Ink per unit of *line length* shows Inter 400 and
Söhne as identical (0.88 each), because Söhne is narrower and packs equal ink
into a shorter line. Perception follows ink per unit of *letter area* — you read
letters, not line-inches. Use the band-normalised figure.

**Not a lever:** `-webkit-font-smoothing`. Ultra's app CSS does not set it while
the other font-lab pages do, which looked like it might explain part of the gap.
It does not — on a Retina Mac macOS uses grayscale antialiasing regardless, so
the property is close to a no-op. Measured and ruled out.

**Knock-on.** Ultra's ladder in `styles.css` is body 400, nav/data/action 500,
label/heading 600, strong 700. Moving body to 430 leaves 70 units to nav instead
of 100, flattening a step the calm hierarchy uses. If body moves, the ladder
should move with it — roughly 430 / 530 / 620 / 700.

## Söhne — why it is a different target from Lausanne

[Söhne](https://klim.co.nz/retail-fonts/soehne/) (Kris Sowersby / Klim Type
Foundry) was added as a reference. Its own name table reads *"Copyright 2025
Klim Type Foundry. All Rights Reserved."*, it carries a trademark notice, and
the supplied files are **trial** fonts under
<https://klim.co.nz/licences/>. Ultra Sans is not and cannot be derived from
it — trial fonts are for evaluation, and the outlines are proprietary
regardless. It is used here only as an on-screen and measured reference, which
is exactly what a trial font is for. Copies live in `refs/`, which is
gitignored and must never be committed or shipped.

Fitting Inter's `opsz` axis to each reference's lowercase advance widths:

| Reference | Best-fit Inter `opsz` | RMS width error |
| --- | ---: | ---: |
| Söhne Buch | **27.0** | 1.86% |
| TWK Lausanne | **16.65** | 2.27% |

**They are opposite ends of the same axis.** Söhne's `o` is 0.564 em against
Lausanne's 0.598 — 5.7% narrower. You cannot chase both. Vertical proportions
do agree, which is why either fit works at all: Söhne x/cap 0.728, Lausanne
0.724, Inter at `opsz` 27 sits at 0.720.

`references.html` shows the fit. At 15px body the fitted Inter and Söhne break
the line at the identical word, while Inter as Ultra ships today runs wider.

**What width-fitting does and does not buy.** Matching advances to under 2% RMS
matches the *rhythm and colour* of a block of text — the thing you perceive
before you read. It does not change the letterforms: Söhne's `a`, `R`, `t` and
`G` stay Söhne's, and Inter's stay Inter's. That gap is outlines, not metrics.

**The tradeoff, stated plainly.** `font-optical-sizing: auto` maps CSS px onto
the `opsz` axis, so Ultra currently renders body copy near `opsz` 15 and
headings clamped at 32. Pinning 27 everywhere holds Söhne's proportions but
gives up Inter's optical compensation at small sizes — the widening and opening
that exists to aid reading. Söhne makes the same compromise by being static.
Pinning at display only, leaving `auto` for body, keeps the compensation at the
cost of not being "true Söhne" throughout.

## Mono: the silent grade jump, and the pin

`mono.html` documents the code-typography pass. The finding: **JetBrains Mono is
static faces, and CSS font-matching rounds a 400–500 target upward** — so when
body moved 400 → 430, all twenty mono rules that inherited body weight silently
jumped to Mono 500. `document.fonts` proved it: only the 500 face was fetched.

| | stem (em) | x-height | stem/x | vs prose@430 | ink density | vs prose |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prose — Inter 430 | 0.0937 | 0.5444 | 0.1721 | — | 111.7 | — |
| **JBM 400** | 0.0900 | 0.5500 | **0.1636** | **−5%** | 89.9 | −19.5% |
| JBM 500 | 0.1080 | 0.5500 | 0.1964 | +14% | 103.4 | −7.4% |

Stroke ratio and texture density disagree, and that disagreement is the point:
mono's fixed 0.6em advance makes any weight read lighter *as a block* while its
strokes read heavier *as letterforms*. Chips are read as letterforms — stroke
parity is the criterion — and the parity solve lands at ≈426. 400 is 2.9×
closer than 500, matches what code editors ship, and stays safe under Night
bloom. Hence `--font-weight-mono: 400`, pinned at every mono `font-family`
site; the contract and `light-theme-ink.test.ts` now refuse any mono rule that
inherits its weight.

Deliberately unchanged: block leading 1.2 (authored in `code-block.tsx`; the
JetBrains IDE default this face was drawn for — ink pitch 13.4px clears the
15.6px box), inline `0.88em` (x-height parity would want 0.99em, but the wide
advance already inflates mono; chip contrast holds at 6.7:1 dark / 5.1:1
light — both clear of AA), and the nine mono rules that pinned explicit weights at authoring time.

## Known limitation: italic

**Inter Italic has no single-storey `a`.** The `geometric` recipe transplants
193 glyphs in roman but only 158 in italic, and `cv11` is reported skipped. Its
roman and italic would not match — an italic word would silently revert to the
double-storey form mid-sentence. `calm` has no such gap; it applies identically
to both faces.

Fixing this means drawing an italic single-storey `a` (and its ~34 accented
composites) in a real font editor against Inter's UFO sources. That is
outside what this pipeline does.

## Licensing

Inter is SIL OFL 1.1 and declares **no Reserved Font Name** — the copyright line
is bare, with no "with Reserved Font Name" clause. Renaming and redistributing a
modified version is therefore permitted outright. Obligations we meet:

- The upstream copyright (name ID 0) and license (IDs 13/14) travel in every
  built binary, with our derivative note appended rather than substituted.
- `OFL-1.1.txt` is copied into `build/` on every run.
- Any derivative must itself ship under OFL 1.1 and may not be sold on its own.

`build_ultra_sans.py` deliberately does not touch name IDs 0, 13 or 14.

## Adopting it

`frontend/scripts/check-typography-contract.mjs` pins the exact Inter v4.1 byte
counts, the SHA-256 in `PROVENANCE.md`, and the `"BisQue Inter Variable"` family
name in both source and built CSS. That guardrail is doing its job — it means a
font swap cannot happen by accident. Adopting Ultra Sans means, in one
deliberate change: new binaries into `src/assets/fonts/`, a rewritten
`PROVENANCE.md`, the `@font-face` family renamed in `typography.css`, and the
contract's pinned bytes and family name updated to match.
