# Vendored font provenance

Ultra ships four families with three jobs. **Ultra Sans** is the primary product face —
UI chrome, reading copy, headings. **Ultra Mono** is the product monospace —
code, data readouts, viewer cues, file contents — behind **JetBrains Mono** as
its coverage net. **Inter** draws the `BisQue Ultra` wordmark and backstops the
glyphs Ultra Sans lacks.

Ultra Sans was designed and authored by **Amil Khan**, a PhD student in the
Department of Electrical and Computer Engineering at the University of
California, Santa Barbara, for **Ultra**, an agentic system for science. This
describes the derivative work; DM Sans, DM Mono, Inter, and JetBrains Mono retain
their original credits and licenses. The UCSB affiliation identifies the author
and does not imply University endorsement.

## Ultra Sans — the product face

Ultra Sans is a derivative of **DM Sans** closing three structural gaps: no
tabular figures (`tnum` generated; stock digits span 310/1000em so
`font-variant-numeric: tabular-nums` silently did nothing), no Greek (grafted in
from Inter — see below), and the weakest `0`/`O` distinction of every candidate
face measured (a generated slashed-zero alternate behind a dormant `zero`
feature).

The source authority is the dedicated
[Ultra Sans repository](https://github.com/amilworks/ultra-sans). This
application pins the last font-producing development snapshot at
[`3078939c29c3298954718a941e3d8dc9e0458525`](https://github.com/amilworks/ultra-sans/tree/3078939c29c3298954718a941e3d8dc9e0458525).
Its `font-lab/build_ultra_tabular.py` verifies the upstream masters, performs the
feature additions and reference C/O/G, lowercase `s`, and question-mark redraws,
renames the family, and emits WOFF2. The build is byte-reproducible
(`head.modified` pinned to upstream), so the output digests below are stable and
re-derivable.

| Asset | Style | Variable axes | Bytes | SHA-256 |
| --- | --- | --- | ---: | --- |
| `UltraSans-Variable.woff2` | normal | `wght` 100–1000; `opsz` 9–40 | 127248 | `b7fff4a81ec342f76d4a88625a450699112a94a2e3280e8d4ad366adbe01221c` |
| `UltraSans-Italic-Variable.woff2` | italic | `wght` 100–1000; `opsz` 9–40 | 155152 | `f136650cad07ed6c74ef2bdaf580cba947f14ef4d4978d27d2063ab72d1783d4` |

Derived from these upstream DM Sans masters:

| Upstream file | Source | SHA-256 |
| --- | --- | --- |
| `DMSans[opsz,wght].ttf` | `https://raw.githubusercontent.com/google/fonts/main/ofl/dmsans/DMSans%5Bopsz,wght%5D.ttf` | `8cd08d97e89c24d0aa92edd2f0f4c8ee6195eee9b7c9f154865a58b02f0c1c0d` |
| `DMSans-Italic[opsz,wght].ttf` | `https://raw.githubusercontent.com/google/fonts/main/ofl/dmsans/DMSans-Italic%5Bopsz,wght%5D.ttf` | `22259c0cc8237221b80f44c76ba8d36e6bce3cda72779f5b2773643d499720ae` |

License: SIL Open Font License 1.1, preserved verbatim as `OFL-DM-Sans.txt`
(4389 bytes). DM Sans declares **no Reserved Font Name** — its copyright line is
bare — so renaming and redistributing a modified version is permitted outright.
Obligations met: the upstream copyright (name ID 0) and license (IDs 13/14) are
carried through untouched, the derivative note is appended to the description
rather than substituted, and Ultra Sans itself remains under OFL 1.1.
Name ID 9 leads with Amil Khan and retains the DM Sans and Inter design lineage;
IDs 10–12 record the full authorship statement and project URLs.

### What was added, and why it is safe

The numeric feature does not alter source outlines: each tabular digit is a
**composite** of the original digit, recentred inside a shared advance, so the
base figure and all of its variation are inherited. The deliberate exceptions
are the documented `C O G Q Ø`, lowercase `s`, and question-mark redraws below.

Three measured facts underpin the construction, all re-checked by the build:

1. **`0` is the widest digit at every axis location** (verified over opsz 9–40 ×
   wght 100–1000). So the tabular advance is exactly `advance(zero)` — a quantity
   read from the font, not modelled.
2. **Advances flow from gvar phantom points; HVAR is dropped.** The italic's
   HVAR diverged from its shipped phantom deltas by up to 4 units on some glyphs,
   so the build rebuilds every glyph's phantom advance deltas from HVAR-true
   advances sampled at all 12 master locations (verified exact afterwards), then
   removes HVAR. Every glyph — original, tabular, grafted — carries its advance
   the same way.
3. The centring shift is a *linear* function of the two advances, and gvar deltas
   are additive, so the shift's deltas are that same linear combination of the
   base digits' advance deltas — again exact, with no modelling.

Digits are centred on their **ink**, not their advance box; centring the advance
box left `1` 44 units off centre in its cell. The ink correction is near-constant
across the axes (worst drift 10 units, 0.15px at 15px), so it is applied
statically. Residual off-centring is ≤8 units (0.12px at 15px) across the weights
Ultra actually uses, rising to 21 units only at wght 1000 / opsz 40, which Ultra
never renders.

Default figures remain **proportional**, as they should for prose; the tabular
set is reached only through `tnum`, which is what `font-variant-numeric:
tabular-nums` activates.

### Greek, grafted in — no longer a CSS-fallback concern

Stock DM Sans covers one Greek codepoint (π) out of 403 total; Ultra renders λ,
σ, Δ, θ constantly, and the old CSS fallback could not reach canvas text
(`ctx.font` names a single family) nor weight-match per master. The build grafts
**104 Greek codepoints** from Inter (OFL, no Reserved Font Name), solved per
location of the two-axis design space: DM Sans's gvar regions map to a grid of
**12 master locations** (opsz {9, 24, 40} × wght {100, 300, 400, 1000},
intermediates included), and at each one Inter's own axes are solved so stems
and x/cap proportions match DM's there — plus a horizontal correction that keeps
Greek tracking DM's width rhythm where Inter's opsz axis saturates. Deltas come
from fontTools' VariationModel over those locations, so the graft is exact at
every master and interpolates with the same support math as the rest of the
font; italic grafts are additionally sheared 0.6° to DM's −10°. Verified by stem
comparison (λ vs l) and x-height alignment (σ vs x) at Ultra's real tokens
(wght 430/690, opsz 13), and at the axis extremes.

### Reference-matched C O G Q Ø — drawn at every master

The supplied small-size reference is optically corrected rather than
mathematically circular. Its `O` is subtly vertical; `C` is narrower and more
open; `G` is narrower than `O` and uses a long horizontal bar. At the normal
text master, the superseded geometric treatment measured `C .927 / O 1.000 /
G 1.000`; the redraw measures `C .825 / O .865 / G .834`. In a like-for-like
30px raster proxy for the reference's 10–11 CSS px capture, Inter had the lowest
mean absolute pixel error of DM Sans, the old Ultra Sans, Inter, and Ultra Mono.

Inter therefore supplies the round-cap skeleton and terminal geometry. At each
of DM Sans's 12 `wght × opsz` master locations, the build solves the matching
Inter optical size, solves Inter's weight again against DM's capital-round side
stroke, scales to DM's round overshoot height, and shears italic outlines the
0.6° needed to land on DM's −10° angle. `C O G Q Ø` are written into the
existing DM glyph slots, retaining their names, original DM advances, and DM ink
centres. GPOS coverage, line lengths, and surrounding spacing therefore remain
stable while the silhouettes change. Ultra Mono / DM Mono is used as the 10px
G-bar clarity check, not an outline donor; its compressed proportions belong in
code, not proportional body text.

All samples share point structure before gvar deltas are emitted. Build-time
verification checks the reference proportions, DM-matched stroke and overshoot,
unchanged advances, and the original mark-to-bowl relationship in `Ç Ö Ğ`
across regular, italic, text, display, and axis extremes. `Q` and `Ø` follow the
same bowl system so the change does not stop at the three specimen letters.

### Reference-matched lowercase s — sentence rhythm at body size

The approved reference was a 24×32-device-pixel crop. At that exact raster
height, the DM-derived Ultra `s` measured `0.782` width/height with nearly equal
upper/lower ink mass (`0.988`); the reference measured `0.750` and `0.948`.
Its smaller upper bowl, fuller lower bowl, stronger diagonal waist, and 2–5%
more open exits keep repeated `s` shapes from becoming wide, dark interruptions
inside a sentence. Inter opsz 20 / wght 430 was the closest measured source;
changing only DM's weight or optical size did not reproduce the white shape.

Inter supplies the roman and italic skeleton. DM opsz `9/24/40` samples Inter
`14/29/32`, placing Ultra's body setting near the approved opsz-20 reference.
At every one of the 12 masters, the source is independently weight-solved
against DM's lowercase stem and scaled uniformly to DM's exact `s` overshoot.
Both original sidebearings are retained; the advance changes only by the
ink-width delta. The normal body result is aspect `0.763` with an 11-unit
advance reduction (about 0.17px at 15px), while extreme bold masters may widen
enough to preserve their counters. Composite deltas keep accented forms centred
over the new base. Build-time verification pins the body silhouette, normal and
italic narrowing, side-space, vertical overshoot, accent relationships, and
interpolation across the complete weight and optical-size grid.

### Reference-matched question marks — open at interface scale

The previous DM-derived `?` turned inward early and placed its neck and dot far
left of the hook's optical centre. In Ultra's welcome headline this could read
like a `P` followed by a detached dot. The replacement uses a broader open hook,
a later inward turn, a short neck, and a circular dot on the same optical axis.

Ultra reconstructs both `?` and `¿` from fitted Inter curve scaffolds at all 12
masters. DM optical sizes `9/24/40` map to Inter `14/29/32`; source weights are
solved after scaling against DM's lowercase stem. An Ultra-specific width curve
then controls the hook while the dot is re-rounded and aligned to the neck in
roman and italic. DM advances, vertical endpoints, and established ink centres
remain unchanged. Verification covers the full master grid and the product
headline setting (`wght` 350, `opsz` 13).

### Slashed zero — capability, dormant by default

`zero.slash` (plus `zero.tnum.slash` so tabular contexts compose) is generated
per master location from the 0's own geometry — slash angle from the counter,
thickness from the 0's own thin stroke, so it tracks both axes like a drawn
glyph. Reached only through the `zero` OpenType feature, which nothing enables
by default: the documented doctrine is that identity does not come from
substituted glyphs, so defaults are untouched and data surfaces opt in with
`font-variant-numeric: slashed-zero` where `0`/`O` ambiguity actually costs
something (run IDs, hashes).

## Ultra Mono — the product monospace

Ultra Mono is a derivative of **DM Mono** (Colophon Foundry for Google Fonts,
the designed monospace counterpart of the DM Sans family Ultra Sans derives
from), with two structural additions DM Mono cannot ship without: a true weight
range, and Greek. Built by `frontend/font-lab/build_ultra_mono.py`; the build is
byte-reproducible (timestamps pinned), so the digests below are stable and
re-derivable.

| Asset | Style | Variable axes | Bytes | SHA-256 |
| --- | --- | --- | ---: | --- |
| `UltraMono-Variable.woff2` | normal | `wght` 300–600 | 40244 | `b3445619fbe749b4384a27af82b47bc159916a75e9a394fa4a9f23af41c6166d` |
| `UltraMono-Italic-Variable.woff2` | italic | `wght` 300–600 | 43744 | `16d1735d9fc8dcb6978182e2b46805a4988db9415a750b1784948e26e9985cb1` |

Derived from the upstream **Glyphs sources**, not the shipped statics — Google
Fonts' DM Mono statics are not point-compatible (62 roman / 76 italic glyphs
mismatch, an artifact of per-instance overlap removal), so the pipeline compiles
the sources with fontmake and instances compatible masters from the result:

| Upstream file | Source | SHA-256 |
| --- | --- | --- |
| `DMMono-MASTER.glyphs` | `https://raw.githubusercontent.com/googlefonts/dm-mono/main/source/DMMono-MASTER.glyphs` | `7e73628b3cd9f3a164eaf3109145a59e15a633f3a9d12a2509c2bb027fc25314` |
| `DMMono-Italics-MASTER.glyphs` | `https://raw.githubusercontent.com/googlefonts/dm-mono/main/source/DMMono-Italics-MASTER.glyphs` | `a3ecd457114537a29921caca4d1a5eea926b031fa581d2d8363fb645fa77d4d5` |

### What was added, and the measurements behind it

**A generated 600 (SemiBold) master.** DM Mono has exactly **two drawn masters**
— Light at stem 70 and Medium at stem 106 — so its weight axis is linear by
construction, and the 600 continues that same vector to stem ~124: the
designers' own direction, extended one drawn-range half-step, not an invented
weight. Six rules in styles.css ask for mono 600 (viewer orientation, axis and
slice cues); before this font, `font-synthesis: none` silently snapped them to
Medium. Every glyph is checked for counter collapse after extrapolation; the
tightest counter in the family keeps **69% of its area** at 600, and stems land
on the family line (70/84/106/124, Regular shipping slightly light of midpoint
exactly as upstream's own instance does).

**Greek, grafted from weight-matched JetBrains Mono instances.** DM Mono ships
exactly one Greek codepoint (π); Ultra renders λ, σ, Δ, θ constantly inside
mono contexts, where the sans coverage face cannot reach. 76 Greek codepoints
are grafted per master from JetBrains Mono (also OFL, no Reserved Font Name),
whose variable weight axis is solved per master so grafted stems match DM
stems — lowercase and capitals solved separately (JB@287–750 lowercase,
JB@252–717 capitals across the range), lowercase scaled by 496/550 to DM's
x-height, capitals by 700/730 to DM's cap height, both about the shared
600-unit cell's center; italic grafts sheared 1° to DM's −10°. The graft is
verified by stem comparison (grafted λ within tolerance of Latin `l`) and by
the grid invariant below.

**Weights are used at their nominal values — measured, not assumed.** By the
band-normalized standard this repo measures ink with (stem/x-height), Ultra
Mono 400 sits at **0.169** against JetBrains 400's **0.165**: the incumbent's
perceived weight within 2.4%, darker if anything, so no re-derived weight
tokens are needed. The visible change of the migration is x-height (496 vs
550/1000em — code reads slightly smaller at equal size), which is DM Mono's
proportions, i.e. the point.

**Invariants held by the build:** every glyph in every instance — grafts
included — advances exactly 600/1000em; default figures and grid survive at
arbitrary axis locations; the fonts are unhinted (upstream statics carried
ttfautohint, the VF path does not — standard for variable fonts, and Ultra's
sizes sit above the hinting-critical range).

License: SIL OFL 1.1. DM Mono declares **no Reserved Font Name**
(`OFL-1.1-DMMono.txt`, 4484 bytes); JetBrains Mono likewise
(`OFL-1.1-JetBrainsMono.txt`, 4399 bytes). Both copyright notices travel in
name ID 0 of every built binary; the derivative note is appended, not
substituted.

## JetBrains Mono — the mono coverage net

The same JetBrains Mono that donated the Greek outlines is vendored once as the
fallback behind Ultra Mono, for the long tail a data viewer meets in arbitrary
files (Cyrillic, box drawing, arrows — 976 codepoints against Ultra Mono's
~480). Container-converted to WOFF2 from the pinned google/fonts TTFs; the
conversion changes no table.

| Asset | Style | Variable axes | Bytes | SHA-256 |
| --- | --- | --- | ---: | --- |
| `JetBrainsMono-Variable.woff2` | normal | `wght` 100–800 | 71736 | `7b7f3419196f675a973d30cb70078749120caddea86c8547ebf54a8db2ca13af` |
| `JetBrainsMono-Italic-Variable.woff2` | italic | `wght` 100–800 | 76452 | `84ef9a6d8b91d130a05c2b5697756d7b8d0d72ce5aef9ddffdd66881fac8e9f1` |

Converted from, and verifiable against:

| Upstream file | SHA-256 |
| --- | --- |
| `JetBrainsMono[wght].ttf` | `48715a42ec242c21e9f02692891e147d022299a52e48d5e413e1a942193ffeda` |
| `JetBrainsMono-Italic[wght].ttf` | `85ae2a5cd3f56baf1ce1c21a851322c58e3d8fbe8e8ad4a4d090a820dd7fe558` |

The coverage face carries **no `size-adjust`**, deliberately and in contrast to
the sans coverage face: both fonts share the same 600/1000em cell, so an
unscaled fallback glyph stays exactly on the mono grid and columns keep
aligning. Scaling JetBrains down to Ultra Mono's x-height would shrink its
advance to ~541 and put every substituted glyph off-grid — for monospace, the
grid outranks optical size matching. The x-height jump on rare fallback glyphs
is the accepted cost, recorded here.

This replaced `@fontsource/jetbrains-mono` — 14 per-weight CSS imports emitting
28 files (508K, `.woff` duplicates included) — with two variable files totalling
148K that load only when a fallback glyph is actually needed.

## Inter — wordmark, and coverage backstop

Ultra vendors the unmodified official Inter Variable v4.1 webfont assets. Inter
draws the wordmark at its native scale and provides a metric-adjusted coverage
backstop behind Ultra Sans. Both families are local WOFF2 assets; the application
makes no network request to a font host.

| Asset | Style | Variable axes | Official source | Bytes | SHA-256 |
| --- | --- | --- | --- | ---: | --- |
| `InterVariable-v4.1.woff2` | normal | `wght` 100–900; `opsz` 14–32 | `https://rsms.me/inter/font-files/InterVariable.woff2?v=4.1` | 352240 | `693b77d4f32ee9b8bfc995589b5fad5e99adf2832738661f5402f9978429a8e3` |
| `InterVariable-Italic-v4.1.woff2` | italic | `wght` 100–900; `opsz` 14–32 | `https://rsms.me/inter/font-files/InterVariable-Italic.woff2?v=4.1` | 387976 | `e564f652916db6c139570fefb9524a77c4d48f30c92928de9db19b6b5c7a262a` |

Version: Inter 4.1. License: SIL Open Font License 1.1; the upstream
`LICENSE.txt` for tag `v4.1` is preserved verbatim as `OFL-1.1.txt`.

The same two files back **two** `@font-face` families, deliberately:

- `BisQue Inter Variable` — Inter at its own scale, consumed only by
  `--font-brand` and therefore only by the wordmark.
- `BisQue Inter Coverage` — the identical bytes with `size-adjust: 96.4%`,
  sitting behind Ultra Sans in `--font-sans` as the LONG-TAIL net (Cyrillic,
  Latin Extended beyond DM's set, symbols) now that Greek lives in Ultra Sans
  itself. The ratio is Ultra Sans's x-height over Inter's (526.0 / 545.9 per
  1000em), so a substituted glyph matches the text around it instead of
  announcing itself. It carries no `unicode-range` — a scoped range would drop
  non-Greek gaps to the platform font. Sharing one URL with the wordmark face
  means the browser fetches each file once, so the backstop costs no extra
  bytes.

Only WOFF2 is shipped. We deliberately do not preload a font today: the benefit
has not yet been measured against the extra early-bandwidth cost on Ultra's
scientific workspaces. If future production measurements justify preloading,
only the Ultra Sans normal face may be considered. Both italic faces must remain
demand-loaded. The post-build typography contract requires any future preload
URL to exactly match the emitted Ultra Sans normal-face URL.
