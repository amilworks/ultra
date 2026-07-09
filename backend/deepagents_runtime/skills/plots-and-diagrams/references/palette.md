# Ultra data palette — the shared source of truth

One calm, colorblind-validated data palette used by **both** runtimes:

- **matplotlib** (this skill, `ultra_style.py`) — sandbox figures, exported
  light-on-white, so they use the **light** column.
- **recharts / shadcn** (frontend `--chart-*` tokens in
  `frontend/src/styles.css`) — the app dashboards, light + dark.

Keep these in lockstep. A data series must be the *same* color in a matplotlib
figure and in a UI chart. If you change a hex here, change both consumers.

## Design intent
Ink is the brand. The palette starts from a near-neutral graphite (so a
single-series chart reads as calm monochrome, not "a colored chart"), and adds
restrained editorial hues (chroma ~0.10–0.12, ~half the loudness of the stock
shadcn defaults). Every slot keeps **one hue across light/dark** — only
lightness/chroma shift for the background. This fixes the prior defect where a
series changed hue entirely when the theme toggled.

## Categorical (fixed order — never cycle, never reorder)

| Slot | Name          | Light     | Dark      | Notes |
|------|---------------|-----------|-----------|-------|
| 1    | graphite      | `#3c414b` | `#b8bec8` | ink anchor; the monochrome default for one series |
| 2    | slate-blue    | `#3a68a5` | `#6294d8` | primary chromatic accent; sequential anchor |
| 3    | terracotta    | `#ae5937` | `#d17956` | warm counterpoint (blue↔orange is the CVD-robust pair) |
| 4    | teal          | `#008e88` | `#00958e` | cyan-green; retains blue so it survives red-green CVD |
| 5    | ochre         | `#ac8128` | `#b88c2e` | muted gold |
| 6    | muted-rose    | `#aa4f58` | `#bf6068` | dusty rose — **not** a status color |
| 7    | dusty-violet  | `#67428c` | `#8c68b5` | desaturated violet |
| 8    | sage          | `#549157` | `#67a469` | muted green; least-used slot |

The slot **order** is the colorblind-safety mechanism — it was derived by
maximizing the minimum adjacent CVD separation across both modes. Do not reorder.

## Chrome (never a hue)

| Role   | Light                 | Dark                    |
|--------|-----------------------|-------------------------|
| grid   | `rgba(23,23,23,0.08)` | `rgba(255,255,255,0.10)`|
| axis   | `rgba(23,23,23,0.22)` | `rgba(255,255,255,0.24)`|
| label  | `#737373`             | `#a1a1aa`               |
| title  | `#171717`             | `#f5f5f5`               |

## Sequential (single hue, low→high)
- light: `#e4eefb #c1d6f2 #96b7e2 #6791ca #3a6aa8 #1f4577 #112845`
- dark:  `#16223b #1f3a63 #2f5991 #4f7cbb #7ba0da #a8c4ec #d3e2f8`

## Diverging (cool ↔ neutral ↔ warm)
- light: `#275591 #6488b9 #b0c3db #eae7e4 #e0b6a4 #bb7051 #9a4729`
- dark:  `#6294d8 #47709f #3a5069 #3a3936 #7a4f3e #b06f52 #d17956`
- The neutral midpoint sits near the light page background — on a light canvas
  rely on cell borders or a colorbar to read the zero band.

## Reserved status (reuse the app's exact meanings)
- `#c62828` danger / failure / negative — never a categorical "series N".
- `#b45309` warning / caveat / threshold (the app's one sanctioned amber).

## Validation (computed, not eyeballed)
Checked with the `dataviz` skill's validator (Machado-2009 CVD + WCAG) and an
independent CIEDE2000/APCA pass:

- Contrast as a mark: all 8 clear **WCAG ≥ 3:1** on both the light (#ffffff) and
  dark (#171717) plot surfaces.
- Chromatic slots 2–8 pass the lightness band and chroma floor in both modes.
  Slot 1 (graphite) is intentionally below the chroma floor — it is the neutral
  anchor, not a hue.
- Adjacent CVD separation is strong (ΔE2000 ≈ 15–17; ΔE76 ≈ 33). Do not quote a
  "2.7× margin" — that was a CIE76 artifact; the honest figure is ~1.4× the floor.
- **Residual weak pairs (need a secondary encoding — marker shape, not color):**
  sage ↔ rose (all-pairs, deuteranopia); and to a lesser degree slate ↔ violet
  and (dark) terracotta ↔ ochre. These only matter when non-adjacent marks can
  neighbor (scatter, bubble, small multiples) — add a marker shape there.

## Re-validate after any edit
```
node <dataviz-skill>/scripts/validate_palette.js \
  "#3a68a5,#ae5937,#008e88,#ac8128,#aa4f58,#67428c,#549157" --mode light --surface "#ffffff"
node <dataviz-skill>/scripts/validate_palette.js \
  "#b8bec8,#6294d8,#d17956,#00958e,#b88c2e,#bf6068,#8c68b5,#67a469" --mode dark --surface "#171717"
```
