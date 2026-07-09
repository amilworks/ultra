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
single-series chart leads with blue, not grey) and uses vibrant editorial hues
(chroma ~0.14–0.17). This is the CVD-tuned hue/luminance structure of the
original calm set with chroma boosted ~1.4× — livelier *and* slightly more
colorblind-robust than the calm version (a from-scratch bright palette collapsed
under CVD; boosting the tuned structure did not). Every slot keeps **one hue
across light/dark** — only lightness/chroma shift for the background.

## Categorical (fixed order — never cycle, never reorder)

| Slot | Name        | Light     | Dark      | Notes |
|------|-------------|-----------|-----------|-------|
| 1    | blue        | `#1e65bd` | `#4a92f2` | primary; a single series is blue; sequential anchor |
| 2    | terracotta  | `#c14701` | `#e66933` | warm counterpoint (blue↔orange is the CVD-robust pair) |
| 3    | teal        | `#00948d` | `#00aca4` | cyan-green; retains blue so it survives red-green CVD |
| 4    | ochre       | `#b97c00` | `#c18200` | gold |
| 5    | rose        | `#bd394e` | `#d34b5d` | crimson-rose — **not** a status color |
| 6    | violet      | `#6e34a0` | `#935ccb` | royal violet |
| 7    | green       | `#399743` | `#4eaa55` | green; the green↔rose all-pairs weak pair needs marker shapes |
| 8    | graphite    | `#3c414b` | `#b8bec8` | neutral / context anchor (grey out de-emphasized data) |

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

## Diagram palette (outlined shapes only — NOT data marks)
For hand-authored diagrams (flow / block / architecture), where a dark outline
carries each shape's edge, use these soft pastels + an **ink outline** (`#171717`,
~1.6px) and ink text. They are deliberately light and **fail 3:1 on white as bare
marks** — never use them for lines/bars/points.

| Name | Hex | | Name | Hex |
|------|-----|-|------|-----|
| blue | `#86c7ea` | | violet | `#b39ddb` |
| green | `#a3dd9b` | | teal | `#43cdaa` |
| pink | `#f19ca6` | | amber | `#f3c98b` |

## Validation (computed, not eyeballed)
Checked with the `dataviz` skill's validator (Machado-2009 CVD + WCAG):

- Contrast as a mark: all 8 clear **WCAG ≥ 3:1** on both the light (#ffffff) and
  dark (#171717) plot surfaces.
- Chromatic slots 1–7 pass the lightness band and chroma floor in both modes.
  Slot 8 (graphite) is intentionally below the chroma floor — it is the neutral
  anchor, not a hue.
- Adjacent CVD separation is strong (ΔE76 ≈ 39). All-pairs worst ≈ 9 (ΔE76,
  deuteranopia: green↔rose light, ochre↔terracotta dark) — the floor band, so
  **scatter/bubble/small-multiples must add a marker shape** (already the rule).
  This is slightly better than the calm palette's ~7.9.
- The vibrant set was derived by boosting the calm palette's chroma ~1.4×; a
  from-scratch evenly-spread bright palette was **rejected** (coral↔green ΔE 1.4).

## Re-validate after any edit
```
node <dataviz-skill>/scripts/validate_palette.js \
  "#1e65bd,#c14701,#00948d,#b97c00,#bd394e,#6e34a0,#399743" --mode light --surface "#ffffff" --pairs all
node <dataviz-skill>/scripts/validate_palette.js \
  "#4a92f2,#e66933,#00aca4,#c18200,#d34b5d,#935ccb,#4eaa55,#b8bec8" --mode dark --surface "#171717" --pairs all
```
