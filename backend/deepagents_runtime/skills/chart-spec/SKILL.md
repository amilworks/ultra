---
name: chart-spec
description: Render a small INTERACTIVE chart inline in the chat conversation (hover tooltips, legend, light/dark aware) by emitting a fenced ```chart block containing a JSON spec — no code execution, no image file. Use for a quick line/bar/area/scatter/pie of small tabular data (a few series, up to ~1000 rows) that you already have in the conversation and want the reader to hover and read. For publication-quality, complex, or scientific figures, or anything needing computation, use plots-and-diagrams (matplotlib, static image via code execution) instead.
---

# Chart spec (interactive in-chat chart)

Emit a JSON chart spec in a fenced ```` ```chart ```` block. The frontend
validates it and draws it with the app's own chart components (recharts + the
brand `--chart-*` palette), so it matches the UI, gets hover tooltips and a
legend, and recolors correctly in light and dark. **No code runs** — the block
is data only.

## When to use this vs a matplotlib figure
- **This (chart-spec):** you already have the numbers in the conversation, the
  dataset is small, and interactivity (hover to read exact values) helps. Fast,
  no sandbox, theme-aware, native look.
- **plots-and-diagrams (matplotlib):** you need to compute the data, the figure
  is complex/scientific/publication-grade, you want a downloadable 300-DPI image,
  or you need a chart type beyond line/bar/area/scatter/pie. That path renders a
  static image via code execution.

## The block
Put ONLY the JSON inside the fence — no prose, no trailing comments:

~~~
```chart
{
  "type": "line",
  "title": "Weekly runs",
  "x": "week",
  "series": [
    { "key": "runs", "label": "Total runs" },
    { "key": "useful", "label": "Useful runs", "colorIndex": 2 }
  ],
  "data": [
    { "week": "W1", "runs": 120, "useful": 88 },
    { "week": "W2", "runs": 143, "useful": 101 },
    { "week": "W3", "runs": 138, "useful": 110 }
  ]
}
```
~~~

## Schema
- `type` *(required)*: one of `line`, `bar`, `area`, `scatter`, `pie`.
- `x` *(required)*: the data key for the category / x-axis (for `pie`, the slice
  name). Must be an **identifier** (letters, digits, `_`, `-`; starts with a
  letter/underscore) and must appear in every data row.
- `series` *(required)*: 1–8 entries, each `{ "key", "label"?, "colorIndex"? }`.
  - `key`: identifier, present in the data rows, distinct from `x`.
  - `label`: display name (free text; optional, defaults to `key`).
  - `colorIndex`: brand palette slot **1–8** (optional; defaults to the series
    position). **You cannot set a raw color** — color always comes from the
    brand tokens, which keeps charts on-brand and theme-correct.
- `data` *(required)*: 1–1000 row objects; values are numbers or short strings.
  Use `null`/omit for gaps.
- `stacked` *(optional)*: `true` to stack `bar`/`area` series.
- `title` *(optional)*: short caption shown above the chart.

## Keep it calm
- **1–4 series** is the norm; the palette starts monochrome and the first slots
  are the most distinct. Don't pack 8 series into one chart — split or aggregate.
- Let the data carry the color; don't set `colorIndex` unless you're encoding a
  specific meaning (e.g. reusing slot 1 for the primary series across charts).
- For **scatter**, series automatically get distinct marker shapes (a colorblind
  secondary encoding) — no action needed.
- One series needs no legend; the `title` names it.

## If it doesn't render
An invalid or incomplete spec shows as a plain JSON code block instead of a
chart (never an error). Common fixes: `x`/`series[].key` must be identifiers and
present in the data; keep to the five `type` values; ≤ 8 series, ≤ 1000 rows.
