---
name: plots-and-diagrams
description: How to render a chart, plot, graph, figure, heatmap, or diagram in the code-execution sandbox so it looks calm and native to Ultra. Covers chart-type choice (form follows data), the shipped Ultra matplotlib style + brand color palette, whisper-quiet chrome, direct labeling over legends, colorblind-safe color used only to encode meaning, light-on-white 300-DPI export to /outputs, and the ONE offline-safe way to make diagrams (render to an image — mermaid/graphviz do not work in the sandbox). Use when generating any data visualization or diagram as an image via code execution. For where files go, captioning, and report structure, use scientific-reporting.
---

# Plots and Diagrams

## When to use
Read this before writing the first line of plotting or diagram code — a chart,
graph, plot, figure, heatmap, distribution, or a flow/architecture/state
diagram that will be rendered as an image. It governs how a figure *looks*, so
every figure reads as one calm system with the rest of Ultra.

## Scope (and what belongs elsewhere)
This skill owns **how a figure looks and how it's produced**: chart choice, the
brand style, color discipline, export. It does **not** own artifact hygiene —
where files live (`/outputs` vs `/workspace/diagnostics`), how a figure is
captioned and referenced from the write-up, or naming discipline. Those are
**scientific-reporting**; read it too whenever a figure ships in a report.

Target = **static images rendered with matplotlib/seaborn in the sandbox**,
harvested from `/outputs` and shown inline in chat. Do **not** use plotly
(its static export needs `kaleido`, which is absent) and do not hand-write web
chart code — the browser dashboards are a separate runtime.

## Apply the Ultra style first
The brand style + palette live in `ultra_style.py`, shipped in this skill. It
sets calm rcParams (quiet grid, despined axes, muted ticks, brand color cycle),
light-on-white 300-DPI export, and colorblind-validated colors that **match the
app's `--chart-*` tokens** — a series is the same color here and in the UI.

Preferred: copy the module into the sandbox workspace and import it. From the
agent side, read `/skills/plots-and-diagrams/ultra_style.py` and write it to the
run workspace, then in sandbox code:

```python
import sys; sys.path.insert(0, "/workspace")   # or "/opt/ultra_style" if image-baked
from ultra_style import apply_ultra_style, PALETTE, highlight, sequential_cmap, DANGER
apply_ultra_style()            # LaTeX/Computer Modern, light-on-white, 300 DPI
# apply_ultra_style(font="sans")   # Inter/sans instead, to match the app UI
```

Fallback (if the module isn't reachable) — paste this; it is what the module applies:

```python
import matplotlib as mpl
from cycler import cycler
mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight",
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#737373", "axes.linewidth": 0.8,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "grid.color": "#171717", "grid.alpha": 0.08, "grid.linewidth": 0.7,
    "xtick.color": "#737373", "ytick.color": "#737373",
    "xtick.labelcolor": "#171717", "ytick.labelcolor": "#171717",
    "axes.labelcolor": "#171717", "text.color": "#171717",
    "axes.titlecolor": "#171717", "axes.titlesize": 12.5, "axes.titleweight": "medium",
    "legend.frameon": False, "lines.linewidth": 2.0,
})
# Ultra brand categorical palette (light-on-white) — same as the app --chart-* tokens.
# Leads with blue; graphite (last) is the neutral/context slot.
mpl.rcParams["axes.prop_cycle"] = cycler(color=[
    "#1e65bd", "#c14701", "#00948d", "#b97c00",
    "#bd394e", "#6e34a0", "#399743", "#3c414b"])
```

**Fonts.** The default is the professional **LaTeX / Computer Modern** look —
`apply_ultra_style()` uses matplotlib's bundled `cmr10` with the `cm` math font,
so `$...$` in titles/labels renders in Computer Modern. This needs **no LaTeX
install** (do NOT set `text.usetex=True` — the sandbox has no TeX binary) and no
font asset. Pass `font="sans"` for Inter/DejaVu Sans if you want the figure to
match the app UI instead. Never try to fetch or install a font (the sandbox is
offline and read-only).

For the LaTeX look without the helper, add these to the fallback rcParams:
`"font.family": "serif", "font.serif": ["cmr10", "STIXGeneral", "DejaVu Serif"],
"mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
"axes.unicode_minus": False`.

**Glyph coverage (LaTeX font).** `cmr10` covers ASCII, basic Latin, and math —
but **not** em/en-dashes (— –), curly quotes (‘ ’ “ ”), ×, or many Unicode
symbols. In titles/labels/annotations use plain ASCII punctuation (hyphen `-`,
straight quotes `"`) or wrap symbols in `$...$` math (`$\times$`, `$\pm$`);
otherwise the character silently drops (a matplotlib "missing glyph" warning).
The minus sign is already handled. If you truly need Unicode punctuation, use
`font="sans"` (full coverage).

## Choose the form (form follows data)
Pick from what the data *is*, not from habit:
- **Trend over a continuous axis (time)** → line. **Comparison across
  categories** → bar (horizontal when labels are long or numerous).
- **Ranking** → sorted horizontal bar. **Two-variable relationship** → scatter
  (add a fit line or marginal only if it earns space).
- **Distribution** → histogram / KDE / box / violin; **ECDF** when tails matter.
- **Part-to-whole** → a stacked bar; avoid pie beyond 2–3 slices, never 3-D or exploded.
- **2-D matrix / density / correlation** → heatmap with `sequential_cmap()`.
- **Uncertainty** → error bars or shaded bands (carry the ± from
  computational-experiment-rigor onto the figure — don't plot bare point estimates).
- **One series** → don't add color by default. **One number** → a stat callout,
  not a chart. But a monochrome single-series bar chart can look flat — when a
  chart is a showcase, add life *without* breaking the rules: color the bars **by
  value** with `sequential_cmap()` (taller = darker; this legitimately encodes
  magnitude), or give each bar its own categorical hue for a nominal set. Both
  beat a wall of identical bars.

## Keep it calm
The app is near-monochrome and editorial; figures match that.
- **Whisper-quiet gridlines**: thin, ~8% ink, behind the data, y-only — drop
  vertical gridlines. No box; keep at most the left + bottom spine (despined).
- **No chartjunk**: no 3-D, gradients, drop shadows, background fills, heavy
  borders, or a value label on every point. Every mark must encode data.
- **Direct labeling over legends**: label a line at its end, annotate the bar,
  put category names on the axis. A legend only when direct labels won't fit —
  then frameless.
- **Restraint**: few ticks, generous whitespace, consistent decimal precision
  no finer than the data supports.

## Multi-panel figures
Composite figures (a 2×2 of related views, panels A–D) are common and are where
most layout bugs live. A few rules keep them clean:
- **Use a managed layout**: `fig, axes = plt.subplots(2, 2, figsize=(11, 8.5),
  layout="constrained")`. It spaces panels and stops labels/titles/annotations
  from colliding with neighbours or the figure edge — prefer it over
  `tight_layout()`, and never hand-tune `subplots_adjust` first.
- **Every panel is a full figure**: each axis still gets its axis labels (with
  units) and a title. Do **not** leave the value axis of a bar chart unlabelled
  because "it's obvious" — name it (e.g. `Condition number \(\kappa\)`).
- **Label panels** A, B, C… in the top-left:
  `ax.text(-0.08, 1.06, "A", transform=ax.transAxes, weight="bold", fontsize=13)`.
- **Don't let annotations collide.** Put block/region labels at the *centre* of
  their region; keep corner counts (`n = 196`, `n_Γ = 14`) clear of other text
  and off the axis line. After saving, look at the render — if two labels touch,
  move or drop one. Overlapping text is the most common defect in these figures.
  When several labels want the *same* spot (limiting cases like `ρ → 1`, or a
  cluster of marked points), consolidate them into **one** text block or a compact
  legend — never stack overlapping callouts/arrows on one region.
- **Be wary of insets** (a small axes floating over another): they get cramped
  and collide at display/chat size. Prefer a **separate panel** unless the inset
  truly sits in empty space.

## Color = meaning
Color still earns its place — don't spray all 8 hues at a 2-series chart. The
palette **leads with blue**, so a single series is blue; add more hues only to
encode more series. Grey the context (`CONTEXT`), color the focus — use
`highlight(n, focus)`.
- **Categorical**: draw in fixed order from `PALETTE` (blue, terracotta, teal,
  ochre, rose, violet, green, graphite). **Keep to 1–4 for the common case** —
  they're the most distinct. At **5+ series, consider small multiples,
  aggregation, or direct labels** before using slots 5–8. Never cycle past 8 —
  fold extras into "other" or facet. `graphite` (slot 8) is the neutral for
  de-emphasis, not a loud series.
- **Sequential** (heatmaps, intensity) → `sequential_cmap()` (single-hue).
- **Diverging** (signed values) → `diverging_cmap()` (cool ↔ neutral ↔ warm).
- **Status is reserved**: red `DANGER` (#c62828) for genuine failure/negative,
  amber `WARNING` (#b45309) for a caveat/threshold. Never reuse them as a
  categorical "series N", and don't also use slot-5 rose as a series in a chart
  that already carries a red failure mark (they read alike). Also **don't spend
  red/amber on a neutral highlight** — a mean, an optimum, a marked point, a
  limiting case: emphasize those with ink/graphite (slot 1) or a categorical hue,
  and keep the alarm colors for things that are actually bad.
- **Colorblind safety**: the palette is validated colorblind-aware, but never
  let color be the *only* channel. For **scatter/bubble** (any two marks can
  neighbor) **add a marker shape** per series — the green (sage) vs rose pair is
  the one residual weak pair and needs it. Do not reorder the palette slots: the
  order is the CVD-safety mechanism.

## Export
- **Light-on-white, always** — a static PNG can't follow the viewer's theme;
  keep a clean white canvas that reads on any chat surface.
- **300 DPI** (style + sandbox `matplotlibrc` already enforce it), `bbox_inches="tight"`.
- **PNG** for raster figures, **SVG** for line art and diagrams. Save to
  `/outputs/` with a figure-hinting name (`revenue_by_quarter.png`,
  `pipeline_diagram.svg`) so the runner reliably harvests it as a figure.
  Files left elsewhere in `/workspace` are only kept if named in the reply.

## Diagrams (flow / architecture / state / DAG)
For structural (non-data) diagrams, use the **pastel diagram palette** —
`DIAGRAM_PALETTE` in `ultra_style` (soft blue/green/pink/violet/teal/amber) —
with an **ink outline** (`DIAGRAM_OUTLINE`, ~1.6px) and ink text on every shape.
Pastels belong here precisely *because* the outline carries the edge; never use
them as data marks (they wash out below 3:1 on white). Reserve saturated
red/amber for genuine status.
- **Render to an image in `/outputs`** — that is the only way it appears inline.
  A raw ```` ```mermaid ```` or DOT block **does not draw** in Ultra chat (no
  mermaid/Graphviz renderer), and the sandbox ships **no `dot` binary and no
  mermaid CLI**, so `graphviz`/`pydot`/`pygraphviz`/mermaid are **not available**.
- Working paths: **`networkx` + matplotlib** (both preinstalled) for graphs/DAGs
  with an explicit layout; or **hand-author a clean SVG** (neutral fills, ink
  strokes, Inter/DejaVu labels) and save it to `/outputs` — often the most
  calm-native result for flow/architecture diagrams.
- **Flowchart text:** put any sub-text *inside* its box (wrapped to the box
  width), never as a separate label beneath the box — beneath-box labels overrun
  into their neighbours and pile into an unreadable smear. Size each box to its
  text (or shorten the text) and space boxes with real gaps.
- Emit Mermaid/DOT *source* only when the destination is a document opened by a
  mermaid/DOT-capable viewer, not Ultra chat. When in doubt, render to SVG/PNG.

## Accessibility
- Every axis carries a **label with units** — including the value axis of a bar
  chart (don't skip it just because the bars are annotated); the title states what
  the figure shows and its key parameters (leave the report-side caption to
  scientific-reporting).
- Legends only when direct labeling can't do the job; keep them frameless.
- Never rely on color alone (see colorblind safety); ensure text/marks have
  enough contrast on the white canvas.
