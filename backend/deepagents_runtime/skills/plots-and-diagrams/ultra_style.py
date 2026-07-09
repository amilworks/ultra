"""Ultra brand matplotlib style + data palette.

This is the single source of truth for how a figure generated inside the
code-execution sandbox looks, so plots read as one calm system with the rest
of the Ultra app (near-monochrome, ink-forward, hue reserved for meaning).

Design contract
---------------
* Colors match the frontend ``--chart-*`` tokens (frontend/src/styles.css). A
  data series is the *same* color in a matplotlib figure and in a recharts
  dashboard. matplotlib figures always export **light-on-white**, so this module
  uses the LIGHT column of the palette; keep it in sync with the CSS tokens.
* The palette is vibrant but disciplined: the CVD-tuned hue/luminance structure
  with chroma ~1.4x the calm original, so charts have life without the colorblind
  collisions a from-scratch bright palette causes. It leads with blue (a single
  series is blue); graphite is the neutral/context slot for greying data out.
* The palette is colorblind-validated (Machado-2009 + WCAG, via the dataviz
  validator). See ``references/palette.md`` in this skill for the proofs and the
  residual weak pairs that require a marker-shape secondary encoding.

Usage (inside the sandbox)
--------------------------
    import sys; sys.path.insert(0, "/workspace")   # or "/opt/ultra_style" if baked
    from ultra_style import apply_ultra_style, PALETTE, highlight, sequential_cmap
    apply_ultra_style()          # calm rcParams, light-on-white, 300 DPI

Import is dependency-light: matplotlib/seaborn are imported lazily inside the
functions that need them, so ``import ultra_style`` (for constants/tests)
succeeds even in the lean worker where matplotlib is not installed.
"""

from __future__ import annotations

# --- Categorical palette -----------------------------------------------------
# Slot -> (light hex, dark hex). Same hue per slot across themes; only L/C shift.
# Vibrant set: the CVD-tuned hue/luminance structure with chroma boosted ~1.4x
# (livelier AND slightly more colorblind-robust than the calm original). The
# palette LEADS WITH BLUE, so a single-series chart is blue, not grey. Graphite
# (slot 8) is the neutral/context anchor (used to grey out de-emphasized data).
# Colorblind-validated (dataviz Machado-2009 + WCAG): adjacent CVD dE >= 39,
# all-pairs dE ~9 (floor band -> scatter uses marker shapes), all >= 3:1 on both
# surfaces. Keep in sync with frontend --chart-* and references/palette.md.
_CATEGORICAL: tuple[tuple[str, str, str], ...] = (
    # name           light      dark
    ("blue",        "#1e65bd", "#4a92f2"),  # 1  primary — single series leads here
    ("terracotta",  "#c14701", "#e66933"),  # 2  warm counterpoint (blue/orange = CVD-robust)
    ("teal",        "#00948d", "#00aca4"),  # 3  cyan-green
    ("ochre",       "#b97c00", "#c18200"),  # 4  gold
    ("rose",        "#bd394e", "#d34b5d"),  # 5  crimson-rose (NOT a status color)
    ("violet",      "#6e34a0", "#935ccb"),  # 6  royal violet
    ("green",       "#399743", "#4eaa55"),  # 7  green
    ("graphite",    "#3c414b", "#b8bec8"),  # 8  neutral / context anchor
)

#: Light-mode categorical colors, in fixed order. Assign in sequence, never
#: cycle: a 9th series folds into "other", small multiples, or a facet.
PALETTE: list[str] = [light for _name, light, _dark in _CATEGORICAL]
PALETTE_DARK: list[str] = [dark for _name, _light, dark in _CATEGORICAL]
PALETTE_NAMES: list[str] = [name for name, _l, _d in _CATEGORICAL]

# --- Reserved / status colors (reuse the app's exact meanings) ----------------
INK = "#171717"          # titles, primary text  (== --text-main)
MUTED = "#737373"        # axis + tick labels    (== --text-muted)
CONTEXT = "#9a9893"      # greyed-out context series (de-emphasized data)
DANGER = "#c62828"       # genuine failure / bad  (== --danger)  -- never "series N"
WARNING = "#b45309"      # caveat / threshold     (the app's one sanctioned amber)

# --- Diagram / illustration palette (outlined shapes on white; NOT data marks) --
# Soft pastels for hand-authored diagrams (flow/block/architecture) where each
# shape has an ink outline carrying the edge. Do NOT use these as line/bar/point
# colors -- they fail contrast on white and wash out. See SKILL.md "Diagrams".
DIAGRAM_PALETTE = ["#86c7ea", "#a3dd9b", "#f19ca6", "#b39ddb", "#43cdaa", "#f3c98b"]
DIAGRAM_NAMES = ["blue", "green", "pink", "violet", "teal", "amber"]
DIAGRAM_OUTLINE = INK    # every diagram shape gets a ~1.6px ink stroke + ink text

# --- Sequential ramp (single hue, low -> high magnitude) ---------------------
SEQUENTIAL: list[str] = [
    "#e4eefb", "#c1d6f2", "#96b7e2", "#6791ca", "#3a6aa8", "#1f4577", "#112845",
]

# --- Diverging ramp (cool <- neutral -> warm; neutral gray midpoint) ---------
DIVERGING: list[str] = [
    "#275591", "#6488b9", "#b0c3db", "#eae7e4", "#e0b6a4", "#bb7051", "#9a4729",
]

# --- Neutral chrome (light-on-white export) ----------------------------------
GRID = INK          # applied at low alpha (see GRID_ALPHA)
GRID_ALPHA = 0.08   # whisper-quiet, == --line
AXIS_EDGE = MUTED

_FONT_STACK = ["Inter", "DejaVu Sans", "sans-serif"]
# Computer Modern (LaTeX look) via matplotlib's bundled cmr10 — no TeX install,
# no font asset. STIX/DejaVu Serif are fuller-glyph fallbacks.
_SERIF_STACK = ["cmr10", "Computer Modern Roman", "STIXGeneral", "DejaVu Serif", "serif"]


def _register_bundled_fonts() -> str:
    """Register a bundled Inter TTF if present; return the resolved family.

    The sandbox is offline + read-only and has no Inter installed (only the
    matplotlib-bundled DejaVu/STIX). If Inter .ttf files are shipped in this
    skill's ``assets/`` dir they are registered at runtime via font_manager;
    otherwise we fall back to DejaVu Sans (matplotlib's default sans). Either
    way the figure renders -- Inter just makes it match the app 1:1.
    """
    try:
        import os
        from matplotlib import font_manager
    except Exception:
        return "DejaVu Sans"

    assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    registered = False
    if os.path.isdir(assets):
        for fn in os.listdir(assets):
            if fn.lower().endswith((".ttf", ".otf")) and "inter" in fn.lower():
                try:
                    font_manager.fontManager.addfont(os.path.join(assets, fn))
                    registered = True
                except Exception:
                    pass
    if registered:
        return "Inter"
    # Is Inter already on the system (e.g. baked into the image)?
    try:
        names = {f.name for f in font_manager.fontManager.ttflist}
        if "Inter" in names:
            return "Inter"
    except Exception:
        pass
    return "DejaVu Sans"


def apply_ultra_style(dark: bool = False, font: str = "latex") -> None:
    """Set the calm Ultra rcParams. Defaults to light-on-white for export.

    ``font``:
      - ``"latex"`` (default): the professional academic look — Computer Modern
        via matplotlib's bundled ``cmr10`` plus the ``cm`` math font. No LaTeX
        install and no font asset needed; renders offline in the sandbox.
      - ``"sans"``: Inter (if available) else DejaVu Sans, to match the app UI.

    ``dark=True`` is provided for parity/preview but static figures shipped to
    chat should stay light-on-white (a PNG can't follow the viewer's theme).
    """
    import matplotlib as mpl
    from cycler import cycler

    if font == "sans":
        font_rc = {
            "font.family": _register_bundled_fonts(),
            "font.sans-serif": _FONT_STACK,
            "mathtext.fontset": "dejavusans",
            "axes.titleweight": "medium",
        }
    else:  # "latex" — Computer Modern, no TeX binary required
        font_rc = {
            "font.family": "serif",
            "font.serif": _SERIF_STACK,
            "mathtext.fontset": "cm",
            # render tick numbers through mathtext (cm) so the minus sign matches;
            # cmr10 lacks U+2212, so also fall back to ASCII minus elsewhere.
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "axes.titleweight": "normal",
        }

    if dark:
        bg, ink, muted, grid_c, grid_a = "#171717", "#f5f5f5", "#a1a1aa", "#ffffff", 0.10
        cycle = PALETTE_DARK
    else:
        bg, ink, muted, grid_c, grid_a = "#ffffff", INK, MUTED, GRID, GRID_ALPHA
        cycle = PALETTE

    mpl.rcParams.update({
        # canvas
        "figure.facecolor": bg, "axes.facecolor": bg, "savefig.facecolor": bg,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
        "figure.dpi": 300, "savefig.dpi": 300, "figure.figsize": (6.4, 4.0),
        # type (font.family / weight come from font_rc below)
        "font.size": 11, "axes.titlesize": 12.5,
        "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.titlecolor": ink, "axes.labelcolor": ink, "text.color": ink,
        # despine: data leads, chrome recedes
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": muted, "axes.linewidth": 0.8,
        # Recede the chrome (grid + tick marks stay quiet) but keep all TEXT
        # black for readability — tick labels get their own color, not the
        # muted tick-mark color.
        "xtick.color": muted, "ytick.color": muted,
        "xtick.labelcolor": ink, "ytick.labelcolor": ink,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.major.width": 0.8,
        # whisper-quiet grid, y-only, behind the data
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": grid_c, "grid.alpha": grid_a, "grid.linewidth": 0.7,
        # legend: frameless, unobtrusive (prefer direct labels)
        "legend.frameon": False, "legend.fontsize": 10, "legend.handlelength": 1.4,
        # marks
        "lines.linewidth": 2.0, "lines.markersize": 6, "patch.linewidth": 0.0,
        "scatter.marker": "o",
        "axes.prop_cycle": cycler(color=cycle),
        **font_rc,
    })


def highlight(n: int, focus: int = 0, dark: bool = False) -> list[str]:
    """Colors for ``n`` series with only ``focus`` colored; the rest greyed.

    The single most calm-honoring move: color the focus series, grey the
    context. Use for "series X vs everything else" charts.
    """
    pal = PALETTE_DARK if dark else PALETTE
    return [pal[i % len(pal)] if i == focus else CONTEXT for i in range(n)]


def _cmap(name: str, stops: list[str]):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, stops)


def sequential_cmap(name: str = "ultra_seq"):
    """Single-hue sequential colormap for heatmaps / continuous magnitude."""
    return _cmap(name, SEQUENTIAL)


def diverging_cmap(name: str = "ultra_div"):
    """Diverging colormap (cool<->neutral<->warm) for signed values.

    Note: the neutral midpoint is near the light page background, so on a
    light canvas rely on cell borders (or a colorbar) to read the zero band.
    """
    return _cmap(name, DIVERGING)


def apply_seaborn_defaults(dark: bool = False) -> None:
    """Best-effort seaborn palette/context to match, if seaborn is present."""
    try:
        import seaborn as sns
    except Exception:
        return
    sns.set_palette(PALETTE_DARK if dark else PALETTE)
    sns.set_context("notebook")
    sns.set_style("ticks", {"axes.grid": True, "grid.linestyle": "-"})
