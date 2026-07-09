"""Regression guard for the plots-and-diagrams brand style module.

Keeps the shipped palette + calm rcParams from silently drifting. matplotlib is
in the [imaging] extra (not the lean worker), so the rcParams checks skip when it
is absent; the pure-constant checks always run.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_MOD = (
    Path(__file__).resolve().parents[1]
    / "skills" / "plots-and-diagrams" / "ultra_style.py"
)

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


@pytest.fixture(scope="module")
def us():
    spec = importlib.util.spec_from_file_location("ultra_style", _MOD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_palette_shape_and_hex(us):
    assert len(us.PALETTE) == 8 and len(us.PALETTE_DARK) == 8
    assert len(us.PALETTE_NAMES) == 8
    for hexes in (us.PALETTE, us.PALETTE_DARK):
        assert all(_HEX.match(h) for h in hexes)
    # slot 1 is the graphite anchor; slot 4 is the CVD-fixed teal.
    assert us.PALETTE[0] == "#3c414b"
    assert us.PALETTE_DARK[3] == "#00958e"


def test_reserved_status_colors_match_app(us):
    assert us.DANGER == "#c62828"
    assert us.WARNING == "#b45309"


def test_sequential_and_diverging_present(us):
    assert len(us.SEQUENTIAL) >= 5 and all(_HEX.match(h) for h in us.SEQUENTIAL)
    assert len(us.DIVERGING) >= 5 and all(_HEX.match(h) for h in us.DIVERGING)


def test_highlight_greys_the_context(us):
    colors = us.highlight(4, focus=1)
    assert colors[1] == us.PALETTE[1]
    assert colors[0] == colors[2] == colors[3] == us.CONTEXT


def test_latex_style_is_the_default(us):
    mpl = pytest.importorskip("matplotlib")
    us.apply_ultra_style()  # default font="latex"
    rc = mpl.rcParams
    assert rc["font.family"] == ["serif"]
    assert rc["font.serif"][0] == "cmr10"
    assert rc["mathtext.fontset"] == "cm"
    # cmr10 lacks U+2212 — these two keep negative ticks clean.
    assert rc["axes.formatter.use_mathtext"] is True
    assert rc["axes.unicode_minus"] is False


def test_calm_chrome_black_text(us):
    mpl = pytest.importorskip("matplotlib")
    us.apply_ultra_style()
    rc = mpl.rcParams
    # text black, tick-marks + grid recede
    assert rc["xtick.labelcolor"] == "#171717"
    assert rc["ytick.labelcolor"] == "#171717"
    assert rc["axes.labelcolor"] == "#171717"
    assert rc["grid.alpha"] == pytest.approx(0.08)
    assert rc["axes.spines.top"] is False and rc["axes.spines.right"] is False
    # brand categorical cycle, in order
    assert rc["axes.prop_cycle"].by_key()["color"] == us.PALETTE


def test_sans_and_dark_modes(us):
    mpl = pytest.importorskip("matplotlib")
    us.apply_ultra_style(font="sans")
    assert mpl.rcParams["mathtext.fontset"] == "dejavusans"
    us.apply_ultra_style(dark=True)
    assert mpl.rcParams["axes.prop_cycle"].by_key()["color"] == us.PALETTE_DARK
    assert mpl.rcParams["figure.facecolor"] == "#171717"
