"""Materials domain-correctness invariants (skip unless the sci stack is present).

Each check fails for the hand-rolled shortcut the computational-materials skill
warns against. Run inside the sandbox image where orix/pymatgen/etc. exist.
"""

from __future__ import annotations

import numpy as np
import pytest


def _close(rgb, target, tol=0.15):
    return all(abs(a - b) <= tol for a, b in zip(rgb, target))


def test_ipf_cubic_key_corners_are_rgb():
    """Cubic IPF-Z key: 001->red, 101->green, 111->blue.

    The naive 'sort xyz then R=x,G=y,B=z' mapping paints the triangle blue/cyan
    and FAILS this — which is exactly the bug this guards against.
    """
    orix = pytest.importorskip("orix")  # noqa: F841
    from orix.quaternion import Orientation, symmetry
    from orix.vector import Vector3d
    from orix.plot import IPFColorKeyTSL

    ckey = IPFColorKeyTSL(symmetry.Oh, direction=Vector3d.zvector())
    # Orientations whose sample-Z (ND) sits at each triangle corner:
    #   Cube (0,0,0) -> ND=[001];  Goss (0,45,0) -> ND=[101];  (30,54.7,45) -> ND=[111]
    euler = np.array([[0.0, 0.0, 0.0], [0.0, 45.0, 0.0], [30.0, 54.7, 45.0]])
    ori = Orientation.from_euler(euler, symmetry=symmetry.Oh, degrees=True)
    rgb = np.asarray(ckey.orientation2color(ori))

    assert _close(rgb[0], (1, 0, 0)), f"001/Cube should be RED, got {tuple(rgb[0])}"
    assert _close(rgb[1], (0, 1, 0)), f"101/Goss should be GREEN, got {tuple(rgb[1])}"
    assert _close(rgb[2], (0, 0, 1)), f"111 should be BLUE, got {tuple(rgb[2])}"


def test_ipf_not_blue_everywhere():
    """Guard the specific failure signature: the whole key is NOT blue-dominant.

    The hand-rolled mapping makes every direction blue-dominant (B is the largest
    channel almost everywhere). The correct key spans red+green+blue.
    """
    pytest.importorskip("orix")
    from orix.quaternion import Orientation, symmetry
    from orix.vector import Vector3d
    from orix.plot import IPFColorKeyTSL

    ckey = IPFColorKeyTSL(symmetry.Oh, direction=Vector3d.zvector())
    rng = np.random.default_rng(0)
    ori = Orientation.from_euler(rng.uniform(0, 360, (200, 3)), symmetry=symmetry.Oh, degrees=True)
    rgb = np.asarray(ckey.orientation2color(ori))
    dominant = np.argmax(rgb, axis=1)  # 0=R,1=G,2=B
    frac_blue = float(np.mean(dominant == 2))
    # A correct key over random orientations is nowhere near all-blue.
    assert frac_blue < 0.6, f"IPF looks blue-dominant ({frac_blue:.0%}) — likely hand-rolled"


def test_random_misorientation_matches_mackenzie():
    """Random cubic misorientations obey the Mackenzie distribution.

    Disorientation angle is capped at ~62.8° for cubic (m-3m) and the
    distribution peaks near 45° (median ~43-44°). A hand-rolled misorientation
    that forgets symmetry reduction runs to 180° and fails the cap.
    """
    pytest.importorskip("orix")
    from orix.quaternion import Orientation, Misorientation, symmetry

    pg = symmetry.Oh
    # verify-pass correction: set .symmetry as an attribute, do NOT pass symmetry= to random()
    a = Orientation.random(20000); a.symmetry = pg
    b = Orientation.random(20000); b.symmetry = pg
    mis = Misorientation(~a * b); mis.symmetry = (pg, pg)
    mis = mis.map_into_symmetry_reduced_zone()
    ang = np.rad2deg(mis.angle)
    assert ang.max() < 62.9, f"cubic disorientation must cap at 62.8°, got {ang.max():.2f}"
    assert 40 < float(np.median(ang)) < 50, f"Mackenzie median ~43-44°, got {np.median(ang):.2f}"
