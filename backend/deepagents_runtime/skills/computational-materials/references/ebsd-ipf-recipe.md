# EBSD / IPF — the correct recipe (use orix; do NOT hand-roll)

Hand-rolling IPF coloring is the single most common EBSD mistake and it produces a
**plausible-but-wrong** figure. Use `orix` (installed in the sandbox). It encodes
the crystal symmetry, the fundamental sector, and the standard TSL color scheme
correctly. `kikuchipy` handles raw patterns; `diffsims` builds dictionaries.

## The anti-pattern that keeps happening (do not do this)
```python
# WRONG — "sort so x<=y<=z, then R=x, G=y, B=z":
#   001 -> (0,0,1) -> BLUE     101 -> (0,.71,.71) -> CYAN     111 -> (.58,.58,.58) -> WHITE
# This paints the whole triangle blue/cyan/white. It is NOT the IPF scheme.
```
The **correct** cubic IPF-Z key is **001 = red, 101 = green, 111 = blue** (TSL/EDAX).
If your color key isn't a red→green→blue triangle, you hand-rolled it — stop and use orix.

## Color key + coloring orientations
```python
import numpy as np
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d
from orix.plot import IPFColorKeyTSL

pg = symmetry.Oh                       # cubic m-3m (state the point group for YOUR phase)
sample_dir = Vector3d.zvector()        # ND for an "IPF-Z" map (use xvector()/yvector() for RD/TD)

ckey = IPFColorKeyTSL(pg, direction=sample_dir)

# 1) the color key figure (correct fundamental sector + TSL colors, rendered by orix)
fig = ckey.plot(return_figure=True)
fig.savefig("/outputs/ipf_key.png", dpi=300, bbox_inches="tight")

# 2) color real orientations. Bunge Euler (phi1, Phi, phi2) in DEGREES here:
euler_deg = np.array([[0,0,0],[0,45,0],[35,45,0],[90,35,45]])  # Cube, Goss, Brass, Copper (FCC)
ori = Orientation.from_euler(euler_deg, symmetry=pg, degrees=True)
rgb = ckey.orientation2color(ori)      # (N, 3) in [0,1] — the correct IPF colors
# -> Cube ND=[001] is RED; a <111>-ND orientation is BLUE; <101>-ND is GREEN.
```

## IPF map from a CrystalMap (per-pixel)
```python
# xmap: an orix CrystalMap (e.g. from kikuchipy indexing or orix.io.load)
ckey = IPFColorKeyTSL(xmap.phases[0].point_group, direction=Vector3d.zvector())
rgb = ckey.orientation2color(xmap.orientations)
fig = xmap.plot(rgb, return_figure=True)      # correct IPF-Z map
fig.savefig("/outputs/ipf_map.png", dpi=300, bbox_inches="tight")
```

## Self-check (catches the hand-rolled bug before you ship the figure)
```python
# Cube {001}<100> has ND = [001] -> must be RED (not blue). Assert it.
cube = Orientation.from_euler([[0,0,0]], symmetry=pg, degrees=True)
r,g,b = ckey.orientation2color(cube)[0]
assert r > 0.8 and g < 0.3 and b < 0.3, f"IPF broken: Cube/ND should be red, got {(r,g,b)}"
```
Also eyeball the key: red corner at 001, green at 101, blue at 111. If it's all blue/cyan, it's wrong.

## Related outputs (also orix — don't hand-roll)
- **Pole figure / IPF density:** `orix.plot` (`plot_pole_figure`-style via `Orientation`/`Vector3d`);
  report intensity in **MRD** vs a uniform reference, never "strong" unquantified.
- **Misorientation:** `orix` misorientation on `Misorientation`/`Orientation`; compare the
  boundary misorientation-angle distribution to the **Mackenzie** random baseline for the symmetry
  (deviation = real texture / special boundaries).
- **Raw Kikuchi → orientations:** `kikuchipy` (state Hough vs dictionary indexing + parameters);
  `diffsims` for the simulated dictionary.

## Conventions to STATE in the write-up (getting these wrong flips colors)
- Point group / Laue class of the phase (e.g. Oh for FCC/BCC cubic).
- Which sample direction the IPF is for (ND=IPF-Z, RD=IPF-X, TD=IPF-Y).
- The orientation convention orix used (`from_euler` direction: `lab2crystal` vs `crystal2lab`)
  and whether Euler angles were degrees or radians. A transpose/convention slip silently
  rotates every color.
