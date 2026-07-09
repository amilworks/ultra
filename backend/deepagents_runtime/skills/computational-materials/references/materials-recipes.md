# Materials — vetted recipes (use the library; don't hand-roll)

Copy these instead of reinventing the analysis. Each has the **trap** (the
plausible-but-wrong shortcut), the **correct** library call, and a **self-check**
you should actually run in-sandbox. EBSD/IPF colouring is in
[ebsd-ipf-recipe.md](ebsd-ipf-recipe.md). Recipes were adversarially reviewed;
where noted, confirm a version-fragile API against the installed package first.

## Misorientation vs the Mackenzie baseline (orix) — VALIDATED
**Trap:** a hand-rolled quaternion/Euler misorientation without symmetry reduction
— angles run to 180° and every boundary looks "high-angle".
```python
import numpy as np
from orix.quaternion import Orientation, Misorientation, symmetry
pg = symmetry.Oh
mis = Misorientation(~o1 * o2); mis.symmetry = (pg, pg)     # o1,o2: Orientations
ang = np.rad2deg(mis.map_into_symmetry_reduced_zone().angle)  # disorientation angles
# random null (Mackenzie): set .symmetry as an ATTRIBUTE — do NOT pass symmetry= to random()
r = Orientation.random(200000); r.symmetry = pg
s = Orientation.random(200000); s.symmetry = pg
null = Misorientation(~r * s); null.symmetry = (pg, pg)
null_ang = np.rad2deg(null.map_into_symmetry_reduced_zone().angle)
```
**Self-check:** cubic disorientation caps at **62.8°** and the Mackenzie median is
**~43°**: `assert null_ang.max() < 62.9 and 40 < np.median(null_ang) < 50`. Compare
your measured boundary distribution to `null_ang`; deviation = real texture/special boundaries.

## Pole figure / ODF texture strength in MRD (orix) — CONFIRM API LIVE
**Trap:** raw pole-point density with no random baseline → "strong texture" that's
just sampling. Report **multiples of random distribution (MRD)** vs uniform.
```python
from orix.quaternion import Orientation, symmetry
# expand crystal-symmetry equivalents in the CRYSTAL frame FIRST, then rotate to sample:
poles = ~ori * m.symmetrise(unique=True)     # ori: Orientation; m: Miller/Vector3d (crystal dir)
# then orix's pole density (MRD-normalised). Confirm signature/kwargs in the installed orix:
#   e.g. Vector3d(poles).pole_density_function(...)  -> returns MRD histogram
```
**Self-check:** a set of `Orientation.random(N)` (build then `.symmetry = pg`) must give
**MRD ≈ 1 everywhere** — but weight by solid angle, not a raw `.mean()` over a lon/lat grid
(pole over-weighting gives false failures). Verify `pole_density_function`'s return + kwargs
against the installed version before trusting.

## Grain segmentation + stereology (scikit-image) — needs skimage ≥ 0.24 for `spacing=`
**Trap:** a bare intensity threshold for "grains", and grain sizes in **pixels**
(ignoring anisotropic TriBeam z-spacing).
```python
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, clear_border
from skimage.measure import label, regionprops_table
vox = (dz, dy, dx)                                   # physical voxel size (µm)
dist = distance_transform_edt(fg, sampling=vox)      # anisotropy-aware
coords = peak_local_max(dist, labels=fg, min_distance=5)
mk = np.zeros(fg.shape, int); mk[tuple(coords.T)] = np.arange(1, len(coords) + 1)
grains = watershed(-dist, markers=mk, mask=fg)
# grains = clear_border(grains)   # drop edge-touching grains; report how many
props = regionprops_table(grains, spacing=vox,       # spacing= requires scikit-image >= 0.24
    properties=("label", "area", "equivalent_diameter_area", "axis_major_length"))
```
If skimage < 0.24: drop `spacing=` and multiply voxel counts by `np.prod(vox)` yourself.
**Self-check:** on a synthetic rasterised ellipsoid, `props["area"]` (with `spacing=`) must equal
`(4/3)π·a·b·c` within ~5%; without spacing it's off by exactly `dz·dy·dx` — catching the
pixels-not-µm bug. Compute any watershed-vs-connected-components count check *before* `clear_border`.

## CALPHAD equilibrium phase fractions (pycalphad) — needs a NAMED .tdb
**Trap:** an ad-hoc "stability score". No TDB is bundled — the user must supply one; never fabricate.
```python
from pycalphad import Database, equilibrium, variables as v
dbf = Database("Ni-Al.tdb")                          # a NAMED database the user provides
comps = ["NI", "AL", "VA"]                           # include VA
phases = list(dbf.phases.keys())
eq = equilibrium(dbf, comps, phases, {v.X("AL"): 0.15, v.T: 1073, v.P: 101325, v.N: 1})
```
**Self-check:** phase fractions are a partition — over **converged** points (`eq.NP` not all-NaN),
`np.nansum(eq.NP, axis=...) ≈ 1` and each in [0,1]. Corroborate γ′ fraction against EBSD/EDS.

## Space-group / phase ID (pymatgen + spglib) — VALIDATED invariant
**Trap:** identifying a phase "by eye" from lattice parameter; collapsing ordered L1₂ γ′ to FCC.
```python
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
sga = SpacegroupAnalyzer(struct, symprec=0.01, angle_tolerance=5)
sg_num = sga.get_space_group_number()   # sweep symprec (0.001..0.1) and report stability
```
**Self-check:** ordered **Ni₃Al L1₂ → 221 (Pm-3m)**; disordered FCC Cu/Ni → **225 (Fm-3m)**.
Getting 225 for ordered Ni₃Al means the ordering (Al at corners, Ni at faces) was lost.

## Magpie composition featurization (matminer)
**Trap:** a hand-averaged atomic-property vector (loses min/max/range/mode/avg-dev spread).
```python
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
feat = ElementProperty.from_preset("magpie")
x = feat.featurize(Composition("Ni3Al")); labels = feat.feature_labels()   # 132 = 22 props × 6 stats
```
(Needs `NUMBA_CPU_NAME=generic` on arm64 to avoid SIGILL — already set in the sandbox.)
**Self-check:** `len(labels) == 132` and the labels include `avg_dev`, `mode`, `range` (a hand-averaged
vector has none). Do **not** cross-check Magpie's mean AtomicWeight against pymatgen masses at 1e-6 —
Magpie ships its own Ward-2016 table; use a loose (~1 amu) tolerance if you compare at all.

## Porosity / pore metrics (porespy) — CONFIRM metric names LIVE
**Trap #1 (the footgun):** porespy's convention is **True = void**. Invert it and every metric flips.
**Trap #2:** a bare threshold "porosity".
```python
import porespy as ps
void = im_bool                       # MUST be True where pore/void
phi = ps.metrics.porosity(void)      # void fraction
lt = ps.filters.local_thickness(void)
# pore_size_distribution / two_point_correlation were renamed across porespy v1->v2 —
# confirm the exact names + return objects in the installed version before use.
```
**Self-check (NOT the tautological `porosity == void.mean()`):** for a nominally dense alloy
`assert phi < 0.5` (mislabeling solid as void trips it), and on a synthetic sphere of radius r,
`local_thickness(sphere).max() ≈ r` (not the diameter).
