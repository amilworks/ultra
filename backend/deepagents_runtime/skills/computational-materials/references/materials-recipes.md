# Microstructure and EBSD vetted recipes

Copy these named-library recipes rather than reimplementing them. Each includes the
plausible-but-wrong shortcut and a self-check. IPF coloring has its own
[recipe](ebsd-ipf-recipe.md).

The legacy structure, CALPHAD, and Magpie recipes moved to
`/skills/materials-structure-thermo/references/structure-thermo-recipes.md`. XRD moved to
`/skills/materials-characterization/references/xrd-recipes.md`.

## Symmetry-reduced misorientation and random baseline (`orix`)

**Trap:** subtracting Euler angles or taking a raw quaternion angle without crystal symmetry.
Those values can run to 180 degrees and make ordinary boundaries appear special.

```python
import numpy as np
from orix.quaternion import Misorientation, Orientation, symmetry

pg = symmetry.Oh
mis = Misorientation(~o1 * o2)  # o1 and o2 are Orientation instances
mis.symmetry = (pg, pg)
angles_deg = np.rad2deg(mis.reduce().angle)

r = Orientation.random(200_000)
r.symmetry = pg
s = Orientation.random(200_000)
s.symmetry = pg
random_mis = Misorientation(~r * s)
random_mis.symmetry = (pg, pg)
null_angles_deg = np.rad2deg(random_mis.reduce().angle)
```

**Self-check:** cubic disorientation is capped near 62.8 degrees and the large-sample Mackenzie
median is near 43 degrees:

```python
assert null_angles_deg.max() < 62.9
assert 40 < np.median(null_angles_deg) < 50
```

Record the random sample count and random-state control if the installed API provides one. Compare
the measured distribution with the null; do not call the null itself a measured grain-boundary
distribution.

## Texture strength relative to random (`orix`)

**Trap:** interpreting raw pole-point density or a latitude/longitude histogram as MRD. A spherical
grid is not equal-area, and raw density has no random reference.

Use the installed `orix` pole-density/ODF API and confirm its live signature before adapting it.
Expand crystal-symmetry-equivalent poles in the crystal frame, rotate them to the sample frame,
and report multiples of random distribution (MRD) or a named texture index. Validate the complete
pipeline on `Orientation.random(N)`; an equal-area-weighted random reference should be consistent
with MRD 1 within its sampling uncertainty.

## Anisotropy-aware grain segmentation and stereology (`scikit-image`)

**Trap:** treating a thresholded foreground mask as individual grains, or reporting voxel counts
as cubic micrometers while ignoring slice spacing.

```python
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.measure import regionprops_table
from skimage.segmentation import clear_border, watershed

voxel_size = (dz, dy, dx)  # physical units, in array-axis order
distance = distance_transform_edt(foreground, sampling=voxel_size)
coordinates = peak_local_max(distance, labels=foreground, min_distance=marker_distance)
markers = np.zeros(foreground.shape, dtype=np.int32)
markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
labels_all = watershed(-distance, markers=markers, mask=foreground)
labels_interior = clear_border(labels_all)
props = regionprops_table(
    labels_interior,
    spacing=voxel_size,
    properties=("label", "area", "equivalent_diameter_area", "axis_major_length"),
)
```

`spacing=` requires a compatible scikit-image release. If unavailable, measure voxel counts and
apply the physical voxel volume explicitly, documenting the fallback. Before specimen analysis,
rasterize an ellipsoid with known physical semi-axes and require recovered volume within a stated
rasterization tolerance (typically a few percent at adequate resolution). Compare counts and
distributions across a marker-distance/threshold sweep and report how many labels `clear_border`
removed.

## Porosity and pore scale (`porespy`)

**Trap:** PoreSpy expects **True = void**. Inverting the mask reverses the material meaning while
still returning plausible numbers.

```python
import porespy as ps

void = void_mask.astype(bool)  # True only where pore/void
porosity = ps.metrics.porosity(void)
local_thickness = ps.filters.local_thickness(void)
```

Confirm any distribution/correlation API name against the installed PoreSpy version because its
result objects changed between major versions. Validate convention and scale on a labeled synthetic
void sphere: the measured void fraction must match the known construction within rasterization
tolerance and the local-thickness maximum must use the installed function's documented radius/
diameter convention. A simple equality with `void.mean()` does not detect mask inversion.
