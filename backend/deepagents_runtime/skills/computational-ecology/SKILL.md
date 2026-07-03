---
name: computational-ecology
description: Domain toolkit and rigor for geospatial and ecological analysis — vector/raster IO, CRS handling, spatial statistics, point patterns, landscape metrics, terrain/hydrology, remote-sensing indices, movement tracks, habitat connectivity, and species distribution modelling. Use for shapefiles, GeoPackage, GeoTIFF, NetCDF/COG rasters, reprojection, clustered-vs-random spatial tests, patch metrics, NDVI/EVI, zonal statistics, GPS telemetry, least-cost corridors, or occurrence prediction. Read before hand-rolling spatial weights, Moran's I, landscape metrics, NDVI loops, or resistance-surface nulls.
---

# Computational Ecology (geospatial)

## When to use
Read this before any geospatial or spatial-ecology analysis — spatial statistics, landscape
metrics, remote-sensing rasters, movement, connectivity, or species distribution. The point
is to use the field's actual tools and inferential safeguards instead of re-deriving weaker
surrogates from numpy/pandas, and to prove any spatial pattern you report is real rather than
an artifact of the weights matrix, the aggregation scale, or an aspatial null. This is domain
tooling; the numerical-rigor protocol in `/skills/computational-experiment-rigor/SKILL.md`
(uncertainty, thresholds, reproducibility, honest accounting) still applies on top of it.

## 0. Environment — one interpreter, and two install blockers
Everything runs in the default `python` (numpy 1.26.4) alongside the baked
scipy/pandas/scikit-image/scikit-learn/networkx/xarray/dask/zarr/pyarrow — no isolated env.
State the interpreter and package versions you used. Two packages are deliberately **not**
baked and must not be reached for: **`fiona`** (its sdist needs GDAL dev headers absent from
the image) and **`richdem`** (no py3.11 wheel). So do vector IO through the **`pyogrio`**
engine (`geopandas.read_file(..., engine="pyogrio")`), never fiona; and do terrain/hydrology
through **`pysheds`** (conditioning, flow direction/accumulation) and **`xarray-spatial`**
(`xrspatial`: slope/aspect/hillshade/curvature), never richdem. If you find yourself importing
fiona or richdem, stop — you are on an unbaked path.

## Protocol

### 1. Use the field-standard named tool, name the method, pin its parameters
Do not hand-roll a spatial primitive when the canonical library exists.
- **Vector/raster IO + CRS:** `geopandas`(pyogrio)/`shapely`/`rioxarray`/`rasterio`; reproject
  with `pyproj`. Reproject to an appropriate **equal-area or projected CRS before any
  distance/area/weights computation** — never compute distance or area in EPSG:4326 degrees.
  State the CRS by EPSG code.
- **Spatial autocorrelation & statistics:** build the weights explicitly with
  `libpysal.weights` (Queen/Rook/KNN/DistanceBand — state which, and k or the band + units)
  and row-standardize; then `esda` for global **Moran's I**, local **LISA**
  (`esda.Moran_Local`), **Getis-Ord G/G\*** (`esda.G_Local`). Do not reimplement I by hand.
- **Spatial regression:** `spreg` (SAR/spatial-error/GNS — name the model and the LM-lag/LM-
  error diagnostics that motivated it), not an OLS that ignores autocorrelated residuals.
- **Point patterns:** `pointpats` (Ripley's K/L, G, F, pair-correlation), not an eyeballed scatter.
- **Landscape metrics:** `pylandstats` (the Python FRAGSTATS analogue) — name the metrics
  (patch/edge density, mean patch area, contagion, LANDSCAPE aggregation) and the neighbor
  rule (4- vs 8-connectivity) and the classified raster.
- **Zonal statistics:** `rasterstats.zonal_stats` (state the statistic and nodata handling).
- **Remote sensing / raster time series:** load with `rioxarray` into `xarray`; spectral
  indices via `xarray-spatial` or explicit band math with the mapping stated (NDVI =
  (NIR−Red)/(NIR+Red); name EVI/NDWI/SAVI coefficients); reduce over time with named reductions.
- **Terrain/hydrology:** `pysheds` (pit-filling, D8 flow direction/accumulation, stream
  extraction) and `xarray-spatial` for slope/aspect/hillshade.
- **Movement ecology:** `movingpandas` `TrajectoryCollection` (state CRS, time index, speed/gap
  splitting), not differenced lat/lon columns.
- **Connectivity:** build the habitat graph on baked `networkx` (nodes = patches/cells, edges
  weighted by a stated resistance/cost); least-cost paths / corridors / betweenness by name.
- **SDM:** baked `scikit-learn` (name the estimator) with an explicit feature stack.
- Basemaps via `contextily` (needs network at map time — figures only), choropleth binning via
  `mapclassify` (name the scheme — Quantiles/FisherJenks/NaturalBreaks — and k).
Name the method at the level you are confident in; never fabricate a citation or benchmark number.

### 2. Validate against a principled spatial null, not an aspatial one
A spatial statistic is meaningless without the right null, and the right null is almost never IID.
- **Autocorrelation:** report Moran's I / G\* / LISA against the tool's **conditional-permutation**
  null (`permutations=999`, fixed seed); the analytic normal-approximation p is a fallback, not
  the headline. Report E[I] = −1/(n−1), observed I, and the pseudo p — a positive I is only
  "clustering" if it beats the permutation null.
- **Point patterns:** compare Ripley's K/L against a **CSR envelope** from ≥99 simulations
  within the *observed study window* (edge correction on), not an unbounded Poisson intuition.
- **Landscape metrics:** one landscape's metric is descriptive; if you claim fragmentation
  *changed* or *differs*, compare against a matched null (neutral landscape / permuted class map)
  or across dates/sites with uncertainty.
- **SDM:** evaluate on **spatially-blocked cross-validation** (spatially disjoint train/test),
  never random k-fold (it leaks through autocorrelation and inflates AUC), and compare against a
  prevalence-only / random-background null.
State the null, the permutation/simulation count, and the seed for every inferential claim.

### 3. Quantify uncertainty and stability
Report pseudo p-values and, where given, standardized z-scores with the permutation count.
Show robustness to the choices known to move spatial results: re-run Moran's I / hot-spot maps
under ≥2 weights definitions (Queen vs KNN-8 vs a distance band) and report whether the
significant-cluster set is stable; flag **MAUP** (aggregation scale/zoning) sensitivity
explicitly. For SDM report mean ± spread across spatial folds (AUC, TSS, Boyce) and
variable-importance stability, not one split. Apply the rigor skill's threshold rule: if an
estimate is within ~3× its spread of its null expectation, label it "marginal / not resolved."

### 4. Respect the geospatial substrate (CRS, resolution, alignment, nodata)
The substrate drives the answer more than the algorithm. Confirm every layer's CRS and
reproject to a common equal-area/projected CRS before area/distance/weights work. When
combining rasters, ensure identical grid/resolution/extent — resample explicitly with a stated
method (nearest for categorical, bilinear/cubic for continuous). Honor nodata/masks (cloud/QA
masks, nan-aware reductions) before every reduction. State pixel size and analysis scale (a
metric at 30 m ≠ at 300 m), and the classification scheme + breaks for landscape metrics and
choropleths.

### 5. Reproducibility record
Record the interpreter and versions (`geopandas.show_versions()`), the IO engine (`pyogrio`),
every EPSG code and reprojection, the weights definition (type + k/band + row-standardization),
permutation counts and seeds, raster resolution / resampling / nodata handling, the
classification scheme and breaks, and the exact commands. Write intermediate spatial data to
inspectable files in `/workspace` (GeoPackage for vector; COG or NetCDF/Zarr for raster).

### 6. Honest accounting
If a canonical tool was unavailable and you used a surrogate, say so and name what was lost
(e.g. "used xarray-spatial slope, not a hydrologically-conditioned flow accumulation"). Every
spatial conclusion (this area is a hotspot; these patches are fragmented; this species will
occur here) gets a confidence level tied to §2–§3 — the correct null, the permutation p, the
weights/scale robustness, the spatial-CV performance — not merely that a function returned a
number. Report which validations you ran and which you did not (e.g. MAUP across scales).

## Delegation note
When delegating a verification subtask, give the subagent this skill's standards explicitly —
the spatial null, the weights/scale robustness, the spatial-CV protocol — and reconcile its
numbers to yours against the spread, not just "consistent".
