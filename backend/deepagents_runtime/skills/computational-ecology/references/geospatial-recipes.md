# Geospatial ecology — vetted recipes (use the library; don't hand-roll)

Copy these instead of reinventing the analysis. Each has the **trap**, the
**correct** call, and a **self-check** to run in-sandbox. Reviewed adversarially;
several self-checks the first draft proposed were tautological or wrong and are
corrected here. Confirm version-fragile APIs (pointpats, pylandstats major) live.

## Area / distance / buffer — project first (geopandas) — VALIDATED
**Trap:** computing `.area`/`.length`/`.buffer(x)` in EPSG:4326 — the numbers are in
**degrees**, not m/m². A ~1° box near 40°N is ~9,400 km², not ~1.
```python
import geopandas as gpd
gdf = gpd.read_file(path, engine="pyogrio")
proj = gdf.to_crs(gdf.estimate_utm_crs())     # local distance/shape; or EPSG:6933 for global equal-AREA
area_km2 = proj.to_crs("EPSG:6933").area / 1e6
```
**Self-check:** `assert not gdf_used_for_area.crs.is_geographic` and sanity-check the magnitude.
(Verified: 1° box → 9,413 km² projected vs a meaningless 1.0 deg².)

## Global Moran's I (esda) — VALIDATED
**Trap:** a Pearson correlation of a variable with itself, or OLS ignoring spatial autocorrelation.
```python
import numpy as np, esda
from libpysal.weights import Queen
w = Queen.from_dataframe(gdf, use_index=True); w.transform = "r"
np.random.seed(42)
mi = esda.Moran(gdf["v"].to_numpy(float), w, permutations=999)   # mi.I, mi.EI, mi.p_sim
```
**Self-check (NOT `EI == -1/(n-1)` — that's tautological, esda always sets it):** recompute I from the
row-standardized lag and require a match, which catches a wrong weights matrix:
`from libpysal.weights import lag_spatial; z = y - y.mean(); assert abs(mi.I - (z @ lag_spatial(w, z))/(z @ z)) < 1e-9`.
Claim clustering only if `mi.p_sim < 0.05 and mi.I > mi.EI`.

## SDM with spatially-blocked CV (sklearn + geopandas)
**Trap:** random K-fold CV on spatially autocorrelated points → train/test leakage → inflated AUC.
```python
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
xy = np.c_[gdf.to_crs(gdf.estimate_utm_crs()).geometry.x, ...y]
block = (xy // block_m).astype(int); bid = block[:,0]*100000 + block[:,1]     # contiguous blocks
auc = cross_val_score(RandomForestClassifier(), X, y, groups=bid, cv=GroupKFold(5), scoring="roc_auc")
```
**Self-check:** verify fold disjointness `set(bid[tr]).isdisjoint(set(bid[te]))` per fold (GroupKFold
guarantees it). Do NOT assert a min train-test distance ≥ block size — adjacent blocks straddle a boundary,
so a legit run has near-zero min distance (false alarm). For true separation, drop training points within a
buffer of the held-out block. Spatial-CV AUC ≤ random-KFold AUC is a useful *warning*, not a hard assert.

## Diversity with rarefaction (Hurlbert) — VALIDATED
**Trap:** comparing raw species richness across samples of different size (richness rises with effort).
```python
import numpy as np
from scipy.special import comb
def erarefy(counts, n):            # Hurlbert E[S_n]
    N = counts.sum()
    return float(np.sum([1 - comb(N - c, n)/comb(N, n) for c in counts if c > 0]))
depth = min(site_totals); rich_rare = {site: erarefy(c, depth) for site, c in counts.items()}
```
**Self-check:** at full depth it must equal observed richness: `abs(erarefy(c, c.sum()) - (c > 0).sum()) < 1e-6`.
Note: rarefy richness; report Shannon/Simpson for context but state they're only mildly effort-biased, not corrected.

## Landscape / fragmentation metrics (pylandstats) — confirm major version
**Trap:** a hand-rolled `scipy.ndimage.label` patch count with undocumented connectivity.
```python
import pylandstats as pls
ls = pls.Landscape(arr, res=(30, 30), nodata=0, neighborhood_rule="8")   # STATE 4- vs 8-connectivity
cls = ls.compute_class_metrics_df(metrics=["total_area", "patch_density", "edge_density", "area_mn"])
```
(Method names differ across pylandstats 1.x vs 2.x/3.x — confirm before quoting.)
**Self-check:** area conservation — `cls["total_area"].sum() ≈ n_valid_pixels * (res_x*res_y/1e4)` ha;
and 8-connectivity patch count ≤ 4-connectivity patch count.

## Point-pattern Ripley's K/L vs CSR (pointpats) — CONFIRM API LIVE
**Trap:** eyeballing "clustered" from a dot map; K without edge correction or a CSR envelope.
pointpats has churned heavily (class `Kenv`/`Ripley` vs functional `k_test`/`l_test`).
```python
# FIRST: import pointpats; print(pointpats.__version__); print(dir(pointpats.distance_statistics))
# then use the confirmed function with edge correction + a >=99-sim CSR envelope on PROJECTED coords.
```
**Self-check:** project first — `assert not gdf.crs.is_geographic` on the **source GeoDataFrame** (a
PointPattern has no `.crs`); CSR simulations drawn inside the observed window; envelope from ≥99 sims.

## DEM flow accumulation (pysheds)
**Trap:** flow routing on an unconditioned DEM (pits/flats break flow paths).
```python
from pysheds.grid import Grid
grid = Grid.from_raster(dem_path); dem = grid.read_raster(dem_path)
dem = grid.fill_pits(dem); dem = grid.fill_depressions(dem); dem = grid.resolve_flats(dem)
fdir = grid.flowdir(dem); acc = grid.accumulation(fdir)     # condition BEFORE routing
```
**Self-check:** `assert acc.max() <= n_valid_cells` and `assert acc.min() >= 0` — **not `>= 1`**:
pysheds doesn't count the cell itself, so headwater cells are 0 (its own docs plot `log(acc+1)`).

## Spatial regression with residual diagnostic (spreg)
**Trap:** plain OLS on spatial data, then trusting the coefficients despite autocorrelated residuals.
```python
import spreg
from libpysal.weights import Queen
w = Queen.from_dataframe(gdf, use_index=True); w.transform = "r"
ols = spreg.OLS(y2d, X2d, w=w, spat_diag=True, moran=True)   # y2d (n,1), X2d (n,k)
# if residual Moran is significant -> escalate to spreg.ML_Lag / spreg.ML_Error
```
**Self-check:** run `esda.Moran` on the fitted residuals; an adequate model has `p_sim > 0.05`. Significant
residual autocorrelation on OLS is exactly the trigger to switch to ML_Lag/ML_Error.
