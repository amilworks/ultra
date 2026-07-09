"""Ecology (geospatial) domain-correctness invariants.

Skips unless geopandas/esda are present (sandbox image). Each check fails for the
hand-rolled shortcut the computational-ecology skill warns against.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_area_requires_projected_crs():
    """Area must be computed in a projected (equal-area) CRS, not degrees.

    A ~1° box near 40°N is ~9,400 km²; computing `.area` in EPSG:4326 returns
    ~1.0 (deg², meaningless). The trap is reporting the degree number as area.
    """
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    g = gpd.GeoDataFrame(geometry=[Polygon([(-100, 40), (-99, 40), (-99, 41), (-100, 41)])], crs="EPSG:4326")
    assert g.crs.is_geographic
    area_km2 = g.to_crs("EPSG:6933").area.iloc[0] / 1e6  # NSIDC EASE-Grid 2.0, equal-area
    assert 8000 < area_km2 < 11000, f"projected area implausible: {area_km2:.0f} km²"


def test_morans_i_matches_recomputation():
    """esda Moran's I equals I recomputed from the row-standardized spatial lag.

    (The tautological `E[I] == -1/(n-1)` check the audit first proposed catches
    nothing — esda always sets that. This recomputation catches a wrong weights
    matrix or statistic, which is the real failure mode.)
    """
    gpd = pytest.importorskip("geopandas")
    esda = pytest.importorskip("esda")
    from libpysal.weights import Queen, lag_spatial
    from shapely.geometry import Polygon

    n = 6
    cells, vals = [], []
    for i in range(n):
        for j in range(n):
            cells.append(Polygon([(j, i), (j + 1, i), (j + 1, i + 1), (j, i + 1)]))
            vals.append(i + j)  # smooth gradient -> autocorrelated
    gdf = gpd.GeoDataFrame({"v": vals}, geometry=cells, crs="EPSG:6933")
    w = Queen.from_dataframe(gdf, use_index=True)
    w.transform = "r"
    y = gdf["v"].to_numpy(dtype=float)
    np.random.seed(42)
    mi = esda.Moran(y, w, permutations=999)
    z = y - y.mean()
    i_manual = (z @ lag_spatial(w, z)) / (z @ z)
    assert abs(mi.I - i_manual) < 1e-9, f"esda I {mi.I} != recomputed {i_manual}"
    assert mi.I > mi.EI and mi.p_sim < 0.05  # genuine positive autocorrelation


def test_hurlbert_rarefaction_full_depth_equals_richness():
    """Hurlbert E[S_n] at n = total count equals observed richness.

    Rarefaction must reduce to observed richness at full depth; a formula that
    doesn't is wrong. (Sample-size-dependent raw richness is the trap.)
    """
    pytest.importorskip("scipy")
    from scipy.special import comb

    counts = np.array([50, 30, 12, 5, 2, 1, 0, 0])

    def erarefy(c, nn):
        N = c.sum()
        return float(np.sum([1 - comb(N - ci, nn) / comb(N, nn) for ci in c if ci > 0]))

    richness = int((counts > 0).sum())
    assert abs(erarefy(counts, counts.sum()) - richness) < 1e-6
