# Single-cell / spatial omics — vetted recipes (scanpy/liana/squidpy)

Copy these instead of reinventing the analysis. Each has the **trap**, the
**correct** call, and a **self-check** to run in-sandbox. Reviewed adversarially;
where a library's surface is volatile, introspect columns/attrs at runtime.

## Preprocessing order (scanpy)
**Trap:** PCA on raw counts, or scaling before HVG. Order is fixed.
```python
import numpy as np, scanpy as sc
sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")  # seurat wants LOG data
adata.raw = adata                       # log-normalised, unscaled — DE/markers/liana read this
adata = adata[:, adata.var.highly_variable]
sc.pp.scale(adata, max_value=10); sc.pp.pca(adata); sc.pp.neighbors(adata)
```
Note: `flavor="seurat_v3"` wants RAW counts and needs `scikit-misc` (NOT installed) → ImportError.
**Self-check:** after a correct run, PC1 should not just track sequencing depth:
`tot = np.asarray(adata.raw.X.sum(1)).ravel(); assert abs(np.corrcoef(adata.obsm["X_pca"][:,0], tot)[0,1]) < 0.6`.

## Leiden clustering with stability (scanpy ≥ 1.10)
**Trap:** a single magic resolution, no seed/robustness check.
```python
import scanpy as sc
sc.tl.leiden(adata, resolution=1.0, flavor="igraph", n_iterations=2, directed=False, random_state=0)
```
**Self-check:** cluster assignment must be stable across seeds — `ARI(seed_a, seed_b) > 0.8`
(sklearn `adjusted_rand_score`). Sweep resolution and report how cluster count moves; do NOT
hard-assert cluster count is monotone in resolution (Leiden isn't strictly monotone — warn, don't fail).

## Differential expression + double-dipping control (scanpy) — VALIDATED
**Trap:** t-test with no multiple-testing correction; reporting "markers" from the same data used to
cluster with no control (double dipping) → inflated significance.
```python
import scanpy as sc
sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon",
                        corr_method="benjamini-hochberg", use_raw=True, pts=True)   # use_raw = lognorm .raw
sc.tl.filter_rank_genes_groups(adata, min_fold_change=1.5, min_in_group_fraction=0.25,
                               max_out_group_fraction=0.5)
```
**Self-check (negative control):** assign RANDOM labels and re-run — a correct BH-corrected test yields
**~0 significant genes**: `padj = adata.uns['rank_genes_groups']['pvals_adj']['0']; assert (padj < 0.05).sum() < 20`.
(Verified: random labels on a homogeneous population → 0/400 BH-significant.)

## Batch integration (scanpy) — only ComBat is installed
**Trap:** naive `ad.concat` across donors, then cluster (clusters = batches). But also: harmonypy /
scanorama / bbknn / scvi-tools are **NOT installed** — those calls ImportError. Out of the box only
`sc.pp.combat` works (weaker than Harmony for strong effects — say so).
```python
import scanpy as sc, anndata as ad          # need anndata import for concat
adata = ad.concat(per_sample, label="batch", keys=sample_ids)
sc.pp.highly_variable_genes(adata, batch_key="batch")
sc.pp.combat(adata, key="batch")            # then scale -> pca -> neighbors -> leiden
```
**Self-check:** mixing, not ARI-vs-batch (ARI false-negatives here). Require most clusters to contain
cells from >1 batch (per-cluster batch entropy > 0). State ComBat is a fallback; kBET/iLISI aren't installed.

## Trajectory / pseudotime (scanpy PAGA + DPT)
**Trap:** a hand-rolled "pseudotime" from a 1-D embedding coordinate.
```python
import scanpy as sc
sc.tl.paga(adata, groups="leiden"); sc.pl.paga(adata, plot=False)
sc.tl.draw_graph(adata, init_pos="paga"); sc.tl.diffmap(adata)
adata.uns["iroot"] = int(root_cell_index); sc.tl.dpt(adata, n_branchings=1)
```
**Self-check:** `"X_diffmap" in adata.obsm` and the root has minimum pseudotime:
`pt = adata.obs["dpt_pseudotime"].values; assert abs(pt[adata.uns["iroot"]] - np.nanmin(pt)) < 1e-6`.

## Marker genes / annotation (scanpy)
**Trap:** calling a gene a "marker" from fold-change alone, ignoring expressed fraction.
```python
import scanpy as sc
sc.tl.rank_genes_groups(adata, "leiden", pts=True)
df = sc.get.rank_genes_groups_df(adata, group="0")
```
**Self-check (defensive — column names are version-dependent):**
`assert {"pct_nz_group","pct_nz_reference"} <= set(df.columns), df.columns` then require a real marker to be
expressed in >25% of in-group cells and >0.1 higher than reference; if the columns are absent, fall back to
`adata.uns['rank_genes_groups']['pts']`.

## Cell-cell communication (liana-py)
**Trap:** a hand-rolled ligand×receptor product with no specificity/permutation null.
```python
import liana as li
li.mt.rank_aggregate(adata, groupby="leiden", expr_prop=0.1, use_raw=True)   # li.mt = method namespace
res = adata.uns["liana_res"]
```
**Self-check:** a valid CCC result reports **both** magnitude and specificity — not one score. Introspect
columns (liana's surface is volatile): assert `res` has a magnitude column (`magnitude_rank`/`magnitude`)
AND a specificity column (`specificity_rank`/`cellphone_pvals`/`specificity`). Confirm `li.mt.rank_aggregate`
exists in the pinned version at first use.

## Spatial stats (squidpy) — MUST run under /opt/biograph/bin/python
**Trap:** squidpy is only in the numpy-2 `/opt/biograph` env → ModuleNotFoundError in the default python.
Also: an ad-hoc "clustering score" instead of Moran's I / neighborhood enrichment.
```python
# /opt/biograph/bin/python
import squidpy as sq                                   # needs adata.obsm["spatial"]
sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
sq.gr.spatial_autocorr(adata, mode="moran")            # -> adata.uns["moranI"] (column "I")
sq.gr.nhood_enrichment(adata, cluster_key="leiden")    # -> adata.uns["leiden_nhood_enrichment"]["zscore"]
```
**Self-check (shuffle control, not a range assert):** shuffle `adata.obsm["spatial"]` rows and re-run —
Moran's I must drop toward 0 for a genuinely spatial signal. Do NOT assert `|I| <= 1` (Moran's I isn't
strictly bounded to [-1,1] and can false-fail).
