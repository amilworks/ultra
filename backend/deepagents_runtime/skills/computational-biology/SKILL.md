---
name: computational-biology
description: Domain toolkit and rigor for single-cell / spatial-omics graph analysis — community detection on cell or gene graphs, spatial neighbor graphs, ligand-receptor inference, and graph neural networks. Use when a task involves single-cell or spatial transcriptomics, clustering cells/genes, detecting modules or communities in a biological graph, testing whether that structure is real, or ligand-receptor / cell-cell communication. Read it before writing code that would otherwise reach for a hand-rolled surrogate (e.g. plain Louvain, a synthetic ligand-receptor score, a home-made SBM null).
---

# Computational Biology

## When to use
Read this before running any single-cell, spatial-transcriptomics, or biological-graph
analysis — community/module detection, spatial structure, or cell-cell communication.
The point is to use the field's actual tools and validation methods instead of
re-deriving weaker surrogates from `numpy`/`networkx`, and to prove that any structure
you report is real rather than an artifact of the algorithm. This skill is domain
tooling; the numerical-rigor protocol in `/skills/computational-experiment-rigor/SKILL.md`
(uncertainty, thresholds, honest accounting) still applies on top of it.

## Environments — two interpreters
Two Python environments are baked into the sandbox. Pick per task; state which you used.

- **Default `python`** (numpy 1.26.4): `scanpy`, `anndata`, `leidenalg` + `python-igraph`
  (Leiden), `liana` (ligand-receptor), `torch-geometric` (GNNs, on the CUDA `torch`),
  plus the full imaging/scientific stack. Use this for most work.
- **`/opt/biograph/bin/python`** (isolated conda env, numpy 2): `graph-tool` (Bayesian
  nested SBM + MDL) and `squidpy` (spatial graphs, Moran's I, neighborhood enrichment),
  alongside a second copy of `scanpy`/`anndata`/`leidenalg`. Use it for SBM/MDL structure
  testing and squidpy spatial analysis. **`graph_tool` and `squidpy` cannot be imported in
  the default `python` (ModuleNotFoundError) — do not try `import graph_tool` there first.**
  Write that work as a standalone script from the start and run it with
  `/opt/biograph/bin/python /workspace/sbm_analysis.py`; do not `pip install` these into the
  default env. Move data between the two via files in `/workspace` (an `.h5ad` written by one
  env reads cleanly in the other).

## Protocol

### 1. Use the field-standard method, name it, and pin its parameters
Do not substitute a generic surrogate when a canonical tool exists.
**Before writing analysis code, copy the vetted recipe** — correct call + a runnable self-check +
the named anti-pattern — from **[references/singlecell-recipes.md](references/singlecell-recipes.md)**
(preprocessing order, Leiden stability, DE double-dipping control, batch integration — ComBat is the
only integrator installed —, PAGA/DPT pseudotime, markers, liana CCC, squidpy under /opt/biograph).
Hand-rolling these is how wrong-but-plausible results ship; run the self-check before trusting output.
- **Community detection:** use **Leiden** (`scanpy.tl.leiden` / `leidenalg`), not Louvain,
  as the default — Leiden guarantees well-connected communities; Louvain does not. State
  the objective (RBConfiguration/modularity/CPM), the resolution, the seed, and the graph
  you ran it on (kNN on PCA? shared-nearest-neighbor? spatial?).
- **Ligand-receptor / cell-cell communication:** use **`liana`** (a real method — its
  rank-aggregate consensus, or a named one: CellPhoneDB, NATMI, Connectome, SingleCellSignalR),
  never a hand-made "expression product" score. Report the method, the permutation/null it
  uses, and both a specificity and a magnitude score — not a single number.
- **Spatial structure:** build the spatial graph with `squidpy.gr.spatial_neighbors`
  (Delaunay or kNN — state which and k), then use `squidpy.gr.spatial_autocorr` (Moran's I)
  and `squidpy.gr.nhood_enrichment`. Distinguish structure in *expression* space from
  structure in *physical* space; do not conflate them.
Name the method at the level you are confident in; never fabricate a citation or a
benchmark number.

### 2. Prove the structure is real, not an artifact (the hallucination test)
Community-detection algorithms return communities on random graphs too. Before reporting
a partition as meaningful, show it beats a principled null.
- **Preferred:** fit a **degree-corrected / nested stochastic block model with MDL** using
  `graph_tool` in `/opt/biograph` (`minimize_blockmodel_dl` /
  `minimize_nested_blockmodel_dl`). In graph-tool ≥2.x the degree-correction flag goes in
  `state_args`, **not** as a top-level kwarg — call
  `gt.minimize_nested_blockmodel_dl(g, state_args=dict(deg_corr=True))` (passing
  `deg_corr=True` directly raises `TypeError: unexpected keyword argument 'deg_corr'`), and
  read the description length with `state.entropy()`. The description length is Occam's razor
  for graphs: if the inferred model collapses to **B = 1 block**, or its description length
  does not beat a single-block model, the community structure is **not resolved** — say so and stop.
- **Always, as a floor:** compare your partition's modularity (or the SBM's DL) against a
  **degree-preserving configuration-model null** (rewire ≥100×, e.g. `igraph`
  `rewire`/`Degree_Sequence`), and report where the observed value sits in the null
  distribution (z-score or empirical p). "Modularity = 0.4" is meaningless without the null.
- Report the number of communities as a **result with uncertainty**, not an input: show it
  is stable across seeds and a resolution sweep, not a single lucky setting.

### 3. Quantify partition agreement and stability
A partition near the noise floor is not a result.
- Compare partitions (across seeds, resolutions, methods, or against known labels) with
  **ARI and NMI** (`sklearn.metrics.adjusted_rand_score`, `normalized_mutual_info_score`).
  Report the value *and* what a random-label baseline would give.
- Run Leiden across ≥5 seeds and ≥3 resolutions; report the community count as
  mean ± spread and the ARI between runs. If ARI between seeds is low, the partition is
  unstable — label it so, and do not over-interpret specific communities.
- When comparing methods (Leiden vs SBM vs spectral), reconcile them explicitly: report
  pairwise ARI and say which cells/nodes move, rather than asserting they "agree".

### 4. Respect the biology of the graph you built
- State how the graph was constructed: features (HVGs? PCA dims?), neighbor count,
  metric, and whether edges are expression-based, spatial, or both. The construction
  drives the communities more than the algorithm does.
- For single-cell, follow the standard preprocessing and say you did: QC filtering,
  normalization (e.g. `sc.pp.normalize_total` + `log1p`), HVG selection, PCA, then
  neighbors — and record the parameters. Skipping steps changes the graph.
- Map communities back to biology (marker genes via `sc.tl.rank_genes_groups`,
  or spatial domains) rather than leaving them as anonymous integer labels; flag any
  community with no coherent markers as possibly technical (batch/QC), not a cell type.

### 5. Reproducibility record
Record which interpreter you used (default vs `/opt/biograph`), package versions
(`scanpy.logging.print_versions()` or explicit `__version__`s), every graph-construction
and clustering parameter, all seeds, and the resolution sweep. A colleague must be able to
regenerate every partition and every number. Write intermediate `AnnData` to `.h5ad` so the
run is inspectable.

### 6. Honest accounting
- If a real tool was unavailable and you fell back to a surrogate, say so explicitly and
  name what was lost — do not present a `networkx` reimplementation as the canonical method.
- Every conclusion (a community is a cell type; a ligand-receptor pair is active; a spatial
  domain exists) gets a confidence level tied to §2–§3 evidence: the null comparison, the
  stability, and the biological coherence — not just that the algorithm returned something.
- Report which validations you ran and which you did not.

## Delegation note
When delegating a verification subtask (e.g. "confirm these communities survive an SBM/MDL
test"), give the subagent this skill's standards explicitly — the null model, the seed/
resolution sweep, the ARI/NMI reconciliation — and compare its numbers to yours against the
spread, not just "consistent".
