"""Biology (single-cell) domain-correctness invariants.

Skips unless scanpy is present (sandbox image). Each check fails for the
hand-rolled/rigor-skipping shortcut the computational-biology skill warns against.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_de_double_dipping_negative_control():
    """Random cluster labels on a homogeneous population yield ~0 DE genes.

    This is the double-dipping guard: proper Wilcoxon + BH correction returns
    almost nothing when the labels are meaningless. A pipeline that reports many
    "significant markers" on random labels is inflated (t-test w/o correction,
    or testing on the same data used to cluster without any control).
    """
    sc = pytest.importorskip("scanpy")
    ad = pytest.importorskip("anndata")

    rng = np.random.default_rng(0)
    X = rng.poisson(2.0, size=(300, 400)).astype(float)  # homogeneous: no real groups
    a = ad.AnnData(X)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.raw = a
    a.obs["rand"] = np.asarray(rng.integers(0, 2, a.n_obs)).astype(str)
    a.obs["rand"] = a.obs["rand"].astype("category")
    sc.tl.rank_genes_groups(
        a, "rand", method="wilcoxon", corr_method="benjamini-hochberg", use_raw=True
    )
    padj = np.asarray(a.uns["rank_genes_groups"]["pvals_adj"]["0"])
    n_sig = int((padj < 0.05).sum())
    assert n_sig < 20, f"random labels gave {n_sig} BH-sig genes — DE pipeline is inflated"
