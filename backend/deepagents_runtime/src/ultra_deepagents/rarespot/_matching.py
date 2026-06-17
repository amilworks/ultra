"""Torch-free bipartite matching helpers shared by the spectral scorer and the
per-detection stability pass. Pure NumPy + Python so the matching logic is testable
without torch/CUDA and reusable from the (torch-free) stability code path."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ultra_deepagents.rarespot.tiling import xyxy_iou


def hungarian_min_cost(cost: np.ndarray) -> tuple[list[int], list[int]]:
    """Minimum-cost one-to-one assignment (Hungarian / Kuhn-Munkres, O(n^3)).

    Pads a rectangular cost matrix to square with neutral (0) cost and returns
    (row_indices, col_indices) for the optimal assignment over the padded square;
    callers drop padded/out-of-range pairs. Standard 1-indexed potentials method.
    Vendored to avoid a scipy dependency.
    """
    cost = np.asarray(cost, dtype=float)
    n_rows, n_cols = cost.shape
    n = max(n_rows, n_cols)
    if n == 0:
        return [], []
    square = np.zeros((n, n), dtype=float)
    square[:n_rows, :n_cols] = cost
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = float(square[i0 - 1, j - 1]) - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    rows: list[int] = []
    cols: list[int] = []
    for j in range(1, n + 1):
        if p[j] != 0:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    return rows, cols


def matched_source_indices(
    boxes_a: list[dict[str, Any]],
    boxes_b: list[dict[str, Any]],
    *,
    iou_thresh: float,
    class_key: str = "class_name",
) -> set[int]:
    """Indices of ``boxes_a`` that have a one-to-one (Hungarian) match in ``boxes_b`` with
    IoU >= ``iou_thresh`` and the SAME class. Matching is per-class so a prairie_dog can
    never "survive" by matching a burrow."""
    matched: set[int] = set()
    if not boxes_a or not boxes_b:
        return matched
    a_by_class: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    b_by_class: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, box in enumerate(boxes_a):
        a_by_class[str(box.get(class_key) or "")].append((index, box))
    for index, box in enumerate(boxes_b):
        b_by_class[str(box.get(class_key) or "")].append((index, box))
    for class_name, a_items in a_by_class.items():
        b_items = b_by_class.get(class_name, [])
        if not b_items:
            continue
        iou = np.zeros((len(a_items), len(b_items)), dtype=float)
        for ai, (_, a_box) in enumerate(a_items):
            a_xyxy = a_box.get("xyxy") or []
            if len(a_xyxy) != 4:
                continue
            for bj, (_, b_box) in enumerate(b_items):
                b_xyxy = b_box.get("xyxy") or []
                if len(b_xyxy) != 4:
                    continue
                iou[ai, bj] = xyxy_iou([float(v) for v in a_xyxy], [float(v) for v in b_xyxy])
        rows, cols = hungarian_min_cost(-iou)
        for r, c in zip(rows, cols):
            if r < len(a_items) and c < len(b_items) and float(iou[r, c]) >= float(iou_thresh):
                matched.add(a_items[r][0])
    return matched
