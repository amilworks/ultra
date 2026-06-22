"""Parity: the Python special-format registry must match backend/shared/special_formats.json
(the canonical spec the Go registry also mirrors)."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ultra_deepagents.imaging.special_formats import SPECIAL_FORMATS, detect_by_name

_SHARED = os.path.join(os.path.dirname(__file__), "..", "..", "shared", "special_formats.json")


def _shared_formats() -> dict[str, dict]:
    with open(_SHARED, encoding="utf-8") as fh:
        data = json.load(fh)
    return {f["id"]: f for f in data["formats"]}


def test_python_registry_matches_shared_json():
    shared = _shared_formats()
    ours = {sf.id: sf for sf in SPECIAL_FORMATS}
    assert set(ours) == set(shared), "registry ids drifted from special_formats.json"
    for fid, sf in ours.items():
        spec = shared[fid]
        assert list(sf.extensions) == list(spec["extensions"]), fid
        assert sf.shape == spec["shape"], fid
        assert list(sf.dir_markers) == list(spec.get("dir_markers", [])), fid
        assert sf.serve == spec["serve"], fid
        assert sf.resource_kind == spec.get("resource_kind", ""), fid


def test_detect_ome_zarr_by_name():
    assert detect_by_name("scan.ome.zarr").id == "ome-zarr"
    assert detect_by_name("scan.zarr").id == "ome-zarr"
    assert detect_by_name("scan.zarr_3").id == "ome-zarr"  # series-suffixed
    assert detect_by_name("scan.tif") is None
