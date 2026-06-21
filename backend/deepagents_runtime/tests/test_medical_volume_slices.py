"""Unit tests for the medical-volume-slices CLI's core (numpy-only) logic.

The CLI is a skill script baked into the codeexec image; the heavy imaging libs
(nibabel/SimpleITK/dipy) are imported lazily, so the modality/windowing/slice-
selection logic is testable here with just numpy. The full pipeline (real NIfTI ->
slices) is validated live in the sandbox image.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_CLI = (
    Path(__file__).resolve().parents[1]
    / "skills" / "medical-volume-slices" / "volume_slices.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("volume_slices", _CLI)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_detect_modality_ct_from_hounsfield(mod):
    # CT: air ~ -1000, soft tissue ~ 40, bone ~ 800 — a real negative tail
    vol = np.random.uniform(-1000, 60, size=(40, 40, 30)).astype(np.float32)
    vol[:5] = -1000  # air
    assert mod.detect_modality("nifti3d", vol) == "ct"


def test_detect_modality_mri_from_nonnegative(mod):
    vol = np.random.uniform(0, 1500, size=(40, 40, 30)).astype(np.float32)
    assert mod.detect_modality("nifti3d", vol) == "mri"


def test_detect_modality_fmri_and_tractography(mod):
    assert mod.detect_modality("nifti4d", np.zeros((4, 4, 4, 3), np.float32)) == "fmri"
    assert mod.detect_modality("tractography", np.zeros(1)) == "tractography"


def test_ct_window_maps_center_to_midgray(mod):
    sl = np.array([[40.0]], dtype=np.float32)  # brain-window center
    out = mod._window(sl, center=40, width=80)
    assert 120 <= int(out[0, 0]) <= 135  # center -> ~mid gray
    # below the window floor clips to black, above ceiling to white
    assert int(mod._window(np.array([[-100.0]]), 40, 80)[0, 0]) == 0
    assert int(mod._window(np.array([[400.0]]), 40, 80)[0, 0]) == 255


def test_robust_norm_uses_percentiles(mod):
    sl = np.concatenate([np.zeros(98), np.array([1000.0, 2000.0])]).astype(np.float32)
    out = mod._robust_norm(sl.reshape(10, 10), lo_p=1, hi_p=99)
    assert out.dtype == np.uint8 and out.max() <= 255


def test_pick_indices_skips_empty_ends(mod):
    # tissue only in the central third along axis 2; picks must land in that band
    vol = np.zeros((20, 20, 60), dtype=np.float32)
    vol[:, :, 20:40] = 100.0
    idx = mod._pick_indices(vol, axis=2, n=4)
    assert all(15 <= i <= 45 for i in idx), idx
    assert len(idx) >= 2


def test_foreground_extent_finds_tissue_band(mod):
    vol = np.zeros((20, 20, 50), dtype=np.float32)
    vol[:, :, 10:30] = 100.0
    lo, hi = mod._foreground_extent(vol, axis=2)
    assert 8 <= lo <= 12 and 28 <= hi <= 32


def test_plane_axis_mapping_is_canonical_ras(mod):
    assert mod._PLANE_AXIS == {"sagittal": 0, "coronal": 1, "axial": 2}
