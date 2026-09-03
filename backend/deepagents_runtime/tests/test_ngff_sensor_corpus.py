"""CI smoke of the synthetic sensor-data corpus + reader contract.

A fast, low-memory subset of ``tools/ngff_sensor_corpus`` (the full stress harness lives
there and is run out-of-band): build one spec-correct OME-Zarr store per STEM domain, assert
the reader/viewer-info/render contract on each, and confirm the whole adversarial set fails
closed. Skipped unless zarr + Pillow (the ngff-service deps) are installed — matching
``test_ngff.py`` — so it runs in the ngff image / CI where those exist.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

try:
    from ngff_sensor_corpus.adversarial import build_adversarial
    from ngff_sensor_corpus.specs import DOMAINS, catalog
    from ngff_sensor_corpus.stress import exercise_adversarial, exercise_valid
    from ngff_sensor_corpus.writer import write_store

    _HAVE_DEPS = True
except Exception:
    _HAVE_DEPS = False

if "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST"):
    import pytest

    pytestmark = pytest.mark.skipif(not _HAVE_DEPS, reason="zarr/Pillow (ngff deps) not installed")


def _fast_subset():
    """One representative store per domain, skipping the slow/huge ones for CI."""
    slow = {"histology_he_rgb"}  # 2560^2 RGB pyramid — exercised out-of-band, not in CI
    chosen: dict[str, object] = {}
    for spec in catalog():
        if spec.modality in slow or spec.domain in chosen:
            continue
        chosen[spec.domain] = spec
    return list(chosen.values())


def test_one_valid_store_per_domain_satisfies_the_contract(tmp_path):
    subset = _fast_subset()
    covered = {spec.domain for spec in subset}
    assert covered == set(DOMAINS), f"missing domain coverage: {set(DOMAINS) - covered}"
    for spec in subset:
        path = write_store(spec, str(tmp_path))
        result = exercise_valid(spec, path)
        assert result["status"] == "PASS", f"{spec.modality}: {result['failures']}"


def test_every_adversarial_store_fails_closed(tmp_path):
    cases = build_adversarial(str(tmp_path))
    assert len(cases) >= 18
    for case in cases:
        result = exercise_adversarial(case)
        # reject cases must PASS (raised the right NgffError); probes must not be FAIL.
        assert result["status"] in ("PASS", "PROBE"), f"{case.name}: {result}"


def test_reader_hardening_budgets_are_live(tmp_path):
    """The corpus's scale probes should be bounded by the reader's plane-read budget."""
    from ngff_sensor_corpus.scale import scale_probes
    from ultra_deepagents.ngff.reader import NgffError, open_ngff
    from ultra_deepagents.ngff.render import render_slice_png

    giga = next(s for s in scale_probes() if s.modality == "gigapixel_single_level")
    path = write_store(giga, str(tmp_path))
    img = open_ngff(path)
    with pytest.raises(NgffError):  # 144 MP single plane must not be read whole
        render_slice_png(img, level=0, max_dim=1024)
