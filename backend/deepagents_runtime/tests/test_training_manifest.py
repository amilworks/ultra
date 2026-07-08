"""Cross-language parity: the Python ModelManifest must match the shared
contract fixture that the Go registry-seed test pins from the other side."""

from __future__ import annotations

import json
from pathlib import Path

from ultra_deepagents.training import MANIFESTS, RARESPOT_MANIFEST
from ultra_deepagents.training.manifests import TRAINING_JOB_PHASES

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "contracts"
    / "training"
    / "yolov5_rarespot.manifest.json"
)


def _load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_rarespot_manifest_matches_shared_fixture() -> None:
    fixture = _load_fixture()
    assert RARESPOT_MANIFEST.to_contract_projection() == fixture["manifest"]


def test_manifest_registry_is_keyed_canonically() -> None:
    for model_key, manifest in MANIFESTS.items():
        assert model_key == manifest.model_key
    assert "yolov5_rarespot" in MANIFESTS
    # The stale spellings must never come back.
    assert "rarespot-prairie-yolo" not in MANIFESTS
    assert "prairie_yolo" not in MANIFESTS


def test_capability_phase_mapping_is_total() -> None:
    phases = RARESPOT_MANIFEST.allowed_phases()
    assert set(phases) == set(TRAINING_JOB_PHASES)
    for phase in phases:
        assert phase.startswith("training.")


def test_job_phases_match_shared_fixture() -> None:
    """The Python half of the 'one enum PR' gate (Go mirror:
    domain.TrainingJobPhases, pinned by TestTrainingJobPhasesMatchSharedContractFixture)."""
    fixture = _load_fixture()
    assert fixture["job_phases"], "fixture job_phases is empty - the enum gate would be vacuous"
    assert list(TRAINING_JOB_PHASES) == fixture["job_phases"]
