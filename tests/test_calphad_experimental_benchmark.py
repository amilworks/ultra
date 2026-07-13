from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "calphad_experimental_benchmark.py"
_SPEC = importlib.util.spec_from_file_location("calphad_experimental_benchmark", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_BENCHMARK = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BENCHMARK
_SPEC.loader.exec_module(_BENCHMARK)

BenchmarkConfigurationError = _BENCHMARK.BenchmarkConfigurationError
load_validated_manifest = _BENCHMARK.load_validated_manifest
score_predictions = _BENCHMARK.score_predictions
validate_manifest = _BENCHMARK.validate_manifest

_FIXTURE = (
    _ROOT / "tests" / "fixtures" / "materials" / "calphad_experimental_benchmark_expected.json"
)


def _manifest() -> dict[str, object]:
    manifest, _, _ = load_validated_manifest(_ROOT)
    return manifest


def _fixture() -> dict[str, object]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def test_source_manifest_is_provenance_bound_and_keeps_lane_claims_separate() -> None:
    manifest, path, validated = load_validated_manifest(_ROOT)

    assert path.name == "experimental_benchmark_manifest.json"
    assert validated["database_sha256"] == (
        "107c7330f0326a334742632f7494c7beadf53370edbc188df1a030853ceab5a8"
    )
    sources = {source["source_id"]: source for source in manifest["sources"]}
    assert sources["nist_lass_2014_900c"]["independent_of_bound_assessment"] is False
    assert sources["nist_lass_2014_900c"]["license_id"] == "CC0-1.0"
    assert sources["tomaszewska_2018_dta"]["independent_of_bound_assessment"] is True
    assert sources["migas_2020_crystallization"]["independent_of_bound_assessment"] is True
    assert sources["tomaszewska_2018_dta"]["license_id"] == "CC-BY-4.0"
    assert sources["migas_2020_crystallization"]["license_id"] == "CC-BY-4.0"

    lanes = {lane["lane_id"]: lane for lane in manifest["lanes"]}
    calibration = lanes[_BENCHMARK.CALIBRATION_LANE_ID]
    held_out = lanes[_BENCHMARK.HELD_OUT_LANE_ID]
    assert calibration["classification"] == "calibration"
    assert "must never be labeled independent" in calibration["promotion_interpretation"]
    assert held_out["classification"] == "held_out"
    assert held_out["independent_of_bound_assessment"] is True
    assert len(calibration["observations"]) == 6
    assert len(held_out["observations"]) == 4
    assert all(
        observation["uncertainty_K"] is None
        and observation["uncertainty_status"] == "not_reported_numerically"
        for observation in held_out["observations"]
    )
    for observation in held_out["observations"]:
        assert observation["temperature_K"] == pytest.approx(
            observation["temperature_degC"] + 273.15,
            abs=1e-12,
        )


def test_retained_pycalphad_predictions_pass_the_locked_two_lane_metrics() -> None:
    manifest = _manifest()
    fixture = _fixture()

    report = score_predictions(
        manifest,
        calibration_predictions=fixture["calibration_predictions_atomic_fraction"],
        held_out_predictions=fixture["held_out_predictions"],
    )

    assert report["status"] == "passed"
    assert report["production_promotion_blocked"] is False
    calibration_metrics = report["lanes"]["calibration"]["metrics"]
    held_out_metrics = report["lanes"]["held_out"]["metrics"]
    expected = fixture["expected_metrics"]
    assert calibration_metrics["weighted_rms_z"] == pytest.approx(
        expected["calibration_weighted_rms_z"], abs=1e-12
    )
    assert calibration_metrics["max_abs_z"] == pytest.approx(
        expected["calibration_max_abs_z"], abs=1e-12
    )
    assert held_out_metrics["mae_K"] == pytest.approx(
        expected["held_out_mae_K"], abs=1e-12
    )
    assert held_out_metrics["max_abs_error_K"] == pytest.approx(
        expected["held_out_max_abs_error_K"], abs=1e-12
    )
    assert held_out_metrics["mae_K_max"] == 20.0
    assert held_out_metrics["max_abs_error_K_max"] == 30.0


def test_independent_holdout_failure_blocks_promotion_without_hiding_residuals() -> None:
    manifest = _manifest()
    fixture = _fixture()
    predictions = copy.deepcopy(fixture["held_out_predictions"])
    predictions["migas_2020_nominal_composition"]["liquidus_K"] = 1800.0

    report = score_predictions(
        manifest,
        calibration_predictions=fixture["calibration_predictions_atomic_fraction"],
        held_out_predictions=predictions,
    )

    assert report["status"] == "failed"
    assert report["production_promotion_blocked"] is True
    held_out = report["lanes"]["held_out"]
    assert held_out["status"] == "failed"
    assert held_out["metrics"]["mae_K_max"] == 20.0
    assert held_out["metrics"]["max_abs_error_K_max"] == 30.0
    failed_observation = next(
        observation
        for observation in held_out["observations"]
        if observation["observation_id"] == "migas_2020_liquidus"
    )
    assert failed_observation["predicted_temperature_K"] == 1800.0
    assert failed_observation["residual_K_predicted_minus_observed"] == pytest.approx(58.85)
    assert any("independent thermometric holdout" in reason for reason in report["blocking_reasons"])


def test_manifest_rejects_post_hoc_tolerance_relaxation() -> None:
    manifest = _manifest()
    tampered = copy.deepcopy(manifest)
    lanes = {lane["lane_id"]: lane for lane in tampered["lanes"]}
    lanes[_BENCHMARK.HELD_OUT_LANE_ID]["metrics"]["mae_K_max"] = 21.0

    with pytest.raises(BenchmarkConfigurationError, match="must remain fixed at 20.0"):
        validate_manifest(tampered, repository_root=_ROOT)


def test_manifest_rejects_temperature_conversion_or_independence_forgery() -> None:
    manifest = _manifest()
    bad_temperature = copy.deepcopy(manifest)
    lanes = {lane["lane_id"]: lane for lane in bad_temperature["lanes"]}
    lanes[_BENCHMARK.HELD_OUT_LANE_ID]["observations"][0]["temperature_K"] += 1.0
    with pytest.raises(BenchmarkConfigurationError, match="Celsius-to-kelvin"):
        validate_manifest(bad_temperature, repository_root=_ROOT)

    bad_independence = copy.deepcopy(manifest)
    sources = {source["source_id"]: source for source in bad_independence["sources"]}
    sources["tomaszewska_2018_dta"]["independent_of_bound_assessment"] = False
    with pytest.raises(BenchmarkConfigurationError, match="source provenance drift"):
        validate_manifest(bad_independence, repository_root=_ROOT)


def test_fixture_records_exact_solver_policy_and_database_identity() -> None:
    fixture = _fixture()

    assert fixture["benchmark_id"] == _BENCHMARK.BENCHMARK_ID
    assert fixture["database_sha256"] == (
        "107c7330f0326a334742632f7494c7beadf53370edbc188df1a030853ceab5a8"
    )
    assert fixture["solver"] == {
        "pycalphad": "0.11.2",
        "numpy": "1.26.4",
        "phase_fraction_epsilon": 1e-8,
        "bisection_iterations": 16,
    }
