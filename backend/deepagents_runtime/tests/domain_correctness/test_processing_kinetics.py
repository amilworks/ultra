from __future__ import annotations

import copy
import hashlib
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import ultra_deepagents.materials.calphad as calphad
import ultra_deepagents.materials.processing_kinetics as kinetics
from ultra_deepagents.materials import (
    CalphadExecutionError,
    CalphadInputError,
    CalphadTimeoutError,
    processing_method_support,
    run_scheil_solidification,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pycalphad") is None or importlib.util.find_spec("scheil") is None,
    reason="Scheil domain tests execute in the pinned materials environment",
)

_SOURCE = "pycalphad Al-Zn test assessment copied only for deterministic solver qualification"
_LICENSE = "MIT pycalphad test fixture; qualification use only"
_SCOPE = "test-only Al-Zn thermodynamic solver qualification"
_REFERENCE = "fixture-defined standard element reference states"

_ALCOCRNI_PHASES = [
    "BCC_A2",
    "BCC_B2",
    "FCC_A1",
    "HCP_A3",
    "L12_FCC",
    "LIQUID",
    "SIGMA_SGTE",
]
_CFE_BROSHE_PHASES = [
    "BCC_A2",
    "CEMENTITE_D011",
    "DIAMOND_A4",
    "FCC_A1",
    "GRAPHITE",
    "HCP_A3",
    "LIQUID",
    "M7C3_D101",
]


def _copy_alzn_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import pycalphad

    source = Path(pycalphad.__file__).resolve().parent / "tests" / "databases" / "alzn_mey.tdb"
    target = tmp_path / "qualification_alzn.tdb"
    target.write_bytes(source.read_bytes())
    monkeypatch.setattr(calphad, "_is_fixture_database", lambda *_args: False)
    return target


def _copy_alcocrni_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import pycalphad

    source = Path(pycalphad.__file__).resolve().parent / "tests" / "databases" / "alcocrni.tdb"
    target = tmp_path / "qualification_alcocrni.tdb"
    target.write_bytes(source.read_bytes())
    monkeypatch.setattr(calphad, "_is_fixture_database", lambda *_args: False)
    return target


def _copy_cfe_broshe_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import pycalphad

    source = Path(pycalphad.__file__).resolve().parent / "tests" / "databases" / "cfe_broshe.tdb"
    target = tmp_path / "qualification_cfe_broshe.tdb"
    target.write_bytes(source.read_bytes())
    monkeypatch.setattr(calphad, "_is_fixture_database", lambda *_args: False)
    return target


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_mass_closed_scheil_result() -> SimpleNamespace:
    return SimpleNamespace(
        method="scheil",
        converged=True,
        temperatures=[1000.0, 900.0, 800.0],
        fraction_solid=[0.0, 0.5, 0.95],
        fraction_liquid=[1.0, 0.5, 0.05],
        phase_amounts={"FCC_A1": [0.0, 0.5, 0.45]},
        cum_phase_amounts={"FCC_A1": [0.0, 0.5, 0.95]},
        phase_compositions={
            "LIQUID": {
                "AL": [float("nan"), 0.85, 0.7],
                "ZN": [float("nan"), 0.15, 0.3],
            },
            "FCC_A1": {
                "AL": [float("nan"), 0.95, 0.8666666666666667],
                "ZN": [float("nan"), 0.05, 0.13333333333333333],
            },
        },
    )


def _validate_synthetic(raw: SimpleNamespace):
    return kinetics._validated_scheil_result(
        raw,
        requested_phases=["FCC_A1", "LIQUID"],
        physical_components=["AL", "ZN"],
        bulk_composition={"AL": 0.9, "ZN": 0.1},
        liquid_phase_name="LIQUID",
        assessment_temperature_limits_K=None,
        stop_liquid_fraction=0.06,
        max_steps=10,
    )


def test_series_point_cap_stops_before_traversing_an_unbounded_result() -> None:
    observed = 0

    def values():
        nonlocal observed
        for value in range(1000):
            observed += 1
            yield value

    with pytest.raises(CalphadExecutionError, match="exceeds its point bound"):
        kinetics._validated_series(values(), name="counting-series", maximum_count=3)
    assert observed == 4


def test_minimum_stop_fraction_cannot_use_the_global_phase_fraction_tolerance() -> None:
    raw = SimpleNamespace(
        method="scheil",
        converged=True,
        temperatures=[1000.0, 900.0, 800.0],
        fraction_solid=[0.0, 0.5, 0.9999995],
        fraction_liquid=[1.0, 0.5, 5e-7],
        phase_amounts={"FCC_A1": [0.0, 0.5, 0.4999995]},
        cum_phase_amounts={"FCC_A1": [0.0, 0.5, 0.9999995]},
        phase_compositions={
            "LIQUID": {
                "AL": [float("nan"), 0.9, 0.9],
                "ZN": [float("nan"), 0.1, 0.1],
            },
            "FCC_A1": {
                "AL": [float("nan"), 0.9, 0.9],
                "ZN": [float("nan"), 0.1, 0.1],
            },
        },
    )

    with pytest.raises(CalphadExecutionError, match="above the requested residual-liquid"):
        kinetics._validated_scheil_result(
            raw,
            requested_phases=["FCC_A1", "LIQUID"],
            physical_components=["AL", "ZN"],
            bulk_composition={"AL": 0.9, "ZN": 0.1},
            liquid_phase_name="LIQUID",
            assessment_temperature_limits_K=None,
            stop_liquid_fraction=1e-8,
            max_steps=10,
        )


def _run(
    path: Path,
    *,
    x_zn: float = 0.05,
    start_temperature_k: float = 1100.0,
    step_temperature_k: float = 5.0,
    **kwargs,
):
    kwargs.setdefault("stop_liquid_fraction", 0.01)
    return run_scheil_solidification(
        path,
        components=["AL", "ZN", "VA"],
        phases=["FCC_A1", "HCP_A3", "LIQUID"],
        independent_composition={"ZN": x_zn},
        start_temperature_K=start_temperature_k,
        step_temperature_K=step_temperature_k,
        source=_SOURCE,
        license_id=_LICENSE,
        artifact_id="resource-test-alzn-scheil",
        expected_sha256=_sha256(path),
        expected_size_bytes=path.stat().st_size,
        assessment_scope=_SCOPE,
        reference_state=_REFERENCE,
        assessment_temperature_limits_K=[298.15, 1700.0],
        assessment_pressure_limits_Pa=[101325.0, 101325.0],
        max_steps=2048,
        wall_time_seconds=15.0,
        **kwargs,
    )


def test_scheil_path_is_provenance_bound_converged_and_mass_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _copy_alzn_database(tmp_path, monkeypatch)

    observed_wall_limits: list[float] = []
    timer_active = False
    original_wall_time_limit = kinetics._wall_time_limit
    original_validate_result = kinetics._validated_scheil_result

    @contextmanager
    def tracked_wall_time_limit(seconds: float):
        nonlocal timer_active
        observed_wall_limits.append(seconds)
        assert timer_active is False
        timer_active = True
        try:
            with original_wall_time_limit(seconds):
                yield
        finally:
            timer_active = False

    def tracked_validate_result(*args, **kwargs):
        assert timer_active is True
        return original_validate_result(*args, **kwargs)

    monkeypatch.setattr(kinetics, "_wall_time_limit", tracked_wall_time_limit)
    monkeypatch.setattr(kinetics, "_validated_scheil_result", tracked_validate_result)

    record = _run(path)

    assert record["schema_version"] == "ultra.materials.scheil-gulliver.v1"
    assert record["method"] == "Scheil-Gulliver"
    assert record["solver"] == {
        "name": "scheil",
        "version": "0.3.0",
        "pycalphad_version": "0.11.2",
        "adaptive_constitution_sampling": True,
        "replay_determinism_claimed": False,
    }
    assert record["database"]["sha256"] == _sha256(path)
    assert record["request"]["bulk_composition_mole_fraction"] == {
        "AL": 0.95,
        "ZN": 0.05,
    }
    assert record["request"]["pressure_Pa"] == 101325.0
    assert record["result"]["converged"] is True
    assert record["result"]["point_count"] > 10
    assert len(record["evidence"]["sha256"]) == 64
    assert record["limits"]["wall_time_scope"] == (
        "shared_liquid_preflight_validation_and_solidification_solve"
    )
    assert observed_wall_limits == [15.0]
    assert "No diffusion in solid phases after they form." in record["assumptions"]

    result = record["result"]
    assert all(
        right <= left for left, right in zip(result["temperatures_K"], result["temperatures_K"][1:])
    )
    assert all(
        right >= left for left, right in zip(result["fraction_solid"], result["fraction_solid"][1:])
    )
    for solid, liquid in zip(result["fraction_solid"], result["fraction_liquid"]):
        assert solid + liquid == pytest.approx(1.0, abs=1e-8)
    for index, solid in enumerate(result["fraction_solid"]):
        phase_sum = sum(
            values[index] for values in result["solid_phase_cumulative_fraction"].values()
        )
        assert phase_sum == pytest.approx(solid, abs=1e-6)
    for phase_compositions in result["phase_composition_mole_fraction"].values():
        for point in zip(*phase_compositions.values()):
            finite = [value for value in point if value is not None]
            if finite:
                assert sum(finite) == pytest.approx(1.0, abs=1e-6)
    mass_balance = result["elemental_mass_balance"]
    assert mass_balance["all_retained_points_closed"] is True
    assert mass_balance["maximum_absolute_component_error"] < 1e-6
    assert mass_balance["final_reconstructed_bulk_composition_mole_fraction"] == pytest.approx(
        {"AL": 0.95, "ZN": 0.05}, abs=1e-6
    )


def test_solver_stage_wall_timeout_keeps_its_typed_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _copy_alzn_database(tmp_path, monkeypatch)

    def timed_out_solver(*_args, **_kwargs):
        raise kinetics._WallTimeExceededError

    monkeypatch.setattr(kinetics, "_simulate_scheil", timed_out_solver)
    with pytest.raises(
        CalphadTimeoutError,
        match="solidification solver exceeded the shared 15.0-second",
    ):
        _run(path)


def test_impossible_result_cardinality_fails_before_solver_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = [f"E{index}" for index in range(31)] + ["VA"]
    phases = [f"PHASE_{index}" for index in range(127)] + ["LIQUID"]
    manifest = {
        "requested_components": components,
        "requested_phases": phases,
        "assessment_temperature_limits_K": [298.15, 2500.0],
        "assessment_pressure_limits_Pa": [101325.0, 101325.0],
    }
    monkeypatch.setattr(
        kinetics,
        "_load_inspected_database",
        lambda *_args, **_kwargs: (object(), manifest),
    )

    def solver_must_not_start(*_args, **_kwargs):
        raise AssertionError("solver entered after a guaranteed result-size rejection")

    monkeypatch.setattr(kinetics, "_simulate_scheil", solver_must_not_start)
    with pytest.raises(CalphadInputError, match="cardinality cannot fit"):
        run_scheil_solidification(
            "unused.tdb",
            components=components,
            phases=phases,
            independent_composition={},
            start_temperature_K=2000.0,
            step_temperature_K=20.0,
            assessment_temperature_limits_K=[298.15, 2500.0],
            assessment_pressure_limits_Pa=[101325.0, 101325.0],
            max_steps=2048,
            max_result_bytes=16 * 1024 * 1024,
        )


def test_result_cardinality_preflight_uses_temperature_bounded_point_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    physical_components = [f"E{index:02d}" for index in range(12)]
    components = [*physical_components, "VA"]
    phases = [f"PHASE_{index:02d}" for index in range(24)] + ["LIQUID"]
    manifest = {
        "requested_components": components,
        "requested_phases": phases,
        "assessment_temperature_limits_K": [298.15, 2500.0],
        "assessment_pressure_limits_Pa": [101325.0, 101325.0],
    }
    result_limit = 16 * 1024 * 1024
    expected_worst_case_points = 257
    full_step_bound = kinetics._scheil_result_upper_bound_bytes(
        max_steps=2048,
        physical_components=physical_components,
        phases=phases,
        database_manifest=manifest,
    )
    temperature_bounded_result = kinetics._scheil_result_upper_bound_bytes(
        max_steps=expected_worst_case_points,
        physical_components=physical_components,
        phases=phases,
        database_manifest=manifest,
    )
    assert full_step_bound > result_limit
    assert temperature_bounded_result < result_limit

    monkeypatch.setattr(
        kinetics,
        "_load_inspected_database",
        lambda *_args, **_kwargs: (object(), manifest),
    )
    observed_point_bounds: list[int] = []

    class BoundObservedError(Exception):
        pass

    def capture_point_bound(*, max_steps: int, **_kwargs) -> int:
        observed_point_bounds.append(max_steps)
        raise BoundObservedError

    monkeypatch.setattr(kinetics, "_scheil_result_upper_bound_bytes", capture_point_bound)
    with pytest.raises(BoundObservedError):
        run_scheil_solidification(
            "unused.tdb",
            components=components,
            phases=phases,
            independent_composition={},
            start_temperature_K=2000.0,
            step_temperature_K=20.0,
            assessment_temperature_limits_K=[298.15, 2500.0],
            assessment_pressure_limits_Pa=[101325.0, 101325.0],
            max_steps=2048,
            max_result_bytes=result_limit,
        )

    assert observed_point_bounds == [expected_worst_case_points]


def test_scheil_multicomponent_alcocrni_control_is_mass_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise a held-out ordered/disordered multicomponent chemistry path.

    The packaged assessment is a deterministic solver control, never evidence for a
    research alloy or proof of transferable superalloy accuracy.
    """

    path = _copy_alcocrni_database(tmp_path, monkeypatch)
    record = run_scheil_solidification(
        path,
        components=["AL", "CO", "CR", "NI", "VA"],
        phases=_ALCOCRNI_PHASES,
        independent_composition={"AL": 0.10, "CO": 0.20, "CR": 0.15},
        start_temperature_K=2000.0,
        step_temperature_K=20.0,
        stop_liquid_fraction=0.05,
        source="pycalphad Al-Co-Cr-Ni test assessment; solver qualification only",
        license_id=_LICENSE,
        artifact_id="qualification-alcocrni-scheil",
        expected_sha256=_sha256(path),
        expected_size_bytes=path.stat().st_size,
        assessment_scope="test-only Al-Co-Cr-Ni multicomponent solver-path qualification",
        reference_state=_REFERENCE,
        assessment_temperature_limits_K=[298.15, 2500.0],
        assessment_pressure_limits_Pa=[101325.0, 101325.0],
        max_steps=2048,
        wall_time_seconds=20.0,
    )

    result = record["result"]
    assert result["converged"] is True
    assert result["point_count"] >= 10
    assert result["fraction_solid"][-1] >= 0.95
    assert set(result["solid_phase_cumulative_fraction"]) == set(_ALCOCRNI_PHASES) - {"LIQUID"}
    mass_balance = result["elemental_mass_balance"]
    assert mass_balance["all_retained_points_closed"] is True
    assert mass_balance["maximum_absolute_component_error"] < 1e-8
    assert mass_balance["final_reconstructed_bulk_composition_mole_fraction"] == pytest.approx(
        {"AL": 0.10, "CO": 0.20, "CR": 0.15, "NI": 0.55},
        abs=1e-8,
    )


def test_scheil_non_al_cfe_broshe_package_fixture_solver_transfer_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qualify Fe-C solver transfer without treating package fixtures as research data."""

    path = _copy_cfe_broshe_database(tmp_path, monkeypatch)
    stop_liquid_fraction = 0.05
    record = run_scheil_solidification(
        path,
        components=["C", "FE", "VA"],
        phases=_CFE_BROSHE_PHASES,
        independent_composition={"C": 0.02},
        start_temperature_K=2200.0,
        step_temperature_K=10.0,
        stop_liquid_fraction=stop_liquid_fraction,
        source=(
            "pycalphad cfe_broshe.tdb package fixture; deterministic non-Al solver-transfer "
            "control only"
        ),
        license_id=_LICENSE,
        artifact_id="qualification-cfe-broshe-scheil-transfer-control",
        expected_sha256=_sha256(path),
        expected_size_bytes=path.stat().st_size,
        assessment_scope=(
            "package-fixture Fe-C Scheil solver-path transfer control; not research evidence"
        ),
        reference_state="fixture-defined Fe-C reference states",
        assessment_temperature_limits_K=[298.15, 6000.0],
        assessment_pressure_limits_Pa=[101325.0, 101325.0],
        max_steps=2048,
        wall_time_seconds=20.0,
    )

    assert record["database"]["sha256"] == _sha256(path)
    assert record["database"]["artifact_id"] == ("qualification-cfe-broshe-scheil-transfer-control")
    assert "not research evidence" in record["database"]["assessment_scope"]
    assert record["request"]["bulk_composition_mole_fraction"] == pytest.approx(
        {"C": 0.02, "FE": 0.98},
        abs=1e-12,
    )

    # Reaching a result proves the wrapper's single-phase-liquid equilibrium
    # preflight accepted the exact 2200 K starting state before solver entry.
    result = record["result"]
    assert result["converged"] is True
    assert result["point_count"] >= 3
    assert result["fraction_liquid"][0] == pytest.approx(1.0, abs=1e-8)
    assert result["fraction_solid"][0] == pytest.approx(0.0, abs=1e-8)
    assert result["fraction_liquid"][-1] <= stop_liquid_fraction + 1e-12
    assert all(
        right <= left for left, right in zip(result["temperatures_K"], result["temperatures_K"][1:])
    )
    assert all(
        right >= left for left, right in zip(result["fraction_solid"], result["fraction_solid"][1:])
    )
    assert all(
        right <= left
        for left, right in zip(result["fraction_liquid"], result["fraction_liquid"][1:])
    )
    for solid, liquid in zip(result["fraction_solid"], result["fraction_liquid"]):
        assert solid + liquid == pytest.approx(1.0, abs=1e-8)

    mass_balance = result["elemental_mass_balance"]
    assert mass_balance["all_retained_points_closed"] is True
    assert mass_balance["maximum_absolute_component_error"] < 1e-8
    assert mass_balance["final_reconstructed_bulk_composition_mole_fraction"] == pytest.approx(
        {"C": 0.02, "FE": 0.98},
        abs=1e-8,
    )
    assert len(record["evidence"]["sha256"]) == 64


def test_scheil_validator_reconstructs_elemental_inventory_from_each_increment() -> None:
    result = _validate_synthetic(_synthetic_mass_closed_scheil_result())

    assert result["elemental_mass_balance"]["all_retained_points_closed"] is True
    assert result["elemental_mass_balance"]["maximum_absolute_component_error"] < 1e-12


def test_scheil_mass_balance_reconstruction_scales_linearly_in_retained_points() -> None:
    class CountingSeries:
        def __init__(self, values: list[float], counter: dict[str, int]) -> None:
            self._values = values
            self._counter = counter

        def __len__(self) -> int:
            return len(self._values)

        def __getitem__(self, index: int) -> float:
            self._counter["accesses"] += 1
            return self._values[index]

    count = 512
    fraction_solid = [0.9 * index / (count - 1) for index in range(count)]
    fraction_liquid = [1.0 - value for value in fraction_solid]
    increments = [0.0, *(right - left for left, right in zip(fraction_solid, fraction_solid[1:]))]
    counter = {"accesses": 0}

    result = kinetics._validated_scheil_mass_balance(
        bulk_composition={"AL": 0.9, "ZN": 0.1},
        fraction_liquid=CountingSeries(fraction_liquid, counter),
        phase_amounts={"FCC_A1": CountingSeries(increments, counter)},
        cumulative_phase_amounts={"FCC_A1": CountingSeries(fraction_solid, counter)},
        phase_compositions={
            "LIQUID": {
                "AL": CountingSeries([0.9] * count, counter),
                "ZN": CountingSeries([0.1] * count, counter),
            },
            "FCC_A1": {
                "AL": CountingSeries([0.9] * count, counter),
                "ZN": CountingSeries([0.1] * count, counter),
            },
        },
        liquid_phase_name="LIQUID",
    )

    assert result["all_retained_points_closed"] is True
    assert result["maximum_absolute_component_error"] < 1e-12
    assert counter["accesses"] < 16 * count


def test_scheil_validator_rejects_normalized_compositions_that_violate_bulk_mass() -> None:
    raw = copy.deepcopy(_synthetic_mass_closed_scheil_result())
    raw.phase_compositions["FCC_A1"]["AL"][-1] = 0.8
    raw.phase_compositions["FCC_A1"]["ZN"][-1] = 0.2

    with pytest.raises(CalphadExecutionError, match="elemental mass balance does not close"):
        _validate_synthetic(raw)


def test_scheil_validator_requires_every_physical_component_for_every_phase() -> None:
    raw = copy.deepcopy(_synthetic_mass_closed_scheil_result())
    del raw.phase_compositions["FCC_A1"]["ZN"]

    with pytest.raises(CalphadExecutionError, match="missing physical components"):
        _validate_synthetic(raw)


def test_scheil_validator_recomputes_cumulative_amounts_from_increments() -> None:
    raw = copy.deepcopy(_synthetic_mass_closed_scheil_result())
    raw.phase_amounts["FCC_A1"] = [0.0, 0.4, 0.55]

    with pytest.raises(CalphadExecutionError, match="disagrees with retained increments"):
        _validate_synthetic(raw)


def test_scheil_validator_rejects_an_ignored_point_zero_solid_inventory() -> None:
    raw = copy.deepcopy(_synthetic_mass_closed_scheil_result())
    raw.fraction_solid[0] = 0.1
    raw.fraction_liquid[0] = 0.9
    raw.phase_amounts["FCC_A1"] = [0.1, 0.4, 0.45]
    raw.cum_phase_amounts["FCC_A1"] = [0.1, 0.5, 0.95]

    with pytest.raises(CalphadExecutionError, match="initial point"):
        _validate_synthetic(raw)


def test_scheil_rejects_a_start_state_that_is_not_single_phase_liquid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _copy_alzn_database(tmp_path, monkeypatch)

    with pytest.raises(CalphadInputError, match="single-phase liquid"):
        _run(path, start_temperature_k=500.0)


def test_scheil_rejects_pressure_outside_the_qualified_solver_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _copy_alzn_database(tmp_path, monkeypatch)

    with pytest.raises(CalphadInputError, match="pressure_Pa=101325"):
        _run(path, pressure_Pa=200000.0)


def test_scheil_rejects_a_nonconverged_upstream_final_fill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _copy_alzn_database(tmp_path, monkeypatch)

    with pytest.raises(CalphadExecutionError, match="did not reach"):
        _run(path, x_zn=0.3, start_temperature_k=850.0)


def test_scheil_translates_upstream_internal_length_assertions_to_typed_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _copy_alzn_database(tmp_path, monkeypatch)

    def inconsistent_solver(*_args, **_kwargs):
        raise AssertionError("phase amount length mismatch")

    monkeypatch.setattr(kinetics, "_simulate_scheil", inconsistent_solver)
    with pytest.raises(CalphadExecutionError, match="inconsistent result lengths"):
        _run(path)


def test_processing_support_matrix_does_not_mislabel_scheil_as_other_kinetics() -> None:
    support = processing_method_support()

    assert support["scheil_gulliver"]["status"] == "qualified_runtime"
    assert support["back_diffusion"] == {
        "status": "qualified_isolated_runtime",
        "solver": "kawin==0.5.0",
        "tool": "materials_run_diffusion_1d",
        "scope": "post_solidification_single_phase_1d_only",
        "required_evidence": [
            "diffusion/mobility data with units and provenance",
            "length scale or dendrite-arm-spacing model",
            "isothermal duration and zero-flux boundary applicability",
        ],
    }
    assert support["mobility_diffusion"]["status"] == "qualified_isolated_runtime"
    assert support["mobility_diffusion"]["tools"] == [
        "materials_transport_coefficients",
        "materials_run_diffusion_1d",
    ]
    assert support["precipitation"]["status"] == "qualified_isolated_runtime"
    assert support["precipitation"]["tool"] == "materials_run_binary_precipitation_kwn"
    assert "binary isothermal spherical KWN" in support["precipitation"]["scope"]
    assert support["phase_field"]["status"] == "requires_external_hpc_solver"
