from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest
from ultra_deepagents.kinetics_runtime import execute_request, runtime_support
from ultra_deepagents.kinetics_runtime.errors import (
    KineticsExecutionError,
    KineticsInputError,
    KineticsUnsupportedError,
)
from ultra_deepagents.kinetics_runtime.runner import _reconstruct_binary_bulk_solute

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kawin") is None,
    reason="Kawin tests execute only in the pinned isolated NumPy-2 kinetics runtime",
)

_SCHEMA = "ultra.materials.kinetics-request.v1"
_LIMITS = {"wall_time_seconds": 20.0, "max_result_bytes": 2 * 1024 * 1024}


@pytest.fixture
def alzr_database(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    from kawin.tests.databases import ALZR_TDB

    path = tmp_path / "qualification-alzr.tdb"
    path.write_text(ALZR_TDB, encoding="utf-8")
    payload = path.read_bytes()
    return path, {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "artifact_id": "kawin-test-alzr",
        "source": "Wang et al. Al-Zr assessment, copied from the Kawin test fixture",
        "license_id": "Kawin test fixture; qualification use only",
        "assessment_scope": "test-only Al-Zr solver qualification",
        "reference_state": "fixture-defined standard element reference states",
        "assessment_temperature_limits_K": [298.15, 6000.0],
        "assessment_pressure_limits_Pa": [101325.0, 101325.0],
    }


@pytest.fixture
def nicral_database(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    from kawin.tests.databases import NICRAL_TDB

    path = tmp_path / "qualification-nicral.tdb"
    path.write_text(NICRAL_TDB, encoding="utf-8")
    payload = path.read_bytes()
    return path, {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "artifact_id": "kawin-test-nicral",
        "source": "Dupin Ni-Cr-Al assessment and mobility data, copied from Kawin tests",
        "license_id": "Kawin test fixture; qualification use only",
        "assessment_scope": "test-only Ni-Cr-Al transport qualification",
        "reference_state": "fixture-defined standard element reference states",
        "assessment_temperature_limits_K": [298.15, 10000.0],
        "assessment_pressure_limits_Pa": [101325.0, 101325.0],
    }


def _transport(database: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA,
        "operation": "transport_coefficients",
        "database": database,
        "components": ["AL", "ZR"],
        "phase": "FCC_A1",
        "independent_composition_mole_fraction": {"ZR": 0.004},
        "temperature_K": 723.15,
        "pressure_Pa": 101325.0,
        "limits": dict(_LIMITS),
    }


def _diffusion(
    database: dict[str, object],
    *,
    mesh_cells: int = 64,
    max_solver_steps: int = 100_000,
) -> dict[str, object]:
    epsilon = 1e-12
    return {
        "schema_version": _SCHEMA,
        "operation": "single_phase_diffusion_1d",
        "database": database,
        "components": ["AL", "ZR"],
        "phase": "FCC_A1",
        "temperature_K": 723.15,
        "pressure_Pa": 101325.0,
        "duration_s": 1e6,
        "domain_m": [-5e-6, 5e-6],
        "mesh_cells": mesh_cells,
        "max_solver_steps": max_solver_steps,
        "boundary_condition": {"kind": "zero_flux"},
        "initial_profile": {
            "coordinates_m": [-5e-6, -epsilon, epsilon, 5e-6],
            "independent_composition_mole_fraction": {"ZR": [0.002, 0.002, 0.006, 0.006]},
            "interpolation": "linear",
            "source": "synthetic step profile for analytical Fick benchmark",
        },
        "application": {
            "kind": "post_solidification_back_diffusion",
            "length_scale_source": "synthetic ten-micrometre qualification domain",
            "solidification_coupling": "post_solidification_only",
        },
        "limits": dict(_LIMITS),
    }


def _precipitation(
    database: dict[str, object],
    *,
    max_solver_steps: int = 200_000,
) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA,
        "operation": "binary_precipitation_kwn",
        "database": database,
        "components": ["AL", "ZR"],
        "matrix_phase": "FCC_A1",
        "precipitate_phase": "AL3ZR",
        "initial_solute_mole_fraction": 0.004,
        "temperature_K": 723.15,
        "temperature_source": "synthetic isotherm for solver qualification",
        "pressure_Pa": 101325.0,
        "duration_s": 100.0,
        "driving_force_method": "tangent",
        "matrix": {
            "molar_volume_m3_per_mol": 1e-5,
            "atoms_per_unit_cell": 4,
            "bulk_nucleation_site_density_per_m3": 1e30,
            "grain_boundary_energy_J_per_m2": 0.3,
            "source": "synthetic qualification parameters",
        },
        "precipitate": {
            "molar_volume_m3_per_mol": 1e-5,
            "atoms_per_unit_cell": 4,
            "interfacial_energy_J_per_m2": 0.1,
            "constant_elastic_strain_energy_J_per_m3": 0.0,
            "infinite_precipitate_diffusion": True,
            "source": "synthetic qualification parameters",
        },
        "nucleation": {
            "site": "bulk",
            "source": "synthetic homogeneous nucleation assumption",
        },
        "population_balance": {
            "min_radius_m": 1e-10,
            "max_radius_m": 1e-8,
            "bins": 50,
            "adaptive": False,
            "max_history_points": 128,
        },
        "max_solver_steps": max_solver_steps,
        "limits": dict(_LIMITS),
    }


def _artifact(path: Path, *, artifact_id: str) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "artifact_id": artifact_id,
        "source": "synthetic phase-field qualification fixture",
        "license_id": "CC0-1.0",
        "assessment_scope": "validator contract qualification only",
    }


def _phase_field_readiness(tmp_path: Path) -> dict[str, object]:
    free_energy = tmp_path / "free-energy.json"
    initial_c = tmp_path / "initial-c.npy"
    mobility = tmp_path / "mobility.json"
    gradient = tmp_path / "gradient.json"
    held_out = tmp_path / "held-out.csv"
    free_energy.write_text('{"model":"double-well"}', encoding="utf-8")
    initial_c.write_bytes(b"synthetic initial condition")
    mobility.write_text('{"M":1e-18}', encoding="utf-8")
    gradient.write_text('{"kappa":1e-14}', encoding="utf-8")
    held_out.write_text("time_s,length_m\n0,1e-6\n", encoding="utf-8")
    return {
        "schema_version": _SCHEMA,
        "operation": "phase_field_readiness",
        "solver_target": {
            "name": "external-phase-field-solver",
            "version": "qualification-pending",
            "execution_environment": "isolated HPC worker with network disabled",
            "image_digest": "sha256:" + "a" * 64,
        },
        "model": {
            "temperature_K": 1000.0,
            "fields": [
                {
                    "name": "c_al",
                    "kind": "conserved",
                    "equation": "Cahn-Hilliard",
                    "physical_quantity": "aluminium mole fraction",
                    "unit": "1",
                    "initial_condition_artifact": _artifact(initial_c, artifact_id="initial-c-v1"),
                }
            ],
            "free_energy": {
                "model": "provenance-bound double-well qualification fixture",
                "energy_density_unit": "J_per_m3",
                "phases": ["matrix", "precipitate"],
                "artifact": _artifact(free_energy, artifact_id="free-energy-v1"),
                "temperature_limits_K": [900.0, 1100.0],
            },
            "kinetic_coefficients": [
                {
                    "field": "c_al",
                    "kind": "mobility",
                    "value": 1e-18,
                    "unit": "m5_per_J_s",
                    "source_artifact": _artifact(mobility, artifact_id="mobility-v1"),
                    "temperature_limits_K": [900.0, 1100.0],
                }
            ],
            "gradient_energy_coefficients": [
                {
                    "field_i": "c_al",
                    "field_j": "c_al",
                    "value": 1e-14,
                    "unit": "J_per_m",
                    "source_artifact": _artifact(gradient, artifact_id="gradient-v1"),
                    "temperature_limits_K": [900.0, 1100.0],
                }
            ],
        },
        "domain_mesh": {
            "dimensions": 2,
            "extent_m": [1e-6, 1e-6],
            "cells": [128, 128],
            "spatial_discretization": "second-order finite elements",
            "mesh_source": "synthetic square qualification domain",
        },
        "boundary_conditions": [
            {
                "field": "c_al",
                "boundary": "all paired faces",
                "kind": "periodic",
                "unit": "1",
                "source": "synthetic periodic qualification model",
            }
        ],
        "integration": {
            "duration_s": 1000.0,
            "initial_time_step_s": 0.01,
            "maximum_time_step_s": 1.0,
            "time_integrator": "BDF2 with startup step",
            "nonlinear_relative_tolerance": 1e-8,
            "linear_relative_tolerance": 1e-10,
            "maximum_nonlinear_iterations": 50,
        },
        "convergence_plan": {
            "mesh_characteristic_lengths_m": [2e-8, 1e-8, 5e-9],
            "maximum_time_steps_s": [1.0, 0.5, 0.25],
            "observables": ["total free energy", "characteristic domain size"],
            "maximum_relative_change": 0.02,
        },
        "validation_plan": {
            "held_out_dataset": _artifact(held_out, artifact_id="held-out-v1"),
            "calibration_artifact_ids": ["calibration-v1"],
            "calibration_and_validation_disjoint": True,
            "metrics": [
                {
                    "name": "relative domain-size error",
                    "unit": "1",
                    "acceptance_operator": "<=",
                    "acceptance_value": 0.1,
                }
            ],
        },
        "limits": dict(_LIMITS),
    }


def test_runtime_is_exactly_isolated_numpy_two() -> None:
    support = runtime_support()

    assert support["runtime"] == {
        "name": "ultra-isolated-kawin",
        "versions": {
            "kawin": "0.5.0",
            "numpy": "2.4.6",
            "pycalphad": "0.11.2",
            "scipy": "1.17.1",
        },
        "shared_numpy_1_26_sandbox_modified": False,
        "pressure_behavior": "Kawin 0.5 thermodynamic calls use a fixed 101325 Pa",
    }
    assert support["operations"]["phase_field"]["status"] == "external_hpc_required"
    assert support["operations"]["coupled_solidification_back_diffusion"]["status"] == (
        "unsupported"
    )
    assert support["operations"]["phase_field_readiness"]["status"] == ("contract_validation_only")


def test_binary_df_dq_reports_only_assessed_solute_and_is_deterministic(
    alzr_database: tuple[Path, dict[str, object]],
) -> None:
    root, database = alzr_database[0].parent, alzr_database[1]

    first = execute_request(_transport(database), workspace_root=root)
    second = execute_request(_transport(database), workspace_root=root)

    assert first["result"]["tracer_diffusivity_m2_per_s"] == {
        "ZR": pytest.approx(2.544961743567114e-19, rel=1e-10)
    }
    assert "AL" not in first["result"]["tracer_diffusivity_m2_per_s"]
    assert first["result"]["interdiffusivity_m2_per_s"] == [
        [pytest.approx(2.544961743567114e-19, rel=1e-10)]
    ]
    assert first["result"]["cross_diffusion_supported"] is False
    assert first["evidence"]["sha256"] == second["evidence"]["sha256"]


def test_multicomponent_mf_mq_transport_has_cross_diffusion_matrix(
    nicral_database: tuple[Path, dict[str, object]],
) -> None:
    root, database = nicral_database[0].parent, nicral_database[1]
    request = {
        "schema_version": _SCHEMA,
        "operation": "transport_coefficients",
        "database": database,
        "components": ["NI", "CR", "AL"],
        "phase": "FCC_A1",
        "independent_composition_mole_fraction": {"CR": 0.05, "AL": 0.05},
        "temperature_K": 1073.0,
        "pressure_Pa": 101325.0,
        "limits": dict(_LIMITS),
    }

    result = execute_request(request, workspace_root=root)["result"]

    assert result["transport_parameter_family_used"] == "MF/MQ mobility"
    assert result["cross_diffusion_supported"] is True
    assert set(result["tracer_diffusivity_m2_per_s"]) == {"NI", "CR", "AL"}
    assert all(value > 0 for value in result["tracer_diffusivity_m2_per_s"].values())
    assert len(result["interdiffusivity_m2_per_s"]) == 2
    assert all(len(row) == 2 for row in result["interdiffusivity_m2_per_s"])


def test_database_digest_and_workspace_containment_fail_closed(
    alzr_database: tuple[Path, dict[str, object]], tmp_path: Path
) -> None:
    root, database = alzr_database[0].parent, alzr_database[1]
    tampered = copy.deepcopy(_transport(database))
    tampered["database"]["sha256"] = "0" * 64
    with pytest.raises(KineticsInputError, match="sha256 does not match"):
        execute_request(tampered, workspace_root=root)

    outside = tmp_path.parent / "outside-kinetics.tdb"
    outside.write_bytes(alzr_database[0].read_bytes())
    escaped = copy.deepcopy(_transport(database))
    escaped["database"]["path"] = str(outside)
    escaped["database"]["sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    escaped["database"]["size_bytes"] = outside.stat().st_size
    with pytest.raises(KineticsInputError, match="escapes the workspace"):
        execute_request(escaped, workspace_root=root)


def test_transport_rejects_missing_kinetics_and_unknown_fields(
    alzr_database: tuple[Path, dict[str, object]],
) -> None:
    root, database = alzr_database[0].parent, alzr_database[1]
    no_kinetics = _transport(database)
    no_kinetics["phase"] = "AL3ZR"
    with pytest.raises(KineticsUnsupportedError, match="no MF/MQ or DF/DQ"):
        execute_request(no_kinetics, workspace_root=root)

    typo = _transport(database)
    typo["temperatur_K"] = typo["temperature_K"]
    with pytest.raises(KineticsInputError, match="unknown keys"):
        execute_request(typo, workspace_root=root)


def test_post_solidification_diffusion_matches_fick_step_and_closes_mass(
    alzr_database: tuple[Path, dict[str, object]],
) -> None:
    import numpy as np
    from scipy.special import erf

    root, database = alzr_database[0].parent, alzr_database[1]
    response = execute_request(_diffusion(database), workspace_root=root)
    result = response["result"]
    coordinates = np.asarray(result["coordinates_m"])
    final = np.asarray(result["final_composition_mole_fraction"]["ZR"])
    diffusivity = execute_request(_transport(database), workspace_root=root)["result"][
        "interdiffusivity_m2_per_s"
    ][0][0]
    analytic = 0.004 + 0.002 * erf(coordinates / (2 * np.sqrt(diffusivity * 1e6)))

    assert np.max(np.abs(final - analytic)) < 7e-6
    assert result["numerical_verification"]["absolute_mass_closure_error"]["ZR"] < 1e-12
    assert result["numerical_verification"]["grid_convergence_assessed"] is False
    assert response["application"]["solidification_coupling"] == "post_solidification_only"
    assert any("no moving solid/liquid interface" in warning for warning in response["warnings"])


def test_diffusion_grid_refinement_reduces_analytical_error(
    alzr_database: tuple[Path, dict[str, object]],
) -> None:
    import numpy as np
    from scipy.special import erf

    root, database = alzr_database[0].parent, alzr_database[1]
    diffusivity = execute_request(_transport(database), workspace_root=root)["result"][
        "interdiffusivity_m2_per_s"
    ][0][0]
    errors: list[float] = []
    for cells in (16, 32, 64):
        result = execute_request(_diffusion(database, mesh_cells=cells), workspace_root=root)[
            "result"
        ]
        coordinates = np.asarray(result["coordinates_m"])
        final = np.asarray(result["final_composition_mole_fraction"]["ZR"])
        analytic = 0.004 + 0.002 * erf(coordinates / (2 * np.sqrt(diffusivity * 1e6)))
        errors.append(float(np.sqrt(np.mean((final - analytic) ** 2))))

    assert errors[2] < errors[1] < errors[0]
    assert errors[0] / errors[1] > 3.0
    assert errors[1] / errors[2] > 3.0


def test_diffusion_step_limit_and_coupled_back_diffusion_fail_closed(
    alzr_database: tuple[Path, dict[str, object]],
) -> None:
    root, database = alzr_database[0].parent, alzr_database[1]
    with pytest.raises(KineticsExecutionError, match="max_solver_steps"):
        execute_request(_diffusion(database, max_solver_steps=1), workspace_root=root)

    request = _diffusion(database)
    request["application"]["solidification_coupling"] = "moving_interface"
    with pytest.raises(KineticsUnsupportedError, match="only after solidification"):
        execute_request(request, workspace_root=root)


def test_binary_kwn_executes_with_mass_closure_and_explicit_assumptions(
    alzr_database: tuple[Path, dict[str, object]],
) -> None:
    root, database = alzr_database[0].parent, alzr_database[1]

    response = execute_request(_precipitation(database), workspace_root=root)
    result = response["result"]

    assert result["final"]["time_s"] == 100.0
    assert result["final"]["precipitate_volume_fraction"] > 0
    assert result["final"]["precipitate_number_density_per_m3"] > 0
    assert result["final"]["average_equivalent_spherical_radius_m"] > 0
    assert result["numerical_verification"]["maximum_absolute_solute_mass_closure_error"] < (1e-12)
    assert result["history"]["retained_point_count"] <= 128
    assert response["scientific_status"].endswith("requires_bin_and_experimental_validation")
    assert "Infinite diffusion within precipitates." in response["assumptions"]
    assert response["result"]["upstream_quantity_contract"]["PBM.PSD"].endswith("per m3")


def test_fraction_weighted_fconc_mass_balance_is_not_double_weighted() -> None:
    import numpy as np

    bulk = 0.1
    volume_fraction = np.asarray([0.0, 0.2, 0.4])
    matrix_composition = np.asarray([0.1, 0.075, 0.05])
    # Kawin's fconc is already the fraction-weighted precipitate contribution.
    fconc = bulk - (1.0 - volume_fraction) * matrix_composition

    reconstructed = _reconstruct_binary_bulk_solute(
        matrix_composition,
        volume_fraction,
        fconc,
    )
    incorrectly_double_weighted = (
        1.0 - volume_fraction
    ) * matrix_composition + volume_fraction * fconc

    assert reconstructed == pytest.approx([bulk, bulk, bulk], abs=1e-15)
    assert not np.allclose(incorrectly_double_weighted[1:], bulk, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize(
    ("mutation", "error_type", "match"),
    [
        (
            lambda request: request["nucleation"].update(site="dislocations"),
            KineticsUnsupportedError,
            "bulk nucleation",
        ),
        (
            lambda request: request["population_balance"].update(adaptive=True),
            KineticsUnsupportedError,
            "adaptive",
        ),
        (
            lambda request: request["precipitate"].update(infinite_precipitate_diffusion=False),
            KineticsUnsupportedError,
            "infinite precipitate diffusion",
        ),
    ],
)
def test_precipitation_model_scope_fails_closed(
    alzr_database: tuple[Path, dict[str, object]],
    mutation,
    error_type,
    match: str,
) -> None:
    root, database = alzr_database[0].parent, alzr_database[1]
    request = _precipitation(database)
    mutation(request)
    with pytest.raises(error_type, match=match):
        execute_request(request, workspace_root=root)


def test_precipitation_step_limit_rejects_partial_result(
    alzr_database: tuple[Path, dict[str, object]],
) -> None:
    root, database = alzr_database[0].parent, alzr_database[1]
    with pytest.raises(KineticsExecutionError, match="max_solver_steps"):
        execute_request(_precipitation(database, max_solver_steps=10), workspace_root=root)


def test_phase_field_and_unknown_operations_never_fall_back_to_toy_solvers(tmp_path: Path) -> None:
    with pytest.raises(KineticsUnsupportedError, match="external solver"):
        execute_request(
            {"schema_version": _SCHEMA, "operation": "phase_field"},
            workspace_root=tmp_path,
        )
    with pytest.raises(KineticsUnsupportedError, match="unknown kinetics operation"):
        execute_request(
            {"schema_version": _SCHEMA, "operation": "renamed_scheil"},
            workspace_root=tmp_path,
        )


def test_phase_field_readiness_validates_contract_without_claiming_execution(
    tmp_path: Path,
) -> None:
    response = execute_request(_phase_field_readiness(tmp_path), workspace_root=tmp_path)

    assert response["result"] == {
        "status": "submission_contract_complete_not_executed",
        "execution_performed": False,
        "pde_solution_available": False,
        "convergence_assessed": False,
        "held_out_validation_performed": False,
        "external_solver_adapter_qualification_required": True,
    }
    assert response["scientific_status"].endswith("external_execution_not_qualified")
    assert response["model"]["kinetic_coefficients"][0]["kind"] == "mobility"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda request: request["model"]["kinetic_coefficients"][0].update(kind="relaxation"),
            "must be 'mobility'",
        ),
        (
            lambda request: request["convergence_plan"].update(
                mesh_characteristic_lengths_m=[2e-8, 1e-8]
            ),
            "3 to 16 refinement levels",
        ),
        (
            lambda request: request["validation_plan"].update(
                calibration_and_validation_disjoint=False
            ),
            "explicitly held out",
        ),
    ],
)
def test_phase_field_readiness_fails_closed_on_scientific_contract_gaps(
    tmp_path: Path, mutation, match: str
) -> None:
    request = _phase_field_readiness(tmp_path)
    mutation(request)
    with pytest.raises(KineticsInputError, match=match):
        execute_request(request, workspace_root=tmp_path)
