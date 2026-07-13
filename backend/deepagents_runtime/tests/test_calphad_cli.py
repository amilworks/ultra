"""Pinned-environment scientific test for the fixed typed CALPHAD CLI."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import ultra_deepagents.materials.calphad as calphad_runtime
import ultra_deepagents.materials.calphad_cli as cli
from ultra_deepagents.materials.calphad_cli import REQUEST_SCHEMA_VERSION

pycalphad = pytest.importorskip("pycalphad")


def test_typed_cli_real_pycalphad_0_11_2_inspection_and_equilibrium(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Exercise inspection chaining and v2 science in the exact release stack."""

    assert pycalphad.__version__ == "0.11.2"
    catalog_root = Path(__file__).resolve().parents[1] / "materials_data" / "calphad"
    output_root = tmp_path / "outputs" / "calphad"
    output_root.mkdir(parents=True)
    monkeypatch.setattr(cli, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(cli, "EMBEDDED_REGISTRY_ROOT", catalog_root)
    database = {"kind": "embedded", "database_id": "nist-al-co-w-wang-2017"}
    registry = json.loads((catalog_root / "manifest.json").read_text())
    registry_entry = next(
        entry for entry in registry["databases"] if entry["database_id"] == database["database_id"]
    )
    all_phases = registry_entry["phases"]
    inspection = cli.execute_request(
        {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "operation": "inspect",
            "runtime_image_id": "sha256:" + "f" * 64,
            "database": database,
            "selection": {
                "components": ["AL", "CO", "W", "VA"],
                "phases": all_phases,
            },
        }
    )
    equilibrium = cli.execute_request(
        {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "operation": "equilibrium",
            "runtime_image_id": "sha256:" + "f" * 64,
            "database": database,
            "selection": {
                "components": ["AL", "CO", "W", "VA"],
                "phases": all_phases,
            },
            "inspection_artifact_sha256": inspection["artifact"]["sha256"],
            "conditions": {
                "temperatures_K": [1173.0],
                "pressures_Pa": [101325.0],
                # This is the historically dangerous parameterization: W is
                # caller-dependent. The typed boundary must rewrite it to the
                # reviewed AL-dependent CO/W checkpoint before hashing.
                "independent_compositions": {"AL": [0.675], "CO": [0.26]},
            },
        }
    )
    evidence_path = output_root / "equilibrium" / f"{equilibrium['artifact']['sha256']}.json"
    evidence = json.loads(evidence_path.read_text())
    assert evidence["request"]["conditions"]["independent_compositions"] == {
        "CO": [0.26],
        "W": [0.065],
    }
    assert evidence["result"]["schema_version"] == "ultra.calphad.equilibrium.v2"
    runtime_request = evidence["result"]["request"]
    assert runtime_request["dependent_component"] == "AL"
    assert runtime_request["conditions"]["independent_compositions"] == {
        "CO": {"units": "mole_fraction", "values": [0.26]},
        "W": {"units": "mole_fraction", "values": [0.065]},
    }
    point = evidence["result"]["result"]["points"][0]
    assert [phase["name"] for phase in point["stable_phases"]] == [
        "AL4W",
        "AL5CO2",
        "BCC_B2",
    ]
    assert point["GM_J_per_mol"] == pytest.approx(-85970.06746, abs=1e-4)
    assert point["stable_phase_vertices"]
    assert point["chemical_potentials_J_per_mol"]
    assert point["maximum_bulk_composition_residual"] <= 1e-8
    assert point["gibbs_euler_residual_J_per_mol"] <= 1e-3


def test_typed_cli_real_dat_resource_format_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise a real ChemSage DAT through the exact resource request/evidence chain."""

    assert pycalphad.__version__ == "0.11.2"
    database_root = Path(pycalphad.__file__).resolve().parent / "tests" / "databases"
    source_paths = sorted(database_root.glob("*.dat"))
    if not source_paths:
        pytest.skip("the pinned pycalphad wheel does not include a ChemSage fixture")
    assert len(source_paths) == 7

    workspace = tmp_path / "workspace"
    staged_root = workspace / ".ultra" / "calphad" / "staged"
    output_root = tmp_path / "outputs" / "calphad"
    staged_root.mkdir(parents=True)
    output_root.mkdir(parents=True)
    source_path = source_paths[0]
    database_path = staged_root / "qualified.dat"
    shutil.copyfile(source_path, database_path)
    payload = database_path.read_bytes()

    monkeypatch.setattr(cli, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(cli, "STAGED_DATABASE_ROOTS", (staged_root,))
    monkeypatch.setattr(cli, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(calphad_runtime, "_is_fixture_database", lambda *_args: False)
    database = {
        "kind": "resource",
        "database_id": "qualified-chemsage-dat",
        "path": str(database_path),
        "resource_id": "resource-qualified-dat",
        "database_format": "dat",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "source": "test-only pinned pycalphad ChemSage parser corpus",
        "license_id": "test-only parser fixture",
        "assessment_scope": "test-only exact DAT request/evidence format qualification",
        "reference_state": "fixture-defined reference state",
        "temperature_limits_K": [1.0, 10_000.0],
        "assessment_pressure_limits_Pa": [101325.0, 101325.0],
        "binding_schema": "ultra.selected_resource.v1",
        "binding_authority": "control_resource_catalog",
        "declaration_authority": "resource_owner",
    }

    result = cli.execute_request(
        {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "operation": "inspect",
            "runtime_image_id": "sha256:" + "f" * 64,
            "database": database,
            "selection": {"components": None, "phases": None},
        }
    )
    evidence_path = output_root / "inspection" / f"{result['artifact']['sha256']}.json"
    evidence = json.loads(evidence_path.read_text())

    assert evidence["database_binding"]["database_format"] == "dat"
    assert evidence["result"]["format"] == "dat"
    assert Path(evidence["result"]["path"]).suffix == ".dat"


def test_typed_cli_real_scheil_alcocrni_is_mass_closed_and_retains_va(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Qualify the typed CLI across a held-out four-component chemistry."""

    assert pycalphad.__version__ == "0.11.2"
    database_root = Path(pycalphad.__file__).resolve().parent / "tests" / "databases"
    source_path = database_root / "alcocrni.tdb"
    assert source_path.is_file()
    workspace = tmp_path / "workspace"
    staged_root = workspace / ".ultra" / "calphad" / "staged"
    output_root = tmp_path / "outputs" / "calphad"
    staged_root.mkdir(parents=True)
    output_root.mkdir(parents=True)
    database_path = staged_root / "qualified-alcocrni.tdb"
    shutil.copyfile(source_path, database_path)
    payload = database_path.read_bytes()
    monkeypatch.setattr(cli, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(cli, "STAGED_DATABASE_ROOTS", (staged_root,))
    monkeypatch.setattr(cli, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(calphad_runtime, "_is_fixture_database", lambda *_args: False)
    phases = [
        "BCC_A2",
        "BCC_B2",
        "FCC_A1",
        "HCP_A3",
        "L12_FCC",
        "LIQUID",
        "SIGMA_SGTE",
    ]
    database = {
        "kind": "resource",
        "database_id": "qualification-alcocrni-v1",
        "path": str(database_path),
        "resource_id": "resource-qualification-alcocrni",
        "database_format": "tdb",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "source": "pycalphad Al-Co-Cr-Ni test assessment; solver qualification only",
        "license_id": "MIT pycalphad test fixture; qualification use only",
        "assessment_scope": ("test-only Al-Co-Cr-Ni multicomponent typed Scheil qualification"),
        "reference_state": "fixture-defined standard element reference states",
        "temperature_limits_K": [298.15, 2500.0],
        "assessment_pressure_limits_Pa": [101325.0, 101325.0],
        "binding_schema": "ultra.selected_resource.v1",
        "binding_authority": "control_resource_catalog",
        "declaration_authority": "resource_owner",
    }
    runtime_image_id = "sha256:" + "f" * 64
    inspection = cli.execute_request(
        {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "operation": "inspect",
            "runtime_image_id": runtime_image_id,
            "database": database,
            "selection": {
                "components": ["AL", "CO", "CR", "NI", "VA"],
                "phases": phases,
            },
        }
    )
    scheil = cli.execute_request(
        {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "operation": "scheil",
            "runtime_image_id": runtime_image_id,
            "database": database,
            "selection": {
                "components": ["AL", "CO", "CR", "NI", "VA"],
                "phases": phases,
            },
            "inspection_artifact_sha256": inspection["artifact"]["sha256"],
            "conditions": {
                "independent_composition_mole_fraction": {
                    "CO": 0.20,
                    "CR": 0.15,
                    "NI": 0.55,
                },
                "start_temperature_K": 2000.0,
                "step_temperature_K": 20.0,
                "pressure_Pa": 101325.0,
                "stop_liquid_fraction": 0.05,
            },
        }
    )

    evidence_path = output_root / "scheil" / f"{scheil['artifact']['sha256']}.json"
    evidence = json.loads(evidence_path.read_text())
    assert evidence["operation"] == "scheil"
    assert evidence["request"]["selection"]["components"] == [
        "AL",
        "CO",
        "CR",
        "NI",
        "VA",
    ]
    result = evidence["result"]["result"]
    assert result["converged"] is True
    assert result["point_count"] >= 10
    assert result["fraction_solid"][-1] >= 0.95
    assert result["elemental_mass_balance"]["all_retained_points_closed"] is True
    assert result["elemental_mass_balance"]["maximum_absolute_component_error"] < 1e-8
    assert result["elemental_mass_balance"][
        "final_reconstructed_bulk_composition_mole_fraction"
    ] == pytest.approx({"AL": 0.10, "CO": 0.20, "CR": 0.15, "NI": 0.55}, abs=1e-8)
    limits = evidence["result"]["limits"]
    assert 0 < limits["conservative_result_upper_bound_bytes"] <= 16 * 1024 * 1024
    assert {
        key: value
        for key, value in limits.items()
        if key != "conservative_result_upper_bound_bytes"
    } == {
        "max_steps": 2048,
        "wall_time_seconds": 30.0,
        "wall_time_scope": "shared_liquid_preflight_validation_and_solidification_solve",
        "max_result_bytes": 16 * 1024 * 1024,
    }
