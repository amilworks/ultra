"""Selected-resource and immutable-image boundaries for Kawin tools."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.tools import ToolRuntime
from ultra_deepagents.code_execution.docker import (
    DockerSandboxBackend,
    DockerSandboxConfig,
)
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.kinetics_tools import (
    FIXED_MAX_RESULT_BYTES,
    FIXED_MAX_SOLVER_STEPS,
    FIXED_WALL_TIME_SECONDS,
    QUALIFIED_VERSIONS,
    KineticsToolError,
    _database_request,
    build_kinetics_tools,
    execute_kinetics_request_typed,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _database() -> dict[str, Any]:
    return {
        "path": "/workspace/.ultra/calphad/staged/" + "b" * 64 + ".tdb",
        "sha256": "b" * 64,
        "size_bytes": 100,
        "artifact_id": "file_kinetics_1",
        "source": "qualification source",
        "license_id": "CC-BY-4.0",
        "assessment_scope": "qualification",
        "reference_state": "SER",
        "assessment_temperature_limits_K": [300.0, 2000.0],
        "assessment_pressure_limits_Pa": [101325.0, 101325.0],
    }


def _request() -> dict[str, Any]:
    return {
        "schema_version": "ultra.materials.kinetics-request.v1",
        "operation": "transport_coefficients",
        "database": _database(),
        "components": ["AL", "ZR"],
        "phase": "FCC_A1",
        "independent_composition_mole_fraction": {"ZR": 0.004},
        "temperature_K": 723.15,
        "pressure_Pa": 101325.0,
        "limits": {
            "wall_time_seconds": FIXED_WALL_TIME_SECONDS,
            "max_result_bytes": FIXED_MAX_RESULT_BYTES,
        },
    }


def _diffusion_request(*, cells: int = 8) -> dict[str, Any]:
    return {
        "schema_version": "ultra.materials.kinetics-request.v1",
        "operation": "single_phase_diffusion_1d",
        "database": _database(),
        "components": ["AL", "ZR"],
        "phase": "FCC_A1",
        "temperature_K": 723.15,
        "pressure_Pa": 101325.0,
        "duration_s": 1e6,
        "domain_m": [-5e-6, 5e-6],
        "mesh_cells": cells,
        "max_solver_steps": FIXED_MAX_SOLVER_STEPS,
        "boundary_condition": {"kind": "zero_flux"},
        "initial_profile": {
            "coordinates_m": [-5e-6, -1e-12, 1e-12, 5e-6],
            "independent_composition_mole_fraction": {"ZR": [0.002, 0.002, 0.006, 0.006]},
            "interpolation": "linear",
            "source": "synthetic qualification profile",
        },
        "application": {"kind": "generic_single_phase_diffusion"},
        "limits": {
            "wall_time_seconds": FIXED_WALL_TIME_SECONDS,
            "max_result_bytes": FIXED_MAX_RESULT_BYTES,
        },
    }


def _kwn_request(*, bins: int = 25) -> dict[str, Any]:
    return {
        "schema_version": "ultra.materials.kinetics-request.v1",
        "operation": "binary_precipitation_kwn",
        "database": _database(),
        "components": ["AL", "ZR"],
        "matrix_phase": "FCC_A1",
        "precipitate_phase": "AL3ZR",
        "initial_solute_mole_fraction": 0.004,
        "temperature_K": 723.15,
        "temperature_source": "synthetic qualification isotherm",
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
            "bins": bins,
            "adaptive": False,
            "max_history_points": 128,
        },
        "max_solver_steps": FIXED_MAX_SOLVER_STEPS,
        "limits": {
            "wall_time_seconds": FIXED_WALL_TIME_SECONDS,
            "max_result_bytes": FIXED_MAX_RESULT_BYTES,
        },
    }


class _FakeKineticsBackend:
    def __init__(self, root: Path) -> None:
        self.workspace_dir = root / "workspace"
        self.outputs_dir = root / "outputs"
        self.workspace_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)
        self.config = SimpleNamespace(
            image="sha256:" + "a" * 64,
            network="none",
            no_new_privileges=True,
            cpus=2.0,
            memory="8g",
            pids_limit=256,
            timeout_seconds=30,
            output_limit_bytes=FIXED_MAX_RESULT_BYTES + 128 * 1024,
            gpus="",
        )
        self.commands: list[str] = []
        self.requests: list[dict[str, Any]] = []
        self.mutation = ""

    def _result(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request["operation"]
        if operation == "transport_coefficients":
            payload: dict[str, Any] = {
                "tracer_diffusivity_m2_per_s": {"ZR": 2.5e-19},
                "interdiffusivity_m2_per_s": [[2.5e-19]],
                "interdiffusivity_rows": ["ZR"],
                "interdiffusivity_columns": ["ZR"],
                "reference_component": "AL",
                "transport_parameter_family_used": "DF/DQ direct diffusivity",
            }
        elif operation == "single_phase_diffusion_1d":
            cells = request["mesh_cells"]
            coordinates = [
                request["domain_m"][0]
                + (index + 0.5) * (request["domain_m"][1] - request["domain_m"][0]) / cells
                for index in range(cells)
            ]
            solute = [0.004] * cells
            payload = {
                "coordinates_m": coordinates,
                "initial_composition_mole_fraction": {
                    "AL": [1.0 - value for value in solute],
                    "ZR": solute,
                },
                "final_composition_mole_fraction": {
                    "AL": [1.0 - value for value in solute],
                    "ZR": solute,
                },
                "time_s": request["duration_s"],
                "solver_steps": 10,
                "numerical_verification": {
                    "absolute_mass_closure_error": {"AL": 0.0, "ZR": 0.0},
                    "mass_closure_tolerance": 1e-8,
                },
            }
        elif operation == "binary_precipitation_kwn":
            bins = request["population_balance"]["bins"]
            payload = {
                "final": {
                    "time_s": request["duration_s"],
                    "matrix_solute_mole_fraction": 0.003999999,
                    "precipitate_volume_fraction": 1e-9,
                    "average_equivalent_spherical_radius_m": 4e-10,
                    "precipitate_number_density_per_m3": 1e18,
                    "nucleation_rate_per_m3_s": 1e12,
                    "driving_force_J_per_m3": 2e8,
                    "reconstructed_bulk_solute_mole_fraction": request[
                        "initial_solute_mole_fraction"
                    ],
                },
                "final_particle_size_distribution": {
                    "equivalent_spherical_radius_m": [
                        1e-10 + index * 1e-10 for index in range(bins)
                    ],
                    "particle_number_density_per_bin_per_m3": [1e10] * bins,
                },
                "solver_steps": 100,
                "numerical_verification": {
                    "maximum_absolute_solute_mass_closure_error": 0.0,
                    "solute_mass_closure_tolerance": 1e-8,
                },
            }
        else:  # pragma: no cover - closed request helpers above
            raise AssertionError(operation)
        request_payload = _canonical(request)
        return {
            "schema_version": "ultra.materials.kinetics-result.v1",
            "operation": operation,
            "input_request_evidence": {
                "algorithm": "sha256",
                "canonicalization": ("UTF-8 JSON, sorted keys, compact separators, finite numbers"),
                "sha256": hashlib.sha256(request_payload).hexdigest(),
                "size_bytes": len(request_payload),
            },
            "database": {
                "artifact_id": request["database"]["artifact_id"],
                "sha256": request["database"]["sha256"],
                "size_bytes": request["database"]["size_bytes"],
            },
            "result": payload,
            "solver": {"name": "kawin", "versions": dict(QUALIFIED_VERSIONS)},
            "limits": {
                **request["limits"],
                **(
                    {"max_solver_steps": request["max_solver_steps"]}
                    if operation != "transport_coefficients"
                    else {}
                ),
            },
        }

    def execute(self, command: str) -> Any:
        self.commands.append(command)
        arguments = shlex.split(command)
        request_path = arguments[arguments.index("--request") + 1]
        request = json.loads(
            (self.workspace_dir / request_path.removeprefix("/workspace/")).read_text()
        )
        self.requests.append(request)
        if self.mutation == "structured_failure":
            return SimpleNamespace(
                output=json.dumps(
                    {
                        "schema_version": "ultra.materials.kinetics-error.v1",
                        "error": {
                            "code": "unsupported_kinetics_scope",
                            "message": "moving-interface coupling is not qualified",
                        },
                    }
                ),
                exit_code=3,
                truncated=False,
            )
        if self.mutation == "nonfinite":
            return SimpleNamespace(output='{"value":NaN}', exit_code=0, truncated=False)
        result = self._result(request)
        if self.mutation == "database_mismatch":
            result["database"]["sha256"] = "0" * 64
        elif self.mutation == "request_mismatch":
            result["input_request_evidence"]["sha256"] = "0" * 64
        elif self.mutation == "diffusion_shape":
            result["result"]["coordinates_m"].pop()
        elif self.mutation == "diffusion_mass_drift":
            cells = request["mesh_cells"]
            result["result"]["final_composition_mole_fraction"] = {
                "AL": [0.995] * cells,
                "ZR": [0.005] * cells,
            }
        elif self.mutation == "kwn_shape":
            result["result"]["final_particle_size_distribution"][
                "particle_number_density_per_bin_per_m3"
            ].pop()
        digest = hashlib.sha256(_canonical(result)).hexdigest()
        result["evidence"] = {
            "algorithm": "sha256",
            "canonicalization": "UTF-8 JSON, sorted keys, compact separators, finite numbers",
            "sha256": "0" * 64 if self.mutation == "tamper_digest" else digest,
        }
        return SimpleNamespace(
            output=_canonical(result).decode(),
            exit_code=0,
            truncated=False,
        )


def _context(
    tmp_path: Path,
    *,
    payload: bytes,
    selected: bool = True,
    database_format: str = "tdb",
    database_id: str = "owner-al-zr-v1",
    source: str = "Owner supplied Al-Zr assessment",
    license_id: str = "CC-BY-4.0",
    assessment_scope: str = "Al-Zr thermodynamics and kinetics",
) -> AgentRunContext:
    resource_id = "file_kinetics_1"
    return AgentRunContext(
        assistant_id="a",
        org_id="org",
        user_id="user",
        project_id="project",
        thread_id="thread",
        run_id="run",
        goal="Calculate tracer diffusivity with Kawin",
        selected_file_ids=(resource_id,) if selected else (),
        resource_descriptors=(
            {
                "type": "selected_resource",
                "binding_schema": "ultra.selected_resource.v1",
                "authority": "control_resource_catalog",
                "resource_id": resource_id,
                "file_id": resource_id,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "original_name": f"assessment.{database_format}",
                "database_format": database_format,
                "content_type": "application/x-thermocalc-tdb",
                "calphad_governance_scope": "read_only_usage",
                "metadata": {
                    "calphad": {
                        "database_id": database_id,
                        "source": source,
                        "license_id": license_id,
                        "assessment_scope": assessment_scope,
                        "reference_state": "SER pure elements",
                        "tdb_temperature_limits_K": [300.0, 2000.0],
                        "assessment_pressure_limits_Pa": [101325.0, 101325.0],
                    }
                },
            },
        ),
        workspace_root=str(tmp_path / "workspace"),
        artifact_root=str(tmp_path / "outputs"),
    )


def _runtime(context: AgentRunContext, tools: list[Any]) -> ToolRuntime[Any, AgentRunContext]:
    return ToolRuntime(
        state={},
        context=context,
        config={},
        stream_writer=lambda _value: None,
        tool_call_id="kinetics-tool-test",
        store=None,
        tools=tools,
    )


def _tool_map(backend: Any, uploads: Path) -> dict[str, Any]:
    tools = build_kinetics_tools(SimpleNamespace(), backend=backend, upload_roots=(uploads,))
    return {item.name: item for item in tools}


def test_typed_transport_executes_fixed_cli_and_persists_verified_evidence(
    tmp_path: Path,
) -> None:
    backend = _FakeKineticsBackend(tmp_path)

    response = execute_kinetics_request_typed(backend, _request())

    assert response["ok"] is True
    assert response["artifact"]["content_addressed"] is True
    assert (backend.outputs_dir / response["artifact"]["path"].removeprefix("/outputs/")).is_file()
    command = shlex.split(backend.commands[0])
    assert command[:5] == [
        "python3",
        "-I",
        "-m",
        "ultra_deepagents.kinetics_runtime.cli",
        "--request",
    ]
    assert "--workspace-root" in command
    assert "file_kinetics_1" not in backend.commands[0]


@pytest.mark.parametrize(
    ("typed_request", "mutation", "message"),
    [
        (_diffusion_request(), "diffusion_shape", "wrong length"),
        (_kwn_request(), "kwn_shape", "wrong length"),
        (_request(), "nonfinite", "returned NaN"),
        (_request(), "database_mismatch", "database provenance mismatch"),
        (_request(), "request_mismatch", "request identity mismatch"),
        (_diffusion_request(), "diffusion_mass_drift", "mass closure failed"),
    ],
)
def test_product_boundary_rejects_malformed_untrusted_runtime_results(
    tmp_path: Path,
    typed_request: dict[str, Any],
    mutation: str,
    message: str,
) -> None:
    backend = _FakeKineticsBackend(tmp_path)
    backend.mutation = mutation

    with pytest.raises(KineticsToolError, match=message):
        execute_kinetics_request_typed(backend, typed_request)


def test_typed_runtime_rejects_structured_failure_tamper_and_unbounded_backend(
    tmp_path: Path,
) -> None:
    backend = _FakeKineticsBackend(tmp_path)
    backend.mutation = "structured_failure"
    with pytest.raises(KineticsToolError, match="moving-interface coupling") as failure:
        execute_kinetics_request_typed(backend, _request())
    assert failure.value.code == "unsupported_kinetics_scope"

    backend.mutation = "tamper_digest"
    with pytest.raises(KineticsToolError, match="digest mismatch"):
        execute_kinetics_request_typed(backend, _request())

    backend.mutation = ""
    backend.config.network = "bridge"
    with pytest.raises(KineticsToolError, match="unbounded_kinetics_backend"):
        execute_kinetics_request_typed(backend, _request())

    backend.config.network = "none"
    backend.config.image = "ultra-materials-kinetics:latest"
    with pytest.raises(KineticsToolError, match="immutable_kinetics_image_required"):
        execute_kinetics_request_typed(backend, _request())


def test_diffusion_tool_constructs_closed_request_and_persists_result(tmp_path: Path) -> None:
    payload = b"ELEMENT VA VACUUM 0 0 0 !\n"
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "file_kinetics_1__assessment.tdb").write_bytes(payload)
    backend = _FakeKineticsBackend(tmp_path)
    context = _context(tmp_path, payload=payload)
    tools = _tool_map(backend, uploads)

    response = json.loads(
        tools["materials_run_diffusion_1d"].func(
            runtime=_runtime(context, list(tools.values())),
            resource_id="file_kinetics_1",
            components=["AL", "ZR"],
            phase="FCC_A1",
            temperature_K=723.15,
            duration_s=1e6,
            domain_m=[-5e-6, 5e-6],
            mesh_cells=8,
            initial_profile_coordinates_m=[-5e-6, -1e-12, 1e-12, 5e-6],
            initial_independent_composition_mole_fraction={"ZR": [0.002, 0.002, 0.006, 0.006]},
            initial_profile_source="synthetic qualification profile",
            application_kind="post_solidification_back_diffusion",
            length_scale_source="ten-micrometre synthetic domain",
        )
    )

    request = backend.requests[-1]
    assert response["ok"] is True
    assert request["operation"] == "single_phase_diffusion_1d"
    assert request["boundary_condition"] == {"kind": "zero_flux"}
    assert request["max_solver_steps"] == FIXED_MAX_SOLVER_STEPS
    assert request["application"] == {
        "kind": "post_solidification_back_diffusion",
        "length_scale_source": "ten-micrometre synthetic domain",
        "solidification_coupling": "post_solidification_only",
    }
    assert Path(
        backend.outputs_dir, response["artifact"]["path"].removeprefix("/outputs/")
    ).is_file()


def test_kwn_tool_constructs_closed_request_and_persists_result(tmp_path: Path) -> None:
    payload = b"ELEMENT VA VACUUM 0 0 0 !\n"
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "file_kinetics_1__assessment.tdb").write_bytes(payload)
    backend = _FakeKineticsBackend(tmp_path)
    context = _context(tmp_path, payload=payload)
    tools = _tool_map(backend, uploads)

    response = json.loads(
        tools["materials_run_binary_precipitation_kwn"].func(
            runtime=_runtime(context, list(tools.values())),
            resource_id="file_kinetics_1",
            components=["AL", "ZR"],
            matrix_phase="FCC_A1",
            precipitate_phase="AL3ZR",
            initial_solute_mole_fraction=0.004,
            temperature_K=723.15,
            temperature_source="synthetic qualification isotherm",
            duration_s=100.0,
            matrix_molar_volume_m3_per_mol=1e-5,
            matrix_atoms_per_unit_cell=4,
            bulk_nucleation_site_density_per_m3=1e30,
            grain_boundary_energy_J_per_m2=0.3,
            matrix_parameter_source="synthetic qualification parameters",
            precipitate_molar_volume_m3_per_mol=1e-5,
            precipitate_atoms_per_unit_cell=4,
            interfacial_energy_J_per_m2=0.1,
            constant_elastic_strain_energy_J_per_m3=0.0,
            precipitate_parameter_source="synthetic qualification parameters",
            nucleation_source="synthetic homogeneous nucleation assumption",
            minimum_radius_m=1e-10,
            maximum_radius_m=1e-8,
            population_balance_bins=25,
        )
    )

    request = backend.requests[-1]
    assert response["ok"] is True
    assert request["operation"] == "binary_precipitation_kwn"
    assert request["driving_force_method"] == "tangent"
    assert request["nucleation"] == {
        "site": "bulk",
        "source": "synthetic homogeneous nucleation assumption",
    }
    assert request["population_balance"] == {
        "min_radius_m": 1e-10,
        "max_radius_m": 1e-8,
        "bins": 25,
        "adaptive": False,
        "max_history_points": 128,
    }
    assert request["max_solver_steps"] == FIXED_MAX_SOLVER_STEPS
    assert Path(
        backend.outputs_dir, response["artifact"]["path"].removeprefix("/outputs/")
    ).is_file()


def test_database_staging_requires_selected_governed_tdb(tmp_path: Path) -> None:
    payload = b"ELEMENT VA VACUUM 0 0 0 !\n"
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "file_kinetics_1__assessment.tdb").write_bytes(payload)
    backend = _FakeKineticsBackend(tmp_path)
    selected = _context(tmp_path, payload=payload)

    database = _database_request(
        SimpleNamespace(),
        selected,
        backend,
        upload_roots=(uploads,),
        resource_id="file_kinetics_1",
    )

    assert database["artifact_id"] == "file_kinetics_1"
    assert database["sha256"] == hashlib.sha256(payload).hexdigest()
    assert database["path"].startswith("/workspace/.ultra/calphad/staged/")
    assert database["path"].endswith(".tdb")

    unselected = _context(tmp_path, payload=payload, selected=False)
    with pytest.raises(KineticsToolError, match="explicitly selected") as selected_error:
        _database_request(
            SimpleNamespace(),
            unselected,
            backend,
            upload_roots=(uploads,),
            resource_id="file_kinetics_1",
        )
    assert selected_error.value.code == "selected_resource_required"

    dat_context = _context(tmp_path, payload=payload, database_format="dat")
    with pytest.raises(KineticsToolError, match="selected, governed") as format_error:
        _database_request(
            SimpleNamespace(),
            dat_context,
            backend,
            upload_roots=(uploads,),
            resource_id="file_kinetics_1",
        )
    assert format_error.value.code == "kinetics_tdb_required"


def test_real_pinned_image_runs_all_three_selected_resource_tools(tmp_path: Path) -> None:
    """Opt-in acceptance through DockerSandboxBackend, not the in-process runner.

    CI or a local qualification run sets the exact freshly built image ID. The
    fixture is copied out of that same pinned Kawin distribution and is marked
    test-only in the selected-resource provenance.
    """

    image = os.getenv("ULTRA_MATERIALS_KINETICS_ACCEPTANCE_IMAGE_ID", "").strip()
    if not image:
        pytest.skip("set ULTRA_MATERIALS_KINETICS_ACCEPTANCE_IMAGE_ID to run Docker acceptance")
    if shutil.which("docker") is None:
        pytest.skip("Docker is unavailable")
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            image,
            "python",
            "-I",
            "-c",
            "from kawin.tests.databases import ALZR_TDB; print(ALZR_TDB, end='')",
        ],
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    payload = completed.stdout
    assert b"FCC_A1" in payload and b"AL3ZR" in payload

    uploads = tmp_path / "uploads"
    workspace = tmp_path / "workspace"
    outputs = tmp_path / "outputs"
    uploads.mkdir()
    workspace.mkdir()
    outputs.mkdir()
    (uploads / "file_kinetics_1__qualification.tdb").write_bytes(payload)
    backend = DockerSandboxBackend(
        workspace_dir=workspace,
        outputs_dir=outputs,
        config=DockerSandboxConfig(
            image=image,
            network="none",
            cpus=2.0,
            memory="8g",
            pids_limit=256,
            no_new_privileges=True,
            gpus="",
            timeout_seconds=30,
            output_limit_bytes=FIXED_MAX_RESULT_BYTES + 128 * 1024,
            worker_id="kinetics-acceptance",
            run_id="kinetics-acceptance",
        ),
    )
    context = _context(
        tmp_path,
        payload=payload,
        database_id="kawin-0.5.0-package-alzr-qualification",
        source="kawin==0.5.0 kawin.tests.databases.ALZR_TDB package fixture",
        license_id="MIT (Kawin package fixture - qualification use only)",
        assessment_scope="test-only Al-Zr solver and product-boundary qualification",
    )
    tools = _tool_map(backend, uploads)
    runtime = _runtime(context, list(tools.values()))

    transport = json.loads(
        tools["materials_transport_coefficients"].func(
            runtime=runtime,
            resource_id="file_kinetics_1",
            components=["AL", "ZR"],
            phase="FCC_A1",
            independent_composition_mole_fraction={"ZR": 0.004},
            temperature_K=723.15,
        )
    )
    diffusion = json.loads(
        tools["materials_run_diffusion_1d"].func(
            runtime=runtime,
            resource_id="file_kinetics_1",
            components=["AL", "ZR"],
            phase="FCC_A1",
            temperature_K=723.15,
            duration_s=1e6,
            domain_m=[-5e-6, 5e-6],
            mesh_cells=16,
            initial_profile_coordinates_m=[-5e-6, -1e-12, 1e-12, 5e-6],
            initial_independent_composition_mole_fraction={"ZR": [0.002, 0.002, 0.006, 0.006]},
            initial_profile_source="synthetic Fick benchmark profile",
            application_kind="post_solidification_back_diffusion",
            length_scale_source="synthetic ten-micrometre qualification domain",
        )
    )
    precipitation = json.loads(
        tools["materials_run_binary_precipitation_kwn"].func(
            runtime=runtime,
            resource_id="file_kinetics_1",
            components=["AL", "ZR"],
            matrix_phase="FCC_A1",
            precipitate_phase="AL3ZR",
            initial_solute_mole_fraction=0.004,
            temperature_K=723.15,
            temperature_source="synthetic qualification isotherm",
            duration_s=100.0,
            matrix_molar_volume_m3_per_mol=1e-5,
            matrix_atoms_per_unit_cell=4,
            bulk_nucleation_site_density_per_m3=1e30,
            grain_boundary_energy_J_per_m2=0.3,
            matrix_parameter_source="synthetic qualification parameters",
            precipitate_molar_volume_m3_per_mol=1e-5,
            precipitate_atoms_per_unit_cell=4,
            interfacial_energy_J_per_m2=0.1,
            constant_elastic_strain_energy_J_per_m3=0.0,
            precipitate_parameter_source="synthetic qualification parameters",
            nucleation_source="synthetic homogeneous nucleation assumption",
            minimum_radius_m=1e-10,
            maximum_radius_m=1e-8,
            population_balance_bins=50,
        )
    )

    assert [transport["ok"], diffusion["ok"], precipitation["ok"]] == [
        True,
        True,
        True,
    ], (transport, diffusion, precipitation)
    assert transport["result"]["result"]["tracer_diffusivity_m2_per_s"]["ZR"] > 0
    assert (
        max(
            diffusion["result"]["result"]["numerical_verification"][
                "absolute_mass_closure_error"
            ].values()
        )
        <= 1e-8
    )
    assert math.isclose(
        precipitation["result"]["result"]["final"]["reconstructed_bulk_solute_mole_fraction"],
        0.004,
        rel_tol=0.0,
        abs_tol=1e-8,
    )
    for response in (transport, diffusion, precipitation):
        artifact = response["artifact"]
        assert (outputs / artifact["path"].removeprefix("/outputs/")).is_file()
