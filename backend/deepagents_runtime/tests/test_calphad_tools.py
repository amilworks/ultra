"""Typed CALPHAD tool, sandbox CLI, and registration regressions."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import ultra_deepagents.materials.calphad_cli as cli
from ultra_deepagents.code_execution.docker import DockerSandboxBackend, DockerSandboxConfig
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.materials.calphad_cli import (
    FAILURE_EVIDENCE_SCHEMA_VERSION,
    FIXED_MAX_RESULT_BYTES,
    FIXED_SCHEIL_MAX_STEPS,
    FIXED_WALL_TIME_SECONDS,
    REQUEST_SCHEMA_VERSION,
    TOOL_EVIDENCE_SCHEMA_VERSION,
    TypedCalphadError,
)
from ultra_deepagents.materials.calphad_tools import (
    CalphadToolError,
    _expected_execution_contract,
    _expected_validation_persistence,
    _governed_calphad_result,
    _persist_calphad_catalog_validation,
    _verified_artifact,
    inspect_calphad_database_typed,
    run_calphad_equilibrium_typed,
    run_calphad_scheil_typed,
)
from ultra_deepagents.materials.processing_kinetics import SCHEIL_ASSUMPTIONS


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _settings(upload_root: Path) -> RuntimeSettings:
    return RuntimeSettings(
        openai_base_url="http://127.0.0.1:9/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        control_worker_token="worker-secret",
        rarespot_upload_roots=(str(upload_root),),
        sandbox_cpus=2,
        sandbox_memory="4g",
        sandbox_pids_limit=256,
    )


def _selected_context(
    tmp_path: Path,
    *,
    resource_id: str,
    payload: bytes,
    complete_provenance: bool = True,
    governance_scope: str = "owner_validation",
    goal: str = "Inspect this CALPHAD TDB and calculate equilibrium",
    original_name: str = "assessment.tdb",
    content_type: str = "application/x-thermocalc-tdb",
    database_format: str | None = None,
) -> AgentRunContext:
    calphad: dict[str, Any] = {
        "database_id": "owner-al-ni-v1",
        "source": "Owner supplied assessment DOI 10.0000/example",
        "license_id": "CC-BY-4.0",
        "assessment_scope": "Assessed Al-Ni equilibrium from 300 to 2000 K",
        "reference_state": "SER pure elements at 298.15 K and 1 bar",
        "tdb_temperature_limits_K": [300.0, 2000.0],
        "assessment_pressure_limits_Pa": [101325.0, 101325.0],
    }
    if not complete_provenance:
        calphad.pop("reference_state")
    return AgentRunContext(
        assistant_id="a",
        org_id="org",
        user_id="user",
        project_id="project",
        thread_id="thread",
        run_id="run",
        goal=goal,
        selected_file_ids=(resource_id,),
        resource_descriptors=(
            {
                "type": "selected_resource",
                "binding_schema": "ultra.selected_resource.v1",
                "authority": "control_resource_catalog",
                "resource_id": resource_id,
                "file_id": resource_id,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "original_name": original_name,
                "database_format": database_format
                if database_format is not None
                else Path(original_name).suffix.casefold().removeprefix("."),
                "content_type": content_type,
                "resource_kind": "document",
                "source_type": "upload",
                "calphad_governance_scope": governance_scope,
                "metadata": {"calphad": calphad},
            },
        ),
        workspace_root=str(tmp_path / "workspace"),
        artifact_root=str(tmp_path / "outputs"),
        run_lease_worker_id="calphad-worker-1",
        run_lease_token="calphad-lease-secret",
    )


def _binding_identity(database: dict[str, Any]) -> dict[str, Any]:
    if database["kind"] == "embedded":
        return {"kind": "embedded", "database_id": database["database_id"]}
    return {
        key: database[key]
        for key in (
            "kind",
            "database_id",
            "resource_id",
            "database_format",
            "sha256",
            "size_bytes",
            "source",
            "license_id",
            "assessment_scope",
            "reference_state",
            "temperature_limits_K",
            "assessment_pressure_limits_Pa",
            "binding_schema",
            "binding_authority",
            "declaration_authority",
        )
    }


class _FakeSandbox:
    """Emulates only the fixed CLI result/artifact contract for host-tool tests."""

    def __init__(self, root: Path) -> None:
        self.workspace_dir = root / "workspace"
        self.outputs_dir = root / "outputs"
        self.config = SimpleNamespace(image="sha256:" + "f" * 64)
        self.workspace_dir.mkdir(parents=True)
        self.outputs_dir.mkdir(parents=True)
        self.requests: list[dict[str, Any]] = []
        self.commands: list[str] = []
        self.mode = "success"
        self.inspection_components = ["AL", "NI", "VA"]
        self.inspection_phases = ["FCC_A1", "LIQUID"]

    def execute(self, command: str) -> Any:
        self.commands.append(command)
        if self.mode == "timeout":
            return SimpleNamespace(output="", exit_code=124, truncated=False)
        sandbox_path = command.rsplit(" ", 1)[-1]
        host_path = self.workspace_dir / sandbox_path.removeprefix("/workspace/")
        request = json.loads(host_path.read_text())
        self.requests.append(request)
        database = request["database"]
        database_format = database.get("database_format", "tdb")
        database_path = database.get("path", f"/opt/ultra-calphad/reference.{database_format}")
        if request["operation"] == "inspect":
            runtime_result = {
                "schema_version": "1",
                "path": database_path,
                "format": database_format,
                "sha256": database.get("sha256", "a" * 64),
                "size_bytes": database.get("size_bytes", 123),
                "artifact_id": database.get("resource_id"),
                "available_components": self.inspection_components,
                "components": self.inspection_components,
                "available_phases": self.inspection_phases,
                "phases": self.inspection_phases,
                "parameter_count": 4,
                "pycalphad_version": "0.11.2",
                "registry_manifest": (
                    {"database_id": database["database_id"]}
                    if database["kind"] == "embedded"
                    else None
                ),
                "assessment_temperature_limits_K": [300.0, 2000.0],
                "assessment_pressure_limits_Pa": [101325.0, 101325.0],
            }
        elif request["operation"] == "equilibrium":
            requested_components = request["selection"]["components"]
            physical_components = sorted(
                component for component in requested_components if component not in {"VA", "/-"}
            )
            runtime_result = {
                "schema_version": "ultra.calphad.equilibrium.v2",
                "pycalphad_version": "0.11.2",
                "database": {
                    "path": database_path,
                    "format": database_format,
                    "sha256": database.get("sha256", "a" * 64),
                    "size_bytes": database.get("size_bytes", 123),
                    "pycalphad_version": "0.11.2",
                    "assessment_temperature_limits_K": [300.0, 2000.0],
                    "assessment_pressure_limits_Pa": [101325.0, 101325.0],
                },
                "request": {
                    "components": requested_components,
                    "dependent_component": physical_components[0],
                    "conditions": {
                        "independent_compositions": {
                            component: {"values": values, "units": "mole_fraction"}
                            for component, values in request["conditions"][
                                "independent_compositions"
                            ].items()
                        }
                    },
                    "phase_selection": {
                        "scope": "all_database_phases",
                        "excluded_database_phases": [],
                        "global_equilibrium_claim_supported": True,
                    },
                },
                "result": {
                    "point_count": 1,
                    "points": [
                        {
                            "conditions": {
                                "T_K": 1000.0,
                                "P_Pa": 101325.0,
                                "N_mol": 1.0,
                                "composition_mole_fraction": {"AL": 0.5, "NI": 0.5},
                            },
                            "stable_phases": [{"name": "FCC_A1", "NP_phase_fraction": 1.0}],
                            "stable_phase_vertices": [
                                {
                                    "vertex_index": 0,
                                    "phase": "FCC_A1",
                                    "NP_phase_fraction": 1.0,
                                    "composition_mole_fraction": {"AL": 0.5, "NI": 0.5},
                                    "composition_sum": 1.0,
                                }
                            ],
                            "phase_fraction_sum": 1.0,
                            "bulk_composition_residual_by_component": {
                                "AL": 0.0,
                                "NI": 0.0,
                            },
                            "maximum_bulk_composition_residual": 0.0,
                            "GM_J_per_mol": -1000.0,
                            "chemical_potentials_J_per_mol": {
                                "AL": -1200.0,
                                "NI": -800.0,
                            },
                            "gibbs_from_chemical_potentials_J_per_mol": -1000.0,
                            "gibbs_euler_residual_J_per_mol": 0.0,
                        }
                    ],
                    "units": {"GM": "J/mol", "MU": "J/mol", "X": "mole_fraction"},
                },
            }
        else:
            requested_components = request["selection"]["components"]
            requested_phases = request["selection"]["phases"]
            independent = request["conditions"]["independent_composition_mole_fraction"]
            dependent = next(
                component
                for component in requested_components
                if component not in {"VA", "/-"} and component not in independent
            )
            bulk = dict(sorted({dependent: 1.0 - sum(independent.values()), **independent}.items()))
            runtime_result = {
                "schema_version": "ultra.materials.scheil-gulliver.v1",
                "method": "Scheil-Gulliver",
                "database": {
                    "path": database_path,
                    "format": database_format,
                    "sha256": database.get("sha256", "a" * 64),
                    "size_bytes": database.get("size_bytes", 123),
                    "artifact_id": database.get("resource_id"),
                    "source": database.get("source"),
                    "license_id": database.get("license_id"),
                    "assessment_scope": database.get("assessment_scope"),
                    "reference_state": database.get("reference_state"),
                    "pycalphad_version": "0.11.2",
                    "registry_manifest": (
                        {"database_id": database["database_id"]}
                        if database["kind"] == "embedded"
                        else None
                    ),
                    "assessment_temperature_limits_K": database.get(
                        "temperature_limits_K", [300.0, 2000.0]
                    ),
                    "assessment_pressure_limits_Pa": database.get(
                        "assessment_pressure_limits_Pa", [101325.0, 101325.0]
                    ),
                },
                "request": {
                    "components": requested_components,
                    "phases": requested_phases,
                    "independent_composition_mole_fraction": independent,
                    "bulk_composition_mole_fraction": bulk,
                    "dependent_component": dependent,
                    "start_temperature_K": request["conditions"]["start_temperature_K"],
                    "step_temperature_K": request["conditions"]["step_temperature_K"],
                    "pressure_Pa": 101325.0,
                    "total_amount_mol": 1.0,
                    "liquid_phase_name": "LIQUID",
                    "stop_liquid_fraction": request["conditions"]["stop_liquid_fraction"],
                },
                "result": {
                    "point_count": 3,
                    "temperatures_K": [1200.0, 1100.0, 1000.0],
                    "fraction_solid": [0.0, 0.5, 0.99995],
                    "fraction_liquid": [1.0, 0.5, 0.00005],
                    "solid_phase_increment_fraction": {"FCC_A1": [0.0, 0.5, 0.49995]},
                    "solid_phase_cumulative_fraction": {"FCC_A1": [0.0, 0.5, 0.99995]},
                    "phase_composition_mole_fraction": {
                        "FCC_A1": {
                            "AL": [None, 0.6, 0.4],
                            "NI": [None, 0.4, 0.6],
                        },
                        "LIQUID": {
                            "AL": [None, 0.4, 0.4],
                            "NI": [None, 0.6, 0.6],
                        },
                    },
                    "elemental_mass_balance": {
                        "basis": "one_mole_initial_bulk",
                        "formula": (
                            "bulk_x[c] = fraction_liquid[i] * liquid_x[c,i] + "
                            "sum_phase,sum_step<=i(solid_increment[phase,step] * "
                            "solid_x[phase,c,step])"
                        ),
                        "absolute_tolerance": 1e-6,
                        "maximum_absolute_component_error": 0.0,
                        "maximum_absolute_error_by_component": {"AL": 0.0, "NI": 0.0},
                        "final_reconstructed_bulk_composition_mole_fraction": {
                            "AL": 0.5,
                            "NI": 0.5,
                        },
                        "all_retained_points_closed": True,
                    },
                    "converged": True,
                    "qualified_terminal_point": "last_residual_liquid_point",
                    "discarded_upstream_terminal_fill_point": False,
                    "closure_tolerances": {
                        "phase_fraction_absolute": 1e-6,
                        "composition_absolute": 1e-6,
                        "elemental_mass_balance_absolute": 1e-6,
                    },
                },
                "assumptions": list(SCHEIL_ASSUMPTIONS),
                "warnings": [
                    "This path is not a back-diffusion, finite-rate diffusion, precipitation, or phase-field calculation.",
                    "A converged numerical path does not validate the thermodynamic assessment or extrapolation domain.",
                ],
                "solver": {
                    "name": "scheil",
                    "version": "0.3.0",
                    "pycalphad_version": "0.11.2",
                    "adaptive_constitution_sampling": True,
                    "replay_determinism_claimed": False,
                },
                "units": {
                    "temperature": "K",
                    "pressure": "Pa",
                    "amount": "mol",
                    "composition": "mole_fraction",
                    "phase_fraction": "fraction_of_one_mole_basis",
                },
                "limits": {
                    "max_steps": FIXED_SCHEIL_MAX_STEPS,
                    "wall_time_seconds": FIXED_WALL_TIME_SECONDS,
                    "max_result_bytes": FIXED_MAX_RESULT_BYTES,
                },
            }
            inner_payload = dict(runtime_result)
            runtime_result["evidence"] = {
                "sha256": hashlib.sha256(_canonical(inner_payload)).hexdigest(),
                "algorithm": "sha256",
                "canonicalization": ("UTF-8 JSON, sorted keys, compact separators, finite numbers"),
            }
        evidence = {
            "schema_version": TOOL_EVIDENCE_SCHEMA_VERSION,
            "operation": request["operation"],
            "database_binding": _binding_identity(database),
            "request": {
                key: value
                for key, value in request.items()
                if key not in {"database", "schema_version"}
            },
            "result": runtime_result,
            "execution_contract": _expected_execution_contract(request),
            "validation_persistence": _expected_validation_persistence(),
        }
        payload = _canonical(evidence)
        digest = hashlib.sha256(payload).hexdigest()
        operation_dir = {
            "inspect": "inspection",
            "equilibrium": "equilibrium",
            "scheil": "scheil",
        }[request["operation"]]
        target = self.outputs_dir / "calphad" / operation_dir / f"{digest}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        envelope = {
            "ok": True,
            "operation": request["operation"],
            "artifact": {
                "path": f"/outputs/calphad/{operation_dir}/{digest}.json",
                "sha256": digest,
                "size_bytes": len(payload),
            },
        }
        if (
            self.mode in {"scheil_missing_phase_component", "scheil_mass_closure_forgery"}
            and request["operation"] == "scheil"
        ):
            if self.mode == "scheil_missing_phase_component":
                evidence["result"]["result"]["phase_composition_mole_fraction"]["FCC_A1"].pop("NI")
            else:
                evidence["result"]["result"]["phase_composition_mole_fraction"]["FCC_A1"]["AL"][
                    1
                ] = 0.7
                evidence["result"]["result"]["phase_composition_mole_fraction"]["FCC_A1"]["NI"][
                    1
                ] = 0.3
            forged_runtime = evidence["result"]
            forged_runtime_payload = dict(forged_runtime)
            forged_runtime_payload.pop("evidence")
            forged_runtime["evidence"]["sha256"] = hashlib.sha256(
                _canonical(forged_runtime_payload)
            ).hexdigest()
            bad_payload = _canonical(evidence)
            bad_digest = hashlib.sha256(bad_payload).hexdigest()
            bad_target = self.outputs_dir / "calphad" / operation_dir / f"{bad_digest}.json"
            bad_target.write_bytes(bad_payload)
            envelope["artifact"] = {
                "path": f"/outputs/calphad/{operation_dir}/{bad_digest}.json",
                "sha256": bad_digest,
                "size_bytes": len(bad_payload),
            }
        if self.mode == "wrong_schema":
            evidence["schema_version"] = "attacker.schema"
            bad_payload = _canonical(evidence)
            bad_digest = hashlib.sha256(bad_payload).hexdigest()
            bad_target = self.outputs_dir / "calphad" / operation_dir / f"{bad_digest}.json"
            bad_target.write_bytes(bad_payload)
            envelope["artifact"] = {
                "path": f"/outputs/calphad/{operation_dir}/{bad_digest}.json",
                "sha256": bad_digest,
                "size_bytes": len(bad_payload),
            }
        if self.mode == "wrong_runtime_schema" and request["operation"] == "equilibrium":
            evidence["result"]["schema_version"] = "ultra.calphad.equilibrium.v1"
            bad_payload = _canonical(evidence)
            bad_digest = hashlib.sha256(bad_payload).hexdigest()
            bad_target = self.outputs_dir / "calphad" / operation_dir / f"{bad_digest}.json"
            bad_target.write_bytes(bad_payload)
            envelope["artifact"] = {
                "path": f"/outputs/calphad/{operation_dir}/{bad_digest}.json",
                "sha256": bad_digest,
                "size_bytes": len(bad_payload),
            }
        if self.mode == "wrong_manifest_format":
            if request["operation"] == "inspect":
                evidence["result"]["format"] = "dat" if database_format == "tdb" else "tdb"
            else:
                evidence["result"]["database"]["format"] = (
                    "dat" if database_format == "tdb" else "tdb"
                )
            bad_payload = _canonical(evidence)
            bad_digest = hashlib.sha256(bad_payload).hexdigest()
            bad_target = self.outputs_dir / "calphad" / operation_dir / f"{bad_digest}.json"
            bad_target.write_bytes(bad_payload)
            envelope["artifact"] = {
                "path": f"/outputs/calphad/{operation_dir}/{bad_digest}.json",
                "sha256": bad_digest,
                "size_bytes": len(bad_payload),
            }
        if (
            self.mode in {"missing_residual", "invalid_hidden_point"}
            and request["operation"] == "equilibrium"
        ):
            if self.mode == "missing_residual":
                evidence["result"]["result"]["points"][0].pop("gibbs_euler_residual_J_per_mol")
            else:
                valid_point = evidence["result"]["result"]["points"][0]
                evidence["result"]["result"]["points"] = [
                    valid_point,
                    *[dict(valid_point) for _ in range(15)],
                    {**valid_point, "maximum_bulk_composition_residual": 0.25},
                ]
                evidence["result"]["result"]["point_count"] = 17
            bad_payload = _canonical(evidence)
            bad_digest = hashlib.sha256(bad_payload).hexdigest()
            bad_target = self.outputs_dir / "calphad" / operation_dir / f"{bad_digest}.json"
            bad_target.write_bytes(bad_payload)
            envelope["artifact"] = {
                "path": f"/outputs/calphad/{operation_dir}/{bad_digest}.json",
                "sha256": bad_digest,
                "size_bytes": len(bad_payload),
            }
        if self.mode in {"wrong_binding", "wrong_request"}:
            if self.mode == "wrong_binding":
                evidence["database_binding"] = {
                    **evidence["database_binding"],
                    "resource_id": "attacker-resource",
                }
            else:
                evidence["request"] = {**evidence["request"], "operation": "equilibrium"}
            bad_payload = _canonical(evidence)
            bad_digest = hashlib.sha256(bad_payload).hexdigest()
            bad_target = self.outputs_dir / "calphad" / operation_dir / f"{bad_digest}.json"
            bad_target.write_bytes(bad_payload)
            envelope["artifact"] = {
                "path": f"/outputs/calphad/{operation_dir}/{bad_digest}.json",
                "sha256": bad_digest,
                "size_bytes": len(bad_payload),
            }
        if self.mode == "artifact_hash_mismatch":
            target.write_bytes(payload + b"tampered")
        return SimpleNamespace(
            output=cli.RESULT_MARKER + json.dumps(envelope),
            exit_code=0,
            truncated=False,
        )


def _prepare_selected(
    tmp_path: Path,
    *,
    complete: bool = True,
    governance_scope: str = "owner_validation",
    original_name: str = "assessment.tdb",
    source_name: str = "assessment.tdb",
    content_type: str = "application/x-thermocalc-tdb",
    database_format: str | None = None,
) -> tuple[Any, ...]:
    payload = b"ELEMENT VA VACUUM 0 0 0 !\n"
    resource_id = "file_calphad_1"
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / f"{resource_id}__{source_name}").write_bytes(payload)
    context = _selected_context(
        tmp_path,
        resource_id=resource_id,
        payload=payload,
        complete_provenance=complete,
        governance_scope=governance_scope,
        original_name=original_name,
        content_type=content_type,
        database_format=database_format,
    )
    backend = _FakeSandbox(tmp_path)
    return payload, resource_id, uploads, context, backend


def test_selected_resource_inspection_uses_fixed_command_and_content_addressed_evidence(
    tmp_path: Path,
) -> None:
    payload, resource_id, uploads, context, backend = _prepare_selected(tmp_path)
    result = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )
    assert result["ok"] is True
    assert result["inventory"]["components"] == ["AL", "NI", "VA"]
    assert result["inspection_artifact"]["content_addressed"] is True
    assert result["catalog_validation_status"] == "pending"
    request = backend.requests[0]
    assert request["database"]["resource_id"] == resource_id
    assert request["database"]["database_format"] == "tdb"
    assert request["database"]["path"].endswith(".tdb")
    assert request["database"]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert request["database"]["source"] == "Owner supplied assessment DOI 10.0000/example"
    assert request["database"]["assessment_pressure_limits_Pa"] == [101325.0, 101325.0]
    assert request["database"]["declaration_authority"] == "resource_owner"
    assert result["database"]["assessment_pressure_limits_Pa"] == [101325.0, 101325.0]
    assert result["database"]["database_format"] == "tdb"
    assert backend.commands[0].startswith("python3 -I -c ")
    assert "/opt/ultra-runtime" in backend.commands[0]
    assert "python3 -m" not in backend.commands[0]
    assert resource_id not in backend.commands[0]


def test_selected_dat_resource_preserves_descriptor_format_through_evidence(
    tmp_path: Path,
) -> None:
    payload, resource_id, uploads, context, backend = _prepare_selected(
        tmp_path,
        original_name="assessment.dat",
        source_name="assessment.dat",
        content_type="application/octet-stream",
    )
    result = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )

    assert result["ok"] is True
    request = backend.requests[0]
    assert request["database"]["database_format"] == "dat"
    assert request["database"]["path"] == (
        f"/workspace/.ultra/calphad/staged/{hashlib.sha256(payload).hexdigest()}.dat"
    )
    artifact_sha = result["inspection_artifact"]["sha256"]
    evidence = json.loads(
        (backend.outputs_dir / "calphad" / "inspection" / f"{artifact_sha}.json").read_text()
    )
    assert evidence["database_binding"]["database_format"] == "dat"
    assert evidence["result"]["format"] == "dat"


def test_selected_descriptor_rejects_db_even_with_tdb_mime_type(tmp_path: Path) -> None:
    _, resource_id, uploads, context, backend = _prepare_selected(
        tmp_path,
        original_name="assessment.db",
        source_name="assessment.db",
        content_type="application/x-thermocalc-tdb",
        database_format="tdb",
    )

    result = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )

    assert result["ok"] is False
    assert result["error"] == "unsupported_calphad_resource_format"
    assert backend.requests == []


def test_selected_descriptor_database_format_must_match_original_name(tmp_path: Path) -> None:
    _, resource_id, uploads, context, backend = _prepare_selected(
        tmp_path,
        original_name="assessment.dat",
        source_name="assessment.dat",
        database_format="tdb",
    )

    result = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )

    assert result["ok"] is False
    assert result["error"] == "calphad_descriptor_format_mismatch"
    assert backend.requests == []


def test_selected_descriptor_requires_server_database_format(tmp_path: Path) -> None:
    _, resource_id, uploads, context, backend = _prepare_selected(
        tmp_path,
        database_format="",
    )

    result = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )

    assert result["ok"] is False
    assert result["error"] == "server_catalog_binding_required"
    assert backend.requests == []


def test_selected_descriptor_and_staged_source_suffix_must_match(tmp_path: Path) -> None:
    _, resource_id, uploads, context, backend = _prepare_selected(
        tmp_path,
        original_name="assessment.tdb",
        source_name="assessment.dat",
    )

    result = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )

    assert result["ok"] is False
    assert result["error"] == "calphad_resource_format_mismatch"
    assert backend.requests == []


def test_catalog_validation_callback_is_run_anchored_and_response_verified(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import httpx

    database_bytes, resource_id, uploads, context, _ = _prepare_selected(tmp_path)
    database_sha256 = hashlib.sha256(database_bytes).hexdigest()
    evidence_bytes = _canonical(
        {
            "schema": "callback-test",
            "database_binding": {
                "resource_id": resource_id,
                "database_format": "tdb",
                "sha256": database_sha256,
                "size_bytes": len(database_bytes),
            },
        }
    )
    digest = hashlib.sha256(evidence_bytes).hexdigest()
    artifact = {
        "path": f"/outputs/calphad/inspection/{digest}.json",
        "sha256": digest,
        "size_bytes": len(evidence_bytes),
    }
    captured: dict[str, Any] = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "revision": {
                    "revision_id": "calphad_revision_1",
                    "resource_id": resource_id,
                    "sha256": database_sha256,
                    "size_bytes": len(database_bytes),
                    "database_format": "tdb",
                },
                "validation": {
                    "validation_id": "calphad_validation_1",
                    "resource_id": resource_id,
                    "database_sha256": database_sha256,
                    "database_size_bytes": len(database_bytes),
                    "database_format": "tdb",
                    "status": "input_validated",
                    "operation": "inspect",
                    "evidence_path": artifact["path"],
                    "evidence_sha256": artifact["sha256"],
                    "evidence_size_bytes": artifact["size_bytes"],
                    "runtime_image_id": "sha256:" + "f" * 64,
                    "pycalphad_version": "0.11.2",
                    "run_id": context.run_id,
                    "evidence_retention": "retained",
                    "promotable": True,
                    "created_by_authority": "trusted_worker",
                },
            }

    class Client:
        def __init__(self, *, timeout: float) -> None:
            captured["timeout"] = timeout

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str]) -> Response:
            captured.update(url=url, json=json, headers=headers)
            return Response()

    monkeypatch.setattr(httpx, "Client", Client)
    ledger = _persist_calphad_catalog_validation(
        _settings(uploads),
        context,
        resource_id=resource_id,
        operation="inspect",
        artifact=artifact,
        evidence_bytes=evidence_bytes,
        runtime_image_id="sha256:" + "f" * 64,
        pycalphad_version="0.11.2",
    )
    assert captured["url"] == (
        f"http://control.test/v2/runs/{context.run_id}/resources/{resource_id}/calphad/validations"
    )
    assert captured["headers"] == {
        "X-Ultra-Worker-Token": "worker-secret",
        "X-Ultra-Run-Id": context.run_id,
        "X-Ultra-Worker-Id": "calphad-worker-1",
        "X-Ultra-Run-Lease-Token": "calphad-lease-secret",
    }
    callback_payload = captured["json"]
    encoded_evidence = callback_payload.pop("evidence_gzip_base64")
    assert gzip.decompress(base64.b64decode(encoded_evidence, validate=True)) == evidence_bytes
    assert callback_payload == {
        "status": "input_validated",
        "operation": "inspect",
        "evidence_path": artifact["path"],
        "evidence_sha256": artifact["sha256"],
        "evidence_size_bytes": len(evidence_bytes),
        "runtime_image_id": "sha256:" + "f" * 64,
        "pycalphad_version": "0.11.2",
    }
    assert ledger == {
        "mode": "server_managed_append_only_ledger",
        "persisted": True,
        "revision_id": "calphad_revision_1",
        "validation_id": "calphad_validation_1",
        "validation_status": "input_validated",
        "created_by_authority": "trusted_worker",
    }


def test_catalog_validation_callback_persists_failure_tuple_as_nonpromotable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import httpx

    database_bytes, resource_id, uploads, context, _ = _prepare_selected(tmp_path)
    database_sha256 = hashlib.sha256(database_bytes).hexdigest()
    outcome = {
        "status": "timeout",
        "failure_domain": "platform",
        "failure_stage": "sandbox_runtime",
        "failure_code": "calphad_sandbox_timeout",
        "exit_code": 124,
        "solver_started": False,
    }
    evidence_bytes = _canonical(
        {
            "schema_version": FAILURE_EVIDENCE_SCHEMA_VERSION,
            "operation": "inspect",
            "database_binding": {
                "resource_id": resource_id,
                "database_format": "tdb",
                "sha256": database_sha256,
                "size_bytes": len(database_bytes),
            },
            "request": {},
            "outcome": outcome,
            "execution_contract": {},
            "validation_persistence": {},
        }
    )
    digest = hashlib.sha256(evidence_bytes).hexdigest()
    artifact = {
        "path": f"/outputs/calphad/inspection/{digest}.json",
        "sha256": digest,
        "size_bytes": len(evidence_bytes),
    }
    captured: dict[str, Any] = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "revision": {
                    "revision_id": "calphad_revision_1",
                    "resource_id": resource_id,
                    "sha256": database_sha256,
                    "size_bytes": len(database_bytes),
                    "database_format": "tdb",
                },
                "validation": {
                    "validation_id": "calphad_validation_failure_1",
                    "resource_id": resource_id,
                    "database_sha256": database_sha256,
                    "database_size_bytes": len(database_bytes),
                    "database_format": "tdb",
                    "status": "timeout",
                    "operation": "inspect",
                    "failure_domain": "platform",
                    "failure_stage": "sandbox_runtime",
                    "failure_code": "calphad_sandbox_timeout",
                    "evidence_path": artifact["path"],
                    "evidence_sha256": artifact["sha256"],
                    "evidence_size_bytes": artifact["size_bytes"],
                    "runtime_image_id": "sha256:" + "f" * 64,
                    "pycalphad_version": "0.11.2",
                    "run_id": context.run_id,
                    "evidence_retention": "retained",
                    "promotable": False,
                    "created_by_authority": "trusted_worker",
                },
            }

    class Client:
        def __init__(self, *, timeout: float) -> None:
            assert timeout > 0

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def post(self, _url: str, *, json: dict[str, Any], headers: dict[str, str]) -> Response:
            captured.update(json=json, headers=headers)
            return Response()

    monkeypatch.setattr(httpx, "Client", Client)
    ledger = _persist_calphad_catalog_validation(
        _settings(uploads),
        context,
        resource_id=resource_id,
        operation="inspect",
        artifact=artifact,
        evidence_bytes=evidence_bytes,
        runtime_image_id="sha256:" + "f" * 64,
        pycalphad_version="0.11.2",
    )

    callback_payload = dict(captured["json"])
    encoded_evidence = callback_payload.pop("evidence_gzip_base64")
    assert gzip.decompress(base64.b64decode(encoded_evidence, validate=True)) == evidence_bytes
    assert callback_payload["status"] == "timeout"
    assert callback_payload["failure_domain"] == "platform"
    assert callback_payload["failure_stage"] == "sandbox_runtime"
    assert callback_payload["failure_code"] == "calphad_sandbox_timeout"
    assert ledger["validation_status"] == "timeout"


@pytest.mark.parametrize("lineage_matches", [True, False])
def test_equilibrium_catalog_callback_requires_exact_inspection_lineage_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    lineage_matches: bool,
) -> None:
    import httpx

    database_bytes, resource_id, uploads, context, _ = _prepare_selected(tmp_path)
    database_sha256 = hashlib.sha256(database_bytes).hexdigest()
    inspection_sha256 = "5" * 64
    evidence_bytes = _canonical(
        {
            "schema_version": TOOL_EVIDENCE_SCHEMA_VERSION,
            "operation": "equilibrium",
            "database_binding": {
                "resource_id": resource_id,
                "database_format": "tdb",
                "sha256": database_sha256,
                "size_bytes": len(database_bytes),
            },
            "request": {"inspection_artifact_sha256": inspection_sha256},
        }
    )
    digest = hashlib.sha256(evidence_bytes).hexdigest()
    artifact = {
        "path": f"/outputs/calphad/equilibrium/{digest}.json",
        "sha256": digest,
        "size_bytes": len(evidence_bytes),
    }

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "revision": {
                    "revision_id": "calphad_revision_1",
                    "resource_id": resource_id,
                    "sha256": database_sha256,
                    "size_bytes": len(database_bytes),
                    "database_format": "tdb",
                },
                "validation": {
                    "validation_id": "calphad_validation_1",
                    "resource_id": resource_id,
                    "database_sha256": database_sha256,
                    "database_size_bytes": len(database_bytes),
                    "database_format": "tdb",
                    "status": "equilibrium_completed",
                    "operation": "equilibrium",
                    "evidence_path": artifact["path"],
                    "evidence_sha256": artifact["sha256"],
                    "evidence_size_bytes": artifact["size_bytes"],
                    "runtime_image_id": "sha256:" + "f" * 64,
                    "pycalphad_version": "0.11.2",
                    "run_id": context.run_id,
                    "inspection_evidence_sha256": (
                        inspection_sha256 if lineage_matches else "6" * 64
                    ),
                    "evidence_retention": "retained",
                    "promotable": True,
                    "created_by_authority": "trusted_worker",
                },
            }

    class Client:
        def __init__(self, *, timeout: float) -> None:
            assert timeout > 0

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def post(self, *_args: Any, **_kwargs: Any) -> Response:
            return Response()

    monkeypatch.setattr(httpx, "Client", Client)

    def invoke() -> dict[str, Any]:
        return _persist_calphad_catalog_validation(
            _settings(uploads),
            context,
            resource_id=resource_id,
            operation="equilibrium",
            artifact=artifact,
            evidence_bytes=evidence_bytes,
            runtime_image_id="sha256:" + "f" * 64,
            pycalphad_version="0.11.2",
        )

    if lineage_matches:
        assert invoke()["validation_status"] == "equilibrium_completed"
    else:
        with pytest.raises(CalphadToolError, match="calphad_governance_persistence_failed"):
            invoke()


def test_governance_persistence_failure_fails_closed_after_artifact_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import ultra_deepagents.materials.calphad_tools as tools_module

    _, resource_id, uploads, context, backend = _prepare_selected(tmp_path)
    evidence_bytes = _canonical({"schema": "governance-failure-test"})
    evidence_sha = hashlib.sha256(evidence_bytes).hexdigest()
    evidence_target = backend.outputs_dir / "calphad" / "inspection" / f"{evidence_sha}.json"
    evidence_target.parent.mkdir(parents=True, exist_ok=True)
    evidence_target.write_bytes(evidence_bytes)
    result = {
        "ok": True,
        "operation": "inspect",
        "database": {"pycalphad_version": "0.11.2"},
        "inspection_artifact": {
            "path": f"/outputs/calphad/inspection/{evidence_sha}.json",
            "sha256": evidence_sha,
            "size_bytes": len(evidence_bytes),
        },
        "catalog_validation_status": "pending",
        "scientific_status": "unverified",
    }

    def fail(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise CalphadToolError("calphad_governance_persistence_failed")

    monkeypatch.setattr(tools_module, "_persist_calphad_catalog_validation", fail)
    governed = _governed_calphad_result(
        _settings(uploads),
        context,
        backend,
        result,
        operation="inspect",
        resource_id=resource_id,
        embedded_database_id="",
    )
    assert governed["ok"] is False
    assert governed["error"] == "calphad_governance_persistence_failed"
    assert governed["artifact_created"] is True
    assert governed["catalog_validation_status"] == "persistence_failed"
    assert governed["catalog_ledger"] == {
        "mode": "server_managed_append_only_ledger",
        "persisted": False,
    }
    assert governed["inspection_artifact"] == result["inspection_artifact"]


def test_embedded_calphad_result_uses_release_registry_not_tenant_ledger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import ultra_deepagents.materials.calphad_tools as tools_module

    _, _, uploads, context, backend = _prepare_selected(tmp_path)

    def unexpected(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("embedded database must not create a tenant resource revision")

    monkeypatch.setattr(tools_module, "_persist_calphad_catalog_validation", unexpected)
    governed = _governed_calphad_result(
        _settings(uploads),
        context,
        backend,
        {"ok": True, "operation": "inspect"},
        operation="inspect",
        resource_id="",
        embedded_database_id="nist-al-co-w-wang-2017",
    )
    assert governed["catalog_ledger"] == {
        "mode": "embedded_release_registry",
        "persisted": False,
    }


def test_shared_calphad_resource_returns_nonpromoting_read_only_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import ultra_deepagents.materials.calphad_tools as tools_module

    _, resource_id, uploads, context, backend = _prepare_selected(
        tmp_path, governance_scope="read_only_usage"
    )
    result = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )
    assert result["ok"] is True

    def unexpected(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("shared read-only analysis must not mutate the owner's ledger")

    monkeypatch.setattr(tools_module, "_persist_calphad_catalog_validation", unexpected)
    governed = _governed_calphad_result(
        _settings(uploads),
        context,
        backend,
        result,
        operation="inspect",
        resource_id=resource_id,
        embedded_database_id="",
    )
    assert governed["ok"] is True
    assert governed["catalog_validation_status"] == "read_only_unpromoted"
    assert governed["catalog_ledger"] == {
        "mode": "shared_read_only_artifact",
        "persisted": False,
        "promotable": False,
    }
    assert governed["inspection_artifact"] == result["inspection_artifact"]


def test_resource_missing_complete_owner_provenance_never_executes(tmp_path: Path) -> None:
    _, resource_id, uploads, context, backend = _prepare_selected(tmp_path, complete=False)
    result = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )
    assert result["ok"] is False
    assert result["error"] == "complete_resource_provenance_required"
    assert backend.commands == []


def test_resource_missing_owner_pressure_scope_never_executes(tmp_path: Path) -> None:
    _, resource_id, uploads, context, backend = _prepare_selected(tmp_path)
    context.resource_descriptors[0]["metadata"]["calphad"].pop("assessment_pressure_limits_Pa")

    result = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )

    assert result["ok"] is False
    assert result["error"] == "complete_resource_provenance_required"
    assert backend.commands == []


def test_resource_catalog_byte_mismatch_never_executes(tmp_path: Path) -> None:
    _, resource_id, uploads, context, backend = _prepare_selected(tmp_path)
    (uploads / f"{resource_id}__assessment.tdb").write_bytes(b"tampered")
    result = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )
    assert result["ok"] is False
    assert result["error"] == "resource_catalog_hash_mismatch"
    assert backend.commands == []


def test_nonselected_readable_catalog_resource_is_rejected(tmp_path: Path) -> None:
    payload = b"ELEMENT VA VACUUM 0 0 0 !\n"
    resource_id = "catalog_calphad_1"
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / f"{resource_id}__catalog.tdb").write_bytes(payload)
    context = AgentRunContext(
        assistant_id="a",
        org_id="org",
        user_id="user",
        project_id="project",
        thread_id="thread",
        run_id="run",
        goal="inspect my catalog CALPHAD database",
        workspace_root=str(tmp_path / "workspace"),
        artifact_root=str(tmp_path / "outputs"),
    )
    backend = _FakeSandbox(tmp_path)
    result = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )
    assert result["ok"] is False
    assert result["error"] == "selected_resource_required"
    assert backend.requests == []
    assert backend.commands == []


def test_backend_timeout_and_nonfinite_input_fail_closed(tmp_path: Path) -> None:
    _, resource_id, uploads, context, backend = _prepare_selected(tmp_path)
    backend.mode = "timeout"
    timed_out = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )
    assert timed_out["ok"] is False
    assert timed_out["error"] == "calphad_sandbox_timeout"
    assert timed_out["status"] == "timeout"
    assert timed_out["failure_domain"] == "platform"
    assert timed_out["failure_stage"] == "sandbox_runtime"
    assert timed_out["solver_started"] is False
    assert timed_out["artifact_created"] is True
    failure_path = (
        backend.outputs_dir
        / "calphad"
        / "inspection"
        / f"{timed_out['inspection_artifact']['sha256']}.json"
    )
    failure_evidence = json.loads(failure_path.read_text())
    assert set(failure_evidence) == {
        "schema_version",
        "operation",
        "database_binding",
        "request",
        "outcome",
        "execution_contract",
        "validation_persistence",
    }
    assert failure_evidence["outcome"] == {
        "status": "timeout",
        "failure_domain": "platform",
        "failure_stage": "sandbox_runtime",
        "failure_code": "calphad_sandbox_timeout",
        "exit_code": 124,
        "solver_started": False,
    }
    retry = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )
    assert retry["inspection_artifact"]["sha256"] == timed_out["inspection_artifact"]["sha256"]
    assert len(list((backend.outputs_dir / "calphad" / "inspection").glob("*.json"))) == 1

    backend.mode = "success"
    nonfinite = run_calphad_equilibrium_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
        inspection_artifact_sha256="a" * 64,
        components=["AL", "NI", "VA"],
        phases=["FCC_A1"],
        temperatures_K=[float("nan")],
        pressures_Pa=[101325.0],
        independent_compositions={"AL": [0.5]},
    )
    assert nonfinite["ok"] is False
    assert nonfinite["error"] == "invalid_typed_input"
    assert len(backend.commands) == 2


def test_equilibrium_summary_exposes_v2_mu_and_phase_compositions(tmp_path: Path) -> None:
    _, resource_id, uploads, context, backend = _prepare_selected(tmp_path)
    backend.inspection_components = ["AL", "CO", "NI", "VA", "W"]
    inspection = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )
    result = run_calphad_equilibrium_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
        inspection_artifact_sha256=inspection["inspection_artifact"]["sha256"],
        components=["AL", "NI", "VA"],
        phases=["FCC_A1", "LIQUID"],
        temperatures_K=[1000.0],
        pressures_Pa=[101325.0],
        independent_compositions={"AL": [0.5]},
    )
    assert result["ok"] is True
    point = result["points"][0]
    assert point["chemical_potentials_J_per_mol"] == {"AL": -1200.0, "NI": -800.0}
    assert point["stable_phase_vertices"][0]["composition_mole_fraction"] == {
        "AL": 0.5,
        "NI": 0.5,
    }
    assert point["maximum_bulk_composition_residual"] == 0.0
    assert point["gibbs_euler_residual_J_per_mol"] == 0.0
    assert result["database_format"] == "tdb"
    assert result["dependent_component"] == "AL"
    assert result["independent_composition_axes"] == {"NI": [0.5]}
    assert result["assessment_pressure_limits_Pa"] == [101325.0, 101325.0]
    assert result["equilibrium_artifact"]["content_addressed"] is True
    assert backend.requests[1]["database"]["database_format"] == "tdb"
    evidence_sha = result["equilibrium_artifact"]["sha256"]
    evidence = json.loads(
        (backend.outputs_dir / "calphad" / "equilibrium" / f"{evidence_sha}.json").read_text()
    )
    assert evidence["database_binding"]["database_format"] == "tdb"

    canonical = run_calphad_equilibrium_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
        inspection_artifact_sha256=inspection["inspection_artifact"]["sha256"],
        components=["AL", "CO", "W"],
        phases=["FCC_A1", "LIQUID"],
        temperatures_K=[1173.0],
        pressures_Pa=[101325.0],
        independent_compositions={"AL": [0.675], "CO": [0.26]},
    )
    assert canonical["ok"] is True
    canonical_request = backend.requests[2]
    assert canonical_request["selection"]["components"] == ["AL", "CO", "VA", "W"]
    assert canonical_request["conditions"]["independent_compositions"] == {
        "CO": [0.26],
        "W": [0.065],
    }
    assert canonical["dependent_component"] == "AL"
    assert canonical["independent_composition_axes"] == {"CO": [0.26], "W": [0.065]}
    request_path = backend.commands[2].rsplit(" ", 1)[-1]
    assert Path(request_path).stem == hashlib.sha256(_canonical(canonical_request)).hexdigest()

    def equilibrium_with_inspection(inspection_sha256: str) -> dict[str, Any]:
        return run_calphad_equilibrium_typed(
            _settings(uploads),
            context,
            backend,
            upload_roots=(uploads,),
            resource_id=resource_id,
            inspection_artifact_sha256=inspection_sha256,
            components=["AL", "NI"],
            phases=["FCC_A1"],
            temperatures_K=[1000.0],
            pressures_Pa=[101325.0],
            independent_compositions={"AL": [0.5]},
        )

    command_count = len(backend.commands)
    missing = equilibrium_with_inspection("b" * 64)
    assert missing["ok"] is False
    assert missing["error"] == "inspection_artifact_unavailable"

    inspection_sha = inspection["inspection_artifact"]["sha256"]
    inspection_path = backend.outputs_dir / "calphad" / "inspection" / f"{inspection_sha}.json"
    inspection_evidence = json.loads(inspection_path.read_text())
    wrong_binding_evidence = json.loads(json.dumps(inspection_evidence))
    wrong_binding_evidence["database_binding"]["resource_id"] = "resource-forged"
    wrong_binding_payload = _canonical(wrong_binding_evidence)
    wrong_binding_sha = hashlib.sha256(wrong_binding_payload).hexdigest()
    (backend.outputs_dir / "calphad" / "inspection" / f"{wrong_binding_sha}.json").write_bytes(
        wrong_binding_payload
    )
    wrong_binding = equilibrium_with_inspection(wrong_binding_sha)
    assert wrong_binding["ok"] is False
    assert wrong_binding["error"] == "inspection_artifact_binding_mismatch"

    wrong_runtime_evidence = json.loads(json.dumps(inspection_evidence))
    wrong_runtime_evidence["request"]["runtime_image_id"] = "sha256:" + "e" * 64
    wrong_runtime_payload = _canonical(wrong_runtime_evidence)
    wrong_runtime_sha = hashlib.sha256(wrong_runtime_payload).hexdigest()
    (backend.outputs_dir / "calphad" / "inspection" / f"{wrong_runtime_sha}.json").write_bytes(
        wrong_runtime_payload
    )
    wrong_runtime = equilibrium_with_inspection(wrong_runtime_sha)
    assert wrong_runtime["ok"] is False
    assert wrong_runtime["error"] == "inspection_artifact_runtime_mismatch"
    assert len(backend.commands) == command_count

    no_va_backend = _FakeSandbox(tmp_path / "no-va-inventory")
    no_va_backend.inspection_components = ["AL", "NI"]
    no_va_inspection = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        no_va_backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )
    no_va_result = run_calphad_equilibrium_typed(
        _settings(uploads),
        context,
        no_va_backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
        inspection_artifact_sha256=no_va_inspection["inspection_artifact"]["sha256"],
        components=["AL", "NI"],
        phases=["FCC_A1"],
        temperatures_K=[1000.0],
        pressures_Pa=[101325.0],
        independent_compositions={"AL": [0.5]},
    )
    assert no_va_result["ok"] is True
    assert no_va_backend.requests[1]["selection"]["components"] == ["AL", "NI"]


def test_equilibrium_rejects_pressure_outside_owner_scope_before_execution(
    tmp_path: Path,
) -> None:
    _, resource_id, uploads, context, backend = _prepare_selected(tmp_path)

    result = run_calphad_equilibrium_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
        inspection_artifact_sha256="a" * 64,
        components=["AL", "NI", "VA"],
        phases=["FCC_A1"],
        temperatures_K=[1000.0],
        pressures_Pa=[101326.0],
        independent_compositions={"AL": [0.5]},
    )

    assert result["ok"] is False
    assert result["error"] == "invalid_typed_input"
    assert backend.commands == []


@pytest.mark.parametrize("mode", ["wrong_schema", "wrong_runtime_schema"])
def test_wrong_tool_or_runtime_evidence_schema_is_rejected(tmp_path: Path, mode: str) -> None:
    _, resource_id, uploads, context, backend = _prepare_selected(tmp_path)
    inspection = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )
    backend.mode = mode
    if mode == "wrong_schema":
        result = inspect_calphad_database_typed(
            _settings(uploads),
            context,
            backend,
            upload_roots=(uploads,),
            resource_id=resource_id,
        )
    else:
        result = run_calphad_equilibrium_typed(
            _settings(uploads),
            context,
            backend,
            upload_roots=(uploads,),
            resource_id=resource_id,
            inspection_artifact_sha256=inspection["inspection_artifact"]["sha256"],
            components=["AL", "NI", "VA"],
            phases=["FCC_A1"],
            temperatures_K=[1000.0],
            pressures_Pa=[101325.0],
            independent_compositions={"AL": [0.5]},
        )
    assert result["ok"] is False
    assert result["error"] == "invalid_artifact_evidence"


def test_result_manifest_format_mismatch_is_rejected_by_host_verifier(tmp_path: Path) -> None:
    _, resource_id, uploads, context, backend = _prepare_selected(tmp_path)
    backend.mode = "wrong_manifest_format"

    result = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )

    assert result["ok"] is False
    assert result["error"] == "invalid_artifact_evidence"


@pytest.mark.parametrize("mode", ["missing_residual", "invalid_hidden_point"])
def test_v2_schema_and_residuals_are_validated_across_full_artifact(
    tmp_path: Path,
    mode: str,
) -> None:
    _, resource_id, uploads, context, backend = _prepare_selected(tmp_path)
    inspection = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )
    backend.mode = mode
    result = run_calphad_equilibrium_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
        inspection_artifact_sha256=inspection["inspection_artifact"]["sha256"],
        components=["AL", "NI", "VA"],
        phases=["FCC_A1"],
        temperatures_K=[1000.0],
        pressures_Pa=[101325.0],
        independent_compositions={"AL": [0.5]},
    )
    assert result["ok"] is False
    assert result["error"] == "invalid_artifact_evidence"


@pytest.mark.parametrize("mode", ["wrong_binding", "wrong_request"])
def test_host_rejects_artifact_binding_or_request_forgery(tmp_path: Path, mode: str) -> None:
    _, resource_id, uploads, context, backend = _prepare_selected(tmp_path)
    backend.mode = mode
    result = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )
    assert result["ok"] is False
    assert result["error"] == "artifact_request_binding_mismatch"


def test_host_rejects_artifact_hash_mismatch(tmp_path: Path) -> None:
    _, resource_id, uploads, context, backend = _prepare_selected(tmp_path)
    backend.mode = "artifact_hash_mismatch"
    result = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )
    assert result["ok"] is False
    assert result["error"] == "artifact_hash_mismatch"


def test_content_addressed_request_symlink_collision_fails_closed(tmp_path: Path) -> None:
    _, resource_id, uploads, context, backend = _prepare_selected(tmp_path)
    first = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )
    assert first["ok"] is True
    request_path = backend.workspace_dir / backend.commands[0].rsplit(" ", 1)[-1].removeprefix(
        "/workspace/"
    )
    replacement = backend.workspace_dir / "attacker-request.json"
    replacement.write_bytes(request_path.read_bytes())
    request_path.unlink()
    request_path.symlink_to(replacement)
    second = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )
    assert second["ok"] is False
    assert second["artifact_created"] is False
    assert len(backend.commands) == 1


def test_isolated_cli_bootstrap_cannot_be_shadowed_from_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import ultra_deepagents.materials.calphad_tools as tools_module

    trusted = tmp_path / "trusted"
    workspace = tmp_path / "workspace"
    for root, marker in ((trusted, "TRUSTED"), (workspace, "FORGED")):
        package = root / "ultra_deepagents" / "materials"
        package.mkdir(parents=True)
        (root / "ultra_deepagents" / "__init__.py").write_text("")
        (package / "__init__.py").write_text("")
        (package / "calphad_cli.py").write_text(
            f"def main(argv=None):\n    print('{marker}')\n    return 0\n"
        )
    monkeypatch.setattr(tools_module, "_TRUSTED_RUNTIME_ROOT", trusted)
    command = tools_module._typed_cli_command(
        "/workspace/.ultra/calphad/requests/" + "a" * 64 + ".json"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(workspace)
    completed = subprocess.run(
        ["bash", "-lc", command],
        cwd=workspace,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "TRUSTED"
    assert "FORGED" not in completed.stdout


def _resource_database(
    path: Path,
    payload: bytes,
    *,
    resource_id: str = "resource-1",
    database_format: str | None = None,
) -> dict:
    return {
        "kind": "resource",
        "database_id": "owner-database",
        "path": str(path),
        "resource_id": resource_id,
        "database_format": database_format or path.suffix.casefold().removeprefix("."),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "source": "Owner assessment source",
        "license_id": "CC0-1.0",
        "assessment_scope": "Binary equilibrium assessment",
        "reference_state": "SER at 298.15 K",
        "temperature_limits_K": [300.0, 2000.0],
        "assessment_pressure_limits_Pa": [101325.0, 101325.0],
        "binding_schema": "ultra.selected_resource.v1",
        "binding_authority": "control_resource_catalog",
        "declaration_authority": "resource_owner",
    }


def _cli_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    staged = workspace / ".ultra" / "calphad" / "staged"
    outputs = tmp_path / "outputs" / "calphad"
    staged.mkdir(parents=True)
    outputs.mkdir(parents=True)
    monkeypatch.setattr(cli, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(cli, "STAGED_DATABASE_ROOTS", (staged,))
    monkeypatch.setattr(cli, "OUTPUT_ROOT", outputs)
    return staged, outputs


def test_cli_requires_fixed_or_bounded_resource_pressure_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    staged, _ = _cli_roots(monkeypatch, tmp_path)
    payload = b"synthetic bounded TDB"
    database_path = staged / "database.tdb"
    database_path.write_bytes(payload)
    database = _resource_database(database_path, payload)

    normalized = cli._validated_database(database)
    assert normalized["database_format"] == "tdb"
    assert normalized["assessment_pressure_limits_Pa"] == [101325.0, 101325.0]
    missing = dict(database)
    missing.pop("assessment_pressure_limits_Pa")
    with pytest.raises(TypedCalphadError, match="complete catalog binding"):
        cli._validated_database(missing)

    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "operation": "equilibrium",
        "runtime_image_id": "sha256:" + "f" * 64,
        "database": database,
        "selection": {"components": ["AL", "NI", "VA"], "phases": ["FCC_A1"]},
        "inspection_artifact_sha256": "a" * 64,
        "conditions": {
            "temperatures_K": [1000.0],
            "pressures_Pa": [101326.0],
            "independent_compositions": {"AL": [0.5]},
        },
    }
    with pytest.raises(TypedCalphadError, match="outside the resource assessment"):
        cli._validated_request(request)

    canonical_request = {
        **request,
        "selection": {
            "components": ["AL", "CO", "W", "VA"],
            "phases": ["FCC_A1"],
        },
        "conditions": {
            "temperatures_K": [1173.0],
            "pressures_Pa": [101325.0],
            "independent_compositions": {"AL": [0.675], "CO": [0.26]},
        },
    }
    normalized_request = cli._validated_request(canonical_request)
    assert normalized_request["conditions"]["independent_compositions"] == {
        "CO": [0.26],
        "W": [0.065],
    }

    coupled_grid = json.loads(json.dumps(canonical_request))
    coupled_grid["conditions"]["independent_compositions"] = {
        "AL": [0.5, 0.6],
        "CO": [0.1, 0.2],
    }
    with pytest.raises(TypedCalphadError, match="cannot be reframed"):
        cli._validated_request(coupled_grid)


def test_cli_requires_exact_resource_format_and_path_suffix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    staged, _ = _cli_roots(monkeypatch, tmp_path)
    payload = b"synthetic bounded TDB"
    database_path = staged / "database.tdb"
    database_path.write_bytes(payload)
    database = _resource_database(database_path, payload)

    missing = dict(database)
    missing.pop("database_format")
    with pytest.raises(TypedCalphadError, match="complete catalog binding"):
        cli._validated_database(missing)

    mismatched = {**database, "database_format": "dat"}
    with pytest.raises(TypedCalphadError, match="suffix does not match"):
        cli._validated_database(mismatched)

    unsupported = {**database, "database_format": "db"}
    with pytest.raises(TypedCalphadError, match="must be tdb or dat"):
        cli._validated_database(unsupported)


@pytest.mark.parametrize(
    ("result_format", "result_suffix", "match"),
    [
        ("dat", ".dat", "does not match the resource binding"),
        ("tdb", ".dat", "path suffix does not match"),
    ],
)
def test_cli_rejects_result_manifest_format_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    result_format: str,
    result_suffix: str,
    match: str,
) -> None:
    staged, _ = _cli_roots(monkeypatch, tmp_path)
    payload = b"synthetic bounded TDB"
    database_path = staged / "database.tdb"
    database_path.write_bytes(payload)
    database = _resource_database(database_path, payload)
    monkeypatch.setattr(
        cli,
        "inspect_calphad_input",
        lambda *a, **k: {
            "schema_version": "1",
            "path": str(staged / f"result{result_suffix}"),
            "format": result_format,
        },
    )

    result = cli.execute_request(
        {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "operation": "inspect",
            "runtime_image_id": "sha256:" + "f" * 64,
            "database": database,
            "selection": {"components": None, "phases": None},
        }
    )
    assert result["ok"] is False
    assert result["outcome"] == {
        "status": "failed",
        "failure_domain": "scientific",
        "failure_stage": "result_validation",
        "failure_code": "calphad_result_invalid",
        "exit_code": 1,
        "solver_started": False,
    }


@pytest.mark.parametrize(
    ("error_type", "expected_outcome"),
    [
        (
            cli.CalphadInputError,
            {
                "status": "failed",
                "failure_domain": "input",
                "failure_stage": "parse",
                "failure_code": "calphad_parse_failed",
                "exit_code": 2,
                "solver_started": False,
            },
        ),
        (
            cli.CalphadTimeoutError,
            {
                "status": "timeout",
                "failure_domain": "scientific",
                "failure_stage": "parse",
                "failure_code": "calphad_parse_timeout",
                "exit_code": 124,
                "solver_started": False,
            },
        ),
        (
            cli.CalphadUnsupportedError,
            {
                "status": "unsupported",
                "failure_domain": "input",
                "failure_stage": "parse",
                "failure_code": "calphad_parse_unsupported",
                "exit_code": 2,
                "solver_started": False,
            },
        ),
    ],
)
def test_cli_retains_exact_bounded_parse_failure_without_raw_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error_type: type[Exception],
    expected_outcome: dict[str, Any],
) -> None:
    staged, outputs = _cli_roots(monkeypatch, tmp_path)
    payload = b"synthetic bounded TDB"
    database_path = staged / "database.tdb"
    database_path.write_bytes(payload)
    database = _resource_database(database_path, payload)
    sensitive_diagnostic = (
        "credential=do-not-retain /private/path command='rm -rf /'\nTraceback: raw stderr"
    )

    def fail_parse(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise error_type(sensitive_diagnostic)

    monkeypatch.setattr(cli, "inspect_calphad_input", fail_parse)
    result = cli.execute_request(
        {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "operation": "inspect",
            "runtime_image_id": "sha256:" + "f" * 64,
            "database": database,
            "selection": {"components": None, "phases": None},
        }
    )

    assert result["ok"] is False
    assert result["outcome"] == expected_outcome
    artifact_path = outputs / "inspection" / f"{result['artifact']['sha256']}.json"
    evidence_bytes = artifact_path.read_bytes()
    evidence = json.loads(evidence_bytes)
    assert evidence["schema_version"] == FAILURE_EVIDENCE_SCHEMA_VERSION
    assert set(evidence) == {
        "schema_version",
        "operation",
        "database_binding",
        "request",
        "outcome",
        "execution_contract",
        "validation_persistence",
    }
    assert evidence["outcome"] == expected_outcome
    assert sensitive_diagnostic.encode() not in evidence_bytes
    for forbidden_key in ("message", "traceback", "command", "env", "stdout", "stderr"):
        assert forbidden_key not in evidence_bytes.decode()


def test_host_verifier_rejects_individually_valid_but_mismatched_failure_tuple(
    tmp_path: Path,
) -> None:
    _, resource_id, uploads, context, backend = _prepare_selected(tmp_path)
    successful = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )
    request = backend.requests[0]
    success_path = (
        backend.outputs_dir
        / "calphad"
        / "inspection"
        / f"{successful['inspection_artifact']['sha256']}.json"
    )
    success_evidence = json.loads(success_path.read_text())
    outcome = {
        "status": "timeout",
        "failure_domain": "input",
        "failure_stage": "parse",
        "failure_code": "calphad_parse_failed",
        "exit_code": 2,
        "solver_started": False,
    }
    forged = {
        "schema_version": FAILURE_EVIDENCE_SCHEMA_VERSION,
        "operation": "inspect",
        "database_binding": success_evidence["database_binding"],
        "request": success_evidence["request"],
        "outcome": outcome,
        "execution_contract": _expected_execution_contract(request),
        "validation_persistence": _expected_validation_persistence(),
    }
    payload = _canonical(forged)
    digest = hashlib.sha256(payload).hexdigest()
    target = backend.outputs_dir / "calphad" / "inspection" / f"{digest}.json"
    target.write_bytes(payload)
    envelope = {
        "ok": False,
        "operation": "inspect",
        "error": outcome["failure_code"],
        "status": outcome["status"],
        "outcome": outcome,
        "artifact": {
            "path": f"/outputs/calphad/inspection/{digest}.json",
            "sha256": digest,
            "size_bytes": len(payload),
        },
    }

    with pytest.raises(CalphadToolError, match="invalid_artifact_evidence"):
        _verified_artifact(backend, envelope, operation="inspect", request=request)


def test_equilibrium_unsupported_is_pre_solver_and_retained_with_solver_started_false(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    staged, outputs = _cli_roots(monkeypatch, tmp_path)
    payload = b"synthetic bounded TDB"
    database_path = staged / "database.tdb"
    database_path.write_bytes(payload)
    database = _resource_database(database_path, payload)
    inspection_result = {
        "schema_version": "1",
        "path": str(database_path),
        "format": "tdb",
        "sha256": database["sha256"],
        "size_bytes": database["size_bytes"],
        "available_components": ["AL", "NI", "VA"],
        "components": ["AL", "NI", "VA"],
        "available_phases": ["FCC_A1"],
        "phases": ["FCC_A1"],
        "pycalphad_version": "0.11.2",
    }
    monkeypatch.setattr(cli, "inspect_calphad_input", lambda *a, **k: inspection_result)
    inspection = cli.execute_request(
        {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "operation": "inspect",
            "runtime_image_id": "sha256:" + "f" * 64,
            "database": database,
            "selection": {"components": None, "phases": None},
        }
    )

    # CalphadUnsupportedError is reserved for database loading before the
    # numerical equilibrium entry point; therefore solver_started must be false.
    def unsupported_before_solver(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise cli.CalphadUnsupportedError("unsupported database construct")

    monkeypatch.setattr(cli, "run_calphad_equilibrium", unsupported_before_solver)
    result = cli.execute_request(
        {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "operation": "equilibrium",
            "runtime_image_id": "sha256:" + "f" * 64,
            "database": database,
            "selection": {"components": ["AL", "NI", "VA"], "phases": ["FCC_A1"]},
            "inspection_artifact_sha256": inspection["artifact"]["sha256"],
            "conditions": {
                "temperatures_K": [1000.0],
                "pressures_Pa": [101325.0],
                "independent_compositions": {"AL": [0.5]},
            },
        }
    )

    assert result["outcome"] == {
        "status": "unsupported",
        "failure_domain": "scientific",
        "failure_stage": "solver",
        "failure_code": "calphad_solver_unsupported",
        "exit_code": 2,
        "solver_started": False,
    }
    evidence_path = outputs / "equilibrium" / f"{result['artifact']['sha256']}.json"
    assert json.loads(evidence_path.read_text())["outcome"]["solver_started"] is False


def test_cli_inspection_chain_rejects_wrong_hash_binding_and_inventory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    staged, outputs = _cli_roots(monkeypatch, tmp_path)
    payload = b"synthetic bounded TDB"
    database_path = staged / "database.tdb"
    database_path.write_bytes(payload)
    database = _resource_database(database_path, payload)
    inspection_result = {
        "schema_version": "1",
        "path": str(database_path),
        "format": "tdb",
        "sha256": database["sha256"],
        "size_bytes": database["size_bytes"],
        "available_components": ["AL", "NI", "VA"],
        "components": ["AL", "NI", "VA"],
        "available_phases": ["FCC_A1"],
        "phases": ["FCC_A1"],
        "pycalphad_version": "0.11.2",
    }
    monkeypatch.setattr(cli, "inspect_calphad_input", lambda *a, **k: inspection_result)
    inspection = cli.execute_request(
        {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "operation": "inspect",
            "runtime_image_id": "sha256:" + "f" * 64,
            "database": database,
            "selection": {"components": None, "phases": None},
        }
    )
    inspection_sha = inspection["artifact"]["sha256"]
    solver_calls: list[bool] = []

    def unexpected_solver(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        solver_calls.append(True)
        raise AssertionError("inspection-lineage rejection must happen before solver entry")

    monkeypatch.setattr(cli, "run_calphad_equilibrium", unexpected_solver)

    base_equilibrium = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "operation": "equilibrium",
        "runtime_image_id": "sha256:" + "f" * 64,
        "database": database,
        "selection": {"components": ["AL", "NI", "VA"], "phases": ["FCC_A1"]},
        "inspection_artifact_sha256": inspection_sha,
        "conditions": {
            "temperatures_K": [1000.0],
            "pressures_Pa": [101325.0],
            "independent_compositions": {"AL": [0.5]},
        },
    }
    omitted_va = json.loads(json.dumps(base_equilibrium))
    omitted_va["selection"]["components"] = ["AL", "NI"]
    with pytest.raises(TypedCalphadError, match="must include VA"):
        cli.execute_request(omitted_va)

    wrong_hash = dict(base_equilibrium)
    wrong_hash["inspection_artifact_sha256"] = "0" * 64
    with pytest.raises(TypedCalphadError, match="inspection evidence"):
        cli.execute_request(wrong_hash)

    wrong_binding = dict(base_equilibrium)
    wrong_binding["database"] = {**database, "resource_id": "different-resource"}
    with pytest.raises(TypedCalphadError, match="different database revision"):
        cli.execute_request(wrong_binding)

    alternate_path = staged / "database.dat"
    alternate_path.write_bytes(payload)
    wrong_format_binding = dict(base_equilibrium)
    wrong_format_binding["database"] = _resource_database(
        alternate_path,
        payload,
        database_format="dat",
    )
    with pytest.raises(TypedCalphadError, match="different database revision"):
        cli.execute_request(wrong_format_binding)

    wrong_inventory = dict(base_equilibrium)
    wrong_inventory["selection"] = {
        "components": ["AL", "NI", "VA"],
        "phases": ["LIQUID"],
    }
    with pytest.raises(TypedCalphadError, match="absent from inspection inventory"):
        cli.execute_request(wrong_inventory)

    # These are trusted preflight rejections, not terminal runtime outcomes:
    # no solver entry and no retained equilibrium failure artifact/event input.
    assert solver_calls == []
    assert not (outputs / "equilibrium").exists()


def test_cli_rejects_resource_byte_mismatch_and_symlink_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    staged, _ = _cli_roots(monkeypatch, tmp_path)
    payload = b"synthetic bounded TDB"
    database_path = staged / "database.tdb"
    database_path.write_bytes(payload)
    database = _resource_database(database_path, payload)
    database_path.write_bytes(b"changed")
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "operation": "inspect",
        "runtime_image_id": "sha256:" + "f" * 64,
        "database": database,
        "selection": {"components": None, "phases": None},
    }
    with pytest.raises(TypedCalphadError, match="catalog SHA-256/size"):
        cli.execute_request(request)

    target = staged / "target.tdb"
    target.write_bytes(payload)
    symlink = staged / "link.tdb"
    symlink.symlink_to(target)
    symlink_request = dict(request)
    symlink_request["database"] = _resource_database(symlink, payload)
    with pytest.raises(TypedCalphadError, match="symbolic link"):
        cli.execute_request(symlink_request)


def test_scheil_typed_tool_retains_va_and_returns_mass_closed_bounded_summary(
    tmp_path: Path,
) -> None:
    _, resource_id, uploads, context, backend = _prepare_selected(tmp_path)
    inspection = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )

    result = run_calphad_scheil_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
        inspection_artifact_sha256=inspection["inspection_artifact"]["sha256"],
        components=["AL", "NI"],
        phases=["LIQUID", "FCC_A1"],
        independent_composition_mole_fraction={"NI": 0.5},
        start_temperature_K=1200.0,
        step_temperature_K=10.0,
        stop_liquid_fraction=1e-4,
    )

    assert result["ok"] is True
    assert result["operation"] == "scheil"
    assert result["method"] == "Scheil-Gulliver"
    assert result["model_scope"]["not_claimed"] == [
        "back_diffusion",
        "finite_rate_solid_diffusion",
        "precipitation",
        "phase_field",
    ]
    assert result["elemental_mass_balance"]["all_retained_points_closed"] is True
    assert result["elemental_mass_balance"]["maximum_absolute_component_error"] == 0.0
    assert result["limits"]["max_steps"] == FIXED_SCHEIL_MAX_STEPS
    assert result["scheil_artifact"]["content_addressed"] is True
    request = backend.requests[-1]
    assert request["operation"] == "scheil"
    assert request["selection"]["components"] == ["AL", "NI", "VA"]
    assert request["conditions"] == {
        "independent_composition_mole_fraction": {"NI": 0.5},
        "start_temperature_K": 1200.0,
        "step_temperature_K": 10.0,
        "pressure_Pa": 101325.0,
        "stop_liquid_fraction": 1e-4,
    }
    assert not ({"path", "code", "options", "max_steps"} & set(request["conditions"]))


@pytest.mark.parametrize("mode", ["scheil_missing_phase_component", "scheil_mass_closure_forgery"])
def test_scheil_host_rejects_complete_hash_bound_but_scientifically_invalid_artifact(
    tmp_path: Path,
    mode: str,
) -> None:
    _, resource_id, uploads, context, backend = _prepare_selected(tmp_path)
    inspection = inspect_calphad_database_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
    )
    backend.mode = mode

    result = run_calphad_scheil_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
        inspection_artifact_sha256=inspection["inspection_artifact"]["sha256"],
        components=["AL", "NI", "VA"],
        phases=["FCC_A1", "LIQUID"],
        independent_composition_mole_fraction={"NI": 0.5},
        start_temperature_K=1200.0,
        step_temperature_K=10.0,
    )

    assert result["ok"] is False
    assert result["error"] == "invalid_artifact_evidence"


@pytest.mark.parametrize(
    ("overrides", "expected_error"),
    [
        ({"phases": ["FCC_A1"]}, "invalid_typed_input"),
        ({"pressure_Pa": 100000.0}, "invalid_typed_input"),
        ({"step_temperature_K": 0.001}, "invalid_typed_input"),
        (
            {"independent_composition_mole_fraction": {"VA": 0.1}},
            "invalid_typed_input",
        ),
    ],
)
def test_scheil_invalid_scientific_conditions_fail_before_sandbox_execution(
    tmp_path: Path,
    overrides: dict[str, Any],
    expected_error: str,
) -> None:
    _, resource_id, uploads, context, backend = _prepare_selected(tmp_path)
    arguments: dict[str, Any] = {
        "inspection_artifact_sha256": "a" * 64,
        "components": ["AL", "NI", "VA"],
        "phases": ["FCC_A1", "LIQUID"],
        "independent_composition_mole_fraction": {"NI": 0.5},
        "start_temperature_K": 1200.0,
        "step_temperature_K": 10.0,
        "pressure_Pa": 101325.0,
    }
    arguments.update(overrides)

    result = run_calphad_scheil_typed(
        _settings(uploads),
        context,
        backend,
        upload_roots=(uploads,),
        resource_id=resource_id,
        **arguments,
    )

    assert result["ok"] is False
    assert result["error"] == expected_error
    assert backend.commands == []


def test_cli_scheil_uses_fixed_kernel_limits_and_retains_inspection_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    staged, outputs = _cli_roots(monkeypatch, tmp_path)
    payload = b"synthetic bounded TDB"
    database_path = staged / "database.tdb"
    database_path.write_bytes(payload)
    database = _resource_database(database_path, payload)
    inspection_result = {
        "schema_version": "1",
        "path": str(database_path),
        "format": "tdb",
        "sha256": database["sha256"],
        "size_bytes": database["size_bytes"],
        "available_components": ["AL", "NI", "VA"],
        "components": ["AL", "NI", "VA"],
        "available_phases": ["FCC_A1", "LIQUID"],
        "phases": ["FCC_A1", "LIQUID"],
        "pycalphad_version": "0.11.2",
    }
    monkeypatch.setattr(cli, "inspect_calphad_input", lambda *a, **k: inspection_result)
    inspection = cli.execute_request(
        {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "operation": "inspect",
            "runtime_image_id": "sha256:" + "f" * 64,
            "database": database,
            "selection": {"components": None, "phases": None},
        }
    )
    captured: dict[str, Any] = {}

    def bounded_scheil(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "schema_version": "ultra.materials.scheil-gulliver.v1",
            "database": {"path": str(database_path), "format": "tdb"},
        }

    monkeypatch.setattr(cli, "run_scheil_solidification", bounded_scheil)
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "operation": "scheil",
        "runtime_image_id": "sha256:" + "f" * 64,
        "database": database,
        "selection": {
            "components": ["AL", "NI", "VA"],
            "phases": ["FCC_A1", "LIQUID"],
        },
        "inspection_artifact_sha256": inspection["artifact"]["sha256"],
        "conditions": {
            "independent_composition_mole_fraction": {"NI": 0.5},
            "start_temperature_K": 1200.0,
            "step_temperature_K": 10.0,
            "pressure_Pa": 101325.0,
            "stop_liquid_fraction": 1e-4,
        },
    }

    result = cli.execute_request(request)

    assert result["ok"] is True
    assert result["operation"] == "scheil"
    assert result["inspection_artifact_sha256"] == inspection["artifact"]["sha256"]
    assert captured["max_steps"] == FIXED_SCHEIL_MAX_STEPS
    assert captured["wall_time_seconds"] == FIXED_WALL_TIME_SECONDS
    assert captured["max_result_bytes"] == FIXED_MAX_RESULT_BYTES
    assert captured["liquid_phase_name"] == "LIQUID"
    assert captured["independent_composition"] == {"NI": 0.5}
    assert (outputs / "scheil" / f"{result['artifact']['sha256']}.json").is_file()

    request_with_options = json.loads(json.dumps(request))
    request_with_options["options"] = {"adaptive": False}
    with pytest.raises(TypedCalphadError, match="schema mismatch"):
        cli.execute_request(request_with_options)


def test_agent_registers_typed_calphad_tools_in_manifest_and_code_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import ultra_deepagents.agent as agent_module

    payload, _, uploads, context, _ = _prepare_selected(tmp_path)
    assert payload
    settings = _settings(uploads)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        agent_module,
        "resolve_docker_image_id",
        lambda _image: "sha256:" + "1" * 64,
    )
    monkeypatch.setattr(
        agent_module,
        "create_deep_agent",
        lambda **kwargs: captured.update(kwargs) or "agent",
    )
    built = agent_module.build_research_agent(
        settings,
        model=object(),
        workspace_dir=tmp_path / "agent-workspace",
        artifact_dir=tmp_path / "agent-outputs",
        context=context,
    )
    assert built == "agent"
    names = {getattr(item, "name", "") for item in captured["tools"]}
    assert {
        "calphad_inspect_database",
        "calphad_run_equilibrium",
        "calphad_run_scheil",
    } <= names
    code_runner = next(item for item in captured["subagents"] if item["name"] == "code-runner")
    subagent_names = {getattr(item, "name", "") for item in code_runner["tools"]}
    assert {
        "calphad_inspect_database",
        "calphad_run_equilibrium",
        "calphad_run_scheil",
    } <= subagent_names
    manifest_tool = next(
        item for item in captured["tools"] if item.name == "tool_capability_manifest"
    )
    manifest = json.loads(manifest_tool.func())
    assert "calphad_inspect_database" in manifest["registered_tools"]
    assert "calphad_run_scheil" in manifest["registered_tools"]
    manifest_code_runner = next(
        item for item in manifest["available_subagents"] if item["name"] == "code-runner"
    )
    assert "calphad_run_equilibrium" in manifest_code_runner["tool_names"]
    assert "calphad_run_scheil" in manifest_code_runner["tool_names"]

    assert agent_module._should_register_calphad_tools(context) is True
    non_calphad = AgentRunContext(
        assistant_id="a",
        org_id="org",
        user_id="user",
        project_id="project",
        thread_id="thread",
        run_id="run",
        goal="Calculate an XRD pattern from this CIF",
    )
    assert agent_module._should_register_calphad_tools(non_calphad) is False
    for goal in (
        "Calculate Al-Co-W equilibrium phase fractions at 1173 K",
        "Compute an isothermal section for this alloy assessment",
        "Evaluate thermodynamic phase stability at one composition",
        "Run a Scheil-Gulliver solidification path for this alloy",
    ):
        routed = AgentRunContext(
            assistant_id="a",
            org_id="org",
            user_id="user",
            project_id="project",
            thread_id="thread",
            run_id="run",
            goal=goal,
        )
        assert agent_module._should_register_calphad_tools(routed) is True


@pytest.mark.parametrize("operator_cap", [0, 21_600, 45])
def test_typed_calphad_backend_has_immutable_nonextensible_outer_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operator_cap: int,
) -> None:
    import ultra_deepagents.agent as agent_module

    def callback(_event: Any) -> None:
        return None

    base = DockerSandboxBackend(
        workspace_dir=tmp_path / f"workspace-{operator_cap}",
        outputs_dir=tmp_path / f"outputs-{operator_cap}",
        config=DockerSandboxConfig(
            image="sha256:" + "a" * 64,
            timeout_seconds=operator_cap,
            network="bridge",
            cpus=2,
            memory="4g",
            pids_limit=256,
            gpus="all",
            run_id="run-cap",
        ),
        progress_callback=callback,
    )
    bounded = agent_module._bounded_calphad_sandbox_backend(base)
    assert bounded is not None
    expected = 45 if operator_cap == 45 else 60
    assert bounded.config.timeout_seconds == expected
    assert bounded.config.image == base.config.image
    assert bounded.config.network == "none"
    assert bounded.config.cpus == 2
    assert bounded.config.memory == "4g"
    assert bounded.config.pids_limit == 256
    assert bounded.config.no_new_privileges is True
    assert bounded.config.gpus == ""
    assert bounded.workspace_dir == base.workspace_dir
    assert bounded.outputs_dir == base.outputs_dir
    assert bounded._progress_callback is callback
    assert base.config.network == "bridge"
    assert base.config.no_new_privileges is None
    assert base.config.gpus == "all"
    monkeypatch.setenv("ULTRA_DEEPAGENTS_SANDBOX_NO_NEW_PRIVILEGES", "false")
    docker_command = bounded.build_docker_command("true")
    assert docker_command[docker_command.index("--network") + 1] == "none"
    assert "no-new-privileges" in docker_command

    mutable = DockerSandboxBackend(
        workspace_dir=tmp_path / f"mutable-workspace-{operator_cap}",
        outputs_dir=tmp_path / f"mutable-outputs-{operator_cap}",
        config=DockerSandboxConfig(image="bisque-ultra-codeexec:py311"),
    )
    assert agent_module._bounded_calphad_sandbox_backend(mutable) is None


@pytest.mark.parametrize(
    ("cpus", "memory", "pids_limit"),
    [
        (0, "4g", 256),
        (2, "", 256),
        (2, "4g", 0),
    ],
)
def test_typed_calphad_backend_rejects_unbounded_resources(
    tmp_path: Path,
    cpus: float,
    memory: str,
    pids_limit: int,
) -> None:
    import ultra_deepagents.agent as agent_module

    base = DockerSandboxBackend(
        workspace_dir=tmp_path / "workspace",
        outputs_dir=tmp_path / "outputs",
        config=DockerSandboxConfig(
            image="sha256:" + "b" * 64,
            cpus=cpus,
            memory=memory,
            pids_limit=pids_limit,
        ),
    )
    assert agent_module._bounded_calphad_sandbox_backend(base) is None
