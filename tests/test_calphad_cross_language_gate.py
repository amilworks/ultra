from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "calphad_cross_language_gate.py"
SPEC = importlib.util.spec_from_file_location("calphad_cross_language_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)

GIT_SHA = "a" * 40
IMAGE_ID = "sha256:" + "b" * 64
RESOURCE_ID = "calphad-cross-language-" + GIT_SHA[:20]
DATABASE_SHA = "c" * 64


def _runtime(*, inspected: bool = True) -> gate.RuntimeAttestation:
    return gate.RuntimeAttestation(
        mode="pinned_image" if inspected else "host_fallback_non_oci",
        runtime_image_id=IMAGE_ID,
        image_ref="materials:test" if inspected else "",
        image_title="Ultra Deep Agents scientific sandbox",
        image_revision=GIT_SHA,
        pythonpath="/opt/ultra-runtime" if inspected else "",
        image_inspected=inspected,
        image_inspection_payload=b"[{}]" if inspected else b"",
    )


def _docker_inspection(
    *,
    image_id: str = IMAGE_ID,
    title: str = "Ultra Deep Agents scientific sandbox",
    revision: str = GIT_SHA,
    pythonpath: str = "/opt/ultra-runtime",
) -> bytes:
    return json.dumps(
        [
            {
                "Id": image_id,
                "Config": {
                    "Env": [f"PYTHONPATH={pythonpath}", "PYTHONHASHSEED=0"],
                    "Labels": {
                        "org.opencontainers.image.title": title,
                        "org.opencontainers.image.revision": revision,
                    },
                },
            }
        ]
    ).encode()


def _database() -> dict[str, object]:
    return {
        "scheme": "postgresql",
        "host": "db.internal",
        "port": 55432,
        "database": "ultra_calphad_qualification",
        "role": "ultra_calphad_serving",
        "serving_role": "ultra_calphad_serving",
        "migration_role": "postgres",
        "credentials_recorded": False,
    }


def _artifact(sha: str, size: int) -> dict[str, object]:
    return {"sha256": sha, "size_bytes": size}


def _resource() -> dict[str, object]:
    return {
        "resource_id": RESOURCE_ID,
        "database_sha256": DATABASE_SHA,
        "database_size_bytes": 21274,
        "database_format": "tdb",
        "assessment_pressure_limits_Pa": [101325.0, 101325.0],
    }


def _backend_marker() -> dict[str, object]:
    return {
        "schema_version": gate.BACKEND_EVIDENCE_SCHEMA,
        "live_http_callback": True,
        "live_postgres": True,
        "database": {
            "name": "ultra_calphad_qualification",
            "server_address": "10.0.0.2",
            "server_port": 5432,
            "connection_target_host": "db.internal",
            "connection_target_port": 55432,
            "transaction_read_only": "off",
            "serving_role": "ultra_calphad_serving",
            "migration_role": "postgres",
            "serving_role_superuser": False,
            "serving_role_create_role": False,
            "serving_role_create_database": False,
            "serving_role_replication": False,
            "serving_role_bypass_rls": False,
            "serving_role_owned_tables": [],
            "serving_role_owned_functions": [],
            "calphad_owner_roles": ["postgres"],
            "calphad_reachable_roles": [],
            "calphad_owner_role_reachable": False,
            "public_schema_owner": "pg_database_owner",
            "public_owner_role_reachable": False,
            "can_create_public_schema": False,
            "serving_role_select_all": True,
            "serving_role_insert_all": False,
            "serving_role_insert_any": False,
            "serving_role_execute_create_revision": True,
            "serving_role_execute_append_validation": True,
            "serving_writer_functions_exact": True,
            "serving_execute_unexpected_writer": False,
            "serving_role_execute_internal": False,
            "serving_role_public_execute": False,
            "serving_unexpected_table_acl_grantees": [],
            "serving_unexpected_function_acl_grantees": [],
            "serving_role_mutation_privilege": False,
        },
        "resource_id": RESOURCE_ID,
        "revision_id": "revision-1",
        "run_id": "run-1",
        "runtime_image_id": IMAGE_ID,
        "pycalphad_version": gate.PYCALPHAD_VERSION,
        "database_sha256": DATABASE_SHA,
        "database_size_bytes": 21274,
        "database_format": "tdb",
        "assessment_pressure_limits_Pa": [101325.0, 101325.0],
        "database_inventory_sha256": "d" * 64,
        "inspect": {
            "evidence_sha256": "e" * 64,
            "evidence_size_bytes": 1234,
            "request_sha256": "1" * 64,
            "evidence_retention": "retained",
            "promotable": True,
            "postgres_bytes_exact": True,
        },
        "equilibrium": {
            "evidence_sha256": "f" * 64,
            "evidence_size_bytes": 4321,
            "request_sha256": "2" * 64,
            "inspection_evidence_sha256": "e" * 64,
            "evidence_retention": "retained",
            "promotable": True,
            "postgres_bytes_exact": True,
        },
    }


def _go_event(action: str, *, output: str = "", package: str = gate.GO_PACKAGE) -> bytes:
    event: dict[str, object] = {
        "Action": action,
        "Package": package,
        "Test": gate.GO_TEST,
    }
    if output:
        event["Output"] = output
    return (json.dumps(event) + "\n").encode()


def _package_event(action: str, *, package: str = gate.GO_PACKAGE) -> bytes:
    return (json.dumps({"Action": action, "Package": package}) + "\n").encode()


def _execution_contract() -> dict[str, object]:
    return {
        "interface": "fixed ultra_deepagents.materials.calphad public surface",
        "caller_code_accepted": False,
        "caller_models_or_solver_options_accepted": False,
        "network": "none",
        "no_new_privileges": True,
        "read_only_root_filesystem": True,
        "cap_drop_all": True,
        "cpus_at_most": 8.0,
        "memory_bytes_at_most": 32 * 1024**3,
        "pids_at_most": 4096,
        "runtime_image_id": IMAGE_ID,
        "max_components": 32,
        "max_phases": 128,
        "max_axis_values": 64,
        "max_grid_points": 256,
        "wall_time_seconds": 30.0,
        "max_result_bytes": 16 * 1024 * 1024,
    }


def test_qualification_database_pair_redacts_credentials_and_requires_role_separation() -> None:
    database = gate.qualification_database_pair(
        "postgresql://ultra_calphad_serving:secret@db.internal:55432/ultra_calphad_qualification",
        "postgresql://postgres:other@db.internal:55432/ultra_calphad_qualification",
    )

    assert database["credentials_recorded"] is False
    assert database["serving_role"] == "ultra_calphad_serving"
    assert database["migration_role"] == "postgres"
    assert "secret" not in repr(database)
    assert "other" not in repr(database)

    with pytest.raises(gate.QualificationError, match="must be distinct"):
        gate.qualification_database_pair(
            "postgresql://same:a@db.internal/ultra_calphad_qualification",
            "postgresql://same:b@db.internal/ultra_calphad_qualification",
        )


def test_source_manifest_covers_the_exact_cross_language_implementation() -> None:
    manifest = gate.build_source_manifest(ROOT)

    assert [record["path"] for record in manifest] == [
        path.as_posix() for path in gate.SOURCE_PATHS
    ]
    assert len(manifest) == 20
    assert all(record["size_bytes"] > 0 for record in manifest)
    assert all(gate.SHA256_RE.fullmatch(record["sha256"]) for record in manifest)


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://serving:x@db.internal/ultra_production",
        "postgresql://serving:x@db.internal/ultra_live_test",
        "postgresql://serving:x@db.internal/ordinary_database",
        "postgresql://serving:x@db.internal/postgres",
    ],
)
def test_qualification_database_identity_rejects_production_or_unmarked_targets(dsn: str) -> None:
    with pytest.raises(gate.QualificationError, match="refusing"):
        gate.qualification_database_identity(dsn)


def test_inspect_image_payload_requires_exact_id_revision_title_and_pythonpath() -> None:
    attestation = gate.inspect_image_payload(
        _docker_inspection(),
        image_ref="materials:test",
        expected_image_id=IMAGE_ID,
        expected_title="Ultra Deep Agents scientific sandbox",
        expected_git_sha=GIT_SHA,
    )

    assert attestation.image_inspected is True
    assert attestation.runtime_image_id == IMAGE_ID
    assert attestation.image_revision == GIT_SHA
    assert attestation.image_inspection_payload == _docker_inspection()

    variants = [
        _docker_inspection(image_id="sha256:" + "9" * 64),
        _docker_inspection(revision="9" * 40),
        _docker_inspection(title="attacker image"),
        _docker_inspection(pythonpath="/workspace"),
    ]
    for payload in variants:
        with pytest.raises(gate.QualificationError):
            gate.inspect_image_payload(
                payload,
                image_ref="materials:test",
                expected_image_id=IMAGE_ID,
                expected_title="Ultra Deep Agents scientific sandbox",
                expected_git_sha=GIT_SHA,
            )


def test_staged_database_filename_is_content_addressed() -> None:
    assert (
        gate.staged_database_filename({"filename": "original.TDB", "sha256": DATABASE_SHA})
        == DATABASE_SHA + ".tdb"
    )
    assert (
        gate.staged_database_filename({"filename": "original.dat", "sha256": DATABASE_SHA})
        == DATABASE_SHA + ".dat"
    )
    with pytest.raises(gate.QualificationError):
        gate.staged_database_filename({"filename": "original.txt", "sha256": DATABASE_SHA})
    with pytest.raises(gate.QualificationError):
        gate.staged_database_filename({"filename": "original.db", "sha256": DATABASE_SHA})


def test_reference_binding_carries_fixed_manifest_pressure_scope() -> None:
    record, _ = gate.load_reference_database(ROOT)

    binding = gate.database_binding(
        record,
        resource_id=RESOURCE_ID,
        path="/workspace/.ultra/calphad/staged/" + record["sha256"] + ".tdb",
    )

    assert record["format"] == "tdb"
    assert record["assessment_pressure_limits_Pa"] == [101325.0, 101325.0]
    assert binding["database_format"] == "tdb"
    assert binding["assessment_pressure_limits_Pa"] == [101325.0, 101325.0]


def test_typed_request_uses_published_three_phase_global_checkpoint() -> None:
    request = gate.typed_request(
        operation="equilibrium",
        runtime_image_id=IMAGE_ID,
        binding={"kind": "resource"},
        inspection_sha256="d" * 64,
    )

    assert request["selection"] == {
        "components": list(gate.REFERENCE_COMPONENTS),
        "phases": list(gate.REFERENCE_PHASES),
    }
    assert request["conditions"] == {
        "temperatures_K": [1173.0],
        "pressures_Pa": [101325.0],
        "independent_compositions": {"CO": [0.26], "W": [0.065]},
    }
    assert len(request["selection"]["phases"]) == 18


def test_reference_checkpoint_requires_expected_three_phase_vertices() -> None:
    evidence = {
        "request": gate.typed_request(
            operation="equilibrium",
            runtime_image_id=IMAGE_ID,
            binding={"kind": "resource"},
            inspection_sha256="d" * 64,
        ),
        "result": {
            "request": {
                "dependent_component": "AL",
                "conditions": {
                    "independent_compositions": {
                        "CO": {"values": [0.26], "units": "mole_fraction"},
                        "W": {"values": [0.065], "units": "mole_fraction"},
                    }
                },
            },
            "result": {
                "points": [
                    {
                        "stable_phases": [
                            {"name": phase} for phase in gate.REFERENCE_STABLE_PHASES
                        ],
                        "stable_phase_vertices": [
                            {"phase": phase} for phase in gate.REFERENCE_STABLE_PHASES
                        ],
                        "GM_J_per_mol": gate.REFERENCE_GM_J_PER_MOL,
                    }
                ]
            },
        },
    }

    checkpoint = gate.validate_reference_equilibrium_checkpoint(evidence)
    assert checkpoint["observed_stable_phases"] == list(gate.REFERENCE_STABLE_PHASES)
    assert checkpoint["global_phase_count"] == 18

    forged = deepcopy(evidence)
    forged["result"]["result"]["points"][0]["stable_phases"] = [{"name": "LIQUID"}]
    with pytest.raises(gate.QualificationError, match="three-phase"):
        gate.validate_reference_equilibrium_checkpoint(forged)

    wrong_minimum = deepcopy(evidence)
    wrong_minimum["result"]["result"]["points"][0]["GM_J_per_mol"] = -85512.6057
    with pytest.raises(gate.QualificationError, match="Gibbs energy"):
        gate.validate_reference_equilibrium_checkpoint(wrong_minimum)

    wrong_dependent = deepcopy(evidence)
    wrong_dependent["result"]["request"]["dependent_component"] = "W"
    with pytest.raises(gate.QualificationError, match="AL-dependent"):
        gate.validate_reference_equilibrium_checkpoint(wrong_dependent)


def test_docker_typed_cli_command_uses_immutable_image_and_exact_isolation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outputs = tmp_path / "outputs"
    request = workspace / ".ultra/calphad/requests" / ("1" * 64 + ".json")
    command = gate.docker_typed_cli_command(
        runtime_image_id=IMAGE_ID,
        trusted_runtime_root="/opt/ultra-runtime",
        workspace=workspace,
        outputs=outputs,
        request_path=request,
    )

    assert command[0:3] == ("docker", "run", "--rm")
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--cpus") + 1] == "8"
    assert command[command.index("--memory") + 1] == "32g"
    assert command[command.index("--pids-limit") + 1] == "4096"
    assert IMAGE_ID in command
    assert "materials:test" not in command
    assert "-I" in command
    assert "/opt/ultra-runtime" in command[command.index("-c") + 1]
    assert "ultra_deepagents.materials.calphad_cli" in command[command.index("-c") + 1]
    assert command[-1] == "/workspace/.ultra/calphad/requests/" + "1" * 64 + ".json"


def test_bundle_writers_reject_final_symlinks(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.write_bytes(b"unchanged")
    target = tmp_path / "artifact.json"
    target.symlink_to(outside)

    with pytest.raises(gate.QualificationError):
        gate.retain_exact_artifact(target, b"evidence", label="test artifact")
    with pytest.raises(gate.QualificationError):
        gate.replace_regular_file(target, b"manifest", label="test manifest")
    assert outside.read_bytes() == b"unchanged"


def test_parse_go_test_json_requires_one_exact_pass_and_one_evidence_marker() -> None:
    marker = _backend_marker()
    payload = (
        _go_event(
            "output",
            output="    calphad_cross_language_test.go:1: "
            + gate.BACKEND_MARKER
            + json.dumps(marker),
        )
        + _go_event("pass")
        + _package_event("pass")
    )

    parsed, test = gate.parse_go_test_json(payload)

    assert parsed == marker
    assert test == {"name": gate.GO_TEST, "package": gate.GO_PACKAGE, "action": "pass"}


def test_parse_go_test_json_reconstructs_chunked_marker_output() -> None:
    marker = _backend_marker()
    line = gate.BACKEND_MARKER + json.dumps(marker) + "\n"
    payload = (
        _go_event("output", output=line[:943])
        + _go_event("output", output=line[943:])
        + _go_event("pass")
        + _package_event("pass")
    )

    parsed, test = gate.parse_go_test_json(payload)

    assert parsed == marker
    assert test == {"name": gate.GO_TEST, "package": gate.GO_PACKAGE, "action": "pass"}


@pytest.mark.parametrize(
    "payload",
    [
        _go_event("skip") + _package_event("pass"),
        _go_event("fail") + _package_event("fail"),
        _go_event("pass", package="attacker.example/forged") + _package_event("pass"),
        _go_event("pass") + _package_event("pass"),
    ],
)
def test_parse_go_test_json_rejects_skip_failure_wrong_package_or_missing_marker(
    payload: bytes,
) -> None:
    with pytest.raises(gate.QualificationError):
        gate.parse_go_test_json(payload)


def test_parse_go_test_json_rejects_duplicate_evidence_markers() -> None:
    marker_output = gate.BACKEND_MARKER + json.dumps(_backend_marker())
    payload = (
        _go_event("output", output=marker_output)
        + _go_event("output", output=marker_output)
        + _go_event("pass")
        + _package_event("pass")
    )

    with pytest.raises(gate.QualificationError, match="exactly one"):
        gate.parse_go_test_json(payload)


def test_parse_go_test_json_rejects_missing_or_failed_package_terminal() -> None:
    marker_output = gate.BACKEND_MARKER + json.dumps(_backend_marker())
    for package_terminal in (b"", _package_event("fail"), _package_event("skip")):
        payload = _go_event("output", output=marker_output) + _go_event("pass") + package_terminal
        with pytest.raises(gate.QualificationError, match="test/package"):
            gate.parse_go_test_json(payload)


def test_validate_backend_marker_binds_exact_retained_bytes_and_lineage() -> None:
    checks = gate.validate_backend_marker(
        _backend_marker(),
        database=_database(),
        runtime=_runtime(),
        resource=_resource(),
        inspection=_artifact("e" * 64, 1234),
        equilibrium=_artifact("f" * 64, 4321),
    )

    assert checks
    assert all(checks.values())


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("live_http_callback",), False),
        (("live_postgres",), False),
        (("runtime_image_id",), "sha256:" + "9" * 64),
        (("database_format",), "dat"),
        (("assessment_pressure_limits_Pa",), [1e-9, 1e12]),
        (("database", "serving_role_superuser"), True),
        (("database", "connection_target_host"), "attacker.internal"),
        (("database", "transaction_read_only"), "on"),
        (("database", "serving_role_create_role"), True),
        (("database", "serving_role_create_database"), True),
        (("database", "serving_role_replication"), True),
        (("database", "serving_role_bypass_rls"), True),
        (("database", "calphad_owner_roles"), []),
        (("database", "calphad_reachable_roles"), ["pg_monitor"]),
        (("database", "calphad_owner_role_reachable"), True),
        (("database", "public_schema_owner"), ""),
        (("database", "public_owner_role_reachable"), True),
        (("database", "can_create_public_schema"), True),
        (("database", "serving_role_insert_all"), True),
        (("database", "serving_role_insert_any"), True),
        (("database", "serving_role_execute_create_revision"), False),
        (("database", "serving_role_execute_append_validation"), False),
        (("database", "serving_writer_functions_exact"), False),
        (("database", "serving_execute_unexpected_writer"), True),
        (("database", "serving_role_execute_internal"), True),
        (("database", "serving_role_public_execute"), True),
        (("database", "serving_unexpected_table_acl_grantees"), ["PUBLIC"]),
        (("database", "serving_unexpected_function_acl_grantees"), ["pg_monitor"]),
        (("database", "unexpected_execute_capability"), True),
        (("database", "serving_role_mutation_privilege"), True),
        (("inspect", "evidence_sha256"), "9" * 64),
        (("inspect", "postgres_bytes_exact"), False),
        (("equilibrium", "evidence_retention"), "unretained"),
        (("equilibrium", "inspection_evidence_sha256"), "9" * 64),
        (("equilibrium", "request_sha256"), "1" * 64),
    ],
)
def test_validate_backend_marker_fails_closed_on_tampering(
    path: tuple[str, ...], replacement: object
) -> None:
    marker = deepcopy(_backend_marker())
    target = marker
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment,index]
    target[path[-1]] = replacement  # type: ignore[index]

    with pytest.raises(gate.QualificationError):
        gate.validate_backend_marker(
            marker,
            database=_database(),
            runtime=_runtime(),
            resource=_resource(),
            inspection=_artifact("e" * 64, 1234),
            equilibrium=_artifact("f" * 64, 4321),
        )


def test_validate_artifact_requires_content_address_pycalphad_and_inspection_lineage(
    tmp_path: Path,
) -> None:
    inspect_evidence = {
        "schema_version": gate.TYPED_EVIDENCE_SCHEMA,
        "operation": "inspect",
        "database_binding": {
            "kind": "resource",
            "resource_id": RESOURCE_ID,
            "database_format": "tdb",
            "sha256": DATABASE_SHA,
            "size_bytes": 21274,
            "assessment_pressure_limits_Pa": [101325.0, 101325.0],
            "binding_schema": "ultra.selected_resource.v1",
            "binding_authority": "control_resource_catalog",
            "declaration_authority": "resource_owner",
        },
        "request": {"operation": "inspect", "runtime_image_id": IMAGE_ID},
        "result": {
            "pycalphad_version": gate.PYCALPHAD_VERSION,
            "format": "tdb",
            "name": DATABASE_SHA + ".tdb",
            "path": "/workspace/.ultra/calphad/staged/" + DATABASE_SHA + ".tdb",
        },
        "execution_contract": _execution_contract(),
    }
    inspect_payload = gate.canonical_json(inspect_evidence)
    inspect_sha = hashlib.sha256(inspect_payload).hexdigest()
    inspect_path = tmp_path / f"{inspect_sha}.json"
    inspect_path.write_bytes(inspect_payload)

    result = gate.validate_artifact(
        inspect_path,
        operation="inspect",
        runtime_image_id=IMAGE_ID,
        resource_id=RESOURCE_ID,
        database_sha256=DATABASE_SHA,
        database_size_bytes=21274,
        require_canonical_staged_path=True,
    )
    assert result["sha256"] == inspect_sha

    pressure_evidence = deepcopy(inspect_evidence)
    pressure_evidence["database_binding"]["assessment_pressure_limits_Pa"] = [
        1e-9,
        1e12,
    ]
    pressure_payload = gate.canonical_json(pressure_evidence)
    pressure_path = tmp_path / f"{hashlib.sha256(pressure_payload).hexdigest()}.json"
    pressure_path.write_bytes(pressure_payload)
    with pytest.raises(gate.QualificationError, match="selected runtime/resource"):
        gate.validate_artifact(
            pressure_path,
            operation="inspect",
            runtime_image_id=IMAGE_ID,
            resource_id=RESOURCE_ID,
            database_sha256=DATABASE_SHA,
            database_size_bytes=21274,
        )

    host_evidence = deepcopy(inspect_evidence)
    host_evidence["result"]["path"] = "/tmp/staged/" + DATABASE_SHA + ".tdb"
    host_payload = gate.canonical_json(host_evidence)
    host_path = tmp_path / f"{hashlib.sha256(host_payload).hexdigest()}.json"
    host_path.write_bytes(host_payload)
    with pytest.raises(gate.QualificationError, match="canonical callback path"):
        gate.validate_artifact(
            host_path,
            operation="inspect",
            runtime_image_id=IMAGE_ID,
            resource_id=RESOURCE_ID,
            database_sha256=DATABASE_SHA,
            database_size_bytes=21274,
            require_canonical_staged_path=True,
        )

    db_evidence = deepcopy(inspect_evidence)
    db_evidence["result"]["name"] = DATABASE_SHA + ".db"
    db_evidence["result"]["path"] = "/workspace/.ultra/calphad/staged/" + DATABASE_SHA + ".db"
    db_payload = gate.canonical_json(db_evidence)
    db_path = tmp_path / f"{hashlib.sha256(db_payload).hexdigest()}.json"
    db_path.write_bytes(db_payload)
    with pytest.raises(gate.QualificationError, match="name is not content-addressed"):
        gate.validate_artifact(
            db_path,
            operation="inspect",
            runtime_image_id=IMAGE_ID,
            resource_id=RESOURCE_ID,
            database_sha256=DATABASE_SHA,
            database_size_bytes=21274,
            require_canonical_staged_path=True,
        )

    equilibrium_evidence = deepcopy(inspect_evidence)
    equilibrium_evidence["operation"] = "equilibrium"
    equilibrium_evidence["request"] = {
        "operation": "equilibrium",
        "runtime_image_id": IMAGE_ID,
        "inspection_artifact_sha256": inspect_sha,
    }
    equilibrium_evidence["result"] = {
        "database": {
            "pycalphad_version": gate.PYCALPHAD_VERSION,
            "format": "tdb",
            "name": DATABASE_SHA + ".tdb",
            "path": "/workspace/.ultra/calphad/staged/" + DATABASE_SHA + ".tdb",
        }
    }
    equilibrium_payload = gate.canonical_json(equilibrium_evidence)
    equilibrium_sha = hashlib.sha256(equilibrium_payload).hexdigest()
    equilibrium_path = tmp_path / f"{equilibrium_sha}.json"
    equilibrium_path.write_bytes(equilibrium_payload)

    gate.validate_artifact(
        equilibrium_path,
        operation="equilibrium",
        runtime_image_id=IMAGE_ID,
        resource_id=RESOURCE_ID,
        database_sha256=DATABASE_SHA,
        database_size_bytes=21274,
        inspection_sha256=inspect_sha,
        require_canonical_staged_path=True,
    )
    with pytest.raises(gate.QualificationError, match="inspection artifact"):
        gate.validate_artifact(
            equilibrium_path,
            operation="equilibrium",
            runtime_image_id=IMAGE_ID,
            resource_id=RESOURCE_ID,
            database_sha256=DATABASE_SHA,
            database_size_bytes=21274,
            inspection_sha256="9" * 64,
        )


def test_only_clean_pinned_image_with_every_live_check_can_qualify() -> None:
    checks = {"live_http": True, "live_postgres": True, "retained": True}
    assert gate.is_production_live_qualified(
        mode="pinned-image",
        repository={"clean": True},
        runtime=_runtime(inspected=True),
        checks=checks,
    )
    assert not gate.is_production_live_qualified(
        mode="host-fallback",
        repository={"clean": True},
        runtime=_runtime(inspected=False),
        checks=checks,
    )
    assert not gate.is_production_live_qualified(
        mode="pinned-image",
        repository={"clean": False},
        runtime=_runtime(inspected=True),
        checks=checks,
    )
    assert not gate.is_production_live_qualified(
        mode="pinned-image",
        repository={"clean": True},
        runtime=_runtime(inspected=True),
        checks={**checks, "retained": False},
    )
    lean_runtime = gate.RuntimeAttestation(
        mode="pinned_image",
        runtime_image_id=IMAGE_ID,
        image_title="Ultra deterministic materials domain gate",
        image_revision=GIT_SHA,
        pythonpath="/opt/ultra/src",
        image_inspected=True,
        image_inspection_payload=b"[{}]",
    )
    assert not gate.is_production_live_qualified(
        mode="pinned-image",
        repository={"clean": True},
        runtime=lean_runtime,
        checks=checks,
    )
