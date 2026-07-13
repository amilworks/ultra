from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "calphad_ledger_gate.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("calphad_ledger_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _event(action: str, test: str, package: str = "") -> bytes:
    if not package:
        package = gate.EXPECTED_GO_TEST_PACKAGES[test.split("/", 1)[0]]
    return (json.dumps({"Action": action, "Test": test, "Package": package}) + "\n").encode()


def _postgres_identity() -> dict[str, object]:
    return {
        "database": "ultra_qualification",
        "server_address": "10.0.0.8",
        "server_port": 5432,
        "role": "ultra_qualification_role",
        "transaction_read_only": "off",
        "role_superuser": False,
        "role_create_role": False,
        "role_create_database": False,
        "role_replication": False,
        "role_bypass_rls": False,
        "calphad_owned_tables": [],
        "calphad_owned_functions": [],
        "calphad_owner_roles": ["ultra_qualification_migration"],
        "calphad_reachable_roles": [],
        "calphad_owner_role_reachable": False,
        "public_schema_owner": "pg_database_owner",
        "public_owner_role_reachable": False,
        "can_create_public_schema": False,
        "calphad_select_all": True,
        "calphad_insert_all": False,
        "calphad_insert_any": False,
        "calphad_execute_create_revision": True,
        "calphad_execute_append_validation": True,
        "calphad_writer_functions_exact": True,
        "calphad_execute_unexpected_writer": False,
        "calphad_execute_internal": False,
        "calphad_public_execute": False,
        "calphad_unexpected_table_acl_grantees": [],
        "calphad_unexpected_function_acl_grantees": [],
        "calphad_mutation_privilege": False,
        "connection_target_host": "db.internal",
        "connection_target_port": 55432,
    }


def test_parse_go_test_json_requires_every_exact_non_skipped_test() -> None:
    payload = b"".join(_event("pass", name) for name in gate.REQUIRED_CALPHAD_LEDGER_TESTS)

    records, failures = gate.parse_go_test_json(payload)

    assert failures == []
    assert [record["name"] for record in records] == list(gate.REQUIRED_CALPHAD_LEDGER_TESTS)
    assert all(record["passed"] and not record["skipped"] for record in records)


def test_postgres_invariants_require_exact_subtests_and_observed_database_identity() -> None:
    events = [
        _event("pass", test)
        for tests in gate.POSTGRES_INVARIANT_TEST_EVIDENCE.values()
        for test in tests
    ]
    identity = _postgres_identity()
    events.append(
        (
            json.dumps(
                {
                    "Action": "output",
                    "Test": gate.POSTGRES_TEST,
                    "Package": gate.STORE_GO_PACKAGE,
                    "Output": gate.POSTGRES_IDENTITY_MARKER + json.dumps(identity),
                }
            )
            + "\n"
        ).encode()
    )

    outcomes, records, observed, failures = gate.parse_postgres_invariant_evidence(
        b"".join(events),
        expected_database={
            "database": "ultra_qualification",
            "host": "db.internal",
            "port": 55432,
            "migration_role": "ultra_qualification_migration",
        },
        expected_role="ultra_qualification_role",
    )

    assert failures == []
    assert all(outcomes.values())
    assert all(record["passed"] for record in records)
    assert observed == identity


def test_postgres_identity_reassembles_adjacent_go_test_output_fragments() -> None:
    events = [
        _event("pass", test)
        for tests in gate.POSTGRES_INVARIANT_TEST_EVIDENCE.values()
        for test in tests
    ]
    identity = _postgres_identity()
    encoded = json.dumps(identity)
    split_at = encoded.index('"role_superuser": false') + len('"role_superuser": fa')
    for fragment in (
        gate.POSTGRES_IDENTITY_MARKER + encoded[:split_at],
        encoded[split_at:] + "\n",
    ):
        events.append(
            (
                json.dumps(
                    {
                        "Action": "output",
                        "Test": gate.POSTGRES_TEST,
                        "Package": gate.STORE_GO_PACKAGE,
                        "Output": fragment,
                    }
                )
                + "\n"
            ).encode()
        )

    outcomes, records, observed, failures = gate.parse_postgres_invariant_evidence(
        b"".join(events),
        expected_database={
            "database": "ultra_qualification",
            "host": "db.internal",
            "port": 55432,
            "migration_role": "ultra_qualification_migration",
        },
        expected_role="ultra_qualification_role",
    )

    assert failures == []
    assert all(outcomes.values())
    assert all(record["passed"] for record in records)
    assert observed == identity


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("role_create_database", True),
        ("role_replication", True),
        ("calphad_reachable_roles", ["pg_monitor"]),
        ("calphad_insert_all", True),
        ("calphad_insert_any", True),
        ("calphad_execute_create_revision", False),
        ("calphad_execute_append_validation", False),
        ("calphad_writer_functions_exact", False),
        ("calphad_execute_unexpected_writer", True),
        ("calphad_execute_internal", True),
        ("calphad_public_execute", True),
        ("calphad_unexpected_table_acl_grantees", ["PUBLIC"]),
        ("calphad_unexpected_function_acl_grantees", ["pg_monitor"]),
    ],
)
def test_postgres_identity_requires_execute_only_calphad_writers(
    field: str, unsafe_value: object
) -> None:
    identity = _postgres_identity()
    identity[field] = unsafe_value
    events = [
        _event("pass", test)
        for tests in gate.POSTGRES_INVARIANT_TEST_EVIDENCE.values()
        for test in tests
    ]
    events.append(
        (
            json.dumps(
                {
                    "Action": "output",
                    "Test": gate.POSTGRES_TEST,
                    "Package": gate.STORE_GO_PACKAGE,
                    "Output": gate.POSTGRES_IDENTITY_MARKER + json.dumps(identity),
                }
            )
            + "\n"
        ).encode()
    )

    _, _, _, failures = gate.parse_postgres_invariant_evidence(
        b"".join(events),
        expected_database={
            "database": "ultra_qualification",
            "host": "db.internal",
            "port": 55432,
            "migration_role": "ultra_qualification_migration",
        },
        expected_role="ultra_qualification_role",
    )

    assert any("writable database identity" in failure for failure in failures)


@pytest.mark.parametrize("action", ["skip", "fail"])
def test_parse_go_test_json_rejects_skip_or_failure(action: str) -> None:
    payload = b"".join(
        _event(action if index == 0 else "pass", name)
        for index, name in enumerate(gate.REQUIRED_CALPHAD_LEDGER_TESTS)
    )

    records, failures = gate.parse_go_test_json(payload)

    assert records[0]["passed"] is False
    assert failures


def test_parse_go_test_json_rejects_forged_package_identity() -> None:
    payload = b"".join(
        _event("pass", name, "attacker.example/forged") if index == 0 else _event("pass", name)
        for index, name in enumerate(gate.REQUIRED_CALPHAD_LEDGER_TESTS)
    )

    _, failures = gate.parse_go_test_json(payload)

    assert any("unexpected package" in failure for failure in failures)


def test_qualification_database_identity_never_records_credentials() -> None:
    identity = gate.qualification_database_identity(
        "postgresql://secret-user:secret-password@db.internal:55432/ultra_qualification"
    )

    assert identity == {
        "scheme": "postgresql",
        "host": "db.internal",
        "port": 55432,
        "database": "ultra_qualification",
        "serving_role": "secret-user",
        "credentials_recorded": False,
    }
    assert "secret-password" not in json.dumps(identity)


def test_qualification_database_pair_requires_distinct_roles_on_exact_target() -> None:
    pair = gate.qualification_database_pair(
        "postgresql://serving:serving-password@db.internal:55432/ultra_qualification",
        "postgresql://migration:migration-password@db.internal:55432/ultra_qualification",
    )
    assert pair["serving_role"] == "serving"
    assert pair["migration_role"] == "migration"
    assert "password" not in json.dumps(pair)

    with pytest.raises(gate.LedgerGateError, match="distinct roles"):
        gate.qualification_database_pair(
            "postgresql://same:one@db.internal:55432/ultra_qualification",
            "postgresql://same:two@db.internal:55432/ultra_qualification",
        )
    with pytest.raises(gate.LedgerGateError, match="same disposable database"):
        gate.qualification_database_pair(
            "postgresql://serving:one@db.internal:55432/ultra_qualification",
            "postgresql://migration:two@other.internal:55432/ultra_qualification",
        )


@pytest.mark.parametrize(
    "database",
    [
        "ultra",
        "production",
        "materials_live",
        "critical",
        "scientific_prod",
        "production_ci",
        "test_production",
        "test-production",
    ],
)
def test_qualification_database_rejects_production_looking_names(database: str) -> None:
    with pytest.raises(gate.LedgerGateError, match="production-looking"):
        gate.qualification_database_identity(f"postgresql://db.internal/{database}")


def test_qualification_database_rejects_invalid_port_without_leaking_url() -> None:
    with pytest.raises(gate.LedgerGateError, match="invalid port") as error:
        gate.qualification_database_identity(
            "postgresql://secret-user:secret-password@db.internal:not-a-port/ultra_qualification"
        )
    assert "secret" not in str(error.value)


def test_write_report_is_content_addressed_and_write_once(tmp_path: Path) -> None:
    report = {
        "schema_version": "1",
        "gate": "calphad-ledger-postgres-qualification",
        "status": "passed",
        "runner": {},
    }
    log = b'{"Action":"pass"}\n'

    first = gate.write_report(tmp_path, report, log)
    second = gate.write_report(tmp_path, report, log)

    assert first == second
    assert first.name.startswith("calphad-ledger-postgres-qualification-")
    stored = json.loads(first.read_text(encoding="utf-8"))
    log_record = stored["runner"]["go_test_log"]
    assert (tmp_path / log_record["path"]).read_bytes() == log
