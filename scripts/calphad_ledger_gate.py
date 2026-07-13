#!/usr/bin/env python3
"""Qualify the append-only CALPHAD ledger against dedicated PostgreSQL.

This runner is intentionally unsafe for a production database: the Go contract
test applies the current schema and writes uniquely named qualification rows.
The database name must therefore clearly identify a test/CI/qualification
database, and the operator must opt in explicitly.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from materials_readiness_gate import (
    REQUIRED_CALPHAD_LEDGER_INVARIANTS,
    REQUIRED_CALPHAD_LEDGER_SOURCE_FILES,
    REQUIRED_CALPHAD_LEDGER_TESTS,
    manifest_hash,
)

SCHEMA_VERSION = "1"
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DATABASE_NAME_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$", re.I)
QUALIFICATION_DATABASE_TOKENS = frozenset({"test", "testing", "ci", "qualification", "sandbox"})
FORBIDDEN_DATABASE_TOKENS = frozenset({"prod", "production", "live", "primary", "critical"})
GO_TEST_PATTERN = "^(" + "|".join(re.escape(name) for name in REQUIRED_CALPHAD_LEDGER_TESTS) + ")$"
MAX_GO_TEST_LOG_BYTES = 32 * 1024 * 1024
POSTGRES_TEST = "TestPostgresStoreCalphadLedgerIsAppendOnlyTenantScopedAndContentBound"
HTTP_TEST = "TestCalphadGovernanceHTTPIsOwnerReadableAndWorkerWritable"
STORE_GO_PACKAGE = "github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
HTTP_GO_PACKAGE = "github.com/amilworks/bisque-ultra/backend/controlplane/internal/httpapi"
EXPECTED_GO_TEST_PACKAGES = {
    name: (
        STORE_GO_PACKAGE
        if name.startswith("TestPostgresStore") or name.startswith("TestCalphadLedgerSchema")
        else HTTP_GO_PACKAGE
    )
    for name in REQUIRED_CALPHAD_LEDGER_TESTS
}
POSTGRES_IDENTITY_MARKER = "CALPHAD_POSTGRES_IDENTITY "
POSTGRES_INVARIANT_TEST_EVIDENCE: dict[str, tuple[str, ...]] = {
    "append_only_update_delete": tuple(
        f"{POSTGRES_TEST}/{suffix}"
        for suffix in (
            "append_only_revision_update",
            "append_only_revision_delete",
            "append_only_validation_update",
            "append_only_validation_delete",
            "append_only_evidence_update",
            "append_only_evidence_delete",
        )
    ),
    "append_only_truncate": tuple(
        f"{POSTGRES_TEST}/{suffix}"
        for suffix in (
            "append_only_revision_truncate",
            "append_only_validation_truncate",
            "append_only_evidence_truncate",
        )
    ),
    "database_bytes_revision_bound": tuple(
        f"{POSTGRES_TEST}/{suffix}"
        for suffix in ("database_revision_binding", "database_digest_binding")
    ),
    "evidence_bytes_server_verified": (
        f"{POSTGRES_TEST}/evidence_blob_content_bound",
        HTTP_TEST,
    ),
    "immutable_runtime_image_required": (
        f"{POSTGRES_TEST}/immutable_runtime_image",
        f"{POSTGRES_TEST}/runtime_policy_authorized",
    ),
    "retry_idempotent": (f"{POSTGRES_TEST}/retry_idempotent",),
    "multiple_equilibria_idempotent": (f"{POSTGRES_TEST}/multiple_equilibria_idempotent",),
    "run_lease_authorized": (f"{POSTGRES_TEST}/run_lease_authorized",),
    "tenant_scoped": (f"{POSTGRES_TEST}/parent_same_tenant",),
    "inspection_lineage_bound": (f"{POSTGRES_TEST}/inspection_lineage_required",),
    "inspection_inventory_bound": (f"{POSTGRES_TEST}/inspection_inventory_bound",),
    "retained_evidence_contract_bound": (f"{POSTGRES_TEST}/evidence_blob_content_bound",),
    "retained_failure_evidence": (
        f"{POSTGRES_TEST}/retained_terminal_statuses",
        "TestVerifyCalphadFailureEvidenceAcceptsOnlyExactBoundedTerminalTuples",
    ),
    "retained_timeout_evidence": (
        f"{POSTGRES_TEST}/retained_terminal_statuses",
        HTTP_TEST,
    ),
    "retained_unsupported_evidence": (
        f"{POSTGRES_TEST}/retained_terminal_statuses",
        "TestVerifyCalphadFailureEvidenceAcceptsOnlyExactBoundedTerminalTuples",
    ),
    "terminal_failure_nonpromotable": (
        f"{POSTGRES_TEST}/retained_terminal_statuses",
        HTTP_TEST,
    ),
    "schema_fingerprint_verified": (f"{POSTGRES_TEST}/schema_fingerprint_verified",),
    "trigger_search_path_pinned": (
        f"{POSTGRES_TEST}/temporary_schema_guarded",
        f"{POSTGRES_TEST}/schema_fingerprint_verified",
    ),
    "serving_role_separated": (f"{POSTGRES_TEST}/serving_role_separated",),
    "public_and_unexpected_acl_grantees_rejected": (
        f"{POSTGRES_TEST}/public_and_unexpected_acl_grantees_rejected",
    ),
    "unexpected_writer_overload_revoked_and_rejected": (
        f"{POSTGRES_TEST}/unexpected_writer_overload_revoked_and_rejected",
    ),
    "equilibrium_reads_require_retained_inspection_event": (
        f"{POSTGRES_TEST}/equilibrium_reads_require_retained_inspection_event",
    ),
}


class LedgerGateError(RuntimeError):
    """The PostgreSQL ledger qualification could not produce trusted evidence."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def qualification_database_identity(dsn: str) -> dict[str, Any]:
    text = str(dsn or "").strip()
    if not text:
        raise LedgerGateError("ULTRA_CONTROL_TEST_DATABASE_URL is required")
    parsed = urlsplit(text)
    database = parsed.path.lstrip("/").split("/", 1)[0]
    try:
        port = parsed.port or 5432
    except ValueError as exc:
        raise LedgerGateError("qualification database URL has an invalid port") from exc
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname or not database:
        raise LedgerGateError("qualification database URL is not a valid PostgreSQL DSN")
    tokens = {token.lower() for token in database.split("_")}
    if (
        DATABASE_NAME_RE.fullmatch(database) is None
        or not (tokens & QUALIFICATION_DATABASE_TOKENS)
        or bool(tokens & FORBIDDEN_DATABASE_TOKENS)
    ):
        raise LedgerGateError(
            "refusing CALPHAD ledger qualification against a production-looking database name"
        )
    return {
        "scheme": "postgresql",
        "host": parsed.hostname,
        "port": port,
        "database": database,
        "serving_role": unquote(parsed.username or "").strip(),
        "credentials_recorded": False,
    }


def qualification_database_pair(serving_dsn: str, migration_dsn: str) -> dict[str, Any]:
    serving = qualification_database_identity(serving_dsn)
    serving_role = str(serving.get("serving_role") or "").strip()
    if not serving_role:
        raise LedgerGateError("qualification database URL must name the non-owner serving role")
    migration = qualification_database_identity(migration_dsn)
    migration_role = str(migration.get("serving_role") or "").strip()
    if not migration_role or migration_role == serving_role:
        raise LedgerGateError("qualification serving and migration URLs must use distinct roles")
    for field in ("scheme", "host", "port", "database"):
        if migration.get(field) != serving.get(field):
            raise LedgerGateError(
                "qualification serving and migration URLs must target the same disposable database"
            )
    serving["migration_role"] = migration_role
    return serving


def inspect_clean_repository(root: Path, expected_git_sha: str) -> dict[str, Any]:
    expected = str(expected_git_sha or "").strip().lower()
    if GIT_SHA_RE.fullmatch(expected) is None:
        raise LedgerGateError("expected Git SHA must be 40 lowercase hexadecimal characters")
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    observed = revision.stdout.strip().lower()
    if revision.returncode != 0 or status.returncode != 0:
        raise LedgerGateError("could not inspect the qualification repository")
    if observed != expected:
        raise LedgerGateError(f"qualification Git SHA {observed or '<missing>'} != {expected}")
    if status.stdout.strip():
        raise LedgerGateError("CALPHAD ledger qualification requires a clean Git checkout")
    return {"git_sha": observed, "clean": True}


def build_source_manifest(root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for relative in REQUIRED_CALPHAD_LEDGER_SOURCE_FILES:
        path = root / relative
        try:
            if path.is_symlink() or not path.is_file():
                raise OSError("not a regular file")
            digest = sha256_file(path)
            size = path.stat().st_size
        except OSError as exc:
            raise LedgerGateError(
                f"required CALPHAD ledger source is unavailable: {relative}"
            ) from exc
        hashes[relative] = digest
        entries.append({"path": relative, "sha256": digest, "size_bytes": size})
    return {
        "file_count": len(entries),
        "aggregate_sha256": manifest_hash(hashes),
        "files": entries,
    }


def parse_go_test_json(output: bytes) -> tuple[list[dict[str, Any]], list[str]]:
    if not output or len(output) > MAX_GO_TEST_LOG_BYTES:
        raise LedgerGateError("Go test log is empty or exceeds its fixed evidence bound")
    outcomes: dict[str, str] = {}
    packages: dict[str, str] = {}
    failures: list[str] = []
    for line_number, raw_line in enumerate(output.splitlines(), start=1):
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise LedgerGateError(f"Go test log line {line_number} is not JSON") from exc
        if not isinstance(event, dict):
            raise LedgerGateError(f"Go test log line {line_number} is not an object")
        test_name = str(event.get("Test") or "")
        action = str(event.get("Action") or "")
        package = str(event.get("Package") or "")
        if test_name in REQUIRED_CALPHAD_LEDGER_TESTS and action in {"pass", "fail", "skip"}:
            outcomes[test_name] = action
            packages[test_name] = package
        if not test_name and action == "fail" and package:
            failures.append(f"Go package failed: {package}")
    records = [
        {
            "name": name,
            "package": packages.get(name, ""),
            "passed": outcomes.get(name) == "pass",
            "skipped": outcomes.get(name) == "skip",
        }
        for name in REQUIRED_CALPHAD_LEDGER_TESTS
    ]
    for record in records:
        expected_package = EXPECTED_GO_TEST_PACKAGES[record["name"]]
        if record["package"] != expected_package:
            failures.append(
                f"required CALPHAD ledger test came from unexpected package: "
                f"{record['name']} ({record['package'] or 'missing'} != {expected_package})"
            )
        if not record["passed"]:
            failures.append(
                f"required CALPHAD ledger test did not pass: {record['name']} "
                f"({outcomes.get(record['name'], 'missing')})"
            )
    return records, failures


def parse_postgres_invariant_evidence(
    output: bytes,
    *,
    expected_database: Mapping[str, Any],
    expected_role: str,
) -> tuple[dict[str, bool], list[dict[str, Any]], dict[str, Any], list[str]]:
    """Bind each claimed PostgreSQL invariant to exact passing Go test events."""

    if not output or len(output) > MAX_GO_TEST_LOG_BYTES:
        raise LedgerGateError("Go test log is empty or exceeds its fixed evidence bound")
    outcomes: dict[str, str] = {}
    packages: dict[str, str] = {}
    observed_identity: dict[str, Any] = {}
    observed_identity_package = ""
    postgres_test_output: list[str] = []
    failures: list[str] = []
    for line_number, raw_line in enumerate(output.splitlines(), start=1):
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise LedgerGateError(f"Go test log line {line_number} is not JSON") from exc
        if not isinstance(event, dict):
            raise LedgerGateError(f"Go test log line {line_number} is not an object")
        test_name = str(event.get("Test") or "")
        action = str(event.get("Action") or "")
        if test_name and action in {"pass", "fail", "skip"}:
            outcomes[test_name] = action
            packages[test_name] = str(event.get("Package") or "")
        if (
            test_name == POSTGRES_TEST
            and action == "output"
            and str(event.get("Package") or "") == STORE_GO_PACKAGE
        ):
            postgres_test_output.append(str(event.get("Output") or ""))

    # `go test -json` may split one long t.Log line across adjacent output
    # events. Reassemble only the exact PostgreSQL identity test's output before
    # parsing the single, line-bounded evidence object.
    postgres_output = "".join(postgres_test_output)
    marker_count = postgres_output.count(POSTGRES_IDENTITY_MARKER)
    if marker_count > 1:
        failures.append("PostgreSQL identity test output contains duplicate evidence")
    elif marker_count == 1:
        encoded = postgres_output.split(POSTGRES_IDENTITY_MARKER, 1)[1].splitlines()[0].strip()
        try:
            parsed = json.loads(encoded)
        except json.JSONDecodeError:
            failures.append("PostgreSQL identity test output is not valid JSON")
        else:
            if isinstance(parsed, dict):
                observed_identity = parsed
                observed_identity_package = STORE_GO_PACKAGE
            else:
                failures.append("PostgreSQL identity test output is not an object")

    invariant_records: list[dict[str, Any]] = []
    invariant_outcomes: dict[str, bool] = {}
    for invariant in REQUIRED_CALPHAD_LEDGER_INVARIANTS:
        tests = POSTGRES_INVARIANT_TEST_EVIDENCE.get(invariant, ())
        passed = bool(tests) and all(
            outcomes.get(test) == "pass"
            and packages.get(test) == EXPECTED_GO_TEST_PACKAGES.get(test.split("/", 1)[0], "")
            for test in tests
        )
        invariant_outcomes[invariant] = passed
        invariant_records.append(
            {
                "name": invariant,
                "passed": passed,
                "test_evidence": [
                    {
                        "name": test,
                        "outcome": outcomes.get(test, "missing"),
                    }
                    for test in tests
                ],
            }
        )
        if not passed:
            failures.append(f"PostgreSQL invariant lacks passing test evidence: {invariant}")

    expected_identity_keys = {
        "database",
        "server_address",
        "server_port",
        "role",
        "transaction_read_only",
        "role_superuser",
        "role_create_role",
        "role_create_database",
        "role_replication",
        "role_bypass_rls",
        "calphad_owned_tables",
        "calphad_owned_functions",
        "calphad_owner_roles",
        "calphad_reachable_roles",
        "calphad_owner_role_reachable",
        "public_schema_owner",
        "public_owner_role_reachable",
        "can_create_public_schema",
        "calphad_select_all",
        "calphad_insert_all",
        "calphad_insert_any",
        "calphad_execute_create_revision",
        "calphad_execute_append_validation",
        "calphad_writer_functions_exact",
        "calphad_execute_unexpected_writer",
        "calphad_execute_internal",
        "calphad_public_execute",
        "calphad_unexpected_table_acl_grantees",
        "calphad_unexpected_function_acl_grantees",
        "calphad_mutation_privilege",
        "connection_target_host",
        "connection_target_port",
    }
    identity_valid = all(
        (
            set(observed_identity) == expected_identity_keys,
            observed_identity.get("database") == expected_database.get("database"),
            isinstance(observed_identity.get("server_port"), int)
            and not isinstance(observed_identity.get("server_port"), bool)
            and 0 <= int(observed_identity.get("server_port", -1)) <= 65535,
            bool(str(observed_identity.get("server_address") or "").strip()),
            observed_identity.get("role") == expected_role,
            observed_identity.get("transaction_read_only") == "off",
            observed_identity.get("role_superuser") is False,
            observed_identity.get("role_create_role") is False,
            observed_identity.get("role_create_database") is False,
            observed_identity.get("role_replication") is False,
            observed_identity.get("role_bypass_rls") is False,
            observed_identity.get("calphad_owned_tables") == [],
            observed_identity.get("calphad_owned_functions") == [],
            observed_identity.get("calphad_owner_roles")
            == [expected_database.get("migration_role")],
            observed_identity.get("calphad_reachable_roles") == [],
            observed_identity.get("calphad_owner_role_reachable") is False,
            bool(str(observed_identity.get("public_schema_owner") or "").strip()),
            observed_identity.get("public_owner_role_reachable") is False,
            observed_identity.get("can_create_public_schema") is False,
            observed_identity.get("calphad_select_all") is True,
            observed_identity.get("calphad_insert_all") is False,
            observed_identity.get("calphad_insert_any") is False,
            observed_identity.get("calphad_execute_create_revision") is True,
            observed_identity.get("calphad_execute_append_validation") is True,
            observed_identity.get("calphad_writer_functions_exact") is True,
            observed_identity.get("calphad_execute_unexpected_writer") is False,
            observed_identity.get("calphad_execute_internal") is False,
            observed_identity.get("calphad_public_execute") is False,
            observed_identity.get("calphad_unexpected_table_acl_grantees") == [],
            observed_identity.get("calphad_unexpected_function_acl_grantees") == [],
            observed_identity.get("calphad_mutation_privilege") is False,
            observed_identity.get("connection_target_host") == expected_database.get("host"),
            observed_identity.get("connection_target_port") == expected_database.get("port"),
            observed_identity_package == STORE_GO_PACKAGE,
        )
    )
    if not identity_valid:
        failures.append(
            "PostgreSQL test output does not prove the connected writable database identity"
        )
    return invariant_outcomes, invariant_records, observed_identity, failures


CommandRunner = Callable[..., subprocess.CompletedProcess[bytes]]


def run_gate(
    *,
    repository_root: Path,
    expected_git_sha: str,
    database_url: str,
    migration_database_url: str,
    qualification_database_confirmed: bool,
    command_runner: CommandRunner = subprocess.run,
) -> tuple[dict[str, Any], bytes]:
    root = repository_root.expanduser().resolve()
    if not qualification_database_confirmed:
        raise LedgerGateError("--qualification-database-confirmed is required")
    database = qualification_database_pair(database_url, migration_database_url)
    expected_role = str(database["serving_role"])
    repository = inspect_clean_repository(root, expected_git_sha)
    source_manifest = build_source_manifest(root)
    command = (
        "go",
        "test",
        "-json",
        "-count=1",
        "./internal/store",
        "./internal/httpapi",
        "-run",
        GO_TEST_PATTERN,
    )
    environment = os.environ.copy()
    environment["ULTRA_CONTROL_TEST_DATABASE_URL"] = database_url
    try:
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = command_runner(
                command,
                cwd=root / "backend/controlplane",
                env=environment,
                stdout=stdout_file,
                stderr=stderr_file,
                check=False,
                timeout=300,
            )
            stdout_size = stdout_file.tell()
            stderr_size = stderr_file.tell()
            if stdout_size > MAX_GO_TEST_LOG_BYTES or stderr_size > MAX_GO_TEST_LOG_BYTES:
                raise LedgerGateError("Go qualification output exceeds its fixed evidence bound")
            stdout_file.seek(0)
            stdout = stdout_file.read(MAX_GO_TEST_LOG_BYTES + 1)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LedgerGateError("CALPHAD ledger Go qualification could not complete") from exc
    tests, failures = parse_go_test_json(stdout)
    invariant_outcomes, invariant_records, observed_database, invariant_failures = (
        parse_postgres_invariant_evidence(
            stdout, expected_database=database, expected_role=expected_role
        )
    )
    failures.extend(invariant_failures)
    if process.returncode != 0:
        failures.append(f"Go qualification exited {process.returncode}")
    passed = len(failures) == 0
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate": "calphad-ledger-postgres-qualification",
        "generated_at_utc": utc_now(),
        "status": "passed" if passed else "failed",
        "qualification_database": True,
        "production_database_used": False,
        "database": database,
        "observed_database": observed_database,
        "git_sha": repository["git_sha"],
        "repository_clean": repository["clean"],
        "source_manifest": source_manifest,
        "tests": tests,
        "summary": {
            "passed": sum(record["passed"] for record in tests),
            "failed": sum(not record["passed"] and not record["skipped"] for record in tests),
            "skipped": sum(record["skipped"] for record in tests),
        },
        "postgres_invariants": invariant_outcomes,
        "postgres_invariant_evidence": invariant_records,
        "runner": {
            "command": list(command),
            "database_credentials_recorded": False,
        },
        "failures": failures,
    }
    return report, stdout


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise LedgerGateError(f"refusing to replace content-addressed evidence: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise LedgerGateError(f"content-addressed evidence collision: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def write_report(output_dir: Path, report: Mapping[str, Any], go_log: bytes) -> Path:
    root = output_dir.expanduser().resolve()
    log_digest = sha256_bytes(go_log)
    log_path = root / f"calphad-ledger-go-test-{log_digest}.jsonl"
    write_once(log_path, go_log)
    complete = dict(report)
    complete["runner"] = {
        **dict(complete.get("runner") or {}),
        "go_test_log": {
            "path": log_path.name,
            "sha256": log_digest,
            "size_bytes": len(go_log),
        },
    }
    payload = canonical_json_bytes(complete)
    digest = sha256_bytes(payload)
    report_path = root / f"calphad-ledger-postgres-qualification-{digest}.json"
    write_once(report_path, payload)
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--qualification-database-confirmed", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report, go_log = run_gate(
        repository_root=args.repository_root,
        expected_git_sha=args.expected_git_sha,
        database_url=os.environ.get("ULTRA_CONTROL_TEST_DATABASE_URL", ""),
        migration_database_url=os.environ.get("ULTRA_CONTROL_TEST_MIGRATION_DATABASE_URL", ""),
        qualification_database_confirmed=args.qualification_database_confirmed,
    )
    path = write_report(args.output_dir, report, go_log)
    print(json.dumps({"status": report["status"], "report": str(path)}, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LedgerGateError as exc:
        print(f"CALPHAD ledger qualification error: {exc}", file=os.sys.stderr)
        raise SystemExit(2) from exc
