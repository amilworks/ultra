"""Promotion-mode behavior for scientific domain invariants."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from typing import Any

import pytest

MATERIALS_INVARIANT_EVIDENCE_PROPERTY = "materials_invariant_evidence"


@pytest.fixture
def materials_evidence(
    request: pytest.FixtureRequest,
    record_property: Callable[[str, object], None],
) -> Callable[..., dict[str, Any]]:
    """Attach one canonical scientific-evidence record to a JUnit testcase.

    The promotion runner validates this schema independently and fails closed when
    a required invariant omits, duplicates, or malforms its record. Tests emit the
    record before their assertions so a scientific failure retains the observed
    values that caused it.
    """

    emitted = False

    def emit(
        *,
        validator_id: str,
        observed: Mapping[str, Any],
        expected: Mapping[str, Any],
        tolerance_rationale: str,
        units: str,
        convention: str,
        library_versions: Mapping[str, str],
        passed: bool,
    ) -> dict[str, Any]:
        nonlocal emitted
        if emitted:
            raise AssertionError("a materials invariant may emit exactly one evidence record")
        payload = {
            "schema_version": "1",
            "validator_id": str(validator_id),
            "test_id": request.node.name,
            "required": True,
            "outcome": "pass" if passed else "fail",
            "observed": dict(observed),
            "expected": dict(expected),
            "tolerance_rationale": str(tolerance_rationale),
            "units": str(units),
            "convention": str(convention),
            "library_versions": dict(sorted(library_versions.items())),
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        record_property(MATERIALS_INVARIANT_EVIDENCE_PROPERTY, encoded)
        emitted = True
        return payload

    return emit


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Turn any scientific skip into a failed promotion job.

    Lean PR environments intentionally skip optional scientific stacks. The full
    release sandbox sets ``ULTRA_FAIL_ON_DOMAIN_SKIP=1`` and must prove that the
    checks actually ran.
    """

    if os.getenv("ULTRA_FAIL_ON_DOMAIN_SKIP") != "1":
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = [] if reporter is None else reporter.stats.get("skipped", [])
    if skipped and session.exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
