"""Promotion-mode behavior for scientific domain invariants."""

from __future__ import annotations

import os

import pytest


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
