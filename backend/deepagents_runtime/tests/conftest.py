"""Shared pytest fixtures for the deepagents runtime suite.

The materials-science platform is OFF by default in production
(``ULTRA_DEEPAGENTS_MATERIALS_ENABLED``); the materials test modules exercise
its behaviour, so enable it for them. Every other test runs with the production
default (materials off), which is exactly the state we want non-materials tests
to verify.
"""

from __future__ import annotations

import pytest

_MATERIALS_TEST_MARKERS = (
    "materials",
    "calphad",
    "kinetics",
    "crystal_plasticity",
    "degradation",
    "characterization",
    "sensor",
    "mattools",
    "live_trace",
)


@pytest.fixture(autouse=True)
def _materials_platform_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = str(getattr(request.node, "fspath", "")).lower()
    if any(marker in module_path for marker in _MATERIALS_TEST_MARKERS):
        monkeypatch.setenv("ULTRA_DEEPAGENTS_MATERIALS_ENABLED", "true")
    else:
        monkeypatch.delenv("ULTRA_DEEPAGENTS_MATERIALS_ENABLED", raising=False)
