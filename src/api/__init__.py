"""FastAPI API layer for Bisque Ultra."""

from __future__ import annotations

from typing import Any

from src.api.client import OrchestratorClient

__all__ = ["app", "create_app", "OrchestratorClient"]


def __getattr__(name: str) -> Any:
    if name == "app":
        from src.api.main import app

        return app
    if name == "create_app":
        from src.api.main import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
