"""Compatibility package for legacy imports.

The active chat runtime now lives under ``src.agno_backend``.
Keep this package lightweight so submodule imports like ``src.agents.contracts``
do not accidentally pull the runtime into unrelated modules.
"""

__all__: list[str] = []
