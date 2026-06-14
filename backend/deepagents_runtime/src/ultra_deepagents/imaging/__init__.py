"""libbioimage-backed image engine for BisQue Ultra.

This package wraps the ViQi/BisQue ``libbioimage`` (``imgcnv``) engine as the
single decode + convert backend behind the V2 image endpoints. It provides:

- :mod:`~ultra_deepagents.imaging.pipelines` — pure pipeline-string builders
  encoding the engine's read-pipeline grammar (tile/region/slice/atlas/...).
- :mod:`~ultra_deepagents.imaging.engine` — an :class:`ImageEngine` protocol with
  a ``LibBioImageEngine`` (real, requires ``libimgcnv.so``) and a ``StubEngine``
  (deterministic, dependency-light) for tests and CI without the native lib.

See ``planning/2026-06-11-libbioimage-engine-understanding.md`` for the engine
contract and ``planning/2026-06-11-libbioimage-image-engine-design.md`` for how
this fits the platform.
"""

from __future__ import annotations

from ultra_deepagents.imaging import pipelines

__all__ = ["pipelines"]
