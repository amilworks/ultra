"""The Lens ``scene3d`` derive: 3D Gaussian splats and point maps into streamable chunks.

The wire contract is frozen in ``backend/contracts/scene3d/CONTRACT.md``; this package is the Python
half of it. The pipeline is deliberately split so each half can be tested without the
other:

- :mod:`~ultra_deepagents.scene3d.ply` — header-driven streaming PLY reader. Every
  property offset is derived from the header, because real writers disagree about the
  record layout (Postshot omits normals: 236 B/splat, not INRIA's 248 B).
- :mod:`~ultra_deepagents.scene3d.spark_encode` — the ``USX1``/``UPC1`` payloads. ``USX1``
  is byte-for-byte what Spark's ``encodeExtSplat`` writes.
- :mod:`~ultra_deepagents.scene3d.chunker` — octree partition, importance ordering, and
  the density tiers. The partition is exact; nothing is dropped.
- :mod:`~ultra_deepagents.scene3d.manifest` — the ``ultra.scene3d.v1`` document, including
  the ``limitations`` sentences that state what the viewer is not doing.
- :mod:`~ultra_deepagents.scene3d.poster` — a CPU orthographic thumbnail for the
  Resources grid, explicitly not the WebGL render.
- :mod:`~ultra_deepagents.scene3d.job` — the ``scene.derive`` runner the NATS worker wraps.
"""

from __future__ import annotations

from ultra_deepagents.scene3d import chunker, manifest, ply, poster, spark_encode

__all__ = ["chunker", "manifest", "ply", "poster", "spark_encode"]
