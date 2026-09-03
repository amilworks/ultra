"""Synthetic OME-NGFF sensor-data corpus + stress harness for the Ultra image/data service.

The corpus stands behind the app's "multimodal sensor data" claim: one spec-correct
OME-Zarr store per real STEM sensor modality (materials, biology, environmental, medical,
geophysics, astronomy, and non-image sensor streams), plus an adversarial/malformed set
and a set of scale probes. The harness runs every store through the application reader,
renderer, viewer-info builder, and the live FastAPI service and asserts invariants.
"""

from __future__ import annotations

from .specs import ChannelSpec, StoreSpec, catalog
from .writer import build_corpus, write_store

__all__ = ["ChannelSpec", "StoreSpec", "build_corpus", "catalog", "write_store"]
