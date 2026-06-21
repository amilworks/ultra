"""Batch model-inference worker (MegaSeg / RareSpot).

Runs as the Data-Agent job worker for analysis.* job types: it processes a batch of
images one at a time, calls the stateless GPU service (MegaSeg) or runs CPU inference
(RareSpot), writes each result into the shared upload root, and registers it with the
control plane so results land on the Resources page grouped + downloadable. Per-image
failures are isolated and the batch is resumable after a worker restart.
"""

from .config import AnalysisSettings
from .processor import AnalysisProcessor

__all__ = ["AnalysisSettings", "AnalysisProcessor"]
