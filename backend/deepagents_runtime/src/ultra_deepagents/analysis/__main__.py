"""Entrypoint for the batch model-inference worker.

Runs the Data-Agent NATS worker with the AnalysisProcessor, which handles analysis.megaseg
(and, later, analysis.rarespot) jobs and falls through to the default processor for other
data-agent job types. Deploy this on the compute node:

    python -m ultra_deepagents.analysis

Environment: ULTRA_DATA_AGENT_NATS_URL, ULTRA_CONTROL_BASE_URL, MEGASEG_SERVICE_URL,
MEGASEG_SERVICE_API_KEY, ULTRA_CONTROL_UPLOAD_ROOT, ULTRA_ANALYSIS_UPLOAD_ROOTS.
"""

from __future__ import annotations

import asyncio
import logging

from ..data_agent.worker import NATSDataAgentWorker
from .processor import AnalysisProcessor


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    worker = NATSDataAgentWorker(processor=AnalysisProcessor())
    asyncio.run(worker.run_forever())


if __name__ == "__main__":
    main()
