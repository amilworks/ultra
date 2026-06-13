"""CLI entry for the image.derive_pyramid convert worker.

Run with::

    python -m ultra_deepagents.image_convert_worker
"""

from __future__ import annotations

import argparse
import asyncio


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ultra_deepagents.image_convert_worker",
        description="Run the image.derive_pyramid convert worker (source -> tiled pyramid).",
    )
    parser.add_argument("--nats-url", default=None, help="NATS URL (default: env ULTRA_CONTROL_NATS_URL).")
    parser.add_argument("--subject", default=None, help="Jobs subject (default: env or ultra.image.jobs).")
    parser.add_argument("--durable", default=None, help="Durable consumer name.")
    args = parser.parse_args()

    from ultra_deepagents.imaging.worker import run_worker_loop

    asyncio.run(run_worker_loop(nats_url=args.nats_url, subject=args.subject, durable=args.durable))


if __name__ == "__main__":
    main()
