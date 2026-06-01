from __future__ import annotations

import argparse
import asyncio

from ultra_deepagents.rarespot.worker import run_worker_loop


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ultra_deepagents.rarespot_worker",
        description="Run the single-concurrency RareSpot ecology inference worker.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        choices=[1],
        help="Worker concurrency. Production v1 is intentionally fixed at 1.",
    )
    parser.parse_args()
    asyncio.run(run_worker_loop())


if __name__ == "__main__":
    main()
