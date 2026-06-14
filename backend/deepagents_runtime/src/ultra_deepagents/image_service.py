"""CLI entry for the libbioimage-backed image service.

Run with::

    python -m ultra_deepagents.image_service --host 0.0.0.0 --port 8090

Use ``--stub`` to force the deterministic stub engine (no native lib required).
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ultra_deepagents.image_service",
        description="Run the libbioimage-backed image service.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--workers", type=int, default=None, help="Process-pool size (default: CPU count).")
    parser.add_argument("--stub", action="store_true", help="Force the deterministic stub engine.")
    args = parser.parse_args()

    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise SystemExit("image service requires uvicorn (install the 'imaging' extra)") from exc

    from ultra_deepagents.imaging.pool import ImagePool
    from ultra_deepagents.imaging.service import create_app

    pool = ImagePool(workers=args.workers, prefer_real=not args.stub)
    app = create_app(runner=pool)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
