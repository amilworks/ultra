"""CLI entry for the native OME-NGFF (OME-Zarr) serving microservice.

Run with::

    python -m ultra_deepagents.ngff_service --host 0.0.0.0 --port 8091

Stateless: reads zarr chunks from paths the control plane resolves + authorizes. Scales
with uvicorn workers (processes) and FastAPI's threadpool (zarr/blosc release the GIL).
``--workers`` defaults to ULTRA_NGFF_WORKERS or CPU count.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ultra_deepagents.ngff_service",
        description="Run the native OME-Zarr (NGFF) serving microservice.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("ULTRA_NGFF_PORT", "8091")))
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("ULTRA_NGFF_WORKERS", "0")) or None,
        help="uvicorn worker processes (default: ULTRA_NGFF_WORKERS or CPU count).",
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise SystemExit("ngff service requires uvicorn") from exc

    workers = args.workers or (os.cpu_count() or 4)
    if workers and workers > 1:
        # Multiple worker processes need an import string (uvicorn re-imports per worker).
        uvicorn.run("ultra_deepagents.ngff_service:app", host=args.host, port=args.port, workers=workers)
    else:
        from ultra_deepagents.ngff.service import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)


# Module-level app for the multi-worker (import-string) path.
def _build_app():
    from ultra_deepagents.ngff.service import create_app

    return create_app()


app = _build_app()


if __name__ == "__main__":
    main()
