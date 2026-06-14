"""Performance harness for the image engine.

Measures per-operation latency (median / p95 / min), output size, and a coarse
process memory high-water mark across the operations that back the V2 endpoints.
Runs against any :class:`~ultra_deepagents.imaging.engine.ImageEngine` — the stub
(to validate the harness) or the real ``LibBioImageEngine`` (for real numbers).

Honesty notes baked into the report:

- ``arch``/``emulated`` are recorded; emulated amd64-on-arm64 numbers are
  functional checks only. Authoritative numbers come from the Linux deploy target.
- ``py_peak_kb`` is Python-side (``tracemalloc``) only and under-reports the
  native engine's C++ allocations; ``max_rss_kb`` is a process high-water mark.
  For authoritative per-op peak RSS of the native engine, run each op in an
  isolated subprocess (a follow-on; latency is the headline metric here).

CLI::

    python -m ultra_deepagents.imaging.benchmark --engine stub --path any
    python -m ultra_deepagents.imaging.benchmark --engine lib --path /data/slide.czi --json out.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
import tracemalloc
from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass
class OpResult:
    op: str
    iterations: int
    ok: bool
    ms_median: float
    ms_p95: float
    ms_min: float
    bytes_out: int
    py_peak_kb: float
    error: str | None = None


def _measure(fn: Callable[[], Any], iterations: int) -> OpResult:
    times: list[float] = []
    out: Any = None
    tracemalloc.start()
    try:
        for _ in range(iterations):
            t0 = time.perf_counter()
            out = fn()
            times.append((time.perf_counter() - t0) * 1000.0)
    except Exception as exc:  # noqa: BLE001 - record and continue
        tracemalloc.stop()
        return OpResult("", iterations=0, ok=False, ms_median=float("nan"), ms_p95=float("nan"),
                        ms_min=float("nan"), bytes_out=0, py_peak_kb=0.0, error=repr(exc))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    times.sort()
    p95 = times[min(len(times) - 1, int(len(times) * 0.95))]
    if isinstance(out, (bytes, bytearray)):
        size = len(out)
    elif isinstance(out, (dict, list)):
        size = len(json.dumps(out).encode("utf-8"))
    else:
        size = 0
    return OpResult("", iterations=iterations, ok=True, ms_median=statistics.median(times),
                    ms_p95=p95, ms_min=times[0], bytes_out=size, py_peak_kb=peak / 1024.0)


def default_ops(engine: Any, path: str) -> dict[str, Callable[[], Any]]:
    """The operation set that backs the V2 image endpoints."""
    return {
        "meta": lambda: engine.meta(path),
        "tile_l0": lambda: engine.tile(path, level=0, col=0, row=0, tile_size=512),
        "tile_l2": lambda: engine.tile(path, level=2, col=0, row=0, tile_size=512),
        "region_half": lambda: engine.region(path, x1=0, y1=0, x2=1024, y2=1024, region_scale=0.5),
        "slice_z0": lambda: engine.slice_plane(path, z=0),
        "thumb_zscrub": lambda: engine.thumbnail(path, max_size=256, z=0),
        "atlas_l2": lambda: engine.atlas(path, level=2),
        "histogram": lambda: engine.histogram(path, bins=256),
    }


def _max_rss_kb() -> float:
    try:
        import resource  # POSIX only

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports kilobytes.
        return rss / 1024.0 if platform.system() == "Darwin" else float(rss)
    except Exception:  # pragma: no cover
        return 0.0


def _emulated() -> bool:
    # amd64 binary on Apple Silicon -> emulated; flag it so numbers aren't trusted as authoritative.
    return platform.machine() in ("arm64", "aarch64") and platform.system() == "Darwin"


def run_benchmark(engine: Any, path: str, *, iterations: int = 20,
                  ops: dict[str, Callable[[], Any]] | None = None) -> dict[str, Any]:
    ops = ops or default_ops(engine, path)
    results: list[OpResult] = []
    for name, fn in ops.items():
        r = _measure(fn, iterations)
        r.op = name
        results.append(r)
    return {
        "engine": type(engine).__name__,
        "arch": platform.machine(),
        "platform": platform.platform(),
        "emulated_guess": _emulated(),
        "path": path,
        "iterations": iterations,
        "max_rss_kb": _max_rss_kb(),
        "results": [asdict(r) for r in results],
    }


def format_table(report: dict[str, Any]) -> str:
    lines = [
        f"engine={report['engine']} arch={report['arch']} "
        f"emulated={report['emulated_guess']} iters={report['iterations']}",
        f"{'op':<14}{'ok':<4}{'median_ms':>11}{'p95_ms':>9}{'min_ms':>9}{'bytes':>10}{'py_peak_kb':>12}",
    ]
    for r in report["results"]:
        if r["ok"]:
            lines.append(
                f"{r['op']:<14}{'y':<4}{r['ms_median']:>11.3f}{r['ms_p95']:>9.3f}"
                f"{r['ms_min']:>9.3f}{r['bytes_out']:>10}{r['py_peak_kb']:>12.1f}"
            )
        else:
            lines.append(f"{r['op']:<14}{'N':<4}  ERROR: {r['error']}")
    lines.append(f"process max_rss_kb={report['max_rss_kb']:.0f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ultra_deepagents.imaging.benchmark",
        description="Benchmark the image engine operations.",
    )
    parser.add_argument("--engine", choices=["stub", "lib"], default="stub")
    parser.add_argument("--path", required=True, help="Image path (any string for --engine stub).")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--json", dest="json_out", default=None, help="Write the report JSON here.")
    args = parser.parse_args()

    from ultra_deepagents.imaging.engine import build_engine

    engine = build_engine(prefer_real=(args.engine == "lib"))
    report = run_benchmark(engine, args.path, iterations=args.iterations)
    print(format_table(report))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
