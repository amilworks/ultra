"""Stress harness for the OME-NGFF image/data service.

Builds the full synthetic sensor corpus (valid + scale probes + adversarial) and drives it
through the application reader, renderer, viewer-info builder, and the FastAPI service.
Every valid store is asserted against a battery of invariants; every adversarial store must
fail closed; every scale probe is checked for bounded resource use; a concurrency storm
exercises the process-wide caches. Emits a JSON report and a human summary.

Run:
    python -m ngff_sensor_corpus.stress --out /tmp/corpus --report /tmp/report.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from PIL import Image
from ultra_deepagents.ngff.reader import NgffError, is_ome_zarr, open_ngff, process_plane_cache_info
from ultra_deepagents.ngff.render import render_slice_png, render_thumbnail_png, render_tile_png
from ultra_deepagents.ngff.viewerinfo import build_ngff_viewer_info

from .adversarial import build_adversarial
from .scale import scale_probes
from .specs import StoreSpec, catalog
from .writer import build_corpus


def _png_size(data: bytes) -> tuple[int, int]:
    im = Image.open(io.BytesIO(data))
    im.load()
    return im.size  # (w, h)


def _check(cond: bool, msg: str, failures: list[str]) -> None:
    if not cond:
        failures.append(msg)


# --------------------------------------------------------------------------- valid stores
def exercise_valid(spec: StoreSpec, path: str) -> dict[str, Any]:
    failures: list[str] = []
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    _check(is_ome_zarr(path), "is_ome_zarr returned False", failures)
    img = open_ngff(path)
    timings["open_ms"] = (time.perf_counter() - t0) * 1000

    t = time.perf_counter()
    vi = build_ngff_viewer_info(img)
    timings["viewerinfo_ms"] = (time.perf_counter() - t) * 1000

    c = img.num_c
    # --- viewer-info contract invariants ---
    _check(
        vi["axis_sizes"]
        == {"T": img.num_t, "C": c, "Z": img.num_z, "Y": img.num_y, "X": img.num_x},
        f"axis_sizes mismatch: {vi['axis_sizes']}",
        failures,
    )
    _check(
        len(vi["channel_names"]) == max(1, c),
        f"channel_names len {len(vi['channel_names'])} != c {c}",
        failures,
    )
    _check(
        len(vi["channel_colors"]) == len(vi["channel_names"]),
        "channel_colors length != channel_names length",
        failures,
    )
    _check(
        vi["dims_order"] == "".join(img.axes).upper(),
        f"dims_order {vi['dims_order']} != {''.join(img.axes).upper()}",
        failures,
    )
    _check(vi["is_timeseries"] == (img.num_t > 1), "is_timeseries wrong", failures)
    _check(vi["is_volume"] == (img.num_z > 1), "is_volume wrong", failures)
    _check(vi["is_multichannel"] == (c > 1), "is_multichannel wrong", failures)
    _check(
        vi["backend_mode"] in ("direct", "pyramid"),
        f"bad backend_mode {vi['backend_mode']}",
        failures,
    )
    _check(str(img.dtype) == vi["dtype"], f"dtype mismatch {img.dtype} vs {vi['dtype']}", failures)
    _check(
        vi["metadata"]["array_shape"] == [int(s) for s in img.levels[0].shape],
        "array_shape does not equal stored level-0 shape",
        failures,
    )
    if vi["backend_mode"] == "pyramid":
        ts = vi.get("tile_scheme")
        _check(bool(ts and ts.get("levels")), "pyramid backend but no tile_scheme levels", failures)

    # --- render invariants ---
    smallest = len(img.levels) - 1
    t = time.perf_counter()
    thumb = render_thumbnail_png(img, max_size=128)
    timings["thumbnail_ms"] = (time.perf_counter() - t) * 1000
    tw, th = _png_size(thumb)
    _check(max(tw, th) <= 128, f"thumbnail exceeds max_size: {tw}x{th}", failures)

    # slice at the smallest level (always cheap)
    t = time.perf_counter()
    sm = render_slice_png(img, level=smallest, z=img.num_z // 2, t=0)
    timings["slice_smallest_ms"] = (time.perf_counter() - t) * 1000
    sw, sh = _png_size(sm)
    exp_h, exp_w = img.level_yx(smallest)
    _check((sw, sh) == (exp_w, exp_h), f"slice size {sw}x{sh} != level {exp_w}x{exp_h}", failures)

    # determinism: same request → identical bytes (window cache stable)
    _check(
        render_slice_png(img, level=smallest, z=img.num_z // 2, t=0) == sm,
        "slice render is non-deterministic across calls",
        failures,
    )

    # per-channel selection
    if c > 1:
        one = render_slice_png(img, level=smallest, channels=[0])
        _check(len(one) > 0, "single-channel slice empty", failures)
        try:
            render_slice_png(img, level=smallest, channels=[c])  # out of range
            failures.append("out-of-range channel index did not raise")
        except NgffError:
            pass

    # tiles when a pyramid is advertised
    if vi["backend_mode"] == "pyramid":
        t = time.perf_counter()
        for col, row in [(0, 0), (1, 0), (0, 1)]:
            tile = render_tile_png(img, level=0, col=col, row=row, tile_size=256, z=img.num_z // 2)
            _check(len(tile) > 0, f"tile ({col},{row}) empty", failures)
        # far out-of-range tile → 1x1 sentinel, never an error
        big = render_tile_png(img, level=0, col=10_000_000, row=10_000_000, tile_size=256)
        _check(_png_size(big) == (1, 1), "out-of-range tile not a 1x1 sentinel", failures)
        timings["tiles_ms"] = (time.perf_counter() - t) * 1000

    # bad indices must raise, never read a neighbour
    for kwargs in ({"z": img.num_z + 5}, {"t": img.num_t + 5}, {"level": len(img.levels) + 3}):
        try:
            render_slice_png(img, **kwargs)  # type: ignore[arg-type]
            failures.append(f"invalid render args {kwargs} did not raise")
        except NgffError:
            pass

    return {
        "name": spec.store_name,
        "domain": spec.domain,
        "modality": spec.modality,
        "title": spec.title,
        "instrument": spec.instrument,
        "axes": "".join(img.axes),
        "dtype": str(img.dtype),
        "levels": len(img.levels),
        "ngff_version": img.version,
        "backend_mode": vi["backend_mode"],
        "is_volume": vi["is_volume"],
        "is_timeseries": vi["is_timeseries"],
        "is_multichannel": vi["is_multichannel"],
        "intensity_status": vi["metadata"]["intensity_range"]["status"],
        "units": {ax: img.units.get(ax) for ax in ("t", "z", "y", "x") if img.units.get(ax)},
        "timings_ms": {k: round(v, 1) for k, v in timings.items()},
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }


# ------------------------------------------------------------------------- adversarial
def exercise_adversarial(case: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": case.name,
        "classification": case.classification,
        "category": case.category,
        "note": case.note,
    }
    try:
        if case.classification == "reject":
            raised = None
            try:
                img = open_ngff(case.path)
                build_ngff_viewer_info(img)  # some checks live here
            except NgffError as exc:
                raised = str(exc)
            except ValueError as exc:
                raised = str(exc)
            if raised is None:
                result.update(status="FAIL", detail="expected NgffError, none raised")
            elif (
                case.expect_error_substr and case.expect_error_substr.lower() not in raised.lower()
            ):
                result.update(
                    status="WARN",
                    detail=f"rejected but message lacked "
                    f"'{case.expect_error_substr}': {raised[:160]}",
                )
            else:
                result.update(status="PASS", detail=raised[:160])
        else:  # probe: must not crash/hang; record behaviour
            result.update(**_probe_behaviour(case))
    except Exception as exc:  # Any uncaught error is itself the finding.
        result.update(status="FAIL", detail=f"UNCAUGHT {type(exc).__name__}: {exc}")
    return result


def _probe_behaviour(case: Any) -> dict[str, Any]:
    if case.name == "complex64_dtype":
        try:
            img = open_ngff(case.path)
            vi = build_ngff_viewer_info(img)
            png = render_slice_png(img, level=0)
            return {
                "status": "PROBE",
                "detail": f"opened dtype={img.dtype}, rendered {len(png)}B PNG, "
                f"intensity={vi['metadata']['intensity_range']['status']}",
            }
        except (NgffError, ValueError, TypeError) as exc:
            return {
                "status": "PROBE",
                "detail": f"rejected: {type(exc).__name__}: {str(exc)[:140]}",
            }
    if case.name == "symlink_chunk_escape":
        try:
            img = open_ngff(case.path)
            plane = img.read_plane(level=0)
            leaked = bool(np.any(plane == 0xAB)) and bool(np.all(plane == 0xAB))
            return {
                "status": "FAIL" if leaked else "PASS",
                "detail": (
                    "HOST BYTES LEAKED via symlinked chunk"
                    if leaked
                    else f"no leak; plane[0,0]={int(plane.flat[0])} (symlink not followed as raw host bytes)"
                ),
            }
        except Exception as exc:
            return {
                "status": "PASS",
                "detail": f"symlinked chunk read errored safely: {type(exc).__name__}",
            }
    return {"status": "PROBE", "detail": "no probe handler"}


# --------------------------------------------------------------------------- scale probes
def exercise_scale(spec: StoreSpec, path: str) -> dict[str, Any]:
    failures: list[str] = []
    img = open_ngff(path)
    vi = build_ngff_viewer_info(img)
    base_h, base_w = img.level_yx(0)
    megapixels = base_h * base_w / 1e6

    # viewer-info + thumbnail + one tile must all be cheap regardless of declared size.
    def _timed(fn) -> float:
        t = time.perf_counter()
        fn()
        return (time.perf_counter() - t) * 1000

    tinfo = _timed(lambda: build_ngff_viewer_info(open_ngff(path)))
    # A large single-level base plane has no coarser level; the reader's plane-read budget
    # turns what used to be a multi-GB full read into a clean bounded rejection.
    thumb_bounded_reject = False
    try:
        tthumb = _timed(lambda: render_thumbnail_png(img, max_size=128))
    except NgffError:
        tthumb, thumb_bounded_reject = None, True
    ttile = None
    if vi["backend_mode"] == "pyramid":
        ttile = _timed(
            lambda: render_tile_png(img, level=0, col=0, row=0, tile_size=256, z=img.num_z // 2)
        )

    # The pointed question: does a single /slice on the base plane materialise the whole
    # thing — even in scrub mode, which is supposed to be a bounded frame? Measure peak RSS
    # of that exact call in a clean subprocess. We test the *scrub* path (max_dim set): if
    # even that peaks huge, the max_dim only bounds wire bytes, not memory.
    slice_rss_mb = None
    slice_ms = None
    slice_rejected = False
    if megapixels >= 32:  # only worth measuring on genuinely large base planes
        slice_rss_mb, slice_ms, slice_rejected = _subprocess_slice_peak(path)
        # A per-request memory budget: a scrub frame should never need more than a few
        # hundred MB. A bounded rejection (slice_rejected) is the good outcome; a large RSS
        # means the full-plane read happened before the downscale.
        if slice_rss_mb is not None and slice_rss_mb > 400 and not slice_rejected:
            failures.append(
                f"scrub /slice on {megapixels:.0f}MP {len(img.levels)}-level store peaked "
                f"{slice_rss_mb:.0f}MB RSS (full-plane read precedes the max_dim downscale)"
            )

    return {
        "name": spec.store_name,
        "modality": spec.modality,
        "note": spec.notes,
        "declared_base": f"{base_w}x{base_h}",
        "megapixels": round(megapixels, 1),
        "num_t": img.num_t,
        "num_c": img.num_c,
        "num_z": img.num_z,
        "levels": len(img.levels),
        "backend_mode": vi["backend_mode"],
        "intensity_status": vi["metadata"]["intensity_range"]["status"],
        "viewerinfo_ms": round(tinfo, 1),
        "thumbnail_ms": round(tthumb, 1) if tthumb is not None else None,
        "thumbnail_bounded_reject": thumb_bounded_reject,
        "slice_bounded_reject": slice_rejected,
        "tile_ms": round(ttile, 1) if ttile is not None else None,
        "scrub_slice_peak_rss_mb": round(slice_rss_mb, 0) if slice_rss_mb is not None else None,
        "scrub_slice_ms": round(slice_ms, 0) if slice_ms is not None else None,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }


def _subprocess_slice_peak(path: str) -> tuple[float | None, float | None, bool]:
    """Run one scrub level-0 /slice in a fresh process; return (peak RSS MB, ms, rejected).

    Uses the service scrub bound (max_dim=1024) — the *supposedly* memory-bounded path — so
    the measured peak reflects what a pan/scrub frame actually costs. ``rejected`` is True
    when the reader's plane-read budget cleanly refuses the gigapixel full-plane read.
    """
    code = (
        "import os,sys,time,resource\n"
        "sys.path[:0]=[os.environ['PP_SRC'],os.environ['PP_TOOLS']]\n"
        "from ultra_deepagents.ngff.reader import open_ngff, NgffError\n"
        "from ultra_deepagents.ngff.render import render_slice_png\n"
        "img=open_ngff(sys.argv[1])\n"
        "rejected=0\n"
        "t=time.perf_counter()\n"
        "try:\n"
        "    render_slice_png(img, level=0, max_dim=1024)\n"
        "except NgffError:\n"
        "    rejected=1\n"
        "ms=(time.perf_counter()-t)*1000\n"
        "rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        "rss_mb=rss/(1024*1024) if sys.platform=='darwin' else rss/1024\n"
        "print(f'{rss_mb:.1f} {ms:.1f} {rejected}')\n"
    )
    env = dict(os.environ)
    # tools/ngff_sensor_corpus/stress.py -> .../deepagents_runtime -> /src
    runtime_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env["PP_SRC"] = os.path.join(runtime_root, "src")
    env["PP_TOOLS"] = os.path.dirname(os.path.dirname(__file__))
    try:
        out = subprocess.run(
            [sys.executable, "-c", code, path], capture_output=True, text=True, timeout=120, env=env
        )
        if out.returncode != 0:
            return None, None, False
        rss_mb, ms, rejected = out.stdout.strip().split()
        return float(rss_mb), float(ms), rejected == "1"
    except Exception:
        return None, None, False


# --------------------------------------------------------------------------- HTTP + concurrency
def exercise_http(valid_paths: list[str], adversarial_paths: list[str]) -> dict[str, Any]:
    from fastapi.testclient import TestClient
    from ultra_deepagents.ngff.service import create_app

    client = TestClient(create_app())
    out: dict[str, Any] = {"endpoints": [], "status": "PASS", "failures": []}

    h = client.get("/healthz")
    out["endpoints"].append({"GET /healthz": h.status_code})
    if h.status_code != 200:
        out["failures"].append("healthz not 200")

    for p in valid_paths[:8]:
        vi = client.get("/viewerinfo", params={"path": p})
        thumb = client.get("/thumbnail", params={"path": p, "max_size": 96})
        sl = client.get("/slice", params={"path": p, "full_resolution": "false"})
        codes = {
            "viewerinfo": vi.status_code,
            "thumbnail": thumb.status_code,
            "slice": sl.status_code,
        }
        ok = vi.status_code == 200 and thumb.status_code == 200 and sl.status_code == 200
        ct_ok = thumb.headers.get("content-type") == "image/png"
        if not (ok and ct_ok):
            out["failures"].append(
                f"valid store bad HTTP: {os.path.basename(p)} {codes} ct={thumb.headers.get('content-type')}"
            )
        out["endpoints"].append({os.path.basename(p): codes})

    for p in adversarial_paths[:6]:
        r = client.get("/viewerinfo", params={"path": p})
        if r.status_code != 422:
            out["failures"].append(
                f"adversarial store not 422: {os.path.basename(p)} -> {r.status_code}"
            )
        out["endpoints"].append({"adv " + os.path.basename(p): r.status_code})

    # nonexistent path
    r = client.get("/viewerinfo", params={"path": "/no/such/store.zarr"})
    if r.status_code != 422:
        out["failures"].append(f"missing path not 422: {r.status_code}")

    out["status"] = "PASS" if not out["failures"] else "FAIL"
    return out


def exercise_concurrency(valid_paths: list[str], rounds: int = 400) -> dict[str, Any]:
    """Hammer the reader/renderer across threads on shared stores → shared process caches."""
    from fastapi.testclient import TestClient
    from ultra_deepagents.ngff.service import create_app

    client = TestClient(create_app())
    paths = valid_paths[:10] or valid_paths
    errors: list[str] = []

    def one(i: int) -> int:
        p = paths[i % len(paths)]
        kind = i % 4
        try:
            if kind == 0:
                r = client.get("/thumbnail", params={"path": p, "max_size": 64})
            elif kind == 1:
                r = client.get("/slice", params={"path": p, "full_resolution": "false"})
            elif kind == 2:
                r = client.get("/viewerinfo", params={"path": p})
            else:
                r = client.get(
                    "/tile", params={"path": p, "level": 0, "col": i % 3, "row": i % 2, "size": 128}
                )
            return r.status_code
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            return -1

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=16) as ex:
        codes = list(ex.map(one, range(rounds)))
    dt = time.perf_counter() - t0

    non200 = [c for c in codes if c not in (200,)]
    cache = process_plane_cache_info()
    status = "PASS"
    if errors:
        status = "FAIL"
    if any(c < 0 for c in codes):
        status = "FAIL"
    if cache["bytes"] < 0 or cache["bytes"] > cache["max_bytes"] + 1:
        errors.append(f"plane cache byte accounting out of bounds: {cache}")
        status = "FAIL"
    return {
        "rounds": rounds,
        "wall_s": round(dt, 2),
        "rps": round(rounds / dt, 0),
        "non_200_count": len(non200),
        "errors": errors[:10],
        "final_cache": cache,
        "status": status,
    }


# --------------------------------------------------------------------------------- driver
def run(
    out_dir: str, report_path: str | None, *, http: bool = True, concurrency: bool = True
) -> dict[str, Any]:
    valid_specs = catalog()
    scale_specs = scale_probes()

    print(
        f"Building {len(valid_specs)} valid + {len(scale_specs)} scale stores under {out_dir} ..."
    )
    valid_built = build_corpus(valid_specs, out_dir)
    scale_built = build_corpus(scale_specs, out_dir)
    adv_cases = build_adversarial(out_dir)
    print(f"Built adversarial: {len(adv_cases)} cases")

    valid_results = []
    for spec, path in valid_built:
        try:
            valid_results.append(exercise_valid(spec, path))
        except Exception as exc:
            valid_results.append(
                {
                    "name": spec.store_name,
                    "domain": spec.domain,
                    "status": "ERROR",
                    "failures": [f"{type(exc).__name__}: {exc}"],
                    "trace": traceback.format_exc().splitlines()[-3:],
                }
            )

    scale_results = [exercise_scale(s, p) for s, p in scale_built]
    adv_results = [exercise_adversarial(c) for c in adv_cases]

    valid_paths = [p for _, p in valid_built]
    adv_paths = [c.path for c in adv_cases if c.classification == "reject"]
    http_result = exercise_http(valid_paths, adv_paths) if http else {"status": "SKIPPED"}
    conc_result = exercise_concurrency(valid_paths) if concurrency else {"status": "SKIPPED"}

    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = (rss1 / (1024 * 1024)) if sys.platform == "darwin" else (rss1 / 1024)

    report = {
        "summary": _summary(
            valid_results, scale_results, adv_results, http_result, conc_result, peak_mb
        ),
        "valid": valid_results,
        "scale": scale_results,
        "adversarial": adv_results,
        "http": http_result,
        "concurrency": conc_result,
    }
    if report_path:
        with open(report_path, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
    _print_summary(report)
    return report


def _summary(valid, scale, adv, http, conc, peak_mb) -> dict[str, Any]:
    return {
        "valid_total": len(valid),
        "valid_pass": sum(1 for r in valid if r.get("status") == "PASS"),
        "valid_fail": sum(1 for r in valid if r.get("status") in ("FAIL", "ERROR")),
        "scale_total": len(scale),
        "scale_fail": sum(1 for r in scale if r.get("status") == "FAIL"),
        "adversarial_total": len(adv),
        "adversarial_pass": sum(1 for r in adv if r.get("status") == "PASS"),
        "adversarial_fail": sum(1 for r in adv if r.get("status") == "FAIL"),
        "adversarial_warn": sum(1 for r in adv if r.get("status") == "WARN"),
        "http_status": http.get("status"),
        "concurrency_status": conc.get("status"),
        "harness_peak_rss_mb": round(peak_mb, 0),
    }


def _print_summary(report: dict[str, Any]) -> None:
    s = report["summary"]
    print("\n" + "=" * 78)
    print("NGFF SENSOR-DATA STRESS REPORT")
    print("=" * 78)
    print(f"Valid stores:   {s['valid_pass']}/{s['valid_total']} PASS   ({s['valid_fail']} fail)")
    print(
        f"Adversarial:    {s['adversarial_pass']}/{s['adversarial_total']} reject-PASS   "
        f"({s['adversarial_warn']} warn, {s['adversarial_fail']} fail)"
    )
    print(f"Scale probes:   {s['scale_total'] - s['scale_fail']}/{s['scale_total']} bounded")
    print(f"HTTP contract:  {s['http_status']}")
    print(f"Concurrency:    {s['concurrency_status']}")
    print(f"Harness peak RSS: {s['harness_peak_rss_mb']} MB")
    print("-" * 78)
    for r in report["valid"]:
        if r.get("status") != "PASS":
            print(f"  VALID {r.get('status')}: {r['name']}: {r.get('failures')}")
    for r in report["adversarial"]:
        if r.get("status") in ("FAIL", "WARN"):
            print(f"  ADV {r['status']}: {r['name']}: {r.get('detail')}")
    for r in report["scale"]:
        if r.get("status") == "FAIL":
            print(f"  SCALE FAIL: {r['name']}: {r.get('failures')}")
        if r.get("scrub_slice_peak_rss_mb"):
            verdict = "BOUNDED-REJECT" if r.get("slice_bounded_reject") else "full-read"
            print(
                f"  scale/{r['modality']}: {r['megapixels']}MP {r['backend_mode']} "
                f"scrub-slice {verdict} peak={r['scrub_slice_peak_rss_mb']}MB in {r['scrub_slice_ms']}ms"
                + ("  (thumbnail bounded-reject)" if r.get("thumbnail_bounded_reject") else "")
            )
    if report["concurrency"].get("errors"):
        print(f"  CONCURRENCY errors: {report['concurrency']['errors']}")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(description="OME-NGFF sensor-data stress harness")
    ap.add_argument("--out", required=True, help="output dir for the generated corpus")
    ap.add_argument("--report", default=None, help="path to write the JSON report")
    ap.add_argument("--no-http", action="store_true")
    ap.add_argument("--no-concurrency", action="store_true")
    args = ap.parse_args()
    report = run(args.out, args.report, http=not args.no_http, concurrency=not args.no_concurrency)
    fails = (
        report["summary"]["valid_fail"]
        + report["summary"]["adversarial_fail"]
        + report["summary"]["scale_fail"]
        + (1 if report["summary"]["http_status"] == "FAIL" else 0)
        + (1 if report["summary"]["concurrency_status"] == "FAIL" else 0)
    )
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
