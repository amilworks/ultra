"""``rarespot.train`` executor — local GPU finetune (replaces the ssh+rsync recipe).

Runs the EXACT plan-6.2 command from ``build_finetune_command`` as a fresh
subprocess inside this GPU container: no ssh, no rsync, no nested docker, no
per-run pip install (the yolov5 fork is BN-patched and the deps are baked into the
image at build time — see services/compute_service/Dockerfile). Each job is a new
process group so cancel kills the trainer AND its DataLoader children cleanly, which
is what makes back-to-back launches reliable.

The worker assembles the dataset (CPU, model-specific) to a shared path and passes
it by reference; this executor only does GPU work.

spec:
    weights_uri   (required) warm-start weights, a path readable in-container
    data_yaml     (required) assembled dataset yaml (path readable in-container)
    new_tile_count(optional) drives 40 vs 60 epochs
    batch_size    (optional) default 16
    hyp_path      (optional) override the baked self-authored hyp
    output_dir    (optional) shared-volume dir to copy best.pt into for the caller
    (GPU is pinned at the container level via CUDA_VISIBLE_DEVICES, not per job.)
returns:
    {weights_uri, epochs, best_map50, best_map50_95, run_dir, log_tail}
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from ultra_deepagents.training.rarespot_adapter import HYP_PATH, build_finetune_command

from .base import ExecutorManifest, JobCancelled, JobContext

YOLOV5_PATH = Path(os.getenv("ULTRA_RARESPOT_YOLOV5_PATH") or "/opt/rarespot/yolov5")
# Hard wall-clock cap: an unattended/hung finetune must eventually free the GPU so
# the next launch can start (the user's back-to-back guarantee). Default 6h.
WALL_CLOCK_BUDGET_S = float(os.getenv("ULTRA_RARESPOT_TRAIN_TIMEOUT_S") or 6 * 3600)

# yolov5 prints "     12/39     3.4G   0.04 ..." per training epoch (0-indexed of
# N-1), and a "                 all   200   450   0.81   0.73   0.785   0.42" line
# after each validation. Parse both for live progress + best-so-far metrics.
_EPOCH_RE = re.compile(r"^\s*(\d+)/(\d+)\s+[\d.]+G")
_MAP_RE = re.compile(r"^\s*all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


class RareSpotTrainExecutor:
    manifest = ExecutorManifest(
        model_key="rarespot",
        capability="train",
        description="yolov5 warm-start finetune (BN-frozen, imgsz 512) producing best.pt",
        resource={"min_vram_gb": 12, "wall_clock_budget_s": 6 * 3600},
    )

    def run(self, spec: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
        weights_uri = str(spec.get("weights_uri") or "").strip()
        data_yaml = str(spec.get("data_yaml") or "").strip()
        if not weights_uri or not Path(weights_uri).is_file():
            raise ValueError(f"rarespot.train: warm-start weights not readable: {weights_uri!r}")
        if not data_yaml or not Path(data_yaml).is_file():
            raise ValueError(f"rarespot.train: assembled data_yaml not readable: {data_yaml!r}")

        hyp_path = str(spec.get("hyp_path") or "").strip() or str(HYP_PATH)
        run_dir = ctx.workdir / "runs"
        run_dir.mkdir(parents=True, exist_ok=True)

        command = build_finetune_command(
            {
                "weights_uri": weights_uri,
                "data_yaml": data_yaml,
                "hyp_path": hyp_path,
                "run_dir": str(run_dir),
                "yolov5_path": str(YOLOV5_PATH),
                "new_tile_count": int(spec.get("new_tile_count") or 0),
                "batch_size": int(spec.get("batch_size") or 16),
                # The image ships python3.10 (no bare `python`); run yolov5 with the
                # interpreter that has torch installed.
                "python": sys.executable,
            }
        )
        total_epochs = int(command[command.index("--epochs") + 1])

        env = os.environ.copy()
        # GPU pinning is at the CONTAINER level (CUDA_VISIBLE_DEVICES in the service
        # env-file, set before torch initializes) — inherited here. Re-setting it to
        # a raw ordinal would double-restrict when --gpus already narrows the view.
        env["GIT_PYTHON_REFRESH"] = "quiet"
        env["YOLOv5_AUTOINSTALL"] = "false"
        # Deliberately NOT overriding YOLOV5_CONFIG_DIR: the proven recipe relies on
        # the default (/root/.config/Ultralytics), where Arial.ttf is seeded at image
        # build so check_font never tries its 404-ing offline download.
        env["MPLCONFIGDIR"] = str((ctx.workdir / ".mpl").resolve())
        env["PYTHONPATH"] = f"{YOLOV5_PATH}{os.pathsep}{env.get('PYTHONPATH', '')}"

        log_path = ctx.workdir / "train.log"
        ctx.progress(total=total_epochs, message="starting finetune", phase="finetune")

        best_map50 = 0.0
        best_map50_95 = 0.0
        tail: list[str] = []
        # start_new_session=True -> the child is a process-group leader so cancel
        # (ctx._kill_processes -> killpg) tears down its DataLoader workers too.
        process = subprocess.Popen(  # noqa: S603 - argv is built from a fixed recipe
            command,
            cwd=str(YOLOV5_PATH),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        ctx.register_process(process)
        # A watchdog kills the process group on cancel OR wall-clock timeout even if
        # the child has gone silent on stdout (dataset cache / a driver hang), so the
        # stdout read loop below never has to be the sole cancellation path and a
        # stuck job can't hold the GPU slot open.
        deadline = time.monotonic() + WALL_CLOCK_BUDGET_S
        watchdog_stop = threading.Event()

        def _watchdog() -> None:
            while not watchdog_stop.wait(2.0):
                if ctx.cancelled or time.monotonic() > deadline:
                    ctx._kill_processes()  # noqa: SLF001
                    return

        watchdog = threading.Thread(target=_watchdog, name=f"train-watchdog-{ctx.job_id}", daemon=True)
        watchdog.start()
        try:
            with log_path.open("w", encoding="utf-8") as log:
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    tail.append(line)
                    if len(tail) > 80:
                        tail.pop(0)
                    epoch_match = _EPOCH_RE.match(line)
                    if epoch_match:
                        completed = int(epoch_match.group(1)) + 1
                        ctx.progress(
                            completed=min(completed, total_epochs),
                            total=total_epochs,
                            message=f"epoch {completed}/{total_epochs}",
                            phase="finetune",
                        )
                    map_match = _MAP_RE.match(line)
                    if map_match:
                        m50 = float(map_match.group(3))
                        m95 = float(map_match.group(4))
                        best_map50 = max(best_map50, m50)
                        best_map50_95 = max(best_map50_95, m95)
                        ctx.progress(metrics={"map50": round(best_map50, 4), "map50_95": round(best_map50_95, 4)})
                    if ctx.cancelled:
                        break
        finally:
            watchdog_stop.set()
            # On a cancel/timeout break the child may still be alive; kill its
            # process group (escalating SIGTERM->SIGKILL) BEFORE reaping so
            # process.wait() can never block here. On a normal EOF this is a no-op.
            if process.poll() is None:
                ctx._kill_processes()  # noqa: SLF001
            returncode = process.wait()
            watchdog.join(timeout=5)

        timed_out = time.monotonic() > deadline and not ctx.cancelled
        if ctx.cancelled:
            raise JobCancelled(f"rarespot.train job {ctx.job_id} cancelled")
        if timed_out:
            raise RuntimeError(
                f"rarespot.train exceeded wall-clock budget {WALL_CLOCK_BUDGET_S:.0f}s; "
                f"log tail:\n{''.join(tail)[-2000:]}"
            )
        if returncode != 0:
            raise RuntimeError(
                f"yolov5 train.py exited {returncode}; log tail:\n{''.join(tail)[-2000:]}"
            )

        best_pt = run_dir / "finetune" / "weights" / "best.pt"
        if not best_pt.is_file():
            raise RuntimeError(
                f"finetune produced no best.pt at {best_pt}; log tail:\n{''.join(tail)[-2000:]}"
            )

        # If the caller gave an explicit shared-volume destination, copy best.pt
        # (and the run log) there so the worker reads a stable path independent of
        # this service's ephemeral, eviction-bounded workspace.
        weights_out = best_pt
        output_dir = str(spec.get("output_dir") or "").strip()
        if output_dir:
            dest = Path(output_dir)
            dest.mkdir(parents=True, exist_ok=True)
            weights_out = dest / "best.pt"
            shutil.copy2(best_pt, weights_out)
            with contextlib.suppress(Exception):
                shutil.copy2(log_path, dest / "train.log")

        ctx.progress(completed=total_epochs, total=total_epochs, message="finetune complete", phase="finetune")
        return {
            "weights_uri": str(weights_out),
            "epochs": total_epochs,
            "best_map50": round(best_map50, 4),
            "best_map50_95": round(best_map50_95, 4),
            "run_dir": str(run_dir / "finetune"),
            "log_tail": "".join(tail)[-2000:],
        }

    @staticmethod
    def _consume_line(*_args: Any, **_kwargs: Any) -> None:
        # Reserved hook for future structured-log parsing; kept explicit so the
        # main loop reads top-to-bottom without a second regex pass helper.
        return None


from . import register  # noqa: E402 - register after the class is defined

register(RareSpotTrainExecutor())
