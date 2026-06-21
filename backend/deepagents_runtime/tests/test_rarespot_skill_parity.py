"""Parity gate for the prairie-dog-detection Skill.

This proves the sandbox Skill CLI (``skills/prairie-dog-detection/rarespot_detect.py``)
reproduces the production RareSpot pipeline's **CPU** output, so the dedicated
``rarespot_worker`` + NATS dispatch can be retired without regressing the science.

Both the Skill CLI and the (retiring) worker funnel through the identical
``run_rarespot_inference``, so parity reduces to *same code + same inputs + same
numerics*. This test pins the device (CPU) and the thread count, runs the CLI as a
subprocess (the real entrypoint), and asserts a **committed, CPU-generated** golden:

* ``counts_by_class``           — EXACT (detector output is deterministic on CPU)
* ``total_detections``          — EXACT
* stability ``label_counts``    — within ±1 per bucket, sum invariant (borderline
                                  boxes can cross the 0.5/0.75 thresholds under the
                                  perturbation passes; a hard-equal here is flaky)
* top spectral review candidate — leading id ordering stable (spectral runs fp32 on
                                  CPU here vs fp16/CUDA on the old worker — a
                                  different numeric path, so only ordering is checked)

Requires the runtime baked into ``bisque-ultra-codeexec`` (torch + torchvision + the
weights + the yolov5 runtime + the fixture image). It **skips** in the bare CI venv
that has none of these — run it inside the sandbox image, where everything is baked.
Do NOT inherit the legacy worker's GPU numbers; regenerate the golden on CPU with::

    python -m tests.regen_rarespot_golden   # (or the snippet in this module's docstring)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_RT_ROOT = Path(__file__).resolve().parents[1]
_CLI = _RT_ROOT / "skills" / "prairie-dog-detection" / "rarespot_detect.py"
_GOLDEN = Path(__file__).resolve().parent / "data" / "rarespot_parity_golden.json"


def _repo_root() -> Path:
    """Find the repo root by walking up for a marker dir, instead of a fixed
    parents[N] — so the module imports whether run from a full checkout or from a
    ``deepagents_runtime``-only mount inside the sandbox image."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "third_party" / "yolov5").exists() or (parent / "data" / "models" / "yolo").exists():
            return parent
    return here.parents[min(2, len(here.parents) - 1)]


_REPO_ROOT = _repo_root()


def _first_existing(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _weights() -> Path | None:
    override = os.getenv("RARESPOT_PARITY_WEIGHTS")
    if override:
        return Path(override) if Path(override).exists() else None
    return _first_existing(
        Path("/opt/rarespot/RareSpotWeights.pt"),
        _REPO_ROOT / "data" / "models" / "yolo" / "RareSpotWeights.pt",
    )


def _yolov5() -> Path | None:
    override = os.getenv("RARESPOT_PARITY_YOLOV5")
    if override:
        return Path(override) if Path(override).exists() else None
    return _first_existing(
        Path("/opt/rarespot/yolov5"),
        _REPO_ROOT / "third_party" / "yolov5",
    )


def _fixture() -> Path | None:
    override = os.getenv("RARESPOT_PARITY_FIXTURE")
    if override:
        return Path(override) if Path(override).exists() else None
    return _first_existing(
        _REPO_ROOT / "data" / "models" / "yolo" / "rarespot_parity_fixture.JPG",
        Path("/opt/rarespot/rarespot_parity_fixture.JPG"),
    )


def _have_torch() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except Exception:
        return False
    return True


_WEIGHTS = _weights()
_YOLOV5 = _yolov5()
_FIXTURE = _fixture()
_RUNTIME_READY = bool(_WEIGHTS and _YOLOV5 and _FIXTURE and _have_torch())

pytestmark = pytest.mark.skipif(
    not _RUNTIME_READY,
    reason=(
        "RareSpot runtime not available "
        f"(torch={_have_torch()} weights={_WEIGHTS} yolov5={_YOLOV5} fixture={_FIXTURE}); "
        "run inside the bisque-ultra-codeexec image."
    ),
)


def _run_cli(out_dir: Path, *, spectral: bool = True) -> dict:
    """Run the Skill CLI as a subprocess (the real sandbox entrypoint) with a pinned
    CPU / single-thread environment for determinism, and return its summary JSON."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # force CPU — the sandbox has no GPU
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["YOLOv5_AUTOINSTALL"] = "false"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(_RT_ROOT / "src"), env.get("PYTHONPATH", "")])
    )
    cmd = [
        sys.executable,
        str(_CLI),
        "--images", str(_FIXTURE),
        "--weights", str(_WEIGHTS),
        "--yolov5", str(_YOLOV5),
        "--out", str(out_dir),
        "--conf", "0.25",
        "--iou", "0.45",
        "--tile-size", "512",
        "--tile-overlap", "0.25",
    ]
    cmd.append("--spectral" if spectral else "--no-spectral")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=2400)
    assert proc.returncode == 0, f"CLI failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr[-3000:]}"
    # The CLI prints the summary JSON on stdout (yolov5 chatter goes to stderr).
    # Decode the first complete JSON object from the first '{' — robust to any
    # trailing newline and to nested braces (do NOT rfind a brace, which lands
    # inside a nested object and yields "Extra data").
    stdout = proc.stdout
    brace = stdout.find("{")
    assert brace >= 0, f"no JSON object in CLI stdout:\n{stdout[:2000]}"
    obj, _ = json.JSONDecoder().raw_decode(stdout[brace:])
    return obj


def test_skill_cli_reproduces_cpu_golden(tmp_path):
    if not _GOLDEN.exists():
        pytest.skip(
            f"Golden {_GOLDEN} not committed yet. Generate it on CPU with the Skill CLI "
            "in the sandbox image, then commit it (see module docstring)."
        )
    golden = json.loads(_GOLDEN.read_text())
    summary = _run_cli(tmp_path / "skill_run", spectral=True)

    # 1) Detection counts: EXACT (deterministic on CPU).
    assert summary["counts_by_class"] == golden["counts_by_class"], (
        f"counts_by_class drift: got {summary['counts_by_class']} vs golden {golden['counts_by_class']}"
    )
    assert summary["total_detections"] == golden["total_detections"]

    # 2) Stability trust buckets: within ±1 per bucket, total preserved.
    got_buckets = summary["stability"]["label_counts"]
    want_buckets = golden["stability"]["label_counts"]
    assert sum(got_buckets.values()) == sum(want_buckets.values()), (
        f"stability total changed: {got_buckets} vs {want_buckets}"
    )
    for label in set(want_buckets) | set(got_buckets):
        assert abs(got_buckets.get(label, 0) - want_buckets.get(label, 0)) <= 1, (
            f"stability bucket '{label}' drift > 1: got {got_buckets} vs golden {want_buckets}"
        )

    # 3) Spectral review ordering: leading candidate file stable (loose — the
    # spectral pass runs fp32/CPU here vs fp16/CUDA on the old worker, a different
    # numeric path, so only the top-ranked file is checked, not the score).
    got_top = [c.get("file_name") for c in summary.get("top_spectral_review_candidates", [])]
    want_top = [c.get("file_name") for c in golden.get("top_spectral_review_candidates", [])]
    if want_top and got_top:
        assert got_top[0] == want_top[0], f"top spectral candidate changed: {got_top[:3]} vs {want_top[:3]}"


def test_network_none_preconditions():
    """A green parity pass must not be able to come from a hidden network install:
    torchvision (NMS) must already be importable, and YOLOv5 auto-install disabled."""
    import torchvision  # noqa: F401  (import error here = the bake is broken)

    assert os.getenv("YOLOv5_AUTOINSTALL", "false").lower() in {"false", "0", "no"}
    # The CLI/inference force YOLOv5_AUTOINSTALL=false; assert the source still does so.
    inference_src = (_RT_ROOT / "src" / "ultra_deepagents" / "rarespot" / "inference.py").read_text()
    assert 'YOLOv5_AUTOINSTALL' in inference_src and 'false' in inference_src, (
        "inference.run_yolov5_detect must set YOLOv5_AUTOINSTALL=false for the network-none sandbox"
    )
