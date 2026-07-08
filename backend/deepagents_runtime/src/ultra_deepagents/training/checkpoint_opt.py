"""Checkpoint-opt verification — GoldGate M1 freeze input #2.

Loads a YOLOv5 checkpoint and reports every piece of embedded training
provenance (the ``opt`` namespace/dict train.py stores, epoch counters, wandb
ids, date, git info) to answer ONE question: did the ``all_overfit.yaml`` run
consume run3? The verdict is written into the inventory's per-run
``trained_on`` field by the caller:

- ``verified_true``   — the embedded manifest provably includes run3 tiles
- ``verified_false``  — it provably excludes them
- ``presumed``        — inconclusive; run3 stays TREATED AS TRAINED-ON
  (quarantined from the replay pool, barred from held-out gold) per plan
  section 5e. Honesty rule: a missing/`opt`-less checkpoint or a bare path
  with no embedded composition is INCONCLUSIVE, never "false".

Requires torch (the code-exec sandbox image has it; the control-plane venvs do
not). Run:  python -m ultra_deepagents.training.checkpoint_opt <weights.pt>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _ensure_yolov5_importable(yolov5_path: Path | None) -> None:
    """YOLOv5 checkpoints pickle the model CLASS (models.yolo.Model), so
    torch.load needs the yolov5 tree on sys.path. Vendored in-repo at
    third_party/yolov5; baked in the sandbox at /opt/rarespot/yolov5."""
    if yolov5_path is None:
        return
    resolved = str(yolov5_path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__") and value.__dict__:
        return {str(key): _to_plain(item) for key, item in vars(value).items()}
    return repr(value)


class _StubBase:
    """Placeholder for classes we cannot (and need not) import — the report
    reads checkpoint METADATA (opt, epoch, date, git), never runs the model."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state


def _stub_pickle_module() -> Any:
    import pickle
    import types

    class StubUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):  # noqa: ANN202
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, AttributeError):
                return type(name, (_StubBase,), {"__module__": module})

    shim = types.SimpleNamespace()
    shim.__name__ = "goldgate_stub_pickle"
    shim.Unpickler = StubUnpickler
    shim.load = pickle.load
    shim.loads = pickle.loads
    shim.dump = pickle.dump
    shim.dumps = pickle.dumps
    return shim


def inspect_checkpoint(weights_path: Path, yolov5_path: Path | None = None) -> dict[str, Any]:
    import torch

    _ensure_yolov5_importable(yolov5_path)
    stubbed = False
    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    except (ModuleNotFoundError, AttributeError):
        # The checkpoint pickles yolov5 model classes whose import chain
        # (cv2, pandas, ...) is not installed here. Stub them: everything the
        # report needs (opt/epoch/date/git/names) survives stubbing.
        checkpoint = torch.load(
            weights_path, map_location="cpu", weights_only=False, pickle_module=_stub_pickle_module()
        )
        stubbed = True
    report: dict[str, Any] = {
        "weights_path": str(weights_path),
        "model_classes_stubbed": stubbed,
        "checkpoint_keys": sorted(str(key) for key in checkpoint.keys()),
        "epoch": _to_plain(checkpoint.get("epoch")),
        "best_fitness": _to_plain(checkpoint.get("best_fitness")),
        "date": _to_plain(checkpoint.get("date")),
        "git": _to_plain(checkpoint.get("git")),
        "wandb_id": _to_plain(checkpoint.get("wandb_id")),
        "opt": _to_plain(checkpoint.get("opt")),
    }
    model = checkpoint.get("model")
    if model is not None:
        names = getattr(model, "names", None)
        report["model_names"] = _to_plain(names)
        report["model_nc"] = _to_plain(getattr(model, "nc", None))
        hyp = getattr(model, "hyp", None)
        if hyp is not None:
            report["model_hyp"] = _to_plain(hyp)

    report["run3_verdict"] = _derive_run3_verdict(report)
    return report


def _derive_run3_verdict(report: dict[str, Any]) -> dict[str, str]:
    """Conservative: only an EMBEDDED train-set composition can verify; a bare
    data-yaml path proves nothing about what the yaml contained at train time."""
    opt = report.get("opt")
    blob = json.dumps(opt, default=str).lower() if opt else ""
    if not blob:
        return {
            "verdict": "presumed",
            "reason": "checkpoint carries no opt/training-args dict - composition unknowable from the artifact",
        }
    mentions_run3 = "run3" in blob
    embeds_composition = any(token in blob for token in ("run1_tiles", "run2_tiles", "train:", "images/"))
    if mentions_run3 and embeds_composition:
        return {
            "verdict": "verified_true",
            "reason": "embedded opt references run3 alongside the train composition",
        }
    data_path = ""
    if isinstance(opt, dict):
        data_path = str(opt.get("data") or "")
    return {
        "verdict": "presumed",
        "reason": (
            f"opt records only a data-yaml path ({data_path or 'unknown'}) with no embedded "
            "composition; the yaml's train-time contents are not recoverable from the checkpoint - "
            "run3 stays treated as trained-on"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--yolov5", type=Path, default=None, help="yolov5 tree for unpickling (third_party/yolov5 or /opt/rarespot/yolov5)")
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    args = parser.parse_args()
    report = inspect_checkpoint(args.weights, args.yolov5)
    payload = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
