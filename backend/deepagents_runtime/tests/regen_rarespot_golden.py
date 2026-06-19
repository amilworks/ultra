"""Regenerate the RareSpot parity golden from the Skill CLI, on CPU.

Run inside the bisque-ultra-codeexec sandbox image (everything baked), or in a
checkout that has the weights + yolov5 + fixture staged under data/models/yolo/::

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
        python -m tests.regen_rarespot_golden

It runs the same pinned-CPU CLI invocation the parity test uses, distills the
detection counts + stability buckets + spectral ordering into the committed golden
(tests/data/rarespot_parity_golden.json), and prints it. The golden MUST be
generated on CPU — do not transcribe the legacy GPU worker's numbers.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tests.test_rarespot_skill_parity import _GOLDEN, _RUNTIME_READY, _run_cli


def main() -> int:
    if not _RUNTIME_READY:
        raise SystemExit(
            "RareSpot runtime not available — run inside the bisque-ultra-codeexec image "
            "or stage weights/yolov5/fixture under data/models/yolo/."
        )
    with tempfile.TemporaryDirectory() as tmp:
        summary = _run_cli(Path(tmp) / "golden_run", spectral=True)
    golden = {
        "_note": "CPU-generated parity golden for the prairie-dog-detection Skill. Regenerate with tests.regen_rarespot_golden.",
        "fixture": "data/models/yolo/rarespot_parity_fixture.JPG (EnrNE_Day2_Run1__0241.JPG)",
        "counts_by_class": summary["counts_by_class"],
        "total_detections": summary["total_detections"],
        "stability": {"label_counts": summary["stability"]["label_counts"]},
        "top_spectral_review_candidates": [
            {"file_name": c.get("file_name")}
            for c in summary.get("top_spectral_review_candidates", [])[:5]
        ],
    }
    _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    _GOLDEN.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
    print(f"Wrote golden -> {_GOLDEN}")
    print(json.dumps(golden, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
