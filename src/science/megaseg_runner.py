#!/usr/bin/env python3
"""Thin CLI wrapper around the shared Megaseg inference core."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.science.megaseg_core import load_megaseg_request, run_megaseg_batch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Megaseg inference on microscopy images.")
    parser.add_argument("--request-json", required=True, help="Path to the runner request JSON.")
    args = parser.parse_args()

    request_path = Path(args.request_json).expanduser().resolve()
    request = load_megaseg_request(request_path)

    result = run_megaseg_batch(
        file_paths=[str(item) for item in list(request.get("file_paths") or [])],
        output_dir=Path(str(request["output_dir"])).expanduser().resolve(),
        checkpoint_path=Path(str(request["checkpoint_path"])).expanduser().resolve(),
        structure_channel=int(request.get("structure_channel") or 4),
        nucleus_channel=(
            int(request.get("nucleus_channel"))
            if request.get("nucleus_channel") is not None
            else None
        ),
        channel_index_base=int(request.get("channel_index_base") or 1),
        mask_threshold=float(request.get("mask_threshold") or 0.5),
        save_visualizations=bool(request.get("save_visualizations", True)),
        generate_report=bool(request.get("generate_report", True)),
        device=str(request.get("device") or "").strip() or None,
        structure_name=str(request.get("structure_name") or "structure"),
        amp_enabled=bool(request.get("amp_enabled", False)),
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
