"""Golden OCR eval: synthetic fixtures with exact ground truth.

OCR is the rare capability with free ground truth — we render the text
ourselves, so accuracy is measured, not asserted. The suite generates fixtures
(clean text, degraded text, a table, a plot-text panel) and scores:

- engine mode (default): tesseract inside the codeexec sandbox image, the same
  binary the ocr-reader subagent uses. Requires docker + the image.
- --vlm mode: the Qwen endpoint (QWEN_VLM_BASE_URL / QWEN_VLM_API_KEY env)
  prompted with the verbatim-transcription contract.

Scores: character error rate (CER) for prose, exact-cell fraction for the
table, expected-string hit rate for plot text. Emits a JSON report; exit 0
always when the eval RAN (thresholds are advisory gates for humans/CI to
interpret), exit 2 when prerequisites are missing.

Usage:
  uv run python scripts/eval_ocr_golden.py [--vlm] [--image TAG] [--out report.json]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

DEFAULT_IMAGE = os.getenv("CODE_EXECUTION_DOCKER_IMAGE", "bisque-ultra-codeexec:py311")

PROSE = (
    "The convolution kernel slides across the input image.\n"
    "Each position computes a dot product with bias 0.125.\n"
    "Feature maps downsample through max pooling layers."
)
TABLE_CELLS = [
    ["layer", "params", "output"],
    ["conv1", "896", "26x26x32"],
    ["pool1", "0", "13x13x32"],
    ["dense", "16640", "128"],
]
PLOT_STRINGS = [
    "Validation accuracy",
    "epoch",
    "accuracy",
    "0.0",
    "0.5",
    "1.0",
    "train",
    "test",
]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_fixtures(directory: Path) -> dict[str, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    fixtures: dict[str, dict] = {}

    clean = Image.new("RGB", (900, 240), "white")
    draw = ImageDraw.Draw(clean)
    draw.multiline_text((30, 30), PROSE, fill="black", font=_font(28), spacing=14)
    clean.save(directory / "prose_clean.png")
    fixtures["prose_clean"] = {"kind": "prose", "truth": PROSE}

    degraded = clean.filter(ImageFilter.GaussianBlur(radius=1.1)).point(
        lambda value: int(value * 0.72 + 60)
    )
    degraded.save(directory / "prose_degraded.png")
    fixtures["prose_degraded"] = {"kind": "prose", "truth": PROSE}

    table = Image.new("RGB", (700, 260), "white")
    draw = ImageDraw.Draw(table)
    for row_index, row in enumerate(TABLE_CELLS):
        for col_index, cell in enumerate(row):
            draw.text(
                (40 + col_index * 220, 30 + row_index * 55),
                cell,
                fill="black",
                font=_font(26),
            )
    for row_index in range(len(TABLE_CELLS) + 1):
        draw.line(
            [(25, 15 + row_index * 55), (675, 15 + row_index * 55)], fill="black"
        )
    table.save(directory / "table.png")
    fixtures["table"] = {
        "kind": "table",
        "truth": [cell for row in TABLE_CELLS for cell in row],
    }

    plot = Image.new("RGB", (800, 500), "white")
    draw = ImageDraw.Draw(plot)
    draw.text((250, 20), "Validation accuracy", fill="black", font=_font(30))
    draw.line([(90, 430), (740, 430)], fill="black", width=2)
    draw.line([(90, 60), (90, 430)], fill="black", width=2)
    draw.text((380, 455), "epoch", fill="black", font=_font(24))
    draw.text((16, 220), "accuracy", fill="black", font=_font(22))
    for label, y in (("1.0", 60), ("0.5", 245), ("0.0", 425)):
        draw.text((48, y - 12), label, fill="black", font=_font(20))
    draw.line([(100, 400), (400, 150), (720, 100)], fill="blue", width=3)
    draw.line([(100, 410), (400, 210), (720, 180)], fill="red", width=3)
    draw.text((600, 120), "train", fill="blue", font=_font(22))
    draw.text((600, 200), "test", fill="red", font=_font(22))
    plot.save(directory / "plot.png")
    fixtures["plot"] = {"kind": "plot", "truth": PLOT_STRINGS}

    return fixtures


def character_error_rate(truth: str, hypothesis: str) -> float:
    reference = " ".join(truth.lower().split())
    candidate = " ".join(hypothesis.lower().split())
    if not reference:
        return 0.0
    previous = list(range(len(candidate) + 1))
    for i, ref_char in enumerate(reference, 1):
        current = [i]
        for j, cand_char in enumerate(candidate, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_char != cand_char),
                )
            )
        previous = current
    return previous[-1] / len(reference)


def score(fixtures: dict[str, dict], transcripts: dict[str, str]) -> dict:
    results: dict[str, dict] = {}
    for name, spec in fixtures.items():
        transcript = transcripts.get(name, "")
        if spec["kind"] == "prose":
            results[name] = {"cer": round(character_error_rate(spec["truth"], transcript), 4)}
        else:
            expected = spec["truth"]
            lowered = transcript.lower()
            hits = sum(1 for item in expected if item.lower() in lowered)
            results[name] = {
                "hit_rate": round(hits / len(expected), 4),
                "missing": [item for item in expected if item.lower() not in lowered],
            }
    return results


def run_engine(fixture_dir: Path, fixtures: dict[str, dict], image: str) -> dict[str, str]:
    if shutil.which("docker") is None:
        sys.exit(2)
    probe = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        timeout=30,
        check=False,
    )
    if probe.returncode != 0:
        print(f"prerequisite missing: docker image {image}", file=sys.stderr)
        sys.exit(2)
    transcripts: dict[str, str] = {}
    for name in fixtures:
        completed = subprocess.run(
            [
                "docker", "run", "--rm", "--network", "none",
                "-v", f"{fixture_dir}:/fixtures:ro",
                image,
                "tesseract", f"/fixtures/{name}.png", "stdout",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        transcripts[name] = completed.stdout
    return transcripts


def run_vlm(fixture_dir: Path, fixtures: dict[str, dict]) -> dict[str, str]:
    base_url = os.getenv("QWEN_VLM_BASE_URL", "").rstrip("/")
    api_key = os.getenv("QWEN_VLM_API_KEY", "")
    model = os.getenv("QWEN_VLM_MODEL", "Qwen3.6-27B")
    if not base_url or not api_key:
        print("prerequisite missing: QWEN_VLM_BASE_URL / QWEN_VLM_API_KEY", file=sys.stderr)
        sys.exit(2)
    import urllib.request

    transcripts: dict[str, str] = {}
    for name in fixtures:
        encoded = base64.b64encode((fixture_dir / f"{name}.png").read_bytes()).decode()
        body = json.dumps(
            {
                "model": model,
                "max_tokens": 1200,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Transcribe ALL text in this image verbatim, top to "
                                    "bottom. Output only the text, no commentary. Mark "
                                    "unreadable spans as [illegible]."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{encoded}"},
                            },
                        ],
                    }
                ],
            }
        ).encode()
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.load(response)
        transcripts[name] = payload["choices"][0]["message"]["content"]
    return transcripts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm", action="store_true", help="score the Qwen VLM instead of tesseract")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="codeexec image for engine mode")
    parser.add_argument("--out", default="", help="write the JSON report here as well")
    args = parser.parse_args()

    fixture_dir = Path(tempfile.mkdtemp(prefix="ocr_golden_"))
    fixtures = make_fixtures(fixture_dir)
    transcripts = (
        run_vlm(fixture_dir, fixtures) if args.vlm else run_engine(fixture_dir, fixtures, args.image)
    )
    results = score(fixtures, transcripts)
    report = {
        "mode": "vlm" if args.vlm else "engine",
        "fixtures": str(fixture_dir),
        "results": results,
        "advisory_thresholds": {
            "prose_clean.cer": "<= 0.02",
            "prose_degraded.cer": "<= 0.15",
            "table.hit_rate": ">= 0.95",
            "plot.hit_rate": ">= 0.85",
        },
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.out:
        Path(args.out).write_text(rendered)


if __name__ == "__main__":
    main()
