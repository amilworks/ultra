---
name: ocr-extraction
description: Verbatim text extraction from images and video (text, tables, plot labels, subtitles) via the ocr-reader subagent and the tesseract+ffmpeg toolchain, plus the plot-digitization round-trip workflow. Use when a task needs the TEXT contained in pixels rather than a judgment about the image.
---

# OCR extraction

## When to use
- The deliverable is text that currently lives in pixels: scanned pages,
  screenshots, photographed documents, tables rendered as images, plot axis
  labels/ticks/legends, signage, handwriting, video subtitles or on-screen text.
- NOT for born-digital PDFs or ingested papers — the paper tools own those
  (`extract_paper_table_evidence` for tables). NOT for visual judgment
  (vision-reasoner) or object counting/measurement (specialist detectors).

## Routing
Delegate extraction to the **ocr-reader** subagent via the task tool with the
image/video path(s) and what text is needed. It owns the two-tier protocol
below and writes durable transcriptions under `/outputs/ocr/`. Only run the
toolchain directly yourself for a trivial single read.

## The two-tier protocol (what makes OCR trustworthy here)
1. **Engine first** for dense printed text: `tesseract <image> stdout` — or
   pytesseract with TSV output for per-word confidence and boxes. Deterministic,
   fast, offline.
2. **VLM second** (inspect_images) for what the engine cannot do — scene text,
   stylized/curved text, handwriting, plot text — and to cross-check
   decision-relevant spans.
3. **Agreement is confidence.** Engine and VLM agreeing → high. Disagreeing →
   report BOTH readings, confidence low; never silently pick one. Unreadable →
   `[illegible]`, never a plausible guess.
4. Bound every command with the execute tool's `timeout` parameter.

## Video
Prefer ffmpeg (preinstalled) for frame extraction:
- Slides/cuts (subtitle changes, scene changes):
  `ffmpeg -i in.mp4 -vf "select='gt(scene,0.30)'" -vsync vfr /workspace/frames/f%04d.png`
- Steady sampling when scene detection misses gradual text:
  `ffmpeg -i in.mp4 -vf fps=1/2 /workspace/frames/f%04d.png`
- Timestamps: add `-frame_pts 1` or map frame index through the sampling rate.
Then OCR frames with the two-tier protocol, dedupe identical text across
consecutive frames, and report each distinct text with its first-seen
timestamp. For >~2 minutes of video, sample first, inspect the frame count,
and chunk the OCR pass with progress checkpoints under /outputs.

## Plot digitization (round-trip verified)
Reading DATA out of a plot image is a coordinator workflow, not a single OCR
call:
1. ocr-reader extracts the calibration text: axis titles, tick values, legend.
2. In the sandbox, detect the plot area and trace the series with cv2/numpy.
3. Map pixel coordinates to data coordinates using the tick calibration.
4. **Round-trip check (required for decision-relevant digitization):** re-plot
   the extracted data with matplotlib and have vision-reasoner compare the
   re-render to the original — shapes, ranges, and crossings must agree. Report
   digitized values as approximate unless the round-trip agrees.

## Outputs
- `/outputs/ocr/<source>.txt` — plain transcription in reading order.
- `/outputs/ocr/<source>.json` — blocks: `{kind: heading|body|table|axis_label|
  legend|tick|caption, text, confidence, agreement: engine|vlm|both}`; tables
  additionally as TSV.
- Reference these files from the final answer; never paste long transcriptions
  into chat.
