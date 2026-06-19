---
name: prairie-dog-detection
description: Run RareSpot prairie-dog and burrow detection on field images and present per-detection reliability for an ecologist — tiled YOLOv5 inference plus perturbation-consensus stability scoring, an energy-adaptive spectral mask, and an EXIF survey map. Use when the user asks to detect, count, or quantify prairie dogs or burrows in one or more images, run RareSpot, or survey colony activity — i.e. running ecological image detection, not writing it up.
---

# Prairie-dog detection (RareSpot)

## When to use
Use this skill when the user wants to **detect, count, or quantify prairie dogs
or burrows** in field imagery, run **RareSpot**, or produce a **colony survey** —
e.g. "run prairie dog detection on this image", "how many burrows are in these
photos", "survey colony activity across the run". For *report-only* follow-ups
("write a combined report across the RareSpot runs in this chat") do **not** re-run
detection — inspect the prior artifacts instead.

## The pipeline is bundled and calibrated — run it, do not re-implement
The full RareSpot pipeline (image tiling → YOLOv5 detection → NMS → per-detection
**perturbation-consensus stability** → **energy-adaptive spectral mask** → Hungarian
matching → EXIF **survey map** → reliability report) is baked into this sandbox at
`/opt/rarespot`. It is calibrated, scientifically load-bearing, and was hardened
against a known false-positive problem. **Do NOT re-derive tiling, NMS, stability,
or the spectral mask in your own code — a prose re-derivation regresses the science.**
Run the bundled CLI.

## Workflow
1. **Get the image(s) into `/workspace`.** For images in the user's Resources
   catalog, `search_resources` then `stage_resource_for_analysis(resource_ids)`
   (lands them under `/workspace/staged_resources/`). For images attached to this
   chat, use `stage_uploaded_files_for_analysis`. You may stage many images — the
   pipeline batches them into one run with one aggregate report and survey map.
2. **Run the bundled detector in ONE `execute()` over all staged images** (a single
   detect pass over all tiles — do not fan out per image):

   ```bash
   YOLOv5_AUTOINSTALL=false python /opt/rarespot/rarespot_detect.py \
     --images /workspace/staged_resources \
     --weights /opt/rarespot/RareSpotWeights.pt \
     --yolov5 /opt/rarespot/yolov5 \
     --out /outputs/rarespot_run
   ```

   It runs on CPU; large surveys are slow but fine for set-and-forget. For a very
   large batch where the spectral pass is too slow, add `--no-spectral` (you keep
   detection + stability). Default config is 512 px tiles at 25% overlap; override
   with `--tile-size --tile-overlap --conf --iou` only if the user asks.
3. **Read the printed JSON summary** (`counts_by_class`, `stability.label_counts`,
   `top_spectral_review_candidates`, artifact paths). Artifacts are written under
   `/outputs/rarespot_run`.

## Presenting results for an ecologist
- **Lead with the per-detection reliability triage**, not raw counts: `trusted`
  (stability ≥ 0.75), `borderline` (≥ 0.5), `unstable` (< 0.5), overall and per
  class — these tell the ecologist which detections to trust.
- **Embed the stability overlay inline** as a markdown image (boxes coloured
  green = trusted / amber = borderline / red = likely false positive); also link
  the class-coloured overlay and the CSV/report.
- **Always include the honest reliability note:** this detector has **no held-out
  validation set** (trained with mAP) and is known to **over-detect**, so
  confidence is a *relative* score (not a calibrated probability) and stability is
  a *triage* signal — recommend hand-verifying the unstable and borderline
  detections to estimate precision. Surface the report's "Reliability & trust"
  section rather than re-deriving these numbers.
- **For multi-image surveys with EXIF GPS,** embed the survey map inline and report
  the spatial metrics (extent, image spacing, totals, prairie-dog:burrow ratio) so
  the ecologist sees where colony activity concentrates, not just per-image counts.

## Artifacts and hygiene
The pipeline writes overlays, `detections.csv`, `predictions.json`, the reliability
`report.md`, and the survey map under `/outputs/rarespot_run` (collected as durable
run artifacts). Answer from these — do **not** create stub/duplicate CSV/JSON/figure
copies of the results. Only produce a new artifact for *derived* synthesis across
multiple runs. See `scientific-reporting` for report contract and figure standards,
and `computational-experiment-rigor` for verification discipline.
