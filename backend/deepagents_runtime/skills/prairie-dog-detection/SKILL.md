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
     --out /outputs/rarespot_run \
     --no-spectral
   ```

   It runs on CPU; large surveys are slow but fine for set-and-forget. Use
   `--no-spectral` by DEFAULT on aerial/survey imagery (as above): the spectral mask
   scores the whole frame at 512 px and finds 0 detections on aerials, producing an
   EMPTY, mislabeled "stability ranking under spectral perturbation" plot. The
   **per-detection perturbation stability** (the coloured overlay + the report triage)
   is the FP signal to present — not the spectral plot. Only enable spectral (drop the
   flag) for close-range, single-subject imagery where it actually scores detections.
   Default config is 512 px tiles at 25% overlap; override with
   `--tile-size --tile-overlap --conf --iou` only if the user asks.
3. **Read the printed JSON summary** (`counts_by_class`, `stability.label_counts`,
   artifact paths). Artifacts are written under `/outputs/rarespot_run`.
4. **(Optional) Bounded false-positive second opinion — burrows + obvious-empty crops only.**
   The detector over-detects, so for a SMALL number of images you may get a grounded
   second opinion from the vision-reasoner on the least-reliable detections — but tightly
   bounded, and only where the VLM has real signal. **Skip this step for large multi-image
   surveys** (latency) and **skip it entirely if there is no false-positive tail**
   (0 `unstable` + 0 `borderline` when stability is present).
   - **Select at most 5 candidates** from `predictions.json` (read each box's
     `xyxy`, `class_name`, `confidence`, `stability_label`; do NOT re-detect). Eligible:
     **(a) burrow boxes** that are `unstable`, else the lowest-stability `borderline`;
     and **(b) any box clearly isolated on visually uniform ground** (dirt/grass/shadow).
     If `stability` is absent, fall back to the **lowest-confidence burrow rows**
     (the tail of the confidence-sorted `detections.csv`). **Do NOT send prairie_dog
     crops** — this VLM cannot resolve small cryptic animals (it returns "none" even on a
     4× crop), so a dog-crop verdict carries no usable signal. Skip degenerate crops.
   - **One `task()` to the vision-reasoner** listing the ≤5 candidates as
     `(index, class_name, score, bbox, source_image)`. Instruct it: for each, call
     `inspect_images(question, image_paths=[that candidate's OWN source image], bbox=that
     box, mode='grounded')` and ask "Centered here the detector flagged a <class>. Is that
     a real <class> or a false positive (bare ground/shadow/vegetation)? Cite ≥2 visual
     observations. End: VERDICT: real|false_positive|uncertain." Tell it: deep-read each,
     do NOT screen, do NOT exceed the list, do NOT hunt for missed detections, do NOT count.
   - **Reconcile as ADVISORY only — never change `counts_by_class`, never delete/add a
     detection, never overturn a trusted/high-confidence box:** a `real` verdict →
     "VLM-corroborated, keep"; a `false_positive` on a burrow or obvious-empty crop →
     surface it FIRST in the hand-verify list as "likely false positive (VLM-refuted,
     advisory)" with its cited observations, still counted; `uncertain` / no-signal →
     leave the detector's own label exactly as-is. The step only **reorders the human's
     review list** — it is labeled "advisory," never "confirmed."

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
  a *triage* signal — the unstable and borderline detections need hand-verification
  to estimate precision. If you ran the optional Step 4 second opinion, **lead the
  hand-verify list with the VLM-refuted candidates** (advisory) and their cited
  observations; otherwise just list the unstable/borderline detections to check.
  Surface the report's "Reliability & trust" section rather than re-deriving these
  numbers.
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
