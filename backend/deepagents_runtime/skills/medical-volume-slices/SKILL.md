---
name: medical-volume-slices
description: Turn a 3D/4D medical imaging volume (CT, MRI, fMRI, DICOM series, or tractography) into clean, VLM-readable 2D anatomical slices, then have the vision-reasoner inspect them. Use when the user uploads or asks about a NIfTI (.nii/.nii.gz), DICOM, or tractography (.trk/.tck) volume — head CT, brain MRI, fMRI, diffusion/tractography, or similar — and wants it described, assessed, or visually inspected.
---

# Medical volume slicing (3D/4D → VLM-optimal slices)

## When to use
The user has a **3D or 4D medical imaging volume** — a NIfTI (`.nii`/`.nii.gz`), a DICOM
series/folder, or tractography (`.trk`/`.tck`) — and wants it inspected, described, or
assessed (e.g. "describe this head CT", "is there ventriculomegaly", "look at this MRI").
A vision-language model cannot read a `.nii.gz` volume directly; it needs 2D slices. **Do
NOT hand-write slice-extraction code** — it tends to produce multi-panel montages that lose
resolution when the VLM downscales them. Use the bundled CLI, which renders single-panel,
upright, correctly-proportioned, modality-windowed slices at anatomically-chosen levels.

## Workflow
1. **Stage the volume** into `/workspace` (`stage_uploaded_files_for_analysis` for an upload,
   `stage_resource_for_analysis` for a catalog resource).
2. **Run the slicer** in one `execute()` call:
   ```bash
   python /opt/medvol/volume_slices.py \
     --input /workspace/staged_uploads/<file> \
     --out /outputs/volume_slices \
     --n-axial 6 --n-coronal 4 --n-sagittal 2
   ```
   It auto-detects modality (CT → brain/bone windows; MRI → robust normalization; 4D fMRI →
   temporal-mean anatomy; tractography → track-density projection), reorients to a consistent
   upright view, and prints a JSON manifest with the slice paths grouped by plane. Tune the
   counts to the question (more axials for ventricle/lesion assessment; add `--planes` to
   restrict). Override modality with `--modality` only if auto-detection is wrong.
3. **Delegate to the vision-reasoner as a SCREEN-then-FOCUS two-pass — never "inspect every
   slice."** In ONE `task()` call, give it the slice directory (`/outputs/volume_slices`), the
   precise clinical question, and this explicit instruction:
   - first **screen the WHOLE slice set in a single `screen_images` pass** (one fast batched
     call) to get a one-line read of each slice and flag the few that bear on the question;
   - then **deep-read with `inspect_images` ONLY the 2–3 decisive slices** the screen flags
     (for a ventricle / NPH question, the mid-ventricle axial brain-window slices; for a lesion,
     the slice(s) that show it).
   Do **NOT** tell it to inspect each slice or hand it a long "inspect all of these" list:
   looping deep `inspect_images` over a whole stack bloats the vision-reasoner's context and
   **stalls its final synthesis** (a real failure mode). Screen-first keeps its context small.
4. **Synthesize** the vision-reasoner's findings into the answer.

## Reading the slices (orientation)
Each slice is captioned with its plane, world-coordinate (mm), and up-direction (axial:
anterior=top; coronal/sagittal: superior=top). Orientation follows canonical RAS — **judge
left/right by symmetry, not absolute side**, unless the caption is verified. CT slices come in
brain and (for axials) bone windows; pick the window that shows the structure of interest.

## Medical safety — this is a qualitative impression, NOT a diagnosis
Imaging interpretation here is a **descriptive, qualitative screening impression to support a
clinician — never a diagnosis**. The vision-language model:
- **cannot measure** (no Evans index, ventricular volume, midline shift, or any metric as fact);
- **cannot reliably distinguish** look-alike entities (e.g. NPH vs. age-related atrophy — both
  show enlarged ventricles; the distinction needs measured indices + DESH + clinical context);
- should report what is **visible** (e.g. "the lateral ventricles appear prominent") with explicit
  uncertainty, and **recommend the appropriate measurement and clinical/radiological correlation**.

State these limits in the answer. Do **not** present a confident diagnostic yes/no. For 4D fMRI
or diffusion data, note that statistical/functional analysis is not assessable from raw slices —
only anatomy is shown.

## Output: a brief clinical-imaging impression in plain prose
Lead with what the images show and the bottom-line impression, then the key caveats (modality/slice
limits, age, missing clinical context) and the recommended next step. Keep it **readable clinical
prose** — this is a qualitative imaging impression, NOT a computational experiment. Do **NOT**:
- produce a multi-section verification/"report contract" write-up, or pull in `scientific-reporting`
  (that contract is for computational experiments — figures, CSVs, reproducible metrics — not an
  imaging read);
- compute the same index (e.g. Evans) several different ways and tabulate the variants — automated
  segmentation on thick-slice CT is approximate; give one honest approximate value with its caveat;
- attach per-claim **ESTABLISHED / SUGGESTED / UNKNOWN** calibration labels to findings.

The vision-reasoner already provides the qualitative read; your job is to synthesize it into one clear,
calm answer a clinician can read — not to re-derive every number or grade every sentence.
