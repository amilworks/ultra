---
name: materials-characterization
description: Materials characterization workflow for simulated or experimental XRD/powder diffraction, SAED and diffraction dictionaries, raw Kikuchi-pattern handling, Raman, XPS, and EDS spectra. Use for peak generation/fitting/indexing, calibration, background and broadening choices, phase-evidence comparison, hyperspectral axes, and instrument-aware uncertainty. Uses pymatgen, diffsims, kikuchipy, HyperSpy/rsciio when available; does not claim Rietveld refinement or database identification without an installed validated engine/reference.
---

# Materials Characterization

## When to use

Use this skill for calculated or measured XRD/diffraction patterns, SAED/diffraction simulations,
Kikuchi-pattern preprocessing/indexing, Raman, XPS, and EDS spectra. For EBSD orientation maps,
IPF/texture, grains, and boundaries also read `/skills/computational-materials/SKILL.md`. For CIF
symmetry or CALPHAD read `/skills/materials-structure-thermo/SKILL.md`.

The release stack supports pymatgen XRD simulation, `diffsims`, `kikuchipy`, and the
HyperSpy/rsciio stack brought by the characterization packages. Verify optional signal-type
extensions at runtime before using them. No installed powder-refinement package or licensed phase
reference database should be assumed.

## Capability maturity

- **Validated executable path:** idealized powder-XRD simulation from a supplied, provenance-bound
  structure with pymatgen, including radiation/wavelength, reflection order, peak-table, and
  deterministic control checks.
- **Research preview only:** measured-XRD calibration/fitting, SAED/diffraction-library matching,
  raw Kikuchi indexing, Raman, XPS, and EDS. The installed readers and processing libraries enable
  bounded analysis, but Ultra does not yet have held-out reference-data qualification for these
  workflows. Report their scientific status as `unverified` unless task-specific independent
  controls are supplied and pass.
- **Unsupported without an added validated engine/database:** Rietveld refinement, quantitative
  phase analysis, automatic phase identification, and licensed reference-database search.

Do not turn a scientifically careful checklist into a capability claim. A library import or a
plausible plot is not evidence that an experimental characterization workflow is calibrated or
validated.

## Required workflow

### 1. Preserve acquisition context and calibration

- Stage the original file unchanged; record path/artifact ID, SHA-256, format, shape/dtype, axis
  names, calibration scale/offset/units, acquisition date when present, instrument/detector, and
  sample geometry.
- Prefer the format's named reader through HyperSpy/rsciio or the vendor-supported parser. Do not
  flatten a calibrated spectrum to row number or manually scrape a binary/XML format when a reader
  is available.
- Save raw counts separately from transformed, background-subtracted, normalized, smoothed, or
  fitted data. Record every transformation in order and preserve parameters.
- Verify at least two known coordinates/peaks or a supplied calibration standard when making a
  calibrated-axis claim. If calibration metadata is absent, report channel/index units.

### 2. X-ray diffraction

For a simulated powder pattern, use `pymatgen.analysis.diffraction.xrd.XRDCalculator` and the
[XRD recipe](references/xrd-recipes.md). Record structure provenance/occupancies, radiation source
and wavelength, geometry, two-theta range, polarization/broadening assumptions, intensity
normalization, and hkl/multiplicity. Emit a machine-readable peak table. Call the result
**simulated**; it is not experimental validation of a phase.

For measured XRD, record the instrument wavelength/optics and justify background model, K-alpha2
handling, zero shift, peak shape, smoothing, and detection threshold. Fit on count-like data with
an error model when counts are available; report parameter uncertainty and residuals, not only R2.
Compare phase candidates by multiple diagnostic reflections and chemistry constraints. A single
peak match or visual resemblance is not phase identification.

Do not claim Rietveld refinement, quantitative phase analysis, or reference-database search unless
an actual validated engine/database is present and named. Pymatgen's calculated stick pattern is
not a Rietveld engine and does not model the instrument by default.

### 3. Electron diffraction and Kikuchi patterns

Use `diffsims` for simulated diffraction libraries and `kikuchipy` for detector geometry,
background correction, pattern processing, and indexing. Record accelerating voltage, camera
length/detector calibration, crystal phase/structure, reciprocal-space convention, zone axis,
orientation sampling, dynamical/kinematical assumption, and matching metric. Validate indexing on
held-out or synthetic known orientations and report angular error/distribution; a visually aligned
overlay is supporting evidence, not a calibrated error estimate.

### 4. Raman, XPS, and EDS

- Raman: record laser wavelength/power, integration/accumulations, spectral resolution, axis
  calibration, cosmic-ray removal, fluorescence/background model, and peak line shape. Report
  peak center/FWHM uncertainty and resolution limits. Do not assign a phase from one band alone.
- XPS: retain counts and binding-energy direction; record charge-reference choice, background
  (for example Shirley/Tougaard only when justified), sensitivity factors, line-shape constraints,
  spin-orbit separation/area constraints, and fit residuals. Label atomic percentages unverified
  if sensitivity/transmission corrections are missing.
- EDS: record detector, beam energy, live time, takeoff geometry, dead time, standards or k-factor/
  correction method, absorption/fluorescence assumptions, overlaps, and detection limits. A peak
  label is not quantitative composition. Align maps on a verified physical grid before combining
  with EBSD or morphology.

For all spectra, distinguish detection, identification, and quantification. Each requires stronger
evidence than the previous level. Carry calibration and fit uncertainty into reported peak and
composition quantities.

### 5. Scientific validation record

Every scientific claim must write `/outputs/materials_validation.json`, separate from run success.
Generate it with `ultra_deepagents.materials.validation.ValidationCheck`, `EvidenceArtifact`,
`assess_scientific_status`, and `canonical_record_json`. The production image asserts this import
at build time. If it fails, report release-infrastructure failure and do not hand-author a verdict.

Each check contains `validator_id`, `outcome` (`pass`, `fail`, `skip`), `observed`, `expected`,
`units`, `tolerance_rationale`, `required`, `critical`, `library_versions`, `evidence` (list of
`name`/`sha256`/`path` or `artifact_id`, optionally `size_bytes`), and `message`. The top-level
record contains `schema_version`, `run_status`, `scientific_status` (`verified`, `failed`,
`unsupported`, `unverified`), `verified`, `silent_success`, `capability_supported`,
`required_validator_ids`, `missing_validator_ids`, `critical_failures`,
`contradiction_failures`, `reasons`, and `checks`:

```json
{
  "schema_version": "1",
  "run_status": "succeeded",
  "scientific_status": "verified",
  "verified": true,
  "silent_success": false,
  "capability_supported": true,
  "required_validator_ids": ["materials.xrd.fcc_ni_cuka.v1"],
  "missing_validator_ids": [],
  "critical_failures": [],
  "contradiction_failures": [],
  "reasons": [],
  "checks": [
    {
      "validator_id": "materials.xrd.fcc_ni_cuka.v1",
      "outcome": "pass",
      "observed": {"first_two_theta_deg": 44.59, "first_hkl": [1, 1, 1]},
      "expected": {"two_theta_deg": [44.0, 45.2], "hkl": [1, 1, 1]},
      "units": "degree 2theta",
      "tolerance_rationale": "Regression range covers lattice and wavelength rounding",
      "required": true,
      "critical": true,
      "library_versions": {"pymatgen": "<version>"},
      "evidence": [
        {"name": "xrd_peaks.csv", "sha256": "0000000000000000000000000000000000000000000000000000000000000000", "path": "/outputs/xrd_peaks.csv"}
      ],
      "message": "Calculated fcc-Ni control begins with the 111 reflection"
    }
  ]
}
```

The all-zero digest is a documentation placeholder only. Replace it with the final artifact's
actual lowercase SHA-256; trace inspection matches it to a durable `/outputs` artifact and rejects
digest or size mismatch. Keep the complete record below the 1,000,000-byte trace limit.

Declare required validators before running the analysis. Missing/skipped required checks and
incomplete evidence produce `unverified`; failed scientific invariants produce `failed`; absent
refinement engines/reference databases produce `unsupported`. Set/report `silent_success` when the
run succeeded but science failed. Hash exact final evidence files.

### 6. Outputs and user-facing status

Write raw-input manifests/hashes, calibrated/preprocessed data, peak tables, fit parameters and
covariance/intervals, residuals, plots, scripts, package versions, and
`materials_validation.json` to `/outputs`. Report processing choices beside the results they
affect. State run status and scientific status separately, list every failed/skipped/missing
required validator, distinguish simulated from measured evidence, and include limitations.
