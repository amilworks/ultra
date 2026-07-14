---
name: materials-characterization-advanced
description: Accuracy-first workflow for measured XRD/Rietveld-profile validation and multimodal materials registration across EBSD, 4D-STEM, TEM, and APT. Use for observed-versus-calculated diffraction residuals, uncertainty-aware R factors, coordinate-frame transforms, calibration fiducials, held-out registration scoring, and advanced microscopy evidence audits. Provides deterministic profile metrics and rigid 2D/3D registration validation today; do not claim refinement, indexing, segmentation, reconstruction, chemical quantification, or feature matching that was not executed by a qualified engine.
---

# Advanced Materials Characterization

## Separate observations, models, and validation

Use measured detector data as the authority. Keep raw/processed/calculated arrays distinct, declare
each with an artifact ID, SHA-256, locator, units, and processing history, and retain masks and
calibration records. A low residual or registration error is evidence about one declared comparison;
it is not proof of phase identity, refinement uniqueness, indexing correctness, chemical accuracy,
or a causal microstructure relationship.

The qualified foundation currently performs two bounded operations:

- observed-versus-calculated powder-profile residual metrics; and
- proper rigid registration of known 2D or 3D point correspondences with a disjoint held-out set.

For an agent run, call `materials_calculate_diffraction_profile_metrics` or
`materials_fit_held_out_rigid_registration` directly before considering execute or code-runner.
Their schemas accept only bounded scientific arrays, frames/units, partitions, and closed caller
provenance declarations. They check digest syntax but do not resolve or re-hash source bytes. Copy
their exact `analysis_artifact.canonical_json` and
`materials_validation_artifact.canonical_json` strings to requested outputs; do not rediscover the
Python API or reconstruct the validation verdict.

Full Rietveld refinement, peak/profile fitting, EBSD indexing, 4D-STEM/TEM reconstruction, APT
reconstruction/quantification, segmentation, non-rigid registration, and automatic correspondence
discovery require their named engines and separate scientific qualification. Never substitute these
validators for those operations.

## Validate an XRD or Rietveld profile

Call `materials_calculate_diffraction_profile_metrics` only after the measured and calculated
intensities are on the same strictly increasing coordinate grid and have exactly matching units.
Supply distinct observed/calculated artifact IDs, exact SHA-256 digests, source locators, and
processing-history IDs, plus the inclusion mask, uncertainty array/semantics, refined-parameter
count, and independent-constraint count when those quantities are known.

Treat those digests as declarations until an independent catalog or resolver retrieves the exact
source bytes and replays SHA-256. The current typed tools therefore emit a passing numerical reducer
check plus required `materials.bounded_tool.provenance_bytes_bound=skip`, making overall
`scientific_status=unverified`; never upgrade this to verified from 64-hex syntax alone.

The mask is inclusion-valued: `True` contributes. Standard Rp is
`sum(abs(y_obs-y_calc))/sum(y_obs)`. Rwp uses inverse-variance weights only when supplied with
independent absolute one-standard-deviation uncertainties; otherwise it is explicitly unit-weighted
and non-statistical. Emit Rexp, χ², reduced χ², and goodness-of-fit only when those uncertainties and
positive `N-P+C` degrees of freedom are known. Do not derive counting uncertainties from a fitted
curve and then describe them as independent observations.

Report the radiation source/wavelength, instrument geometry and calibration standard, coordinate
definition, background/profile model, phase/model set, refined parameters/constraints, excluded
regions and rationale, detector corrections, and provenance. If an external refinement engine was
not actually run, say “profile comparison,” not “Rietveld refinement.”

## Register characterized volumes without leakage

Use `materials_fit_held_out_rigid_registration` for known corresponding landmarks only. Supply
distinct source/target frame IDs, identical explicit coordinate units, caller-declared provenance
for both point arrays, and fixed calibration/held-out index lists that together cover every row.

The fit uses a Kabsch/SVD proper rotation and translation from source to target. Source and target
frames must be distinct, units must already match, calibration points must span the full dimension,
and calibration/held-out indices must be unique and disjoint. A reflection-only optimum fails
closed. Treat held-out RMSE, mean, median, maximum, residual plots, and their physical scale as the
primary generalization evidence; calibration residual alone is not sufficient.

This primitive does not prove that landmarks correspond. Record how correspondences were obtained,
operator blinding, localization uncertainty, distortion corrections, reconstruction versions, and
selection criteria. When tuning landmarks or transforms after seeing held-out errors, invalidate the
holdout and create a new untouched test set.

## Modality-specific minimum evidence

- **Measured XRD/Rietveld:** raw counts, calibration standard, wavelength/spectrum, geometry,
  background and profile functions, phase/model candidates, parameter covariance/correlation,
  residual-versus-angle structure, and an external reference or held-out pattern.
- **EBSD:** detector geometry/pattern center, phase dictionary, indexing engine/version, confidence
  or pattern-quality fields, cleanup steps, crystal/sample/vendor orientation convention, spatial
  step and units, and held-out raw-pattern reindexing checks.
- **4D-STEM/TEM:** detector calibration, accelerating voltage/camera length, scan and reciprocal
  frames, distortion/rotation corrections, dose and drift, masks, reconstruction algorithm and
  regularization, convergence evidence, and simulated/standard or held-out validation.
- **APT:** reconstruction parameters/version, detector efficiency, ranging decisions and overlaps,
  background/correction model, spatial calibration, local-magnification limitations, uncertainty,
  and independent microscopy or standard comparison.

Do not treat a visualization as a calibrated measurement. Preserve masks, saturation, missing data,
uncertainty, transforms, and every raw-to-derived processing step.

## Accuracy and regression gates

For profile metrics, test analytical Rp/Rwp/Rexp/χ² cases, masked points, zero denominators,
nonfinite values, mismatched grids/units, uncertainty semantics, and invalid degrees of freedom. For
registration, test known 2D and 3D transforms, noise, frame/unit mismatch, train/holdout leakage,
reflections, duplicate/out-of-range indices, and rank-deficient or nearly collinear calibration.

Benchmark representative array sizes before broad sweeps. Save scripts, immutable inputs,
per-point residuals, transforms, calibration and held-out tables, plots, package versions, and
`materials_validation.json` under `/outputs`. Report run completion, internal numerical validation,
held-out experimental validation, and scientific conclusion status separately.
