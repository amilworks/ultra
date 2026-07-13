---
name: materials-sensor-data
description: Evidence-preserving workflow for governed materials sensor series stored in Zarr, including acoustic-emission waveforms, mechanical stress/strain channels, thermal and process telemetry, calibration, uncertainty, clocks, quality flags, coordinate transforms, and links to OME-NGFF imagery. Use for sensor ingestion, waveform or telemetry analysis, multimodal time alignment, downsampling, and sensor-to-microscopy correlation. The selected-resource tool validates a server-authored whole-tree identity plus the scientific sensor contract; do not claim object-store-native transport, a signal viewer, or cross-device clock synchronization that is not bound.
---

# Materials Sensor Data

## Preserve measurement meaning before analysis

Use the `ultra.sensor-series.v1` contract for one-dimensional, chunked materials measurements such
as acoustic emission, load/displacement/stress/strain, furnace or chamber telemetry, laser power,
photodiode traces, thermocouples, and frame-synchronous thermal-camera telemetry. OME-NGFF remains
the authoritative representation for multidimensional microscopy or thermal image pyramids. Link
those image resources to a sensor series by immutable resource ID and SHA-256; do not duplicate an
image stack as generic waveform channels. Sensor metadata only declares that link. It is returned
as `declared_unverified` until the sensor tool independently finds exactly one tenant-authorized,
run-selected catalog descriptor for the linked resource and its digest matches exactly.

The current runtime can validate selected Zarr v2/v3 groups, inspect every signal value in bounded
chunks, verify a deterministic tree manifest against the control-plane catalog digest, and generate
spike-preserving min/max envelopes through `inspect_selected_sensor_series`. Folder-upload
finalization rehashes the exact regular-file tree and propagates its closed identity through
selection, staging, and delegation. The catalog emits a digest-bound sensor marker only when a
bounded root-attribute read finds the exact `ultra.sensor-series.v1` schema; a generic `.zarr` or
OME-NGFF resource is not sensor-routed unless the prompt explicitly asks for sensor analysis. It
does not yet provide object-store-native transport, a
waveform signal viewer, or a general cross-device clock transform. State those remaining
integration gaps explicitly whenever a request depends on them. Legacy directory resources that
predate whole-tree identity remain `unbound` until re-finalized or re-uploaded.

Keep sensor-tree lineage and linked-resource lineage separate. `tree_verified` authenticates only
the selected sensor directory. `catalog_identity_verified` authenticates only a linked catalog
resource identity, not its NGFF schema, pixels, timing semantics, or scientific interpretation.
`verified_cross_resource_identity` requires the sensor tree to be verified and every declared
linked resource to be independently selected and digest-matched. Validate linked OME-NGFF content
with the NGFF reader before making multimodal claims.

The qualified semantic-unit registry covers the common SI/UCUM scales used by materials DAQ:
time, voltage, pressure/stress/modulus, strain, temperature, power, displacement/length, force,
current, frequency, linear velocity, acceleration, torque, energy, mass, and angle. It includes
ordinary scaled identities such as microvolts, millimetres, kilonewtons, milliamperes, kilohertz,
kilopascals/megapascals/gigapascals, and kelvin. The parser checks each exact UCUM/QUDT pair and its
QUDT quantity kind; it deliberately refuses an unregistered unit rather than accepting a
syntax-valid but dimensionally unknown identifier. Extend that closed registry with an
authoritative unit source and adversarial mismatch tests when a new instrument needs a quantity
outside this set.

## Required contract

Before calculating any material response, require and preserve:

- a unique `series_id`, modality, specimen/material identity, and source-resource lineage;
- one or more regular or explicit clocks in seconds, with sample count and a relative or
  instrument-local reference;
- quantified standard timing uncertainty and method, or an explicit reason it was not quantified;
- one-dimensional channel arrays whose length matches the referenced clock;
- a QUDT quantity-kind URI and UCUM code and/or QUDT unit URI for every channel;
- an applied identity or linear calibration, its revision, canonical parameter SHA-256, and any
  calibration-certificate SHA-256;
- scalar or array standard uncertainty, or an explicit reason uncertainty was not quantified;
- optional validity and saturation arrays with exact shape and Boolean type;
- named coordinate frames and homogeneous affine transforms for spatially located sensors;
- optional precomputed min/max multiscale levels whose factor, paths, shapes, and extrema close
  against the source channels; and
- immutable linked resources, such as an OME-NGFF thermal video, including its SHA-256 and shared
  frame-clock ID; these are declarations and must never contain caller-supplied verification
  status or authority.

Do not silently convert an uncalibrated ADC trace into volts, force to zero an omitted uncertainty,
infer units from a channel name, or treat acquisition index as calibrated time. Absolute UTC/TAI
coordinates are unsupported in v1 because epoch and clock-scale semantics are not represented.
Keep them unsupported until an explicit epoch/scale contract exists.

Read [the sensor contract reference](references/sensor-contract.md) when creating or auditing
metadata.

## Validate before using values

For a natural-prompt run, select the uploaded directory and call the bounded typed tool before
general execution:

```text
inspect_selected_sensor_series(validate_values=true,
                               envelope_channel_id="ae-1",
                               max_buckets=512)
```

The tool accepts only a run-selected resource ID, never a filesystem path; it returns bounded
metadata and envelopes rather than raw waveforms or host paths. For direct sandbox qualification,
open the staged group with the strict value scan enabled:

```python
from ultra_deepagents.sensors import open_sensor_series

series = open_sensor_series(
    "/workspace/test-run.sensor.zarr",
    expected_tree_manifest_sha256="<trusted out-of-band manifest SHA-256>",
    validate_values=True,
)
```

The expected manifest digest must come from the governed resource record or another trusted
out-of-band source. A digest found only inside the same untrusted directory is not an authority
binding. With an expected digest, require `series.lineage.status == "tree_verified"`; without one,
report `lineage.status == "unbound"` and do not describe the tree as tamper-evident.

The strict scan reads arrays in blocks of at most 65,536 values and validates finiteness,
timestamps, uncertainty, quality flags, and stored envelope extrema. One cumulative budget covers
every decoded value/byte/read across explicit clocks, signals, quality and uncertainty arrays, and
all multiscale proofs; the run-scoped tool also rejects an oversized whole-run plan before the
value-reading open. Local staging first enforces catalog and observed aggregate-byte, entry-count,
and wall-time limits before copying. Use
`validate_values=False` only for an explicit metadata preflight. That result has
`values_validated=false` and cannot support a numerical scientific claim.

Reject the series on any schema, shape, dtype, clock, unit, calibration-hash, reference, transform,
or manifest error. Report the validator's stable error code and field path; do not patch malformed
metadata in memory and continue as though the original resource passed.

## Analyze without erasing transients

For bounded previews or agent summaries, use `build_min_max_envelope` instead of stride sampling or
averaging:

```python
from ultra_deepagents.sensors import build_min_max_envelope

preview = build_min_max_envelope(
    values,
    max_buckets=2_000,
    validity=validity_flags,
    saturation=saturation_flags,
)
```

Each bucket retains source-index bounds, minimum/maximum values and indices, and invalid/saturated
counts. A saturated peak remains saturated evidence; it is not a measured peak amplitude. Invalid
values are excluded from extrema but counted. NaN is accepted only with a matching false validity
flag, and infinity is always rejected.

Preserve the raw array as the authority. Any filter, baseline subtraction, trigger, FFT, event
detector, strain conversion, compliance correction, emissivity correction, or time alignment is a
derived operation. Record exact parameters, software version, input/output hashes, and the retained
raw interval. Validate event detectors and peak measurements on synthetic spikes, known signals,
and held-out experimental records rather than only on the data used to tune thresholds.

## Align modalities only when the clock evidence allows it

Channels sharing the same clock may be compared at common sample coordinates. A linked thermal
video may be aligned only when its frame timing is explicitly bound to `frame_clock_id`. Different
instrument clocks require an independently justified offset/drift transform and uncertainty
propagation; v1 has no such transform. Do not correlate AE events, load frames, EBSD/SEM images, or
thermal frames across clocks merely because their arrays have similar lengths or nominal rates.

For stress-strain interpretation, retain whether each channel is raw load/displacement,
engineering stress/strain, or true stress/strain and preserve specimen geometry, extensometer/DIC
gauge definition, machine compliance correction, and their uncertainties outside the generic
channel contract. For thermal data, preserve emissivity, integration time, spectral response,
ambient/reflected-temperature assumptions, and calibration evidence. For AE, preserve sensor,
preamplifier, coupling, gain, bandwidth, threshold, trigger, and pre-trigger context. Missing
modality metadata limits interpretation even when the generic series validates.

## Accuracy and regression gates

Qualification must include acoustic-emission, stress-strain, and thermal-linked fixtures plus
adversarial cases for nonmonotone clocks, invalid units, calibration-hash mismatch, malformed
chunks, NaN/infinity, saturation, manifest tampering, symlinks, and Zarr v2/v3 I/O. Confirm full
value scans use bounded reads and metadata preflight reads no payload. Confirm a one-sample spike is
retained by every requested envelope budget.

Before claiming an analysis result, report separately:

1. resource/tree lineage status;
2. linked-resource catalog-identity status and whether every link was run-selected;
3. linked-content scientific validation status, such as independent OME-NGFF validation;
4. metadata-contract validity;
5. value-scan status;
6. clock/calibration/uncertainty adequacy for the intended quantity;
7. algorithm validation on synthetic and held-out evidence; and
8. remaining scientific limitations.

Write derived tables, plots, algorithm parameters, content hashes, and a machine-readable
validation record under `/outputs`. A series can be structurally valid while a fatigue,
fracture, thermal-history, or process-causality conclusion remains unverified.
