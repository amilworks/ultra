# `ultra.sensor-series.v1` reference

The contract lives in the Zarr root attributes at either
`attributes.ultra.sensor_series` or `attributes["ultra.sensor_series"]`.

## Top-level records

- `schema`: exactly `ultra.sensor-series.v1`.
- `series_id`, `modality`: non-empty identifiers.
- `specimen`: required closed identity record containing non-empty stable `specimen_id` and
  `material_id` identifiers. Unknown specimen fields are rejected rather than dropped.
- `clocks`: regular or explicit clock records.
- `channels`: calibrated one-dimensional measurement arrays.
- `coordinate_frames`, `coordinate_transforms`: optional spatial frames and homogeneous affine
  transforms.
- `multiscales`: optional stored min/max envelope levels.
- `linked_resources`: optional immutable resource-link declarations, including OME-NGFF thermal
  video. Each declaration contains only `role`, `resource_id`, `sha256`, and optional
  `frame_clock_id`; caller-supplied verification or authority fields are forbidden.
- `lineage.tree_manifest_path`: optional deterministic tree-manifest path.

## Clock records

A regular clock has `clock_id`, `kind="regular"`, `sample_count`, `reference`, a seconds
`time_unit`, `start_time_seconds`, and positive `sample_rate_hz`. An explicit clock replaces rate
and start with `timestamp_array`. Timestamps must be finite and strictly increasing. Supported
references are relative and instrument-local; absolute UTC/TAI requires a future epoch/clock-scale
contract.

`accuracy.status="quantified"` requires positive finite `standard_uncertainty_seconds` and a
method. `accuracy.status="not_quantified"` requires a reason and produces a warning.

## Channel records

Each channel binds `array`, `clock_id`, a QUDT `quantity_kind_uri`, a machine-readable unit, an
applied calibration, and uncertainty. The array must be one-dimensional, numeric, chunked, and
match the clock sample count. Calibration output unit must match channel unit. The
`parameters_sha256` is the canonical SHA-256 of the calibration mapping before that field is
inserted. Optional validity/saturation/uncertainty arrays must match the source exactly.

## Lineage authority levels

Sensor-tree lineage is independent of linked-resource lineage:

- `unbound`: no trusted digest was supplied.
- `manifest_verified`: trusted digest matched canonical manifest bytes, but file closure was not
  checked (in-memory parser path).
- `tree_verified`: the local opener matched the trusted digest and verified the full regular-file
  tree closure, excluding the manifest itself.

A manifest must be canonical JSON. Every entry is a relative regular-file path with exact byte
size and lowercase SHA-256. Symlinks, duplicate paths, traversal, missing/extra files, and content
mismatches fail validation.

At format-read time, every linked resource is `declared_unverified`; a SHA-256 written inside the
same sensor metadata is not an authority. The selected-resource tool may upgrade an individual
link to `catalog_identity_verified` only when the linked resource is also selected for the same
tenant-scoped run, has exactly one `control_resource_catalog` descriptor, and that descriptor's
SHA-256 exactly matches the declaration. A selected descriptor with a malformed identity or a
digest mismatch fails closed. An unselected link remains available as declared metadata but stays
unverified.

Overall `verified_cross_resource_identity` requires both a `tree_verified` selected sensor
descriptor and independently selected, digest-matched descriptors for every linked resource.
This status covers resource identity only. It does not validate the linked resource's OME-NGFF
schema, pixel values, clock semantics, coordinate transforms, or scientific interpretation; those
require the appropriate linked-content reader and a separate validation result.

## Current integration boundary

This package validates local Zarr v2/v3 sensor groups. It does not itself upload folders, store
bytes in an object store, register the schema in Postgres, expose a control-plane endpoint, render a
signal viewer, or estimate transforms between independent clocks. Existing OME-NGFF support owns
multidimensional image pyramids and can be linked by resource identity and digest.

The run-scoped local path is intentionally bounded rather than object-store-native. Before copying,
it enforces the catalog size plus an observed aggregate byte, entry, and wall-time preflight. Value
validation uses one cumulative decoded-value, decoded-byte, read-operation, and wall-time budget for
explicit clocks, signals, quality flags, uncertainty, and all stored multiscale proofs. A closed
`ultra.sensor-format-binding.v1` selected-resource marker is server-authored only after bounded root
attribute inspection and is bound to the resource SHA-256. File suffixes and generic OME-NGFF
metadata are not type authority.
