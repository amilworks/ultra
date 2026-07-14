"""Immutable records returned by the sensor-series validator.

These records describe validated metadata and validation outcomes. They intentionally do not
claim that a viewer, object store, or control-plane API exists for the series.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnitRef:
    """A unit bound to at least one machine-readable authority."""

    label: str
    ucum_code: str | None
    qudt_uri: str | None


@dataclass(frozen=True)
class ClockRecord:
    """A validated sampling coordinate and its retained timing-accuracy statement.

    V1 supports relative or instrument-local seconds. Absolute UTC/TAI coordinates are
    rejected until an epoch and clock-scale transform can be represented without ambiguity.
    """

    clock_id: str
    kind: str
    sample_count: int
    reference: str
    time_unit: UnitRef
    accuracy_status: str
    standard_uncertainty_seconds: float | None
    accuracy_method: str | None
    accuracy_reason: str | None
    sample_rate_hz: float | None = None
    start_time_seconds: float | None = None
    timestamp_array: str | None = None


@dataclass(frozen=True)
class CalibrationRecord:
    """An applied calibration where stored output = ``scale * input + offset``."""

    kind: str
    calibration_id: str
    revision: str
    input_unit: UnitRef
    output_unit: UnitRef
    scale: float
    offset: float
    parameters_sha256: str
    certificate_sha256: str | None = None


@dataclass(frozen=True)
class ChannelRecord:
    """A calibrated 1-D measurement channel with uncertainty and quality provenance."""

    channel_id: str
    name: str
    array_path: str
    clock_id: str
    quantity_kind_uri: str
    unit: UnitRef
    calibration: CalibrationRecord
    uncertainty_kind: str
    uncertainty_value: float | None
    uncertainty_array: str | None
    uncertainty_reason: str | None
    validity_array: str | None
    saturation_array: str | None
    coordinate_frame_id: str | None
    invalid_count: int | None
    saturation_count: int | None


@dataclass(frozen=True)
class CoordinateAxis:
    name: str
    unit: UnitRef
    quantity_kind_uri: str | None


@dataclass(frozen=True)
class CoordinateFrame:
    frame_id: str
    axes: tuple[CoordinateAxis, ...]


@dataclass(frozen=True)
class CoordinateTransform:
    transform_id: str
    kind: str
    input_frame_id: str
    output_frame_id: str
    matrix: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class EnvelopeChannel:
    channel_id: str
    min_array: str
    max_array: str


@dataclass(frozen=True)
class MultiscaleLevel:
    level: int
    kind: str
    factor: int
    channels: tuple[EnvelopeChannel, ...]


@dataclass(frozen=True)
class LinkedResource:
    """A sensor-declared link whose catalog authority is resolved by the run tool.

    Format parsing can preserve a declaration, but it cannot authenticate another
    catalog resource. ``catalog_identity_verified`` is therefore issued only by
    the selected-resource tool after it matches a second, tenant-authorized run
    descriptor. It verifies resource identity, not the linked content's scientific
    interpretation.
    """

    role: str
    resource_id: str
    sha256: str
    frame_clock_id: str | None
    verification_status: str
    verification_authority: str | None


@dataclass(frozen=True)
class LineageBinding:
    """Authority level for the deterministic tree manifest.

    ``unbound`` means no trusted, out-of-band digest was supplied. ``manifest_verified``
    means that digest matched canonical manifest bytes, but no file closure was checked.
    ``tree_verified`` is issued only by the local opener after it checks the complete regular-
    file directory closure with :func:`verify_tree_manifest`.
    """

    status: str
    tree_manifest_path: str | None
    expected_tree_manifest_sha256: str | None
    computed_tree_manifest_sha256: str | None
    entry_count: int | None


@dataclass(frozen=True)
class SpecimenRecord:
    """Stable specimen and material identities retained with every sensor series."""

    specimen_id: str
    material_id: str


@dataclass(frozen=True)
class SensorSeries:
    schema: str
    series_id: str
    modality: str
    specimen: SpecimenRecord
    clocks: tuple[ClockRecord, ...]
    channels: tuple[ChannelRecord, ...]
    coordinate_frames: tuple[CoordinateFrame, ...]
    coordinate_transforms: tuple[CoordinateTransform, ...]
    multiscales: tuple[MultiscaleLevel, ...]
    linked_resources: tuple[LinkedResource, ...]
    lineage: LineageBinding
    warnings: tuple[str, ...]
    values_validated: bool


@dataclass(frozen=True)
class EnvelopeBucket:
    start_index: int
    stop_index: int
    minimum: float | None
    maximum: float | None
    minimum_index: int | None
    maximum_index: int | None
    valid_count: int
    invalid_count: int
    saturation_count: int


@dataclass(frozen=True)
class MinMaxEnvelope:
    source_count: int
    factor: int
    buckets: tuple[EnvelopeBucket, ...]


@dataclass(frozen=True)
class TreeManifestVerification:
    entry_count: int
    size_bytes: int
