"""Strict, format-level validation for chunked materials sensor series.

This module defines ``ultra.sensor-series.v1`` as a small metadata convention on top of
Zarr arrays. It complements OME-NGFF instead of replacing it: thermal image pixels remain
OME-NGFF, while calibrated detector telemetry, frame timestamps, acoustic waveforms, and
mechanical curves can live in a linked sensor-series group.

The parser is deliberately fail-closed about units, clocks, calibration, uncertainty, and
quality flags. It never treats a self-declared digest as server authority. A caller may pass
an out-of-band tree-manifest SHA-256; only a match to a deterministic manifest can upgrade
lineage from ``unbound`` to ``manifest_verified``. :func:`open_sensor_series` additionally
verifies every regular file in the directory closure before returning ``tree_verified``.

This milestone is a reader/validator only. It does not provide object-store access, a
control-plane endpoint, PostgreSQL extraction, or a signal viewer.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, cast

from ultra_deepagents.sensors.model import (
    CalibrationRecord,
    ChannelRecord,
    ClockRecord,
    CoordinateAxis,
    CoordinateFrame,
    CoordinateTransform,
    EnvelopeBucket,
    EnvelopeChannel,
    LineageBinding,
    LinkedResource,
    MinMaxEnvelope,
    MultiscaleLevel,
    SensorSeries,
    SpecimenRecord,
    TreeManifestVerification,
    UnitRef,
)

SENSOR_SCHEMA = "ultra.sensor-series.v1"
TREE_MANIFEST_SCHEMA = "ultra.tree-manifest.v1"
CANONICAL_JSON_SCHEMA = "ultra.canonical-json.v1"

# A chunk may be larger than its array at the edge, but accepting a declared decoded chunk
# larger than this would make even a bounded metadata validation vulnerable to a huge read.
MAX_DECODED_CHUNK_BYTES = 64 * 1024 * 1024
VALIDATION_BLOCK_VALUES = 65_536
MAX_SENSOR_ROOT_ATTRIBUTES_BYTES = 4 * 1024 * 1024
MAX_TREE_MANIFEST_BYTES = 64 * 1024 * 1024

# Aggregate validation limits apply to every decoded array read in one parse, including
# explicit clocks, signal values, quality flags, uncertainty arrays, and every stored
# multiscale envelope verification.  The per-chunk limit above is necessary but not
# sufficient: without a shared budget, many individually bounded reads can still amplify
# into an unbounded validation job.
DEFAULT_VALIDATION_MAX_VALUES = 64_000_000
DEFAULT_VALIDATION_MAX_DECODED_BYTES = 512 * 1024 * 1024
DEFAULT_VALIDATION_MAX_READS = 4_096
DEFAULT_VALIDATION_MAX_WALL_SECONDS = 30.0


@dataclass
class SensorValidationBudget:
    """One cumulative admission and accounting budget for sensor array validation.

    ``ensure_scan_fits`` admits a complete array scan before its first slice is read;
    ``consume_slice`` then records the reads that actually occurred.  This keeps failures
    fail-closed before a single oversized scan while retaining truthful success counters.
    The same instance must be threaded through the whole parse.
    """

    max_values: int = DEFAULT_VALIDATION_MAX_VALUES
    max_decoded_bytes: int = DEFAULT_VALIDATION_MAX_DECODED_BYTES
    max_reads: int = DEFAULT_VALIDATION_MAX_READS
    max_wall_seconds: float = DEFAULT_VALIDATION_MAX_WALL_SECONDS
    decoded_values: int = field(default=0, init=False)
    decoded_bytes: int = field(default=0, init=False)
    read_operations: int = field(default=0, init=False)
    _started_at: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("max_values", "max_decoded_bytes", "max_reads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.max_wall_seconds, bool)
            or not isinstance(self.max_wall_seconds, int | float)
            or not math.isfinite(float(self.max_wall_seconds))
            or float(self.max_wall_seconds) <= 0
        ):
            raise ValueError("max_wall_seconds must be finite and positive")
        self._started_at = time.monotonic()

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self._started_at)

    def check_wall(self, path: str) -> None:
        if self.elapsed_seconds > float(self.max_wall_seconds):
            _fail(
                "validation_budget_exceeded",
                path,
                f"sensor validation exceeded the {self.max_wall_seconds:g} second wall budget",
            )

    @staticmethod
    def _itemsize(array: Any, path: str) -> int:
        try:
            itemsize = operator.index(getattr(getattr(array, "dtype", None), "itemsize"))
        except TypeError:
            _fail("invalid_array_dtype", path, "array dtype must expose a positive item size")
        if itemsize <= 0:
            _fail("invalid_array_dtype", path, "array dtype must expose a positive item size")
        return itemsize

    def _ensure_fits(self, *, values: int, decoded_bytes: int, reads: int, path: str) -> None:
        self.check_wall(path)
        projected = (
            ("decoded value", self.decoded_values + values, self.max_values),
            ("decoded byte", self.decoded_bytes + decoded_bytes, self.max_decoded_bytes),
            ("array read", self.read_operations + reads, self.max_reads),
        )
        for label, total, limit in projected:
            if total > limit:
                _fail(
                    "validation_budget_exceeded",
                    path,
                    f"sensor validation {label} budget would be exceeded: {total} > {limit}",
                )

    def ensure_plan_fits(self, *, values: int, reads: int, path: str = "validation_plan") -> None:
        """Reject a metadata-derived whole-run plan before any value array is read."""

        self._ensure_fits(values=values, decoded_bytes=0, reads=reads, path=path)

    def ensure_scan_fits(self, array: Any, count: int, path: str) -> None:
        reads = math.ceil(count / VALIDATION_BLOCK_VALUES) if count else 0
        self._ensure_fits(
            values=count,
            decoded_bytes=count * self._itemsize(array, path),
            reads=reads,
            path=path,
        )

    def consume_slice(self, array: Any, count: int, path: str) -> None:
        decoded_bytes = count * self._itemsize(array, path)
        self._ensure_fits(values=count, decoded_bytes=decoded_bytes, reads=1, path=path)
        self.decoded_values += count
        self.decoded_bytes += decoded_bytes
        self.read_operations += 1

    def snapshot(self) -> dict[str, int | float]:
        return {
            "decoded_values": self.decoded_values,
            "decoded_bytes": self.decoded_bytes,
            "read_operations": self.read_operations,
            "elapsed_seconds": self.elapsed_seconds,
            "max_values": self.max_values,
            "max_decoded_bytes": self.max_decoded_bytes,
            "max_reads": self.max_reads,
            "max_wall_seconds": float(self.max_wall_seconds),
        }


_ROOT_FIELDS = frozenset(
    {
        "schema",
        "series_id",
        "modality",
        "specimen",
        "clocks",
        "channels",
        "coordinate_frames",
        "coordinate_transforms",
        "multiscales",
        "linked_resources",
        "lineage",
    }
)
_UNIT_FIELDS = frozenset({"label", "ucum_code", "qudt_uri"})
_CLOCK_COMMON_FIELDS = frozenset(
    {"clock_id", "kind", "sample_count", "reference", "time_unit", "accuracy"}
)
_CLOCK_REGULAR_FIELDS = _CLOCK_COMMON_FIELDS | {"sample_rate_hz", "start_time_seconds"}
_CLOCK_EXPLICIT_FIELDS = _CLOCK_COMMON_FIELDS | {"timestamp_array"}
_ACCURACY_QUANTIFIED_FIELDS = frozenset({"status", "standard_uncertainty_seconds", "method"})
_ACCURACY_UNQUANTIFIED_FIELDS = frozenset({"status", "reason"})
_CALIBRATION_FIELDS = frozenset(
    {
        "kind",
        "applied",
        "calibration_id",
        "revision",
        "input_unit",
        "output_unit",
        "scale",
        "offset",
        "parameters_sha256",
        "certificate_sha256",
    }
)
_FRAME_FIELDS = frozenset({"frame_id", "axes"})
_COORDINATE_AXIS_FIELDS = frozenset({"name", "unit", "quantity_kind_uri"})
_TRANSFORM_FIELDS = frozenset(
    {"transform_id", "kind", "input_frame_id", "output_frame_id", "matrix"}
)
_CHANNEL_FIELDS = frozenset(
    {
        "channel_id",
        "name",
        "array",
        "clock_id",
        "quantity_kind_uri",
        "unit",
        "calibration",
        "uncertainty",
        "quality",
        "coordinate_frame_id",
    }
)
_STANDARD_UNCERTAINTY_FIELDS = frozenset({"kind", "value", "array", "unit"})
_UNQUANTIFIED_UNCERTAINTY_FIELDS = frozenset({"kind", "reason"})
_QUALITY_FIELDS = frozenset({"validity_array", "saturation_array"})
_MULTISCALE_FIELDS = frozenset({"level", "kind", "factor", "channels"})
_ENVELOPE_CHANNEL_FIELDS = frozenset({"channel_id", "min_array", "max_array"})
_LINKED_RESOURCE_FIELDS = frozenset({"role", "resource_id", "sha256", "frame_clock_id"})
_TREE_MANIFEST_FIELDS = frozenset({"schema", "entries"})
_TREE_MANIFEST_ENTRY_FIELDS = frozenset({"path", "sha256", "size_bytes"})
_LINEAGE_FIELDS = frozenset({"tree_manifest_path"})
_SPECIMEN_FIELDS = frozenset({"specimen_id", "material_id"})

_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_UCUM_RE = re.compile(r"^[A-Za-z0-9%_.'{}\[\]()/+*^=-]{1,64}$")
_QUDT_UNIT_RE = re.compile(r"^http://qudt\.org/vocab/unit/[A-Za-z0-9._-]{1,128}$")
_QUDT_QUANTITY_RE = re.compile(r"^http://qudt\.org/vocab/quantitykind/[A-Za-z0-9._-]{1,128}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Closed semantic crosswalk for the sensor quantities qualified in v1. Syntax-only
# ontology identifiers are unsafe: two individually valid identifiers can still name
# incompatible units, and a valid unit can be dimensionally wrong for a quantity kind.
#
# The entries intentionally use the exact UCUM codes published on their corresponding
# QUDT unit resources. The sole explicit bridge is UCUM ``1`` to QUDT ``UNITLESS``;
# that QUDT resource is dimensionless but currently publishes no ``qudt:ucumCode``.
# Scaled units are separate identities: a millimetre is compatible
# with a displacement quantity, but is not silently treated as numerically identical to
# a metre in uncertainty or coordinate-transform checks. A governed linear calibration
# is the explicit place to perform such a conversion.
_CURATED_UNIT_QUANTITIES: dict[tuple[str, str], frozenset[str]] = {
    ("s", "SEC"): frozenset({"Time"}),
    ("ms", "MilliSEC"): frozenset({"Time"}),
    ("us", "MicroSEC"): frozenset({"Time"}),
    ("V", "V"): frozenset({"ElectricPotential", "ElectricPotentialDifference", "Voltage"}),
    ("mV", "MilliV"): frozenset({"ElectricPotential", "ElectricPotentialDifference", "Voltage"}),
    ("uV", "MicroV"): frozenset({"ElectricPotential", "ElectricPotentialDifference", "Voltage"}),
    ("Pa", "PA"): frozenset(
        {
            "BulkModulus",
            "ForcePerArea",
            "GaugePressure",
            "ModulusOfElasticity",
            "NormalStress",
            "Pressure",
            "ShearModulus",
            "ShearStress",
            "StaticPressure",
            "Stress",
        }
    ),
    ("kPa", "KiloPA"): frozenset(
        {
            "BulkModulus",
            "ForcePerArea",
            "GaugePressure",
            "ModulusOfElasticity",
            "NormalStress",
            "Pressure",
            "ShearModulus",
            "ShearStress",
            "StaticPressure",
            "Stress",
        }
    ),
    ("MPa", "MegaPA"): frozenset(
        {
            "BulkModulus",
            "ForcePerArea",
            "GaugePressure",
            "ModulusOfElasticity",
            "NormalStress",
            "Pressure",
            "ShearModulus",
            "ShearStress",
            "StaticPressure",
            "Stress",
        }
    ),
    ("GPa", "GigaPA"): frozenset(
        {
            "BulkModulus",
            "ForcePerArea",
            "ModulusOfElasticity",
            "NormalStress",
            "Pressure",
            "ShearModulus",
            "ShearStress",
            "Stress",
        }
    ),
    ("1", "UNITLESS"): frozenset({"Dimensionless", "DimensionlessRatio", "Strain"}),
    ("Cel", "DEG_C"): frozenset({"Temperature", "ThermodynamicTemperature"}),
    ("K", "K"): frozenset({"Temperature", "ThermodynamicTemperature"}),
    ("W", "W"): frozenset({"Power"}),
    ("mW", "MilliW"): frozenset({"Power"}),
    ("kW", "KiloW"): frozenset({"Power"}),
    ("m", "M"): frozenset(
        {"Depth", "Diameter", "Displacement", "Distance", "Length", "PositionVector", "Thickness"}
    ),
    ("mm", "MilliM"): frozenset(
        {"Depth", "Diameter", "Displacement", "Distance", "Length", "PositionVector", "Thickness"}
    ),
    ("um", "MicroM"): frozenset(
        {"Depth", "Diameter", "Displacement", "Distance", "Length", "PositionVector", "Thickness"}
    ),
    ("nm", "NanoM"): frozenset(
        {"Depth", "Diameter", "Displacement", "Distance", "Length", "PositionVector", "Thickness"}
    ),
    ("N", "N"): frozenset({"Force", "ForceMagnitude", "Thrust"}),
    ("mN", "MilliN"): frozenset({"Force", "ForceMagnitude", "Thrust"}),
    ("kN", "KiloN"): frozenset({"Force", "ForceMagnitude", "Thrust"}),
    ("A", "A"): frozenset({"ElectricCurrent"}),
    ("mA", "MilliA"): frozenset({"ElectricCurrent"}),
    ("uA", "MicroA"): frozenset({"ElectricCurrent"}),
    ("Hz", "HZ"): frozenset({"Frequency", "RotationalFrequency"}),
    ("kHz", "KiloHZ"): frozenset({"Frequency", "RotationalFrequency"}),
    ("MHz", "MegaHZ"): frozenset({"Frequency", "RotationalFrequency"}),
    ("m.s-1", "M-PER-SEC"): frozenset({"LinearVelocity", "Speed", "Velocity"}),
    ("mm.s-1", "MilliM-PER-SEC"): frozenset({"LinearVelocity", "Speed", "Velocity"}),
    ("um.s-1", "MicroM-PER-SEC"): frozenset({"LinearVelocity", "Speed", "Velocity"}),
    ("m.s-2", "M-PER-SEC2"): frozenset({"Acceleration", "LinearAcceleration"}),
    ("N.m", "N-M"): frozenset({"MomentOfForce", "Torque"}),
    ("J", "J"): frozenset({"Energy", "Work"}),
    ("mJ", "MilliJ"): frozenset({"Energy", "Work"}),
    ("kg", "KiloGM"): frozenset({"Mass"}),
    ("g", "GM"): frozenset({"Mass"}),
    ("mg", "MilliGM"): frozenset({"Mass"}),
    ("rad", "RAD"): frozenset({"Angle", "PlaneAngle"}),
    ("deg", "DEG"): frozenset({"Angle", "PlaneAngle"}),
}
_CURATED_BY_UCUM = {ucum: (ucum, qudt) for ucum, qudt in _CURATED_UNIT_QUANTITIES}
_CURATED_BY_QUDT = {qudt: (ucum, qudt) for ucum, qudt in _CURATED_UNIT_QUANTITIES}


class SensorValidationError(ValueError):
    """A stable, machine-classifiable sensor contract failure."""

    def __init__(self, code: str, path: str, message: str) -> None:
        self.code = code
        self.path = path
        self.message = message
        super().__init__(f"{code} at {path}: {message}")


def _fail(code: str, path: str, message: str) -> NoReturn:
    raise SensorValidationError(code, path, message)


def _mapping(value: Any, path: str, *, code: str = "invalid_object") -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code, path, "expected an object")
    return cast(Mapping[str, Any], value)


def _reject_unknown_fields(
    value: Mapping[str, Any],
    allowed: frozenset[str] | set[str],
    path: str,
    *,
    code: str = "unknown_sensor_fields",
) -> None:
    """Fail on the first undeclared key in linear time and bounded error space."""

    for key in value:
        if not isinstance(key, str) or key not in allowed:
            _fail(code, path, f"unsupported field {key!r}")


def _sequence(value: Any, path: str, *, code: str = "invalid_array") -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail(code, path, "expected an array")
    return cast(Sequence[Any], value)


def _string(value: Any, path: str, *, max_length: int = 512) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("invalid_string", path, "expected a non-empty string")
    text = cast(str, value).strip()
    if len(text) > max_length or any(ord(char) < 32 or char == "\x7f" for char in text):
        _fail("invalid_string", path, "string is too long or contains control characters")
    return text


def _token(value: Any, path: str) -> str:
    text = _string(value, path, max_length=128)
    if not _TOKEN_RE.fullmatch(text):
        _fail("invalid_token", path, "expected a stable ASCII identifier")
    return text


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        _fail("invalid_sha256", path, "expected 64 lowercase hexadecimal characters")
    return cast(str, value)


def _finite(value: Any, path: str, *, code: str = "nonfinite_value") -> float:
    if isinstance(value, bool):
        _fail(code, path, "boolean is not a numeric measurement")
    try:
        number = float(value)
    except (TypeError, ValueError):
        _fail(code, path, "expected a finite number")
    if not math.isfinite(number):
        _fail(code, path, "expected a finite number")
    return number


def _positive_int(value: Any, path: str, *, minimum: int = 1, code: str = "invalid_integer") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(code, path, f"expected an integer >= {minimum}")
    return cast(int, value)


def _relative_path(value: Any, path: str) -> str:
    text = _string(value, path, max_length=2048)
    if "\\" in text or "\x00" in text or text.startswith("/"):
        _fail("unsafe_path", path, "path must be a relative POSIX path")
    pure = PurePosixPath(text)
    if str(pure) != text or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        _fail("unsafe_path", path, "path must be normalized and traversal-free")
    return text


def canonical_json_bytes(value: Any) -> bytes:
    """Encode the repository's deterministic JSON profile.

    This is explicitly ``ultra.canonical-json.v1`` (sorted UTF-8 object keys, compact
    separators, no NaN/Infinity), not a claim of RFC 8785 conformance.
    """

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _fail(
            "noncanonical_json", "$", f"value cannot be encoded as {CANONICAL_JSON_SCHEMA}: {exc}"
        )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _parse_unit(value: Any, path: str) -> UnitRef:
    raw = _mapping(value, path, code="unit_unbound")
    _reject_unknown_fields(raw, _UNIT_FIELDS, path)
    ucum_raw = raw.get("ucum_code")
    qudt_raw = raw.get("qudt_uri")
    ucum: str | None = None
    qudt: str | None = None
    if ucum_raw is not None:
        if not isinstance(ucum_raw, str) or not _UCUM_RE.fullmatch(ucum_raw):
            _fail("invalid_ucum_unit", f"{path}.ucum_code", "invalid bounded UCUM code")
        ucum = ucum_raw
    if qudt_raw is not None:
        if not isinstance(qudt_raw, str) or not _QUDT_UNIT_RE.fullmatch(qudt_raw):
            _fail("invalid_qudt_unit", f"{path}.qudt_uri", "invalid QUDT unit URI")
        qudt = qudt_raw
    if ucum is None and qudt is None:
        _fail("unit_unbound", path, "unit requires ucum_code and/or a QUDT unit URI")
    if ucum is not None and qudt is not None:
        qudt_name = qudt.rsplit("/", 1)[-1]
        if (ucum, qudt_name) not in _CURATED_UNIT_QUANTITIES:
            _fail(
                "unit_identity_mismatch",
                path,
                "dual UCUM/QUDT identifiers are not one qualified semantic unit identity",
            )
    label_raw = raw.get("label")
    if label_raw is not None:
        label = _string(label_raw, f"{path}.label", max_length=64)
    elif ucum is not None:
        label = ucum
    else:
        assert qudt is not None
        label = qudt.rsplit("/", 1)[-1]
    return UnitRef(label=label, ucum_code=ucum, qudt_uri=qudt)


def _curated_unit_identity(value: UnitRef) -> tuple[str, str] | None:
    by_ucum = _CURATED_BY_UCUM.get(value.ucum_code) if value.ucum_code is not None else None
    qudt_name = value.qudt_uri.rsplit("/", 1)[-1] if value.qudt_uri is not None else None
    by_qudt = _CURATED_BY_QUDT.get(qudt_name) if qudt_name is not None else None
    if by_ucum is not None and by_qudt is not None and by_ucum != by_qudt:
        return None
    return by_ucum or by_qudt


def _units_equivalent(left: UnitRef, right: UnitRef) -> bool:
    left_identity = _curated_unit_identity(left)
    right_identity = _curated_unit_identity(right)
    if left_identity is not None and right_identity is not None:
        return left_identity == right_identity
    compared = False
    if left.ucum_code is not None and right.ucum_code is not None:
        compared = True
        if left.ucum_code != right.ucum_code:
            return False
    if left.qudt_uri is not None and right.qudt_uri is not None:
        compared = True
        if left.qudt_uri != right.qudt_uri:
            return False
    return compared


def _validate_quantity_unit(quantity_kind_uri: str, unit: UnitRef, path: str) -> None:
    identity = _curated_unit_identity(unit)
    if identity is None:
        _fail(
            "unsupported_quantity_unit",
            path,
            "quantity-bearing units must use a qualified v1 UCUM/QUDT identity",
        )
    quantity_name = quantity_kind_uri.rsplit("/", 1)[-1]
    if quantity_name not in _CURATED_UNIT_QUANTITIES[identity]:
        _fail(
            "quantity_unit_mismatch",
            path,
            f"unit {identity[0]!r} is not qualified for quantity kind {quantity_name!r}",
        )


def _is_seconds(unit: UnitRef) -> bool:
    return bool(unit.ucum_code == "s" or unit.qudt_uri == "http://qudt.org/vocab/unit/SEC")


def _quantity_kind(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _QUDT_QUANTITY_RE.fullmatch(value):
        _fail("invalid_quantity_kind", path, "expected an authoritative QUDT quantity-kind URI")
    return cast(str, value)


def _array_lookup(arrays: Any, array_path: str, metadata_path: str) -> Any:
    try:
        if isinstance(arrays, Mapping):
            if array_path not in arrays:
                raise KeyError(array_path)
            array = arrays[array_path]
        else:
            array = arrays[array_path]
    except Exception as exc:  # noqa: BLE001 - adapters expose heterogeneous lookup errors
        _fail("array_not_found", metadata_path, f"array {array_path!r} is not available: {exc}")
    if array is None:
        _fail("array_not_found", metadata_path, f"array {array_path!r} is not available")
    return array


def _shape(array: Any, path: str) -> tuple[int, ...]:
    raw = getattr(array, "shape", None)
    if raw is None:
        _fail("invalid_array_shape", path, "array does not expose an integer shape")
    try:
        shape = tuple(operator.index(value) for value in cast(Sequence[Any], raw))
    except TypeError:
        _fail("invalid_array_shape", path, "array does not expose an integer shape")
    if any(value < 0 for value in shape):
        _fail("invalid_array_shape", path, "array shape contains a negative dimension")
    return shape


def _dtype_kind(array: Any, path: str) -> str:
    dtype = getattr(array, "dtype", None)
    kind = getattr(dtype, "kind", None)
    if not isinstance(kind, str) or len(kind) != 1:
        _fail("invalid_array_dtype", path, "array does not expose a supported dtype kind")
    return cast(str, kind)


def _validate_chunks(array: Any, path: str, ndim: int) -> tuple[int, ...]:
    raw = getattr(array, "chunks", None)
    if raw is None:
        _fail("invalid_chunks", path, "array must declare an integer chunk shape")
    try:
        chunks = tuple(operator.index(value) for value in cast(Sequence[Any], raw))
    except TypeError:
        _fail("invalid_chunks", path, "array must declare an integer chunk shape")
    if len(chunks) != ndim or any(value <= 0 for value in chunks):
        _fail("invalid_chunks", path, "chunk dimensions must be positive and match array rank")
    dtype = getattr(array, "dtype", None)
    try:
        itemsize = operator.index(getattr(dtype, "itemsize"))
    except TypeError:
        _fail("invalid_array_dtype", path, "array dtype must expose a positive item size")
    if itemsize <= 0:
        _fail("invalid_array_dtype", path, "array dtype must expose a positive item size")
    decoded = itemsize
    for value in chunks:
        decoded *= value
    if decoded > MAX_DECODED_CHUNK_BYTES:
        _fail(
            "chunk_too_large",
            path,
            f"decoded chunk size {decoded} exceeds {MAX_DECODED_CHUNK_BYTES} bytes",
        )
    return chunks


def _validate_signal_array(array: Any, expected_count: int, path: str) -> None:
    shape = _shape(array, path)
    if len(shape) != 1:
        _fail("signal_rank_mismatch", path, "v1 sensor channels must be one-dimensional")
    if shape[0] != expected_count:
        _fail(
            "sample_count_mismatch",
            path,
            f"array length {shape[0]} != clock sample_count {expected_count}",
        )
    if _dtype_kind(array, path) not in {"i", "u", "f"}:
        _fail("signal_dtype", path, "signal arrays must use a real numeric dtype")
    _validate_chunks(array, path, 1)


def _validate_timestamp_array(array: Any, expected_count: int, path: str) -> None:
    shape = _shape(array, path)
    if len(shape) != 1 or shape[0] != expected_count:
        _fail(
            "clock_sample_count_mismatch",
            path,
            "timestamp array must be 1-D and match sample_count",
        )
    if _dtype_kind(array, path) not in {"i", "u", "f"}:
        _fail("clock_dtype", path, "timestamps must use a real numeric dtype")
    _validate_chunks(array, path, 1)


def _validate_flag_array(array: Any, expected_count: int, path: str) -> None:
    shape = _shape(array, path)
    if len(shape) != 1 or shape[0] != expected_count:
        _fail("quality_shape_mismatch", path, "quality flag array must be 1-D and match the signal")
    if _dtype_kind(array, path) != "b":
        _fail("flag_dtype", path, "quality flags require a boolean dtype")
    _validate_chunks(array, path, 1)


def _validate_uncertainty_array(array: Any, expected_count: int, path: str) -> None:
    shape = _shape(array, path)
    if len(shape) != 1 or shape[0] != expected_count:
        _fail(
            "uncertainty_shape_mismatch", path, "uncertainty array must be 1-D and match the signal"
        )
    if _dtype_kind(array, path) not in {"i", "u", "f"}:
        _fail("uncertainty_dtype", path, "uncertainty array must use a real numeric dtype")
    _validate_chunks(array, path, 1)


def _slice_values(
    array: Any,
    start: int,
    stop: int,
    path: str,
    *,
    budget: SensorValidationBudget,
) -> list[Any]:
    count = stop - start
    budget.consume_slice(array, count, path)
    try:
        block = array[start:stop]
        if hasattr(block, "reshape"):
            block = block.reshape(-1)
        if hasattr(block, "tolist"):
            block = block.tolist()
        values = list(block)
    except Exception as exc:  # noqa: BLE001
        _fail("array_read_failed", path, f"bounded read [{start}:{stop}] failed: {exc}")
    budget.check_wall(path)
    if len(values) != count:
        _fail("array_read_shape", path, "bounded read returned an unexpected value count")
    return values


def _blocks(array: Any, count: int, path: str, *, budget: SensorValidationBudget):
    budget.ensure_scan_fits(array, count, path)
    for start in range(0, count, VALIDATION_BLOCK_VALUES):
        stop = min(count, start + VALIDATION_BLOCK_VALUES)
        yield start, stop, _slice_values(array, start, stop, path, budget=budget)


def _as_bool(value: Any, path: str) -> bool:
    if isinstance(value, bool):
        return value
    # numpy.bool_ is intentionally supported without importing numpy in the base runtime.
    if type(value).__name__ == "bool_" and type(value).__module__.startswith("numpy"):
        return bool(value)
    _fail("flag_value", path, "boolean quality array yielded a non-boolean value")


def _as_float(value: Any, path: str, *, code: str) -> float:
    if isinstance(value, bool):
        _fail(code, path, "boolean is not a numeric sample")
    try:
        return float(value)
    except (TypeError, ValueError):
        _fail(code, path, "array yielded a non-numeric value")


def _parse_clock(
    value: Any,
    index: int,
    arrays: Any,
    *,
    validate_values: bool,
    validation_budget: SensorValidationBudget | None,
    warnings: list[str],
) -> ClockRecord:
    path = f"clocks[{index}]"
    raw = _mapping(value, path)
    clock_id = _token(raw.get("clock_id"), f"{path}.clock_id")
    kind = _string(raw.get("kind"), f"{path}.kind", max_length=32)
    if kind not in {"regular", "explicit"}:
        _fail("unsupported_clock_kind", f"{path}.kind", "supported values are regular and explicit")
    _reject_unknown_fields(
        raw,
        _CLOCK_REGULAR_FIELDS if kind == "regular" else _CLOCK_EXPLICIT_FIELDS,
        path,
    )
    sample_count = _positive_int(raw.get("sample_count"), f"{path}.sample_count")
    reference = _string(raw.get("reference"), f"{path}.reference", max_length=32)
    if reference in {"utc", "tai"}:
        _fail(
            "absolute_clock_epoch_required",
            f"{path}.reference",
            "v1 has no absolute epoch/clock-scale transform; use relative or instrument coordinates",
        )
    if reference not in {"relative", "instrument"}:
        _fail("invalid_clock_reference", f"{path}.reference", "unsupported time reference")
    time_unit = _parse_unit(raw.get("time_unit"), f"{path}.time_unit")
    if not _is_seconds(time_unit):
        _fail(
            "clock_unit_not_seconds",
            f"{path}.time_unit",
            "v1 clock coordinates are expressed in seconds",
        )

    accuracy = _mapping(raw.get("accuracy"), f"{path}.accuracy")
    accuracy_status = _string(accuracy.get("status"), f"{path}.accuracy.status", max_length=32)
    if accuracy_status == "quantified":
        _reject_unknown_fields(accuracy, _ACCURACY_QUANTIFIED_FIELDS, f"{path}.accuracy")
    elif accuracy_status == "not_quantified":
        _reject_unknown_fields(accuracy, _ACCURACY_UNQUANTIFIED_FIELDS, f"{path}.accuracy")
    standard_uncertainty_seconds: float | None = None
    accuracy_method: str | None = None
    accuracy_reason: str | None = None
    if accuracy_status == "quantified":
        standard_uncertainty_seconds = _finite(
            accuracy.get("standard_uncertainty_seconds"),
            f"{path}.accuracy.standard_uncertainty_seconds",
            code="invalid_clock_uncertainty",
        )
        if standard_uncertainty_seconds < 0:
            _fail(
                "invalid_clock_uncertainty",
                f"{path}.accuracy.standard_uncertainty_seconds",
                "uncertainty cannot be negative",
            )
        accuracy_method = _string(accuracy.get("method"), f"{path}.accuracy.method", max_length=256)
    elif accuracy_status == "not_quantified":
        accuracy_reason = _string(accuracy.get("reason"), f"{path}.accuracy.reason", max_length=512)
        warnings.append(f"clock_accuracy_not_quantified:{clock_id}")
    else:
        _fail(
            "invalid_clock_accuracy",
            f"{path}.accuracy.status",
            "status must be quantified or not_quantified",
        )

    if kind == "regular":
        sample_rate_hz = _finite(
            raw.get("sample_rate_hz"), f"{path}.sample_rate_hz", code="invalid_sample_rate"
        )
        if sample_rate_hz <= 0:
            _fail("invalid_sample_rate", f"{path}.sample_rate_hz", "sample rate must be > 0 Hz")
        start = _finite(
            raw.get("start_time_seconds"), f"{path}.start_time_seconds", code="invalid_clock_start"
        )
        if raw.get("timestamp_array") is not None:
            _fail(
                "ambiguous_clock",
                f"{path}.timestamp_array",
                "regular clock must not also declare timestamps",
            )
        return ClockRecord(
            clock_id=clock_id,
            kind=kind,
            sample_count=sample_count,
            reference=reference,
            time_unit=time_unit,
            accuracy_status=accuracy_status,
            standard_uncertainty_seconds=standard_uncertainty_seconds,
            accuracy_method=accuracy_method,
            accuracy_reason=accuracy_reason,
            sample_rate_hz=sample_rate_hz,
            start_time_seconds=start,
        )

    timestamp_path = _relative_path(raw.get("timestamp_array"), f"{path}.timestamp_array")
    timestamp_array = _array_lookup(arrays, timestamp_path, f"{path}.timestamp_array")
    _validate_timestamp_array(timestamp_array, sample_count, timestamp_path)
    if validate_values:
        assert validation_budget is not None
        previous: float | None = None
        for start, _, values in _blocks(
            timestamp_array,
            sample_count,
            timestamp_path,
            budget=validation_budget,
        ):
            for offset, value in enumerate(values):
                current = _as_float(
                    value, f"{timestamp_path}[{start + offset}]", code="nonfinite_timestamp"
                )
                if not math.isfinite(current):
                    _fail(
                        "nonfinite_timestamp",
                        f"{timestamp_path}[{start + offset}]",
                        "timestamp is NaN or infinite",
                    )
                if previous is not None and current <= previous:
                    _fail(
                        "clock_not_strictly_increasing",
                        f"{timestamp_path}[{start + offset}]",
                        "timestamps must increase strictly; duplicates are not accepted",
                    )
                previous = current
    return ClockRecord(
        clock_id=clock_id,
        kind=kind,
        sample_count=sample_count,
        reference=reference,
        time_unit=time_unit,
        accuracy_status=accuracy_status,
        standard_uncertainty_seconds=standard_uncertainty_seconds,
        accuracy_method=accuracy_method,
        accuracy_reason=accuracy_reason,
        timestamp_array=timestamp_path,
    )


def _parse_calibration(value: Any, channel_unit: UnitRef, path: str) -> CalibrationRecord:
    raw = _mapping(value, path)
    _reject_unknown_fields(raw, _CALIBRATION_FIELDS, path)
    kind = _string(raw.get("kind"), f"{path}.kind", max_length=32)
    if kind not in {"identity", "linear"}:
        _fail(
            "unsupported_calibration",
            f"{path}.kind",
            "v1 supports applied identity and linear calibrations",
        )
    if raw.get("applied") is not True:
        _fail(
            "calibration_not_applied",
            f"{path}.applied",
            "stored values must already have the declared calibration applied",
        )
    calibration_id = _token(raw.get("calibration_id"), f"{path}.calibration_id")
    revision = _string(raw.get("revision"), f"{path}.revision", max_length=128)
    input_unit = _parse_unit(raw.get("input_unit"), f"{path}.input_unit")
    output_unit = _parse_unit(raw.get("output_unit"), f"{path}.output_unit")
    if not _units_equivalent(output_unit, channel_unit):
        _fail(
            "calibration_output_unit_mismatch",
            f"{path}.output_unit",
            "calibration output must equal the channel unit",
        )
    scale = _finite(raw.get("scale"), f"{path}.scale", code="invalid_calibration")
    offset = _finite(raw.get("offset"), f"{path}.offset", code="invalid_calibration")
    if kind == "identity" and (
        scale != 1.0 or offset != 0.0 or not _units_equivalent(input_unit, output_unit)
    ):
        _fail(
            "invalid_identity_calibration",
            path,
            "identity requires scale=1, offset=0, and identical units",
        )
    if kind == "linear" and scale == 0:
        _fail("invalid_calibration", f"{path}.scale", "linear calibration scale cannot be zero")
    parameters_sha256 = _sha256(raw.get("parameters_sha256"), f"{path}.parameters_sha256")
    hash_payload = dict(raw)
    hash_payload.pop("parameters_sha256", None)
    computed = canonical_sha256(hash_payload)
    if computed != parameters_sha256:
        _fail(
            "calibration_hash_mismatch",
            f"{path}.parameters_sha256",
            f"declared {parameters_sha256} != computed {computed}",
        )
    certificate_raw = raw.get("certificate_sha256")
    certificate = (
        _sha256(certificate_raw, f"{path}.certificate_sha256")
        if certificate_raw is not None
        else None
    )
    return CalibrationRecord(
        kind=kind,
        calibration_id=calibration_id,
        revision=revision,
        input_unit=input_unit,
        output_unit=output_unit,
        scale=scale,
        offset=offset,
        parameters_sha256=parameters_sha256,
        certificate_sha256=certificate,
    )


def _parse_coordinate_frames(value: Any) -> tuple[CoordinateFrame, ...]:
    raw_frames = _sequence(value, "coordinate_frames")
    frames: list[CoordinateFrame] = []
    ids: set[str] = set()
    for frame_index, frame_value in enumerate(raw_frames):
        path = f"coordinate_frames[{frame_index}]"
        raw = _mapping(frame_value, path)
        _reject_unknown_fields(raw, _FRAME_FIELDS, path)
        frame_id = _token(raw.get("frame_id"), f"{path}.frame_id")
        if frame_id in ids:
            _fail("duplicate_coordinate_frame", f"{path}.frame_id", f"duplicate frame {frame_id!r}")
        ids.add(frame_id)
        raw_axes = _sequence(raw.get("axes"), f"{path}.axes")
        if not 1 <= len(raw_axes) <= 8:
            _fail(
                "invalid_coordinate_frame", f"{path}.axes", "coordinate frame requires 1 to 8 axes"
            )
        axes: list[CoordinateAxis] = []
        axis_names: set[str] = set()
        for axis_index, axis_value in enumerate(raw_axes):
            axis_path = f"{path}.axes[{axis_index}]"
            axis_raw = _mapping(axis_value, axis_path)
            _reject_unknown_fields(axis_raw, _COORDINATE_AXIS_FIELDS, axis_path)
            name = _token(axis_raw.get("name"), f"{axis_path}.name")
            if name in axis_names:
                _fail("duplicate_coordinate_axis", f"{axis_path}.name", f"duplicate axis {name!r}")
            axis_names.add(name)
            unit = _parse_unit(axis_raw.get("unit"), f"{axis_path}.unit")
            quantity_raw = axis_raw.get("quantity_kind_uri")
            quantity = (
                _quantity_kind(quantity_raw, f"{axis_path}.quantity_kind_uri")
                if quantity_raw is not None
                else None
            )
            if quantity is not None:
                _validate_quantity_unit(quantity, unit, f"{axis_path}.unit")
            axes.append(CoordinateAxis(name=name, unit=unit, quantity_kind_uri=quantity))
        frames.append(CoordinateFrame(frame_id=frame_id, axes=tuple(axes)))
    return tuple(frames)


def _parse_coordinate_transforms(
    value: Any,
    frames: tuple[CoordinateFrame, ...],
) -> tuple[CoordinateTransform, ...]:
    raw_transforms = _sequence(value, "coordinate_transforms")
    frames_by_id = {frame.frame_id: frame for frame in frames}
    transforms: list[CoordinateTransform] = []
    ids: set[str] = set()
    for index, transform_value in enumerate(raw_transforms):
        path = f"coordinate_transforms[{index}]"
        raw = _mapping(transform_value, path)
        _reject_unknown_fields(raw, _TRANSFORM_FIELDS, path)
        transform_id = _token(raw.get("transform_id"), f"{path}.transform_id")
        if transform_id in ids:
            _fail(
                "duplicate_coordinate_transform",
                f"{path}.transform_id",
                "transform IDs must be unique",
            )
        ids.add(transform_id)
        kind = _string(raw.get("kind"), f"{path}.kind", max_length=32)
        if kind != "affine":
            _fail(
                "unsupported_coordinate_transform", f"{path}.kind", "v1 supports affine transforms"
            )
        input_id = _token(raw.get("input_frame_id"), f"{path}.input_frame_id")
        output_id = _token(raw.get("output_frame_id"), f"{path}.output_frame_id")
        if input_id not in frames_by_id or output_id not in frames_by_id:
            _fail("coordinate_frame_not_found", path, "transform references an undeclared frame")
        input_dims = len(frames_by_id[input_id].axes)
        output_dims = len(frames_by_id[output_id].axes)
        matrix_raw = _sequence(raw.get("matrix"), f"{path}.matrix")
        if len(matrix_raw) != output_dims + 1:
            _fail(
                "transform_dimension_mismatch",
                f"{path}.matrix",
                "affine row count does not match output frame",
            )
        matrix: list[tuple[float, ...]] = []
        for row_index, row_value in enumerate(matrix_raw):
            row_raw = _sequence(row_value, f"{path}.matrix[{row_index}]")
            if len(row_raw) != input_dims + 1:
                _fail(
                    "transform_dimension_mismatch",
                    f"{path}.matrix[{row_index}]",
                    "affine column count does not match input frame",
                )
            row: list[float] = []
            for column_index, item in enumerate(row_raw):
                number = _finite(
                    item,
                    f"{path}.matrix[{row_index}][{column_index}]",
                    code="nonfinite_transform",
                )
                row.append(number)
            matrix.append(tuple(row))
        bottom = matrix[-1]
        if any(abs(value) > 1e-15 for value in bottom[:-1]) or abs(bottom[-1] - 1.0) > 1e-15:
            _fail(
                "invalid_affine_homogeneous_row",
                f"{path}.matrix[-1]",
                "last row must be [0, ..., 0, 1]",
            )
        input_axes = frames_by_id[input_id].axes
        output_axes = frames_by_id[output_id].axes
        for output_index, matrix_row in enumerate(matrix[:-1]):
            output_axis = output_axes[output_index]
            for input_index, coefficient in enumerate(matrix_row[:-1]):
                if abs(coefficient) <= 1e-15:
                    continue
                input_axis = input_axes[input_index]
                if not _units_equivalent(input_axis.unit, output_axis.unit):
                    _fail(
                        "transform_axis_unit_mismatch",
                        f"{path}.matrix[{output_index}][{input_index}]",
                        "a nonzero affine coefficient cannot mix incompatible axis units",
                    )
                if (
                    input_axis.quantity_kind_uri is not None
                    and output_axis.quantity_kind_uri is not None
                    and input_axis.quantity_kind_uri != output_axis.quantity_kind_uri
                ):
                    _fail(
                        "transform_axis_quantity_mismatch",
                        f"{path}.matrix[{output_index}][{input_index}]",
                        "a nonzero affine coefficient cannot mix incompatible quantity kinds",
                    )
        transforms.append(
            CoordinateTransform(
                transform_id=transform_id,
                kind=kind,
                input_frame_id=input_id,
                output_frame_id=output_id,
                matrix=tuple(matrix),
            )
        )
    return tuple(transforms)


def _parse_channel(
    value: Any,
    index: int,
    arrays: Any,
    clocks_by_id: Mapping[str, ClockRecord],
    frame_ids: set[str],
    *,
    validate_values: bool,
    validation_budget: SensorValidationBudget | None,
    warnings: list[str],
) -> ChannelRecord:
    path = f"channels[{index}]"
    raw = _mapping(value, path)
    _reject_unknown_fields(raw, _CHANNEL_FIELDS, path)
    channel_id = _token(raw.get("channel_id"), f"{path}.channel_id")
    name = _string(raw.get("name"), f"{path}.name", max_length=256)
    array_path = _relative_path(raw.get("array"), f"{path}.array")
    clock_id = _token(raw.get("clock_id"), f"{path}.clock_id")
    clock = clocks_by_id.get(clock_id)
    if clock is None:
        _fail("clock_not_found", f"{path}.clock_id", f"unknown clock {clock_id!r}")
    quantity_kind = _quantity_kind(raw.get("quantity_kind_uri"), f"{path}.quantity_kind_uri")
    unit = _parse_unit(raw.get("unit"), f"{path}.unit")
    _validate_quantity_unit(quantity_kind, unit, f"{path}.unit")
    calibration = _parse_calibration(raw.get("calibration"), unit, f"{path}.calibration")

    uncertainty = _mapping(raw.get("uncertainty"), f"{path}.uncertainty")
    uncertainty_kind = _string(uncertainty.get("kind"), f"{path}.uncertainty.kind", max_length=32)
    if uncertainty_kind == "standard":
        _reject_unknown_fields(uncertainty, _STANDARD_UNCERTAINTY_FIELDS, f"{path}.uncertainty")
    elif uncertainty_kind == "not_quantified":
        _reject_unknown_fields(uncertainty, _UNQUANTIFIED_UNCERTAINTY_FIELDS, f"{path}.uncertainty")
    uncertainty_value: float | None = None
    uncertainty_array_path: str | None = None
    uncertainty_array: Any | None = None
    uncertainty_reason: str | None = None
    if uncertainty_kind == "standard":
        has_value = uncertainty.get("value") is not None
        has_array = uncertainty.get("array") is not None
        if has_value == has_array:
            _fail(
                "invalid_uncertainty",
                f"{path}.uncertainty",
                "standard uncertainty requires exactly one of value or array",
            )
        uncertainty_unit = _parse_unit(uncertainty.get("unit"), f"{path}.uncertainty.unit")
        if not _units_equivalent(uncertainty_unit, unit):
            _fail(
                "uncertainty_unit_mismatch",
                f"{path}.uncertainty.unit",
                "uncertainty unit must equal channel unit",
            )
        if has_value:
            uncertainty_value = _finite(
                uncertainty.get("value"),
                f"{path}.uncertainty.value",
                code="invalid_uncertainty",
            )
            if uncertainty_value < 0:
                _fail(
                    "invalid_uncertainty",
                    f"{path}.uncertainty.value",
                    "uncertainty cannot be negative",
                )
        else:
            uncertainty_array_path = _relative_path(
                uncertainty.get("array"), f"{path}.uncertainty.array"
            )
            uncertainty_array = _array_lookup(
                arrays, uncertainty_array_path, f"{path}.uncertainty.array"
            )
            _validate_uncertainty_array(
                uncertainty_array, clock.sample_count, uncertainty_array_path
            )
    elif uncertainty_kind == "not_quantified":
        uncertainty_reason = _string(
            uncertainty.get("reason"), f"{path}.uncertainty.reason", max_length=512
        )
        if uncertainty.get("value") is not None or uncertainty.get("array") is not None:
            _fail(
                "invalid_uncertainty",
                f"{path}.uncertainty",
                "not_quantified must not carry numeric uncertainty",
            )
        warnings.append(f"uncertainty_not_quantified:{channel_id}")
    else:
        _fail(
            "invalid_uncertainty",
            f"{path}.uncertainty.kind",
            "kind must be standard or not_quantified",
        )

    quality = _mapping(raw.get("quality", {}), f"{path}.quality")
    _reject_unknown_fields(quality, _QUALITY_FIELDS, f"{path}.quality")
    validity_path = (
        _relative_path(quality.get("validity_array"), f"{path}.quality.validity_array")
        if quality.get("validity_array") is not None
        else None
    )
    saturation_path = (
        _relative_path(quality.get("saturation_array"), f"{path}.quality.saturation_array")
        if quality.get("saturation_array") is not None
        else None
    )
    signal = _array_lookup(arrays, array_path, f"{path}.array")
    _validate_signal_array(signal, clock.sample_count, array_path)
    validity = (
        _array_lookup(arrays, validity_path, f"{path}.quality.validity_array")
        if validity_path
        else None
    )
    saturation = (
        _array_lookup(arrays, saturation_path, f"{path}.quality.saturation_array")
        if saturation_path
        else None
    )
    if validity is not None:
        assert validity_path is not None
        _validate_flag_array(validity, clock.sample_count, validity_path)
    if saturation is not None:
        assert saturation_path is not None
        _validate_flag_array(saturation, clock.sample_count, saturation_path)

    coordinate_frame_raw = raw.get("coordinate_frame_id")
    coordinate_frame = (
        _token(coordinate_frame_raw, f"{path}.coordinate_frame_id")
        if coordinate_frame_raw is not None
        else None
    )
    if coordinate_frame is not None and coordinate_frame not in frame_ids:
        _fail(
            "coordinate_frame_not_found",
            f"{path}.coordinate_frame_id",
            "channel references an undeclared frame",
        )

    invalid_count: int | None = None
    saturation_count: int | None = None
    if validate_values:
        assert validation_budget is not None
        invalid_count = 0
        saturation_count = 0
        if validity is not None:
            assert validity_path is not None
        if saturation is not None:
            assert saturation_path is not None
        for start, stop, signal_values in _blocks(
            signal,
            clock.sample_count,
            array_path,
            budget=validation_budget,
        ):
            if validity is not None:
                assert validity_path is not None
                validity_values = _slice_values(
                    validity,
                    start,
                    stop,
                    validity_path,
                    budget=validation_budget,
                )
            else:
                validity_values = None
            if saturation is not None:
                assert saturation_path is not None
                saturation_values = _slice_values(
                    saturation,
                    start,
                    stop,
                    saturation_path,
                    budget=validation_budget,
                )
            else:
                saturation_values = None
            for offset, raw_value in enumerate(signal_values):
                position = start + offset
                valid = (
                    _as_bool(validity_values[offset], f"{validity_path}[{position}]")
                    if validity_values is not None
                    else True
                )
                saturated = (
                    _as_bool(saturation_values[offset], f"{saturation_path}[{position}]")
                    if saturation_values is not None
                    else False
                )
                number = _as_float(raw_value, f"{array_path}[{position}]", code="nonfinite_signal")
                if math.isinf(number):
                    _fail(
                        "infinite_signal",
                        f"{array_path}[{position}]",
                        "infinite signal values are never valid measurements",
                    )
                if math.isnan(number) and valid:
                    _fail(
                        "nan_without_invalid_flag",
                        f"{array_path}[{position}]",
                        "NaN requires a matching false validity flag",
                    )
                if not valid:
                    invalid_count += 1
                if saturated:
                    saturation_count += 1
        if uncertainty_array is not None:
            assert uncertainty_array_path is not None
            for start, _, values in _blocks(
                uncertainty_array,
                clock.sample_count,
                uncertainty_array_path,
                budget=validation_budget,
            ):
                for offset, raw_value in enumerate(values):
                    position = start + offset
                    number = _as_float(
                        raw_value,
                        f"{uncertainty_array_path}[{position}]",
                        code="invalid_uncertainty_array",
                    )
                    if not math.isfinite(number) or number < 0:
                        _fail(
                            "invalid_uncertainty_array",
                            f"{uncertainty_array_path}[{position}]",
                            "uncertainty samples must be finite and nonnegative",
                        )

    return ChannelRecord(
        channel_id=channel_id,
        name=name,
        array_path=array_path,
        clock_id=clock_id,
        quantity_kind_uri=quantity_kind,
        unit=unit,
        calibration=calibration,
        uncertainty_kind=uncertainty_kind,
        uncertainty_value=uncertainty_value,
        uncertainty_array=uncertainty_array_path,
        uncertainty_reason=uncertainty_reason,
        validity_array=validity_path,
        saturation_array=saturation_path,
        coordinate_frame_id=coordinate_frame,
        invalid_count=invalid_count,
        saturation_count=saturation_count,
    )


def _verify_min_max_envelope(
    source: Any,
    validity: Any | None,
    minimum: Any,
    maximum: Any,
    *,
    source_count: int,
    factor: int,
    source_path: str,
    validity_path: str | None,
    min_path: str,
    max_path: str,
    metadata_path: str,
    validation_budget: SensorValidationBudget,
) -> None:
    """Prove stored extrema against the source signal with bounded reads.

    Shape-valid but stale envelopes are scientifically unsafe for sparse transients such as
    acoustic-emission hits. This verifier excludes explicitly invalid samples, requires an
    all-NaN pair for an all-invalid bucket, and compares every stored extremum to the exact
    source value. Its source and envelope buffers are each capped by
    ``VALIDATION_BLOCK_VALUES`` values.
    """

    pending_lows: list[float | None] = []
    pending_highs: list[float | None] = []
    pending_start = 0

    def flush() -> None:
        nonlocal pending_start
        if not pending_lows:
            return
        stop = pending_start + len(pending_lows)
        stored_lows = _slice_values(
            minimum,
            pending_start,
            stop,
            min_path,
            budget=validation_budget,
        )
        stored_highs = _slice_values(
            maximum,
            pending_start,
            stop,
            max_path,
            budget=validation_budget,
        )
        for offset, (expected_low, expected_high, low_raw, high_raw) in enumerate(
            zip(pending_lows, pending_highs, stored_lows, stored_highs, strict=True)
        ):
            bucket = pending_start + offset
            low = _as_float(low_raw, f"{min_path}[{bucket}]", code="nonfinite_envelope")
            high = _as_float(high_raw, f"{max_path}[{bucket}]", code="nonfinite_envelope")
            if math.isnan(low) and math.isnan(high):
                if expected_low is None:
                    continue
                _fail(
                    "stale_envelope",
                    metadata_path,
                    f"bucket {bucket} is NaN but contains valid source samples",
                )
            if not math.isfinite(low) or not math.isfinite(high):
                _fail(
                    "nonfinite_envelope",
                    metadata_path,
                    "envelope bounds must be finite or both NaN",
                )
            if low > high:
                _fail(
                    "invalid_envelope_bounds",
                    metadata_path,
                    f"minimum {low} exceeds maximum {high}",
                )
            if expected_low is None or low != expected_low or high != expected_high:
                _fail(
                    "stale_envelope",
                    metadata_path,
                    f"bucket {bucket} stored [{low}, {high}] != source [{expected_low}, {expected_high}]",
                )
        pending_start = stop
        pending_lows.clear()
        pending_highs.clear()

    current_bucket = -1
    current_low: float | None = None
    current_high: float | None = None
    for start, stop, source_values in _blocks(
        source,
        source_count,
        source_path,
        budget=validation_budget,
    ):
        if validity is not None:
            assert validity_path is not None
            validity_values = _slice_values(
                validity,
                start,
                stop,
                validity_path,
                budget=validation_budget,
            )
        else:
            validity_values = None
        for offset, raw_value in enumerate(source_values):
            position = start + offset
            bucket = position // factor
            if bucket != current_bucket:
                if current_bucket >= 0:
                    pending_lows.append(current_low)
                    pending_highs.append(current_high)
                    if len(pending_lows) >= VALIDATION_BLOCK_VALUES:
                        flush()
                current_bucket = bucket
                current_low = None
                current_high = None
            valid = (
                _as_bool(validity_values[offset], f"{validity_path}[{position}]")
                if validity_values is not None
                else True
            )
            number = _as_float(raw_value, f"{source_path}[{position}]", code="nonfinite_signal")
            if not valid:
                continue
            if current_low is None or number < current_low:
                current_low = number
            if current_high is None or number > current_high:
                current_high = number
    pending_lows.append(current_low)
    pending_highs.append(current_high)
    flush()


def _parse_multiscales(
    value: Any,
    arrays: Any,
    channels_by_id: Mapping[str, ChannelRecord],
    clocks_by_id: Mapping[str, ClockRecord],
    *,
    validate_values: bool,
    validation_budget: SensorValidationBudget | None,
) -> tuple[MultiscaleLevel, ...]:
    raw_levels = _sequence(value, "multiscales")
    levels: list[MultiscaleLevel] = []
    seen_levels: set[int] = set()
    previous_factor = 1
    for index, level_value in enumerate(raw_levels):
        path = f"multiscales[{index}]"
        raw = _mapping(level_value, path)
        _reject_unknown_fields(raw, _MULTISCALE_FIELDS, path)
        level = _positive_int(raw.get("level"), f"{path}.level")
        if level in seen_levels:
            _fail("duplicate_multiscale_level", f"{path}.level", "level IDs must be unique")
        seen_levels.add(level)
        kind = _string(raw.get("kind"), f"{path}.kind", max_length=64)
        if kind != "min_max_envelope":
            _fail("unsupported_multiscale_kind", f"{path}.kind", "v1 supports min_max_envelope")
        factor = _positive_int(raw.get("factor"), f"{path}.factor", minimum=2)
        if factor <= previous_factor:
            _fail("multiscale_factor_order", f"{path}.factor", "factors must increase strictly")
        previous_factor = factor
        raw_channels = _sequence(raw.get("channels"), f"{path}.channels")
        if not raw_channels:
            _fail("empty_multiscale", f"{path}.channels", "multiscale level requires channels")
        parsed_channels: list[EnvelopeChannel] = []
        seen_channels: set[str] = set()
        for channel_index, value in enumerate(raw_channels):
            channel_path = f"{path}.channels[{channel_index}]"
            channel_raw = _mapping(value, channel_path)
            _reject_unknown_fields(channel_raw, _ENVELOPE_CHANNEL_FIELDS, channel_path)
            channel_id = _token(channel_raw.get("channel_id"), f"{channel_path}.channel_id")
            if channel_id in seen_channels:
                _fail(
                    "duplicate_multiscale_channel",
                    f"{channel_path}.channel_id",
                    "channel appears twice at one level",
                )
            seen_channels.add(channel_id)
            channel = channels_by_id.get(channel_id)
            if channel is None:
                _fail(
                    "channel_not_found",
                    f"{channel_path}.channel_id",
                    "multiscale references an unknown channel",
                )
            count = clocks_by_id[channel.clock_id].sample_count
            expected = math.ceil(count / factor)
            min_path = _relative_path(channel_raw.get("min_array"), f"{channel_path}.min_array")
            max_path = _relative_path(channel_raw.get("max_array"), f"{channel_path}.max_array")
            minimum = _array_lookup(arrays, min_path, f"{channel_path}.min_array")
            maximum = _array_lookup(arrays, max_path, f"{channel_path}.max_array")
            _validate_signal_array(minimum, expected, min_path)
            _validate_signal_array(maximum, expected, max_path)
            if validate_values:
                assert validation_budget is not None
                source = _array_lookup(arrays, channel.array_path, channel_path)
                validity = (
                    _array_lookup(arrays, channel.validity_array, channel_path)
                    if channel.validity_array is not None
                    else None
                )
                _verify_min_max_envelope(
                    source,
                    validity,
                    minimum,
                    maximum,
                    source_count=count,
                    factor=factor,
                    source_path=channel.array_path,
                    validity_path=channel.validity_array,
                    min_path=min_path,
                    max_path=max_path,
                    metadata_path=channel_path,
                    validation_budget=validation_budget,
                )
            parsed_channels.append(
                EnvelopeChannel(channel_id=channel_id, min_array=min_path, max_array=max_path)
            )
        levels.append(
            MultiscaleLevel(level=level, kind=kind, factor=factor, channels=tuple(parsed_channels))
        )
    return tuple(levels)


def _parse_linked_resources(value: Any, clock_ids: set[str]) -> tuple[LinkedResource, ...]:
    raw_resources = _sequence(value, "linked_resources")
    resources: list[LinkedResource] = []
    identities: set[tuple[str, str]] = set()
    for index, resource_value in enumerate(raw_resources):
        path = f"linked_resources[{index}]"
        raw = _mapping(resource_value, path)
        verification_claims = {
            "authority",
            "verification_authority",
            "verification_status",
            "verified",
        }.intersection(raw)
        if verification_claims:
            _fail(
                "caller_linked_resource_verification_forbidden",
                path,
                "linked-resource verification is issued only from tenant-authorized "
                "run-selected catalog descriptors",
            )
        _reject_unknown_fields(
            raw,
            _LINKED_RESOURCE_FIELDS,
            path,
            code="unknown_linked_resource_fields",
        )
        role = _token(raw.get("role"), f"{path}.role")
        resource_id = _token(raw.get("resource_id"), f"{path}.resource_id")
        identity = (role, resource_id)
        if identity in identities:
            _fail("duplicate_linked_resource", path, "linked resource role/id must be unique")
        identities.add(identity)
        digest = _sha256(raw.get("sha256"), f"{path}.sha256")
        frame_clock_raw = raw.get("frame_clock_id")
        frame_clock = (
            _token(frame_clock_raw, f"{path}.frame_clock_id")
            if frame_clock_raw is not None
            else None
        )
        if frame_clock is not None and frame_clock not in clock_ids:
            _fail(
                "clock_not_found",
                f"{path}.frame_clock_id",
                "linked resource references an unknown clock",
            )
        resources.append(
            LinkedResource(
                role=role,
                resource_id=resource_id,
                sha256=digest,
                frame_clock_id=frame_clock,
                verification_status="declared_unverified",
                verification_authority=None,
            )
        )
    return tuple(resources)


def _manifest_entries(manifest: Any) -> tuple[Mapping[str, Any], ...]:
    raw = _mapping(manifest, "tree_manifest", code="invalid_tree_manifest")
    _reject_unknown_fields(raw, _TREE_MANIFEST_FIELDS, "tree_manifest")
    if raw.get("schema") != TREE_MANIFEST_SCHEMA:
        _fail(
            "invalid_tree_manifest_schema",
            "tree_manifest.schema",
            f"expected {TREE_MANIFEST_SCHEMA}",
        )
    entries_raw = _sequence(
        raw.get("entries"), "tree_manifest.entries", code="invalid_tree_manifest"
    )
    entries: list[Mapping[str, Any]] = []
    paths: list[str] = []
    for index, entry_value in enumerate(entries_raw):
        path = f"tree_manifest.entries[{index}]"
        entry = _mapping(entry_value, path, code="invalid_tree_manifest")
        _reject_unknown_fields(entry, _TREE_MANIFEST_ENTRY_FIELDS, path)
        entry_path = _relative_path(entry.get("path"), f"{path}.path")
        _sha256(entry.get("sha256"), f"{path}.sha256")
        size = entry.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            _fail(
                "invalid_tree_manifest", f"{path}.size_bytes", "size must be a nonnegative integer"
            )
        paths.append(entry_path)
        entries.append(entry)
    if len(set(paths)) != len(paths):
        _fail("duplicate_tree_entry", "tree_manifest.entries", "manifest paths must be unique")
    if paths != sorted(paths):
        _fail(
            "nondeterministic_tree_manifest",
            "tree_manifest.entries",
            "entries must be sorted by path",
        )
    return tuple(entries)


def _parse_lineage(
    value: Any,
    *,
    tree_manifest: Any | None,
    expected_tree_manifest_sha256: str | None,
    warnings: list[str],
) -> LineageBinding:
    raw = _mapping(value, "lineage")
    _reject_unknown_fields(raw, _LINEAGE_FIELDS, "lineage")
    path_raw = raw.get("tree_manifest_path")
    manifest_path = (
        _relative_path(path_raw, "lineage.tree_manifest_path") if path_raw is not None else None
    )
    if expected_tree_manifest_sha256 is None:
        if tree_manifest is not None:
            _manifest_entries(tree_manifest)
        warnings.append("lineage_unbound")
        return LineageBinding(
            status="unbound",
            tree_manifest_path=manifest_path,
            expected_tree_manifest_sha256=None,
            computed_tree_manifest_sha256=None,
            entry_count=None,
        )
    expected = _sha256(expected_tree_manifest_sha256, "expected_tree_manifest_sha256")
    if manifest_path is None or tree_manifest is None:
        _fail(
            "lineage_manifest_required",
            "lineage.tree_manifest_path",
            "trusted expected digest requires a declared path and supplied manifest",
        )
    entries = _manifest_entries(tree_manifest)
    computed = canonical_sha256(tree_manifest)
    if computed != expected:
        _fail(
            "lineage_manifest_hash_mismatch",
            "expected_tree_manifest_sha256",
            f"expected {expected} != computed {computed}",
        )
    return LineageBinding(
        status="manifest_verified",
        tree_manifest_path=manifest_path,
        expected_tree_manifest_sha256=expected,
        computed_tree_manifest_sha256=computed,
        entry_count=len(entries),
    )


def _parse_specimen(value: Any) -> SpecimenRecord:
    """Validate the closed v1 specimen/material identity record.

    These fields are scientific join keys, not descriptive labels.  Requiring both and
    rejecting silently ignored extras prevents a sensor trace from becoming detached from
    the specimen or material to which downstream mechanics/characterization results refer.
    """

    raw = _mapping(value, "specimen", code="invalid_specimen")
    _reject_unknown_fields(
        raw,
        _SPECIMEN_FIELDS,
        "specimen",
        code="unknown_specimen_fields",
    )
    return SpecimenRecord(
        specimen_id=_token(raw.get("specimen_id"), "specimen.specimen_id"),
        material_id=_token(raw.get("material_id"), "specimen.material_id"),
    )


def parse_sensor_series(
    metadata: Any,
    arrays: Any,
    *,
    tree_manifest: Any | None = None,
    expected_tree_manifest_sha256: str | None = None,
    validate_values: bool = True,
    validation_budget: SensorValidationBudget | None = None,
) -> SensorSeries:
    """Validate an ``ultra.sensor-series.v1`` metadata object and its lazy arrays.

    ``arrays`` may be a mapping or any object supporting ``arrays[path]``. Array objects
    need the standard Zarr-like ``shape``, ``dtype``, ``chunks``, and slice interface.
    Full value validation streams in bounded blocks; it never silently samples. Set
    ``validate_values=False`` only for an explicitly metadata-only preflight—the returned
    ``values_validated`` flag and channel counts remain honest about that weaker result.
    When values are validated, one :class:`SensorValidationBudget` accounts for all array
    reads. Supplying a budget lets a run-scoped caller choose stricter admission limits;
    otherwise the module's conservative aggregate defaults apply.
    """

    if not isinstance(validate_values, bool):
        _fail("invalid_validation_mode", "validate_values", "validate_values must be boolean")
    if validation_budget is not None and not isinstance(validation_budget, SensorValidationBudget):
        _fail(
            "invalid_validation_budget",
            "validation_budget",
            "validation_budget must be a SensorValidationBudget",
        )
    active_budget = (
        validation_budget or SensorValidationBudget() if validate_values else validation_budget
    )
    if active_budget is not None:
        active_budget.check_wall("sensor_metadata")

    raw = _mapping(metadata, "$", code="invalid_sensor_metadata")
    _reject_unknown_fields(raw, _ROOT_FIELDS, "$")
    if raw.get("schema") != SENSOR_SCHEMA:
        _fail("invalid_sensor_schema", "schema", f"expected {SENSOR_SCHEMA}")
    series_id = _token(raw.get("series_id"), "series_id")
    modality = _token(raw.get("modality"), "modality")
    specimen = _parse_specimen(raw.get("specimen"))
    warnings: list[str] = []

    frames = _parse_coordinate_frames(raw.get("coordinate_frames", []))
    transforms = _parse_coordinate_transforms(raw.get("coordinate_transforms", []), frames)

    raw_clocks = _sequence(raw.get("clocks"), "clocks")
    if not raw_clocks:
        _fail("missing_clocks", "clocks", "at least one sample clock is required")
    clocks: list[ClockRecord] = []
    clock_ids: set[str] = set()
    for index, value in enumerate(raw_clocks):
        clock = _parse_clock(
            value,
            index,
            arrays,
            validate_values=validate_values,
            validation_budget=active_budget,
            warnings=warnings,
        )
        if clock.clock_id in clock_ids:
            _fail("duplicate_clock", f"clocks[{index}].clock_id", "clock IDs must be unique")
        clock_ids.add(clock.clock_id)
        clocks.append(clock)
    clocks_by_id = {clock.clock_id: clock for clock in clocks}

    raw_channels = _sequence(raw.get("channels"), "channels")
    if not raw_channels:
        _fail("missing_channels", "channels", "at least one sensor channel is required")
    channels: list[ChannelRecord] = []
    channel_ids: set[str] = set()
    frame_ids = {frame.frame_id for frame in frames}
    for index, value in enumerate(raw_channels):
        channel = _parse_channel(
            value,
            index,
            arrays,
            clocks_by_id,
            frame_ids,
            validate_values=validate_values,
            validation_budget=active_budget,
            warnings=warnings,
        )
        if channel.channel_id in channel_ids:
            _fail(
                "duplicate_channel", f"channels[{index}].channel_id", "channel IDs must be unique"
            )
        channel_ids.add(channel.channel_id)
        channels.append(channel)
    channels_by_id = {channel.channel_id: channel for channel in channels}

    multiscales = _parse_multiscales(
        raw.get("multiscales", []),
        arrays,
        channels_by_id,
        clocks_by_id,
        validate_values=validate_values,
        validation_budget=active_budget,
    )
    linked = _parse_linked_resources(raw.get("linked_resources", []), clock_ids)
    warnings.extend(
        f"linked_resource_declared_unverified:{resource.role}:{resource.resource_id}"
        for resource in linked
    )
    lineage = _parse_lineage(
        raw.get("lineage", {}),
        tree_manifest=tree_manifest,
        expected_tree_manifest_sha256=expected_tree_manifest_sha256,
        warnings=warnings,
    )

    return SensorSeries(
        schema=SENSOR_SCHEMA,
        series_id=series_id,
        modality=modality,
        specimen=specimen,
        clocks=tuple(clocks),
        channels=tuple(channels),
        coordinate_frames=frames,
        coordinate_transforms=transforms,
        multiscales=multiscales,
        linked_resources=linked,
        lineage=lineage,
        warnings=tuple(warnings),
        values_validated=bool(validate_values),
    )


def verify_tree_manifest(
    root: str | os.PathLike[str],
    manifest: Any,
    *,
    manifest_path: str,
    validation_budget: SensorValidationBudget | None = None,
) -> TreeManifestVerification:
    """Verify exact regular-file closure, sizes, and hashes for a directory tree.

    The manifest file is excluded to avoid a self-hash cycle. Every other filesystem
    entry must be a declared regular file; symlinks and undeclared files fail closed.
    """

    entries = _manifest_entries(manifest)
    if validation_budget is not None:
        validation_budget.check_wall("tree_manifest")
    safe_manifest_path = _relative_path(manifest_path, "manifest_path")
    root_path = Path(root)
    if root_path.is_symlink() or not root_path.is_dir():
        _fail("invalid_tree_root", "root", "tree root must be a real directory, not a symlink")
    root_resolved = root_path.resolve()
    manifest_file = root_path / PurePosixPath(safe_manifest_path)
    if manifest_file.is_symlink() or not manifest_file.is_file():
        _fail(
            "tree_manifest_missing",
            "manifest_path",
            "declared manifest file is missing or not regular",
        )

    declared = {str(entry["path"]): entry for entry in entries}
    if safe_manifest_path in declared:
        _fail("tree_manifest_self_entry", "tree_manifest.entries", "manifest must exclude itself")
    actual: set[str] = set()
    for directory, dirnames, filenames in os.walk(root_path, followlinks=False):
        if validation_budget is not None:
            validation_budget.check_wall("tree_manifest")
        directory_path = Path(directory)
        for dirname in dirnames:
            if (directory_path / dirname).is_symlink():
                relative = (directory_path / dirname).relative_to(root_path).as_posix()
                _fail(
                    "tree_symlink_forbidden",
                    relative,
                    "tree closure cannot contain symlink directories",
                )
        for filename in filenames:
            candidate = directory_path / filename
            relative = candidate.relative_to(root_path).as_posix()
            if candidate.is_symlink():
                _fail(
                    "tree_symlink_forbidden", relative, "tree closure cannot contain symlink files"
                )
            if relative == safe_manifest_path:
                continue
            if not candidate.is_file():
                _fail("tree_nonregular_entry", relative, "tree closure accepts regular files only")
            actual.add(relative)
    declared_paths = set(declared)
    if actual != declared_paths:
        missing = sorted(declared_paths - actual)
        extra = sorted(actual - declared_paths)
        _fail(
            "tree_manifest_not_closed",
            "tree_manifest.entries",
            f"missing={missing[:8]} extra={extra[:8]}",
        )

    total_size = 0
    for relative in sorted(declared):
        entry = declared[relative]
        candidate = root_path / PurePosixPath(relative)
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root_resolved)
        except (OSError, ValueError):
            _fail("unsafe_tree_entry", relative, "entry escapes the verified root")
        stat = candidate.stat()
        expected_size = int(entry["size_bytes"])
        if stat.st_size != expected_size:
            _fail(
                "tree_entry_size_mismatch",
                relative,
                f"expected {expected_size} != actual {stat.st_size}",
            )
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            while True:
                if validation_budget is not None:
                    validation_budget.check_wall(relative)
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        actual_sha = digest.hexdigest()
        expected_sha = str(entry["sha256"])
        if actual_sha != expected_sha:
            _fail(
                "tree_entry_hash_mismatch",
                relative,
                f"expected {expected_sha} != actual {actual_sha}",
            )
        total_size += stat.st_size
    return TreeManifestVerification(entry_count=len(entries), size_bytes=total_size)


def _read_root_attributes(root: Path) -> Mapping[str, Any]:
    zattrs = root / ".zattrs"
    zarr_json = root / "zarr.json"
    try:
        if zattrs.is_file() and not zattrs.is_symlink():
            if zattrs.stat().st_size > MAX_SENSOR_ROOT_ATTRIBUTES_BYTES:
                _fail(
                    "sensor_metadata_budget_exceeded",
                    ".zattrs",
                    f"root attributes exceed {MAX_SENSOR_ROOT_ATTRIBUTES_BYTES} bytes",
                )
            value = json.loads(zattrs.read_text(encoding="utf-8"))
            return _mapping(value, ".zattrs", code="invalid_sensor_metadata")
        if zarr_json.is_file() and not zarr_json.is_symlink():
            if zarr_json.stat().st_size > MAX_SENSOR_ROOT_ATTRIBUTES_BYTES:
                _fail(
                    "sensor_metadata_budget_exceeded",
                    "zarr.json",
                    f"root attributes exceed {MAX_SENSOR_ROOT_ATTRIBUTES_BYTES} bytes",
                )
            value = _mapping(json.loads(zarr_json.read_text(encoding="utf-8")), "zarr.json")
            return _mapping(value.get("attributes", {}), "zarr.json.attributes")
    except (OSError, json.JSONDecodeError) as exc:
        _fail("invalid_sensor_metadata", str(root), f"could not read root attributes: {exc}")
    _fail(
        "invalid_sensor_metadata",
        str(root),
        "Zarr root has neither .zattrs nor zarr.json attributes",
    )


def _extract_sensor_contract(attributes: Mapping[str, Any]) -> Mapping[str, Any]:
    ultra = attributes.get("ultra")
    if isinstance(ultra, Mapping) and isinstance(ultra.get("sensor_series"), Mapping):
        return cast(Mapping[str, Any], ultra["sensor_series"])
    dotted = attributes.get("ultra.sensor_series")
    if isinstance(dotted, Mapping):
        return cast(Mapping[str, Any], dotted)
    _fail(
        "sensor_metadata_missing",
        "attributes.ultra.sensor_series",
        f"missing {SENSOR_SCHEMA} metadata",
    )


class _ZarrArrays:
    def __init__(self, group: Any) -> None:
        self._group = group

    def __getitem__(self, path: str) -> Any:
        value = self._group[path]
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise KeyError(f"{path!r} is not an array")
        return value


def open_sensor_series(
    path: str | os.PathLike[str],
    *,
    expected_tree_manifest_sha256: str | None = None,
    validate_values: bool = True,
    validation_budget: SensorValidationBudget | None = None,
) -> SensorSeries:
    """Open and validate a local Zarr sensor-series group.

    Zarr remains an optional dependency of the base worker. The NGFF/sandbox runtimes already
    install it; callers in a lean runtime receive a classified error instead of an import-time
    package failure.
    """

    root = Path(path)
    if not isinstance(validate_values, bool):
        _fail("invalid_validation_mode", "validate_values", "validate_values must be boolean")
    if validation_budget is not None and not isinstance(validation_budget, SensorValidationBudget):
        _fail(
            "invalid_validation_budget",
            "validation_budget",
            "validation_budget must be a SensorValidationBudget",
        )
    active_budget = (
        validation_budget or SensorValidationBudget() if validate_values else validation_budget
    )
    if active_budget is not None:
        active_budget.check_wall("sensor_root")
    if root.is_symlink() or not root.is_dir():
        _fail("invalid_sensor_root", str(path), "sensor-series path must be a real directory")
    attributes = _read_root_attributes(root)
    metadata = _extract_sensor_contract(attributes)

    tree_manifest: Any | None = None
    if expected_tree_manifest_sha256 is not None:
        lineage = _mapping(metadata.get("lineage"), "lineage")
        manifest_path = _relative_path(
            lineage.get("tree_manifest_path"), "lineage.tree_manifest_path"
        )
        manifest_file = root / PurePosixPath(manifest_path)
        try:
            manifest_size = manifest_file.stat().st_size
            if manifest_size > MAX_TREE_MANIFEST_BYTES:
                _fail(
                    "tree_manifest_budget_exceeded",
                    manifest_path,
                    f"tree manifest exceeds {MAX_TREE_MANIFEST_BYTES} bytes",
                )
            raw_manifest = manifest_file.read_bytes()
            tree_manifest = json.loads(raw_manifest)
        except (OSError, json.JSONDecodeError) as exc:
            _fail(
                "lineage_manifest_required",
                manifest_path,
                f"could not read deterministic manifest: {exc}",
            )
        canonical = canonical_json_bytes(tree_manifest)
        if raw_manifest != canonical:
            _fail(
                "nondeterministic_tree_manifest",
                manifest_path,
                f"manifest bytes must use {CANONICAL_JSON_SCHEMA}",
            )
        computed = hashlib.sha256(canonical).hexdigest()
        expected = _sha256(expected_tree_manifest_sha256, "expected_tree_manifest_sha256")
        if computed != expected:
            _fail(
                "lineage_manifest_hash_mismatch",
                manifest_path,
                f"expected {expected} != computed {computed}",
            )
        verify_tree_manifest(
            root,
            tree_manifest,
            manifest_path=manifest_path,
            validation_budget=active_budget,
        )

    try:
        import zarr  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        _fail("zarr_dependency_missing", str(path), f"install the ngff runtime extra: {exc}")
    try:
        group = zarr.open_group(str(root), mode="r")
    except Exception as exc:  # noqa: BLE001
        _fail("invalid_sensor_root", str(path), f"zarr could not open the group: {exc}")
    result = parse_sensor_series(
        metadata,
        _ZarrArrays(group),
        tree_manifest=tree_manifest,
        expected_tree_manifest_sha256=expected_tree_manifest_sha256,
        validate_values=validate_values,
        validation_budget=active_budget,
    )
    if expected_tree_manifest_sha256 is not None:
        result = replace(result, lineage=replace(result.lineage, status="tree_verified"))
    return result


def build_min_max_envelope(
    values: Sequence[Any],
    *,
    max_buckets: int,
    validity: Sequence[Any] | None = None,
    saturation: Sequence[Any] | None = None,
) -> MinMaxEnvelope:
    """Build a bounded, deterministic min/max envelope without losing spikes.

    Invalid samples are counted but excluded from extrema. NaN is accepted only when a
    corresponding validity flag is false; infinity is always rejected. Each bucket retains
    source index bounds and extremum indices so a later UI can disclose exact provenance.
    """

    if isinstance(max_buckets, bool) or not isinstance(max_buckets, int) or max_buckets <= 0:
        _fail("invalid_envelope_budget", "max_buckets", "max_buckets must be a positive integer")
    count = len(values)
    if validity is not None and len(validity) != count:
        _fail("quality_shape_mismatch", "validity", "validity length must equal source length")
    if saturation is not None and len(saturation) != count:
        _fail("quality_shape_mismatch", "saturation", "saturation length must equal source length")
    factor = max(1, math.ceil(count / max_buckets)) if count else 1
    buckets: list[EnvelopeBucket] = []
    for start in range(0, count, factor):
        stop = min(count, start + factor)
        minimum: float | None = None
        maximum: float | None = None
        minimum_index: int | None = None
        maximum_index: int | None = None
        valid_count = 0
        invalid_count = 0
        saturation_count = 0
        for index in range(start, stop):
            valid = (
                _as_bool(validity[index], f"validity[{index}]") if validity is not None else True
            )
            saturated = (
                _as_bool(saturation[index], f"saturation[{index}]")
                if saturation is not None
                else False
            )
            number = _as_float(values[index], f"values[{index}]", code="nonfinite_signal")
            if math.isinf(number):
                _fail(
                    "infinite_signal",
                    f"values[{index}]",
                    "infinite signal values are never valid measurements",
                )
            if math.isnan(number) and valid:
                _fail(
                    "nan_without_invalid_flag",
                    f"values[{index}]",
                    "NaN requires a matching false validity flag",
                )
            if saturated:
                saturation_count += 1
            if not valid:
                invalid_count += 1
                continue
            valid_count += 1
            if minimum is None or number < minimum:
                minimum = number
                minimum_index = index
            if maximum is None or number > maximum:
                maximum = number
                maximum_index = index
        buckets.append(
            EnvelopeBucket(
                start_index=start,
                stop_index=stop,
                minimum=minimum,
                maximum=maximum,
                minimum_index=minimum_index,
                maximum_index=maximum_index,
                valid_count=valid_count,
                invalid_count=invalid_count,
                saturation_count=saturation_count,
            )
        )
    return MinMaxEnvelope(source_count=count, factor=factor, buckets=tuple(buckets))
