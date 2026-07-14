"""Scientific-contract tests for generic chunked sensor series.

The fixtures deliberately span three unlike materials workflows so the contract cannot
quietly collapse into an acoustic-emission-only or mechanical-test-only schema.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest
from ultra_deepagents.sensors import (
    SensorValidationBudget,
    SensorValidationError,
    build_min_max_envelope,
    canonical_json_bytes,
    canonical_sha256,
    open_sensor_series,
    parse_sensor_series,
    verify_tree_manifest,
)


class _FakeDType:
    def __init__(self, name: str, kind: str, itemsize: int) -> None:
        self.name = name
        self.kind = kind
        self.itemsize = itemsize

    def __str__(self) -> str:
        return self.name


class _FakeArray:
    """Small in-memory stand-in for the subset of the zarr.Array API we validate."""

    def __init__(
        self,
        values: list[Any],
        *,
        dtype: str = "float64",
        kind: str = "f",
        itemsize: int = 8,
        chunks: tuple[int, ...] | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> None:
        self._values = values
        self.shape = shape if shape is not None else (len(values),)
        self.dtype = _FakeDType(dtype, kind, itemsize)
        self.chunks = chunks if chunks is not None else (max(1, min(4, len(values))),)

    def __getitem__(self, key: slice | int):
        return self._values[key]


class _TrackingArray(_FakeArray):
    """Records reads so tests can enforce the parser's bounded-I/O contract."""

    def __init__(self, values: list[Any], **kwargs: Any) -> None:
        super().__init__(values, **kwargs)
        self.reads: list[slice | int] = []

    def __getitem__(self, key: slice | int):
        self.reads.append(key)
        return super().__getitem__(key)


def _unit(ucum_code: str, qudt_name: str) -> dict[str, str]:
    return {
        "label": ucum_code,
        "ucum_code": ucum_code,
        "qudt_uri": f"http://qudt.org/vocab/unit/{qudt_name}",
    }


SECOND = _unit("s", "SEC")
VOLT = _unit("V", "V")
MEGAPASCAL = _unit("MPa", "MegaPA")
STRAIN = _unit("1", "UNITLESS")
CELSIUS = _unit("Cel", "DEG_C")
WATT = _unit("W", "W")
METRE = _unit("m", "M")


def _calibration(unit: dict[str, str], *, calibration_id: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        "kind": "identity",
        "applied": True,
        "calibration_id": calibration_id,
        "revision": "1",
        "input_unit": copy.deepcopy(unit),
        "output_unit": copy.deepcopy(unit),
        "scale": 1.0,
        "offset": 0.0,
    }
    value["parameters_sha256"] = canonical_sha256(value)
    return value


def _clock_accuracy(uncertainty_seconds: float = 1e-8) -> dict[str, Any]:
    return {
        "status": "quantified",
        "standard_uncertainty_seconds": uncertainty_seconds,
        "method": "manufacturer_specification",
    }


def _base_ae_fixture() -> tuple[dict[str, Any], dict[str, _FakeArray]]:
    metadata = {
        "schema": "ultra.sensor-series.v1",
        "series_id": "ae-coupon-17",
        "modality": "acoustic_emission",
        "specimen": {"specimen_id": "coupon-17", "material_id": "IN718"},
        "clocks": [
            {
                "clock_id": "ae-daq",
                "kind": "regular",
                "sample_count": 8,
                "reference": "relative",
                "time_unit": copy.deepcopy(SECOND),
                "start_time_seconds": -2e-6,
                "sample_rate_hz": 2_000_000.0,
                "accuracy": _clock_accuracy(),
            }
        ],
        "channels": [
            {
                "channel_id": "ae-1",
                "name": "AE sensor 1 voltage",
                "array": "signals/ae-1",
                "clock_id": "ae-daq",
                "quantity_kind_uri": "http://qudt.org/vocab/quantitykind/Voltage",
                "unit": copy.deepcopy(VOLT),
                "calibration": _calibration(VOLT, calibration_id="ae-chain-1"),
                "uncertainty": {"kind": "standard", "value": 0.002, "unit": copy.deepcopy(VOLT)},
                "quality": {"saturation_array": "quality/ae-1-saturated"},
                "coordinate_frame_id": "sensor-head",
            }
        ],
        "coordinate_frames": [
            {
                "frame_id": "sensor-head",
                "axes": [
                    {"name": "x", "unit": copy.deepcopy(METRE)},
                    {"name": "y", "unit": copy.deepcopy(METRE)},
                    {"name": "z", "unit": copy.deepcopy(METRE)},
                ],
            },
            {
                "frame_id": "specimen",
                "axes": [
                    {"name": "x", "unit": copy.deepcopy(METRE)},
                    {"name": "y", "unit": copy.deepcopy(METRE)},
                    {"name": "z", "unit": copy.deepcopy(METRE)},
                ],
            },
        ],
        "coordinate_transforms": [
            {
                "transform_id": "sensor-to-specimen",
                "kind": "affine",
                "input_frame_id": "sensor-head",
                "output_frame_id": "specimen",
                "matrix": [
                    [1.0, 0.0, 0.0, 0.012],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.004],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        ],
        "multiscales": [],
        "linked_resources": [],
        "lineage": {"tree_manifest_path": ".ultra/tree-manifest.json"},
    }
    arrays = {
        "signals/ae-1": _FakeArray([0.0, 0.02, -0.03, 3.2, -2.7, 0.04, 0.01, 0.0]),
        "quality/ae-1-saturated": _FakeArray(
            [False, False, False, True, True, False, False, False],
            dtype="bool",
            kind="b",
            itemsize=1,
        ),
    }
    return metadata, arrays


@pytest.fixture
def ae_fixture() -> tuple[dict[str, Any], dict[str, _FakeArray]]:
    return _base_ae_fixture()


@pytest.fixture
def stress_strain_fixture() -> tuple[dict[str, Any], dict[str, _FakeArray]]:
    metadata, arrays = _base_ae_fixture()
    metadata["series_id"] = "tensile-304l-01"
    metadata["modality"] = "mechanical_test"
    metadata["clocks"][0].update({"clock_id": "frame", "sample_rate_hz": 10.0, "sample_count": 6})
    metadata["channels"] = [
        {
            "channel_id": "engineering-strain",
            "name": "Engineering strain",
            "array": "signals/strain",
            "clock_id": "frame",
            "quantity_kind_uri": "http://qudt.org/vocab/quantitykind/Strain",
            "unit": copy.deepcopy(STRAIN),
            "calibration": _calibration(STRAIN, calibration_id="dic-2026-01"),
            "uncertainty": {"kind": "standard", "value": 0.0001, "unit": copy.deepcopy(STRAIN)},
            "quality": {"validity_array": "quality/valid"},
        },
        {
            "channel_id": "engineering-stress",
            "name": "Engineering stress",
            "array": "signals/stress",
            "clock_id": "frame",
            "quantity_kind_uri": "http://qudt.org/vocab/quantitykind/Stress",
            "unit": copy.deepcopy(MEGAPASCAL),
            "calibration": _calibration(MEGAPASCAL, calibration_id="loadcell-88"),
            "uncertainty": {
                "kind": "standard",
                "array": "uncertainty/stress",
                "unit": copy.deepcopy(MEGAPASCAL),
            },
            "quality": {"validity_array": "quality/valid"},
        },
    ]
    metadata["coordinate_frames"] = []
    metadata["coordinate_transforms"] = []
    arrays = {
        "signals/strain": _FakeArray([0.0, 0.002, 0.004, 0.01, 0.03, 0.08]),
        "signals/stress": _FakeArray([0.0, 392.0, 425.0, 452.0, 518.0, 471.0]),
        "uncertainty/stress": _FakeArray([0.8, 0.8, 0.8, 1.0, 1.2, 1.3]),
        "quality/valid": _FakeArray([True] * 6, dtype="bool", kind="b", itemsize=1),
    }
    return metadata, arrays


@pytest.fixture
def thermal_fixture() -> tuple[dict[str, Any], dict[str, _FakeArray]]:
    metadata, _ = _base_ae_fixture()
    metadata["series_id"] = "lpbf-thermal-run-4"
    metadata["modality"] = "thermal_telemetry"
    metadata["clocks"] = [
        {
            "clock_id": "camera",
            "kind": "explicit",
            "sample_count": 5,
            "reference": "instrument",
            "time_unit": copy.deepcopy(SECOND),
            "timestamp_array": "coordinates/frame-time",
            "accuracy": _clock_accuracy(2e-6),
        }
    ]
    metadata["channels"] = [
        {
            "channel_id": "detector-temperature",
            "name": "Detector temperature",
            "array": "signals/detector-temperature",
            "clock_id": "camera",
            "quantity_kind_uri": "http://qudt.org/vocab/quantitykind/Temperature",
            "unit": copy.deepcopy(CELSIUS),
            "calibration": _calibration(CELSIUS, calibration_id="ir-detector-cal-9"),
            "uncertainty": {"kind": "standard", "value": 0.4, "unit": copy.deepcopy(CELSIUS)},
            "quality": {},
        },
        {
            "channel_id": "laser-power",
            "name": "Commanded laser power",
            "array": "signals/laser-power",
            "clock_id": "camera",
            "quantity_kind_uri": "http://qudt.org/vocab/quantitykind/Power",
            "unit": copy.deepcopy(WATT),
            "calibration": _calibration(WATT, calibration_id="laser-monitor-2"),
            "uncertainty": {
                "kind": "not_quantified",
                "reason": "Controller export omitted uncertainty.",
            },
            "quality": {},
        },
    ]
    metadata["coordinate_frames"] = []
    metadata["coordinate_transforms"] = []
    metadata["linked_resources"] = [
        {
            "role": "thermal_video",
            "resource_id": "file_ir_ngff_4",
            "sha256": "a" * 64,
            "frame_clock_id": "camera",
        }
    ]
    arrays = {
        "coordinates/frame-time": _FakeArray([0.0, 0.0002, 0.000401, 0.000603, 0.000806]),
        "signals/detector-temperature": _FakeArray([31.2, 31.3, 31.4, 31.5, 31.6]),
        "signals/laser-power": _FakeArray([0.0, 180.0, 180.0, 180.0, 0.0]),
    }
    return metadata, arrays


def _assert_code(exc: pytest.ExceptionInfo[SensorValidationError], code: str) -> None:
    assert exc.value.code == code, str(exc.value)


def test_accepts_ae_waveform_with_clock_calibration_quality_and_transform(ae_fixture):
    metadata, arrays = ae_fixture
    result = parse_sensor_series(metadata, arrays)
    assert result.modality == "acoustic_emission"
    assert result.specimen.specimen_id == "coupon-17"
    assert result.specimen.material_id == "IN718"
    assert result.clocks[0].sample_rate_hz == 2_000_000.0
    assert result.clocks[0].accuracy_method == "manufacturer_specification"
    assert result.channels[0].saturation_count == 2
    assert result.channels[0].invalid_count == 0
    assert result.lineage.status == "unbound"
    assert "lineage_unbound" in result.warnings


def test_rejects_missing_invalid_or_silently_ignored_specimen_identity(ae_fixture):
    metadata, arrays = ae_fixture
    metadata.pop("specimen")
    with pytest.raises(SensorValidationError) as exc:
        parse_sensor_series(metadata, arrays)
    _assert_code(exc, "invalid_specimen")

    metadata, arrays = _base_ae_fixture()
    metadata["specimen"]["material_id"] = "IN 718"
    with pytest.raises(SensorValidationError) as exc:
        parse_sensor_series(metadata, arrays)
    _assert_code(exc, "invalid_token")

    metadata, arrays = _base_ae_fixture()
    metadata["specimen"]["lot_id"] = "heat-44"
    with pytest.raises(SensorValidationError) as exc:
        parse_sensor_series(metadata, arrays)
    _assert_code(exc, "unknown_specimen_fields")


def test_accepts_mechanical_stress_strain_with_array_uncertainty(stress_strain_fixture):
    metadata, arrays = stress_strain_fixture
    result = parse_sensor_series(metadata, arrays)
    assert [channel.channel_id for channel in result.channels] == [
        "engineering-strain",
        "engineering-stress",
    ]
    assert result.channels[1].uncertainty_kind == "standard"
    assert result.channels[1].unit.ucum_code == "MPa"


def test_accepts_thermal_telemetry_linked_to_ngff_video(thermal_fixture):
    metadata, arrays = thermal_fixture
    result = parse_sensor_series(metadata, arrays)
    assert result.clocks[0].kind == "explicit"
    assert result.linked_resources[0].role == "thermal_video"
    assert result.linked_resources[0].verification_status == "declared_unverified"
    assert result.linked_resources[0].verification_authority is None
    assert result.channels[1].uncertainty_reason == "Controller export omitted uncertainty."
    assert "uncertainty_not_quantified:laser-power" in result.warnings
    assert "linked_resource_declared_unverified:thermal_video:file_ir_ngff_4" in result.warnings


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("authority", "control_resource_catalog"),
        ("verification_authority", "control_resource_catalog"),
        ("verification_status", "catalog_identity_verified"),
        ("verified", True),
    ],
)
def test_rejects_caller_claims_of_linked_resource_verification(
    thermal_fixture,
    field: str,
    value: object,
) -> None:
    metadata, arrays = thermal_fixture
    metadata["linked_resources"][0][field] = value

    with pytest.raises(SensorValidationError) as exc:
        parse_sensor_series(metadata, arrays)

    _assert_code(exc, "caller_linked_resource_verification_forbidden")


def test_rejects_unknown_linked_resource_fields(thermal_fixture) -> None:
    metadata, arrays = thermal_fixture
    metadata["linked_resources"][0]["model_hint"] = "trust-this-link"

    with pytest.raises(SensorValidationError) as exc:
        parse_sensor_series(metadata, arrays)

    _assert_code(exc, "unknown_linked_resource_fields")


@pytest.mark.parametrize(
    ("mutate", "expected_path"),
    [
        (lambda metadata: metadata.update({"extension": True}), "$"),
        (lambda metadata: metadata["clocks"][0].update({"epoch_hint": 0}), "clocks[0]"),
        (
            lambda metadata: metadata["clocks"][0]["accuracy"].update({"confidence": 0.95}),
            "clocks[0].accuracy",
        ),
        (
            lambda metadata: metadata["clocks"][0]["time_unit"].update({"symbol": "s"}),
            "clocks[0].time_unit",
        ),
        (lambda metadata: metadata["channels"][0].update({"gain_hint": 12}), "channels[0]"),
        (
            lambda metadata: metadata["channels"][0]["unit"].update({"symbol": "V"}),
            "channels[0].unit",
        ),
        (
            lambda metadata: metadata["channels"][0]["calibration"].update({"fit_residual": 0.1}),
            "channels[0].calibration",
        ),
        (
            lambda metadata: metadata["channels"][0]["uncertainty"].update({"confidence": 0.95}),
            "channels[0].uncertainty",
        ),
        (
            lambda metadata: metadata["channels"][0]["quality"].update(
                {"interpolated_array": "quality/interpolated"}
            ),
            "channels[0].quality",
        ),
        (
            lambda metadata: metadata["coordinate_frames"][0].update({"handedness": "right"}),
            "coordinate_frames[0]",
        ),
        (
            lambda metadata: metadata["coordinate_frames"][0]["axes"][0].update(
                {"positive": "right"}
            ),
            "coordinate_frames[0].axes[0]",
        ),
        (
            lambda metadata: metadata["coordinate_transforms"][0].update(
                {"convention": "column-vector"}
            ),
            "coordinate_transforms[0]",
        ),
        (
            lambda metadata: metadata["lineage"].update({"producer": "caller"}),
            "lineage",
        ),
    ],
)
def test_sensor_contract_rejects_unknown_fields_at_each_governed_object_level(
    ae_fixture, mutate, expected_path
):
    metadata, arrays = ae_fixture
    mutate(metadata)

    with pytest.raises(SensorValidationError) as exc:
        parse_sensor_series(metadata, arrays)

    _assert_code(exc, "unknown_sensor_fields")
    assert exc.value.path == expected_path


@pytest.mark.parametrize("target", ["level", "envelope_channel"])
def test_multiscale_contract_rejects_unknown_fields(ae_fixture, target):
    metadata, arrays = ae_fixture
    metadata["multiscales"] = [
        {
            "level": 1,
            "kind": "min_max_envelope",
            "factor": 4,
            "channels": [
                {
                    "channel_id": "ae-1",
                    "min_array": "envelopes/4/ae-1-min",
                    "max_array": "envelopes/4/ae-1-max",
                }
            ],
        }
    ]
    if target == "level":
        metadata["multiscales"][0]["reducer"] = "minimum-maximum"
        expected_path = "multiscales[0]"
    else:
        metadata["multiscales"][0]["channels"][0]["color"] = "FF0000"
        expected_path = "multiscales[0].channels[0]"

    with pytest.raises(SensorValidationError) as exc:
        parse_sensor_series(metadata, arrays, validate_values=False)

    _assert_code(exc, "unknown_sensor_fields")
    assert exc.value.path == expected_path


@pytest.mark.parametrize("target", ["manifest", "entry"])
def test_tree_manifest_contract_rejects_unknown_fields(ae_fixture, target):
    metadata, arrays = ae_fixture
    manifest = {
        "schema": "ultra.tree-manifest.v1",
        "entries": [{"path": ".zattrs", "size_bytes": 2, "sha256": "1" * 64}],
    }
    if target == "manifest":
        manifest["digest_algorithm"] = "sha256"
        expected_path = "tree_manifest"
    else:
        manifest["entries"][0]["media_type"] = "application/json"
        expected_path = "tree_manifest.entries[0]"

    with pytest.raises(SensorValidationError) as exc:
        parse_sensor_series(metadata, arrays, tree_manifest=manifest)

    _assert_code(exc, "unknown_sensor_fields")
    assert exc.value.path == expected_path


def test_large_unknown_field_attack_fails_after_first_key(ae_fixture):
    metadata, arrays = ae_fixture

    class CountingMetadata(dict):
        yielded = 0

        def __iter__(self):
            for key in super().__iter__():
                self.yielded += 1
                yield key

    adversarial = CountingMetadata({"untrusted-extension": "x"})
    adversarial.update({f"extra-{index}": index for index in range(50_000)})
    adversarial.update(metadata)

    with pytest.raises(SensorValidationError) as exc:
        parse_sensor_series(adversarial, arrays)

    _assert_code(exc, "unknown_sensor_fields")
    assert adversarial.yielded == 1


@pytest.mark.parametrize(
    ("quantity_kind", "ucum_code", "qudt_name"),
    [
        ("Force", "kN", "KiloN"),
        ("Displacement", "mm", "MilliM"),
        ("ElectricCurrent", "mA", "MilliA"),
        ("Frequency", "kHz", "KiloHZ"),
        ("ThermodynamicTemperature", "K", "K"),
        ("Pressure", "kPa", "KiloPA"),
        ("ModulusOfElasticity", "GPa", "GigaPA"),
        ("LinearVelocity", "mm.s-1", "MilliM-PER-SEC"),
        ("Acceleration", "m.s-2", "M-PER-SEC2"),
        ("Torque", "N.m", "N-M"),
        ("Energy", "J", "J"),
        ("Mass", "kg", "KiloGM"),
        ("PlaneAngle", "rad", "RAD"),
    ],
)
def test_accepts_common_materials_sensor_quantity_unit_identities(
    ae_fixture, quantity_kind: str, ucum_code: str, qudt_name: str
):
    metadata, arrays = ae_fixture
    unit = _unit(ucum_code, qudt_name)
    channel = metadata["channels"][0]
    channel["quantity_kind_uri"] = f"http://qudt.org/vocab/quantitykind/{quantity_kind}"
    channel["unit"] = copy.deepcopy(unit)
    channel["calibration"] = _calibration(unit, calibration_id="qualified-common-unit")
    channel["uncertainty"] = {
        "kind": "standard",
        "value": 0.002,
        "unit": copy.deepcopy(unit),
    }

    result = parse_sensor_series(metadata, arrays)

    assert result.channels[0].quantity_kind_uri.endswith(f"/{quantity_kind}")
    assert result.channels[0].unit.ucum_code == ucum_code
    assert result.channels[0].unit.qudt_uri.endswith(f"/{qudt_name}")


def test_preserves_reason_when_clock_accuracy_is_not_quantified(ae_fixture):
    metadata, arrays = ae_fixture
    metadata["clocks"][0]["accuracy"] = {
        "status": "not_quantified",
        "reason": "DAQ export did not include the synchronization budget.",
    }
    result = parse_sensor_series(metadata, arrays)
    assert result.clocks[0].accuracy_method is None
    assert (
        result.clocks[0].accuracy_reason == "DAQ export did not include the synchronization budget."
    )
    assert "clock_accuracy_not_quantified:ae-daq" in result.warnings


@pytest.mark.parametrize("reference", ["utc", "tai"])
def test_rejects_absolute_clock_reference_without_an_explicit_epoch_contract(ae_fixture, reference):
    metadata, arrays = ae_fixture
    metadata["clocks"][0]["reference"] = reference
    with pytest.raises(SensorValidationError) as exc:
        parse_sensor_series(metadata, arrays)
    _assert_code(exc, "absolute_clock_epoch_required")


def test_value_validation_uses_bounded_reads_and_metadata_preflight_reads_no_payload(ae_fixture):
    metadata, arrays = ae_fixture
    count = 65_537
    metadata["clocks"][0]["sample_count"] = count
    signal = _TrackingArray([0.0] * count, chunks=(4096,))
    saturation = _TrackingArray([False] * count, dtype="bool", kind="b", itemsize=1, chunks=(4096,))
    arrays["signals/ae-1"] = signal
    arrays["quality/ae-1-saturated"] = saturation

    preflight = parse_sensor_series(metadata, arrays, validate_values=False)
    assert not preflight.values_validated
    assert signal.reads == []
    assert saturation.reads == []

    validated = parse_sensor_series(metadata, arrays)
    assert validated.values_validated
    for reads in (signal.reads, saturation.reads):
        slices = [read for read in reads if isinstance(read, slice)]
        assert len(slices) == 2
        assert max(int(read.stop) - int(read.start) for read in slices) <= 65_536


def test_one_budget_accounts_for_explicit_clock_signal_quality_and_uncertainty(
    thermal_fixture,
) -> None:
    metadata, arrays = thermal_fixture
    budget = SensorValidationBudget(
        max_values=100,
        max_decoded_bytes=10_000,
        max_reads=20,
        max_wall_seconds=5.0,
    )

    result = parse_sensor_series(metadata, arrays, validation_budget=budget)

    assert result.values_validated is True
    snapshot = budget.snapshot()
    assert snapshot["decoded_values"] == 15
    assert snapshot["decoded_bytes"] == 120
    assert snapshot["read_operations"] == 3


def test_unused_explicit_clock_is_budgeted_before_its_first_value_read(ae_fixture) -> None:
    metadata, arrays = ae_fixture
    timestamps = _TrackingArray([float(index) for index in range(11)])
    arrays["coordinates/unused-time"] = timestamps
    metadata["clocks"].append(
        {
            "clock_id": "unused-clock",
            "kind": "explicit",
            "sample_count": 11,
            "reference": "relative",
            "time_unit": copy.deepcopy(SECOND),
            "timestamp_array": "coordinates/unused-time",
            "accuracy": _clock_accuracy(),
        }
    )
    budget = SensorValidationBudget(
        max_values=10,
        max_decoded_bytes=1_000,
        max_reads=10,
        max_wall_seconds=5.0,
    )

    with pytest.raises(SensorValidationError) as exc:
        parse_sensor_series(metadata, arrays, validation_budget=budget)

    _assert_code(exc, "validation_budget_exceeded")
    assert timestamps.reads == []
    assert budget.snapshot()["decoded_values"] == 0


def test_quality_uncertainty_and_multiscale_rereads_share_the_same_budget(
    stress_strain_fixture,
    ae_fixture,
) -> None:
    stress_metadata, stress_arrays = stress_strain_fixture
    tracked_validity = _TrackingArray(
        [True] * 6,
        dtype="bool",
        kind="b",
        itemsize=1,
    )
    tracked_stress = _TrackingArray([0.0, 392.0, 425.0, 452.0, 518.0, 471.0])
    stress_arrays["quality/valid"] = tracked_validity
    stress_arrays["signals/stress"] = tracked_stress
    quality_budget = SensorValidationBudget(
        max_values=20,
        max_decoded_bytes=10_000,
        max_reads=20,
        max_wall_seconds=5.0,
    )

    with pytest.raises(SensorValidationError) as quality_exc:
        parse_sensor_series(
            stress_metadata,
            stress_arrays,
            validation_budget=quality_budget,
        )
    _assert_code(quality_exc, "validation_budget_exceeded")
    assert len(tracked_validity.reads) == 1
    assert len(tracked_stress.reads) == 1
    assert quality_budget.snapshot()["decoded_values"] == 18

    envelope_metadata, envelope_arrays = ae_fixture
    tracked_signal = _TrackingArray([0.0, 0.02, -0.03, 3.2, -2.7, 0.04, 0.01, 0.0])
    envelope_arrays["signals/ae-1"] = tracked_signal
    envelope_metadata["multiscales"] = [
        {
            "level": 1,
            "kind": "min_max_envelope",
            "factor": 4,
            "channels": [
                {
                    "channel_id": "ae-1",
                    "min_array": "envelopes/4/min",
                    "max_array": "envelopes/4/max",
                }
            ],
        }
    ]
    minimum = _TrackingArray([-0.03, -2.7], chunks=(2,))
    maximum = _TrackingArray([3.2, 0.04], chunks=(2,))
    envelope_arrays["envelopes/4/min"] = minimum
    envelope_arrays["envelopes/4/max"] = maximum
    multiscale_budget = SensorValidationBudget(
        max_values=20,
        max_decoded_bytes=10_000,
        max_reads=20,
        max_wall_seconds=5.0,
    )

    with pytest.raises(SensorValidationError) as multiscale_exc:
        parse_sensor_series(
            envelope_metadata,
            envelope_arrays,
            validation_budget=multiscale_budget,
        )
    _assert_code(multiscale_exc, "validation_budget_exceeded")
    assert len(tracked_signal.reads) == 1
    assert minimum.reads == []
    assert maximum.reads == []


def test_decoded_byte_and_wall_budgets_fail_before_array_read(
    ae_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, arrays = ae_fixture
    signal = _TrackingArray([0.0] * 8)
    arrays["signals/ae-1"] = signal
    byte_budget = SensorValidationBudget(
        max_values=100,
        max_decoded_bytes=4,
        max_reads=10,
        max_wall_seconds=5.0,
    )
    with pytest.raises(SensorValidationError) as byte_exc:
        parse_sensor_series(metadata, arrays, validation_budget=byte_budget)
    _assert_code(byte_exc, "validation_budget_exceeded")
    assert signal.reads == []

    ticks = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(
        "ultra_deepagents.sensors.reader.time.monotonic",
        lambda: next(ticks, 2.0),
    )
    wall_budget = SensorValidationBudget(
        max_values=100,
        max_decoded_bytes=1_000,
        max_reads=10,
        max_wall_seconds=1.0,
    )
    with pytest.raises(SensorValidationError) as wall_exc:
        parse_sensor_series(metadata, arrays, validation_budget=wall_budget)
    _assert_code(wall_exc, "validation_budget_exceeded")
    assert signal.reads == []


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda m, a: m["channels"][0].update({"unit": {"label": "volts-ish"}}), "unit_unbound"),
        (
            lambda m, a: m["channels"][0].update(
                {
                    "unit": {
                        "label": "V",
                        "ucum_code": "V",
                        "qudt_uri": "https://qudt.org/vocab/unit/V",
                    }
                }
            ),
            "invalid_qudt_unit",
        ),
        (
            lambda m, a: m["channels"][0].update(
                {"quantity_kind_uri": "https://qudt.org/vocab/quantitykind/Voltage"}
            ),
            "invalid_quantity_kind",
        ),
        (
            lambda m, a: m["channels"][0]["unit"].update(
                {"qudt_uri": "http://qudt.org/vocab/unit/KiloGM"}
            ),
            "unit_identity_mismatch",
        ),
        (
            lambda m, a: m["channels"][0].update(
                {"quantity_kind_uri": "http://qudt.org/vocab/quantitykind/Mass"}
            ),
            "quantity_unit_mismatch",
        ),
        (
            lambda m, a: m["channels"][0]["calibration"].update({"parameters_sha256": "0" * 64}),
            "calibration_hash_mismatch",
        ),
        (
            lambda m, a: m["channels"][0]["uncertainty"].update({"value": -0.1}),
            "invalid_uncertainty",
        ),
        (lambda m, a: setattr(a["signals/ae-1"], "chunks", (0,)), "invalid_chunks"),
        (lambda m, a: setattr(a["signals/ae-1"], "chunks", (4.5,)), "invalid_chunks"),
        (lambda m, a: setattr(a["signals/ae-1"], "chunks", (9_000_000,)), "chunk_too_large"),
        (lambda m, a: setattr(a["signals/ae-1"], "shape", (8.5,)), "invalid_array_shape"),
        (lambda m, a: setattr(a["signals/ae-1"].dtype, "itemsize", 8.5), "invalid_array_dtype"),
        (lambda m, a: a.update({"signals/ae-1": _FakeArray([0.0] * 7)}), "sample_count_mismatch"),
    ],
)
def test_rejects_adversarial_units_calibration_uncertainty_chunks_and_lengths(
    ae_fixture, mutate, code
):
    metadata, arrays = ae_fixture
    mutate(metadata, arrays)
    with pytest.raises(SensorValidationError) as exc:
        parse_sensor_series(metadata, arrays)
    _assert_code(exc, code)


def test_affine_transform_rejects_nonzero_length_to_time_axis_mixing(ae_fixture):
    metadata, arrays = ae_fixture
    for axis in metadata["coordinate_frames"][1]["axes"]:
        axis["unit"] = copy.deepcopy(SECOND)

    with pytest.raises(SensorValidationError) as exc:
        parse_sensor_series(metadata, arrays)
    _assert_code(exc, "transform_axis_unit_mismatch")


@pytest.mark.parametrize(
    ("timestamps", "code"),
    [
        ([0.0, 0.2, 0.19, 0.3, 0.4], "clock_not_strictly_increasing"),
        ([0.0, 0.2, math.nan, 0.3, 0.4], "nonfinite_timestamp"),
        ([0.0, 0.2, 0.2, 0.3, 0.4], "clock_not_strictly_increasing"),
    ],
)
def test_rejects_nonmonotonic_nonfinite_or_duplicate_explicit_timestamps(
    thermal_fixture, timestamps, code
):
    metadata, arrays = thermal_fixture
    arrays["coordinates/frame-time"] = _FakeArray(timestamps)
    with pytest.raises(SensorValidationError) as exc:
        parse_sensor_series(metadata, arrays)
    _assert_code(exc, code)


def test_nan_signal_requires_matching_false_validity_flag(ae_fixture):
    metadata, arrays = ae_fixture
    arrays["signals/ae-1"] = _FakeArray([0.0, math.nan, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(SensorValidationError) as exc:
        parse_sensor_series(metadata, arrays)
    _assert_code(exc, "nan_without_invalid_flag")

    metadata["channels"][0]["quality"]["validity_array"] = "quality/ae-1-valid"
    arrays["quality/ae-1-valid"] = _FakeArray(
        [True, False, True, True, True, True, True, True], dtype="bool", kind="b", itemsize=1
    )
    result = parse_sensor_series(metadata, arrays)
    assert result.channels[0].invalid_count == 1


def test_rejects_infinite_signal_even_when_marked_invalid(ae_fixture):
    metadata, arrays = ae_fixture
    metadata["channels"][0]["quality"]["validity_array"] = "quality/valid"
    arrays["signals/ae-1"] = _FakeArray([0.0, math.inf, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    arrays["quality/valid"] = _FakeArray(
        [True, False] + [True] * 6, dtype="bool", kind="b", itemsize=1
    )
    with pytest.raises(SensorValidationError) as exc:
        parse_sensor_series(metadata, arrays)
    _assert_code(exc, "infinite_signal")


@pytest.mark.parametrize(
    ("replacement", "code"),
    [
        (_FakeArray([0, 0, 1, 0, 0, 0, 0, 0], dtype="uint8", kind="u", itemsize=1), "flag_dtype"),
        (_FakeArray([False] * 7, dtype="bool", kind="b", itemsize=1), "quality_shape_mismatch"),
    ],
)
def test_rejects_nonboolean_or_misaligned_saturation_flags(ae_fixture, replacement, code):
    metadata, arrays = ae_fixture
    arrays["quality/ae-1-saturated"] = replacement
    with pytest.raises(SensorValidationError) as exc:
        parse_sensor_series(metadata, arrays)
    _assert_code(exc, code)


def test_rejects_nonfinite_or_negative_uncertainty_array(stress_strain_fixture):
    metadata, arrays = stress_strain_fixture
    arrays["uncertainty/stress"] = _FakeArray([0.8, 0.8, -0.1, 1.0, 1.2, 1.3])
    with pytest.raises(SensorValidationError) as exc:
        parse_sensor_series(metadata, arrays)
    _assert_code(exc, "invalid_uncertainty_array")


@pytest.mark.parametrize(
    ("matrix", "code"),
    [
        ([[1.0, 0.0], [0.0, 1.0]], "transform_dimension_mismatch"),
        (
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, math.nan, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "nonfinite_transform",
        ),
    ],
)
def test_rejects_invalid_coordinate_transforms(ae_fixture, matrix, code):
    metadata, arrays = ae_fixture
    metadata["coordinate_transforms"][0]["matrix"] = matrix
    with pytest.raises(SensorValidationError) as exc:
        parse_sensor_series(metadata, arrays)
    _assert_code(exc, code)


def test_validates_min_max_multiscale_arrays(ae_fixture):
    metadata, arrays = ae_fixture
    metadata["multiscales"] = [
        {
            "level": 1,
            "kind": "min_max_envelope",
            "factor": 4,
            "channels": [
                {
                    "channel_id": "ae-1",
                    "min_array": "envelopes/4/ae-1-min",
                    "max_array": "envelopes/4/ae-1-max",
                }
            ],
        }
    ]
    arrays["envelopes/4/ae-1-min"] = _FakeArray([-0.03, -2.7], chunks=(2,))
    arrays["envelopes/4/ae-1-max"] = _FakeArray([3.2, 0.04], chunks=(2,))
    result = parse_sensor_series(metadata, arrays)
    assert result.multiscales[0].factor == 4

    arrays["envelopes/4/ae-1-min"] = _FakeArray([4.0, -2.7], chunks=(2,))
    with pytest.raises(SensorValidationError) as exc:
        parse_sensor_series(metadata, arrays)
    _assert_code(exc, "invalid_envelope_bounds")

    arrays["envelopes/4/ae-1-min"] = _FakeArray([-0.03, -2.7], chunks=(2,))
    arrays["envelopes/4/ae-1-max"] = _FakeArray([3.1, 0.04], chunks=(2,))
    with pytest.raises(SensorValidationError) as stale:
        parse_sensor_series(metadata, arrays)
    _assert_code(stale, "stale_envelope")


def test_expected_manifest_digest_is_required_and_authoritatively_bound(ae_fixture):
    metadata, arrays = ae_fixture
    manifest = {
        "schema": "ultra.tree-manifest.v1",
        "entries": [
            {"path": ".zattrs", "size_bytes": 17, "sha256": "1" * 64},
            {"path": "signals/ae-1/0", "size_bytes": 64, "sha256": "2" * 64},
        ],
    }
    digest = canonical_sha256(manifest)
    bound = parse_sensor_series(
        metadata,
        arrays,
        tree_manifest=manifest,
        expected_tree_manifest_sha256=digest,
    )
    assert bound.lineage.status == "manifest_verified"
    assert bound.lineage.expected_tree_manifest_sha256 == digest

    with pytest.raises(SensorValidationError) as missing:
        parse_sensor_series(metadata, arrays, expected_tree_manifest_sha256=digest)
    _assert_code(missing, "lineage_manifest_required")

    with pytest.raises(SensorValidationError) as mismatch:
        parse_sensor_series(
            metadata,
            arrays,
            tree_manifest=manifest,
            expected_tree_manifest_sha256="f" * 64,
        )
    _assert_code(mismatch, "lineage_manifest_hash_mismatch")


def _write_manifest_entry(root: Path, relative_path: str, payload: bytes) -> dict[str, Any]:
    destination = root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return {
        "path": relative_path,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_tree_manifest_verification_detects_tamper_extra_files_and_symlinks(tmp_path):
    entries = [
        _write_manifest_entry(tmp_path, ".zattrs", b"{}"),
        _write_manifest_entry(tmp_path, "signals/a/0", b"12345678"),
    ]
    entries.sort(key=lambda item: item["path"])
    manifest = {"schema": "ultra.tree-manifest.v1", "entries": entries}
    manifest_path = ".ultra/tree-manifest.json"
    destination = tmp_path / manifest_path
    destination.parent.mkdir(parents=True)
    destination.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )

    verified = verify_tree_manifest(tmp_path, manifest, manifest_path=manifest_path)
    assert verified.entry_count == 2
    assert verified.size_bytes == 10

    (tmp_path / "signals/a/0").write_bytes(b"tampered")
    with pytest.raises(SensorValidationError) as tampered:
        verify_tree_manifest(tmp_path, manifest, manifest_path=manifest_path)
    _assert_code(tampered, "tree_entry_hash_mismatch")

    (tmp_path / "signals/a/0").write_bytes(b"12345678")
    (tmp_path / "extra").write_bytes(b"not declared")
    with pytest.raises(SensorValidationError) as extra:
        verify_tree_manifest(tmp_path, manifest, manifest_path=manifest_path)
    _assert_code(extra, "tree_manifest_not_closed")

    (tmp_path / "extra").unlink()
    (tmp_path / "link").symlink_to(tmp_path / ".zattrs")
    with pytest.raises(SensorValidationError) as symlink:
        verify_tree_manifest(tmp_path, manifest, manifest_path=manifest_path)
    _assert_code(symlink, "tree_symlink_forbidden")


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_opens_real_zarr_v2_and_v3_with_verified_tree_closure(tmp_path, zarr_format):
    np = pytest.importorskip("numpy")
    zarr = pytest.importorskip("zarr")
    root = tmp_path / f"ae-v{zarr_format}.zarr"
    group = zarr.open_group(str(root), mode="w", zarr_format=zarr_format)
    signals = group.require_group("signals")
    signals.create_array(
        "ae-1",
        data=np.asarray([0.0, 0.25, -0.5, 1.0], dtype="float64"),
        chunks=(2,),
    )
    metadata, _ = _base_ae_fixture()
    metadata["series_id"] = f"real-zarr-v{zarr_format}"
    metadata["clocks"][0]["sample_count"] = 4
    metadata["channels"][0]["quality"] = {}
    metadata["channels"][0].pop("coordinate_frame_id", None)
    metadata["coordinate_frames"] = []
    metadata["coordinate_transforms"] = []
    group.attrs["ultra"] = {"sensor_series": metadata}

    manifest_path = ".ultra/tree-manifest.json"
    entries: list[dict[str, Any]] = []
    for candidate in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = candidate.relative_to(root).as_posix()
        payload = candidate.read_bytes()
        entries.append(
            {
                "path": relative,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = {"schema": "ultra.tree-manifest.v1", "entries": entries}
    destination = root / manifest_path
    destination.parent.mkdir(parents=True)
    destination.write_bytes(canonical_json_bytes(manifest))

    result = open_sensor_series(
        root,
        expected_tree_manifest_sha256=canonical_sha256(manifest),
    )
    assert result.lineage.status == "tree_verified"
    assert result.lineage.entry_count == len(entries)
    assert result.values_validated


def test_min_max_envelope_is_bounded_deterministic_and_preserves_spikes_quality():
    values = [0.0] * 25
    values[7] = 1000.0
    values[8] = -800.0
    values[15] = math.nan
    validity = [True] * 25
    validity[15] = False
    saturation = [False] * 25
    saturation[7] = True

    first = build_min_max_envelope(values, max_buckets=5, validity=validity, saturation=saturation)
    second = build_min_max_envelope(values, max_buckets=5, validity=validity, saturation=saturation)
    assert first == second
    assert len(first.buckets) == 5
    assert first.factor == 5
    assert max(bucket.maximum for bucket in first.buckets if bucket.maximum is not None) == 1000.0
    assert min(bucket.minimum for bucket in first.buckets if bucket.minimum is not None) == -800.0
    assert sum(bucket.invalid_count for bucket in first.buckets) == 1
    assert sum(bucket.saturation_count for bucket in first.buckets) == 1


def test_min_max_envelope_rejects_infinite_values_and_flag_length_mismatch():
    with pytest.raises(SensorValidationError) as infinite:
        build_min_max_envelope([0.0, math.inf], max_buckets=2)
    _assert_code(infinite, "infinite_signal")

    with pytest.raises(SensorValidationError) as shape:
        build_min_max_envelope([0.0, 1.0], max_buckets=2, validity=[True])
    _assert_code(shape, "quality_shape_mismatch")
