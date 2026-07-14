from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest
import zarr
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.sensors import canonical_json_bytes, canonical_sha256
from ultra_deepagents.sensors.tools import (
    MAX_SOURCE_TREE_BYTES,
    MAX_VALUE_VALIDATION_SAMPLES,
    SENSOR_FORMAT_BINDING_SCHEMA,
    build_sensor_tools,
    inspect_selected_sensor_series_resource,
    inspect_selected_sensor_series_text,
    should_register_sensor_tools,
)


def _unit(ucum_code: str, qudt_name: str) -> dict[str, str]:
    return {
        "label": ucum_code,
        "ucum_code": ucum_code,
        "qudt_uri": f"http://qudt.org/vocab/unit/{qudt_name}",
    }


def _sensor_format_marker(digest: str) -> dict[str, str]:
    return {
        "schema": SENSOR_FORMAT_BINDING_SCHEMA,
        "authority": "control_resource_catalog",
        "container": "zarr",
        "sensor_schema": "ultra.sensor-series.v1",
        "resource_sha256": digest,
        "detection": "bounded_root_attributes",
    }


def _metadata(sample_count: int) -> dict[str, object]:
    volt = _unit("V", "V")
    calibration: dict[str, object] = {
        "kind": "identity",
        "applied": True,
        "calibration_id": "ae-chain-1",
        "revision": "2026-07",
        "input_unit": volt,
        "output_unit": volt,
        "scale": 1.0,
        "offset": 0.0,
    }
    calibration["parameters_sha256"] = canonical_sha256(calibration)
    return {
        "schema": "ultra.sensor-series.v1",
        "series_id": "ae-coupon-17",
        "modality": "acoustic_emission",
        "specimen": {"specimen_id": "coupon-17", "material_id": "IN718"},
        "clocks": [
            {
                "clock_id": "ae-daq",
                "kind": "regular",
                "sample_count": sample_count,
                "reference": "relative",
                "time_unit": _unit("s", "SEC"),
                "start_time_seconds": -2e-6,
                "sample_rate_hz": 2_000_000.0,
                "accuracy": {
                    "status": "quantified",
                    "standard_uncertainty_seconds": 1e-8,
                    "method": "manufacturer_specification",
                },
            }
        ],
        "channels": [
            {
                "channel_id": "ae-1",
                "name": "AE sensor voltage",
                "array": "signals/ae-1",
                "clock_id": "ae-daq",
                "quantity_kind_uri": "http://qudt.org/vocab/quantitykind/Voltage",
                "unit": volt,
                "calibration": calibration,
                "uncertainty": {"kind": "standard", "value": 0.002, "unit": volt},
                "quality": {
                    "validity_array": "quality/valid",
                    "saturation_array": "quality/saturated",
                },
            }
        ],
        "coordinate_frames": [],
        "coordinate_transforms": [],
        "multiscales": [],
        "linked_resources": [],
        "lineage": {"tree_manifest_path": ".ultra/tree-manifest.json"},
    }


def _write_sensor_bundle(
    upload_root: Path,
    *,
    file_id: str = "file_sensor",
    values: np.ndarray | None = None,
    sparse_sample_count: int | None = None,
    linked_resources: list[dict[str, object]] | None = None,
) -> Path:
    root = upload_root / "bundles" / file_id / "ae-coupon.zarr"
    group = zarr.open_group(str(root), mode="w", zarr_format=2)
    signals = group.require_group("signals")
    quality = group.require_group("quality")
    if sparse_sample_count is not None:
        count = sparse_sample_count
        signals.create_array(
            "ae-1",
            shape=(count,),
            chunks=(65_536,),
            dtype="float64",
            fill_value=0.0,
        )
        quality.create_array(
            "valid",
            shape=(count,),
            chunks=(65_536,),
            dtype="bool",
            fill_value=True,
        )
        quality.create_array(
            "saturated",
            shape=(count,),
            chunks=(65_536,),
            dtype="bool",
            fill_value=False,
        )
    else:
        data = (
            np.asarray([0.0, 0.1, 1000.0, -800.0, np.nan, 0.0], dtype="float64")
            if values is None
            else np.asarray(values, dtype="float64")
        )
        count = int(data.size)
        signals.create_array("ae-1", data=data, chunks=(min(4096, count),))
        validity = np.ones(count, dtype="bool")
        if count == 6:
            validity[4] = False
        saturated = np.zeros(count, dtype="bool")
        if count >= 3:
            saturated[2] = True
        quality.create_array("valid", data=validity, chunks=(min(4096, count),))
        quality.create_array("saturated", data=saturated, chunks=(min(4096, count),))
    metadata = _metadata(count)
    if linked_resources is not None:
        metadata["linked_resources"] = [dict(resource) for resource in linked_resources]
    group.attrs["ultra"] = {"sensor_series": metadata}
    return root


def _write_tree_manifest(root: Path) -> str:
    manifest_path = ".ultra/tree-manifest.json"
    entries: list[dict[str, object]] = []
    for candidate in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = candidate.relative_to(root).as_posix()
        if relative == manifest_path:
            continue
        payload = candidate.read_bytes()
        entries.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    manifest = {"entries": entries, "schema": "ultra.tree-manifest.v1"}
    destination = root / manifest_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(manifest))
    return canonical_sha256(manifest)


def _context(
    tmp_path: Path,
    *,
    selected_file_ids: tuple[str, ...] = ("file_sensor",),
    tree_manifest_sha256: str | None = None,
    descriptor_sha256s: dict[str, str] | None = None,
    descriptor_original_names: dict[str, str] | None = None,
) -> AgentRunContext:
    descriptor_sha256s = descriptor_sha256s or {}
    descriptor_original_names = descriptor_original_names or {}
    descriptors_list: list[dict[str, object]] = []
    for file_id in selected_file_ids:
        descriptor: dict[str, object] = {
            "type": "selected_resource",
            "binding_schema": "ultra.selected_resource.v1",
            "authority": "control_resource_catalog",
            "resource_id": file_id,
            "file_id": file_id,
            "original_name": descriptor_original_names.get(file_id, "ae-coupon.zarr"),
            "content_type": "application/octet-stream",
            "resource_kind": "dataset",
            "sha256": descriptor_sha256s.get(file_id, "a" * 64),
            "size_bytes": 1234,
        }
        if tree_manifest_sha256 is not None and file_id == "file_sensor":
            descriptor["sha256"] = tree_manifest_sha256
            descriptor["tree_identity"] = {
                "schema": "ultra.resource-tree-identity.v1",
                "authority": "control_resource_catalog",
                "canonical_json_schema": "ultra.canonical-json.v1",
                "tree_manifest_schema": "ultra.tree-manifest.v1",
                "tree_manifest_path": ".ultra/tree-manifest.json",
                "tree_manifest_sha256": tree_manifest_sha256,
                "scope": "all_regular_files_except_tree_manifest",
            }
        descriptors_list.append(descriptor)
    return AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="org-1",
        user_id="user-1",
        project_id="project-1",
        thread_id="thread-1",
        run_id="run-1",
        goal="Inspect the selected acoustic-emission sensor series.",
        selected_file_ids=selected_file_ids,
        resource_descriptors=tuple(descriptors_list),
        workspace_root=str(tmp_path / "workspace"),
        artifact_root=str(tmp_path / "artifacts" / "run-1"),
    )


def test_selected_sensor_tool_validates_contract_and_returns_spike_preserving_envelope(
    tmp_path: Path,
) -> None:
    upload_root = tmp_path / "uploads"
    _write_sensor_bundle(upload_root)
    context = _context(tmp_path)

    result = inspect_selected_sensor_series_resource(
        context,
        upload_roots=(upload_root,),
        envelope_channel_id="ae-1",
        max_buckets=3,
    )

    assert result["ok"] is True
    assert result["schema"] == "ultra.sensor-inspection.v1"
    assert result["resource"]["file_id"] == "file_sensor"
    assert result["resource"]["sandbox_path"].startswith("/workspace/staged_uploads/")
    assert result["series"]["schema"] == "ultra.sensor-series.v1"
    assert result["series"]["specimen"] == {
        "specimen_id": "coupon-17",
        "material_id": "IN718",
    }
    assert result["series"]["clocks"][0]["sample_rate_hz"] == 2_000_000.0
    assert result["series"]["channels"][0]["invalid_count"] == 1
    assert result["series"]["channels"][0]["saturation_count"] == 1
    assert result["validation"]["values_validated"] is True
    assert result["validation"]["lineage_status"] == "unbound"
    assert result["validation"]["lineage_authority"] == "no_out_of_band_tree_digest"
    source_preflight = result["validation"]["source_staging"]
    assert source_preflight["entry_count"] > 0
    assert source_preflight["size_bytes"] > 0
    assert source_preflight["filesystem_scanned"] is True
    assert source_preflight["copied"] is True
    validation_budget = result["validation"]["validation_budget"]
    assert validation_budget["decoded_values"] == 18
    assert validation_budget["decoded_bytes"] == 60
    assert validation_budget["read_operations"] == 3
    buckets = result["generated_envelope"]["envelope"]["buckets"]
    assert max(bucket["maximum"] for bucket in buckets if bucket["maximum"] is not None) == 1000.0
    assert min(bucket["minimum"] for bucket in buckets if bucket["minimum"] is not None) == -800.0
    assert sum(bucket["invalid_count"] for bucket in buckets) == 1
    assert sum(bucket["saturation_count"] for bucket in buckets) == 1

    serialized = inspect_selected_sensor_series_text(
        context,
        upload_roots=(upload_root,),
        envelope_channel_id="ae-1",
        max_buckets=3,
    )
    assert str(tmp_path) not in serialized
    assert '"source_path"' not in serialized
    assert '"staged_path"' not in serialized


def test_selected_sensor_tool_upgrades_catalog_bound_tree_to_verified_lineage(
    tmp_path: Path,
) -> None:
    upload_root = tmp_path / "uploads"
    sensor_root = _write_sensor_bundle(upload_root)
    digest = _write_tree_manifest(sensor_root)
    context = _context(tmp_path, tree_manifest_sha256=digest)

    result = inspect_selected_sensor_series_resource(
        context,
        upload_roots=(upload_root,),
        validate_values=False,
    )

    assert result["ok"] is True
    assert result["resource"]["tree_identity"]["tree_manifest_sha256"] == digest
    assert result["series"]["lineage"]["status"] == "tree_verified"
    assert result["validation"]["lineage_status"] == "tree_verified"
    assert result["validation"]["lineage_authority"] == "out_of_band_tree_digest_verified"
    assert result["validation"]["linked_resource_lineage"]["status"] == "not_applicable"


def test_sensor_tool_verifies_multimodal_identity_only_with_both_selected_descriptors(
    tmp_path: Path,
) -> None:
    upload_root = tmp_path / "uploads"
    linked_digest = "b" * 64
    sensor_root = _write_sensor_bundle(
        upload_root,
        linked_resources=[
            {
                "role": "thermal_video",
                "resource_id": "file_ir_ngff_4",
                "sha256": linked_digest,
                "frame_clock_id": "ae-daq",
            }
        ],
    )
    sensor_digest = _write_tree_manifest(sensor_root)
    context = _context(
        tmp_path,
        selected_file_ids=("file_sensor", "file_ir_ngff_4"),
        tree_manifest_sha256=sensor_digest,
        descriptor_sha256s={"file_ir_ngff_4": linked_digest},
        descriptor_original_names={"file_ir_ngff_4": "thermal-video.ome.zarr"},
    )

    result = inspect_selected_sensor_series_resource(
        context,
        upload_roots=(upload_root,),
        file_id="file_sensor",
        validate_values=False,
    )

    assert result["ok"] is True
    assert result["validation"]["tree_lineage"]["status"] == "tree_verified"
    linked = result["series"]["linked_resources"][0]
    assert linked["verification_status"] == "catalog_identity_verified"
    assert linked["verification_authority"] == "control_resource_catalog"
    linked_lineage = result["validation"]["linked_resource_lineage"]
    assert linked_lineage["status"] == "verified_cross_resource_identity"
    assert linked_lineage["catalog_identity_verified_count"] == 1
    assert linked_lineage["all_linked_resources_run_selected"] is True
    assert linked_lineage["cross_resource_identity_verified"] is True


def test_sensor_tree_verification_does_not_verify_an_unselected_ngff_link(
    tmp_path: Path,
) -> None:
    upload_root = tmp_path / "uploads"
    sensor_root = _write_sensor_bundle(
        upload_root,
        linked_resources=[
            {
                "role": "thermal_video",
                "resource_id": "file_ir_ngff_4",
                "sha256": "b" * 64,
                "frame_clock_id": "ae-daq",
            }
        ],
    )
    sensor_digest = _write_tree_manifest(sensor_root)
    context = _context(tmp_path, tree_manifest_sha256=sensor_digest)

    result = inspect_selected_sensor_series_resource(
        context,
        upload_roots=(upload_root,),
        validate_values=False,
    )

    assert result["ok"] is True
    assert result["validation"]["metadata_validated"] is True
    assert result["validation"]["tree_lineage"]["status"] == "tree_verified"
    linked = result["series"]["linked_resources"][0]
    assert linked["verification_status"] == "declared_unverified"
    assert linked["verification_authority"] is None
    linked_lineage = result["validation"]["linked_resource_lineage"]
    assert linked_lineage["status"] == "declared_unverified"
    assert linked_lineage["sensor_tree_verified"] is True
    assert linked_lineage["all_linked_resources_run_selected"] is False
    assert linked_lineage["cross_resource_identity_verified"] is False


def test_sensor_tool_rejects_selected_ngff_link_with_catalog_digest_mismatch(
    tmp_path: Path,
) -> None:
    upload_root = tmp_path / "uploads"
    _write_sensor_bundle(
        upload_root,
        linked_resources=[
            {
                "role": "thermal_video",
                "resource_id": "file_ir_ngff_4",
                "sha256": "b" * 64,
                "frame_clock_id": "ae-daq",
            }
        ],
    )
    context = _context(
        tmp_path,
        selected_file_ids=("file_sensor", "file_ir_ngff_4"),
        descriptor_sha256s={"file_ir_ngff_4": "c" * 64},
    )

    result = inspect_selected_sensor_series_resource(
        context,
        upload_roots=(upload_root,),
        file_id="file_sensor",
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "linked_resource_catalog_digest_mismatch"


def test_selected_sensor_tool_rejects_same_size_tree_tamper_against_catalog_digest(
    tmp_path: Path,
) -> None:
    upload_root = tmp_path / "uploads"
    sensor_root = _write_sensor_bundle(upload_root)
    digest = _write_tree_manifest(sensor_root)
    chunk = next(
        path
        for path in (sensor_root / "signals" / "ae-1").iterdir()
        if path.is_file() and not path.name.startswith(".")
    )
    payload = bytearray(chunk.read_bytes())
    payload[0] ^= 0x01
    chunk.write_bytes(payload)
    context = _context(tmp_path, tree_manifest_sha256=digest)

    result = inspect_selected_sensor_series_resource(
        context,
        upload_roots=(upload_root,),
        validate_values=False,
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "tree_entry_hash_mismatch"


@pytest.mark.parametrize("requested", ["file_secret", "../../file_secret", "/tmp/data.zarr"])
def test_sensor_tool_rejects_unselected_ids_and_arbitrary_path_shapes_before_staging(
    tmp_path: Path,
    requested: str,
) -> None:
    upload_root = tmp_path / "uploads"
    _write_sensor_bundle(upload_root, file_id="file_secret")
    context = _context(tmp_path, selected_file_ids=("file_allowed",))

    result = inspect_selected_sensor_series_resource(
        context,
        upload_roots=(upload_root,),
        file_id=requested,
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "selected_sensor_resource_required"
    assert not (tmp_path / "workspace").exists()
    assert str(tmp_path) not in json.dumps(result)


def test_sensor_tool_requires_an_explicit_id_for_multiple_selected_resources(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, selected_file_ids=("file_a", "file_b"))
    result = inspect_selected_sensor_series_resource(context, upload_roots=(tmp_path / "uploads",))
    assert result["error"]["code"] == "selected_sensor_resource_required"
    assert "more than one" in result["error"]["message"]


def test_sensor_tool_rejects_source_and_workspace_symlinks(tmp_path: Path) -> None:
    upload_root = tmp_path / "uploads"
    sensor_root = _write_sensor_bundle(upload_root)
    outside = tmp_path / "outside.txt"
    outside.write_text("must never be staged", encoding="utf-8")
    (sensor_root / "external-link").symlink_to(outside)
    context = _context(tmp_path)

    source_result = inspect_selected_sensor_series_resource(context, upload_roots=(upload_root,))
    assert source_result["error"]["code"] == "unsafe_sensor_source"
    assert not (tmp_path / "workspace").exists()

    (sensor_root / "external-link").unlink()
    outside_dir = tmp_path / "outside-stage"
    outside_dir.mkdir()
    token_root = tmp_path / "workspace" / "staged_uploads"
    token_root.mkdir(parents=True)
    (token_root / "file_sensor").symlink_to(outside_dir, target_is_directory=True)
    stage_result = inspect_selected_sensor_series_resource(context, upload_roots=(upload_root,))
    assert stage_result["error"]["code"] == "unsafe_sensor_staging_path"
    assert list(outside_dir.iterdir()) == []


def test_sensor_tool_rejects_catalog_oversize_before_lookup_scan_or_copy(tmp_path: Path) -> None:
    context = _context(tmp_path)
    descriptor = dict(context.resource_descriptors[0])
    descriptor["size_bytes"] = MAX_SOURCE_TREE_BYTES + 1
    context = AgentRunContext(**{**context.to_payload(), "resource_descriptors": [descriptor]})

    result = inspect_selected_sensor_series_resource(
        context,
        upload_roots=(tmp_path / "uploads-that-need-not-exist",),
    )

    assert result["error"]["code"] == "sensor_source_budget_exceeded"
    source = result["preflight"]["source_staging"]
    assert source["filesystem_scanned"] is False
    assert source["copied"] is False
    assert not (tmp_path / "workspace").exists()


def test_sensor_tool_rejects_actual_sparse_oversize_before_copy(tmp_path: Path) -> None:
    upload_root = tmp_path / "uploads"
    sensor_root = _write_sensor_bundle(upload_root)
    with (sensor_root / "oversized-chunk").open("wb") as stream:
        stream.truncate(MAX_SOURCE_TREE_BYTES + 1)
    context = _context(tmp_path)

    result = inspect_selected_sensor_series_resource(context, upload_roots=(upload_root,))

    assert result["error"]["code"] == "sensor_source_budget_exceeded"
    assert result["preflight"]["source_staging"]["filesystem_scanned"] is True
    assert result["preflight"]["source_staging"]["copied"] is False
    assert not (tmp_path / "workspace").exists()


def test_sensor_tool_rejects_entry_and_preflight_wall_amplification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upload_root = tmp_path / "uploads"
    _write_sensor_bundle(upload_root)
    context = _context(tmp_path)

    monkeypatch.setattr("ultra_deepagents.sensors.tools.MAX_SOURCE_TREE_ENTRIES", 1)
    entries = inspect_selected_sensor_series_resource(context, upload_roots=(upload_root,))
    assert entries["error"]["code"] == "sensor_source_budget_exceeded"
    assert entries["preflight"]["source_staging"]["copied"] is False
    assert not (tmp_path / "workspace").exists()

    monkeypatch.setattr("ultra_deepagents.sensors.tools.MAX_SOURCE_TREE_ENTRIES", 200_000)
    ticks = iter((0.0, 6.0))
    monkeypatch.setattr(
        "ultra_deepagents.sensors.tools.time.monotonic",
        lambda: next(ticks, 6.0),
    )
    wall = inspect_selected_sensor_series_resource(context, upload_roots=(upload_root,))
    assert wall["error"]["code"] == "sensor_source_budget_exceeded"
    assert "wall-time" in wall["error"]["message"]
    assert wall["preflight"]["source_staging"]["copied"] is False
    assert not (tmp_path / "workspace").exists()


def test_sensor_tool_fails_closed_before_an_over_budget_value_scan(tmp_path: Path) -> None:
    upload_root = tmp_path / "uploads"
    count = MAX_VALUE_VALIDATION_SAMPLES + 1
    _write_sensor_bundle(upload_root, sparse_sample_count=count)
    context = _context(tmp_path)

    blocked = inspect_selected_sensor_series_resource(context, upload_roots=(upload_root,))
    assert blocked["ok"] is False
    assert blocked["error"]["code"] == "value_validation_budget_exceeded"
    assert blocked["preflight"]["aggregate_channel_samples"] == count
    assert blocked["preflight"]["values_validated"] is False

    metadata_only = inspect_selected_sensor_series_resource(
        context,
        upload_roots=(upload_root,),
        validate_values=False,
    )
    assert metadata_only["ok"] is True
    assert metadata_only["validation"]["values_validated"] is False
    assert metadata_only["series"]["channels"][0]["invalid_count"] is None


def test_sensor_tool_rejects_non_sensor_zarr_without_leaking_host_paths(tmp_path: Path) -> None:
    upload_root = tmp_path / "uploads"
    root = upload_root / "bundles" / "file_sensor" / "ordinary.ome.zarr"
    group = zarr.open_group(str(root), mode="w", zarr_format=2)
    group.attrs["multiscales"] = []
    context = _context(tmp_path)

    result = inspect_selected_sensor_series_resource(context, upload_roots=(upload_root,))
    assert result["ok"] is False
    assert result["error"]["code"] == "sensor_metadata_missing"
    assert str(tmp_path) not in json.dumps(result)


def test_sensor_tool_schema_exposes_no_filesystem_path_argument() -> None:
    sensor_tool = build_sensor_tools()[0]
    properties = set(sensor_tool.args)
    assert properties == {
        "file_id",
        "validate_values",
        "envelope_channel_id",
        "max_buckets",
    }
    assert not properties.intersection({"path", "root", "staged_path", "source_path"})


def test_sensor_tool_registration_gate_requires_selection_and_sensor_shape(tmp_path: Path) -> None:
    natural = _context(tmp_path)
    assert should_register_sensor_tools(natural) is True

    no_selection = AgentRunContext(
        **{
            **natural.to_payload(),
            "goal": "Analyze this acoustic-emission waveform.",
            "selected_file_ids": [],
            "resource_descriptors": [],
        }
    )
    assert should_register_sensor_tools(no_selection) is False

    terse_selected_zarr = AgentRunContext(
        **{
            **natural.to_payload(),
            "goal": "Inspect the selected file.",
            "resource_descriptors": [
                {
                    **natural.resource_descriptors[0],
                    "sensor_format": _sensor_format_marker(
                        str(natural.resource_descriptors[0]["sha256"])
                    ),
                }
            ],
        }
    )
    assert should_register_sensor_tools(terse_selected_zarr) is True

    biology_ome_ngff = AgentRunContext(
        **{
            **natural.to_payload(),
            "goal": "Inspect the selected Zarr.",
            "resource_descriptors": [
                {
                    **natural.resource_descriptors[0],
                    "original_name": "organoid.ome.zarr",
                }
            ],
        }
    )
    assert should_register_sensor_tools(biology_ome_ngff) is False

    unrelated = AgentRunContext(
        **{
            **natural.to_payload(),
            "goal": "Compute CALPHAD phase equilibrium.",
            "resource_descriptors": [
                {
                    **natural.resource_descriptors[0],
                    "original_name": "alloy.tdb",
                }
            ],
        }
    )
    assert should_register_sensor_tools(unrelated) is False


def test_sensor_tool_100k_value_validation_and_envelope_stays_bounded(tmp_path: Path) -> None:
    upload_root = tmp_path / "uploads"
    values = np.zeros(100_000, dtype="float64")
    values[49_999] = 19_000.0
    values[50_000] = -17_000.0
    _write_sensor_bundle(upload_root, values=values)
    context = _context(tmp_path)

    started = time.perf_counter()
    result = inspect_selected_sensor_series_resource(
        context,
        upload_roots=(upload_root,),
        envelope_channel_id="ae-1",
        max_buckets=512,
    )
    elapsed = time.perf_counter() - started

    assert result["ok"] is True
    assert len(result["generated_envelope"]["envelope"]["buckets"]) <= 512
    assert len(json.dumps(result)) < 250_000
    # This is a regression tripwire, not a production SLO; it leaves ample room for CI noise.
    assert elapsed < 10.0
