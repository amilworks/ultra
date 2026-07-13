"""Core-only production-worker smoke for selected sensor Zarr inspection.

This file deliberately uses no pytest-only or imaging-extra imports. Dockerfile.worker
executes it after installing the core project, and pytest also collects the same proof.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import zarr
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.sensors import canonical_sha256
from ultra_deepagents.sensors.tools import inspect_selected_sensor_series_resource


def _unit() -> dict[str, str]:
    return {
        "label": "V",
        "ucum_code": "V",
        "qudt_uri": "http://qudt.org/vocab/unit/V",
    }


def _metadata(sample_count: int) -> dict[str, object]:
    volt = _unit()
    calibration: dict[str, object] = {
        "kind": "identity",
        "applied": True,
        "calibration_id": "worker-smoke-calibration",
        "revision": "2026-07",
        "input_unit": volt,
        "output_unit": volt,
        "scale": 1.0,
        "offset": 0.0,
    }
    calibration["parameters_sha256"] = canonical_sha256(calibration)
    return {
        "schema": "ultra.sensor-series.v1",
        "series_id": "worker-core-smoke",
        "modality": "acoustic_emission",
        "specimen": {"specimen_id": "coupon-smoke", "material_id": "IN718"},
        "clocks": [
            {
                "clock_id": "daq-clock",
                "kind": "regular",
                "sample_count": sample_count,
                "reference": "relative",
                "time_unit": {
                    "label": "s",
                    "ucum_code": "s",
                    "qudt_uri": "http://qudt.org/vocab/unit/SEC",
                },
                "start_time_seconds": 0.0,
                "sample_rate_hz": 1_000_000.0,
                "accuracy": {
                    "status": "quantified",
                    "standard_uncertainty_seconds": 1.0e-8,
                    "method": "worker_smoke_control",
                },
            }
        ],
        "channels": [
            {
                "channel_id": "ae-1",
                "name": "AE voltage",
                "array": "signals/ae-1",
                "clock_id": "daq-clock",
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


def _write_bundle(upload_root: Path) -> None:
    root = upload_root / "bundles" / "file_sensor_smoke" / "worker-smoke.zarr"
    group = zarr.open_group(str(root), mode="w", zarr_format=2)
    values = np.asarray([0.0, 0.25, 9.0, -7.0, 0.5, 0.0], dtype="float64")
    group.require_group("signals").create_array("ae-1", data=values, chunks=(3,))
    quality = group.require_group("quality")
    quality.create_array("valid", data=np.ones(values.size, dtype="bool"), chunks=(3,))
    quality.create_array(
        "saturated",
        data=np.asarray([False, False, True, False, False, False], dtype="bool"),
        chunks=(3,),
    )
    group.attrs["ultra"] = {"sensor_series": _metadata(int(values.size))}


def _context(root: Path) -> AgentRunContext:
    return AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="org-smoke",
        user_id="user-smoke",
        project_id="project-smoke",
        thread_id="thread-smoke",
        run_id="run-sensor-worker-smoke",
        goal="Inspect the selected acoustic-emission sensor series.",
        selected_file_ids=("file_sensor_smoke",),
        resource_descriptors=(
            {
                "type": "selected_resource",
                "binding_schema": "ultra.selected_resource.v1",
                "authority": "control_resource_catalog",
                "resource_id": "file_sensor_smoke",
                "file_id": "file_sensor_smoke",
                "original_name": "worker-smoke.zarr",
                "content_type": "application/octet-stream",
                "resource_kind": "dataset",
                "sha256": "a" * 64,
                "size_bytes": 4096,
            },
        ),
        workspace_root=str(root / "workspace"),
        artifact_root=str(root / "artifacts" / "run-sensor-worker-smoke"),
    )


def _assert_worker_core_smoke(root: Path) -> None:
    upload_root = root / "uploads"
    _write_bundle(upload_root)
    result = inspect_selected_sensor_series_resource(
        _context(root),
        upload_roots=(upload_root,),
        envelope_channel_id="ae-1",
        max_buckets=3,
    )
    assert result["ok"] is True, result
    assert result["series"]["specimen"] == {
        "specimen_id": "coupon-smoke",
        "material_id": "IN718",
    }
    assert result["validation"]["values_validated"] is True
    assert result["series"]["channels"][0]["saturation_count"] == 1
    envelope = result["generated_envelope"]["envelope"]["buckets"]
    assert max(bucket["maximum"] for bucket in envelope) == 9.0
    assert min(bucket["minimum"] for bucket in envelope) == -7.0


def test_worker_core_can_open_and_inspect_tiny_sensor_zarr(tmp_path: Path) -> None:
    _assert_worker_core_smoke(tmp_path)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="ultra-sensor-worker-smoke-") as temporary:
        _assert_worker_core_smoke(Path(temporary))
    print("worker core sensor Zarr inspection ok")
