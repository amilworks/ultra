from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path
from types import ModuleType

from ultra_deepagents.sensors import canonical_json_bytes, canonical_sha256, open_sensor_series

_BUILDER_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "materials_natural_prompts" / "build_fixtures.py"
)
_MANIFEST_PATH = ".ultra/tree-manifest.json"


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("zarr_tree_identity_fixture", _BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _server_manifest_for_tree(root: Path) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for candidate in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = candidate.relative_to(root).as_posix()
        if relative == _MANIFEST_PATH:
            continue
        payload = candidate.read_bytes()
        entries.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    return {"entries": entries, "schema": "ultra.tree-manifest.v1"}


def test_server_tree_identity_is_exact_out_of_band_sensor_manifest_digest(tmp_path: Path) -> None:
    builder = _load_builder()
    fixture_root = tmp_path / "fixture"
    gold = builder.build(fixture_root)
    sensor_gold = gold["sensor"]
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(fixture_root / sensor_gold["archive"]) as archive:
        archive.extractall(extracted)
    sensor_root = extracted / sensor_gold["directory_name"]

    internal_manifest_path = sensor_root / _MANIFEST_PATH
    # Exercise the normal browser-upload shape: clients are not required to
    # submit the reserved manifest. The control-plane contract derives it from
    # every other regular member and installs these exact canonical bytes.
    internal_manifest_path.unlink()
    server_manifest = _server_manifest_for_tree(sensor_root)
    internal_bytes = canonical_json_bytes(server_manifest)
    internal_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    internal_manifest_path.write_bytes(internal_bytes)
    internal_manifest = json.loads(internal_manifest_path.read_bytes())

    # This is the cross-boundary invariant: Go finalization and the Python
    # reader use the same canonical object, with the manifest excluded from its
    # own entries. Including the manifest file here would make the digest
    # unusable as expected_tree_manifest_sha256.
    assert server_manifest == internal_manifest
    assert internal_bytes == canonical_json_bytes(internal_manifest)
    server_digest = canonical_sha256(server_manifest)
    assert server_digest == sensor_gold["tree_manifest_sha256"]

    series = open_sensor_series(
        sensor_root,
        expected_tree_manifest_sha256=server_digest,
    )
    assert series.lineage.status == "tree_verified"
    assert series.lineage.expected_tree_manifest_sha256 == server_digest
