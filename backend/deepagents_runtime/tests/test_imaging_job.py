"""Unit tests for the image.derive_pyramid job runner (no engine binary needed)."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from ultra_deepagents import safe_storage
from ultra_deepagents.imaging import derivative_manifest
from ultra_deepagents.imaging.convert import ConvertResult
from ultra_deepagents.imaging.derivative_manifest import (
    DeterministicDerivativeError,
    StaleDerivativeJobError,
    TransientDerivativeError,
)
from ultra_deepagents.imaging.job import DerivePyramidJob, run_derive_pyramid_job
from ultra_deepagents.imaging.transcode import TranscodeResult
from ultra_deepagents.imaging.worker import extract_derive_pyramid_payload


def _viewer_info(*, dtype="uint16", names=None, spacing=None, scene="scene-0", tiled=True):
    names = names or ["DAPI", "FITC"]
    spacing = spacing or {"x": 0.25, "y": 0.25, "z": 1.0}
    tile_scheme = {"levels": [{"level": 0, "width": 16, "height": 8}]}
    return {
        "dims_order": "TCZYX",
        "axis_sizes": {"T": 2, "C": 2, "Z": 3, "Y": 8, "X": 16},
        "dtype": dtype,
        "channel_names": names,
        "physical_spacing": spacing,
        "metadata": {
            "selected_scene_id": scene,
            "spacing_units": {"x": "um", "y": "um", "z": "um"},
        },
        "viewer": {
            "tile_scheme": tile_scheme if tiled else None,
            "atlas_scheme": {"columns": 2, "rows": 2, "cell_w": 16, "cell_h": 8},
        },
        "tile_scheme": tile_scheme if tiled else None,
    }


def _strict_job(src, dst):
    payload = src.read_bytes()
    return {
        "resource_id": "file-1",
        "src_path": str(src),
        "dst_path": str(dst),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_size_bytes": len(payload),
        "fmt": "bigtiff",
    }


def _writing_convert(calls):
    def convert(src, dst, *, spec):
        calls.append((src, dst, spec))
        with open(dst, "wb") as artifact:
            artifact.write(b"strict pyramid bytes")
        return ConvertResult(src, dst, 0, "", "")

    return convert


def _assert_valid_manifest_artifact_pair(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    artifact = manifest_path.parent / manifest["artifact"]["basename"]
    payload = artifact.read_bytes()
    assert artifact.stat().st_size == manifest["artifact"]["size_bytes"]
    assert hashlib.sha256(payload).hexdigest() == manifest["artifact"]["sha256"]
    return manifest, artifact


def test_managed_directory_creation_fsyncs_each_parent(monkeypatch, tmp_path):
    events = []
    original_mkdir = safe_storage.os.mkdir
    original_fsync = safe_storage.os.fsync

    def tracked_mkdir(component, *, mode, dir_fd):
        original_mkdir(component, mode=mode, dir_fd=dir_fd)
        events.append(("mkdir", component))

    def tracked_fsync(descriptor):
        events.append(("fsync", descriptor))
        return original_fsync(descriptor)

    monkeypatch.setattr(safe_storage.os, "mkdir", tracked_mkdir)
    monkeypatch.setattr(safe_storage.os, "fsync", tracked_fsync)
    with safe_storage.open_directory_chain_no_follow(
        tmp_path,
        (".staging", "file-1", "pyramid"),
        create=True,
    ):
        pass

    assert [kind for kind, _value in events] == [
        "mkdir",
        "fsync",
        "mkdir",
        "fsync",
        "mkdir",
        "fsync",
    ]


def test_managed_directory_creation_retries_parent_fsync_after_mkdir_succeeds(
    monkeypatch, tmp_path
):
    original_fsync = safe_storage.os.fsync
    attempts = 0

    def fail_first_fsync(descriptor):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(errno.EIO, "injected parent sync failure")
        return original_fsync(descriptor)

    monkeypatch.setattr(safe_storage.os, "fsync", fail_first_fsync)

    with pytest.raises(OSError, match="injected parent sync failure"):
        with safe_storage.open_directory_chain_no_follow(
            tmp_path,
            (".staging", "file-1", "pyramid"),
            create=True,
        ):
            pass

    with safe_storage.open_directory_chain_no_follow(
        tmp_path,
        (".staging", "file-1", "pyramid"),
        create=True,
    ):
        pass

    assert attempts == 4
    assert (tmp_path / ".staging" / "file-1" / "pyramid").is_dir()


def test_strict_derivative_publishes_immutable_artifact_manifest_last_and_replays(tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-generation-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    calls = []

    first = run_derive_pyramid_job(
        _strict_job(src, dst),
        convert_fn=_writing_convert(calls),
        meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
        viewer_info_fn=lambda _path: _viewer_info(),
    )

    manifest_path = dst.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    artifact = manifest_path.parent / manifest["artifact"]["basename"]
    assert not dst.exists()
    assert artifact.is_file() and artifact.read_bytes() == b"strict pyramid bytes"
    assert manifest["source"] == {
        "sha256": _strict_job(src, dst)["source_sha256"],
        "size_bytes": len(b"source-generation-one"),
    }
    assert manifest["conversion_spec"] == {
        "requested": {
            "compression": "lzw",
            "fmt": "bigtiff",
            "layout": "topdirs",
            "tile_size": 512,
        },
        "effective": {
            "compression": "lzw",
            "fmt": "bigtiff",
            "layout": "topdirs",
            "tile_size": 512,
        },
        "producer_revision": "ultra-deepagents.image-pyramid-publisher.v1",
        "converter_revision": "libbioimage.imgcnv-pyramid.v1",
    }
    assert manifest["producer"]["reader"] == "libbioimage"
    assert manifest["semantics"]["axis_sizes"] == {"T": 2, "C": 2, "Z": 3, "Y": 8, "X": 16}
    assert manifest["capabilities"] == {
        "atlas": True,
        "atlas_t": True,
        "lut": True,
        "ordered_channels": True,
        "slice": True,
        "thumbnail": True,
        "tile": True,
        "tile_t": False,
        "tile_z": False,
    }
    assert first["manifest_path"] == str(manifest_path)

    replay = run_derive_pyramid_job(
        _strict_job(src, dst),
        convert_fn=_writing_convert(calls),
        meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
        viewer_info_fn=lambda _path: _viewer_info(),
    )
    assert replay["status"] == "replayed"
    assert replay["derived_path"] == str(artifact)
    assert len(calls) == 1


@pytest.mark.parametrize("tombstone_kind", ["permanent", "deleted"])
def test_strict_derivative_cannot_publish_after_resource_tombstone(tmp_path, tombstone_kind):
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    src = upload_root / "file-1__source.ome.tif"
    src.write_bytes(b"source-generation-one")
    dst = upload_root / "derived" / "file-1__pyramid.tif"
    tombstone = upload_root / ".tombstones" / tombstone_kind / "file-1"
    tombstone.parent.mkdir(parents=True)
    tombstone.write_text("deleted\n", encoding="utf-8")
    calls = []

    with pytest.raises(StaleDerivativeJobError) as excinfo:
        run_derive_pyramid_job(
            _strict_job(src, dst),
            convert_fn=_writing_convert(calls),
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=lambda _path: _viewer_info(),
        )

    assert excinfo.value.code == "resource_deleted"
    assert calls == []
    assert not dst.with_suffix(".manifest.json").exists()


def test_strict_derivative_conversion_does_not_block_delete_and_fails_closed(tmp_path):
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    src = upload_root / "file-1__source.ome.tif"
    src.write_bytes(b"source-generation-one")
    dst = upload_root / "derived" / "file-1__pyramid.tif"
    conversion_started = threading.Event()
    release_conversion = threading.Event()

    def blocking_convert(_src, staged_path, *, spec):
        del spec
        conversion_started.set()
        assert release_conversion.wait(timeout=5), "test did not release conversion"
        Path(staged_path).write_bytes(b"strict pyramid bytes")
        return ConvertResult(str(src), str(staged_path), 0, "", "")

    outcome = []

    def publish():
        try:
            run_derive_pyramid_job(
                _strict_job(src, dst),
                convert_fn=blocking_convert,
                meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
                viewer_info_fn=lambda _path: _viewer_info(),
            )
        except BaseException as exc:  # capture the worker-visible typed result
            outcome.append(exc)

    publisher = threading.Thread(target=publish)
    publisher.start()
    assert conversion_started.wait(timeout=5), "conversion did not start"

    # This is the deletion critical section: it must acquire the same lock while
    # conversion is still blocked, publish the tombstone, and remove the source.
    tombstone = upload_root / ".tombstones" / "permanent" / "file-1"
    with derivative_manifest._publication_lock(dst, src):
        tombstone.parent.mkdir(parents=True)
        tombstone.write_text("deleted\n", encoding="utf-8")
        src.unlink()
    assert not release_conversion.is_set(), "delete waited for conversion"

    release_conversion.set()
    publisher.join(timeout=5)
    assert not publisher.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], StaleDerivativeJobError)
    assert outcome[0].code == "resource_deleted"
    assert not dst.with_suffix(".manifest.json").exists()
    assert not any(".tmp-" in entry.name for entry in dst.parent.iterdir())


def test_catalog_hashing_does_not_hold_lifecycle_lock(monkeypatch, tmp_path):
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    src = upload_root / "file-1__source.ome.tif"
    src.write_bytes(b"source-generation-one")
    dst = upload_root / "derived" / "file-1__pyramid.tif"
    hashing_started = threading.Event()
    release_hashing = threading.Event()
    original_verify = derivative_manifest._verify_catalog_source

    def blocking_verify(*args, **kwargs):
        result = original_verify(*args, **kwargs)
        hashing_started.set()
        assert release_hashing.wait(timeout=5)
        return result

    monkeypatch.setattr(derivative_manifest, "_verify_catalog_source", blocking_verify)
    outcome = []

    def publish():
        try:
            run_derive_pyramid_job(
                _strict_job(src, dst),
                convert_fn=_writing_convert([]),
                meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
                viewer_info_fn=lambda _path: _viewer_info(),
            )
        except BaseException as exc:
            outcome.append(exc)

    publisher = threading.Thread(target=publish)
    publisher.start()
    assert hashing_started.wait(timeout=5)
    tombstone = upload_root / ".tombstones" / "permanent" / "file-1"
    with derivative_manifest._publication_lock(dst, src):
        tombstone.parent.mkdir(parents=True)
        tombstone.write_text("deleted\n", encoding="utf-8")
        src.unlink()
    release_hashing.set()
    publisher.join(timeout=5)

    assert not publisher.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], StaleDerivativeJobError)
    assert outcome[0].code == "resource_deleted"
    assert not dst.with_suffix(".manifest.json").exists()
    assert not dst.parent.exists() or not any(dst.parent.iterdir())


def test_artifact_hashing_does_not_hold_lifecycle_lock(monkeypatch, tmp_path):
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    src = upload_root / "file-1__source.ome.tif"
    src.write_bytes(b"source-generation-one")
    dst = upload_root / "derived" / "file-1__pyramid.tif"
    hashing_started = threading.Event()
    release_hashing = threading.Event()
    original_digest = derivative_manifest._open_digest_and_sync_regular_file

    def blocking_digest(path, **kwargs):
        result = original_digest(path, **kwargs)
        if kwargs.get("label") == "temporary derivative artifact":
            hashing_started.set()
            assert release_hashing.wait(timeout=5)
        return result

    monkeypatch.setattr(derivative_manifest, "_open_digest_and_sync_regular_file", blocking_digest)
    outcome = []

    def publish():
        try:
            run_derive_pyramid_job(
                _strict_job(src, dst),
                convert_fn=_writing_convert([]),
                meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
                viewer_info_fn=lambda _path: _viewer_info(),
            )
        except BaseException as exc:
            outcome.append(exc)

    publisher = threading.Thread(target=publish)
    publisher.start()
    assert hashing_started.wait(timeout=5)
    tombstone = upload_root / ".tombstones" / "permanent" / "file-1"
    with derivative_manifest._publication_lock(dst, src):
        tombstone.parent.mkdir(parents=True)
        tombstone.write_text("deleted\n", encoding="utf-8")
        src.unlink()
    release_hashing.set()
    publisher.join(timeout=5)

    assert not publisher.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], StaleDerivativeJobError)
    assert outcome[0].code == "resource_deleted"
    assert not dst.with_suffix(".manifest.json").exists()
    assert not any(".tmp-" in entry.name for entry in dst.parent.iterdir())


def test_derivative_tombstone_identity_is_injective_for_dotted_resource_ids(tmp_path):
    destination = tmp_path / "uploads" / "derived" / "file-1__pyramid.tif"
    permanent, reversible = derivative_manifest._publication_tombstone_paths(destination)

    assert permanent == tmp_path / "uploads" / ".tombstones" / "permanent" / "file-1"
    assert reversible == tmp_path / "uploads" / ".tombstones" / "deleted" / "file-1"
    assert permanent != tmp_path / "uploads" / ".tombstones" / "permanent" / "file-1.deleted"


@pytest.mark.parametrize("ancestor", [".locks", ".tombstones"])
def test_strict_derivative_rejects_symlinked_lifecycle_ancestor(tmp_path, ancestor):
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (upload_root / ancestor).symlink_to(external, target_is_directory=True)
    src = upload_root / "file-1__source.ome.tif"
    src.write_bytes(b"source-generation-one")
    dst = upload_root / "derived" / "file-1__pyramid.tif"
    calls = []

    with pytest.raises(TransientDerivativeError):
        run_derive_pyramid_job(
            _strict_job(src, dst),
            convert_fn=_writing_convert(calls),
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=lambda _path: _viewer_info(),
        )

    assert calls == []
    assert list(external.iterdir()) == []


def test_concurrent_strict_publication_singleflights_and_leaves_no_temporary_files(tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-generation-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    calls = []
    start = threading.Barrier(2)

    def publish():
        start.wait()
        return run_derive_pyramid_job(
            _strict_job(src, dst),
            convert_fn=_writing_convert(calls),
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=lambda _path: _viewer_info(),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: publish(), range(2)))

    assert sorted(result["status"] for result in results) == ["replayed", "succeeded"]
    assert len(calls) == 1
    entries = list(dst.parent.iterdir())
    assert len([entry for entry in entries if ".sha256-" in entry.name]) == 1
    assert dst.with_suffix(".manifest.json") in entries
    assert not any(".tmp-" in entry.name or ".transcode" in entry.name for entry in entries)


def test_new_work_owner_reclaims_private_workspace_without_touching_other_resource(tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-generation-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    dst.parent.mkdir()
    workspace = tmp_path / ".staging" / "file-1" / "pyramid"
    workspace.mkdir(parents=True)
    abandoned = workspace / "artifact.tif"
    abandoned_transcode = workspace / "artifact.tif.transcode.ome.tif"
    abandoned_manifest = workspace / "manifest.tmp"
    unrelated = tmp_path / ".staging" / "file-10" / "pyramid" / "artifact.tif"
    unrelated.parent.mkdir(parents=True)
    for path in (abandoned, abandoned_transcode, abandoned_manifest):
        path.write_bytes(b"partial")
    unrelated.write_bytes(b"keep")

    result = run_derive_pyramid_job(
        _strict_job(src, dst),
        convert_fn=_writing_convert([]),
        meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
        viewer_info_fn=lambda _path: _viewer_info(),
    )

    assert result["status"] == "succeeded"
    assert not abandoned.exists()
    assert not abandoned_transcode.exists()
    assert not abandoned_manifest.exists()
    assert unrelated.read_bytes() == b"keep"


def test_cross_process_derivation_lock_runs_converter_once(tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-generation-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    counter = tmp_path / "converter-calls"
    start = tmp_path / "start"
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
import hashlib, os, pathlib, sys, time
sys.path.insert(0, sys.argv[1])
from ultra_deepagents.imaging.derivative_manifest import run_strict_publication
src, dst, counter, start = map(pathlib.Path, sys.argv[2:])
while not start.exists():
    time.sleep(0.005)
payload = src.read_bytes()
def viewer(_path):
    return {
        "dims_order": "TCZYX",
        "axis_sizes": {"T": 2, "C": 2, "Z": 3, "Y": 8, "X": 16},
        "dtype": "uint16",
        "channel_names": ["DAPI", "FITC"],
        "physical_spacing": {"x": 0.25, "y": 0.25, "z": 1.0},
        "metadata": {
            "selected_scene_id": "scene-0",
            "spacing_units": {"x": "um", "y": "um", "z": "um"},
        },
        "viewer": {
            "tile_scheme": {"levels": [{"level": 0, "width": 16, "height": 8}]},
            "atlas_scheme": {"columns": 2, "rows": 2, "cell_w": 16, "cell_h": 8},
        },
        "tile_scheme": {"levels": [{"level": 0, "width": 16, "height": 8}]},
    }
def produce(path):
    descriptor = os.open(counter, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    os.write(descriptor, b"call\n")
    os.close(descriptor)
    time.sleep(0.1)
    pathlib.Path(path).write_bytes(b"strict pyramid bytes")
    return {}
run_strict_publication(
    resource_id="file-1",
    src_path=str(src),
    dst_path=str(dst),
    source_sha256=hashlib.sha256(payload).hexdigest(),
    source_size_bytes=len(payload),
    viewer_info_fn=viewer,
    source_viewer_info_fn=None,
    source_reader="libbioimage",
    conversion_spec={
        "tile_size": 512,
        "compression": "lzw",
        "layout": "topdirs",
        "fmt": "bigtiff",
    },
    produce_fn=produce,
)
"""
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(source_root),
                str(src),
                str(dst),
                str(counter),
                str(start),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    start.write_text("go", encoding="utf-8")
    failures = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=10)
        if process.returncode != 0:
            failures.append((process.returncode, stdout, stderr))

    assert failures == []
    assert counter.read_text(encoding="utf-8").splitlines() == ["call"]


def test_publication_lock_is_cross_process_and_rechecks_source_after_wait(tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-generation-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    lock_path = tmp_path / ".locks" / ".file-1__pyramid.lock"
    ready = tmp_path / "lock-ready"
    release = tmp_path / "lock-release"
    script = """
import fcntl, os, pathlib, sys, time
lock_path, ready, release = map(pathlib.Path, sys.argv[1:])
lock_path.parent.mkdir(parents=True, exist_ok=True)
descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
fcntl.flock(descriptor, fcntl.LOCK_EX)
ready.write_text("ready")
while not release.exists():
    time.sleep(0.01)
fcntl.flock(descriptor, fcntl.LOCK_UN)
os.close(descriptor)
"""
    holder = subprocess.Popen(
        [sys.executable, "-c", script, str(lock_path), str(ready), str(release)]
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "child process did not acquire the lifecycle lock"
        entered = threading.Event()
        outcome = []

        def wait_for_lock():
            try:
                with derivative_manifest._publication_lock(dst, src):
                    entered.set()
            except BaseException as exc:  # capture the worker-visible typed result
                outcome.append(exc)

        waiter = threading.Thread(target=wait_for_lock)
        waiter.start()
        time.sleep(0.1)
        assert not entered.is_set(), "publisher ignored the cross-process lock"
        src.unlink()
        release.write_text("release")
        waiter.join(timeout=5)
        assert not waiter.is_alive()
        assert not entered.is_set()
        assert len(outcome) == 1
        assert isinstance(outcome[0], StaleDerivativeJobError)
        assert outcome[0].code == "source_generation_changed"
    finally:
        release.write_text("release")
        holder.wait(timeout=5)


def test_strict_publication_rejects_artifact_path_swap_after_semantic_probe(tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-generation-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"

    def swapping_viewer_info(path):
        artifact = os.fspath(path)
        if Path(artifact).name == "artifact.tif":
            replacement = tmp_path / "replacement.tif"
            replacement.write_bytes(b"replaced pyramid data")
            os.replace(replacement, artifact)
        return _viewer_info()

    with pytest.raises(DeterministicDerivativeError) as excinfo:
        run_derive_pyramid_job(
            _strict_job(src, dst),
            convert_fn=_writing_convert([]),
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=swapping_viewer_info,
        )

    assert excinfo.value.code == "conversion_artifact_changed"
    assert not dst.with_suffix(".manifest.json").exists()
    assert not any(".tmp-" in entry.name for entry in dst.parent.iterdir())


def test_strict_publication_fences_source_swap_during_source_semantic_probe(tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-one")
    initial_mtime = src.stat().st_mtime_ns
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    calls = []

    def swapping_source_viewer(path):
        if os.fspath(path) == os.fspath(src):
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"source-two")
            os.utime(replacement, ns=(initial_mtime, initial_mtime))
            os.replace(replacement, src)
        return _viewer_info()

    with pytest.raises(StaleDerivativeJobError) as excinfo:
        run_derive_pyramid_job(
            _strict_job(src, dst),
            convert_fn=_writing_convert(calls),
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=swapping_source_viewer,
        )

    assert excinfo.value.code == "source_generation_changed"
    assert calls == []


def test_strict_publication_preserves_old_manifest_when_source_swaps_during_derived_probe(
    tmp_path,
):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    base_job = _strict_job(src, dst)
    run_derive_pyramid_job(
        base_job,
        convert_fn=_writing_convert([]),
        meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
        viewer_info_fn=lambda _path: _viewer_info(),
    )
    manifest_path = dst.with_suffix(".manifest.json")
    old_manifest = manifest_path.read_bytes()
    initial_mtime = src.stat().st_mtime_ns

    def swapping_derived_viewer(path):
        if Path(path).name == "artifact.tif":
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"source-two")
            os.utime(replacement, ns=(initial_mtime, initial_mtime))
            os.replace(replacement, src)
        return _viewer_info()

    with pytest.raises(StaleDerivativeJobError) as excinfo:
        run_derive_pyramid_job(
            {**base_job, "force": True},
            convert_fn=_writing_convert([]),
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=swapping_derived_viewer,
        )

    assert excinfo.value.code == "source_generation_changed"
    assert manifest_path.read_bytes() == old_manifest


def test_strict_publication_accepts_display_default_changes_without_changing_pixel_semantics(
    tmp_path,
):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    source_info = _viewer_info()
    source_info["display_defaults"] = {"channels": [0, 1]}
    derived_info = _viewer_info()
    derived_info["display_defaults"] = {"channels": [0]}

    result = run_derive_pyramid_job(
        _strict_job(src, dst),
        convert_fn=_writing_convert([]),
        meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
        viewer_info_fn=lambda path: source_info
        if os.fspath(path) == os.fspath(src)
        else derived_info,
    )

    manifest = json.loads(dst.with_suffix(".manifest.json").read_text())
    assert result["status"] == "succeeded"
    assert manifest["semantics"]["display"]["default_channels"] == [0, 1]


def test_final_manifest_commit_recovers_equal_size_artifact_swap_and_preserves_valid_old_pair(
    monkeypatch, tmp_path
):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    job = _strict_job(src, dst)
    first = run_derive_pyramid_job(
        job,
        convert_fn=_writing_convert([]),
        meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
        viewer_info_fn=lambda _path: _viewer_info(),
    )
    manifest_path = dst.with_suffix(".manifest.json")
    old_manifest = manifest_path.read_bytes()
    artifact_path = os.fspath(first["derived_path"])
    original_replace = derivative_manifest.os.replace
    swapped = False

    def swap_artifact_at_commit(source, target):
        nonlocal swapped
        if os.fspath(target) == os.fspath(manifest_path) and not swapped:
            swapped = True
            current = os.stat(artifact_path)
            replacement = tmp_path / "equal-size-artifact"
            replacement.write_bytes(b"mutant pyramid bytes")
            os.utime(replacement, ns=(current.st_atime_ns, current.st_mtime_ns))
            original_replace(replacement, artifact_path)
        return original_replace(source, target)

    monkeypatch.setattr(derivative_manifest.os, "replace", swap_artifact_at_commit)

    with pytest.raises(TransientDerivativeError) as excinfo:
        run_derive_pyramid_job(
            {**job, "force": True},
            convert_fn=_writing_convert([]),
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=lambda _path: _viewer_info(),
        )

    assert excinfo.value.code == "artifact_publication_changed"
    assert manifest_path.read_bytes() == old_manifest
    _, restored_artifact = _assert_valid_manifest_artifact_pair(manifest_path)
    assert restored_artifact.read_bytes() == b"strict pyramid bytes"


def test_first_manifest_commit_removes_record_after_equal_size_artifact_swap(monkeypatch, tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    manifest_path = dst.with_suffix(".manifest.json")
    original_replace = derivative_manifest.os.replace
    swapped = False

    def swap_artifact_at_commit(source, target):
        nonlocal swapped
        if os.fspath(target) == os.fspath(manifest_path) and not swapped:
            swapped = True
            artifact_paths = list(dst.parent.glob("*.sha256-*.tif"))
            assert len(artifact_paths) == 1
            artifact_path = artifact_paths[0]
            current = artifact_path.stat()
            replacement = tmp_path / "equal-size-first-artifact"
            replacement.write_bytes(b"mutant pyramid bytes")
            os.utime(replacement, ns=(current.st_atime_ns, current.st_mtime_ns))
            original_replace(replacement, artifact_path)
        return original_replace(source, target)

    monkeypatch.setattr(derivative_manifest.os, "replace", swap_artifact_at_commit)

    with pytest.raises(TransientDerivativeError) as excinfo:
        run_derive_pyramid_job(
            _strict_job(src, dst),
            convert_fn=_writing_convert([]),
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=lambda _path: _viewer_info(),
        )

    assert excinfo.value.code == "artifact_publication_changed"
    assert not manifest_path.exists()


def test_failed_commit_does_not_restore_manifest_with_invalid_prior_referent(monkeypatch, tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    job = _strict_job(src, dst)
    first = run_derive_pyramid_job(
        job,
        convert_fn=_writing_convert([]),
        meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
        viewer_info_fn=lambda _path: _viewer_info(),
    )
    prior_artifact = tmp_path / "derived" / Path(first["derived_path"]).name
    prior_artifact.write_bytes(b"mutant pyramid bytes")
    manifest_path = dst.with_suffix(".manifest.json")
    original_replace = derivative_manifest.os.replace
    swapped = False

    def write_second_generation(src_path, dst_path, *, spec):
        del spec
        with open(dst_path, "wb") as artifact:
            artifact.write(b"second pyramid bytes")
        return ConvertResult(src_path, dst_path, 0, "", "")

    def swap_new_artifact_at_commit(source, target):
        nonlocal swapped
        if os.fspath(target) == os.fspath(manifest_path) and not swapped:
            swapped = True
            artifact_paths = [
                path for path in dst.parent.glob("*.sha256-*.tif") if path != prior_artifact
            ]
            assert len(artifact_paths) == 1
            new_artifact = artifact_paths[0]
            current = new_artifact.stat()
            replacement = tmp_path / "equal-size-new-artifact"
            replacement.write_bytes(b"mutant pyramid bytes")
            os.utime(replacement, ns=(current.st_atime_ns, current.st_mtime_ns))
            original_replace(replacement, new_artifact)
        return original_replace(source, target)

    monkeypatch.setattr(derivative_manifest.os, "replace", swap_new_artifact_at_commit)

    with pytest.raises(TransientDerivativeError) as excinfo:
        run_derive_pyramid_job(
            {**job, "force": True},
            convert_fn=write_second_generation,
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=lambda _path: _viewer_info(),
        )

    assert excinfo.value.code == "artifact_publication_changed"
    assert not manifest_path.exists()


def test_final_manifest_commit_rolls_back_when_source_becomes_symlink(monkeypatch, tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    job = _strict_job(src, dst)
    run_derive_pyramid_job(
        job,
        convert_fn=_writing_convert([]),
        meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
        viewer_info_fn=lambda _path: _viewer_info(),
    )
    manifest_path = dst.with_suffix(".manifest.json")
    old_manifest = manifest_path.read_bytes()
    original_replace = derivative_manifest.os.replace
    swapped = False

    def swap_source_after_commit(source, target):
        nonlocal swapped
        result = original_replace(source, target)
        if os.fspath(target) == os.fspath(manifest_path) and not swapped:
            swapped = True
            replacement = tmp_path / "replacement-source"
            replacement.write_bytes(b"source-one")
            src.unlink()
            src.symlink_to(replacement)
        return result

    monkeypatch.setattr(derivative_manifest.os, "replace", swap_source_after_commit)

    with pytest.raises(StaleDerivativeJobError) as excinfo:
        run_derive_pyramid_job(
            {**job, "force": True},
            convert_fn=_writing_convert([]),
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=lambda _path: _viewer_info(),
        )

    assert excinfo.value.code == "source_generation_changed"
    assert manifest_path.read_bytes() == old_manifest


def test_source_eio_from_filesystem_adapter_is_retryable(monkeypatch, tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    original_open = derivative_manifest.os.open

    def fail_source_open(path, flags, *args, **kwargs):
        if os.fspath(path) == os.fspath(src):
            raise OSError(errno.EIO, "injected source read failure")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(derivative_manifest.os, "open", fail_source_open)

    with pytest.raises(TransientDerivativeError) as excinfo:
        run_derive_pyramid_job(
            _strict_job(src, dst),
            convert_fn=_writing_convert([]),
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=lambda _path: _viewer_info(),
        )

    assert excinfo.value.code == "source_io_unavailable"


def test_artifact_fsync_eio_from_filesystem_adapter_is_retryable(monkeypatch, tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    original_fsync = derivative_manifest.os.fsync
    failed = False

    def fail_first_fsync(descriptor):
        nonlocal failed
        if not failed and stat.S_ISREG(os.fstat(descriptor).st_mode):
            failed = True
            raise OSError(errno.EIO, "injected artifact sync failure")
        return original_fsync(descriptor)

    monkeypatch.setattr(derivative_manifest.os, "fsync", fail_first_fsync)

    with pytest.raises(TransientDerivativeError) as excinfo:
        run_derive_pyramid_job(
            _strict_job(src, dst),
            convert_fn=_writing_convert([]),
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=lambda _path: _viewer_info(),
        )

    assert excinfo.value.code == "artifact_io_unavailable"


def test_strict_replay_fences_source_and_artifact_generations_around_semantic_probe(tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    calls = []
    base_job = _strict_job(src, dst)
    first = run_derive_pyramid_job(
        base_job,
        convert_fn=_writing_convert(calls),
        meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
        viewer_info_fn=lambda _path: _viewer_info(),
    )
    artifact_path = first["derived_path"]

    def swap_artifact(path):
        if os.fspath(path) == artifact_path:
            replacement = tmp_path / "replacement-artifact"
            replacement.write_bytes(b"strict pyramid bytes")
            os.replace(replacement, artifact_path)
        return _viewer_info()

    regenerated = run_derive_pyramid_job(
        base_job,
        convert_fn=_writing_convert(calls),
        meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
        viewer_info_fn=swap_artifact,
    )
    assert regenerated["status"] == "succeeded"
    assert len(calls) == 2

    initial_mtime = src.stat().st_mtime_ns

    def swap_source_during_replay(path):
        if os.fspath(path) == regenerated["derived_path"]:
            replacement = tmp_path / "replacement-source"
            replacement.write_bytes(b"source-two")
            os.utime(replacement, ns=(initial_mtime, initial_mtime))
            os.replace(replacement, src)
        return _viewer_info()

    with pytest.raises(StaleDerivativeJobError) as excinfo:
        run_derive_pyramid_job(
            base_job,
            convert_fn=_writing_convert(calls),
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=swap_source_during_replay,
        )
    assert excinfo.value.code == "source_generation_changed"


def test_strict_replay_is_invalidated_by_spec_change_and_force(tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-generation-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    calls = []
    base = _strict_job(src, dst)
    kwargs = {
        "convert_fn": _writing_convert(calls),
        "meta_fn": lambda _path: {"image_num_x": 16, "image_num_y": 8},
        "viewer_info_fn": lambda _path: _viewer_info(),
    }

    run_derive_pyramid_job(base, **kwargs)
    changed = {**base, "tile_size": 256}
    result = run_derive_pyramid_job(changed, **kwargs)
    assert result["status"] == "succeeded"
    assert len(calls) == 2
    manifest = json.loads(dst.with_suffix(".manifest.json").read_text())
    assert manifest["conversion_spec"]["requested"]["tile_size"] == 256
    assert manifest["conversion_spec"]["effective"]["tile_size"] == 256

    forced = run_derive_pyramid_job({**changed, "force": True}, **kwargs)
    assert forced["status"] == "succeeded"
    assert len(calls) == 3

    manifest = json.loads(dst.with_suffix(".manifest.json").read_text())
    manifest["conversion_spec"]["producer_revision"] = "mutable-build-label"
    dst.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    regenerated = run_derive_pyramid_job(changed, **kwargs)
    assert regenerated["status"] == "succeeded"
    assert len(calls) == 4


def test_forced_envelope_redelivery_replays_once_and_reclaims_previous_digest(tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-generation-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    calls = []
    payloads = [b"first forced artifact", b"second forced artifact"]

    def convert(source, destination, *, spec):
        payload = payloads[len(calls)]
        calls.append((source, destination, spec))
        Path(destination).write_bytes(payload)
        return ConvertResult(source, destination, 0, "", "")

    def envelope(job_id):
        return {
            "job_id": job_id,
            "job_type": "image.derive_pyramid",
            "metadata": {**_strict_job(src, dst), "force": True},
        }

    kwargs = {
        "convert_fn": convert,
        "meta_fn": lambda _path: {"image_num_x": 16, "image_num_y": 8},
        "viewer_info_fn": lambda _path: _viewer_info(),
        "require_manifest": True,
    }
    first_job = extract_derive_pyramid_payload(envelope("derive-request-1"))
    assert first_job is not None and first_job["force_id"] == "derive-request-1"
    first = run_derive_pyramid_job(first_job, **kwargs)
    redelivery = run_derive_pyramid_job(first_job, **kwargs)

    assert first["status"] == "succeeded"
    assert redelivery["status"] == "replayed"
    assert len(calls) == 1

    second_job = extract_derive_pyramid_payload(envelope("derive-request-2"))
    assert second_job is not None
    second = run_derive_pyramid_job(second_job, **kwargs)
    manifest = json.loads(dst.with_suffix(".manifest.json").read_text())
    artifacts = list(dst.parent.glob("file-1__pyramid.sha256-*.tif"))

    assert second["status"] == "succeeded"
    assert len(calls) == 2
    assert manifest["schema"] == derivative_manifest.SCHEMA_V1
    assert "request" not in manifest
    force_record = json.loads(
        (tmp_path / ".staging" / "file-1" / "pyramid" / "request.json").read_text()
    )
    assert force_record["force_id"] == "derive-request-2"
    assert artifacts == [Path(second["derived_path"])]
    assert not Path(first["derived_path"]).exists()


def test_forced_redelivery_repairs_private_record_from_durable_journal(monkeypatch, tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-generation-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    calls = []
    envelope = {
        "job_id": "derive-request-crash-window",
        "job_type": "image.derive_pyramid",
        "metadata": {**_strict_job(src, dst), "force": True},
    }
    job = extract_derive_pyramid_payload(envelope)
    assert job is not None
    kwargs = {
        "convert_fn": _writing_convert(calls),
        "meta_fn": lambda _path: {"image_num_x": 16, "image_num_y": 8},
        "viewer_info_fn": lambda _path: _viewer_info(),
        "require_manifest": True,
    }
    original_cleanup = derivative_manifest._cleanup_publication_workspace
    original_write_force_record = derivative_manifest._write_force_record
    simulate_crash_window = True

    def maybe_cleanup(destination):
        if simulate_crash_window:
            return None
        return original_cleanup(destination)

    def maybe_write_force_record(destination, force_id, manifest_payload):
        if simulate_crash_window:
            return None
        return original_write_force_record(destination, force_id, manifest_payload)

    monkeypatch.setattr(derivative_manifest, "_cleanup_publication_workspace", maybe_cleanup)
    monkeypatch.setattr(derivative_manifest, "_write_force_record", maybe_write_force_record)

    first = run_derive_pyramid_job(job, **kwargs)
    workspace = tmp_path / ".staging" / "file-1" / "pyramid"
    journal = json.loads((workspace / "journal.json").read_text())

    assert first["status"] == "succeeded"
    assert journal["schema"] == derivative_manifest.PUBLICATION_JOURNAL_SCHEMA_V2
    assert journal["force_id"] == "derive-request-crash-window"
    assert (
        journal["manifest_sha256"]
        == hashlib.sha256(dst.with_suffix(".manifest.json").read_bytes()).hexdigest()
    )
    assert not (workspace / "request.json").exists()

    simulate_crash_window = False
    redelivery = run_derive_pyramid_job(job, **kwargs)
    force_record = json.loads((workspace / "request.json").read_text())

    assert redelivery["status"] == "replayed"
    assert len(calls) == 1
    assert force_record["force_id"] == "derive-request-crash-window"
    assert force_record["manifest_sha256"] == journal["manifest_sha256"]
    assert not (workspace / "journal.json").exists()


def test_transient_force_record_read_failure_retries_without_conversion(monkeypatch, tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-generation-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    calls = []
    envelope = {
        "job_id": "derive-request-read-eio",
        "job_type": "image.derive_pyramid",
        "metadata": {**_strict_job(src, dst), "force": True},
    }
    job = extract_derive_pyramid_payload(envelope)
    assert job is not None
    kwargs = {
        "convert_fn": _writing_convert(calls),
        "meta_fn": lambda _path: {"image_num_x": 16, "image_num_y": 8},
        "viewer_info_fn": lambda _path: _viewer_info(),
        "require_manifest": True,
    }
    first = run_derive_pyramid_job(job, **kwargs)
    assert first["status"] == "succeeded"
    original_read_at = derivative_manifest._read_stable_regular_file_at

    def fail_force_record_read(directory, name, *, label, max_bytes):
        if name == "request.json":
            raise OSError(errno.EIO, "injected force-record read failure")
        return original_read_at(directory, name, label=label, max_bytes=max_bytes)

    monkeypatch.setattr(
        derivative_manifest,
        "_read_stable_regular_file_at",
        fail_force_record_read,
    )

    with pytest.raises(TransientDerivativeError) as excinfo:
        run_derive_pyramid_job(job, **kwargs)

    assert excinfo.value.code == "manifest_io_unavailable"
    assert len(calls) == 1


def test_rollout_publication_remains_v1_and_replayable(tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-generation-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    calls = []
    job = _strict_job(src, dst)
    kwargs = {
        "convert_fn": _writing_convert(calls),
        "meta_fn": lambda _path: {"image_num_x": 16, "image_num_y": 8},
        "viewer_info_fn": lambda _path: _viewer_info(),
    }
    run_derive_pyramid_job(job, **kwargs)
    manifest_path = dst.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == derivative_manifest.SCHEMA_V1
    assert "request" not in manifest

    replay = run_derive_pyramid_job(job, **kwargs)

    assert replay["status"] == "replayed"
    assert len(calls) == 1


def test_crash_journal_reclaims_only_uncommitted_resource_workspace(tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-generation-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    job = _strict_job(src, dst)
    kwargs = {
        "convert_fn": _writing_convert([]),
        "meta_fn": lambda _path: {"image_num_x": 16, "image_num_y": 8},
        "viewer_info_fn": lambda _path: _viewer_info(),
    }
    first = run_derive_pyramid_job(job, **kwargs)
    committed = Path(first["derived_path"])
    orphan = dst.parent / f"{dst.stem}.sha256-{'f' * 64}{dst.suffix}"
    orphan.write_bytes(b"uncommitted")
    unrelated = dst.parent / f"file-10__pyramid.sha256-{'e' * 64}.tif"
    unrelated.write_bytes(b"neighbor")
    workspace = tmp_path / ".staging" / "file-1" / "pyramid"
    workspace.mkdir(parents=True, exist_ok=True)
    for name in (
        "artifact.tif",
        "artifact.tif.transcode.ome.tif",
        "manifest.tmp",
        "publish.link",
        "recovery.link",
        "journal.tmp",
    ):
        (workspace / name).write_bytes(b"abandoned")
    (workspace / "journal.json").write_text(
        json.dumps(
            {
                "schema": "ultra.image-pyramid-publication-journal.v1",
                "artifact_basename": orphan.name,
                "previous_artifact_basename": committed.name,
            }
        )
    )

    replay = run_derive_pyramid_job(job, **kwargs)

    assert replay["status"] == "replayed"
    assert committed.is_file()
    assert not orphan.exists()
    assert unrelated.read_bytes() == b"neighbor"
    assert {path.name for path in workspace.iterdir()} == {"request.json"}


def test_crash_recovery_rejects_symlinked_derived_without_touching_referent(tmp_path):
    upload_root = tmp_path / "uploads"
    outside = tmp_path / "outside"
    outside.mkdir()
    resource_id = "file-1"
    destination = upload_root / "derived" / f"{resource_id}__pyramid.tif"
    artifact_name = f"{destination.stem}.sha256-{'f' * 64}{destination.suffix}"
    outside_artifact = outside / artifact_name
    outside_artifact.write_bytes(b"outside artifact")
    workspace = upload_root / ".staging" / resource_id / "pyramid"
    workspace.mkdir(parents=True)
    (workspace / "journal.json").write_text(
        json.dumps(
            {
                "schema": "ultra.image-pyramid-publication-journal.v1",
                "artifact_basename": artifact_name,
                "previous_artifact_basename": "",
            }
        )
    )
    os.symlink(outside, upload_root / "derived")

    with pytest.raises(TransientDerivativeError) as excinfo:
        derivative_manifest._cleanup_publication_workspace(destination)

    assert excinfo.value.code == "publication_recovery_unavailable"
    assert outside_artifact.read_bytes() == b"outside artifact"
    assert (workspace / "journal.json").is_file()


def test_spec_change_reclaims_prior_digest_even_when_not_replay_compatible(tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-generation-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    payloads = [b"old pyramid bytes", b"new pyramid bytes"]
    calls = []

    def convert(source, destination, *, spec):
        Path(destination).write_bytes(payloads[len(calls)])
        calls.append((source, destination, spec))
        return ConvertResult(source, destination, 0, "", "")

    base = _strict_job(src, dst)
    kwargs = {
        "convert_fn": convert,
        "meta_fn": lambda _path: {"image_num_x": 16, "image_num_y": 8},
        "viewer_info_fn": lambda _path: _viewer_info(),
    }
    first = run_derive_pyramid_job(base, **kwargs)
    second = run_derive_pyramid_job({**base, "tile_size": 256}, **kwargs)

    assert second["status"] == "succeeded"
    assert len(calls) == 2
    assert not Path(first["derived_path"]).exists()
    assert list(dst.parent.glob("file-1__pyramid.sha256-*.tif")) == [Path(second["derived_path"])]


def test_equal_size_in_place_artifact_mutation_after_commit_is_rejected(monkeypatch, tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-generation-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    manifest_path = dst.with_suffix(".manifest.json")
    original_replace = derivative_manifest.os.replace
    mutated = False

    def mutate_artifact_after_manifest_commit(source, target):
        nonlocal mutated
        result = original_replace(source, target)
        if os.fspath(target) == os.fspath(manifest_path) and not mutated:
            mutated = True
            artifact = next(dst.parent.glob("file-1__pyramid.sha256-*.tif"))
            before = artifact.stat()
            with artifact.open("r+b") as stream:
                stream.write(b"x" * before.st_size)
                stream.flush()
                os.fsync(stream.fileno())
            os.utime(artifact, ns=(before.st_atime_ns, before.st_mtime_ns))
        return result

    monkeypatch.setattr(derivative_manifest.os, "replace", mutate_artifact_after_manifest_commit)

    with pytest.raises(TransientDerivativeError) as excinfo:
        run_derive_pyramid_job(
            _strict_job(src, dst),
            convert_fn=_writing_convert([]),
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=lambda _path: _viewer_info(),
        )

    assert excinfo.value.code == "artifact_publication_changed"
    assert not manifest_path.exists()
    assert not list(dst.parent.glob("file-1__pyramid.sha256-*.tif"))


def test_strict_auto_format_records_requested_and_effective_specs(tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-generation-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"

    run_derive_pyramid_job(
        {**_strict_job(src, dst), "fmt": "auto"},
        convert_fn=_writing_convert([]),
        meta_fn=lambda _path: {
            "image_num_x": 16,
            "image_num_y": 8,
            "image_num_z": 3,
            "image_num_t": 2,
        },
        viewer_info_fn=lambda _path: _viewer_info(),
    )

    conversion = json.loads(dst.with_suffix(".manifest.json").read_text())["conversion_spec"]
    assert conversion["requested"]["fmt"] == "auto"
    assert conversion["effective"]["fmt"] == "ome-bigtiff"


@pytest.mark.parametrize(("channel_count", "admitted"), [(9000, False), (400, True)])
def test_manifest_size_admission_precedes_conversion(tmp_path, channel_count, admitted):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-generation-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    calls = []
    info = _viewer_info(names=[f"Channel {index}" for index in range(channel_count)])
    info["axis_sizes"]["C"] = channel_count

    if admitted:
        result = run_derive_pyramid_job(
            _strict_job(src, dst),
            convert_fn=_writing_convert(calls),
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=lambda _path: info,
        )
        assert result["status"] == "succeeded"
        assert len(calls) == 1
        manifest = json.loads(dst.with_suffix(".manifest.json").read_text())
        assert len(manifest["semantics"]["channels"]) == channel_count
        return

    with pytest.raises(DeterministicDerivativeError) as excinfo:
        run_derive_pyramid_job(
            _strict_job(src, dst),
            convert_fn=_writing_convert(calls),
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=lambda _path: info,
        )
    assert excinfo.value.code == "manifest_too_large"
    assert calls == []
    assert not dst.parent.exists()


def test_strict_preferred_reader_binds_czi_scene_provenance(tmp_path):
    src = tmp_path / "source.czi"
    src.write_bytes(b"czi-source")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    source_info = _viewer_info(scene="Scene0")
    source_info["metadata"].update({"reader": "bioio", "scene_count": 3, "selected_scene_index": 0})

    def transcode(source, destination, **kwargs):
        assert kwargs["prefer"] == "first"
        with open(destination, "wb") as stream:
            stream.write(b"intermediate")
        return TranscodeResult(
            path=destination,
            series_count=3,
            series_index=0,
            series_name="Scene0",
            num_c=2,
            num_z=3,
            dtype="uint16",
            series_names=["Scene0", "Scene1", "Scene2"],
            num_t=2,
        )

    run_derive_pyramid_job(
        _strict_job(src, dst),
        convert_fn=_writing_convert([]),
        meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
        transcode_fn=transcode,
        viewer_info_fn=lambda _path: _viewer_info(),
        source_viewer_info_fn=lambda _path: source_info,
    )
    manifest = json.loads(dst.with_suffix(".manifest.json").read_text())
    assert manifest["producer"] == {
        "reader": "bioio",
        "series_count": 3,
        "series_index": 0,
        "series_name": "Scene0",
    }
    assert manifest["semantics"]["scene"] == {
        "count": 3,
        "id": "Scene0",
        "index": 0,
    }


def test_strict_nd2_does_not_fall_back_to_native_when_bioio_fails(tmp_path):
    from ultra_deepagents.imaging.transcode import TranscodeInputError

    src = tmp_path / "source.nd2"
    src.write_bytes(b"nd2-source")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    native_calls = []

    with pytest.raises(DeterministicDerivativeError) as excinfo:
        run_derive_pyramid_job(
            _strict_job(src, dst),
            convert_fn=_writing_convert(native_calls),
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            transcode_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                TranscodeInputError("native fallback forbidden")
            ),
            viewer_info_fn=lambda _path: _viewer_info(),
            source_viewer_info_fn=lambda _path: _viewer_info(),
        )
    assert excinfo.value.code == "unsupported_source"
    assert native_calls == []


def test_strict_source_replacement_with_same_size_and_mtime_is_rejected(tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-one")
    initial_mtime = src.stat().st_mtime_ns
    dst = tmp_path / "derived" / "file-1__pyramid.tif"

    def replace_source(source, destination, *, spec):
        with open(destination, "wb") as stream:
            stream.write(b"artifact")
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"source-two")
        os.utime(replacement, ns=(initial_mtime, initial_mtime))
        os.replace(replacement, src)
        return ConvertResult(source, destination, 0, "", "")

    with pytest.raises(StaleDerivativeJobError) as excinfo:
        run_derive_pyramid_job(
            _strict_job(src, dst),
            convert_fn=replace_source,
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=lambda _path: _viewer_info(),
        )
    assert excinfo.value.code == "source_generation_changed"


def test_strict_source_symlink_is_rejected_before_decode(tmp_path):
    target = tmp_path / "target.ome.tif"
    target.write_bytes(b"source")
    src = tmp_path / "source.ome.tif"
    src.symlink_to(target)
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    called = []

    with pytest.raises(StaleDerivativeJobError) as excinfo:
        run_derive_pyramid_job(
            _strict_job(target, dst) | {"src_path": str(src)},
            convert_fn=_writing_convert(called),
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=lambda _path: _viewer_info(),
        )
    assert excinfo.value.code == "source_generation_changed"
    assert called == []


@pytest.mark.parametrize(
    "corruption",
    ["unknown", "trailing", "duplicate", "oversize", "unsafe", "symlink", "capability"],
)
def test_strict_derivative_rejects_noncanonical_or_unsafe_manifest_and_regenerates(
    tmp_path, corruption
):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    calls = []
    kwargs = {
        "convert_fn": _writing_convert(calls),
        "meta_fn": lambda _path: {"image_num_x": 16, "image_num_y": 8},
        "viewer_info_fn": lambda _path: _viewer_info(),
    }
    run_derive_pyramid_job(_strict_job(src, dst), **kwargs)
    manifest_path = dst.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if corruption == "unknown":
        manifest["unexpected"] = True
        manifest_path.write_text(json.dumps(manifest))
    elif corruption == "trailing":
        manifest_path.write_text(json.dumps(manifest) + "\n{}")
    elif corruption == "duplicate":
        manifest_path.write_text(
            json.dumps(manifest).replace('"schema":', '"schema":"duplicate","schema":', 1)
        )
    elif corruption == "oversize":
        manifest_path.write_text("{" + " " * (1 << 20) + "}")
    elif corruption == "unsafe":
        manifest["artifact"]["basename"] = "../outside.tif"
        manifest_path.write_text(json.dumps(manifest))
    elif corruption == "capability":
        manifest["capabilities"]["tile_t"] = True
        manifest_path.write_text(json.dumps(manifest))
    else:
        artifact = manifest_path.parent / manifest["artifact"]["basename"]
        artifact.unlink()
        artifact.symlink_to(src)

    result = run_derive_pyramid_job(_strict_job(src, dst), **kwargs)

    assert result["status"] == "succeeded"
    assert len(calls) == 2


def test_strict_derivative_failure_and_source_fence_preserve_old_manifest(tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-one")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    run_derive_pyramid_job(
        _strict_job(src, dst),
        convert_fn=_writing_convert([]),
        meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
        viewer_info_fn=lambda _path: _viewer_info(),
    )
    manifest_path = dst.with_suffix(".manifest.json")
    old_manifest = manifest_path.read_bytes()

    src.write_bytes(b"source-two")

    def mutate_source(_src, temp_dst, *, spec):
        with open(temp_dst, "wb") as artifact:
            artifact.write(b"new generation")
        src.write_bytes(b"source-mutated-during-conversion")
        return ConvertResult(_src, temp_dst, 0, "", "")

    with pytest.raises(StaleDerivativeJobError) as excinfo:
        run_derive_pyramid_job(
            _strict_job(src, dst),
            convert_fn=mutate_source,
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=lambda _path: _viewer_info(),
        )
    assert excinfo.value.code == "source_generation_changed"
    assert manifest_path.read_bytes() == old_manifest


def test_strict_derivative_rejects_wrong_catalog_sha_before_conversion(tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    job = _strict_job(src, dst)
    job["source_sha256"] = "0" * 64
    calls = []

    with pytest.raises(StaleDerivativeJobError) as excinfo:
        run_derive_pyramid_job(
            job,
            convert_fn=_writing_convert(calls),
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=lambda _path: _viewer_info(),
        )
    assert excinfo.value.code == "catalog_source_digest_mismatch"
    assert calls == []


def test_job_from_dict_defaults_and_spec():
    job = DerivePyramidJob.from_dict(
        {"resource_id": "r1", "src_path": "/a.czi", "dst_path": "/d.tif"}
    )
    assert job.tile_size == 512 and job.layout == "topdirs" and job.fmt == "bigtiff"
    assert job.spec().options() == "compression lzw tiles 512 pyramid topdirs"


def test_runner_converts_and_reports_metadata():
    seen: dict = {}

    def fake_convert(src, dst, *, spec):
        seen["src"], seen["dst"], seen["spec"] = src, dst, spec
        return ConvertResult(src, dst, 0, "", "")

    def fake_meta(path):
        if path == "/a.lsm":
            # source: decodable, not a native tiled pyramid -> proceed to convert
            return {"image_num_x": 2048, "image_num_y": 2048}
        assert path == "/d.tif"
        return {
            "image_num_resolution_levels": 5,
            "image_res_l_scales": "1,0.5,0.25,0.125,0.0625",
            "image_num_x": 2048,
            "image_num_y": 2048,
        }

    out = run_derive_pyramid_job(
        {"resource_id": "r1", "src_path": "/a.lsm", "dst_path": "/d.tif", "tile_size": 256},
        convert_fn=fake_convert,
        meta_fn=fake_meta,
    )
    assert out["status"] == "succeeded"
    assert out["resource_id"] == "r1"
    assert out["derived_path"] == "/d.tif"
    assert out["levels"] == 5
    assert out["num_x"] == 2048
    assert seen["src"] == "/a.lsm"
    assert seen["spec"].tile_size == 256


def test_auto_fmt_volume_derives_ome_bigtiff():
    # A z-stack source must derive to OME-BigTIFF so its Z planes survive (plain
    # BigTIFF flattens a multichannel OME hyperstack to a single plane).
    seen = {}

    def fake_convert(src, dst, *, spec):
        seen["fmt"] = spec.fmt
        return ConvertResult(src, dst, 0, "", "")

    out = run_derive_pyramid_job(
        {"resource_id": "r", "src_path": "/v.ome.tiff", "dst_path": "/d.tif", "fmt": "auto"},
        convert_fn=fake_convert,
        meta_fn=lambda _p: {
            "image_num_x": 4096,
            "image_num_y": 4096,
            "image_num_z": 80,
            "image_num_c": 7,
        },
    )
    assert seen["fmt"] == "ome-bigtiff" and out["fmt"] == "ome-bigtiff"


def test_auto_fmt_flat_2d_derives_bigtiff():
    # A flat 2D slide stays BigTIFF (tile-addressable; OME wrapper breaks -tile).
    seen = {}

    def fake_convert(src, dst, *, spec):
        seen["fmt"] = spec.fmt
        return ConvertResult(src, dst, 0, "", "")

    run_derive_pyramid_job(
        {"resource_id": "r", "src_path": "/big.svs", "dst_path": "/d.tif", "fmt": "auto"},
        convert_fn=fake_convert,
        meta_fn=lambda _p: {
            "image_num_x": 50000,
            "image_num_y": 40000,
            "image_num_z": 1,
            "image_num_c": 3,
        },
    )
    assert seen["fmt"] == "bigtiff"


def test_auto_fmt_paged_zstack_derives_ome_bigtiff():
    seen = {}

    def fake_convert(src, dst, *, spec):
        seen["fmt"] = spec.fmt
        return ConvertResult(src, dst, 0, "", "")

    run_derive_pyramid_job(
        {"resource_id": "r", "src_path": "/pages.tif", "dst_path": "/d.tif", "fmt": "auto"},
        convert_fn=fake_convert,
        meta_fn=lambda _p: {
            "image_num_x": 1024,
            "image_num_y": 1024,
            "image_num_z": 1,
            "image_num_c": 1,
            "image_num_p": 40,
        },
    )
    assert seen["fmt"] == "ome-bigtiff"


def test_native_tiled_pyramid_source_skips_convert():
    # A source already exposing a tiled multi-resolution pyramid (e.g. a COG/orthomosaic)
    # is served tile-by-tile directly, so the potentially huge convert is skipped.
    called = {"convert": False}

    def fake_convert(src, dst, *, spec):
        called["convert"] = True
        return ConvertResult(src, dst, 0, "", "")

    out = run_derive_pyramid_job(
        {"resource_id": "r", "src_path": "/ortho.tif", "dst_path": "/d.tif", "fmt": "auto"},
        convert_fn=fake_convert,
        meta_fn=lambda _p: {"image_num_resolution_levels": 8, "tile_num_x": 256},
    )
    assert out["status"] == "skipped_native_pyramid_no_manifest"
    assert out["derived_path"] is None
    assert called["convert"] is False


def test_strict_native_pyramid_terminal_is_fenced_to_source_generation(tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source-one")
    initial_mtime = src.stat().st_mtime_ns
    dst = tmp_path / "derived" / "file-1__pyramid.tif"

    def swapping_meta(_path):
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"source-two")
        os.utime(replacement, ns=(initial_mtime, initial_mtime))
        os.replace(replacement, src)
        return {"image_num_resolution_levels": 8, "tile_num_x": 256}

    with pytest.raises(StaleDerivativeJobError) as excinfo:
        run_derive_pyramid_job(
            _strict_job(src, dst),
            convert_fn=_writing_convert([]),
            meta_fn=swapping_meta,
            viewer_info_fn=lambda _path: _viewer_info(),
        )

    assert excinfo.value.code == "source_generation_changed"


def test_strict_missing_viewer_info_dependency_is_retryable_before_convert(tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"
    calls = []

    with pytest.raises(TransientDerivativeError) as excinfo:
        run_derive_pyramid_job(
            _strict_job(src, dst),
            convert_fn=_writing_convert(calls),
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=None,
            require_manifest=True,
        )

    assert excinfo.value.code == "viewer_info_unavailable"
    assert calls == []
    assert not dst.parent.exists()


def test_strict_unknown_runtime_convert_failure_is_retryable(tmp_path):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"

    def fail_convert(*_args, **_kwargs):
        raise RuntimeError("opaque lower-layer failure")

    with pytest.raises(TransientDerivativeError) as excinfo:
        run_derive_pyramid_job(
            _strict_job(src, dst),
            convert_fn=fail_convert,
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=lambda _path: _viewer_info(),
        )
    assert excinfo.value.code == "conversion_unavailable"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tile_size", 0),
        ("compression", "not-a-real-compression"),
        ("layout", "not-a-real-layout"),
        ("fmt", "not-a-real-format"),
    ],
)
def test_strict_invalid_conversion_spec_is_typed_before_publication(tmp_path, field, value):
    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"

    with pytest.raises(DeterministicDerivativeError) as excinfo:
        run_derive_pyramid_job(
            {**_strict_job(src, dst), field: value},
            convert_fn=_writing_convert([]),
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=lambda _path: _viewer_info(),
        )

    assert excinfo.value.code == "invalid_conversion_spec"
    assert not dst.parent.exists()


def test_strict_structured_convert_disposition_and_lower_layer_error_preservation(tmp_path):
    from ultra_deepagents.imaging.convert import (
        ConversionDependencyError,
        ConversionInputError,
        ConversionResourceError,
    )

    src = tmp_path / "source.ome.tif"
    src.write_bytes(b"source")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"

    def run_with(exc):
        def fail_convert(*_args, **_kwargs):
            raise exc

        return run_derive_pyramid_job(
            _strict_job(src, dst),
            convert_fn=fail_convert,
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            viewer_info_fn=lambda _path: _viewer_info(),
        )

    with pytest.raises(DeterministicDerivativeError) as deterministic:
        run_with(ConversionInputError("unsupported input"))
    assert deterministic.value.code == "conversion_rejected"
    for exc in (
        ConversionDependencyError("imgcnv missing"),
        ConversionResourceError("imgcnv killed"),
    ):
        with pytest.raises(TransientDerivativeError):
            run_with(exc)

    with pytest.raises(TransientDerivativeError) as preserved:
        run_with(TransientDerivativeError("lower_layer_retry"))
    assert preserved.value.code == "lower_layer_retry"


def test_strict_missing_bioio_dependency_is_retryable(tmp_path):
    from ultra_deepagents.imaging.transcode import TranscodeDependencyError

    src = tmp_path / "source.nd2"
    src.write_bytes(b"source")
    dst = tmp_path / "derived" / "file-1__pyramid.tif"

    with pytest.raises(TransientDerivativeError) as excinfo:
        run_derive_pyramid_job(
            _strict_job(src, dst),
            convert_fn=_writing_convert([]),
            meta_fn=lambda _path: {"image_num_x": 16, "image_num_y": 8},
            transcode_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                TranscodeDependencyError("bioio missing")
            ),
            viewer_info_fn=lambda _path: _viewer_info(),
            source_viewer_info_fn=lambda _path: _viewer_info(),
        )
    assert excinfo.value.code == "transcode_unavailable"


def test_pyramidal_but_untiled_source_still_converts():
    # Multi-resolution but NOT tiled (no tile grid) -> -tile needs a derived tiled pyramid.
    called = {"convert": False}

    def fake_convert(src, dst, *, spec):
        called["convert"] = True
        return ConvertResult(src, dst, 0, "", "")

    out = run_derive_pyramid_job(
        {"resource_id": "r", "src_path": "/striped.tif", "dst_path": "/d.tif", "fmt": "auto"},
        convert_fn=fake_convert,
        meta_fn=lambda _p: {
            "image_num_x": 8000,
            "image_num_y": 8000,
            "image_num_resolution_levels": 6,
        },  # no tile_num_x
    )
    assert called["convert"] is True and out["status"] == "succeeded"


def test_auto_fmt_without_engine_falls_back_to_bigtiff():
    # No meta_fn (native engine absent) keeps the tile-serving default.
    seen = {}

    def fake_convert(src, dst, *, spec):
        seen["fmt"] = spec.fmt
        return ConvertResult(src, dst, 0, "", "")

    run_derive_pyramid_job(
        {"resource_id": "r", "src_path": "/x.tif", "dst_path": "/d.tif", "fmt": "auto"},
        convert_fn=fake_convert,
        meta_fn=None,
    )
    assert seen["fmt"] == "bigtiff"


def test_undecodable_source_transcodes_via_bioio(tmp_path):
    # libbioimage can't decode the source (empty meta) -> bioio transcodes it to an
    # intermediate OME-TIFF, and the pyramid is built from THAT (not the original).
    src = str(tmp_path / "scan.lif")
    dst = str(tmp_path / "d.tif")
    intermediate = dst + ".transcode.ome.tif"
    seen: dict = {}

    def fake_meta(path):
        if path.endswith(".transcode.ome.tif"):
            return {"image_num_x": 3000, "image_num_y": 3000, "image_num_z": 2, "image_num_c": 2}
        return {}  # source: undecodable by libbioimage

    def fake_transcode(s, d, **_kw):
        open(d, "w").close()  # create the intermediate so the cleanup has a real file
        return TranscodeResult(
            path=d,
            series_count=16,
            series_index=4,
            series_name="Series005",
            num_c=2,
            num_z=2,
            dtype="uint8",
            series_names=[f"S{i}" for i in range(16)],
        )

    def fake_convert(s, d, *, spec):
        seen["convert_src"] = s
        seen["fmt"] = spec.fmt
        seen["intermediate_present_at_convert"] = os.path.exists(intermediate)
        return ConvertResult(s, d, 0, "", "")

    out = run_derive_pyramid_job(
        {"resource_id": "r", "src_path": src, "dst_path": dst, "fmt": "auto"},
        convert_fn=fake_convert,
        meta_fn=fake_meta,
        transcode_fn=fake_transcode,
    )
    assert out["status"] == "succeeded"
    assert out["transcoded"] is True and out["source_reader"] == "bioio"
    assert (
        out["series_count"] == 16 and out["series_index"] == 4 and out["series_name"] == "Series005"
    )
    # The pyramid is derived from the transcoded intermediate, not the .lif.
    assert seen["convert_src"] == intermediate
    # Multichannel/volume series must stay OME-BigTIFF (preserve channels + z planes).
    assert seen["fmt"] == "ome-bigtiff"
    assert seen["intermediate_present_at_convert"] is True
    # The redundant intermediate is reclaimed after the pyramid exists.
    assert not os.path.exists(intermediate)


def test_decodable_source_does_not_transcode():
    # A source libbioimage CAN read must never invoke the bioio fallback.
    called = {"transcode": False}

    def fake_transcode(s, d, **_kw):
        called["transcode"] = True
        raise AssertionError("transcode must not run for a decodable source")

    out = run_derive_pyramid_job(
        {"resource_id": "r", "src_path": "/a.tif", "dst_path": "/d.tif", "fmt": "auto"},
        convert_fn=lambda s, d, *, spec: ConvertResult(s, d, 0, "", ""),
        meta_fn=lambda _p: {
            "image_num_x": 2048,
            "image_num_y": 2048,
            "image_num_z": 1,
            "image_num_c": 1,
        },
        transcode_fn=fake_transcode,
    )
    assert called["transcode"] is False
    assert out.get("transcoded") is None


def test_prefer_bioio_extension_forces_transcode(tmp_path):
    # .czi is in PREFER_BIOIO_EXTENSIONS -> route through bioio even though libbioimage
    # COULD decode it; the pyramid is built from the bioio transcode (Zeiss mosaics read
    # correctly there). meta_fn(source) is not even consulted for the routing decision.
    src = str(tmp_path / "scene.czi")
    dst = str(tmp_path / "d.tif")
    seen: dict = {}

    def fake_meta(path):
        # libbioimage CAN read the czi (real geometry), but prefer-bioio overrides it.
        if path.endswith(".transcode.ome.tif"):
            return {"image_num_x": 5913, "image_num_y": 5679, "image_num_z": 1, "image_num_c": 2}
        return {"image_num_x": 5913, "image_num_y": 5679, "image_num_z": 1, "image_num_c": 2}

    def fake_transcode(s, d, **_kw):
        open(d, "w").close()
        return TranscodeResult(
            path=d,
            series_count=1,
            series_index=0,
            series_name="Scene0",
            num_c=2,
            num_z=1,
            dtype="uint16",
            series_names=["Scene0"],
        )

    def fake_convert(s, d, *, spec):
        seen["convert_src"] = s
        seen["fmt"] = spec.fmt
        return ConvertResult(s, d, 0, "", "")

    out = run_derive_pyramid_job(
        {"resource_id": "r", "src_path": src, "dst_path": dst, "fmt": "auto"},
        convert_fn=fake_convert,
        meta_fn=fake_meta,
        transcode_fn=fake_transcode,
    )
    assert out["transcoded"] is True and out["source_reader"] == "bioio"
    assert seen["convert_src"].endswith(".transcode.ome.tif")  # built from bioio, not the czi
    assert seen["fmt"] == "ome-bigtiff"  # 2 channels -> keep them


def test_prefer_bioio_soft_falls_back_to_libbioimage(tmp_path):
    # If bioio cannot read a prefer-bioio source, fall back to a normal libbioimage
    # convert of the source — don't discard a working native render.
    from ultra_deepagents.imaging.transcode import TranscodeInputError

    src = str(tmp_path / "scene.czi")
    dst = str(tmp_path / "d.tif")
    seen: dict = {}

    def fake_meta(_path):
        return {"image_num_x": 4000, "image_num_y": 4000, "image_num_z": 1, "image_num_c": 1}

    def failing_transcode(s, d, **_kw):
        raise TranscodeInputError("bioio cannot read this czi variant")

    def fake_convert(s, d, *, spec):
        seen["convert_src"] = s
        seen["fmt"] = spec.fmt
        return ConvertResult(s, d, 0, "", "")

    out = run_derive_pyramid_job(
        {"resource_id": "r", "src_path": src, "dst_path": dst, "fmt": "auto"},
        convert_fn=fake_convert,
        meta_fn=fake_meta,
        transcode_fn=failing_transcode,
    )
    assert out["status"] == "succeeded"
    assert out.get("transcoded") is None  # bioio failed -> libbioimage path taken
    assert seen["convert_src"] == src  # converted the original czi, no intermediate
    assert seen["fmt"] == "bigtiff"  # flat single-channel 2D -> bigtiff
    assert not os.path.exists(dst + ".transcode.ome.tif")  # intermediate cleaned up


def test_runner_propagates_convert_failure():
    def failing(src, dst, *, spec):
        raise RuntimeError("imgcnv conversion failed")

    with pytest.raises(RuntimeError, match="imgcnv conversion failed"):
        run_derive_pyramid_job(
            {"resource_id": "r", "src_path": "/a", "dst_path": "/d"}, convert_fn=failing
        )


def test_extract_payload_from_data_agent_envelope():
    env = {
        "job_id": "derive-request-7",
        "job_type": "image.derive_pyramid",
        "metadata": {
            "resource_id": "r",
            "src_path": "/a.lsm",
            "dst_path": "/d.tif",
            "tile_size": 256,
        },
    }
    job = extract_derive_pyramid_payload(env)
    assert job is not None and job["src_path"] == "/a.lsm" and job["tile_size"] == 256
    assert job["force_id"] == "derive-request-7"


def test_extract_payload_skips_other_job_types():
    assert (
        extract_derive_pyramid_payload(
            {"job_type": "caption.generate", "metadata": {"src_path": "/x"}}
        )
        is None
    )


def test_extract_payload_accepts_direct_job_dict():
    direct = {"resource_id": "r", "src_path": "/a", "dst_path": "/d"}
    assert extract_derive_pyramid_payload(direct) == direct


def test_runner_metadata_is_best_effort():
    def ok_convert(src, dst, *, spec):
        return ConvertResult(src, dst, 0, "", "")

    def bad_meta(path):
        raise ValueError("cannot read meta")

    out = run_derive_pyramid_job(
        {"resource_id": "r", "src_path": "/a", "dst_path": "/d"},
        convert_fn=ok_convert,
        meta_fn=bad_meta,
    )
    assert out["status"] == "succeeded"
    assert "meta_warning" in out


def test_prefer_bioio_default_set_covers_proprietary_microscopy_formats():
    # The baked-in default routes the proprietary microscopy formats (that libbioimage
    # reads poorly or not at all) through bioio deterministically, and does NOT include
    # plain tiff/svs (libbioimage-native) or zarr (native ngff, never transcoded).
    from ultra_deepagents.imaging.job import _prefer_bioio_extensions

    exts = _prefer_bioio_extensions()
    assert {"czi", "nd2", "lif", "dv", "r3d"} <= exts
    assert "zarr" not in exts and "ome.zarr" not in exts
    assert "tif" not in exts and "tiff" not in exts and "svs" not in exts


def test_prefer_bioio_env_override_replaces_default():
    from ultra_deepagents.imaging.job import _prefer_bioio_extensions

    prev = os.environ.get("ULTRA_IMGSVC_PREFER_BIOIO_EXTS")
    os.environ["ULTRA_IMGSVC_PREFER_BIOIO_EXTS"] = ".czi, nd2 ,lif"
    try:
        assert _prefer_bioio_extensions() == frozenset({"czi", "nd2", "lif"})
    finally:
        if prev is None:
            del os.environ["ULTRA_IMGSVC_PREFER_BIOIO_EXTS"]
        else:
            os.environ["ULTRA_IMGSVC_PREFER_BIOIO_EXTS"] = prev


def test_nd2_routes_through_bioio_transcode(tmp_path):
    # A Nikon .nd2 (no libbioimage reader) must go straight to the bioio transcode lane —
    # meta_fn(source) is never consulted for the routing decision.
    src = str(tmp_path / "acquisition.nd2")
    dst = str(tmp_path / "d.tif")
    seen: dict = {}

    def fake_meta(path):
        if path.endswith(".transcode.ome.tif"):
            return {"image_num_x": 2048, "image_num_y": 2048, "image_num_z": 1, "image_num_c": 3}
        raise AssertionError("meta_fn must NOT be called on the .nd2 source (prefer-bioio)")

    def fake_transcode(s, d, **_kw):
        open(d, "w").close()
        return TranscodeResult(
            path=d,
            series_count=1,
            series_index=0,
            series_name="XYPos0",
            num_c=3,
            num_z=1,
            dtype="uint16",
            series_names=["XYPos0"],
        )

    def fake_convert(s, d, *, spec):
        seen["convert_src"] = s
        return ConvertResult(s, d, 0, "", "")

    out = run_derive_pyramid_job(
        {"resource_id": "r", "src_path": src, "dst_path": dst, "fmt": "auto"},
        convert_fn=fake_convert,
        meta_fn=fake_meta,
        transcode_fn=fake_transcode,
    )
    assert out["transcoded"] is True and out["source_reader"] == "bioio"
    assert seen["convert_src"].endswith(".transcode.ome.tif")
