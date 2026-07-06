from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import sys
import tarfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "services" / "megaseg_service" / "app.py"
APP_SPEC = importlib.util.spec_from_file_location("megaseg_service_app_under_test", APP_PATH)
assert APP_SPEC is not None and APP_SPEC.loader is not None
megaseg_app = importlib.util.module_from_spec(APP_SPEC)
sys.modules[APP_SPEC.name] = megaseg_app
APP_SPEC.loader.exec_module(megaseg_app)


def _settings(tmp_path: Path) -> megaseg_app.ServiceSettings:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    return megaseg_app.ServiceSettings(
        checkpoint_path=checkpoint,
        device="cpu",
        job_root=tmp_path / "jobs",
        artifact_root=tmp_path / "artifacts",
        api_key="test-key",
        max_concurrent_jobs=1,
    )


def _app_for_runtime(runtime: megaseg_app.ServiceRuntime) -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(megaseg_runtime=runtime))


def _queued_record(
    runtime: megaseg_app.ServiceRuntime,
    job_id: str,
    source_path: Path,
) -> megaseg_app.JobRecord:
    now = megaseg_app._utc_now()
    return megaseg_app._save_job_record(
        runtime,
        megaseg_app.JobRecord(
            job_id=job_id,
            status="queued",
            created_at=now,
            updated_at=now,
            request=megaseg_app.JobRequest(),
            resolved_sources=[str(source_path)],
        ),
    )


def test_extract_uploaded_archive_rejects_symlink_escape(tmp_path: Path) -> None:
    inputs_dir = tmp_path / "inputs"
    outside_dir = tmp_path / "outside"
    inputs_dir.mkdir()
    outside_dir.mkdir()
    archive_path = inputs_dir / "sample.zarr.tar.gz"

    with tarfile.open(archive_path, "w:gz") as archive:
        root = tarfile.TarInfo("sample.zarr")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)

        link = tarfile.TarInfo("sample.zarr/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"
        archive.addfile(link)

        payload = b"escaped"
        nested = tarfile.TarInfo("sample.zarr/link/escape.txt")
        nested.size = len(payload)
        archive.addfile(nested, io.BytesIO(payload))

    with pytest.raises(RuntimeError):
        megaseg_app._extract_uploaded_archive(archive_path, inputs_dir)

    assert not (outside_dir / "escape.txt").exists()


def test_extract_uploaded_archive_rejects_top_level_sibling_and_preserves_existing_file(tmp_path: Path) -> None:
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    victim = inputs_dir / "victim.tif"
    victim.write_bytes(b"original")
    archive_path = inputs_dir / "sample.zarr.tar.gz"

    with tarfile.open(archive_path, "w:gz") as archive:
        root = tarfile.TarInfo("sample.zarr")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)

        zgroup = tarfile.TarInfo("sample.zarr/.zgroup")
        zgroup_payload = b"{}"
        zgroup.size = len(zgroup_payload)
        archive.addfile(zgroup, io.BytesIO(zgroup_payload))

        sibling = tarfile.TarInfo("victim.tif")
        sibling_payload = b"overwritten"
        sibling.size = len(sibling_payload)
        archive.addfile(sibling, io.BytesIO(sibling_payload))

    with pytest.raises(RuntimeError, match="expected root"):
        megaseg_app._extract_uploaded_archive(archive_path, inputs_dir)

    assert victim.read_bytes() == b"original"


def test_safe_extract_tar_rejects_too_many_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(megaseg_app, "MAX_EXTRACTED_TAR_MEMBERS", 1)
    inputs_dir = tmp_path / "inputs"
    archive_path = tmp_path / "sample.tar.gz"
    payload = b"x"

    with tarfile.open(archive_path, "w:gz") as archive:
        root = tarfile.TarInfo("sample.zarr")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        member = tarfile.TarInfo("sample.zarr/a.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with tarfile.open(archive_path, "r:gz") as archive:
        monkeypatch.setattr(
            archive,
            "getmembers",
            lambda: (_ for _ in ()).throw(AssertionError("getmembers should not be used")),
        )
        with pytest.raises(RuntimeError, match="too many"):
            megaseg_app._safe_extract_tar(archive, inputs_dir)

    assert not (inputs_dir / "sample.zarr" / "a.txt").exists()


def test_safe_extract_tar_rejects_oversize_uncompressed_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(megaseg_app, "MAX_EXTRACTED_TAR_UNCOMPRESSED_BYTES", 3)
    inputs_dir = tmp_path / "inputs"
    archive_path = tmp_path / "sample.tar.gz"
    payload = b"abcd"

    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("sample.zarr/a.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with tarfile.open(archive_path, "r:gz") as archive:
        with pytest.raises(RuntimeError, match="uncompressed"):
            megaseg_app._safe_extract_tar(archive, inputs_dir)

    assert not (inputs_dir / "sample.zarr" / "a.txt").exists()


def test_worker_loop_survives_bad_record_and_processes_next_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> megaseg_app.JobRecord:
        runtime = megaseg_app.ServiceRuntime(_settings(tmp_path))
        runtime.infer_semaphore = asyncio.Semaphore(1)
        runtime.settings.job_root.mkdir(parents=True, exist_ok=True)
        (runtime.settings.job_root / "bad.json").write_text("{not-json", encoding="utf-8")

        source = tmp_path / "input.tif"
        source.write_bytes(b"input")
        _queued_record(runtime, "good", source)

        monkeypatch.setattr(
            megaseg_app,
            "_get_model_for_request",
            lambda runtime, request: (object(), runtime.settings.checkpoint_path, "cpu"),
        )
        monkeypatch.setattr(megaseg_app, "run_megaseg_batch", lambda **kwargs: {"success": True})

        worker = asyncio.create_task(megaseg_app._worker_loop(_app_for_runtime(runtime)))
        try:
            await runtime.queue.put("bad")
            await runtime.queue.put("good")
            await asyncio.wait_for(runtime.queue.join(), timeout=1.0)
            return megaseg_app._load_job_record(runtime, "good")
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await worker

    record = asyncio.run(scenario())

    assert record.status == "succeeded"


def test_worker_loop_uses_infer_semaphore_for_queued_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> int:
        runtime = megaseg_app.ServiceRuntime(_settings(tmp_path))
        runtime.infer_semaphore = asyncio.Semaphore(1)
        source = tmp_path / "input.tif"
        source.write_bytes(b"input")
        _queued_record(runtime, "queued", source)

        monkeypatch.setattr(
            megaseg_app,
            "_get_model_for_request",
            lambda runtime, request: (object(), runtime.settings.checkpoint_path, "cpu"),
        )

        lock = threading.Lock()
        entered = threading.Event()
        active = 0
        max_active = 0

        def fake_run_megaseg_batch(**kwargs):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
                entered.set()
            try:
                time.sleep(0.25)
                output_dir = Path(kwargs["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)
                return {"success": True}
            finally:
                with lock:
                    active -= 1

        monkeypatch.setattr(megaseg_app, "run_megaseg_batch", fake_run_megaseg_batch)

        async def infer_call() -> None:
            async with runtime.infer_semaphore:
                await asyncio.to_thread(
                    megaseg_app._run_infer_sync,
                    runtime,
                    megaseg_app.JobRequest(),
                    source,
                    tmp_path / "infer-results",
                )

        worker = asyncio.create_task(megaseg_app._worker_loop(_app_for_runtime(runtime)))
        infer_task = asyncio.create_task(infer_call())
        try:
            assert await asyncio.to_thread(entered.wait, 1.0)
            await runtime.queue.put("queued")
            await asyncio.wait_for(runtime.queue.join(), timeout=2.0)
            await asyncio.wait_for(infer_task, timeout=2.0)
            return max_active
        finally:
            worker.cancel()
            infer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await worker
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await infer_task

    assert asyncio.run(scenario()) == 1
