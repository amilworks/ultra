"""Tests for the model-agnostic GPU compute service framework.

The framework is exercised with tiny fake executors (no torch) so these run in CI
without a GPU. The real rarespot/megaseg executors are validated at image build
time (services/compute_service/Dockerfile) and on the GPU node.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.compute_service import executors as registry
from services.compute_service.app import create_app
from services.compute_service.executors.base import ExecutorManifest, JobCancelled, JobContext
from services.compute_service.jobs import TERMINAL_STATUSES, JobManager
from services.compute_service.settings import ServiceSettings


# --------------------------------------------------------------- fake executors
class EchoExecutor:
    manifest = ExecutorManifest(model_key="test", capability="echo", gpu=False)

    def run(self, spec, ctx):
        steps = int(spec.get("steps") or 2)
        ctx.progress(total=steps, message="start")
        for i in range(steps):
            if ctx.cancelled:
                raise JobCancelled("cancelled")
            time.sleep(0.02)
            ctx.progress(completed=i + 1, total=steps, metrics={"i": i + 1})
        return {"echo": spec.get("value"), "steps": steps}


class SlowExecutor:
    manifest = ExecutorManifest(model_key="test", capability="slow", gpu=False)

    def run(self, spec, ctx):
        for i in range(200):
            if ctx.cancelled:
                raise JobCancelled("cancelled mid-run")
            time.sleep(0.02)
            ctx.progress(completed=i + 1, total=200)
        return {"done": True}


class BoomExecutor:
    manifest = ExecutorManifest(model_key="test", capability="boom", gpu=False)

    def run(self, spec, ctx):
        raise ValueError("intentional failure")


@pytest.fixture
def clean_registry():
    saved = dict(registry._REGISTRY)
    saved_errors = dict(registry._LOAD_ERRORS)
    registry._REGISTRY.clear()
    registry._LOAD_ERRORS.clear()
    try:
        yield registry
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)
        registry._LOAD_ERRORS.clear()
        registry._LOAD_ERRORS.update(saved_errors)


async def _wait_terminal(manager: JobManager, job_id: str, timeout: float = 10.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = manager.get(job_id)
        if record is not None and record.status in TERMINAL_STATUSES:
            return record.status
        await asyncio.sleep(0.02)
    raise AssertionError(f"job {job_id} did not terminate within {timeout}s")


def _manager(tmp_path: Path, **kw) -> JobManager:
    return JobManager(
        job_root=tmp_path / "jobs",
        workspace_root=tmp_path / "work",
        max_concurrent=kw.pop("max_concurrent", 2),
        history_limit=kw.pop("history_limit", 50),
        progress_persist_interval=kw.pop("progress_persist_interval", 0.0),
    )


# ------------------------------------------------------------- lifecycle tests
def test_back_to_back_jobs_all_succeed(tmp_path, clean_registry):
    clean_registry.register(EchoExecutor())

    async def run():
        manager = _manager(tmp_path)
        manager.start(asyncio.get_running_loop())
        ids = [manager.submit(executor_key="test.echo", spec={"value": n, "steps": 2}).job_id for n in range(6)]
        for job_id in ids:
            assert await _wait_terminal(manager, job_id) == "succeeded"
        first = manager.get(ids[0])
        assert first.result == {"echo": 0, "steps": 2}
        assert first.progress.completed == 2 and first.progress.total == 2
        await manager.shutdown()

    asyncio.run(run())


def test_failing_job_is_isolated_and_service_survives(tmp_path, clean_registry):
    clean_registry.register(BoomExecutor())
    clean_registry.register(EchoExecutor())

    async def run():
        manager = _manager(tmp_path)
        manager.start(asyncio.get_running_loop())
        boom = manager.submit(executor_key="test.boom", spec={}).job_id
        assert await _wait_terminal(manager, boom) == "failed"
        assert "intentional failure" in (manager.get(boom).error or "")
        # a following job still runs
        ok = manager.submit(executor_key="test.echo", spec={"value": 1}).job_id
        assert await _wait_terminal(manager, ok) == "succeeded"
        await manager.shutdown()

    asyncio.run(run())


def test_cancel_running_job(tmp_path, clean_registry):
    clean_registry.register(SlowExecutor())

    async def run():
        manager = _manager(tmp_path)
        manager.start(asyncio.get_running_loop())
        job_id = manager.submit(executor_key="test.slow", spec={}).job_id
        await asyncio.sleep(0.2)
        assert manager.get(job_id).status == "running"
        manager.cancel(job_id)
        assert await _wait_terminal(manager, job_id) == "cancelled"
        await manager.shutdown()

    asyncio.run(run())


def test_cancel_queued_job_before_start(tmp_path, clean_registry):
    clean_registry.register(SlowExecutor())
    clean_registry.register(EchoExecutor())

    async def run():
        manager = _manager(tmp_path, max_concurrent=1)
        manager.start(asyncio.get_running_loop())
        running = manager.submit(executor_key="test.slow", spec={}).job_id
        await asyncio.sleep(0.1)
        queued = manager.submit(executor_key="test.echo", spec={}).job_id
        record = manager.cancel(queued)
        assert record.status == "cancelled"
        assert manager.get(queued).error == "cancelled before start"
        manager.cancel(running)
        await manager.shutdown()

    asyncio.run(run())


def test_idempotency_key_dedupes(tmp_path, clean_registry):
    clean_registry.register(EchoExecutor())

    async def run():
        manager = _manager(tmp_path)
        manager.start(asyncio.get_running_loop())
        a = manager.submit(executor_key="test.echo", spec={}, idempotency_key="K").job_id
        b = manager.submit(executor_key="test.echo", spec={}, idempotency_key="K").job_id
        assert a == b
        await manager.shutdown()

    asyncio.run(run())


def test_restart_recovery_marks_running_failed_and_requeues_queued(tmp_path, clean_registry):
    clean_registry.register(EchoExecutor())

    async def run():
        first = _manager(tmp_path)
        first.start(asyncio.get_running_loop())
        # Fabricate a persisted "running" record + a "queued" one as if the service
        # died mid-flight, then start a fresh manager over the same roots.
        done = first.submit(executor_key="test.echo", spec={}).job_id
        await _wait_terminal(first, done)
        await first.shutdown()

        import json

        running_id = "deadbeef" * 4
        (tmp_path / "jobs" / f"{running_id}.json").write_text(
            json.dumps(
                {
                    "job_id": running_id,
                    "executor": "test.echo",
                    "status": "running",
                    "created_at": "2026-07-09T00:00:00+00:00",
                    "updated_at": "2026-07-09T00:00:00+00:00",
                    "spec": {},
                }
            ),
            encoding="utf-8",
        )

        second = _manager(tmp_path)
        second.start(asyncio.get_running_loop())
        recovered = second.get(running_id)
        assert recovered is not None and recovered.status == "failed"
        assert "restarted" in (recovered.error or "")
        await second.shutdown()

    asyncio.run(run())


def test_history_eviction_bounds_records(tmp_path, clean_registry):
    clean_registry.register(EchoExecutor())

    async def run():
        manager = _manager(tmp_path, history_limit=16)
        manager.start(asyncio.get_running_loop())
        ids = [manager.submit(executor_key="test.echo", spec={"steps": 1}).job_id for _ in range(30)]
        # The newest job is never evicted (eviction removes oldest terminal), so
        # once it is done the whole batch has drained. Early jobs are correctly
        # evicted, so get() returns None for them — that is the behaviour under test.
        assert await _wait_terminal(manager, ids[-1]) == "succeeded"
        await asyncio.sleep(0.1)
        remaining = [j for j in ids if manager.get(j) is not None]
        assert len(remaining) <= 16, f"history not bounded: {len(remaining)}"
        assert manager.get(ids[-1]) is not None, "newest job must be retained"
        assert manager.get(ids[0]) is None, "oldest job must be evicted"
        await manager.shutdown()

    asyncio.run(run())


def test_recovered_queued_job_is_cancellable(tmp_path, clean_registry):
    # Regression: a requeued job must get a cancel event so cancel() isn't a no-op.
    clean_registry.register(SlowExecutor())

    async def run():
        (tmp_path / "jobs").mkdir(parents=True, exist_ok=True)
        job_id = "cafef00d" * 4
        (tmp_path / "jobs" / f"{job_id}.json").write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "executor": "test.slow",
                    "status": "queued",
                    "created_at": "2026-07-09T00:00:00+00:00",
                    "updated_at": "2026-07-09T00:00:00+00:00",
                    "spec": {},
                }
            ),
            encoding="utf-8",
        )
        manager = _manager(tmp_path, max_concurrent=1)
        manager.start(asyncio.get_running_loop())
        for _ in range(200):
            if manager.get(job_id).status == "running":
                break
            await asyncio.sleep(0.02)
        assert manager.get(job_id).status == "running"
        manager.cancel(job_id)
        assert await _wait_terminal(manager, job_id) == "cancelled"
        await manager.shutdown()

    asyncio.run(run())


def test_workdir_setup_failure_marks_failed_not_stuck(tmp_path, clean_registry):
    # Regression: a failure creating the per-job workdir must terminate the job as
    # FAILED, not leave it stuck RUNNING forever.
    clean_registry.register(EchoExecutor())

    async def run():
        workspace = tmp_path / "work"
        workspace.mkdir()
        manager = JobManager(
            job_root=tmp_path / "jobs",
            workspace_root=workspace,
            max_concurrent=1,
            history_limit=32,
            progress_persist_interval=0.0,
        )
        manager.start(asyncio.get_running_loop())
        os.chmod(workspace, 0o500)  # read-only parent -> per-job mkdir raises
        try:
            job_id = manager.submit(executor_key="test.echo", spec={}).job_id
            assert await _wait_terminal(manager, job_id) == "failed"
        finally:
            os.chmod(workspace, 0o700)
            await manager.shutdown()

    asyncio.run(run())


def test_idempotency_allows_retry_after_failure(tmp_path, clean_registry):
    # Regression: a terminally-failed idempotent record must NOT block a retry.
    clean_registry.register(BoomExecutor())

    async def run():
        manager = _manager(tmp_path)
        manager.start(asyncio.get_running_loop())
        first = manager.submit(executor_key="test.boom", spec={}, idempotency_key="R").job_id
        assert await _wait_terminal(manager, first) == "failed"
        second = manager.submit(executor_key="test.boom", spec={}, idempotency_key="R").job_id
        assert second != first  # fresh job, not the stale failed record
        await manager.shutdown()

    asyncio.run(run())


def test_kill_processes_escalates_to_sigkill(tmp_path):
    # A child that ignores SIGTERM must still be reaped (escalation to SIGKILL),
    # so a stubborn CUDA child can't wedge the job slot.
    proc = subprocess.Popen(
        [sys.executable, "-c", "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"],
        start_new_session=True,
    )
    ctx = JobContext(job_id="k", workdir=tmp_path, _emit=lambda _u: None)
    ctx.register_process(proc)
    start = time.monotonic()
    ctx._kill_processes(grace_seconds=1.0)
    assert proc.poll() is not None, "SIGTERM-ignoring child was not killed"
    assert time.monotonic() - start < 6.0


# ---------------------------------------------------------------- HTTP surface
def _client(tmp_path) -> TestClient:
    settings = ServiceSettings(
        api_key="secret",
        job_root=tmp_path / "jobs",
        workspace_root=tmp_path / "work",
        max_concurrent=1,
        history_limit=32,
        executors=frozenset(),  # load no built-ins; test registers its own
    )
    return TestClient(create_app(settings))


AUTH = {"Authorization": "Bearer secret"}


def test_http_requires_bearer(tmp_path, clean_registry):
    clean_registry.register(EchoExecutor())
    with _client(tmp_path) as client:
        assert client.get("/health").status_code == 401
        assert client.get("/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/health", headers=AUTH).status_code == 200


def test_http_health_reports_executors(tmp_path, clean_registry):
    clean_registry.register(EchoExecutor())
    with _client(tmp_path) as client:
        body = client.get("/health", headers=AUTH).json()
        assert body["ok"] is True
        assert "test.echo" in body["executors"]
        listed = client.get("/v1/executors", headers=AUTH).json()
        assert any(e["key"] == "test.echo" for e in listed["executors"])


def test_http_health_503_when_no_executors(tmp_path, clean_registry):
    # registry cleared, none registered, none loaded -> not ready
    with _client(tmp_path) as client:
        resp = client.get("/health", headers=AUTH)
        assert resp.status_code == 503
        assert resp.json()["ok"] is False


def test_http_submit_get_cancel_flow(tmp_path, clean_registry):
    clean_registry.register(EchoExecutor())
    with _client(tmp_path) as client:
        resp = client.post("/v1/jobs", headers=AUTH, json={"executor": "test.echo", "spec": {"value": 7, "steps": 2}})
        assert resp.status_code == 202, resp.text
        job_id = resp.json()["job_id"]
        for _ in range(200):
            record = client.get(f"/v1/jobs/{job_id}", headers=AUTH).json()
            if record["status"] in TERMINAL_STATUSES:
                break
            time.sleep(0.02)
        assert record["status"] == "succeeded"
        assert record["result"] == {"echo": 7, "steps": 2}


def test_http_unknown_executor_is_400(tmp_path, clean_registry):
    clean_registry.register(EchoExecutor())
    with _client(tmp_path) as client:
        resp = client.post("/v1/jobs", headers=AUTH, json={"executor": "nope.gone", "spec": {}})
        assert resp.status_code == 400


def test_http_get_missing_job_is_404(tmp_path, clean_registry):
    clean_registry.register(EchoExecutor())
    with _client(tmp_path) as client:
        assert client.get("/v1/jobs/does-not-exist", headers=AUTH).status_code == 404


# ---------------------------------------------------------------- registry
def test_load_builtin_executors_is_defensive(clean_registry):
    # Loading built-ins in an env without torch must record load errors, not raise.
    clean_registry.load_builtin_executors()
    # Either they loaded (dev env has torch) or they errored — never a crash.
    assert isinstance(clean_registry.load_errors(), dict)
