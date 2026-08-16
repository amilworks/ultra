"""Tier-2 chaos harness: the real worker on a real JetStream, faults injected.

Tier 1 proved the run-loop invariants by driving ``run_job`` directly. Tier 2
moves the trust boundary out one ring: jobs flow through a REAL NATS JetStream
(dockerized, per-session) into the REAL ``NATSDeepAgentsWorker`` consume loop —
durable pull consumer, ack/NAK/ack-extension, duplicate-delivery guards, the
shared run sequencer, worker terminal events — while the model and sandbox stay
the deterministic Tier-1 fakes.

What plays the control plane: ``ChaosControlPlane``, a thin in-test stand-in
wired through the worker's own constructor injection points (`run_status_func`,
`run_lease_func`, ...). Its status answers are DERIVED from the event stream
the same way the Go control plane's are (terminal event seen => terminal
status), which is exactly the cooperation the duplicate-redelivery contract
depends on. The durable checkpoint "database" is the Tier-1 in-memory store
shared across worker instances; the resume sequence floor is served from the
collector's view of the stream, mirroring the control plane's persisted-events
query.

Worker death is simulated at the same boundary production uses: cancelling the
serve task trips the worker's own shutdown path (``_shutting_down`` → NAK →
JetStream redelivery), and a second, fresh worker instance — new process
memory, same durable store — picks the run up.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import shutil
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import nats
from longhorizon_harness import (
    EventLog,
    ScriptedChatModel,
    ScriptedSandbox,
    TurnPolicy,
)
from ultra_deepagents.agent import build_research_agent
from ultra_deepagents.checkpointing import DurableCheckpointer, InMemoryCheckpointStateStore
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.nats_worker import (
    ControlPlaneRunLease,
    NATSDeepAgentsWorker,
    RunLeaseConflict,
)
from ultra_deepagents.runner import run_job
from ultra_deepagents.schemas import RunJobEnvelope

NATS_IMAGE = "nats:2.10-alpine"
_namespace_counter = itertools.count(1)


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class NatsServerContainer:
    """One dockerized JetStream server per test session; tests isolate via
    per-test stream names and subject namespaces."""

    def __init__(self) -> None:
        self.port = _free_port()
        self.name = f"ultra-chaos-nats-{self.port}"
        self.url = f"nats://127.0.0.1:{self.port}"

    def start(self) -> None:
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                self.name,
                "-p",
                f"{self.port}:4222",
                NATS_IMAGE,
                "-js",
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )

    def stop(self) -> None:
        subprocess.run(
            ["docker", "rm", "-f", self.name],
            check=False,
            capture_output=True,
            timeout=30,
        )

    async def restart(self) -> None:
        """Bounce the server process. ``docker restart`` keeps the container
        filesystem, so JetStream streams/consumers/unacked state survive —
        the same shape as a NATS node reboot in production."""
        await asyncio.to_thread(
            subprocess.run,
            ["docker", "restart", self.name],
            check=True,
            capture_output=True,
            timeout=60,
        )
        await self.wait_ready()

    async def wait_ready(self, *, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                connection = await nats.connect(self.url, connect_timeout=1)
            except Exception as exc:  # noqa: BLE001 - retry any startup error
                last_error = exc
                await asyncio.sleep(0.15)
            else:
                await connection.close()
                return
        raise RuntimeError(f"NATS container {self.name} never became ready: {last_error!r}")


@dataclass
class ChaosNamespace:
    """Per-test isolation on the shared server: unique stream + subject tree +
    durable, so overlapping-subject stream conflicts are impossible."""

    stream: str
    jobs_subject: str
    events_subject: str
    cancel_subject: str
    durable: str

    @classmethod
    def fresh(cls) -> ChaosNamespace:
        index = next(_namespace_counter)
        token = f"{index}{uuid.uuid4().hex[:6]}"
        return cls(
            stream=f"ULTRA_CHAOS_{token}".upper(),
            jobs_subject=f"chaos{token}.runs.jobs",
            events_subject=f"chaos{token}.runs.events",
            cancel_subject=f"chaos{token}.runs.cancel",
            durable=f"chaos-worker-{token}",
        )


def chaos_settings(
    tmp: Path,
    server: NatsServerContainer,
    namespace: ChaosNamespace,
    **overrides: Any,
) -> RuntimeSettings:
    uploads = tmp / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    values: dict[str, Any] = dict(
        openai_base_url="http://scripted.invalid/v1",
        openai_model="scripted-longhorizon",
        model_max_input_tokens=60_000,
        workspace_root=str(tmp / "workspaces"),
        artifact_root=str(tmp / "artifacts"),
        memory_root=str(tmp / "memory"),
        rarespot_upload_roots=(str(uploads),),
        title_generation_enabled=False,
        nats_url=server.url,
        nats_stream=namespace.stream,
        nats_jobs_subject=namespace.jobs_subject,
        nats_events_subject=namespace.events_subject,
        nats_cancel_subject=namespace.cancel_subject,
        nats_worker_durable=namespace.durable,
        worker_max_concurrency=4,
        # No periodic docker sweeps during tests (the startup sweep still runs
        # and is best-effort).
        sandbox_reaper_interval_seconds=0,
    )
    values.update(overrides)
    return RuntimeSettings(**values)


@dataclass
class ChaosWorld:
    """Shared durable world across worker instances (duck-typed for
    ScriptedChatModel: provides ``invocation`` and ``model_calls``)."""

    tmp: Path
    sandbox: ScriptedSandbox = field(default_factory=ScriptedSandbox)
    store: InMemoryCheckpointStateStore = field(default_factory=InMemoryCheckpointStateStore)
    model_calls: list[Any] = field(default_factory=list)
    invocation: int = 1


class EventCollector:
    """The test's view of the event stream — the same view the control plane
    ingests. Serves both assertions and the worker's resume-floor query."""

    def __init__(self, url: str, namespace: ChaosNamespace) -> None:
        self._url = url
        self._namespace = namespace
        self._connection: Any = None
        self._js: Any = None
        self._subscription: Any = None
        self._task: asyncio.Task | None = None
        self._seen_event_ids: set[str] = set()
        self.events: list[dict[str, Any]] = []

    async def start(self) -> None:
        self._connection = await nats.connect(self._url)
        js = self._connection.jetstream()
        # The worker provisions the stream; make collection order-independent.
        try:
            await js.add_stream(
                name=self._namespace.stream,
                subjects=[
                    self._namespace.jobs_subject,
                    self._namespace.events_subject,
                    f"{self._namespace.events_subject}.>",
                    self._namespace.cancel_subject,
                ],
            )
        except Exception as exc:
            if "stream name already in use" not in str(exc).lower():
                raise
        self._js = js
        await self._open_subscription()

    async def _open_subscription(self) -> None:
        self._subscription = await self._js.subscribe(
            f"{self._namespace.events_subject}.>",
            stream=self._namespace.stream,
            ordered_consumer=True,
        )
        self._task = asyncio.create_task(self._pump(self._subscription))

    async def resync(self) -> None:
        """Re-read the stream from the beginning on a fresh ordered consumer.

        A server restart destroys ephemeral ordered consumers; the replacement
        replays the full history, and event_id dedupe makes the replay additive
        (``events`` is never cleared, so the worker's resume-floor query never
        observes a transiently empty view)."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        with contextlib.suppress(Exception):
            await self._subscription.unsubscribe()
        await self._open_subscription()

    async def _pump(self, subscription: Any) -> None:
        while True:
            try:
                message = await subscription.next_msg(timeout=0.5)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                raise
            try:
                event = json.loads(message.data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            event_id = str(event.get("event_id") or "")
            if event_id and event_id in self._seen_event_ids:
                continue
            if event_id:
                self._seen_event_ids.add(event_id)
            self.events.append(event)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._connection is not None:
            with contextlib.suppress(Exception):
                await self._connection.close()

    def for_run(self, run_id: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event.get("run_id") == run_id]

    def max_sequence(self, run_id: str) -> int:
        return max(
            (int(event.get("sequence") or 0) for event in self.for_run(run_id)),
            default=0,
        )

    def kinds(self, run_id: str) -> list[str]:
        return [str(event.get("event_kind")) for event in self.for_run(run_id)]

    def of_kind(self, run_id: str, kind: str) -> list[dict[str, Any]]:
        return [event for event in self.for_run(run_id) if event.get("event_kind") == kind]

    def terminal_kind(self, run_id: str) -> str | None:
        for event in self.for_run(run_id):
            kind = str(event.get("event_kind") or "")
            if kind in {"run.completed", "run.failed", "run.canceled", "run.worker_skipped"}:
                return kind
        return None

    def to_event_log(self, run_id: str) -> EventLog:
        """Adapt to the Tier-1 EventLog so Tier-1 invariants apply verbatim."""
        log = EventLog()
        log.events = list(self.for_run(run_id))
        return log

    async def wait_for(
        self,
        predicate,
        *,
        timeout: float,
        description: str,
    ) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate(self):
                return
            await asyncio.sleep(0.05)
        raise AssertionError(
            f"timed out after {timeout:g}s waiting for {description}; "
            f"observed events: {[(e.get('run_id'), e.get('event_kind')) for e in self.events]}"
        )


class ChaosControlPlane:
    """Stand-in for the Go control plane, answering through the worker's own
    injection seams. Status is derived from the ingested event stream exactly
    like production: a terminal event makes the run non-runnable, which is what
    stops a late redelivery from re-executing a completed run.

    Lease chaos: with ``grant_leases=True`` every delivery gets a real
    ``ControlPlaneRunLease`` (a new tenure per grant) and the keepalive thread's
    renewals go through ``renew_lease_sync``. Arm ``conflict_after_renewals`` to
    make the FIRST tenure's Nth renewal raise ``RunLeaseConflict`` — the
    authoritative "the run was handed elsewhere" signal, and the only kill a
    worker trusts."""

    def __init__(self, collector: EventCollector, *, grant_leases: bool = False) -> None:
        self._collector = collector
        self._grant_leases = grant_leases
        self.status_overrides: dict[str, str] = {}
        self.lease_tenures = 0
        self.lease_renewals = 0
        self.conflict_after_renewals: int | None = None
        self.renewal_conflict_gate: Callable[[], bool] | None = None

    async def run_status(self, run_id: str, settings: RuntimeSettings) -> str | None:
        del settings
        if run_id in self.status_overrides:
            return self.status_overrides[run_id]
        terminal = self._collector.terminal_kind(run_id)
        if terminal is None:
            return None
        # The control plane's status vocabulary (TERMINAL_CONTROL_PLANE_RUN_
        # STATUSES): a completed run reads "succeeded", NOT "completed" — using
        # the wrong word here silently disables the worker's skip guard and a
        # post-completion redelivery re-enters compute.
        return {
            "run.completed": "succeeded",
            "run.failed": "failed",
            "run.canceled": "canceled",
            "run.worker_skipped": "canceled",
        }[terminal]

    async def run_lease(self, run_id: str, settings: RuntimeSettings):
        if not self._grant_leases:
            return None
        self.lease_tenures += 1
        return ControlPlaneRunLease(
            run_id=run_id,
            worker_id=settings.worker_id,
            lease_token=f"lease-{self.lease_tenures}",
        )

    async def release_lease(self, lease: Any, settings: RuntimeSettings) -> None:
        del lease, settings

    def renew_lease_sync(self, lease: Any, settings: RuntimeSettings):
        # Runs on the _LeaseKeepalive thread, mirroring the sync HTTP renewal.
        del settings
        self.lease_renewals += 1
        if self.conflict_after_renewals is not None and (
            self.renewal_conflict_gate is None or self.renewal_conflict_gate()
        ):
            self.conflict_after_renewals -= 1
            if self.conflict_after_renewals < 0:
                self.conflict_after_renewals = None  # one-shot: later tenures renew fine
                raise RunLeaseConflict("scripted 409: run handed to another worker")
        return lease

    async def worker_heartbeat(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    async def user_profile(self, run_id: str, settings: RuntimeSettings) -> None:
        del run_id, settings
        return None


class ChaosWorker(NATSDeepAgentsWorker):
    """The production worker with its injectable edges pointed at the harness:
    scripted model via a real ``build_research_agent`` factory, the shared
    in-memory durable store, and the collector-backed resume floor."""

    def __init__(
        self,
        settings: RuntimeSettings,
        *,
        world: ChaosWorld,
        collector: EventCollector,
        control_plane: ChaosControlPlane,
        policy: TurnPolicy,
    ) -> None:
        self._world = world
        self._collector = collector
        model = ScriptedChatModel(
            policy=policy,
            world=world,
            window_tokens=settings.model_max_input_tokens,
            profile={"max_input_tokens": settings.model_max_input_tokens},
        )

        async def run_job_with_scripted_agent(
            job: RunJobEnvelope,
            job_settings: RuntimeSettings,
            **kwargs: Any,
        ) -> str:
            def factory(factory_settings: RuntimeSettings, **factory_kwargs: Any) -> Any:
                return build_research_agent(factory_settings, model=model, **factory_kwargs)

            return await run_job(
                job,
                job_settings,
                agent_factory=factory,
                **kwargs,
            )

        super().__init__(
            settings,
            run_job_func=run_job_with_scripted_agent,
            run_status_func=control_plane.run_status,
            run_lease_func=control_plane.run_lease,
            lease_renew_sync_func=control_plane.renew_lease_sync,
            release_run_lease_func=control_plane.release_lease,
            worker_heartbeat_func=control_plane.worker_heartbeat,
            user_profile_func=control_plane.user_profile,
        )

    async def _ensure_checkpointer(self) -> Any:
        # Production gates this on the control-plane Postgres DSN; the chaos rig
        # substitutes the shared in-memory store. Zero debounce so every
        # checkpoint write is durable the moment the graph makes it — the
        # deterministic worst-case-honest resume point.
        if self._checkpointer is None:
            self._checkpointer = DurableCheckpointer(
                self._world.store, persist_debounce_seconds=0.0
            )
        return self._checkpointer

    async def _run_events_snapshot(self, run_id: str) -> tuple[int, dict[str, Any] | None]:
        # Production pages the run's persisted events out of the control plane;
        # the collector holds the same view here.
        return self._collector.max_sequence(run_id), None


async def start_worker(worker: ChaosWorker) -> asyncio.Task:
    task = asyncio.create_task(worker._serve_one_connection())
    # Give the worker one beat to connect and provision before callers publish.
    await asyncio.sleep(0.3)
    return task


async def stop_worker(task: asyncio.Task) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


async def publish_job(
    server: NatsServerContainer,
    settings: RuntimeSettings,
    job: RunJobEnvelope,
    *,
    copies: int = 1,
) -> None:
    connection = await nats.connect(server.url)
    try:
        js = connection.jetstream()
        payload = json.dumps(
            {
                "run_id": job.run_id,
                "thread_id": job.thread_id,
                "user_id": job.user_id,
                "goal": job.goal,
                "messages": job.messages,
            }
        ).encode("utf-8")
        for _ in range(copies):
            await js.publish(settings.nats_jobs_subject, payload)
    finally:
        await connection.close()


async def consumer_state(
    server: NatsServerContainer,
    settings: RuntimeSettings,
) -> tuple[int, int]:
    """(num_ack_pending, num_redelivered) for the worker's durable consumer."""
    connection = await nats.connect(server.url)
    try:
        js = connection.jetstream()
        info = await js.consumer_info(settings.nats_stream, settings.nats_worker_durable)
        return int(info.num_ack_pending), int(info.num_redelivered)
    finally:
        await connection.close()
