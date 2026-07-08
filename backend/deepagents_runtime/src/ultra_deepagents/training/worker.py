"""GoldGate NATS training worker (plan sections 4.6, 9.2).

A sibling consumer of the data-agent worker — same envelope/consumer/lease
supervision shape — with the load-bearing differences the plan mandates:

- **Five generic phase branches** ``training.{sync, assemble, finetune,
  benchmark, gold_freeze}``; each resolves the adapter from the registry by the
  envelope's ``model_key`` and hard-fails (``no adapter registered``) BEFORE
  leasing work it can't do. No model-named branches, ever.
- **OS-thread lease+ack keepalive** (keepalive.py, ported from nats_worker's
  ``_LeaseKeepalive``): lease renewal AND ``message.in_progress()`` fire from a
  wall-clock thread immune to event-loop stalls — mandatory for multi-hour
  runs. ``ack_wait``/lease-TTL default to 3600s (well above the data-agent's
  300/600s, per plan 9.2).
- **Every processor step offloads via ``asyncio.to_thread``** — never a raw
  blocking call on the loop (belt-and-suspenders next to the keepalive).
- **SIGTERM drains until idle**: the signal handler sets a stop flag, in-flight
  jobs run to their ``finally`` (kill children, release lease), then the worker
  unsubscribes. Never SIGKILL mid-train.

Entry point: ``python -m ultra_deepagents.training.worker``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import signal
import socket
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import nats
from nats.js.api import AckPolicy, ConsumerConfig

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.nats_worker import (
    WorkerHeartbeatFunc,
    fetch_job_messages,
    post_control_plane_worker_heartbeat,
)
from ultra_deepagents.training.adapters import (
    MaterializeContext,
    ModelAdapter,
    NoAdapterRegistered,
    TrainContext,
    get_adapter,
)
from ultra_deepagents.training.control_client import (
    TrainingControlClient,
    TrainingLeaseConflict,
    TrainingLeaseUnavailable,
)
from ultra_deepagents.training.envelope import TrainingJobEnvelope
from ultra_deepagents.training.keepalive import TrainingLeaseKeepalive
from ultra_deepagents.training.manifests import MANIFESTS, ModelManifest

logger = logging.getLogger(__name__)

TERMINAL_TRAINING_JOB_STATUSES = {"succeeded", "failed", "canceled"}


@dataclass(frozen=True)
class TrainingWorkerSettings:
    """Training-worker knobs, composed over RuntimeSettings (control base URL,
    worker token, and the shared NATS fallbacks come from there)."""

    runtime: RuntimeSettings
    nats_url: str = "nats://127.0.0.1:4222"
    stream: str = "ULTRA_RUNS"
    jobs_subject: str = "ultra.training.jobs"
    durable: str = "ultra-training-worker"
    ack_wait_seconds: float = 3600.0
    lease_ttl_seconds: float = 3600.0
    redelivery_delay_seconds: float = 30.0
    worker_id: str = "ultra-training-worker"
    worker_kind: str = "training"
    max_deliver: int = 5
    max_concurrency: int = 1
    workdir_root: str = ""

    @classmethod
    def from_env(cls, runtime: RuntimeSettings | None = None) -> TrainingWorkerSettings:
        runtime = runtime or RuntimeSettings.from_env()
        return cls(
            runtime=runtime,
            nats_url=os.getenv("ULTRA_TRAINING_NATS_URL", runtime.data_agent_nats_url),
            stream=os.getenv("ULTRA_CONTROL_NATS_TRAINING_STREAM", runtime.data_agent_nats_stream),
            jobs_subject=os.getenv(
                "ULTRA_CONTROL_NATS_TRAINING_JOBS_SUBJECT", "ultra.training.jobs"
            ),
            durable=os.getenv(
                "ULTRA_CONTROL_NATS_TRAINING_WORKER_DURABLE", "ultra-training-worker"
            ),
            # Plan 9.2: well above the data-agent's 300/600s defaults — a
            # multi-hour epoch must never race the redelivery clock even if the
            # keepalive misses ticks.
            ack_wait_seconds=max(
                1.0, float(os.getenv("ULTRA_TRAINING_NATS_ACK_WAIT_SECONDS", "3600"))
            ),
            lease_ttl_seconds=max(
                1.0, float(os.getenv("ULTRA_TRAINING_CONTROL_LEASE_TTL_SECONDS", "3600"))
            ),
            redelivery_delay_seconds=max(
                0.0, float(os.getenv("ULTRA_TRAINING_REDELIVERY_DELAY_SECONDS", "30"))
            ),
            worker_id=os.getenv(
                "ULTRA_TRAINING_WORKER_ID",
                f"ultra-training-worker@{socket.gethostname()}:{os.getpid()}",
            ),
            worker_kind=os.getenv("ULTRA_TRAINING_WORKER_KIND", "training"),
            max_deliver=max(1, int(os.getenv("ULTRA_TRAINING_WORKER_MAX_DELIVER", "5"))),
            max_concurrency=max(1, int(os.getenv("ULTRA_TRAINING_WORKER_MAX_CONCURRENCY", "1"))),
            workdir_root=os.getenv("ULTRA_TRAINING_WORKDIR_ROOT", "").strip(),
        )


def build_training_consumer_config(settings: TrainingWorkerSettings) -> ConsumerConfig:
    return ConsumerConfig(
        durable_name=settings.durable,
        filter_subject=settings.jobs_subject,
        ack_policy=AckPolicy.EXPLICIT,
        ack_wait=settings.ack_wait_seconds,
        max_deliver=settings.max_deliver,
        max_ack_pending=settings.max_concurrency,
    )


def compute_gold_content_hash(items: list[dict[str, Any]]) -> str:
    """SHA256 over the sorted (content_sha256, gt_label_sha256) pairs — the
    canonical-benchmark identity: identical bytes+labels, identically for boxes
    and masks; any drift changes the hash (plan 4.5)."""
    pairs = sorted(
        (str(item.get("content_sha256") or ""), str(item.get("gt_label_sha256") or ""))
        for item in items
    )
    digest = hashlib.sha256()
    for content_sha, label_sha in pairs:
        digest.update(content_sha.encode("utf-8"))
        digest.update(b":")
        digest.update(label_sha.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def run_generic_leakage_defenses(
    manifest: ModelManifest,
    train_pool: list[dict[str, Any]],
    gold_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The always-on generic defenses (plan 4.4): source-id disjointness,
    content-hash disjointness, and the structural requires_phash NULL assertion
    (a buggy materializer must not silently skip the near-dup defense)."""
    violations: list[dict[str, Any]] = []
    train_ids = {_primary_source_id(row) for row in train_pool or []} - {""}
    train_shas = {str(row.get("content_sha256") or "") for row in train_pool or []} - {""}
    for item in gold_items:
        item_id = str(item.get("item_id") or "")
        source_id = _primary_source_id(item)
        if source_id and source_id in train_ids:
            violations.append(
                {
                    "check": "source_id_disjointness",
                    "item_id": item_id,
                    "reason": f"gold source id {source_id!r} appears in the assembled training pool",
                }
            )
        content_sha = str(item.get("content_sha256") or "")
        if content_sha and content_sha in train_shas:
            violations.append(
                {
                    "check": "content_hash_disjointness",
                    "item_id": item_id,
                    "reason": "gold content_sha256 appears in the assembled training pool",
                }
            )
        if manifest.requires_phash and not str(item.get("phash") or "").strip():
            violations.append(
                {
                    "check": "requires_phash",
                    "item_id": item_id,
                    "reason": "manifest requires_phash but the gold item's phash is NULL - "
                    "the near-dup defense cannot be skipped silently",
                }
            )
    return violations


def _primary_source_id(row: dict[str, Any]) -> str:
    source_ref = row.get("source_ref") or {}
    return str(
        source_ref.get("bisque_image_id")
        or source_ref.get("barrel_path")
        or row.get("item_id")
        or ""
    ).strip()


class TrainingJobProcessor:
    """Phase → adapter-hook dispatch (five generic branches). Every step is
    offloaded via ``asyncio.to_thread``; blocking work never rides the loop."""

    def __init__(
        self, settings: TrainingWorkerSettings, client: TrainingControlClient | Any
    ) -> None:
        self._settings = settings
        self._client = client
        self._children: set[Any] = set()

    def register_child(self, process: Any) -> None:
        """Adapters running long-lived subprocesses (the M3 dockerized train.py)
        register Popen handles here so the SIGTERM drain's finally can kill them
        instead of orphaning them (plan 9.2 worker-roll safety)."""
        self._children.add(process)

    def terminate_children(self) -> None:
        for process in list(self._children):
            self._children.discard(process)
            with contextlib.suppress(Exception):
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10.0)
                    except subprocess.TimeoutExpired:
                        process.kill()

    async def process(self, job: TrainingJobEnvelope, adapter: ModelAdapter) -> dict[str, Any]:
        handlers: dict[str, Callable[..., Any]] = {
            "training.sync": self._sync,
            "training.assemble": self._train_phase,
            "training.finetune": self._train_phase,
            "training.benchmark": self._benchmark,
            "training.gold_freeze": self._gold_freeze,
        }
        handler = handlers.get(job.job_type)
        if handler is None:
            return {
                "terminal_status": "failed",
                "error": f"unknown training phase {job.job_type!r}",
            }
        return await handler(job, adapter)

    def _job_workdir(self, job: TrainingJobEnvelope) -> Path:
        configured = str((job.params or {}).get("workdir") or "").strip()
        if configured:
            path = Path(configured)
            path.mkdir(parents=True, exist_ok=True)
            return path
        root = self._settings.workdir_root or tempfile.gettempdir()
        path = Path(root) / f"goldgate-{job.job_type.rsplit('.', 1)[-1]}-{job.job_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------ sync

    async def _sync(self, job: TrainingJobEnvelope, adapter: ModelAdapter) -> dict[str, Any]:
        ctx = MaterializeContext(
            purpose="sync",
            params=dict(job.params),
            workdir=self._job_workdir(job),
            settings=self._settings.runtime,
        )
        manifest = await asyncio.to_thread(adapter.materialize_dataset, ctx)
        items = list(manifest.get("items") or [])
        per_class: dict[str, int] = {}
        total_objects = 0
        for item in items:
            for name, count in (
                (item.get("label_stats") or {}).get("per_class_box_count") or {}
            ).items():
                per_class[str(name)] = per_class.get(str(name), 0) + int(count)
                total_objects += int(count)
        # Gate A counts skeleton: the control plane evaluates the policy row +
        # the fixed active-passes-gold precondition against these (plan 7.1).
        status_payload = {
            "model_key": job.model_key,
            "reviewed_images": len(items),
            "per_class_new_objects": per_class,
            "unsupported_class_counts": dict(manifest.get("unsupported_class_counts") or {}),
            "retrain_gate_counts": {
                "reviewed_images": len(items),
                "total_objects": total_objects,
                "per_class": per_class,
            },
        }
        await asyncio.to_thread(self._client.upsert_model_status, job.job_id, status_payload)
        return {
            "terminal_status": "succeeded",
            "item_count": len(items),
            "total_objects": total_objects,
            "per_class_new_objects": per_class,
            "unsupported_class_counts": dict(manifest.get("unsupported_class_counts") or {}),
        }

    # ----------------------------------------------------------- gold freeze

    async def _gold_freeze(self, job: TrainingJobEnvelope, adapter: ModelAdapter) -> dict[str, Any]:
        manifest = MANIFESTS.get(job.model_key)
        if manifest is None:
            return {
                "terminal_status": "failed",
                "error": f"no ModelManifest registered for model_key={job.model_key}",
            }
        params = dict(job.params)
        items = [dict(item) for item in (params.get("items") or [])]
        if params.get("materialize", not items):
            ctx = MaterializeContext(
                purpose="gold_cut",
                params=params,
                workdir=self._job_workdir(job),
                settings=self._settings.runtime,
            )
            dataset = await asyncio.to_thread(adapter.materialize_dataset, ctx)
            items = list(dataset.get("items") or [])

        assignments = dict(params.get("slice_assignments") or {})
        default_slice = str(params.get("default_slice") or "prior_train")
        for item in items:
            item.setdefault(
                "slice", str(assignments.get(str(item.get("item_id"))) or default_slice)
            )

        train_pool = list(params.get("train_pool") or [])
        violations = run_generic_leakage_defenses(manifest, train_pool, items)

        # Extra-check availability is asserted at freeze EXECUTION, not boot:
        # a manifest name with no implementation fails the freeze (plan 4.4).
        missing = [
            name
            for name in manifest.leakage_defenses_extra
            if name not in adapter.implemented_leakage_checks
        ]
        if missing:
            violations.append(
                {
                    "check": "extra_defense_unresolved",
                    "item_id": "",
                    "reason": f"manifest leakage_defenses_extra {missing} not implemented by the adapter",
                }
            )
        else:
            try:
                extra = await asyncio.to_thread(
                    adapter.extra_leakage_checks, train_pool, items, params=params
                )
            except Exception as exc:
                # A refusal (inventory sha/kernel mismatch, missing config) is
                # fail-closed: the freeze fails with the reason recorded.
                violations.append(
                    {
                        "check": "extra_defense_refused",
                        "item_id": "",
                        "reason": str(exc) or exc.__class__.__name__,
                    }
                )
            else:
                violations.extend(extra or [])

        content_hash = compute_gold_content_hash(items)
        label_stats = _aggregate_gold_label_stats(items)
        gold_status = "failed" if violations else "frozen"
        provenance = dict(params.get("provenance") or {})
        for key in (
            "exclusion_inventory_uri",
            "inventory_sha256",
            "geo_radius_m",
            "held_out_state",
        ):
            if params.get(key) is not None:
                provenance.setdefault(key, params[key])
        result_payload = {
            "status": gold_status,
            "content_hash": content_hash,
            "item_count": len(items),
            "label_stats": label_stats,
            "strata_summary": dict(params.get("strata_summary") or {}),
            "provenance": provenance,
            "split_manifest_uri": str(params.get("split_manifest_uri") or ""),
            "failure_reasons": violations,
            "items": items,
        }
        await asyncio.to_thread(self._client.finalize_gold_set, job.job_id, result_payload)
        summary: dict[str, Any] = {
            "gold_status": gold_status,
            "content_hash": content_hash,
            "item_count": len(items),
            "violation_count": len(violations),
        }
        if violations:
            summary["terminal_status"] = "failed"
            summary["error"] = (
                f"gold freeze failed: {len(violations)} leakage violation(s); "
                f"first: {violations[0].get('check')}: {violations[0].get('reason')}"
            )
        else:
            summary["terminal_status"] = "succeeded"
        return summary

    # ------------------------------------------------------------- benchmark

    async def _benchmark(self, job: TrainingJobEnvelope, adapter: ModelAdapter) -> dict[str, Any]:
        manifest = MANIFESTS.get(job.model_key)
        params = dict(job.params)
        weights_uri = str(params.get("weights_uri") or "").strip()
        if not weights_uri:
            return {"terminal_status": "failed", "error": "benchmark requires params.weights_uri"}
        gold_manifest = params.get("gold_manifest")
        if not isinstance(gold_manifest, dict):
            manifest_uri = str(params.get("gold_manifest_uri") or "").strip()
            if not manifest_uri:
                return {
                    "terminal_status": "failed",
                    "error": "benchmark requires params.gold_manifest or params.gold_manifest_uri",
                }
            gold_manifest = await asyncio.to_thread(_load_json_file, Path(manifest_uri))
        metrics = await asyncio.to_thread(
            adapter.evaluate, weights_uri, gold_manifest, params.get("slice")
        )
        # Guardrails (Gate B) evaluate control-plane side (M2); the worker sends
        # metrics + report only.
        run_payload = {
            "model_version_id": str(params.get("model_version_id") or ""),
            "gold_set_id": str(params.get("gold_set_id") or ""),
            "gold_set_content_hash": str(params.get("gold_set_content_hash") or ""),
            "metric_schema": manifest.metric_schema if manifest else "",
            "kernel_version": str((metrics.get("eval") or {}).get("kernel") or ""),
            "metrics": metrics,
            "report_uri": str(params.get("report_uri") or ""),
        }
        await asyncio.to_thread(self._client.insert_benchmark_run, job.job_id, run_payload)
        return {
            "terminal_status": "succeeded",
            "metric_schema": run_payload["metric_schema"],
            "kernel_version": run_payload["kernel_version"],
            "aggregate": dict(metrics.get("aggregate") or {}),
        }

    # ------------------------------------------------- assemble / finetune (M3)

    async def _train_phase(self, job: TrainingJobEnvelope, adapter: ModelAdapter) -> dict[str, Any]:
        ctx = TrainContext(
            params=dict(job.params),
            workdir=self._job_workdir(job),
            settings=self._settings.runtime,
        )
        try:
            artifact = await asyncio.to_thread(adapter.train, ctx)
        except NotImplementedError as exc:
            # The M3 branch: fail cleanly with the reason, never wedge the job.
            return {"terminal_status": "failed", "error": str(exc) or "trainer not implemented"}
        # Reached once an adapter's trainer exists: smoke-test BEFORE version
        # registration (plan 6.2), then register the candidate.
        weights_uri = str((artifact or {}).get("weights_uri") or "")
        gold_sample = job.params.get("gold_sample")
        if gold_sample is not None:
            await asyncio.to_thread(adapter.smoke_test, weights_uri, gold_sample)
        await asyncio.to_thread(
            self._client.register_model_version,
            job.model_key,
            {
                "status": "candidate",
                "weights_uri": weights_uri,
                "provenance": {
                    "job_id": job.job_id,
                    "job_type": job.job_type,
                    "hyp_sha256": (artifact or {}).get("hyp_sha256"),
                    "data_manifest_sha256": (artifact or {}).get("data_manifest_sha256"),
                },
            },
        )
        return {"terminal_status": "succeeded", "weights_uri": weights_uri}


def _aggregate_gold_label_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    tile_count = 0
    box_count = 0
    per_class: dict[str, int] = {}
    per_slice: dict[str, int] = {}
    for item in items:
        stats = item.get("label_stats") or {}
        tile_count += int(stats.get("tile_count") or 0)
        box_count += int(stats.get("box_count") or 0)
        for name, count in (stats.get("per_class_box_count") or {}).items():
            per_class[str(name)] = per_class.get(str(name), 0) + int(count)
        slice_name = str(item.get("slice") or "")
        if slice_name:
            per_slice[slice_name] = per_slice.get(slice_name, 0) + 1
    return {
        "tile_count": tile_count,
        "box_count": box_count,
        "per_class_box_count": per_class,
        "per_slice_item_count": per_slice,
    }


def _load_json_file(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


ClientFactory = Callable[[TrainingJobEnvelope], Any]


class NATSTrainingWorker:
    def __init__(
        self,
        settings: TrainingWorkerSettings | None = None,
        *,
        client_factory: ClientFactory | None = None,
        processor_factory: Callable[[TrainingWorkerSettings, Any], TrainingJobProcessor]
        | None = None,
        worker_heartbeat_func: WorkerHeartbeatFunc = post_control_plane_worker_heartbeat,
    ) -> None:
        self.settings = settings or TrainingWorkerSettings.from_env()
        self._client_factory = client_factory or self._default_client_factory
        self._processor_factory = processor_factory or TrainingJobProcessor
        self._worker_heartbeat = worker_heartbeat_func
        self._stop_requested = False
        self._shutting_down = False
        self._active_tasks: set[asyncio.Task] = set()

    def _default_client_factory(self, job: TrainingJobEnvelope) -> TrainingControlClient:
        return TrainingControlClient(
            self.settings.runtime,
            worker_id=self.settings.worker_id,
            principal_headers=job.principal_headers(self.settings.runtime),
        )

    def request_stop(self) -> None:
        """SIGTERM drain: stop fetching, let in-flight jobs finish (their
        finally releases the lease and kills children), then unsubscribe."""
        logger.info("Training worker stop requested; draining in-flight jobs.")
        self._stop_requested = True

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.add_signal_handler(signum, self.request_stop)

    async def run_forever(self) -> None:
        self._install_signal_handlers()
        nc = await nats.connect(self.settings.nats_url)
        js = nc.jetstream()
        try:
            await self._ensure_stream(js)
            subscription = await self._subscribe(js)
            max_concurrency = max(1, self.settings.max_concurrency)
            last_idle_heartbeat_at = 0.0
            while not self._stop_requested:
                self._active_tasks = _harvest_finished_tasks(self._active_tasks)
                available = max_concurrency - len(self._active_tasks)
                if available <= 0:
                    self._active_tasks = await _wait_for_task_capacity(
                        self._active_tasks, timeout=0.2
                    )
                    continue
                messages = await fetch_job_messages(subscription, batch=available, timeout=2.0)
                if not messages:
                    if self._active_tasks:
                        self._active_tasks = await _wait_for_task_capacity(
                            self._active_tasks, timeout=0.2
                        )
                    else:
                        last_idle_heartbeat_at = await self._maybe_post_idle_heartbeat(
                            last_idle_heartbeat_at
                        )
                        await asyncio.sleep(0.2)
                    continue
                for message in messages:
                    self._active_tasks.add(asyncio.create_task(self._process_message(message)))
            # Drain-until-idle: in-flight jobs finish; never cancel mid-train.
            if self._active_tasks:
                logger.info(
                    "Training worker draining %d in-flight job(s) before shutdown.",
                    len(self._active_tasks),
                )
                await asyncio.gather(*self._active_tasks, return_exceptions=True)
            with contextlib.suppress(Exception):
                await subscription.unsubscribe()
        finally:
            self._shutting_down = True
            pending = {task for task in self._active_tasks if not task.done()}
            if pending:
                await _cancel_tasks(pending)
            await nc.drain()

    async def _ensure_stream(self, js: Any) -> None:
        try:
            await js.add_stream(name=self.settings.stream, subjects=[self.settings.jobs_subject])
        except Exception as exc:
            if "stream name already in use" not in str(exc).lower():
                raise

    async def _subscribe(self, js: Any) -> Any:
        config = build_training_consumer_config(self.settings)
        return await js.pull_subscribe(
            self.settings.jobs_subject,
            durable=self.settings.durable,
            stream=self.settings.stream,
            config=config,
        )

    async def _process_message(self, message: Any) -> None:
        should_ack = True
        keepalive: TrainingLeaseKeepalive | None = None
        lease = None
        client: Any = None
        processor: TrainingJobProcessor | None = None
        try:
            payload = json.loads(message.data.decode("utf-8"))
            job = TrainingJobEnvelope.from_dict(payload)
        except Exception:
            # Poison message (malformed payload or an out-of-enum job_type):
            # terminate the delivery, never redeliver garbage.
            logger.exception("Discarding malformed training job payload.")
            await _terminate_message(message)
            return

        client = self._client_factory(job)

        # Resolve the adapter BEFORE leasing work this worker can't do
        # (plan 3.5/9.2). Unknown model_key is terminal, not retryable.
        try:
            adapter = get_adapter(job.model_key)
        except NoAdapterRegistered as exc:
            logger.error(
                "No adapter for training job; failing terminally.",
                extra={"job_id": job.job_id, "model_key": job.model_key},
            )
            with contextlib.suppress(Exception):
                await asyncio.to_thread(
                    client.append_event,
                    job.job_id,
                    event_type="training.job.adapter_missing",
                    message=str(exc),
                    metadata={
                        "model_key": job.model_key,
                        "job_type": job.job_type,
                        "worker_id": self.settings.worker_id,
                    },
                )
            with contextlib.suppress(Exception):
                await asyncio.to_thread(
                    client.update_job_status, job.job_id, status="failed", error=str(exc)
                )
            await _terminate_message(message)
            return

        try:
            lease = await asyncio.to_thread(
                client.acquire_lease, job.job_id, ttl_seconds=self.settings.lease_ttl_seconds
            )
        except (TrainingLeaseConflict, TrainingLeaseUnavailable):
            await _nak_message(message, delay_seconds=self.settings.redelivery_delay_seconds)
            return

        current_task = asyncio.current_task()
        lease_lost = False

        def mark_lease_lost() -> None:
            nonlocal lease_lost
            lease_lost = True
            if current_task is not None and not current_task.done():
                current_task.cancel()

        if lease is not None:
            keepalive = TrainingLeaseKeepalive(
                lease,
                client,
                loop=asyncio.get_running_loop(),
                ttl_seconds=self.settings.lease_ttl_seconds,
                message=message,
                on_lost=mark_lease_lost,
            )
            keepalive.start()

        processor = self._processor_factory(self.settings, client)
        await self._post_worker_heartbeat("busy", current_run_id=job.job_id, job_type=job.job_type)
        try:
            await asyncio.to_thread(client.update_job_status, job.job_id, status="running")
            summary = dict(await processor.process(job, adapter) or {})
            terminal_status = str(summary.pop("terminal_status", "succeeded"))
            if terminal_status not in TERMINAL_TRAINING_JOB_STATUSES:
                terminal_status = "succeeded"
            error_text = str(summary.get("error") or "") if terminal_status == "failed" else ""
            await asyncio.to_thread(
                client.update_job_status,
                job.job_id,
                status=terminal_status,
                error=error_text,
                output_summary=summary,
            )
        except asyncio.CancelledError:
            if lease_lost or (keepalive is not None and keepalive.lost):
                # Authoritative hand-off: another worker owns the job now.
                should_ack = False
                await _nak_message(message, delay_seconds=self.settings.redelivery_delay_seconds)
                return
            if self._shutting_down:
                should_ack = False
                await _nak_message(message)
                return
            raise
        except Exception as exc:
            try:
                await asyncio.to_thread(
                    client.update_job_status,
                    job.job_id,
                    status="failed",
                    error=str(exc) or exc.__class__.__name__,
                )
            except Exception:
                should_ack = False
                await _nak_message(message)
                raise
        finally:
            if keepalive is not None:
                await asyncio.to_thread(keepalive.stop)
            if processor is not None:
                await asyncio.to_thread(processor.terminate_children)
            release_lease = keepalive.current_lease() if keepalive is not None else lease
            if release_lease is not None:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(client.release_lease, release_lease)
            await self._post_worker_heartbeat("idle")
        if should_ack:
            await _ack_message(message)

    async def _maybe_post_idle_heartbeat(self, last_posted_at: float) -> float:
        interval_seconds = float(self.settings.runtime.worker_heartbeat_interval_seconds)
        if interval_seconds <= 0:
            return last_posted_at
        now = time.monotonic()
        if now - last_posted_at < interval_seconds:
            return last_posted_at
        await self._post_worker_heartbeat("idle")
        return now

    async def _post_worker_heartbeat(
        self,
        status: str,
        *,
        current_run_id: str | None = None,
        job_type: str = "",
    ) -> None:
        runtime = replace(
            self.settings.runtime,
            worker_id=self.settings.worker_id,
            worker_kind=self.settings.worker_kind,
        )
        metadata: dict[str, Any] = {"active_tasks": len(self._active_tasks)}
        if job_type:
            metadata["job_type"] = job_type
        try:
            await self._worker_heartbeat(
                runtime,
                status=status,
                current_run_id=current_run_id,
                metadata=metadata,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Training worker heartbeat failed; keeping worker alive.",
                extra={"worker_id": self.settings.worker_id, "status": status},
                exc_info=True,
            )


async def _terminate_message(message: Any) -> None:
    if hasattr(message, "term"):
        with contextlib.suppress(Exception):
            await message.term()
            return
    await _ack_message(message)


async def _ack_message(message: Any) -> None:
    if hasattr(message, "ack"):
        with contextlib.suppress(Exception):
            await message.ack()


async def _nak_message(message: Any, *, delay_seconds: float | None = None) -> None:
    if not hasattr(message, "nak"):
        return
    with contextlib.suppress(Exception):
        if delay_seconds is None or delay_seconds <= 0:
            await message.nak()
        else:
            try:
                await message.nak(delay=delay_seconds)
            except TypeError:
                await message.nak()


def _harvest_finished_tasks(tasks: set[asyncio.Task]) -> set[asyncio.Task]:
    pending: set[asyncio.Task] = set()
    for task in tasks:
        if task.done():
            _log_task_result(task)
        else:
            pending.add(task)
    return pending


async def _wait_for_task_capacity(tasks: set[asyncio.Task], *, timeout: float) -> set[asyncio.Task]:
    if not tasks:
        return tasks
    done, pending = await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
    for task in done:
        _log_task_result(task)
    return set(pending)


async def _cancel_tasks(tasks: set[asyncio.Task]) -> None:
    for task in tasks:
        task.cancel()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, asyncio.CancelledError):
            continue
        if isinstance(result, BaseException):
            logger.error(
                "Training worker message task failed during shutdown.",
                exc_info=(type(result), result, result.__traceback__),
            )


def _log_task_result(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except BaseException as exc:
        logger.error(
            "Training worker message task failed.",
            exc_info=(type(exc), exc, exc.__traceback__),
        )


async def amain() -> None:
    await NATSTrainingWorker().run_forever()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(amain())


if __name__ == "__main__":
    main()
