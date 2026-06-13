"""NATS worker for ``image.derive_pyramid`` jobs.

Consumes convert jobs from a JetStream subject, runs the (tested)
:func:`~ultra_deepagents.imaging.job.run_derive_pyramid_job` off-thread (the
convert shells out to ``imgcnv``), and acks. Pyramid metadata is read back via
the real engine when available. The job *logic* is unit-tested in
``test_imaging_job.py``; this module is the transport and needs a running NATS to
exercise end-to-end.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Callable

from ultra_deepagents.imaging.job import run_derive_pyramid_job

logger = logging.getLogger(__name__)

DEFAULT_SUBJECT = "ultra.image.jobs"
DEFAULT_STREAM = "ULTRA_IMAGE"
DEFAULT_DURABLE = "ultra-image-convert-worker"


def _resolve_meta_fn() -> Callable[[str], dict[str, Any]] | None:
    """Return the real engine's ``meta`` if the native lib is present, else None."""
    try:
        from ultra_deepagents.imaging.engine import LibBioImageEngine, build_engine

        engine = build_engine(prefer_real=True)
        if isinstance(engine, LibBioImageEngine):
            return engine.meta
    except Exception:  # noqa: BLE001
        logger.warning("imaging worker: real engine unavailable; pyramid metadata will be skipped")
    return None


def extract_derive_pyramid_payload(envelope: dict[str, Any]) -> dict[str, Any] | None:
    """Return the derive-pyramid job params from a raw message, or None if not ours.

    Accepts either a direct job dict or a Data Agent job envelope (where the
    convert params live under ``metadata`` and ``job_type`` selects the handler).
    """
    job_type = envelope.get("job_type")
    if job_type and job_type != "image.derive_pyramid":
        return None
    meta = envelope.get("metadata")
    if isinstance(meta, dict) and "src_path" in meta:
        return meta
    return envelope


async def _handle_message(msg: Any, *, meta_fn: Callable[[str], dict[str, Any]] | None) -> None:
    try:
        envelope = json.loads(msg.data.decode("utf-8"))
        job = extract_derive_pyramid_payload(envelope)
        if job is None:
            await msg.ack()  # not our job type; ack so JetStream stops redelivering it to us
            return
        result = await asyncio.to_thread(run_derive_pyramid_job, job, meta_fn=meta_fn)
        logger.info(
            "derive_pyramid done: resource=%s derived=%s levels=%s",
            result.get("resource_id"), result.get("derived_path"), result.get("levels"),
        )
        await msg.ack()
    except Exception as exc:  # noqa: BLE001 - redeliver on failure
        logger.exception("derive_pyramid job failed: %r", exc)
        with_nak = getattr(msg, "nak", None)
        if callable(with_nak):
            try:
                await msg.nak()
            except Exception:  # noqa: BLE001
                pass


async def run_worker_loop(
    *, nats_url: str | None = None, subject: str | None = None, durable: str | None = None
) -> None:
    import nats  # lazy: nats-py is a runtime dep, not needed to import this module

    nats_url = nats_url or os.environ.get("ULTRA_CONTROL_NATS_URL", "nats://127.0.0.1:4222")
    subject = subject or os.environ.get("ULTRA_CONTROL_NATS_IMAGE_JOBS_SUBJECT", DEFAULT_SUBJECT)
    durable = durable or os.environ.get("ULTRA_CONTROL_NATS_IMAGE_WORKER_DURABLE", DEFAULT_DURABLE)
    meta_fn = _resolve_meta_fn()

    nc = await nats.connect(nats_url)
    js = nc.jetstream()
    try:
        await js.add_stream(name=DEFAULT_STREAM, subjects=[subject])
    except Exception:  # noqa: BLE001 - stream may already exist
        pass
    sub = await js.pull_subscribe(subject, durable=durable)
    logger.info("imaging convert worker subscribed: subject=%s durable=%s", subject, durable)
    try:
        while True:
            try:
                msgs = await sub.fetch(1, timeout=5)
            except Exception:  # noqa: BLE001 - fetch timeout when idle
                continue
            for msg in msgs:
                await _handle_message(msg, meta_fn=meta_fn)
    finally:
        await nc.drain()
