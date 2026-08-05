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
import re
import tempfile
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from ultra_deepagents.imaging.derivative_manifest import (
    DeterministicDerivativeError,
    StaleDerivativeJobError,
)
from ultra_deepagents.imaging.job import run_derive_pyramid_job

logger = logging.getLogger(__name__)

DEFAULT_SUBJECT = "ultra.image.jobs"
DEFAULT_STREAM = "ULTRA_IMAGE"
DEFAULT_DURABLE = "ultra-image-convert-worker"
# Redelivery cap for a failing convert job, mirroring the main data-agent worker's
# ConsumerConfig.max_deliver. Bounds a poison job (a source that always fails to
# convert) so it cannot loop forever. Override with ULTRA_IMGSVC_MAX_DELIVER.
DEFAULT_MAX_DELIVER = 5
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _resolve_meta_fn() -> Callable[[str], dict[str, Any]] | None:
    """Return the real engine's ``meta`` if the native lib is present, else None."""
    try:
        from ultra_deepagents.imaging.engine import LibBioImageEngine, build_engine

        engine = build_engine(prefer_real=True)
        if isinstance(engine, LibBioImageEngine):

            def resolved(path: str) -> dict[str, Any]:
                return dict(engine.meta(path))

            return resolved
    except Exception:
        logger.warning("imaging worker: real engine unavailable; pyramid metadata will be skipped")
    return None


def _resolve_viewer_info_fn() -> Callable[[str], dict[str, Any]] | None:
    """Return the real engine's ``viewer_info`` for semantic publication checks.

    Strict derivative jobs fail closed when this is unavailable: the worker must prove
    that conversion preserved T/Z/C, dtype, scene, channel order, and spacing before it
    can publish a serving manifest.
    """
    try:
        from ultra_deepagents.imaging.engine import build_engine

        engine = build_engine(prefer_real=True)
        viewer_info = getattr(engine, "viewer_info", None)
        if callable(viewer_info):

            def resolved(path: str) -> dict[str, Any]:
                return dict(viewer_info(path))

            return resolved
    except Exception:
        return None
    return None


def _resolve_source_viewer_info_fn() -> Callable[[str], dict[str, Any]] | None:
    """Resolve BioIO source semantics for proprietary preferred-reader jobs."""

    try:
        from ultra_deepagents.imaging.bioio_engine import BioioEngine

        engine = BioioEngine()

        def resolved(path: str) -> dict[str, Any]:
            return dict(engine.strict_publication_viewer_info(path))

        return resolved
    except Exception:
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


def _delivery_attempt(msg: Any) -> int:
    """Best-effort JetStream delivery count (1 on the first delivery). Falls back to
    1 when the transport doesn't expose metadata, so a missing count never terminates
    a message prematurely (we'd rather retry once more than drop a recoverable job)."""
    try:
        num = int(msg.metadata.num_delivered)
    except Exception:
        return 1
    return num if num >= 1 else 1


async def _safe_ack_op(msg: Any, op: str) -> None:
    """Invoke msg.ack/nak/term if present, swallowing transport errors — a failed ack
    operation must never crash the worker loop."""
    fn = getattr(msg, op, None)
    if not callable(fn):
        return
    with suppress(Exception):
        await fn()


def _failure_marker_path(dst_path: str) -> str:
    """Sidecar path the control plane checks to back off a permanently-failed derive.
    Mirrors Go's derivedPyramidFailedName: ``<fileID>__pyramid.failed`` next to the
    would-be ``<fileID>__pyramid.tif`` pyramid."""
    return dst_path[:-4] + ".failed" if dst_path.endswith(".tif") else dst_path + ".failed"


def _write_failure_marker(job: dict[str, Any] | None, exc: DeterministicDerivativeError) -> None:
    """Best-effort: record that this resource's pyramid derivation permanently failed,
    so the control plane stops re-enqueuing a doomed conversion on every viewer open
    (which would keep burning the engine — the failure mode behind the 707k-redelivery
    incident). The marker's mtime gives the control plane a backoff window; a later
    successful derive or an explicit re-derive clears it. Never raises."""
    try:
        dst = (job or {}).get("dst_path")
        if not dst:
            return
        marker = _failure_marker_path(str(dst))
        directory = os.path.dirname(marker)
        os.makedirs(directory, exist_ok=True)
        payload = {
            "schema": "ultra.image-derived-pyramid-failure.v1",
            "resource_id": (job or {}).get("resource_id"),
            "source_sha256": (job or {}).get("source_sha256"),
            "source_size_bytes": (job or {}).get("source_size_bytes"),
            "conversion_spec": {
                "tile_size": (job or {}).get("tile_size", 512),
                "compression": (job or {}).get("compression", "lzw"),
                "layout": (job or {}).get("layout", "topdirs"),
                "fmt": (job or {}).get("fmt", "auto"),
            },
            "code": exc.code,
        }
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{os.path.basename(marker)}.", dir=directory
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                os.fchmod(stream.fileno(), 0o644)
                json.dump(payload, stream, allow_nan=False, separators=(",", ":"), sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, marker)
            directory_descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            with suppress(OSError):
                os.unlink(temporary)
    except Exception:
        logger.debug("imaging worker: failed to write deterministic failure marker")


def _has_strict_source_identity(job: dict[str, Any]) -> bool:
    sha256 = job.get("source_sha256")
    size = job.get("source_size_bytes")
    return (
        isinstance(sha256, str)
        and _SHA256_RE.fullmatch(sha256) is not None
        and isinstance(size, int)
        and not isinstance(size, bool)
        and size >= 0
    )


def _failure_class(exc: BaseException) -> str:
    if isinstance(exc, StaleDerivativeJobError):
        return "stale"
    if isinstance(exc, DeterministicDerivativeError):
        return "deterministic"
    return "transient"


async def _handle_message(
    msg: Any,
    *,
    meta_fn: Callable[[str], dict[str, Any]] | None,
    viewer_info_fn: Callable[[str], dict[str, Any]] | None = None,
    source_viewer_info_fn: Callable[[str], dict[str, Any]] | None = None,
    max_deliver: int = DEFAULT_MAX_DELIVER,
) -> None:
    job: dict[str, Any] | None = None
    try:
        envelope = json.loads(msg.data.decode("utf-8"))
        job = extract_derive_pyramid_payload(envelope)
        if job is None:
            await _safe_ack_op(
                msg, "ack"
            )  # not our job type; ack so JetStream stops redelivering it to us
            return
        if not _has_strict_source_identity(job):
            logger.info(
                "retiring stale derive_pyramid job without immutable source identity: resource=%s",
                job.get("resource_id"),
            )
            await _safe_ack_op(msg, "ack")
            return
        result = await asyncio.to_thread(
            run_derive_pyramid_job,
            job,
            meta_fn=meta_fn,
            viewer_info_fn=viewer_info_fn,
            source_viewer_info_fn=source_viewer_info_fn,
            require_manifest=True,
        )
        logger.info(
            "derive_pyramid done: resource=%s derived=%s levels=%s",
            result.get("resource_id"),
            result.get("derived_path"),
            result.get("levels"),
        )
        await _safe_ack_op(msg, "ack")
        # Pre-warm the source's viewer-info sidecar so the FIRST viewer open is instant
        # (engine.viewer_info persists it to a shared sidecar). Best-effort; never blocks.
        src = (job or {}).get("src_path")
        if viewer_info_fn is not None and src:
            try:
                await asyncio.to_thread(viewer_info_fn, str(src))
            except Exception:
                logger.debug("viewer-info pre-warm skipped for %s", src)
        # Pre-populate the LOCAL pyramid cache so the FIRST viewer open is instant: the
        # pyramid is served from local disk (it hangs over remote NFS), so copy it now,
        # off the user's request path. Best-effort; never blocks the job.
        derived_path = result.get("derived_path")
        if derived_path:
            try:
                from ultra_deepagents.imaging.service import localize_pyramid

                await asyncio.to_thread(localize_pyramid, str(derived_path))
            except Exception:
                logger.debug("pyramid cache pre-warm skipped for %s", derived_path)
    except Exception as exc:
        # Bound redelivery so a POISON job (corrupt / unsupported / OOM source that will
        # always fail) cannot wedge the consumer in an infinite nak->redeliver loop
        # (criterion C4: "a poison job does not wedge the consumer forever"). Below the
        # cap we nak() for a transient retry; at the cap we term() so JetStream stops
        # redelivering this message for good, logged as a dead-letter for operators. The
        # main data-agent worker gets this bound from ConsumerConfig.max_deliver; this
        # pull consumer enforces it message-side so it holds even against a pre-existing
        # server-side consumer whose config predates the cap.
        failure_class = _failure_class(exc)
        if failure_class == "stale":
            logger.info(
                "retiring stale derive_pyramid job: resource=%s code=%s",
                (job or {}).get("resource_id"),
                getattr(exc, "code", "stale_job"),
            )
            await _safe_ack_op(msg, "ack")
            return
        attempt = _delivery_attempt(msg)
        if failure_class == "deterministic" or attempt >= max_deliver:
            logger.error(
                "derive_pyramid permanently failed after %d attempt(s); "
                "terminating (dead-letter): resource=%s code=%s",
                attempt,
                (job or {}).get("resource_id"),
                getattr(exc, "code", "retry_limit_exhausted"),
            )
            if isinstance(exc, DeterministicDerivativeError):
                _write_failure_marker(
                    job, exc
                )  # source/spec-bound marker suppresses only the same doomed conversion
            await _safe_ack_op(msg, "term")
        else:
            logger.warning(
                "derive_pyramid failed (attempt %d/%d); will retry: resource=%s code=%s",
                attempt,
                max_deliver,
                (job or {}).get("resource_id"),
                getattr(exc, "code", "transient_failure"),
            )
            await _safe_ack_op(msg, "nak")


async def run_worker_loop(
    *, nats_url: str | None = None, subject: str | None = None, durable: str | None = None
) -> None:
    import nats  # type: ignore[import-not-found]  # lazy runtime dependency

    nats_url = nats_url or os.environ.get("ULTRA_CONTROL_NATS_URL", "nats://127.0.0.1:4222")
    subject = subject or os.environ.get("ULTRA_CONTROL_NATS_IMAGE_JOBS_SUBJECT", DEFAULT_SUBJECT)
    durable = durable or os.environ.get("ULTRA_CONTROL_NATS_IMAGE_WORKER_DURABLE", DEFAULT_DURABLE)
    meta_fn = _resolve_meta_fn()
    viewer_info_fn = _resolve_viewer_info_fn()
    source_viewer_info_fn = _resolve_source_viewer_info_fn()
    max_deliver = DEFAULT_MAX_DELIVER
    raw_max = os.environ.get("ULTRA_IMGSVC_MAX_DELIVER")
    if raw_max:
        try:
            parsed = int(raw_max)
            if parsed >= 1:
                max_deliver = parsed
        except ValueError:
            pass

    nc = await nats.connect(nats_url)
    js = nc.jetstream()
    with suppress(Exception):
        await js.add_stream(name=DEFAULT_STREAM, subjects=[subject])
    sub = await js.pull_subscribe(subject, durable=durable)
    logger.info("imaging convert worker subscribed: subject=%s durable=%s", subject, durable)
    try:
        while True:
            try:
                msgs = await sub.fetch(1, timeout=5)
            except Exception:
                continue
            for msg in msgs:
                await _handle_message(
                    msg,
                    meta_fn=meta_fn,
                    viewer_info_fn=viewer_info_fn,
                    source_viewer_info_fn=source_viewer_info_fn,
                    max_deliver=max_deliver,
                )
    finally:
        await nc.drain()
