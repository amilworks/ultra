"""Mid-run steering (Phase 1): the worker half.

While a run is in flight the control plane accumulates durable steering
messages (`control_run_steer_messages`). This module delivers them into the
running agent loop:

- ``SteeringInbox`` talks to the control plane (list / ack / close-barrier)
  with the worker token, via sync urllib offloaded to a thread — the same
  transport as the lease client, immune to model-call concurrency.
- ``SteeringInboxMiddleware.abefore_model`` is the injection point: a
  checkpointed graph node that runs before *every* coordinator model call.
  It returns a ``{"messages": [...]}`` state update whose messages carry the
  steer's ``message_id`` as the LangChain message id.

The id is the whole idempotency story. ``add_messages`` upserts by id, so a
steer injected here, seeded by a recovery requeue (the transcript row carries
the same id via ``_effective_job_messages`` normalization), or appended by
the finalization barrier collapses to ONE graph message no matter how the
attempt was restarted. Injection is simply "inject iff the id is absent";
acks are re-sent until the control plane confirms the pending→applied flip,
so a lost ack HTTP response self-heals on the next model call.

A steer poll can never crash or stall a run: every network failure is logged
and swallowed, and the next ``abefore_model`` retries.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error as urllib_error
import urllib.request as urllib_request
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware

from ultra_deepagents.config import RuntimeSettings

logger = logging.getLogger(__name__)

_STEER_TIMEOUT_SECONDS = 5.0


def _steer_headers(settings: RuntimeSettings, *, json_body: bool = False) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if json_body:
        headers["Content-Type"] = "application/json"
    token = str(getattr(settings, "control_worker_token", "") or "").strip()
    if token:
        headers["X-Ultra-Worker-Token"] = token
    return headers


class SteeringInbox:
    """Control-plane client for one run's steering messages."""

    def __init__(self, settings: RuntimeSettings, *, run_id: str) -> None:
        self._settings = settings
        self._run_id = run_id
        self._base_url = str(settings.control_base_url or "").rstrip("/")
        self._acked_steer_ids: set[str] = set()

    @property
    def run_id(self) -> str:
        return self._run_id

    def _url(self, suffix: str = "") -> str:
        return f"{self._base_url}/v2/runs/{self._run_id}/steer{suffix}"

    def _request_sync(
        self,
        url: str,
        *,
        method: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib_request.Request(
            url,
            data=body,
            method=method,
            headers=_steer_headers(self._settings, json_body=payload is not None),
        )
        with urllib_request.urlopen(request, timeout=_STEER_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))

    async def fetch(self) -> list[dict[str, Any]]:
        """List this run's steers, oldest first. Empty list on any failure."""
        try:
            data = await asyncio.to_thread(self._request_sync, self._url(), method="GET")
        except Exception:
            logger.warning(
                "Steering fetch failed; retrying at the next model call.",
                extra={"run_id": self._run_id},
                exc_info=True,
            )
            return []
        rows = data.get("steer_messages") if isinstance(data, dict) else None
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []

    async def ack(self, steer_id: str) -> bool:
        """Confirm pending→applied. Cached per-inbox; safe to re-send."""
        if steer_id in self._acked_steer_ids:
            return True
        try:
            await asyncio.to_thread(
                self._request_sync,
                self._url(f"/{steer_id}/ack"),
                method="POST",
                payload={"worker_id": ""},
            )
        except Exception:
            logger.warning(
                "Steering ack failed; the steer stays pending and the next poll retries.",
                extra={"run_id": self._run_id, "steer_id": steer_id},
                exc_info=True,
            )
            return False
        self._acked_steer_ids.add(steer_id)
        return True

    async def close_barrier(self) -> list[dict[str, Any]] | None:
        """Atomically stop steer acceptance; return still-pending steers.

        None (not []) on failure so the caller can tell "nothing pending"
        from "could not ask" — the latter is loud, and the control plane's
        terminal sweep will mark any stranded steer missed rather than let
        it vanish silently.
        """
        for attempt in range(3):
            try:
                data = await asyncio.to_thread(
                    self._request_sync,
                    self._url("/barrier"),
                    method="POST",
                    payload={},
                )
            except urllib_error.HTTPError as exc:
                logger.error(
                    "Steer barrier request rejected (HTTP %s); finishing without it.",
                    exc.code,
                    extra={"run_id": self._run_id},
                )
                return None
            except Exception:
                if attempt == 2:
                    logger.error(
                        "Steer barrier unreachable after retries; finishing without it. "
                        "Pending steers will be marked missed by the terminal sweep.",
                        extra={"run_id": self._run_id},
                        exc_info=True,
                    )
                    return None
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            rows = data.get("pending") if isinstance(data, dict) else None
            return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
        return None


def steer_message_id(steer: dict[str, Any]) -> str:
    return str(steer.get("message_id") or "").strip()


# Live-trace finding: injected RAW, a mid-run steer gets rationalized away —
# the model's reasoning literally said "that was not in the user's request.
# I'm done!" and skipped it. The frame makes authority explicit. It is applied
# identically by every worker-side path (middleware injection, finalization
# barrier, requeue seed normalization) so the model sees one consistent voice;
# the UI transcript renders the raw text from the thread row.
STEER_CONTENT_PREFIX = (
    "[Mid-run steering update from the user — this is now part of the request. "
    "Apply it to the current work without discarding progress.]"
)


def steer_agent_content(text: str) -> str:
    return f"{STEER_CONTENT_PREFIX}\n{text}"


def steer_ids_in_messages(messages: Any) -> set[str]:
    """Every steer message id already represented in the state's messages.

    Two shapes count: a message whose LangChain id IS a steer message_id
    (middleware/barrier copies), and a requeue-seeded transcript dict whose
    steering metadata rode into additional_kwargs with its row message_id.
    """
    ids: set[str] = set()
    for message in messages or []:
        message_id = getattr(message, "id", None)
        if isinstance(message, dict):
            message_id = message.get("id") or message_id
        if isinstance(message_id, str) and message_id:
            ids.add(message_id)
        kwargs = getattr(message, "additional_kwargs", None)
        if isinstance(message, dict):
            kwargs = message.get("additional_kwargs") or kwargs
        if not isinstance(kwargs, dict):
            continue
        metadata = kwargs.get("metadata")
        if isinstance(metadata, dict) and metadata.get("kind") == "steering":
            row_id = kwargs.get("message_id")
            if isinstance(row_id, str) and row_id:
                ids.add(row_id)
    return ids


class SteeringInboxMiddleware(AgentMiddleware):
    """Injects pending steers into the coordinator loop before model calls.

    Coordinator-only by construction: it is registered on the top-level
    agent's middleware list and never propagated to subagent stacks — a
    steer lands between coordinator steps, not inside a delegation.
    """

    def __init__(self, inbox: SteeringInbox) -> None:
        super().__init__()
        self._inbox = inbox

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        try:
            return await self._poll(state)
        except Exception:
            # Belt over the inbox's own suspenders: steering must never take
            # down a run.
            logger.warning(
                "Steering middleware poll failed; continuing run.",
                extra={"run_id": self._inbox.run_id},
                exc_info=True,
            )
            return None

    async def _poll(self, state: Any) -> dict[str, Any] | None:
        steers = await self._inbox.fetch()
        if not steers:
            return None
        present = steer_ids_in_messages((state or {}).get("messages"))
        to_inject: list[dict[str, Any]] = []
        for steer in steers:
            message_id = steer_message_id(steer)
            if not message_id:
                continue
            status = str(steer.get("status") or "").strip()
            in_state = message_id in present
            injecting = False
            # Inject anything absent — including an already-applied steer
            # whose state was rebuilt by a continuation pass: the model must
            # keep seeing it.
            if not in_state and status != "missed":
                content = str(steer.get("content") or "")
                if content.strip():
                    to_inject.append(
                        {
                            "role": "user",
                            "content": steer_agent_content(content),
                            "id": message_id,
                        }
                    )
                    injecting = True
            if status == "pending" and (in_state or injecting):
                # Present now (or about to be): confirm application. Failures
                # leave it pending; the next model call retries.
                await self._inbox.ack(str(steer.get("steer_id") or ""))
        if not to_inject:
            return None
        logger.info(
            "Applying %d steering message(s) to the running agent.",
            len(to_inject),
            extra={"run_id": self._inbox.run_id},
        )
        return {"messages": to_inject}


def build_steering_inbox(settings: RuntimeSettings, *, run_id: str) -> SteeringInbox | None:
    """Inbox for runs with a control-plane identity; None otherwise.

    Gated on the worker token (the steer endpoints are worker-scoped): bare
    local/test runs without a control plane get no middleware and no
    per-model-call HTTP chatter.
    """
    if not str(getattr(settings, "control_worker_token", "") or "").strip():
        return None
    if not str(getattr(settings, "control_base_url", "") or "").strip():
        return None
    return SteeringInbox(settings, run_id=run_id)
