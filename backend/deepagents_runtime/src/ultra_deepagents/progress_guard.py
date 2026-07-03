"""Within-turn progress-stall detection and the durable attempt ledger.

Why this exists: a live comp-bio run livelocked to 6.7M tokens over ~3h by
cycling over a small set of sandbox commands whose outputs never changed
(measured: 114 executes / 22 distinct in one 20-minute window — ~80% repeats).
Every anti-churn guard the runner has (the completion-guard no-progress break,
the model-stream idle recovery) lives in the BETWEEN-turn loop and only runs
after ``_stream_agent_attempt`` returns — a turn that never ends is invisible
to them. The idle watchdog cannot help either: it is a dead-transport detector
that resets on every event, and a busy livelock emits a perfectly healthy event
stream. This module supplies the missing WITHIN-turn signal.

Progress definition (what resets the stall counter) — chosen so the guard can
never interrupt genuinely progressing long scientific work:

- a NOVEL sandbox command (never seen in this agent scope) is forward motion;
- a repeated command whose OUTPUT CHANGED (preview or size) is forward motion
  (covers legitimate polling: job monitors produce moving output);
- any workspace MUTATION (``write_file``/``edit_file``) or completed delegation
  (``task``) is forward motion — after an edit, re-running the same command is
  a genuinely new attempt (the edit-and-rerun debugging loop stays untouched);
- a single long-running command never advances the counter (one event pair).

Only a monotone run of repeats-with-unchanged-output advances the counter, and
the states are kept PER AGENT SCOPE (coordinator vs. each subagent namespace)
so one healthy stream cannot mask another scope's churn. Even when the guard
fires it ends the TURN, not the run: the runner injects a corrective prompt
(capped recoveries) and finally falls through to finalization with the partial
results — mirroring the proven model-stream idle recovery arm.

The ``AttemptLedger`` is the durable, compaction-proof memory of what already
failed: the runner appends an entry whenever an execute errors or repeats with
unchanged output, and ``UltraAttemptLedgerMiddleware`` (agent.py) re-derives a
compact digest into the system prompt on every model call. Advisory by design —
a previously failing command may succeed after the environment changes, so
nothing is ever blocked; the model is only reminded of what it already tried.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The sandbox execution tool (live event histograms confirm the name). Repeats of
# these are what a livelock is made of.
EXECUTE_TOOL_NAMES = frozenset({"execute"})
# Forward-motion tools: a completed mutation or delegation means the next re-run
# is a new attempt, not churn.
PROGRESS_TOOL_NAMES = frozenset({"write_file", "edit_file", "task"})

_COMMAND_PREVIEW_CHARS = 160
_ERROR_PREVIEW_CHARS = 200
_LEDGER_MAX_ENTRIES = 500
_DIGEST_MAX_COMMANDS = 10


@dataclass(frozen=True)
class ProgressStallVerdict:
    """Raised-worthy stall: ``stall_count`` consecutive non-progressing executes."""

    scope: str
    stall_count: int
    # (command preview, repeat count) — most-repeated first, bounded.
    repeated_commands: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ProgressObservation:
    """Classification of one observed execute completion."""

    scope: str
    command_preview: str
    stagnant_repeat: bool = False
    error_preview: str = ""
    verdict: ProgressStallVerdict | None = None


@dataclass
class _ScopeState:
    last_output_by_command: dict[str, str] = field(default_factory=dict)
    pending_command_by_call: dict[str, tuple[str, str]] = field(default_factory=dict)
    stall_count: int = 0
    repeats: dict[str, int] = field(default_factory=dict)


class ProgressStallDetector:
    """Counts consecutive non-progressing sandbox executes per agent scope."""

    def __init__(self, threshold: int) -> None:
        self._threshold = max(0, int(threshold))
        self._scopes: dict[str, _ScopeState] = {}

    @property
    def enabled(self) -> bool:
        return self._threshold > 0

    def observe(self, tool_event: dict[str, Any]) -> ProgressObservation | None:
        """Feed one normalized tool event; returns a classification for execute
        completions (with a verdict when the stall threshold is reached), else None."""
        if not self.enabled:
            return None
        payload = tool_event.get("payload")
        if not isinstance(payload, dict):
            return None
        tool_name = str(payload.get("tool_name") or "").strip()
        status = str(payload.get("status") or "").strip().lower()
        scope = _event_scope(payload)
        state = self._scopes.setdefault(scope, _ScopeState())

        if tool_name in PROGRESS_TOOL_NAMES:
            # The workspace or delegation state changed: whatever gets re-run next
            # is a new attempt. (Errors here change nothing, so only completions reset.)
            if status == "completed":
                state.stall_count = 0
                state.repeats.clear()
            return None
        if tool_name not in EXECUTE_TOOL_NAMES:
            return None

        call_id = str(payload.get("tool_call_id") or "").strip()
        if status == "started":
            signature, preview = _command_signature(payload)
            if call_id and signature:
                state.pending_command_by_call[call_id] = (signature, preview)
            return None
        if status not in {"completed", "failed", "error"}:
            return None

        signature, preview = state.pending_command_by_call.pop(call_id, ("", ""))
        if not signature:
            # Completion events may not carry the input; without the started-event
            # correlation, fall back to any input present, else stay neutral.
            signature, preview = _command_signature(payload)
        error_preview = str(payload.get("error") or "")[:_ERROR_PREVIEW_CHARS]
        if not signature:
            return None

        output_signature = _output_signature(payload, status)
        previous = state.last_output_by_command.get(signature)
        state.last_output_by_command[signature] = output_signature
        if previous is None or previous != output_signature:
            # Novel command, or a repeat whose output moved: forward progress.
            state.stall_count = 0
            state.repeats.clear()
            return ProgressObservation(
                scope=scope,
                command_preview=preview,
                error_preview=error_preview,
            )

        state.stall_count += 1
        if preview:
            state.repeats[preview] = state.repeats.get(preview, 0) + 1
        verdict: ProgressStallVerdict | None = None
        if state.stall_count >= self._threshold:
            verdict = ProgressStallVerdict(
                scope=scope,
                stall_count=state.stall_count,
                repeated_commands=tuple(
                    sorted(state.repeats.items(), key=lambda kv: -kv[1])[:5]
                ),
            )
            # Re-arm: the runner aborts the turn and injects a corrective prompt;
            # the next window starts clean (bounded overall by max recoveries).
            state.stall_count = 0
            state.repeats.clear()
        return ProgressObservation(
            scope=scope,
            command_preview=preview,
            stagnant_repeat=True,
            error_preview=error_preview,
            verdict=verdict,
        )


class AttemptLedger:
    """Append-only, per-run JSONL record of failed/stagnant sandbox attempts.

    Written by the runner as tool events stream; read back (mtime-cached) by
    ``UltraAttemptLedgerMiddleware`` to inject a bounded digest into the system
    prompt each model call — so the memory of what already failed survives
    SummarizationMiddleware compaction, which rewrites ``messages`` but never
    the per-request system prompt. Best-effort on I/O: a ledger failure must
    never fail a run.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries_written = 0

    @property
    def path(self) -> Path:
        return self._path

    def record(
        self,
        *,
        scope: str,
        command_preview: str,
        kind: str,
        detail: str = "",
    ) -> None:
        if not command_preview or self._entries_written >= _LEDGER_MAX_ENTRIES:
            return
        entry = {
            "scope": scope,
            "command": command_preview,
            "kind": kind,  # "error" | "stagnant_repeat"
            "detail": detail[:_ERROR_PREVIEW_CHARS],
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._entries_written += 1
        except OSError:
            logger.debug("attempt ledger write failed (ignored)", exc_info=True)


def read_attempt_ledger_digest(path: Path, *, max_commands: int = _DIGEST_MAX_COMMANDS) -> str:
    """Aggregate the ledger into a compact advisory digest (empty when no entries).

    Healthy runs produce a near-empty ledger, so this costs nothing when there is
    no churn. Deliberately advisory wording: re-running after a real change is fine.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    counts: dict[str, int] = {}
    details: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        command = str(entry.get("command") or "").strip()
        if not command:
            continue
        counts[command] = counts.get(command, 0) + 1
        detail = str(entry.get("detail") or "").strip()
        if detail:
            details[command] = detail
    if not counts:
        return ""
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:max_commands]
    lines = [
        "ATTEMPT LEDGER (advisory; durable record of this run's sandbox history — it "
        "survives context compaction):",
        "These commands already FAILED or repeated with UNCHANGED output. Do not blindly "
        "re-run them — read the failing output/logs, change the code or inputs first, or "
        "choose a different approach. Re-running after a real change is fine.",
    ]
    for command, count in ranked:
        detail = details.get(command, "")
        suffix = f"  [last: {detail}]" if detail else ""
        lines.append(f"- {count}x: {command}{suffix}")
    dropped = len(counts) - len(ranked)
    if dropped > 0:
        lines.append(f"- (+{dropped} more repeated/failed commands not shown)")
    return "\n".join(lines)


def _event_scope(payload: dict[str, Any]) -> str:
    subagent = str(payload.get("subagent_name") or "").strip()
    if subagent:
        return subagent
    namespace = payload.get("namespace")
    if isinstance(namespace, (list, tuple)) and namespace:
        return "/".join(str(part) for part in namespace)
    return ""


def _command_signature(payload: dict[str, Any]) -> tuple[str, str]:
    """(stable hash, human preview) of an execute input; ("", "") when absent."""
    raw = payload.get("input")
    if raw is None:
        raw = payload.get("command")
    if raw is None:
        return "", ""
    if isinstance(raw, dict):
        preview_source = raw.get("command") or raw.get("code") or raw
        try:
            canonical = json.dumps(raw, sort_keys=True, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            canonical = str(raw)
    else:
        preview_source = raw
        canonical = str(raw)
    preview = " ".join(str(preview_source).split())[:_COMMAND_PREVIEW_CHARS]
    digest = hashlib.sha256(canonical.encode("utf-8", errors="replace")).hexdigest()
    return digest, preview


def _output_signature(payload: dict[str, Any], status: str) -> str:
    """Fingerprint of what an execute produced. Includes the untruncated output
    size so movement beyond the preview cutoff still reads as progress."""
    parts = [
        status,
        str(payload.get("output_preview") or ""),
        str(payload.get("output_size_chars") or ""),
        str(payload.get("error") or ""),
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8", errors="replace")).hexdigest()
