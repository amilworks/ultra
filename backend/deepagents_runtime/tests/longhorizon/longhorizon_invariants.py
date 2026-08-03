"""Invariant checkers for compressed-horizon runs.

Each checker asserts one clause of the long-horizon SLO in a form that fails
with a diagnosable message rather than a bare boolean:

- event-stream integrity: per-run source sequences are unique, strictly
  increasing in publish order, and contiguous (the control plane's
  UNIQUE(run_id, source_sequence) + strict partition consumer contract —
  a gap or collision here is what wedges ingest in production).
- bounded context: compaction actually cycles, and the conversation portion
  of every model call stays bounded by the declared window instead of growing
  monotonically with turn count.
- constraint retention: a canary planted in the goal is still visible to the
  model after the last compaction cycle — proving the summarize -> keep ->
  next-prompt transport, not the model's goodwill.
- terminal honesty: exactly the expected terminal event, with usage accounted.
"""

from __future__ import annotations

from typing import Any

from longhorizon_harness import CANARY_PATTERN, EventLog, LongHorizonWorld, ModelCallRecord


def assert_event_stream_integrity(log: EventLog) -> None:
    sequences = [int(event.get("sequence") or 0) for event in log.events]
    assert sequences, "no events were published"
    for position, (previous, current) in enumerate(zip(sequences, sequences[1:])):
        assert current > previous, (
            f"sequence not strictly increasing at publish position {position + 1}: "
            f"{previous} -> {current} "
            f"(kinds: {log.kinds()[position]} -> {log.kinds()[position + 1]})"
        )
    expected = set(range(min(sequences), max(sequences) + 1))
    missing = sorted(expected - set(sequences))
    assert not missing, f"sequence gaps detected (control-plane ingest would stall): {missing}"

    event_ids = [str(event.get("event_id") or "") for event in log.events]
    assert all(event_ids), "an event was published without an event_id"
    duplicates = {event_id for event_id in event_ids if event_ids.count(event_id) > 1}
    assert not duplicates, f"duplicate event_ids (dedup would drop real events): {duplicates}"


def summary_calls(world: LongHorizonWorld) -> list[ModelCallRecord]:
    return [record for record in world.model_calls if record.kind == "summary"]


def turn_calls(world: LongHorizonWorld) -> list[ModelCallRecord]:
    return [record for record in world.model_calls if record.kind == "turn"]


def assert_compaction_cycled(world: LongHorizonWorld, *, minimum_cycles: int) -> None:
    observed = len(summary_calls(world))
    assert observed >= minimum_cycles, (
        f"expected >= {minimum_cycles} compaction cycles at this window/turn count, "
        f"saw {observed} — either the window knob is not reaching the middleware "
        f"or compaction silently stopped"
    )


def assert_conversation_bounded(world: LongHorizonWorld, *, window_tokens: int) -> None:
    """Compaction must keep the conversation sawtoothing under a fixed multiple
    of the window; unbounded growth here is the exact failure mode that makes
    week-long runs impossible regardless of model quality."""
    turns = turn_calls(world)
    assert turns, "no turn-kind model calls recorded"
    # One uncompacted turn can legitimately overshoot the 85% trigger by the
    # size of the newest tool result; 2x the window is far below unbounded
    # growth (a linear-growth bug at these turn counts lands 10-50x over).
    bound = 2 * window_tokens
    worst = max(turns, key=lambda record: record.approx_conversation_tokens)
    assert worst.approx_conversation_tokens <= bound, (
        f"conversation grew to ~{worst.approx_conversation_tokens} tokens at model call "
        f"#{worst.index} (bound {bound}, window {window_tokens}) — compaction is not "
        f"holding the context; a week-long run would OOM the model input"
    )


def assert_canary_retained_after_compaction(world: LongHorizonWorld, canary: str) -> None:
    assert CANARY_PATTERN.fullmatch(canary), (
        f"canary {canary!r} must match {CANARY_PATTERN.pattern} or the scripted "
        f"summarizer cannot echo it"
    )
    summaries = summary_calls(world)
    assert summaries, "cannot check post-compaction retention: no compaction happened"
    last_summary_index = summaries[-1].index
    late_turns = [record for record in turn_calls(world) if record.index > last_summary_index]
    assert late_turns, "no turn calls after the last compaction cycle"
    for record in late_turns:
        assert canary in record.prompt_text, (
            f"constraint canary {canary} vanished from the model's context by call "
            f"#{record.index} (after the last compaction at call #{last_summary_index}) — "
            f"an early constraint would be silently dropped in week 2 of a real run"
        )


def run_completed_payload(log: EventLog) -> dict[str, Any]:
    completed = log.of_kind("run.completed")
    assert len(completed) == 1, (
        f"expected exactly one run.completed, saw {len(completed)} "
        f"(kinds: {sorted(set(log.kinds()))})"
    )
    payload = completed[0].get("payload")
    assert isinstance(payload, dict)
    return payload


def assert_terminal_success(log: EventLog, *, expected_failures: int = 0) -> dict[str, Any]:
    """The run ends in exactly one run.completed; ``expected_failures`` counts
    the run.failed events a crash/redelivery scenario deliberately produced."""
    failures = log.of_kind("run.failed")
    assert len(failures) == expected_failures, (
        f"expected {expected_failures} run.failed event(s), saw {len(failures)}: "
        f"{[event.get('message') for event in failures]}"
    )
    payload = run_completed_payload(log)
    usage = payload.get("usage")
    assert isinstance(usage, dict) and int(usage.get("total_tokens") or 0) > 0, (
        f"run.completed carries no token usage (usage plumbing broke): {usage!r}"
    )
    return payload


def executed_commands(world: LongHorizonWorld) -> list[str]:
    return list(world.sandbox.calls)


def assert_no_duplicate_side_effects(world: LongHorizonWorld) -> None:
    commands = executed_commands(world)
    duplicates = sorted({command for command in commands if commands.count(command) > 1})
    assert not duplicates, (
        f"sandbox commands executed more than once across attempt boundaries "
        f"(resume re-ran completed work): {duplicates}"
    )
