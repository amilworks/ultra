"""Tier-1 compressed-horizon scenarios.

Each test runs the REAL run_job -> build_research_agent -> deepagents graph
with a deterministic scripted model + sandbox (see longhorizon_harness) and
asserts one clause of the long-horizon SLO:

- compaction keeps context bounded and early constraints reachable
- a worker crash + redelivery resumes with zero duplicated side effects
- the progress-stall breaker trips on real stagnation, injects its corrective
  prompt, persists the attempt ledger — and never fires on healthy polling
- the idle watchdog recovers a dead-silent model stream
- a week-scale turn count (hundreds of turns, dozens of compaction cycles)
  completes inside a CI time budget with every invariant intact

Wall-clock is compressed via RuntimeSettings knobs only; the machinery under
test is production code.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from deepagents.backends.protocol import ExecuteResponse
from longhorizon_harness import (
    IDLE_RECOVERY_MARKER,
    STALL_EXHAUSTED_MARKER,
    STALL_RECOVERY_MARKER,
    CompressedConfig,
    LongHorizonWorld,
    TurnDecision,
    TurnRequest,
    staged_pipeline_policy,
)
from longhorizon_invariants import (
    assert_canary_retained_after_compaction,
    assert_compaction_cycled,
    assert_conversation_bounded,
    assert_event_stream_integrity,
    assert_no_duplicate_side_effects,
    assert_terminal_success,
    summary_calls,
    turn_calls,
)

from ultra_deepagents.agent import attempt_ledger_path
from ultra_deepagents.progress_guard import read_attempt_ledger_digest


def _chunky_output(command: str, nth: int) -> ExecuteResponse:
    """Distinct, verbose output per call: grows context fast (forcing frequent
    compaction) while always reading as forward progress to the stall guard."""
    filler = f"stage metrics for call {nth}: " + ("telemetry sample; " * 50)
    return ExecuteResponse(output=f"{filler}\nchecksum {nth:08d}", exit_code=0)


def test_permanent_compaction_regime_retains_constraints(tmp_path) -> None:
    """Hardest compaction case: a window BELOW the fixed per-call overhead
    (system prompt + tool schemas, which the deepagents trigger counts), so the
    middleware summarizes on essentially every turn. Even in this regime the
    canary must survive and events must stay gapless — this is the maximum-churn
    stress on the summarize -> keep -> next-prompt transport."""
    world = LongHorizonWorld(tmp=tmp_path, behavior=_chunky_output)
    config = CompressedConfig(context_window_tokens=3000)
    canary = "ULTRA_CONSTRAINT_ALPHA7"
    policy = staged_pipeline_policy(
        world,
        rounds=40,
        final_answer=(
            f"Diagnostic sequence finished: 40 stages nominal. {canary} honored throughout."
        ),
    )
    goal = (
        "Run the staged diagnostic sequence to completion and then state the "
        f"outcome plainly. Always honor {canary} while working."
    )

    response = world.run_sync(policy, goal=goal, config=config)

    assert "40 stages nominal" in response
    assert_terminal_success(world.log)
    assert_event_stream_integrity(world.log)
    # Window < fixed overhead => near-every-turn summarization by design.
    assert_compaction_cycled(world, minimum_cycles=20)
    assert_conversation_bounded(world, window_tokens=config.context_window_tokens)
    assert_canary_retained_after_compaction(world, canary)
    assert world.sandbox.calls == [f"python stage.py --index {n}" for n in range(1, 41)]
    encoded = asyncio.run(world.checkpoint_encoded_bytes())
    assert 0 < encoded < 400_000, (
        f"final durable checkpoint is {encoded} bytes — compaction should keep the "
        f"thread slice small at any turn count"
    )


def test_worker_crash_and_redelivery_resume_without_duplicate_work(tmp_path) -> None:
    world = LongHorizonWorld(tmp=tmp_path)
    # A generous window keeps this scenario purely about the attempt boundary;
    # compaction interplay is covered by the compaction and soak scenarios.
    config = CompressedConfig(context_window_tokens=60_000)
    policy = staged_pipeline_policy(
        world,
        rounds=12,
        final_answer="Pipeline finished after resume: 12 stages, each executed once.",
        crash_on_round=8,
        crash_only_in_invocation=1,
    )
    goal = "Run the staged pipeline to completion and state the outcome plainly."

    async def scenario() -> tuple[BaseException | None, str]:
        crash: BaseException | None = None
        try:
            await world.run(policy, goal=goal, config=config)
        except RuntimeError as exc:
            crash = exc
        response = await world.run(policy, goal=goal, config=config)
        return crash, response

    crash, response = asyncio.run(scenario())

    assert crash is not None and "simulated worker crash" in str(crash)
    assert len(world.log.of_kind("run.resumed")) == 1, (
        "redelivered run did not resume from the durable checkpoint "
        f"(kinds: {sorted(set(world.log.kinds()))})"
    )
    # Stages 1-7 ran before the crash and were NOT re-executed after resume.
    assert world.sandbox.calls == [f"python stage.py --index {n}" for n in range(1, 13)]
    assert_no_duplicate_side_effects(world)
    assert_event_stream_integrity(world.log)
    assert_terminal_success(world.log, expected_failures=1)
    assert "each executed once" in response


def _constant_output(command: str, nth: int) -> ExecuteResponse:
    return ExecuteResponse(output="pipeline still warming up", exit_code=0)


def test_progress_stall_guard_trips_recovers_and_finalizes_honestly(tmp_path) -> None:
    world = LongHorizonWorld(tmp=tmp_path, behavior=_constant_output)
    config = CompressedConfig(
        context_window_tokens=60_000,
        progress_stall_threshold=4,
        progress_stall_max_recoveries=1,
    )

    def policy(request: TurnRequest) -> TurnDecision:
        if request.saw(STALL_RECOVERY_MARKER) or request.saw(STALL_EXHAUSTED_MARKER):
            return TurnDecision(
                text=(
                    "Monitoring stopped: the status output never changed across polls. "
                    "Reporting the partial result honestly."
                )
            )
        return TurnDecision(execute_command="poll status")

    response = world.run_sync(
        policy,
        goal="Watch the pipeline warm-up and state the outcome plainly.",
        config=config,
    )

    stall_events = [
        event
        for event in world.log.events
        if event.get("node_name") == "progress_stall_recovery"
    ]
    assert stall_events, "the stall breaker never fired on genuinely stagnant executes"
    payload = stall_events[0].get("payload")
    assert isinstance(payload, dict) and payload.get("reason") == "progress_stall"
    assert payload.get("stall_count") == config.progress_stall_threshold
    # 1 novel execute + `threshold` stagnant repeats, then the corrective prompt
    # lands and the run finalizes — a livelock would have burned hundreds.
    assert len(world.sandbox.calls) == 1 + config.progress_stall_threshold
    assert "honestly" in response
    assert_terminal_success(world.log)
    assert_event_stream_integrity(world.log)

    # The durable attempt ledger recorded the churn, and the ledger middleware
    # injected its digest into the recovery attempt's prompt — the cross-attempt
    # memory that survives compaction.
    workspace_dir = tmp_path / "workspaces" / world.run_id
    digest = read_attempt_ledger_digest(attempt_ledger_path(workspace_dir))
    assert "poll status" in digest
    recovery_turns = [
        record
        for record in turn_calls(world)
        if STALL_RECOVERY_MARKER in record.prompt_text
    ]
    assert recovery_turns, "no model call ever saw the stall-recovery prompt"
    assert any(
        "ATTEMPT LEDGER" in record.prompt_text for record in recovery_turns
    ), "the attempt-ledger digest never reached the recovery attempt's prompt"


def _moving_output(command: str, nth: int) -> ExecuteResponse:
    return ExecuteResponse(output=f"warm-up progress {nth * 5}%", exit_code=0)


def test_healthy_polling_never_trips_the_stall_guard(tmp_path) -> None:
    """False-positive control: a guard that kills legitimate long-running work
    is a long-horizon bug of the same rank as the livelock it prevents."""
    world = LongHorizonWorld(tmp=tmp_path, behavior=_moving_output)
    config = CompressedConfig(
        context_window_tokens=60_000,
        progress_stall_threshold=4,
        progress_stall_max_recoveries=1,
    )

    def policy(request: TurnRequest) -> TurnDecision:
        if len(world.sandbox.calls) < 12:
            return TurnDecision(execute_command="poll status")
        return TurnDecision(text="Warm-up completed after 12 polls with moving output.")

    response = world.run_sync(
        policy,
        goal="Watch the pipeline warm-up and state the outcome plainly.",
        config=config,
    )

    assert not [
        event
        for event in world.log.events
        if event.get("node_name") == "progress_stall_recovery"
    ], "stall guard false-positived on output that was visibly changing"
    assert len(world.sandbox.calls) == 12
    assert "12 polls" in response
    assert_terminal_success(world.log)
    assert_event_stream_integrity(world.log)


def test_idle_watchdog_recovers_a_dead_silent_stream(tmp_path) -> None:
    world = LongHorizonWorld(tmp=tmp_path)
    config = CompressedConfig(
        context_window_tokens=60_000,
        idle_timeout_seconds=0.75,
        idle_max_recoveries=1,
    )

    def policy(request: TurnRequest) -> TurnDecision:
        if request.saw(IDLE_RECOVERY_MARKER):
            return TurnDecision(text="Recovered from the stalled stream and finished the task.")
        # Dead transport: the model call blocks silently well past the watchdog
        # window before producing anything.
        return TurnDecision(text="warming up", sleep_seconds=2.5)

    started = time.monotonic()
    response = world.run_sync(
        policy,
        goal="Answer the diagnostic question plainly.",
        config=config,
    )
    elapsed = time.monotonic() - started

    recovery_events = [
        event
        for event in world.log.events
        if event.get("node_name") == "model_stream_recovery"
    ]
    assert recovery_events, "idle watchdog never fired on a silent stream"
    assert "Recovered from the stalled stream" in response
    assert_terminal_success(world.log)
    assert_event_stream_integrity(world.log)
    assert elapsed < 30, f"idle-recovery scenario took {elapsed:.1f}s — watchdog not bounding"


def test_week_scale_soak_completes_bounded_and_gapless(tmp_path) -> None:
    """The headline claim at compressed scale: hundreds of coordinator turns and
    dozens of compaction cycles, with context bounded, constraints retained,
    events gapless, side effects exactly-once, and CI-scale wall clock."""
    world = LongHorizonWorld(tmp=tmp_path, behavior=_chunky_output)
    # Window comfortably above the fixed per-call overhead (system prompt +
    # tool schemas) => the production sawtooth regime: compaction every ~dozen
    # turns, not every turn (measured: ~35 cycles at these parameters).
    config = CompressedConfig(context_window_tokens=24_000)
    canary = "ULTRA_CONSTRAINT_SOAK42"
    rounds = 400
    policy = staged_pipeline_policy(
        world,
        rounds=rounds,
        final_answer=(
            f"Soak finished: {rounds} stages nominal. {canary} honored to the end."
        ),
    )
    goal = (
        "Run the very long staged sequence to completion and then state the "
        f"outcome plainly. Always honor {canary} while working."
    )

    started = time.monotonic()
    response = world.run_sync(policy, goal=goal, config=config)
    elapsed = time.monotonic() - started

    assert f"{rounds} stages nominal" in response
    assert_terminal_success(world.log)
    assert_event_stream_integrity(world.log)
    assert_compaction_cycled(world, minimum_cycles=8)
    cycles = len(summary_calls(world))
    assert cycles <= rounds // 4, (
        f"{cycles} compaction cycles for {rounds} rounds — the window no longer "
        f"clears the fixed system-prompt + tool-schema overhead (permanent-"
        f"compaction regime); raise context_window_tokens to restore the sawtooth"
    )
    assert_conversation_bounded(world, window_tokens=config.context_window_tokens)
    assert_canary_retained_after_compaction(world, canary)
    assert world.sandbox.calls == [f"python stage.py --index {n}" for n in range(1, rounds + 1)]
    assert_no_duplicate_side_effects(world)
    turns = len(turn_calls(world))
    assert rounds + 1 <= turns <= rounds + 5, (
        f"expected ~{rounds + 1} coordinator turns, saw {turns} — a hidden retry "
        f"or continuation loop is inflating the turn count"
    )
    encoded = asyncio.run(world.checkpoint_encoded_bytes())
    assert 0 < encoded < 400_000
    assert elapsed < 240, (
        f"soak took {elapsed:.1f}s — the compressed tier must stay CI-fast; "
        f"profile before raising this budget"
    )


def test_render_proof_gate_blocks_unproven_html_then_accepts_proof(tmp_path) -> None:
    """End-to-end through the real completion loop: producing an .html
    deliverable without render evidence draws a completion-guard continuation
    demanding render proof; writing a passing, fresh proof satisfies the gate
    and the run completes. Pins the gate the same way the harness pins the
    stall/idle guards."""
    world = LongHorizonWorld(tmp=tmp_path)
    workspace = tmp_path / "workspaces" / world.run_id

    def behavior(command: str, nth: int) -> ExecuteResponse:
        if "build page" in command:
            page = workspace / "demo.html"
            page.parent.mkdir(parents=True, exist_ok=True)
            page.write_text("<html><body><button id='b'>go</button></body></html>")
            return ExecuteResponse(output="page written", exit_code=0)
        if "verify page" in command:
            proof = workspace / "diagnostics" / "report_preview" / "demo.console.json"
            proof.parent.mkdir(parents=True, exist_ok=True)
            proof.write_text('{"console_errors": [], "page_errors": []}')
            return ExecuteResponse(output="render check passed", exit_code=0)
        return ExecuteResponse(output=f"ok {nth}", exit_code=0)

    world.sandbox._behavior = behavior

    def policy(request: TurnRequest) -> TurnDecision:
        if request.saw("render proof"):
            # The gate's continuation prompt is in context: run the (scripted)
            # headless check, then restate the answer.
            if not any("verify page" in call for call in world.sandbox.calls):
                return TurnDecision(execute_command="verify page")
            return TurnDecision(
                text="The demo page demo.html is verified: zero console errors."
            )
        if not world.sandbox.calls:
            return TurnDecision(execute_command="build page")
        return TurnDecision(text="Built the demo page demo.html with one button.")

    response = world.run_sync(
        policy,
        goal="Make a small HTML page with a button and deliver it.",
        config=CompressedConfig(context_window_tokens=60_000),
    )

    guard_events = [
        event
        for event in world.log.events
        if event.get("node_name") == "completion_guard"
        and "render proof" in str((event.get("payload") or {}).get("text") or "").lower()
    ]
    assert guard_events, (
        "the completion guard never demanded render proof for the html deliverable "
        f"(kinds: {sorted(set(world.log.kinds()))})"
    )
    assert (
        (guard_events[0].get("payload") or {}).get("missing_artifact_kinds") is not None
    )
    assert "verify page" in " ".join(world.sandbox.calls)
    assert "verified" in response
    assert_terminal_success(world.log)
    assert_event_stream_integrity(world.log)


def test_render_proof_gate_stays_quiet_without_html_artifacts(tmp_path) -> None:
    """Negative control: ordinary non-HTML runs never see the render demand."""
    world = LongHorizonWorld(tmp=tmp_path)
    policy = staged_pipeline_policy(
        world,
        rounds=3,
        final_answer="Diagnostics done: 3 stages nominal.",
    )
    world.run_sync(
        policy,
        goal="Run the diagnostic sequence and state the outcome plainly.",
        config=CompressedConfig(context_window_tokens=60_000),
    )
    assert not [
        event
        for event in world.log.events
        if event.get("node_name") == "completion_guard"
        and "render proof" in str((event.get("payload") or {}).get("text") or "").lower()
    ]
    assert_terminal_success(world.log)
