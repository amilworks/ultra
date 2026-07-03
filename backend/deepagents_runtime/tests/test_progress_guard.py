"""Trace tests for the within-turn progress-stall guard + durable attempt ledger.

These encode the fix for the 6.7M-token/3h livelock (a run cycling over ~22
commands with unchanged outputs at 80% repeats per window while every existing
guard — all between-turn — stayed silent). The safety fixtures are the load-
bearing ones: the guard must NEVER fire on genuinely progressing long work
(edit-and-rerun debugging, output-moving polling, novel-command exploration).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from langchain.agents.middleware import ModelRequest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage

from ultra_deepagents.agent import UltraAttemptLedgerMiddleware, attempt_ledger_path
from ultra_deepagents.progress_guard import (
    AttemptLedger,
    ProgressStallDetector,
    read_attempt_ledger_digest,
)


def _exec_pair(
    call_id: str,
    command: str,
    *,
    output: str = "",
    size: int | None = None,
    error: str = "",
    scope: str = "",
) -> list[dict[str, Any]]:
    started_payload: dict[str, Any] = {
        "tool_name": "execute",
        "status": "started",
        "tool_call_id": call_id,
        "input": {"command": command},
    }
    finished_payload: dict[str, Any] = {
        "tool_name": "execute",
        "status": "failed" if error else "completed",
        "tool_call_id": call_id,
        # Deliberately NO input on the finished event: the live tools protocol
        # often omits it, so detection must correlate via tool_call_id.
        "output_preview": output,
        "output_size_chars": len(output) if size is None else size,
    }
    if error:
        finished_payload["error"] = error
    if scope:
        started_payload["subagent_name"] = scope
        finished_payload["subagent_name"] = scope
    return [{"payload": started_payload}, {"payload": finished_payload}]


def _tool_completed(tool_name: str, *, scope: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "tool_name": tool_name,
        "status": "completed",
        "tool_call_id": f"{tool_name}-call",
    }
    if scope:
        payload["subagent_name"] = scope
    return {"payload": payload}


def _drive(detector: ProgressStallDetector, events: list[dict[str, Any]]):
    """Feed events; return the first verdict (or None)."""
    for event in events:
        observation = detector.observe(event)
        if observation is not None and observation.verdict is not None:
            return observation.verdict
    return None


# --- Fixture 1: the livelock (must fire) -------------------------------------


def test_fires_on_identical_failing_repeats() -> None:
    detector = ProgressStallDetector(threshold=12)
    events: list[dict[str, Any]] = []
    # 1 novel + 12 identical repeats with unchanged failing output.
    for index in range(13):
        events.extend(
            _exec_pair(
                f"c{index}",
                "python run_experiment.py --all",
                output="",
                error="results.csv is empty",
            )
        )
    verdict = _drive(detector, events)
    assert verdict is not None
    assert verdict.stall_count == 12
    assert verdict.repeated_commands
    assert verdict.repeated_commands[0][0] == "python run_experiment.py --all"


def test_fires_on_cycling_over_small_command_set() -> None:
    """The REAL livelock shape: not one command repeated back-to-back, but a
    CYCLE over a small set (measured: 114 execs / 22 distinct). Any consecutive-
    identical counter misses this; the guard counts non-progressing repeats."""
    detector = ProgressStallDetector(threshold=12)
    commands = [
        "ls results/",
        "cat results/results.csv",
        "python run_experiment.py",
        "python check_results.py",
    ]
    events: list[dict[str, Any]] = []
    call = 0
    for cycle in range(5):  # cycle 0 is all-novel; cycles 1+ are pure repeats
        for command in commands:
            events.extend(
                _exec_pair(f"c{call}", command, output=f"static output of {command}")
            )
            call += 1
    verdict = _drive(detector, events)
    assert verdict is not None
    # 4 cycles x 4 repeated commands = 16 non-progressing execs; fires at 12.
    assert verdict.stall_count == 12


# --- Fixture 2: edit-and-rerun debugging (must NEVER fire) -------------------


def test_edit_and_rerun_debugging_never_fires() -> None:
    detector = ProgressStallDetector(threshold=12)
    events: list[dict[str, Any]] = []
    for index in range(30):
        events.extend(
            _exec_pair(f"c{index}", "python x.py", output="", error="Traceback: KeyError")
        )
        events.append(_tool_completed("edit_file"))
    assert _drive(detector, events) is None


def test_write_file_and_task_reset_the_counter() -> None:
    for progress_tool in ("write_file", "task"):
        detector = ProgressStallDetector(threshold=3)
        events: list[dict[str, Any]] = []
        for index in range(20):
            events.extend(_exec_pair(f"c{index}", "python x.py", output="same"))
            if index % 2 == 1:
                events.append(_tool_completed(progress_tool))
        assert _drive(detector, events) is None, progress_tool


# --- Fixture 3: polling with moving output (must NEVER fire) -----------------


def test_polling_with_changing_output_never_fires() -> None:
    detector = ProgressStallDetector(threshold=12)
    events: list[dict[str, Any]] = []
    for index in range(40):
        events.extend(
            _exec_pair(f"c{index}", "squeue -j 42", output=f"job 42 running {index}s")
        )
    assert _drive(detector, events) is None


def test_output_size_change_beyond_preview_counts_as_progress() -> None:
    """A growing log whose PREVIEW is identical (truncation) still reads as
    progress via the untruncated size channel."""
    detector = ProgressStallDetector(threshold=12)
    events: list[dict[str, Any]] = []
    for index in range(40):
        events.extend(
            _exec_pair(f"c{index}", "tail training.log", output="epoch...", size=1000 + index)
        )
    assert _drive(detector, events) is None


# --- Fixture 4: healthy exploration vs measured livelock shape ----------------


def test_healthy_window_shape_never_fires_and_livelock_shape_fires() -> None:
    """Replay of the measured run's two windows. Healthy 12:20 window: 52 execs /
    39 distinct (few, short repeat streaks). Livelocked 14:00 window: 114 execs /
    22 distinct, cycling with unchanged output."""
    healthy = ProgressStallDetector(threshold=12)
    events: list[dict[str, Any]] = []
    call = 0
    # 39 distinct commands; 13 of them re-checked once with unchanged output,
    # interleaved (worst healthy streak well below threshold).
    for index in range(39):
        events.extend(_exec_pair(f"h{call}", f"cmd-{index}", output=f"out-{index}"))
        call += 1
        if index % 3 == 0:
            events.extend(_exec_pair(f"h{call}", f"cmd-{index}", output=f"out-{index}"))
            call += 1
    assert _drive(healthy, events) is None

    livelocked = ProgressStallDetector(threshold=12)
    events = []
    call = 0
    for index in range(22):  # first pass: all novel
        events.extend(_exec_pair(f"l{call}", f"cmd-{index}", output="unchanged"))
        call += 1
    fired = None
    for _cycle in range(5):  # 92 more execs cycling the same 22, output unchanged
        for index in range(22):
            events.extend(_exec_pair(f"l{call}", f"cmd-{index}", output="unchanged"))
            call += 1
    fired = _drive(livelocked, events)
    assert fired is not None
    assert fired.stall_count == 12


# --- Scope isolation ----------------------------------------------------------


def test_scope_isolation_healthy_coordinator_cannot_mask_stuck_subagent() -> None:
    detector = ProgressStallDetector(threshold=6)
    events: list[dict[str, Any]] = []
    for index in range(10):
        # Coordinator makes genuine progress...
        events.extend(_exec_pair(f"coord-{index}", f"novel-{index}", output=f"o{index}"))
        # ...while code-runner re-runs the same failing command.
        events.extend(
            _exec_pair(
                f"cr-{index}",
                "python broken.py",
                output="",
                error="empty",
                scope="code-runner",
            )
        )
    verdict = _drive(detector, events)
    assert verdict is not None
    assert verdict.scope == "code-runner"


# --- Guard mechanics ----------------------------------------------------------


def test_threshold_zero_disables() -> None:
    detector = ProgressStallDetector(threshold=0)
    events: list[dict[str, Any]] = []
    for index in range(50):
        events.extend(_exec_pair(f"c{index}", "python x.py", output="same"))
    assert _drive(detector, events) is None
    assert not detector.enabled


def test_verdict_rearms_for_the_next_window() -> None:
    detector = ProgressStallDetector(threshold=3)
    events: list[dict[str, Any]] = []
    for index in range(8):
        events.extend(_exec_pair(f"c{index}", "python x.py", output="same"))
    verdicts = []
    for event in events:
        observation = detector.observe(event)
        if observation is not None and observation.verdict is not None:
            verdicts.append(observation.verdict)
    # 1 novel + 7 repeats -> fires at repeat 3 and again at repeat 6.
    assert len(verdicts) == 2


def test_exec_without_any_input_stays_neutral() -> None:
    detector = ProgressStallDetector(threshold=2)
    finished_only = {
        "payload": {
            "tool_name": "execute",
            "status": "completed",
            "tool_call_id": "orphan",
            "output_preview": "same",
        }
    }
    for _ in range(10):
        observation = detector.observe(dict(finished_only))
        assert observation is None or observation.verdict is None


# --- Attempt ledger + middleware ----------------------------------------------


def test_ledger_records_and_digest_aggregates(tmp_path: Path) -> None:
    ledger = AttemptLedger(tmp_path / ".ultra" / "attempt_ledger.jsonl")
    for _ in range(7):
        ledger.record(
            scope="code-runner",
            command_preview="python run_experiment.py",
            kind="stagnant_repeat",
            detail="results.csv is empty",
        )
    ledger.record(scope="", command_preview="python other.py", kind="error", detail="Traceback")
    digest = read_attempt_ledger_digest(ledger.path)
    assert "ATTEMPT LEDGER" in digest
    assert "7x: python run_experiment.py" in digest
    assert "results.csv is empty" in digest
    assert "1x: python other.py" in digest
    # Advisory wording, never a hard prohibition.
    assert "Re-running after a real change is fine." in digest


def test_ledger_digest_empty_when_no_entries(tmp_path: Path) -> None:
    assert read_attempt_ledger_digest(tmp_path / "missing.jsonl") == ""


def test_ledger_bounds_entries(tmp_path: Path) -> None:
    ledger = AttemptLedger(tmp_path / "ledger.jsonl")
    for index in range(1000):
        ledger.record(scope="", command_preview=f"cmd-{index}", kind="error", detail="")
    lines = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 500


def test_ledger_write_failure_never_raises(tmp_path: Path) -> None:
    target = tmp_path / "blocked"
    target.write_text("a file, not a directory", encoding="utf-8")
    ledger = AttemptLedger(target / "ledger.jsonl")  # parent is a file -> OSError
    ledger.record(scope="", command_preview="cmd", kind="error", detail="")  # no raise


def test_middleware_appends_digest_to_system_message(tmp_path: Path) -> None:
    ledger_path = attempt_ledger_path(tmp_path)
    AttemptLedger(ledger_path).record(
        scope="",
        command_preview="python run_experiment.py",
        kind="stagnant_repeat",
        detail="results.csv is empty",
    )
    middleware = UltraAttemptLedgerMiddleware(ledger_path)
    captured: list[Any] = []

    def handler(request: ModelRequest) -> ModelResponse:
        captured.append(request.system_message)
        return ModelResponse(result=[AIMessage(content="ok")])

    request = ModelRequest(
        model=cast(BaseChatModel, object()),
        messages=[],
        system_message=SystemMessage(content="base prompt"),
        runtime=cast(Any, None),
    )
    middleware.wrap_model_call(request, handler)
    text = str(captured[0].content)
    assert "base prompt" in text
    assert "ATTEMPT LEDGER" in text
    assert "python run_experiment.py" in text


def test_middleware_no_ledger_no_prompt_change(tmp_path: Path) -> None:
    middleware = UltraAttemptLedgerMiddleware(attempt_ledger_path(tmp_path))
    captured: list[Any] = []

    def handler(request: ModelRequest) -> ModelResponse:
        captured.append(request.system_message)
        return ModelResponse(result=[AIMessage(content="ok")])

    request = ModelRequest(
        model=cast(BaseChatModel, object()),
        messages=[],
        system_message=SystemMessage(content="base prompt"),
        runtime=cast(Any, None),
    )
    middleware.wrap_model_call(request, handler)
    assert str(captured[0].content) == "base prompt"


def test_middleware_digest_cache_refreshes_on_write(tmp_path: Path) -> None:
    ledger_path = attempt_ledger_path(tmp_path)
    ledger = AttemptLedger(ledger_path)
    ledger.record(scope="", command_preview="cmd-a", kind="error", detail="")
    middleware = UltraAttemptLedgerMiddleware(ledger_path)
    assert "cmd-a" in middleware._digest()
    ledger.record(scope="", command_preview="cmd-b", kind="error", detail="")
    # mtime moved -> re-derived.
    refreshed = middleware._digest()
    assert "cmd-b" in refreshed


def test_ledger_entries_are_json_lines(tmp_path: Path) -> None:
    ledger = AttemptLedger(tmp_path / "ledger.jsonl")
    ledger.record(
        scope="code-runner",
        command_preview="python x.py",
        kind="stagnant_repeat",
        detail="empty",
    )
    entry = json.loads((tmp_path / "ledger.jsonl").read_text(encoding="utf-8").strip())
    assert entry == {
        "scope": "code-runner",
        "command": "python x.py",
        "kind": "stagnant_repeat",
        "detail": "empty",
    }
