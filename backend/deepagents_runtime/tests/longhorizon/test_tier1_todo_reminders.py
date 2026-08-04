"""Tier-1 scenario: todo state-echo + staleness nudge through the real stack.

Proves the three mechanism claims of ``todo_reminders.py`` end-to-end
(run_job -> build_research_agent -> deepagents graph, scripted model):

1. after the model writes a plan, every subsequent coordinator request carries
   the CURRENT todo list (state-derived echo, updated statuses included);
2. once the configured number of tool results lands without a write_todos
   update, the request additionally carries the staleness nudge — and not
   one turn earlier;
3. the echo survives compaction: after SummarizationMiddleware rewrites the
   conversation, the plan still reaches the model via the per-request prompt.

The scripted policy also *reacts* to the nudge with a status-update rewrite,
which lets us assert the echo reflects the NEW statuses on later turns —
the state-derived (not history-derived) proof.
"""

from __future__ import annotations

from deepagents.backends.protocol import ExecuteResponse
from longhorizon_harness import (
    CompressedConfig,
    LongHorizonWorld,
    TurnDecision,
    TurnRequest,
)
from longhorizon_invariants import (
    assert_compaction_cycled,
    assert_event_stream_integrity,
    assert_terminal_success,
    summary_calls,
    turn_calls,
)

ECHO_HEADER = "## Current todo list (write_todos state)"
NUDGE_MARKER = "statuses above may be stale"

_PLAN = [
    {"content": "Survey diagnostic fixtures", "status": "in_progress"},
    {"content": "Run the staged diagnostic sequence", "status": "pending"},
    {"content": "Report the outcome", "status": "pending"},
]

_UPDATED = [
    {"content": "Survey diagnostic fixtures", "status": "completed"},
    {"content": "Run the staged diagnostic sequence", "status": "in_progress"},
    {"content": "Report the outcome", "status": "pending"},
]


def _todo_pipeline_policy(rounds: int, final_answer: str):
    state = {"planned": False, "updated": False, "stage": 0}

    def policy(request: TurnRequest) -> TurnDecision:
        if not state["planned"]:
            state["planned"] = True
            return TurnDecision(
                text="Planning first.",
                tool_name="write_todos",
                tool_args={"todos": _PLAN},
            )
        if NUDGE_MARKER in request.full_text and not state["updated"]:
            state["updated"] = True
            return TurnDecision(
                text="Nudge received; reconciling statuses.",
                tool_name="write_todos",
                tool_args={"todos": _UPDATED},
            )
        if state["stage"] < rounds:
            state["stage"] += 1
            return TurnDecision(execute_command=f"python stage.py --index {state['stage']}")
        return TurnDecision(text=final_answer)

    return policy


def _chunky_output(command: str, nth: int) -> ExecuteResponse:
    """Verbose per-call output so the 24k sawtooth window actually compacts
    within the scenario's turn budget (same shape as the compaction tests)."""
    filler = f"stage metrics for call {nth}: " + ("telemetry sample; " * 50)
    return ExecuteResponse(output=f"{filler}\nchecksum {nth:08d}", exit_code=0)


def test_todo_echo_nudge_and_compaction_survival(tmp_path) -> None:
    world = LongHorizonWorld(tmp=tmp_path, behavior=_chunky_output)
    # Sawtooth compaction regime (window well above fixed overhead) with enough
    # rounds to cross the 12-tool-result staleness threshold both before and
    # after the mid-run status rewrite.
    config = CompressedConfig(context_window_tokens=24_000)
    policy = _todo_pipeline_policy(
        rounds=34,
        final_answer="Diagnostic sequence finished: 34 stages nominal, plan reconciled.",
    )
    goal = "Run the staged diagnostic sequence to completion and state the outcome plainly."

    response = world.run_sync(policy, goal=goal, config=config)

    assert "34 stages nominal" in response
    assert_terminal_success(world.log)
    assert_event_stream_integrity(world.log)
    assert_compaction_cycled(world, minimum_cycles=1)

    turns = turn_calls(world)
    plan_index = next(
        record.index for record in turns if "Survey diagnostic fixtures" not in record.prompt_text
    )
    # Claim 1: before the plan there is no echo; after it, every coordinator
    # turn carries the current list.
    assert ECHO_HEADER not in turns[0].prompt_text
    post_plan = [record for record in turns if record.index > plan_index]
    assert post_plan, "scenario must run turns after the plan"
    for record in post_plan:
        assert ECHO_HEADER in record.prompt_text, (
            f"turn {record.index} lost the todo echo"
        )

    # Claim 2: the nudge fires exactly at the configured staleness window,
    # never before.
    nudged = [record for record in post_plan if NUDGE_MARKER in record.prompt_text]
    assert nudged, "staleness nudge never reached the model"
    first_nudged = min(nudged, key=lambda record: record.index)
    for record in post_plan:
        if record.index < first_nudged.index:
            assert NUDGE_MARKER not in record.prompt_text
    assert first_nudged.tool_result_count >= 12

    # Claim 3 (state-derived echo): after the scripted status rewrite the echo
    # shows the NEW statuses on later turns.
    reconciled = [
        record
        for record in post_plan
        if "- [completed] Survey diagnostic fixtures" in record.prompt_text
    ]
    assert reconciled, "echo never reflected the updated statuses"

    # Claim 3 (compaction survival): turns AFTER the last summarization still
    # carry the echo — the plan's transport is the per-request prompt, not the
    # compacted message history.
    summaries = summary_calls(world)
    assert summaries, "scenario must compact at least once"
    last_summary_index = max(record.index for record in summaries)
    post_compaction_turns = [
        record for record in turns if record.index > last_summary_index
    ]
    assert post_compaction_turns, "scenario must keep working after compaction"
    for record in post_compaction_turns:
        assert ECHO_HEADER in record.prompt_text, (
            f"turn {record.index} lost the todo echo after compaction"
        )
