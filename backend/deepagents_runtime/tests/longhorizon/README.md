# Tier 1 — compressed-horizon harness

Tests the long-horizon claim ("Ultra can work a problem for hours/days/weeks")
at the timescale CI can afford: a "week" is driven as **counts** — coordinator
turns, compaction cycles, checkpoint writes, attempt/redelivery boundaries —
not wall-clock. The full suite runs in seconds and gates PRs.

## What is real, what is scripted

Real (production code under test): `run_job`'s attempt loop, the full
`build_research_agent` deepagents graph (middleware, subagents, skills routes,
system prompt), summarization/compaction, the progress-stall breaker, the
completion guard, the idle watchdog, the attempt ledger, event normalization +
the run sequencer, and the durable checkpointer.

Scripted (deterministic, at the outer edges only): the chat model
(`ScriptedChatModel`, policy-driven) and the sandbox (`ScriptedSandbox`, no
docker). Compression comes from `RuntimeSettings` knobs — never patched logic.

## Files

- `longhorizon_harness.py` — scripted model/sandbox, event log, `LongHorizonWorld`
  (simulated worker restarts over one shared durable store), `CompressedConfig`.
- `longhorizon_invariants.py` — the SLO clauses as assertions: gapless/contiguous
  event sequences, bounded conversation, compaction cycling, canary retention
  after compaction, exactly-once side effects, terminal honesty with usage.
- `test_tier1_compressed_horizon.py` — the scenarios.

## Scenarios

| Scenario | Claim under test |
|---|---|
| permanent-compaction regime | constraints survive summarize-every-turn churn |
| crash + redelivery resume | zero duplicated side effects, `run.resumed`, gapless sequences across the boundary |
| stall guard trips | livelock breaker fires, corrective prompt lands, ledger digest reaches the next attempt |
| healthy polling control | the breaker never false-positives on moving output |
| idle watchdog | a dead-silent stream is recovered, run still completes |
| week-scale soak | 400 turns / ~35 compaction cycles / ~1.3k events, all invariants, seconds of wall clock |

## Sizing note

The deepagents compaction trigger counts messages **plus the system prompt and
tool schemas**. With the full Ultra agent that fixed overhead is ~12–14k
approx-tokens: `context_window_tokens` above it produces the production
sawtooth; below it, a permanent-compaction stress regime. Both are covered
deliberately — see `CompressedConfig`'s docstring.

## Adding a scenario

Write a `TurnPolicy` (a function from `TurnRequest` to `TurnDecision`: answer
text, an `execute` command, a sleep, or a raised crash), pick a sandbox
`behavior`, run it through `LongHorizonWorld.run_sync`, and assert with the
invariant helpers. Derive policy progress from `world.sandbox.calls` (the
side-effect ledger), never from visible context — compaction rewrites context,
and resume replays it.
