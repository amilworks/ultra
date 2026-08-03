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

# Tier 2 — chaos on real JetStream

`longhorizon_nats.py` + `test_tier2_chaos_nats.py` move the trust boundary out
one ring: jobs flow through a REAL dockerized NATS JetStream into the REAL
`NATSDeepAgentsWorker` consume loop (durable pull consumer, ack/NAK/extension,
duplicate guards, worker terminal events), with the Tier-1 scripted model and
sandbox at the edges and a thin `ChaosControlPlane` answering the worker's own
injection seams. Needs a docker daemon; marked `tier2_chaos` and self-skipping.

| Scenario | Claim under test |
|---|---|
| happy path | job → consumer → run → partitioned events → ack, gapless, no redeliveries |
| worker death | shutdown NAK → JetStream redelivery → fresh worker resumes from durable checkpoint, 12 stages exactly once |
| duplicate delivery | active-run duplicate parried (delayed NAK); post-completion redelivery `run.worker_skipped` via control-plane status, never re-executed |
| NATS server restart | broker bounced mid-run (`docker restart`, JetStream state persists); worker recovers via reconnect or NAK→redelivery→resume; every stage present, ≤1 replayed |
| cancel subject | mid-run cancel message → prompt abort, `run.canceled` with the caller's reason, compute stops (no later stages), delivery acked |
| lease loss | scripted 409 on keepalive renewal (the one kill a worker trusts) → prompt abort, silent NAK (no user-facing terminal), redelivery under a fresh lease tenure resumes and completes |

First blood: the worker-death scenario found a real production bug — the
shutdown path's immediate NAK raced the dying worker's own outstanding pull
request, so JetStream redelivered into the doomed buffer and the run stayed
checked out to a ghost until AckWait expired (5 minutes at production
settings), with a 30s DrainTimeoutError on every such shutdown. Fixed by
delaying the shutdown NAK past the pull max-wait
(`_SHUTDOWN_NAK_DELAY_SECONDS`); teardown is now instant and the redelivery
reaches a live worker.

Gotchas encoded in the harness: control-plane status vocabulary is
`succeeded` (not `completed`); the skip terminal event is
`run.worker_skipped`; `num_redelivered` is a gauge of UNACKED redeliveries
(reads 0 after ack), so redelivery evidence comes from `run.resumed` /
`run.worker_skipped` events instead.
