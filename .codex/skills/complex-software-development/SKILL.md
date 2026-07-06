---
name: complex-software-development
description: Method for carrying out complex software development in the Ultra repo — any multi-file or multi-system change, feature build, refactor, performance fix, or hard bug hunt across the Go control plane, Deep Agents runtime, React frontend, NATS/JetStream, or Postgres. Use this whenever a task touches auth, worker tokens, uploads, retention, NATS subjects/streams/consumers, run leases, run events, migrations, artifacts, or the frontend trace; whenever a change could plausibly break a component you are not editing or violate an invariant nobody wrote down; and whenever the work spans more than one file or subsystem or a mistake would be hard to notice — even when the request never says "complex". When in doubt, use it.
---

# Complex Software Development in Ultra

Ultra is frontier scientific-imaging infrastructure: a Go control plane
(`backend/controlplane`), a Python Deep Agents runtime
(`backend/deepagents_runtime`), a React frontend (`frontend/`), NATS/JetStream
between them, and Postgres as durable truth. Complex changes here fail in a
characteristic way: not from typos, but from unstated invariants, hidden
coupling across the Go/Python/TS boundary, replay and idempotency mistakes,
permission leaks, and regressions that produce no error — just a quietly wrong
trace, a duplicated event, or a file another tenant can read.

This skill is the engineering method. The orchestration rules — read-only
recon swarm, one plan, exactly one writer (`ultra_implementer`), read-only
review, at most two repair loops, narrow verification — live in
`.codex/README.md`. Follow them; this document tells you how to think inside
each phase. You run at xhigh reasoning effort: concentrate that depth where a
wrong call is expensive to reverse — plan shape, invariant extraction,
reviewing your own diff, debugging hypotheses — not uniformly across
mechanical edits.

## 1. Frame the task, then extract the invariants

Before spawning anything, write down in the parent thread: the **observable
outcome** (what a user, test, or database row will show when this is done),
the **blast-radius guess** (which of the five systems this plausibly touches),
and the **risk surfaces in scope** (see the tiers below). Naming the risk
surfaces now determines the reviewer roster later, so you do not discover
mid-review that a security pass was needed all along. If the task starts from
a misbehaving run, capture the concrete identifiers — run_id, thread_id,
worker_id — because durable state is the only reliable narrator of what
happened.

Then extract invariants. Most Ultra defects are violations of rules the code
enforces but no comment states. Read the code paths you will change and write
down, explicitly, the invariants they currently uphold. If you cannot state
them, you are not ready to edit — you would be preserving behavior you have
not identified.

Invariants that recur across this repo (verify against current code, do not
trust this list blindly):

- **At-least-once delivery is the baseline.** NATS redelivers; the control
  plane and `nats_worker.py` may see the same message twice. Correctness comes
  from idempotency keys, MsgId deduplication, durable event IDs, lease tokens,
  and terminal-state guards — never from assuming single delivery.
- **Per-run event order is by `sequence_number` in Postgres**, not by arrival
  order across subjects or workers. Any code that assumes cross-subject
  ordering is wrong until a concrete mechanism proves otherwise.
- **Terminal run events are immutable.** Once a run reaches a terminal state,
  late or replayed events must not resurrect or mutate it.
- **Leases gate the single active worker.** Dispatch, ack extension, lease
  renewal, and cancellation in `internal/runcontrol` and `nats_worker.py` form
  one state machine; changing any leg changes the others.
- **Ownership is the authorization unit.** User/org/project scoping and
  resource-ownership checks in `internal/httpapi` are the tenant boundary;
  worker-token routes are a separate, narrower trust channel and must never
  become a general-purpose bypass.
- **Token deltas never become durable frontend state.** Structural events and
  artifacts are the trust surface; raw token chatter must stay ephemeral or
  the UI degrades at millions of tokens.

Recon discipline for the fan-out:

- **Two to five specialists is typical.** Eight threads is the ceiling, not
  the target — every agent you spawn is a report you must synthesize, and
  past a point more recon lowers the quality of the plan.
- **Ask sharp questions with named deliverables** — "which function enforces
  lease ownership on event ingest, and which test proves it?" — not "look at
  the control plane". Ask each specialist to return invariants, not just file
  lists: `ultra_architect` exists to name boundaries and invariants,
  `ultra_nats_expert` for delivery/ordering semantics (it reads its memory in
  `.codex/memory/` and consults the `natsDocs` MCP server),
  `ultra_security_data` for the permission boundary.
- **Include `ultra_test_verifier` in recon, not just at the end.** Knowing the
  narrowest gate for each surface changes how you slice the plan: an increment
  you cannot cheaply prove is an increment shaped wrong.
- **For debugging tasks, sequence `ultra_trace_debugger` first** — hand it the
  run_id and let other recon wait for its timeline. Its blast-radius map tells
  you which other specialists you actually need; spawning them earlier is
  guessing.
- If a recon agent hands back a diff, treat it as a proposal. Only
  `ultra_implementer` writes.

Also find the **nearest working example** of what you are about to build.
Ultra has strong local conventions (sqlc query patterns, event emission
helpers, handler auth wrappers, frontend hydration paths). A change that
imitates the adjacent working code inherits its invariants for free; a novel
pattern must re-earn every one of them.

## 2. Risk-tier the surfaces you will touch

Not all code deserves equal paranoia. Tier the touched surfaces first, because
the tier determines how much recon, testing, and review each part of the
change needs. In this repo:

**Tier 1 — evidence-heavy, mistakes are silent or unrecoverable:**

- *Auth, sessions, worker tokens* (`internal/httpapi`, `internal/config`):
  a missing ownership check produces no failing test and no error — just
  cross-tenant access. Every change here needs a negative test: prove the
  wrong principal is rejected, not only that the right one succeeds.
- *Uploads, staging, retention, deletion* (bundle upload handlers, storage
  paths, sandbox staging in the runtime): path traversal, host-path leaks, and
  premature deletion destroy scientific data that cannot be regenerated.
  Reason about symlinks, `..`, and partial-failure cleanup explicitly.
- *NATS subjects, streams, consumers* (`internal/eventbus`, `nats_worker.py`):
  renaming a subject (e.g. `ultra.runs.cancel`), changing a durable consumer,
  or altering ack behavior can strand in-flight messages or double-process
  runs. Any change here must explain redelivery, duplicate collapse, and
  replay before it merges — pull `ultra_nats_expert` in, always.
- *Run lifecycle: leases, dispatch markers, cancellation, terminal state*
  (`internal/runcontrol`): errors here appear as stuck or duplicated runs
  hours later, under load, never in unit tests that mock time.
- *Migrations* (`backend/controlplane/migrations/`, applied via
  `make control-migrate`): numbered up/down pairs. Prefer additive migrations;
  a destructive change needs an explicit story for data already in production
  and a real down path. Migrations couple to the sqlc queries in
  `internal/store` — change them together or the build lies to you.
- *Secrets and credentials* (WorkOS, BisQue, model, worker tokens): never in
  code, logs, tool output, or error messages.

Per the `ultra_implementer` contract, changes to secrets, credentials,
destructive storage paths, auth policy, migrations, deployment-critical
config, or data-deletion behavior need explicit approval in the parent thread
before the writer touches them. Grant those approvals per increment, in
writing, in the plan, so the writer never has to guess — these are exactly
the surfaces where a confident agent does irreversible damage.

**Tier 2 — contract surfaces, drift is the failure mode:**

- *OpenAPI* (`backend/controlplane/api/openapi.yaml`): the spec, the generated
  Go bindings, and the frontend client (`frontend/src/lib/api.ts`) form one
  contract. Change any leg, regenerate with `make control-generate`, and check
  the other legs for drift — the compiler will not catch a frontend call
  shaped for the old response.
- *Event and trace shapes* (`events.py`, `live_trace.py`, run-event ingest in
  the control plane, frontend trace rendering): three languages consume one
  shape. A field rename that "works" in Python silently blanks a UI panel.
- *Runtime config* (`config.py`, `internal/config`, `docker-compose.yml`):
  env-var parsing drift between local stack and deployment fails only at
  startup on someone else's machine.

**Tier 3 — everything else:** normal engineering care, narrow tests, move on.
Do not spend Tier-1 ceremony on a copy change; the budget you save funds the
paranoia Tier 1 needs.

## 3. Plan as invariant-preserving increments

Synthesize recon into one plan in the parent thread before `ultra_implementer`
writes anything. Recon output is hypothesis, not fact: when two specialists
describe the same boundary differently, that seam is usually where the bug or
the design gap lives — read the contested file yourself before adjudicating,
and spot-check any claim the plan will load weight on.

A good plan states the invariants at risk and how each step preserves them,
then slices the change into ordered increments where each increment lands
alone, passes a named narrow gate, and leaves the tree consistent while later
increments are still pending. Slicing heuristics that fit Ultra:

- **Contracts before consumers.** Change
  `backend/controlplane/api/openapi.yaml` and regenerate
  (`make control-generate`) before touching `frontend/src/lib/api.ts`; land
  sqlc/store changes and `make control-migrate` before the Go handlers that
  need them. The generated layer is the ratchet that keeps both sides honest.
- **Additive before destructive.** New columns, event kinds, and subjects land
  first; removals are their own later increment after all consumers migrate.
  This is not caution for its own sake: JetStream is at-least-once with real
  retention, so events produced by old code *will* replay into your new code.
  Consumers must tolerate both shapes during the transition — which also
  means changing consumer tolerance before the producer whenever a payload
  shifts.
- **Backend before frontend.** The frontend hydrates from durable events and
  artifacts, so it can only be honestly verified against real backend output.
- **Keep each increment reviewable in one sitting.** A diff a reviewer cannot
  hold in their head produces reviews that are theater, and a sprawling patch
  exhausts the two repair loops on noise.

For every increment, write the acceptance evidence **before** implementation:
the exact command and the observation that proves it. This makes checkpoints
mechanical instead of improvised, and it forces you to notice unprovable
increments while re-slicing is still cheap.

Treat the plan as working memory for the whole session. Keep a decision
journal in it — "chose X over Y because Z" — and when the same fork recurs
hours later, read the entry instead of re-deriving it; reopen the decision
only if Z is no longer true. Re-anchor at every phase boundary (recon done,
plan approved, increment landed, review returned): reread the plan, mark what
is done, record surprises. If reality diverged, change the plan deliberately
and say so — the failure mode is not changing course, it is changing course
silently.

Scope the diff with one test: every hunk should trace to a plan step or a
named invariant; if you cannot say which, delete the hunk. "Production-grade"
here means replay-safe, idempotent, correctly auth-scoped, bounded in memory
and I/O for large datasets, and observable when it fails — not new
abstraction layers, configuration knobs nobody requested, or defensive
rewrites of adjacent working code. One writer is not bureaucracy: two agents
editing the same seam produce merge states neither reasoned about, and in a
system built on replay-safe handlers, an unreasoned intermediate state is a
live bug.

## 4. Tests: decide first-or-after with judgment, not ritual

Write the test **first** when the test is the falsifiable statement of the
thing you are changing:

- Bug fixes: reproduce the failure in a test before fixing it. A fix without
  a prior red test is a claim, not evidence — you may have fixed a different
  bug, or nothing.
- Replay/idempotency/lease semantics: encode the invariant (duplicate message
  collapses, terminal state rejects late events, lease renewal survives
  restart) as a test first, because these behaviors are invisible in a happy
  path and the test is the only place the invariant is written down.
- Permission boundaries: write the negative case first — the forbidden
  request that must 401/403/404 — because implementation naturally makes the
  positive case pass and nothing else forces the negative one to exist.

Test **after** is fine when existing coverage already pins the behavior:
mechanical refactors under green suites, UI layout shaping, log/message
wording. Forcing test-first there produces snapshot churn, not safety.

Map each risky part to the narrowest real gate:

- Go control plane: `make control-test`; Postgres+NATS semantics →
  `make control-integration`; run-lifecycle durability → `make control-soak`.
- Deep Agents runtime: `make deepagents-test`; worker transport, lease, and
  redelivery behavior → `make deepagents-worker-test`; agent routing/quality →
  `make deepagents-autonomy-test`.
- Frontend: `make frontend-lint frontend-type-check frontend-test-unit`
  (or `make frontend-quality`), `make frontend-test-smoke` when render paths
  change, and `make frontend-autonomy-test` for autonomous-chat recovery.
- Python quality: `make lint`, `make type-check`.
- Cross-cutting run-lifecycle changes: `make autonomy-gate` (the CI gate in
  `.github/workflows/autonomy-gate.yml`) is the honest local approximation;
  `make test-chat-stack` for prod-like chat-stack behavior.
- Touched `.codex/` config or agents: `python3 scripts/verify_codex_agents.py`
  — its contract (no pinned models, xhigh effort, depth 1, read-only roster,
  required memory files) is enforced, not advisory.

Delegate gate mapping to `ultra_test_verifier` when the change spans systems;
its job is to find the narrowest command set that proves the change, and to
say plainly when tests prove structure but not semantics.

## 5. Self-review, layered review, and the repair budget

Before spawning reviewers, read your own full diff as an adversary. This has
the best exchange rate in the workflow: a defect found now costs one edit;
the same defect found by a reviewer costs a spawned review plus one of your
two repair loops. Ask of the diff, concretely: What happens on redelivery or
replay? On a 10 GB input? Under two concurrent runs against the same
resource? For the wrong user? On partial failure halfway through? Did any
event shape change that the frontend hydrates? Did an error path start
swallowing the failure it used to surface? Fix what you find, then send the
reviewers a diff you already believe in — their job is to find what you could
not see, not what you did not look for.

Freeze the diff before review — no writer running while reviewers read, or
findings will point at code that no longer exists. Spawn the roster
`.codex/README.md` prescribes for the surfaces you touched, then triage the
findings yourself in the parent thread. Blocking means correctness, security,
data integrity, or replay/idempotency defects. Everything else is advisory
and goes into the final report, not into a repair pass — repairing advisory
nits burns your bounded loops on noise.

Each repair loop is one focused `ultra_implementer` pass scoped strictly to
the blocking findings, followed by re-running the affected gate and
re-reviewing only what changed. Two loops is a hard budget because repair
churn is a signal, not noise: if reviewers still find blockers after two
loops, the defect is in the plan, not the patch. Return to the plan with what
you learned, or report the residual risk honestly and stop — a third
mechanical loop does not add safety, it launders uncertainty into false
confidence.

## 6. Debugging: evidence over story

For hard bugs, the failure mode is narrative debugging — forming a plausible
story and patching where the symptom appears. Ultra's architecture punishes
this: the symptom (a stuck run, a duplicated step, a blank trace panel) is
usually three boundaries away from the cause.

- Work **backward from the observed bad value**, not forward from suspicion.
- Build the timeline from durable state first: Postgres rows
  (`control_runs`, `control_run_events` ordered by `sequence_number`,
  artifacts, token usage, leases, worker heartbeats), then NATS
  stream/consumer state, then logs. Durable state cannot lie about order the
  way your memory of the logs can.
- Trace the boundary chain in order: frontend request → Go handler →
  runcontrol → store → eventbus/NATS → worker lease → Python runtime →
  sandbox → artifact → frontend hydration. Find the first boundary where the
  data is already wrong; the bug is at or before it, and the fix belongs
  there — not at the boundary where you noticed it.
- Hold one root-cause hypothesis at a time and name the smallest observation
  that would falsify it. If you cannot name a falsifier, you have a story,
  not a hypothesis. If three hypotheses die in a row, stop generating new
  ones and rebuild the timeline — the boundary you mapped is probably wrong.
- For anything tied to a real run, hand `ultra_trace_debugger` the `run_id`
  and let it produce the timeline and blast-radius map before anyone proposes
  a fix. For delivery, ordering, or lag anomalies, pair it with
  `ultra_nats_expert`.
- Compare against the nearest working sibling (a run that succeeded, a
  handler that behaves) — diffing against a working example localizes faults
  faster than reading either side alone.

A fix earns merge only when you can state the root cause in one sentence, the
mechanism from cause to symptom, and the test that failed before the fix and
passes after. Never ship a symptom fix while the mechanism is unexplained —
in an at-least-once, replayed system, an unexplained fix usually relocates
the bug rather than removing it.

## 7. Verification, stopping, and the report

Distinguish three grades of evidence, and only report the first as done:

- **Real evidence:** a command you actually ran, its exit status, and the
  specific output line that bears on the change — a test that was red before
  and is green after, `make control-integration` passing after a NATS-touching
  change, regenerated bindings showing no drift after `make control-generate`.
- **Weak evidence:** a green full suite that never exercises the changed path;
  a typecheck pass on a behavioral change. Report it as what it is — a
  structural check — and say what it does not prove.
- **Non-evidence:** "should work", "tests pass" without naming which test
  exercises the change, or describing a command's expected output instead of
  its actual output. Fabricated or hand-waved verification is worse than none,
  because it discharges review attention that the change still needs.

Match evidence depth to the tier from step 2: Tier-1 surfaces need
integration or negative-case proof; Tier-3 needs a narrow unit test and a
lint pass.

Stop and report rather than loop when: both repair loops are spent; a
journaled assumption is falsified in a way that invalidates the plan shape;
verification requires resources you do not have (live stack, production
credentials, GPU nodes); or the fix keeps growing past the approved scope.
A precise report of a partial change with residual risks is a good outcome
for a complex task. A sprawling, low-confidence diff is not — someone will
trust it more than you do.

The final report states: files changed and why; commands run with their
**actual** results — never claim a gate you did not run, and if a needed gate
cannot run in this environment, say so plainly; invariants checked and how;
residual risks and untested paths; deferred advisory findings and
out-of-scope cleanups; and **Memory updates** — when a durable lesson emerged
about tracing or NATS, propose concise bullets for
`.codex/memory/ultra-trace-debugger.md` or
`.codex/memory/ultra-nats-expert.md`. Specialists never edit their own
memory; the parent or `ultra_implementer` applies accepted updates. An honest
"green on these gates, unverified on that path" protects the next session; a
false "verified" poisons it.
