# Ultra Orchestrator Memory

Purpose: persistent memory for the PARENT thread that runs the bounded swarm.
Read this at session start, before the first fan-out. Unlike the specialist
memories, this file is about how to orchestrate — fan-out sizing, brief
quality, sequencing, and repair-loop economics — not about any one subsystem.

## Orchestration Lessons

- Sharp briefs with named deliverables beat broad ones: "which function
  enforces lease ownership on event ingest, and which test proves it?"
  returns evidence; "look at the control plane" returns essays. Write the
  deliverable into every brief.
- Typical recon fan-out is 2-5 specialists; the thread ceiling is not the
  target. Every spawned agent is a report the parent must synthesize, and
  synthesis quality is the bottleneck, not parallelism.
- For debugging tasks, sequence `ultra_trace_debugger` first and hold other
  recon until its timeline and blast-radius map arrive; they determine which
  specialists are actually needed.
- Include `ultra_test_verifier` in recon, not only at the end — the narrowest
  gate per surface shapes how the plan is sliced.
- Exactly one writer (`ultra_implementer`), always; recon diffs are proposals.
  Freeze the diff before spawning reviewers or findings point at dead code.
- Blocking findings are correctness, security, data integrity, or
  replay/idempotency defects; everything else is advisory and goes in the
  report, not a repair loop. Two repair loops is the budget; blockers after
  two loops mean the plan is wrong, not the patch.
- Grant approval-gated items (secrets, auth policy, migrations, deletion
  behavior, deployment-critical config) per increment, in writing, in the
  plan, so the writer never guesses.
- Specialists never edit their own memory files; the parent or
  `ultra_implementer` applies accepted "Memory updates" — including to this
  file.

## Session Log

Append one dated bullet per significant swarm session: task shape, fan-out
used, what worked, what was wasted, repair loops spent, and any brief pattern
worth reusing.

- 2026-07-05: Memory initialized alongside the complex-software-development
  skill and the roster expansion to 13 agents (imaging pipeline, perf
  engineer). No swarm sessions logged yet against the new roster.
- 2026-07-05: Batch inference swarm used 7 requested recon agents plus
  implementer, reviewer, security, imaging, runtime, and NATS follow-up. The
  useful pattern was contracts/security recon before writing, then a narrow
  runtime-only shipment while export/frontend stayed approval-gated. Repair
  pressure came from review findings, not broad taste: worker-token omission,
  RareSpot terminal/cancel semantics, unsafe path segments, and tar streaming.
  Final remaining blockers were control-plane terminal-state monotonicity and
  `/outputs` cancel preconditions; those require named approval because they
  touch route/store/auth-policy behavior.
- 2026-07-05: Live prairie-dog/RareSpot trace used no fan-out after the
  initial timeline showed one active durable Deep Agents run was the whole
  blast radius. Reusable pattern: first distinguish `control_runs` from
  `/v2/data-agent/jobs`, then verify run status, lease freshness, sandbox CPU,
  command parameters, artifact promotion, artifact contents, and browser
  console separately. The run completed successfully with 50% tile overlap,
  while the only product issues were non-fatal BisQue metadata fallback noise
  and React duplicate-key errors during streaming.
