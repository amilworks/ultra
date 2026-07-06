---
name: go-nats-development
description: How to carry out Go and NATS/JetStream development on the ultra control plane (backend/controlplane) and its messaging contract with the Python deepagents workers. Use whenever writing, changing, or reviewing Go code in this repo; touching NATS subjects, streams, consumers, or publish paths; working on run-event ingest, run leases, recovery, or cancellation; editing the store layer (schema.sql, queries.sql, sqlc, migrations); adding or changing /v2 endpoints or backend/controlplane/api/openapi.yaml; or changing anything the Python worker publishes or consumes (nats_worker.py, events.py). Use it even for "small" changes — this codebase has strict cross-language invariants (partition ordering, idempotency keys, retry budgets, sequencer discipline) that break silently and wedge production partitions. Also use when debugging stuck runs, wedged event partitions, duplicate or lost events, or lease/recovery misbehavior.
---

# Go + NATS Development (pointer)

The canonical, fact-checked content for this skill lives in the repo's shared
skill directory and is maintained in one place to avoid drift:

- `.claude/skills/go-nats-development/SKILL.md` — the golden rules
- `.claude/skills/go-nats-development/references/architecture.md`
- `.claude/skills/go-nats-development/references/nats-jetstream.md`
- `.claude/skills/go-nats-development/references/worker-contract.md`
- `.claude/skills/go-nats-development/references/lessons.md`

Read the canonical SKILL.md now, then open the reference file that matches
your task (architecture for control-plane structure, nats-jetstream for
stream/consumer semantics, worker-contract for the Go/Python messaging
contract, lessons for known failure modes). Treat that content as
authoritative over this stub.

Related specialist memory: `.codex/memory/ultra-nats-expert.md`. Related
agents: `ultra_nats_expert`, `ultra_control_plane`.
