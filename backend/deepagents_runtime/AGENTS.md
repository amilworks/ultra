# Deep Agents runtime agent contract

Apply these instructions to every file under `backend/deepagents_runtime/`.

## Load before editing

- Load `$complex-software-development` for runtime, worker, planner, sandbox, artifact, trace, or multi-file changes.
- Load `$go-nats-development` for `nats_worker.py`, `events.py`, leases, cancellation, subjects, envelopes, sequencing, retry/ACK behavior, or any Go↔Python contract.
- Load `$ultra-contract-change` for shared payload/config/schema changes.
- Load `$ultra-release-qualification` before selecting final gates.

## Authority and invariants

- Python executes leased work; it does not become product, run-status, scheduling, budget, or JetStream-topology authority.
- Every run event producer must use the run’s one shared `RunEventSequencer`. `event_id` remains the end-to-end idempotency key. Do not introduce a second counter or derive ordering from arrival time.
- Preserve delayed NAKs, bounded retry classes, reconnect supervision, lease-token fencing, checkpoint-before-ACK ordering, and cancellation settlement. A worker may inspect streams/consumers but must not create or mutate shared topology.
- Keep model prose outside authority. Typed plans, tool inputs/results, artifacts, host-written receipts, policy decisions, and durable events carry authority.
- Sandbox paths and resource locators are hostile input. Preserve no-follow traversal, run/user ownership, bounded reads/writes, immutable inputs, and cleanup after durable publication.
- Never emit credentials, cookies, DSNs, provider request bodies, host paths, restricted resource metadata, or raw evidence locators into traces or model-visible payloads.
- Token deltas are ephemeral. Durable trace events remain bounded and structural; large results become artifacts.

## Sources of truth

- Runtime package: `src/ultra_deepagents/`.
- Worker transport: `src/ultra_deepagents/nats_worker.py`.
- Run event construction/shape: `src/ultra_deepagents/events.py`; Go consumers live under `backend/controlplane/internal/runcontrol` and `internal/domain`.
- Configuration: `src/ultra_deepagents/config.py` plus matching Go/deployment environment consumers.
- Tests: `tests/`. Locked environment: this directory’s `pyproject.toml` and `uv.lock`.

## Change protocol

1. Find the producer, durable consumer, frontend projection, and replay test for every shared field.
2. State lease, retry, idempotency, sequencing, budget, and cancellation behavior before editing a worker path.
3. Add adversarial tests for duplicates, response loss, reconnect, stale leases, shutdown, malformed payloads, oversized outputs, and cross-run isolation as applicable.
4. Keep production paths real. A stub engine, fake model, memory checkpointer, or mocked transport cannot qualify the corresponding production behavior.
5. For scientific output, bind source identity, selectors, units, policy/runtime version, and exact output bytes; plausible prose or rendering is not verification.

## Commands

Run root Make targets from the repository root:

- Full runtime: `make deepagents-test`.
- Worker transport/lease/redelivery: `make deepagents-worker-test`.
- Agent routing/quality: `make deepagents-autonomy-test`.
- Smoke: `make deepagents-smoke`.
- Cross-system: `make autonomy-gate`.

For a focused test from this directory, use:

`uv run --python 3.11 --extra dev python -m pytest <exact paths or node ids>`

Confirm current node IDs before invoking them. Run root Ruff/MyPy targets from the repository root; this subproject’s dev extra is not the quality-tool authority.
