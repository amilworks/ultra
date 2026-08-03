"""Tier-2 chaos scenarios: the real worker + real JetStream under faults.

Requires a docker daemon (a throwaway ``nats -js`` container per session);
skipped cleanly where docker is unavailable. Model + sandbox remain the
deterministic Tier-1 fakes — every ack, NAK, redelivery, sequence, checkpoint,
and terminal event below is produced by production code.

Scenarios:
- a job flows jobs-subject -> durable pull consumer -> run_job -> partitioned
  event subjects -> ack, with per-run sequences gapless and usage accounted
- a worker dies mid-run (shutdown-cancel => the worker's own NAK path), the
  message is redelivered, and a FRESH worker instance resumes from the durable
  checkpoint with every pipeline stage executed exactly once
- a duplicate delivery during an active run is parried (NAK with delay), and
  its post-completion redelivery is skipped via the control-plane status
  cooperation instead of re-executing the finished run
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from deepagents.backends.protocol import ExecuteResponse
from longhorizon_harness import TurnDecision, TurnRequest
from longhorizon_invariants import (
    assert_event_stream_integrity,
    assert_terminal_success,
)
from longhorizon_nats import (
    ChaosControlPlane,
    ChaosNamespace,
    ChaosWorker,
    ChaosWorld,
    EventCollector,
    NatsServerContainer,
    chaos_settings,
    consumer_state,
    docker_available,
    publish_job,
    start_worker,
    stop_worker,
)
from unittest import mock

from ultra_deepagents.schemas import RunJobEnvelope

pytestmark = [
    pytest.mark.tier2_chaos,
    pytest.mark.skipif(not docker_available(), reason="tier-2 chaos needs a docker daemon"),
]


@pytest.fixture(scope="session")
def nats_server():
    server = NatsServerContainer()
    server.start()
    try:
        asyncio.run(server.wait_ready())
        yield server
    finally:
        server.stop()


def _staged_policy(world: ChaosWorld, rounds: int, final_answer: str):
    def policy(request: TurnRequest) -> TurnDecision:
        next_stage = len(world.sandbox.calls) + 1
        if next_stage <= rounds:
            return TurnDecision(execute_command=f"python stage.py --index {next_stage}")
        return TurnDecision(text=final_answer)

    return policy


def _sandbox_patch(world: ChaosWorld):
    return mock.patch(
        "ultra_deepagents.agent.build_sandbox_backend",
        return_value=world.sandbox,
    )


def test_job_flows_through_real_jetstream_and_completes(tmp_path, nats_server) -> None:
    namespace = ChaosNamespace.fresh()
    settings = chaos_settings(tmp_path, nats_server, namespace)
    world = ChaosWorld(tmp=tmp_path)
    job = RunJobEnvelope(
        run_id="run-chaos-happy",
        thread_id="thread-chaos",
        user_id="chaos-tester",
        goal="Run the staged pipeline to completion and state the outcome plainly.",
    )

    async def scenario():
        collector = EventCollector(nats_server.url, namespace)
        await collector.start()
        control_plane = ChaosControlPlane(collector)
        worker = ChaosWorker(
            settings,
            world=world,
            collector=collector,
            control_plane=control_plane,
            policy=_staged_policy(world, 8, "Pipeline done: 8 stages nominal."),
        )
        worker_task = await start_worker(worker)
        try:
            await publish_job(nats_server, settings, job)
            await collector.wait_for(
                lambda c: c.of_kind(job.run_id, "run.completed"),
                timeout=60,
                description="run.completed on the event stream",
            )
            await asyncio.sleep(0.3)  # let the collector drain trailing events
            return collector.to_event_log(job.run_id)
        finally:
            await stop_worker(worker_task)
            await collector.stop()

    with _sandbox_patch(world):
        log = asyncio.run(scenario())

    assert world.sandbox.calls == [f"python stage.py --index {n}" for n in range(1, 9)]
    assert_event_stream_integrity(log)
    assert_terminal_success(log)
    ack_pending, redelivered = asyncio.run(consumer_state(nats_server, settings))
    assert ack_pending == 0, "job message was not acked after terminal completion"
    assert redelivered == 0, "happy path saw unexpected redeliveries"


def test_worker_death_redelivery_and_fresh_worker_resume(tmp_path, nats_server) -> None:
    namespace = ChaosNamespace.fresh()
    settings = chaos_settings(tmp_path, nats_server, namespace)
    world = ChaosWorld(tmp=tmp_path)
    job = RunJobEnvelope(
        run_id="run-chaos-resume",
        thread_id="thread-chaos",
        user_id="chaos-tester",
        goal="Run the staged pipeline to completion and state the outcome plainly.",
    )
    reached_stage_six = threading.Event()

    def policy(request: TurnRequest) -> TurnDecision:
        next_stage = len(world.sandbox.calls) + 1
        if next_stage <= 12:
            if next_stage == 6 and request.invocation == 1:
                # Signal the test, then stall inside the MODEL node: the kill
                # lands here, so stages 1-5 are checkpointed and stage 6 has not
                # executed — the resume must replay only this model step.
                reached_stage_six.set()
                return TurnDecision(
                    execute_command="python stage.py --index 6",
                    sleep_seconds=5.0,
                )
            return TurnDecision(execute_command=f"python stage.py --index {next_stage}")
        return TurnDecision(text="Pipeline finished after resume: 12 stages, each executed once.")

    async def scenario() -> None:
        collector = EventCollector(nats_server.url, namespace)
        await collector.start()
        control_plane = ChaosControlPlane(collector)

        worker_one = ChaosWorker(
            settings,
            world=world,
            collector=collector,
            control_plane=control_plane,
            policy=policy,
        )
        worker_one_task = await start_worker(worker_one)
        await publish_job(nats_server, settings, job)

        deadline = time.monotonic() + 30
        while not reached_stage_six.is_set():
            if time.monotonic() > deadline:
                raise AssertionError("run never reached stage 6 on worker one")
            await asyncio.sleep(0.05)
        # Kill worker one mid-model-call. Cancellation drives the worker's own
        # shutdown classification: should_ack=False -> NAK -> redelivery.
        await stop_worker(worker_one_task)

        world.invocation = 2
        worker_two = ChaosWorker(
            settings,
            world=world,
            collector=collector,
            control_plane=control_plane,
            policy=policy,
        )
        worker_two_task = await start_worker(worker_two)
        try:
            await collector.wait_for(
                lambda c: c.of_kind(job.run_id, "run.completed"),
                timeout=60,
                description="run.completed after redelivery to the fresh worker",
            )
            await asyncio.sleep(0.3)  # let the collector drain trailing events
        finally:
            await stop_worker(worker_two_task)

        log = collector.to_event_log(job.run_id)
        assert collector.of_kind(job.run_id, "run.resumed"), (
            "the fresh worker restarted the run from scratch instead of resuming "
            f"the durable checkpoint (kinds: {collector.kinds(job.run_id)})"
        )
        assert_event_stream_integrity(log)
        assert_terminal_success(log)
        await collector.stop()

    with _sandbox_patch(world):
        asyncio.run(scenario())

    # The heart of the claim: twelve stages, each executed exactly once, across
    # a worker death + JetStream redelivery + checkpoint resume.
    assert world.sandbox.calls == [f"python stage.py --index {n}" for n in range(1, 13)]

    ack_pending, _redelivered = asyncio.run(consumer_state(nats_server, settings))
    # num_redelivered is a gauge of UNACKED redelivered messages, so it reads 0
    # again after the resumed run acks; the redelivery itself is evidenced by
    # run.resumed above (a fresh worker can only resume a redelivered job).
    assert ack_pending == 0, "redelivered job was never terminally acked"


def test_duplicate_delivery_parried_and_post_completion_redelivery_skipped(
    tmp_path, nats_server
) -> None:
    namespace = ChaosNamespace.fresh()
    # Small ack_wait => duplicate-NAK delay (ack_wait/2) stays test-sized.
    settings = chaos_settings(tmp_path, nats_server, namespace, worker_ack_wait_seconds=8.0)

    def slow_behavior(command: str, nth: int) -> ExecuteResponse:
        time.sleep(0.25)  # keep the run alive long enough to meet its duplicate
        return ExecuteResponse(output=f"ok #{nth}: {command}", exit_code=0)

    world = ChaosWorld(tmp=tmp_path)
    world.sandbox._behavior = slow_behavior
    job = RunJobEnvelope(
        run_id="run-chaos-duplicate",
        thread_id="thread-chaos",
        user_id="chaos-tester",
        goal="Run the staged pipeline to completion and state the outcome plainly.",
    )

    async def scenario() -> None:
        collector = EventCollector(nats_server.url, namespace)
        await collector.start()
        control_plane = ChaosControlPlane(collector)
        worker = ChaosWorker(
            settings,
            world=world,
            collector=collector,
            control_plane=control_plane,
            policy=_staged_policy(world, 6, "Pipeline done: 6 stages nominal."),
        )
        worker_task = await start_worker(worker)
        try:
            await publish_job(nats_server, settings, job, copies=2)
            await collector.wait_for(
                lambda c: c.of_kind(job.run_id, "run.completed"),
                timeout=60,
                description="run.completed despite the duplicate delivery",
            )
            # The duplicate is NAK'd with ~ack_wait/2 delay; after completion it
            # redelivers and must be SKIPPED via control-plane status, never
            # re-executed.
            await collector.wait_for(
                lambda c: c.of_kind(job.run_id, "run.worker_skipped"),
                timeout=30,
                description="run.worker_skipped for the post-completion redelivery",
            )
        finally:
            await stop_worker(worker_task)
            await collector.stop()

        assert len(collector.of_kind(job.run_id, "run.started")) == 1, (
            "the duplicate delivery entered compute — exactly-once was violated"
        )
        assert len(collector.of_kind(job.run_id, "run.completed")) == 1

    with _sandbox_patch(world):
        asyncio.run(scenario())

    assert world.sandbox.calls == [f"python stage.py --index {n}" for n in range(1, 7)], (
        "the duplicate delivery re-executed pipeline stages"
    )
    # The run.skipped event above is the redelivery evidence (only a redelivered
    # duplicate reaches the skip path); num_redelivered is an unacked gauge and
    # reads 0 again once the skip acks.
    ack_pending, _redelivered = asyncio.run(consumer_state(nats_server, settings))
    assert ack_pending == 0, "a delivery was left unacked after skip handling"
