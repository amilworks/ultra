"""Mid-run steering (Phase 1) — worker-side contracts.

The invariant under test everywhere: a steer is applied exactly once per
state topology, across middleware injection, requeue-seeded transcripts,
continuation re-submission, and the finalization barrier — because every
copy carries the steer's message_id as the LangChain message id and
add_messages upserts by id.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langgraph.graph.message import add_messages

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.runner import (
    _effective_job_messages,
    _run_request_text,
    _steer_barrier_continuation,
)
from ultra_deepagents.schemas import RunJobEnvelope
from ultra_deepagents.steering import (
    STEER_CONTENT_PREFIX,
    SteeringInboxMiddleware,
    build_steering_inbox,
    steer_agent_content,
    steer_ids_in_messages,
)


class FakeInbox:
    def __init__(self, steers: list[dict[str, Any]], *, barrier: Any = ()) -> None:
        self.steers = steers
        self.acked: list[str] = []
        self.barrier_result = list(barrier) if barrier is not None else None
        self.barrier_calls = 0
        self.fetch_error = False

    @property
    def run_id(self) -> str:
        return "run_test"

    async def fetch(self) -> list[dict[str, Any]]:
        if self.fetch_error:
            return []  # the real inbox swallows errors into []
        return list(self.steers)

    async def ack(self, steer_id: str) -> bool:
        self.acked.append(steer_id)
        return True

    async def close_barrier(self) -> list[dict[str, Any]] | None:
        self.barrier_calls += 1
        return None if self.barrier_result is None else list(self.barrier_result)

    async def reopen_barrier(self) -> bool:
        return True


def steer(steer_id: str, message_id: str, content: str, status: str = "pending") -> dict[str, Any]:
    return {
        "steer_id": steer_id,
        "message_id": message_id,
        "content": content,
        "status": status,
    }


def run_middleware(inbox: FakeInbox, messages: list[Any]) -> dict[str, Any] | None:
    middleware = SteeringInboxMiddleware(inbox)  # type: ignore[arg-type]
    return asyncio.run(middleware.abefore_model({"messages": messages}, None))


class TestMiddlewareInjection:
    def test_pending_steer_injects_with_deterministic_id_and_acks(self) -> None:
        inbox = FakeInbox([steer("steer_a", "msg_a", "Also compare baselines.")])
        update = run_middleware(inbox, [])
        assert update is not None
        (injected,) = update["messages"]
        assert injected["id"] == "msg_a"
        assert injected["role"] == "user"
        # Framed, not raw: a bare mid-run user message gets rationalized away
        # ("that was not in the user's request") — live-trace finding.
        assert injected["content"] == steer_agent_content("Also compare baselines.")
        assert injected["content"].startswith(STEER_CONTENT_PREFIX)
        assert inbox.acked == ["steer_a"]

    def test_present_id_never_reinjects_but_heals_lost_ack(self) -> None:
        inbox = FakeInbox([steer("steer_a", "msg_a", "Once only.")])
        state_messages = add_messages(
            [], [{"role": "user", "content": "Once only.", "id": "msg_a"}]
        )
        update = run_middleware(inbox, state_messages)
        assert update is None
        # Status still pending + present in state = the ack was lost — re-ack.
        assert inbox.acked == ["steer_a"]

    def test_requeue_seeded_copy_detected_via_additional_kwargs(self) -> None:
        # A requeue seeds the transcript dict through LangChain coercion:
        # extra keys (metadata, message_id) land in additional_kwargs.
        seeded = add_messages(
            [],
            [
                {
                    "role": "user",
                    "content": "Steered earlier.",
                    "message_id": "msg_a",
                    "metadata": {"kind": "steering", "steer_id": "steer_a"},
                }
            ],
        )
        inbox = FakeInbox([steer("steer_a", "msg_a", "Steered earlier.", status="applied")])
        assert run_middleware(inbox, seeded) is None

    def test_applied_but_absent_reinjects_after_state_rebuild(self) -> None:
        # A continuation pass can rebuild state without the checkpointer; an
        # applied steer must keep being visible to the model.
        inbox = FakeInbox([steer("steer_a", "msg_a", "Keep me.", status="applied")])
        update = run_middleware(inbox, [{"role": "user", "content": "original", "id": "u1"}])
        assert update is not None
        assert update["messages"][0]["id"] == "msg_a"
        assert inbox.acked == []  # applied steers are not re-acked

    def test_missed_steers_never_inject(self) -> None:
        inbox = FakeInbox([steer("steer_a", "msg_a", "Too late.", status="missed")])
        assert run_middleware(inbox, []) is None

    def test_double_injection_collapses_via_add_messages(self) -> None:
        # The id IS the idempotency: apply the same update twice; state holds
        # one copy.
        update = {"role": "user", "content": "steer", "id": "msg_a"}
        once = add_messages([], [update])
        twice = add_messages(once, [update])
        assert len(twice) == 1
        assert twice[0].id == "msg_a"

    def test_fetch_failure_yields_none_and_never_raises(self) -> None:
        inbox = FakeInbox([steer("steer_a", "msg_a", "x")])
        inbox.fetch_error = True
        assert run_middleware(inbox, []) is None


class TestSteerIdsInMessages:
    def test_reads_object_ids_dict_ids_and_seeded_metadata(self) -> None:
        objects = add_messages([], [{"role": "user", "content": "a", "id": "msg_obj"}])
        mixed = [
            *objects,
            {"role": "user", "content": "b", "id": "msg_dict"},
            {
                "role": "user",
                "content": "c",
                "additional_kwargs": {
                    "metadata": {"kind": "steering"},
                    "message_id": "msg_seeded",
                },
            },
        ]
        ids = steer_ids_in_messages(mixed)
        assert {"msg_obj", "msg_dict", "msg_seeded"} <= ids


class TestSeedNormalization:
    def make_job(self, messages: list[dict[str, Any]]) -> RunJobEnvelope:
        return RunJobEnvelope(
            run_id="run_1", thread_id="t_1", user_id="u_1", goal="Plot the data.",
            messages=messages,
        )

    def test_steering_rows_carry_their_message_id_as_graph_id(self) -> None:
        job = self.make_job(
            [
                {"role": "user", "content": "Plot the data.", "message_id": "msg_u1"},
                {
                    "role": "user",
                    "content": "Label the axes.",
                    "message_id": "msg_steer",
                    "metadata": {"kind": "steering", "steer_id": "steer_a"},
                },
            ]
        )
        messages = _effective_job_messages(job)
        assert "id" not in messages[0]  # ordinary rows untouched
        assert messages[1]["id"] == "msg_steer"
        # Seeded copies speak with the same framed voice as live injections.
        assert messages[1]["content"] == steer_agent_content("Label the axes.")
        # Re-normalizing never double-frames.
        again = _effective_job_messages(self.make_job(messages))
        assert again[1]["content"] == messages[1]["content"]

    def test_request_classification_skips_steers_but_keeps_their_demands(self) -> None:
        # The reversed scan must not let "also label the axes" REPLACE the
        # original "plot" request — the steer appends, never erases.
        job = self.make_job(
            [
                {"role": "user", "content": "Plot the data as a figure."},
                {"role": "assistant", "content": "Working on it."},
                {
                    "role": "user",
                    "content": "Also save it as a CSV.",
                    "message_id": "msg_steer",
                    "run_id": "run_1",
                    "metadata": {"kind": "steering", "steer_id": "steer_a"},
                },
            ]
        )
        text = _run_request_text(job)
        assert "Plot the data as a figure." in text
        assert "Also save it as a CSV." in text

    def test_prior_turns_steers_add_no_demands(self) -> None:
        # A steer from a PAST run is ordinary history: folding it into the
        # demand classification resurrected long-satisfied artifact demands
        # on every requeue.
        job = self.make_job(
            [
                {
                    "role": "user",
                    "content": "Make a 3D model of the part.",
                    "message_id": "msg_old_steer",
                    "run_id": "run_0_previous",
                    "metadata": {"kind": "steering", "steer_id": "steer_old"},
                },
                {"role": "assistant", "content": "Model built."},
                {"role": "user", "content": "Now just summarize the findings."},
            ]
        )
        text = _run_request_text(job)
        assert "Now just summarize the findings." in text
        assert "Make a 3D model" not in text


class TestBarrierContinuation:
    def run_barrier(
        self,
        inbox: FakeInbox | None,
        messages: list[dict[str, Any]],
        rounds_used: int = 0,
        prior_answer_parts: list[str] | None = None,
    ) -> bool:
        from ultra_deepagents.context import AgentRunContext
        from ultra_deepagents.runner import AgentAttemptResult, RunEventSequencer

        events: list[dict[str, Any]] = []

        async def publish(event: dict[str, Any]) -> None:
            events.append(event)

        context = AgentRunContext(
            assistant_id="a", org_id="o", user_id="u", project_id="p",
            thread_id="t_1", run_id="run_test", goal="g",
        )
        result = asyncio.run(
            _steer_barrier_continuation(
                inbox,
                messages=messages,
                attempt_result=AgentAttemptResult(
                    final_response_text="The answer.",
                    streamed_response_text="",
                    post_tool_streamed_response_text="",
                ),
                artifact_events=[],
                context=context,
                sequencer=RunEventSequencer("run_test"),
                publish_event=publish,
                rounds_used=rounds_used,
                prior_answer_parts=prior_answer_parts,
            )
        )
        self.events = events
        return result

    def test_none_inbox_is_a_no_op(self) -> None:
        assert self.run_barrier(None, []) is False

    def test_fresh_pending_steers_append_and_continue(self) -> None:
        inbox = FakeInbox([], barrier=[steer("steer_a", "msg_a", "Last thing.")])
        messages = [{"role": "user", "content": "original"}]
        assert self.run_barrier(inbox, messages) is True
        assert messages[-1] == {
            "role": "user",
            "content": steer_agent_content("Last thing."),
            "id": "msg_a",
        }
        # The prior answer stays in the conversation the steer responds to.
        assert any(m.get("role") == "assistant" for m in messages)
        # Visible on the event stream — no silent application.
        assert any(e.get("event_kind") == "trace.message.delta" for e in self.events)

    def test_already_present_steers_reack_without_looping(self) -> None:
        inbox = FakeInbox([], barrier=[steer("steer_a", "msg_a", "Seen already.")])
        messages = [{"role": "user", "content": "Seen already.", "id": "msg_a"}]
        assert self.run_barrier(inbox, messages) is False
        assert inbox.acked == ["steer_a"]

    def test_round_cap_stops_the_loop(self) -> None:
        inbox = FakeInbox([], barrier=[steer("steer_a", "msg_a", "Again.")])
        assert self.run_barrier(inbox, [], rounds_used=3) is False
        assert inbox.barrier_calls == 0  # cap short-circuits before HTTP

    def test_barrier_failure_finishes_run_without_steers(self) -> None:
        inbox = FakeInbox([], barrier=None)  # close_barrier -> None (unreachable)
        assert self.run_barrier(inbox, []) is False

    def test_prior_answer_is_kept_for_stitching(self) -> None:
        # The continuation's reply is usually just the steer delta; the
        # published answer must keep the pre-steer response.
        from ultra_deepagents.runner import _stitch_continuation_answers

        inbox = FakeInbox([], barrier=[steer("steer_a", "msg_a", "Add a title.")])
        messages = [{"role": "user", "content": "original"}]
        parts: list[str] = []
        assert self.run_barrier(inbox, messages, prior_answer_parts=parts) is True
        assert parts == ["The answer."]
        stitched = _stitch_continuation_answers(parts, "Done — added the title.")
        assert stitched == "The answer.\n\nDone — added the title."
        # A model that re-emitted the full answer wins without duplication.
        assert (
            _stitch_continuation_answers(parts, "The answer. And the title.")
            == "The answer. And the title."
        )
        assert _stitch_continuation_answers([], "unchanged") == "unchanged"


class TestInboxGating:
    def make_settings(self, **overrides: Any) -> RuntimeSettings:
        return RuntimeSettings(
            openai_base_url="http://model.local/v1",
            openai_model="test-model",
            **overrides,
        )

    def test_no_worker_token_disables_steering(self) -> None:
        assert build_steering_inbox(self.make_settings(), run_id="run_1") is None

    def test_token_and_base_url_enable_steering(self) -> None:
        inbox = build_steering_inbox(
            self.make_settings(control_worker_token="tok"), run_id="run_1"
        )
        assert inbox is not None
        assert inbox.run_id == "run_1"
