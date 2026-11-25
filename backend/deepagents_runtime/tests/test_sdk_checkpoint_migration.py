"""Replay a trusted, synthetic durable checkpoint made by the previous worker.

Generate the fixture with this module and the 0.7.5 runtime on PYTHONPATH.
Only the hard-coded repository fixture can be deserialized by this test.
"""

import asyncio
import base64
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from deepagents.backends import StateBackend
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from ultra_deepagents.agent import build_research_agent
from ultra_deepagents.checkpointing import (
    DurableCheckpointer,
    InMemoryCheckpointStateStore,
    run_graph_config,
)
from ultra_deepagents.config import RuntimeSettings

RUN = "sdk-update-regression"
FIXTURE = Path(__file__).with_name("fixtures") / "checkpoint-deepagents-075.json"


class MigrationModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


async def exercise(directory, *, capture=False):
    receipts = []

    @tool
    def record_result() -> str:
        """Record a result once and return its immutable receipt."""
        receipts.append("saved")
        return "measurement-accepted"

    class InterruptAfterCommit(MigrationModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if any(isinstance(message, ToolMessage) for message in messages):
                raise RuntimeError("fixture restart after commit")
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    call = AIMessage(
        content="", tool_calls=[{"name": "record_result", "args": {}, "id": "measurement"}]
    )
    model = (
        InterruptAfterCommit(responses=[call])
        if capture
        else MigrationModel(responses=[AIMessage(content="complete")])
    )
    store = InMemoryCheckpointStateStore()
    saver = DurableCheckpointer(store)
    if not capture:
        checkpoint = json.loads(FIXTURE.read_text())
        assert checkpoint["producer"] == "0.7.5"
        await store.save(RUN, base64.b64decode(checkpoint["state"], validate=True))
        assert await saver.hydrate(RUN)
    settings = RuntimeSettings(
        openai_base_url="https://example.test/v1",
        openai_model="local-test",
        memory_root=str(directory),
        builder_enabled=False,
    )
    graph = build_research_agent(
        settings, model=model, tools=[record_result], backend=StateBackend(), checkpointer=saver
    )
    config = run_graph_config(RUN, recursion_limit=40)
    if capture:
        assert version("deepagents") == "0.7.5"
        try:
            await graph.ainvoke({"messages": [("user", "Record this result.")]}, config)
        except RuntimeError as exc:
            assert str(exc) == "fixture restart after commit"
        else:
            raise AssertionError("previous worker did not stop at the checkpoint")
        await saver.flush()
        assert receipts == ["saved"]
        blob = await store.load(RUN)
        FIXTURE.write_text(
            json.dumps(
                {"producer": version("deepagents"), "state": base64.b64encode(blob).decode("ascii")}
            )
            + "\n"
        )
        return

    state = await graph.aget_state(config)
    assert state.next and any(
        isinstance(m, ToolMessage) and m.content == "measurement-accepted"
        for m in state.values["messages"]
    )
    resumed = await graph.ainvoke(None, config)
    await saver.flush()
    assert resumed["messages"][-1].content == "complete"
    assert not (await graph.aget_state(config)).next
    assert receipts == []


def test_prior_worker_checkpoint_keeps_receipt_without_replaying_mutation(tmp_path):
    asyncio.run(exercise(tmp_path))


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        asyncio.run(exercise(Path(directory), capture=True))
