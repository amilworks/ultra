from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib import error as urllib_error

import nats.errors
import pytest
import ultra_deepagents.nats_worker as nats_worker_module
from nats.js.api import AckPolicy
from PIL import Image
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.nats_worker import (
    _RECOVERABLE_NATS_ERRORS,
    ControlPlaneRunLease,
    NATSDeepAgentsWorker,
    RunLeaseConflict,
    RunLeaseUnavailable,
    build_job_consumer_config,
    fetch_control_plane_run_max_sequence,
    fetch_control_plane_run_status,
    fetch_control_plane_run_usage_summary,
    fetch_job_messages,
    job_ack_extension_interval,
    post_control_plane_worker_heartbeat,
    try_acquire_run_lock,
)
from ultra_deepagents.runner import CheckpointReconciliationPendingError, run_job
from ultra_deepagents.schemas import RunJobEnvelope

_REAL_RUN_EVENTS_SNAPSHOT = NATSDeepAgentsWorker._run_events_snapshot
_REAL_ACQUIRE_RUN_LEASE_FOR_DELIVERY = NATSDeepAgentsWorker._acquire_run_lease_for_delivery


@pytest.fixture(autouse=True)
def _make_worker_transport_tests_independent_of_control_plane(monkeypatch, request):
    """Most transport tests exercise one lifecycle branch, not HTTP replay.

    Resume-specific tests restore or override the real method explicitly. This
    keeps the rest hermetic now that production correctly waits for durable
    event authority instead of degrading to a fresh floor on connection error.
    """

    async def empty_snapshot(_self, _run_id):
        return 0, None

    async def no_control_plane_lease(_self, _run_id):
        return None

    monkeypatch.setattr(NATSDeepAgentsWorker, "_run_events_snapshot", empty_snapshot)
    if "lease" not in request.node.name:
        monkeypatch.setattr(
            NATSDeepAgentsWorker,
            "_acquire_run_lease_for_delivery",
            no_control_plane_lease,
        )


def test_run_job_envelope_preserves_control_plane_context(tmp_path: Path):
    job = RunJobEnvelope.from_dict(
        {
            "run_id": "run-1",
            "thread_id": "thread-1",
            "user_id": "researcher-1",
            "goal": "Train a model.",
            "messages": [{"role": "user", "content": "Train a model."}],
            "file_ids": ["file-1"],
            "resource_uris": ["resource://file-1"],
            "dataset_uris": ["dataset://cells"],
            "selected_tool_names": ["python"],
            "knowledge_context": {"active_paper": "arxiv:2509.26626"},
            "workflow_hint": {"kind": "training"},
            "selection_context": {"source": "chat"},
            "reasoning_mode": "deep",
            "budgets": {"max_runtime_seconds": 0},
            "benchmark": {"suite": "autonomy"},
            "metadata": {
                "principal": {"org_id": "allen", "role": "researcher"},
                "runtime_facts": {
                    "current_datetime_utc": "2026-06-25T00:42:05Z",
                    "current_date_utc": "Thursday, June 25, 2026",
                    "user_timezone": "America/Los_Angeles",
                    "product_name": "Ultra",
                    "public_url": "https://ultra.example.edu",
                },
            },
        }
    )

    context = job.to_context(
        artifact_root=str(tmp_path / "artifacts"),
        workspace_root=str(tmp_path / "workspace"),
        run_lease_worker_id="worker-secret-id",
        run_lease_token="lease-secret-token",
    )

    assert context.run_id == "run-1"
    assert context.thread_id == "thread-1"
    assert context.user_id == "researcher-1"
    assert context.org_id == "allen"
    assert context.goal == "Train a model."
    assert context.selected_file_ids == ("file-1",)
    assert context.selected_resource_uris == ("resource://file-1",)
    assert context.selected_dataset_uris == ("dataset://cells",)
    assert context.allowed_tool_packs == ("python",)
    assert context.knowledge_context == {"active_paper": "arxiv:2509.26626"}
    assert context.workflow_hint == {"kind": "training"}
    assert context.selection_context == {"source": "chat"}
    assert context.reasoning_mode == "deep"
    assert context.budget == {"max_runtime_seconds": 0}
    assert context.benchmark == {"suite": "autonomy"}
    assert context.runtime_facts == {
        "current_datetime_utc": "2026-06-25T00:42:05Z",
        "current_date_utc": "Thursday, June 25, 2026",
        "user_timezone": "America/Los_Angeles",
        "product_name": "Ultra",
        "public_url": "https://ultra.example.edu",
    }
    assert context.auth_claims["role"] == "researcher"
    assert context.run_lease_worker_id == "worker-secret-id"
    assert context.run_lease_token == "lease-secret-token"
    assert "worker-secret-id" not in repr(context)
    assert "lease-secret-token" not in repr(context)
    assert "run_lease_worker_id" not in context.to_payload()
    assert "run_lease_token" not in context.to_payload()


def test_run_job_envelope_uses_only_typed_remote_mutation_scope(tmp_path: Path):
    job = RunJobEnvelope.from_dict(
        {
            "run_id": "run-mutation",
            "thread_id": "thread-mutation",
            "user_id": "researcher-1",
            "goal": "Upload the selected result and create a BisQue dataset.",
            "remote_mutation_intents": [
                "bisque.create_dataset",
                "bisque.upload",
                "bisque.upload",
            ],
            "metadata": {
                "principal": {"org_id": "allen", "role": "researcher"},
                "remote_mutation_intents": ["attacker.arbitrary_write"],
            },
        }
    )

    assert job.remote_mutation_intents == (
        "bisque.upload",
        "bisque.create_dataset",
    )
    context = job.to_context(
        artifact_root=str(tmp_path / "artifacts"),
        workspace_root=str(tmp_path / "workspace"),
    )
    assert context.remote_mutation_intents == (
        "bisque.upload",
        "bisque.create_dataset",
    )
    assert "remote_mutation_intents" not in context.run_metadata
    assert "remote_mutation_intents" not in context.to_payload()


@pytest.mark.parametrize(
    "raw_scope",
    ["bisque.upload", ["bisque.upload", "unknown.write"], ["bisque.upload", 1]],
)
def test_run_job_envelope_fails_closed_for_malformed_remote_mutation_scope(raw_scope):
    job = RunJobEnvelope.from_dict(
        {
            "run_id": "run-malformed",
            "thread_id": "thread-malformed",
            "user_id": "researcher-1",
            "goal": "Upload the selected result to BisQue.",
            "remote_mutation_intents": raw_scope,
        }
    )

    assert job.remote_mutation_intents == ()


class FakeStreamingAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        assert context.run_id == "run-1"
        assert payload["messages"][0]["content"] == "Say hello."
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": type("Chunk", (), {"content": "Hello"})()},
            "metadata": {"langgraph_node": "coordinator"},
        }


class FakeV3ProtocolStreamingAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        assert context.run_id == "run-1"
        assert payload["messages"][0]["content"] == "Say hello."
        yield {
            "type": "event",
            "method": "messages",
            "params": {
                "namespace": [],
                "data": [
                    {
                        "event": "content-block-delta",
                        "index": 0,
                        "delta": {"type": "text-delta", "text": "Hello"},
                    },
                    {"lc_agent_name": "ultra-research-agent"},
                ],
            },
        }
        yield {
            "type": "event",
            "method": "messages",
            "params": {
                "namespace": ["general-purpose"],
                "data": [
                    {
                        "event": "content-block-delta",
                        "index": 0,
                        "delta": {"type": "text-delta", "text": " subagent scratch"},
                    },
                    {"lc_agent_name": "general-purpose"},
                ],
            },
        }


class FakeV3ProtocolSubagentToolAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        assert context.run_id == "run-1"
        assert payload["messages"][0]["content"] == "Analyze the staged data."
        yield {
            "type": "event",
            "method": "messages",
            "params": {
                "namespace": [],
                "data": [
                    {
                        "event": "content-block-delta",
                        "index": 0,
                        "delta": {"type": "text-delta", "text": "Delegating"},
                    },
                    {"lc_agent_name": "ultra-research-agent"},
                ],
            },
        }
        yield {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "data": {
                    "event": "started",
                    "tool_call_id": "task-call-1",
                    "tool_name": "task",
                    "input": {
                        "subagent_type": "data-analyst",
                        "description": "Inspect the staged data and summarize shape.",
                    },
                },
            },
        }
        yield {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": ["data-analyst"],
                "data": {
                    "event": "started",
                    "tool_call_id": "execute-call-1",
                    "tool_name": "execute",
                    "input": {"command": "python inspect_data.py"},
                },
            },
        }
        yield {
            "type": "event",
            "method": "messages",
            "params": {
                "namespace": ["data-analyst"],
                "data": [
                    {
                        "event": "content-block-delta",
                        "index": 0,
                        "delta": {"type": "text-delta", "text": " saw 128 rows"},
                    },
                    {"lc_agent_name": "data-analyst"},
                ],
            },
        }
        yield {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": ["data-analyst"],
                "data": {
                    "event": "completed",
                    "tool_call_id": "execute-call-1",
                    "tool_name": "execute",
                    "output": "shape=(128, 6)",
                },
            },
        }
        yield {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "data": {
                    "event": "completed",
                    "tool_call_id": "task-call-1",
                    "tool_name": "task",
                    "output": "Data analyst inspected staged data: shape=(128, 6).",
                },
            },
        }
        yield {
            "type": "event",
            "method": "messages",
            "params": {
                "namespace": [],
                "data": [
                    {
                        "event": "content-block-delta",
                        "index": 0,
                        "delta": {"type": "text-delta", "text": " and reconciling."},
                    },
                    {"lc_agent_name": "ultra-research-agent"},
                ],
            },
        }


class FakeV3ProtocolDynamicTaskNamespaceAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        assert context.run_id == "run-1"
        assert payload["messages"][0]["content"] == "Compute statistics."
        dynamic_namespace = "tools:f9fde1a0-7bf1-5231-cabb-b5815b5fbb51"
        yield {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "data": {
                    "event": "started",
                    "tool_call_id": "task-call-1",
                    "tool_name": "task",
                    "input": {
                        "subagent_type": "code-runner",
                        "description": "Compute the requested statistics.",
                    },
                },
            },
        }
        yield {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [dynamic_namespace],
                "data": {
                    "event": "started",
                    "tool_call_id": "execute-call-1",
                    "tool_name": "execute",
                    "input": {"command": "python compute_stats.py"},
                },
            },
        }
        yield {
            "type": "event",
            "method": "messages",
            "params": {
                "namespace": [dynamic_namespace],
                "data": [
                    {
                        "event": "content-block-delta",
                        "index": 0,
                        "delta": {"type": "text-delta", "text": " computed regression"},
                    },
                    {"lc_agent_name": dynamic_namespace},
                ],
            },
        }
        yield {
            "event": "on_chat_model_end",
            "run_id": "subagent-model-call-1",
            "namespace": [dynamic_namespace],
            "data": {
                "output": _FakeUsageMessage(
                    {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16}
                )
            },
            "metadata": {
                "lc_agent_name": dynamic_namespace,
                "langgraph_node": dynamic_namespace,
                "ls_model_name": "deepseek_v4",
            },
        }
        yield {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [dynamic_namespace],
                "data": {
                    "event": "completed",
                    "tool_call_id": "execute-call-1",
                    "tool_name": "execute",
                    "output": "mean_y=4.0 slope=0.6 intercept=2.2",
                },
            },
        }
        yield {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "data": {
                    "event": "completed",
                    "tool_call_id": "task-call-1",
                    "tool_name": "task",
                    "output": "code-runner computed the statistics.",
                },
            },
        }
        yield {
            "type": "event",
            "method": "messages",
            "params": {
                "namespace": [],
                "data": [
                    {
                        "event": "content-block-delta",
                        "index": 0,
                        "delta": {"type": "text-delta", "text": " done"},
                    },
                    {"lc_agent_name": "ultra-research-agent"},
                ],
            },
        }


class FakeV3ProtocolAsyncDelegationAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        assert context.run_id == "run-1"
        assert payload["messages"][0]["content"] == "Train a classifier in the background."
        yield {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "data": {
                    "event": "started",
                    "tool_call_id": "async-start-1",
                    "tool_name": "start_async_task",
                    "input": {
                        "subagent_type": "remote-training-runner",
                        "description": "Train the classifier and report validation metrics.",
                    },
                },
            },
        }
        yield {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "data": {
                    "event": "completed",
                    "tool_call_id": "async-start-1",
                    "tool_name": "start_async_task",
                    "output": "Launched async subagent. task_id: async-thread-1",
                },
            },
        }
        yield {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "data": {
                    "event": "started",
                    "tool_call_id": "async-list-1",
                    "tool_name": "list_async_tasks",
                    "input": {"status_filter": "all"},
                },
            },
        }
        yield {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "data": {
                    "event": "completed",
                    "tool_call_id": "async-list-1",
                    "tool_name": "list_async_tasks",
                    "output": (
                        "1 tracked task(s):\n"
                        "- task_id: async-thread-1  agent: remote-training-runner  "
                        "status: running"
                    ),
                },
            },
        }
        yield {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "data": {
                    "event": "started",
                    "tool_call_id": "async-check-1",
                    "tool_name": "check_async_task",
                    "input": {"task_id": "async-thread-1"},
                },
            },
        }
        yield {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "data": {
                    "event": "completed",
                    "tool_call_id": "async-check-1",
                    "tool_name": "check_async_task",
                    "output": json.dumps(
                        {
                            "status": "error",
                            "thread_id": "async-thread-1",
                            "error": "CUDA out of memory",
                        }
                    ),
                },
            },
        }
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": payload["messages"][0]["content"]},
                        {"role": "assistant", "content": "The background worker failed."},
                    ]
                },
            },
        }


class FakeV3ProtocolAsyncValidationFailureAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        assert context.run_id == "run-1"
        assert payload["messages"][0]["content"] == "Validate async delegation failures."
        yield {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "data": {
                    "event": "completed",
                    "tool_call_id": "async-start-blank",
                    "tool_name": "start_async_task",
                    "output": (
                        "start_async_task description is required for async subagent delegation."
                    ),
                },
            },
        }
        yield {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "data": {
                    "event": "completed",
                    "tool_call_id": "async-update-blank",
                    "tool_name": "update_async_task",
                    "input": {"task_id": "async-thread-1"},
                    "output": (
                        "update_async_task message is required for async subagent delegation."
                    ),
                },
            },
        }
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": payload["messages"][0]["content"]},
                        {"role": "assistant", "content": "Delegation validation failed."},
                    ]
                },
            },
        }


class FakeV3ProtocolAsyncCancelAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        assert context.run_id == "run-1"
        assert payload["messages"][0]["content"] == "Cancel the background task."
        yield {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "data": {
                    "event": "started",
                    "tool_call_id": "async-cancel-1",
                    "tool_name": "cancel_async_task",
                    "input": {"task_id": "async-thread-1"},
                },
            },
        }
        yield {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "data": {
                    "event": "completed",
                    "tool_call_id": "async-cancel-1",
                    "tool_name": "cancel_async_task",
                    "output": "Cancelled async subagent task: async-thread-1",
                },
            },
        }
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": payload["messages"][0]["content"]},
                        {"role": "assistant", "content": "Cancelled the background worker."},
                    ]
                },
            },
        }


class FakeFailedTaskThenFallbackDisclosureAgent:
    def __init__(self) -> None:
        self.calls = 0
        self.continuation_prompt = ""

    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        self.calls += 1
        if self.calls == 1:
            yield {
                "type": "event",
                "method": "tools",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "started",
                        "tool_call_id": "task-call-1",
                        "tool_name": "task",
                        "input": {
                            "subagent_type": "code-runner",
                            "description": "Verify the Lyapunov estimate.",
                        },
                    },
                },
            }
            yield {
                "type": "event",
                "method": "tools",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "failed",
                        "tool_call_id": "task-call-1",
                        "tool_name": "task",
                        "error": "subagent stream closed",
                    },
                },
            }
            answer = (
                "Classification table: ICs=3, seeds=2, durations=500 and 1000 periods, "
                "step size h=T/200. lambda = 0.112 ± 0.005. Decision rule: classified "
                "only when |lambda| > 3× spread and an independent Poincare discriminator "
                "agrees; otherwise label the row marginal.\n\nLimitations: finite observation time.\n\n"
                "Delegated verification confirms the chaotic classification."
            )
        else:
            self.continuation_prompt = payload["messages"][-1]["content"]
            assert "task delegation failed" in self.continuation_prompt.lower()
            answer = (
                "Task delegation failed, so I used local fallback verification instead. "
                "Classification table: ICs=3, seeds=2, durations=500 and 1000 periods, "
                "step size h=T/200. lambda = 0.112 ± 0.005. Decision rule: classified "
                "only when |lambda| > 3× spread and an independent Poincare discriminator "
                "agrees; otherwise label the row marginal.\n\nLimitations: finite observation time.\n\n"
                "The local fallback verification confirms the chaotic classification."
            )
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": payload["messages"][0]["content"]},
                        {"role": "assistant", "content": answer},
                    ]
                },
            },
        }


class FakeCompletedInvalidTaskThenFallbackDisclosureAgent:
    def __init__(self) -> None:
        self.calls = 0
        self.continuation_prompt = ""

    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        self.calls += 1
        if self.calls == 1:
            yield {
                "type": "event",
                "method": "tools",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "started",
                        "tool_call_id": "task-call-invalid",
                        "tool_name": "task",
                        "input": {
                            "subagent_type": "missing-agent-fallback-probe",
                            "description": "Intentional invalid subagent probe.",
                        },
                    },
                },
            }
            yield {
                "type": "event",
                "method": "tools",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "completed",
                        "tool_call_id": "task-call-invalid",
                        "tool_name": "task",
                        "output": (
                            "We cannot invoke subagent missing-agent-fallback-probe "
                            "because it does not exist; the only allowed types are "
                            "`code-runner`, `data-analyst`."
                        ),
                    },
                },
            }
            answer = "Delegated verification confirms the debug workflow is safe."
        else:
            self.continuation_prompt = payload["messages"][-1]["content"]
            assert "task delegation failed" in self.continuation_prompt.lower()
            answer = (
                "Task delegation failed, so I used local fallback verification instead. "
                "The tool output says the requested subagent type does not exist."
            )
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": payload["messages"][0]["content"]},
                        {"role": "assistant", "content": answer},
                    ]
                },
            },
        }


class FakeMetadataOnlySubagentToolAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        assert context.run_id == "run-1"
        assert payload["messages"][0]["content"] == "Analyze the staged data."
        yield {
            "event": "on_tool_start",
            "name": "execute",
            "run_id": "execute-call-1",
            "data": {"input": {"command": "python inspect_data.py"}},
            "metadata": {"lc_agent_name": "data-analyst"},
        }
        yield {
            "event": "on_tool_end",
            "name": "execute",
            "run_id": "execute-call-1",
            "data": {"output": "shape=(128, 6)"},
            "metadata": {"lc_agent_name": "data-analyst"},
        }
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": "Analyze the staged data."},
                        {"role": "assistant", "content": "Analysis complete."},
                    ]
                },
            },
        }


class FakeConfigAwareStreamingAgent:
    async def astream_events(self, payload, config=None, *, context=None, version=None):
        assert version == "v3"
        assert context.run_id == "run-1"
        assert payload["messages"][0]["content"] == "Say hello."
        assert config["recursion_limit"] == 1234
        assert config["configurable"]["thread_id"] == "run-1"
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": "Say hello."},
                        {"role": "assistant", "content": "Hello with high recursion limit."},
                    ]
                },
            },
        }


class FakeConversationTitleResponse:
    content = '{"title": "RareSpot Prairie Dog Analysis"}'


class FakeConversationTitleModel:
    async def ainvoke(self, messages):
        joined = "\n".join(str(message.get("content", "")) for message in messages)
        assert "Run RareSpot on this prairie dog image" in joined
        # The title call runs concurrently with the run, so its prompt contains
        # only the request — never the assistant result.
        assert "RareSpot completed with burrow overlays." not in joined
        return FakeConversationTitleResponse()


class FakeRareSpotCompletionAgent:
    async def astream_events(self, payload, config=None, *, context=None, version=None):
        assert version == "v3"
        assert context.run_id == "run-1"
        assert "Run RareSpot on this prairie dog image" in payload["messages"][0]["content"]
        assert config["recursion_limit"] == 1000
        assert config["configurable"]["thread_id"] == "run-1"
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": payload["messages"][0]["content"]},
                        {
                            "role": "assistant",
                            "content": "RareSpot completed with burrow overlays.",
                        },
                    ]
                },
            },
        }


class FakeToolLifecycleAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        assert context.run_id == "run-1"
        yield {
            "event": "on_tool_start",
            "name": "execute",
            "run_id": "tool-run-1",
            "data": {"input": {"command": "python analysis.py"}},
            "metadata": {"lc_agent_name": "ultra-research-agent"},
        }
        yield {
            "event": "on_tool_end",
            "name": "execute",
            "run_id": "tool-run-1",
            "data": {
                "output": (
                    "analysis complete\nsaved plot.png\n[Command succeeded with exit code 0]"
                )
            },
            "metadata": {"lc_agent_name": "ultra-research-agent"},
        }
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": payload["messages"][0]["content"]},
                        {"role": "assistant", "content": "Finished analysis."},
                    ]
                },
            },
        }


class FakeProcessTextThenToolNoFinalAgent:
    def __init__(self) -> None:
        self.calls = 0
        self.continuation_prompt = ""

    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        self.calls += 1
        if self.calls == 1:
            yield {
                "type": "event",
                "method": "messages",
                "params": {
                    "namespace": [],
                    "data": [
                        {
                            "event": "content-block-delta",
                            "index": 0,
                            "delta": {
                                "type": "text-delta",
                                "text": "Let me inspect the script and fix the visualization.",
                            },
                        },
                        {"lc_agent_name": "ultra-research-agent"},
                    ],
                },
            }
            yield {
                "type": "event",
                "method": "tools",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "tool-started",
                        "tool_call_id": "call-edit-1",
                        "tool_name": "edit_file",
                        "input": {"file_path": "/workspace/analysis.py"},
                    },
                },
            }
            yield {
                "type": "event",
                "method": "tools",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "tool-finished",
                        "tool_call_id": "call-edit-1",
                        "output": "Successfully replaced 1 instance.",
                    },
                },
            }
            return

        self.continuation_prompt = payload["messages"][-1]["content"]
        assert "missing requested final response" in self.continuation_prompt
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": payload["messages"][-1]["content"]},
                        {
                            "role": "assistant",
                            "content": (
                                "Bubble sort works by repeatedly comparing adjacent elements, "
                                "swapping inverted pairs, and shrinking the unsorted suffix after "
                                "each pass."
                            ),
                        },
                    ]
                },
            },
        }


class FakeEmptyThenFinalResponseAgent:
    def __init__(self) -> None:
        self.calls = 0
        self.continuation_prompt = ""

    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        self.calls += 1
        if self.calls == 1:
            yield {
                "type": "event",
                "method": "messages",
                "params": {
                    "namespace": [],
                    "data": [
                        {
                            "event": "message-finish",
                            "metadata": {"finish_reason": "stop"},
                        },
                        {},
                    ],
                },
            }
            return

        self.continuation_prompt = payload["messages"][-1]["content"]
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": payload["messages"][-1]["content"]},
                        {
                            "role": "assistant",
                            "content": (
                                "Delta attention updates its fast-weight matrix with the "
                                "prediction error before reading the next output."
                            ),
                        },
                    ]
                },
            },
        }


class FakeAlwaysEmptyAgent:
    def __init__(self) -> None:
        self.calls = 0

    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        self.calls += 1
        if False:
            yield payload


class FakeDeepAgentsToolsProtocolAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        yield {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "data": {
                    "event": "tool-started",
                    "tool_call_id": "call-write-1",
                    "tool_name": "write_file",
                    "input": {
                        "file_path": "/workspace/analysis.py",
                        "content": "print('hello')",
                    },
                },
            },
        }
        yield {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "data": {
                    "event": "tool-finished",
                    "tool_call_id": "call-write-1",
                    "output": "content='Updated file /workspace/analysis.py' name='write_file'",
                },
            },
        }
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": payload["messages"][0]["content"]},
                        {"role": "assistant", "content": "Saved the analysis script."},
                    ]
                },
            },
        }


class FakeHangingAfterToolAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        yield {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "data": {
                    "event": "tool-started",
                    "tool_call_id": "call-hang-1",
                    "tool_name": "execute",
                    "input": {"command": "python plot.py"},
                },
            },
        }
        yield {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "data": {
                    "event": "tool-finished",
                    "tool_call_id": "call-hang-1",
                    "output": "script completed",
                },
            },
        }
        await asyncio.Event().wait()


class FakeUnderlyingTimeoutAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        _ = payload, context
        raise TimeoutError("upstream subagent stream timed out")
        yield  # pragma: no cover


class FakeIdleOnceThenRecoversAgent:
    def __init__(self) -> None:
        self.calls = 0
        self.recovery_prompt = ""

    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        self.calls += 1
        if self.calls == 1:
            yield {
                "type": "event",
                "method": "tools",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "tool-started",
                        "tool_call_id": "call-plot-1",
                        "tool_name": "execute",
                        "input": {"command": "python plot.py"},
                    },
                },
            }
            yield {
                "type": "event",
                "method": "tools",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "tool-finished",
                        "tool_call_id": "call-plot-1",
                        "output": "saved /workspace/outputs/plot.png",
                    },
                },
            }
            await asyncio.Event().wait()
            return

        self.recovery_prompt = payload["messages"][-1]["content"]
        assert "model stream went idle" in self.recovery_prompt
        assert "same workspace" in self.recovery_prompt
        # The recovery prompt must steer AWAY from blindly re-running existing work
        # (the empty-output -> re-run re-discovery loop seen in the 6.7M-token trace).
        assert "do NOT re-run a step whose output already exists" in self.recovery_prompt
        assert "KNOWN failure" in self.recovery_prompt
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": payload["messages"][-1]["content"]},
                        {
                            "role": "assistant",
                            "content": "Recovered from the idle stream and summarized the saved plot.",
                        },
                    ]
                },
            },
        }


class FakeReasoningIdleOnceThenRecoversAgent:
    def __init__(self) -> None:
        self.calls = 0
        self.recovery_prompt = ""

    def astream_events(
        self,
        payload,
        *,
        config=None,
        context=None,
        version=None,
    ):
        assert version == "v3"
        self.calls += 1
        callbacks = list((config or {}).get("callbacks") or [])

        async def generate():
            if self.calls == 1:
                run_id = "reasoning-dead-open-call"
                for handler in callbacks:
                    if hasattr(handler, "on_chat_model_start"):
                        await handler.on_chat_model_start(
                            {},
                            [],
                            run_id=run_id,
                            metadata={},
                        )
                for handler in callbacks:
                    if hasattr(handler, "on_llm_new_token"):
                        await handler.on_llm_new_token(
                            "",
                            chunk=SimpleNamespace(
                                message=SimpleNamespace(
                                    additional_kwargs={
                                        "reasoning_content": "Reasoning without answering. " * 8
                                    }
                                )
                            ),
                            run_id=run_id,
                        )
                await asyncio.Event().wait()
                return

            self.recovery_prompt = payload["messages"][-1]["content"]
            yield {
                "type": "event",
                "method": "values",
                "params": {
                    "namespace": [],
                    "data": {
                        "messages": [
                            {"role": "user", "content": self.recovery_prompt},
                            {
                                "role": "assistant",
                                "content": "Recovered and returned the complete delta-attention answer.",
                            },
                        ]
                    },
                },
            }

        return generate()


class FakeDegenerateReasoningOnceThenRecoversAgent:
    def __init__(self) -> None:
        self.calls = 0
        self.recovery_prompt = ""

    def astream_events(
        self,
        payload,
        *,
        config=None,
        context=None,
        version=None,
    ):
        assert version == "v3"
        self.calls += 1
        callbacks = list((config or {}).get("callbacks") or [])

        async def generate():
            if self.calls == 1:
                run_id = "degenerate-reasoning-call"
                for handler in callbacks:
                    if hasattr(handler, "on_chat_model_start"):
                        await handler.on_chat_model_start(
                            {},
                            [],
                            run_id=run_id,
                            metadata={},
                        )
                bad_fragment = "the — wait — the — — continue — " * 24
                for _ in range(8):
                    for handler in callbacks:
                        if hasattr(handler, "on_llm_new_token"):
                            await handler.on_llm_new_token(
                                "",
                                chunk=SimpleNamespace(
                                    message=SimpleNamespace(
                                        additional_kwargs={
                                            "reasoning_content": bad_fragment,
                                        }
                                    )
                                ),
                                run_id=run_id,
                            )
                return

            self.recovery_prompt = payload["messages"][-1]["content"]
            yield {
                "type": "event",
                "method": "values",
                "params": {
                    "namespace": [],
                    "data": {
                        "messages": [
                            {"role": "user", "content": self.recovery_prompt},
                            {
                                "role": "assistant",
                                "content": "Recovered with one complete, coherent answer.",
                            },
                        ]
                    },
                },
            }

        return generate()


class FakeRepeatedReasoningOnceThenRecoversAgent:
    def __init__(self) -> None:
        self.calls = 0

    def astream_events(
        self,
        payload,
        *,
        config=None,
        context=None,
        version=None,
    ):
        assert version == "v3"
        self.calls += 1
        callbacks = list((config or {}).get("callbacks") or [])

        async def generate():
            if self.calls == 1:
                run_id = "repeated-reasoning-call"
                for handler in callbacks:
                    if hasattr(handler, "on_chat_model_start"):
                        await handler.on_chat_model_start({}, [], run_id=run_id, metadata={})
                repeated = ("Retrieval Retrieval Retrieval Retrieval | Delta " * 180).strip()
                for handler in callbacks:
                    if hasattr(handler, "on_llm_new_token"):
                        await handler.on_llm_new_token(
                            "",
                            chunk=SimpleNamespace(
                                message=SimpleNamespace(
                                    additional_kwargs={"reasoning_content": repeated}
                                )
                            ),
                            run_id=run_id,
                        )
                return

            yield {
                "type": "event",
                "method": "values",
                "params": {
                    "namespace": [],
                    "data": {
                        "messages": [
                            {"role": "user", "content": payload["messages"][-1]["content"]},
                            {
                                "role": "assistant",
                                "content": "Recovered with a coherent explanation of the plot.",
                            },
                        ]
                    },
                },
            }

        return generate()


class FakeIdleTwiceThenMalformedProtocolThenRecoversAgent:
    """Exact live failure shape: two dead-open turns, then raw DSML text."""

    def __init__(self) -> None:
        self.calls = 0
        self.protocol_recovery_prompt = ""

    def astream_events(
        self,
        payload,
        *,
        config=None,
        context=None,
        version=None,
    ):
        assert version == "v3"
        self.calls += 1

        async def generate():
            if self.calls <= 2:
                await asyncio.Event().wait()
                return
            if self.calls == 3:
                for text in (
                    "Computation verified. Now the figures.\n\n</｜DS",
                    'ML｜tool_calls>\n<｜DSML｜invoke name="execute">',
                ):
                    yield {
                        "event": "on_chat_model_stream",
                        "data": {"chunk": SimpleNamespace(content=text)},
                        "metadata": {},
                    }
                return

            self.protocol_recovery_prompt = payload["messages"][-1]["content"]
            assert "malformed internal tool protocol" in self.protocol_recovery_prompt
            assert "DSML" not in json.dumps(payload["messages"][:-1])
            yield {
                "type": "event",
                "method": "values",
                "params": {
                    "namespace": [],
                    "data": {
                        "messages": [
                            {"role": "user", "content": self.protocol_recovery_prompt},
                            {
                                "role": "assistant",
                                "content": (
                                    "For q=(1,0), k=(1,0), and v=(2,3), the retrieval "
                                    "is (2,3); the delta update subtracts the current "
                                    "prediction before adding the residual outer product."
                                ),
                            },
                        ]
                    },
                },
            }

        return generate()


class FakePartialResponseIdleOnceThenRecoversAgent:
    def __init__(self) -> None:
        self.calls = 0

    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        self.calls += 1
        if self.calls == 1:
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": SimpleNamespace(content="First complete numerical section.")},
                "metadata": {},
            }
            await asyncio.Event().wait()
            return

        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": payload["messages"][-1]["content"]},
                        {
                            "role": "assistant",
                            "content": "Second recovered numerical section.",
                        },
                    ]
                },
            },
        }


def _stall_exec_events(call_id: str, command: str, output: str) -> list[dict]:
    return [
        {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "data": {
                    "event": "tool-started",
                    "tool_call_id": call_id,
                    "tool_name": "execute",
                    "input": {"command": command},
                },
            },
        },
        {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "data": {
                    "event": "tool-finished",
                    "tool_call_id": call_id,
                    "output": output,
                },
            },
        },
    ]


class FakeLivelocksThenRecoversAgent:
    """First attempt re-runs the same command with unchanged output until the
    within-turn progress-stall guard trips; the corrective attempt finishes."""

    def __init__(self) -> None:
        self.calls = 0
        self.recovery_prompt = ""

    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        self.calls += 1
        if self.calls == 1:
            for index in range(10):
                for event in _stall_exec_events(
                    f"call-{index}", "python run_experiment.py", "results.csv is empty"
                ):
                    yield event
            # The guard must abort the turn well before this hang is reached.
            await asyncio.Event().wait()
            return
        self.recovery_prompt = payload["messages"][-1]["content"]
        assert "Progress stall detected" in self.recovery_prompt
        assert "python run_experiment.py" in self.recovery_prompt
        assert "Do not re-run any repeated command" in self.recovery_prompt
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": payload["messages"][-1]["content"]},
                        {
                            "role": "assistant",
                            "content": "Diagnosed the empty CSV and finalized with current results.",
                        },
                    ]
                },
            },
        }


class FakeAlwaysLivelockedAgent:
    """Every attempt churns the same command: recoveries must exhaust and the run
    must still COMPLETE with partial results (the guard ends turns, never the run)."""

    def __init__(self) -> None:
        self.calls = 0

    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        self.calls += 1
        for index in range(10):
            for event in _stall_exec_events(
                f"call-{self.calls}-{index}", "python broken.py", "no output"
            ):
                yield event
        await asyncio.Event().wait()


class FakeV3FinalValuesOnlyAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        assert context.run_id == "run-1"
        assert payload["messages"][0]["content"] == "Say hello."
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": "Say hello."},
                        {"role": "assistant", "content": "Hello from final state"},
                    ]
                },
            },
        }


class FakeV3StreamThenFinalAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        yield {
            "type": "event",
            "method": "messages",
            "params": {
                "namespace": [],
                "data": [
                    {
                        "event": "content-block-delta",
                        "index": 0,
                        "delta": {"type": "text-delta", "text": "Let me recreate the script."},
                    },
                    {"lc_agent_name": "ultra-research-agent"},
                ],
            },
        }
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": "Change the plot color."},
                        {
                            "role": "assistant",
                            "content": "Updated the plot to use a green line and saved the revised code and figure.",
                        },
                    ]
                },
            },
        }


class FakeV3FollowupMissingFinalAssistantAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        assert payload["messages"][-1]["content"] == "Add a reference line."
        yield {
            "type": "event",
            "method": "messages",
            "params": {
                "namespace": [],
                "data": [
                    {
                        "event": "content-block-delta",
                        "index": 0,
                        "delta": {
                            "type": "text-delta",
                            "text": "Updated the plot with a dashed y=10 reference line.",
                        },
                    },
                    {"lc_agent_name": "ultra-research-agent"},
                ],
            },
        }
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": "Create the first plot."},
                        {"role": "assistant", "content": "Created the first plot."},
                        {"role": "user", "content": "Add a reference line."},
                    ]
                },
            },
        }


class FakeOutputWritingAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        output_dir = Path(context.workspace_root) / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "plot_squared.py").write_text("print('plot')\n")
        (output_dir / "plot_squared.png").write_bytes(b"\x89PNG\r\n\x1a\nplot")
        (output_dir / "frame_006.png").write_bytes(b"\x89PNG\r\n\x1a\nframe")
        (Path(context.workspace_root) / "matplotlibrc").write_text("savefig.dpi: 300\n")
        (Path(context.workspace_root) / "frame_007.png").write_bytes(b"\x89PNG\r\n\x1a\nframe")
        yield {
            "type": "event",
            "method": "messages",
            "params": {
                "namespace": [],
                "data": [
                    {
                        "event": "content-block-delta",
                        "index": 0,
                        "delta": {"type": "text-delta", "text": "Created plot."},
                    },
                    {"lc_agent_name": "ultra-research-agent"},
                ],
            },
        }


class FakeLowDpiPlotAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        output_dir = Path(context.workspace_root) / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (600, 400), "white")
        image.save(output_dir / "low_dpi_plot.png", dpi=(72, 72))
        yield {
            "type": "event",
            "method": "messages",
            "params": {
                "namespace": [],
                "data": [
                    {
                        "event": "content-block-delta",
                        "index": 0,
                        "delta": {"type": "text-delta", "text": "Created plot."},
                    },
                    {"lc_agent_name": "ultra-research-agent"},
                ],
            },
        }


class FakeMarkdownReportAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        output_dir = Path(context.workspace_root) / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "rarespot_combined_report.md").write_text(
            "# RareSpot report\n\nMetrics table.\n"
        )
        yield {
            "type": "event",
            "method": "messages",
            "params": {
                "namespace": [],
                "data": [
                    {
                        "event": "content-block-delta",
                        "index": 0,
                        "delta": {"type": "text-delta", "text": "Saved report."},
                    },
                    {"lc_agent_name": "ultra-research-agent"},
                ],
            },
        }


class FakeDuplicateRootAndOutputsAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        workspace = Path(context.workspace_root)
        output_dir = workspace / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        code = "print('plot')\n"
        figure = b"\x89PNG\r\n\x1a\nplot"
        (output_dir / "plot_squared.py").write_text(code)
        (output_dir / "plot_squared.png").write_bytes(figure)
        (workspace / "plot_squared.py").write_text(code)
        (workspace / "plot_squared.png").write_bytes(figure)
        yield {
            "type": "event",
            "method": "messages",
            "params": {
                "namespace": [],
                "data": [
                    {
                        "event": "content-block-delta",
                        "index": 0,
                        "delta": {"type": "text-delta", "text": "Created plot."},
                    },
                    {"lc_agent_name": "ultra-research-agent"},
                ],
            },
        }


class FakeWhitespaceOnlyOutputAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        output_dir = Path(context.workspace_root) / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "plot_x_squared.py").write_text("print('plot')\n")
        yield {
            "type": "event",
            "method": "messages",
            "params": {
                "namespace": [],
                "data": [
                    {
                        "event": "content-block-delta",
                        "index": 0,
                        "delta": {"type": "text-delta", "text": "\n\n"},
                    },
                    {"lc_agent_name": "ultra-research-agent"},
                ],
            },
        }
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": payload["messages"][0]["content"]},
                        {"role": "assistant", "content": "\n\n"},
                    ]
                },
            },
        }


class FakeRootOutputWritingAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        workspace = Path(context.workspace_root)
        (workspace / "plot_squared.py").write_text("print('plot')\n")
        (workspace / "plot_squared.png").write_bytes(b"\x89PNG\r\n\x1a\nplot")
        (workspace / "frame_007.png").write_bytes(b"\x89PNG\r\n\x1a\nframe")
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": "Create a plot."},
                        {
                            "role": "assistant",
                            "content": (
                                "Created root-level deliverables: plot_squared.png and "
                                "plot_squared.py."
                            ),
                        },
                    ]
                },
            },
        }


class FakePrematureCodeOnlyThenFigureAgent:
    def __init__(self) -> None:
        self.calls = 0
        self.continuation_prompt = ""

    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        self.calls += 1
        workspace = Path(context.workspace_root)
        if self.calls == 1:
            (workspace / "plot_x2.py").write_text(
                "import matplotlib.pyplot as plt\n"
                "plt.plot([0, 1, 2], [0, 1, 4])\n"
                "plt.savefig('/workspace/plot_x2.png')\n"
            )
            yield {
                "type": "event",
                "method": "values",
                "params": {
                    "namespace": [],
                    "data": {
                        "messages": [
                            {"role": "user", "content": payload["messages"][0]["content"]},
                            {"role": "assistant", "content": "Saved the script plot_x2.py."},
                        ]
                    },
                },
            }
            return

        self.continuation_prompt = payload["messages"][-1]["content"]
        assert "missing requested durable outputs" in self.continuation_prompt
        assert "figure" in self.continuation_prompt
        assert "execute" in self.continuation_prompt
        (workspace / "plot_x2.png").write_bytes(b"\x89PNG\r\n\x1a\nplot")
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": payload["messages"][-1]["content"]},
                        {
                            "role": "assistant",
                            "content": (
                                "Executed the plotting script and saved the code "
                                "(plot_x2.py) and figure (plot_x2.png)."
                            ),
                        },
                    ]
                },
            },
        }


class FakePrematureTrainingNoWeightsThenCheckpointAgent:
    def __init__(self) -> None:
        self.calls = 0
        self.continuation_prompt = ""

    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        self.calls += 1
        output_dir = Path(context.workspace_root) / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.calls == 1:
            (output_dir / "train_unet.py").write_text("print('train')\n")
            (output_dir / "training_curves.png").write_bytes(b"\x89PNG\r\n\x1a\ncurves")
            yield {
                "type": "event",
                "method": "values",
                "params": {
                    "namespace": [],
                    "data": {
                        "messages": [
                            {"role": "user", "content": payload["messages"][0]["content"]},
                            {
                                "role": "assistant",
                                "content": "Saved the training script and training curves.",
                            },
                        ]
                    },
                },
            }
            return

        self.continuation_prompt = payload["messages"][-1]["content"]
        assert "missing requested durable outputs" in self.continuation_prompt
        assert "model" in self.continuation_prompt
        assert "execute" in self.continuation_prompt
        (output_dir / "best_model.pth").write_bytes(b"model weights")
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": payload["messages"][-1]["content"]},
                        {
                            "role": "assistant",
                            "content": "Saved the trained model weights, code, and figures.",
                        },
                    ]
                },
            },
        }


class FakeArtifactsOnlyThenExplanationAgent:
    def __init__(self) -> None:
        self.calls = 0
        self.continuation_prompt = ""

    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        self.calls += 1
        output_dir = Path(context.workspace_root) / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.calls == 1:
            (output_dir / "plot_x_squared.py").write_text("print('plot')\n")
            (output_dir / "plot_x_squared.png").write_bytes(b"\x89PNG\r\n\x1a\nplot")
            if False:
                yield {}
            return

        self.continuation_prompt = payload["messages"][-1]["content"]
        assert "missing requested final response" in self.continuation_prompt
        assert "explain" in self.continuation_prompt
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": payload["messages"][-1]["content"]},
                        {
                            "role": "assistant",
                            "content": (
                                "The plot demonstrates quadratic growth: y = x^2 "
                                "rises slowly near zero and faster as x increases."
                            ),
                        },
                    ]
                },
            },
        }


class BlockingStreamingAgent:
    def __init__(self, started: asyncio.Event):
        self.started = started

    async def astream_events(self, payload, *, context=None, version=None):
        self.started.set()
        await asyncio.Event().wait()
        if False:
            yield {}


async def _run_fake_job(tmp_path: Path):
    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        workspace_root=str(tmp_path / "workspaces"),
        artifact_root=str(tmp_path / "artifacts"),
    )
    job = RunJobEnvelope(
        run_id="run-1",
        thread_id="thread-1",
        user_id="researcher-1",
        goal="Say hello.",
        messages=[{"role": "user", "content": "Say hello."}],
    )
    published = []

    async def publish(event):
        published.append(event)

    await run_job(
        job,
        settings,
        publish_event=publish,
        agent_factory=lambda *_args, **_kwargs: FakeStreamingAgent(),
    )
    return published


def test_run_job_streams_started_delta_and_completed(tmp_path: Path):
    events = asyncio.run(_run_fake_job(tmp_path))

    assert [event["event_kind"] for event in events] == [
        "run.started",
        "message.delta",
        "run.completed",
    ]
    assert [event["sequence"] for event in events] == [1, 2, 3]
    assert [event["event_id"] for event in events] == [
        "evt_run-1_000001",
        "evt_run-1_000002",
        "evt_run-1_000003",
    ]
    assert events[1]["payload"]["text"] == "Hello"
    assert events[-1]["payload"]["response_text"] == "Hello"


class _FakeUsageMessage:
    """Stand-in for the aggregated AIMessage on ``on_chat_model_end``."""

    def __init__(self, usage_metadata, model="deepseek_v4"):
        self.usage_metadata = usage_metadata
        self.response_metadata = {"model_name": model}


class FakeTokenUsageAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        assert context.run_id == "run-1"
        # Two model calls (e.g. coordinator turn + a post-tool turn); usage is
        # reported per call and must be summed at the run level.
        yield {
            "event": "on_chat_model_end",
            "run_id": "model-call-1",
            "data": {
                "output": _FakeUsageMessage(
                    {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
                )
            },
            "metadata": {"langgraph_node": "coordinator", "ls_model_name": "deepseek_v4"},
        }
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": type("Chunk", (), {"content": "Hello"})()},
            "metadata": {"langgraph_node": "coordinator"},
        }
        yield {
            "event": "on_chat_model_end",
            "run_id": "model-call-2",
            "data": {
                "output": _FakeUsageMessage(
                    {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60}
                )
            },
            "metadata": {"langgraph_node": "coordinator", "ls_model_name": "deepseek_v4"},
        }
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": "Say hello."},
                        {"role": "assistant", "content": "Hello from the model."},
                    ]
                },
            },
        }


class FakeSubagentTokenUsageAgent:
    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        assert context.run_id == "run-1"
        assert payload["messages"][0]["content"] == "Analyze the staged data."
        yield {
            "event": "on_chat_model_end",
            "run_id": "subagent-model-call-1",
            "namespace": ["data-analyst"],
            "data": {
                "output": _FakeUsageMessage(
                    {"input_tokens": 30, "output_tokens": 12, "total_tokens": 42}
                )
            },
            "metadata": {
                "lc_agent_name": "data-analyst",
                "langgraph_node": "data-analyst",
                "ls_model_name": "deepseek_v4",
            },
        }
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": "Analyze the staged data."},
                        {"role": "assistant", "content": "Analysis complete."},
                    ]
                },
            },
        }


def test_run_job_completed_event_carries_summed_token_usage(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Say hello.",
            messages=[{"role": "user", "content": "Say hello."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeTokenUsageAgent(),
        )
        return published

    events = asyncio.run(scenario())

    completed = events[-1]
    assert completed["event_kind"] == "run.completed"
    assert completed["payload"]["response_text"] == "Hello from the model."
    assert completed["payload"]["usage"] == {
        "input_tokens": 150,
        "output_tokens": 30,
        "total_tokens": 180,
        "model": "deepseek_v4",
    }
    usage_events = [event for event in events if event["event_kind"] == "run.token_usage"]
    assert [event["payload"]["usage_event_id"] for event in usage_events] == [
        "run-1:model:model-call-1",
        "run-1:model:model-call-2",
    ]
    assert all(event["payload"]["usage_event_id"] != event["event_id"] for event in usage_events)
    assert [event["payload"]["total_tokens"] for event in usage_events] == [120, 60]
    assert all(event["message"] == "" for event in usage_events)


def test_run_job_scopes_subagent_token_usage_events(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Analyze the staged data.",
            messages=[{"role": "user", "content": "Analyze the staged data."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeSubagentTokenUsageAgent(),
        )
        return published

    events = asyncio.run(scenario())

    usage_event = next(event for event in events if event["event_kind"] == "run.token_usage")
    assert usage_event["agent_role"] == "data-analyst"
    assert usage_event["node_name"] == "data-analyst:model"
    assert usage_event["payload"]["subagent_name"] == "data-analyst"
    assert usage_event["payload"]["namespace"] == ["data-analyst"]
    assert usage_event["payload"]["total_tokens"] == 42


def _v3_message_finish_event(
    usage, *, checkpoint_ns, namespace=None, model="deepseek_v4", provider="openai"
):
    """A real v3 Pregel 'messages' event carrying per-call token usage.

    Mirrors the shape the compiled graph actually emits under
    ``astream_events(version="v3")``: ``params.data[0]`` is the
    ``message-finish`` chunk with ``usage``; ``params.data[1]`` is the
    LangGraph metadata. The legacy ``on_chat_model_end`` event the older
    extractor keyed on is never emitted by v3, so these tests guard the real
    production path rather than a synthetic v2 stand-in.
    """
    return {
        "type": "event",
        "method": "messages",
        "params": {
            "namespace": list(namespace or []),
            "data": [
                {"event": "message-finish", "usage": dict(usage), "metadata": {}},
                {
                    "langgraph_node": "model",
                    "langgraph_checkpoint_ns": checkpoint_ns,
                    "checkpoint_ns": checkpoint_ns,
                    "ls_model_name": model,
                    "ls_provider": provider,
                    **({"lc_agent_name": namespace[-1]} if namespace else {}),
                },
            ],
        },
    }


class FakeV3TokenUsageAgent:
    """Coordinator + subagent model calls in the real v3 message-finish shape."""

    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        assert context.run_id == "run-1"
        yield _v3_message_finish_event(
            {"input_tokens": 6940, "output_tokens": 20, "total_tokens": 6960},
            checkpoint_ns="model:aaaa1111-2222-3333-4444-555566667777",
        )
        # A subagent ("task" delegation) model call streams under a namespace.
        yield _v3_message_finish_event(
            {"input_tokens": 500, "output_tokens": 40, "total_tokens": 540},
            checkpoint_ns="model:bbbb1111-2222-3333-4444-555566667777",
            namespace=["data-analyst"],
        )
        # A duplicate of the first call (e.g. NATS redelivery) must not double-count.
        yield _v3_message_finish_event(
            {"input_tokens": 6940, "output_tokens": 20, "total_tokens": 6960},
            checkpoint_ns="model:aaaa1111-2222-3333-4444-555566667777",
        )
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": "Analyze the staged data."},
                        {"role": "assistant", "content": "Done."},
                    ]
                },
            },
        }


def test_run_job_captures_v3_message_finish_token_usage(tmp_path: Path):
    """Regression guard: usage must be extracted from the v3 protocol the
    runner actually streams, not only the v2 on_chat_model_end shape."""

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            model_provider_id="self-hosted-vllm",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Analyze the staged data.",
            messages=[{"role": "user", "content": "Analyze the staged data."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeV3TokenUsageAgent(),
        )
        return published

    events = asyncio.run(scenario())

    usage_events = [event for event in events if event["event_kind"] == "run.token_usage"]
    # Two distinct model calls captured; the redelivered duplicate dedupes by
    # the checkpoint-ns-derived usage_event_id.
    ids = [event["payload"]["usage_event_id"] for event in usage_events]
    assert len(usage_events) == 3
    assert ids[0] == ids[2] != ids[1]
    assert {event["payload"]["provider"] for event in usage_events} == {"self-hosted-vllm"}
    # Coordinator call is unscoped; the subagent call is scoped by namespace.
    scoped = {
        event["payload"]["usage_event_id"]: event["payload"].get("subagent_name")
        for event in usage_events
    }
    assert scoped[ids[1]] == "data-analyst"

    completed = events[-1]
    assert completed["event_kind"] == "run.completed"
    # 6960 (coordinator) + 540 (subagent) + 6960 (duplicate, still summed in the
    # attempt total) — the control plane dedupes persistence by usage_event_id.
    assert completed["payload"]["usage"]["total_tokens"] == 14460
    assert completed["payload"]["usage"]["model"] == "deepseek_v4"


def test_run_job_completed_usage_includes_prior_persisted_usage(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Say hello.",
            messages=[{"role": "user", "content": "Say hello."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeTokenUsageAgent(),
            prior_usage={
                "input_tokens": 25,
                "output_tokens": 5,
                "total_tokens": 30,
                "model": "deepseek_v4",
            },
        )
        return published

    events = asyncio.run(scenario())

    completed = events[-1]
    assert completed["event_kind"] == "run.completed"
    assert completed["payload"]["usage"] == {
        "input_tokens": 175,
        "output_tokens": 35,
        "total_tokens": 210,
        "model": "deepseek_v4",
    }


def test_seed_user_profile_memory_writes_profile_without_clobbering_preferences(tmp_path: Path):
    from ultra_deepagents.runner import _seed_user_profile_memory

    memory_root = tmp_path / "users" / "ada"
    learned_preferences = memory_root / "preferences.md"
    memory_root.mkdir(parents=True)
    learned_preferences.write_text(
        "## Learned preference\nUse concise equations.\n", encoding="utf-8"
    )
    _seed_user_profile_memory(
        memory_root,
        {
            "display_name": "Ada Lovelace",
            "title": "Principal Investigator",
            "institution": "Analytical Engine Lab",
            "research_interests": "symbolic computation",
            "bio": "Studies general-purpose computation.",
        },
    )
    profile = (memory_root / "user_profile.md").read_text()
    assert profile.startswith("# User profile")
    assert "Ada Lovelace" in profile
    assert "Analytical Engine Lab" in profile
    assert "symbolic computation" in profile
    assert (
        learned_preferences.read_text(encoding="utf-8")
        == "## Learned preference\nUse concise equations.\n"
    )


def test_seed_user_profile_memory_skips_empty_or_blank_profiles(tmp_path: Path):
    from ultra_deepagents.runner import _seed_user_profile_memory

    memory_root = tmp_path / "users" / "ada"
    _seed_user_profile_memory(memory_root, None)
    _seed_user_profile_memory(memory_root, {})
    _seed_user_profile_memory(memory_root, {"display_name": "   ", "bio": ""})
    assert not (memory_root / "user_profile.md").exists()


def test_run_job_seeds_user_profile_into_per_user_memory(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            memory_root=str(tmp_path / "memory"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-7",
            goal="Say hello.",
            messages=[{"role": "user", "content": "Say hello."}],
        )

        async def publish(event):
            return None

        await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeStreamingAgent(),
            user_profile={
                "display_name": "Dr. Test",
                "research_interests": "astrobiology",
            },
        )
        return settings

    settings = asyncio.run(scenario())
    profile = Path(settings.memory_root) / "users" / "researcher-7" / "user_profile.md"
    assert profile.exists()
    text = profile.read_text()
    assert "Dr. Test" in text
    assert "astrobiology" in text


def test_run_job_streams_deepagents_v3_raw_protocol_messages(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Say hello.",
            messages=[{"role": "user", "content": "Say hello."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeV3ProtocolStreamingAgent(),
        )
        return result, published

    result, events = asyncio.run(scenario())

    assert result == "Hello"
    assert [event["event_kind"] for event in events] == [
        "run.started",
        "message.delta",
        "subagent.message.delta",
        "run.completed",
    ]
    assert events[1]["payload"]["text"] == "Hello"
    assert events[2]["payload"]["text"] == " subagent scratch"
    assert events[2]["payload"]["source"] == "general-purpose"
    assert events[-1]["payload"]["response_text"] == "Hello"


def test_run_job_scopes_subagent_tool_events_in_stream(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Analyze the staged data.",
            messages=[{"role": "user", "content": "Analyze the staged data."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeV3ProtocolSubagentToolAgent(),
        )
        return result, published

    result, events = asyncio.run(scenario())

    assert result == " and reconciling."
    execute_started = next(
        event
        for event in events
        if event["event_kind"] == "tool_call.started" and event["payload"]["tool_name"] == "execute"
    )
    assert execute_started["agent_role"] == "data-analyst"
    assert execute_started["node_name"] == "data-analyst:tool:execute"
    assert execute_started["payload"]["subagent_name"] == "data-analyst"
    assert execute_started["payload"]["namespace"] == ["data-analyst"]
    assert execute_started["payload"]["tool_call_id"] == "execute-call-1"

    subagent_delta = next(
        event for event in events if event["event_kind"] == "subagent.message.delta"
    )
    assert subagent_delta["agent_role"] == "data-analyst"
    assert subagent_delta["payload"]["text"] == " saw 128 rows"

    task_started = next(
        event
        for event in events
        if event["event_kind"] == "tool_call.started" and event["payload"]["tool_name"] == "task"
    )
    assert task_started["agent_role"] == "tool"
    assert task_started["payload"]["subagent_type"] == "data-analyst"
    assert events[-1]["payload"]["response_text"] == " and reconciling."


def test_run_job_maps_dynamic_task_namespace_to_parent_subagent_type(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Compute statistics.",
            messages=[{"role": "user", "content": "Compute statistics."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeV3ProtocolDynamicTaskNamespaceAgent(),
        )
        return result, published

    result, events = asyncio.run(scenario())

    assert result == " done"
    dynamic_namespace = "tools:f9fde1a0-7bf1-5231-cabb-b5815b5fbb51"
    task_started = next(
        event
        for event in events
        if event["event_kind"] == "tool_call.started" and event["payload"]["tool_name"] == "task"
    )
    assert task_started["payload"]["subagent_type"] == "code-runner"

    execute_started = next(
        event
        for event in events
        if event["event_kind"] == "tool_call.started" and event["payload"]["tool_name"] == "execute"
    )
    assert execute_started["agent_role"] == "code-runner"
    assert execute_started["node_name"] == "code-runner:tool:execute"
    assert execute_started["payload"]["subagent_name"] == "code-runner"
    assert execute_started["payload"]["namespace"] == [dynamic_namespace]

    subagent_delta = next(
        event for event in events if event["event_kind"] == "subagent.message.delta"
    )
    assert subagent_delta["agent_role"] == "code-runner"
    assert subagent_delta["node_name"] == "code-runner"
    assert subagent_delta["payload"]["source"] == "code-runner"
    assert subagent_delta["payload"]["namespace"] == [dynamic_namespace]

    usage_event = next(event for event in events if event["event_kind"] == "run.token_usage")
    assert usage_event["agent_role"] == "code-runner"
    assert usage_event["node_name"] == "code-runner:model"
    assert usage_event["payload"]["subagent_name"] == "code-runner"
    assert usage_event["payload"]["namespace"] == [dynamic_namespace]


def test_run_job_enriches_async_delegation_tool_events_with_structured_evidence(
    tmp_path: Path,
):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Train a classifier in the background.",
            messages=[{"role": "user", "content": "Train a classifier in the background."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeV3ProtocolAsyncDelegationAgent(),
        )
        return published

    events = asyncio.run(scenario())

    start_started = next(
        event
        for event in events
        if event["event_kind"] == "tool_call.started"
        and event["payload"]["tool_name"] == "start_async_task"
    )
    assert start_started["payload"]["delegation_mode"] == "async_subagent"
    assert start_started["payload"]["async_subagent_name"] == "remote-training-runner"

    launch_completed = next(
        event
        for event in events
        if event["event_kind"] == "tool_call.completed"
        and event["payload"]["tool_name"] == "start_async_task"
    )
    assert launch_completed["payload"]["delegation_mode"] == "async_subagent"
    assert launch_completed["payload"]["async_task_id"] == "async-thread-1"
    assert launch_completed["payload"]["async_task_ids"] == ["async-thread-1"]

    list_completed = next(
        event
        for event in events
        if event["event_kind"] == "tool_call.completed"
        and event["payload"]["tool_name"] == "list_async_tasks"
    )
    assert list_completed["payload"]["delegation_mode"] == "async_subagent"
    assert list_completed["payload"]["async_task_id"] == "async-thread-1"
    assert list_completed["payload"]["async_subagent_name"] == "remote-training-runner"
    assert list_completed["payload"]["async_status"] == "running"

    check_completed = next(
        event
        for event in events
        if event["event_kind"] == "tool_call.completed"
        and event["payload"]["tool_name"] == "check_async_task"
    )
    assert check_completed["payload"]["delegation_mode"] == "async_subagent"
    assert check_completed["payload"]["async_task_id"] == "async-thread-1"
    assert check_completed["payload"]["async_status"] == "error"
    assert check_completed["payload"]["async_error"] == "CUDA out of memory"
    assert check_completed["payload"]["async_failure"] is True


def test_run_job_marks_async_required_instruction_errors_as_failures(
    tmp_path: Path,
):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Validate async delegation failures.",
            messages=[{"role": "user", "content": "Validate async delegation failures."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeV3ProtocolAsyncValidationFailureAgent(),
        )
        return published

    events = asyncio.run(scenario())

    start_completed = next(
        event
        for event in events
        if event["event_kind"] == "tool_call.completed"
        and event["payload"]["tool_name"] == "start_async_task"
    )
    assert start_completed["payload"]["delegation_mode"] == "async_subagent"
    assert start_completed["payload"]["async_failure"] is True
    assert start_completed["payload"]["async_error"] == (
        "start_async_task description is required for async subagent delegation."
    )

    update_completed = next(
        event
        for event in events
        if event["event_kind"] == "tool_call.completed"
        and event["payload"]["tool_name"] == "update_async_task"
    )
    assert update_completed["payload"]["delegation_mode"] == "async_subagent"
    assert update_completed["payload"]["async_task_id"] == "async-thread-1"
    assert update_completed["payload"]["async_failure"] is True
    assert update_completed["payload"]["async_error"] == (
        "update_async_task message is required for async subagent delegation."
    )


def test_run_job_marks_cancelled_async_delegation_events_as_terminal(
    tmp_path: Path,
):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Cancel the background task.",
            messages=[{"role": "user", "content": "Cancel the background task."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeV3ProtocolAsyncCancelAgent(),
        )
        return published

    events = asyncio.run(scenario())

    cancel_completed = next(
        event
        for event in events
        if event["event_kind"] == "tool_call.completed"
        and event["payload"]["tool_name"] == "cancel_async_task"
    )
    assert cancel_completed["payload"]["delegation_mode"] == "async_subagent"
    assert cancel_completed["payload"]["async_task_id"] == "async-thread-1"
    assert cancel_completed["payload"]["async_task_ids"] == ["async-thread-1"]
    assert cancel_completed["payload"]["async_status"] == "cancelled"
    assert cancel_completed["payload"]["async_statuses"] == ["cancelled"]


def test_run_job_scopes_metadata_only_subagent_tool_events_in_stream(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Analyze the staged data.",
            messages=[{"role": "user", "content": "Analyze the staged data."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeMetadataOnlySubagentToolAgent(),
        )
        return published

    events = asyncio.run(scenario())

    execute_events = [
        event
        for event in events
        if event["event_kind"].startswith("tool_call.")
        and event["payload"]["tool_name"] == "execute"
    ]
    assert [event["event_kind"] for event in execute_events] == [
        "tool_call.started",
        "tool_call.completed",
    ]
    assert [event["agent_role"] for event in execute_events] == [
        "data-analyst",
        "data-analyst",
    ]
    assert [event["node_name"] for event in execute_events] == [
        "data-analyst:tool:execute",
        "data-analyst:tool:execute",
    ]
    assert execute_events[0]["payload"]["subagent_name"] == "data-analyst"
    assert execute_events[0]["payload"].get("namespace") is None


def test_run_job_passes_configured_langgraph_recursion_limit(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            langgraph_recursion_limit=1234,
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Say hello.",
            messages=[{"role": "user", "content": "Say hello."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeConfigAwareStreamingAgent(),
        )
        return result, published

    result, events = asyncio.run(scenario())

    assert result == "Hello with high recursion limit."
    assert [event["event_kind"] for event in events] == ["run.started", "run.completed"]
    assert events[-1]["payload"]["response_text"] == result


def test_run_job_includes_generated_conversation_title_in_completed_event(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            title_generation_enabled=True,
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Run RareSpot on this prairie dog image and summarize burrow detections.",
            messages=[
                {
                    "role": "user",
                    "content": "Run RareSpot on this prairie dog image and summarize burrow detections.",
                }
            ],
        )
        published = []

        async def publish(event):
            published.append(event)

        await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeRareSpotCompletionAgent(),
            title_model_factory=lambda _settings: FakeConversationTitleModel(),
        )
        return published

    events = asyncio.run(scenario())

    assert events[-1]["event_kind"] == "run.completed"
    assert events[-1]["payload"]["conversation_title"] == "RareSpot Prairie Dog Analysis"
    assert events[-1]["payload"]["title_generation"] == {
        "strategy": "llm",
        "model": "deepseek_v4",
    }


def test_notes_run_title_generation_never_receives_note_derived_answer(tmp_path: Path):
    sentinel = "NOTE_ONLY_SENTINEL_MUST_STAY_OUT_OF_TITLE"
    title_prompts: list[str] = []

    class NotesAnswerAgent:
        async def astream_events(self, _payload, *, context=None, version=None, **_kwargs):
            assert context.selection_context["note_access"]["mode"] == "selected"
            assert version == "v3"
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": SimpleNamespace(content=sentinel)},
                "metadata": {"langgraph_node": "coordinator"},
            }

    class CapturingTitleModel:
        model_name = "deepseek_v4"

        async def ainvoke(self, messages):
            title_prompts.append("\n".join(str(message.get("content", "")) for message in messages))
            return SimpleNamespace(content='{"title": "Data Analysis"}')

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            title_generation_enabled=True,
        )
        job = RunJobEnvelope(
            run_id="run-notes-title",
            thread_id="thread-notes-title",
            user_id="researcher-1",
            goal="Use my attached note as context.",
            messages=[{"role": "user", "content": "Use my attached note as context."}],
            selection_context={
                "note_access": {
                    "mode": "selected",
                    "notes": [{"note_id": "note-private", "revision": 1}],
                }
            },
        )
        published: list[dict[str, Any]] = []

        async def publish(event):
            published.append(event)

        response = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: NotesAnswerAgent(),
            title_model_factory=lambda _settings: CapturingTitleModel(),
        )
        return response, published

    response, events = asyncio.run(scenario())

    assert response == sentinel
    assert title_prompts
    assert all(sentinel not in prompt for prompt in title_prompts)
    completed = events[-1]
    assert completed["event_kind"] == "run.completed"
    assert completed["payload"]["response_text"] == sentinel
    assert sentinel not in completed["payload"]["conversation_title"]
    assert len(title_prompts) == 1


def test_notes_run_ignores_checkpoint_and_steering_while_general_run_retains_them(
    tmp_path: Path,
    monkeypatch,
):
    factory_kwargs: dict[str, dict[str, Any]] = {}
    hydrated_runs: list[str] = []
    steering_inbox_runs: list[str] = []

    class TrackingCheckpointer:
        async def hydrate(self, run_id: str) -> bool:
            hydrated_runs.append(run_id)
            return False

    class EmptySteeringInbox:
        async def reopen_barrier(self) -> None:
            return None

        async def close_barrier(self) -> list[dict[str, Any]]:
            return []

    class AnswerAgent:
        async def astream_events(self, _payload, *, version=None, **_kwargs):
            assert version == "v3"
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": SimpleNamespace(content="done")},
                "metadata": {"langgraph_node": "coordinator"},
            }

    def build_inbox(_settings, *, run_id: str):
        steering_inbox_runs.append(run_id)
        return EmptySteeringInbox()

    def agent_factory(_settings, **kwargs):
        context = kwargs["context"]
        factory_kwargs[context.run_id] = kwargs
        return AnswerAgent()

    monkeypatch.setattr("ultra_deepagents.runner.build_steering_inbox", build_inbox)
    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        workspace_root=str(tmp_path / "workspaces"),
        artifact_root=str(tmp_path / "artifacts"),
        memory_root=str(tmp_path / "memory"),
        title_generation_enabled=False,
    )
    notes_job = RunJobEnvelope(
        run_id="run-notes-private-state",
        thread_id="thread-notes-private-state",
        user_id="researcher-1",
        goal="Use my Note.",
        messages=[{"role": "user", "content": "Use my Note."}],
        selection_context={
            "note_access": {
                "mode": "selected",
                "notes": [{"note_id": "note-private", "revision": 1}],
            }
        },
    )
    general_job = RunJobEnvelope(
        run_id="run-general-durable-state",
        thread_id="thread-general-durable-state",
        user_id="researcher-1",
        goal="Analyze this.",
        messages=[{"role": "user", "content": "Analyze this."}],
    )
    checkpointer = TrackingCheckpointer()

    async def scenario() -> None:
        async def publish(_event):
            return None

        await run_job(
            notes_job,
            settings,
            publish_event=publish,
            agent_factory=agent_factory,
            checkpointer=checkpointer,
        )
        await run_job(
            general_job,
            settings,
            publish_event=publish,
            agent_factory=agent_factory,
            checkpointer=checkpointer,
        )

    asyncio.run(scenario())

    assert factory_kwargs[notes_job.run_id].get("checkpointer") is None
    assert factory_kwargs[notes_job.run_id].get("steering_inbox") is None
    assert notes_job.run_id not in hydrated_runs
    assert notes_job.run_id not in steering_inbox_runs

    assert factory_kwargs[general_job.run_id]["checkpointer"] is checkpointer
    assert factory_kwargs[general_job.run_id]["steering_inbox"] is not None
    assert hydrated_runs == [general_job.run_id]
    assert steering_inbox_runs == [general_job.run_id]


def test_notes_run_failure_redacts_exception_and_workspace_lease(tmp_path: Path):
    sentinel = "NOTE_SENTINEL_ECHOED_BY_PROVIDER_ERROR"

    class FailingNotesAgent:
        def astream_events(self, _payload, **_kwargs):
            async def generate():
                raise RuntimeError(sentinel)
                yield

            return generate()

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            title_generation_enabled=False,
        )
        job = RunJobEnvelope(
            run_id="run-notes-failed",
            thread_id="thread-notes-failed",
            user_id="researcher-1",
            goal="Use my attached note as context.",
            messages=[{"role": "user", "content": "Use my attached note as context."}],
            selection_context={
                "note_access": {
                    "mode": "selected",
                    "notes": [{"note_id": "note-private", "revision": 1}],
                }
            },
        )
        published: list[dict[str, Any]] = []

        async def publish(event):
            published.append(event)

        with pytest.raises(RuntimeError, match=sentinel):
            await run_job(
                job,
                settings,
                publish_event=publish,
                agent_factory=lambda *_args, **_kwargs: FailingNotesAgent(),
            )
        return settings, published

    settings, events = asyncio.run(scenario())

    failed = events[-1]
    assert failed["event_kind"] == "run.failed"
    assert failed["message"] == "Notes-enabled run failed."
    assert failed["payload"] == {
        "error": "Notes-enabled run failed.",
        "error_type": "RuntimeError",
        "redacted": True,
    }
    lease = Path(settings.workspace_root, "run-notes-failed", "lease.json").read_text()
    assert "notes_run_failed" in lease
    assert sentinel not in json.dumps(events)
    assert sentinel not in lease


def test_run_job_publishes_tool_lifecycle_without_polluting_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image_id = "sha256:" + "b" * 64
    monkeypatch.setattr(
        "ultra_deepagents.runner.resolve_docker_image_id",
        lambda _image_ref: image_id,
    )

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Run analysis.",
            messages=[{"role": "user", "content": "Run analysis."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeToolLifecycleAgent(),
        )
        return result, published

    result, events = asyncio.run(scenario())

    assert result == "Finished analysis."
    assert [event["event_kind"] for event in events] == [
        "run.started",
        "tool_call.started",
        "tool_call.completed",
        "run.completed",
    ]
    assert events[1]["payload"]["tool_name"] == "execute"
    assert events[1]["payload"]["command"] == "python analysis.py"
    assert events[1]["payload"]["tool_call_id"] == "tool-run-1"
    assert events[1]["payload"]["runtime_image_digest"] == image_id
    assert events[2]["payload"]["output_preview"] == (
        "analysis complete\nsaved plot.png\n[Command succeeded with exit code 0]"
    )
    assert events[2]["payload"]["exit_code"] == 0
    assert events[2]["payload"]["runtime_image_digest"] == image_id
    assert events[-1]["payload"]["response_text"] == "Finished analysis."


def test_run_job_publishes_deepagents_tools_protocol_events(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Write analysis code.",
            messages=[{"role": "user", "content": "Write analysis code."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeDeepAgentsToolsProtocolAgent(),
        )
        return result, published

    result, events = asyncio.run(scenario())

    assert result == "Saved the analysis script."
    assert [event["event_kind"] for event in events] == [
        "run.started",
        "tool_call.started",
        "tool_call.completed",
        "run.completed",
    ]
    assert events[1]["payload"]["tool_name"] == "write_file"
    assert events[1]["payload"]["file_path"] == "/workspace/analysis.py"
    assert events[1]["payload"]["tool_call_id"] == "call-write-1"
    assert events[2]["payload"]["tool_name"] == "write_file"
    assert "Updated file /workspace/analysis.py" in events[2]["payload"]["output_preview"]
    assert events[-1]["payload"]["response_text"] == "Saved the analysis script."


def test_run_job_fails_terminal_when_deepagents_stream_goes_idle(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            model_stream_idle_timeout_seconds=0.01,
            model_stream_idle_max_recoveries=0,
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Create a plot.",
            messages=[{"role": "user", "content": "Create a plot."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        with pytest.raises(TimeoutError, match="Deep Agents stream produced no events"):
            await asyncio.wait_for(
                run_job(
                    job,
                    settings,
                    publish_event=publish,
                    agent_factory=lambda *_args, **_kwargs: FakeHangingAfterToolAgent(),
                ),
                timeout=1.0,
            )
        return published

    events = asyncio.run(scenario())

    assert [event["event_kind"] for event in events] == [
        "run.started",
        "tool_call.started",
        "tool_call.completed",
        "trace.model.stalled",
        "run.failed",
    ]
    assert events[-2]["payload"]["idle_scope"] == "stream"
    assert events[-2]["payload"]["recoveries_exhausted"] is True
    assert "Deep Agents stream produced no events" in events[-1]["message"]
    lease = json.loads((tmp_path / "workspaces" / "run-1" / "lease.json").read_text())
    assert lease["status"] == "failed"


def test_run_job_does_not_label_underlying_timeout_as_idle_recovery(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            model_stream_idle_timeout_seconds=0.0,
            model_stream_idle_max_recoveries=2,
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Run a delegated audit.",
            messages=[{"role": "user", "content": "Run a delegated audit."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        with pytest.raises(TimeoutError, match="upstream subagent stream timed out"):
            await run_job(
                job,
                settings,
                publish_event=publish,
                agent_factory=lambda *_args, **_kwargs: FakeUnderlyingTimeoutAgent(),
            )
        return published

    events = asyncio.run(scenario())

    assert [event["event_kind"] for event in events] == ["run.started", "run.failed"]
    assert events[-1]["payload"]["error"] == "upstream subagent stream timed out"
    assert not any(
        event.get("payload", {}).get("reason") == "model_stream_idle" for event in events
    )


def test_run_job_recovers_from_one_idle_model_stream_before_failing(tmp_path: Path):
    async def scenario():
        fake_agent = FakeIdleOnceThenRecoversAgent()
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            model_stream_idle_timeout_seconds=0.01,
            model_stream_idle_max_recoveries=1,
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Run the analysis and explain it.",
            messages=[{"role": "user", "content": "Run the analysis and explain it."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        result = await asyncio.wait_for(
            run_job(
                job,
                settings,
                publish_event=publish,
                agent_factory=lambda *_args, **_kwargs: fake_agent,
            ),
            timeout=1.0,
        )
        return result, published, fake_agent

    result, events, fake_agent = asyncio.run(scenario())

    assert fake_agent.calls == 2
    assert result == "Recovered from the idle stream and summarized the saved plot."
    assert [event["event_kind"] for event in events] == [
        "run.started",
        "tool_call.started",
        "tool_call.completed",
        "trace.model.stalled",
        "trace.message.delta",
        "run.completed",
    ]
    assert events[3]["payload"]["idle_scope"] == "stream"
    assert events[4]["payload"]["recovery_index"] == 1
    assert events[4]["payload"]["reason"] == "model_stream_idle"
    assert events[-1]["payload"]["response_text"] == result
    lease = json.loads((tmp_path / "workspaces" / "run-1" / "lease.json").read_text())
    assert lease["status"] == "succeeded"


def test_run_job_recovers_reasoning_only_dead_open_model_stream(tmp_path: Path):
    async def scenario():
        fake_agent = FakeReasoningIdleOnceThenRecoversAgent()
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            model_stream_idle_timeout_seconds=1.0,
            model_output_idle_timeout_seconds=0.02,
            model_stream_idle_max_recoveries=1,
        )
        prompt = "Can you provide real computations for delta attention?"
        job = RunJobEnvelope(
            run_id="run-reasoning-dead-open",
            thread_id="thread-reasoning-dead-open",
            user_id="researcher-1",
            goal=prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        published = []

        async def publish(event):
            published.append(event)

        result = await asyncio.wait_for(
            run_job(
                job,
                settings,
                publish_event=publish,
                agent_factory=lambda *_args, **_kwargs: fake_agent,
            ),
            timeout=1.0,
        )
        return result, published, fake_agent

    result, events, fake_agent = asyncio.run(scenario())

    assert fake_agent.calls == 2
    assert result == "Recovered and returned the complete delta-attention answer."
    assert "model stream went idle" in fake_agent.recovery_prompt
    recovery = next(
        event for event in events if event.get("payload", {}).get("reason") == "model_stream_idle"
    )
    assert recovery["payload"]["timeout_seconds"] == 0.02
    stalled = next(event for event in events if event["event_kind"] == "trace.model.stalled")
    assert stalled["payload"]["idle_scope"] == "model_output"
    assert stalled["payload"]["output_classification"] == "reasoning_only"
    assert [
        event["payload"]["status"]
        for event in events
        if event["event_kind"] == "trace.reasoning.delta"
    ] == ["running", "completed"]
    assert events[-1]["event_kind"] == "run.completed"
    assert events[-1]["payload"]["response_text"] == result


def test_run_job_aborts_degenerate_reasoning_and_returns_full_recovery(tmp_path: Path):
    async def scenario():
        fake_agent = FakeDegenerateReasoningOnceThenRecoversAgent()
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            model_stream_idle_max_recoveries=1,
        )
        prompt = "Can you provide real computations for delta attention?"
        job = RunJobEnvelope(
            run_id="run-degenerate-reasoning",
            thread_id="thread-degenerate-reasoning",
            user_id="researcher-1",
            goal=prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: fake_agent,
        )
        return result, published, fake_agent

    result, events, fake_agent = asyncio.run(scenario())

    assert fake_agent.calls == 2
    assert "malformed repetitive token loop" in fake_agent.recovery_prompt
    assert result == "Recovered with one complete, coherent answer."
    stalled = next(event for event in events if event["event_kind"] == "trace.model.stalled")
    assert stalled["payload"]["idle_scope"] == "reasoning_quality"
    assert stalled["payload"]["quality_signal"] == "dashlike_density"
    assert stalled["payload"]["recoveries_exhausted"] is False
    recovery = next(
        event
        for event in events
        if event.get("payload", {}).get("reason") == "reasoning_degeneration"
    )
    assert recovery["payload"]["recovery_index"] == 1
    assert events[-1]["event_kind"] == "run.completed"
    assert events[-1]["payload"]["response_text"] == result
    assert not any(event["event_kind"] == "run.failed" for event in events)


def test_run_job_aborts_repeated_word_reasoning_and_returns_full_recovery(tmp_path: Path):
    async def scenario():
        fake_agent = FakeRepeatedReasoningOnceThenRecoversAgent()
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            model_stream_idle_max_recoveries=1,
        )
        prompt = "Explain the delta-attention retrieval calculation."
        job = RunJobEnvelope(
            run_id="run-repeated-reasoning",
            thread_id="thread-repeated-reasoning",
            user_id="researcher-1",
            goal=prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: fake_agent,
        )
        return result, published, fake_agent

    result, events, fake_agent = asyncio.run(scenario())

    assert fake_agent.calls == 2
    assert result == "Recovered with a coherent explanation of the plot."
    stalled = next(event for event in events if event["event_kind"] == "trace.model.stalled")
    assert stalled["payload"]["idle_scope"] == "reasoning_quality"
    assert stalled["payload"]["quality_signal"] == "lexical_repetition"
    assert stalled["payload"]["quality_max_repeated_trigram"] >= 48
    assert stalled["payload"]["quality_token_diversity"] <= 0.08
    assert "Retrieval Retrieval Retrieval" not in json.dumps(events)
    assert events[-1]["event_kind"] == "run.completed"
    assert events[-1]["payload"]["response_text"] == result


def test_run_job_recovers_malformed_protocol_after_idle_budget_is_exhausted(tmp_path: Path):
    async def scenario():
        fake_agent = FakeIdleTwiceThenMalformedProtocolThenRecoversAgent()
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            model_stream_idle_timeout_seconds=0.01,
            model_stream_idle_max_recoveries=2,
            model_protocol_max_recoveries=1,
        )
        prompt = "Can you provide real computations for delta attention?"
        job = RunJobEnvelope(
            run_id="run-malformed-protocol",
            thread_id="thread-malformed-protocol",
            user_id="researcher-1",
            goal=prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        published = []

        async def publish(event):
            published.append(event)

        result = await asyncio.wait_for(
            run_job(
                job,
                settings,
                publish_event=publish,
                agent_factory=lambda *_args, **_kwargs: fake_agent,
            ),
            timeout=1.0,
        )
        return result, published, fake_agent

    result, events, fake_agent = asyncio.run(scenario())

    assert fake_agent.calls == 4
    assert fake_agent.protocol_recovery_prompt
    assert result.startswith("For q=(1,0), k=(1,0), and v=(2,3)")
    stalled = [event for event in events if event["event_kind"] == "trace.model.stalled"]
    assert [event["payload"]["idle_scope"] for event in stalled] == [
        "stream",
        "stream",
        "model_protocol",
    ]
    protocol_stall = stalled[-1]
    assert protocol_stall["payload"]["quality_signal"] == "deepseek_dsml_control_token"
    assert protocol_stall["payload"]["recoveries_exhausted"] is False
    protocol_recovery = next(
        event for event in events if event.get("payload", {}).get("reason") == "model_protocol_leak"
    )
    assert protocol_recovery["payload"]["recovery_index"] == 1
    terminal = events[-1]
    assert terminal["event_kind"] == "run.completed"
    assert terminal["payload"]["response_text"] == result
    assert "DSML" not in result
    assert "tool_calls" not in result
    assert "DSML" not in json.dumps(events)
    assert not any(event["event_kind"] == "run.failed" for event in events)


def test_run_job_fails_closed_instead_of_completing_with_protocol_text(tmp_path: Path):
    class _MalformedProtocolAgent:
        def astream_events(self, _payload, *, version=None, **_kwargs):
            assert version == "v3"

            async def generate():
                for text in ("Results follow.\n\n</｜DS", "ML｜tool_calls>"):
                    yield {
                        "event": "on_chat_model_stream",
                        "data": {"chunk": SimpleNamespace(content=text)},
                        "metadata": {},
                    }

            return generate()

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            model_protocol_max_recoveries=0,
        )
        job = RunJobEnvelope(
            run_id="run-malformed-protocol-exhausted",
            thread_id="thread-malformed-protocol-exhausted",
            user_id="researcher-1",
            goal="Return the calculation.",
            messages=[{"role": "user", "content": "Return the calculation."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        with pytest.raises(
            RuntimeError,
            match="malformed internal tool protocol text",
        ):
            await run_job(
                job,
                settings,
                publish_event=publish,
                agent_factory=lambda *_args, **_kwargs: _MalformedProtocolAgent(),
            )
        return published

    events = asyncio.run(scenario())

    stalled = next(event for event in events if event["event_kind"] == "trace.model.stalled")
    assert stalled["payload"]["idle_scope"] == "model_protocol"
    assert stalled["payload"]["recoveries_exhausted"] is True
    assert events[-1]["event_kind"] == "run.failed"
    assert not any(event["event_kind"] == "run.completed" for event in events)
    visible = "".join(
        str(event.get("payload", {}).get("text") or "")
        for event in events
        if event["event_kind"] == "message.delta"
    )
    assert visible == "Results follow.\n\n"
    assert "DSML" not in visible
    assert "DSML" not in json.dumps(events)


def test_run_job_quarantines_legacy_protocol_response_from_model_history(tmp_path: Path):
    class _HistoryCapturingAgent:
        def __init__(self) -> None:
            self.messages = []

        def astream_events(self, payload, *, version=None, **_kwargs):
            assert version == "v3"
            self.messages = list(payload["messages"])

            async def generate():
                yield {
                    "type": "event",
                    "method": "values",
                    "params": {
                        "namespace": [],
                        "data": {
                            "messages": [
                                {"role": "user", "content": self.messages[-1]["content"]},
                                {
                                    "role": "assistant",
                                    "content": "The first plot shows the retrieval residual.",
                                },
                            ]
                        },
                    },
                }

            return generate()

    async def scenario():
        agent = _HistoryCapturingAgent()
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            completion_max_continuations=0,
        )
        job = RunJobEnvelope(
            run_id="run-legacy-protocol-history",
            thread_id="thread-legacy-protocol-history",
            user_id="researcher-1",
            goal="Explain the first plot.",
            messages=[
                {"role": "user", "content": "Compute delta attention."},
                {
                    "role": "assistant",
                    "content": (
                        "Computation verified.\n\n</｜DSML｜tool_calls>"
                        '<｜DSML｜invoke name="execute">'
                    ),
                },
                {"role": "user", "content": "Explain the first plot."},
            ],
        )
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: agent,
        )
        return result, published, agent

    result, events, agent = asyncio.run(scenario())

    assert result == "The first plot shows the retrieval residual."
    assert [message["role"] for message in agent.messages] == ["user", "user"]
    assert "DSML" not in json.dumps(agent.messages)
    quarantine = next(
        event for event in events if event["event_kind"] == "trace.history.quarantined"
    )
    assert quarantine["payload"] == {
        "reason": "model_protocol",
        "message_count": 1,
    }
    assert events[-1]["event_kind"] == "run.completed"


def test_run_job_preserves_partial_response_across_idle_recovery(tmp_path: Path):
    async def scenario():
        fake_agent = FakePartialResponseIdleOnceThenRecoversAgent()
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            model_stream_idle_timeout_seconds=1.0,
            model_output_idle_timeout_seconds=0.02,
            model_stream_idle_max_recoveries=1,
        )
        job = RunJobEnvelope(
            run_id="run-partial-response-idle",
            thread_id="thread-partial-response-idle",
            user_id="researcher-1",
            goal="Return both numerical sections.",
            messages=[{"role": "user", "content": "Return both numerical sections."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        result = await asyncio.wait_for(
            run_job(
                job,
                settings,
                publish_event=publish,
                agent_factory=lambda *_args, **_kwargs: fake_agent,
            ),
            timeout=1.0,
        )
        return result, published, fake_agent

    result, events, fake_agent = asyncio.run(scenario())

    assert fake_agent.calls == 2
    assert result == ("First complete numerical section.\n\nSecond recovered numerical section.")
    stalled = next(event for event in events if event["event_kind"] == "trace.model.stalled")
    assert stalled["payload"]["idle_scope"] == "model_output"
    assert stalled["payload"]["output_classification"] == "partial_response"
    assert stalled["payload"]["visible_response_observed"] is True
    assert events[-1]["event_kind"] == "run.completed"
    assert events[-1]["payload"]["response_text"] == result


def test_run_job_recovers_from_within_turn_progress_stall(tmp_path: Path):
    """The 6.7M-token livelock class: a turn that keeps re-running the same command
    with unchanged output must be ENDED (not the run) and corrected via one
    injected prompt — the within-turn complement to the idle recovery above."""

    async def scenario():
        fake_agent = FakeLivelocksThenRecoversAgent()
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            progress_stall_threshold=3,
            progress_stall_max_recoveries=1,
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Run the analysis and explain it.",
            messages=[{"role": "user", "content": "Run the analysis and explain it."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        result = await asyncio.wait_for(
            run_job(
                job,
                settings,
                publish_event=publish,
                agent_factory=lambda *_args, **_kwargs: fake_agent,
            ),
            timeout=2.0,
        )
        return result, published, fake_agent

    result, events, fake_agent = asyncio.run(scenario())

    assert fake_agent.calls == 2
    assert result == "Diagnosed the empty CSV and finalized with current results."
    stall_events = [
        event for event in events if event.get("payload", {}).get("reason") == "progress_stall"
    ]
    assert len(stall_events) == 1
    payload = stall_events[0]["payload"]
    assert payload["recovery_index"] == 1
    assert payload["recoveries_exhausted"] is False
    assert payload["stall_count"] == 3
    assert payload["repeated_commands"][0]["command"] == "python run_experiment.py"
    assert events[-1]["event_kind"] == "run.completed"
    # The turn was aborted at the threshold: 1 novel + 3 repeats = 4 exec pairs,
    # never the fake's full 10 (the guard, not the hang, ended the turn).
    tool_completed = [e for e in events if e["event_kind"] == "tool_call.completed"]
    assert len(tool_completed) == 4
    lease = json.loads((tmp_path / "workspaces" / "run-1" / "lease.json").read_text())
    assert lease["status"] == "succeeded"


def test_run_job_progress_stall_recoveries_exhausted_fail_without_empty_success(
    tmp_path: Path,
):
    """A model that ignores recovery is bounded and never reports empty success."""

    async def scenario():
        fake_agent = FakeAlwaysLivelockedAgent()
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            progress_stall_threshold=3,
            progress_stall_max_recoveries=1,
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Run the analysis.",
            messages=[{"role": "user", "content": "Run the analysis."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        with pytest.raises(RuntimeError, match="without a user-visible response"):
            await asyncio.wait_for(
                run_job(
                    job,
                    settings,
                    publish_event=publish,
                    agent_factory=lambda *_args, **_kwargs: fake_agent,
                ),
                timeout=2.0,
            )
        return published, fake_agent

    events, fake_agent = asyncio.run(scenario())

    stall_events = [
        event for event in events if event.get("payload", {}).get("reason") == "progress_stall"
    ]
    assert len(stall_events) >= 2
    assert stall_events[-1]["payload"]["recoveries_exhausted"] is True
    assert "finalizing the run honestly" in stall_events[-1]["message"]
    kinds = [event["event_kind"] for event in events]
    assert "run.completed" not in kinds
    assert kinds[-1] == "run.failed"
    # Bounded: initial attempt + 1 recovery (+ possibly bounded completion-guard
    # continuations that re-trip the guard and exhaust immediately).
    assert fake_agent.calls <= 4


def test_run_job_uses_deepagents_v3_final_values_when_no_delta_stream(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Say hello.",
            messages=[{"role": "user", "content": "Say hello."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeV3FinalValuesOnlyAgent(),
        )
        return result, published

    result, events = asyncio.run(scenario())

    assert result == "Hello from final state"
    assert [event["event_kind"] for event in events] == ["run.started", "run.completed"]
    assert events[-1]["payload"]["response_text"] == "Hello from final state"


def test_run_job_prefers_final_values_answer_over_process_stream(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Change the plot color.",
            messages=[{"role": "user", "content": "Change the plot color."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeV3StreamThenFinalAgent(),
        )
        return result, published

    result, events = asyncio.run(scenario())

    assert result == "Updated the plot to use a green line and saved the revised code and figure."
    assert events[1]["event_kind"] == "message.delta"
    assert events[1]["payload"]["text"] == "Let me recreate the script."
    assert events[-1]["event_kind"] == "run.completed"
    assert events[-1]["payload"]["response_text"] == result


def test_run_job_ignores_prior_assistant_when_followup_final_state_has_no_new_answer(
    tmp_path: Path,
):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Add a reference line.",
            messages=[
                {"role": "user", "content": "Create the first plot."},
                {"role": "assistant", "content": "Created the first plot."},
                {"role": "user", "content": "Add a reference line."},
            ],
        )
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeV3FollowupMissingFinalAssistantAgent(),
        )
        return result, published

    result, events = asyncio.run(scenario())

    assert result == "Updated the plot with a dashed y=10 reference line."
    assert events[-1]["event_kind"] == "run.completed"
    assert events[-1]["payload"]["response_text"] == result


def test_run_job_ignores_leading_whitespace_and_falls_back_to_saved_outputs(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            completion_max_continuations=0,
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Create a plot.",
            messages=[{"role": "user", "content": "Create a plot."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeWhitespaceOnlyOutputAgent(),
        )
        return result, published

    result, events = asyncio.run(scenario())

    assert result == (
        "Saved durable outputs:\n- Code: Plot X Squared "
        "([`outputs/plot_x_squared.py`]"
        "(/v2/artifacts/artifact_run_1_5d21e99b3940252f/download))"
    )
    assert [event["event_kind"] for event in events] == [
        "run.started",
        "trace.model.terminal",
        "artifact.created",
        "run.completed",
    ]
    assert events[-1]["payload"]["response_text"] == result


def test_link_response_artifact_paths_uses_unique_download_ids():
    from ultra_deepagents.runner import _link_response_artifact_paths

    response = (
        "Download `/outputs/results/report.md` and `predictions.jsonl`; "
        "leave `/outputs/results/` as a directory reference and `report.md` ambiguous."
    )
    artifacts = [
        {
            "payload": {
                "artifact_id": "artifact-run-report",
                "path": "results/report.md",
            }
        },
        {
            "payload": {
                "artifact_id": "artifact-run-predictions",
                "path": "results/predictions.jsonl",
            }
        },
        {"payload": {"artifact_id": "artifact-run-report-2", "path": "other/report.md"}},
        {"payload": {"artifact_id": "artifact-run-report-3", "path": "third/report.md"}},
    ]

    assert _link_response_artifact_paths(response, artifacts) == (
        "Download [`/outputs/results/report.md`]"
        "(/v2/artifacts/artifact-run-report/download) and "
        "[`predictions.jsonl`](/v2/artifacts/artifact-run-predictions/download); "
        "leave `/outputs/results/` as a directory reference and `report.md` ambiguous."
    )


def test_run_job_publishes_artifacts_from_explicit_outputs_directory(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Create a plot.",
            messages=[{"role": "user", "content": "Create a plot."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeOutputWritingAgent(),
        )
        return result, published

    result, events = asyncio.run(scenario())

    assert result == "Created plot."
    assert [event["event_kind"] for event in events] == [
        "run.started",
        "message.delta",
        "artifact.created",
        "artifact.created",
        "run.completed",
    ]
    artifact_payloads = [
        event["payload"] for event in events if event["event_kind"] == "artifact.created"
    ]
    assert [payload["path"] for payload in artifact_payloads] == [
        "outputs/plot_squared.png",
        "outputs/plot_squared.py",
    ]
    assert [payload["kind"] for payload in artifact_payloads] == ["figure", "code"]
    for payload in artifact_payloads:
        copied = tmp_path / "artifacts" / "run-1" / payload["path"]
        assert copied.exists()
        assert payload["relative_path"] == payload["path"]
        assert payload["source_path"] == str(copied)
        assert payload["storage_uri"] == f"file://{copied}"
        assert payload["sha256"]
        assert payload["size_bytes"] == copied.stat().st_size
    assert not (tmp_path / "artifacts" / "run-1" / "outputs" / "frame_006.png").exists()
    assert not (tmp_path / "artifacts" / "run-1" / "frame_007.png").exists()


def test_run_job_preserves_collected_plot_bytes_and_reports_source_ppi(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Create a publication-quality plot.",
            messages=[{"role": "user", "content": "Create a publication-quality plot."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeLowDpiPlotAgent(),
        )
        return result, published

    result, events = asyncio.run(scenario())

    assert result == "Created plot."
    artifact_payload = next(
        event["payload"] for event in events if event["event_kind"] == "artifact.created"
    )
    copied = tmp_path / "artifacts" / "run-1" / artifact_payload["path"]
    source = tmp_path / "workspaces" / "run-1" / "outputs" / "low_dpi_plot.png"
    with Image.open(copied) as image:
        dpi = image.info.get("dpi")

    assert dpi is not None
    assert dpi[0] == pytest.approx(72.0, abs=0.5)
    assert dpi[1] == pytest.approx(72.0, abs=0.5)
    assert copied.read_bytes() == source.read_bytes()
    quality = artifact_payload["metadata"]["figure_quality"]
    assert quality["minimum_ppi"] == 300
    assert quality["dpi_metadata_normalized"] is False
    assert quality["meets_minimum_ppi"] is False
    assert quality["original_ppi"] == pytest.approx([72.0, 72.0], abs=0.5)
    assert quality["final_ppi"] == pytest.approx([72.0, 72.0], abs=0.5)


def test_run_job_collects_markdown_reports_with_markdown_mime_type(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Write a combined research report.",
            messages=[{"role": "user", "content": "Write a combined research report."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeMarkdownReportAgent(),
        )
        return published

    events = asyncio.run(scenario())

    [artifact_payload] = [
        event["payload"] for event in events if event["event_kind"] == "artifact.created"
    ]
    assert artifact_payload["path"] == "outputs/rarespot_combined_report.md"
    assert artifact_payload["kind"] == "report"
    assert artifact_payload["mime_type"] == "text/markdown"


def test_run_job_deduplicates_same_file_saved_in_workspace_root_and_outputs(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Create a plot.",
            messages=[{"role": "user", "content": "Create a plot."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeDuplicateRootAndOutputsAgent(),
        )
        return published

    events = asyncio.run(scenario())

    artifact_payloads = [
        event["payload"] for event in events if event["event_kind"] == "artifact.created"
    ]
    assert [payload["path"] for payload in artifact_payloads] == [
        "outputs/plot_squared.png",
        "outputs/plot_squared.py",
    ]
    assert not (tmp_path / "artifacts" / "run-1" / "plot_squared.png").exists()
    assert not (tmp_path / "artifacts" / "run-1" / "plot_squared.py").exists()


def test_run_job_collects_top_level_workspace_deliverables_without_frames(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Create a plot.",
            messages=[{"role": "user", "content": "Create a plot."}],
        )
        published = []

        async def publish(event):
            published.append(event)

        await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeRootOutputWritingAgent(),
        )
        return published

    events = asyncio.run(scenario())

    artifact_payloads = [
        event["payload"] for event in events if event["event_kind"] == "artifact.created"
    ]
    assert [payload["path"] for payload in artifact_payloads] == [
        "plot_squared.png",
        "plot_squared.py",
    ]
    assert [payload["kind"] for payload in artifact_payloads] == ["figure", "code"]
    assert (tmp_path / "artifacts" / "run-1" / "plot_squared.png").exists()
    assert (tmp_path / "artifacts" / "run-1" / "plot_squared.py").exists()
    assert not (tmp_path / "artifacts" / "run-1" / "frame_007.png").exists()


def test_run_job_continues_when_explicit_plot_request_only_saves_code(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            completion_max_continuations=1,
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Create a matplotlib plot, save the code and plot as durable outputs.",
            messages=[
                {
                    "role": "user",
                    "content": "Create a matplotlib plot, save the code and plot as durable outputs.",
                }
            ],
        )
        fake_agent = FakePrematureCodeOnlyThenFigureAgent()
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: fake_agent,
        )
        return result, published, fake_agent

    result, events, fake_agent = asyncio.run(scenario())

    assert fake_agent.calls == 2
    # Terminal assembly stitches the pre-continuation answer in front of the
    # continuation's delta reply — the guard must not erase attempt 1's answer.
    assert result == (
        "Saved the script plot_x2.py.\n\n"
        "Executed the plotting script and saved the code (plot_x2.py) and figure (plot_x2.png)."
    )
    assert "figure" in fake_agent.continuation_prompt
    assert [event["event_kind"] for event in events] == [
        "run.started",
        "trace.message.delta",
        "artifact.created",
        "artifact.created",
        "run.completed",
    ]
    artifact_payloads = [
        event["payload"] for event in events if event["event_kind"] == "artifact.created"
    ]
    assert [payload["kind"] for payload in artifact_payloads] == ["figure", "code"]
    assert {payload["path"] for payload in artifact_payloads} == {"plot_x2.png", "plot_x2.py"}


def test_run_job_continues_when_training_request_omits_model_weights(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            completion_max_continuations=1,
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal=(
                "Train a UNet model, save the trained model weights, save plots, "
                "and save the training code as durable outputs."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Train a UNet model, save the trained model weights, save plots, "
                        "and save the training code as durable outputs."
                    ),
                }
            ],
        )
        fake_agent = FakePrematureTrainingNoWeightsThenCheckpointAgent()
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: fake_agent,
        )
        return result, published, fake_agent

    result, events, fake_agent = asyncio.run(scenario())

    assert fake_agent.calls == 2
    # Attempt 1's answer survives the completion-guard continuation.
    assert result == (
        "Saved the training script and training curves.\n\n"
        "Saved the trained model weights, code, and figures."
    )
    assert "model" in fake_agent.continuation_prompt
    trace_events = [event for event in events if event["event_kind"] == "trace.message.delta"]
    assert [event["payload"]["missing_artifact_kinds"] for event in trace_events] == [["model"]]
    artifact_payloads = [
        event["payload"] for event in events if event["event_kind"] == "artifact.created"
    ]
    assert {payload["kind"] for payload in artifact_payloads} == {"code", "figure", "model"}
    assert {payload["path"] for payload in artifact_payloads} == {
        "outputs/best_model.pth",
        "outputs/train_unet.py",
        "outputs/training_curves.png",
    }


def test_run_job_continues_when_requested_explanation_is_missing(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            completion_max_continuations=1,
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal=(
                "Use Python to create a small matplotlib figure showing y = x^2 "
                "for x = 0..5. Save the plotting code and plot as durable outputs, "
                "and briefly explain what the plot demonstrates."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Use Python to create a small matplotlib figure showing y = x^2 "
                        "for x = 0..5. Save the plotting code and plot as durable outputs, "
                        "and briefly explain what the plot demonstrates."
                    ),
                }
            ],
        )
        fake_agent = FakeArtifactsOnlyThenExplanationAgent()
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: fake_agent,
        )
        return result, published, fake_agent

    result, events, fake_agent = asyncio.run(scenario())

    assert fake_agent.calls == 2
    assert "quadratic growth" in result
    assert "Saved durable outputs:" not in result
    assert "final response" in fake_agent.continuation_prompt
    trace_events = [event for event in events if event["event_kind"] == "trace.message.delta"]
    assert [event["payload"]["missing_artifact_kinds"] for event in trace_events] == [["response"]]
    artifact_payloads = [
        event["payload"] for event in events if event["event_kind"] == "artifact.created"
    ]
    assert {payload["kind"] for payload in artifact_payloads} == {"code", "figure"}


def test_run_job_continues_when_only_process_text_precedes_final_tool(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            completion_max_continuations=1,
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Can you explain more of the theory?",
            messages=[
                {"role": "user", "content": "Write code and visualize how bubble sort works"},
                {"role": "assistant", "content": "Created code and visualizations."},
                {"role": "user", "content": "Can you explain more of the theory?"},
            ],
        )
        fake_agent = FakeProcessTextThenToolNoFinalAgent()
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: fake_agent,
        )
        return result, published, fake_agent

    result, events, fake_agent = asyncio.run(scenario())

    assert fake_agent.calls == 2
    assert "repeatedly comparing adjacent elements" in result
    assert "Let me inspect" not in result
    assert events[-1]["event_kind"] == "run.completed"
    assert events[-1]["payload"]["response_text"] == result
    trace_events = [event for event in events if event["event_kind"] == "trace.message.delta"]
    assert [event["payload"]["missing_artifact_kinds"] for event in trace_events] == [["response"]]


def test_run_job_recovers_empty_response_for_generic_visible_prompt(tmp_path: Path):
    prompt = "Can you provide some real computations on how delta attention is computed"

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            completion_max_continuations=2,
        )
        job = RunJobEnvelope(
            run_id="run-empty-recovery",
            thread_id="thread-empty-recovery",
            user_id="researcher-1",
            goal=prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        agent = FakeEmptyThenFinalResponseAgent()
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: agent,
        )
        return result, published, agent

    result, events, agent = asyncio.run(scenario())

    assert agent.calls == 2
    assert "Delta attention updates" in result
    assert "missing requested final response" in agent.continuation_prompt
    assert [
        event["payload"]["missing_artifact_kinds"]
        for event in events
        if event["event_kind"] == "trace.message.delta"
    ] == [["response"]]
    terminal_traces = [event for event in events if event["event_kind"] == "trace.model.terminal"]
    assert [event["payload"]["output_classification"] for event in terminal_traces] == ["empty"]
    assert terminal_traces[0]["payload"]["finish_reason"] == "stop"
    assert events[-1]["event_kind"] == "run.completed"
    assert events[-1]["payload"]["response_text"] == result


def test_run_job_fails_closed_when_visible_response_recovery_is_exhausted(tmp_path: Path):
    prompt = "Can you provide some real computations on how delta attention is computed"

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            completion_max_continuations=8,
        )
        job = RunJobEnvelope(
            run_id="run-empty-exhausted",
            thread_id="thread-empty-exhausted",
            user_id="researcher-1",
            goal=prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        agent = FakeAlwaysEmptyAgent()
        published = []

        async def publish(event):
            published.append(event)

        with pytest.raises(RuntimeError, match="without a user-visible response"):
            await run_job(
                job,
                settings,
                publish_event=publish,
                agent_factory=lambda *_args, **_kwargs: agent,
            )
        return published, agent

    events, agent = asyncio.run(scenario())

    assert agent.calls == 2
    assert not [event for event in events if event["event_kind"] == "run.completed"]
    assert events[-1]["event_kind"] == "run.failed"
    assert events[-1]["payload"]["error_type"] == "AgentEmptyResponseError"
    assert [
        event["payload"]["output_classification"]
        for event in events
        if event["event_kind"] == "trace.model.terminal"
    ] == ["empty", "empty"]


def test_run_job_preserves_empty_completion_for_explicit_internal_run(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-internal-empty",
            thread_id="thread-internal-empty",
            user_id="researcher-1",
            goal="Perform the internal tool handoff.",
            messages=[{"role": "tool", "content": "Internal tool handoff."}],
            metadata={"internal": True, "visible_in_thread": False},
        )
        agent = FakeAlwaysEmptyAgent()
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: agent,
        )
        return result, published, agent

    result, events, agent = asyncio.run(scenario())

    assert result == ""
    assert agent.calls == 1
    assert not [event for event in events if event["event_kind"] == "run.failed"]
    assert events[-1]["event_kind"] == "run.completed"
    assert events[-1]["payload"]["response_text"] == ""


def test_run_job_writes_workspace_lease_lifecycle(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Say hello.",
            messages=[{"role": "user", "content": "Say hello."}],
            metadata={"principal": {"org_id": "allen", "role": "researcher"}},
        )
        lease_path = tmp_path / "workspaces" / "run-1" / "lease.json"
        statuses = []

        async def publish(event):
            if event["event_kind"] in {"run.started", "run.completed"}:
                statuses.append(json.loads(lease_path.read_text())["status"])

        await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeStreamingAgent(),
        )
        return lease_path, statuses

    lease_path, statuses = asyncio.run(scenario())
    lease = json.loads(lease_path.read_text())

    assert statuses == ["running", "succeeded"]
    assert lease["run_id"] == "run-1"
    assert lease["thread_id"] == "thread-1"
    assert lease["user_id"] == "researcher-1"
    assert lease["org_id"] == "allen"
    assert lease["status"] == "succeeded"
    assert lease["cleanup_state"] == "active"
    assert lease["workspace_root"] == str(tmp_path / "workspaces" / "run-1")
    assert lease["artifact_root"] == str(tmp_path / "artifacts" / "run-1")
    assert lease["created_at"]
    assert lease["updated_at"]


def test_run_job_marks_workspace_lease_canceled_when_task_is_canceled(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Wait until canceled.",
            messages=[{"role": "user", "content": "Wait until canceled."}],
        )
        started = asyncio.Event()

        async def publish(_event):
            pass

        task = asyncio.create_task(
            run_job(
                job,
                settings,
                publish_event=publish,
                agent_factory=lambda *_args, **_kwargs: BlockingStreamingAgent(started),
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        lease_path = tmp_path / "workspaces" / "run-1" / "lease.json"
        return json.loads(lease_path.read_text())

    lease = asyncio.run(scenario())

    assert lease["status"] == "canceled"
    assert lease["error"] == "canceled"


def test_run_job_replay_produces_same_event_ids_for_deduplication(tmp_path: Path):
    first = asyncio.run(_run_fake_job(tmp_path))
    replay = asyncio.run(_run_fake_job(tmp_path))

    assert [event["event_id"] for event in replay] == [event["event_id"] for event in first]


def test_run_job_publishes_failed_terminal_event_on_agent_error(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-1",
            thread_id="thread-1",
            user_id="researcher-1",
            goal="Fail cleanly.",
            messages=[{"role": "user", "content": "Fail cleanly."}],
        )
        lease_path = tmp_path / "workspaces" / "run-1" / "lease.json"
        published = []

        async def publish(event):
            published.append(event)

        def failing_agent_factory(*_args, **_kwargs):
            raise RuntimeError("sandbox unavailable")

        with pytest.raises(RuntimeError, match="sandbox unavailable"):
            await run_job(
                job,
                settings,
                publish_event=publish,
                agent_factory=failing_agent_factory,
            )
        return published, json.loads(lease_path.read_text())

    events, lease = asyncio.run(scenario())

    assert [event["event_kind"] for event in events] == ["run.started", "run.failed"]
    assert events[-1]["payload"]["error"] == "sandbox unavailable"
    assert lease["status"] == "failed"
    assert lease["error"] == "sandbox unavailable"


class TimeoutSubscription:
    async def fetch(self, *_args, **_kwargs):
        raise nats.errors.TimeoutError


class BareAsyncioTimeoutSubscription:
    """What JetStream's own _fetch_n raises when an idle pull finds nothing:
    a BARE asyncio.TimeoutError, not the nats.errors flavour."""

    async def fetch(self, *_args, **_kwargs):
        raise asyncio.TimeoutError


class ConnectionClosedSubscription:
    async def fetch(self, *_args, **_kwargs):
        raise nats.errors.ConnectionClosedError


def test_fetch_job_messages_treats_nats_timeout_as_empty_poll():
    assert asyncio.run(fetch_job_messages(TimeoutSubscription())) == []


def test_fetch_job_messages_treats_bare_asyncio_timeout_as_empty_poll():
    """Regression: an idle JetStream fetch raises a bare asyncio.TimeoutError, which is
    NOT a nats.errors.TimeoutError. It used to escape this helper and then match the
    OSError arm of _RECOVERABLE_NATS_ERRORS (asyncio.TimeoutError is builtin
    TimeoutError, an OSError subclass, on 3.11+), so every idle poll was misreported as
    'NATS connection lost' and reconnected the pump."""

    assert asyncio.run(fetch_job_messages(BareAsyncioTimeoutSubscription())) == []


def test_bare_asyncio_timeout_would_be_mistaken_for_a_recoverable_nats_error():
    """Pins the hazard the fix guards against: if a bare timeout ever escapes a call
    site, the supervisor's OSError arm swallows it as a connection failure. This is why
    timeouts must be caught where they are raised, not left to the supervisor."""

    assert issubclass(asyncio.TimeoutError, OSError)
    assert not issubclass(asyncio.TimeoutError, nats.errors.TimeoutError)
    assert isinstance(asyncio.TimeoutError(), _RECOVERABLE_NATS_ERRORS)


def test_fetch_job_messages_still_propagates_real_connection_failures():
    """The pump must NOT go silent-busy-loop on a dead link: a closed connection is a
    genuine transport failure and has to reach the supervisor so it reconnects."""

    with pytest.raises(nats.errors.ConnectionClosedError):
        asyncio.run(fetch_job_messages(ConnectionClosedSubscription()))


def test_job_consumer_config_uses_long_running_ack_settings():
    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        nats_worker_durable="worker-a",
        nats_jobs_subject="ultra.test.jobs",
        worker_ack_wait_seconds=300.0,
        worker_ack_progress_interval_seconds=60.0,
        worker_max_deliver=7,
        worker_max_concurrency=3,
    )

    config = build_job_consumer_config(settings)

    assert config.durable_name == "worker-a"
    assert config.filter_subject == "ultra.test.jobs"
    assert config.ack_policy == AckPolicy.EXPLICIT
    assert config.ack_wait == 300.0
    assert config.max_deliver == 7
    assert config.max_ack_pending == 3
    assert job_ack_extension_interval(settings) == 60.0


def test_job_consumer_config_match_rejects_non_explicit_ack_policy():
    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        nats_worker_durable="worker-a",
        nats_jobs_subject="ultra.test.jobs",
    )
    desired = build_job_consumer_config(settings)
    existing = SimpleNamespace(
        filter_subject=desired.filter_subject,
        ack_wait=desired.ack_wait,
        max_deliver=desired.max_deliver,
        max_ack_pending=desired.max_ack_pending,
        ack_policy=AckPolicy.NONE,
    )

    assert not nats_worker_module._consumer_config_matches(existing, desired)


def test_job_ack_extension_interval_stays_below_ack_wait_for_long_running_leases():
    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        worker_ack_wait_seconds=300.0,
        worker_ack_progress_interval_seconds=600.0,
    )

    assert job_ack_extension_interval(settings) == 150.0


class FakeNATSMessage:
    def __init__(self, payload: bytes):
        self.data = payload
        self.acked = 0
        self.naked = 0
        self.nak_delays = []
        self.in_progress_calls = 0

    async def ack(self):
        self.acked += 1

    async def nak(self, delay=None):
        self.naked += 1
        self.nak_delays.append(delay)

    async def in_progress(self):
        self.in_progress_calls += 1


class FailingAckNATSMessage(FakeNATSMessage):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.ack_attempts = 0

    async def ack(self):
        self.ack_attempts += 1
        raise RuntimeError("simulated acknowledgement loss")


class CapturingJetStream:
    def __init__(self):
        self.published = []

    async def publish(self, subject: str, payload: bytes, **kwargs):
        self.published.append((subject, payload, kwargs.get("headers") or {}))


class StreamConfigCapturingJetStream:
    def __init__(self):
        self.stream_name = ""
        self.subjects = []

    async def add_stream(self, *, name: str, subjects: list[str]):
        self.stream_name = name
        self.subjects = subjects


class HeartbeatCapturingJetStream(CapturingJetStream):
    def __init__(self):
        super().__init__()
        self.heartbeat_seen = asyncio.Event()

    async def publish(self, subject: str, payload: bytes, **kwargs):
        await super().publish(subject, payload, **kwargs)
        event = json.loads(payload.decode("utf-8"))
        if event.get("event_kind") == "run.heartbeat":
            self.heartbeat_seen.set()


class FailingJetStream:
    async def publish(self, subject: str, payload: bytes, **_kwargs):
        raise RuntimeError("nats publish unavailable")


def _published_events(js: CapturingJetStream):
    return [json.loads(payload.decode("utf-8")) for _subject, payload, _headers in js.published]


def _published_headers(js: CapturingJetStream):
    return [headers for _subject, _payload, headers in js.published]


def _published_subjects(js: CapturingJetStream):
    return [subject for subject, _payload, _headers in js.published]


def _assert_worker_skipped_event(
    event,
    *,
    run_id: str,
    thread_id: str,
    control_status: str,
    stage: str,
):
    assert event == {
        "event_id": f"evt_{run_id}_worker_skipped",
        "sequence": 1,
        "run_id": run_id,
        "thread_id": thread_id,
        "event_kind": "run.worker_skipped",
        "event_type": "run",
        "node_name": "worker",
        "agent_role": "worker",
        "level": "warning",
        "message": "Worker skipped job because the control plane run is not runnable.",
        "payload": {
            "ack_action": "ack",
            "control_status": control_status,
            "reason": "control_plane_not_runnable",
            "stage": stage,
            "worker_id": "ultra-deepagents-worker",
        },
    }


def test_worker_publishes_events_with_deterministic_jetstream_message_id():
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="test-model",
            nats_url="nats://example.test:4222",
            nats_events_subject="ultra.runs.events",
        )
        worker = NATSDeepAgentsWorker(settings)
        js = CapturingJetStream()
        await worker._publish_event(
            js,
            {
                "event_id": "evt-run-1-started",
                "run_id": "run-1",
                "sequence": 1,
                "event_kind": "run.started",
            },
        )
        return js

    js = asyncio.run(scenario())

    assert _published_headers(js)[0]["Nats-Msg-Id"] == "event:evt-run-1-started"


def test_worker_sanitizes_nul_bytes_before_publishing_run_events():
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="test-model",
            nats_url="nats://example.test:4222",
            nats_events_subject="ultra.runs.events",
        )
        worker = NATSDeepAgentsWorker(settings)
        js = CapturingJetStream()
        await worker._publish_event(
            js,
            {
                "event_id": "evt-run-nul-progress",
                "run_id": "run-nul-progress",
                "sequence": 1,
                "event_kind": "tool_call.progress",
                "message": "/workspace/train.py\x00155:def train():",
                "payload": {
                    "text": "/workspace/train.py\x00155:def train():",
                    "nested": ["safe", {"key\x00suffix": "value\x00suffix"}],
                },
            },
        )
        return js

    js = asyncio.run(scenario())

    raw_payload = js.published[0][1].decode("utf-8")
    event = _published_events(js)[0]
    assert "\\u0000" not in raw_payload
    assert "\x00" not in raw_payload
    assert event["message"] == "/workspace/train.py\\0155:def train():"
    assert event["payload"]["text"] == "/workspace/train.py\\0155:def train():"
    assert event["payload"]["nested"][1] == {"key\\0suffix": "value\\0suffix"}


def test_worker_publishes_run_events_to_deterministic_partition_subject():
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="test-model",
            nats_url="nats://example.test:4222",
            nats_events_subject="ultra.runs.events",
            nats_event_partitions=8,
        )
        worker = NATSDeepAgentsWorker(settings)
        js = CapturingJetStream()
        event = {
            "event_id": "evt-run-partition-started",
            "run_id": "run-partitioned",
            "sequence": 1,
            "event_kind": "run.started",
        }

        await worker._publish_event(js, event)
        await worker._publish_event(
            js, {**event, "event_id": "evt-run-partition-second", "sequence": 2}
        )

        return js

    js = asyncio.run(scenario())

    subjects = _published_subjects(js)
    assert subjects == [subjects[0], subjects[0]]
    assert subjects[0].startswith("ultra.runs.events.p.")
    partition = int(subjects[0].removeprefix("ultra.runs.events.p."))
    assert 0 <= partition < 8


def test_worker_stream_includes_partitioned_run_event_subjects():
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="test-model",
            nats_stream="ULTRA_TEST",
            nats_jobs_subject="ultra.runs.jobs",
            nats_events_subject="ultra.runs.events",
            nats_cancel_subject="ultra.runs.cancel",
        )
        worker = NATSDeepAgentsWorker(settings)
        js = StreamConfigCapturingJetStream()

        await worker._ensure_stream(js)

        return js

    js = asyncio.run(scenario())

    assert js.stream_name == "ULTRA_TEST"
    assert "ultra.runs.events" in js.subjects
    assert "ultra.runs.events.p.*" in js.subjects


def test_worker_acks_failed_job_without_crashing_loop():
    async def failing_run_job(*_args, **_kwargs):
        raise RuntimeError("agent failed")

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            worker_ack_progress_interval_seconds=30.0,
        )
        worker = NATSDeepAgentsWorker(settings, run_job_func=failing_run_job)
        message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"fail"}'
        )
        js = CapturingJetStream()
        await worker._process_message(message, js)
        return message, _published_events(js)

    message, events = asyncio.run(scenario())

    assert message.acked == 1
    assert message.naked == 0
    assert events == [
        {
            "event_id": "evt_run-1_worker_failed",
            "sequence": 1,
            "run_id": "run-1",
            "thread_id": "thread-1",
            "event_kind": "run.failed",
            "event_type": "run",
            "node_name": "coordinator",
            "agent_role": "coordinator",
            "level": "error",
            "message": "Run failed.",
            "payload": {
                "error": "agent failed",
                "error_type": "RuntimeError",
                "stage": "worker",
            },
        }
    ]


def test_worker_redacts_notes_failure_fallback_and_logs(caplog):
    sentinel = "NOTE_SENTINEL_IN_WORKER_EXCEPTION"

    async def failing_run_job(*_args, **_kwargs):
        raise RuntimeError(sentinel)

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            worker_ack_progress_interval_seconds=30.0,
        )
        worker = NATSDeepAgentsWorker(settings, run_job_func=failing_run_job)
        message = FakeNATSMessage(
            json.dumps(
                {
                    "run_id": "run-notes-failed",
                    "thread_id": "thread-notes-failed",
                    "user_id": "user-1",
                    "goal": "Use my attached note.",
                    "selection_context": {
                        "note_access": {
                            "mode": "selected",
                            "notes": [{"note_id": "note-private", "revision": 1}],
                        }
                    },
                }
            ).encode()
        )
        js = CapturingJetStream()
        await worker._process_message(message, js)
        return message, _published_events(js)

    with caplog.at_level("ERROR"):
        message, events = asyncio.run(scenario())

    assert message.acked == 1
    assert events[-1]["payload"] == {
        "error": "Notes-enabled run failed.",
        "error_type": "RuntimeError",
        "stage": "worker",
        "redacted": True,
    }
    assert sentinel not in json.dumps(events)
    assert sentinel not in caplog.text


def test_worker_does_not_publish_fallback_failure_after_terminal_event_then_cleanup_error():
    async def run_job_publishes_terminal_then_raises(*_args, **kwargs):
        await kwargs["publish_event"](
            {
                "event_id": "evt-run-1-completed",
                "run_id": "run-1",
                "thread_id": "thread-1",
                "event_kind": "run.completed",
                "payload": {"response_text": "done"},
            }
        )
        raise RuntimeError("cleanup failed after terminal event")

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            worker_ack_progress_interval_seconds=30.0,
        )
        worker = NATSDeepAgentsWorker(
            settings,
            run_job_func=run_job_publishes_terminal_then_raises,
        )
        message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"finish"}'
        )
        js = CapturingJetStream()
        await worker._process_message(message, js)
        return message, _published_events(js)

    message, events = asyncio.run(scenario())

    assert message.acked == 1
    assert message.naked == 0
    assert [event["event_kind"] for event in events] == ["run.completed"]
    assert events[0]["event_id"] == "evt-run-1-completed"


def test_worker_publishes_fallback_failure_after_nonterminal_event_then_error():
    async def run_job_publishes_nonterminal_then_raises(*_args, **kwargs):
        await kwargs["publish_event"](
            {
                "event_id": "evt-run-1-started",
                "sequence": 5,
                "run_id": "run-1",
                "thread_id": "thread-1",
                "event_kind": "run.started",
                "payload": {},
            }
        )
        raise RuntimeError("agent failed after start")

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            worker_ack_progress_interval_seconds=30.0,
        )
        worker = NATSDeepAgentsWorker(
            settings,
            run_job_func=run_job_publishes_nonterminal_then_raises,
        )
        message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"fail"}'
        )
        js = CapturingJetStream()
        await worker._process_message(message, js)
        return message, _published_events(js)

    message, events = asyncio.run(scenario())

    assert message.acked == 1
    assert message.naked == 0
    assert [event["event_kind"] for event in events] == ["run.started", "run.failed"]
    assert [event["sequence"] for event in events] == [5, 6]
    assert events[-1]["event_id"] == "evt_run-1_worker_failed"
    assert events[-1]["payload"]["error"] == "agent failed after start"


def test_worker_publishes_completion_if_runner_returns_without_terminal_event():
    async def run_job_returns_without_terminal(*_args, **kwargs):
        await kwargs["publish_event"](
            {
                "event_id": "evt-run-1-started",
                "sequence": 5,
                "run_id": "run-1",
                "thread_id": "thread-1",
                "event_kind": "run.started",
                "payload": {},
            }
        )
        return "runner returned a final answer"

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            worker_ack_progress_interval_seconds=30.0,
        )
        worker = NATSDeepAgentsWorker(
            settings,
            run_job_func=run_job_returns_without_terminal,
        )
        message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"finish"}'
        )
        js = CapturingJetStream()
        await worker._process_message(message, js)
        return message, _published_events(js)

    message, events = asyncio.run(scenario())

    assert message.acked == 1
    assert message.naked == 0
    assert [event["event_kind"] for event in events] == ["run.started", "run.completed"]
    assert [event["sequence"] for event in events] == [5, 6]
    assert events[-1]["event_id"] == "evt_run-1_worker_completed"
    assert events[-1]["payload"]["response_text"] == "runner returned a final answer"


def test_worker_seeds_sequence_floor_without_checkpointer():
    async def run_job_returns_without_terminal(*_args, **_kwargs):
        return "runner returned after redelivery"

    class SnapshotWorker(NATSDeepAgentsWorker):
        async def _ensure_checkpointer(self):
            return None

        async def _run_events_snapshot(self, run_id: str):
            assert run_id == "run-redelivered"
            return 42, None

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            worker_ack_progress_interval_seconds=0,
        )
        worker = SnapshotWorker(
            settings,
            run_job_func=run_job_returns_without_terminal,
        )
        message = FakeNATSMessage(
            b'{"run_id":"run-redelivered","thread_id":"thread-1","user_id":"user-1","goal":"finish"}'
        )
        js = CapturingJetStream()
        await worker._process_message(message, js)
        return message, _published_events(js)

    message, events = asyncio.run(scenario())

    assert message.acked == 1
    assert message.naked == 0
    assert [event["event_kind"] for event in events] == ["run.completed"]
    assert events[-1]["sequence"] == 43


@pytest.mark.parametrize("status_code", [404, 500])
def test_worker_waits_without_compute_when_resume_snapshot_is_temporarily_unavailable(
    monkeypatch,
    status_code,
):
    monkeypatch.setattr(
        NATSDeepAgentsWorker,
        "_run_events_snapshot",
        _REAL_RUN_EVENTS_SNAPSHOT,
    )
    run_job_calls = 0
    snapshot_calls = 0
    js = CapturingJetStream()

    async def run_job_starts_after_snapshot(*_args, **_kwargs):
        nonlocal run_job_calls
        assert snapshot_calls == 7
        run_job_calls += 1
        return "safe resumed compute"

    async def snapshot_temporarily_unavailable(_run_id, _settings):
        nonlocal snapshot_calls
        snapshot_calls += 1
        if snapshot_calls <= 6:
            assert _published_events(js) == []
            raise urllib_error.HTTPError(
                "http://control.test/v2/runs/run-redelivered/events",
                status_code,
                "temporarily unavailable",
                {},
                None,
            )
        return 42, None

    async def run_status(_run_id, _settings):
        return None

    async def run_lease(_run_id, _settings):
        return None

    async def worker_heartbeat(_settings, *, status, current_run_id=None, metadata=None):
        return None

    async def user_profile(_run_id, _settings):
        return None

    monkeypatch.setattr(
        nats_worker_module,
        "fetch_control_plane_run_events_snapshot",
        snapshot_temporarily_unavailable,
    )
    monkeypatch.setattr(
        nats_worker_module,
        "_run_events_snapshot_retry_delay",
        lambda _attempt: 0.002,
    )

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            control_base_url="http://control.test",
            worker_ack_progress_interval_seconds=0.001,
        )
        worker = NATSDeepAgentsWorker(
            settings,
            run_job_func=run_job_starts_after_snapshot,
            run_status_func=run_status,
            run_lease_func=run_lease,
            worker_heartbeat_func=worker_heartbeat,
            user_profile_func=user_profile,
        )
        message = FakeNATSMessage(
            b'{"run_id":"run-redelivered","thread_id":"thread-1","user_id":"user-1","goal":"resume"}'
        )
        await worker._process_message(message, js)
        return message, _published_events(js)

    message, events = asyncio.run(scenario())

    assert snapshot_calls == 7
    assert run_job_calls == 1
    assert message.acked == 1
    assert message.naked == 0
    assert message.in_progress_calls > 0
    assert events[-1]["event_kind"] == "run.completed"
    assert events[-1]["sequence"] >= 43


@pytest.mark.parametrize(
    "failure",
    [
        urllib_error.HTTPError("http://control.test/events", 401, "unauthorized", {}, None),
        ValueError("malformed authority page"),
    ],
)
def test_worker_redelivers_nonretryable_snapshot_authority_failure(
    monkeypatch,
    failure,
):
    monkeypatch.setattr(
        NATSDeepAgentsWorker,
        "_run_events_snapshot",
        _REAL_RUN_EVENTS_SNAPSHOT,
    )
    run_job_calls = 0

    async def fail_snapshot(_run_id, _settings):
        raise failure

    async def run_job(*_args, **_kwargs):
        nonlocal run_job_calls
        run_job_calls += 1

    monkeypatch.setattr(
        nats_worker_module,
        "fetch_control_plane_run_events_snapshot",
        fail_snapshot,
    )
    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
    )
    worker = NATSDeepAgentsWorker(settings, run_job_func=run_job)
    message = FakeNATSMessage(
        b'{"run_id":"run-authority","thread_id":"thread-1","user_id":"user-1","goal":"resume"}'
    )
    js = CapturingJetStream()

    asyncio.run(worker._process_message(message, js))

    assert run_job_calls == 0
    assert message.acked == 0
    assert message.naked == 1
    assert _published_events(js) == []


def test_worker_redelivers_after_transient_snapshot_authority_budget_expires(monkeypatch):
    monkeypatch.setattr(
        NATSDeepAgentsWorker,
        "_run_events_snapshot",
        _REAL_RUN_EVENTS_SNAPSHOT,
    )
    monotonic_values = iter([0.0, 301.0])

    async def fail_snapshot(_run_id, _settings):
        raise urllib_error.HTTPError("http://control.test/events", 500, "unavailable", {}, None)

    monkeypatch.setattr(
        nats_worker_module, "fetch_control_plane_run_events_snapshot", fail_snapshot
    )
    monkeypatch.setattr(nats_worker_module, "_authority_monotonic", lambda: next(monotonic_values))
    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
    )
    worker = NATSDeepAgentsWorker(settings, run_job_func=lambda *_args, **_kwargs: None)
    message = FakeNATSMessage(
        b'{"run_id":"run-budget","thread_id":"thread-1","user_id":"user-1","goal":"resume"}'
    )

    asyncio.run(worker._process_message(message, CapturingJetStream()))

    assert message.acked == 0
    assert message.naked == 1


@pytest.mark.parametrize("mode", ["success", "failure", "canceled"])
def test_worker_clears_checkpoint_runtime_state_after_terminal_job(mode: str, tmp_path: Path):
    class CleanupTrackingCheckpointer:
        def __init__(self) -> None:
            self.cleared: list[str] = []

        def clear_thread(self, thread_id: str) -> None:
            self.cleared.append(thread_id)

    checkpointer = CleanupTrackingCheckpointer()
    run_job_calls = 0

    async def run_job_func(*_args, **kwargs):
        nonlocal run_job_calls
        run_job_calls += 1
        assert kwargs["checkpointer"] is checkpointer
        if mode == "failure":
            raise RuntimeError("agent failed")
        return "ok"

    async def run_status(_run_id, _settings):
        return None

    async def run_lease(_run_id, _settings):
        return None

    async def worker_heartbeat(_settings, *, status, current_run_id=None, metadata=None):
        return None

    class CleanupWorker(NATSDeepAgentsWorker):
        async def _ensure_checkpointer(self):
            return checkpointer

        async def _run_events_snapshot(self, run_id: str):
            assert run_id == "run-cleanup"
            return 0, None

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / f"workspaces-{mode}"),
            worker_ack_progress_interval_seconds=0,
        )
        worker = CleanupWorker(
            settings,
            run_job_func=run_job_func,
            run_status_func=run_status,
            run_lease_func=run_lease,
            worker_heartbeat_func=worker_heartbeat,
        )
        worker._checkpointer = checkpointer
        if mode == "canceled":
            worker._canceled_run_reasons["run-cleanup"] = "user requested cancel"
        message = FakeNATSMessage(
            b'{"run_id":"run-cleanup","thread_id":"thread-1","user_id":"user-1","goal":"cleanup"}'
        )
        js = CapturingJetStream()
        await worker._process_message(message, js)
        return message, _published_events(js)

    message, events = asyncio.run(scenario())

    assert message.acked == 1
    assert message.naked == 0
    assert checkpointer.cleared == ["run-cleanup"]
    if mode == "canceled":
        assert run_job_calls == 0
        assert [event["event_kind"] for event in events] == ["run.canceled"]
    elif mode == "failure":
        assert run_job_calls == 1
        assert [event["event_kind"] for event in events] == ["run.failed"]
    else:
        assert run_job_calls == 1
        assert [event["event_kind"] for event in events] == ["run.completed"]


def test_worker_retains_checkpoint_when_terminal_ack_fails_then_cleans_on_redelivery(
    tmp_path: Path,
):
    calls: list[tuple[str, str]] = []
    compute_calls = 0
    terminal = False

    class TrackingCheckpointer:
        async def flush(self, run_id: str) -> bool:
            calls.append(("flush", run_id))
            return True

        def clear_thread(self, run_id: str) -> None:
            calls.append(("clear", run_id))

        async def delete_thread(self, run_id: str) -> None:
            calls.append(("delete", run_id))

    checkpointer = TrackingCheckpointer()

    async def run_job_func(*_args, **_kwargs):
        nonlocal compute_calls, terminal
        compute_calls += 1
        terminal = True
        return "ok"

    async def run_status(_run_id, _settings):
        return "succeeded" if terminal else None

    async def run_lease(_run_id, _settings):
        return None

    async def worker_heartbeat(_settings, *, status, current_run_id=None, metadata=None):
        return None

    class TrackingWorker(NATSDeepAgentsWorker):
        async def _ensure_checkpointer(self):
            return checkpointer

        async def _run_events_snapshot(self, run_id: str):
            assert run_id == "run-ack-loss"
            return 0, None

    async def scenario():
        worker = TrackingWorker(
            RuntimeSettings(
                openai_base_url="http://example.test/v1",
                openai_model="deepseek_v4",
                workspace_root=str(tmp_path / "workspaces"),
                worker_ack_progress_interval_seconds=0,
            ),
            run_job_func=run_job_func,
            run_status_func=run_status,
            run_lease_func=run_lease,
            worker_heartbeat_func=worker_heartbeat,
        )
        worker._checkpointer = checkpointer
        payload = (
            b'{"run_id":"run-ack-loss","thread_id":"thread-1",'
            b'"user_id":"user-1","goal":"resume safely"}'
        )
        first = FailingAckNATSMessage(payload)
        await worker._process_message(first, CapturingJetStream())
        second = FakeNATSMessage(payload)
        await worker._process_message(second, CapturingJetStream())
        return first, second

    first, second = asyncio.run(scenario())

    assert compute_calls == 1
    assert first.ack_attempts == 1
    assert first.acked == 0
    assert second.acked == 1
    assert calls == [
        ("flush", "run-ack-loss"),
        ("clear", "run-ack-loss"),
        ("delete", "run-ack-loss"),
    ]


def test_completed_checkpoint_waits_for_terminal_ingest_without_restarting_graph(
    tmp_path: Path,
):
    calls: list[tuple[str, str]] = []
    run_job_calls = 0
    control_status = "running"

    class TrackingCheckpointer:
        async def flush(self, run_id: str) -> bool:
            calls.append(("flush", run_id))
            return True

        def clear_thread(self, run_id: str) -> None:
            calls.append(("clear", run_id))

        async def delete_thread(self, run_id: str) -> None:
            calls.append(("delete", run_id))

    checkpointer = TrackingCheckpointer()

    async def completed_checkpoint_guard(*_args, **_kwargs):
        nonlocal run_job_calls
        run_job_calls += 1
        # Let several enabled heartbeat intervals elapse before completed-state
        # reconciliation returns. The runner has not signaled event authority,
        # so no heartbeat may claim the terminal event's source sequence.
        await asyncio.sleep(0.05)
        raise CheckpointReconciliationPendingError(
            "completed checkpoint is awaiting terminal reconciliation"
        )

    async def run_status(_run_id, _settings):
        return control_status

    async def worker_heartbeat(_settings, *, status, current_run_id=None, metadata=None):
        return None

    class ReconciliationWorker(NATSDeepAgentsWorker):
        async def _ensure_checkpointer(self):
            self._checkpointer = checkpointer
            return checkpointer

    async def scenario():
        nonlocal control_status
        worker = ReconciliationWorker(
            RuntimeSettings(
                openai_base_url="http://example.test/v1",
                openai_model="deepseek_v4",
                workspace_root=str(tmp_path / "workspaces-terminal-lag"),
                worker_ack_wait_seconds=1.0,
                worker_ack_progress_interval_seconds=0.01,
            ),
            run_job_func=completed_checkpoint_guard,
            run_status_func=run_status,
            worker_heartbeat_func=worker_heartbeat,
        )
        payload = (
            b'{"run_id":"run-terminal-lag","thread_id":"thread-1",'
            b'"user_id":"user-1","goal":"do not replay"}'
        )
        first = FakeNATSMessage(payload)
        first_events = CapturingJetStream()
        await worker._process_message(first, first_events)
        control_status = "succeeded"
        second = FakeNATSMessage(payload)
        second_events = CapturingJetStream()
        await worker._process_message(second, second_events)
        return first, second, _published_events(first_events), _published_events(second_events)

    first, second, first_events, second_events = asyncio.run(scenario())

    assert run_job_calls == 1
    assert first.acked == 0
    assert first.naked == 1
    assert first_events == []
    assert second.acked == 1
    assert second.naked == 0
    assert [event["event_kind"] for event in second_events] == ["run.worker_skipped"]
    assert calls == [
        ("flush", "run-terminal-lag"),
        ("clear", "run-terminal-lag"),
        ("delete", "run-terminal-lag"),
    ]


def test_worker_periodically_reaps_abandoned_checkpoints():
    class TrackingCheckpointer:
        def __init__(self) -> None:
            self.retentions: list[int] = []
            self.reaped_twice = asyncio.Event()

        async def gc(self, retention_seconds: int) -> int:
            self.retentions.append(retention_seconds)
            if len(self.retentions) >= 2:
                self.reaped_twice.set()
            return 1

    checkpointer = TrackingCheckpointer()

    class CheckpointGCWorker(NATSDeepAgentsWorker):
        async def _ensure_checkpointer(self):
            return checkpointer

    async def scenario() -> list[int]:
        worker = CheckpointGCWorker(
            RuntimeSettings(
                openai_base_url="http://example.test/v1",
                openai_model="deepseek_v4",
                checkpoint_retention_seconds=72 * 3600,
                checkpoint_gc_interval_seconds=0.01,
            )
        )
        task = asyncio.create_task(worker._checkpoint_gc_loop())
        await asyncio.wait_for(checkpointer.reaped_twice.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return checkpointer.retentions

    assert asyncio.run(scenario())[:2] == [72 * 3600, 72 * 3600]


def test_worker_flushes_then_clears_resumable_checkpoint_from_memory():
    calls: list[tuple[str, str]] = []

    class TrackingCheckpointer:
        async def flush(self, run_id: str) -> bool:
            calls.append(("flush", run_id))
            return True

        def clear_thread(self, run_id: str) -> None:
            calls.append(("clear", run_id))

    worker = NATSDeepAgentsWorker(
        RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
        )
    )
    worker._checkpointer = TrackingCheckpointer()

    asyncio.run(worker._release_checkpointer_thread("run-resumable"))

    assert calls == [("flush", "run-resumable"), ("clear", "run-resumable")]


def test_worker_retains_freshest_runtime_checkpoint_when_flush_fails():
    calls: list[tuple[str, str]] = []

    class FailingFlushCheckpointer:
        async def flush(self, run_id: str) -> bool:
            calls.append(("flush", run_id))
            return False

        def clear_thread(self, run_id: str) -> None:
            calls.append(("clear", run_id))

    worker = NATSDeepAgentsWorker(
        RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
        )
    )
    worker._checkpointer = FailingFlushCheckpointer()

    asyncio.run(worker._release_checkpointer_thread("run-flush-failed"))

    assert calls == [("flush", "run-flush-failed")]


def test_worker_claims_and_releases_control_plane_run_lease_around_compute(tmp_path: Path):
    calls: list[tuple[str, str]] = []
    run_job_kwargs: dict[str, Any] = {}

    async def run_job_returns(*_args, **kwargs):
        calls.append(("run_job", "run-1"))
        run_job_kwargs.update(kwargs)
        return "ok"

    async def acquire_lease(run_id, settings):
        calls.append(("acquire", run_id))
        return ControlPlaneRunLease(
            run_id=run_id,
            worker_id="worker-a",
            lease_token="lease-token-1",
        )

    def renew_lease_sync(lease, settings):
        calls.append(("renew", lease.lease_token))
        return lease

    async def release_lease(lease, settings):
        calls.append(("release", lease.lease_token))

    async def run_status(_run_id, _settings):
        return None

    async def worker_heartbeat(_settings, **_kwargs):
        return None

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            control_base_url="http://control.test",
            worker_id="worker-a",
            workspace_root=str(tmp_path / "workspaces"),
            worker_ack_progress_interval_seconds=0,
        )
        worker = NATSDeepAgentsWorker(
            settings,
            run_job_func=run_job_returns,
            run_status_func=run_status,
            run_lease_func=acquire_lease,
            lease_renew_sync_func=renew_lease_sync,
            release_run_lease_func=release_lease,
            worker_heartbeat_func=worker_heartbeat,
        )
        message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"lease"}'
        )
        await worker._process_message(message, CapturingJetStream())
        return message

    message = asyncio.run(scenario())

    assert message.acked == 1
    assert message.naked == 0
    assert calls == [
        ("acquire", "run-1"),
        ("run_job", "run-1"),
        ("release", "lease-token-1"),
    ]
    assert run_job_kwargs["run_lease_worker_id"] == "worker-a"
    assert run_job_kwargs["run_lease_token"] == "lease-token-1"


def test_worker_retries_optional_control_plane_lease_before_snapshot_or_compute(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        NATSDeepAgentsWorker,
        "_acquire_run_lease_for_delivery",
        _REAL_ACQUIRE_RUN_LEASE_FOR_DELIVERY,
    )
    monkeypatch.setattr(
        nats_worker_module,
        "_run_events_snapshot_retry_delay",
        lambda _attempt: 0,
    )
    lease_calls = 0
    snapshot_calls = 0
    compute_calls = 0

    async def acquire_lease(run_id, _settings):
        nonlocal lease_calls
        lease_calls += 1
        if lease_calls <= 2:
            return None
        return ControlPlaneRunLease(run_id, "worker-a", "lease-token")

    async def release_lease(_lease, _settings):
        return None

    class SnapshotWorker(NATSDeepAgentsWorker):
        async def _run_events_snapshot(self, _run_id):
            nonlocal snapshot_calls
            snapshot_calls += 1
            assert lease_calls == 3
            return 0, None

    async def run_job(*_args, **_kwargs):
        nonlocal compute_calls
        compute_calls += 1
        return "ok"

    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        worker_id="worker-a",
        workspace_root=str(tmp_path / "workspaces"),
    )
    worker = SnapshotWorker(
        settings,
        run_job_func=run_job,
        run_lease_func=acquire_lease,
        release_run_lease_func=release_lease,
    )
    message = FakeNATSMessage(
        b'{"run_id":"run-lease-retry","thread_id":"thread-1","user_id":"user-1","goal":"resume"}'
    )

    asyncio.run(worker._process_message(message, CapturingJetStream()))

    assert lease_calls == 3
    assert snapshot_calls == 1
    assert compute_calls == 1
    assert message.acked == 1
    assert message.naked == 0


def test_worker_retries_required_transient_lease_failure_in_same_delivery(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        NATSDeepAgentsWorker,
        "_acquire_run_lease_for_delivery",
        _REAL_ACQUIRE_RUN_LEASE_FOR_DELIVERY,
    )
    monkeypatch.setattr(nats_worker_module, "_run_events_snapshot_retry_delay", lambda _: 0.002)
    calls = 0

    async def acquire_lease(run_id, _settings):
        nonlocal calls
        calls += 1
        if calls <= 6:
            raise RunLeaseUnavailable("connection refused", retryable=True)
        return ControlPlaneRunLease(run_id, "worker-a", "lease-token")

    async def release_lease(_lease, _settings):
        return None

    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        control_run_lease_required=True,
        worker_id="worker-a",
        workspace_root=str(tmp_path / "workspaces"),
        worker_ack_progress_interval_seconds=0.001,
    )
    worker = NATSDeepAgentsWorker(
        settings,
        run_job_func=lambda *_args, **_kwargs: asyncio.sleep(0, result="ok"),
        run_lease_func=acquire_lease,
        release_run_lease_func=release_lease,
    )
    message = FakeNATSMessage(
        b'{"run_id":"run-required-lease","thread_id":"thread-1","user_id":"user-1","goal":"resume"}'
    )

    asyncio.run(worker._process_message(message, CapturingJetStream()))

    assert calls == 7
    assert message.in_progress_calls > 0
    assert message.acked == 1
    assert message.naked == 0


def test_worker_rejects_nonretryable_required_lease_failure_without_compute(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        NATSDeepAgentsWorker,
        "_acquire_run_lease_for_delivery",
        _REAL_ACQUIRE_RUN_LEASE_FOR_DELIVERY,
    )
    lease_calls = 0
    compute_calls = 0

    async def acquire_lease(_run_id, _settings):
        nonlocal lease_calls
        lease_calls += 1
        raise RunLeaseUnavailable("HTTP 401", retryable=False)

    async def run_job(*_args, **_kwargs):
        nonlocal compute_calls
        compute_calls += 1
        return "unsafe"

    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        control_run_lease_required=True,
        worker_id="worker-a",
        workspace_root=str(tmp_path / "workspaces"),
    )
    worker = NATSDeepAgentsWorker(
        settings,
        run_job_func=run_job,
        run_lease_func=acquire_lease,
    )
    message = FakeNATSMessage(
        b'{"run_id":"run-lease-auth","thread_id":"thread-1","user_id":"user-1","goal":"resume"}'
    )
    js = CapturingJetStream()

    asyncio.run(worker._process_message(message, js))

    assert lease_calls == 1
    assert compute_calls == 0
    assert message.acked == 0
    assert message.naked == 1
    assert _published_events(js) == []


def test_worker_requires_control_url_when_run_leases_are_required(tmp_path: Path):
    required_settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        control_base_url="",
        control_run_lease_required=True,
        workspace_root=str(tmp_path / "required-workspaces"),
    )

    with pytest.raises(ValueError, match="control_base_url"):
        NATSDeepAgentsWorker(required_settings)

    local_settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        control_base_url="",
        control_run_lease_required=False,
        workspace_root=str(tmp_path / "local-workspaces"),
    )
    assert NATSDeepAgentsWorker(local_settings).settings.control_base_url == ""


@pytest.mark.parametrize(
    "lease",
    [
        ControlPlaneRunLease("wrong-run", "worker-a", "token"),
        ControlPlaneRunLease("run-malformed-lease", "wrong-worker", "token"),
        ControlPlaneRunLease("run-malformed-lease", "worker-a", ""),
        ControlPlaneRunLease("run-malformed-lease", "worker-a", 123),
    ],
)
def test_worker_rejects_malformed_or_mismatched_lease_authority(
    monkeypatch,
    tmp_path: Path,
    lease,
):
    monkeypatch.setattr(
        NATSDeepAgentsWorker,
        "_acquire_run_lease_for_delivery",
        _REAL_ACQUIRE_RUN_LEASE_FOR_DELIVERY,
    )

    async def acquire_lease(_run_id, _settings):
        return lease

    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        worker_id="worker-a",
        workspace_root=str(tmp_path / "workspaces"),
    )
    worker = NATSDeepAgentsWorker(settings, run_lease_func=acquire_lease)
    message = FakeNATSMessage(
        b'{"run_id":"run-malformed-lease","thread_id":"thread-1","user_id":"user-1","goal":"resume"}'
    )
    js = CapturingJetStream()

    asyncio.run(worker._process_message(message, js))

    assert message.acked == 0
    assert message.naked == 1
    assert _published_events(js) == []


def test_worker_releases_late_lease_despite_repeated_cancellation(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        NATSDeepAgentsWorker,
        "_acquire_run_lease_for_delivery",
        _REAL_ACQUIRE_RUN_LEASE_FOR_DELIVERY,
    )
    acquisition_started = threading.Event()
    grant_lease = threading.Event()
    release_started = threading.Event()
    finish_release = threading.Event()
    lease_requests: list[tuple[str, str | None]] = []
    deleted: list[str] = []
    compute_calls = 0

    def request_lease(_url, method, payload, _settings):
        lease_requests.append((method, payload.get("lease_token")))
        if method == "DELETE":
            release_started.set()
            assert finish_release.wait(timeout=1.0)
            return None
        acquisition_started.set()
        assert grant_lease.wait(timeout=1.0)
        return ControlPlaneRunLease("run-late-lease", "worker-a", "late-token")

    class CleanupTrackingWorker(NATSDeepAgentsWorker):
        async def _delete_checkpointer_thread(self, run_id):
            deleted.append(run_id)

    async def run_job(*_args, **_kwargs):
        nonlocal compute_calls
        compute_calls += 1
        return "unsafe"

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            control_base_url="http://control.test",
            worker_id="worker-a",
            workspace_root=str(tmp_path / "workspaces"),
        )
        worker = CleanupTrackingWorker(
            settings,
            run_job_func=run_job,
        )
        message = FakeNATSMessage(
            b'{"run_id":"run-late-lease","thread_id":"thread-1","user_id":"user-1","goal":"resume"}'
        )
        js = CapturingJetStream()
        task = asyncio.create_task(worker._process_message(message, js))
        async with asyncio.timeout(1.0):
            while not acquisition_started.is_set():
                await asyncio.sleep(0)
        worker._shutting_down = True
        task.cancel()
        await asyncio.sleep(0)
        # Repeated shutdown/cancel signals while the retained POST is still in
        # flight must not cancel its cleanup waiter or leak the returned lease.
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
        grant_lease.set()
        async with asyncio.timeout(1.0):
            while not release_started.is_set():
                await asyncio.sleep(0)
        # The compensating DELETE is also a to_thread call and needs the same
        # protection from repeated cancellation.
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
        finish_release.set()
        await asyncio.wait_for(task, timeout=1.0)
        return worker, message, js

    monkeypatch.setattr(
        nats_worker_module,
        "_request_control_plane_run_lease",
        request_lease,
    )
    worker, message, js = asyncio.run(scenario())

    assert compute_calls == 0
    assert _published_events(js) == []
    assert lease_requests == [("POST", None), ("DELETE", "late-token")]
    assert deleted == []
    assert message.acked == 0
    assert message.naked == 1
    assert message.nak_delays == [nats_worker_module._SHUTDOWN_NAK_DELAY_SECONDS]
    lock = try_acquire_run_lock(worker.settings, "run-late-lease")
    assert lock is not None
    lock.release()


def test_worker_naks_when_control_plane_run_lease_is_held(tmp_path: Path):
    calls = 0

    async def run_job_should_not_start(*_args, **_kwargs):
        nonlocal calls
        calls += 1

    async def acquire_lease(_run_id, _settings):
        raise RunLeaseConflict("run already leased")

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            worker_id="worker-a",
            workspace_root=str(tmp_path / "workspaces"),
            worker_ack_progress_interval_seconds=30.0,
        )
        worker = NATSDeepAgentsWorker(
            settings,
            run_job_func=run_job_should_not_start,
            run_lease_func=acquire_lease,
        )
        message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"lease"}'
        )
        await worker._process_message(message, CapturingJetStream())
        return message

    message = asyncio.run(scenario())

    assert calls == 0
    assert message.acked == 0
    assert message.naked == 1
    assert message.nak_delays == [30.0]


def test_worker_naks_and_stops_compute_when_control_plane_run_lease_is_lost(tmp_path: Path):
    calls: list[str] = []
    compute_cancelled = asyncio.Event()

    async def long_running_run_job(*_args, **_kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            compute_cancelled.set()
            raise

    async def acquire_lease(run_id, settings):
        calls.append(f"acquire:{run_id}")
        return ControlPlaneRunLease(
            run_id=run_id,
            worker_id="worker-a",
            lease_token="lease-token-1",
        )

    def renew_lost_lease_sync(lease, settings):
        # Called from the keepalive thread; a 409 means the run was handed to
        # another worker and this worker must stop.
        calls.append(f"renew:{lease.lease_token}")
        raise RunLeaseConflict("lease moved to another worker")

    async def release_lease(lease, settings):
        calls.append(f"release:{lease.lease_token}")

    async def run_status(_run_id, _settings):
        return None

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            worker_id="worker-a",
            workspace_root=str(tmp_path / "workspaces"),
            worker_ack_wait_seconds=1.0,
            worker_ack_progress_interval_seconds=0.01,
        )
        worker = NATSDeepAgentsWorker(
            settings,
            run_job_func=long_running_run_job,
            run_status_func=run_status,
            run_lease_func=acquire_lease,
            lease_renew_sync_func=renew_lost_lease_sync,
            release_run_lease_func=release_lease,
        )
        message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"lease"}'
        )
        js = CapturingJetStream()
        await asyncio.wait_for(worker._process_message(message, js), timeout=2.0)
        return message, _published_events(js)

    message, events = asyncio.run(scenario())

    assert compute_cancelled.is_set()
    assert calls == ["acquire:run-1", "renew:lease-token-1", "release:lease-token-1"]
    assert message.acked == 0
    assert message.naked == 1
    assert message.nak_delays == [0.01]
    # Lease lost -> NAK so another worker picks the run up; this worker must NOT
    # publish any terminal event. Best-effort progress heartbeats may have fired
    # before the loop-independent keepalive observed the 409, which is fine.
    terminal_kinds = {"run.completed", "run.failed", "run.canceled", "run.skipped"}
    assert terminal_kinds.isdisjoint(event["event_kind"] for event in events)


def test_worker_keeps_computing_through_transient_lease_renewal_outage(tmp_path: Path):
    # A control-plane replica restart must not abort hours of compute: the
    # stored lease stays valid for its TTL, so failed renewals are retried
    # until the validity window closes instead of cancelling the job.
    renewal_attempts = 0
    renewals_seen = asyncio.Event()

    async def long_running_run_job(*_args, **_kwargs):
        await asyncio.wait_for(renewals_seen.wait(), timeout=2.0)
        return "completed despite renewal outage"

    async def acquire_lease(run_id, settings):
        return ControlPlaneRunLease(
            run_id=run_id,
            worker_id="worker-a",
            lease_token="lease-token-1",
        )

    def renew_unreachable_lease_sync(lease, settings):
        nonlocal renewal_attempts
        renewal_attempts += 1
        if renewal_attempts >= 3:
            renewals_seen.set()
        # Transient outage (control-plane replica restart): NOT a 409, so the
        # keepalive keeps the run alive on the still-valid lease and retries.
        raise RunLeaseUnavailable("control plane connection refused")

    async def release_lease(lease, settings):
        return None

    async def run_status(_run_id, _settings):
        return None

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            control_base_url="http://control.test",
            worker_id="worker-a",
            workspace_root=str(tmp_path / "workspaces"),
            worker_ack_wait_seconds=1.0,
            worker_ack_progress_interval_seconds=0.01,
            control_run_lease_ttl_seconds=600.0,
        )
        worker = NATSDeepAgentsWorker(
            settings,
            run_job_func=long_running_run_job,
            run_status_func=run_status,
            run_lease_func=acquire_lease,
            lease_renew_sync_func=renew_unreachable_lease_sync,
            release_run_lease_func=release_lease,
        )
        message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"lease"}'
        )
        js = CapturingJetStream()
        await asyncio.wait_for(worker._process_message(message, js), timeout=3.0)
        return message, _published_events(js)

    message, events = asyncio.run(scenario())

    assert renewal_attempts >= 3
    assert message.acked == 1
    assert message.naked == 0
    assert "run.completed" in [event["event_kind"] for event in events]


def _keepalive_settings() -> RuntimeSettings:
    return RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
    )


def test_lease_keepalive_renews_rotates_token_and_posts_worker_heartbeat():
    # The keepalive is the sole renewer: it must follow a rotating lease token
    # forward AND post the worker heartbeat each tick (the heartbeat is what
    # keeps the run out of the "stale worker heartbeat" recovery path).
    def scenario():
        async def body():
            loop = asyncio.get_running_loop()
            seen: list[str] = []
            heartbeats: list[str | None] = []
            rotated_twice = asyncio.Event()

            def renew_rotates(lease, _settings):
                index = len(seen)
                seen.append(lease.lease_token)
                if index + 1 >= 2:
                    loop.call_soon_threadsafe(rotated_twice.set)
                return ControlPlaneRunLease(
                    run_id=lease.run_id,
                    worker_id=lease.worker_id,
                    lease_token=f"t{index + 1}",
                )

            def record_heartbeat(_settings, *, status, current_run_id, metadata=None):
                heartbeats.append((status, current_run_id))

            keepalive = nats_worker_module._LeaseKeepalive(
                ControlPlaneRunLease(run_id="run-x", worker_id="w", lease_token="t0"),
                _keepalive_settings(),
                loop=loop,
                interval_seconds=0.02,
                on_lost=lambda: None,
                renew_func=renew_rotates,
                heartbeat_func=record_heartbeat,
            )
            keepalive.start()
            await asyncio.wait_for(rotated_twice.wait(), timeout=2.0)
            await asyncio.to_thread(keepalive.stop)
            return seen, heartbeats, keepalive.current_lease(), keepalive.lost

        return asyncio.run(body())

    seen, heartbeats, current, lost = scenario()
    assert lost is False
    assert seen[:2] == ["t0", "t1"]  # first renewal used the seed token, then followed the rotation
    assert current.lease_token != "t0"
    assert heartbeats  # posted the worker heartbeat alongside renewal
    assert heartbeats[0] == ("busy", "run-x")


def test_lease_keepalive_aborts_the_run_on_lease_conflict():
    # A 409 means the control plane handed the run to another worker: the
    # keepalive must fire on_lost exactly once and stop renewing.
    def scenario():
        async def body():
            loop = asyncio.get_running_loop()
            lost_signal = asyncio.Event()
            renew_calls = 0
            heartbeats = 0

            def renew_conflict(_lease, _settings):
                nonlocal renew_calls
                renew_calls += 1
                raise RunLeaseConflict("handed to another worker")

            def count_heartbeat(_settings, *, status, current_run_id, metadata=None):
                nonlocal heartbeats
                heartbeats += 1

            keepalive = nats_worker_module._LeaseKeepalive(
                ControlPlaneRunLease(run_id="run-x", worker_id="w", lease_token="t0"),
                _keepalive_settings(),
                loop=loop,
                interval_seconds=0.02,
                on_lost=lost_signal.set,
                renew_func=renew_conflict,
                heartbeat_func=count_heartbeat,
            )
            keepalive.start()
            await asyncio.wait_for(lost_signal.wait(), timeout=2.0)
            # Give the thread a beat to prove it does not keep renewing.
            await asyncio.sleep(0.1)
            await asyncio.to_thread(keepalive.stop)
            return renew_calls, heartbeats, keepalive.lost

        return asyncio.run(body())

    renew_calls, heartbeats, lost = scenario()
    assert lost is True
    assert renew_calls == 1  # stopped after the conflict, no further renewals
    assert heartbeats == 0  # the conflict aborts before the heartbeat post


def test_lease_keepalive_survives_transient_renewal_failures():
    # A control-plane replica restart (not a 409) must not abort the run: the
    # lease is still valid for its TTL, so the keepalive retries and keeps the
    # last-good token.
    def scenario():
        async def body():
            loop = asyncio.get_running_loop()
            enough = asyncio.Event()
            attempts = 0
            heartbeats = 0
            lost_called = False

            def renew_flaky(_lease, _settings):
                nonlocal attempts
                attempts += 1
                if attempts >= 3:
                    loop.call_soon_threadsafe(enough.set)
                raise RunLeaseUnavailable("control plane connection refused")

            def count_heartbeat(_settings, *, status, current_run_id, metadata=None):
                nonlocal heartbeats
                heartbeats += 1

            def on_lost():
                nonlocal lost_called
                lost_called = True

            keepalive = nats_worker_module._LeaseKeepalive(
                ControlPlaneRunLease(run_id="run-x", worker_id="w", lease_token="t0"),
                _keepalive_settings(),
                loop=loop,
                interval_seconds=0.02,
                on_lost=on_lost,
                renew_func=renew_flaky,
                heartbeat_func=count_heartbeat,
            )
            keepalive.start()
            await asyncio.wait_for(enough.wait(), timeout=2.0)
            current = keepalive.current_lease()
            await asyncio.to_thread(keepalive.stop)
            return attempts, heartbeats, current, keepalive.lost, lost_called

        return asyncio.run(body())

    attempts, heartbeats, current, lost, lost_called = scenario()
    assert attempts >= 3
    assert lost is False
    assert lost_called is False
    assert current.lease_token == "t0"  # kept the last-good token through the outage
    # The worker heartbeat is independent of lease renewal: it keeps posting even
    # while renewals transiently fail, so the run stays out of the stale-heartbeat
    # recovery path.
    assert heartbeats >= 1


def test_lease_keepalive_interval_stays_inside_the_ttl():
    # Production defaults renew every 60s against a 600s TTL (ample headroom);
    # the interval never exceeds the configured progress cadence.
    default_interval = nats_worker_module.lease_keepalive_interval(
        RuntimeSettings(openai_base_url="http://example.test/v1", openai_model="deepseek_v4")
    )
    assert default_interval == 60.0
    fast = nats_worker_module.lease_keepalive_interval(
        RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            worker_ack_progress_interval_seconds=0.01,
        )
    )
    assert fast == pytest.approx(0.1)


def test_worker_publishes_run_heartbeat_during_silent_long_running_compute():
    async def scenario():
        js = HeartbeatCapturingJetStream()

        async def long_running_run_job(*_args, **kwargs):
            kwargs["on_event_emission_ready"]()
            await asyncio.wait_for(js.heartbeat_seen.wait(), timeout=1.0)
            return "finished after heartbeat"

        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            worker_ack_wait_seconds=1.0,
            worker_ack_progress_interval_seconds=0.01,
        )
        worker = NATSDeepAgentsWorker(settings, run_job_func=long_running_run_job)
        message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"silent compute"}'
        )
        await worker._process_message(message, js)
        return message, _published_events(js)

    message, events = asyncio.run(scenario())

    assert message.acked == 1
    assert message.naked == 0
    assert events[0]["event_kind"] == "run.heartbeat"
    assert events[0]["event_id"] == "evt_run-1_worker_heartbeat_000001"
    assert events[-1]["event_kind"] == "run.completed"
    assert events[-1]["payload"]["response_text"] == "finished after heartbeat"
    sequences = [event["sequence"] for event in events]
    assert sequences == sorted(sequences)
    assert len(sequences) == len(set(sequences))


def test_resumed_worker_heartbeat_id_uses_the_seeded_sequence_floor():
    class ResumedWorker(NATSDeepAgentsWorker):
        async def _run_events_snapshot(self, run_id: str):
            assert run_id == "run-resumed"
            return 42, None

    async def scenario():
        js = HeartbeatCapturingJetStream()

        async def long_running_run_job(*_args, **kwargs):
            kwargs["on_event_emission_ready"]()
            await asyncio.wait_for(js.heartbeat_seen.wait(), timeout=1.0)
            return "finished after resumed heartbeat"

        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            worker_ack_wait_seconds=1.0,
            worker_ack_progress_interval_seconds=0.01,
        )
        worker = ResumedWorker(settings, run_job_func=long_running_run_job)
        message = FakeNATSMessage(
            b'{"run_id":"run-resumed","thread_id":"thread-1","user_id":"user-1","goal":"resume"}'
        )
        await worker._process_message(message, js)
        return message, _published_events(js), _published_headers(js)

    message, events, headers = asyncio.run(scenario())

    assert message.acked == 1
    assert events[0]["event_kind"] == "run.heartbeat"
    assert events[0]["sequence"] == 43
    assert events[0]["event_id"] == "evt_run-resumed_worker_heartbeat_000043"
    assert headers[0]["Nats-Msg-Id"] == "event:evt_run-resumed_worker_heartbeat_000043"


def test_worker_posts_control_plane_heartbeats_around_compute(tmp_path: Path):
    heartbeat_calls: list[dict[str, object]] = []

    async def run_job_returns(*_args, **_kwargs):
        assert heartbeat_calls[-1]["status"] == "busy"
        assert heartbeat_calls[-1]["current_run_id"] == "run-1"
        return "ok"

    async def post_worker_heartbeat(settings, *, status, current_run_id=None, metadata=None):
        heartbeat_calls.append(
            {
                "worker_id": settings.worker_id,
                "status": status,
                "current_run_id": current_run_id,
                "metadata": metadata or {},
            }
        )

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            worker_id="worker-test-1",
            worker_ack_progress_interval_seconds=30.0,
        )
        worker = NATSDeepAgentsWorker(
            settings,
            run_job_func=run_job_returns,
            worker_heartbeat_func=post_worker_heartbeat,
        )
        message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"heartbeat"}'
        )
        js = CapturingJetStream()
        await worker._process_message(message, js)
        return message

    message = asyncio.run(scenario())

    assert message.acked == 1
    assert [call["status"] for call in heartbeat_calls] == ["busy", "idle"]
    assert heartbeat_calls[0]["current_run_id"] == "run-1"
    assert heartbeat_calls[0]["metadata"] == {"active_tasks": 1, "max_concurrency": 64}
    assert heartbeat_calls[1]["current_run_id"] is None
    assert heartbeat_calls[1]["metadata"] == {"active_tasks": 0, "max_concurrency": 64}


def test_control_plane_worker_heartbeat_posts_json(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"worker_id":"worker-test-1","status":"busy"}'

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(nats_worker_module.urllib_request, "urlopen", fake_urlopen)
    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        control_status_timeout_seconds=3.5,
        worker_id="worker-test-1",
        worker_kind="deepagents",
    )

    asyncio.run(
        post_control_plane_worker_heartbeat(
            settings,
            status="busy",
            current_run_id="run-1",
            metadata={"active_tasks": 1},
        )
    )

    assert captured["url"] == "http://control.test/v2/workers/heartbeat"
    assert captured["method"] == "POST"
    assert captured["timeout"] == 3.5
    payload = captured["payload"]
    assert payload["worker_id"] == "worker-test-1"
    assert payload["worker_kind"] == "deepagents"
    assert payload["status"] == "busy"
    assert payload["current_run_id"] == "run-1"
    assert payload["metadata"] == {"active_tasks": 1}


def test_worker_naks_job_when_fallback_failed_event_publish_fails():
    async def failing_run_job(*_args, **_kwargs):
        raise RuntimeError("agent failed")

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            worker_ack_progress_interval_seconds=30.0,
        )
        worker = NATSDeepAgentsWorker(settings, run_job_func=failing_run_job)
        message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"fail"}'
        )
        await worker._process_message(message, FailingJetStream())
        return message

    message = asyncio.run(scenario())

    assert message.acked == 0
    assert message.naked == 1


def test_worker_naks_job_when_event_publish_fails():
    async def run_job_that_publishes(*_args, **kwargs):
        await kwargs["publish_event"](
            {
                "event_id": "evt-run-1-started",
                "run_id": "run-1",
                "thread_id": "thread-1",
                "event_kind": "run.started",
                "payload": {},
            }
        )
        return "ok"

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            worker_ack_progress_interval_seconds=30.0,
        )
        worker = NATSDeepAgentsWorker(settings, run_job_func=run_job_that_publishes)
        message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"publish"}'
        )
        await worker._process_message(message, FailingJetStream())
        return message

    message = asyncio.run(scenario())

    assert message.acked == 0
    assert message.naked == 1


def test_worker_naks_duplicate_delivery_for_active_run_without_starting_second_compute():
    async def scenario():
        calls = 0
        first_started = asyncio.Event()
        release_first = asyncio.Event()

        async def blocking_run_job(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                await release_first.wait()
            return "ok"

        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            worker_ack_progress_interval_seconds=30.0,
        )
        worker = NATSDeepAgentsWorker(settings, run_job_func=blocking_run_job)
        first_message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"long"}'
        )
        duplicate_message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"long"}'
        )
        first_task = asyncio.create_task(
            worker._process_message(first_message, CapturingJetStream())
        )
        await asyncio.wait_for(first_started.wait(), timeout=1.0)

        await worker._process_message(duplicate_message, CapturingJetStream())
        release_first.set()
        await asyncio.wait_for(first_task, timeout=1.0)

        return calls, first_message, duplicate_message

    calls, first_message, duplicate_message = asyncio.run(scenario())

    assert calls == 1
    assert first_message.acked == 1
    assert first_message.naked == 0
    assert duplicate_message.acked == 0
    assert duplicate_message.naked == 1
    assert duplicate_message.nak_delays == [30.0]


def test_worker_naks_cross_worker_duplicate_delivery_for_locked_run(tmp_path: Path):
    async def scenario():
        calls = 0
        first_started = asyncio.Event()
        release_first = asyncio.Event()

        async def blocking_run_job(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                await release_first.wait()
            return "ok"

        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            worker_ack_progress_interval_seconds=30.0,
        )
        first_worker = NATSDeepAgentsWorker(settings, run_job_func=blocking_run_job)
        second_worker = NATSDeepAgentsWorker(settings, run_job_func=blocking_run_job)
        first_message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"long"}'
        )
        duplicate_message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"long"}'
        )

        first_task = asyncio.create_task(
            first_worker._process_message(first_message, CapturingJetStream())
        )
        await asyncio.wait_for(first_started.wait(), timeout=1.0)

        await second_worker._process_message(duplicate_message, CapturingJetStream())
        release_first.set()
        await asyncio.wait_for(first_task, timeout=1.0)

        return calls, first_message, duplicate_message

    calls, first_message, duplicate_message = asyncio.run(scenario())

    assert calls == 1
    assert first_message.acked == 1
    assert first_message.naked == 0
    assert duplicate_message.acked == 0
    assert duplicate_message.naked == 1
    assert duplicate_message.nak_delays == [30.0]


def test_worker_fetches_new_jobs_while_previous_job_is_still_running(monkeypatch):
    async def scenario():
        first_started = asyncio.Event()
        second_started = asyncio.Event()
        release_first = asyncio.Event()

        first_message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"long"}'
        )
        second_message = FakeNATSMessage(
            b'{"run_id":"run-2","thread_id":"thread-2","user_id":"user-1","goal":"short"}'
        )

        class FakeSubscription:
            def __init__(self):
                self.fetch_calls = 0

            async def fetch(self, *_args, **_kwargs):
                self.fetch_calls += 1
                if self.fetch_calls == 1:
                    return [first_message]
                if self.fetch_calls == 2:
                    return [second_message]
                await asyncio.sleep(0.01)
                return []

        class FakeConnection:
            def __init__(self):
                self.subscription = FakeSubscription()

            def jetstream(self):
                return CapturingJetStream()

            async def subscribe(self, *_args, **_kwargs):
                return None

            async def drain(self):
                pass

        class ConcurrentWorker(NATSDeepAgentsWorker):
            def __init__(self, settings):
                super().__init__(settings)
                self.connection = FakeConnection()

            async def _ensure_stream(self, _js):
                pass

            async def _reap_sandbox_containers_once(self):
                pass

            async def _subscribe(self, _js):
                return self.connection.subscription

            async def _process_message(self, message, _js):
                payload = json.loads(message.data.decode("utf-8"))
                if payload["run_id"] == "run-1":
                    first_started.set()
                    await release_first.wait()
                if payload["run_id"] == "run-2":
                    second_started.set()

        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            worker_max_concurrency=2,
            worker_ack_progress_interval_seconds=0,
        )
        worker = ConcurrentWorker(settings)

        async def fake_connect(_url):
            return worker.connection

        monkeypatch.setattr(nats_worker_module.nats, "connect", fake_connect)

        worker_task = asyncio.create_task(worker.run_forever())
        try:
            await asyncio.wait_for(first_started.wait(), timeout=1.0)
            await asyncio.wait_for(second_started.wait(), timeout=0.25)
        finally:
            release_first.set()
            worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker_task

    asyncio.run(scenario())


def test_worker_reconnects_after_recoverable_nats_error(monkeypatch):
    async def scenario():
        sleeps: list[float] = []

        class ReconnectingWorker(NATSDeepAgentsWorker):
            def __init__(self, settings):
                super().__init__(settings)
                self.serve_calls = 0

            async def _serve_one_connection(self):
                self.serve_calls += 1
                if self.serve_calls == 1:
                    raise nats.errors.ConnectionClosedError()

        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
        )
        worker = ReconnectingWorker(settings)

        async def fake_sleep(delay):
            sleeps.append(delay)

        monkeypatch.setattr(nats_worker_module.asyncio, "sleep", fake_sleep)

        await asyncio.wait_for(worker.run_forever(), timeout=1.0)

        return worker.serve_calls, sleeps

    serve_calls, sleeps = asyncio.run(scenario())

    assert serve_calls == 2
    assert sleeps == [1.0]


def test_worker_shutdown_naks_active_job_without_marking_user_canceled(monkeypatch):
    async def scenario():
        started = asyncio.Event()
        message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"long"}'
        )
        js = CapturingJetStream()

        class FakeSubscription:
            def __init__(self):
                self.sent = False

            async def fetch(self, *_args, **_kwargs):
                if not self.sent:
                    self.sent = True
                    return [message]
                await asyncio.sleep(0.05)
                return []

        class FakeConnection:
            def __init__(self):
                self.subscription = FakeSubscription()

            def jetstream(self):
                return js

            async def subscribe(self, *_args, **_kwargs):
                return None

            async def drain(self):
                pass

        async def blocking_run_job(*_args, **_kwargs):
            started.set()
            await asyncio.Event().wait()

        class ShutdownWorker(NATSDeepAgentsWorker):
            def __init__(self, settings):
                super().__init__(settings, run_job_func=blocking_run_job)
                self.connection = FakeConnection()

            async def _ensure_stream(self, _js):
                pass

            async def _reap_sandbox_containers_once(self):
                pass

            async def _subscribe(self, _js):
                return self.connection.subscription

        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            worker_ack_progress_interval_seconds=0,
        )
        worker = ShutdownWorker(settings)

        async def fake_connect(_url):
            return worker.connection

        monkeypatch.setattr(nats_worker_module.nats, "connect", fake_connect)

        worker_task = asyncio.create_task(worker.run_forever())
        await asyncio.wait_for(started.wait(), timeout=1.0)
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task
        return message, _published_events(js)

    message, events = asyncio.run(scenario())

    assert message.acked == 0
    assert message.naked == 1
    # The shutdown NAK must delay redelivery beyond the dying worker's last
    # outstanding pull request; an immediate NAK redelivers into this worker's
    # own doomed buffer and wedges the run until AckWait expires.
    assert message.nak_delays == [nats_worker_module._SHUTDOWN_NAK_DELAY_SECONDS]
    assert [event["event_kind"] for event in events] == []


@pytest.mark.parametrize("window", ["busy_heartbeat", "snapshot"])
def test_worker_shutdown_during_authority_startup_retains_checkpoint_and_redelivers(
    monkeypatch,
    tmp_path: Path,
    window,
):
    monkeypatch.setattr(
        NATSDeepAgentsWorker,
        "_acquire_run_lease_for_delivery",
        _REAL_ACQUIRE_RUN_LEASE_FOR_DELIVERY,
    )
    entered = asyncio.Event()
    released: list[str] = []
    deleted: list[str] = []
    compute_calls = 0

    async def acquire_lease(run_id, _settings):
        return ControlPlaneRunLease(run_id, "worker-a", "current-token")

    async def release_lease(lease, _settings):
        released.append(lease.lease_token)

    async def run_status(_run_id, _settings):
        return None

    class WindowWorker(NATSDeepAgentsWorker):
        async def _post_worker_heartbeat(
            self,
            status,
            *,
            current_run_id=None,
            metadata=None,
        ):
            if window == "busy_heartbeat" and status == "busy":
                entered.set()
                await asyncio.Event().wait()

        async def _run_events_snapshot(self, _run_id):
            if window == "snapshot":
                entered.set()
                await asyncio.Event().wait()
            return 0, None

        async def _delete_checkpointer_thread(self, run_id):
            deleted.append(run_id)

    async def run_job(*_args, **_kwargs):
        nonlocal compute_calls
        compute_calls += 1
        return "unsafe"

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            control_base_url="http://control.test",
            worker_id="worker-a",
            workspace_root=str(tmp_path / "workspaces"),
            worker_ack_progress_interval_seconds=0,
        )
        worker = WindowWorker(
            settings,
            run_job_func=run_job,
            run_status_func=run_status,
            run_lease_func=acquire_lease,
            release_run_lease_func=release_lease,
        )
        message = FakeNATSMessage(
            b'{"run_id":"run-startup-cancel","thread_id":"thread-1","user_id":"user-1","goal":"resume"}'
        )
        js = CapturingJetStream()
        task = asyncio.create_task(worker._process_message(message, js))
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        worker._shutting_down = True
        task.cancel()
        await asyncio.wait_for(task, timeout=1.0)
        return worker, message, js

    worker, message, js = asyncio.run(scenario())

    assert compute_calls == 0
    assert _published_events(js) == []
    assert message.acked == 0
    assert message.naked == 1
    assert message.nak_delays == [nats_worker_module._SHUTDOWN_NAK_DELAY_SECONDS]
    assert deleted == []
    assert released == ["current-token"]
    assert worker._active_tasks == {}
    lock = try_acquire_run_lock(worker.settings, "run-startup-cancel")
    assert lock is not None
    lock.release()


def test_worker_registers_async_cancel_subscription_callback(monkeypatch):
    async def scenario():
        callback_was_coroutine = asyncio.Event()

        class EmptySubscription:
            async def fetch(self, *_args, **_kwargs):
                await asyncio.sleep(0.05)
                return []

        class FakeConnection:
            def jetstream(self):
                return CapturingJetStream()

            async def subscribe(self, *_args, **kwargs):
                callback = kwargs.get("cb")
                assert inspect.iscoroutinefunction(callback)
                callback_was_coroutine.set()
                return None

            async def drain(self):
                pass

        class SubscribeWorker(NATSDeepAgentsWorker):
            async def _ensure_stream(self, _js):
                pass

            async def _reap_sandbox_containers_once(self):
                pass

            async def _subscribe(self, _js):
                return EmptySubscription()

        async def fake_connect(_url):
            return FakeConnection()

        monkeypatch.setattr(nats_worker_module.nats, "connect", fake_connect)
        worker = SubscribeWorker(
            RuntimeSettings(openai_base_url="http://example.test/v1", openai_model="deepseek_v4")
        )

        worker_task = asyncio.create_task(worker.run_forever())
        try:
            await asyncio.wait_for(callback_was_coroutine.wait(), timeout=1.0)
        finally:
            worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker_task

    asyncio.run(scenario())


def test_worker_cancel_signal_cancels_active_run_and_publishes_canceled():
    started = asyncio.Event()

    async def blocking_run_job(*_args, **kwargs):
        await kwargs["publish_event"](
            {
                "event_id": "evt-run-1-started",
                "sequence": 3,
                "run_id": "run-1",
                "thread_id": "thread-1",
                "event_kind": "run.started",
                "payload": {},
            }
        )
        started.set()
        await asyncio.Event().wait()

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            worker_ack_progress_interval_seconds=0,
        )
        worker = NATSDeepAgentsWorker(settings, run_job_func=blocking_run_job)
        message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"wait"}'
        )
        js = CapturingJetStream()
        task = asyncio.create_task(worker._process_message(message, js))
        await asyncio.wait_for(started.wait(), timeout=1.0)
        await worker._handle_cancel_payload({"run_id": "run-1", "reason": "user stop"})
        await asyncio.wait_for(task, timeout=1.0)
        return message, _published_events(js)

    message, events = asyncio.run(scenario())

    assert message.acked == 1
    assert events[-1]["event_kind"] == "run.canceled"
    assert events[-1]["event_id"] == "evt_run-1_canceled"
    assert events[-1]["sequence"] == 4
    assert events[-1]["payload"]["reason"] == "user stop"


def test_worker_does_not_start_previously_canceled_run():
    calls = 0

    async def run_job_should_not_start(*_args, **_kwargs):
        nonlocal calls
        calls += 1

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            worker_ack_progress_interval_seconds=0,
        )
        worker = NATSDeepAgentsWorker(settings, run_job_func=run_job_should_not_start)
        js = CapturingJetStream()
        await worker._handle_cancel_payload({"run_id": "run-1", "reason": "pre-canceled"})
        message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"skip"}'
        )
        await worker._process_message(message, js)
        return message, _published_events(js)

    message, events = asyncio.run(scenario())

    assert calls == 0
    assert message.acked == 1
    assert events[-1]["event_kind"] == "run.canceled"
    assert events[-1]["event_id"] == "evt_run-1_canceled"


def test_worker_acks_terminal_control_plane_run_without_starting_compute():
    calls = 0
    deleted: list[str] = []

    async def run_job_should_not_start(*_args, **_kwargs):
        nonlocal calls
        calls += 1

    async def terminal_status(_run_id, _settings):
        return "canceled"

    class CleanupWorker(NATSDeepAgentsWorker):
        async def _delete_checkpointer_thread(self, run_id):
            deleted.append(run_id)

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            worker_ack_progress_interval_seconds=0,
        )
        worker = CleanupWorker(
            settings,
            run_job_func=run_job_should_not_start,
            run_status_func=terminal_status,
        )
        js = CapturingJetStream()
        message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"skip"}'
        )
        await worker._process_message(message, js)
        return message, _published_events(js)

    message, events = asyncio.run(scenario())

    assert calls == 0
    assert message.acked == 1
    assert message.naked == 0
    assert deleted == ["run-1"]
    assert len(events) == 1
    _assert_worker_skipped_event(
        events[0],
        run_id="run-1",
        thread_id="thread-1",
        control_status="canceled",
        stage="initial_status_check",
    )


def test_fresh_worker_initializes_checkpoint_store_after_terminal_ack_for_cleanup():
    calls: list[tuple[str, str]] = []

    async def terminal_status(_run_id, _settings):
        return "succeeded"

    async def run_job_should_not_start(*_args, **_kwargs):
        raise AssertionError("terminal redelivery must not restart compute")

    class CleanupCheckpointer:
        async def delete_thread(self, run_id: str) -> None:
            assert message.acked == 1, "checkpoint deletion must remain ACK-after-confirmation"
            calls.append(("delete", run_id))

    checkpointer = CleanupCheckpointer()

    class FreshWorker(NATSDeepAgentsWorker):
        async def _ensure_checkpointer(self):
            assert self._checkpointer is None
            calls.append(("initialize", "run-terminal-fresh"))
            self._checkpointer = checkpointer
            return checkpointer

    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        worker_ack_progress_interval_seconds=0,
    )
    worker = FreshWorker(
        settings,
        run_job_func=run_job_should_not_start,
        run_status_func=terminal_status,
    )
    message = FakeNATSMessage(
        b'{"run_id":"run-terminal-fresh","thread_id":"thread-1",'
        b'"user_id":"user-1","goal":"do not replay"}'
    )

    asyncio.run(worker._process_message(message, CapturingJetStream()))

    assert message.acked == 1
    assert message.naked == 0
    assert calls == [
        ("initialize", "run-terminal-fresh"),
        ("delete", "run-terminal-fresh"),
    ]


def test_worker_rechecks_control_plane_status_after_acquiring_run_lock(tmp_path: Path):
    calls = 0
    statuses = iter([None, "canceled"])

    async def run_job_should_not_start(*_args, **_kwargs):
        nonlocal calls
        calls += 1

    async def status_changes_after_first_check(_run_id, _settings):
        return next(statuses)

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            worker_ack_progress_interval_seconds=0,
        )
        worker = NATSDeepAgentsWorker(
            settings,
            run_job_func=run_job_should_not_start,
            run_status_func=status_changes_after_first_check,
        )
        js = CapturingJetStream()
        message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"skip"}'
        )
        await worker._process_message(message, js)
        return message, _published_events(js)

    message, events = asyncio.run(scenario())

    assert calls == 0
    assert message.acked == 1
    assert message.naked == 0
    assert len(events) == 1
    _assert_worker_skipped_event(
        events[0],
        run_id="run-1",
        thread_id="thread-1",
        control_status="canceled",
        stage="pre_compute_recheck",
    )


def test_worker_starts_compute_when_initial_control_status_lookup_fails(tmp_path: Path):
    calls = 0

    async def run_job_should_start(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return "ok"

    async def status_lookup_fails(_run_id, _settings):
        raise RuntimeError("control plane temporarily unavailable")

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            worker_ack_progress_interval_seconds=0,
        )
        worker = NATSDeepAgentsWorker(
            settings,
            run_job_func=run_job_should_start,
            run_status_func=status_lookup_fails,
        )
        js = CapturingJetStream()
        message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"keep going"}'
        )
        await worker._process_message(message, js)
        return message

    message = asyncio.run(scenario())

    assert calls == 1
    assert message.acked == 1
    assert message.naked == 0


def test_worker_starts_compute_when_rechecked_control_status_lookup_fails(tmp_path: Path):
    calls = 0
    status_calls = 0

    async def run_job_should_start(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return "ok"

    async def status_lookup_fails_on_recheck(_run_id, _settings):
        nonlocal status_calls
        status_calls += 1
        if status_calls == 1:
            return None
        raise RuntimeError("control plane temporarily unavailable")

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            worker_ack_progress_interval_seconds=0,
        )
        worker = NATSDeepAgentsWorker(
            settings,
            run_job_func=run_job_should_start,
            run_status_func=status_lookup_fails_on_recheck,
        )
        js = CapturingJetStream()
        message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"keep going"}'
        )
        await worker._process_message(message, js)
        return message

    message = asyncio.run(scenario())

    assert calls == 1
    assert status_calls == 2
    assert message.acked == 1
    assert message.naked == 0


def test_worker_status_monitor_cancels_active_run_when_control_plane_becomes_terminal(
    tmp_path: Path,
):
    started = asyncio.Event()
    status_calls = 0

    async def blocking_run_job(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    async def status_becomes_canceled(_run_id, _settings):
        nonlocal status_calls
        status_calls += 1
        if started.is_set() and status_calls >= 3:
            return "canceled"
        return None

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            worker_ack_progress_interval_seconds=0,
            control_status_poll_interval_seconds=0.01,
        )
        worker = NATSDeepAgentsWorker(
            settings,
            run_job_func=blocking_run_job,
            run_status_func=status_becomes_canceled,
        )
        js = CapturingJetStream()
        message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"long"}'
        )
        await worker._process_message(message, js)
        return message, _published_events(js)

    message, events = asyncio.run(scenario())

    assert started.is_set()
    assert message.acked == 1
    assert message.naked == 0
    assert events[-1]["event_kind"] == "run.canceled"
    assert events[-1]["event_id"] == "evt_run-1_canceled"


@pytest.mark.parametrize("terminal_status", ["failed", "succeeded", "not_found"])
def test_worker_status_monitor_stops_without_publishing_cancel_for_non_cancel_terminal_status(
    tmp_path: Path,
    terminal_status: str,
):
    started = asyncio.Event()
    status_calls = 0

    async def blocking_run_job(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    async def status_becomes_terminal(_run_id, _settings):
        nonlocal status_calls
        status_calls += 1
        if started.is_set() and status_calls >= 3:
            return terminal_status
        return None

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            worker_ack_progress_interval_seconds=0,
            control_status_poll_interval_seconds=0.01,
        )
        worker = NATSDeepAgentsWorker(
            settings,
            run_job_func=blocking_run_job,
            run_status_func=status_becomes_terminal,
        )
        js = CapturingJetStream()
        message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"long"}'
        )
        await worker._process_message(message, js)
        return message, _published_events(js)

    message, events = asyncio.run(scenario())

    assert started.is_set()
    assert message.acked == 1
    assert message.naked == 0
    assert len(events) == 1
    _assert_worker_skipped_event(
        events[0],
        run_id="run-1",
        thread_id="thread-1",
        control_status=terminal_status,
        stage="status_monitor",
    )


def test_control_plane_status_lookup_reads_v2_run_status(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"status":"succeeded"}'

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(nats_worker_module.urllib_request, "urlopen", fake_urlopen)

    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        control_status_timeout_seconds=0.5,
    )

    status = asyncio.run(fetch_control_plane_run_status("run with space", settings))

    assert status == "succeeded"
    assert captured == {
        "url": "http://control.test/v2/runs/run%20with%20space",
        "timeout": 0.5,
    }


def test_worker_acks_missing_control_plane_run_without_starting_compute():
    calls = 0

    async def run_job_should_not_start(*_args, **_kwargs):
        nonlocal calls
        calls += 1

    async def missing_status(_run_id, _settings):
        return "not_found"

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            worker_ack_progress_interval_seconds=0,
        )
        worker = NATSDeepAgentsWorker(
            settings,
            run_job_func=run_job_should_not_start,
            run_status_func=missing_status,
        )
        js = CapturingJetStream()
        message = FakeNATSMessage(
            b'{"run_id":"run-missing","thread_id":"thread-1","user_id":"user-1","goal":"stale"}'
        )
        await worker._process_message(message, js)
        return message, _published_events(js)

    message, events = asyncio.run(scenario())

    assert calls == 0
    assert message.acked == 1
    assert message.naked == 0
    assert len(events) == 1
    _assert_worker_skipped_event(
        events[0],
        run_id="run-missing",
        thread_id="thread-1",
        control_status="not_found",
        stage="initial_status_check",
    )


def test_control_plane_status_lookup_maps_authenticated_404_to_not_found(monkeypatch):
    def fake_urlopen(_request, timeout=None):
        _ = timeout
        raise urllib_error.HTTPError(
            url="http://control.test/v2/runs/run-missing",
            code=404,
            msg="not found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(nats_worker_module.urllib_request, "urlopen", fake_urlopen)

    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        control_worker_token="trace-worker-secret",
    )

    status = asyncio.run(fetch_control_plane_run_status("run-missing", settings))

    assert status == "not_found"


def test_control_plane_status_lookup_treats_anonymous_404_as_unknown(monkeypatch):
    # Without any worker identity a 404 is indistinguishable from an auth
    # failure (the control plane hides runs it cannot scope to the caller).
    # Treating it as authoritative would ack and silently drop the job.
    def fake_urlopen(_request, timeout=None):
        _ = timeout
        raise urllib_error.HTTPError(
            url="http://control.test/v2/runs/run-missing",
            code=404,
            msg="not found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(nats_worker_module.urllib_request, "urlopen", fake_urlopen)

    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
    )

    status = asyncio.run(fetch_control_plane_run_status("run-missing", settings))

    assert status is None


def test_control_plane_requests_attach_worker_token_and_job_user(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"status":"running"}'

    def fake_urlopen(request, timeout=None):
        _ = timeout
        captured["headers"] = {key.lower(): value for key, value in request.header_items()}
        return FakeResponse()

    monkeypatch.setattr(nats_worker_module.urllib_request, "urlopen", fake_urlopen)

    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        control_worker_token="trace-worker-secret",
    )

    async def scenario() -> str | None:
        token = nats_worker_module._control_plane_user_id.set("bisque:researcher")
        try:
            return await fetch_control_plane_run_status("run-1", settings)
        finally:
            nats_worker_module._control_plane_user_id.reset(token)

    status = asyncio.run(scenario())

    assert status == "running"
    assert captured["headers"]["x-ultra-worker-token"] == "trace-worker-secret"
    assert captured["headers"]["x-ultra-user-id"] == "bisque:researcher"


def test_control_plane_run_sequence_floor_uses_source_sequence_while_pagination_uses_store_sequence(
    monkeypatch,
):
    calls: list[dict[str, object]] = []

    class FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

    def fake_urlopen(request, timeout=None):
        call_index = len(calls)
        calls.append(
            {
                "url": request.full_url,
                "headers": {key.lower(): value for key, value in request.header_items()},
                "timeout": timeout,
            }
        )
        if call_index == 0:
            return FakeResponse(
                {
                    "events": [
                        {"sequence": sequence, "source_sequence": sequence}
                        for sequence in range(1, 501)
                    ]
                }
            )
        return FakeResponse(
            {
                "events": [
                    # Source sequences can be sparse and exceed the store
                    # cursor after gate bypass/replay. Producer authority is
                    # independent of reader pagination order.
                    {"sequence": 501, "source_sequence": 900},
                    # A later control-plane-authored event has no producer
                    # source sequence. It advances the reader pagination
                    # cursor, but must not make the resumed worker skip 502-513.
                    {"sequence": 513, "event_kind": "run.requeued"},
                ]
            }
        )

    monkeypatch.setattr(nats_worker_module.urllib_request, "urlopen", fake_urlopen)

    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        control_worker_token="trace-worker-secret",
        control_status_timeout_seconds=4.0,
    )

    async def scenario() -> int:
        token = nats_worker_module._control_plane_user_id.set("bisque:researcher")
        try:
            return await fetch_control_plane_run_max_sequence("run with space", settings)
        finally:
            nats_worker_module._control_plane_user_id.reset(token)

    max_sequence = asyncio.run(scenario())

    assert max_sequence == 900
    assert [call["url"] for call in calls] == [
        "http://control.test/v2/runs/run%20with%20space/events?limit=500&after_sequence=0",
        "http://control.test/v2/runs/run%20with%20space/events?limit=500&after_sequence=500",
    ]
    for call in calls:
        assert call["timeout"] == 4.0
        headers = call["headers"]
        assert headers["x-ultra-worker-token"] == "trace-worker-secret"
        assert headers["x-ultra-user-id"] == "bisque:researcher"


@pytest.mark.parametrize(
    "payload",
    [
        "not-an-object",
        {},
        {"events": "not-a-list"},
        {"events": [{"sequence": 1}, "not-an-event"]},
        {"events": [{"sequence": 0, "source_sequence": 9}]},
        {"events": [{"sequence": 1, "source_sequence": "9"}]},
        {"events": [{"sequence": 1, "source_sequence": True}]},
    ],
)
def test_control_plane_run_sequence_floor_rejects_malformed_authority_pages(
    monkeypatch,
    payload,
):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(
        nats_worker_module.urllib_request,
        "urlopen",
        lambda _request, timeout=None: FakeResponse(),
    )
    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
    )

    with pytest.raises(ValueError):
        asyncio.run(fetch_control_plane_run_max_sequence("run-1", settings))


def test_control_plane_run_usage_summary_dedupes_token_usage_events(monkeypatch):
    calls: list[str] = []

    class FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

    def fake_urlopen(request, timeout=None):
        _ = timeout
        calls.append(request.full_url)
        if len(calls) == 1:
            return FakeResponse(
                {
                    "events": [
                        {
                            "sequence": 1,
                            "event_kind": "run.token_usage",
                            "event_id": "evt-1",
                            "payload": {
                                "usage_event_id": "usage-1",
                                "input_tokens": 100,
                                "output_tokens": 20,
                                "total_tokens": 120,
                                "model": "deepseek_v4",
                            },
                        },
                        {
                            "sequence": 2,
                            "event_kind": "run.token_usage",
                            "event_id": "evt-duplicate",
                            "payload": {
                                "usage_event_id": "usage-1",
                                "input_tokens": 100,
                                "output_tokens": 20,
                                "total_tokens": 120,
                                "model": "deepseek_v4",
                            },
                        },
                    ]
                    + [
                        {"sequence": sequence, "event_kind": "trace.message.delta", "payload": {}}
                        for sequence in range(3, 501)
                    ]
                }
            )
        return FakeResponse(
            {
                "events": [
                    {
                        "sequence": 503,
                        "event_kind": "run.token_usage",
                        "event_id": "evt-2",
                        "payload": {
                            "usage_event_id": "usage-2",
                            "input_tokens": 50,
                            "output_tokens": 10,
                            "total_tokens": 60,
                            "model": "deepseek_v4",
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr(nats_worker_module.urllib_request, "urlopen", fake_urlopen)

    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        control_worker_token="trace-worker-secret",
    )

    async def scenario() -> dict[str, object] | None:
        token = nats_worker_module._control_plane_user_id.set("bisque:researcher")
        try:
            return await fetch_control_plane_run_usage_summary("run-1", settings)
        finally:
            nats_worker_module._control_plane_user_id.reset(token)

    usage = asyncio.run(scenario())

    assert usage == {
        "input_tokens": 150,
        "output_tokens": 30,
        "total_tokens": 180,
        "model": "deepseek_v4",
    }
    assert calls == [
        "http://control.test/v2/runs/run-1/events?limit=500&after_sequence=0",
        "http://control.test/v2/runs/run-1/events?limit=500&after_sequence=500",
    ]


def test_worker_acks_malformed_job_without_crashing_loop():
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            worker_ack_progress_interval_seconds=0,
        )
        worker = NATSDeepAgentsWorker(settings)
        message = FakeNATSMessage(b"not-json")
        await worker._process_message(message, CapturingJetStream())
        return message

    message = asyncio.run(scenario())

    assert message.acked == 1


# --- Rigor results-contract enforcement (Intelligence: Pro) -----------------


def _attempt(text: str):
    from ultra_deepagents.runner import AgentAttemptResult

    return AgentAttemptResult(
        final_response_text=text,
        streamed_response_text="",
        post_tool_streamed_response_text="",
    )


def _study_job(workflow_hint: dict | None = None) -> RunJobEnvelope:
    return RunJobEnvelope(
        run_id="run-rigor",
        thread_id="thread-rigor",
        user_id="researcher-1",
        goal=(
            "Simulate the Duffing oscillator and classify dynamical regimes from "
            "Lyapunov exponents and Poincare sections."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    "Simulate the Duffing oscillator and classify dynamical regimes from "
                    "Lyapunov exponents and Poincare sections."
                ),
            }
        ],
        workflow_hint=dict(workflow_hint or {}),
    )


def _context_for_job(job: RunJobEnvelope, tmp_path: Path):
    from ultra_deepagents.context import AgentRunContext

    return AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="org-1",
        user_id=job.user_id,
        project_id="proj-1",
        thread_id=job.thread_id,
        run_id=job.run_id,
        goal=job.goal,
        workflow_hint=dict(job.workflow_hint),
        workspace_root=str(tmp_path / "ws"),
        artifact_root=str(tmp_path / "art"),
    )


def test_missing_rigor_contract_kinds_flags_absent_markers(tmp_path: Path):
    from ultra_deepagents.runner import _missing_rigor_contract_kinds

    job = _study_job({"id": "pro_mode"})
    context = _context_for_job(job, tmp_path)

    missing = _missing_rigor_contract_kinds(context, job, _attempt("Only A=1.5 is chaotic. Done."))
    assert missing == [
        "rigor:uncertainty",
        "rigor:sampling",
        "rigor:step_size",
        "rigor:decision_rule",
        "rigor:discriminator",
        "rigor:limitations",
    ]

    satisfied = _attempt(
        "Classification table: ICs=6, seeds=2, durations=500 and 1000 periods, "
        "step size h=T/200. lambda = 0.112 ± 0.005. Decision rule: classified "
        "only when |lambda| > 3× spread and an independent Poincare discriminator "
        "agrees; otherwise label the row marginal.\n\nLimitations: finite observation time."
    )
    assert _missing_rigor_contract_kinds(context, job, satisfied) == []


def test_missing_rigor_contract_kinds_rejects_keyword_only_rigor(tmp_path: Path):
    from ultra_deepagents.runner import _missing_rigor_contract_kinds

    job = _study_job({"id": "pro_mode"})
    context = _context_for_job(job, tmp_path)
    keyword_only = _attempt(
        "Classification table: ICs and seeds were considered across durations. "
        "The step size was documented. lambda = 0.112 ± 0.005. Decision rule: "
        "use the 3× spread threshold. Poincare section. Limitations: finite time."
    )

    assert _missing_rigor_contract_kinds(context, job, keyword_only) == [
        "rigor:sampling",
        "rigor:step_size",
        "rigor:decision_rule",
        "rigor:discriminator",
    ]


def test_missing_rigor_contract_kinds_accepts_structured_quantitative_summary(tmp_path: Path):
    from ultra_deepagents.runner import _missing_rigor_contract_kinds

    job = _study_job({"id": "pro_mode"})
    context = _context_for_job(job, tmp_path)
    structured = _attempt(
        "Decision rule: definitive chaotic only when |lambda| > 3× spread and "
        "the independent Poincare discriminator agrees; otherwise marginal.\n\n"
        "| A | ICs/seeds | durations | step size | lambda ± spread | discriminator | class |\n"
        "| 1.50 | 3 initial conditions | 80 and 160 drive periods | h=T/200 | "
        "0.112 ± 0.005 | Poincare section agrees | chaotic |\n\n"
        "Artifacts: /outputs/metrics.csv and /outputs/report.md.\n\n"
        "Limitations: finite observation time and limited IC coverage."
    )
    artifact_events = [{"payload": {"kind": "table", "path": "outputs/metrics.csv"}}]

    assert (
        _missing_rigor_contract_kinds(
            context,
            job,
            structured,
            artifact_events=artifact_events,
        )
        == []
    )


def test_missing_rigor_contract_kinds_accepts_markdown_sampling_table(tmp_path: Path):
    from ultra_deepagents.runner import _missing_rigor_contract_kinds

    job = _study_job({"id": "pro_mode"})
    context = _context_for_job(job, tmp_path)
    answer = _attempt(
        "Classification table:\n\n"
        "| A | N_obs | ICs | lambda +/- spread | discriminator | class |\n"
        "|---|---|---|---|---|---|\n"
        "| 1.35 | 6T | 3 | -0.241 +/- 0.004 | Poincare agrees 0/3 | marginal |\n"
        "| 1.50 | 9T | 3 | 0.120 +/- 0.033 | Poincare agrees 3/3 | chaotic |\n\n"
        "Step size h = T/80. Decision rule: definitive chaotic only when "
        "|estimate| > 3× spread and an independent discriminator agrees; "
        "otherwise label marginal.\n\n"
        "Artifacts: /outputs/metrics.csv and /outputs/report.md.\n\n"
        "Limitations: finite observation time and small IC coverage."
    )
    artifact_events = [{"payload": {"kind": "table", "path": "metrics.csv"}}]

    assert (
        _missing_rigor_contract_kinds(
            context,
            job,
            answer,
            artifact_events=artifact_events,
        )
        == []
    )


def test_missing_rigor_contract_kinds_requires_artifact_references(tmp_path: Path):
    from ultra_deepagents.runner import _missing_rigor_contract_kinds

    job = _study_job({"id": "pro_mode"})
    context = _context_for_job(job, tmp_path)
    artifact_events = [{"payload": {"kind": "table", "path": "metrics.csv"}}]
    answer = _attempt(
        "Classification table: ICs=6, seeds=2, durations=500 and 1000 periods, "
        "step size h=T/200. lambda = 0.112 ± 0.005. Decision rule: classified "
        "only when |lambda| > 3× spread and an independent Poincare discriminator "
        "agrees; otherwise label the row marginal.\n\nLimitations: finite observation time."
    )

    assert _missing_rigor_contract_kinds(
        context,
        job,
        answer,
        artifact_events=artifact_events,
    ) == ["rigor:artifact_references"]

    referenced = _attempt(answer.final_response_text + "\n\nArtifacts: /outputs/metrics.csv")
    assert (
        _missing_rigor_contract_kinds(
            context,
            job,
            referenced,
            artifact_events=artifact_events,
        )
        == []
    )


def test_missing_rigor_contract_kinds_gates_on_intelligence_and_goal(tmp_path: Path):
    from ultra_deepagents.runner import _missing_rigor_contract_kinds

    bare_answer = _attempt("Only A=1.5 is chaotic.")

    high_job = _study_job()
    high_context = _context_for_job(high_job, tmp_path)
    assert _missing_rigor_contract_kinds(high_context, high_job, bare_answer) == []

    chat_job = RunJobEnvelope(
        run_id="run-chat",
        thread_id="thread-chat",
        user_id="researcher-1",
        goal="Say hello.",
        messages=[{"role": "user", "content": "Say hello."}],
        workflow_hint={"id": "pro_mode"},
    )
    chat_context = _context_for_job(chat_job, tmp_path)
    assert _missing_rigor_contract_kinds(chat_context, chat_job, bare_answer) == []

    code_debug_job = RunJobEnvelope(
        run_id="run-code-debug",
        thread_id="thread-code-debug",
        user_id="researcher-1",
        goal="Analyze this Python code and debug the workflow.",
        messages=[{"role": "user", "content": "Analyze this Python code and debug the workflow."}],
        workflow_hint={"id": "pro_mode"},
    )
    code_debug_context = _context_for_job(code_debug_job, tmp_path)
    assert _missing_rigor_contract_kinds(code_debug_context, code_debug_job, bare_answer) == []

    diagram_review_job = RunJobEnvelope(
        run_id="run-dynamics-diagram-review",
        thread_id="thread-dynamics-diagram-review",
        user_id="researcher-1",
        goal="Analyze the attached bifurcation diagram for the Lorenz system.",
        messages=[
            {
                "role": "user",
                "content": "Analyze the attached bifurcation diagram for the Lorenz system.",
            }
        ],
        workflow_hint={"id": "pro_mode"},
    )
    diagram_review_context = _context_for_job(diagram_review_job, tmp_path)
    assert (
        _missing_rigor_contract_kinds(
            diagram_review_context,
            diagram_review_job,
            bare_answer,
        )
        == []
    )

    incidental_nph_goal = (
        "Quantitatively assess this CT, estimate the Evans index, and classify NPH; "
        "the radiology report mentions a bifurcation in the disease course."
    )
    dynamics_request = (
        "Simulate the Duffing oscillator and classify dynamical regimes from "
        "Lyapunov exponents and Poincare sections."
    )
    nph_job = RunJobEnvelope(
        run_id="run-nph",
        thread_id="thread-nph",
        user_id="researcher-1",
        goal=incidental_nph_goal,
        messages=[
            {"role": "user", "content": dynamics_request},
            {
                "role": "user",
                "content": dynamics_request,
                "metadata": {"kind": "steering"},
                "run_id": "run-nph",
            },
        ],
        workflow_hint={"id": "pro_mode"},
    )
    nph_context = _context_for_job(nph_job, tmp_path)
    assert _missing_rigor_contract_kinds(nph_context, nph_job, bare_answer) == []

    dynamics_job = RunJobEnvelope(
        run_id="run-dynamics-base-goal",
        thread_id="thread-dynamics-base-goal",
        user_id="researcher-1",
        goal=dynamics_request,
        messages=[
            {"role": "user", "content": incidental_nph_goal},
            {
                "role": "user",
                "content": "Do not run a dynamics study.",
                "metadata": {"kind": "steering"},
                "run_id": "run-dynamics-base-goal",
            },
        ],
        workflow_hint={"id": "pro_mode"},
    )
    dynamics_context = _context_for_job(dynamics_job, tmp_path)
    assert _missing_rigor_contract_kinds(dynamics_context, dynamics_job, bare_answer) == [
        "rigor:uncertainty",
        "rigor:sampling",
        "rigor:step_size",
        "rigor:decision_rule",
        "rigor:discriminator",
        "rigor:limitations",
    ]

    pro_job = _study_job({"id": "pro_mode"})
    pro_context = _context_for_job(pro_job, tmp_path)
    assert _missing_rigor_contract_kinds(pro_context, pro_job, _attempt("")) == []


@pytest.mark.parametrize(
    "goal",
    [
        ("Simulate the Lorenz system, then inspect HRV metrics and classify oscillatory regimes."),
        ("Simulate the Lorenz system, then process RNA-seq data and classify stable regimes."),
        (
            "Simulate the Lorenz system, then interpret the attached paper and classify "
            "its chaotic regimes."
        ),
        ("Simulate the Lorenz system, then evaluate the CT and classify stable regimes of NPH."),
        (
            "Simulate the Lorenz system, then inspect an attached bifurcation diagram "
            "and classify its regimes."
        ),
        (
            "Simulate the Lorenz system, then inspect an HRV Poincare map and classify "
            "oscillatory regimes."
        ),
        (
            "Simulate the Lorenz system, then inspect ECG Lyapunov exponents and classify "
            "oscillatory regimes."
        ),
        (
            "Simulate the Lorenz system, then inspect a bifurcation diagram from the "
            "attached paper and classify its regimes."
        ),
        (
            "Simulate the Lorenz system, then inspect Poincare maps from an HRV recording "
            "and classify oscillatory regimes."
        ),
        (
            "Simulate the Lorenz system, then inspect Lyapunov exponents from an ECG "
            "recording and classify oscillatory regimes."
        ),
        (
            "Simulate the Lorenz system, then analyze Lyapunov exponents from the attached "
            "article and classify chaotic regimes."
        ),
        (
            "Simulate the Lorenz system, then measure Lyapunov exponents for this ECG signal "
            "and classify oscillatory regimes."
        ),
        (
            "Simulate the Lorenz system, then inspect the Duffing oscillator from the "
            "attached paper and classify chaotic regimes."
        ),
    ],
)
def test_missing_rigor_contract_kinds_rejects_coordinated_foreign_segments(
    tmp_path: Path,
    goal: str,
):
    from ultra_deepagents.runner import _missing_rigor_contract_kinds

    job = RunJobEnvelope(
        run_id="run-coordinated-foreign-segment",
        thread_id="thread-coordinated-foreign-segment",
        user_id="researcher-1",
        goal=goal,
        messages=[{"role": "user", "content": goal}],
        workflow_hint={"id": "pro_mode"},
    )
    context = _context_for_job(job, tmp_path)

    assert _missing_rigor_contract_kinds(context, job, _attempt("Only one result.")) == []


@pytest.mark.parametrize(
    "goal",
    [
        (
            "Simulate the Lorenz system, then analyze the Lyapunov spectrum and classify "
            "chaotic regimes."
        ),
        (
            "Simulate the Lorenz system, then measure Lyapunov exponents and classify "
            "chaotic regimes."
        ),
        (
            "Simulate the Lorenz system, then compute Lyapunov exponents and Poincare "
            "sections, then classify chaotic regimes."
        ),
        (
            "Simulate the Lorenz system, then analyze its Lyapunov spectrum and classify "
            "chaotic regimes."
        ),
        (
            "Simulate the Lorenz system, then compute Lyapunov exponents from its trajectories "
            "and classify chaotic regimes."
        ),
        (
            "Simulate the Lorenz system, then measure Lyapunov exponents for the Lorenz "
            "attractor and classify chaotic regimes."
        ),
        (
            "Simulate the logistic map, then compute Lyapunov exponents over parameter "
            "values and classify chaotic regimes."
        ),
        (
            "Simulate the logistic map, then measure Lyapunov exponents across the parameter "
            "range and classify chaotic regimes."
        ),
        (
            "Simulate the logistic map, then compute Lyapunov exponents at r=3.9 and "
            "classify chaotic regimes."
        ),
    ],
)
def test_missing_rigor_contract_kinds_accepts_coordinated_dynamics_evidence(
    tmp_path: Path,
    goal: str,
):
    from ultra_deepagents.runner import _missing_rigor_contract_kinds

    job = RunJobEnvelope(
        run_id="run-coordinated-dynamics-evidence",
        thread_id="thread-coordinated-dynamics-evidence",
        user_id="researcher-1",
        goal=goal,
        messages=[{"role": "user", "content": goal}],
        workflow_hint={"id": "pro_mode"},
    )
    context = _context_for_job(job, tmp_path)

    assert _missing_rigor_contract_kinds(context, job, _attempt("Only one result.")) == [
        "rigor:uncertainty",
        "rigor:sampling",
        "rigor:step_size",
        "rigor:decision_rule",
        "rigor:discriminator",
        "rigor:limitations",
    ]


def test_requested_artifacts_ignore_code_blocks_and_negated_plot_requests():
    from ultra_deepagents.runner import _requested_artifact_kinds

    prompt = """Pro mode debug request. Analyze this Python function for correctness.
Do not run a numerical experiment and do not create plots or CSVs.

```python
def normalize_rows(rows):
    totals = [sum(row) for row in rows]
    return [[x / totals[i] for x in row] for i, row in enumerate(rows)]
```

Please answer with findings and a corrected implementation."""

    assert _requested_artifact_kinds(prompt) == []


def test_requested_artifacts_do_not_treat_inference_inputs_or_tool_names_as_outputs():
    from ultra_deepagents.runner import _requested_artifact_kinds

    prompt = """Run bulk inference using the attached user-trained YOLO model checkpoint.
Provide a report, predictions, summary, verification, and annotated-image archive as durable
outputs. Delegate implementation review to qwen-code-runner if available, but continue locally
if that tool is unavailable. Do not copy the input checkpoint into the outputs."""

    assert _requested_artifact_kinds(prompt) == []


def test_requested_artifacts_require_direct_durable_code_and_model_requests():
    from ultra_deepagents.runner import _requested_artifact_kinds

    prompt = (
        "Train a UNet, save the training code, and return the trained model weights "
        "as durable outputs."
    )

    assert _requested_artifact_kinds(prompt) == ["code", "model"]


def test_completion_continuation_prompt_renders_rigor_requirements():
    from ultra_deepagents.runner import _completion_continuation_prompt

    rigor_only = _completion_continuation_prompt(
        missing_kinds=["rigor:uncertainty", "rigor:limitations", "rigor:step_size"],
        artifact_events=[],
    )
    assert "results contract" in rigor_only
    assert "mean ± spread" in rigor_only
    assert "Limitations paragraph" in rigor_only
    assert "step size" in rigor_only
    assert "missing requested durable outputs" not in rigor_only

    mixed = _completion_continuation_prompt(
        missing_kinds=["figure", "rigor:decision_rule"],
        artifact_events=[],
    )
    assert "missing artifact kinds exist: figure" in mixed
    assert "decision rule" in mixed


def test_collect_output_artifacts_skips_unreferenced_top_level_scripts(tmp_path: Path):
    from ultra_deepagents.runner import (
        _artifact_reference_text,
        _collect_output_artifacts,
    )

    job = _study_job({"id": "pro_mode"})
    context = _context_for_job(job, tmp_path)
    workspace = Path(context.workspace_root)
    artifact_dir = Path(context.artifact_root)
    (workspace / "outputs").mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    (workspace / "final_sim.py").write_text("print('final')\n")
    (workspace / "scratch_v2.py").write_text("print('scratch')\n")
    (workspace / "diagnostics").mkdir()
    (workspace / "diagnostics" / "probe.py").write_text("print('probe')\n")
    (workspace / "outputs" / "report.md").write_text("# Report\nFinal code: final_sim.py\n")

    reference_text = _artifact_reference_text(_attempt("Study complete; see report.md."), workspace)
    events = _collect_output_artifacts(
        context, workspace, artifact_dir, reference_text=reference_text
    )
    paths = [event["payload"]["path"] for event in events]

    assert "final_sim.py" in paths
    assert "scratch_v2.py" not in paths
    assert all("diagnostics" not in path for path in paths)
    assert "outputs/report.md" in paths


def test_collect_output_artifacts_skips_unreferenced_scratch_crops(tmp_path: Path):
    from ultra_deepagents.runner import (
        _artifact_reference_text,
        _collect_output_artifacts,
    )

    job = _study_job({"id": "pro_mode"})
    context = _context_for_job(job, tmp_path)
    workspace = Path(context.workspace_root)
    artifact_dir = Path(context.artifact_root)
    ocr_dir = workspace / "outputs" / "ocr"
    (ocr_dir / "crops").mkdir(parents=True)
    artifact_dir.mkdir(parents=True)

    def _write_png(path: Path, color: str) -> None:
        Image.new("RGB", (8, 8), color).save(path)

    _write_png(ocr_dir / "crop_350_450.png", "red")
    _write_png(ocr_dir / "crop_x5.png", "green")
    _write_png(ocr_dir / "crops" / "quote.png", "blue")
    _write_png(ocr_dir / "crop_referenced.png", "yellow")
    _write_png(workspace / "outputs" / "cropland_yield.png", "white")
    (ocr_dir / "crop_notes.txt").write_text("engine vs VLM disagreements\n")

    reference_text = _artifact_reference_text(
        _attempt(
            "Transcription in outputs/ocr/page.txt; the disputed glyph is shown "
            "in crop_referenced.png."
        ),
        workspace,
    )
    events = _collect_output_artifacts(
        context, workspace, artifact_dir, reference_text=reference_text
    )
    paths = [event["payload"]["path"] for event in events]

    assert "outputs/ocr/crop_350_450.png" not in paths
    assert "outputs/ocr/crop_x5.png" not in paths
    assert "outputs/ocr/crops/quote.png" not in paths
    assert "outputs/ocr/crop_referenced.png" in paths
    assert "outputs/cropland_yield.png" in paths
    assert "outputs/ocr/crop_notes.txt" in paths


def test_run_job_enforces_rigor_contract_with_one_continuation(tmp_path: Path):
    class FakeStudyThenContractAgent:
        def __init__(self) -> None:
            self.calls = 0
            self.continuation_prompt = ""

        async def astream_events(self, payload, *, context=None, version=None):
            assert version == "v3"
            self.calls += 1
            if self.calls == 1:
                answer = "Only A=1.5 is chaotic. Saved everything."
            else:
                self.continuation_prompt = payload["messages"][-1]["content"]
                answer = (
                    "Classification table: ICs=3, seeds=2, durations=500 and 1000 periods, "
                    "step size h=T/200. lambda = 0.112 ± 0.005, classified by the "
                    "decision rule |lambda| > 3× spread with independent Poincare "
                    "discriminator agreement; otherwise label the row marginal.\n\n"
                    "Limitations: finite observation window."
                )
            yield {
                "type": "event",
                "method": "values",
                "params": {
                    "namespace": [],
                    "data": {
                        "messages": [
                            {"role": "user", "content": payload["messages"][0]["content"]},
                            {"role": "assistant", "content": answer},
                        ]
                    },
                },
            }

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-rigor-e2e",
            thread_id="thread-rigor-e2e",
            user_id="researcher-1",
            goal=(
                "Simulate the Duffing oscillator and classify dynamical regimes from "
                "Lyapunov exponents and Poincare sections."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Simulate the Duffing oscillator and classify dynamical regimes from "
                        "Lyapunov exponents and Poincare sections."
                    ),
                }
            ],
            workflow_hint={"id": "pro_mode"},
        )
        agent = FakeStudyThenContractAgent()
        published = []

        async def publish(event):
            published.append(event)

        await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: agent,
        )
        return agent, published

    agent, events = asyncio.run(scenario())

    assert agent.calls == 2
    assert "results contract" in agent.continuation_prompt
    completed = next(event for event in events if event["event_kind"] == "run.completed")
    assert "±" in completed["payload"]["response_text"]
    assert "Limitations" in completed["payload"]["response_text"]


def test_run_job_does_not_force_dynamics_rigor_for_nph_prompt(tmp_path: Path):
    response = (
        "The measured Evans index is 0.34 on the selected axial slice. "
        "That measurement can support ventriculomegaly, but it does not by itself "
        "establish normal-pressure hydrocephalus; clinical correlation is required."
    )

    class FakeNPHAgent:
        def __init__(self) -> None:
            self.calls = 0

        async def astream_events(self, payload, *, context=None, version=None):
            assert version == "v3"
            self.calls += 1
            yield {
                "type": "event",
                "method": "values",
                "params": {
                    "namespace": [],
                    "data": {
                        "messages": [
                            {"role": "user", "content": payload["messages"][0]["content"]},
                            {"role": "assistant", "content": response},
                        ]
                    },
                },
            }

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        prompt = (
            "Quantitatively assess this CT, estimate the Evans index, and classify NPH; "
            "the radiology report mentions a bifurcation in the disease course."
        )
        job = RunJobEnvelope(
            run_id="run-nph-e2e",
            thread_id="thread-nph-e2e",
            user_id="researcher-1",
            goal=prompt,
            messages=[{"role": "user", "content": prompt}],
            workflow_hint={"id": "pro_mode"},
        )
        agent = FakeNPHAgent()
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: agent,
        )
        return result, agent, published

    result, agent, events = asyncio.run(scenario())

    assert agent.calls == 1
    assert not [event for event in events if event.get("node_name") == "completion_guard"]
    assert result == response
    completed = next(event for event in events if event["event_kind"] == "run.completed")
    assert completed["payload"]["response_text"] == response


def test_run_job_does_not_force_dynamics_rigor_for_lorenz_literature_review(
    tmp_path: Path,
):
    response = (
        "The reviewed sources use the Lorenz system as a canonical example of deterministic "
        "chaos and discuss sensitivity to initial conditions. This is a literature summary, "
        "not a new numerical regime classification."
    )

    class FakeLiteratureReviewAgent:
        def __init__(self) -> None:
            self.calls = 0

        async def astream_events(self, payload, *, context=None, version=None):
            assert version == "v3"
            self.calls += 1
            yield {
                "type": "event",
                "method": "values",
                "params": {
                    "namespace": [],
                    "data": {
                        "messages": [
                            {"role": "user", "content": payload["messages"][0]["content"]},
                            {"role": "assistant", "content": response},
                        ]
                    },
                },
            }

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        prompt = "Run a literature review of the Lorenz chaos study."
        job = RunJobEnvelope(
            run_id="run-lorenz-literature-review",
            thread_id="thread-lorenz-literature-review",
            user_id="researcher-1",
            goal=prompt,
            messages=[{"role": "user", "content": prompt}],
            workflow_hint={"id": "pro_mode"},
        )
        agent = FakeLiteratureReviewAgent()
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: agent,
        )
        return result, agent, published

    result, agent, events = asyncio.run(scenario())

    assert agent.calls == 1
    assert not [event for event in events if event.get("node_name") == "completion_guard"]
    assert result == response
    completed = next(event for event in events if event["event_kind"] == "run.completed")
    assert completed["payload"]["response_text"] == response


def test_run_job_does_not_force_dynamics_rigor_for_hrv_with_lorenz_citation(
    tmp_path: Path,
):
    response = (
        "The HRV feature summary shows an oscillatory component in the measured signal. "
        "The cited Lorenz example is contextual and does not turn this into a new "
        "dynamical-system regime experiment."
    )

    class FakeHRVAgent:
        def __init__(self) -> None:
            self.calls = 0

        async def astream_events(self, payload, *, context=None, version=None):
            assert version == "v3"
            self.calls += 1
            yield {
                "type": "event",
                "method": "values",
                "params": {
                    "namespace": [],
                    "data": {
                        "messages": [
                            {"role": "user", "content": payload["messages"][0]["content"]},
                            {"role": "assistant", "content": response},
                        ]
                    },
                },
            }

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        prompt = (
            "Compute HRV metrics and classify oscillatory regimes while citing the "
            "Lorenz system chaos study."
        )
        job = RunJobEnvelope(
            run_id="run-hrv-lorenz-citation",
            thread_id="thread-hrv-lorenz-citation",
            user_id="researcher-1",
            goal=prompt,
            messages=[{"role": "user", "content": prompt}],
            workflow_hint={"id": "pro_mode"},
        )
        agent = FakeHRVAgent()
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: agent,
        )
        return result, agent, published

    result, agent, events = asyncio.run(scenario())

    assert agent.calls == 1
    assert not [event for event in events if event.get("node_name") == "completion_guard"]
    assert result == response
    completed = next(event for event in events if event["event_kind"] == "run.completed")
    assert completed["payload"]["response_text"] == response


@pytest.mark.parametrize(
    "prompt",
    [
        (
            "Simulate the Lorenz system for illustration, and compute HRV metrics and "
            "classify oscillatory regimes."
        ),
        ("Simulate the Lorenz system, then inspect HRV metrics and classify oscillatory regimes."),
    ],
)
def test_run_job_does_not_force_dynamics_rigor_across_same_sentence_task_switch(
    tmp_path: Path,
    prompt: str,
):
    response = (
        "The Lorenz simulation is illustrative only. The HRV calculation reports the "
        "requested signal metrics without treating them as a Lorenz-regime experiment."
    )

    class FakeCrossObjectAgent:
        def __init__(self) -> None:
            self.calls = 0

        async def astream_events(self, payload, *, context=None, version=None):
            assert version == "v3"
            self.calls += 1
            yield {
                "type": "event",
                "method": "values",
                "params": {
                    "namespace": [],
                    "data": {
                        "messages": [
                            {"role": "user", "content": payload["messages"][0]["content"]},
                            {"role": "assistant", "content": response},
                        ]
                    },
                },
            }

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-cross-object-task-switch",
            thread_id="thread-cross-object-task-switch",
            user_id="researcher-1",
            goal=prompt,
            messages=[{"role": "user", "content": prompt}],
            workflow_hint={"id": "pro_mode"},
        )
        agent = FakeCrossObjectAgent()
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: agent,
        )
        return result, agent, published

    result, agent, events = asyncio.run(scenario())

    assert agent.calls == 1
    assert not [event for event in events if event.get("node_name") == "completion_guard"]
    assert result == response
    completed = next(event for event in events if event["event_kind"] == "run.completed")
    assert completed["payload"]["response_text"] == response


def test_run_job_does_not_force_rigor_or_artifacts_for_negated_code_debug_prompt(tmp_path: Path):
    class FakeCodeDebugAgent:
        def __init__(self) -> None:
            self.calls = 0

        async def astream_events(self, payload, *, context=None, version=None):
            assert version == "v3"
            self.calls += 1
            answer = (
                "Findings: the function divides by zero for empty or zero-sum rows "
                "and has unclear input validation.\n\n"
                "Corrected implementation:\n"
                "```python\n"
                "def normalize_rows(rows):\n"
                "    out = []\n"
                "    for row in rows:\n"
                "        total = sum(row)\n"
                "        if total == 0:\n"
                "            raise ZeroDivisionError('row sum is zero')\n"
                "        out.append([x / total for x in row])\n"
                "    return out\n"
                "```"
            )
            yield {
                "type": "event",
                "method": "values",
                "params": {
                    "namespace": [],
                    "data": {
                        "messages": [
                            {"role": "user", "content": payload["messages"][0]["content"]},
                            {"role": "assistant", "content": answer},
                        ]
                    },
                },
            }

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        prompt = """Pro mode debug request. Analyze this Python function for correctness.
Do not run a numerical experiment and do not create plots or CSVs.

```python
def normalize_rows(rows):
    totals = [sum(row) for row in rows]
    return [[x / totals[i] for x in row] for i, row in enumerate(rows)]
```

Please answer with findings and a corrected implementation."""
        job = RunJobEnvelope(
            run_id="run-code-debug-negated",
            thread_id="thread-code-debug-negated",
            user_id="researcher-1",
            goal=prompt,
            messages=[{"role": "user", "content": prompt}],
            workflow_hint={"id": "pro_mode"},
        )
        agent = FakeCodeDebugAgent()
        published = []

        async def publish(event):
            published.append(event)

        await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: agent,
        )
        return agent, published

    agent, events = asyncio.run(scenario())

    assert agent.calls == 1
    assert not [event for event in events if event.get("node_name") == "completion_guard"]
    completed = next(event for event in events if event["event_kind"] == "run.completed")
    assert "Corrected implementation" in completed["payload"]["response_text"]


def test_run_job_requires_failed_task_delegation_fallback_disclosure(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-delegation-fallback",
            thread_id="thread-delegation-fallback",
            user_id="researcher-1",
            goal="Run a simulation and verify the classification.",
            messages=[
                {"role": "user", "content": "Run a simulation and verify the classification."}
            ],
            workflow_hint={"id": "pro_mode"},
        )
        agent = FakeFailedTaskThenFallbackDisclosureAgent()
        published = []

        async def publish(event):
            published.append(event)

        await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: agent,
        )
        return agent, published

    agent, events = asyncio.run(scenario())

    assert agent.calls == 2
    assert "task delegation failed" in agent.continuation_prompt.lower()
    completed = next(event for event in events if event["event_kind"] == "run.completed")
    response_text = completed["payload"]["response_text"]
    assert "Task delegation failed" in response_text
    assert "local fallback verification" in response_text


def test_run_job_treats_invalid_subagent_task_output_as_failed_delegation(tmp_path: Path):
    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
        )
        job = RunJobEnvelope(
            run_id="run-invalid-subagent-fallback",
            thread_id="thread-invalid-subagent-fallback",
            user_id="researcher-1",
            goal="Analyze this debug workflow and verify the delegation handling.",
            messages=[
                {
                    "role": "user",
                    "content": "Analyze this debug workflow and verify the delegation handling.",
                }
            ],
            workflow_hint={"id": "pro_mode"},
        )
        agent = FakeCompletedInvalidTaskThenFallbackDisclosureAgent()
        published = []

        async def publish(event):
            published.append(event)

        await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: agent,
        )
        return agent, published

    agent, events = asyncio.run(scenario())

    assert agent.calls == 2
    assert "task delegation failed" in agent.continuation_prompt.lower()
    task_completed = [
        event
        for event in events
        if event["event_kind"] == "tool_call.completed" and event["payload"]["tool_name"] == "task"
    ]
    assert task_completed[0]["payload"]["delegation_failure"] is True
    completed = next(event for event in events if event["event_kind"] == "run.completed")
    response_text = completed["payload"]["response_text"]
    assert "Task delegation failed" in response_text
    assert "local fallback verification" in response_text


class FakeDegenerateThenObservesFallbackAgent(FakeDegenerateReasoningOnceThenRecoversAgent):
    """Records whether the thinking fallback was armed when each attempt ran."""

    def __init__(self) -> None:
        super().__init__()
        self.fallback_state_per_call: list[bool] = []

    def astream_events(self, payload, *, config=None, context=None, version=None):
        from ultra_deepagents.model import thinking_fallback_armed

        self.fallback_state_per_call.append(thinking_fallback_armed())
        return super().astream_events(payload, config=config, context=context, version=version)


def test_degeneration_recovery_arms_thinking_fallback_run_scoped(tmp_path: Path):
    """Attempt 1 runs unarmed; the retry after a degeneration recovery runs with
    the thinking fallback armed; a FRESH run starts unarmed again (run-scoped
    reset), and the recovery event advertises the escalation."""

    async def scenario():
        fake_agent = FakeDegenerateThenObservesFallbackAgent()
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            model_stream_idle_max_recoveries=1,
        )
        prompt = "Can you provide real computations for delta attention?"
        job = RunJobEnvelope(
            run_id="run-fallback-escalation",
            thread_id="thread-fallback-escalation",
            user_id="researcher-1",
            goal=prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: fake_agent,
        )

        # A brand-new run on the same task context must start unarmed.
        fresh_agent = FakeDegenerateThenObservesFallbackAgent()
        fresh_job = RunJobEnvelope(
            run_id="run-fallback-reset",
            thread_id="thread-fallback-reset",
            user_id="researcher-1",
            goal=prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        await run_job(
            fresh_job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: fresh_agent,
        )
        return result, published, fake_agent, fresh_agent

    result, events, fake_agent, fresh_agent = asyncio.run(scenario())

    assert result == "Recovered with one complete, coherent answer."
    assert fake_agent.fallback_state_per_call == [False, True]
    assert fresh_agent.fallback_state_per_call[0] is False
    recovery = next(
        event
        for event in events
        if event.get("payload", {}).get("reason") == "reasoning_degeneration"
    )
    assert recovery["payload"]["thinking_fallback_armed"] is True


def test_degeneration_recovery_respects_disabled_fallback(tmp_path: Path):
    async def scenario():
        fake_agent = FakeDegenerateThenObservesFallbackAgent()
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            model_stream_idle_max_recoveries=1,
            degeneration_thinking_fallback=False,
        )
        prompt = "Can you provide real computations for delta attention?"
        job = RunJobEnvelope(
            run_id="run-fallback-disabled",
            thread_id="thread-fallback-disabled",
            user_id="researcher-1",
            goal=prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        published = []

        async def publish(event):
            published.append(event)

        await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: fake_agent,
        )
        return published, fake_agent

    events, fake_agent = asyncio.run(scenario())

    assert fake_agent.fallback_state_per_call == [False, False]
    recovery = next(
        event
        for event in events
        if event.get("payload", {}).get("reason") == "reasoning_degeneration"
    )
    assert recovery["payload"]["thinking_fallback_armed"] is False


class FakeDisconnectOnceThenRecoversAgent:
    """First attempt raises a mid-stream APIConnectionError (the live
    2026-08-16 httpx.ReadError failure shape); second attempt completes."""

    def __init__(self) -> None:
        self.calls = 0
        self.recovery_prompt = ""

    async def astream_events(self, payload, *, context=None, version=None):
        import httpx
        from openai import APIConnectionError

        assert version == "v3"
        self.calls += 1
        if self.calls == 1:
            yield {
                "type": "event",
                "method": "tools",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "tool-finished",
                        "tool_call_id": "call-eig-1",
                        "output": "eigenvalues: [-2.14, -0.87, 0.13, 1.02, 2.55, 3.31]",
                    },
                },
            }
            raise APIConnectionError(request=httpx.Request("POST", "http://example.test/v1"))

        self.recovery_prompt = payload["messages"][-1]["content"]
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": payload["messages"][-1]["content"]},
                        {
                            "role": "assistant",
                            "content": "Resumed after the connection drop: the eigenvalues are [-2.14, -0.87, 0.13, 1.02, 2.55, 3.31].",
                        },
                    ]
                },
            },
        }


def test_run_job_recovers_from_mid_stream_connection_drop(tmp_path: Path):
    async def scenario():
        fake_agent = FakeDisconnectOnceThenRecoversAgent()
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            model_stream_idle_max_recoveries=1,
        )
        prompt = "Compute the eigenvalues of the matrix and report them."
        job = RunJobEnvelope(
            run_id="run-disconnect-recovery",
            thread_id="thread-disconnect-recovery",
            user_id="researcher-1",
            goal=prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        published = []

        async def publish(event):
            published.append(event)

        result = await run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: fake_agent,
        )
        return result, published, fake_agent

    result, events, fake_agent = asyncio.run(scenario())

    assert fake_agent.calls == 2
    assert (
        result
        == "Resumed after the connection drop: the eigenvalues are [-2.14, -0.87, 0.13, 1.02, 2.55, 3.31]."
    )
    assert "transient provider connection drop" in fake_agent.recovery_prompt
    assert "do NOT re-run a step whose output already exists" in fake_agent.recovery_prompt
    stalled = next(event for event in events if event["event_kind"] == "trace.model.stalled")
    assert stalled["payload"]["idle_scope"] == "model_disconnect"
    recovery = next(
        event
        for event in events
        if event.get("payload", {}).get("reason") == "model_stream_disconnect"
    )
    assert recovery["payload"]["recovery_index"] == 1
    # A pure transport drop is NOT a thinking-path collapse: no escalation.
    assert recovery["payload"]["thinking_fallback_armed"] is False
    assert events[-1]["event_kind"] == "run.completed"


class FakeAlwaysDisconnectsAgent:
    def __init__(self) -> None:
        self.calls = 0

    async def astream_events(self, payload, *, context=None, version=None):
        import httpx
        from openai import APIConnectionError

        self.calls += 1
        if False:
            yield {}
        raise APIConnectionError(request=httpx.Request("POST", "http://example.test/v1"))


def test_run_job_fails_after_exhausting_disconnect_recoveries(tmp_path: Path):
    async def scenario():
        fake_agent = FakeAlwaysDisconnectsAgent()
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            artifact_root=str(tmp_path / "artifacts"),
            model_stream_idle_max_recoveries=1,
        )
        job = RunJobEnvelope(
            run_id="run-disconnect-exhausted",
            thread_id="thread-disconnect-exhausted",
            user_id="researcher-1",
            goal="anything",
            messages=[{"role": "user", "content": "anything"}],
        )
        published = []

        async def publish(event):
            published.append(event)

        with pytest.raises(Exception):
            await run_job(
                job,
                settings,
                publish_event=publish,
                agent_factory=lambda *_args, **_kwargs: fake_agent,
            )
        return fake_agent

    fake_agent = asyncio.run(scenario())
    # initial attempt + exactly one bounded recovery, never an infinite loop
    assert fake_agent.calls == 2
