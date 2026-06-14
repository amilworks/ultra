from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
from pathlib import Path
from urllib import error as urllib_error

import nats.errors
import pytest
import ultra_deepagents.nats_worker as nats_worker_module
from PIL import Image
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.nats_worker import (
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
)
from ultra_deepagents.runner import run_job
from ultra_deepagents.schemas import RunJobEnvelope


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
            "metadata": {"principal": {"org_id": "allen", "role": "researcher"}},
        }
    )

    context = job.to_context(artifact_root=str(tmp_path / "artifacts"), workspace_root=str(tmp_path / "workspace"))

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
    assert context.auth_claims["role"] == "researcher"


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
                        "start_async_task description is required for async "
                        "subagent delegation."
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
                        "update_async_task message is required for async "
                        "subagent delegation."
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
            "data": {"output": "analysis complete\nsaved plot.png"},
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
        assert "do not repeat completed tool work" in self.recovery_prompt
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
        (output_dir / "rarespot_combined_report.md").write_text("# RareSpot report\n\nMetrics table.\n")
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
    assert all(
        event["payload"]["usage_event_id"] != event["event_id"] for event in usage_events
    )
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
    learned_preferences.write_text("## Learned preference\nUse concise equations.\n", encoding="utf-8")
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
    assert learned_preferences.read_text(encoding="utf-8") == "## Learned preference\nUse concise equations.\n"


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
        if event["event_kind"] == "tool_call.started"
        and event["payload"]["tool_name"] == "execute"
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
        if event["event_kind"] == "tool_call.started"
        and event["payload"]["tool_name"] == "task"
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
        if event["event_kind"] == "tool_call.started"
        and event["payload"]["tool_name"] == "task"
    )
    assert task_started["payload"]["subagent_type"] == "code-runner"

    execute_started = next(
        event
        for event in events
        if event["event_kind"] == "tool_call.started"
        and event["payload"]["tool_name"] == "execute"
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


def test_run_job_publishes_tool_lifecycle_without_polluting_response(tmp_path: Path):
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
    assert events[2]["payload"]["output_preview"] == "analysis complete\nsaved plot.png"
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
        "run.failed",
    ]
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
        event.get("payload", {}).get("reason") == "model_stream_idle"
        for event in events
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
        "trace.message.delta",
        "run.completed",
    ]
    assert events[3]["payload"]["recovery_index"] == 1
    assert events[3]["payload"]["reason"] == "model_stream_idle"
    assert events[-1]["payload"]["response_text"] == result
    lease = json.loads((tmp_path / "workspaces" / "run-1" / "lease.json").read_text())
    assert lease["status"] == "succeeded"


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


def test_run_job_ignores_prior_assistant_when_followup_final_state_has_no_new_answer(tmp_path: Path):
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

    assert result == "Saved durable outputs:\n- Code: Plot X Squared (`outputs/plot_x_squared.py`)"
    assert [event["event_kind"] for event in events] == [
        "run.started",
        "artifact.created",
        "run.completed",
    ]
    assert events[-1]["payload"]["response_text"] == result


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
    artifact_payloads = [event["payload"] for event in events if event["event_kind"] == "artifact.created"]
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


def test_run_job_normalizes_collected_plot_artifacts_to_publication_ppi(tmp_path: Path):
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
    artifact_payload = next(event["payload"] for event in events if event["event_kind"] == "artifact.created")
    copied = tmp_path / "artifacts" / "run-1" / artifact_payload["path"]
    with Image.open(copied) as image:
        dpi = image.info.get("dpi")

    assert dpi is not None
    assert dpi[0] >= 299.5
    assert dpi[1] >= 299.5
    quality = artifact_payload["metadata"]["figure_quality"]
    assert quality["minimum_ppi"] == 300
    assert quality["dpi_metadata_normalized"] is True


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

    artifact_payloads = [event["payload"] for event in events if event["event_kind"] == "artifact.created"]
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

    artifact_payloads = [event["payload"] for event in events if event["event_kind"] == "artifact.created"]
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
    assert result == (
        "Executed the plotting script and saved the code (plot_x2.py) and "
        "figure (plot_x2.png)."
    )
    assert "figure" in fake_agent.continuation_prompt
    assert [event["event_kind"] for event in events] == [
        "run.started",
        "trace.message.delta",
        "artifact.created",
        "artifact.created",
        "run.completed",
    ]
    artifact_payloads = [event["payload"] for event in events if event["event_kind"] == "artifact.created"]
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
    assert result == "Saved the trained model weights, code, and figures."
    assert "model" in fake_agent.continuation_prompt
    trace_events = [event for event in events if event["event_kind"] == "trace.message.delta"]
    assert [event["payload"]["missing_artifact_kinds"] for event in trace_events] == [["model"]]
    artifact_payloads = [event["payload"] for event in events if event["event_kind"] == "artifact.created"]
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
    assert [event["payload"]["missing_artifact_kinds"] for event in trace_events] == [
        ["response"]
    ]
    artifact_payloads = [event["payload"] for event in events if event["event_kind"] == "artifact.created"]
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
    assert [event["payload"]["missing_artifact_kinds"] for event in trace_events] == [
        ["response"]
    ]


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


def test_fetch_job_messages_treats_nats_timeout_as_empty_poll():
    assert asyncio.run(fetch_job_messages(TimeoutSubscription())) == []


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
    assert config.ack_wait == 300.0
    assert config.max_deliver == 7
    assert config.max_ack_pending == 3
    assert job_ack_extension_interval(settings) == 60.0


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


class CapturingJetStream:
    def __init__(self):
        self.published = []

    async def publish(self, subject: str, payload: bytes, **kwargs):
        self.published.append((subject, payload, kwargs.get("headers") or {}))


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

        async def _run_event_sequence_floor(self, run_id: str) -> int:
            assert run_id == "run-cleanup"
            return 0

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


def test_worker_claims_and_releases_control_plane_run_lease_around_compute(tmp_path: Path):
    calls: list[tuple[str, str]] = []

    async def run_job_returns(*_args, **_kwargs):
        calls.append(("run_job", "run-1"))
        return "ok"

    async def acquire_lease(run_id, settings):
        calls.append(("acquire", run_id))
        return ControlPlaneRunLease(
            run_id=run_id,
            worker_id="worker-a",
            lease_token="lease-token-1",
        )

    async def renew_lease(lease, settings):
        calls.append(("renew", lease.lease_token))
        return lease

    async def release_lease(lease, settings):
        calls.append(("release", lease.lease_token))

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            workspace_root=str(tmp_path / "workspaces"),
            worker_ack_progress_interval_seconds=0,
        )
        worker = NATSDeepAgentsWorker(
            settings,
            run_job_func=run_job_returns,
            run_lease_func=acquire_lease,
            renew_run_lease_func=renew_lease,
            release_run_lease_func=release_lease,
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

    async def renew_lost_lease(lease, settings):
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
            workspace_root=str(tmp_path / "workspaces"),
            worker_ack_wait_seconds=1.0,
            worker_ack_progress_interval_seconds=0.01,
        )
        worker = NATSDeepAgentsWorker(
            settings,
            run_job_func=long_running_run_job,
            run_status_func=run_status,
            run_lease_func=acquire_lease,
            renew_run_lease_func=renew_lost_lease,
            release_run_lease_func=release_lease,
        )
        message = FakeNATSMessage(
            b'{"run_id":"run-1","thread_id":"thread-1","user_id":"user-1","goal":"lease"}'
        )
        js = CapturingJetStream()
        await asyncio.wait_for(worker._process_message(message, js), timeout=0.5)
        return message, _published_events(js)

    message, events = asyncio.run(scenario())

    assert compute_cancelled.is_set()
    assert calls == ["acquire:run-1", "renew:lease-token-1", "release:lease-token-1"]
    assert message.acked == 0
    assert message.naked == 1
    assert message.nak_delays == [0.01]
    assert [event["event_kind"] for event in events] == []


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

    async def renew_unreachable_lease(lease, settings):
        nonlocal renewal_attempts
        renewal_attempts += 1
        if renewal_attempts >= 3:
            renewals_seen.set()
        raise RunLeaseUnavailable("control plane connection refused")

    async def release_lease(lease, settings):
        return None

    async def run_status(_run_id, _settings):
        return None

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
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
            renew_run_lease_func=renew_unreachable_lease,
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


def test_worker_publishes_run_heartbeat_during_silent_long_running_compute():
    async def scenario():
        js = HeartbeatCapturingJetStream()

        async def long_running_run_job(*_args, **_kwargs):
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
        first_task = asyncio.create_task(worker._process_message(first_message, CapturingJetStream()))
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

        first_task = asyncio.create_task(first_worker._process_message(first_message, CapturingJetStream()))
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
    assert [event["event_kind"] for event in events] == []


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
        await worker._handle_cancel_payload({"run_id": "run-1", "reason": "user stop"}, js)
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
        await worker._handle_cancel_payload({"run_id": "run-1", "reason": "pre-canceled"}, js)
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

    async def run_job_should_not_start(*_args, **_kwargs):
        nonlocal calls
        calls += 1

    async def terminal_status(_run_id, _settings):
        return "canceled"

    async def scenario():
        settings = RuntimeSettings(
            openai_base_url="http://example.test/v1",
            openai_model="deepseek_v4",
            worker_ack_progress_interval_seconds=0,
        )
        worker = NATSDeepAgentsWorker(
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
    assert events == []


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
    assert events == []


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
        if status_calls >= 3:
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
        if status_calls >= 3:
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
    assert events == []


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
    assert events == []


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


def test_control_plane_run_sequence_floor_paginates_events_with_worker_identity(monkeypatch):
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
                {"events": [{"sequence": sequence} for sequence in range(1, 501)]}
            )
        return FakeResponse({"events": [{"sequence": 501}, {"sequence": 513}]})

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

    assert max_sequence == 513
    assert [call["url"] for call in calls] == [
        "http://control.test/v2/runs/run%20with%20space/events?limit=500&after_sequence=0",
        "http://control.test/v2/runs/run%20with%20space/events?limit=500&after_sequence=500",
    ]
    for call in calls:
        assert call["timeout"] == 4.0
        headers = call["headers"]
        assert headers["x-ultra-worker-token"] == "trace-worker-secret"
        assert headers["x-ultra-user-id"] == "bisque:researcher"


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
                        "sequence": 3,
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
        goal="Run a simulation, analyze metrics, and plot results.",
        messages=[{"role": "user", "content": "Run a simulation and analyze metrics."}],
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

    missing = _missing_rigor_contract_kinds(
        context, job, _attempt("Only A=1.5 is chaotic. Done.")
    )
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
    assert _missing_rigor_contract_kinds(
        context,
        job,
        referenced,
        artifact_events=artifact_events,
    ) == []


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

    pro_job = _study_job({"id": "pro_mode"})
    pro_context = _context_for_job(pro_job, tmp_path)
    assert _missing_rigor_contract_kinds(pro_context, pro_job, _attempt("")) == []


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
    (workspace / "outputs" / "report.md").write_text(
        "# Report\nFinal code: final_sim.py\n"
    )

    reference_text = _artifact_reference_text(
        _attempt("Study complete; see report.md."), workspace
    )
    events = _collect_output_artifacts(
        context, workspace, artifact_dir, reference_text=reference_text
    )
    paths = [event["payload"]["path"] for event in events]

    assert "final_sim.py" in paths
    assert "scratch_v2.py" not in paths
    assert all("diagnostics" not in path for path in paths)
    assert "outputs/report.md" in paths


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
            goal="Run a simulation and analyze metrics.",
            messages=[{"role": "user", "content": "Run a simulation and analyze metrics."}],
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
            messages=[{"role": "user", "content": "Run a simulation and verify the classification."}],
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
        if event["event_kind"] == "tool_call.completed"
        and event["payload"]["tool_name"] == "task"
    ]
    assert task_completed[0]["payload"]["delegation_failure"] is True
    completed = next(event for event in events if event["event_kind"] == "run.completed")
    response_text = completed["payload"]["response_text"]
    assert "Task delegation failed" in response_text
    assert "local fallback verification" in response_text
