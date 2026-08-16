from __future__ import annotations

import asyncio
import json

import pytest
from langchain_core.tools import tool
from ultra_deepagents.harness_plugins import (
    HarnessPlugin,
    HarnessPluginRegistry,
    ProgramToolPolicy,
    ToolConcurrency,
)
from ultra_deepagents.tool_program import (
    ToolProgram,
    ToolProgramLimits,
    execute_tool_program,
)


@tool
def read_metric(name: str) -> dict[str, str]:
    """Read a metric by name."""
    return {"name": name}


@tool
def write_note(text: str) -> dict[str, str]:
    """Write a note in an external system."""
    return {"text": text}


def _capabilities():
    surface = HarnessPluginRegistry(
        (
            HarnessPlugin(
                name="metrics",
                tools=(read_metric, write_note),
                program_tools=(
                    ProgramToolPolicy(
                        tool_name="read_metric",
                        concurrency=ToolConcurrency.PARALLEL,
                    ),
                    ProgramToolPolicy(
                        tool_name="write_note",
                        concurrency=ToolConcurrency.EXCLUSIVE,
                    ),
                ),
            ),
        )
    ).freeze()
    return surface.program_tools


def _run(coroutine):
    return asyncio.run(coroutine)


def test_program_parallelizes_calls_and_keeps_unselected_intermediates_out_of_context():
    active = 0
    max_active = 0

    async def invoke(capability, arguments, step_id):
        nonlocal active, max_active
        assert capability.tool.name == "read_metric"
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        if step_id == "forwarded":
            return {"name": arguments["name"]}
        if step_id == "sample_a":
            return {
                "items": [
                    {"label": "a", "score": 1.0},
                    {"label": "b", "score": 2.0},
                    {"label": "c", "score": 3.0},
                ],
                "private_intermediate": "must-not-enter-the-model-context",
            }
        return {"items": [{"label": "control", "score": 9.0}]}

    program = ToolProgram.model_validate(
        {
            "operations": [
                {
                    "kind": "call",
                    "id": "sample_a",
                    "tool": "read_metric",
                    "arguments": {"name": "a"},
                },
                {
                    "kind": "call",
                    "id": "sample_b",
                    "tool": "read_metric",
                    "arguments": {"name": "b"},
                },
                {
                    "kind": "transform",
                    "id": "filtered",
                    "operation": "filter",
                    "source": {"step": "sample_a", "pointer": "/items"},
                    "where": {
                        "pointer": "/score",
                        "operator": "gte",
                        "value": 2.0,
                    },
                },
                {
                    "kind": "transform",
                    "id": "average",
                    "operation": "aggregate",
                    "source": {"step": "filtered"},
                    "aggregate": "mean",
                    "value_pointer": "/score",
                },
                {
                    "kind": "call",
                    "id": "forwarded",
                    "tool": "read_metric",
                    "arguments": {"name": {"$ref": "sample_a#/items/1/label"}},
                },
            ],
            "outputs": [
                {
                    "name": "mean_score",
                    "source": {"step": "average", "pointer": "/value"},
                },
                {
                    "name": "forwarded_label",
                    "source": {"step": "forwarded", "pointer": "/name"},
                },
            ],
        }
    )

    result = _run(execute_tool_program(program, _capabilities(), invoke=invoke))

    assert max_active == 2
    assert result["status"] == "succeeded"
    assert result["outputs"] == {"mean_score": 2.5, "forwarded_label": "b"}
    assert [receipt["status"] for receipt in result["receipts"]] == [
        "succeeded",
        "succeeded",
        "succeeded",
        "succeeded",
        "succeeded",
    ]
    assert "private_intermediate" not in json.dumps(result)
    assert "must-not-enter-the-model-context" not in json.dumps(result)


def test_exclusive_tools_are_ordered_barriers_between_parallel_batches():
    active: set[str] = set()
    overlaps: list[tuple[str, tuple[str, ...]]] = []
    completed: list[str] = []

    async def invoke(capability, arguments, step_id):
        _ = arguments
        overlaps.append((step_id, tuple(sorted(active))))
        active.add(step_id)
        await asyncio.sleep(0.01)
        if capability.concurrency is ToolConcurrency.EXCLUSIVE:
            assert active == {step_id}
        active.remove(step_id)
        completed.append(step_id)
        return {"step": step_id}

    program = ToolProgram.model_validate(
        {
            "operations": [
                {"kind": "call", "id": "read_a", "tool": "read_metric"},
                {"kind": "call", "id": "read_b", "tool": "read_metric"},
                {
                    "kind": "call",
                    "id": "mutation",
                    "tool": "write_note",
                    "arguments": {"text": "done"},
                },
                {"kind": "call", "id": "read_c", "tool": "read_metric"},
            ],
            "outputs": [{"name": "last", "source": {"step": "read_c", "pointer": "/step"}}],
        }
    )

    result = _run(execute_tool_program(program, _capabilities(), invoke=invoke))

    assert result["status"] == "succeeded"
    assert set(completed[:2]) == {"read_a", "read_b"}
    assert completed[2:] == ["mutation", "read_c"]
    assert ("mutation", ()) in overlaps
    assert result["outputs"] == {"last": "read_c"}


def test_parallel_tool_failure_is_isolated_from_independent_sibling():
    async def invoke(capability, arguments, step_id):
        _ = capability, arguments
        if step_id == "failed":
            raise RuntimeError("sensitive upstream detail must be redacted")
        return {"value": 42}

    program = ToolProgram.model_validate(
        {
            "operations": [
                {"kind": "call", "id": "failed", "tool": "read_metric"},
                {"kind": "call", "id": "succeeded", "tool": "read_metric"},
            ],
            "outputs": [
                {
                    "name": "value",
                    "source": {"step": "succeeded", "pointer": "/value"},
                }
            ],
        }
    )

    result = _run(execute_tool_program(program, _capabilities(), invoke=invoke))

    assert result["status"] == "partial"
    assert result["outputs"] == {"value": 42}
    assert result["receipts"][0]["error_code"] == "tool_failed"
    assert result["receipts"][1]["status"] == "succeeded"
    assert "sensitive upstream detail" not in json.dumps(result)


def test_program_conditions_branch_without_calling_the_skipped_tool():
    invoked: list[str] = []

    async def invoke(capability, arguments, step_id):
        _ = capability, arguments
        invoked.append(step_id)
        return {"continue": False}

    program = ToolProgram.model_validate(
        {
            "operations": [
                {"kind": "call", "id": "gate", "tool": "read_metric"},
                {
                    "kind": "call",
                    "id": "blocked_mutation",
                    "tool": "write_note",
                    "when": {
                        "source": {"step": "gate", "pointer": "/continue"},
                        "operator": "truthy",
                    },
                },
            ],
            "outputs": [{"name": "decision", "source": {"step": "gate", "pointer": "/continue"}}],
        }
    )

    result = _run(execute_tool_program(program, _capabilities(), invoke=invoke))

    assert invoked == ["gate"]
    assert result["status"] == "succeeded"
    assert result["receipts"][1]["status"] == "skipped"
    assert result["outputs"] == {"decision": False}


def test_exclusive_call_does_not_detach_work_behind_an_inner_timeout():
    async def slow_exclusive(capability, arguments, step_id):
        _ = capability, arguments
        await asyncio.sleep(0.01)
        return {"step": step_id}

    program = ToolProgram.model_validate(
        {
            "operations": [
                {
                    "kind": "call",
                    "id": "mutation",
                    "tool": "write_note",
                    "arguments": {"text": "ordered"},
                }
            ],
            "outputs": [{"name": "step", "source": {"step": "mutation", "pointer": "/step"}}],
        }
    )

    result = _run(
        execute_tool_program(
            program,
            _capabilities(),
            invoke=slow_exclusive,
            limits=ToolProgramLimits(call_timeout_seconds=0.001),
        )
    )

    assert result["status"] == "succeeded"
    assert result["outputs"] == {"step": "mutation"}


def test_program_fails_closed_for_unknown_tools_forward_refs_and_oversized_results():
    invoked = 0

    async def never_invoke(capability, arguments, step_id):
        nonlocal invoked
        _ = capability, arguments, step_id
        invoked += 1
        return {"payload": "x" * 5000}

    unknown = ToolProgram.model_validate(
        {
            "operations": [{"kind": "call", "id": "bad", "tool": "shell"}],
            "outputs": [{"name": "bad", "source": {"step": "bad"}}],
        }
    )
    unknown_result = _run(execute_tool_program(unknown, _capabilities(), invoke=never_invoke))
    assert unknown_result["status"] == "rejected"
    assert unknown_result["error"]["code"] == "tool_not_allowed"
    assert invoked == 0

    malformed_reference = ToolProgram.model_validate(
        {
            "operations": [
                {
                    "kind": "call",
                    "id": "bad_ref",
                    "tool": "read_metric",
                    "arguments": {"name": {"$ref": "later#/value", "extra": True}},
                }
            ],
            "outputs": [{"name": "value", "source": {"step": "bad_ref"}}],
        }
    )
    malformed_result = _run(
        execute_tool_program(
            malformed_reference,
            _capabilities(),
            invoke=never_invoke,
        )
    )
    assert malformed_result["status"] == "rejected"
    assert malformed_result["error"]["code"] == "invalid_reference"
    assert invoked == 0

    invalid_pointer_reference = ToolProgram.model_validate(
        {
            "operations": [
                {"kind": "call", "id": "seed", "tool": "read_metric"},
                {
                    "kind": "call",
                    "id": "bad_pointer",
                    "tool": "read_metric",
                    "arguments": {"name": {"$ref": "seed#/bad~2"}},
                },
            ],
            "outputs": [{"name": "value", "source": {"step": "bad_pointer"}}],
        }
    )
    invalid_pointer_result = _run(
        execute_tool_program(
            invalid_pointer_reference,
            _capabilities(),
            invoke=never_invoke,
        )
    )
    assert invalid_pointer_result["status"] == "rejected"
    assert invalid_pointer_result["error"]["code"] == "invalid_reference"
    assert invoked == 0

    forward = ToolProgram.model_validate(
        {
            "operations": [
                {
                    "kind": "call",
                    "id": "first",
                    "tool": "read_metric",
                    "arguments": {"name": {"$ref": "later#/value"}},
                },
                {"kind": "call", "id": "later", "tool": "read_metric"},
            ],
            "outputs": [{"name": "value", "source": {"step": "first"}}],
        }
    )
    forward_result = _run(execute_tool_program(forward, _capabilities(), invoke=never_invoke))
    assert forward_result["status"] == "rejected"
    assert forward_result["error"]["code"] == "invalid_reference"
    assert invoked == 0

    oversized = ToolProgram.model_validate(
        {
            "operations": [{"kind": "call", "id": "large", "tool": "read_metric"}],
            "outputs": [{"name": "large", "source": {"step": "large"}}],
        }
    )
    oversized_result = _run(
        execute_tool_program(
            oversized,
            _capabilities(),
            invoke=never_invoke,
            limits=ToolProgramLimits(max_result_bytes=128, max_state_bytes=256),
        )
    )
    assert invoked == 1
    assert oversized_result["status"] == "failed"
    assert oversized_result["receipts"][0]["error_code"] == "result_too_large"
    assert oversized_result["outputs"] == {}
    assert "xxxxx" not in json.dumps(oversized_result)

    async def oversized_json_text(capability, arguments, step_id):
        _ = capability, arguments, step_id
        return json.dumps({"payload": "y" * 5000})

    oversized_text_result = _run(
        execute_tool_program(
            oversized,
            _capabilities(),
            invoke=oversized_json_text,
            limits=ToolProgramLimits(max_result_bytes=128, max_state_bytes=256),
        )
    )
    assert oversized_text_result["status"] == "failed"
    assert oversized_text_result["receipts"][0]["error_code"] == "result_too_large"
    assert "yyyyy" not in json.dumps(oversized_text_result)


def test_program_does_not_swallow_cancellation():
    async def cancel(capability, arguments, step_id):
        _ = capability, arguments, step_id
        raise asyncio.CancelledError

    program = ToolProgram.model_validate(
        {
            "operations": [{"kind": "call", "id": "cancelled", "tool": "read_metric"}],
            "outputs": [{"name": "value", "source": {"step": "cancelled"}}],
        }
    )

    with pytest.raises(asyncio.CancelledError):
        _run(execute_tool_program(program, _capabilities(), invoke=cancel))


@pytest.mark.parametrize(
    "operation,output",
    [
        (
            {
                "kind": "call",
                "id": "step",
                "tool": "read_metric",
                "depends_on": ["not a valid id"],
            },
            {"name": "value", "source": {"step": "step"}},
        ),
        (
            {"kind": "call", "id": "step", "tool": "x" * 129},
            {"name": "value", "source": {"step": "step"}},
        ),
        (
            {"kind": "call", "id": "step", "tool": "read_metric"},
            {"name": "value", "source": {"step": "step", "pointer": "/bad~2"}},
        ),
    ],
)
def test_program_schema_bounds_identifiers_and_json_pointers(operation, output):
    with pytest.raises(ValueError):
        ToolProgram.model_validate({"operations": [operation], "outputs": [output]})
