"""Bounded typed tool programs for efficient multi-tool coordinator steps.

This is intentionally a declarative program, not an arbitrary-code runtime.
Every callable is an already-registered LangChain tool with explicit plugin
policy, and every intermediate value remains inside this host-owned executor
unless the program selects it as an output.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Annotated, Any, Literal, TypeAlias

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ultra_deepagents.harness_plugins import BoundProgramTool, ToolConcurrency

_STEP_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_OUTPUT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_REFERENCE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]{0,63})#(.*)$")
_MAX_JSON_DEPTH = 32
_MAX_ERROR_TEXT = 240
_MAX_POINTER_LENGTH = 2048


def _validated_pointer(value: str) -> str:
    if value and not value.startswith("/"):
        raise ValueError("JSON pointer must be empty or start with '/'")
    if len(value) > _MAX_POINTER_LENGTH:
        raise ValueError("JSON pointer is too long")
    if re.search(r"~(?![01])", value):
        raise ValueError("JSON pointer contains an invalid escape")
    return value


def _validated_dependencies(values: list[str]) -> list[str]:
    if len(values) > 64 or any(not _STEP_ID_RE.fullmatch(value) for value in values):
        raise ValueError("invalid program dependency")
    return values


class _ProgramModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProgramReference(_ProgramModel):
    step: str
    pointer: str = ""

    @field_validator("step")
    @classmethod
    def _valid_step(cls, value: str) -> str:
        if not _STEP_ID_RE.fullmatch(value):
            raise ValueError("invalid program step id")
        return value

    @field_validator("pointer")
    @classmethod
    def _valid_pointer(cls, value: str) -> str:
        return _validated_pointer(value)


class ProgramCondition(_ProgramModel):
    source: ProgramReference
    operator: Literal["truthy", "falsy", "eq", "ne", "exists", "not_exists"]
    value: Any = None


class ProgramPredicate(_ProgramModel):
    pointer: str = ""
    operator: Literal["eq", "ne", "lt", "lte", "gt", "gte", "contains"]
    value: Any

    @field_validator("pointer")
    @classmethod
    def _valid_pointer(cls, value: str) -> str:
        return _validated_pointer(value)


class ToolCallOperation(_ProgramModel):
    kind: Literal["call"] = "call"
    id: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    when: ProgramCondition | None = None

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if not _STEP_ID_RE.fullmatch(value):
            raise ValueError("invalid program step id")
        return value

    @field_validator("tool")
    @classmethod
    def _valid_tool(cls, value: str) -> str:
        value = value.strip()
        if not value or len(value) > 128:
            raise ValueError("tool name is required")
        return value

    @field_validator("depends_on")
    @classmethod
    def _valid_dependencies(cls, value: list[str]) -> list[str]:
        return _validated_dependencies(value)


class TransformOperation(_ProgramModel):
    kind: Literal["transform"] = "transform"
    id: str
    operation: Literal["select", "filter", "aggregate"]
    source: ProgramReference
    where: ProgramPredicate | None = None
    aggregate: Literal["count", "sum", "min", "max", "mean"] | None = None
    value_pointer: str = ""
    depends_on: list[str] = Field(default_factory=list)
    when: ProgramCondition | None = None

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if not _STEP_ID_RE.fullmatch(value):
            raise ValueError("invalid program step id")
        return value

    @field_validator("value_pointer")
    @classmethod
    def _valid_value_pointer(cls, value: str) -> str:
        return _validated_pointer(value)

    @field_validator("depends_on")
    @classmethod
    def _valid_dependencies(cls, value: list[str]) -> list[str]:
        return _validated_dependencies(value)

    @model_validator(mode="after")
    def _operation_fields_match(self) -> TransformOperation:
        if self.operation == "filter" and self.where is None:
            raise ValueError("filter transform requires where")
        if self.operation == "aggregate" and self.aggregate is None:
            raise ValueError("aggregate transform requires aggregate")
        if self.operation != "filter" and self.where is not None:
            raise ValueError("where is only valid for filter transforms")
        if self.operation != "aggregate" and self.aggregate is not None:
            raise ValueError("aggregate is only valid for aggregate transforms")
        return self


ProgramOperation: TypeAlias = Annotated[
    ToolCallOperation | TransformOperation,
    Field(discriminator="kind"),
]


class ProgramOutput(_ProgramModel):
    name: str
    source: ProgramReference

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _OUTPUT_NAME_RE.fullmatch(value):
            raise ValueError("invalid program output name")
        return value


class ToolProgram(_ProgramModel):
    operations: list[ProgramOperation] = Field(min_length=1, max_length=64)
    outputs: list[ProgramOutput] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def _unique_names(self) -> ToolProgram:
        operation_ids = [operation.id for operation in self.operations]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("program operation ids must be unique")
        output_names = [output.name for output in self.outputs]
        if len(output_names) != len(set(output_names)):
            raise ValueError("program output names must be unique")
        return self


@dataclass(frozen=True, slots=True)
class ToolProgramLimits:
    max_operations: int = 16
    max_concurrency: int = 4
    max_argument_bytes: int = 32 * 1024
    max_result_bytes: int = 256 * 1024
    max_state_bytes: int = 1024 * 1024
    max_output_bytes: int = 256 * 1024
    max_collection_items: int = 10_000
    call_timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        for name in (
            "max_operations",
            "max_concurrency",
            "max_argument_bytes",
            "max_result_bytes",
            "max_state_bytes",
            "max_output_bytes",
            "max_collection_items",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.call_timeout_seconds <= 0:
            raise ValueError("call_timeout_seconds must be positive")


ProgramInvoker: TypeAlias = Callable[
    [BoundProgramTool, dict[str, Any], str],
    Awaitable[Any],
]


@dataclass(frozen=True, slots=True)
class _StepResult:
    value: Any | None
    receipt: dict[str, Any]
    encoded_bytes: int = 0


class _ProgramError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


async def execute_tool_program(
    program: ToolProgram,
    capabilities: Sequence[BoundProgramTool],
    *,
    invoke: ProgramInvoker,
    limits: ToolProgramLimits | None = None,
) -> dict[str, Any]:
    """Validate and execute a tool program without exposing intermediates."""

    resolved_limits = limits or ToolProgramLimits()
    by_name = {capability.name: capability for capability in capabilities}
    if len(by_name) != len(capabilities):
        return _rejected("duplicate_capability")
    rejection = _validate_program(program, by_name, resolved_limits)
    if rejection is not None:
        return rejection

    state: dict[str, Any] = {}
    statuses: dict[str, str] = {}
    receipts: list[dict[str, Any]] = []
    state_bytes = 0
    operations = program.operations
    cursor = 0

    while cursor < len(operations):
        operation = operations[cursor]
        if isinstance(operation, ToolCallOperation):
            capability = by_name[operation.tool]
            if capability.concurrency is ToolConcurrency.PARALLEL:
                batch: list[ToolCallOperation] = []
                batch_ids: set[str] = set()
                scan = cursor
                while scan < len(operations):
                    candidate = operations[scan]
                    if not isinstance(candidate, ToolCallOperation):
                        break
                    candidate_capability = by_name[candidate.tool]
                    if candidate_capability.concurrency is not ToolConcurrency.PARALLEL:
                        break
                    if _operation_references(candidate) & batch_ids:
                        break
                    batch.append(candidate)
                    batch_ids.add(candidate.id)
                    scan += 1

                semaphore = asyncio.Semaphore(resolved_limits.max_concurrency)

                async def run_parallel(item: ToolCallOperation) -> _StepResult:
                    async with semaphore:
                        return await _execute_call(
                            item,
                            by_name[item.tool],
                            state,
                            statuses,
                            invoke,
                            resolved_limits,
                        )

                results = await asyncio.gather(*(run_parallel(item) for item in batch))
                for item, step_result in zip(batch, results, strict=True):
                    state_bytes = _record_step(
                        item.id,
                        step_result,
                        state=state,
                        statuses=statuses,
                        receipts=receipts,
                        state_bytes=state_bytes,
                        limits=resolved_limits,
                    )
                cursor = scan
                continue

            result = await _execute_call(
                operation,
                capability,
                state,
                statuses,
                invoke,
                resolved_limits,
            )
        else:
            result = _execute_transform(operation, state, statuses, resolved_limits)

        state_bytes = _record_step(
            operation.id,
            result,
            state=state,
            statuses=statuses,
            receipts=receipts,
            state_bytes=state_bytes,
            limits=resolved_limits,
        )
        cursor += 1

    outputs: dict[str, Any] = {}
    output_errors: dict[str, dict[str, str]] = {}
    for output in program.outputs:
        if statuses.get(output.source.step) != "succeeded":
            output_errors[output.name] = {"code": "source_unavailable"}
            continue
        try:
            outputs[output.name] = _resolve_pointer(
                state[output.source.step],
                output.source.pointer,
            )
        except _ProgramError:
            output_errors[output.name] = {"code": "pointer_not_found"}

    failed = any(status == "failed" for status in statuses.values()) or bool(output_errors)
    succeeded = any(status == "succeeded" for status in statuses.values())
    result_status = "succeeded"
    if failed:
        result_status = "partial" if succeeded and outputs else "failed"
    program_result: dict[str, Any] = {
        "version": 1,
        "status": result_status,
        "outputs": outputs,
        "receipts": receipts,
    }
    if output_errors:
        program_result["output_errors"] = output_errors
    if _json_size(program_result) > resolved_limits.max_output_bytes:
        return {
            "version": 1,
            "status": "failed",
            "outputs": {},
            "receipts": receipts,
            "error": {"code": "selected_output_too_large"},
        }
    return program_result


def build_tool_program_tool(
    capabilities: Sequence[BoundProgramTool],
    *,
    limits: ToolProgramLimits | None = None,
) -> Any:
    """Build the single model-visible entry point for a frozen capability set."""

    frozen_capabilities = tuple(capabilities)
    resolved_limits = limits or ToolProgramLimits()
    for capability in frozen_capabilities:
        _validate_tool_injections(capability)

    async def run_tool_program_async(
        runtime: ToolRuntime[Any],
        operations: list[ProgramOperation],
        outputs: list[ProgramOutput],
    ) -> str:
        program = ToolProgram(operations=operations, outputs=outputs)

        async def invoke(
            capability: BoundProgramTool,
            arguments: dict[str, Any],
            step_id: str,
        ) -> Any:
            parent_id = str(runtime.tool_call_id or "tool-program")
            child_id = f"{parent_id}:{step_id}"
            child_runtime = replace(runtime, tool_call_id=child_id)
            child_arguments = dict(arguments)
            hidden = _hidden_tool_fields(capability.tool)
            if "runtime" in hidden:
                child_arguments["runtime"] = child_runtime
            child_call = {
                "name": capability.name,
                "args": child_arguments,
                "id": child_id,
                "type": "tool_call",
            }
            return await capability.tool.ainvoke(
                child_call,
                config=child_runtime.config,
            )

        result = await execute_tool_program(
            program,
            frozen_capabilities,
            invoke=invoke,
            limits=resolved_limits,
        )
        return json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def run_tool_program_sync(
        runtime: ToolRuntime[Any],
        operations: list[ProgramOperation],
        outputs: list[ProgramOutput],
    ) -> str:
        return asyncio.run(run_tool_program_async(runtime, operations, outputs))

    return StructuredTool.from_function(
        func=run_tool_program_sync,
        coroutine=run_tool_program_async,
        name="run_tool_program",
        description=(
            "Run a bounded typed multi-tool program. Independent parallel-safe calls "
            "are concurrent and exclusive calls are ordering barriers. Arguments may "
            'reference an earlier result with {"$ref":"step_id#/json/pointer"}. Use '
            "select/filter/aggregate transforms and conditional `when` branches. Only "
            "requested outputs return to model context; all other intermediates are "
            "discarded after this call."
        ),
        handle_validation_error="Invalid tool program schema.",
    )


def build_tool_program_prompt(program_sdk: str) -> str:
    """Render the deterministic, cacheable model projection of program tools."""

    return (
        "Typed tool-program SDK (deterministic, host-authorized):\n"
        "Use run_tool_program when two or more registered calls can be composed. "
        "The harness may parallelize only tools marked parallel; tools marked exclusive "
        "are serialized barriers. Use earlier-result references, conditional `when` "
        "branches, and select/filter/aggregate transforms to keep intermediate data out "
        "of model context. Request only the final values needed for reasoning. A failed "
        "step never grants fallback access to an undeclared tool.\n"
        f"SDK_JSON={program_sdk}"
    )


def _validate_program(
    program: ToolProgram,
    capabilities: Mapping[str, BoundProgramTool],
    limits: ToolProgramLimits,
) -> dict[str, Any] | None:
    if len(program.operations) > limits.max_operations:
        return _rejected("operation_limit_exceeded")

    known: set[str] = set()
    for operation in program.operations:
        if isinstance(operation, ToolCallOperation) and operation.tool not in capabilities:
            return _rejected("tool_not_allowed")
        if len(operation.depends_on) != len(set(operation.depends_on)):
            return _rejected("duplicate_dependency")
        try:
            references = _operation_references(operation)
            if not set(operation.depends_on).issubset(known) or not references.issubset(known):
                return _rejected("invalid_reference")
            if isinstance(operation, ToolCallOperation):
                _validate_json_tree(operation.arguments)
                if _json_size(operation.arguments) > limits.max_argument_bytes:
                    return _rejected("argument_too_large")
            if operation.when is not None:
                _validate_json_tree(operation.when.value)
            if isinstance(operation, TransformOperation) and operation.where is not None:
                _validate_json_tree(operation.where.value)
            if _json_size(operation.model_dump()) > limits.max_argument_bytes:
                return _rejected("operation_too_large")
        except _ProgramError as exc:
            return _rejected(exc.code)
        known.add(operation.id)

    if any(output.source.step not in known for output in program.outputs):
        return _rejected("invalid_reference")
    if _json_size([output.model_dump() for output in program.outputs]) > limits.max_argument_bytes:
        return _rejected("output_spec_too_large")
    return None


async def _execute_call(
    operation: ToolCallOperation,
    capability: BoundProgramTool,
    state: Mapping[str, Any],
    statuses: Mapping[str, str],
    invoke: ProgramInvoker,
    limits: ToolProgramLimits,
) -> _StepResult:
    precondition = _precondition_receipt(operation, state, statuses)
    if precondition is not None:
        return precondition
    started = time.monotonic()
    try:
        arguments = _resolve_references(operation.arguments, state, statuses)
        if not isinstance(arguments, dict):
            raise _ProgramError("invalid_arguments")
        if _json_size(arguments) > limits.max_argument_bytes:
            raise _ProgramError("argument_too_large")
        if capability.concurrency is ToolConcurrency.EXCLUSIVE:
            # Most LangChain sync tools run in a worker thread that Python
            # cannot safely kill. Let the outer run watchdog/cancellation own
            # an exclusive call so a timed-out thread can never overlap the
            # next mutation behind the barrier.
            raw = await invoke(capability, arguments, operation.id)
        else:
            async with asyncio.timeout(limits.call_timeout_seconds):
                raw = await invoke(capability, arguments, operation.id)
        value = _normalize_tool_result(raw, limits)
        encoded_bytes = _bounded_result_size(value, limits)
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        return _failed_receipt(operation, "tool_timeout", started)
    except _ProgramError as exc:
        return _failed_receipt(operation, exc.code, started)
    except Exception:  # noqa: BLE001 - tool exceptions are intentionally redacted
        return _failed_receipt(operation, "tool_failed", started)
    return _StepResult(
        value=value,
        encoded_bytes=encoded_bytes,
        receipt={
            "id": operation.id,
            "kind": "call",
            "tool": capability.name,
            "concurrency": capability.concurrency.value,
            "status": "succeeded",
            "result_bytes": encoded_bytes,
            "duration_ms": _duration_ms(started),
        },
    )


def _execute_transform(
    operation: TransformOperation,
    state: Mapping[str, Any],
    statuses: Mapping[str, str],
    limits: ToolProgramLimits,
) -> _StepResult:
    precondition = _precondition_receipt(operation, state, statuses)
    if precondition is not None:
        return precondition
    started = time.monotonic()
    try:
        source = _resolve_reference(operation.source, state, statuses)
        if operation.operation == "select":
            value = source
        elif operation.operation == "filter":
            value = _filter_values(source, operation.where, limits)
        else:
            value = _aggregate_values(
                source,
                operation.aggregate,
                operation.value_pointer,
                limits,
            )
        value = _canonical_json_value(value)
        encoded_bytes = _bounded_result_size(value, limits)
    except _ProgramError as exc:
        return _failed_receipt(operation, exc.code, started)
    return _StepResult(
        value=value,
        encoded_bytes=encoded_bytes,
        receipt={
            "id": operation.id,
            "kind": "transform",
            "operation": operation.operation,
            "status": "succeeded",
            "result_bytes": encoded_bytes,
            "duration_ms": _duration_ms(started),
        },
    )


def _record_step(
    step_id: str,
    result: _StepResult,
    *,
    state: dict[str, Any],
    statuses: dict[str, str],
    receipts: list[dict[str, Any]],
    state_bytes: int,
    limits: ToolProgramLimits,
) -> int:
    status = str(result.receipt["status"])
    if status == "succeeded" and state_bytes + result.encoded_bytes > limits.max_state_bytes:
        statuses[step_id] = "failed"
        receipts.append(
            {
                "id": step_id,
                "kind": result.receipt["kind"],
                "status": "failed",
                "error_code": "state_limit_exceeded",
            }
        )
        return state_bytes
    statuses[step_id] = status
    receipts.append(result.receipt)
    if status == "succeeded":
        state[step_id] = result.value
        return state_bytes + result.encoded_bytes
    return state_bytes


def _precondition_receipt(
    operation: ToolCallOperation | TransformOperation,
    state: Mapping[str, Any],
    statuses: Mapping[str, str],
) -> _StepResult | None:
    referenced = _operation_references(operation)
    if any(statuses.get(step) != "succeeded" for step in referenced):
        return _skipped_receipt(operation, "dependency_unavailable")
    if operation.when is None:
        return None
    try:
        should_run = _condition_matches(operation.when, state, statuses)
    except _ProgramError:
        return _skipped_receipt(operation, "condition_unavailable")
    if not should_run:
        return _skipped_receipt(operation, "condition_false")
    return None


def _condition_matches(
    condition: ProgramCondition,
    state: Mapping[str, Any],
    statuses: Mapping[str, str],
) -> bool:
    source = state.get(condition.source.step)
    if statuses.get(condition.source.step) != "succeeded":
        raise _ProgramError("condition_unavailable")
    try:
        value = _resolve_pointer(source, condition.source.pointer)
        exists = True
    except _ProgramError:
        value = None
        exists = False
    if condition.operator == "exists":
        return exists
    if condition.operator == "not_exists":
        return not exists
    if not exists:
        return False
    if condition.operator == "truthy":
        return bool(value)
    if condition.operator == "falsy":
        return not bool(value)
    if condition.operator == "eq":
        return bool(value == condition.value)
    return bool(value != condition.value)


def _filter_values(
    source: Any,
    predicate: ProgramPredicate | None,
    limits: ToolProgramLimits,
) -> list[Any]:
    if not isinstance(source, list) or predicate is None:
        raise _ProgramError("filter_requires_array")
    if len(source) > limits.max_collection_items:
        raise _ProgramError("collection_limit_exceeded")
    selected: list[Any] = []
    for item in source:
        try:
            candidate = _resolve_pointer(item, predicate.pointer)
        except _ProgramError:
            continue
        if _predicate_matches(candidate, predicate.operator, predicate.value):
            selected.append(item)
    return selected


def _predicate_matches(value: Any, operator: str, expected: Any) -> bool:
    if operator == "eq":
        return bool(value == expected)
    if operator == "ne":
        return bool(value != expected)
    if operator == "contains":
        try:
            return bool(expected in value)
        except TypeError:
            return False
    try:
        if operator == "lt":
            return bool(value < expected)
        if operator == "lte":
            return bool(value <= expected)
        if operator == "gt":
            return bool(value > expected)
        return bool(value >= expected)
    except TypeError:
        return False


def _aggregate_values(
    source: Any,
    operation: str | None,
    pointer: str,
    limits: ToolProgramLimits,
) -> dict[str, Any]:
    if not isinstance(source, list) or operation is None:
        raise _ProgramError("aggregate_requires_array")
    if len(source) > limits.max_collection_items:
        raise _ProgramError("collection_limit_exceeded")
    if operation == "count":
        return {"operation": "count", "count": len(source), "value": len(source)}
    values: list[float | int] = []
    for item in source:
        try:
            value = _resolve_pointer(item, pointer)
        except _ProgramError as exc:
            raise _ProgramError("aggregate_value_missing") from exc
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise _ProgramError("aggregate_requires_numbers")
        try:
            finite = math.isfinite(float(value))
        except OverflowError as exc:
            raise _ProgramError("aggregate_requires_finite_numbers") from exc
        if not finite:
            raise _ProgramError("aggregate_requires_finite_numbers")
        values.append(value)
    if not values:
        raise _ProgramError("aggregate_requires_values")
    try:
        if operation == "sum":
            result: float | int = sum(values)
        elif operation == "min":
            result = min(values)
        elif operation == "max":
            result = max(values)
        else:
            result = sum(values) / len(values)
    except OverflowError as exc:
        raise _ProgramError("aggregate_overflow") from exc
    return {"operation": operation, "count": len(values), "value": result}


def _operation_references(operation: ToolCallOperation | TransformOperation) -> set[str]:
    references = set(operation.depends_on)
    if operation.when is not None:
        references.add(operation.when.source.step)
    if isinstance(operation, ToolCallOperation):
        references.update(reference.step for reference in _argument_references(operation.arguments))
    else:
        references.add(operation.source.step)
    return references


def _argument_references(value: Any, *, depth: int = 0) -> list[ProgramReference]:
    if depth > _MAX_JSON_DEPTH:
        raise _ProgramError("json_depth_exceeded")
    if isinstance(value, Mapping):
        if "$ref" in value:
            if set(value) != {"$ref"} or not isinstance(value["$ref"], str):
                raise _ProgramError("invalid_reference")
            match = _REFERENCE_RE.fullmatch(value["$ref"])
            if match is None:
                raise _ProgramError("invalid_reference")
            pointer = match.group(2)
            if pointer and not pointer.startswith("/"):
                raise _ProgramError("invalid_reference")
            try:
                reference = ProgramReference(step=match.group(1), pointer=pointer)
            except ValueError as exc:
                raise _ProgramError("invalid_reference") from exc
            return [reference]
        references: list[ProgramReference] = []
        for child in value.values():
            references.extend(_argument_references(child, depth=depth + 1))
        return references
    if isinstance(value, list | tuple):
        references = []
        for child in value:
            references.extend(_argument_references(child, depth=depth + 1))
        return references
    return []


def _resolve_references(
    value: Any,
    state: Mapping[str, Any],
    statuses: Mapping[str, str],
    *,
    depth: int = 0,
) -> Any:
    if depth > _MAX_JSON_DEPTH:
        raise _ProgramError("json_depth_exceeded")
    if isinstance(value, Mapping):
        references = _argument_references(value, depth=depth)
        if "$ref" in value:
            return _resolve_reference(references[0], state, statuses)
        return {
            str(key): _resolve_references(child, state, statuses, depth=depth + 1)
            for key, child in value.items()
        }
    if isinstance(value, list | tuple):
        return [_resolve_references(child, state, statuses, depth=depth + 1) for child in value]
    return value


def _resolve_reference(
    reference: ProgramReference,
    state: Mapping[str, Any],
    statuses: Mapping[str, str],
) -> Any:
    if statuses.get(reference.step) != "succeeded" or reference.step not in state:
        raise _ProgramError("source_unavailable")
    return _resolve_pointer(state[reference.step], reference.pointer)


def _resolve_pointer(value: Any, pointer: str) -> Any:
    if not pointer:
        return value
    if not pointer.startswith("/"):
        raise _ProgramError("invalid_pointer")
    current = value
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if token not in current:
                raise _ProgramError("pointer_not_found")
            current = current[token]
            continue
        if isinstance(current, list):
            if not token.isdigit():
                raise _ProgramError("pointer_not_found")
            index = int(token)
            if index >= len(current):
                raise _ProgramError("pointer_not_found")
            current = current[index]
            continue
        raise _ProgramError("pointer_not_found")
    return current


def _normalize_tool_result(value: Any, limits: ToolProgramLimits) -> Any:
    if isinstance(value, ToolMessage):
        if str(getattr(value, "status", "success")) != "success":
            raise _ProgramError("tool_reported_error")
        value = value.content
    if isinstance(value, str):
        if len(value.encode("utf-8")) > limits.max_result_bytes:
            raise _ProgramError("result_too_large")
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                value = json.loads(stripped)
            except (json.JSONDecodeError, RecursionError):
                value = {"text": value}
        else:
            value = {"text": value}
    return _canonical_json_value(value)


def _canonical_json_value(value: Any) -> Any:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return json.loads(encoded)
    except (RecursionError, TypeError, ValueError) as exc:
        raise _ProgramError("non_json_result") from exc


def _validate_json_tree(value: Any, *, depth: int = 0) -> None:
    if depth > _MAX_JSON_DEPTH:
        raise _ProgramError("json_depth_exceeded")
    if value is None or isinstance(value, bool | int | str):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _ProgramError("non_finite_json")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise _ProgramError("non_json_argument")
            _validate_json_tree(child, depth=depth + 1)
        return
    if isinstance(value, list | tuple):
        for child in value:
            _validate_json_tree(child, depth=depth + 1)
        return
    raise _ProgramError("non_json_argument")


def _bounded_result_size(value: Any, limits: ToolProgramLimits) -> int:
    size = _json_size(value)
    if size > limits.max_result_bytes:
        raise _ProgramError("result_too_large")
    return size


def _json_size(value: Any) -> int:
    try:
        return len(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
    except (RecursionError, TypeError, ValueError) as exc:
        raise _ProgramError("non_json_value") from exc


def _validate_tool_injections(capability: BoundProgramTool) -> None:
    hidden = _hidden_tool_fields(capability.tool)
    unsupported = hidden - {"runtime", "tool_call_id"}
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(
            f"program tool {capability.name!r} has unsupported injected fields: {names}"
        )


def _hidden_tool_fields(tool_object: Any) -> set[str]:
    args_schema = getattr(tool_object, "args_schema", None)
    tool_call_schema = getattr(tool_object, "tool_call_schema", None)
    all_fields = set(getattr(args_schema, "model_fields", {}) or {})
    visible_fields = set(getattr(tool_call_schema, "model_fields", {}) or {})
    return all_fields - visible_fields


def _failed_receipt(
    operation: ToolCallOperation | TransformOperation,
    code: str,
    started: float,
) -> _StepResult:
    return _StepResult(
        value=None,
        receipt={
            "id": operation.id,
            "kind": operation.kind,
            "status": "failed",
            "error_code": str(code)[:_MAX_ERROR_TEXT],
            "duration_ms": _duration_ms(started),
        },
    )


def _skipped_receipt(
    operation: ToolCallOperation | TransformOperation,
    code: str,
) -> _StepResult:
    return _StepResult(
        value=None,
        receipt={
            "id": operation.id,
            "kind": operation.kind,
            "status": "skipped",
            "reason_code": code,
        },
    )


def _duration_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))


def _rejected(code: str) -> dict[str, Any]:
    return {
        "version": 1,
        "status": "rejected",
        "outputs": {},
        "receipts": [],
        "error": {"code": code},
    }
