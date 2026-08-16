"""Content-free, opt-in metadata for the local Trace Lens tool.

The recorder observes LangChain callback inputs, retains only bounded structural
metadata, and lets the runner attach that metadata to an already-existing token
usage event.  It never publishes an event and never retains prompt, response,
schema, tool-description, URL, header, or credential values.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, cast
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler

TRACE_LENS_METADATA_SCHEMA_VERSION = 2
TRACE_LENS_INPUT_PHASE = "logical_model_invocation_start"
TRACE_LENS_MAX_PENDING_CALLS = 512
TRACE_LENS_MAX_SERIALIZED_BYTES = 4096

_MAX_COUNT = 1_000_000_000
_MAX_IDENTIFIER_LENGTH = 96
_MAX_FUNCTION_NAMES = 32
_MAX_DOMAIN_TOOL_NAMES = 64
_MAX_SUBAGENT_NAMES = 32
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,95}$")
_SAFE_MODEL_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,63}(?:/[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,63})?$"
)
_COORDINATOR_NODES = {"agent", "coordinator", "main", "model", "ultra-research-agent"}
_COORDINATOR_NAMES = {"ultra-research-agent", "main-agent", "coordinator"}
_CODE_OWNED_SUBAGENT_NAMES = {
    "builder",
    "code-runner",
    "literature-reviewer",
    "qwen-code-runner",
    "vision-reasoner",
}
_NON_FUNCTION_CLASSES = ("custom", "web_search", "file_search", "code_interpreter", "other")

_MatchKey = tuple[str, str]


def _bounded_count(value: int) -> int:
    return min(_MAX_COUNT, max(0, int(value)))


def _safe_identifier(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = cast(str, value).strip()
    if len(text) > _MAX_IDENTIFIER_LENGTH or not _SAFE_IDENTIFIER_RE.fullmatch(text):
        return ""
    return text


def _safe_model_id(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = cast(str, value).strip()
    return text if len(text) <= 129 and _SAFE_MODEL_ID_RE.fullmatch(text) else ""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _message_value(message: Any, name: str) -> Any:
    if isinstance(message, Mapping):
        return message.get(name)
    return getattr(message, name, None)


def _message_role(message: Any) -> str:
    raw = _message_value(message, "role") or _message_value(message, "type")
    role = str(raw or "").strip().lower()
    return {
        "ai": "assistant",
        "assistant": "assistant",
        "human": "user",
        "system": "system",
        "tool": "tool",
        "user": "user",
    }.get(role, "unknown")


def _content_shape(content: Any) -> tuple[int, int]:
    """Return ``(block_count, text_char_count)`` without retaining content."""
    if isinstance(content, str):
        return (1 if content else 0), _bounded_count(len(content))
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        blocks = 0
        chars = 0
        for item in content:
            item_blocks, item_chars = _content_shape(item)
            blocks += item_blocks
            chars += item_chars
        return _bounded_count(blocks), _bounded_count(chars)
    if isinstance(content, Mapping):
        chars = 0
        for key in ("text", "content"):
            value = content.get(key)
            if isinstance(value, str):
                chars += len(value)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                _, nested_chars = _content_shape(value)
                chars += nested_chars
        return 1, _bounded_count(chars)
    return 0, 0


def _flatten_messages(messages: Any) -> list[Any]:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        return []
    flattened: list[Any] = []
    for item in messages:
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray, Mapping)):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def _scope_from_metadata(metadata: Mapping[str, Any]) -> tuple[str, str]:
    node = str(metadata.get("langgraph_node") or "").strip().lower()
    agent_name = str(metadata.get("lc_agent_name") or "").strip().lower()
    checkpoint_ns = str(
        metadata.get("langgraph_checkpoint_ns") or metadata.get("checkpoint_ns") or ""
    )
    is_coordinator = (
        (not node or node in _COORDINATOR_NODES)
        and (not agent_name or agent_name in _COORDINATOR_NAMES)
        and "|" not in checkpoint_ns
    )
    if is_coordinator:
        return "coordinator", "coordinator"
    if node or agent_name or checkpoint_ns:
        return "subagent", "subagent"
    return "unknown", "unknown"


def _is_summary_message(message: Any) -> bool:
    additional = _mapping(_message_value(message, "additional_kwargs"))
    return additional.get("lc_source") == "summarization"


def _has_reasoning_continuity(message: Any) -> bool:
    additional = _mapping(_message_value(message, "additional_kwargs"))
    reasoning = additional.get("reasoning_content")
    return isinstance(reasoning, str) and bool(reasoning)


def _tool_choice_mode(invocation_params: Mapping[str, Any]) -> str:
    if "tool_choice" not in invocation_params:
        return "unavailable"
    choice = invocation_params.get("tool_choice")
    if isinstance(choice, str) and choice in {"auto", "none", "required"}:
        return choice
    if isinstance(choice, Mapping):
        function = choice.get("function")
        if choice.get("type") == "function" and isinstance(function, Mapping):
            return "specific_function" if _safe_identifier(function.get("name")) else "unavailable"
    return "unavailable"


def _provider_bound_tools(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": TRACE_LENS_METADATA_SCHEMA_VERSION,
        "capture_scope": "logical_model_invocation",
        "content_captured": False,
        "availability": "unavailable",
        "function_tool_names": [],
        "non_function_classifications": {key: 0 for key in _NON_FUNCTION_CLASSES},
        "structured_response_tool_count": 0,
        "tool_choice_mode": "unavailable",
    }
    if "invocation_params" not in kwargs:
        return base
    invocation_params = kwargs.get("invocation_params")
    if not isinstance(invocation_params, Mapping):
        return {**base, "availability": "malformed"}
    base["tool_choice_mode"] = _tool_choice_mode(invocation_params)
    if "tools" not in invocation_params:
        return base
    tools = invocation_params.get("tools")
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        return {**base, "availability": "malformed"}

    function_names: list[str] = []
    omitted_function_names = 0
    function_tool_count = 0
    structured_response_tool_count = 0
    non_function_count = 0
    classifications = {key: 0 for key in _NON_FUNCTION_CLASSES}
    for tool in tools:
        if not isinstance(tool, Mapping):
            return {**base, "availability": "malformed"}
        tool_type = tool.get("type")
        if tool_type == "function":
            function_tool_count += 1
            function = tool.get("function")
            raw_name = function.get("name") if isinstance(function, Mapping) else ""
            if isinstance(raw_name, str) and raw_name.startswith("response_format_"):
                structured_response_tool_count += 1
                name = "structured_response"
            else:
                name = _safe_identifier(raw_name)
            if name and len(function_names) < _MAX_FUNCTION_NAMES:
                function_names.append(name)
            else:
                omitted_function_names += 1
            continue
        if not isinstance(tool_type, str):
            return {**base, "availability": "malformed"}
        non_function_count += 1
        classification = tool_type if tool_type in classifications else "other"
        classifications[classification] += 1

    return {
        **base,
        "availability": "observed",
        "tool_count": _bounded_count(len(tools)),
        "function_tool_count": _bounded_count(function_tool_count),
        "function_tool_names": function_names,
        "omitted_function_name_count": _bounded_count(omitted_function_names),
        "non_function_tool_count": _bounded_count(non_function_count),
        "structured_response_tool_count": _bounded_count(structured_response_tool_count),
        "non_function_classifications": {
            key: _bounded_count(value) for key, value in classifications.items()
        },
    }


def build_logical_model_input(
    messages: Any,
    metadata: Mapping[str, Any] | None,
    callback_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project a model callback into counts and fixed classifications only."""
    role_counts = {role: 0 for role in ("system", "user", "assistant", "tool", "unknown")}
    content_blocks = 0
    text_chars = 0
    summary_messages = 0
    reasoning_continuity_messages = 0
    flattened = _flatten_messages(messages)
    for message in flattened:
        role = _message_role(message)
        role_counts[role] += 1
        blocks, chars = _content_shape(_message_value(message, "content"))
        content_blocks += blocks
        text_chars += chars
        summary_messages += int(_is_summary_message(message))
        reasoning_continuity_messages += int(_has_reasoning_continuity(message))
    scope, agent_role = _scope_from_metadata(_mapping(metadata))
    return {
        "schema_version": TRACE_LENS_METADATA_SCHEMA_VERSION,
        "phase": TRACE_LENS_INPUT_PHASE,
        "capture_scope": "logical_model_invocation",
        "content_captured": False,
        "scope": scope,
        "agent_role": agent_role,
        "message_count": _bounded_count(len(flattened)),
        "role_counts": {key: _bounded_count(value) for key, value in role_counts.items()},
        "content_block_count": _bounded_count(content_blocks),
        "text_char_count": _bounded_count(text_chars),
        "system_message_count": _bounded_count(role_counts["system"]),
        "tool_result_message_count": _bounded_count(role_counts["tool"]),
        "summary_message_count": _bounded_count(summary_messages),
        "reasoning_continuity_message_count": _bounded_count(reasoning_continuity_messages),
        "provider_bound_tools": _provider_bound_tools(_mapping(callback_kwargs)),
    }


def _callback_match_key(run_id: UUID, metadata: Mapping[str, Any]) -> _MatchKey:
    checkpoint_ns = metadata.get("langgraph_checkpoint_ns") or metadata.get("checkpoint_ns")
    if isinstance(checkpoint_ns, str) and checkpoint_ns:
        return "checkpoint_namespace", checkpoint_ns
    return "callback_run_id", str(run_id)


def _stream_event_match_key(event: Mapping[str, Any]) -> _MatchKey | None:
    if event.get("method") == "messages":
        params = _mapping(event.get("params"))
        data = params.get("data")
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            chunk = data[0] if data else None
            metadata = _mapping(data[1] if len(data) > 1 else None)
            if isinstance(chunk, Mapping) and chunk.get("event") == "message-finish":
                checkpoint_ns = metadata.get("langgraph_checkpoint_ns") or metadata.get(
                    "checkpoint_ns"
                )
                if isinstance(checkpoint_ns, str) and checkpoint_ns:
                    return "checkpoint_namespace", checkpoint_ns
                return None
    if event.get("event") == "on_chat_model_end":
        run_id = event.get("run_id")
        if isinstance(run_id, UUID):
            return "callback_run_id", str(run_id)
        if isinstance(run_id, str) and run_id:
            return "callback_run_id", run_id
    return None


class TraceLensInputCallback(AsyncCallbackHandler):
    """Collect bounded input metadata for exact usage-event correlation."""

    def __init__(self) -> None:
        super().__init__()
        self._records: dict[_MatchKey, dict[str, Any]] = {}
        self._ambiguous: set[_MatchKey] = set()
        self._saturated = False

    def clear(self) -> None:
        """Release all retained metadata at attempt finalization."""
        self._records.clear()
        self._ambiguous.clear()

    def _saturate(self) -> None:
        self.clear()
        self._saturated = True

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del serialized, parent_run_id, tags
        if self._saturated:
            return
        try:
            key = _callback_match_key(run_id, _mapping(metadata))
            record = build_logical_model_input(messages, metadata, kwargs)
        except Exception:
            return
        if key in self._records or key in self._ambiguous:
            self._records.pop(key, None)
            self._ambiguous.add(key)
        else:
            self._records[key] = record
        if len(self._records) + len(self._ambiguous) > TRACE_LENS_MAX_PENDING_CALLS:
            self._saturate()

    def take_exact_match(
        self, event: Mapping[str, Any], *, logical_call_id: str
    ) -> dict[str, Any] | None:
        if self._saturated:
            return None
        key = _stream_event_match_key(event)
        if key is None or key in self._ambiguous:
            return None
        record = self._records.pop(key, None)
        if record is None:
            return None
        return {**record, "logical_call_id": logical_call_id}


def _surface_name(value: Any) -> str:
    if isinstance(value, Mapping):
        return _safe_identifier(value.get("name"))
    return _safe_identifier(getattr(value, "name", None))


def build_agent_configuration(
    *,
    model_id: str,
    provider_id: str,
    registered_tools: Sequence[Any],
    subagents: Sequence[Any],
    async_subagent_count: int,
    builder_enabled: bool,
    declared_memory_count: int,
    declared_policy_count: int,
    declared_skill_count: int,
) -> dict[str, Any]:
    """Build the pre-invocation declared configuration without prompt content."""
    tool_names = sorted({name for tool in registered_tools if (name := _surface_name(tool))})
    code_owned_names = sorted(
        {
            name
            for subagent in subagents
            if (name := _surface_name(subagent)) in _CODE_OWNED_SUBAGENT_NAMES
        }
    )
    configuration: dict[str, Any] = {
        "schema_version": TRACE_LENS_METADATA_SCHEMA_VERSION,
        "capture_scope": "configuration",
        "content_captured": False,
        "domain_tool_declaration_count": _bounded_count(len(registered_tools)),
        "domain_tool_names": tool_names[:_MAX_DOMAIN_TOOL_NAMES],
        "domain_tool_name_omitted_count": _bounded_count(
            len(registered_tools) - min(len(tool_names), _MAX_DOMAIN_TOOL_NAMES)
        ),
        "declared_local_subagent_count": _bounded_count(len(subagents)),
        "code_owned_subagent_names": code_owned_names[:_MAX_SUBAGENT_NAMES],
        "configured_async_subagent_count": _bounded_count(async_subagent_count),
        "builder_configured": bool(builder_enabled),
        "declared_memory_count": _bounded_count(declared_memory_count),
        "declared_policy_count": _bounded_count(declared_policy_count),
        "skill_source_count": _bounded_count(declared_skill_count),
    }
    if safe_model_id := _safe_model_id(model_id):
        configuration["model_id"] = safe_model_id
    if safe_provider_id := _safe_identifier(provider_id):
        configuration["provider_id"] = safe_provider_id
    return configuration


# Compatibility for the initial internal prototype; callers should use the
# configuration name because these values are declarations, not observations.
build_agent_surface = build_agent_configuration


def _publisher_serialized_size(value: Mapping[str, Any]) -> int:
    return len(json.dumps(value, default=str).encode("utf-8"))


def _compact_trace_names(trace_metadata: dict[str, Any]) -> dict[str, Any]:
    compacted = dict(trace_metadata)
    configuration = compacted.get("agent_configuration")
    if isinstance(configuration, Mapping):
        compact_configuration = dict(configuration)
        domain_names = compact_configuration.pop("domain_tool_names", None)
        if isinstance(domain_names, Sequence) and not isinstance(
            domain_names, (str, bytes, bytearray)
        ):
            compact_configuration["domain_tool_name_omitted_count"] = _bounded_count(
                int(compact_configuration.get("domain_tool_name_omitted_count") or 0)
                + len(domain_names)
            )
            compact_configuration["domain_tool_names_omitted_due_to_size"] = True
        subagent_names = compact_configuration.pop("code_owned_subagent_names", None)
        if isinstance(subagent_names, Sequence) and not isinstance(
            subagent_names, (str, bytes, bytearray)
        ):
            compact_configuration["code_owned_subagent_name_omitted_count"] = _bounded_count(
                int(compact_configuration.get("code_owned_subagent_name_omitted_count") or 0)
                + len(subagent_names)
            )
            compact_configuration["code_owned_subagent_names_omitted_due_to_size"] = True
        compacted["agent_configuration"] = compact_configuration

    logical_input = compacted.get("logical_model_input")
    if isinstance(logical_input, Mapping):
        compact_input = dict(logical_input)
        provider_tools = compact_input.get("provider_bound_tools")
        if isinstance(provider_tools, Mapping):
            compact_tools = dict(provider_tools)
            function_names = compact_tools.pop("function_tool_names", None)
            if isinstance(function_names, Sequence) and not isinstance(
                function_names, (str, bytes, bytearray)
            ):
                compact_tools["omitted_function_name_count"] = _bounded_count(
                    int(compact_tools.get("omitted_function_name_count") or 0) + len(function_names)
                )
                compact_tools["function_names_omitted_due_to_size"] = True
            compact_input["provider_bound_tools"] = compact_tools
        compacted["logical_model_input"] = compact_input
    return compacted


def attach_trace_lens_metadata(
    usage_event: dict[str, Any],
    *,
    logical_model_input: Mapping[str, Any],
    agent_configuration: Mapping[str, Any] | None = None,
) -> bool:
    """Attach bounded metadata to an unstamped event, failing closed on any error."""
    try:
        payload = usage_event.get("payload")
        if not isinstance(payload, dict) or "trace_lens" in payload:
            return False
        trace_metadata: dict[str, Any] = {
            "schema_version": TRACE_LENS_METADATA_SCHEMA_VERSION,
            "capture_scope": "attempt",
            "content_captured": False,
            "logical_model_input": dict(logical_model_input),
        }
        if agent_configuration:
            trace_metadata["agent_configuration"] = dict(agent_configuration)
        candidate = trace_metadata
        if _publisher_serialized_size(candidate) > TRACE_LENS_MAX_SERIALIZED_BYTES:
            candidate = _compact_trace_names(candidate)
        if _publisher_serialized_size(candidate) > TRACE_LENS_MAX_SERIALIZED_BYTES:
            candidate = dict(candidate)
            if "agent_configuration" in candidate:
                candidate.pop("agent_configuration")
                candidate["agent_configuration_omitted_due_to_size"] = True
        if _publisher_serialized_size(candidate) > TRACE_LENS_MAX_SERIALIZED_BYTES:
            return False
        payload["trace_lens"] = candidate
        return True
    except Exception:
        return False
