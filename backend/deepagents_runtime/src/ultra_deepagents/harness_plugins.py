"""Deterministic, fail-closed composition for Ultra harness capabilities.

The registry is deliberately run-local.  It does not mutate process globals and
does not grant execution authority: a plugin contributes already-constructed
tools, stable prompt sections, and (optionally) an explicit policy describing
which of those tools may participate in a typed tool program.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

_PLUGIN_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


class ToolConcurrency(StrEnum):
    """Host-owned scheduling class for a program-callable tool."""

    PARALLEL = "parallel"
    EXCLUSIVE = "exclusive"


@dataclass(frozen=True, slots=True)
class ProgramToolPolicy:
    """Authority for exposing one plugin tool to the program executor."""

    tool_name: str
    concurrency: ToolConcurrency = ToolConcurrency.PARALLEL
    result_schema: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.tool_name or "").strip()
        if not name:
            raise ValueError("program tool policy requires a tool name")
        object.__setattr__(self, "tool_name", name)
        try:
            concurrency = ToolConcurrency(self.concurrency)
        except ValueError as exc:
            raise ValueError(f"invalid tool concurrency for {name}") from exc
        object.__setattr__(self, "concurrency", concurrency)
        object.__setattr__(
            self,
            "result_schema",
            _canonical_json_object(self.result_schema, label=f"{name} result schema"),
        )


@dataclass(frozen=True, slots=True)
class HarnessPlugin:
    """One replaceable harness contribution with no hidden global state."""

    name: str
    version: str = "1"
    order: int = 100
    tools: tuple[Any, ...] = ()
    program_tools: tuple[ProgramToolPolicy, ...] = ()
    prompt_sections: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        if not _PLUGIN_NAME_RE.fullmatch(name):
            raise ValueError(f"invalid harness plugin name: {name!r}")
        version = str(self.version or "").strip()
        if not version:
            raise ValueError(f"harness plugin {name!r} requires a version")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "program_tools", tuple(self.program_tools))
        object.__setattr__(
            self,
            "prompt_sections",
            tuple(section.strip() for section in self.prompt_sections if section.strip()),
        )


@runtime_checkable
class HarnessPluginProvider(Protocol):
    """Extension seam for packages that construct a run-local plugin."""

    def build_harness_plugin(self) -> HarnessPlugin: ...


@dataclass(frozen=True, slots=True)
class BoundProgramTool:
    """A program policy bound to the exact registered tool object."""

    plugin_name: str
    tool: Any
    concurrency: ToolConcurrency
    input_schema: Mapping[str, Any]
    result_schema: Mapping[str, Any]

    @property
    def name(self) -> str:
        return _tool_name(self.tool)


@dataclass(frozen=True, slots=True)
class HarnessSurface:
    """Frozen, deterministic model/tool surface for one agent construction."""

    tools: tuple[Any, ...]
    program_tools: tuple[BoundProgramTool, ...]
    prompt_sections: tuple[str, ...]
    plugin_manifest: tuple[dict[str, str], ...]
    program_sdk: str


class HarnessPluginRegistry:
    """Compose plugins deterministically and reject ambiguous authority."""

    def __init__(self, plugins: Iterable[HarnessPlugin] = ()) -> None:
        self._plugins: dict[str, HarnessPlugin] = {}
        self._tool_owners: dict[str, str] = {}
        self._program_owners: dict[str, str] = {}
        for plugin in plugins:
            self.register(plugin)

    def register_provider(self, provider: HarnessPluginProvider) -> None:
        self.register(provider.build_harness_plugin())

    def register(self, plugin: HarnessPlugin) -> None:
        if plugin.name in self._plugins:
            raise ValueError(f"duplicate harness plugin name: {plugin.name}")

        local_tools: dict[str, Any] = {}
        for tool in plugin.tools:
            name = _tool_name(tool)
            if not name:
                # Preserve compatibility with callers that pass opaque tool-like
                # objects to Deep Agents tests. Such objects can never receive
                # program authority because they have no stable name.
                continue
            if name in local_tools:
                raise ValueError(f"duplicate tool name in plugin {plugin.name}: {name}")
            owner = self._tool_owners.get(name)
            if owner is not None:
                raise ValueError(
                    f"duplicate tool name {name!r} in plugins {owner!r} and {plugin.name!r}"
                )
            local_tools[name] = tool

        local_program_names: set[str] = set()
        for policy in plugin.program_tools:
            name = policy.tool_name
            if name not in local_tools:
                raise ValueError(
                    f"program tool {name!r} is not registered by plugin {plugin.name!r}"
                )
            if name in local_program_names or name in self._program_owners:
                raise ValueError(f"duplicate program tool policy for {name!r}")
            local_program_names.add(name)

        self._plugins[plugin.name] = plugin
        self._tool_owners.update({name: plugin.name for name in local_tools})
        self._program_owners.update({name: plugin.name for name in local_program_names})

    def freeze(self) -> HarnessSurface:
        ordered_plugins = sorted(
            self._plugins.values(),
            key=lambda plugin: (plugin.order, plugin.name),
        )
        tools = tuple(tool for plugin in ordered_plugins for tool in plugin.tools)
        prompt_sections = tuple(
            section for plugin in ordered_plugins for section in plugin.prompt_sections
        )

        bound: list[BoundProgramTool] = []
        for plugin in ordered_plugins:
            tools_by_name = {_tool_name(tool): tool for tool in plugin.tools if _tool_name(tool)}
            for policy in plugin.program_tools:
                tool = tools_by_name[policy.tool_name]
                bound.append(
                    BoundProgramTool(
                        plugin_name=plugin.name,
                        tool=tool,
                        concurrency=policy.concurrency,
                        input_schema=_tool_input_schema(tool),
                        result_schema=policy.result_schema,
                    )
                )
        bound.sort(key=lambda item: item.name)

        sdk_payload = [
            {
                "concurrency": item.concurrency.value,
                "input_schema": item.input_schema,
                "name": item.name,
                "plugin": item.plugin_name,
                "result_schema": item.result_schema,
            }
            for item in bound
        ]
        return HarnessSurface(
            tools=tools,
            program_tools=tuple(bound),
            prompt_sections=prompt_sections,
            plugin_manifest=tuple(
                {"name": plugin.name, "version": plugin.version} for plugin in ordered_plugins
            ),
            program_sdk=json.dumps(
                sdk_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ),
        )


def program_policy_from_tool_metadata(tool: Any) -> ProgramToolPolicy | None:
    """Read an explicit opt-in policy from a custom LangChain tool.

    No policy is inferred from a name or description.  The caller owns the
    metadata, while Ultra owns validation and scheduling semantics.
    """

    metadata = getattr(tool, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    raw = metadata.get("ultra_tool_program")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("ultra_tool_program metadata must be an object")
    try:
        concurrency = ToolConcurrency(str(raw.get("concurrency") or "parallel"))
    except ValueError as exc:
        raise ValueError("invalid ultra_tool_program concurrency") from exc
    result_schema = raw.get("result_schema") or {}
    if not isinstance(result_schema, Mapping):
        raise ValueError("ultra_tool_program result_schema must be an object")
    return ProgramToolPolicy(
        tool_name=_tool_name(tool),
        concurrency=concurrency,
        result_schema=result_schema,
    )


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or "").strip()


def _tool_input_schema(tool: Any) -> dict[str, Any]:
    schema_type = getattr(tool, "tool_call_schema", None)
    if schema_type is None or not hasattr(schema_type, "model_json_schema"):
        raise ValueError(f"program tool {_tool_name(tool)!r} has no typed input schema")
    try:
        schema = schema_type.model_json_schema()
    except Exception as exc:  # noqa: BLE001 - converted to a safe construction error
        raise ValueError(
            f"program tool {_tool_name(tool)!r} input schema is not JSON-serializable"
        ) from exc
    return _canonical_json_object(schema, label=f"{_tool_name(tool)} input schema")


def _canonical_json_object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only finite JSON values") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be an object")
    return decoded
