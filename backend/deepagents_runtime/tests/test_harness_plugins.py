from __future__ import annotations

import json

import pytest
from langchain_core.tools import tool
from ultra_deepagents.harness_plugins import (
    HarnessPlugin,
    HarnessPluginRegistry,
    ProgramToolPolicy,
    ToolConcurrency,
    program_policy_from_tool_metadata,
)


@tool
def alpha_lookup(value: int) -> dict[str, int]:
    """Return a small deterministic lookup result."""
    return {"value": value}


@tool
def beta_lookup(query: str) -> dict[str, str]:
    """Return a small deterministic search result."""
    return {"query": query}


def _registry(*, reverse: bool = False) -> HarnessPluginRegistry:
    plugins = [
        HarnessPlugin(
            name="alpha",
            version="1",
            order=10,
            tools=(alpha_lookup,),
            program_tools=(
                ProgramToolPolicy(
                    tool_name="alpha_lookup",
                    concurrency=ToolConcurrency.PARALLEL,
                    result_schema={
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                    },
                ),
            ),
            prompt_sections=("Alpha guidance.",),
        ),
        HarnessPlugin(
            name="beta",
            version="2",
            order=20,
            tools=(beta_lookup,),
            program_tools=(
                ProgramToolPolicy(
                    tool_name="beta_lookup",
                    concurrency=ToolConcurrency.EXCLUSIVE,
                ),
            ),
            prompt_sections=("Beta guidance.",),
        ),
    ]
    registry = HarnessPluginRegistry()
    for plugin in reversed(plugins) if reverse else plugins:
        registry.register(plugin)
    return registry


def test_plugin_surface_is_deterministic_and_cache_stable():
    forward = _registry().freeze()
    reverse = _registry(reverse=True).freeze()

    assert [tool.name for tool in forward.tools] == ["alpha_lookup", "beta_lookup"]
    assert forward.tools == reverse.tools
    assert forward.prompt_sections == ("Alpha guidance.", "Beta guidance.")
    assert forward.prompt_sections == reverse.prompt_sections
    assert forward.plugin_manifest == (
        {"name": "alpha", "version": "1"},
        {"name": "beta", "version": "2"},
    )
    assert forward.program_sdk == reverse.program_sdk
    assert forward.program_sdk.index("alpha_lookup") < forward.program_sdk.index("beta_lookup")

    sdk_payload = json.loads(forward.program_sdk)
    assert sdk_payload[0]["concurrency"] == "parallel"
    assert sdk_payload[0]["input_schema"]["properties"]["value"]["type"] == "integer"
    assert sdk_payload[0]["result_schema"]["properties"]["value"]["type"] == "integer"
    assert sdk_payload[1]["concurrency"] == "exclusive"


def test_plugin_registry_rejects_ambiguous_names_and_capabilities():
    registry = HarnessPluginRegistry()
    registry.register(HarnessPlugin(name="first", tools=(alpha_lookup,)))

    with pytest.raises(ValueError, match="plugin name"):
        registry.register(HarnessPlugin(name="first", tools=(beta_lookup,)))

    duplicate_tool = HarnessPlugin(name="second", tools=(alpha_lookup,))
    with pytest.raises(ValueError, match="tool name"):
        registry.register(duplicate_tool)

    missing_tool = HarnessPlugin(
        name="missing",
        tools=(beta_lookup,),
        program_tools=(ProgramToolPolicy(tool_name="not_registered"),),
    )
    with pytest.raises(ValueError, match="not_registered"):
        HarnessPluginRegistry((missing_tool,))


def test_custom_tool_program_policy_requires_explicit_valid_metadata():
    beta_lookup.metadata = {
        "ultra_tool_program": {
            "concurrency": "parallel",
            "result_schema": {"type": "object"},
        }
    }
    try:
        policy = program_policy_from_tool_metadata(beta_lookup)
        assert policy == ProgramToolPolicy(
            tool_name="beta_lookup",
            concurrency=ToolConcurrency.PARALLEL,
            result_schema={"type": "object"},
        )

        beta_lookup.metadata = {"ultra_tool_program": {"concurrency": "unknown"}}
        with pytest.raises(ValueError, match="concurrency"):
            program_policy_from_tool_metadata(beta_lookup)
    finally:
        beta_lookup.metadata = None


def test_tool_program_metadata_is_opt_in():
    beta_lookup.metadata = {"unrelated": True}
    try:
        assert program_policy_from_tool_metadata(beta_lookup) is None
    finally:
        beta_lookup.metadata = None
