from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agno.agent import Agent
from agno.models.base import Model as AgnoModel


def build_codeexec_reasoning_agent(
    *,
    model_builder: Callable[..., AgnoModel],
    tools: list[Any],
    max_runtime_seconds: int,
    session_state: dict[str, Any] | None = None,
    debug_mode: bool = False,
) -> Agent:
    """Build the dedicated reasoning agent for hard computational prompts.

    Keep deterministic execution on explicit tools while letting the model reason
    about whether execution is needed, how to validate outputs, and how to fail
    closed if the computation does not complete.
    """

    return Agent(
        name="pro-mode-codeexec-reasoner",
        model=model_builder(
            reasoning_mode="fast",
            reasoning_effort_override="low",
            max_runtime_seconds=max_runtime_seconds,
        ),
        reasoning_model=model_builder(
            reasoning_mode="deep",
            reasoning_effort_override="high",
            max_runtime_seconds=max_runtime_seconds,
        ),
        tools=list(tools or []) or None,
        instructions=[
            "You are the dedicated Pro Mode reasoning agent for hard computational tasks.",
            "Use the first reasoning pass to decide whether code execution is actually necessary.",
            "When execution is necessary, generate a plan, run code, inspect outputs, and validate results before answering.",
            "Prefer the smallest executable workflow that can produce the requested evidence.",
            "Treat execute_python_job outputs as the source of truth for measured values and produced artifacts.",
            "If execution fails or returns incomplete results, say so clearly and do not fabricate measured outputs.",
        ],
        session_state=dict(session_state or {}),
        add_session_state_to_context=True,
        markdown=True,
        telemetry=False,
        retries=0,
        store_events=False,
        store_history_messages=False,
        reasoning=True,
        reasoning_min_steps=2,
        reasoning_max_steps=10,
        debug_mode=bool(debug_mode),
    )
