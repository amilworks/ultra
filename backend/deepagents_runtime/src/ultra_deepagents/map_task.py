"""map_task: deterministic batched subagent dispatch.

The one capability the interpreter's dynamic subagents would add that Ultra
actually needs — fanning N similar delegations out under code control — built
on the NORMAL subagent path instead of an eval bridge, so everything the
runtime guarantees for `task` holds per dispatch: streamed events with correct
namespaces (guards see it, steering interleaves, the UI renders it), usage
accounting with valid dedup keys, cancellation propagation, and Tier-1
harness testability.

Contract highlights:
- results aggregate IN ORDER; one failed item becomes an error entry, never a
  failed run;
- full outputs are offloaded to ``<workspace>/map_task/<call>.jsonl`` and only
  compact previews enter the transcript (the context-economics point of the
  tool);
- concurrency is semaphore-bounded and item counts are capped.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from deepagents.middleware import subagents as deepagents_subagents
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

MAP_TASK_MAX_ITEMS = 64
MAP_TASK_MAX_CONCURRENCY = 8
MAP_TASK_DEFAULT_CONCURRENCY = 4
_PREVIEW_CHARS = 240
_RESULT_MARKER = "map_task result"

# Parent-state keys that must not leak into subagent state — mirror the task
# tool's exclusions, with a defensive fallback if the private name moves.
_EXCLUDED_STATE_KEYS: frozenset[str] = getattr(
    deepagents_subagents, "_EXCLUDED_STATE_KEYS", frozenset({"messages", "todos"})
)


class MapTaskSchema(BaseModel):
    subagent_type: str = Field(description="Which configured subagent runs every item.")
    items: list[str] = Field(
        description=(
            "One self-contained delegation prompt per item (each runs in an "
            "isolated subagent with no shared context)."
        )
    )
    max_concurrency: int = Field(
        default=MAP_TASK_DEFAULT_CONCURRENCY,
        description=f"Simultaneous dispatches, 1..{MAP_TASK_MAX_CONCURRENCY}.",
    )
    item_timeout_seconds: float = Field(
        default=0.0,
        description="Per-item wall-clock bound; 0 inherits no per-item bound.",
    )


def _compile_subagent_graphs(
    subagents: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """name -> runnable for LOCAL subagent specs, mirroring the task tool.

    Compiled specs ("runnable" present) are used as-is; raw specs compile
    through deepagents' own ``create_sub_agent`` so map_task and task can
    never drift in how a subagent is materialized.
    """
    graphs: dict[str, Any] = {}
    for spec in subagents:
        name = str(spec.get("name") or "").strip()
        if not name or "graph_id" in spec:  # async subagents are not mappable
            continue
        if "runnable" in spec:
            graphs[name] = spec["runnable"]
        else:
            # Callers must pass ENRICHED specs (model + tools present) — the
            # same precondition deepagents' own compile path enforces.
            graphs[name] = deepagents_subagents.create_sub_agent(spec)
    return graphs


def _result_text(result: Any) -> str:
    if isinstance(result, dict):
        structured = result.get("structured_response")
        if structured is not None:
            try:
                return json.dumps(structured, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                return str(structured)
        messages = result.get("messages") or []
        for message in reversed(messages):
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                return content
    return str(result)


def _safe_call_slug(tool_call_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]", "", str(tool_call_id or ""))[:40]
    return slug or "call"


def build_map_task_tool(
    subagents: Sequence[dict[str, Any]],
    *,
    workspace_dir: str | Path | None,
) -> BaseTool:
    graphs_cache: dict[str, Any] = {}

    def _graphs() -> dict[str, Any]:
        # Lazy: compiling shares the run's first map_task call, and runs that
        # never map pay nothing.
        if not graphs_cache:
            graphs_cache.update(_compile_subagent_graphs(subagents))
        return graphs_cache

    async def amap_task(
        subagent_type: str,
        items: list[str],
        max_concurrency: int = MAP_TASK_DEFAULT_CONCURRENCY,
        item_timeout_seconds: float = 0.0,
        runtime: Any = None,
    ) -> str:
        graphs = _graphs()
        if subagent_type not in graphs:
            allowed = ", ".join(f"`{name}`" for name in sorted(graphs))
            return (
                f"We cannot map over subagent {subagent_type} because it does not "
                f"exist, the only allowed types are {allowed}"
            )
        cleaned = [str(item) for item in (items or []) if str(item).strip()]
        if not cleaned:
            return "map_task requires at least one non-empty item."
        if len(cleaned) > MAP_TASK_MAX_ITEMS:
            return (
                f"map_task accepts at most {MAP_TASK_MAX_ITEMS} items per call "
                f"(got {len(cleaned)}); chunk the batch and call again."
            )
        concurrency = max(1, min(int(max_concurrency or 1), MAP_TASK_MAX_CONCURRENCY))
        timeout = float(item_timeout_seconds or 0.0)

        parent_state = {}
        state = getattr(runtime, "state", None) or {}
        parent_state = {key: value for key, value in state.items() if key not in _EXCLUDED_STATE_KEYS}
        graph = graphs[subagent_type]
        semaphore = asyncio.Semaphore(concurrency)

        async def dispatch(index: int, description: str) -> dict[str, Any]:
            async with semaphore:
                subagent_state = {**parent_state, "messages": [HumanMessage(content=description)]}
                config = {"configurable": {"ls_agent_type": "subagent"}}
                try:
                    invocation = graph.ainvoke(subagent_state, config)
                    result = (
                        await asyncio.wait_for(invocation, timeout=timeout)
                        if timeout > 0
                        else await invocation
                    )
                except TimeoutError:
                    return {
                        "index": index,
                        "status": "error",
                        "error": f"item timed out after {timeout:g}s",
                    }
                except Exception as exc:  # noqa: BLE001 - one item must never fail the batch
                    return {
                        "index": index,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                return {"index": index, "status": "ok", "text": _result_text(result)}

        results = await asyncio.gather(
            *(dispatch(position + 1, item) for position, item in enumerate(cleaned))
        )

        jsonl_note = ""
        if workspace_dir is not None:
            try:
                out_dir = Path(workspace_dir) / "map_task"
                out_dir.mkdir(parents=True, exist_ok=True)
                call_id = _safe_call_slug(getattr(runtime, "tool_call_id", "") or "")
                jsonl_path = out_dir / f"{subagent_type}_{call_id}.jsonl"
                with jsonl_path.open("w", encoding="utf-8") as handle:
                    for entry in results:
                        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                jsonl_note = str(jsonl_path)
            except OSError:
                jsonl_note = ""  # offload is best-effort; previews still return

        ok_count = sum(1 for entry in results if entry["status"] == "ok")
        compact = [
            {
                "index": entry["index"],
                "status": entry["status"],
                **(
                    {"preview": entry["text"][:_PREVIEW_CHARS]}
                    if entry["status"] == "ok"
                    else {"error": entry["error"]}
                ),
            }
            for entry in results
        ]
        payload = {
            "subagent_type": subagent_type,
            "items": len(cleaned),
            "ok": ok_count,
            "errors": len(cleaned) - ok_count,
            "results_jsonl": jsonl_note,
            "entries": compact,
        }
        return (
            f"{_RESULT_MARKER} ({ok_count}/{len(cleaned)} ok; full outputs in "
            f"{jsonl_note or 'previews below — offload unavailable'}):\n"
            + json.dumps(payload, ensure_ascii=False, indent=1)
        )

    return StructuredTool.from_function(
        name="map_task",
        coroutine=amap_task,
        description=(
            "Deterministically fan one subagent out over MANY independent items "
            "and get one aggregated, order-preserving result. Use this instead "
            "of serial task calls whenever you have three or more similar, "
            "independent delegations (per-paper review, per-claim verification, "
            "per-image reads): it dispatches concurrently under a cap, isolates "
            "per-item failures as error entries, writes full outputs to a "
            "workspace JSONL, and returns only compact previews so the batch "
            "never floods your context. Items must be self-contained prompts — "
            "subagents share no state. For a single delegation, or steps that "
            "depend on each other's outputs, keep using task."
        ),
        args_schema=MapTaskSchema,
        infer_schema=False,
    )
