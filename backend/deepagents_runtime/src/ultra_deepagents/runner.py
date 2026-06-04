from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import mimetypes
import re
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from ultra_deepagents.agent import build_research_agent
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.context_tools import stage_uploaded_files
from ultra_deepagents.events import (
    RunEvent,
    normalize_artifact_created,
    normalize_message_delta,
    normalize_run_completed,
    normalize_run_failed,
    normalize_subagent_message_delta,
    normalize_tool_call,
)
from ultra_deepagents.papers.tools import (
    ingest_arxiv_pdf,
    ingest_pdf_file,
    normalize_arxiv_id,
    paper_id_from_pdf_path,
)
from ultra_deepagents.rarespot.tools import (
    build_rarespot_idempotency_key,
    compact_config_overrides,
    create_rarespot_run,
    looks_report_only_rarespot_goal,
    wait_for_rarespot_run,
)
from ultra_deepagents.schemas import RunJobEnvelope

PublishEvent = Callable[[dict[str, Any]], Awaitable[None]]


class RunEventSequencer:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.sequence = 0

    def stamp(self, event: dict[str, Any]) -> dict[str, Any]:
        self.sequence += 1
        stamped = dict(event)
        stamped.setdefault("sequence", self.sequence)
        stamped.setdefault("event_id", f"evt_{self.run_id}_{self.sequence:06d}")
        return stamped


@dataclass(frozen=True)
class AgentAttemptResult:
    final_response_text: str
    streamed_response_text: str
    post_tool_streamed_response_text: str


class AgentStreamIdleTimeoutError(TimeoutError):
    def __init__(self, timeout_seconds: float) -> None:
        super().__init__(
            f"Deep Agents stream produced no events for {timeout_seconds:g}s "
            "while waiting for the model."
        )
        self.timeout_seconds = timeout_seconds


def _extract_arxiv_references(*, goal: str, messages: list[dict[str, Any]]) -> list[str]:
    """Extract unique arXiv IDs/URLs from a run goal and transcript messages."""
    texts = [goal]
    for message in messages:
        texts.extend(_message_text_fragments(message.get("content")))
    combined = "\n".join(text for text in texts if text)
    candidates: list[str] = []
    candidates.extend(
        match.group(0)
        for match in re.finditer(
            r"https?://(?:www\.)?(?:export\.)?arxiv\.org/(?:abs|pdf)/[A-Za-z0-9._/-]+(?:\.pdf)?",
            combined,
            flags=re.IGNORECASE,
        )
    )
    candidates.extend(
        match.group(1)
        for match in re.finditer(
            r"\barxiv\s*:\s*([A-Za-z-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?",
            combined,
            flags=re.IGNORECASE,
        )
    )
    candidates.extend(
        match.group(0)
        for match in re.finditer(r"(?<![\w./-])\d{4}\.\d{4,5}(?:v\d+)?(?![\w/-])", combined)
    )

    seen_ids: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        try:
            normalized = normalize_arxiv_id(candidate)
        except ValueError:
            continue
        if normalized in seen_ids:
            continue
        seen_ids.add(normalized)
        unique.append(candidate)
    return unique


def _preload_arxiv_papers_for_context(
    context: AgentRunContext,
    *,
    goal: str,
    messages: list[dict[str, Any]],
    cache_root: str | Path,
) -> AgentRunContext:
    references = _extract_arxiv_references(goal=goal, messages=messages)
    if not references:
        return context

    ingested: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []
    for reference in references:
        try:
            result = ingest_arxiv_pdf(context, reference, cache_root=cache_root)
        except Exception as exc:
            warnings.append({"source": reference, "error": str(exc)})
            continue
        ingested.append(
            {
                "paper_id": str(result.get("paper_id") or ""),
                "source": reference,
                "page_count": int(result.get("page_count") or 0),
                "chunk_count": int(result.get("chunk_count") or 0),
                "extraction_status": str(result.get("extraction_status") or "unknown"),
            }
        )

    if not ingested and not warnings:
        return context
    knowledge_context = dict(context.knowledge_context)
    existing = [
        item for item in knowledge_context.get("ingested_papers", [])
        if isinstance(item, dict)
    ]
    knowledge_context["ingested_papers"] = existing + ingested
    if warnings:
        knowledge_context["paper_ingest_warnings"] = warnings
    return replace(context, knowledge_context=knowledge_context)


def _preload_uploaded_pdf_papers_for_context(
    context: AgentRunContext,
    *,
    upload_roots: tuple[str | Path, ...],
    cache_root: str | Path,
) -> AgentRunContext:
    if not context.selected_file_ids:
        return context

    ingested: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []
    try:
        staged = stage_uploaded_files(context, upload_roots=upload_roots)
    except Exception as exc:
        warnings.append({"source": "selected_uploads", "error": str(exc)})
        staged = {"staged_files": [], "missing_file_ids": []}

    for missing_file_id in staged.get("missing_file_ids") or []:
        warnings.append({"source": f"upload:{missing_file_id}", "error": "upload_file_missing"})

    for file_info in staged.get("staged_files") or []:
        if not isinstance(file_info, dict):
            continue
        staged_path = Path(str(file_info.get("staged_path") or ""))
        file_id = str(file_info.get("file_id") or "").strip()
        if staged_path.suffix.lower() != ".pdf":
            continue
        try:
            result = ingest_pdf_file(
                context,
                staged_path,
                paper_id=paper_id_from_pdf_path(staged_path),
                cache_root=cache_root,
                title=staged_path.name,
                source_url=f"upload:{file_id}" if file_id else "",
            )
        except Exception as exc:
            warnings.append({"source": f"upload:{file_id}" if file_id else str(staged_path), "error": str(exc)})
            continue
        ingested.append(
            {
                "paper_id": str(result.get("paper_id") or ""),
                "source": f"upload:{file_id}" if file_id else str(staged_path),
                "page_count": int(result.get("page_count") or 0),
                "chunk_count": int(result.get("chunk_count") or 0),
                "extraction_status": str(result.get("extraction_status") or "unknown"),
            }
        )

    if not ingested and not warnings:
        return context
    knowledge_context = dict(context.knowledge_context)
    existing = [
        item for item in knowledge_context.get("ingested_papers", [])
        if isinstance(item, dict)
    ]
    knowledge_context["ingested_papers"] = _dedupe_ingested_papers(existing + ingested)
    if warnings:
        existing_warnings = [
            item for item in knowledge_context.get("paper_ingest_warnings", [])
            if isinstance(item, dict)
        ]
        knowledge_context["paper_ingest_warnings"] = existing_warnings + warnings
    return replace(context, knowledge_context=knowledge_context)


def _dedupe_ingested_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for paper in papers:
        paper_id = str(paper.get("paper_id") or "").strip()
        key = paper_id or str(paper.get("source") or "").strip()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(paper)
    return deduped


async def _preload_rarespot_for_context(
    context: AgentRunContext,
    *,
    settings: RuntimeSettings,
    sequencer: RunEventSequencer,
    publish_event: PublishEvent,
) -> dict[str, Any] | None:
    if not _should_preload_rarespot(context, settings=settings):
        return None

    text = _rarespot_request_text(context)
    config_overrides = _rarespot_config_overrides_from_text(text)
    idempotency_key = build_rarespot_idempotency_key(
        parent_run_id=context.run_id,
        selected_file_ids=list(context.selected_file_ids),
        local_paths=[],
        resource_uris=list(context.selected_resource_uris),
        dataset_uris=list(context.selected_dataset_uris),
        config_overrides=config_overrides,
    )
    body = {
        "user_id": context.user_id or "researcher",
        "goal": "Run RareSpot ecology inference.",
        "messages": [{"role": "tool", "content": "RareSpot ecology inference requested."}],
        "file_ids": list(context.selected_file_ids),
        "resource_uris": list(context.selected_resource_uris),
        "dataset_uris": list(context.selected_dataset_uris),
        "selected_tool_names": ["rarespot_ecology_inference"],
        "workflow_hint": {"id": "rarespot_ecology"},
        "idempotency_key": idempotency_key,
        "metadata": {
            "idempotency_key": idempotency_key,
            "file_ids": list(context.selected_file_ids),
            "local_paths": [],
            "tool_name": "rarespot_ecology_inference",
            "tile_overlap": 0.25,
            "spectral": True,
            "rarespot_config": config_overrides,
        },
    }

    await publish_event(
        sequencer.stamp(
            normalize_tool_call(
                context,
                "rarespot_ecology_inference",
                "started",
                {
                    "source": "deterministic_preload",
                    "input": body,
                    "tool_call_id": f"preload_rarespot_{context.run_id}",
                },
            )
        )
    )
    try:
        run = await asyncio.to_thread(
            create_rarespot_run,
            settings,
            thread_id=context.thread_id,
            body=body,
        )
        result = await asyncio.to_thread(
            wait_for_rarespot_run,
            settings,
            run_id=str(run["run_id"]),
            timeout_seconds=3600,
        )
    except Exception as exc:
        await publish_event(
            sequencer.stamp(
                normalize_tool_call(
                    context,
                    "rarespot_ecology_inference",
                    "failed",
                    {
                        "source": "deterministic_preload",
                        "error": str(exc),
                        "tool_call_id": f"preload_rarespot_{context.run_id}",
                    },
                )
            )
        )
        raise

    output = json.dumps(result, default=str, sort_keys=True)
    if _rarespot_result_failed(result):
        error = _rarespot_result_error(result)
        await publish_event(
            sequencer.stamp(
                normalize_tool_call(
                    context,
                    "rarespot_ecology_inference",
                    "failed",
                    {
                        "source": "deterministic_preload",
                        "error": error,
                        "output": output,
                        "output_size_chars": len(output),
                        "tool_call_id": f"preload_rarespot_{context.run_id}",
                    },
                )
            )
        )
        raise RuntimeError(f"RareSpot inference failed: {error}")
    await publish_event(
        sequencer.stamp(
            normalize_tool_call(
                context,
                "rarespot_ecology_inference",
                "completed",
                {
                    "source": "deterministic_preload",
                    "output": output,
                    "output_size_chars": len(output),
                    "tool_call_id": f"preload_rarespot_{context.run_id}",
                },
            )
        )
    )
    return result


def _rarespot_result_failed(result: dict[str, Any]) -> bool:
    status = str(result.get("status") or "").strip().lower()
    return bool(status) and status != "succeeded"


def _rarespot_result_error(result: dict[str, Any]) -> str:
    for key in ("error", "reason", "message"):
        value = str(result.get(key) or "").strip()
        if value:
            return value
    status = str(result.get("status") or "failed").strip()
    return f"RareSpot run {status}."


def _should_preload_rarespot(context: AgentRunContext, *, settings: RuntimeSettings) -> bool:
    if not settings.rarespot_tool_enabled:
        return False
    if not (
        context.selected_file_ids
        or context.selected_resource_uris
        or context.selected_dataset_uris
    ):
        return False
    text = _rarespot_request_text(context)
    if looks_report_only_rarespot_goal(text):
        return False
    return _looks_like_rarespot_detection_request(text)


def _rarespot_request_text(context: AgentRunContext) -> str:
    return str(context.goal or "")


def _looks_like_rarespot_detection_request(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return False
    detection_word = re.search(r"\b(detect|detection|inference|run|analy[sz]e)\b", normalized)
    if "rarespot" in normalized and detection_word:
        return True
    if re.search(r"\bprairie\s+dogs?\b", normalized) and detection_word:
        return True
    if re.search(r"\bburrows?\b", normalized) and detection_word:
        return True
    return False


def _rarespot_config_overrides_from_text(text: str) -> dict[str, Any]:
    normalized = " ".join(str(text or "").lower().split())
    confidence_threshold = _first_float_match(
        normalized,
        (
            r"\b(?:confidence|conf)\s*(?:threshold)?\s*(?:=|:|of|to)?\s*(0?\.\d+|1(?:\.0+)?)\b",
            r"\bthreshold\s*(?:=|:|of|to)?\s*(0?\.\d+|1(?:\.0+)?)\b",
        ),
    )
    image_size = _first_int_match(
        normalized,
        (
            r"\b(?:tile|image|imgsz|size)\D{0,12}([1-9]\d{2,3})\s*px\b",
            r"\b([1-9]\d{2,3})\s*px\s*(?:tile|tiles|image|imgsz|size)\b",
        ),
    )
    overlap_percent = _first_float_match(
        normalized,
        (
            r"\b(\d+(?:\.\d+)?)\s*(?:%|percent)\s*overlap\b",
            r"\boverlap\s*(?:=|:|of|to)?\s*(\d+(?:\.\d+)?)\s*(?:%|percent)\b",
        ),
    )
    tile_overlap = None
    if overlap_percent is not None:
        tile_overlap = overlap_percent / 100 if overlap_percent > 1 else overlap_percent
    return compact_config_overrides(
        confidence_threshold=confidence_threshold,
        image_size=image_size,
        tile_overlap=tile_overlap,
    )


def _first_float_match(text: str, patterns: tuple[str, ...]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        try:
            return float(match.group(1))
        except (TypeError, ValueError):
            continue
    return None


def _first_int_match(text: str, patterns: tuple[str, ...]) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            continue
    return None


def _append_preloaded_rarespot_messages(
    messages: list[dict[str, Any]],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    if _rarespot_result_failed(result):
        raise ValueError("Failed RareSpot results cannot be appended as verified context.")
    result_json = json.dumps(result, default=str, indent=2, sort_keys=True)
    return [
        *messages,
        {
            "role": "assistant",
            "content": (
                "RareSpot ecology inference has already completed through the production "
                "RareSpot tool. Use the verified tool result below as the evidence base. "
                "Do not rerun the same RareSpot configuration or search the sandbox "
                "filesystem for outputs."
            ),
        },
        {
            "role": "user",
            "content": (
                "Synthesize the final user-facing answer from this verified RareSpot "
                "result. Include quantitative counts, confidence summary, the 512 px "
                "tile / 25% overlap configuration when present, and direct artifact "
                "download links for overlay, CSV, and report. Use relative download_url "
                "values exactly as provided; do not add or invent a host, scheme, or "
                "placeholder domain. Keep the answer technical and concise.\n\n"
                f"```json\n{result_json}\n```"
            ),
        },
    ]


def _message_text_fragments(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        fragments: list[str] = []
        for item in content:
            if isinstance(item, dict):
                fragments.extend(_message_text_fragments(item.get("text") or item.get("content")))
            else:
                fragments.extend(_message_text_fragments(item))
        return fragments
    if isinstance(content, dict):
        return _message_text_fragments(content.get("text") or content.get("content"))
    return []


async def run_job(
    job: RunJobEnvelope,
    settings: RuntimeSettings,
    *,
    publish_event: PublishEvent,
    agent_factory: Callable[..., Any] = build_research_agent,
) -> str:
    workspace_dir = Path(settings.workspace_root).expanduser() / job.run_id
    artifact_dir = Path(settings.artifact_root).expanduser() / job.run_id
    workspace_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    context = job.to_context(
        artifact_root=str(artifact_dir),
        workspace_root=str(workspace_dir),
    )
    sequencer = RunEventSequencer(context.run_id)
    _write_workspace_lease(context, status="running")

    await publish_event(
        sequencer.stamp(
            RunEvent(
                run_id=context.run_id,
                thread_id=context.thread_id,
                event_kind="run.started",
                event_type="run",
                node_name="coordinator",
                agent_role="coordinator",
                level="info",
                message="Deep Agents run started.",
                payload={
                    "workspace_root": context.workspace_root,
                    "artifact_root": context.artifact_root,
                },
            ).to_dict()
        )
    )

    try:
        messages = list(job.messages or [{"role": "user", "content": job.goal}])
        context = _preload_arxiv_papers_for_context(
            context,
            goal=job.goal,
            messages=messages,
            cache_root=Path(settings.memory_root).expanduser() / "papers",
        )
        context = _preload_uploaded_pdf_papers_for_context(
            context,
            upload_roots=settings.rarespot_upload_roots,
            cache_root=Path(settings.memory_root).expanduser() / "papers",
        )
        preloaded_rarespot_result = await _preload_rarespot_for_context(
            context,
            settings=settings,
            sequencer=sequencer,
            publish_event=publish_event,
        )
        if preloaded_rarespot_result is not None:
            messages = _append_preloaded_rarespot_messages(
                messages,
                preloaded_rarespot_result,
            )
        agent = agent_factory(
            settings,
            workspace_dir=workspace_dir,
            artifact_dir=artifact_dir,
            context=context,
        )
        artifact_events: list[dict[str, Any]] = []
        attempt_result = AgentAttemptResult(
            final_response_text="",
            streamed_response_text="",
            post_tool_streamed_response_text="",
        )
        completion_continuations_used = 0
        model_stream_recoveries_used = 0

        while True:
            try:
                attempt_result = await _stream_agent_attempt(
                    agent,
                    messages=messages,
                    context=context,
                    sequencer=sequencer,
                    publish_event=publish_event,
                    model_stream_idle_timeout_seconds=settings.model_stream_idle_timeout_seconds,
                    langgraph_recursion_limit=settings.langgraph_recursion_limit,
                )
            except AgentStreamIdleTimeoutError as exc:
                if model_stream_recoveries_used >= settings.model_stream_idle_max_recoveries:
                    raise
                model_stream_recoveries_used += 1
                artifact_events = _collect_output_artifacts(context, workspace_dir, artifact_dir)
                current_response_text = _choose_response_text(
                    final_response_text=attempt_result.final_response_text,
                    streamed_response_text=attempt_result.streamed_response_text,
                    post_tool_streamed_response_text=attempt_result.post_tool_streamed_response_text,
                    artifact_events=artifact_events,
                )
                recovery_prompt = _model_stream_idle_recovery_prompt(
                    timeout_seconds=exc.timeout_seconds,
                    artifact_events=artifact_events,
                )
                await publish_event(
                    sequencer.stamp(
                        RunEvent(
                            run_id=context.run_id,
                            thread_id=context.thread_id,
                            event_kind="trace.message.delta",
                            event_type="trace",
                            node_name="model_stream_recovery",
                            agent_role="model_stream_recovery",
                            level="warning",
                            message=recovery_prompt,
                            payload={
                                "text": recovery_prompt,
                                "reason": "model_stream_idle",
                                "recovery_index": model_stream_recoveries_used,
                                "timeout_seconds": exc.timeout_seconds,
                            },
                        ).to_dict()
                    )
                )
                if current_response_text.strip():
                    messages.append({"role": "assistant", "content": current_response_text})
                messages.append({"role": "user", "content": recovery_prompt})
                continue

            artifact_events = _collect_output_artifacts(context, workspace_dir, artifact_dir)
            missing_kinds = [
                *_missing_requested_artifact_kinds(job, artifact_events),
                *_missing_requested_response_kinds(job, attempt_result),
            ]
            if not missing_kinds or completion_continuations_used >= settings.completion_max_continuations:
                break

            current_response_text = _choose_response_text(
                final_response_text=attempt_result.final_response_text,
                streamed_response_text=attempt_result.streamed_response_text,
                post_tool_streamed_response_text=attempt_result.post_tool_streamed_response_text,
                artifact_events=artifact_events,
            )
            continuation_prompt = _completion_continuation_prompt(
                missing_kinds=missing_kinds,
                artifact_events=artifact_events,
            )
            await publish_event(
                sequencer.stamp(
                    RunEvent(
                        run_id=context.run_id,
                        thread_id=context.thread_id,
                        event_kind="trace.message.delta",
                        event_type="trace",
                        node_name="completion_guard",
                        agent_role="completion_guard",
                        level="info",
                        message=continuation_prompt,
                        payload={
                            "text": continuation_prompt,
                            "missing_artifact_kinds": missing_kinds,
                            "continuation_index": completion_continuations_used + 1,
                        },
                    ).to_dict()
                )
            )
            if current_response_text.strip():
                messages.append({"role": "assistant", "content": current_response_text})
            messages.append({"role": "user", "content": continuation_prompt})
            completion_continuations_used += 1

        response_text = _choose_response_text(
            final_response_text=attempt_result.final_response_text,
            streamed_response_text=attempt_result.streamed_response_text,
            post_tool_streamed_response_text=attempt_result.post_tool_streamed_response_text,
            artifact_events=artifact_events,
        )
        for artifact_event in artifact_events:
            await publish_event(sequencer.stamp(artifact_event))
    except asyncio.CancelledError:
        _write_workspace_lease(context, status="canceled", error="canceled")
        raise
    except Exception as exc:
        _write_workspace_lease(context, status="failed", error=str(exc))
        await publish_event(sequencer.stamp(normalize_run_failed(context, exc)))
        raise

    _write_workspace_lease(context, status="succeeded")
    await publish_event(sequencer.stamp(normalize_run_completed(context, response_text)))
    return response_text


async def _stream_agent_attempt(
    agent: Any,
    *,
    messages: list[dict[str, Any]],
    context: AgentRunContext,
    sequencer: RunEventSequencer,
    publish_event: PublishEvent,
    model_stream_idle_timeout_seconds: float = 0.0,
    langgraph_recursion_limit: int = 1000,
) -> AgentAttemptResult:
    response_parts: list[str] = []
    post_tool_response_parts: list[str] = []
    final_state: dict[str, Any] | None = None
    tool_names_by_call_id: dict[str, str] = {}
    payload = {"messages": messages}
    stream = _start_agent_event_stream(
        agent,
        payload,
        context=context,
        version="v3",
        config={"recursion_limit": max(1, int(langgraph_recursion_limit))},
    )
    if inspect.isawaitable(stream):
        stream = await stream
    has_streamed_text = False
    saw_tool_event = False
    active_tool_calls: set[str] = set()
    stream_iter = stream.__aiter__()
    while True:
        try:
            if model_stream_idle_timeout_seconds > 0 and not active_tool_calls:
                event = await asyncio.wait_for(
                    stream_iter.__anext__(),
                    timeout=model_stream_idle_timeout_seconds,
                )
            else:
                event = await stream_iter.__anext__()
        except StopAsyncIteration:
            break
        except TimeoutError as exc:
            raise AgentStreamIdleTimeoutError(model_stream_idle_timeout_seconds) from exc
        tool_event = _tool_event_from_stream_event(context, event, tool_names_by_call_id)
        if tool_event is not None:
            await publish_event(sequencer.stamp(tool_event))
            _track_active_tool_call(active_tool_calls, tool_event)
            saw_tool_event = True
            post_tool_response_parts = []
            continue
        subagent_event = _subagent_message_event_from_stream_event(context, event)
        if subagent_event is not None:
            await publish_event(sequencer.stamp(subagent_event))
            continue
        if not _is_coordinator_stream(event):
            continue
        values_state = _deepagents_values_state(event)
        if values_state is not None:
            final_state = values_state
            continue
        text = _stream_text_delta(event)
        if not text:
            continue
        if not has_streamed_text and not text.strip():
            continue
        if text.strip():
            has_streamed_text = True
        response_parts.append(text)
        post_tool_response_parts.append(text)
        await publish_event(sequencer.stamp(normalize_message_delta(context, text)))

    return AgentAttemptResult(
        final_response_text=_response_text_from_final_state(final_state),
        streamed_response_text="".join(response_parts),
        post_tool_streamed_response_text=(
            "".join(post_tool_response_parts) if saw_tool_event else "".join(response_parts)
        ),
    )


def _start_agent_event_stream(
    agent: Any,
    payload: dict[str, Any],
    *,
    context: AgentRunContext,
    version: str,
    config: dict[str, Any],
) -> Any:
    method = agent.astream_events
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(payload, config=config, context=context, version=version)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if "config" in signature.parameters or accepts_var_kwargs:
        return method(payload, config=config, context=context, version=version)
    return method(payload, context=context, version=version)


def _track_active_tool_call(active_tool_calls: set[str], tool_event: dict[str, Any]) -> None:
    payload = tool_event.get("payload")
    if not isinstance(payload, dict):
        return
    key = _first_nonempty_string(payload.get("tool_call_id"), payload.get("tool_name"))
    if not key:
        return
    status = str(payload.get("status") or "").strip().lower()
    if status == "started":
        active_tool_calls.add(key)
    elif status in {"completed", "failed", "error"}:
        active_tool_calls.discard(key)


def _write_workspace_lease(
    context: AgentRunContext,
    *,
    status: str,
    cleanup_state: str = "active",
    error: str | None = None,
) -> None:
    workspace = Path(context.workspace_root)
    workspace.mkdir(parents=True, exist_ok=True)
    lease_path = workspace / "lease.json"
    now = _utc_now()
    created_at = now
    if lease_path.exists():
        try:
            existing = json.loads(lease_path.read_text())
            if isinstance(existing, dict) and isinstance(existing.get("created_at"), str):
                created_at = existing["created_at"]
        except (OSError, json.JSONDecodeError):
            created_at = now
    lease: dict[str, Any] = {
        "run_id": context.run_id,
        "thread_id": context.thread_id,
        "user_id": context.user_id,
        "org_id": context.org_id,
        "status": status,
        "cleanup_state": cleanup_state,
        "workspace_root": context.workspace_root,
        "artifact_root": context.artifact_root,
        "created_at": created_at,
        "updated_at": now,
    }
    if error:
        lease["error"] = error
    tmp_path = lease_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(lease, indent=2, sort_keys=True))
    tmp_path.replace(lease_path)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


_ANIMATION_FRAME_RE = re.compile(
    r"(^|[_-])frames?[_-]?\d+\.(?:png|jpe?g|webp)$|^frame[_-]?\d+\.(?:png|jpe?g|webp)$",
    re.IGNORECASE,
)
_SKIP_OUTPUT_DIRS = {".cache", ".deepagents", "__pycache__", "frames", "animation_frames", "tmp_frames"}
_SKIP_OUTPUT_FILES = {"lease.json", "run.lock", "matplotlibrc"}
_CODE_SUFFIXES = {".py", ".ipynb", ".r", ".R", ".js", ".ts", ".tsx", ".jsx", ".jl", ".m", ".sh"}
_TABLE_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".jsonl"}
_REPORT_SUFFIXES = {".md", ".txt", ".pdf", ".html"}
_MODEL_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx"}
_MINIMUM_FIGURE_PPI = 300
_DPI_CAPABLE_FIGURE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_MIME_TYPES_BY_SUFFIX = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".py": "text/x-python",
    ".r": "text/x-r",
    ".sh": "text/x-shellscript",
}


def _collect_output_artifacts(
    context: AgentRunContext,
    workspace_dir: Path,
    artifact_dir: Path,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_hashes: set[str] = set()
    output_roots = [
        (workspace_dir / "outputs", Path("outputs"), True),
        (workspace_dir, Path("."), False),
        (artifact_dir, Path("."), True),
    ]
    for root, prefix, recursive in output_roots:
        if not root.exists():
            continue
        candidates = root.rglob("*") if recursive else root.iterdir()
        for source in sorted(path for path in candidates if path.is_file()):
            if not _should_collect_output_file(source, root):
                continue
            relative = (prefix / source.relative_to(root)).as_posix()
            if relative.startswith("./"):
                relative = relative[2:]
            if not relative or relative in seen:
                continue
            seen.add(relative)
            target = artifact_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            copied = source.resolve() != target.resolve()
            if copied:
                shutil.copy2(source, target)
            figure_quality = _ensure_publication_figure_ppi(target)
            content_hash = _file_sha256(target)
            if content_hash in seen_hashes:
                if copied:
                    try:
                        target.unlink()
                    except OSError:
                        pass
                continue
            seen_hashes.add(content_hash)
            events.append(
                _artifact_event_for_path(
                    context,
                    target,
                    relative,
                    figure_quality=figure_quality,
                )
            )
    return events


def _should_collect_output_file(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    if path.is_symlink():
        return False
    if path.name in _SKIP_OUTPUT_FILES:
        return False
    if any(part.startswith(".") or part in _SKIP_OUTPUT_DIRS for part in relative.parts):
        return False
    lowered_parts = {part.lower() for part in relative.parts[:-1]}
    if lowered_parts & _SKIP_OUTPUT_DIRS:
        return False
    if _ANIMATION_FRAME_RE.search(path.name):
        return False
    return True


def _artifact_event_for_path(
    context: AgentRunContext,
    path: Path,
    relative_path: str,
    *,
    figure_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sha256 = _file_sha256(path)
    mime_type = _artifact_mime_type(path)
    kind = _artifact_kind(path, mime_type)
    path_digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]
    safe_run_id = re.sub(r"[^A-Za-z0-9_]+", "_", context.run_id).strip("_") or "run"
    artifact_id = f"artifact_{safe_run_id}_{path_digest}"
    output_id = f"output_{safe_run_id}_{path_digest}"
    title = _title_from_path(path)
    resolved_path = str(path.resolve())
    payload: dict[str, Any] = {
        "output_id": output_id,
        "kind": kind,
        "relative_path": relative_path,
        "source_path": resolved_path,
        "storage_uri": f"file://{resolved_path}",
        "mime_type": mime_type,
        "size_bytes": path.stat().st_size,
        "sha256": sha256,
        "tool_name": "outputs_collector",
    }
    if kind == "figure" and figure_quality is not None:
        payload["metadata"] = {"figure_quality": figure_quality}
    return normalize_artifact_created(
        context,
        artifact_id=artifact_id,
        path=relative_path,
        title=title,
        payload=payload,
    )


def _ensure_publication_figure_ppi(path: Path) -> dict[str, Any] | None:
    if path.suffix.lower() not in _DPI_CAPABLE_FIGURE_SUFFIXES:
        return None
    if not _artifact_mime_type(path).startswith("image/"):
        return None
    try:
        with Image.open(path) as image:
            image_format = image.format
            original_ppi = _image_ppi(image)
            if _ppi_meets_minimum(original_ppi):
                return _figure_quality_metadata(
                    original_ppi=original_ppi,
                    final_ppi=original_ppi,
                    normalized=False,
                )
            image.load()
            save_image = image
            save_kwargs: dict[str, Any] = {"dpi": (_MINIMUM_FIGURE_PPI, _MINIMUM_FIGURE_PPI)}
            if image_format == "JPEG":
                save_kwargs["quality"] = 95
                if image.mode in {"P", "LA", "RGBA"}:
                    save_image = image.convert("RGB")
            save_image.save(path, format=image_format, **save_kwargs)
        with Image.open(path) as updated_image:
            final_ppi = _image_ppi(updated_image)
    except (OSError, UnidentifiedImageError):
        return None
    return _figure_quality_metadata(
        original_ppi=original_ppi,
        final_ppi=final_ppi,
        normalized=True,
    )


def _image_ppi(image: Image.Image) -> tuple[float, float] | None:
    value = image.info.get("dpi")
    if isinstance(value, int | float):
        ppi = float(value)
        return (ppi, ppi)
    if isinstance(value, tuple | list) and len(value) >= 2:
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return None
    return None


def _ppi_meets_minimum(ppi: tuple[float, float] | None) -> bool:
    if ppi is None:
        return False
    return min(ppi) >= _MINIMUM_FIGURE_PPI - 0.5


def _figure_quality_metadata(
    *,
    original_ppi: tuple[float, float] | None,
    final_ppi: tuple[float, float] | None,
    normalized: bool,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "minimum_ppi": _MINIMUM_FIGURE_PPI,
        "dpi_metadata_normalized": normalized,
    }
    if original_ppi is not None:
        metadata["original_ppi"] = [round(original_ppi[0], 3), round(original_ppi[1], 3)]
    if final_ppi is not None:
        metadata["final_ppi"] = [round(final_ppi[0], 3), round(final_ppi[1], 3)]
    return metadata


def _artifact_mime_type(path: Path) -> str:
    return (
        _MIME_TYPES_BY_SUFFIX.get(path.suffix.lower())
        or mimetypes.guess_type(path.name)[0]
        or "application/octet-stream"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_kind(path: Path, mime_type: str) -> str:
    suffix = path.suffix
    suffix_lower = suffix.lower()
    if mime_type.startswith("image/"):
        return "figure"
    if suffix in _CODE_SUFFIXES:
        return "code"
    if suffix_lower in _MODEL_SUFFIXES:
        return "model"
    if suffix_lower in _TABLE_SUFFIXES:
        return "table"
    if suffix_lower in _REPORT_SUFFIXES:
        return "report"
    return "artifact"


def _choose_response_text(
    *,
    final_response_text: str,
    streamed_response_text: str,
    post_tool_streamed_response_text: str,
    artifact_events: list[dict[str, Any]],
) -> str:
    if final_response_text.strip():
        return final_response_text
    if post_tool_streamed_response_text.strip():
        return post_tool_streamed_response_text
    artifact_fallback = _fallback_response_from_artifacts(artifact_events)
    if artifact_fallback.strip():
        return artifact_fallback
    return streamed_response_text


def _fallback_response_from_artifacts(artifact_events: list[dict[str, Any]]) -> str:
    if not artifact_events:
        return ""
    lines = ["Saved durable outputs:"]
    for event in artifact_events:
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        title = str(payload.get("title") or event.get("message") or "Output").strip()
        kind = str(payload.get("kind") or "artifact").strip()
        path = str(payload.get("path") or payload.get("relative_path") or "").strip()
        label = f"{kind.title()}: {title}" if title else kind.title()
        lines.append(f"- {label}{f' (`{path}`)' if path else ''}")
    return "\n".join(lines)


_FIGURE_REQUEST_RE = re.compile(
    r"\b("
    r"plot|plots|figure|figures|visuali[sz]e|visuali[sz]ation|graph|graphs|"
    r"chart|charts|gif|animation|animations|qualitative\s+results?"
    r")\b",
    re.IGNORECASE,
)
_CODE_REQUEST_RE = re.compile(r"\b(code|script|notebook|python)\b", re.IGNORECASE)
_MODEL_REQUEST_RE = re.compile(
    r"\b("
    r"weights?|model\s+weights?|trained\s+model|saved\s+model|"
    r"checkpoints?|ckpt|safetensors|onnx"
    r")\b",
    re.IGNORECASE,
)
_RESPONSE_REQUEST_RE = re.compile(
    r"\b("
    r"briefly|explain|explanation|summari[sz]e|summary|interpret|interpretation|"
    r"compare|comparison|discuss|describe|walk\s+through|takeaways?|conclusions?|report"
    r")\b",
    re.IGNORECASE,
)
_DURABLE_OUTPUT_REQUEST_RE = re.compile(
    r"\b(save|saved|return|provide|download|durable|outputs?|artifacts?|deliverables?)\b",
    re.IGNORECASE,
)


def _missing_requested_artifact_kinds(
    job: RunJobEnvelope,
    artifact_events: list[dict[str, Any]],
) -> list[str]:
    requested = _requested_artifact_kinds(_run_request_text(job))
    if not requested:
        return []
    present = _present_artifact_kinds(artifact_events)
    return [kind for kind in requested if kind not in present]


def _missing_requested_response_kinds(
    job: RunJobEnvelope,
    attempt_result: AgentAttemptResult,
) -> list[str]:
    if not _RESPONSE_REQUEST_RE.search(_run_request_text(job)):
        return []
    if (
        attempt_result.final_response_text.strip()
        or attempt_result.post_tool_streamed_response_text.strip()
    ):
        return []
    return ["response"]


def _run_request_text(job: RunJobEnvelope) -> str:
    parts = [job.goal]
    for message in reversed(job.messages or []):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or message.get("type") or "").lower()
        if role in {"user", "human"}:
            parts.append(_chunk_text(message.get("content")))
            break
    return "\n".join(part for part in parts if part)


def _requested_artifact_kinds(text: str) -> list[str]:
    requested: list[str] = []
    if _FIGURE_REQUEST_RE.search(text):
        requested.append("figure")
    if _CODE_REQUEST_RE.search(text) and _DURABLE_OUTPUT_REQUEST_RE.search(text):
        requested.append("code")
    if _MODEL_REQUEST_RE.search(text) and _DURABLE_OUTPUT_REQUEST_RE.search(text):
        requested.append("model")
    return requested


def _present_artifact_kinds(artifact_events: list[dict[str, Any]]) -> set[str]:
    present: set[str] = set()
    for event in artifact_events:
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        kind = str(payload.get("kind") or "").strip()
        if kind:
            present.add(kind)
    return present


def _completion_continuation_prompt(
    *,
    missing_kinds: list[str],
    artifact_events: list[dict[str, Any]],
) -> str:
    current_outputs = _current_output_summary(artifact_events)
    current_outputs_text = "\n".join(current_outputs) if current_outputs else "- none yet"
    missing = ", ".join(missing_kinds)
    if missing_kinds == ["response"]:
        intro = "The previous attempt ended with a missing requested final response."
    else:
        intro = (
            "The previous attempt ended with missing requested durable outputs "
            f"or final response: {missing}."
        )
    return (
        f"{intro}\n"
        "Continue autonomously in the same workspace. If code was written to generate "
        "the missing output, call execute to run it, fix any errors, and verify the "
        "requested files exist with ls or another lightweight file check. Save durable "
        "deliverables under /outputs, /workspace/outputs, or top-level /workspace so "
        "the backend can collect them. Do not treat the task as complete until the "
        f"missing artifact kinds exist: {missing}.\n"
        "If the only missing item is response, do not rewrite files unnecessarily; "
        "write the requested final answer now, including the requested explain, "
        "summary, comparison, interpretation, or limitations text based only on the "
        "verified outputs.\n"
        "Current durable outputs detected:\n"
        f"{current_outputs_text}"
    )


def _model_stream_idle_recovery_prompt(
    *,
    timeout_seconds: float,
    artifact_events: list[dict[str, Any]],
) -> str:
    current_outputs = _current_output_summary(artifact_events)
    current_outputs_text = "\n".join(current_outputs) if current_outputs else "- none yet"
    return (
        "The previous Deep Agents model stream went idle after "
        f"{timeout_seconds:g}s with no new events while no tool call was active. "
        "Continue autonomously in the same workspace. Inspect existing files or "
        "outputs if needed, do not repeat completed tool work unless verification "
        "shows it is necessary, and either finish the requested final answer or "
        "continue the next missing step.\n"
        "Current durable outputs detected:\n"
        f"{current_outputs_text}"
    )


def _current_output_summary(artifact_events: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for event in artifact_events:
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        kind = str(payload.get("kind") or "artifact").strip()
        path = str(payload.get("path") or payload.get("relative_path") or "").strip()
        title = str(payload.get("title") or event.get("message") or path or "output").strip()
        if path:
            lines.append(f"- {kind}: {title} ({path})")
        else:
            lines.append(f"- {kind}: {title}")
    return lines


def _title_from_path(path: Path) -> str:
    title = re.sub(r"[_-]+", " ", path.stem).strip()
    return title.title() if title else path.name


def _is_coordinator_stream(event: dict[str, Any]) -> bool:
    metadata = event.get("metadata") or {}
    node = str(metadata.get("langgraph_node") or "").lower()
    agent_name = str(metadata.get("lc_agent_name") or "").lower()
    namespace = _event_namespace(event)
    if node and node not in {"agent", "coordinator", "main", "ultra-research-agent"}:
        return False
    if agent_name and agent_name not in {"ultra-research-agent", "main-agent", "coordinator"}:
        return False
    return not namespace


def _tool_event_from_stream_event(
    context: AgentRunContext,
    event: dict[str, Any],
    tool_names_by_call_id: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    raw_event = str(event.get("event") or event.get("method") or "").strip()
    if event.get("method") == "tools":
        return _tool_event_from_deepagents_tools_protocol(
            context,
            event,
            tool_names_by_call_id if tool_names_by_call_id is not None else {},
        )
    if raw_event in {"on_tool_start", "tool_call.started", "tool_call_start"}:
        tool_name = _tool_name_from_event(event)
        if not tool_name:
            return None
        return normalize_tool_call(
            context,
            tool_name=tool_name,
            status="started",
            payload=_tool_start_payload(event),
        )
    if raw_event in {"on_tool_end", "tool_call.completed", "tool_call_end"}:
        tool_name = _tool_name_from_event(event)
        if not tool_name:
            return None
        return normalize_tool_call(
            context,
            tool_name=tool_name,
            status="completed",
            payload=_tool_end_payload(event),
        )
    if raw_event in {"on_tool_error", "tool_call.failed", "tool_call.error", "tool_call_error"}:
        tool_name = _tool_name_from_event(event)
        if not tool_name:
            return None
        return normalize_tool_call(
            context,
            tool_name=tool_name,
            status="failed",
            payload=_tool_error_payload(event),
        )
    if event.get("method") == "tool_calls":
        return _tool_event_from_deepagents_tool_call(context, event)
    return None


def _tool_event_from_deepagents_tools_protocol(
    context: AgentRunContext,
    event: dict[str, Any],
    tool_names_by_call_id: dict[str, str],
) -> dict[str, Any] | None:
    params = event.get("params")
    if not isinstance(params, dict):
        return None
    data = params.get("data")
    if not isinstance(data, dict):
        return None
    raw_status = _first_nonempty_string(data.get("event"), data.get("status"))
    status = _normalize_tool_status(raw_status)
    tool_call_id = _first_nonempty_string(data.get("tool_call_id"), data.get("id"), data.get("run_id"))
    tool_name = _first_nonempty_string(
        data.get("tool_name"),
        data.get("name"),
        tool_names_by_call_id.get(tool_call_id),
    )
    if tool_call_id and tool_name:
        tool_names_by_call_id[tool_call_id] = tool_name
    if not tool_name:
        return None
    payload: dict[str, Any] = {"tool_call_id": tool_call_id}
    tool_input = data.get("input")
    if isinstance(tool_input, dict):
        payload.update(_compact_payload_mapping(tool_input))
        payload["input"] = _compact_payload_value(tool_input)
    elif tool_input is not None:
        payload["input"] = _compact_payload_value(tool_input)
    output = data.get("output")
    if output is not None:
        payload["output_preview"] = _payload_preview_text(output)
        payload["output_size_chars"] = len(str(output))
    error = data.get("error")
    if error is not None:
        payload["error"] = _payload_preview_text(error)
    return normalize_tool_call(
        context,
        tool_name=tool_name,
        status=status,
        payload=_drop_empty_payload_values(payload),
    )


def _tool_event_from_deepagents_tool_call(
    context: AgentRunContext,
    event: dict[str, Any],
) -> dict[str, Any] | None:
    params = event.get("params")
    if not isinstance(params, dict):
        return None
    data = params.get("data")
    if isinstance(data, list | tuple):
        candidates = [item for item in data if isinstance(item, dict)]
    elif isinstance(data, dict):
        candidates = [data]
    else:
        candidates = []
    for candidate in candidates:
        tool_name = _first_nonempty_string(
            candidate.get("name"),
            candidate.get("tool_name"),
            candidate.get("tool"),
        )
        if not tool_name:
            continue
        status = _first_nonempty_string(candidate.get("status"), candidate.get("state"))
        if not status:
            status = "completed" if candidate.get("completed") is True else "started"
        normalized_status = _normalize_tool_status(status)
        payload = _tool_payload_from_mapping(candidate)
        return normalize_tool_call(
            context,
            tool_name=tool_name,
            status=normalized_status,
            payload=payload,
        )
    return None


def _tool_name_from_event(event: dict[str, Any]) -> str:
    data = event.get("data")
    data_mapping = data if isinstance(data, dict) else {}
    return _first_nonempty_string(
        event.get("name"),
        event.get("tool_name"),
        data_mapping.get("name"),
        data_mapping.get("tool_name"),
    )


def _tool_start_payload(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    payload: dict[str, Any] = {
        "tool_call_id": _first_nonempty_string(event.get("run_id"), data.get("run_id"), data.get("id")),
    }
    tool_input = data.get("input")
    if isinstance(tool_input, dict):
        payload.update(_compact_payload_mapping(tool_input))
        payload["input"] = _compact_payload_value(tool_input)
    elif tool_input is not None:
        payload["input"] = _compact_payload_value(tool_input)
    return _drop_empty_payload_values(payload)


def _tool_end_payload(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    output = data.get("output")
    payload: dict[str, Any] = {
        "tool_call_id": _first_nonempty_string(event.get("run_id"), data.get("run_id"), data.get("id")),
    }
    if output is not None:
        output_text = _payload_preview_text(output)
        payload["output_preview"] = output_text
        payload["output_size_chars"] = len(str(output))
    return _drop_empty_payload_values(payload)


def _tool_error_payload(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    error = data.get("error") or event.get("error")
    payload: dict[str, Any] = {
        "tool_call_id": _first_nonempty_string(event.get("run_id"), data.get("run_id"), data.get("id")),
    }
    if error is not None:
        payload["error"] = _payload_preview_text(error)
    return _drop_empty_payload_values(payload)


def _tool_payload_from_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in ("id", "run_id", "tool_call_id", "call_id"):
        value = _first_nonempty_string(mapping.get(key))
        if value:
            payload["tool_call_id"] = value
            break
    tool_input = mapping.get("input") or mapping.get("args") or mapping.get("arguments")
    if isinstance(tool_input, dict):
        payload.update(_compact_payload_mapping(tool_input))
        payload["input"] = _compact_payload_value(tool_input)
    elif tool_input is not None:
        payload["input"] = _compact_payload_value(tool_input)
    output = mapping.get("output") or mapping.get("result")
    if output is not None:
        payload["output_preview"] = _payload_preview_text(output)
        payload["output_size_chars"] = len(str(output))
    error = mapping.get("error")
    if error is not None:
        payload["error"] = _payload_preview_text(error)
    return payload


def _normalize_tool_status(status: str) -> str:
    normalized = status.strip().lower()
    if normalized in {"end", "complete", "completed", "success", "succeeded", "tool-finished"}:
        return "completed"
    if normalized in {"error", "errored", "failed", "failure", "tool-error", "tool-failed"}:
        return "failed"
    return "started"


def _subagent_message_event_from_stream_event(
    context: AgentRunContext,
    event: dict[str, Any],
) -> dict[str, Any] | None:
    source = _subagent_source_from_event(event)
    if not source:
        return None
    text = _stream_text_delta(event)
    if not text:
        return None
    namespace = _event_namespace_list(event)
    task_id = _first_nonempty_string(event.get("run_id"), _metadata_value(event, "run_id"))
    return normalize_subagent_message_delta(
        context,
        name=source,
        text=text,
        task_id=task_id or None,
        payload={"namespace": namespace} if namespace else None,
    )


def _subagent_source_from_event(event: dict[str, Any]) -> str:
    namespace = _event_namespace_list(event)
    if namespace:
        return str(namespace[-1]).strip()
    agent_name = _metadata_value(event, "lc_agent_name")
    if not isinstance(agent_name, str) or not agent_name.strip():
        return ""
    normalized = agent_name.strip()
    if normalized.lower() in {"ultra-research-agent", "main-agent", "coordinator"}:
        return ""
    return normalized


def _event_namespace_list(event: dict[str, Any]) -> list[Any]:
    namespace = _event_namespace(event)
    if isinstance(namespace, list | tuple):
        return list(namespace)
    if namespace:
        return [namespace]
    return []


def _metadata_value(event: dict[str, Any], key: str) -> Any:
    metadata = event.get("metadata")
    if isinstance(metadata, dict) and key in metadata:
        return metadata.get(key)
    params = event.get("params")
    if isinstance(params, dict):
        data = params.get("data")
        if isinstance(data, list | tuple):
            for item in data:
                if isinstance(item, dict) and key in item:
                    return item.get(key)
        if isinstance(data, dict) and key in data:
            return data.get(key)
    return None


def _first_nonempty_string(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _compact_payload_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _compact_payload_value(item) for key, item in value.items()}


def _drop_empty_payload_values(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None and value != ""}


def _compact_payload_value(value: Any, *, max_chars: int = 2000) -> Any:
    if isinstance(value, dict):
        return _compact_payload_mapping(value)
    if isinstance(value, list | tuple):
        return [_compact_payload_value(item) for item in value[:20]]
    if isinstance(value, str):
        return value if len(value) <= max_chars else f"{value[:max_chars]}... [truncated]"
    if isinstance(value, int | float | bool) or value is None:
        return value
    return _payload_preview_text(value, max_chars=max_chars)


def _payload_preview_text(value: Any, *, max_chars: int = 2000) -> str:
    text = _chunk_text(value) or str(value)
    return text if len(text) <= max_chars else f"{text[:max_chars]}... [truncated]"


def _event_namespace(event: dict[str, Any]) -> Any:
    if "namespace" in event:
        return event.get("namespace")
    params = event.get("params")
    if isinstance(params, dict):
        return params.get("namespace")
    return None


def _deepagents_values_state(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("method") != "values":
        return None
    params = event.get("params")
    if not isinstance(params, dict):
        return None
    data = params.get("data")
    if isinstance(data, dict):
        return data
    return None


def _stream_text_delta(event: dict[str, Any]) -> str:
    if event.get("event") == "on_chat_model_stream":
        return _chunk_text((event.get("data") or {}).get("chunk"))
    if event.get("method") != "messages":
        return ""
    params = event.get("params")
    if not isinstance(params, dict):
        return ""
    data = params.get("data")
    if not isinstance(data, list | tuple) or not data:
        return ""
    chunk = data[0]
    if not isinstance(chunk, dict):
        return _chunk_text(chunk)
    if chunk.get("event") != "content-block-delta":
        return ""
    return _chunk_text(chunk.get("delta"))


def _response_text_from_final_state(state: dict[str, Any] | None) -> str:
    if not isinstance(state, dict):
        return ""
    messages = state.get("messages")
    if not isinstance(messages, list | tuple):
        return ""
    last_user_index = _last_user_message_index(messages)
    candidate_messages = messages[last_user_index + 1 :] if last_user_index >= 0 else messages
    for message in reversed(candidate_messages):
        if not _is_assistant_message(message):
            continue
        text = _message_content_text(message)
        if text.strip():
            return text
    return ""


def _last_user_message_index(messages: list[Any] | tuple[Any, ...]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if _is_user_message(messages[index]):
            return index
    return -1


def _is_user_message(message: Any) -> bool:
    if isinstance(message, dict):
        role = str(message.get("role") or message.get("type") or "").lower()
        return role in {"user", "human"}
    role = str(getattr(message, "role", "") or getattr(message, "type", "") or "").lower()
    if role in {"user", "human"}:
        return True
    return message.__class__.__name__ == "HumanMessage"


def _is_assistant_message(message: Any) -> bool:
    if isinstance(message, dict):
        role = str(message.get("role") or message.get("type") or "").lower()
        return role in {"assistant", "ai"}
    role = str(getattr(message, "role", "") or getattr(message, "type", "") or "").lower()
    if role in {"assistant", "ai"}:
        return True
    return message.__class__.__name__ == "AIMessage"


def _message_content_text(message: Any) -> str:
    if isinstance(message, dict):
        return _chunk_text(message.get("content"))
    return _chunk_text(getattr(message, "content", None))


def _chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list | tuple):
        return "".join(_content_block_text(item) for item in content)
    if isinstance(content, dict):
        return _content_block_text(content)
    return ""


def _content_block_text(block: Any) -> str:
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return ""
    if isinstance(block.get("text"), str):
        return block["text"]
    if isinstance(block.get("content"), str):
        return block["content"]
    return ""
