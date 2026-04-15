"""Tool-calling engine for Bisque Ultra (LLM + deterministic tools)."""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

from src.logger import logger
from src.tooling.progress import encode_progress_chunk, reset_progress_callback, set_progress_callback

_STREAMING_TOOLS = {"yolo_finetune_detect"}


def _derive_bisque_client_view_url(resource_uri: str | None, existing_url: str | None = None) -> str | None:
    candidate = str(existing_url or "").strip()
    if candidate:
        return candidate
    normalized_resource_uri = str(resource_uri or "").strip()
    if not normalized_resource_uri:
        return None
    try:
        parsed = urlparse(normalized_resource_uri)
    except Exception:
        return None
    if not parsed.scheme or not parsed.netloc:
        return None
    normalized = normalized_resource_uri
    if "/image_service/" in normalized:
        normalized = normalized.replace("/image_service/", "/data_service/", 1)
    elif "/data_service/" not in normalized:
        return None
    return f"{parsed.scheme}://{parsed.netloc}/client_service/view?resource={normalized}"

@dataclass
class ToolState:
    """Minimal shared state for chaining and summarization."""

    last_search: list[dict[str, Any]] = field(default_factory=list)
    last_resource: str | None = None
    last_downloads: list[str] = field(default_factory=list)
    last_yolo_models: list[dict[str, Any]] = field(default_factory=list)
    latest_result_refs: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summarize(self) -> str:
        parts: list[str] = []
        if self.last_search:
            sample = self.last_search[:3]
            parts.append(
                "Recent BisQue search results (sample):\n"
                + "\n".join(
                    f"- {r.get('name') or r.get('uri') or 'resource'}"
                    for r in sample
                )
            )
        if self.last_resource:
            parts.append(f"Last resource URI: {self.last_resource}")
        if self.last_downloads:
            parts.append(
                "Downloaded paths (recent):\n"
                + "\n".join(f"- {p}" for p in self.last_downloads[-3:])
            )
        if self.last_yolo_models:
            latest = self.last_yolo_models[0]
            model_name = latest.get("model_name") or "unknown"
            model_path = latest.get("model_path") or ""
            parts.append(f"Latest local finetuned YOLO model: {model_name} ({model_path})")
        if self.latest_result_refs:
            ref_lines: list[str] = []
            for key, value in self.latest_result_refs.items():
                if isinstance(value, str):
                    ref_lines.append(f"- {key}: {value}")
            if ref_lines:
                parts.append("Latest result refs:\n" + "\n".join(ref_lines[:12]))
        return "\n".join(parts)


def _coerce_tool_argument_dict(raw_args: Any) -> dict[str, Any] | None:
    if isinstance(raw_args, dict):
        return dict(raw_args)
    if not isinstance(raw_args, str):
        return None
    token = str(raw_args or "").strip()
    if not token:
        return {}
    try:
        parsed = json.loads(token)
    except Exception:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _serialize_tool_arguments(args: dict[str, Any], *, original: Any) -> str | dict[str, Any]:
    if isinstance(original, dict):
        return args
    return json.dumps(args, ensure_ascii=False, default=str)


def _looks_like_mask_path(path: str) -> bool:
    lowered = str(path or "").strip().lower()
    if not lowered:
        return False
    normalized = str(Path(lowered).name)
    normalized = " ".join(filter(None, re.sub(r"[^a-z0-9]+", " ", normalized).split()))
    words = set(normalized.split())
    if "mask" in words or "segmentation" in words or "pred" in words or "prediction" in words:
        return True
    if lowered.endswith(".nii.gz") and words.intersection({"mask", "label", "labels", "gt", "target", "truth"}):
        return True
    if lowered.endswith((".npy", ".npz", ".nii")):
        return True
    if lowered.endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        return bool(words.intersection({"mask", "label", "labels", "gt", "target", "truth", "segmentation"}))
    return False


def _collect_latest_segmentation_mask_paths(state: ToolState) -> list[str]:
    refs = state.latest_result_refs
    candidates: list[str] = []
    for key in (
        "latest_segmentation_mask_paths",
        "segment_image_sam2.mask_paths",
        "segment_image_sam2.preferred_upload_paths",
        "segment_image_sam3.mask_paths",
        "segment_image_sam3.preferred_upload_paths",
        "sam2_prompt_image.mask_paths",
        "sam2_prompt_image.preferred_upload_paths",
    ):
        values = refs.get(key)
        if isinstance(values, list):
            for item in values:
                token = str(item or "").strip()
                if token:
                    candidates.append(token)
    for key in (
        "latest_segmentation_mask_path",
        "segment_image_sam2.latest_mask_path",
        "segment_image_sam3.latest_mask_path",
        "sam2_prompt_image.latest_mask_path",
        "latest_mask_path",
    ):
        token = str(refs.get(key) or "").strip()
        if token:
            candidates.append(token)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if not _looks_like_mask_path(item):
            continue
        path = Path(item).expanduser()
        try:
            if not path.exists() or not path.is_file():
                continue
            normalized = str(path.resolve())
        except Exception:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _coerce_existing_mask_paths(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    paths: list[str] = []
    seen: set[str] = set()
    for item in values:
        token = str(item or "").strip()
        if not token or not _looks_like_mask_path(token):
            continue
        path = Path(token).expanduser()
        try:
            if not path.exists() or not path.is_file():
                continue
            normalized = str(path.resolve())
        except Exception:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        paths.append(normalized)
    return paths


def _enrich_tool_arguments_from_state(
    *,
    tool_name: str,
    tool_args: Any,
    tool_state: ToolState,
) -> Any:
    if tool_name != "quantify_segmentation_masks":
        return tool_args

    args = _coerce_tool_argument_dict(tool_args)
    if not isinstance(args, dict):
        return tool_args

    existing_mask_paths = _coerce_existing_mask_paths(args.get("mask_paths"))
    inferred_mask_paths = _collect_latest_segmentation_mask_paths(tool_state)
    if existing_mask_paths:
        if existing_mask_paths != args.get("mask_paths"):
            args["mask_paths"] = existing_mask_paths
            return _serialize_tool_arguments(args, original=tool_args)
        return tool_args
    if not inferred_mask_paths:
        return tool_args

    args["mask_paths"] = inferred_mask_paths[:8]
    logger.info(
        "Injected %s inferred mask path(s) into quantify_segmentation_masks from prior tool state",
        len(args["mask_paths"]),
    )
    return _serialize_tool_arguments(args, original=tool_args)


class ToolEngine:
    """LLM tool-calling loop with simple chaining state."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        tools: list[dict[str, Any]],
        tool_executor: Callable[[str, str | dict, list[str] | None], str],
        max_iterations: int = 6,
        stream_delta_fix: bool = True,
    ) -> None:
        self.client = client
        self.model = model
        self.tools = tools
        self.tool_executor = tool_executor
        self.max_iterations = max_iterations
        self.stream_delta_fix = stream_delta_fix

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        uploaded_files: list[str] | None = None,
        extra_body: dict[str, Any] | None = None,
        tool_state: ToolState | None = None,
        scratchpad_path: str | None = None,
        refine_response: bool = True,
        suppress_draft_output: bool = False,
        emit_suppressed_draft_when_no_refine: bool = True,
        workpad_mode: str | None = None,
    ) -> Iterator[str]:
        tool_state = tool_state or ToolState()
        scratchpad = None
        suppress_output = bool(suppress_draft_output)
        if scratchpad_path:
            init_started = time.monotonic()
            phase_payload: dict[str, Any] = {
                "event": "phase",
                "phase": "workpad_init",
                "status": "started",
                "message": "Initializing prompt workpad.",
                "ts": datetime.utcnow().isoformat() + "Z",
            }
            if workpad_mode:
                phase_payload["mode"] = str(workpad_mode)
            yield encode_progress_chunk(phase_payload)
            question = _latest_user_question(messages)
            scratchpad = _ScratchpadWriter(scratchpad_path, question)
            yield encode_progress_chunk(
                {
                    "event": "phase",
                    "phase": "workpad_init",
                    "status": "completed",
                    "message": "Prompt workpad initialized.",
                    "duration_seconds": round(time.monotonic() - init_started, 3),
                    "mode": str(workpad_mode or "legacy"),
                    "ts": datetime.utcnow().isoformat() + "Z",
                }
            )

        iteration = 0
        overall_has_content = False
        last_tool_error: str | None = None
        last_assistant_text: str | None = None
        required_explicit_tools = _explicit_requested_tool_names(
            _latest_user_question(messages) or "",
            _tool_name_set(self.tools),
        )
        called_tool_names: set[str] = set()
        explicit_retry_count = 0
        tool_result_cache: dict[str, tuple[dict[str, Any], str]] = {}

        while iteration < self.max_iterations:
            iteration += 1
            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "tools": self.tools,
                "tool_choice": "auto",
            }
            if extra_body:
                request_params["extra_body"] = extra_body

            stream = self.client.chat.completions.create(**request_params)
            tool_calls = []
            current_tool_call: dict[str, Any] | None = None
            assembled_text = ""

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if delta.content:
                    text = delta.content
                    if self.stream_delta_fix and assembled_text and text.startswith(assembled_text):
                        new_text = text[len(assembled_text) :]
                        assembled_text = text
                    else:
                        new_text = text
                        assembled_text += text
                    if new_text:
                        overall_has_content = True
                        if not suppress_output:
                            yield new_text

                if delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        if tool_call_delta.index is not None:
                            if current_tool_call is None or tool_call_delta.index != current_tool_call.get("index"):
                                if current_tool_call:
                                    tool_calls.append(current_tool_call)
                                current_tool_call = {
                                    "index": tool_call_delta.index,
                                    "id": tool_call_delta.id or "",
                                    "function": {
                                        "name": tool_call_delta.function.name or "",
                                        "arguments": tool_call_delta.function.arguments or "",
                                    },
                                }
                            else:
                                if tool_call_delta.function.arguments:
                                    current_tool_call["function"]["arguments"] += tool_call_delta.function.arguments

            if current_tool_call:
                tool_calls.append(current_tool_call)

            if assembled_text.strip():
                last_assistant_text = assembled_text.strip()
                if scratchpad:
                    scratchpad.append_section(
                        f"Draft (iteration {iteration})",
                        assembled_text.strip(),
                    )

            if not tool_calls:
                missing_explicit_tools = sorted(
                    [name for name in required_explicit_tools if name not in called_tool_names]
                )
                if (
                    missing_explicit_tools
                    and explicit_retry_count < 2
                    and iteration < self.max_iterations
                ):
                    explicit_retry_count += 1
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "You must call explicitly requested tool(s) before finalizing: "
                                + ", ".join(missing_explicit_tools)
                                + ". Call each at least once. If inputs are missing, call anyway and return the error."
                            ),
                        }
                    )
                    continue
                break

            assistant_message = {"role": "assistant", "content": None, "tool_calls": []}
            tool_results_with_viz: list[dict[str, Any]] = []

            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = _enrich_tool_arguments_from_state(
                    tool_name=tool_name,
                    tool_args=tool_call["function"]["arguments"],
                    tool_state=tool_state,
                )
                tool_args_text = (
                    json.dumps(tool_args, ensure_ascii=False, default=str)
                    if isinstance(tool_args, dict)
                    else str(tool_args or "")
                )
                tool_call["function"]["arguments"] = tool_args_text
                called_tool_names.add(str(tool_name))
                if not tool_call.get("id"):
                    tool_call["id"] = f"tool_{iteration}_{tool_calls.index(tool_call)}"
                status_label = _tool_status_label(tool_name)
                tool_fingerprint = _tool_call_fingerprint(tool_name, tool_args_text)

                cached_result = tool_result_cache.get(tool_fingerprint)
                if cached_result is not None:
                    result_dict, tool_content = cached_result
                    tool_failed = isinstance(result_dict, dict) and result_dict.get("success") is False
                    progress_artifacts = _progress_artifacts_from_result(result_dict)
                    progress_summary = _progress_summary_from_result(tool_name, result_dict)
                    if tool_failed:
                        _set_tool_status(status_label, "error")
                        last_tool_error = str(result_dict.get("error") or "Tool failed")
                        yield encode_progress_chunk(
                            {
                                "event": "error",
                                "tool": tool_name,
                                "message": f"{status_label} failed",
                                "error": last_tool_error,
                                "cached": True,
                                "ts": datetime.utcnow().isoformat() + "Z",
                            }
                        )
                    else:
                        _set_tool_status(status_label, "complete")
                        yield encode_progress_chunk(
                            {
                                "event": "completed",
                                "tool": tool_name,
                                "message": f"{status_label} complete",
                                "artifacts": progress_artifacts,
                                "summary": progress_summary,
                                "cached": True,
                                "ts": datetime.utcnow().isoformat() + "Z",
                            }
                        )

                    _update_tool_state(tool_state, tool_name, result_dict)
                    _collect_ui_artifacts(result_dict, tool_results_with_viz)
                    if scratchpad:
                        scratchpad.append_section(
                            f"Tool: {tool_name}",
                            _summarize_tool_output(tool_name, result_dict),
                        )
                    assistant_message["tool_calls"].append(
                        {
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {"name": tool_name, "arguments": tool_args_text},
                        }
                    )
                    messages.append(assistant_message)
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call["id"], "content": tool_content}
                    )
                    assistant_message = {"role": "assistant", "content": None, "tool_calls": []}
                    continue

                _set_tool_status(status_label, "running")
                logger.info("Executing tool: %s", tool_name)

                if tool_name not in _STREAMING_TOOLS:
                    yield encode_progress_chunk(
                        {
                            "event": "started",
                            "tool": tool_name,
                            "message": f"{status_label}...",
                            "ts": datetime.utcnow().isoformat() + "Z",
                        }
                    )

                raw_result: Any = None
                tool_runtime_error: str | None = None
                try:
                    if tool_name in _STREAMING_TOOLS:
                        progress_queue: queue.Queue[object] = queue.Queue()
                        progress_done = object()
                        outcome: dict[str, Any] = {"raw": None, "exc": None}
                        start_ts = time.time()

                        def _progress_cb(payload: dict[str, Any]) -> None:
                            if not isinstance(payload, dict):
                                return
                            payload.setdefault("tool", tool_name)
                            progress_queue.put(payload)

                        def _run_tool() -> None:
                            token = set_progress_callback(_progress_cb)
                            try:
                                outcome["raw"] = self.tool_executor(
                                    tool_name, tool_args_text, uploaded_files
                                )
                            except Exception as exc:  # noqa: BLE001
                                outcome["exc"] = exc
                            finally:
                                reset_progress_callback(token)
                                progress_queue.put(progress_done)

                        thread = threading.Thread(target=_run_tool, daemon=True)
                        thread.start()

                        yield encode_progress_chunk(
                            {
                                "event": "started",
                                "tool": tool_name,
                                "message": f"{status_label}...",
                                "ts": datetime.utcnow().isoformat() + "Z",
                            }
                        )

                        last_heartbeat = start_ts
                        heartbeat_every_s = 2.0
                        while True:
                            try:
                                item = progress_queue.get(timeout=0.25)
                            except queue.Empty:
                                now = time.time()
                                if now - last_heartbeat >= heartbeat_every_s:
                                    elapsed = round(now - start_ts, 1)
                                    yield encode_progress_chunk(
                                        {
                                            "event": "heartbeat",
                                            "tool": tool_name,
                                            "message": f"{status_label}... ({elapsed}s)",
                                            "elapsed_s": elapsed,
                                            "ts": datetime.utcnow().isoformat() + "Z",
                                        }
                                    )
                                    last_heartbeat = now
                                if not thread.is_alive():
                                    break
                                continue
                            if item is progress_done:
                                break
                            if isinstance(item, dict):
                                yield encode_progress_chunk(item)

                        thread.join()
                        if outcome.get("exc") is not None:
                            raise outcome["exc"]
                        raw_result = outcome["raw"]
                    else:
                        raw_result = self.tool_executor(tool_name, tool_args_text, uploaded_files)
                except Exception as exc:  # noqa: BLE001
                    tool_runtime_error = str(exc)
                    logger.exception("Tool execution failed: %s", tool_name)
                    raw_result = {
                        "success": False,
                        "error": tool_runtime_error,
                        "tool_name": tool_name,
                    }

                result_dict, tool_content = _coerce_tool_result(raw_result, tool_name=tool_name)
                tool_result_cache[tool_fingerprint] = (result_dict, tool_content)
                tool_failed = isinstance(result_dict, dict) and result_dict.get("success") is False
                progress_artifacts = _progress_artifacts_from_result(result_dict)
                progress_summary = _progress_summary_from_result(tool_name, result_dict)
                if tool_failed:
                    _set_tool_status(status_label, "error")
                    last_tool_error = str(result_dict.get("error") or "Tool failed")
                    yield encode_progress_chunk(
                        {
                            "event": "error",
                            "tool": tool_name,
                            "message": f"{status_label} failed",
                            "error": last_tool_error,
                            "ts": datetime.utcnow().isoformat() + "Z",
                        }
                    )
                else:
                    _set_tool_status(status_label, "complete")
                    yield encode_progress_chunk(
                        {
                            "event": "completed",
                            "tool": tool_name,
                            "message": f"{status_label} complete",
                            "artifacts": progress_artifacts,
                            "summary": progress_summary,
                            "ts": datetime.utcnow().isoformat() + "Z",
                        }
                    )

                _update_tool_state(tool_state, tool_name, result_dict)
                _collect_ui_artifacts(result_dict, tool_results_with_viz)
                if scratchpad:
                    scratchpad.append_section(
                        f"Tool: {tool_name}",
                        _summarize_tool_output(tool_name, result_dict),
                    )

                assistant_message["tool_calls"].append(
                    {
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {"name": tool_name, "arguments": tool_args_text},
                    }
                )

                messages.append(assistant_message)
                messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": tool_content})
                assistant_message = {"role": "assistant", "content": None, "tool_calls": []}

            _push_ui_artifacts(tool_results_with_viz)

            summary = tool_state.summarize()
            if summary:
                messages.append(
                    {
                        "role": "system",
                        "content": "TOOL CONTEXT (do not quote verbatim):\n" + summary,
                    }
                )
                if scratchpad:
                    scratchpad.append_section("Tool context summary", summary)

        if scratchpad and refine_response:
            if not last_assistant_text:
                scratchpad.append_section(
                    "Draft (auto)",
                    _auto_draft_from_state(tool_state, last_tool_error),
                )
            refine_started = time.monotonic()
            yield encode_progress_chunk(
                {
                    "event": "phase",
                    "phase": "workpad_refinement",
                    "status": "started",
                    "message": "Refining final response from prompt workpad.",
                    "mode": str(workpad_mode or "legacy"),
                    "ts": datetime.utcnow().isoformat() + "Z",
                }
            )
            for chunk in _refine_from_scratchpad(
                client=self.client,
                model=self.model,
                scratchpad_path=scratchpad.path,
                fallback_text=last_assistant_text,
                error_text=last_tool_error,
            ):
                yield chunk
            yield encode_progress_chunk(
                {
                    "event": "phase",
                    "phase": "workpad_refinement",
                    "status": "completed",
                    "message": "Prompt workpad refinement completed.",
                    "duration_seconds": round(time.monotonic() - refine_started, 3),
                    "mode": str(workpad_mode or "legacy"),
                    "ts": datetime.utcnow().isoformat() + "Z",
                }
            )
            return
        if scratchpad and not refine_response:
            yield encode_progress_chunk(
                {
                    "event": "phase",
                    "phase": "workpad_refinement",
                    "status": "skipped",
                    "reason": "disabled_by_mode",
                    "mode": str(workpad_mode or "phased"),
                    "message": "Prompt workpad refinement skipped by mode.",
                    "ts": datetime.utcnow().isoformat() + "Z",
                }
            )
            if (
                suppress_output
                and bool(emit_suppressed_draft_when_no_refine)
                and last_assistant_text
            ):
                yield last_assistant_text
                return

        if not overall_has_content:
            if last_tool_error:
                yield f"Tool execution failed: {last_tool_error}"
            else:
                yield "Tools executed successfully. See outputs above."


def _collect_ui_artifacts(result: dict[str, Any], sink: list[dict[str, Any]]) -> None:
    if not isinstance(result, dict):
        return
    ui_artifacts = result.get("ui_artifacts")
    if isinstance(ui_artifacts, list):
        for item in ui_artifacts[:25]:
            if isinstance(item, dict):
                sink.append(item)
    visualization_paths = result.get("visualization_paths")
    if isinstance(visualization_paths, list):
        for item in visualization_paths[:50]:
            if isinstance(item, dict):
                sink.append(item)
    prediction_images = result.get("prediction_images")
    if isinstance(prediction_images, list):
        counts = result.get("counts_by_class") or {}
        pretty_counts = ", ".join(
            [f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:6]]
        )
        for img_path in prediction_images[:50]:
            sink.append(
                {
                    "path": img_path,
                    "file": img_path.split("/")[-1],
                    "caption": (
                        f"🛰️ YOLO detections ({pretty_counts})" if pretty_counts else "🛰️ YOLO detections"
                    ),
                }
            )


def _progress_artifacts_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []

    image_exts = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg", ".tif", ".tiff")
    by_path: dict[str, dict[str, Any]] = {}

    def _add(path_value: Any, *, title: str | None = None, kind: str | None = None) -> None:
        path = str(path_value or "").strip()
        if not path:
            return
        lower = path.lower()
        inferred_kind = kind or ("image" if lower.endswith(image_exts) else "file")
        by_path[path] = {
            "path": path,
            "title": title or Path(path).name,
            "kind": inferred_kind,
        }

    for key in (
        "overlay_path",
        "side_by_side_path",
        "mask_preview_path",
        "mask_path",
        "mask_volume_path",
        "preferred_upload_path",
        "output_path",
        "report_markdown_path",
        "report_json_path",
    ):
        if key in result:
            _add(result.get(key))

    for key in ("preferred_upload_paths", "mask_paths", "depth_map_paths", "output_files"):
        values = result.get(key)
        if isinstance(values, list):
            for value in values[:120]:
                _add(value)

    preferred_entries = result.get("preferred_upload_entries")
    if isinstance(preferred_entries, list):
        for item in preferred_entries[:120]:
            if not isinstance(item, dict):
                continue
            _add(
                item.get("path"),
                title=str(item.get("file") or item.get("title") or "").strip() or None,
            )

    ui_artifacts = result.get("ui_artifacts")
    if isinstance(ui_artifacts, list):
        for item in ui_artifacts[:80]:
            if not isinstance(item, dict):
                continue
            _add(item.get("path"), title=str(item.get("title") or "").strip() or None, kind=item.get("type"))

    visualization_paths = result.get("visualization_paths")
    if isinstance(visualization_paths, list):
        for item in visualization_paths[:120]:
            if isinstance(item, dict):
                _add(
                    item.get("path"),
                    title=str(item.get("title") or item.get("caption") or "").strip() or None,
                    kind=item.get("kind"),
                )
            else:
                _add(item)

    files_processed = result.get("files_processed")
    if isinstance(files_processed, list):
        for row in files_processed[:120]:
            if not isinstance(row, dict):
                continue
            for key in (
                "preferred_upload_path",
                "mask_path",
                "mask_volume_path",
                "visualization",
                "output_path",
            ):
                if key in row:
                    _add(
                        row.get(key),
                        title=str(row.get("file") or row.get("title") or "").strip() or None,
                    )
            visualizations = row.get("visualizations")
            if isinstance(visualizations, list):
                for item in visualizations[:40]:
                    if isinstance(item, dict):
                        _add(
                            item.get("path"),
                            title=str(item.get("title") or item.get("caption") or "").strip() or None,
                            kind=item.get("kind"),
                        )
                    else:
                        _add(item)

    prediction_images = result.get("prediction_images")
    if isinstance(prediction_images, list):
        for path in prediction_images[:120]:
            _add(path, title="YOLO prediction image", kind="image")

    return list(by_path.values())[:120]


def _coerce_result_count(result: dict[str, Any], fallback: int) -> int:
    value = result.get("count")
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, str):
        token = value.strip()
        if token.isdigit():
            return int(token)
    return max(0, fallback)


def _resource_rows_for_summary(resources: Any, *, limit: int = 10) -> list[dict[str, Any]]:
    if not isinstance(resources, list):
        return []
    rows: list[dict[str, Any]] = []
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        resource_uri = str(resource.get("resource_uri") or resource.get("uri") or "").strip()
        client_view_url = _derive_bisque_client_view_url(
            str(resource.get("resource_uri") or resource.get("uri") or "").strip(),
            str(resource.get("client_view_url") or resource.get("view_url") or "").strip() or None,
        ) or ""
        image_service_url = str(resource.get("image_service_url") or "").strip()
        display_uri = client_view_url or resource_uri
        raw_name = str(resource.get("name") or resource.get("file") or "").strip()
        fallback_name = resource_uri.split("/")[-1] if resource_uri else "resource"
        row: dict[str, Any] = {"name": raw_name or fallback_name}
        owner = str(resource.get("owner") or "").strip()
        if owner:
            row["owner"] = owner
        created = _format_timestamp(str(resource.get("created") or resource.get("ts") or "").strip() or None)
        if created:
            row["created"] = created
        resource_type = str(resource.get("resource_type") or resource.get("type") or "").strip()
        if resource_type:
            row["resource_type"] = resource_type
        if resource_uri:
            row["resource_uri"] = resource_uri
        if client_view_url:
            row["client_view_url"] = client_view_url
        if image_service_url:
            row["image_service_url"] = image_service_url
        if display_uri:
            row["uri"] = display_uri
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def _download_rows_for_summary(downloads: Any, *, limit: int = 10) -> list[dict[str, Any]]:
    if not isinstance(downloads, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in downloads:
        if not isinstance(item, dict):
            continue
        resource_uri = str(item.get("resource_uri") or item.get("uri") or "").strip()
        client_view_url = _derive_bisque_client_view_url(
            str(item.get("resource_uri") or item.get("uri") or "").strip(),
            str(item.get("client_view_url") or item.get("view_url") or "").strip() or None,
        ) or ""
        image_service_url = str(item.get("image_service_url") or "").strip()
        output_path = str(item.get("output_path") or item.get("path") or "").strip()
        success = bool(item.get("success"))
        row: dict[str, Any] = {
            "status": "ok" if success else "failed",
        }
        if resource_uri:
            row["resource_uri"] = resource_uri
        if client_view_url:
            row["client_view_url"] = client_view_url
        if image_service_url:
            row["image_service_url"] = image_service_url
        if output_path:
            row["output_path"] = output_path
        error_text = str(item.get("error") or "").strip()
        if error_text:
            row["error"] = error_text
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def _bisque_query_scope_summary(query: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(query, dict):
        return {}
    payload: dict[str, Any] = {}
    for key in (
        "requested_resource_type",
        "normalized_resource_type",
        "resource_type",
        "resource_types_tried",
        "text",
        "text_terms",
        "original_tag_query",
        "tag_query",
        "tag_filters",
        "metadata_filters",
        "expanded_tag_filters",
        "limit",
        "offset",
    ):
        value = query.get(key)
        if value in (None, "", [], {}):
            continue
        payload[key] = value
    return payload


def _bisque_query_scope_text(query: dict[str, Any] | None) -> str:
    payload = _bisque_query_scope_summary(query)
    if not payload:
        return ""
    parts: list[str] = []
    requested_type = str(payload.get("requested_resource_type") or "").strip()
    resolved_type = str(payload.get("resource_type") or "").strip()
    if requested_type and requested_type != resolved_type:
        parts.append(f"requested type '{requested_type}' (resolved as '{resolved_type}')")
    elif resolved_type:
        parts.append(f"type '{resolved_type}'")
    text_value = payload.get("text")
    if isinstance(text_value, list):
        rendered_text = ", ".join(str(item).strip() for item in text_value[:8] if str(item or "").strip())
    else:
        rendered_text = str(text_value or "").strip()
    if rendered_text:
        parts.append(f'text "{rendered_text}"')
    original_tag_query = str(payload.get("original_tag_query") or "").strip()
    if original_tag_query:
        parts.append(f"tag query {original_tag_query}")
    return ", ".join(parts)


def _progress_summary_from_result(tool_name: str, result: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    if tool_name == "upload_to_bisque":
        summary = {
            "success": bool(result.get("success")),
            "kind": "bisque_upload",
            "uploaded": result.get("uploaded"),
            "total": result.get("total"),
            "dataset_action": result.get("dataset_action"),
            "dataset_success": result.get("dataset_success"),
            "dataset_name": result.get("dataset_name"),
            "dataset_uri": result.get("dataset_uri"),
            "dataset_members_added": result.get("dataset_members_added"),
            "dataset_client_view_url": str(result.get("dataset_client_view_url") or "").strip() or None,
            "rows": _resource_rows_for_summary(result.get("results"), limit=12),
        }
        error_text = str(result.get("error") or "").strip()
        if error_text:
            summary["error"] = error_text
        return summary
    if tool_name == "delete_bisque_resource":
        deleted_uri = str(result.get("deleted_uri") or result.get("resource_uri") or "").strip() or None
        deleted_client_view_url = _derive_bisque_client_view_url(
            deleted_uri,
            str(result.get("deleted_client_view_url") or result.get("client_view_url") or "").strip() or None,
        )
        resource_name = str(result.get("resource_name") or "").strip() or None
        dataset_cleanup = result.get("dataset_cleanup") if isinstance(result.get("dataset_cleanup"), dict) else None
        rows = _resource_rows_for_summary(
            [
                {
                    "name": resource_name or deleted_uri or "resource",
                    "resource_uri": deleted_uri,
                    "client_view_url": deleted_client_view_url,
                    "uri": deleted_client_view_url or deleted_uri,
                }
            ],
            limit=1,
        )
        return {
            "success": bool(result.get("success")),
            "kind": "bisque_delete",
            "resource_name": resource_name,
            "resource_uri": deleted_uri,
            "client_view_url": deleted_client_view_url,
            "rows": rows,
            "deletion_verified": bool(result.get("deletion_verified")),
            "deletion_verification_attempts": result.get("deletion_verification_attempts"),
            "dataset_cleanup": dataset_cleanup,
            "error": str(result.get("error") or "").strip() or None,
        }
    if result.get("success") is False:
        return {"success": False, "error": str(result.get("error") or "Tool failed")}

    if tool_name == "segment_image_sam3":
        files_processed = result.get("files_processed")
        files_summary: list[dict[str, Any]] = []
        if isinstance(files_processed, list):
            for row in files_processed[:8]:
                if not isinstance(row, dict):
                    continue
                files_summary.append(
                    {
                        "file": row.get("file"),
                        "coverage_scope": row.get("coverage_scope"),
                        "coverage_percent": row.get("coverage_percent"),
                        "total_masks": row.get("total_masks"),
                        "instance_count_reported": row.get("instance_count_reported"),
                        "instance_count_measured": row.get("instance_count_measured"),
                        "instance_count_scope": row.get("instance_count_scope"),
                        "instance_coverage_percent_mean": row.get("instance_coverage_percent_mean"),
                        "instance_coverage_percent_min": row.get("instance_coverage_percent_min"),
                        "instance_coverage_percent_max": row.get("instance_coverage_percent_max"),
                        "instance_area_voxels_mean": row.get("instance_area_voxels_mean"),
                        "instance_area_voxels_min": row.get("instance_area_voxels_min"),
                        "instance_area_voxels_max": row.get("instance_area_voxels_max"),
                        "avg_points_per_window": row.get("avg_points_per_window"),
                        "min_points": row.get("min_points"),
                        "max_points": row.get("max_points"),
                        "point_density": row.get("point_density"),
                        "preferred_upload_path": row.get("preferred_upload_path"),
                        "success": row.get("success"),
                    }
                )
        return {
            "success": True,
            "kind": "sam3",
            "processed": result.get("processed"),
            "total_files": result.get("total_files"),
            "total_masks_generated": result.get("total_masks_generated"),
            "concept_prompt": result.get("concept_prompt"),
            "concept_prompt_source": result.get("concept_prompt_source"),
            "coverage_scope": result.get("coverage_scope"),
            "coverage_percent_mean": result.get("coverage_percent_mean"),
            "coverage_percent_min": result.get("coverage_percent_min"),
            "coverage_percent_max": result.get("coverage_percent_max"),
            "instance_count_reported_total": result.get("instance_count_reported_total"),
            "instance_count_measured_total": result.get("instance_count_measured_total"),
            "instance_count_mismatch_files": result.get("instance_count_mismatch_files"),
            "instance_count_scope": result.get("instance_count_scope"),
            "instance_coverage_percent_mean": result.get("instance_coverage_percent_mean"),
            "instance_coverage_percent_min": result.get("instance_coverage_percent_min"),
            "instance_coverage_percent_max": result.get("instance_coverage_percent_max"),
            "instance_area_voxels_mean": result.get("instance_area_voxels_mean"),
            "instance_area_voxels_min": result.get("instance_area_voxels_min"),
            "instance_area_voxels_max": result.get("instance_area_voxels_max"),
            "min_points": result.get("min_points"),
            "max_points": result.get("max_points"),
            "point_density": result.get("point_density"),
            "files": files_summary,
            "preferred_upload_paths": result.get("preferred_upload_paths"),
            "model": result.get("model"),
            "output_directory": result.get("output_directory"),
        }

    if tool_name == "yolo_detect":
        counts = result.get("counts_by_class") if isinstance(result.get("counts_by_class"), dict) else {}
        counts_rows = [
            {"class_name": str(name), "count": int(value)}
            for name, value in sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[:10]
        ]
        predictions = result.get("predictions") if isinstance(result.get("predictions"), list) else []
        detections: list[dict[str, Any]] = []
        for pred in predictions[:4]:
            if not isinstance(pred, dict):
                continue
            file_name = Path(str(pred.get("path") or "")).name or "image"
            boxes = pred.get("boxes")
            if not isinstance(boxes, list):
                continue
            for box in boxes[:10]:
                if not isinstance(box, dict):
                    continue
                xyxy = box.get("xyxy")
                xyxy_values: list[float] = []
                if isinstance(xyxy, list):
                    for value in xyxy[:4]:
                        try:
                            xyxy_values.append(float(value))
                        except Exception:
                            pass
                detections.append(
                    {
                        "file": file_name,
                        "class_name": box.get("class_name"),
                        "confidence": box.get("confidence"),
                        "xyxy": xyxy_values,
                    }
                )
                if len(detections) >= 20:
                    break
            if len(detections) >= 20:
                break

        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        return {
            "success": True,
            "kind": "yolo",
            "model_name": result.get("model_name"),
            "output_directory": result.get("output_directory"),
            "predictions_json": result.get("predictions_json"),
            "prediction_images": result.get("prediction_images"),
            "prediction_images_raw": result.get("prediction_images_raw"),
            "prediction_image_records": result.get("prediction_image_records"),
            "total_boxes": metrics.get("total_boxes"),
            "avg_confidence": metrics.get("avg_confidence"),
            "finetune_recommended": metrics.get("finetune_recommended"),
            "classes": counts_rows,
            "detections": detections,
            "predictions": result.get("predictions"),
            "analysis_summary": result.get("analysis_summary"),
            "scientific_summary": result.get("scientific_summary"),
            "spatial_analysis": result.get("spatial_analysis"),
            "metadata_context_by_image": result.get("metadata_context_by_image"),
            "inference_configuration": result.get("inference_configuration"),
        }

    if tool_name == "estimate_depth_pro":
        files_processed = result.get("files_processed")
        files_summary: list[dict[str, Any]] = []
        if isinstance(files_processed, list):
            for row in files_processed[:8]:
                if not isinstance(row, dict):
                    continue
                files_summary.append(
                    {
                        "file": row.get("file"),
                        "depth_min": row.get("depth_min"),
                        "depth_max": row.get("depth_max"),
                        "depth_mean": row.get("depth_mean"),
                        "depth_std": row.get("depth_std"),
                        "field_of_view": row.get("field_of_view"),
                        "focal_length": row.get("focal_length"),
                        "success": row.get("success"),
                    }
                )
        return {
            "success": True,
            "kind": "depth_pro",
            "processed": result.get("processed"),
            "total_files": result.get("total_files"),
            "depth_mean_average": result.get("depth_mean_average"),
            "model": result.get("model"),
            "output_directory": result.get("output_directory"),
            "files": files_summary,
        }

    if tool_name == "quantify_segmentation_masks":
        summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
        evaluation = result.get("evaluation") if isinstance(result.get("evaluation"), dict) else {}
        metrics_mean = evaluation.get("metrics_mean") if isinstance(evaluation.get("metrics_mean"), dict) else {}
        rows = result.get("rows") if isinstance(result.get("rows"), list) else []
        return {
            "success": True,
            "kind": "quantify_segmentation_masks",
            "summary": summary,
            "metrics_mean": metrics_mean,
            "row_count": len(rows),
            "evaluation_pairing_fallback_used": result.get("evaluation_pairing_fallback_used"),
        }

    if tool_name == "structure_report":
        functional_groups = result.get("functional_groups") if isinstance(result.get("functional_groups"), dict) else {}
        nonzero_groups = [
            {"name": str(name), "count": int(count)}
            for name, count in functional_groups.items()
            if int(count) > 0
        ][:10]
        rings = result.get("rings") if isinstance(result.get("rings"), dict) else {}
        return {
            "success": True,
            "kind": "chemistry_structure",
            "canonical_smiles": result.get("canonical_smiles"),
            "formula": result.get("formula"),
            "exact_mass": result.get("exact_mass"),
            "degree_of_unsaturation": result.get("degree_of_unsaturation"),
            "ring_count": rings.get("count"),
            "ring_sizes": rings.get("sizes"),
            "bridgehead_atom_count": rings.get("bridgehead_atom_count"),
            "spiro_atom_count": rings.get("spiro_atom_count"),
            "functional_groups": nonzero_groups,
            "strain_flags": result.get("strain_flags"),
        }

    if tool_name == "compare_structures":
        return {
            "success": True,
            "kind": "chemistry_compare",
            "substrate_formula": (
                result.get("substrate", {}).get("formula")
                if isinstance(result.get("substrate"), dict)
                else None
            ),
            "product_formula": (
                result.get("product", {}).get("formula")
                if isinstance(result.get("product"), dict)
                else None
            ),
            "formula_delta": result.get("formula_delta_product_minus_substrate"),
            "functional_group_delta": result.get("functional_group_delta_product_minus_substrate"),
            "ring_size_delta": result.get("ring_size_delta_product_minus_substrate"),
            "heuristic_transformation_labels": result.get("heuristic_transformation_labels"),
            "mcs": result.get("mcs"),
        }

    if tool_name == "propose_reactive_sites":
        return {
            "success": True,
            "kind": "chemistry_reactive_sites",
            "canonical_smiles": result.get("canonical_smiles"),
            "condition_classes": result.get("condition_classes"),
            "global_motifs": result.get("global_motifs"),
            "top_reactive_sites": result.get("top_reactive_sites"),
            "strain_flags": result.get("strain_flags"),
        }

    if tool_name == "formula_balance_check":
        return {
            "success": True,
            "kind": "chemistry_balance",
            "balanced": bool(result.get("balanced")),
            "reactant_element_counts": result.get("reactant_element_counts"),
            "product_element_counts": result.get("product_element_counts"),
            "product_minus_reactant": result.get("product_minus_reactant"),
        }

    if tool_name in {"search_bisque_resources", "bisque_advanced_search"}:
        resources = result.get("resources")
        rows = _resource_rows_for_summary(resources, limit=12)
        query = result.get("query") if isinstance(result.get("query"), dict) else {}
        resource_type = str(query.get("resource_type") or "").strip() or None
        payload = {
            "success": True,
            "kind": "bisque_search",
            "count": _coerce_result_count(result, len(resources) if isinstance(resources, list) else 0),
            "resource_type": resource_type,
            "rows": rows,
        }
        payload.update(_bisque_query_scope_summary(query))
        return payload

    if tool_name == "bisque_find_assets":
        resources = result.get("resources")
        metadata = result.get("metadata")
        downloads = result.get("downloads")
        downloads_list = downloads if isinstance(downloads, list) else []
        query = result.get("query") if isinstance(result.get("query"), dict) else {}
        payload = {
            "success": True,
            "kind": "bisque_find_assets",
            "count": _coerce_result_count(result, len(resources) if isinstance(resources, list) else 0),
            "rows": _resource_rows_for_summary(resources, limit=12),
            "metadata_loaded": len(metadata) if isinstance(metadata, list) else 0,
            "downloads_total": len(downloads_list),
            "downloads_success": sum(
                1
                for item in downloads_list
                if isinstance(item, dict) and bool(item.get("success"))
            ),
            "download_rows": _download_rows_for_summary(downloads_list, limit=12),
        }
        payload.update(_bisque_query_scope_summary(query))
        return payload

    if tool_name == "load_bisque_resource":
        resource = result.get("resource") if isinstance(result.get("resource"), dict) else {}
        tags = resource.get("tags") if isinstance(resource.get("tags"), list) else []
        dimensions = resource.get("dimensions") if isinstance(resource.get("dimensions"), dict) else {}
        rows = _resource_rows_for_summary(
            [
                {
                    **resource,
                    "resource_uri": resource.get("uri"),
                    "client_view_url": result.get("view_url"),
                    "image_service_url": result.get("image_service_url"),
                }
            ],
            limit=1,
        )
        return {
            "success": True,
            "kind": "bisque_metadata",
            "rows": rows,
            "tag_count": len(tags),
            "dimensions": dimensions,
        }

    if tool_name == "bisque_download_resource":
        return {
            "success": True,
            "kind": "bisque_download",
            "download_rows": _download_rows_for_summary([result], limit=1),
        }

    if tool_name == "add_tags_to_resource":
        resource_uri = str(result.get("resource_uri") or "").strip() or None
        tags_added = result.get("tags_added") if isinstance(result.get("tags_added"), list) else []
        return {
            "success": True,
            "kind": "bisque_tags",
            "resource_uri": resource_uri,
            "tag_count": result.get("total_tags"),
            "tags": tags_added[:20],
            "rows": _resource_rows_for_summary(
                [{"name": "Tagged resource", "resource_uri": resource_uri, "uri": resource_uri}],
                limit=1,
            ),
        }

    if tool_name == "bisque_fetch_xml":
        resource_uri = str(result.get("resource_uri") or "").strip() or None
        saved_path = str(result.get("saved_path") or "").strip() or None
        return {
            "success": True,
            "kind": "bisque_xml",
            "resource_uri": resource_uri,
            "saved_path": saved_path,
            "truncated": bool(result.get("truncated")),
            "rows": _resource_rows_for_summary(
                [{"name": "XML source", "resource_uri": resource_uri, "uri": resource_uri}],
                limit=1,
            ),
        }

    if tool_name == "bisque_download_dataset":
        results = result.get("results") if isinstance(result.get("results"), list) else []
        return {
            "success": True,
            "kind": "bisque_dataset_download",
            "dataset_uri": str(result.get("dataset_uri") or "").strip() or None,
            "output_dir": str(result.get("output_dir") or "").strip() or None,
            "total_members": result.get("total_members"),
            "downloaded": result.get("downloaded"),
            "download_rows": _download_rows_for_summary(results, limit=12),
        }

    if tool_name in {"bisque_create_dataset", "bisque_add_to_dataset"}:
        return {
            "success": True,
            "kind": "bisque_dataset_write",
            "action": result.get("action"),
            "dataset_name": result.get("dataset_name"),
            "dataset_uri": result.get("dataset_uri"),
            "dataset_client_view_url": str(result.get("dataset_client_view_url") or "").strip() or None,
            "members": result.get("members"),
            "added": result.get("added"),
            "total_resources": result.get("total_resources"),
            "rows": _resource_rows_for_summary(
                [
                    {
                        "name": result.get("dataset_name") or "dataset",
                        "resource_type": "dataset",
                        "resource_uri": result.get("dataset_uri"),
                        "client_view_url": result.get("dataset_client_view_url"),
                        "uri": result.get("dataset_client_view_url") or result.get("dataset_uri"),
                    }
                ],
                limit=1,
            ),
        }

    if tool_name == "bisque_add_gobjects":
        added = result.get("added") if isinstance(result.get("added"), list) else []
        counts_by_type: dict[str, int] = {}
        for item in added:
            if not isinstance(item, dict):
                continue
            name = str(item.get("type") or "annotation").strip() or "annotation"
            counts_by_type[name] = counts_by_type.get(name, 0) + 1
        return {
            "success": bool(result.get("success")),
            "kind": "bisque_annotations",
            "resource_uri": str(result.get("resource_uri") or "").strip() or None,
            "added_total": len(added),
            "counts_by_type": counts_by_type,
            "verified": result.get("verified"),
            "verification_attempts": result.get("verification_attempts"),
            "rows": _resource_rows_for_summary(
                [
                    {
                        "name": "Annotated resource",
                        "resource_uri": result.get("resource_uri"),
                        "uri": result.get("resource_uri"),
                    }
                ],
                limit=1,
            ),
            "error": str(result.get("error") or "").strip() or None,
        }

    if tool_name == "run_bisque_module":
        output_path = str(
            result.get("output_path") or result.get("output_local_path") or ""
        ).strip()
        output_resource_uri = str(result.get("output_resource_uri") or "").strip()
        output_client_view_url = str(result.get("output_client_view_url") or "").strip()
        output_image_service_url = str(result.get("output_image_service_url") or "").strip()
        return {
            "success": True,
            "kind": "bisque_module",
            "module_name": result.get("module_name"),
            "status": result.get("status"),
            "output_path": output_path or None,
            "output_resource_uri": output_resource_uri or None,
            "output_client_view_url": output_client_view_url or None,
            "output_image_service_url": output_image_service_url or None,
            "downloaded_output": bool(output_path),
        }

    if tool_name == "execute_python_job":
        output_files_raw = result.get("output_files")
        output_files: list[str] = []
        if isinstance(output_files_raw, list):
            output_files = [str(item) for item in output_files_raw if str(item or "").strip()][:20]

        expected_outputs_raw = result.get("expected_outputs")
        expected_outputs: list[str] = []
        if isinstance(expected_outputs_raw, list):
            expected_outputs = [
                str(item) for item in expected_outputs_raw if str(item or "").strip()
            ][:20]

        missing_expected_raw = result.get("missing_expected_outputs")
        missing_expected_outputs: list[str] = []
        if isinstance(missing_expected_raw, list):
            missing_expected_outputs = [
                str(item) for item in missing_expected_raw if str(item or "").strip()
            ][:20]

        key_measurements_raw = result.get("key_measurements")
        key_measurements: list[dict[str, Any]] = []
        if isinstance(key_measurements_raw, list):
            for item in key_measurements_raw[:16]:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                value = item.get("value")
                if not name or value is None:
                    continue
                key_measurements.append({"name": name, "value": value})

        analysis_outputs_raw = result.get("analysis_outputs")
        analysis_outputs: list[dict[str, Any]] = []
        if isinstance(analysis_outputs_raw, list):
            for item in analysis_outputs_raw[:8]:
                if not isinstance(item, dict):
                    continue
                analysis_outputs.append(
                    {
                        "path": item.get("path"),
                        "kind": item.get("kind"),
                        "parse_status": item.get("parse_status"),
                        "top_level_keys": item.get("top_level_keys"),
                        "measurement_count": item.get("measurement_count"),
                    }
                )

        attempt_history = result.get("attempt_history")
        attempts = len(attempt_history) if isinstance(attempt_history, list) else 1
        return {
            "success": True,
            "kind": "code_execution",
            "job_id": result.get("job_id"),
            "durable_execution": result.get("durable_execution"),
            "durable_run_id": result.get("durable_run_id"),
            "exit_code": result.get("exit_code"),
            "runtime_seconds": result.get("runtime_seconds"),
            "attempts": attempts,
            "repair_cycles_used": result.get("repair_cycles_used"),
            "output_files": output_files,
            "expected_outputs": expected_outputs,
            "missing_expected_outputs": missing_expected_outputs,
            "measurements": key_measurements,
            "analysis_outputs": analysis_outputs,
        }

    return None


def _push_ui_artifacts(artifacts: list[dict[str, Any]]) -> None:
    del artifacts


def _set_tool_status(label: str | None, state: str) -> None:
    del label, state


_STATUS_LABELS = {
    "upload_to_bisque": "Uploading to BisQue",
    "bisque_ping": "Checking BisQue",
    "bisque_download_resource": "Downloading from BisQue",
    "bisque_find_assets": "Searching BisQue",
    "search_bisque_resources": "Searching BisQue",
    "load_bisque_resource": "Loading metadata",
    "delete_bisque_resource": "Deleting resource",
    "add_tags_to_resource": "Adding metadata tags",
    "bisque_fetch_xml": "Fetching XML",
    "bisque_download_dataset": "Downloading dataset",
    "bisque_create_dataset": "Creating dataset",
    "bisque_add_to_dataset": "Updating dataset",
    "bisque_add_gobjects": "Adding annotations",
    "bisque_advanced_search": "Querying BisQue",
    "run_bisque_module": "Running BisQue module",
    "bioio_load_image": "Loading scientific image",
    "segment_image_megaseg": "Segmenting with Megaseg",
    "segment_image_sam3": "Segmenting with SAM3",
    "estimate_depth_pro": "Estimating depth",
    "segment_evaluate_batch": "Running segmentation + evaluation workflow",
    "evaluate_segmentation_masks": "Evaluating segmentation masks",
    "quantify_segmentation_masks": "Quantifying segmentation masks",
    "segment_image_sam2": "Segmenting with MedSAM2",
    "sam2_prompt_image": "Prompting MedSAM2",
    "segment_video_sam2": "Tracking video with SAM-family model",
    "yolo_list_finetuned_models": "Listing YOLO models",
    "yolo_detect": "Running YOLO detection",
    "yolo_finetune_detect": "Finetuning YOLO detector",
    "structure_report": "Inspecting molecular structure",
    "compare_structures": "Comparing candidate structures",
    "propose_reactive_sites": "Scoring reactive sites",
    "formula_balance_check": "Checking formula balance",
    "codegen_python_plan": "Generating Python analysis plan",
    "execute_python_job": "Executing Python analysis in sandbox",
}


def _tool_status_label(tool_name: str) -> str:
    return _STATUS_LABELS.get(tool_name, f"Running {tool_name}")


def _coerce_tool_result(raw_result: Any, *, tool_name: str) -> tuple[dict[str, Any], str]:
    if isinstance(raw_result, str):
        try:
            parsed = json.loads(raw_result)
        except Exception:
            parsed = {
                "success": False,
                "error": f"Tool {tool_name} returned non-JSON output.",
                "raw": _truncate_text(str(raw_result), 4000),
            }
        if isinstance(parsed, dict):
            return parsed, raw_result
        wrapped = {"success": True, "result": parsed}
        return wrapped, json.dumps(wrapped, ensure_ascii=False, default=str)

    if isinstance(raw_result, dict):
        return raw_result, json.dumps(raw_result, ensure_ascii=False, default=str)

    wrapped = {"success": True, "result": raw_result}
    return wrapped, json.dumps(wrapped, ensure_ascii=False, default=str)


def _update_tool_state(state: ToolState, tool_name: str, result: Any) -> None:
    if not isinstance(result, dict):
        return
    refs = result.get("latest_result_refs")
    if isinstance(refs, dict):
        for key, value in refs.items():
            if value is None:
                continue
            state.latest_result_refs[str(key)] = value
    if tool_name in {"search_bisque_resources", "bisque_advanced_search"}:
        resources = result.get("resources") or []
        if isinstance(resources, list):
            state.last_search = resources[:20]
    if tool_name in {"load_bisque_resource"}:
        resource = result.get("resource") or {}
        if isinstance(resource, dict):
            state.last_resource = resource.get("uri") or state.last_resource
    if tool_name in {"bisque_download_resource"}:
        output_path = result.get("output_path")
        if output_path:
            state.last_downloads.append(str(output_path))
    if tool_name in {"yolo_list_finetuned_models"}:
        models = result.get("models") or []
        if isinstance(models, list):
            state.last_yolo_models = models[:10]
    if tool_name in {"yolo_finetune_detect"}:
        model_name = result.get("model_name")
        model_path = result.get("model_path")
        if model_name or model_path:
            state.last_yolo_models = [
                {"model_name": model_name, "model_path": model_path}
            ]
    if tool_name in {"yolo_detect"}:
        model_name = result.get("model_name")
        model_path = result.get("model_path")
        if model_name or model_path:
            state.notes.append(f"YOLO detection used model {model_name or model_path}.")
    if tool_name in {"estimate_depth_pro"}:
        model_ref = result.get("model")
        if model_ref:
            state.notes.append(f"Depth estimation used model {model_ref}.")
    if tool_name in {"segment_image_sam2", "segment_image_sam3", "sam2_prompt_image"}:
        mask_paths: list[str] = []

        preferred_upload_paths = result.get("preferred_upload_paths")
        if isinstance(preferred_upload_paths, list):
            for item in preferred_upload_paths:
                token = str(item or "").strip()
                if token:
                    mask_paths.append(token)

        files_processed = result.get("files_processed")
        if isinstance(files_processed, list):
            for row in files_processed:
                if not isinstance(row, dict):
                    continue
                for key in ("preferred_upload_path", "mask_path", "mask_volume_path"):
                    token = str(row.get(key) or "").strip()
                    if token:
                        mask_paths.append(token)

        deduped_mask_paths: list[str] = []
        seen_paths: set[str] = set()
        for token in mask_paths:
            if token in seen_paths:
                continue
            seen_paths.add(token)
            deduped_mask_paths.append(token)

        if deduped_mask_paths:
            state.latest_result_refs[f"{tool_name}.mask_paths"] = deduped_mask_paths[:24]
            state.latest_result_refs[f"{tool_name}.preferred_upload_paths"] = (
                deduped_mask_paths[:24]
            )
            state.latest_result_refs[f"{tool_name}.latest_mask_path"] = deduped_mask_paths[0]
            state.latest_result_refs["latest_segmentation_mask_paths"] = deduped_mask_paths[:24]
            state.latest_result_refs["latest_segmentation_mask_path"] = deduped_mask_paths[0]
            state.latest_result_refs["latest_mask_path"] = deduped_mask_paths[0]
    if result.get("message"):
        state.notes.append(str(result.get("message")))


class _ScratchpadWriter:
    def __init__(self, path: str, question: str | None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.reset(question)

    def reset(self, question: str | None) -> None:
        if self.path.exists():
            try:
                existing = self.path.read_text(encoding="utf-8")
            except Exception:
                existing = ""
            if existing.strip():
                return
        content = "# Scratchpad\n\n"
        if question:
            content += "## User question\n" + question.strip() + "\n"
        self.path.write_text(content, encoding="utf-8")

    def append_section(self, title: str, body: str | None) -> None:
        if not body:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n## {title}\n")
            handle.write(body.rstrip() + "\n")


def _tool_name_set(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return names


def _tool_call_fingerprint(tool_name: str, tool_args: Any) -> str:
    args_value: Any = tool_args
    if isinstance(tool_args, str):
        token = tool_args.strip()
        if token:
            try:
                args_value = json.loads(token)
            except Exception:
                args_value = token
        else:
            args_value = ""
    try:
        args_text = json.dumps(args_value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        args_text = str(args_value)
    return f"{str(tool_name or '').strip()}|{args_text}"


def _explicit_requested_tool_names(user_text: str, tool_names: set[str]) -> set[str]:
    text = str(user_text or "").lower()
    found: set[str] = set()
    for name in tool_names:
        lowered = name.lower()
        if lowered in text:
            found.add(name)
    # Semantic aliasing for BisQue module execution requests where users
    # describe the action (e.g., edge detection) without naming the tool id.
    if "run_bisque_module" in tool_names and any(
        token in text
        for token in (
            "edge detection",
            "canny",
            "edge map",
            "run module",
            "bisque module",
            "module on this image",
        )
    ):
        found.add("run_bisque_module")
    return found


def _latest_user_question(messages: list[dict[str, Any]]) -> str | None:
    for msg in reversed(messages):
        if msg.get("role") == "user" and msg.get("content"):
            return str(msg.get("content"))
    return None


def _format_timestamp(ts: str | None) -> str | None:
    if not ts:
        return None
    try:
        norm = str(ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(norm)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 10].rstrip() + "\n\n[truncated]"


def _summarize_tool_output(tool_name: str, result: Any) -> str:
    if not isinstance(result, dict):
        return f"- Result: {str(result)[:500]}"

    if result.get("success") is False:
        return f"- Error: {result.get('error') or 'Tool failed.'}"

    if tool_name in {"search_bisque_resources", "bisque_advanced_search"}:
        resources = result.get("resources") or []
        query = result.get("query") if isinstance(result.get("query"), dict) else {}
        count = _coerce_result_count(result, len(resources) if isinstance(resources, list) else 0)
        lines = [f"- Matches: {count}"]
        scope_text = _bisque_query_scope_text(query)
        if scope_text:
            lines.append(f"- Search scope: {scope_text}")
        rows = _resource_rows_for_summary(resources, limit=6)
        if rows:
            lines.extend(
                [
                    "",
                    "| Name | Owner | Created | URI |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for row in rows:
                lines.append(
                    "| "
                    + f"{row.get('name') or '-'} | "
                    + f"{row.get('owner') or '-'} | "
                    + f"{row.get('created') or '-'} | "
                    + f"{row.get('uri') or '-'} |"
                )
        return "\n".join(lines)

    if tool_name == "bisque_find_assets":
        resources = result.get("resources") or []
        meta = result.get("metadata") or []
        downloads = result.get("downloads") or []
        query = result.get("query") if isinstance(result.get("query"), dict) else {}
        lines = [f"- Matches: {_coerce_result_count(result, len(resources) if isinstance(resources, list) else 0)}"]
        scope_text = _bisque_query_scope_text(query)
        if scope_text:
            lines.append(f"- Search scope: {scope_text}")
        if meta:
            lines.append(f"- Metadata loaded: {len(meta)}")
        if downloads:
            ok = sum(1 for d in downloads if d.get("success"))
            lines.append(f"- Downloads: {ok}/{len(downloads)} succeeded")
        rows = _resource_rows_for_summary(resources, limit=6)
        if rows:
            lines.extend(
                [
                    "",
                    "| Name | Owner | Created | URI |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for row in rows:
                lines.append(
                    "| "
                    + f"{row.get('name') or '-'} | "
                    + f"{row.get('owner') or '-'} | "
                    + f"{row.get('created') or '-'} | "
                    + f"{row.get('uri') or '-'} |"
                )
        return "\n".join(lines)

    if tool_name == "load_bisque_resource":
        res = result.get("resource") or {}
        lines = []
        name = res.get("name") or res.get("uri") or "resource"
        created = _format_timestamp(res.get("created"))
        if created:
            lines.append(f"- {name} — {created}")
        else:
            lines.append(f"- {name}")
        tags = res.get("tags") or []
        if tags:
            lines.append(f"- Tags: {len(tags)}")
        dims = res.get("dimensions") or {}
        if dims:
            dims_text = ", ".join(f"{k}={v}" for k, v in dims.items() if v is not None)
            if dims_text:
                lines.append(f"- Dimensions: {dims_text}")
        return "\n".join(lines) if lines else "- Loaded resource."

    if tool_name == "bisque_download_resource":
        path = result.get("output_path")
        uri = result.get("resource_uri")
        if path and uri:
            return f"- Downloaded `{path}` from `{uri}`."
        if path:
            return f"- Downloaded `{path}`."
        return "- Download completed."

    if tool_name == "run_bisque_module":
        lines: list[str] = []
        module_name = result.get("module_name")
        if module_name:
            lines.append(f"- Module: {module_name}")
        status = result.get("status")
        if status:
            lines.append(f"- Status: {status}")
        mex_url = result.get("mex_url")
        if mex_url:
            lines.append(f"- MEX: {mex_url}")
        output_resource_uri = str(result.get("output_resource_uri") or "").strip() or None
        output_client_view_url = _derive_bisque_client_view_url(
            output_resource_uri,
            str(result.get("output_client_view_url") or "").strip() or None,
        )
        if output_client_view_url or output_resource_uri:
            lines.append(f"- Output resource: {output_client_view_url or output_resource_uri}")
        output_path = result.get("output_path") or result.get("output_local_path")
        if output_path:
            lines.append(f"- Output file: {output_path}")
        if lines:
            return "\n".join(lines)
        return "- Module execution completed."

    if tool_name == "yolo_list_finetuned_models":
        models = result.get("models") or []
        lines = [f"- Local finetuned models: {len(models)}"]
        for model in models[:5]:
            name = model.get("model_name") or "model"
            created = _format_timestamp(model.get("created_at_utc"))
            detail = name
            if created:
                detail += f" — {created}"
            lines.append(f"- {detail}")
        return "\n".join(lines)

    if tool_name == "codegen_python_plan":
        lines: list[str] = []
        entrypoint = str(result.get("entrypoint") or "").strip()
        if entrypoint:
            lines.append(f"- Entrypoint: {entrypoint}")
        command = str(result.get("command") or "").strip()
        if command:
            lines.append(f"- Command: {command}")
        dependencies = result.get("dependencies")
        if isinstance(dependencies, list) and dependencies:
            lines.append(
                "- Dependencies: " + ", ".join(str(item) for item in dependencies[:10])
            )
        expected_outputs = result.get("expected_outputs")
        if isinstance(expected_outputs, list) and expected_outputs:
            lines.append(
                "- Expected outputs: "
                + ", ".join(str(item) for item in expected_outputs[:10])
            )
        attempt_index = result.get("attempt_index")
        if isinstance(attempt_index, int):
            lines.append(f"- Attempt index: {attempt_index}")
        return "\n".join(lines) if lines else "- Python job specification prepared."

    if tool_name == "execute_python_job":
        lines: list[str] = []
        runtime_seconds = result.get("runtime_seconds")
        if isinstance(runtime_seconds, (int, float)):
            lines.append(f"- Runtime (s): {float(runtime_seconds):.3f}")
        exit_code = result.get("exit_code")
        if isinstance(exit_code, int):
            lines.append(f"- Exit code: {exit_code}")
        repair_cycles_used = result.get("repair_cycles_used")
        if isinstance(repair_cycles_used, int):
            lines.append(f"- Repair cycles used: {repair_cycles_used}")

        output_files = result.get("output_files")
        if isinstance(output_files, list) and output_files:
            lines.append(f"- Output files: {len(output_files)}")
            for output_file in output_files[:8]:
                text = str(output_file or "").strip()
                if text:
                    lines.append(f"- Output: {text}")

        key_measurements = result.get("key_measurements")
        if isinstance(key_measurements, list) and key_measurements:
            lines.append("- Key quantitative results:")
            for item in key_measurements[:10]:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                value = item.get("value")
                if not name or value is None:
                    continue
                if isinstance(value, float):
                    lines.append(f"- {name}: {value:.6g}")
                else:
                    lines.append(f"- {name}: {value}")

        missing_expected_outputs = result.get("missing_expected_outputs")
        if isinstance(missing_expected_outputs, list) and missing_expected_outputs:
            lines.append(
                "- Missing expected outputs: "
                + ", ".join(str(item) for item in missing_expected_outputs[:10])
            )

        analysis_outputs = result.get("analysis_outputs")
        if isinstance(analysis_outputs, list) and analysis_outputs:
            parsed_ok = sum(
                1
                for item in analysis_outputs
                if isinstance(item, dict) and str(item.get("parse_status") or "") == "ok"
            )
            lines.append(
                f"- Parsed JSON output summaries: {parsed_ok}/{len(analysis_outputs)}"
            )
        return "\n".join(lines) if lines else "- Python sandbox execution completed."

    if tool_name in {"segment_image_sam3", "segment_image_sam2", "sam2_prompt_image"}:
        files = result.get("files_processed") or []
        coverage_mean = result.get("coverage_percent_mean")
        coverage_min = result.get("coverage_percent_min")
        coverage_max = result.get("coverage_percent_max")
        instance_coverage_mean = result.get("instance_coverage_percent_mean")
        instance_coverage_min = result.get("instance_coverage_percent_min")
        instance_coverage_max = result.get("instance_coverage_percent_max")
        instance_count_reported_total = result.get("instance_count_reported_total")
        instance_count_measured_total = result.get("instance_count_measured_total")
        instance_count_mismatch_files = result.get("instance_count_mismatch_files")
        total_masks = result.get("total_masks_generated")
        visualizations = result.get("visualization_paths") or []
        lines: list[str] = []
        if isinstance(total_masks, (int, float)):
            lines.append(f"- Reported masks: {int(total_masks)}")
        if isinstance(coverage_mean, (int, float)):
            lines.append(
                f"- Image coverage % (union mask, mean/min/max): {float(coverage_mean):.4f} / "
                f"{float(coverage_min or 0.0):.4f} / {float(coverage_max or 0.0):.4f}"
            )
        if isinstance(instance_coverage_mean, (int, float)):
            lines.append(
                "- Per-instance coverage % (mean/min/max): "
                f"{float(instance_coverage_mean):.4f} / "
                f"{float(instance_coverage_min or 0.0):.4f} / "
                f"{float(instance_coverage_max or 0.0):.4f}"
            )
        if isinstance(instance_count_reported_total, (int, float)) or isinstance(
            instance_count_measured_total, (int, float)
        ):
            lines.append(
                "- Instance counts (reported/measured): "
                f"{int(instance_count_reported_total or 0)} / {int(instance_count_measured_total or 0)}"
            )
        if isinstance(instance_count_mismatch_files, (int, float)) and int(instance_count_mismatch_files) > 0:
            lines.append(
                f"- Files with reported-vs-measured count mismatch: {int(instance_count_mismatch_files)}"
            )
        if isinstance(files, list) and files:
            lines.append(f"- Files processed: {len(files)}")
            for row in files[:5]:
                if not isinstance(row, dict):
                    continue
                if row.get("success") is False:
                    lines.append(f"- {row.get('file') or 'file'}: {row.get('error') or 'failed'}")
                    continue
                reported = row.get("instance_count_reported")
                measured = row.get("instance_count_measured")
                lines.append(
                    f"- {row.get('file') or 'file'}: image_coverage={row.get('coverage_percent')}, "
                    f"reported_masks={row.get('total_masks')}, "
                    f"measured_instances={measured if measured is not None else reported}"
                )
        elif result.get("file"):
            lines.append(f"- File: {result.get('file')}")
            if isinstance(result.get("best_iou_scores"), list):
                lines.append(f"- Prompted score count: {len(result.get('best_iou_scores') or [])}")
        if isinstance(visualizations, list) and visualizations:
            lines.append(f"- Visualizations: {len(visualizations)}")
        return "\n".join(lines) if lines else "- Segmentation completed."

    if tool_name == "estimate_depth_pro":
        files = result.get("files_processed") or []
        depth_mean = result.get("depth_mean_average")
        visualizations = result.get("visualization_paths") or []
        lines: list[str] = []
        if isinstance(files, list) and files:
            lines.append(f"- Files processed: {len(files)}")
            for row in files[:5]:
                if not isinstance(row, dict):
                    continue
                if row.get("success") is False:
                    lines.append(f"- {row.get('file') or 'file'}: {row.get('error') or 'failed'}")
                    continue
                lines.append(
                    f"- {row.get('file') or 'file'}: mean={row.get('depth_mean')}, "
                    f"range=({row.get('depth_min')}, {row.get('depth_max')})"
                )
        if isinstance(depth_mean, (int, float)):
            lines.append(f"- Mean depth (average across files): {float(depth_mean):.6f}")
        if isinstance(visualizations, list) and visualizations:
            lines.append(f"- Visualizations: {len(visualizations)}")
        return "\n".join(lines) if lines else "- Depth estimation completed."

    generic_lines: list[str] = []
    scalar_keys: list[tuple[str, str]] = [
        ("status", "Status"),
        ("message", "Message"),
        ("count", "Count"),
        ("processed", "Processed"),
        ("total_files", "Total files"),
        ("total", "Total"),
        ("duration_seconds", "Duration (s)"),
    ]
    for key, label in scalar_keys:
        value = result.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        generic_lines.append(f"- {label}: {value}")

    path_keys: list[tuple[str, str]] = [
        ("resource_uri", "Resource"),
        ("output_resource_uri", "Output resource"),
        ("client_view_url", "Client view"),
        ("output_client_view_url", "Output BisQue view"),
        ("image_service_url", "Image service"),
        ("mex_url", "MEX"),
        ("output_path", "Output file"),
        ("output_local_path", "Output local file"),
        ("output_directory", "Output directory"),
    ]
    for key, label in path_keys:
        value = str(result.get(key) or "").strip()
        if not value:
            continue
        if key == "client_view_url":
            derived = _derive_bisque_client_view_url(
                str(result.get("resource_uri") or "").strip() or None,
                value,
            )
            if derived and str(result.get("resource_uri") or "").strip():
                continue
        if key == "output_client_view_url":
            derived = _derive_bisque_client_view_url(
                str(result.get("output_resource_uri") or "").strip() or None,
                value,
            )
            if derived and str(result.get("output_resource_uri") or "").strip():
                continue
        display_value = value
        if key == "resource_uri":
            display_value = _derive_bisque_client_view_url(
                value,
                str(result.get("client_view_url") or "").strip() or None,
            ) or value
        elif key == "output_resource_uri":
            display_value = _derive_bisque_client_view_url(
                value,
                str(result.get("output_client_view_url") or "").strip() or None,
            ) or value
        generic_lines.append(f"- {label}: {display_value}")

    list_keys: list[tuple[str, str]] = [
        ("resources", "Resources"),
        ("downloads", "Downloads"),
        ("files_processed", "Files processed"),
        ("visualization_paths", "Visualizations"),
        ("prediction_images", "Prediction images"),
        ("ui_artifacts", "Artifacts"),
    ]
    for key, label in list_keys:
        value = result.get(key)
        if isinstance(value, list) and value:
            generic_lines.append(f"- {label}: {len(value)}")

    warnings = result.get("warnings")
    if isinstance(warnings, list) and warnings:
        generic_lines.append(f"- Warnings: {len(warnings)}")
        for item in warnings[:2]:
            text = str(item).strip()
            if text:
                generic_lines.append(f"- Warning detail: {text}")

    metrics = result.get("metrics")
    if isinstance(metrics, dict) and metrics:
        metric_parts: list[str] = []
        for key, value in metrics.items():
            if len(metric_parts) >= 5:
                break
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                metric_parts.append(f"{key}={value}")
        if metric_parts:
            generic_lines.append("- Metrics: " + ", ".join(metric_parts))

    if generic_lines:
        return "\n".join(generic_lines)

    return "- Completed successfully."


def _auto_draft_from_state(state: ToolState, last_error: str | None) -> str:
    lines: list[str] = []
    if state.last_search:
        lines.append(f"Found {len(state.last_search)} matching asset(s).")
        for res in state.last_search[:5]:
            name = res.get("name") or res.get("uri") or "resource"
            created = _format_timestamp(res.get("created"))
            detail = f"{name}"
            if created:
                detail += f" — {created}"
            lines.append(f"- {detail}")
    if state.last_resource and not state.last_search:
        lines.append(f"Loaded metadata for {state.last_resource}.")
    if state.last_downloads:
        lines.append("Downloaded files:")
        for path in state.last_downloads[-3:]:
            lines.append(f"- {path}")
    if not lines:
        if last_error:
            lines.append(f"Tool execution failed: {last_error}")
        else:
            lines.append("No tool results were returned.")
    return "\n".join(lines)


def _refine_from_scratchpad(
    *,
    client: Any,
    model: str,
    scratchpad_path: Path,
    fallback_text: str | None,
    error_text: str | None,
) -> Iterator[str]:
    try:
        scratchpad_text = scratchpad_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("Failed to read scratchpad: %s", exc)
        if error_text:
            yield f"Tool execution failed: {error_text}"
        elif fallback_text:
            yield fallback_text
        else:
            yield "Unable to generate a response."
        return

    system_prompt = (
        "You are a scientific assistant. Read the provided context and return exactly one strict JSON object.\n"
        "Output contract schema (all keys required):\n"
        "{\n"
        '  "result": string,\n'
        '  "evidence": array,\n'
        '  "measurements": array,\n'
        '  "statistical_analysis": array,\n'
        '  "confidence": {"level":"low|medium|high","why":[string,...]},\n'
        '  "qc_warnings": [string,...],\n'
        '  "limitations": [string,...],\n'
        '  "next_steps": [{"action": string}, ...]\n'
        "}\n"
        "Quality rules:\n"
        "- Do NOT mention scratchpad, internal logs, or tool IDs.\n"
        "- The result must summarize what was completed, what outputs were produced, and any missing/failed pieces.\n"
        "- Assess output quality from available signals (status, warnings, errors, metric completeness, artifact availability).\n"
        "- Put concrete quality concerns in qc_warnings and scientific/operational constraints in limitations.\n"
        "- Use exact filenames/URIs when available; do not expose private local filesystem roots.\n"
        "- Include at least 1 evidence item and at least 1 next_steps action.\n"
        "- If equations are helpful, render them in LaTeX. Use valid LaTeX only with explicit delimiters: inline \\\\(...\\\\) and display \\\\[...\\\\]. Avoid pseudo-LaTeX, unmatched delimiters, or bare bracket-wrapped math. Keep long chemical or IUPAC names in ordinary prose rather than math mode; if plain text must appear inside math, use \\\\text{...}.\n"
        "- When using LaTeX in JSON strings, escape backslashes (for example use \\\\mu, \\\\sigma, \\\\frac{a}{b}).\n"
        "- Keep notation consistent and define symbols on first use.\n"
        "- Keep text concise, actionable, and user-facing.\n"
        "- Return JSON only (no markdown fences, no prose outside JSON).\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scratchpad_text},
    ]

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
    except Exception as exc:
        logger.error("Refinement failed: %s", exc)
        if fallback_text:
            yield fallback_text
        elif error_text:
            yield f"Tool execution failed: {error_text}"
        else:
            yield "Unable to generate a response."
