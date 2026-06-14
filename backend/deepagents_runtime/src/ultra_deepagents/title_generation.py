from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.model import build_chat_model


MAX_TITLE_CHARS = 52
TITLE_PROMPT_SYSTEM = (
    "You write compact sidebar titles for a scientific AI workspace. "
    "Return exactly one JSON object with a title field. The title must be "
    "3 to 7 words, <=52 characters, specific to the scientific task, and "
    "must not include quotes, trailing punctuation, dates, or generic words "
    "such as chat, conversation, request, help, analysis task, or discussion."
)


@dataclass(frozen=True)
class ConversationTitleResult:
    title: str
    strategy: str
    model: str
    reason: str = ""

    def to_payload(self) -> dict[str, str]:
        payload = {
            "strategy": self.strategy,
            "model": self.model,
        }
        if self.reason:
            payload["reason"] = self.reason
        return payload


def _run_fallback_text(
    goal: str,
    messages: list[dict[str, Any]],
    response_text: str,
    artifact_events: list[dict[str, Any]],
) -> str:
    return "\n".join(
        fragment
        for fragment in [
            goal,
            _first_user_message(messages),
            response_text,
            _artifact_summary(artifact_events),
        ]
        if fragment.strip()
    )


def is_initial_conversation_turn(messages: list[dict[str, Any]] | None) -> bool:
    """True when the transcript has no assistant turns yet.

    The control plane only applies generated titles to threads whose title
    state is still initial/auto, so once a prior run completed (an assistant
    message exists) a new generated title would be ignored anyway.
    """
    for message in messages or []:
        if isinstance(message, dict):
            role = str(message.get("role") or "")
        else:
            role = str(getattr(message, "role", "") or "")
        if role.strip().lower() == "assistant":
            return False
    return True


def start_conversation_title_task(
    *,
    settings: RuntimeSettings,
    goal: str,
    messages: list[dict[str, Any]],
    model_factory: Callable[[RuntimeSettings], Any] = build_chat_model,
) -> asyncio.Task | None:
    """Begin title generation concurrently with the run itself.

    The sidebar title is derived from the request (goal + user messages),
    which is fully known at run start, so the model call overlaps the run
    instead of extending it. Returns ``None`` when no model call should be
    made: title generation disabled, or a follow-up turn whose generated
    title the control plane would ignore.
    """
    if not getattr(settings, "title_generation_enabled", True):
        return None
    if not is_initial_conversation_turn(messages):
        return None
    snapshot = [dict(message) if isinstance(message, dict) else message for message in messages or []]
    return asyncio.create_task(
        generate_conversation_title(
            settings=settings,
            goal=goal,
            messages=snapshot,
            response_text="",
            artifact_events=[],
            model_factory=model_factory,
        )
    )


async def resolve_conversation_title_task(
    task: asyncio.Task | None,
    *,
    settings: RuntimeSettings,
    goal: str,
    messages: list[dict[str, Any]],
    response_text: str,
    artifact_events: list[dict[str, Any]],
    grace_seconds: float = 2.0,
) -> ConversationTitleResult:
    """Join the early title task at run completion.

    By completion the task has had the entire run to finish, so the small
    grace only matters for runs shorter than the title call itself. A skipped,
    unresolved, or fallback result is recomputed from the full run outcome
    (goal + response + artifacts) so the deterministic title keeps the same
    quality it had when generation ran inline.
    """
    fallback = fallback_conversation_title(
        _run_fallback_text(goal, messages, response_text, artifact_events)
    )
    if task is None:
        reason = (
            "disabled"
            if not getattr(settings, "title_generation_enabled", True)
            else "thread_already_titled"
        )
        return ConversationTitleResult(
            title=fallback,
            strategy="fallback",
            model=settings.openai_model,
            reason=reason,
        )
    try:
        result = await asyncio.wait_for(task, timeout=max(0.0, grace_seconds))
    except TimeoutError:
        task.cancel()
        return ConversationTitleResult(
            title=fallback,
            strategy="fallback",
            model=settings.openai_model,
            reason="early_title_unresolved",
        )
    if result.strategy == "fallback":
        # The early task lacked the run outcome; rebuild its deterministic
        # fallback from the richer post-run context.
        return ConversationTitleResult(
            title=fallback,
            strategy="fallback",
            model=result.model,
            reason=result.reason,
        )
    return result


async def generate_conversation_title(
    *,
    settings: RuntimeSettings,
    goal: str,
    messages: list[dict[str, Any]],
    response_text: str,
    artifact_events: list[dict[str, Any]],
    model_factory: Callable[[RuntimeSettings], Any] = build_chat_model,
) -> ConversationTitleResult:
    fallback = fallback_conversation_title(
        _run_fallback_text(goal, messages, response_text, artifact_events)
    )
    if not getattr(settings, "title_generation_enabled", True):
        return ConversationTitleResult(
            title=fallback,
            strategy="fallback",
            model=settings.openai_model,
            reason="disabled",
        )
    try:
        title = await _call_title_model(
            settings=settings,
            model_factory=model_factory,
            prompt=_title_prompt(
                goal=goal,
                messages=messages,
                response_text=response_text,
                artifact_events=artifact_events,
            ),
        )
    except Exception as exc:  # Title generation must never block run completion.
        return ConversationTitleResult(
            title=fallback,
            strategy="fallback",
            model=settings.openai_model,
            reason=str(exc),
        )
    sanitized = sanitize_generated_title(title)
    if not sanitized:
        return ConversationTitleResult(
            title=fallback,
            strategy="fallback",
            model=settings.openai_model,
            reason="empty_title",
        )
    return ConversationTitleResult(
        title=sanitized,
        strategy="llm",
        model=settings.openai_model,
    )


def fallback_conversation_title(text: str) -> str:
    normalized = _normalize_text(text)
    lowered = normalized.lower()
    if "rarespot" in lowered and "burrow" in lowered:
        return "RareSpot Burrow Quantification"
    if "rarespot" in lowered and "prairie dog" in lowered:
        return "RareSpot Prairie Dog Analysis"
    if "unet" in lowered and ("segmentation" in lowered or "mask" in lowered):
        return "UNet Segmentation Training" if _contains_any(lowered, ("train", "weights", "curves")) else "UNet Segmentation"
    if _contains_any(lowered, ("arxiv", "paper", "papers")) and "attention" in lowered:
        return "Attention Paper Comparison" if _contains_any(lowered, ("compare", "limitations", "table")) else "Attention Paper Review"
    if "matplotlib" in lowered and _contains_any(lowered, ("plot", "figure", "visualize")):
        return "Matplotlib Function Plot"
    if _contains_any(lowered, ("ome-tiff", "ome tiff", "ome-tif")) and "channel" in lowered:
        return "OME-TIFF Channel Drift" if "drift" in lowered else "OME-TIFF Channel Metadata"
    if "bubble sort" in lowered:
        return "Bubble Sort Visualization"
    if "pca" in lowered and _contains_any(lowered, ("iris", "plot", "table")):
        return "PCA Visualization"
    if "ct" in lowered and _contains_any(lowered, ("slice", "scan", "alignment")):
        return "CT Slice Alignment"
    return _keyword_title(normalized)


def sanitize_generated_title(value: str) -> str:
    title = _normalize_text(value)
    if not title:
        return ""
    parsed = _json_title(title)
    if parsed:
        title = parsed
    title = re.sub(r"^(title|conversation title|name)\s*:\s*", "", title, flags=re.IGNORECASE)
    title = title.strip(" \t\r\n\"'`“”‘’.:;,-")
    title = re.sub(r"\s+", " ", title)
    title = _strip_generic_edges(title)
    if not title or title.lower() in {"new conversation", "chat", "conversation"}:
        return ""
    if len(title) <= MAX_TITLE_CHARS:
        return title
    truncated = title[:MAX_TITLE_CHARS].rstrip(" -,:;")
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated.strip() or title[:MAX_TITLE_CHARS].rstrip()


def _build_title_model(
    settings: RuntimeSettings,
    model_factory: Callable[[RuntimeSettings], Any],
) -> Any:
    """The run model, re-bound for a cheap single-purpose title call.

    Hybrid-reasoning models think for thousands of tokens before a 12-token
    title, which is what used to eat the whole title timeout; vLLM-style chat
    templates accept ``chat_template_kwargs.enable_thinking`` to skip that
    phase (servers whose template lacks the variable ignore it; endpoints that
    reject the body land in the existing fallback path). Factories used in
    tests may return plain fakes without ``bind`` — use them as-is.
    """
    model = model_factory(settings)
    bind = getattr(model, "bind", None)
    if bind is None:
        return model
    bind_kwargs: dict[str, Any] = {}
    max_tokens = int(getattr(settings, "title_max_tokens", 0) or 0)
    if max_tokens > 0 and getattr(settings, "title_thinking_disabled", True):
        # Only cap tokens when thinking is off: with thinking on, the cap
        # would truncate mid-reasoning and return an empty title.
        bind_kwargs["max_tokens"] = max_tokens
    if getattr(settings, "title_thinking_disabled", True):
        bind_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    if not bind_kwargs:
        return model
    try:
        return bind(**bind_kwargs)
    except Exception:
        return model


async def _call_title_model(
    *,
    settings: RuntimeSettings,
    model_factory: Callable[[RuntimeSettings], Any],
    prompt: str,
) -> str:
    model = _build_title_model(settings, model_factory)
    messages = [
        {"role": "system", "content": TITLE_PROMPT_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    timeout_seconds = getattr(settings, "title_generation_timeout_seconds", 8.0)
    if timeout_seconds and timeout_seconds > 0:
        async with asyncio.timeout(timeout_seconds):
            response = await _invoke_model(model, messages)
    else:
        response = await _invoke_model(model, messages)
    return _response_content(response)


async def _invoke_model(model: Any, messages: list[dict[str, str]]) -> Any:
    if hasattr(model, "ainvoke"):
        return await model.ainvoke(messages)
    if hasattr(model, "invoke"):
        return await asyncio.to_thread(model.invoke, messages)
    raise TypeError("title model does not support invoke")


def _title_prompt(
    *,
    goal: str,
    messages: list[dict[str, Any]],
    response_text: str,
    artifact_events: list[dict[str, Any]],
) -> str:
    user_messages = [
        _normalize_text(_message_content(message))
        for message in messages
        if str(message.get("role", "")).lower() == "user"
    ]
    sections = [
        ("Goal", goal),
        ("First user message", user_messages[0] if user_messages else ""),
        ("Latest user message", user_messages[-1] if user_messages else ""),
        ("Assistant result", _trim(response_text, 900)),
        ("Artifacts/tools", _artifact_summary(artifact_events)),
    ]
    lines = [f"{label}: {_trim(value, 900)}" for label, value in sections if _normalize_text(value)]
    lines.append('Return JSON only, for example: {"title":"RareSpot Burrow Quantification"}')
    return "\n".join(lines)


def _artifact_summary(artifact_events: list[dict[str, Any]]) -> str:
    summaries: list[str] = []
    for event in artifact_events[:8]:
        payload = event.get("payload") if isinstance(event, dict) else None
        if not isinstance(payload, dict):
            continue
        parts = [
            str(payload.get("kind") or "").strip(),
            str(payload.get("title") or "").strip(),
            str(payload.get("path") or "").strip(),
            str(payload.get("tool_name") or "").strip(),
        ]
        summary = " ".join(part for part in parts if part)
        if summary:
            summaries.append(summary)
    return "; ".join(summaries)


def _first_user_message(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if str(message.get("role", "")).lower() == "user":
            return _message_content(message)
    return ""


def _message_content(message: Any) -> str:
    if isinstance(message, str):
        return message
    content = getattr(message, "content", None)
    if content is not None:
        return _content_text(content)
    if isinstance(message, dict):
        return _content_text(message.get("content"))
    return str(message)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list | tuple):
        return " ".join(_content_text(item) for item in content)
    if isinstance(content, dict):
        return _content_text(content.get("text") or content.get("content") or "")
    return "" if content is None else str(content)


def _response_content(response: Any) -> str:
    content = getattr(response, "content", None)
    if content is not None:
        return _content_text(content)
    return _content_text(response)


def _json_title(text: str) -> str:
    if not text.startswith("{"):
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return ""
    if isinstance(parsed, dict):
        return str(parsed.get("title") or "")
    return ""


def _keyword_title(text: str) -> str:
    normalized = _normalize_text(text)
    tokens = [
        _display_token(token)
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+.-]*|x\^\d+|y\s*=\s*x\^\d+", normalized)
        if _keep_token(token)
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(token)
        if len(deduped) >= 5:
            break
    title = " ".join(deduped).strip()
    return sanitize_generated_title(title) or "New conversation"


def _display_token(token: str) -> str:
    normalized = token.strip()
    acronym = {
        "api": "API",
        "arxiv": "arXiv",
        "bisque": "BisQue",
        "ct": "CT",
        "hdf5": "HDF5",
        "ome": "OME",
        "ome-tiff": "OME-TIFF",
        "pca": "PCA",
        "rarespot": "RareSpot",
        "unet": "UNet",
    }.get(normalized.lower())
    if acronym:
        return acronym
    return normalized[:1].upper() + normalized[1:]


def _keep_token(token: str) -> bool:
    value = token.strip().lower()
    return len(value) >= 3 and value not in {
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "please",
        "create",
        "generate",
        "write",
        "make",
        "show",
        "run",
        "analyze",
        "analysis",
        "summarize",
        "discuss",
        "conversation",
        "chat",
        "uploaded",
        "latest",
    }


def _strip_generic_edges(title: str) -> str:
    generic_prefixes = ("Chat About ", "Conversation About ", "Analysis Of ", "Discussion Of ")
    generic_suffixes = (" Discussion", " Chat", " Conversation", " Request")
    changed = True
    while changed:
        changed = False
        for prefix in generic_prefixes:
            if title.lower().startswith(prefix.lower()):
                title = title[len(prefix) :].strip()
                changed = True
        for suffix in generic_suffixes:
            if title.lower().endswith(suffix.lower()) and len(title) > len(suffix) + 4:
                title = title[: -len(suffix)].strip()
                changed = True
    return title


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _trim(value: str, max_chars: int) -> str:
    text = _normalize_text(value)
    return text if len(text) <= max_chars else f"{text[:max_chars].rstrip()}..."


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)
