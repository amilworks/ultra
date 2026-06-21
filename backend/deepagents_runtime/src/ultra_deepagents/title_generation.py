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
    "You write short, specific sidebar titles for a scientific AI workspace. "
    'Return exactly one JSON object: {"title": "..."}. '
    "Title the user's core task or question, not incidental details from the "
    "answer. The title must be 2 to 6 words, Title Case, at most 52 characters, "
    "and name the concrete subject (dataset, file type, method, or organism) "
    "when there is one. Write the title in English. Do not use quotes, trailing "
    "punctuation, dates, or generic words such as chat, conversation, request, "
    "help, or discussion."
)

# Canonical casing for domain terms so titles read consistently regardless of
# how the model (or the deterministic fallback) emitted them.
_ACRONYMS: dict[str, str] = {
    "2d": "2D",
    "3d": "3D",
    "api": "API",
    "arxiv": "arXiv",
    "bisque": "BisQue",
    "csv": "CSV",
    "ct": "CT",
    "dicom": "DICOM",
    "fmri": "fMRI",
    "hdf5": "HDF5",
    "json": "JSON",
    "mri": "MRI",
    "nifti": "NIfTI",
    "npm1": "NPM1",
    "nph": "NPH",
    "ome": "OME",
    "ome-tiff": "OME-TIFF",
    "pca": "PCA",
    "png": "PNG",
    "rarespot": "RareSpot",
    "roi": "ROI",
    "sha": "SHA",
    "tif": "TIF",
    "tiff": "TIFF",
    "unet": "UNet",
    "yolo": "YOLO",
}

# Lowercased mid-title connectors (kept lowercase unless first/last word).
_SMALL_WORDS = frozenset(
    {"a", "an", "and", "as", "at", "but", "by", "for", "in", "of", "on", "or", "the", "to", "via", "vs", "with"}
)

# Structural/filler words that carry no concrete subject. A title made entirely
# of these (e.g. "Image Content Analysis") is generic and worth regenerating
# with the answer once it exists.
_GENERIC_TITLE_WORDS = frozenset(
    {
        "analysis", "analyze", "assessment", "content", "contents", "data", "dataset",
        "describe", "description", "details", "file", "files", "identification", "image",
        "images", "info", "information", "inquiry", "inspection", "overview", "result",
        "results", "review", "summary", "task",
    }
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
    model_factory: Callable[[RuntimeSettings], Any] = build_chat_model,
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
    # Smart upgrade: a request-only title that came out generic (no concrete
    # subject) gets one response-aware regeneration now that the answer exists.
    # Clear prompts already produce specific titles and skip the extra call.
    if response_text.strip() and _is_generic_title(result.title):
        upgraded = await generate_conversation_title(
            settings=settings,
            goal=goal,
            messages=messages,
            response_text=response_text,
            artifact_events=artifact_events,
            model_factory=model_factory,
        )
        if upgraded.strategy == "llm" and not _is_generic_title(upgraded.title):
            return ConversationTitleResult(
                title=upgraded.title,
                strategy="llm",
                model=upgraded.model,
                reason="response_aware_upgrade",
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
    # Reasoning models can wrap the answer in <think>...</think> before the JSON;
    # strip that first so the title parse never trips on the chain-of-thought.
    title = _normalize_text(_strip_reasoning(str(value or "")))
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
    title = _to_title_case(title)
    if len(title) <= MAX_TITLE_CHARS:
        return title
    truncated = title[:MAX_TITLE_CHARS].rstrip(" -,:;")
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated.strip() or title[:MAX_TITLE_CHARS].rstrip()


def _strip_reasoning(text: str) -> str:
    """Remove <think>...</think> reasoning blocks (closed or trailing-unclosed)."""
    cleaned = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*$", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _to_title_case(title: str) -> str:
    words = title.split()
    last = len(words) - 1
    out: list[str] = []
    for index, word in enumerate(words):
        out.append(_title_case_word(word, is_edge=index == 0 or index == last))
    return " ".join(out)


def _title_case_word(word: str, *, is_edge: bool) -> str:
    lowered = word.lower()
    if lowered in _ACRONYMS:
        return _ACRONYMS[lowered]
    # Preserve intentional internal casing or alphanumerics (NPM1, fMRI, x^2).
    if any(ch.isdigit() for ch in word) or any(ch.isupper() for ch in word[1:]):
        return word
    if not is_edge and lowered in _SMALL_WORDS:
        return lowered
    if "-" in word:
        return "-".join(_capitalize_token(part) for part in word.split("-"))
    return _capitalize_token(word)


def _capitalize_token(token: str) -> str:
    lowered = token.lower()
    if lowered in _ACRONYMS:
        return _ACRONYMS[lowered]
    return f"{token[:1].upper()}{token[1:].lower()}" if token else token


def _is_generic_title(title: str) -> bool:
    """True when a title names no concrete subject — only filler words.

    Such a title is what a request-only call produces for a vague prompt
    ("describe this image" -> "Image Content Analysis"); regenerating it with
    the answer usually recovers a specific subject (the modality, dataset, etc.).
    """
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9+/.#-]*", title)
    significant = [word for word in words if word.lower() not in _SMALL_WORDS]
    if not significant:
        return True
    for word in significant:
        lowered = word.lower()
        if lowered in _ACRONYMS:  # NPH, CT, NPM1, BisQue-family acronyms
            return False
        if any(ch.isdigit() for ch in word):
            return False
        if any(ch.isupper() for ch in word[1:]):  # internal caps: BisQue, fMRI
            return False
        if lowered not in _GENERIC_TITLE_WORDS:
            return False
    return True


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
        ("Assistant result", _trim(response_text, 400)),
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
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and parsed.get("title"):
            return str(parsed["title"])
    # The title JSON can be embedded in surrounding prose; find the first object
    # that carries a title field.
    for match in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL):
        try:
            candidate = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("title"):
            return str(candidate["title"])
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
    acronym = _ACRONYMS.get(normalized.lower())
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
