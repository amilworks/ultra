"""Conversation title helpers for chat surfaces."""

from __future__ import annotations

import re

from src.config import get_settings
from src.llm_client import _resolved_model, get_openai_client


def _latest_user_message(messages: list[dict[str, str]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user" and msg.get("content"):
            return str(msg.get("content"))
    return ""


def _fallback_title_from_messages(messages: list[dict[str, str]], max_words: int) -> str:
    source = _latest_user_message(messages).strip()
    if not source:
        return "New conversation"
    cleaned = re.sub(r"\s+", " ", source).strip().strip("\"'`")
    words = cleaned.split()
    if not words:
        return "New conversation"
    return " ".join(words[:max_words])


def _sanitize_title_text(raw: str, max_words: int) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    text = text.splitlines()[0].strip()
    text = re.sub(r"^(title)\s*[:\\-]\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("\"'`“”‘’")
    text = re.sub(r"\s+", " ", text).strip()
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_'’\\-]*", text)
    if not tokens:
        return ""
    return " ".join(tokens[:max_words]).strip()


def generate_chat_title(
    messages: list[dict[str, str]],
    max_words: int = 4,
) -> tuple[str, str]:
    settings = get_settings()
    if bool(getattr(settings, "llm_mock_mode", False)):
        return _fallback_title_from_messages(messages, max_words=max_words), "fallback"

    client = get_openai_client()
    max_words_clamped = max(2, min(int(max_words or 4), 8))
    fallback = _fallback_title_from_messages(messages, max_words_clamped)

    user_messages = [
        str(item.get("content") or "").strip()
        for item in messages
        if str(item.get("role") or "") == "user" and str(item.get("content") or "").strip()
    ]
    if not user_messages:
        return fallback, "fallback"

    context_block = "\n".join(f"- {entry}" for entry in user_messages[-6:])
    title_prompt = [
        {
            "role": "system",
            "content": (
                "Generate a short conversation title.\n"
                f"Rules: max {max_words_clamped} words, no quotes, no trailing punctuation, title only."
            ),
        },
        {
            "role": "user",
            "content": f"Conversation snippets:\n{context_block}",
        },
    ]

    request_kwargs = {
        "model": _resolved_model(settings),
        "messages": title_prompt,
        "max_tokens": max(8, max_words_clamped * 5),
        "temperature": 0.1,
        "stream": False,
    }

    raw_title = ""
    try:
        completion = client.chat.completions.create(
            **request_kwargs,
            extra_body={"reasoning_effort": "low"},
        )
    except Exception:
        completion = client.chat.completions.create(**request_kwargs)

    try:
        if completion.choices and completion.choices[0].message:
            raw_title = str(completion.choices[0].message.content or "")
    except Exception:
        raw_title = ""

    normalized_title = _sanitize_title_text(raw_title, max_words_clamped)
    if normalized_title:
        return normalized_title, "llm"
    return fallback, "fallback"
