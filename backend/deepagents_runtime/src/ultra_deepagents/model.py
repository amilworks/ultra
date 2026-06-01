from __future__ import annotations

from langchain_openai import ChatOpenAI

from ultra_deepagents.config import RuntimeSettings


def build_chat_model(settings: RuntimeSettings) -> ChatOpenAI:
    request_timeout = (
        settings.request_timeout_seconds if settings.request_timeout_seconds > 0 else None
    )
    return ChatOpenAI(
        model=settings.openai_model,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        timeout=request_timeout,
        max_retries=settings.max_retries,
    )
