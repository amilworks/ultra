from __future__ import annotations

from langchain_openai import ChatOpenAI

from ultra_deepagents.config import RuntimeSettings


def build_chat_model(settings: RuntimeSettings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        timeout=settings.request_timeout_seconds,
        max_retries=settings.max_retries,
    )
