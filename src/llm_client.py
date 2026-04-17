"""OpenAI-compatible client helpers for the active backend paths."""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI, OpenAIError

from src.config import get_settings
from src.logger import logger


def _resolved_provider(settings: object) -> str:
    return str(getattr(settings, "llm_provider", "openai"))


def _resolved_base_url(settings: object) -> str:
    value = getattr(settings, "resolved_llm_base_url", None) or getattr(
        settings, "openai_base_url", "http://localhost:8000/v1"
    )
    return str(value)


def _resolved_model(settings: object) -> str:
    value = (
        getattr(settings, "resolved_llm_model", None)
        or getattr(settings, "llm_model", None)
        or getattr(settings, "openai_model", "gpt-oss-120b")
    )
    return str(value)


def _resolved_api_key(settings: object) -> str | None:
    provider = _resolved_provider(settings)
    if hasattr(settings, "resolved_llm_api_key"):
        value = getattr(settings, "resolved_llm_api_key")
        if value:
            return str(value)
        if provider in {"ollama", "vllm"}:
            return "EMPTY"
        return None
    value = getattr(settings, "llm_api_key", None)
    if value:
        return str(value)
    value = getattr(settings, "openai_api_key", None)
    if value:
        return str(value)
    if provider in {"ollama", "vllm"}:
        return "EMPTY"
    return None


@lru_cache
def get_openai_client() -> OpenAI:
    """Return a cached OpenAI-compatible client for the configured backend."""

    settings = get_settings()
    base_url = _resolved_base_url(settings)
    api_key = _resolved_api_key(settings)
    model_name = _resolved_model(settings)
    provider = _resolved_provider(settings)
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=settings.openai_timeout,
        max_retries=settings.openai_max_retries,
    )

    logger.info(
        "OpenAI-compatible client initialized: provider=%s base_url=%s model=%s",
        provider,
        base_url,
        model_name,
    )
    return client


def test_connection() -> tuple[bool, str]:
    """Exercise the configured OpenAI-compatible backend with a tiny request."""

    try:
        settings = get_settings()
        client = get_openai_client()

        logger.info("Testing API connection...")

        response = client.chat.completions.create(
            model=_resolved_model(settings),
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
            stream=False,
        )

        if response.choices:
            logger.info("API connection test successful")
            return True, "✅ Connection successful!"
        logger.warning("API returned empty response")
        return False, "⚠️ API returned empty response"

    except OpenAIError as exc:
        error_msg = f"API Error: {exc}"
        logger.error("Connection test failed: %s", error_msg)
        return False, f"❌ {error_msg}"
    except Exception as exc:
        error_msg = f"Unexpected error: {exc}"
        logger.error("Connection test failed: %s", error_msg)
        return False, f"❌ {error_msg}"


def get_available_models() -> list[str]:
    """Return model ids from the active OpenAI-compatible backend."""

    try:
        client = get_openai_client()
        models = client.models.list()
        model_list = [model.id for model in models.data]
        logger.info("Retrieved %s models from API", len(model_list))
        return model_list
    except Exception as exc:
        logger.error("Failed to retrieve models: %s", exc)
        settings = get_settings()
        return [_resolved_model(settings)]
