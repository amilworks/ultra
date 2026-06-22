from __future__ import annotations

import logging
from typing import Any

from langchain_openai import ChatOpenAI

from ultra_deepagents.config import RuntimeSettings

logger = logging.getLogger(__name__)


class UltraChatOpenAI(ChatOpenAI):
    """ChatOpenAI that preserves vLLM reasoning stream deltas.

    vLLM serves hybrid-reasoning models with the thinking stream in
    ``choices[].delta.reasoning`` (older reasoning parsers used
    ``reasoning_content``), a field the upstream chunk converter discards.
    Without it the run produces no events for the entire thinking phase. Lift
    the text into the LangChain-conventional
    ``additional_kwargs["reasoning_content"]`` so the runner can surface it.
    """

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> Any:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None:
            return generation_chunk
        reasoning = _chunk_reasoning_text(chunk)
        if reasoning:
            kwargs = generation_chunk.message.additional_kwargs
            kwargs["reasoning_content"] = (
                str(kwargs.get("reasoning_content") or "") + reasoning
            )
        return generation_chunk


def _chunk_reasoning_text(chunk: dict) -> str:
    if not isinstance(chunk, dict):
        return ""
    choices = chunk.get("choices") or (chunk.get("chunk") or {}).get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    delta = choices[0].get("delta")
    if not isinstance(delta, dict):
        return ""
    for key in ("reasoning", "reasoning_content"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def build_chat_model(settings: RuntimeSettings) -> ChatOpenAI:
    request_timeout = (
        settings.request_timeout_seconds if settings.request_timeout_seconds > 0 else None
    )
    stream_chunk_timeout = request_timeout
    model = UltraChatOpenAI(
        model=settings.openai_model,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        timeout=request_timeout,
        stream_chunk_timeout=stream_chunk_timeout,
        max_retries=settings.max_retries,
        # Ask the OpenAI-compatible endpoint to report token usage on the
        # streamed response so the runner can account per-run token spend.
        stream_usage=True,
    )
    # Publish the served model's context window so deepagents summarization uses
    # adaptive fraction-based compaction (85% trigger / 10% keep) instead of the
    # conservative no-profile fallback (170k tokens / keep 6 messages), which on a
    # large-window model compacts long autonomous/batch runs far too early. We set
    # ONLY max_input_tokens (never "structured_output"), so the subagent
    # ToolStrategy auto-retry handoff is unaffected (langchain factory gates
    # ProviderStrategy on profile["structured_output"], not on max_input_tokens).
    if settings.model_max_input_tokens > 0:
        model.profile = {"max_input_tokens": settings.model_max_input_tokens}
    return model


def build_vision_chat_model(settings: RuntimeSettings) -> ChatOpenAI:
    """The vision-reasoner's own Qwen3.6-27B VLM (on-prem vLLM, OpenAI-compatible),
    distinct from the text coordinator's ``build_chat_model``.

    Sampling presets (thinking/reasoning vs precise) are applied PER REQUEST by the
    vision tool via ``.bind`` — this returns the bare transport. ``UltraChatOpenAI``
    already lifts the streamed ``reasoning`` delta, and ``stream_usage`` routes the
    VLM's token spend through the existing per-run accounting (deduped by a distinct
    ``checkpoint_ns`` and grouped by ``model_name``). Like ``build_chat_model`` we set
    ONLY ``max_input_tokens`` on the profile (never ``structured_output``), preserving
    the subagent ToolStrategy auto-retry handoff.
    """
    timeout = (
        settings.qwen_vlm_request_timeout_seconds
        if settings.qwen_vlm_request_timeout_seconds > 0
        else None
    )
    model = UltraChatOpenAI(
        model=settings.qwen_vlm_model,
        base_url=settings.qwen_vlm_base_url,
        api_key=settings.qwen_vlm_api_key,
        timeout=timeout,
        stream_chunk_timeout=timeout,
        max_retries=settings.max_retries,
        stream_usage=True,
    )
    if settings.qwen_vlm_max_input_tokens > 0:
        model.profile = {"max_input_tokens": settings.qwen_vlm_max_input_tokens}
    return model


def build_builder_model(settings: RuntimeSettings) -> ChatOpenAI:
    """The Builder sub-coordinator's loop model - MODEL-AGNOSTIC by design.

    Points at whatever OpenAI-compatible endpoint ``builder_base_url``/``builder_model``
    name (swap Qwen, deepseek, or any other without touching the system). When those are
    unset, the Builder falls back to the coordinator's own ``build_chat_model`` so the
    system still runs with a single model configured. Same UltraChatOpenAI transport
    (reasoning-delta lift + per-run token accounting) as the other model builders.
    """
    if not settings.builder_base_url or not settings.builder_model:
        # A partially-configured endpoint (exactly one of base_url/model set) is almost
        # certainly an operator mistake; fall back, but say so loudly rather than silently
        # running the Builder on the coordinator model.
        if settings.builder_base_url or settings.builder_model:
            logger.warning(
                "Builder model partially configured (base_url=%r, model=%r): BOTH are required "
                "to use a separate endpoint; falling back to the coordinator model.",
                settings.builder_base_url,
                settings.builder_model,
            )
        return build_chat_model(settings)
    timeout = settings.request_timeout_seconds if settings.request_timeout_seconds > 0 else None
    model = UltraChatOpenAI(
        model=settings.builder_model,
        base_url=settings.builder_base_url,
        api_key=settings.builder_api_key,
        timeout=timeout,
        stream_chunk_timeout=timeout,
        max_retries=settings.max_retries,
        stream_usage=True,
    )
    if settings.builder_max_input_tokens > 0:
        model.profile = {"max_input_tokens": settings.builder_max_input_tokens}
    return model
