from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from ultra_deepagents.config import RuntimeSettings


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
