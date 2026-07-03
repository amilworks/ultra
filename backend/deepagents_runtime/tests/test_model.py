from langchain_core.messages import AIMessageChunk
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.model import build_chat_model


def test_build_chat_model_uses_openai_compatible_base_url():
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        openai_api_key="EMPTY",
        request_timeout_seconds=42.0,
        max_retries=0,
    )

    model = build_chat_model(settings)

    assert model.openai_api_base == "http://127.0.0.1:8003/v1"
    assert model.model_name == "deepseek_v4"
    assert model.openai_api_key.get_secret_value() == "EMPTY"
    # timeout is now an httpx.Timeout: connect/write/pool are bounded (fail fast on a
    # degraded resolver / slow connect so max_retries can re-dispatch), while read
    # carries the configured request timeout for the (streamed) response body.
    assert model.request_timeout.read == 42.0
    assert model.request_timeout.connect == 10.0
    assert model.stream_chunk_timeout == 42.0
    assert model.max_retries == 0


def test_build_chat_model_disables_request_timeout_when_setting_is_zero():
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        request_timeout_seconds=0.0,
    )

    model = build_chat_model(settings)

    # read is unbounded (0 = no request cap → long streams), but connect stays bounded
    # so a slow DNS/TCP/TLS setup never silently becomes time-to-first-token.
    assert model.request_timeout.read is None
    assert model.request_timeout.connect == 10.0
    assert model.stream_chunk_timeout is None


def _stream_chunk(delta: dict) -> dict:
    return {
        "id": "chatcmpl-test",
        "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
        "model": "deepseek_v4",
        "object": "chat.completion.chunk",
    }


def test_chunk_conversion_lifts_vllm_reasoning_delta():
    model = build_chat_model(
        RuntimeSettings(openai_base_url="http://127.0.0.1:8003/v1", openai_model="deepseek_v4")
    )

    generation_chunk = model._convert_chunk_to_generation_chunk(
        _stream_chunk({"content": "", "reasoning": "Considering the request."}),
        AIMessageChunk,
        None,
    )

    assert generation_chunk is not None
    assert (
        generation_chunk.message.additional_kwargs["reasoning_content"]
        == "Considering the request."
    )
    assert generation_chunk.message.content == ""


def test_chunk_conversion_supports_reasoning_content_field():
    model = build_chat_model(
        RuntimeSettings(openai_base_url="http://127.0.0.1:8003/v1", openai_model="deepseek_v4")
    )

    generation_chunk = model._convert_chunk_to_generation_chunk(
        _stream_chunk({"content": "", "reasoning_content": "Thinking."}),
        AIMessageChunk,
        None,
    )

    assert generation_chunk is not None
    assert generation_chunk.message.additional_kwargs["reasoning_content"] == "Thinking."


def test_chunk_conversion_leaves_content_chunks_unchanged():
    model = build_chat_model(
        RuntimeSettings(openai_base_url="http://127.0.0.1:8003/v1", openai_model="deepseek_v4")
    )

    generation_chunk = model._convert_chunk_to_generation_chunk(
        _stream_chunk({"content": "Hello"}),
        AIMessageChunk,
        None,
    )

    assert generation_chunk is not None
    assert generation_chunk.message.content == "Hello"
    assert "reasoning_content" not in generation_chunk.message.additional_kwargs


def test_build_chat_model_publishes_context_window_for_adaptive_summarization():
    """A configured context window must flip deepagents summarization to adaptive
    fraction-based compaction WITHOUT enabling native structured output (which
    would drop the subagent ToolStrategy auto-retry handoff)."""
    from deepagents.middleware.summarization import compute_summarization_defaults
    from langchain.agents.factory import _supports_provider_strategy
    from ultra_deepagents.config import RuntimeSettings
    from ultra_deepagents.model import build_chat_model

    base = dict(openai_base_url="http://127.0.0.1:9/v1", openai_model="deepseek_v4")

    # Default (0): no profile -> conservative fallback (170k tokens / keep 6 messages).
    unset = build_chat_model(RuntimeSettings(**base))
    assert unset.profile is None
    d_unset = compute_summarization_defaults(unset)
    assert d_unset["trigger"] == ("tokens", 170000)
    assert d_unset["keep"] == ("messages", 6)

    # Configured: profile carries max_input_tokens -> adaptive 85% trigger / 10% keep.
    on = build_chat_model(RuntimeSettings(**base, model_max_input_tokens=786432))
    assert on.profile == {"max_input_tokens": 786432}
    d_on = compute_summarization_defaults(on)
    assert d_on["trigger"] == ("fraction", 0.85)
    assert d_on["keep"] == ("fraction", 0.10)

    # Structured-output strategy must stay ToolStrategy (no "structured_output" key),
    # so SCOPED_DELEGATION_RESPONSE_FORMAT keeps its auto-retry path.
    assert _supports_provider_strategy(on, tools=[object()]) is False
    assert "structured_output" not in on.profile
