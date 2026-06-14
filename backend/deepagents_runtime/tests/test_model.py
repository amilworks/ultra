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
    assert model.request_timeout == 42.0
    assert model.stream_chunk_timeout == 42.0
    assert model.max_retries == 0


def test_build_chat_model_disables_request_timeout_when_setting_is_zero():
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        request_timeout_seconds=0.0,
    )

    model = build_chat_model(settings)

    assert model.request_timeout is None
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
