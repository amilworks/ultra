from __future__ import annotations

from typing import Any, cast

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from ultra_deepagents.multimodal import (
    TextOnlyMultimodalMiddleware,
    sanitize_messages_for_text_only_model,
)


def test_sanitize_tool_image_result_preserves_path_without_base64() -> None:
    message = ToolMessage(
        content_blocks=[{"type": "image", "base64": "abc123", "mime_type": "image/png"}],
        tool_call_id="call-1",
        name="read_file",
        additional_kwargs={
            "read_file_path": "/workspace/outputs/bubble_sort_pass_01.png",
            "read_file_media_type": "image/png",
        },
    )

    sanitized = sanitize_messages_for_text_only_model([message])[0]

    assert isinstance(sanitized.content, str)
    assert "bubble_sort_pass_01.png" in sanitized.content
    assert "image/png" in sanitized.content
    assert "text-only" in sanitized.content
    assert "abc123" not in sanitized.content
    assert sanitized.tool_call_id == "call-1"
    assert sanitized.name == "read_file"


def test_sanitize_human_mixed_content_keeps_text_and_replaces_image() -> None:
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Change the plot to include error bars."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
        ]
    )

    sanitized = sanitize_messages_for_text_only_model([message])[0]

    assert isinstance(sanitized.content, str)
    assert "Change the plot to include error bars." in sanitized.content
    assert "multimodal content omitted" in sanitized.content
    assert "abc123" not in sanitized.content


def test_text_only_middleware_sanitizes_model_request_before_handler() -> None:
    message = ToolMessage(
        content_blocks=[{"type": "image", "base64": "abc123", "mime_type": "image/png"}],
        tool_call_id="call-1",
        name="read_file",
        additional_kwargs={"read_file_path": "/workspace/outputs/plot.png"},
    )
    middleware = TextOnlyMultimodalMiddleware()
    captured: list[Any] = []

    def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        captured.extend(request.messages)
        return ModelResponse(result=[AIMessage(content="ok")])

    request = ModelRequest(
        model=cast(BaseChatModel, object()),
        messages=[message],
        runtime=cast(Any, None),
    )

    response = middleware.wrap_model_call(request, handler)

    assert response.result[0].content == "ok"
    assert len(captured) == 1
    assert isinstance(captured[0].content, str)
    assert "plot.png" in captured[0].content
    assert "abc123" not in captured[0].content
