from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from ultra_deepagents.multimodal import (
    BoundedImageMultimodalMiddleware,
    QwenModelCallTimeoutError,
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


def test_bounded_image_middleware_keeps_only_the_four_newest_images() -> None:
    messages = [
        ToolMessage(
            content_blocks=[
                {"type": "image", "base64": f"image-{index}", "mime_type": "image/png"}
            ],
            tool_call_id=f"call-{index}",
            name="read_file",
            additional_kwargs={"read_file_path": f"/workspace/outputs/plot-{index}.png"},
        )
        for index in range(5)
    ]
    middleware = BoundedImageMultimodalMiddleware(max_images=4)
    captured: list[Any] = []

    def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        captured.extend(request.messages)
        return ModelResponse(result=[AIMessage(content="ok")])

    request = ModelRequest(
        model=cast(BaseChatModel, object()),
        messages=messages,
        runtime=cast(Any, None),
    )

    response = middleware.wrap_model_call(request, handler)

    assert response.result[0].content == "ok"
    image_blocks = [
        block
        for message in captured
        for block in (message.content if isinstance(message.content, list) else [])
        if isinstance(block, dict) and block.get("type") in {"image", "image_url", "input_image"}
    ]
    assert len(image_blocks) == 4
    assert isinstance(captured[0].content, list)
    first_notice = captured[0].content[0]
    assert isinstance(first_notice, dict) and first_notice.get("type") == "text"
    assert "plot-0.png" in str(first_notice.get("text"))
    assert "at most 4 images" in str(first_notice.get("text"))
    assert "image-0" not in str(first_notice)
    assert [block.get("base64") for block in image_blocks] == [
        "image-1",
        "image-2",
        "image-3",
        "image-4",
    ]
    assert [message.tool_call_id for message in captured] == [
        "call-0",
        "call-1",
        "call-2",
        "call-3",
        "call-4",
    ]
    assert captured[1:] == messages[1:]


def test_bounded_image_middleware_preserves_non_image_blocks_in_place() -> None:
    message = HumanMessage(
        id="message-1",
        content=[
            {"type": "text", "text": "before"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,old"}},
            {"type": "audio", "base64": "audio-bytes", "mime_type": "audio/wav"},
            {"type": "text", "text": "after"},
        ],
    )
    middleware = BoundedImageMultimodalMiddleware(max_images=0)
    captured: list[Any] = []

    def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        captured.extend(request.messages)
        return ModelResponse(result=[AIMessage(content="ok")])

    request = ModelRequest(
        model=cast(BaseChatModel, object()),
        messages=[message],
        runtime=cast(Any, None),
    )
    middleware.wrap_model_call(request, handler)

    bounded = captured[0]
    assert bounded.id == "message-1"
    assert [block["type"] for block in bounded.content] == ["text", "text", "audio", "text"]
    assert bounded.content[0] == message.content[0]
    assert bounded.content[2] == message.content[2]
    assert bounded.content[3] == message.content[3]
    assert "base64,old" not in str(bounded.content[1])


def test_bounded_image_middleware_replaces_unsupported_file_blocks() -> None:
    message = ToolMessage(
        content_blocks=[
            {
                "type": "file",
                "base64": "checkpoint-bytes",
                "mime_type": "application/octet-stream",
                "filename": "yolov5n.pt",
            },
            {"type": "image", "base64": "image-bytes", "mime_type": "image/png"},
        ],
        tool_call_id="call-read-checkpoint",
        name="read_file",
        additional_kwargs={
            "read_file_path": "/workspace/staged_uploads/yolov5n.pt",
            "read_file_media_type": "application/octet-stream",
        },
    )
    middleware = BoundedImageMultimodalMiddleware(max_images=4)
    captured: list[Any] = []

    def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        captured.extend(request.messages)
        return ModelResponse(result=[AIMessage(content="ok")])

    request = ModelRequest(
        model=cast(BaseChatModel, object()),
        messages=[message],
        runtime=cast(Any, None),
    )
    middleware.wrap_model_call(request, handler)

    bounded = captured[0]
    assert bounded.tool_call_id == "call-read-checkpoint"
    assert isinstance(bounded.content, list)
    assert [block["type"] for block in bounded.content] == ["text", "image"]
    notice = bounded.content[0]["text"]
    assert "yolov5n.pt" in notice
    assert "/workspace/staged_uploads/yolov5n.pt" in notice
    assert "application/octet-stream" in notice
    assert "checkpoint-bytes" not in str(bounded.content)
    assert bounded.content[1] == message.content[1]


def test_bounded_image_middleware_enforces_async_model_call_timeout() -> None:
    middleware = BoundedImageMultimodalMiddleware(
        max_images=4,
        async_timeout_seconds=0.01,
    )
    request = ModelRequest(
        model=cast(BaseChatModel, object()),
        messages=[HumanMessage(content="wait")],
        runtime=cast(Any, None),
    )

    async def handler(_request: ModelRequest[Any]) -> ModelResponse[Any]:
        await asyncio.Event().wait()
        return ModelResponse(result=[AIMessage(content="unreachable")])

    async def scenario() -> None:
        with pytest.raises(QwenModelCallTimeoutError) as exc_info:
            await middleware.awrap_model_call(request, handler)
        assert str(exc_info.value) == "Qwen model call exceeded 0.01s."

    asyncio.run(scenario())


def test_bounded_image_middleware_does_not_relabel_provider_timeout() -> None:
    middleware = BoundedImageMultimodalMiddleware(
        max_images=4,
        async_timeout_seconds=1.0,
    )
    request = ModelRequest(
        model=cast(BaseChatModel, object()),
        messages=[HumanMessage(content="provider timeout")],
        runtime=cast(Any, None),
    )

    async def handler(_request: ModelRequest[Any]) -> ModelResponse[Any]:
        raise TimeoutError("provider read timed out")

    async def scenario() -> None:
        with pytest.raises(TimeoutError, match="provider read timed out") as exc_info:
            await middleware.awrap_model_call(request, handler)
        assert not isinstance(exc_info.value, QwenModelCallTimeoutError)

    asyncio.run(scenario())


def test_bounded_image_middleware_preserves_external_cancellation() -> None:
    middleware = BoundedImageMultimodalMiddleware(
        max_images=4,
        async_timeout_seconds=1.0,
    )
    request = ModelRequest(
        model=cast(BaseChatModel, object()),
        messages=[HumanMessage(content="cancel")],
        runtime=cast(Any, None),
    )

    async def handler(_request: ModelRequest[Any]) -> ModelResponse[Any]:
        raise asyncio.CancelledError

    async def scenario() -> None:
        with pytest.raises(asyncio.CancelledError):
            await middleware.awrap_model_call(request, handler)

    asyncio.run(scenario())
