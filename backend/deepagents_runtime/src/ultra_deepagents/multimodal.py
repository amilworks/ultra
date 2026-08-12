from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain_core.messages import BaseMessage

_MULTIMODAL_BLOCK_TYPES = {"image", "image_url", "audio", "video", "file"}
_IMAGE_BLOCK_TYPES = {"image", "image_url", "input_image"}


class QwenModelCallTimeoutError(RuntimeError):
    """The local Qwen model-call wall clock expired."""


def sanitize_messages_for_text_only_model(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Replace media blocks with text notices before a text-only model call."""

    sanitized: list[BaseMessage] = []
    changed = False
    for message in messages:
        sanitized_message = _sanitize_message(message)
        changed = changed or sanitized_message is not message
        sanitized.append(sanitized_message)
    return sanitized if changed else messages


class TextOnlyMultimodalMiddleware(AgentMiddleware[Any, Any, Any]):
    """Prevent text-only chat models from receiving multimodal content blocks."""

    tools = ()

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Any],
    ) -> Any:
        return handler(_sanitize_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[Any]],
    ) -> Any:
        return await handler(_sanitize_request(request))


class BoundedImageMultimodalMiddleware(AgentMiddleware[Any, Any, Any]):
    """Keep only the newest images in a multimodal model request.

    Older image bytes are replaced in place with a small path-preserving text
    breadcrumb. Message order, message identity fields, and every non-image
    block are preserved. The optional async timeout bounds the entire downstream
    model call; it is used for the Qwen coding delegate because provider request
    timeouts do not bound all async client failure modes.
    """

    tools = ()

    def __init__(
        self,
        *,
        max_images: int,
        async_timeout_seconds: float | None = None,
    ) -> None:
        if max_images < 0:
            raise ValueError("max_images must be non-negative")
        self.max_images = int(max_images)
        self.async_timeout_seconds = (
            float(async_timeout_seconds)
            if async_timeout_seconds is not None and async_timeout_seconds > 0
            else None
        )

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Any],
    ) -> Any:
        return handler(_bound_image_request(request, max_images=self.max_images))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[Any]],
    ) -> Any:
        bounded_request = _bound_image_request(request, max_images=self.max_images)
        if self.async_timeout_seconds is None:
            return await handler(bounded_request)
        timeout = asyncio.timeout(self.async_timeout_seconds)
        try:
            async with timeout:
                return await handler(bounded_request)
        except TimeoutError as exc:
            if not timeout.expired():
                raise
            raise QwenModelCallTimeoutError(
                f"Qwen model call exceeded {self.async_timeout_seconds:g}s."
            ) from exc


def _sanitize_request(request: ModelRequest[Any]) -> ModelRequest[Any]:
    sanitized_messages = sanitize_messages_for_text_only_model(request.messages)
    sanitized_system_message = (
        _sanitize_message(request.system_message) if request.system_message is not None else None
    )
    if (
        sanitized_messages is request.messages
        and sanitized_system_message is request.system_message
    ):
        return request
    return request.override(
        messages=sanitized_messages,
        system_message=sanitized_system_message,
    )


def _bound_image_request(
    request: ModelRequest[Any],
    *,
    max_images: int,
) -> ModelRequest[Any]:
    ordered_messages = [
        *([request.system_message] if request.system_message is not None else []),
        *request.messages,
    ]
    image_count = sum(
        1
        for message in ordered_messages
        if isinstance(message.content, list)
        for block in message.content
        if isinstance(block, dict) and _is_image_block(block)
    )
    images_to_replace = max(0, image_count - max_images)
    if images_to_replace == 0:
        return request

    bounded_messages: list[BaseMessage] = []
    remaining = images_to_replace
    for message in ordered_messages:
        bounded_message, replaced = _replace_oldest_images(
            message,
            count=remaining,
            max_images=max_images,
        )
        bounded_messages.append(bounded_message)
        remaining -= replaced

    offset = 1 if request.system_message is not None else 0
    bounded_system_message = bounded_messages[0] if offset else request.system_message
    return request.override(
        messages=bounded_messages[offset:],
        system_message=bounded_system_message,
    )


def _replace_oldest_images(
    message: BaseMessage,
    *,
    count: int,
    max_images: int,
) -> tuple[BaseMessage, int]:
    if count <= 0 or not isinstance(message.content, list):
        return message, 0

    content: list[Any] = []
    replaced = 0
    for block in message.content:
        if replaced < count and isinstance(block, dict) and _is_image_block(block):
            content.append(_bounded_image_notice(message, block, max_images=max_images))
            replaced += 1
        else:
            content.append(block)
    if replaced == 0:
        return message, 0
    return message.model_copy(update={"content": content}), replaced


def _is_image_block(block: dict[str, Any]) -> bool:
    block_type = str(block.get("type") or "").strip().lower()
    if block_type in _IMAGE_BLOCK_TYPES:
        return True
    if "image_url" in block:
        return True
    mime_type = str(block.get("mime_type") or block.get("media_type") or "").lower()
    return bool("base64" in block and mime_type.startswith("image/"))


def _bounded_image_notice(
    message: BaseMessage,
    block: dict[str, Any],
    *,
    max_images: int,
) -> dict[str, str]:
    path = str(
        message.additional_kwargs.get("read_file_path")
        or block.get("path")
        or block.get("file_path")
        or ""
    ).strip()
    mime_type = (
        str(message.additional_kwargs.get("read_file_media_type") or "").strip()
        or str(block.get("mime_type") or block.get("media_type") or "").strip()
        or _mime_type_from_image_url(block)
    )
    source = f" Source path: {path}." if path else ""
    mime = f" MIME type: {mime_type}." if mime_type else ""
    return {
        "type": "text",
        "text": (
            f"[Older image bytes omitted to keep this model request at most "
            f"{max_images} images.{source}{mime} Re-read the source path if the "
            "older image is needed.]"
        ),
    }


def _sanitize_message(message: BaseMessage) -> BaseMessage:
    content = message.content
    if not isinstance(content, list):
        return message

    parts: list[str] = []
    changed = False
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        if _is_text_block(block):
            text = _text_from_block(block)
            if text:
                parts.append(text)
            continue
        if _is_multimodal_block(block):
            changed = True
            parts.append(_multimodal_notice(message, block))
            continue
        parts.append(str(block))

    if not changed:
        return message
    replacement = "\n\n".join(part for part in parts if part).strip()
    if not replacement:
        replacement = (
            "[multimodal content omitted: the active model is text-only. "
            "Use artifact paths, metadata, and code-derived summaries instead.]"
        )
    return message.model_copy(update={"content": replacement})


def _is_text_block(block: dict[str, Any]) -> bool:
    return str(block.get("type") or "").strip().lower() == "text"


def _text_from_block(block: dict[str, Any]) -> str:
    for key in ("text", "content", "body"):
        value = block.get(key)
        if value is not None:
            return str(value)
    return ""


def _is_multimodal_block(block: dict[str, Any]) -> bool:
    block_type = str(block.get("type") or "").strip().lower()
    if block_type in _MULTIMODAL_BLOCK_TYPES:
        return True
    if "base64" in block or "image_url" in block:
        return True
    return bool(block.get("mime_type") and block_type != "text")


def _multimodal_notice(message: BaseMessage, block: dict[str, Any]) -> str:
    block_type = str(block.get("type") or "media").strip().lower() or "media"
    mime_type = (
        str(message.additional_kwargs.get("read_file_media_type") or "").strip()
        or str(block.get("mime_type") or "").strip()
        or _mime_type_from_image_url(block)
    )
    path = str(message.additional_kwargs.get("read_file_path") or "").strip()
    source = f" from {path}" if path else ""
    mime = f" ({mime_type})" if mime_type else ""
    return (
        f"[{block_type}{mime} multimodal content omitted{source}: "
        "the active model is text-only. Use the artifact path, source code, "
        "metadata, or computed summary for captions instead of visual inspection.]"
    )


def _mime_type_from_image_url(block: dict[str, Any]) -> str:
    image_url = block.get("image_url")
    if not isinstance(image_url, dict):
        return ""
    url = str(image_url.get("url") or "").strip()
    if not url.startswith("data:"):
        return ""
    prefix = url.split(",", 1)[0]
    media_type = prefix.removeprefix("data:").split(";", 1)[0].strip()
    return media_type
