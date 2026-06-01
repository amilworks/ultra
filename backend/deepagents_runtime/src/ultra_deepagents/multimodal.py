from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain_core.messages import BaseMessage

_MULTIMODAL_BLOCK_TYPES = {"image", "image_url", "audio", "video", "file"}


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


def _sanitize_request(request: ModelRequest[Any]) -> ModelRequest[Any]:
    sanitized_messages = sanitize_messages_for_text_only_model(request.messages)
    sanitized_system_message = (
        _sanitize_message(request.system_message) if request.system_message is not None else None
    )
    if sanitized_messages is request.messages and sanitized_system_message is request.system_message:
        return request
    return request.override(
        messages=sanitized_messages,
        system_message=sanitized_system_message,
    )


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
