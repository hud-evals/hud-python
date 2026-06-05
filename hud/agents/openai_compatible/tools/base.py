"""OpenAI-compatible tool spec + result formatting."""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import mcp.types as mcp_types

from hud.agents.tools.base import AgentToolSpec

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

    from hud.types import MCPToolCall, MCPToolResult

OpenAICompatibleToolParam: TypeAlias = "ChatCompletionToolParam"
_TOOL_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


def format_chat_result(
    call: MCPToolCall,
    result: MCPToolResult,
) -> ChatCompletionMessageParam | list[ChatCompletionMessageParam]:
    """Format a tool result for OpenAI-compatible chat completions."""
    text_parts: list[str] = []
    image_parts: list[dict[str, Any]] = []
    items: list[Any] = list(result.content)
    if not result.content and result.structuredContent:
        items = [result.structuredContent.get("result", result.content)]

    for item in items:
        if isinstance(item, dict):
            item_dict = cast("dict[str, Any]", item)
            if item_dict.get("type") == "text":
                text_parts.append(str(item_dict.get("text", "")))
            elif item_dict.get("type") == "image":
                mime_type = str(item_dict.get("mimeType", "image/png"))
                data = str(item_dict.get("data", ""))
                image_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{data}"},
                    }
                )
        elif isinstance(item, mcp_types.TextContent):
            text_parts.append(item.text)
        elif isinstance(item, mcp_types.ImageContent):
            image_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{item.mimeType};base64,{item.data}"},
                }
            )

    tool_message = cast(
        "ChatCompletionMessageParam",
        {
            "role": "tool",
            "tool_call_id": call.id,
            "content": "".join(text_parts) if text_parts else "Tool executed successfully",
        },
    )
    if not image_parts:
        return tool_message
    return [
        tool_message,
        cast(
            "ChatCompletionMessageParam",
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Tool returned the following:"},
                    image_parts[-1],
                ],
            },
        ),
    ]


def openai_compatible_tool_name(name: str) -> str:
    sanitized = _TOOL_NAME_PATTERN.sub("_", name).strip("_") or "tool"
    if sanitized == name and len(sanitized) <= 64:
        return sanitized
    digest = hashlib.sha256(name.encode()).hexdigest()[:8]
    prefix = sanitized[: 64 - len(digest) - 1].rstrip("_") or "tool"
    return f"{prefix}_{digest}"


def _sanitize_schema_for_openai(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert MCP JSON Schema to the OpenAI-compatible chat tool subset."""
    sanitized: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "anyOf" and isinstance(value, list):
            any_of = cast("list[Any]", value)
            non_null = [
                cast("dict[str, Any]", item)
                for item in any_of
                if isinstance(item, dict) and cast("dict[str, Any]", item).get("type") != "null"
            ]
            if non_null:
                sanitized.update(_sanitize_schema_for_openai(non_null[0]))
            else:
                sanitized["type"] = "string"
        elif key == "prefixItems" and isinstance(value, list):
            sanitized["type"] = "array"
            prefix_items = cast("list[Any]", value)
            if prefix_items:
                first: Any = prefix_items[0]
                if isinstance(first, dict):
                    sanitized["items"] = {
                        "type": cast("dict[str, Any]", first).get("type", "string")
                    }
                else:
                    sanitized["items"] = {"type": "string"}
        elif key == "properties" and isinstance(value, dict):
            sanitized[key] = {
                k: _sanitize_schema_for_openai(cast("dict[str, Any]", v))
                for k, v in cast("dict[str, Any]", value).items()
                if isinstance(v, dict)
            }
        elif key == "items" and isinstance(value, dict):
            sanitized[key] = _sanitize_schema_for_openai(cast("dict[str, Any]", value))
        elif key in (
            "type",
            "description",
            "enum",
            "required",
            "default",
            "minimum",
            "maximum",
            "minItems",
            "maxItems",
        ):
            sanitized[key] = value
    return sanitized or {"type": "object"}


def openai_compatible_tool_param(
    tool: mcp_types.Tool,
    *,
    name: str | None = None,
) -> OpenAICompatibleToolParam:
    parameters = tool.inputSchema
    sanitized = (
        _sanitize_schema_for_openai(parameters)
        if parameters
        else {"type": "object", "properties": {}}
    )
    return cast(
        "OpenAICompatibleToolParam",
        {
            "type": "function",
            "function": {
                "name": name or openai_compatible_tool_name(tool.name),
                "description": tool.description or f"Call {tool.name}",
                "parameters": sanitized,
            },
        },
    )


__all__ = [
    "AgentToolSpec",
    "OpenAICompatibleToolParam",
    "format_chat_result",
    "openai_compatible_tool_name",
    "openai_compatible_tool_param",
]
