"""OpenAI-compatible agent-owned tool setup."""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import mcp.types as mcp_types

from hud.agents.tools import AgentTool, AgentToolSpec

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

    from hud.types import MCPToolCall, MCPToolResult

    from .qwen_computer import QwenComputerUseToolParam

OpenAICompatibleToolParam: TypeAlias = "ChatCompletionToolParam | QwenComputerUseToolParam"
_TOOL_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


class OpenAICompatibleTool(AgentTool[OpenAICompatibleToolParam, "ChatCompletionMessageParam"]):
    """Agent-side OpenAI-compatible tool backed by an environment tool."""

    def format_result(
        self, call: MCPToolCall, result: MCPToolResult
    ) -> ChatCompletionMessageParam | list[ChatCompletionMessageParam]:
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


class OpenAICompatibleFunctionTool(OpenAICompatibleTool):
    """Regular environment tool exposed as an OpenAI-compatible function."""

    name = "function"
    capability = "function"

    def __init__(
        self,
        *,
        env_tool_name: str,
        provider_name: str,
        params: OpenAICompatibleToolParam,
    ) -> None:
        super().__init__(
            env_tool_name=env_tool_name,
            spec=AgentToolSpec(api_type="function", api_name=env_tool_name),
        )
        self._provider_name = provider_name
        self.params = params

    @classmethod
    def from_tool(cls, tool: mcp_types.Tool) -> OpenAICompatibleFunctionTool:
        provider_name = openai_compatible_tool_name(tool.name)
        return cls(
            env_tool_name=tool.name,
            provider_name=provider_name,
            params=openai_compatible_tool_param(tool, name=provider_name),
        )

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def to_params(self) -> OpenAICompatibleToolParam:
        return self.params


def openai_compatible_tool_name(name: str) -> str:
    sanitized = _TOOL_NAME_PATTERN.sub("_", name).strip("_") or "tool"
    if sanitized == name and len(sanitized) <= 64:
        return sanitized

    digest = hashlib.sha256(name.encode()).hexdigest()[:8]
    prefix = sanitized[: 64 - len(digest) - 1].rstrip("_") or "tool"
    return f"{prefix}_{digest}"


def openai_compatible_tool_param(
    tool: mcp_types.Tool,
    *,
    name: str | None = None,
) -> OpenAICompatibleToolParam:
    parameters = tool.inputSchema
    sanitized_params: dict[str, Any] = (
        _sanitize_openai_compatible_schema(parameters)
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
                "parameters": sanitized_params,
            },
        },
    )


def _sanitize_openai_compatible_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert MCP JSON Schema to the OpenAI-compatible chat tool subset."""
    sanitized: dict[str, Any] = {}

    for key, value in schema.items():
        if key == "anyOf" and isinstance(value, list):
            any_of_items = cast("list[Any]", value)
            non_null_types: list[dict[str, Any]] = [
                cast("dict[str, Any]", item)
                for item in any_of_items
                if isinstance(item, dict) and cast("dict[str, Any]", item).get("type") != "null"
            ]
            if non_null_types:
                sanitized.update(_sanitize_openai_compatible_schema(non_null_types[0]))
            else:
                sanitized["type"] = "string"

        elif key == "prefixItems" and isinstance(value, list):
            sanitized["type"] = "array"
            prefix_items = cast("list[Any]", value)
            if prefix_items:
                first_item: Any = prefix_items[0]
                if isinstance(first_item, dict):
                    first_schema = cast("dict[str, Any]", first_item)
                    sanitized["items"] = {"type": first_schema.get("type", "string")}
                else:
                    sanitized["items"] = {"type": "string"}

        elif key == "properties" and isinstance(value, dict):
            properties = cast("dict[str, Any]", value)
            sanitized[key] = {
                prop_name: _sanitize_openai_compatible_schema(cast("dict[str, Any]", prop_schema))
                for prop_name, prop_schema in properties.items()
                if isinstance(prop_schema, dict)
            }

        elif key == "items" and isinstance(value, dict):
            sanitized[key] = _sanitize_openai_compatible_schema(cast("dict[str, Any]", value))

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
