"""LiteLLM MCP Agent implementation.

Same OpenAI chat-completions shape + MCP tool plumbing,
but transport is LiteLLM and (optionally) tools are shaped by LiteLLM's MCP transformer.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import litellm
import mcp.types as types

from hud.types import MCPToolCall, MCPToolResult

from .openai_chat_generic import GenericOpenAIChatAgent

logger = logging.getLogger(__name__)

# Prefer LiteLLM's built-in MCP -> OpenAI tool transformer (handles Bedrock nuances)
try:
    from litellm.experimental_mcp_client.tools import (
        transform_mcp_tool_to_openai_tool,
    )
except Exception:  # pragma: no cover - optional dependency
    transform_mcp_tool_to_openai_tool = None  # type: ignore


class LiteAgent(GenericOpenAIChatAgent):
    """
    Same OpenAI chat-completions shape + MCP tool plumbing,
    but transport is LiteLLM and (optionally) tools are shaped by LiteLLM's MCP transformer.
    """

    metadata: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        *,
        model_name: str = "gpt-4o-mini",
        completion_kwargs: dict[str, Any] | None = None,
        **agent_kwargs: Any,
    ) -> None:
        # We don't need an OpenAI client; pass None
        super().__init__(
            openai_client=None,
            model_name=model_name,
            completion_kwargs=completion_kwargs,
            **agent_kwargs,
        )

    def get_tool_schemas(self) -> list[dict]:
        # Prefer LiteLLM's stricter transformer (handles Bedrock & friends)
        if transform_mcp_tool_to_openai_tool is not None:
            return [
                transform_mcp_tool_to_openai_tool(t)  # returns ChatCompletionToolParam-like dict
                for t in self.get_available_tools()
            ]
        # Fallback to the generic OpenAI sanitizer
        return GenericOpenAIChatAgent.get_tool_schemas(self)

    async def get_system_messages(self) -> list[Any]:
        """Get system messages with caching support."""
        return [{
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        }]

    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[Any]:
        """Format blocks with caching support."""
        content = []
        for block in blocks:
            if isinstance(block, types.TextContent):
                content.append({
                    "type": "text",
                    "text": block.text,
                    "cache_control": {"type": "ephemeral"}
                })
            elif isinstance(block, types.ImageContent):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{block.mimeType};base64,{block.data}"},
                    "cache_control": {"type": "ephemeral"}
                })

        return [{"role": "user", "content": content}]

    async def format_tool_results(
        self,
        tool_calls: list[MCPToolCall],
        tool_results: list[MCPToolResult],
    ) -> list[Any]:
        """Render MCP tool results with caching support."""
        rendered: list[dict[str, Any]] = []
        image_parts = []

        for call, res in zip(tool_calls, tool_results, strict=False):
            text_parts = []
            items = res.content
            if not res.content and res.structuredContent:
                items = [res.structuredContent.get("result", res.content)]

            for item in items:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image":
                        mime_type = item.get("mimeType", "image/png")
                        data = item.get("data", "")
                        image_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{data}"},
                            "cache_control": {"type": "ephemeral"}
                        })
                elif isinstance(item, types.TextContent):
                    text_parts.append(item.text)
                elif isinstance(item, types.ImageContent):
                    image_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{item.mimeType};base64,{item.data}"},
                        "cache_control": {"type": "ephemeral"}
                    })

            text_content = "".join(text_parts) if text_parts else "Tool executed successfully"
            rendered.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": text_content,
            })

        if image_parts:
            content_with_images = [
                {
                    "type": "text",
                    "text": "Tool returned the following:",
                    "cache_control": {"type": "ephemeral"}
                },
                image_parts[-1],
            ]
            rendered.append({
                "role": "user",
                "content": content_with_images,
            })

        return rendered

    async def _invoke_chat_completion(
        self,
        *,
        messages: list[Any],
        tools: list[dict] | None,
        extra: dict[str, Any],
    ) -> Any:
        return await litellm.acompletion(
            model=self.model_name,
            messages=messages,
            tools=tools or None,  # LiteLLM tolerates None better than []
            tool_choice="auto",
            **extra,
        )
