"""Claude Agent implementation compatible with HUD MCPAgent.

Replicates logic from reference_claude_agent.py including streaming and prompt caching.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from copy import deepcopy
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import (
    Base64ImageSourceParam,
    ImageBlockParam,
    MessageParam,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
)
import mcp.types as types

from hud.agents.base import MCPAgent
from hud.settings import settings
from hud.types import AgentResponse, MCPToolCall, MCPToolResult
from hud.utils.hud_console import HUDConsole

logger = logging.getLogger(__name__)


def truncate_base64_in_data(data: Any, max_length: int = 100) -> Any:
    """Recursively truncate base64 strings in data structures for logging."""
    if isinstance(data, dict):
        return {k: truncate_base64_in_data(v, max_length) for k, v in data.items()}
    if isinstance(data, list):
        return [truncate_base64_in_data(item, max_length) for item in data]
    if isinstance(data, str):
        # Pattern to match base64 strings (common patterns used in image data)
        base64_pattern = r"data:image/[^;]+;base64,[A-Za-z0-9+/]+=*|[A-Za-z0-9+/]{50,}=*"

        def replace_base64(match: re.Match) -> str:
            b64_str = match.group(0)
            if len(b64_str) > max_length:
                return f"[BASE64_DATA_TRUNCATED_{len(b64_str)}_CHARS]"
            return b64_str

        return re.sub(base64_pattern, replace_base64, data)
    return data


def mcp_content_block_to_messages_format(
    content: types.TextContent | types.ImageContent,
) -> TextBlockParam | ImageBlockParam:
    if isinstance(content, types.TextContent):
        return TextBlockParam(type="text", text=content.text)
    if isinstance(content, types.ImageContent):
        if content.mimeType not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
            raise ValueError(f"Invalid image mime type: {content.mimeType}")
        return ImageBlockParam(
            type="image",
            source=Base64ImageSourceParam(
                data=content.data,
                media_type=content.mimeType,  # type: ignore
                type="base64",
            ),
        )
    raise ValueError(f"Invalid content type: {type(content)}")


class AnthropicToolCallHandler:
    """Handles tool calls from Anthropic API."""

    def __init__(self) -> None:
        self.current_partial_json = ""
        self.current_tool_use: dict[str, Any] | None = None

    def process_stream_chunk(self, chunk: Any) -> dict[str, Any] | None:
        """Process Anthropic stream chunk."""
        if chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
            return {"type": "text", "content": chunk.delta.text}

        if chunk.type == "content_block_start" and chunk.content_block.type == "tool_use":
            if not isinstance(chunk, RawContentBlockStartEvent):
                raise ValueError(f"Expected RawContentBlockStartEvent, got {type(chunk)}")
            self.current_tool_use = {
                "id": chunk.content_block.id,
                "name": chunk.content_block.name,
                "input": {},
                "error": None,
            }
            return {"type": "tool_use_start", "tool": chunk.content_block.name}

        if chunk.type == "content_block_delta" and chunk.delta.type == "input_json_delta":
            if not isinstance(chunk, RawContentBlockDeltaEvent):
                raise ValueError(f"Expected RawContentBlockDeltaEvent, got {type(chunk)}")
            if chunk.delta.partial_json:
                if not self.current_tool_use:
                    logger.warning(
                        "No current tool use found for JSON delta: %s", chunk.delta.partial_json
                    )
                    return None
                self.current_partial_json += chunk.delta.partial_json
                return {"type": "tool_use_param", "value": chunk.delta.partial_json}

        if chunk.type == "content_block_stop" and self.current_tool_use:
            if not isinstance(chunk, RawContentBlockStopEvent):
                raise ValueError(f"Expected RawContentBlockStopEvent, got {type(chunk)}")
            # Try to parse the JSON
            try:
                parsed_partial_json = json.loads(self.current_partial_json)
                self.current_tool_use["input"] = parsed_partial_json
            except json.JSONDecodeError as e:
                self.current_tool_use["error"] = (
                    f"Unable to parse JSON: {str(e)}. Raw JSON:\n```\n{self.current_partial_json}\n```"
                )
            # Reset for next tool call
            tool_call = self.current_tool_use
            self.current_tool_use = None
            self.current_partial_json = ""
            return {"type": "tool_call_complete", "tool_call": tool_call}

        # Handle max_tokens stop to propagate partial tool call error like normal flow
        if (
            getattr(chunk, "type", None) == "message_delta"
            and getattr(getattr(chunk, "delta", None), "stop_reason", None) == "max_tokens"
        ):
            if self.current_tool_use:
                self.current_tool_use["error"] = "Response exceeded max_tokens during stream."
                tool_call = self.current_tool_use
                self.current_tool_use = None
                self.current_partial_json = ""
                return {"type": "tool_call_complete", "tool_call": tool_call}

        return None


class Claude2Agent(MCPAgent):
    """
    Claude Agent that closely follows the reference implementation (reference_claude_agent.py).
    Includes prompt caching, parallel tool use, and robust retry logic.
    """

    anthropic_client: AsyncAnthropic

    def __init__(
        self,
        model_client: AsyncAnthropic | None = None,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 8192,
        validate_api_key: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Initialize client if not provided
        if model_client is None:
            api_key = settings.anthropic_api_key
            if not api_key:
                raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY.")
            self.anthropic_client = AsyncAnthropic(api_key=api_key)
        else:
            self.anthropic_client = model_client

        self.model = model
        self.max_tokens = max_tokens
        self.hud_console = HUDConsole(logger=logger)

        self.model_name = "Claude2"
        self.checkpoint_name = self.model

    async def get_system_messages(self) -> list[Any]:
        """
        Get system messages.

        In reference implementation, system prompt is handled via the API parameter,
        not as a message in the messages list. We return empty list here and
        handle system prompt in get_response.
        """
        return []

    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[MessageParam]:
        """Format content blocks into Anthropic messages."""
        anthropic_blocks: list[Any] = []

        for block in blocks:
            if isinstance(block, types.TextContent):
                anthropic_blocks.append(TextBlockParam(type="text", text=block.text))
            elif isinstance(block, types.ImageContent):
                anthropic_blocks.append(
                    ImageBlockParam(
                        type="image",
                        source=Base64ImageSourceParam(
                            type="base64",
                            media_type=block.mimeType,  # type: ignore
                            data=block.data,
                        ),
                    )
                )
            else:
                self.hud_console.log(
                    f"Unknown content block type: {type(block)}", level="warning"
                )
                # Best effort fallback
                anthropic_blocks.append(block)  # type: ignore

        return [
            {
                "role": "user",
                "content": anthropic_blocks,
            }
        ]

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[MessageParam]:
        """Format tool results into Anthropic messages."""
        content: list[ToolResultBlockParam] = []

        for tool_call, result in zip(tool_calls, tool_results, strict=True):
            # Extract tool_use_id from tool_call
            tool_use_id = tool_call.id
            if not tool_use_id:
                self.hud_console.warning(f"No tool_use_id found for {tool_call.name}")
                continue

            tool_content: list[TextBlockParam | ImageBlockParam] = []

            if result.isError:
                error_msg = "Tool execution failed"
                for c in result.content:
                    if isinstance(c, types.TextContent):
                        error_msg = c.text
                        break
                tool_content.append(TextBlockParam(type="text", text=f"Error: {error_msg}"))
            else:
                for c in result.content:
                    if isinstance(c, (types.TextContent, types.ImageContent)):
                        tool_content.append(mcp_content_block_to_messages_format(c))

            content.append(
                ToolResultBlockParam(
                    type="tool_result", tool_use_id=tool_use_id, content=tool_content
                )
            )

        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    def _convert_tools(self) -> list[ToolParam]:
        """Convert MCP tools to Anthropic tools."""
        tools: list[ToolParam] = []
        for tool in self.get_available_tools():
            input_schema = tool.inputSchema or {"type": "object", "properties": {}}
            if "type" not in input_schema:
                input_schema = {"type": "object", **input_schema}

            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": input_schema,  # type: ignore
                }
            )
        return tools

    async def get_response(self, messages: list[Any]) -> AgentResponse:
        """
        Get response from Claude with retry logic and streaming.
        Replicates the stream processing loop from reference_claude_agent.py.
        """
        # Convert messages to MessageParam
        current_messages = cast(list[MessageParam], deepcopy(messages))

        # Get tools
        tools = self._convert_tools()

        # Apply tool caching (ephemeral) to the last tool
        if tools:
            tools[-1]["cache_control"] = {"type": "ephemeral"}  # type: ignore

        # Apply prompt caching to the last user message
        last_msg = current_messages[-1]
        if last_msg.get("role") == "user":
            content = last_msg.get("content")
            if isinstance(content, list):
                # We can attach cache_control to the last content block of the last user message
                # or strictly follow reference implementation which iterates blocks
                for block in content:
                    if isinstance(block, dict):
                        block["cache_control"] = {"type": "ephemeral"}  # type: ignore

        # Retry logic
        retries = 0
        max_retries = 10
        base_delay = 1
        response_stream: Any = None

        extra_headers = {
            "anthropic-beta": "computer-use-2025-01-24,fine-grained-tool-streaming-2025-05-14",
        }

        while True:
            try:
                create_kwargs = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": current_messages,
                    "extra_headers": extra_headers,
                    "stream": True,
                }

                if tools:
                    create_kwargs["tools"] = tools

                if self.system_prompt:
                    create_kwargs["system"] = self.system_prompt

                response_stream = await self.anthropic_client.messages.create(**create_kwargs)
                break
            except Exception as e:
                logger.exception("API call failed: %s", e)
                if retries >= max_retries:
                    raise
                retries += 1
                delay = base_delay * (2 ** (retries - 1))
                logger.warning(
                    "API call failed, attempt %d/%d. Retrying in %ss...",
                    retries,
                    max_retries,
                    delay,
                )
                await asyncio.sleep(delay)

        # Process stream
        tool_calls: list[dict[str, Any]] = []
        current_text = ""

        handler = AnthropicToolCallHandler()

        if response_stream is None:
            raise RuntimeError("Failed to get response stream")

        async for chunk in response_stream:
            result = handler.process_stream_chunk(chunk)
            if result:
                if result["type"] == "text":
                    current_text += result["content"]
                elif result["type"] == "tool_call_complete":
                    tool_calls.append(result["tool_call"])
                # We can ignore other event types like tool_use_start/param for the final response construction
                # but they are useful for real-time feedback if we wanted to pipe that to console

        # Build AgentResponse
        final_content = current_text
        final_tool_calls: list[MCPToolCall] = []

        for tc in tool_calls:
            final_tool_calls.append(
                MCPToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc["input"],
                )
            )

        # Replicate check for empty input
        for tc in tool_calls:
            if tc["input"] == {}:
                logger.warning("Tool call %s has no input", tc["name"])

        return AgentResponse(
            content=final_content,
            tool_calls=final_tool_calls,
            done=len(final_tool_calls) == 0,
        )
