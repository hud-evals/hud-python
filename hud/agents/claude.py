"""Claude MCP Agent implementation."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, cast

from anthropic import AsyncAnthropic, BadRequestError

import hud

if TYPE_CHECKING:
    from anthropic.types.beta import (
        BetaCacheControlEphemeralParam,
        BetaImageBlockParam,
        BetaMessageParam,
        BetaTextBlockParam,
        BetaToolResultBlockParam,
    )

    from hud.datasets import TaskConfig

import mcp.types as types

from hud.agent import MCPAgent
from hud.settings import settings
from hud.tools.computer.settings import computer_settings
from hud.types import AgentResponse, MCPToolCall, MCPToolResult

logger = logging.getLogger(__name__)


def base64_to_content_block(base64: str) -> BetaImageBlockParam:
    """Convert base64 image to Claude content block."""
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64},
    }


def text_to_content_block(text: str) -> BetaTextBlockParam:
    """Convert text to Claude content block."""
    return {"type": "text", "text": text}


def tool_use_content_block(
    tool_use_id: str, content: list[BetaTextBlockParam | BetaImageBlockParam]
) -> BetaToolResultBlockParam:
    """Create tool result content block."""
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}


class ClaudeAgent(MCPAgent):
    """
    Claude agent that uses MCP servers for tool execution.

    This agent uses Claude's native tool calling capabilities but executes
    tools through MCP servers instead of direct implementation.
    """

    def __init__(
        self,
        model_client: AsyncAnthropic | None = None,
        model: str = "claude-3-7-sonnet-20250219",
        max_tokens: int = 4096,
        display_width_px: int = computer_settings.ANTHROPIC_COMPUTER_WIDTH,
        display_height_px: int = computer_settings.ANTHROPIC_COMPUTER_HEIGHT,
        use_computer_beta: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Claude MCP agent.

        Args:
            model_client: AsyncAnthropic client (created if not provided)
            model: Claude model to use
            max_tokens: Maximum tokens for response
            display_width_px: Display width for computer use tools
            display_height_px: Display height for computer use tools
            use_computer_beta: Whether to use computer-use beta features
            **kwargs: Additional arguments passed to BaseMCPAgent (including mcp_client)
        """
        super().__init__(**kwargs)

        # Initialize client if not provided
        if model_client is None:
            api_key = settings.anthropic_api_key
            if not api_key:
                raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY.")
            model_client = AsyncAnthropic(api_key=api_key)

        self.anthropic_client = model_client
        self.model = model
        self.max_tokens = max_tokens
        self.display_width_px = display_width_px
        self.display_height_px = display_height_px
        self.use_computer_beta = use_computer_beta

        self.model_name = self.model

        # Track mapping from Claude tool names to MCP tool names
        self._claude_to_mcp_tool_map: dict[str, str] = {}

    async def initialize(self, task: str | TaskConfig | None = None) -> None:
        """Initialize the agent and build tool mappings."""
        await super().initialize(task)
        # Build tool mappings after tools are discovered
        self._convert_tools_for_claude()

    async def create_initial_messages(
        self, prompt: str, screenshot: str | None = None
    ) -> list[BetaMessageParam]:
        """Create initial messages for Claude."""
        user_content: list[BetaImageBlockParam | BetaTextBlockParam] = []

        # Add prompt text
        user_content.append(text_to_content_block(prompt))

        # Add screenshot if available
        if screenshot:
            user_content.append(base64_to_content_block(screenshot))

        # Return initial user message
        return [
            cast(
                "BetaMessageParam",
                {
                    "role": "user",
                    "content": user_content,
                },
            )
        ]

    @hud.instrument(
        span_type="agent",
        record_args=False,  # Messages can be large
        record_result=True,
    )
    async def get_model_response(self, messages: list[BetaMessageParam]) -> AgentResponse:
        """Get response from Claude including any tool calls."""
        # Get Claude tools
        claude_tools = self._convert_tools_for_claude()

        # Make API call with retry for prompt length
        current_messages = messages.copy()

        while True:
            messages_cached = self._add_prompt_caching(current_messages)

            # Build create kwargs
            create_kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": self.get_system_prompt(),
                "messages": messages_cached,
                "tools": claude_tools,
                "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            }

            # Add beta features if using computer tools
            if self.use_computer_beta and any(
                t.get("type") == "computer_20250124" for t in claude_tools
            ):
                create_kwargs["betas"] = ["computer-use-2025-01-24"]

            try:
                response = await self.anthropic_client.beta.messages.create(**create_kwargs)
                break
            except BadRequestError as e:
                if e.message.startswith("prompt is too long"):
                    logger.warning("Prompt too long, truncating message history")
                    # Keep first message and last 20 messages
                    if len(current_messages) > 21:
                        current_messages = [current_messages[0], *current_messages[-20:]]
                    else:
                        raise
                else:
                    raise

        messages.append(
            cast(
                "BetaMessageParam",
                {
                    "role": "assistant",
                    "content": response.content,
                },
            )
        )

        # Process response
        result = AgentResponse(content="", tool_calls=[], done=True)

        # Extract text content and reasoning
        text_content = ""
        thinking_content = ""

        for block in response.content:
            if block.type == "tool_use":
                # Map Claude tool name back to MCP tool name
                mcp_tool_name = self._claude_to_mcp_tool_map.get(block.name, block.name)

                # Create MCPToolCall object with Claude metadata as extra fields
                # Pyright will complain but the tool class accepts extra fields
                tool_call = MCPToolCall(
                    id=block.id,  # canonical identifier for telemetry
                    name=mcp_tool_name,
                    arguments=block.input,
                    claude_name=block.name,  # type: ignore
                )
                result.tool_calls.append(tool_call)
                result.done = False
            elif block.type == "text":
                text_content += block.text
            elif hasattr(block, "type") and block.type == "thinking":
                thinking_content += f"Thinking: {block.thinking}\n"

        # Combine text and thinking for final content
        if thinking_content:
            result.content = thinking_content + text_content
        else:
            result.content = text_content

        return result

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[BetaMessageParam]:
        """Format tool results into Claude messages."""
        # Process each tool result
        user_content = []

        for tool_call, result in zip(tool_calls, tool_results, strict=True):
            # Extract Claude-specific metadata from extra fields
            tool_use_id = tool_call.id
            if not tool_use_id:
                logger.warning("No tool_use_id found for %s", tool_call.name)
                continue

            # Convert MCP tool results to Claude format
            claude_blocks = []

            if result.isError:
                # Extract error message from content
                error_msg = "Tool execution failed"
                for content in result.content:
                    if isinstance(content, types.TextContent):
                        error_msg = content.text
                        break
                claude_blocks.append(text_to_content_block(f"Error: {error_msg}"))
            else:
                # Process success content
                for content in result.content:
                    if isinstance(content, types.TextContent):
                        claude_blocks.append(text_to_content_block(content.text))
                    elif isinstance(content, types.ImageContent):
                        claude_blocks.append(base64_to_content_block(content.data))

            # Add tool result
            user_content.append(tool_use_content_block(tool_use_id, claude_blocks))

        # Return as a user message containing all tool results
        return [
            cast(
                "BetaMessageParam",
                {
                    "role": "user",
                    "content": user_content,
                },
            )
        ]

    async def create_user_message(self, text: str) -> BetaMessageParam:
        """Create a user message in Claude's format."""
        return cast("BetaMessageParam", {"role": "user", "content": text})

    def _convert_tools_for_claude(self) -> list[dict]:
        """Convert MCP tools to Claude tool format."""
        claude_tools = []
        self._claude_to_mcp_tool_map = {}  # Reset mapping

        for tool in self._available_tools:
            # Special handling for computer use tools
            if tool.name in ["computer", "computer_anthropic", "anthropic_computer"]:
                # Use Claude's native computer use format with configurable dimensions
                claude_tool = {
                    "type": "computer_20250124",
                    "name": "computer",
                    "display_width_px": self.display_width_px,
                    "display_height_px": self.display_height_px,
                }
                # Map Claude's "computer" back to the actual MCP tool name
                self._claude_to_mcp_tool_map["computer"] = tool.name
            elif tool.name not in self.lifecycle_tools:
                # Convert regular tools
                claude_tool = {
                    "name": tool.name,
                    "description": tool.description or f"Execute {tool.name}",
                    "input_schema": tool.inputSchema
                    or {
                        "type": "object",
                        "properties": {},
                    },
                }
                # Direct mapping for non-computer tools
                self._claude_to_mcp_tool_map[tool.name] = tool.name
            else:
                continue

            claude_tools.append(claude_tool)

        return claude_tools

    def _add_prompt_caching(self, messages: list[BetaMessageParam]) -> list[BetaMessageParam]:
        """Add prompt caching to messages."""
        messages_cached = copy.deepcopy(messages)

        # Mark last user message with cache control
        if messages_cached and messages_cached[-1].get("role") == "user":
            last_content = messages_cached[-1]["content"]
            if isinstance(last_content, list):
                for block in last_content:
                    if block.get("type") not in ["thinking", "redacted_thinking"]:
                        cache_control: BetaCacheControlEphemeralParam = {"type": "ephemeral"}
                        block["cache_control"] = cache_control  # type: ignore[reportGeneralTypeIssues]

        return messages_cached
