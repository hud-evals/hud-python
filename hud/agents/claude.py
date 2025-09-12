"""Claude MCP Agent implementation."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, ClassVar, cast

from anthropic import AsyncAnthropic, BadRequestError
from anthropic.types.beta import BetaContentBlockParam, BetaImageBlockParam, BetaTextBlockParam

import hud

if TYPE_CHECKING:
    from anthropic.types.beta import (
        BetaCacheControlEphemeralParam,
        BetaContentBlockParam,
        BetaImageBlockParam,
        BetaMessageParam,
        BetaTextBlockParam,
        BetaToolResultBlockParam,
    )

    from hud.datasets import Task

import mcp.types as types

from hud.settings import settings
from hud.tools.computer.settings import computer_settings
from hud.types import AgentResponse, MCPToolCall, MCPToolResult

from .base import MCPAgent

logger = logging.getLogger(__name__)


class ClaudeAgent(MCPAgent):
    """
    Claude agent that uses MCP servers for tool execution.

    This agent uses Claude's native tool calling capabilities but executes
    tools through MCP servers instead of direct implementation.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "display_width": computer_settings.ANTHROPIC_COMPUTER_WIDTH,
        "display_height": computer_settings.ANTHROPIC_COMPUTER_HEIGHT,
    }

    def __init__(
        self,
        model_client: AsyncAnthropic | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        use_computer_beta: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Claude MCP agent.

        Args:
            model_client: AsyncAnthropic client (created if not provided)
            model: Claude model to use
            max_tokens: Maximum tokens for response
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
        self.use_computer_beta = use_computer_beta

        self.model_name = self.model

        # Track mapping from Claude tool names to MCP tool names
        self._claude_to_mcp_tool_map: dict[str, str] = {}
        self.claude_tools: list[dict] = []

        # Append Claude-specific instructions to the base system prompt
        claude_instructions = """
        You are Claude, an AI assistant created by Anthropic. You are helpful, harmless, and honest.
        
        When working on tasks:
        1. Be thorough and systematic in your approach
        2. Complete tasks autonomously without asking for confirmation
        3. Use available tools efficiently to accomplish your goals
        4. Verify your actions and ensure task completion
        5. Be precise and accurate in all operations
        
        Remember: You are expected to complete tasks autonomously. The user trusts you to accomplish what they asked.
        """.strip()  # noqa: E501

        # Append Claude instructions to any base system prompt
        if self.system_prompt:
            self.system_prompt = f"{self.system_prompt}\n\n{claude_instructions}"
        else:
            self.system_prompt = claude_instructions

    async def initialize(self, task: str | Task | None = None) -> None:
        """Initialize the agent and build tool mappings."""
        await super().initialize(task)
        # Build tool mappings after tools are discovered
        self._convert_tools_for_claude()

    async def get_system_messages(self) -> list[Any]:
        """No system messages for Claude because applied in get_response"""
        return []

    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[Any]:
        """Format messages for Claude."""
        # Convert MCP content types to Anthropic content types
        anthropic_blocks: list[BetaContentBlockParam] = []

        for block in blocks:
            if isinstance(block, types.TextContent):
                # Only include fields that Anthropic expects
                anthropic_blocks.append(
                    cast(
                        "BetaTextBlockParam",
                        {
                            "type": "text",
                            "text": block.text,
                        },
                    )
                )
            elif isinstance(block, types.ImageContent):
                # Convert MCP ImageContent to Anthropic format
                anthropic_blocks.append(
                    cast(
                        "BetaImageBlockParam",
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": block.mimeType,
                                "data": block.data,
                            },
                        },
                    )
                )
            else:
                # For other types, try to cast but log a warning
                logger.warning("Unknown content block type: %s", type(block))
                anthropic_blocks.append(cast("BetaContentBlockParam", block))

        return [
            cast(
                "BetaMessageParam",
                {
                    "role": "user",
                    "content": anthropic_blocks,
                },
            )
        ]

    @hud.instrument(
        span_type="agent",
        record_args=False,  # Messages can be large
        record_result=True,
    )
    async def get_response(self, messages: list[BetaMessageParam]) -> AgentResponse:
        """Get response from Claude including any tool calls."""

        # Make API call with retry for prompt length
        current_messages = messages.copy()

        while True:
            messages_cached = self._add_prompt_caching(current_messages)

            # Build create kwargs
            create_kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": self.system_prompt,
                "messages": messages_cached,
                "tools": self.claude_tools,
                "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            }

            # Add beta features if using computer tools
            if self.use_computer_beta and any(
                tool.get("type") == "computer_20250124" for tool in self.claude_tools
            ):
                create_kwargs["betas"] = ["computer-use-2025-01-24"]

            try:
                response = await self.anthropic_client.beta.messages.create(**create_kwargs)
                break
            except BadRequestError as e:
                if (
                    "prompt is too long" in str(e)
                    or "request_too_large" in str(e)
                    or e.status_code == 413
                ):
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

        # Find computer tool by priority
        computer_tool_priority = ["anthropic_computer", "computer_anthropic", "computer"]
        selected_computer_tool = None

        for priority_name in computer_tool_priority:
            for tool in self._available_tools:
                # Check both exact match and suffix match (for prefixed tools)
                if tool.name == priority_name or tool.name.endswith(f"_{priority_name}"):
                    selected_computer_tool = tool
                    break
            if selected_computer_tool:
                break

        # Add the selected computer tool if found
        if selected_computer_tool:
            claude_tool = {
                "type": "computer_20250124",
                "name": "computer",
                "display_width_px": self.metadata["display_width"],
                "display_height_px": self.metadata["display_height"],
            }
            # Map Claude's "computer" back to the actual MCP tool name
            self._claude_to_mcp_tool_map["computer"] = selected_computer_tool.name
            claude_tools.append(claude_tool)
            logger.debug("Using %s as computer tool for Claude", selected_computer_tool.name)

        # Add other non-computer tools
        for tool in self._available_tools:
            # Skip computer tools (already handled) and lifecycle tools
            is_computer_tool = any(
                tool.name == priority_name or tool.name.endswith(f"_{priority_name}")
                for priority_name in computer_tool_priority
            )
            if is_computer_tool or tool.name in self.lifecycle_tools:
                continue

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
            claude_tools.append(claude_tool)

        self.claude_tools = claude_tools
        return claude_tools

    def _add_prompt_caching(self, messages: list[BetaMessageParam]) -> list[BetaMessageParam]:
        """Add prompt caching to messages."""
        messages_cached = copy.deepcopy(messages)

        # Mark last user message with cache control
        if (
            messages_cached
            and isinstance(messages_cached[-1], dict)
            and messages_cached[-1].get("role") == "user"
        ):
            last_content = messages_cached[-1]["content"]
            # Content is formatted to be list of ContentBlock in format_blocks and format_message
            if isinstance(last_content, list):
                for block in last_content:
                    # Only add cache control to dict-like block types that support it
                    if isinstance(block, dict):
                        block_type = block.get("type")
                        if block_type in ["text", "image", "tool_use", "tool_result"]:
                            cache_control: BetaCacheControlEphemeralParam = {"type": "ephemeral"}
                            block["cache_control"] = cache_control  # type: ignore[reportGeneralTypeIssues]

        return messages_cached


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
