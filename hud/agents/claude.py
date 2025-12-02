"""Claude MCP Agent implementation."""

from __future__ import annotations

import copy
import logging
from inspect import cleandoc
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from anthropic import Anthropic, AsyncAnthropic, Omit
from anthropic.types import CacheControlEphemeralParam
from anthropic.types.beta import (
    BetaBase64ImageSourceParam,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessageParam,
    BetaTextBlockParam,
    BetaToolBash20250124Param,
    BetaToolComputerUse20250124Param,
    BetaToolParam,
    BetaToolResultBlockParam,
    BetaToolTextEditor20250728Param,
    BetaToolUnionParam,
)

import hud

if TYPE_CHECKING:
    from hud.datasets import Task

import mcp.types as types
from pydantic import ConfigDict

from hud.settings import settings
from hud.tools.computer.settings import computer_settings
from hud.types import AgentResponse, BaseAgentConfig, MCPToolCall, MCPToolResult
from hud.utils.hud_console import HUDConsole
from hud.utils.types import with_signature

from .base import BaseCreateParams, MCPAgent

logger = logging.getLogger(__name__)


class ClaudeConfig(BaseAgentConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "Claude"
    checkpoint_name: str = "claude-sonnet-4-5"
    model_client: AsyncAnthropic | None = None
    max_tokens: int = 16384
    use_computer_beta: bool = True
    validate_api_key: bool = True


class ClaudeCreateParams(BaseCreateParams, ClaudeConfig):
    pass


class ClaudeAgent(MCPAgent):
    """
    Claude agent that uses MCP servers for tool execution.

    This agent uses Claude's native tool calling capabilities but executes
    tools through MCP servers instead of direct implementation.
    """

    metadata: ClassVar[dict[str, Any] | None] = {
        "display_width": computer_settings.ANTHROPIC_COMPUTER_WIDTH,
        "display_height": computer_settings.ANTHROPIC_COMPUTER_HEIGHT,
    }
    config_cls: ClassVar[type[BaseAgentConfig]] = ClaudeConfig

    @with_signature(ClaudeCreateParams)
    @classmethod
    def create(cls, **kwargs: Any) -> ClaudeAgent:  # pyright: ignore[reportIncompatibleMethodOverride]
        return MCPAgent.create.__func__(cls, **kwargs)  # type: ignore[return-value]

    def __init__(self, params: ClaudeCreateParams | None = None, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        self.config: ClaudeConfig

        model_client = self.config.model_client
        if model_client is None:
            api_key = settings.anthropic_api_key
            if not api_key:
                raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY.")
            model_client = AsyncAnthropic(api_key=api_key)

        if self.config.validate_api_key:
            try:
                Anthropic(api_key=model_client.api_key).models.list()
            except Exception as e:
                raise ValueError(f"Anthropic API key is invalid: {e}") from e

        self.anthropic_client = model_client
        self.max_tokens = self.config.max_tokens
        self.use_computer_beta = self.config.use_computer_beta
        self.hud_console = HUDConsole(logger=logger)

        # these will be initialized in _convert_tools_for_claude
        self.has_computer_tool = False
        self.tool_mapping: dict[str, str] = {}
        self.claude_tools: list[BetaToolUnionParam] = []

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
                    BetaTextBlockParam(
                        type="text",
                        text=block.text,
                    )
                )
            elif isinstance(block, types.ImageContent):
                # Convert MCP ImageContent to Anthropic format
                anthropic_blocks.append(
                    BetaImageBlockParam(
                        type="image",
                        source=BetaBase64ImageSourceParam(
                            type="base64",
                            media_type=cast(
                                "Literal['image/jpeg', 'image/png', 'image/gif', 'image/webp']",
                                block.mimeType,
                            ),
                            data=block.data,
                        ),
                    )
                )
            else:
                raise ValueError(f"Unknown content block type: {type(block)}")

        return [BetaMessageParam(role="user", content=anthropic_blocks)]

    @hud.instrument(
        span_type="agent",
        record_args=False,  # Messages can be large
        record_result=True,
    )
    async def get_response(self, messages: list[BetaMessageParam]) -> AgentResponse:
        """Get response from Claude including any tool calls."""

        messages_cached = self._add_prompt_caching(messages)

        # betas to use
        betas = ["fine-grained-tool-streaming-2025-05-14"]
        if self.has_computer_tool:
            betas.append("computer-use-2025-01-24")

        async with self.anthropic_client.beta.messages.stream(
            model=self.config.checkpoint_name,
            system=self.system_prompt if self.system_prompt is not None else Omit(),
            max_tokens=self.max_tokens,
            messages=messages_cached,
            tools=self.claude_tools,
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            betas=betas,
        ) as stream:
            # allow backend to accumulate message content
            async for _ in stream:
                pass
            # get final message
            response = await stream.get_final_message()
            messages.append(BetaMessageParam(role="assistant", content=response.content))

        # Process response
        result = AgentResponse(content="", tool_calls=[], done=True)

        # Extract text content and reasoning
        text_content = ""
        thinking_content = ""

        for block in response.content:
            if block.type == "tool_use":
                tool_call = MCPToolCall(
                    id=block.id,
                    # look up name in tool_mapping if available, otherwise use block name
                    name=self.tool_mapping.get(block.name, block.name),
                    arguments=block.input
                    if isinstance(block.input, dict)
                    else block.input.__dict__,
                )
                result.tool_calls.append(tool_call)
                result.done = False
            elif block.type == "text":
                text_content += block.text
            elif hasattr(block, "type") and block.type == "thinking":
                thinking_content += f"Thinking: {block.thinking}\n"

        result.content = thinking_content + text_content

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
                self.hud_console.warning(f"No tool_use_id found for {tool_call.name}")
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
            BetaMessageParam(
                role="user",
                content=user_content,
            )
        ]

    async def create_user_message(self, text: str) -> BetaMessageParam:
        """Create a user message in Claude's format."""
        return BetaMessageParam(role="user", content=text)

    def _convert_tools_for_claude(self) -> None:
        """Convert MCP tools to Claude API tools."""

        # First pass: identify all computer tools and find the longest match
        available_tools = self.get_available_tools()

        # find potential computer tools by priority
        selected_computer_tool = None
        computer_tool_names_by_priority = ["anthropic_computer", "computer_anthropic", "computer"]
        for computer_tool_name in computer_tool_names_by_priority:
            for tool in available_tools:
                # Check both exact match and suffix match (for prefixed tools)
                if tool.name == computer_tool_name or tool.name.endswith(f"_{computer_tool_name}"):
                    selected_computer_tool = tool
                    break
            if selected_computer_tool:
                break

        def to_api_tool(tool: types.Tool) -> BetaToolUnionParam | None:
            if tool.name == "str_replace_based_edit_tool":
                return BetaToolTextEditor20250728Param(
                    type="text_editor_20250728",
                    name="str_replace_based_edit_tool",
                )
            if tool.name == "bash":
                return BetaToolBash20250124Param(
                    type="bash_20250124",
                    name="bash",
                )
            if selected_computer_tool is not None:
                if tool.name == selected_computer_tool.name:
                    return BetaToolComputerUse20250124Param(
                        type="computer_20250124",
                        name="computer",
                        display_number=1,
                        display_width_px=computer_settings.ANTHROPIC_COMPUTER_WIDTH,
                        display_height_px=computer_settings.ANTHROPIC_COMPUTER_HEIGHT,
                    )
                elif tool.name == "computer":
                    logger.warning(
                        "Renamed tool %s to 'computer', dropping original 'computer' tool",
                        selected_computer_tool.name,
                    )
                    return None

            if tool.description is None or tool.inputSchema is None:
                raise ValueError(
                    cleandoc(f"""MCP tool {tool.name} requires both a description and inputSchema.
                    Add these by:
                    1. Adding a docstring to your @mcp.tool decorated function for the description
                    2. Using pydantic Field() annotations on function parameters for the schema
                    """)
                )

            return BetaToolParam(
                name=tool.name,
                description=tool.description,
                input_schema=tool.inputSchema,
            )

        self.has_computer_tool = False
        self.tool_mapping = {}
        self.claude_tools = []
        for tool in available_tools:
            claude_tool = to_api_tool(tool)
            if claude_tool is None or "name" not in claude_tool:
                continue
            if claude_tool["name"] == "computer":
                self.has_computer_tool = True
            self.tool_mapping[claude_tool["name"]] = tool.name
            self.claude_tools.append(claude_tool)

    def _add_prompt_caching(self, messages: list[BetaMessageParam]) -> list[BetaMessageParam]:
        """Add prompt caching to messages."""
        messages_cached = copy.deepcopy(messages)
        cache_control = CacheControlEphemeralParam(type="ephemeral")

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
                        match block["type"]:
                            case "redacted_thinking" | "thinking":
                                pass
                            case _:
                                block["cache_control"] = cache_control

        return messages_cached


def base64_to_content_block(base64: str) -> BetaImageBlockParam:
    """Convert base64 image to Claude content block."""
    return BetaImageBlockParam(
        type="image",
        source=BetaBase64ImageSourceParam(
            type="base64",
            media_type="image/png",
            data=base64,
        ),
    )


def text_to_content_block(text: str) -> BetaTextBlockParam:
    """Convert text to Claude content block."""
    return {"type": "text", "text": text}


def tool_use_content_block(
    tool_use_id: str, content: list[BetaTextBlockParam | BetaImageBlockParam]
) -> BetaToolResultBlockParam:
    """Create tool result content block."""
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
