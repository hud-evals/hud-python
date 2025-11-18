"""Claude MCP Agent implementation."""

from __future__ import annotations

import copy
import logging
from inspect import cleandoc
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import (
    Base64ImageSourceParam,
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
    ToolBash20250124Param,
    ToolParam,
    ToolTextEditor20250124Param,
    ToolTextEditor20250728Param,
    ToolUnionParam,
)

import hud

if TYPE_CHECKING:
    from anthropic.types import (
        CacheControlEphemeralParam,
        ContentBlockParam,
        ToolResultBlockParam,
    )

    from hud.datasets import Task

import mcp.types as types

from hud.settings import settings
from hud.tools.computer.settings import computer_settings
from hud.types import AgentResponse, MCPToolCall, MCPToolResult
from hud.utils.hud_console import HUDConsole

from .base import MCPAgent

logger = logging.getLogger(__name__)


class Claude2Agent(MCPAgent):
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
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 16384,
        use_computer_beta: bool = True,
        validate_api_key: bool = True,
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

        # validate api key if requested
        if validate_api_key:
            try:
                Anthropic(api_key=model_client.api_key).models.list()
            except Exception as e:
                raise ValueError(f"Anthropic API key is invalid: {e}") from e

        self.anthropic_client = model_client
        self.model = model
        self.max_tokens = max_tokens
        self.use_computer_beta = use_computer_beta
        self.hud_console = HUDConsole(logger=logger)

        self.model_name = "Claude2"
        self.checkpoint_name = self.model

        # Track mapping from Claude tool names to MCP tool names
        self.claude_tools: list[ToolUnionParam] = []

    async def initialize(self, task: str | Task | None = None) -> None:
        """Initialize the agent and build tool mappings."""
        await super().initialize(task)
        # Build tool mappings after tools are discovered
        self.claude_tools = self._convert_tools_for_claude()

    async def get_system_messages(self) -> list[Any]:
        """No system messages for Claude because applied in get_response"""
        return []

    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[Any]:
        """Format messages for Claude."""
        # Convert MCP content types to Anthropic content types
        anthropic_blocks: list[ContentBlockParam] = []

        for block in blocks:
            if isinstance(block, types.TextContent):
                # Only include fields that Anthropic expects
                anthropic_blocks.append(
                    TextBlockParam(
                        type="text",
                        text=block.text,
                    )
                )
            elif isinstance(block, types.ImageContent):
                # Convert MCP ImageContent to Anthropic format
                anthropic_blocks.append(
                    ImageBlockParam(
                        type="image",
                        source=Base64ImageSourceParam(
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

        return [MessageParam(role="user", content=anthropic_blocks)]

    @hud.instrument(
        span_type="agent",
        record_args=False,  # Messages can be large
        record_result=True,
    )
    async def get_response(self, messages: list[MessageParam]) -> AgentResponse:
        """Get response from Claude including any tool calls."""

        messages_cached = self._add_prompt_caching(messages)

        response = await self.anthropic_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages_cached,
            tools=self.claude_tools,
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        )

        messages.append(
            MessageParam(
                role="assistant",
                content=response.content,
            )
        )

        # Process response
        result = AgentResponse(content="", tool_calls=[], done=True)

        # Extract text content and reasoning
        text_content = ""
        thinking_content = ""

        for block in response.content:
            if block.type == "tool_use":
                # Create MCPToolCall object with Claude metadata as extra fields
                # Pyright will complain but the tool class accepts extra fields
                tool_call = MCPToolCall(
                    id=block.id,  # canonical identifier for telemetry
                    name=block.name,
                    arguments=block.input,
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
    ) -> list[MessageParam]:
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
            MessageParam(
                role="user",
                content=user_content,
            )
        ]

    async def create_user_message(self, text: str) -> MessageParam:
        """Create a user message in Claude's format."""
        return MessageParam(role="user", content=text)

    def _convert_tools_for_claude(self) -> list[ToolUnionParam]:
        def to_api_tool(tool: types.Tool) -> ToolUnionParam:
            if tool.name == "str_replace_editor":
                return ToolTextEditor20250124Param(
                    type="text_editor_20250124", name="str_replace_editor"
                )
            if tool.name == "str_replace_based_edit_tool":
                return ToolTextEditor20250728Param(
                    type="text_editor_20250728", name="str_replace_based_edit_tool"
                )
            if tool.name == "bash":
                return ToolBash20250124Param(type="bash_20250124", name="bash")
            if tool.name == "computer":
                raise ValueError("Computer tool is not supported for Claude2")

            if not tool.description or not tool.inputSchema:
                raise ValueError(
                    cleandoc(f"""Custom MCP tool {tool.name} requires both a description and inputSchema.
                    Add these by:
                    1. Adding a docstring to your @mcp.tool decorated function for the description
                    2. Using pydantic Field() annotations on function parameters for the schema
                    
                    Example:
                    @mcp.tool()
                    def search_files(
                        query: str = Field(description="Search term to look for"),
                        max_results: int = Field(default=10, description="Maximum number of results to return")
                    ):
                        '''Search for files matching the given query.''' <-- this is processed as the description
                        # implementation here
                    """)
                )
            """Convert a tool to the API format"""
            return ToolParam(
                name=tool.name,
                description=tool.description,
                input_schema=tool.inputSchema,
            )

        return [to_api_tool(tool) for tool in self.get_available_tools()]

    def _add_prompt_caching(self, messages: list[MessageParam]) -> list[MessageParam]:
        """Add prompt caching to messages."""
        messages_cached = copy.deepcopy(messages)
        cache_control: CacheControlEphemeralParam = {"type": "ephemeral"}

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


def base64_to_content_block(base64: str) -> ImageBlockParam:
    """Convert base64 image to Claude content block."""
    return ImageBlockParam(
        type="image",
        source=Base64ImageSourceParam(
            type="base64",
            media_type="image/png",
            data=base64,
        ),
    )


def text_to_content_block(text: str) -> TextBlockParam:
    """Convert text to Claude content block."""
    return {"type": "text", "text": text}


def tool_use_content_block(
    tool_use_id: str, content: list[TextBlockParam | ImageBlockParam]
) -> ToolResultBlockParam:
    """Create tool result content block."""
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
