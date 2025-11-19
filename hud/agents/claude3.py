"""Claude MCP Agent implementation."""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime
from inspect import cleandoc
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import httpx
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import (
    Base64ImageSourceParam,
    ContentBlock,
    ImageBlockParam,
    MessageParam,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    TextBlock,
    TextBlockParam,
    ToolBash20250124Param,
    ToolParam,
    ToolTextEditor20250124Param,
    ToolTextEditor20250728Param,
    ToolUnionParam,
    ToolUseBlock,
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
from hud.types import AgentResponse, MCPToolCall, MCPToolResult
from hud.utils.hud_console import HUDConsole

from .base import MCPAgent

logger = logging.getLogger(__name__)


def create_logging_httpx_client(run_id: str) -> httpx.AsyncClient:
    """Create an httpx client that logs Anthropic API requests to files."""
    # Create logs directory for this run
    run_logs_dir = Path("logs") / run_id
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    
    request_counter = {"count": 1}
    
    async def log_request(request: httpx.Request) -> None:
        """Log request if it's going to Anthropic API."""
        # Only log requests to Anthropic API
        if "api.anthropic.com" not in str(request.url):
            return
        
        try:
            # Parse request body as JSON
            if request.content:
                body_json = json.loads(request.content.decode("utf-8"))
                
                # Write to logs/run_id/n.json
                log_file = run_logs_dir / f"{request_counter['count']}.json"
                with open(log_file, "w") as f:  # noqa: ASYNC230
                    json.dump(body_json, f, indent=2)
                
                logger.info("Logged Anthropic request to %s", log_file)
                request_counter["count"] += 1
        except Exception as e:
            logger.warning("Failed to log request: %s", e)
    
    # Create client with event hooks
    return httpx.AsyncClient(
        event_hooks={"request": [log_request]},
        timeout=600.0,  # Match Anthropic's default timeout
    )


class Claude3Agent(MCPAgent):
    """
    Claude agent that uses MCP servers for tool execution.

    This agent uses Claude's native tool calling capabilities but executes
    tools through MCP servers instead of direct implementation.
    """

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

        # Create a unique run identifier for this agent instance
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        logger.info("Created agent with run_id: %s", self.run_id)

        # Initialize client if not provided
        if model_client is None:
            api_key = settings.anthropic_api_key
            if not api_key:
                raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY.")
            
            # Create custom httpx client with request logging
            logging_client = create_logging_httpx_client(self.run_id)
            model_client = AsyncAnthropic(api_key=api_key, http_client=logging_client)
            # model_client = AsyncAnthropic(api_key=api_key)

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

        # Use streaming API with fine-grained tool streaming
        extra_headers = {
            "anthropic-beta": "computer-use-2025-01-24,fine-grained-tool-streaming-2025-05-14",
        }

        response = await self.anthropic_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages_cached,
            tools=self.claude_tools,
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            extra_headers=extra_headers,
            stream=True,
        )

        # Process the stream chunks - exact logic from reference_claude_agent.py
        tool_calls: list[dict[str, Any]] = []
        current_partial_json = ""
        current_text = ""
        current_tool_use: dict[str, Any] | None = None

        async for chunk in response:
            if chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                current_text += chunk.delta.text

            elif chunk.type == "content_block_start" and chunk.content_block.type == "tool_use":
                assert isinstance(chunk, RawContentBlockStartEvent)
                current_tool_use = {
                    "id": chunk.content_block.id,
                    "name": chunk.content_block.name,
                    "input": {},
                    "error": None,
                }

            elif chunk.type == "content_block_delta" and chunk.delta.type == "input_json_delta":
                assert isinstance(chunk, RawContentBlockDeltaEvent)
                if chunk.delta.partial_json:
                    if not current_tool_use:
                        logger.warning(
                            "No current tool use found for JSON delta: %s",
                            chunk.delta.partial_json,
                        )
                        continue
                    current_partial_json += chunk.delta.partial_json

            elif chunk.type == "content_block_stop" and current_tool_use:
                assert isinstance(chunk, RawContentBlockStopEvent)
                # Try to parse the JSON for the tool call
                try:
                    parsed_partial_json = json.loads(current_partial_json)
                    current_tool_use["input"] = parsed_partial_json
                except json.JSONDecodeError as e:
                    error_msg = (
                        f"Unable to parse JSON: {e}. Raw JSON:\n```\n{current_partial_json}\n```"
                    )
                    current_tool_use["error"] = error_msg

                tool_calls.append(current_tool_use)
                current_tool_use = None
                current_partial_json = ""

            elif (
                getattr(chunk, "type", None) == "message_delta"
                and getattr(getattr(chunk, "delta", None), "stop_reason", None) == "max_tokens"
            ) and current_tool_use:
                current_tool_use["error"] = "Response exceeded max_tokens during stream."
                tool_calls.append(current_tool_use)
                current_tool_use = None

        # Build assistant content from streamed chunks

        assistant_content: list[ContentBlock] = []
        if current_text:
            assistant_content.append(TextBlock(type="text", text=current_text))
        for tool_call in tool_calls:
            assistant_content.append(  # noqa: PERF401
                ToolUseBlock(
                    type="tool_use",
                    id=tool_call["id"],
                    name=tool_call["name"],
                    input=tool_call["input"],
                )
            )

        # Add assistant message to conversation
        messages.append(MessageParam(role="assistant", content=assistant_content))

        # Process response into AgentResponse format
        result = AgentResponse(content="", tool_calls=[], done=True)

        # Extract text content
        result.content = current_text

        # Convert tool calls to MCPToolCall format
        for tool_call in tool_calls:
            if tool_call.get("error"):
                logger.warning(
                    "Tool call %s has error: %s",
                    tool_call["name"],
                    tool_call["error"],
                )
            mcp_tool_call = MCPToolCall(
                id=tool_call["id"],
                name=tool_call["name"],
                arguments=tool_call["input"],
            )
            result.tool_calls.append(mcp_tool_call)
            result.done = False

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
