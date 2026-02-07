"""Claude MCP Agent implementation."""

from __future__ import annotations

import copy
import logging
from inspect import cleandoc
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import mcp.types as types
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, Omit
from anthropic.types import CacheControlEphemeralParam
from anthropic.types.beta import (
    BetaBase64ImageSourceParam,
    BetaBase64PDFSourceParam,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessageParam,
    BetaRequestDocumentBlockParam,
    BetaTextBlockParam,
    BetaToolBash20250124Param,
    BetaToolComputerUse20250124Param,
    BetaToolComputerUse20251124Param,
    BetaToolParam,
    BetaToolResultBlockParam,
    BetaToolTextEditor20250728Param,
    BetaToolUnionParam,
)

from hud.settings import settings
from hud.tools.computer.settings import computer_settings
from hud.tools.native_types import NativeToolSpec
from hud.types import AgentResponse, AgentType, BaseAgentConfig, MCPToolCall, MCPToolResult
from hud.utils.hud_console import HUDConsole
from hud.utils.types import with_signature

from .base import MCPAgent
from .types import ClaudeConfig, ClaudeCreateParams

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


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

    @classmethod
    def agent_type(cls) -> AgentType:
        """Return the AgentType for Claude."""
        return AgentType.CLAUDE

    # Legacy tool name patterns for backwards compatibility
    _LEGACY_COMPUTER_NAMES = ("anthropic_computer", "computer_anthropic", "computer")
    _LEGACY_BASH_NAMES = ("bash",)
    _LEGACY_EDITOR_NAMES = ("str_replace_based_edit_tool", "text_editor", "edit")

    def _legacy_native_spec_fallback(self, tool: types.Tool) -> NativeToolSpec | None:
        """Detect Claude native tools by name for backwards compatibility.

        Supports old environments that expose tools like 'anthropic_computer',
        'bash', or 'str_replace_based_edit_tool' without native_tools metadata.
        """
        import fnmatch

        name = tool.name

        # Check for computer tool patterns
        for pattern in self._LEGACY_COMPUTER_NAMES:
            if name == pattern or name.endswith(f"_{pattern}"):
                logger.debug("Legacy fallback: detected %s as computer tool", name)
                # Opus 4.5/4.6 use the new computer tool version
                model_lower = (self.model or "").lower()
                if any(
                    fnmatch.fnmatch(model_lower, p)
                    for p in ("claude-opus-4-5*", "claude-opus-4-6*")
                ):
                    return NativeToolSpec(
                        api_type="computer_20251124",
                        api_name="computer",
                        beta="computer-use-2025-11-24",
                        role="computer",
                    )
                return NativeToolSpec(
                    api_type="computer_20250124",
                    api_name="computer",
                    beta="computer-use-2025-01-24",
                    role="computer",
                )

        # Check for bash tool patterns
        for pattern in self._LEGACY_BASH_NAMES:
            if name == pattern or name.endswith(f"_{pattern}"):
                logger.debug("Legacy fallback: detected %s as bash tool", name)
                return NativeToolSpec(
                    api_type="bash_20250124",
                    api_name="bash",
                    role="shell",
                )

        # Check for text editor tool patterns
        for pattern in self._LEGACY_EDITOR_NAMES:
            if name == pattern or name.endswith(f"_{pattern}"):
                logger.debug("Legacy fallback: detected %s as text_editor tool", name)
                return NativeToolSpec(
                    api_type="text_editor_20250728",
                    api_name="str_replace_based_edit_tool",
                    role="editor",
                )

        return None

    @with_signature(ClaudeCreateParams)
    @classmethod
    def create(cls, **kwargs: Any) -> ClaudeAgent:  # pyright: ignore[reportIncompatibleMethodOverride]
        return MCPAgent.create.__func__(cls, **kwargs)  # type: ignore[return-value]

    def __init__(self, params: ClaudeCreateParams | None = None, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        self.config: ClaudeConfig

        model_client = self.config.model_client
        if model_client is None:
            # Default to HUD gateway when HUD_API_KEY is available
            if settings.api_key:
                from hud.agents.gateway import build_gateway_client

                model_client = build_gateway_client("anthropic")
            elif settings.anthropic_api_key:
                model_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
            else:
                raise ValueError(
                    "No API key found. Set HUD_API_KEY for HUD gateway, "
                    "or ANTHROPIC_API_KEY for direct Anthropic access."
                )

        self.anthropic_client: AsyncAnthropic | AsyncAnthropicBedrock = model_client
        self.max_tokens = self.config.max_tokens
        self.use_computer_beta = self.config.use_computer_beta
        self.hud_console = HUDConsole(logger=logger)

        # these will be initialized in _convert_tools_for_claude
        self.has_computer_tool = False
        self.tool_mapping: dict[str, str] = {}
        self.claude_tools: list[BetaToolUnionParam] = []
        self._required_betas: set[str] = set()

    def _on_tools_ready(self) -> None:
        """Build Claude-specific tool mappings after tools are discovered."""
        self._convert_tools_for_claude()

    async def get_system_messages(self) -> list[BetaMessageParam]:
        """No system messages for Claude because applied in get_response"""
        return []

    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[BetaMessageParam]:
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

    async def get_response(self, messages: list[BetaMessageParam]) -> AgentResponse:
        """Get response from Claude including any tool calls."""
        messages_cached = self._add_prompt_caching(messages)

        # betas to use - collected during tool conversion based on native specs
        # Pass NOT_GIVEN when empty to avoid sending an empty Anthropic-Beta header
        betas = list(self._required_betas) if self._required_betas else Omit()

        # Bedrock doesn't support .stream() - use create(stream=True) instead
        if isinstance(self.anthropic_client, AsyncAnthropicBedrock):
            try:
                response = await self.anthropic_client.beta.messages.create(
                    model=self.config.model,
                    system=self.system_prompt if self.system_prompt is not None else Omit(),
                    max_tokens=self.max_tokens,
                    messages=messages_cached,
                    tools=self.claude_tools,
                    tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                    betas=betas,
                )
                messages.append(BetaMessageParam(role="assistant", content=response.content))
            except ModuleNotFoundError:
                raise ValueError(
                    "boto3 is required for AWS Bedrock. Use `pip install hud[bedrock]`"
                ) from None
        else:
            # Regular Anthropic client supports .stream()
            async with self.anthropic_client.beta.messages.stream(
                model=self.config.model,
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
                if thinking_content:
                    thinking_content += "\n"
                thinking_content += block.thinking

        result.content = text_content
        if thinking_content:
            result.reasoning = thinking_content

        return result

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[BetaMessageParam]:
        """Format tool results into Claude messages.

        Handles EmbeddedResource (PDFs), images, and text content.
        """
        # Process each tool result
        user_content = []

        for tool_call, result in zip(tool_calls, tool_results, strict=True):
            # Extract Claude-specific metadata from extra fields
            tool_use_id = tool_call.id
            if not tool_use_id:
                self.hud_console.warning(f"No tool_use_id found for {tool_call.name}")
                continue

            # Convert MCP tool results to Claude format
            claude_blocks: list[
                BetaTextBlockParam | BetaImageBlockParam | BetaRequestDocumentBlockParam
            ] = []

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
                    elif isinstance(content, types.EmbeddedResource):
                        # Handle embedded resources (PDFs)
                        resource = content.resource
                        if (
                            isinstance(resource, types.BlobResourceContents)
                            and resource.mimeType == "application/pdf"
                        ):
                            claude_blocks.append(
                                document_to_content_block(base64_data=resource.blob)
                            )

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
        """Convert MCP tools to Claude API tools using native specs.

        Uses shared categorize_tools() for role-based exclusion.
        """
        self.has_computer_tool = False
        self.tool_mapping: dict[str, str] = {}
        self.claude_tools: list[BetaToolUnionParam] = []
        self._required_betas: set[str] = set()

        categorized = self.categorize_tools()

        # Log skipped tools at debug level
        for tool, reason in categorized.skipped:
            logger.debug("Skipping tool %s: %s", tool.name, reason)

        # Log skipped hosted tools (Claude doesn't support hosted tools currently)
        for tool, _spec in categorized.hosted:
            logger.debug("Skipping hosted tool %s for Claude", tool.name)

        # Process native tools
        for tool, spec in categorized.native:
            claude_tool = self._build_native_tool(tool, spec)
            if spec.beta:
                self._required_betas.add(spec.beta)

            api_name = self._get_native_api_name(spec)
            self.tool_mapping[api_name] = tool.name
            self.claude_tools.append(claude_tool)

            if spec.api_type and spec.api_type.startswith("computer"):
                self.has_computer_tool = True

        # Process generic tools
        for tool in categorized.generic:
            if tool.description is None or tool.inputSchema is None:
                raise ValueError(
                    cleandoc(f"""MCP tool {tool.name} requires both a description and inputSchema.
                    Add these by:
                    1. Adding a docstring to your @mcp.tool decorated function for the description
                    2. Using pydantic Field() annotations on function parameters for the schema
                    """)
                )

            claude_tool = BetaToolParam(
                name=tool.name,
                description=tool.description,
                input_schema=tool.inputSchema,
                eager_input_streaming=True,
            )
            self.tool_mapping[tool.name] = tool.name
            self.claude_tools.append(claude_tool)

        # Log actual tools being used
        tool_names = sorted(self.tool_mapping.keys())
        self.console.info(
            f"Agent initialized with {len(tool_names)} tools: {', '.join(tool_names)}"
        )

    def _get_native_api_name(self, spec: NativeToolSpec) -> str:
        """Get the literal API name for a native tool spec.

        Claude's native tools have fixed names that must be used exactly.
        """
        match spec.api_type:
            case "computer_20250124" | "computer_20251124":
                return "computer"
            case "bash_20250124":
                return "bash"
            case "text_editor_20250728":
                return "str_replace_based_edit_tool"
            case _:
                return spec.api_name or spec.api_type or "unknown"

    def _build_native_tool(self, tool: types.Tool, spec: NativeToolSpec) -> BetaToolUnionParam:
        """Build a Claude native tool from a NativeToolSpec.

        Args:
            tool: The MCP tool
            spec: The native spec for Claude

        Returns:
            Claude-specific tool parameter
        """
        match spec.api_type:
            case "computer_20251124":
                display_width = spec.extra.get("display_width")
                display_height = spec.extra.get("display_height")

                if display_width is None or display_height is None:
                    import warnings

                    warnings.warn(
                        "Computer tool missing display dimensions in native_specs.extra. "
                        "Falling back to computer_settings. This fallback will be removed "
                        "in v0.6.0. Update your tool to pass display_width/display_height.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    display_width = display_width or computer_settings.ANTHROPIC_COMPUTER_WIDTH
                    display_height = display_height or computer_settings.ANTHROPIC_COMPUTER_HEIGHT

                return BetaToolComputerUse20251124Param(
                    type="computer_20251124",
                    name="computer",
                    display_number=1,
                    display_width_px=display_width,
                    display_height_px=display_height,
                    enable_zoom=True,
                )
            case "computer_20250124":
                display_width = spec.extra.get("display_width")
                display_height = spec.extra.get("display_height")

                if display_width is None or display_height is None:
                    import warnings

                    warnings.warn(
                        "Computer tool missing display dimensions in native_specs.extra. "
                        "Falling back to computer_settings. This fallback will be removed "
                        "in v0.6.0. Update your tool to pass display_width/display_height.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    display_width = display_width or computer_settings.ANTHROPIC_COMPUTER_WIDTH
                    display_height = display_height or computer_settings.ANTHROPIC_COMPUTER_HEIGHT

                return BetaToolComputerUse20250124Param(
                    type="computer_20250124",
                    name="computer",
                    display_number=1,
                    display_width_px=display_width,
                    display_height_px=display_height,
                )
            case "bash_20250124":
                # Claude expects name to be literal "bash"
                return BetaToolBash20250124Param(
                    type="bash_20250124",
                    name="bash",
                )
            case "text_editor_20250728":
                # Claude expects name to be literal "str_replace_based_edit_tool"
                return BetaToolTextEditor20250728Param(
                    type="text_editor_20250728",
                    name="str_replace_based_edit_tool",
                )
            case _:
                # Unknown native type - fall back to generic function tool
                logger.warning(
                    "Unknown native tool type %s for tool %s, using generic format",
                    spec.api_type,
                    tool.name,
                )
                if tool.description is None or tool.inputSchema is None:
                    raise ValueError(
                        f"MCP tool {tool.name} requires both a description and inputSchema."
                    )
                return BetaToolParam(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.inputSchema,
                )

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


def document_to_content_block(base64_data: str) -> BetaRequestDocumentBlockParam:
    """Convert base64 PDF to Claude document content block."""
    return BetaRequestDocumentBlockParam(
        type="document",
        source=BetaBase64PDFSourceParam(
            type="base64",
            media_type="application/pdf",
            data=base64_data,
        ),
    )


def tool_use_content_block(
    tool_use_id: str,
    content: Sequence[BetaTextBlockParam | BetaImageBlockParam | BetaRequestDocumentBlockParam],
) -> BetaToolResultBlockParam:
    """Create tool result content block."""
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}  # pyright: ignore[reportReturnType]
