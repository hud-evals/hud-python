"""Claude MCP Agent implementation."""

from __future__ import annotations

import copy
import json
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
    BetaPlainTextSourceParam,
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
from hud.types import AgentType, BaseAgentConfig, InferenceResult, MCPToolCall, MCPToolResult
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

        Each tuple is ordered by preference — first name that exists wins.
        Only returns a spec if this tool IS that preferred match.
        """
        import fnmatch

        available = {t.name for t in (self._available_tools or [])} | {tool.name}
        preferred = lambda names: next((n for n in names if n in available), None) == tool.name

        if preferred(self._LEGACY_COMPUTER_NAMES):
            logger.debug("Legacy fallback: detected %s as computer tool", tool.name)
            model_lower = (self.model or "").lower()
            if any(
                fnmatch.fnmatch(model_lower, p)
                for p in (
                    "claude-opus-4-5*",
                    "claude-opus-4-6*",
                    "claude-sonnet-4-6*",
                )
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

        if preferred(self._LEGACY_BASH_NAMES):
            logger.debug("Legacy fallback: detected %s as bash tool", tool.name)
            return NativeToolSpec(
                api_type="bash_20250124",
                api_name="bash",
                role="shell",
            )

        if preferred(self._LEGACY_EDITOR_NAMES):
            logger.debug("Legacy fallback: detected %s as text_editor tool", tool.name)
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
        self._tool_search_threshold: int | None = None
        self._gated_screenshot_tools: set[str] = set()

    def _on_tools_ready(self) -> None:
        """Build Claude-specific tool mappings after tools are discovered."""
        self._convert_tools_for_claude()

    async def get_system_messages(self) -> list[types.ContentBlock]:
        """No system messages for Claude because applied in get_response"""
        return []

    def _result_from_response_blocks(self, response_blocks: list[Any]) -> InferenceResult:
        """Extract text/tool calls/citations from Anthropic response blocks."""
        result = InferenceResult(content="", tool_calls=[], done=True)
        text_content = ""
        thinking_content = ""
        citations: list[dict[str, Any]] = []

        for block in response_blocks:
            block_type = getattr(block, "type", None)
            if block_type == "tool_use":
                block_input = getattr(block, "input", {})
                mcp_name = self.tool_mapping.get(
                    getattr(block, "name", ""),
                    getattr(block, "name", ""),
                )
                arguments = block_input if isinstance(block_input, dict) else block_input.__dict__
                if mcp_name in self._gated_screenshot_tools:
                    arguments = {**arguments, "take_screenshot_on_click": False}
                    logger.debug(
                        "Injected take_screenshot_on_click=False for gated tool %s", mcp_name
                    )
                tool_call = MCPToolCall(
                    id=getattr(block, "id", ""),
                    name=mcp_name,
                    arguments=arguments,
                )
                result.tool_calls.append(tool_call)
                result.done = False
            elif block_type == "text":
                text = getattr(block, "text", "") or ""
                text_content += text
                block_citations = getattr(block, "citations", None) or []
                for cit in block_citations:
                    cit_dict = {
                        "type": "document_citation",
                        "text": getattr(cit, "cited_text", "") or "",
                        "source": (
                            str(idx)
                            if (idx := getattr(cit, "document_index", None)) is not None
                            else getattr(cit, "document_title", "") or ""
                        ),
                        "title": getattr(cit, "document_title", None),
                        "start_index": getattr(cit, "start_char_index", None),
                        "end_index": getattr(cit, "end_char_index", None),
                    }
                    normalized = self._normalize_citation(cit_dict)
                    if normalized is not None:
                        citations.append(normalized.model_dump(exclude={"provider_data"}))
            elif block_type == "thinking":
                thinking = getattr(block, "thinking", "") or ""
                if thinking:
                    if thinking_content:
                        thinking_content += "\n"
                    thinking_content += thinking

        result.content = text_content
        result.citations = citations
        if thinking_content:
            result.reasoning = thinking_content
        return result

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

    @staticmethod
    def _extract_invalid_tool_json(exc: Exception) -> str | None:
        """Extract malformed tool JSON payload from Anthropic stream errors.

        Returns None when the exception is unrelated to tool JSON parsing.
        """
        message = str(exc)
        parse_error_prefix = "Unable to parse tool parameter JSON from model."
        if parse_error_prefix not in message:
            return None

        marker = "JSON: "
        marker_index = message.find(marker)
        if marker_index == -1:
            return ""

        return message[marker_index + len(marker) :].strip()

    @staticmethod
    def _build_invalid_tool_json_retry_message(invalid_json: str) -> BetaMessageParam:
        """Build a user message prompting the model to re-emit valid tool JSON."""
        wrapped = json.dumps({"INVALID_JSON": invalid_json}, ensure_ascii=True)
        retry_text = (
            "Your previous tool-call arguments were invalid JSON and could not be parsed.\n"
            "Retry the same intended tool call once with valid JSON arguments only.\n"
            "Ensure all strings are quoted and all arrays/objects are valid JSON.\n"
            f"Malformed payload (wrapped): {wrapped}"
        )
        return BetaMessageParam(
            role="user",
            content=[text_to_content_block(retry_text)],
        )

    async def get_response(self, messages: list[BetaMessageParam]) -> InferenceResult:
        """Get response from Claude including any tool calls."""
        messages_cached = self._add_prompt_caching(messages)
        # betas to use - collected during tool conversion based on native specs
        # Only pass betas when non-empty; an empty list can produce an empty
        # anthropic-beta header which the API rejects.
        betas: list[str] | Omit = list(self._required_betas) if self._required_betas else Omit()

        effective_tools: list[BetaToolUnionParam] = list(self.claude_tools)
        if self._tool_search_threshold is not None:
            generic_count = sum(
                1 for t in effective_tools if isinstance(t, dict) and "input_schema" in t
            )
            if generic_count > self._tool_search_threshold:
                logger.debug(
                    "tool_search: %d generic tools > threshold %d, applying defer_loading",
                    generic_count,
                    self._tool_search_threshold,
                )
                effective_tools = [
                    {**t, "defer_loading": True}
                    if isinstance(t, dict) and "input_schema" in t
                    else t
                    for t in effective_tools
                ]

        # Bedrock doesn't support .stream() - use create(stream=True) instead
        if isinstance(self.anthropic_client, AsyncAnthropicBedrock):
            try:
                response = await self.anthropic_client.beta.messages.create(
                    model=self.config.model,
                    system=self.system_prompt if self.system_prompt is not None else Omit(),
                    max_tokens=self.max_tokens,
                    messages=messages_cached,
                    tools=effective_tools,
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
            response = None
            invalid_json_failures = 0
            for _ in range(3):
                messages_cached = self._add_prompt_caching(messages)
                try:
                    async with self.anthropic_client.beta.messages.stream(
                        model=self.config.model,
                        system=self.system_prompt if self.system_prompt is not None else Omit(),
                        max_tokens=self.max_tokens,
                        messages=messages_cached,
                        tools=effective_tools,
                        tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                        betas=betas,
                    ) as stream:
                        # allow backend to accumulate message content
                        async for _ in stream:
                            pass
                        # get final message
                        response = await stream.get_final_message()
                        messages.append(
                            BetaMessageParam(
                                role="assistant",
                                content=response.content,
                            )
                        )
                        break
                except ValueError as exc:
                    invalid_json = self._extract_invalid_tool_json(exc)
                    is_retryable = invalid_json is not None
                    if not is_retryable:
                        raise

                    invalid_json_failures += 1
                    if invalid_json_failures == 1:
                        logger.warning(
                            "Claude returned invalid streamed tool JSON; "
                            "retrying same generation once"
                        )
                        continue

                    if invalid_json_failures == 2:
                        logger.warning(
                            "Claude returned invalid streamed tool JSON twice; "
                            "retrying once with INVALID_JSON guidance"
                        )
                        messages.append(self._build_invalid_tool_json_retry_message(invalid_json))
                        continue

                    raise

            if response is None:
                raise ValueError("Claude response missing after stream retries")

        # Process response
        result = self._result_from_response_blocks(list(response.content))

        return result

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[BetaMessageParam]:
        """Format tool results into Claude messages.

        Handles EmbeddedResource (PDFs), images, and text content.
        """
        citations_enabled = bool(
            getattr(self.ctx, "scenario_enable_citations", False) if self.ctx else False
        )

        # Process each tool result
        user_content: list[BetaToolResultBlockParam | BetaRequestDocumentBlockParam] = []

        for tool_call, result in zip(tool_calls, tool_results, strict=True):
            tool_use_id = tool_call.id
            if not tool_use_id:
                self.hud_console.warning(f"No tool_use_id found for {tool_call.name}")
                continue

            # Blocks placed inside the tool_result (text, images)
            claude_blocks: list[
                BetaTextBlockParam | BetaImageBlockParam | BetaRequestDocumentBlockParam
            ] = []
            # Citable document blocks placed as siblings after the tool_result
            # so Claude's citation system indexes them properly.
            sibling_docs: list[BetaRequestDocumentBlockParam] = []

            if result.isError:
                error_msg = "Tool execution failed"
                for content in result.content:
                    if isinstance(content, types.TextContent):
                        error_msg = content.text
                        break
                claude_blocks.append(text_to_content_block(f"Error: {error_msg}"))
            else:
                for content in result.content:
                    if isinstance(content, types.TextContent):
                        claude_blocks.append(text_to_content_block(content.text))
                        if citations_enabled:
                            sibling_docs.append(
                                text_document_block(content.text, title=tool_call.name)
                            )
                    elif isinstance(content, types.ImageContent):
                        claude_blocks.append(base64_to_content_block(content.data))
                    elif isinstance(content, types.EmbeddedResource):
                        resource = content.resource
                        if (
                            isinstance(resource, types.BlobResourceContents)
                            and resource.mimeType == "application/pdf"
                        ):
                            target = sibling_docs if citations_enabled else claude_blocks
                            target.append(
                                document_to_content_block(
                                    base64_data=resource.blob,
                                    enable_citations=citations_enabled,
                                )
                            )

            user_content.append(tool_use_content_block(tool_use_id, claude_blocks))
            user_content.extend(sibling_docs)

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
        self._tool_search_threshold = None
        self._gated_screenshot_tools: set[str] = set()

        categorized = self._categorized_tools

        # Process hosted tools
        for tool, spec in categorized.hosted:
            if not spec.api_type:
                logger.debug("Skipping hosted tool %s: no api_type", tool.name)
                continue
            tool_def: dict[str, Any] = {
                "type": spec.api_type,
                "name": spec.api_name or tool.name,
            }
            api_extra = {k: v for k, v in spec.extra.items() if k != "threshold"}
            tool_def.update(api_extra)
            if spec.beta:
                self._required_betas.add(spec.beta)
            if "threshold" in spec.extra:
                self._tool_search_threshold = spec.extra["threshold"]
            self.claude_tools.append(tool_def)  # type: ignore[arg-type]
            logger.debug("Added hosted tool %s (%s) for Claude", tool.name, spec.api_type)

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
            if spec.api_type == "computer_20251124":
                self._gated_screenshot_tools.add(tool.name)
                logger.debug("Screenshot gating enabled for tool %s (computer_20251124)", tool.name)

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


def text_document_block(text: str, *, title: str | None = None) -> BetaRequestDocumentBlockParam:
    """Wrap plain text as a citable document block."""
    block = BetaRequestDocumentBlockParam(
        type="document",
        source=BetaPlainTextSourceParam(
            type="text",
            media_type="text/plain",
            data=text,
        ),
        citations={"enabled": True},
    )
    if title:
        block["title"] = title
    return block


def document_to_content_block(
    base64_data: str, *, enable_citations: bool = False
) -> BetaRequestDocumentBlockParam:
    """Convert base64 PDF to Claude document content block."""
    block = BetaRequestDocumentBlockParam(
        type="document",
        source=BetaBase64PDFSourceParam(
            type="base64",
            media_type="application/pdf",
            data=base64_data,
        ),
    )
    if enable_citations:
        block["citations"] = {"enabled": True}
    return block


def tool_use_content_block(
    tool_use_id: str,
    content: Sequence[BetaTextBlockParam | BetaImageBlockParam | BetaRequestDocumentBlockParam],
) -> BetaToolResultBlockParam:
    """Create tool result content block."""
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}  # pyright: ignore[reportReturnType]
