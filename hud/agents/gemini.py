"""Gemini MCP Agent implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, cast

import mcp.types as types
from google import genai
from google.genai import types as genai_types

from hud.settings import settings
from hud.tools.computer.gemini import (
    PREDEFINED_COMPUTER_USE_FUNCTIONS,
    SUPPORTED_GEMINI_COMPUTER_USE_MODELS,
    normalize_gemini_computer_use_args,
)
from hud.tools.computer.settings import computer_settings
from hud.types import AgentType, BaseAgentConfig, InferenceResult, MCPToolCall, MCPToolResult
from hud.utils.hud_console import HUDConsole
from hud.utils.types import with_signature

from .base import MCPAgent
from .types import GeminiConfig, GeminiCreateParams

if TYPE_CHECKING:
    from hud.tools.native_types import NativeToolSpec

logger = logging.getLogger(__name__)


class GeminiAgent(MCPAgent):
    """
    Gemini agent that uses MCP servers for tool execution.

    This agent uses Gemini's native tool calling capabilities but executes
    tools through MCP servers instead of direct implementation.
    """

    metadata: ClassVar[dict[str, Any] | None] = None
    config_cls: ClassVar[type[BaseAgentConfig]] = GeminiConfig

    @classmethod
    def agent_type(cls) -> AgentType:
        """Return the AgentType for Gemini."""
        return AgentType.GEMINI

    # Legacy tool name patterns for backwards compatibility
    _LEGACY_COMPUTER_NAMES = ("gemini_computer", "computer_gemini", "computer")

    def _legacy_native_spec_fallback(self, tool: types.Tool) -> NativeToolSpec | None:
        """Detect Gemini native tools by name for backwards compatibility.

        Supports old environments that expose tools like 'gemini_computer'
        without native_tools metadata.

        Each tuple is ordered by preference — first name that exists wins.
        Only returns a spec if this tool IS that preferred match.
        """
        from hud.tools.native_types import NativeToolSpec

        available = {t.name for t in (self._available_tools or [])} | {tool.name}
        preferred = lambda names: next((n for n in names if n in available), None) == tool.name

        if preferred(self._LEGACY_COMPUTER_NAMES):
            logger.debug("Legacy fallback: detected %s as computer tool", tool.name)
            return NativeToolSpec(
                api_type="computer_use",
                api_name="gemini_computer",
                role="computer",
                supported_models=SUPPORTED_GEMINI_COMPUTER_USE_MODELS,
            )

        return None

    @with_signature(GeminiCreateParams)
    @classmethod
    def create(cls, **kwargs: Any) -> GeminiAgent:  # pyright: ignore[reportIncompatibleMethodOverride]
        return MCPAgent.create.__func__(cls, **kwargs)  # type: ignore[return-value]

    def __init__(self, params: GeminiCreateParams | None = None, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        self.config: GeminiConfig

        model_client = self.config.model_client
        if model_client is None:
            if settings.api_key:
                from hud.agents.gateway import build_gateway_client

                model_client = build_gateway_client("gemini")
            elif settings.gemini_api_key:
                model_client = genai.Client(api_key=settings.gemini_api_key)
                if self.config.validate_api_key:
                    try:
                        list(
                            model_client.models.list(
                                config=genai_types.ListModelsConfig(page_size=1)
                            )
                        )
                    except Exception as e:
                        raise ValueError(f"Gemini API key is invalid: {e}") from e
            else:
                raise ValueError(
                    "No API key found for Gemini.\n"
                    "  • Set HUD_API_KEY to use HUD Gateway"
                    " (add your Gemini key at"
                    " hud.ai/project/secrets for BYOK)\n"
                    "  • Or set GEMINI_API_KEY for direct"
                    " access"
                )

        self.gemini_client: genai.Client = model_client
        self.temperature = self.config.temperature
        self.top_p = self.config.top_p
        self.top_k = self.config.top_k
        self.max_output_tokens = self.config.max_output_tokens
        self.hud_console = HUDConsole(logger=logger)

        # Track mapping from Gemini tool names to MCP tool names
        self._gemini_to_mcp_tool_map: dict[str, str] = {}
        self._computer_tool_name: str | None = None
        self.excluded_predefined_functions = list(self.config.excluded_predefined_functions)
        self.max_recent_turn_with_screenshots = (
            computer_settings.GEMINI_MAX_RECENT_TURN_WITH_SCREENSHOTS
        )
        self.gemini_tools: genai_types.ToolListUnion = []

    def _on_tools_ready(self) -> None:
        """Build Gemini-specific tool mappings after tools are discovered."""
        self._convert_tools_for_gemini()

    async def get_system_messages(self) -> list[genai_types.Content]:
        """No system messages for Gemini because applied in get_response"""
        return []

    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[genai_types.Content]:
        """Format messages for Gemini."""
        # Convert MCP content types to Gemini content types
        gemini_parts: list[genai_types.Part] = []

        for block in blocks:
            if isinstance(block, types.TextContent):
                gemini_parts.append(genai_types.Part(text=block.text))
            elif isinstance(block, types.ImageContent):
                # Convert MCP ImageContent to Gemini format
                # Need to decode base64 string to bytes
                import base64

                image_bytes = base64.b64decode(block.data)
                gemini_parts.append(
                    genai_types.Part.from_bytes(data=image_bytes, mime_type=block.mimeType)
                )
            else:
                # For other types, try to handle but log a warning
                self.hud_console.log(f"Unknown content block type: {type(block)}", level="warning")

        return [genai_types.Content(role="user", parts=gemini_parts)]

    async def get_response(self, messages: list[genai_types.Content]) -> InferenceResult:
        """Get response from Gemini including any tool calls."""
        self._remove_old_screenshots(messages)
        tools = self.gemini_tools

        citations_enabled = bool(
            getattr(self.ctx, "scenario_enable_citations", False) if self.ctx else False
        )
        if citations_enabled and not self._has_google_search_tool():
            tools = [*list(tools), genai_types.Tool(google_search=genai_types.GoogleSearch())]

        # Build generate content config
        generate_config = genai_types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=self.max_output_tokens,
            tools=tools,
            system_instruction=self.system_prompt,
        )

        # Use async API to avoid blocking the event loop
        response = await self.gemini_client.aio.models.generate_content(
            model=self.config.model,
            contents=cast("Any", messages),
            config=generate_config,
        )

        # Append assistant response (including any function_call) so that
        # subsequent FunctionResponse messages correspond to a prior FunctionCall
        if response.candidates and len(response.candidates) > 0 and response.candidates[0].content:
            messages.append(response.candidates[0].content)

        # Process response
        result = InferenceResult(content="", tool_calls=[], done=True)
        collected_tool_calls: list[MCPToolCall] = []

        if not response.candidates:
            self.hud_console.warning("Response has no candidates")
            return result

        candidate = response.candidates[0]

        # Extract text content and function calls
        text_content = ""
        thinking_content = ""

        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if part.function_call:
                    tool_call = self._extract_tool_call(part)
                    if tool_call is not None:
                        collected_tool_calls.append(tool_call)
                elif part.thought is True and part.text:
                    if thinking_content:
                        thinking_content += "\n"
                    thinking_content += part.text
                elif part.text:
                    text_content += part.text

        # Assign collected tool calls and mark done status
        if collected_tool_calls:
            result.tool_calls = collected_tool_calls
            result.done = False

        result.content = text_content
        if thinking_content:
            result.reasoning = thinking_content

        # Extract grounding citations from groundingMetadata
        grounding_meta = getattr(candidate, "grounding_metadata", None)
        if grounding_meta:
            citations: list[dict[str, Any]] = []

            # Build a lookup from chunk index → source info
            chunks = getattr(grounding_meta, "grounding_chunks", None) or []
            chunk_sources: list[dict[str, Any]] = []
            for chunk in chunks:
                web = getattr(chunk, "web", None)
                if web:
                    chunk_sources.append(
                        {
                            "source": getattr(web, "uri", "") or "",
                            "title": getattr(web, "title", None),
                        }
                    )
                else:
                    chunk_sources.append({"source": "", "title": None})

            # Walk groundingSupports for text-segment anchoring
            supports = getattr(grounding_meta, "grounding_supports", None) or []
            seen_chunk_indices: set[int] = set()
            for support in supports:
                segment = getattr(support, "segment", None)
                support_chunk_indices = getattr(support, "grounding_chunk_indices", None) or []
                segment_text = getattr(segment, "text", "") or "" if segment else ""
                start_idx = getattr(segment, "start_index", None) if segment else None
                end_idx = getattr(segment, "end_index", None) if segment else None

                for idx in support_chunk_indices:
                    seen_chunk_indices.add(idx)
                    source_info = chunk_sources[idx] if idx < len(chunk_sources) else {}
                    citations.append(
                        {
                            "type": "grounding",
                            "text": segment_text,
                            "source": source_info.get("source", ""),
                            "title": source_info.get("title"),
                            "start_index": start_idx,
                            "end_index": end_idx,
                        }
                    )

            # Include any chunks not referenced by a support entry
            for idx, src in enumerate(chunk_sources):
                if idx not in seen_chunk_indices and src.get("source"):
                    citations.append(
                        {
                            "type": "grounding",
                            "text": "",
                            "source": src["source"],
                            "title": src.get("title"),
                        }
                    )

            result.citations = citations

        return result

    def _extract_tool_call(self, part: genai_types.Part) -> MCPToolCall | None:
        """Extract an MCPToolCall from a function call part.

        Subclasses can override to customize tool call extraction (e.g., normalizing
        computer use calls to a different schema).
        """
        if not part.function_call:
            return None

        func_name = part.function_call.name or ""
        raw_args = dict(part.function_call.args) if part.function_call.args else {}

        if self._computer_tool_name and func_name in PREDEFINED_COMPUTER_USE_FUNCTIONS:
            return MCPToolCall(
                name=self._computer_tool_name,
                arguments=normalize_gemini_computer_use_args(func_name, raw_args),
                gemini_name=func_name,  # type: ignore[arg-type]
            )

        mcp_tool_name = self._gemini_to_mcp_tool_map.get(func_name, func_name)
        return MCPToolCall(
            name=mcp_tool_name,
            arguments=raw_args,
        )

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[genai_types.Content]:
        """Format tool results into Gemini messages."""
        # Process each tool result
        function_responses = []

        for tool_call, result in zip(tool_calls, tool_results, strict=True):
            # Get the Gemini function name from metadata
            gemini_name = getattr(tool_call, "gemini_name", tool_call.name)

            # Convert MCP tool results to Gemini format
            response_dict: dict[str, Any] = {}
            is_computer_call = (
                self._computer_tool_name is not None and tool_call.name == self._computer_tool_name
            )

            if result.isError:
                # Extract error message from content
                error_msg = "Tool execution failed"
                for content in result.content:
                    if isinstance(content, types.TextContent):
                        if content.text.startswith("__URL__:"):
                            continue
                        error_msg = content.text
                        break
                response_dict["error"] = error_msg
                if is_computer_call:
                    response_dict["url"] = self._extract_url(result) or "about:blank"
            else:
                # Process success content
                response_dict["success"] = True

            screenshot_parts: list[genai_types.FunctionResponsePart] = []
            if is_computer_call:
                url = self._extract_url(result)
                for content in result.content:
                    if isinstance(content, types.ImageContent):
                        import base64

                        image_bytes = base64.b64decode(content.data)
                        screenshot_parts.append(
                            genai_types.FunctionResponsePart(
                                inline_data=genai_types.FunctionResponseBlob(
                                    mime_type=content.mimeType or "image/png",
                                    data=image_bytes,
                                )
                            )
                        )

                response_dict["url"] = url or "about:blank"
                if tool_call.arguments and tool_call.arguments.get("safety_decision"):
                    response_dict["safety_acknowledgement"] = True
            else:
                # Add text content to response
                for content in result.content:
                    if isinstance(content, types.TextContent):
                        response_dict["output"] = content.text
                        break

            # Create function response
            function_response = genai_types.FunctionResponse(
                name=gemini_name,
                response=response_dict,
                parts=screenshot_parts if screenshot_parts else None,
            )
            function_responses.append(function_response)

        # Return as a user message containing all function responses
        return [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(function_response=fr) for fr in function_responses],
            )
        ]

    @staticmethod
    def _extract_url(result: MCPToolResult) -> str | None:
        for content in result.content:
            if isinstance(content, types.TextContent) and content.text.startswith("__URL__:"):
                return content.text.replace("__URL__:", "", 1)
        return None

    def _map_role(self, role: str) -> str:
        """Gemini uses 'model' instead of 'assistant' for non-user turns."""
        if role == "assistant":
            return "model"
        return role

    async def create_user_message(self, text: str) -> genai_types.Content:
        """Create a user message in Gemini's format."""
        return genai_types.Content(role="user", parts=[genai_types.Part(text=text)])

    def _has_google_search_tool(self) -> bool:
        """Check if google_search is already in the tool list."""
        return any(getattr(tool, "google_search", None) is not None for tool in self.gemini_tools)

    def _convert_tools_for_gemini(self) -> None:
        """Convert MCP tools to Gemini tool format using native specs.

        Uses shared categorize_tools() for role-based exclusion.
        """
        self._gemini_to_mcp_tool_map = {}
        self._computer_tool_name = None
        self.gemini_tools = []

        categorized = self._categorized_tools

        # Process hosted tools
        for tool, spec in categorized.hosted:
            gemini_tool = self._build_hosted_tool(spec)
            if gemini_tool:
                self.gemini_tools.append(gemini_tool)
                logger.debug("Added hosted tool %s (%s) for Gemini", tool.name, spec.api_type)

        # Process native client-executed tools
        for tool, spec in categorized.native:
            gemini_tool = self._build_native_tool(tool, spec)
            if gemini_tool:
                self._gemini_to_mcp_tool_map[tool.name] = tool.name
                self.gemini_tools.append(gemini_tool)

        # Process generic function tools
        for tool in categorized.generic:
            gemini_tool = self._to_gemini_tool(tool)
            if gemini_tool:
                self._gemini_to_mcp_tool_map[tool.name] = tool.name
                self.gemini_tools.append(gemini_tool)

        # Log actual tools being used
        tool_names = sorted(self._gemini_to_mcp_tool_map.keys())
        self.console.info(
            f"Agent initialized with {len(tool_names)} tools: {', '.join(tool_names)}"
        )

    def _build_hosted_tool(self, spec: NativeToolSpec) -> genai_types.Tool | None:
        """Build a Gemini hosted tool from a NativeToolSpec.

        Args:
            spec: The native spec with hosted=True

        Returns:
            Gemini Tool with the appropriate hosted configuration
        """
        match spec.api_type:
            case "google_search":
                return genai_types.Tool(google_search=genai_types.GoogleSearch(**spec.extra))
            case "code_execution":
                return genai_types.Tool(code_execution=genai_types.ToolCodeExecution())
            case "url_context":
                return genai_types.Tool(url_context=genai_types.UrlContext())
            case _:
                logger.warning("Unknown hosted tool type: %s", spec.api_type)
                return None

    def _build_native_tool(self, tool: types.Tool, spec: NativeToolSpec) -> genai_types.Tool | None:
        """Build a Gemini native tool from a NativeToolSpec.

        Args:
            tool: The MCP tool
            spec: The native spec for Gemini

        Returns:
            Gemini-specific tool or None if not supported
        """
        match spec.api_type:
            case "computer_use":
                self._computer_tool_name = tool.name
                return genai_types.Tool(
                    computer_use=genai_types.ComputerUse(
                        environment=genai_types.Environment.ENVIRONMENT_BROWSER,
                        excluded_predefined_functions=self.excluded_predefined_functions,
                    )
                )
            case _:
                # Unknown native type - try as function tool
                logger.debug(
                    "Native tool type %s for %s, using function declaration",
                    spec.api_type,
                    tool.name,
                )
                return self._to_gemini_tool(tool)

    def _remove_old_screenshots(self, messages: list[genai_types.Content]) -> None:
        """Drop older Gemini Computer Use screenshots to keep context growth bounded."""
        if self._computer_tool_name is None:
            return

        turn_with_screenshots_found = 0
        for content in reversed(messages):
            if content.role != "user" or not content.parts:
                continue

            has_screenshot = any(
                part.function_response
                and part.function_response.parts
                and part.function_response.name in PREDEFINED_COMPUTER_USE_FUNCTIONS
                for part in content.parts
            )
            if not has_screenshot:
                continue

            turn_with_screenshots_found += 1
            if turn_with_screenshots_found <= self.max_recent_turn_with_screenshots:
                continue

            for part in content.parts:
                if (
                    part.function_response
                    and part.function_response.parts
                    and part.function_response.name in PREDEFINED_COMPUTER_USE_FUNCTIONS
                ):
                    part.function_response.parts = None

    def _to_gemini_tool(self, tool: types.Tool) -> genai_types.Tool | None:
        """Convert a single MCP tool to Gemini function tool format.

        Args:
            tool: MCP tool to convert

        Returns:
            Gemini Tool with function declaration
        """
        if tool.description is None or tool.inputSchema is None:
            raise ValueError(f"MCP tool {tool.name} requires both a description and inputSchema.")

        function_decl = genai_types.FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters_json_schema=tool.inputSchema,
        )
        return genai_types.Tool(function_declarations=[function_decl])
