"""Gemini MCP Agent implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, cast

from google import genai
from google.genai import types as genai_types
from pydantic import ConfigDict

import hud

if TYPE_CHECKING:
    from hud.datasets import Task

import mcp.types as types

from hud.settings import settings
from hud.types import AgentResponse, BaseAgentConfig, MCPToolCall, MCPToolResult
from hud.utils.hud_console import HUDConsole
from hud.utils.types import with_signature

from .base import BaseCreateParams, MCPAgent

logger = logging.getLogger(__name__)


class GeminiConfig(BaseAgentConfig):
    """Configuration for `GeminiAgent`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "Gemini"
    checkpoint_name: str = "gemini-3-pro-preview"
    model_client: genai.Client | None = None
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 8192
    validate_api_key: bool = True


class GeminiCreateParams(BaseCreateParams, GeminiConfig):
    pass


class GeminiAgent(MCPAgent):
    """
    Gemini agent that uses MCP servers for tool execution.

    This agent uses Gemini's native tool calling capabilities but executes
    tools through MCP servers instead of direct implementation.
    """

    metadata: ClassVar[dict[str, Any] | None] = None
    config_cls: ClassVar[type[BaseAgentConfig]] = GeminiConfig

    @with_signature(GeminiCreateParams)
    @classmethod
    def create(cls, **kwargs: Any) -> GeminiAgent:  # pyright: ignore[reportIncompatibleMethodOverride]
        return MCPAgent.create.__func__(cls, **kwargs)  # type: ignore[return-value]

    def __init__(self, params: GeminiCreateParams | None = None, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        self.config: GeminiConfig

        model_client = self.config.model_client
        if model_client is None:
            api_key = settings.gemini_api_key
            if not api_key:
                raise ValueError("Gemini API key not found. Set GEMINI_API_KEY.")
            model_client = genai.Client(api_key=api_key)

        if self.config.validate_api_key:
            try:
                list(model_client.models.list(config=genai_types.ListModelsConfig(page_size=1)))
            except Exception as e:
                raise ValueError(f"Gemini API key is invalid: {e}") from e

        self.gemini_client = model_client
        self.temperature = self.config.temperature
        self.top_p = self.config.top_p
        self.top_k = self.config.top_k
        self.max_output_tokens = self.config.max_output_tokens
        self.hud_console = HUDConsole(logger=logger)

        # Track mapping from Gemini tool names to MCP tool names
        self._gemini_to_mcp_tool_map: dict[str, str] = {}
        self.gemini_tools: genai_types.ToolListUnion = []

    async def initialize(self, task: str | Task | None = None) -> None:
        """Initialize the agent and build tool mappings."""
        await super().initialize(task)
        # Build tool mappings after tools are discovered
        self._convert_tools_for_gemini()

    async def get_system_messages(self) -> list[Any]:
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

    @hud.instrument(
        span_type="agent",
        record_args=False,  # Messages can be large
        record_result=True,
    )
    async def get_response(self, messages: list[genai_types.Content]) -> AgentResponse:
        """Get response from Gemini including any tool calls."""

        # Build generate content config
        generate_config = genai_types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=self.max_output_tokens,
            tools=self.gemini_tools,
            system_instruction=self.system_prompt,
        )

        # Make API call
        response = self.gemini_client.models.generate_content(
            model=self.config.checkpoint_name,
            contents=cast("Any", messages),
            config=generate_config,
        )

        # Append assistant response (including any function_call) so that
        # subsequent FunctionResponse messages correspond to a prior FunctionCall
        if response.candidates and len(response.candidates) > 0 and response.candidates[0].content:
            messages.append(response.candidates[0].content)

        # Process response
        result = AgentResponse(content="", tool_calls=[], done=True)
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
                elif part.text:
                    text_content += part.text
                elif hasattr(part, "thought") and part.thought:
                    thinking_content += f"Thinking: {part.thought}\n"

        # Assign collected tool calls and mark done status
        if collected_tool_calls:
            result.tool_calls = collected_tool_calls
            result.done = False

        # Combine text and thinking for final content
        if thinking_content:
            result.content = thinking_content + text_content
        else:
            result.content = text_content

        return result

    def _extract_tool_call(self, part: genai_types.Part) -> MCPToolCall | None:
        """Extract an MCPToolCall from a function call part.

        Subclasses can override to customize tool call extraction (e.g., normalizing
        computer use calls to a different schema).
        """
        if not part.function_call:
            return None

        func_name = part.function_call.name or ""
        mcp_tool_name = self._gemini_to_mcp_tool_map.get(func_name, func_name)
        raw_args = dict(part.function_call.args) if part.function_call.args else {}

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

            if result.isError:
                # Extract error message from content
                error_msg = "Tool execution failed"
                for content in result.content:
                    if isinstance(content, types.TextContent):
                        error_msg = content.text
                        break
                response_dict["error"] = error_msg
            else:
                # Process success content
                response_dict["success"] = True
                # Add text content to response
                for content in result.content:
                    if isinstance(content, types.TextContent):
                        response_dict["output"] = content.text
                        break

            # Create function response
            function_response = genai_types.FunctionResponse(
                name=gemini_name,
                response=response_dict,
            )
            function_responses.append(function_response)

        # Return as a user message containing all function responses
        return [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(function_response=fr) for fr in function_responses],
            )
        ]

    async def create_user_message(self, text: str) -> genai_types.Content:
        """Create a user message in Gemini's format."""
        return genai_types.Content(role="user", parts=[genai_types.Part(text=text)])

    def _convert_tools_for_gemini(self) -> genai_types.ToolListUnion:
        """Convert MCP tools to Gemini tool format."""
        self._gemini_to_mcp_tool_map = {}  # Reset mapping
        self.gemini_tools = []

        for tool in self.get_available_tools():
            gemini_tool = self._to_gemini_tool(tool)
            if gemini_tool is None:
                continue

            self._gemini_to_mcp_tool_map[tool.name] = tool.name
            self.gemini_tools.append(gemini_tool)

        return self.gemini_tools

    def _to_gemini_tool(self, tool: types.Tool) -> genai_types.Tool | None:
        """Convert a single MCP tool to Gemini tool format.

        Subclasses can override to customize tool conversion (e.g., for computer use).
        """
        # Ensure parameters have proper Schema format
        if tool.description is None or tool.inputSchema is None:
            raise ValueError(f"MCP tool {tool.name} requires both a description and inputSchema.")
        function_decl = genai_types.FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters_json_schema=tool.inputSchema,
        )
        return genai_types.Tool(function_declarations=[function_decl])
