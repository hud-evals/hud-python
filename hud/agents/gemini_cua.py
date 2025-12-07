"""Gemini Computer Use Agent implementation."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import mcp.types as types
from google.genai import types as genai_types
from pydantic import ConfigDict, Field

from hud.tools.computer.settings import computer_settings
from hud.types import BaseAgentConfig, MCPToolCall, MCPToolResult
from hud.utils.types import with_signature

from .base import BaseCreateParams, MCPAgent
from .gemini import GeminiAgent, GeminiConfig

logger = logging.getLogger(__name__)

# Predefined Gemini computer use functions
PREDEFINED_COMPUTER_USE_FUNCTIONS = [
    "open_web_browser",
    "click_at",
    "hover_at",
    "type_text_at",
    "scroll_document",
    "scroll_at",
    "wait_5_seconds",
    "go_back",
    "go_forward",
    "search",
    "navigate",
    "key_combination",
    "drag_and_drop",
]

GEMINI_CUA_INSTRUCTIONS = """
You are an autonomous computer-using agent. Follow these guidelines:

1. NEVER ask for confirmation. Complete all tasks autonomously.
2. Do NOT send messages like "I need to confirm before..." or "Do you want me to
   continue?" - just proceed.
3. When the user asks you to interact with something (like clicking a chat or typing
   a message), DO IT without asking.
4. Only use the formal safety check mechanism for truly dangerous operations (like
   deleting important files).
5. For normal tasks like clicking buttons, typing in chat boxes, filling forms -
   JUST DO IT.
6. The user has already given you permission by running this agent. No further
   confirmation is needed.
7. Be decisive and action-oriented. Complete the requested task fully.

Remember: You are expected to complete tasks autonomously. The user trusts you to do
what they asked.
""".strip()


class GeminiCUAConfig(GeminiConfig):
    """Configuration for `GeminiCUAAgent`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "GeminiCUA"
    checkpoint_name: str = "gemini-2.5-computer-use-preview"
    excluded_predefined_functions: list[str] = Field(default_factory=list)


class GeminiCUACreateParams(BaseCreateParams, GeminiCUAConfig):
    pass


class GeminiCUAAgent(GeminiAgent):
    """
    Gemini Computer Use Agent that extends GeminiAgent with computer use capabilities.

    This agent uses Gemini's native computer use capabilities but executes
    tools through MCP servers instead of direct implementation.
    """

    metadata: ClassVar[dict[str, Any] | None] = {
        "display_width": computer_settings.GEMINI_COMPUTER_WIDTH,
        "display_height": computer_settings.GEMINI_COMPUTER_HEIGHT,
    }
    required_tools: ClassVar[list[str]] = ["gemini_computer"]
    config_cls: ClassVar[type[BaseAgentConfig]] = GeminiCUAConfig

    @with_signature(GeminiCUACreateParams)
    @classmethod
    def create(cls, **kwargs: Any) -> GeminiCUAAgent:  # pyright: ignore[reportIncompatibleMethodOverride]
        return MCPAgent.create.__func__(cls, **kwargs)  # type: ignore[return-value]

    def __init__(self, params: GeminiCUACreateParams | None = None, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)  # type: ignore[arg-type]
        self.config: GeminiCUAConfig  # type: ignore[assignment]

        self._computer_tool_name = "gemini_computer"
        self.excluded_predefined_functions = list(self.config.excluded_predefined_functions)

        # Context management: Maximum number of recent turns to keep screenshots for
        # Configurable via GEMINI_MAX_RECENT_TURN_WITH_SCREENSHOTS environment variable
        self.max_recent_turn_with_screenshots = (
            computer_settings.GEMINI_MAX_RECENT_TURN_WITH_SCREENSHOTS
        )

        # Add computer use instructions
        if self.system_prompt:
            self.system_prompt = f"{self.system_prompt}\n\n{GEMINI_CUA_INSTRUCTIONS}"
        else:
            self.system_prompt = GEMINI_CUA_INSTRUCTIONS

    def _to_gemini_tool(self, tool: types.Tool) -> genai_types.Tool | None:
        """Convert a single MCP tool to Gemini tool format.

        Handles gemini_computer tool specially by using Gemini's native ComputerUse.
        """
        if tool.name == self._computer_tool_name:
            # Use Gemini's native computer use capability
            return genai_types.Tool(
                computer_use=genai_types.ComputerUse(
                    environment=genai_types.Environment.ENVIRONMENT_BROWSER,
                    excluded_predefined_functions=self.excluded_predefined_functions,
                )
            )

        # For non-computer tools, use the parent implementation
        return super()._to_gemini_tool(tool)

    async def get_response(self, messages: list[genai_types.Content]) -> Any:
        """Get response from Gemini including any tool calls.

        Extends parent to trim old screenshots before making API call.
        """
        # Trim screenshots from older turns to manage context growth
        self._remove_old_screenshots(messages)

        return await super().get_response(messages)

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[genai_types.Content]:
        """Format tool results into Gemini messages.

        Handles computer tool results specially with screenshots and URLs.
        """
        # Process each tool result
        function_responses = []

        for tool_call, result in zip(tool_calls, tool_results, strict=True):
            # Get the Gemini function name from metadata
            gemini_name = getattr(tool_call, "gemini_name", tool_call.name)

            # Check if this is a computer use tool call
            is_computer_call = tool_call.name == self._computer_tool_name

            # Convert MCP tool results to Gemini format
            response_dict: dict[str, Any] = {}
            url = None

            if result.isError:
                # Extract error message from content
                error_msg = "Tool execution failed"
                for content in result.content:
                    if isinstance(content, types.TextContent):
                        # Check if this is a URL metadata block
                        if content.text.startswith("__URL__:"):
                            url = content.text.replace("__URL__:", "")
                        else:
                            error_msg = content.text
                            break
                response_dict["error"] = error_msg
            else:
                # Process success content
                response_dict["success"] = True

            # Extract URL and screenshot from content (for computer use)
            screenshot_parts = []
            if is_computer_call:
                for content in result.content:
                    if isinstance(content, types.TextContent):
                        # Check if this is a URL metadata block
                        if content.text.startswith("__URL__:"):
                            url = content.text.replace("__URL__:", "")
                    elif isinstance(content, types.ImageContent):
                        # Decode base64 string to bytes for FunctionResponseBlob
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

                # Add URL to response dict (required by Gemini Computer Use model)
                # URL must ALWAYS be present per Gemini API requirements
                response_dict["url"] = url if url else "about:blank"

                # For Gemini Computer Use actions, always acknowledge safety decisions
                requires_ack = False
                if tool_call.arguments:
                    requires_ack = bool(tool_call.arguments.get("safety_decision"))
                if requires_ack:
                    response_dict["safety_acknowledgement"] = True
            else:
                # For non-computer tools, add text content to response
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

    def _extract_tool_call(self, part: genai_types.Part) -> MCPToolCall | None:
        """Extract an MCPToolCall from a function call part.

        Routes predefined Gemini Computer Use functions to the gemini_computer tool
        and normalizes the arguments to MCP tool schema.
        """
        if not part.function_call:
            return None

        func_name = part.function_call.name or ""
        raw_args = dict(part.function_call.args) if part.function_call.args else {}

        # Route predefined computer use functions to the computer tool
        if func_name in PREDEFINED_COMPUTER_USE_FUNCTIONS:
            # Normalize Gemini Computer Use calls to MCP tool schema
            # Ensure 'action' is present and equals the Gemini function name
            normalized_args: dict[str, Any] = {"action": func_name}

            # Map common argument shapes used by Gemini Computer Use
            # 1) Coordinate arrays → x/y
            coord = raw_args.get("coordinate") or raw_args.get("coordinates")
            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                try:
                    normalized_args["x"] = int(coord[0])
                    normalized_args["y"] = int(coord[1])
                except (TypeError, ValueError):
                    # Fall back to raw if casting fails
                    pass

            # Destination coordinate arrays → destination_x/destination_y
            dest = (
                raw_args.get("destination")
                or raw_args.get("destination_coordinate")
                or raw_args.get("destinationCoordinate")
            )
            if isinstance(dest, (list, tuple)) and len(dest) >= 2:
                try:
                    normalized_args["destination_x"] = int(dest[0])
                    normalized_args["destination_y"] = int(dest[1])
                except (TypeError, ValueError):
                    pass

            # Pass through supported fields if present (including direct coords)
            for key in (
                "text",
                "press_enter",
                "clear_before_typing",
                "safety_decision",
                "direction",
                "magnitude",
                "url",
                "keys",
                "x",
                "y",
                "destination_x",
                "destination_y",
            ):
                if key in raw_args:
                    normalized_args[key] = raw_args[key]

            return MCPToolCall(
                name=self._computer_tool_name,
                arguments=normalized_args,
                gemini_name=func_name,  # type: ignore[arg-type]
            )

        # Non-computer tools: use parent implementation
        return super()._extract_tool_call(part)

    def _remove_old_screenshots(self, messages: list[genai_types.Content]) -> None:
        """
        Remove screenshots from old turns to manage context length.
        Keeps only the last N turns with screenshots (configured via
        self.max_recent_turn_with_screenshots).
        """
        turn_with_screenshots_found = 0

        for content in reversed(messages):
            if content.role == "user" and content.parts:
                # Check if content has screenshots (function responses with images)
                has_screenshot = False
                for part in content.parts:
                    if (
                        part.function_response
                        and part.function_response.parts
                        and part.function_response.name in PREDEFINED_COMPUTER_USE_FUNCTIONS
                    ):
                        has_screenshot = True
                        break

                if has_screenshot:
                    turn_with_screenshots_found += 1
                    # Remove the screenshot image if the number of screenshots exceeds the limit
                    if turn_with_screenshots_found > self.max_recent_turn_with_screenshots:
                        for part in content.parts:
                            if (
                                part.function_response
                                and part.function_response.parts
                                and part.function_response.name in PREDEFINED_COMPUTER_USE_FUNCTIONS
                            ):
                                # Clear the parts (screenshots)
                                part.function_response.parts = None
