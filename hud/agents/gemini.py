"""Gemini MCP Agent implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, cast

from google import genai
from google.genai import types as genai_types

import hud

if TYPE_CHECKING:
    from hud.datasets import Task

import mcp.types as types

from hud.settings import settings
from hud.tools.computer.settings import computer_settings
from hud.types import AgentResponse, MCPToolCall, MCPToolResult
from hud.utils.hud_console import HUDConsole

from .base import MCPAgent

logger = logging.getLogger(__name__)

# Maximum number of recent turns to keep screenshots for
MAX_RECENT_TURN_WITH_SCREENSHOTS = 3

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


class GeminiAgent(MCPAgent):
    """
    Gemini agent that uses MCP servers for tool execution.

    This agent uses Gemini's native computer use capabilities but executes
    tools through MCP servers instead of direct implementation.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "display_width": computer_settings.GEMINI_COMPUTER_WIDTH,
        "display_height": computer_settings.GEMINI_COMPUTER_HEIGHT,
    }

    def __init__(
        self,
        model_client: genai.Client | None = None,
        model: str = "gemini-2.5-computer-use-preview-10-2025",
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
        max_output_tokens: int = 8192,
        validate_api_key: bool = True,
        excluded_predefined_functions: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Gemini MCP agent.

        Args:
            model_client: Gemini client (created if not provided)
            model: Gemini model to use
            temperature: Temperature for response generation
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_output_tokens: Maximum tokens for response
            validate_api_key: Whether to validate API key on initialization
            excluded_predefined_functions: List of predefined functions to exclude
            **kwargs: Additional arguments passed to BaseMCPAgent (including mcp_client)
        """
        super().__init__(**kwargs)

        # Initialize client if not provided
        if model_client is None:
            api_key = settings.gemini_api_key
            if not api_key:
                raise ValueError("Gemini API key not found. Set GEMINI_API_KEY.")
            model_client = genai.Client(api_key=api_key)

        # Validate API key if requested
        if validate_api_key:
            try:
                # Simple validation - try to list models
                list(model_client.models.list(config=genai_types.ListModelsConfig(page_size=1)))
            except Exception as e:
                raise ValueError(f"Gemini API key is invalid: {e}") from e

        self.gemini_client = model_client
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.excluded_predefined_functions = excluded_predefined_functions or []
        self.hud_console = HUDConsole(logger=logger)

        self.model_name = self.model

        # Track mapping from Gemini tool names to MCP tool names
        self._gemini_to_mcp_tool_map: dict[str, str] = {}
        self.gemini_tools: list[genai_types.Tool] = []

        # Append Gemini-specific instructions to the base system prompt
        gemini_instructions = """
        You are Gemini, a helpful AI assistant created by Google. You are capable of interacting with computer interfaces.
        
        When working on tasks:
        1. Be thorough and systematic in your approach
        2. Complete tasks autonomously without asking for confirmation
        3. Use available tools efficiently to accomplish your goals
        4. Verify your actions and ensure task completion
        5. Be precise and accurate in all operations
        6. Adapt to the environment and the task at hand
        
        Remember: You are expected to complete tasks autonomously. The user trusts you to accomplish what they asked.
        """.strip()

        # Append Gemini instructions to any base system prompt
        if self.system_prompt:
            self.system_prompt = f"{self.system_prompt}\n\n{gemini_instructions}"
        else:
            self.system_prompt = gemini_instructions

    async def initialize(self, task: str | Task | None = None) -> None:
        """Initialize the agent and build tool mappings."""
        await super().initialize(task)
        # Build tool mappings after tools are discovered
        self._convert_tools_for_gemini()

    async def get_system_messages(self) -> list[Any]:
        """No system messages for Gemini because applied in get_response"""
        return []

    async def format_blocks(
        self, blocks: list[types.ContentBlock]
    ) -> list[genai_types.Content]:
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
    async def get_response(
        self, messages: list[genai_types.Content]
    ) -> AgentResponse:
        """Get response from Gemini including any tool calls."""

        # Build generate content config
        generate_config = genai_types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=self.max_output_tokens,
            tools=cast(Any, self.gemini_tools),
            system_instruction=self.system_prompt,
        )

        # Make API call - using a simpler call pattern
        response = self.gemini_client.models.generate_content(
            model=self.model,
            contents=cast(Any, messages),
            config=generate_config,
        )

        # Append assistant response (including any function_call) so that
        # subsequent FunctionResponse messages correspond to a prior FunctionCall
        if response.candidates and len(response.candidates) > 0 and response.candidates[0].content:
            cast(list[genai_types.Content], messages).append(response.candidates[0].content)

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
                    # Map Gemini tool name back to MCP tool name
                    func_name = part.function_call.name or ""
                    mcp_tool_name = self._gemini_to_mcp_tool_map.get(func_name, func_name)

                    # Create MCPToolCall object with Gemini metadata
                    raw_args = dict(part.function_call.args) if part.function_call.args else {}

                    # Normalize Gemini Computer Use calls to MCP tool schema
                    if part.function_call.name in PREDEFINED_COMPUTER_USE_FUNCTIONS:
                        # Ensure 'action' is present and equals the Gemini function name
                        normalized_args: dict[str, Any] = {"action": part.function_call.name}

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

                        # Use normalized args for computer tool calls
                        final_args = normalized_args
                    else:
                        # Non-computer tools: pass args as-is
                        final_args = raw_args

                    tool_call = MCPToolCall(
                        name=mcp_tool_name,
                        arguments=final_args,
                        id=func_name or mcp_tool_name,  # type: ignore[arg-type]
                        gemini_name=func_name,  # type: ignore[arg-type]
                    )
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

            # Extract URL and screenshot from content
            screenshot_parts = []
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
            # Some actions (e.g., click_at) may require explicit acknowledgement
            # in the corresponding FunctionResponse to proceed.
            if gemini_name in PREDEFINED_COMPUTER_USE_FUNCTIONS:
                # Provide common acknowledgement flags expected by the API
                # (include multiple canonical spellings to maximize compatibility)
                response_dict["acknowledged"] = True
                response_dict["acknowledged_safety"] = True
                response_dict["acknowledgedSafetyDecision"] = True

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

    async def create_user_message(self, text: str) -> genai_types.Content:
        """Create a user message in Gemini's format."""
        return genai_types.Content(role="user", parts=[genai_types.Part(text=text)])

    def _convert_tools_for_gemini(self) -> list[genai_types.Tool]:
        """Convert MCP tools to Gemini tool format."""
        gemini_tools = []
        self._gemini_to_mcp_tool_map = {}  # Reset mapping

        # Find computer tool by priority
        computer_tool_priority = ["gemini_computer", "computer_gemini", "computer"]
        selected_computer_tool = None

        for priority_name in computer_tool_priority:
            for tool in self.get_available_tools():
                # Check both exact match and suffix match (for prefixed tools)
                if tool.name == priority_name or tool.name.endswith(f"_{priority_name}"):
                    selected_computer_tool = tool
                    break
            if selected_computer_tool:
                break

        # Add the selected computer tool if found
        if selected_computer_tool:
            gemini_tool = genai_types.Tool(
                computer_use=genai_types.ComputerUse(
                    environment=genai_types.Environment.ENVIRONMENT_BROWSER,
                    excluded_predefined_functions=self.excluded_predefined_functions,
                )
            )
            # Map Gemini's computer use functions back to the actual MCP tool name
            for func_name in PREDEFINED_COMPUTER_USE_FUNCTIONS:
                if func_name not in self.excluded_predefined_functions:
                    self._gemini_to_mcp_tool_map[func_name] = selected_computer_tool.name

            gemini_tools.append(gemini_tool)
            self.hud_console.debug(
                f"Using {selected_computer_tool.name} as computer tool for Gemini"
            )

        # Add other non-computer tools as custom functions
        for tool in self.get_available_tools():
            # Skip computer tools (already handled)
            if any(
                tool.name == priority_name or tool.name.endswith(f"_{priority_name}")
                for priority_name in computer_tool_priority
            ):
                continue

            # Convert MCP tool schema to Gemini function declaration
            try:
                # Ensure parameters have proper Schema format
                params = tool.inputSchema or {"type": "object", "properties": {}}
                function_decl = genai_types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description or f"Execute {tool.name}",
                    parameters=genai_types.Schema(**params) if isinstance(params, dict) else params,  # type: ignore
                )
                custom_tool = genai_types.Tool(function_declarations=[function_decl])
                gemini_tools.append(custom_tool)
                # Direct mapping for non-computer tools
                self._gemini_to_mcp_tool_map[tool.name] = tool.name
            except Exception:  # noqa: S110
                self.hud_console.warning(f"Failed to convert tool {tool.name} to Gemini format")
                pass

        self.gemini_tools = gemini_tools
        return gemini_tools

    def _remove_old_screenshots(self, messages: list[genai_types.Content]) -> None:
        """
        Remove screenshots from old turns to manage context length.
        Keeps only the last MAX_RECENT_TURN_WITH_SCREENSHOTS turns with screenshots.
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
                    if turn_with_screenshots_found > MAX_RECENT_TURN_WITH_SCREENSHOTS:
                        for part in content.parts:
                            if (
                                part.function_response
                                and part.function_response.parts
                                and part.function_response.name in PREDEFINED_COMPUTER_USE_FUNCTIONS
                            ):
                                # Clear the parts (screenshots)
                                part.function_response.parts = None

