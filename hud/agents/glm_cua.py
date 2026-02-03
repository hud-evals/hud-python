"""GLM Computer Use Agent implementation.

GLM-4.6V GUI Agent using native PC action space.
Uses predefined functions like left_click, type, scroll, etc.
with start_box='[x,y]' coordinate format (0-999 range).
"""

from __future__ import annotations

import logging
import re
from typing import Any, ClassVar

import mcp.types as types

from hud.tools.computer.settings import computer_settings
from hud.tools.native_types import NativeToolSpec
from hud.types import AgentResponse, AgentType, BaseAgentConfig, MCPToolCall, MCPToolResult
from hud.utils.types import with_signature

from .base import MCPAgent
from .openai_chat import OpenAIChatAgent
from .types import GLMCUAConfig, GLMCUACreateParams

logger = logging.getLogger(__name__)

# GLM predefined PC action functions
PREDEFINED_GLM_FUNCTIONS = [
    "left_click",
    "right_click",
    "middle_click",
    "hover",
    "left_double_click",
    "left_drag",
    "key",
    "type",
    "scroll",
    "WAIT",
    "DONE",
    "FAIL",
]

# GLM CUA system instructions (from agent.md)
GLM_CUA_INSTRUCTIONS = """You are a GUI Agent, and your primary task is to respond accurately to user requests or questions. In addition to directly answering the user's queries, you can also use tools or perform GUI operations directly until you fulfill the user's request or provide a correct answer. You should carefully read and understand the images and questions provided by the user, and engage in thinking and reflection when appropriate. The coordinates involved are all represented in thousandths (0-999).

# Task Platform
Ubuntu

# Tool Call Format
IMPORTANT: Always use valid JSON for function arguments. Do NOT use XML tags.
Correct: {"action": "left_click", "start_box": "[500, 300]"}
Wrong: {"action": "left_click<arg_key>start_box</arg_key><arg_value>[500, 300]"}

# Some Additional Notes
- Complete tasks autonomously without asking for confirmation.
- If a task cannot be completed, use FAIL().
- Coordinates use 0-999 scale (thousandths of screen dimensions).
""".strip()


class GLMCUAAgent(OpenAIChatAgent):
    """
    GLM Computer Use Agent using native PC action space.
    
    Routes GLM's predefined functions (left_click, type, scroll, etc.)
    to the glm_computer MCP tool with automatic coordinate rescaling.
    """

    metadata: ClassVar[dict[str, Any] | None] = {
        "display_width": computer_settings.GLM_COMPUTER_WIDTH,
        "display_height": computer_settings.GLM_COMPUTER_HEIGHT,
        "coordinate_space": 999,
    }
    required_tools: ClassVar[list[str]] = ["glm_computer"]
    config_cls: ClassVar[type[BaseAgentConfig]] = GLMCUAConfig

    @classmethod
    def agent_type(cls) -> AgentType:
        """Return the AgentType for GLM CUA."""
        return AgentType.GLM_CUA

    # Legacy tool name patterns for backwards compatibility
    _LEGACY_COMPUTER_NAMES = ("glm_computer", "computer_glm", "computer")

    def _legacy_native_spec_fallback(self, tool: types.Tool) -> NativeToolSpec | None:
        """Detect GLM CUA native tools by name for backwards compatibility."""
        name = tool.name
        for pattern in self._LEGACY_COMPUTER_NAMES:
            if name == pattern or name.endswith(f"_{pattern}"):
                logger.debug("Legacy fallback: detected %s as glm_computer tool", name)
                return NativeToolSpec(
                    api_type="function",
                    api_name="glm_computer",
                    role="computer",
                )
        return None

    @with_signature(GLMCUACreateParams)
    @classmethod
    def create(cls, **kwargs: Any) -> GLMCUAAgent:
        """Create a GLMCUAAgent instance."""
        return MCPAgent.create.__func__(cls, **kwargs)  # type: ignore[return-value]

    def __init__(self, params: GLMCUACreateParams | None = None, **kwargs: Any) -> None:
        # Set default system prompt if not provided
        if params and not params.system_prompt:
            params.system_prompt = GLM_CUA_INSTRUCTIONS
        elif "system_prompt" not in kwargs or kwargs.get("system_prompt") is None:
            kwargs["system_prompt"] = GLM_CUA_INSTRUCTIONS

        super().__init__(params, **kwargs)  # type: ignore[arg-type]
        self.config: GLMCUAConfig  # type: ignore[assignment]
        
        self._computer_tool_name = "glm_computer"
        logger.info("GLMCUAAgent initialized with coordinate space 0-999")

    async def get_response(self, messages: list[dict[str, Any]]) -> AgentResponse:
        """Get response from GLM model.
        
        Intercepts predefined GLM functions and routes them to glm_computer.
        Handles DONE/FAIL actions to terminate the task.
        """
        response = await super().get_response(messages)
        
        if response.isError:
            return response

        # Process tool calls - route predefined functions to glm_computer
        if response.tool_calls:
            processed_calls = []
            for tc in response.tool_calls:
                logger.info("GLM raw tool call: %s(%s)", tc.name, tc.arguments)
                
                # First fix any XML-style arguments
                if tc.arguments:
                    tc.arguments = self._fix_xml_args(tc.arguments)
                
                # Check for terminal actions (after fixing XML)
                action = self._get_action(tc)
                if action in ("DONE", "FAIL"):
                    logger.info("Task %s", action)
                    response.done = True
                    response.tool_calls = []  # Don't execute DONE/FAIL as tool
                    if not response.content:
                        response.content = f"Task {action.lower()}."
                    return response
                
                processed = self._process_tool_call(tc)
                logger.info("GLM processed tool call: %s(%s)", processed.name, processed.arguments)
                processed_calls.append(processed)
            response.tool_calls = processed_calls

        return response
    
    def _get_action(self, tool_call: MCPToolCall) -> str | None:
        """Extract action from tool call (handles both direct and routed calls)."""
        # Direct function call (e.g., DONE, FAIL)
        if tool_call.name in ("DONE", "FAIL"):
            return tool_call.name
        
        # glm_computer call with action argument
        if tool_call.name == self._computer_tool_name and tool_call.arguments:
            return tool_call.arguments.get("action")
        
        return None
    
    def _process_tool_call(self, tool_call: MCPToolCall) -> MCPToolCall:
        """Process a tool call, routing predefined GLM functions to glm_computer.
        
        Handles GLM's native PC action space:
        - left_click(start_box='[x,y]', element_info='')
        - type(content='')
        - scroll(start_box='[x,y]', direction='', step=5)
        - etc.
        """
        func_name = tool_call.name
        raw_args = dict(tool_call.arguments) if tool_call.arguments else {}
        
        # Route predefined GLM functions to glm_computer
        if func_name in PREDEFINED_GLM_FUNCTIONS:
            normalized_args: dict[str, Any] = {"action": func_name}
            
            # Pass through GLM's native parameters directly
            # Tool handles start_box/end_box parsing
            if "start_box" in raw_args:
                normalized_args["start_box"] = raw_args["start_box"]
            if "end_box" in raw_args:
                normalized_args["end_box"] = raw_args["end_box"]
            if "content" in raw_args:
                normalized_args["content"] = raw_args["content"]
            if "keys" in raw_args:
                normalized_args["keys"] = raw_args["keys"]
            if "direction" in raw_args:
                normalized_args["direction"] = raw_args["direction"]
            if "step" in raw_args:
                normalized_args["step"] = raw_args["step"]
            if "element_info" in raw_args:
                normalized_args["element_info"] = raw_args["element_info"]
            
            return MCPToolCall(
                name=self._computer_tool_name,
                arguments=normalized_args,
                glm_name=func_name,  # type: ignore[arg-type]
            )
        
        # For glm_computer direct calls, fix any XML-style args
        if func_name == self._computer_tool_name:
            fixed_args = self._fix_xml_args(raw_args)
            return MCPToolCall(
                name=func_name,
                arguments=fixed_args,
            )
        
        # Other tools pass through unchanged
        return tool_call
    
    def _fix_xml_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Fix XML-style arguments from GLM output.
        
        Handles cases like:
        {"action": "left_click\n<arg_key>start_box</arg_key>\n<arg_value>[114, 167]"}
        
        Converts to:
        {"action": "left_click", "start_box": "[114, 167]"}
        """
        fixed = {}
        
        for key, value in args.items():
            if not isinstance(value, str):
                fixed[key] = value
                continue
            
            # Check if value contains XML tags
            if "<arg_key>" not in value:
                fixed[key] = value
                continue
            
            # Extract the actual value before XML tags
            # e.g., "left_click\n<arg_key>..." -> "left_click"
            main_value = value.split("<arg_key>")[0].strip()
            if main_value:
                fixed[key] = main_value
            
            # Parse XML-style key-value pairs
            # Pattern: <arg_key>key</arg_key> followed by optional <arg_value>value</arg_value>
            # Value ends at: </arg_value>, next <arg_key>, or end of string
            # Use greedy match and capture everything after <arg_value> until terminator
            pattern = r"<arg_key>(\w+)</arg_key>\s*<arg_value>([^\"<]+)"
            matches = re.findall(pattern, value)
            
            for arg_name, arg_val in matches:
                arg_name = arg_name.strip()
                arg_val = arg_val.strip()
                if arg_name and arg_val:
                    fixed[arg_name] = arg_val
            
            logger.warning("Fixed XML args: %s -> %s", args, fixed)
        
        return fixed

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[dict[str, Any]]:
        """Format tool results for the next turn.
        
        Simply passes results back - screenshots are already in the results.
        """
        messages: list[dict[str, Any]] = []

        for tool_call, result in zip(tool_calls, tool_results, strict=True):
            # Get the original GLM function name if available
            glm_name = getattr(tool_call, "glm_name", tool_call.name)
            
            # Build content from result
            content_parts: list[dict[str, Any]] = []
            
            # Add text content
            text_parts = []
            for content in result.content:
                if isinstance(content, types.TextContent):
                    text_parts.append(content.text)
                elif isinstance(content, types.ImageContent):
                    # Add screenshot
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{content.mimeType};base64,{content.data}"}
                    })
            
            if text_parts:
                content_parts.insert(0, {"type": "text", "text": "\n".join(text_parts)})
            
            if not content_parts:
                content_parts.append({"type": "text", "text": f"{glm_name} completed"})

            messages.append({
                "role": "user", 
                "content": content_parts,
            })

        return messages


# Alias for backwards compatibility
GLMCUA = GLMCUAAgent
