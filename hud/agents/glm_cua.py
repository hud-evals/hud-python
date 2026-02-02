"""GLM Computer Use Agent implementation for GLM-4.5V."""

from __future__ import annotations

import base64
import logging
import re
from io import BytesIO
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


# PC Action Space for Desktop Environment (from GLM-4.5V agent.md)
PC_ACTION_SPACE = ("""
### {left,right,middle}_click

Call rule: `{left,right,middle}_click(start_box='[x,y]', element_info='')`
{
    'name': ['left_click', 'right_click', 'middle_click'],
    'description': 'Perform a left/right/middle mouse click at the specified coordinates on the screen.',
    'parameters': {
        'type': 'object',
        'properties': {
            'start_box': {
                'type': 'array',
                'items': {'type': 'integer'},
                'description': 'Coordinates [x,y] where to perform the click, normalized to 0-999 range.'
            },
            'element_info': {
                'type': 'string',
                'description': 'Optional text description of the UI element being clicked.'
            }
        },
        'required': ['start_box']
    }
}

### hover

Call rule: `hover(start_box='[x,y]', element_info='')`
{
    'name': 'hover',
    'description': 'Move the mouse pointer to the specified coordinates without performing any click action.',
    'parameters': {'type': 'object', 'properties': {'start_box': {'type': 'array', 'items': {'type': 'integer'}, 'description': 'Coordinates [x,y] where to move the mouse pointer, normalized to 0-999 range.'}, 'element_info': {'type': 'string', 'description': 'Optional text description of the UI element being hovered over.'}}, 'required': ['start_box']}
}

### left_double_click

Call rule: `left_double_click(start_box='[x,y]', element_info='')`
{
    'name': 'left_double_click',
    'description': 'Perform a left mouse double-click at the specified coordinates on the screen.',
    'parameters': {'type': 'object', 'properties': {'start_box': {'type': 'array', 'items': {'type': 'integer'}, 'description': 'Coordinates [x,y] where to perform the double-click, normalized to 0-999 range.'}, 'element_info': {'type': 'string', 'description': 'Optional text description of the UI element being double-clicked.'}}, 'required': ['start_box']}
}

### left_drag

Call rule: `left_drag(start_box='[x1,y1]', end_box='[x2,y2]', element_info='')`
{
    'name': 'left_drag',
    'description': 'Drag the mouse from starting coordinates to ending coordinates while holding the left mouse button.',
    'parameters': {'type': 'object', 'properties': {'start_box': {'type': 'array', 'items': {'type': 'integer'}, 'description': 'Starting coordinates [x1,y1] for the drag operation, normalized to 0-999 range.'}, 'end_box': {'type': 'array', 'items': {'type': 'integer'}, 'description': 'Ending coordinates [x2,y2] for the drag operation, normalized to 0-999 range.'}, 'element_info': {'type': 'string', 'description': 'Optional text description of the UI element being dragged.'}}, 'required': ['start_box', 'end_box']}
}

### key

Call rule: `key(keys='')`
{
    'name': 'key',
    'description': 'Simulate pressing a single key or combination of keys on the keyboard.',
    'parameters': {'type': 'object', 'properties': {'keys': {'type': 'string', 'description': "The key or key combination to press. Use '+' to separate keys in combinations (e.g., 'ctrl+c', 'alt+tab')."}}, 'required': ['keys']}
}

### type

Call rule: `type(content='')`
{
    'name': 'type',
    'description': 'Type text content into the currently focused text input field.',
    'parameters': {'type': 'object', 'properties': {'content': {'type': 'string', 'description': 'The text content to be typed into the active text field.'}}, 'required': ['content']}
}

### scroll

Call rule: `scroll(start_box='[x,y]', direction='', step=5, element_info='')`
{
    'name': 'scroll',
    'description': 'Scroll an element at the specified coordinates in the specified direction.',
    'parameters': {'type': 'object', 'properties': {'start_box': {'type': 'array', 'items': {'type': 'integer'}, 'description': 'Coordinates [x,y] of the element or area to scroll, normalized to 0-999 range.'}, 'direction': {'type': 'string', 'enum': ['down', 'up'], 'description': "The direction to scroll: 'down' or 'up'."}, 'step': {'type': 'integer', 'default': 5, 'description': 'Number of wheel steps to scroll, default is 5.'}, 'element_info': {'type': 'string', 'description': 'Optional text description of the UI element being scrolled.'}}, 'required': ['start_box', 'direction']}
}

### WAIT

Call rule: `WAIT()`
{
    'name': 'WAIT',
    'description': 'Wait for 5 seconds before proceeding to the next action.',
    'parameters': {'type': 'object', 'properties': {}, 'required': []}
}

### DONE

Call rule: `DONE()`
{
    'name': 'DONE',
    'description': 'Indicate that the current task has been completed successfully.',
    'parameters': {'type': 'object', 'properties': {}, 'required': []}
}

### FAIL

Call rule: `FAIL()`
{
    'name': 'FAIL',
    'description': 'Indicate that the current task cannot be completed.',
    'parameters': {'type': 'object', 'properties': {}, 'required': []}
}
""").strip()


GLM_CUA_SYSTEM_PROMPT = ("""You are a GUI Agent, and your primary task is to respond accurately to user requests or questions. In addition to directly answering the user's queries, you can also use tools or perform GUI operations directly until you fulfill the user's request or provide a correct answer. You should carefully read and understand the images and questions provided by the user, and engage in thinking and reflection when appropriate. The coordinates involved are all represented in thousandths (0-999).

# Task Platform
Ubuntu

# Action Space
{action_space}

# Output Format
Plain text explanation with action(param='...')
Memory:
[{{"key": "value"}}, ...]

# Some Additional Notes
- I'll give you the most recent 4 history screenshots(shrunked to 50%*50%) along with the historical action steps.
- You should put the key information you *have to remember* in a separated memory part. The content in this part should be a dict list.
- My computer's password is "password", feel free to use it when you need sudo rights.
""").strip()


class GLMCUA(OpenAIChatAgent):
    """
    GLM Computer Use Agent for GLM-4.5V.

    Uses prompt-based action selection with text parsing, routing actions
    to the GLMComputerTool MCP tool.
    """

    metadata: ClassVar[dict[str, Any] | None] = {
        "display_width": computer_settings.GLM_COMPUTER_WIDTH,
        "display_height": computer_settings.GLM_COMPUTER_HEIGHT,
    }
    required_tools: ClassVar[list[str]] = ["glm_computer"]
    config_cls: ClassVar[type[BaseAgentConfig]] = GLMCUAConfig

    # Legacy tool name patterns for backwards compatibility
    _LEGACY_COMPUTER_NAMES = ("glm_computer", "computer_glm", "computer")

    @classmethod
    def agent_type(cls) -> AgentType:
        """Return the AgentType for GLM CUA."""
        return AgentType.GLM_CUA

    def _legacy_native_spec_fallback(self, tool: types.Tool) -> NativeToolSpec | None:
        """Detect GLM CUA native tools by name for backwards compatibility."""
        name = tool.name

        for pattern in self._LEGACY_COMPUTER_NAMES:
            if name == pattern or name.endswith(f"_{pattern}"):
                logger.debug("Legacy fallback: detected %s as glm_computer tool", name)
                return NativeToolSpec(
                    api_type="glm_computer",
                    api_name="glm_computer",
                    role="computer",
                )

        return None

    @with_signature(GLMCUACreateParams)
    @classmethod
    def create(cls, **kwargs: Any) -> GLMCUA:  # pyright: ignore[reportIncompatibleMethodOverride]
        return MCPAgent.create.__func__(cls, **kwargs)  # type: ignore[return-value]

    def __init__(self, params: GLMCUACreateParams | None = None, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)  # type: ignore[arg-type]
        self.config: GLMCUAConfig  # type: ignore[assignment]

        self._computer_tool_name = "glm_computer"

        # History management
        self.max_history_screenshots = self.config.max_history_screenshots
        self.history_image_scale = self.config.history_image_scale

        # History storage: list of (thought, action, screenshot_base64)
        self._history: list[tuple[str, str, str | None]] = []
        self._memory: str = "[]"

        # Set system prompt with action space
        action_prompt = GLM_CUA_SYSTEM_PROMPT.format(action_space=PC_ACTION_SPACE)
        if self.system_prompt:
            self.system_prompt = f"{self.system_prompt}\n\n{action_prompt}"
        else:
            self.system_prompt = action_prompt

    def _format_as_function_call(self, args: dict[str, Any]) -> str:
        """Format parsed args as a function-call string for logging."""
        action = args.get("action", "unknown")
        params = []
        for k, v in args.items():
            if k == "action":
                continue
            if isinstance(v, str):
                # Truncate long values
                display_v = v if len(v) <= 50 else v[:47] + "..."
                params.append(f'{k}="{display_v}"')
            else:
                params.append(f"{k}={v}")
        return f"{action}({', '.join(params)})"

    def _parse_action(self, response: str) -> dict[str, Any] | None:
        """Parse action from GLM response text.

        Supports multiple formats:
        1. <|begin_of_box|>action(...)<|end_of_box|> - original GLM-4.5V format
        2. action(start_box='[x,y]', ...) - function-call style
        3. action <arg_key>key</arg_key> <arg_value>val</arg_value> - HUD gateway XML format
        4. Standalone action name on its own line (for truncated responses)
        """
        result = None

        # Format 1: extract from special tokens
        pattern = r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>"
        match = re.search(pattern, response)
        if match:
            result = self._parse_function_style(match.group(1).strip())
            if result:
                logger.info("Parsed (box tokens): %s", self._format_as_function_call(result))
                return result

        # Format 3: HUD gateway XML-style format (complete or partial/malformed)
        # e.g. "left_click <arg_key>start_box</arg_key> <arg_value>[246, 349]</arg_value>"
        # Also handles malformed: "left_click start_box</arg_key> <arg_value>..."
        if "<arg_key>" in response or "</arg_key>" in response or "<arg_value>" in response:
            result = self._parse_xml_style(response)
            if result and len(result) > 1:  # Has action + at least one argument
                logger.info("Parsed (XML): %s", self._format_as_function_call(result))
                return result

        # Format 2: function-call style fallback
        fallback_pattern = r"[\w_]+\([^)]*\)"
        matched = re.findall(fallback_pattern, response)
        if matched:
            result = self._parse_function_style(matched[0])
            if result:
                logger.info("Parsed (function): %s", self._format_as_function_call(result))
                return result

        # Format 4: Standalone action name (truncated response or simple action)
        # Look for known action names on their own line or at end
        for action in self._GLM_ACTIONS:
            # Match action name as whole word, possibly at end of response
            standalone_pattern = rf"\b({action})\b\s*(?:\n|$|<)"
            standalone_match = re.search(standalone_pattern, response)
            if standalone_match:
                logger.info("Parsed (standalone/truncated): %s()", action)
                return {"action": action}

        return None

    def _parse_function_style(self, action_str: str) -> dict[str, Any] | None:
        """Parse function-call style: action(arg='value', ...)"""
        action_match = re.match(r"(\w+)\((.*)\)", action_str)
        if not action_match:
            # Might be just action name like "DONE" or "WAIT"
            action_name_only = re.match(r"^(\w+)$", action_str.strip())
            if action_name_only:
                return {"action": action_name_only.group(1)}
            return None

        action_name = action_match.group(1)
        args_str = action_match.group(2)

        args: dict[str, Any] = {"action": action_name}

        # Extract named arguments like start_box='[500,500]'
        arg_pattern = r"(\w+)='([^']*)'"
        for arg_match in re.finditer(arg_pattern, args_str):
            args[arg_match.group(1)] = arg_match.group(2)

        # Also check for unquoted arguments like step=5
        unquoted_pattern = r"(\w+)=(\d+)"
        for arg_match in re.finditer(unquoted_pattern, args_str):
            args[arg_match.group(1)] = int(arg_match.group(2))

        return args

    def _parse_xml_style(self, response: str) -> dict[str, Any] | None:
        """Parse HUD gateway XML-style format.

        Handles both well-formed and malformed XML:
        - "left_click <arg_key>start_box</arg_key> <arg_value>[246, 349]</arg_value>"
        - "left_click start_box</arg_key> <arg_value>[494, 254]</arg_value>" (missing opening tag)
        """
        # Extract action name - look for known action followed by space/tag
        action_pattern = r"\b(" + "|".join(self._GLM_ACTIONS) + r")\b"
        action_match = re.search(action_pattern, response)
        if not action_match:
            return None

        action_name = action_match.group(1)
        args: dict[str, Any] = {"action": action_name}

        # Pattern 1: Well-formed <arg_key>key</arg_key> <arg_value>val</arg_value>
        kv_pattern = r"<arg_key>(\w+)</arg_key>\s*<arg_value>([^<]*)</arg_value>"
        for kv_match in re.finditer(kv_pattern, response):
            key = kv_match.group(1)
            value = kv_match.group(2).strip()
            args[key] = value

        # Pattern 2: Malformed - key</arg_key> <arg_value>val</arg_value> (missing opening tag)
        malformed_pattern = r"\b(\w+)</arg_key>\s*<arg_value>([^<]*)</arg_value>"
        for kv_match in re.finditer(malformed_pattern, response):
            key = kv_match.group(1)
            # Skip if already found via well-formed pattern
            if key not in args:
                value = kv_match.group(2).strip()
                args[key] = value

        return args

    def _extract_memory(self, response: str) -> str:
        """Extract memory section from response."""
        memory_pattern = r"Memory:(.*?)$"
        memory_match = re.search(memory_pattern, response, re.DOTALL)
        return memory_match.group(1).strip() if memory_match else "[]"

    def _extract_thought(self, response: str) -> str:
        """Extract thought/reasoning from response."""
        # Try to get text before Memory:
        if "</think>" in response:
            thought_pattern = r"</think>(.*?)Memory:"
        else:
            thought_pattern = r"^(.*?)Memory:"

        thought_match = re.search(thought_pattern, response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
            # Clean up special tokens
            thought = (
                thought.replace("<|begin_of_box|>", "")
                .replace("<|end_of_box|>", "")
                .strip()
            )
            return thought
        return ""

    def _resize_image(self, base64_data: str, scale: float) -> str:
        """Resize base64 image by scale factor."""
        try:
            from PIL import Image

            # Decode base64
            image_data = base64.b64decode(base64_data)
            image = Image.open(BytesIO(image_data))

            # Resize
            new_size = (int(image.width * scale), int(image.height * scale))
            resized = image.resize(new_size, Image.Resampling.LANCZOS)

            # Encode back to base64
            buffer = BytesIO()
            resized.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            logger.warning("Failed to resize image: %s", e)
            return base64_data

    def _build_history_content(self) -> list[dict[str, Any]]:
        """Build history content with screenshots for the prompt."""
        content: list[dict[str, Any]] = []

        # Add history header
        content.append({"type": "text", "text": "# Historical Actions and Current Memory\nHistory:"})

        # Add history entries (last N with screenshots)
        total_history = len(self._history)
        for i, (thought, action, screenshot) in enumerate(self._history):
            step_num = i + 1

            # For older entries beyond max_history_screenshots, omit screenshot
            if total_history - i > self.max_history_screenshots:
                content.append({
                    "type": "text",
                    "text": f"step {step_num}: (Omitted in context.)\n Thought: {thought}\nAction: {action}"
                })
            else:
                content.append({
                    "type": "text",
                    "text": f"step {step_num}: Screenshot:"
                })
                if screenshot:
                    # Add resized screenshot
                    resized = self._resize_image(screenshot, self.history_image_scale)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{resized}"}
                    })
                content.append({
                    "type": "text",
                    "text": f"Thought: {thought}\nAction: {action}"
                })

        # Add current memory
        content.append({"type": "text", "text": f"\nMemory:\n{self._memory}\n\nCurrent Screenshot:"})

        return content

    # GLM action names that should be routed to glm_computer
    _GLM_ACTIONS = {
        "left_click", "right_click", "middle_click", "left_double_click",
        "hover", "left_drag", "key", "type", "scroll", "WAIT", "DONE", "FAIL",
        "screenshot",
    }

    async def get_response(self, messages: list[dict[str, Any]]) -> AgentResponse:
        """Get response from GLM and parse action from text output.

        Handles both:
        1. Native function calls from HUD gateway (remaps to glm_computer)
        2. Text-based action output (parses and routes to glm_computer)
        """
        # Get response from parent (may have native tool_calls)
        response = await super().get_response(messages)

        # If there was an error, return as-is
        if response.isError:
            return response

        text_response = response.content or ""

        # Extract memory and thought for history
        self._memory = self._extract_memory(text_response)
        thought = self._extract_thought(text_response)

        # Case 1: Parent returned native tool calls - remap them to glm_computer
        if response.tool_calls:
            remapped_calls = []
            for tc in response.tool_calls:
                if tc.name in self._GLM_ACTIONS:
                    # Remap action like "left_click" to "glm_computer" tool
                    args = dict(tc.arguments) if tc.arguments else {}
                    args["action"] = tc.name
                    # Log the remapped function call
                    logger.info("Parsed (native): %s", self._format_as_function_call(args))
                    remapped_calls.append(MCPToolCall(
                        name=self._computer_tool_name,
                        arguments=args,
                    ))
                else:
                    # Pass through non-GLM tool calls
                    remapped_calls.append(tc)

            return AgentResponse(
                content=text_response,
                reasoning=thought,
                tool_calls=remapped_calls,
                raw=response.raw,
            )

        # Case 2: No native tool calls - try parsing from text
        action_args = self._parse_action(text_response)

        if not action_args:
            # No action found - might be a final answer or thinking
            logger.warning("Could not parse action from GLM response: %s", text_response[:200])
            return AgentResponse(
                content=text_response,
                reasoning=thought,
                tool_calls=[],
                done=False,
                raw=response.raw,
            )

        action_name = action_args.get("action", "")

        # Handle terminal actions
        if action_name == "DONE":
            return AgentResponse(
                content=text_response,
                reasoning=thought,
                done=True,
                raw=response.raw,
            )
        elif action_name == "FAIL":
            return AgentResponse(
                content=text_response,
                reasoning=thought,
                done=True,
                isError=True,
                raw=response.raw,
            )

        # Create tool call to glm_computer
        tool_call = MCPToolCall(
            name=self._computer_tool_name,
            arguments=action_args,
        )

        return AgentResponse(
            content=text_response,
            reasoning=thought,
            tool_calls=[tool_call],
            raw=response.raw,
        )

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[dict[str, Any]]:
        """Format tool results into messages for the next turn.

        Stores screenshot in history for context management.
        """
        messages: list[dict[str, Any]] = []

        for tool_call, result in zip(tool_calls, tool_results, strict=True):
            # Extract action string for history
            action_str = ""
            if tool_call.arguments:
                action_name = tool_call.arguments.get("action", "")
                # Reconstruct action string
                args_parts = []
                for k, v in tool_call.arguments.items():
                    if k != "action":
                        args_parts.append(f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}")
                action_str = f"{action_name}({', '.join(args_parts)})"

            # Extract screenshot from result
            screenshot_base64: str | None = None
            for content in result.content:
                if isinstance(content, types.ImageContent):
                    screenshot_base64 = content.data
                    break

            # Store in history
            thought = self._extract_thought(str(self._last_response_content or ""))
            self._history.append((thought, action_str, screenshot_base64))

            # Build message with history and current screenshot
            content_parts = self._build_history_content()

            # Add current screenshot
            if screenshot_base64:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{screenshot_base64}"}
                })

            messages.append({
                "role": "user",
                "content": content_parts,
            })

        return messages

    async def format_initial_messages(self, task: str) -> list[dict[str, Any]]:
        """Format initial messages with task prompt."""
        # Reset history for new task
        self._history = []
        self._memory = "[]"

        # Build initial message
        content: list[dict[str, Any]] = [
            {"type": "text", "text": f"# Task:\n{task}\n\n{self._build_history_content()[0]['text']}"},
        ]

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]

    # Track last response content for history
    _last_response_content: str | None = None

    async def step(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[AgentResponse, list[dict[str, Any]]]:
        """Execute one agent step."""
        response = await self.get_response(messages)
        self._last_response_content = response.content
        return response, messages
