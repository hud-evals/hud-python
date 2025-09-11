"""Grounded computer tool that resolves element descriptions to coordinates."""

from __future__ import annotations

import logging
from typing import Any

from mcp import ErrorData, McpError
from mcp.types import INVALID_PARAMS, ContentBlock

from hud.clients.base import AgentMCPClient  # noqa: TC001
from hud.tools.grounding.grounder import Grounder  # noqa: TC001
from hud.types import MCPToolCall

logger = logging.getLogger(__name__)


class GroundedComputerTool:
    """Computer tool wrapper that grounds element descriptions to coordinates.

    This tool acts as a local wrapper that:
    1. Accepts natural language element descriptions from the agent
    2. Calls the environment's computer tool via MCP to take screenshots
    3. Uses a grounding model to resolve descriptions to coordinates
    4. Calls the environment's computer tool via MCP with resolved coordinates
    5. Returns the result to the agent

    This allows the agent to use element descriptions while ensuring all
    computer actions happen in the correct environment.
    """

    def __init__(
        self,
        *,
        grounder: Grounder,
        mcp_client: AgentMCPClient,
        computer_tool_name: str = "computer",
    ) -> None:
        """Initialize the grounded computer tool.

        Args:
            grounder: Grounder instance for visual grounding
            mcp_client: MCP client to call the environment's computer tool
            computer_tool_name: Name of the computer tool in the environment
        """
        self._grounder = grounder
        self._mcp_client = mcp_client
        self._computer_tool_name = computer_tool_name

    def get_openai_tool_schema(self) -> dict:
        """Get the OpenAI tool schema for the grounded computer tool.

        Returns:
            Dictionary containing the tool schema in OpenAI format
        """
        return {
            "type": "function",
            "function": {
                "name": "computer",
                "description": (
                    "Control a computer by interacting with UI elements. This tool uses "
                    "element descriptions to locate and interact with UI elements on the "
                    "screen (e.g., 'red submit button', 'search text field', 'hamburger menu "
                    "icon', 'close button in top right corner')."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "click",
                                "double_click",
                                "move",
                                "scroll",
                                "drag",
                                "type",
                                "keypress",
                                "wait",
                                "screenshot",
                                "get_current_url",
                                "get_dimensions",
                                "get_environment",
                            ],
                            "description": "The action to perform",
                        },
                        "element_description": {
                            "type": "string",
                            "description": (
                                "Natural language description of the element for "
                                "click/move/scroll actions"
                            ),
                        },
                        "start_element_description": {
                            "type": "string",
                            "description": "Description of the start element for drag actions",
                        },
                        "end_element_description": {
                            "type": "string",
                            "description": "Description of the end element for drag actions",
                        },
                        "text": {"type": "string", "description": "Text to type"},
                        "keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keys to press (e.g., ['ctrl', 'a'] for Ctrl+A)",
                        },
                        "button": {
                            "type": "string",
                            "enum": ["left", "right", "middle"],
                            "description": "Mouse button to use",
                        },
                        "scroll_x": {"type": "integer", "description": "Horizontal scroll amount"},
                        "scroll_y": {"type": "integer", "description": "Vertical scroll amount"},
                    },
                    "required": ["action"],
                },
            },
        }

    async def __call__(
        self,
        action: str,
        # Screenshot from conversation
        screenshot_b64: str | None = None,
        # Grounding-specific parameters
        element_description: str | None = None,
        start_element_description: str | None = None,
        end_element_description: str | None = None,
        # Pass-through parameters
        text: str | None = None,
        keys: list[str] | None = None,
        button: str | None = None,
        scroll_x: int | None = None,
        scroll_y: int | None = None,
        **kwargs: Any,
    ) -> list[ContentBlock]:
        """Execute a computer action, grounding element descriptions to coordinates first.

        This method calls the environment's computer tool through MCP to ensure
        actions happen in the correct environment.

        Args:
            action: The action to perform
            element_description: Description of element for click/move/scroll actions
            start_element_description: Start element for drag actions
            end_element_description: End element for drag actions
            text: Text to type for type actions
            keys: Keys to press for keypress actions
            button: Mouse button (left, right, middle)
            scroll_x: Horizontal scroll amount
            scroll_y: Vertical scroll amount
            **kwargs: Additional arguments

        Returns:
            List of ContentBlocks with action results from the environment
        """
        try:
            # For actions that don't need grounding, call environment tool directly
            if action in (
                "screenshot",
                "type",
                "keypress",
                "wait",
                "get_current_url",
                "get_dimensions",
                "get_environment",
            ):
                computer_args: dict[str, Any] = {"action": action}
                if text is not None:
                    computer_args["text"] = text
                if keys is not None:
                    computer_args["keys"] = keys

                result = await self._mcp_client.call_tool(
                    MCPToolCall(
                        name=self._computer_tool_name, arguments={**computer_args, **kwargs}
                    )
                )
                return result.content

            # For actions that need coordinates, we need to ground element descriptions
            if action in ("click", "double_click", "move", "scroll"):
                if not element_description:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS,
                            message=f"element_description is required for {action} action",
                        )
                    )

                if not screenshot_b64:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS, message="No screenshot available for grounding"
                        )
                    )

                # Ground the element description to coordinates
                coords = await self._grounder.predict_click(
                    image_b64=screenshot_b64, instruction=element_description
                )

                if not coords:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS,
                            message=(
                                f"Could not locate element: '{element_description}'. "
                                "Try a more specific description or different identifying "
                                "features (color, position, text, etc.)"
                            ),
                        )
                    )

                x, y = coords

                # Execute action with resolved coordinates
                computer_args: dict[str, Any] = {"action": action, "x": x, "y": y}
                if button:
                    computer_args["button"] = button
                if scroll_x is not None:
                    computer_args["scroll_x"] = scroll_x
                if scroll_y is not None:
                    computer_args["scroll_y"] = scroll_y

                result = await self._mcp_client.call_tool(
                    MCPToolCall(
                        name=self._computer_tool_name, arguments={**computer_args, **kwargs}
                    )
                )
                return result.content

            elif action == "drag":
                if not start_element_description or not end_element_description:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS,
                            message=(
                                "start_element_description and end_element_description "
                                "are required for drag action"
                            ),
                        )
                    )

                if not screenshot_b64:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS, message="No screenshot available for grounding"
                        )
                    )

                # Ground both start and end points
                start_coords = await self._grounder.predict_click(
                    image_b64=screenshot_b64, instruction=start_element_description
                )

                if not start_coords:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS,
                            message=(
                                f"Could not locate start element: '{start_element_description}'. "
                                "Try a more specific description or different identifying features."
                            ),
                        )
                    )

                end_coords = await self._grounder.predict_click(
                    image_b64=screenshot_b64, instruction=end_element_description
                )

                if not end_coords:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS,
                            message=(
                                f"Could not locate end element: '{end_element_description}'. "
                                "Try a more specific description or different identifying features."
                            ),
                        )
                    )

                # Execute drag with resolved coordinates
                computer_args: dict[str, Any] = {
                    "action": "drag",
                    "path": [
                        (start_coords[0], start_coords[1]),
                        (end_coords[0], end_coords[1]),
                    ],
                }
                if button:
                    computer_args["button"] = button

                result = await self._mcp_client.call_tool(
                    MCPToolCall(
                        name=self._computer_tool_name, arguments={**computer_args, **kwargs}
                    )
                )
                return result.content

            else:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message=f"Unsupported action: {action}")
                )

        except McpError:
            # Re-raise MCP errors
            raise
        except Exception as e:
            logger.error("Grounded tool failed: %s", e)
            raise McpError(
                ErrorData(code=INVALID_PARAMS, message=f"Grounding failed: {e!s}")
            ) from e
