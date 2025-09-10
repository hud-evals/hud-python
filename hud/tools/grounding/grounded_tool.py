"""Grounded computer tool that resolves element descriptions to coordinates."""

from __future__ import annotations

import json
import logging

from mcp import ErrorData, McpError
from mcp.types import INVALID_PARAMS, ContentBlock, TextContent

from hud.tools.base import BaseTool

from .grounder import Grounder

logger = logging.getLogger(__name__)


class GroundedComputerTool(BaseTool):
    """Computer tool that grounds element descriptions to coordinates without executing actions.
    
    This tool:
    - Accepts CUA-compatible parameters (action, element_description, etc.)
    - Uses a grounding model to resolve descriptions to XY coordinates
    - Returns a JSON payload describing the resolved action
    - Does NOT execute any actual computer actions (no side effects)
    
    The returned payload can be used by an agent to construct actual computer tool calls.
    """
    
    def __init__(
        self,
        *,
        grounder: Grounder,
        name: str = "computer",
        title: str = "Grounded Computer",
        description: str = "CUA-compatible grounded computer tool that resolves descriptions to coordinates"
    ) -> None:
        """Initialize the grounded computer tool.
        
        Args:
            grounder: GrounderLiteLLM instance for visual grounding
            name: Tool name for registration
            title: Human-readable title
            description: Tool description
        """
        super().__init__(
            env=None,
            name=name,
            title=title,
            description=description
        )
        self._grounder = grounder
    
    async def __call__(
        self,
        action: str,
        screenshot_b64: str,  # Required screenshot parameter
        element_description: str | None = None,
        start_element_description: str | None = None,
        end_element_description: str | None = None,
        text: str | None = None,
        keys: list[str] | None = None,
        button: str | None = None,
        scroll_x: int | None = None,
        scroll_y: int | None = None,
    ) -> list[ContentBlock]:
        """Process a CUA-compatible computer action and return resolved coordinates.
        
        Args:
            action: The action to perform (click, double_click, move, scroll, drag, type, keypress, wait, screenshot)
            screenshot_b64: Base64-encoded screenshot for grounding
            element_description: Description of element for click/move/scroll actions
            start_element_description: Start element for drag actions
            end_element_description: End element for drag actions
            text: Text to type for type actions
            keys: Keys to press for keypress actions
            button: Mouse button (left, right, middle)
            scroll_x: Horizontal scroll amount
            scroll_y: Vertical scroll amount
            
        Returns:
            List containing a TextContent block with JSON payload
        """
        try:
            # Validate screenshot is provided
            if not screenshot_b64:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "No screenshot provided"})
                )]
            
            payload: dict[str, any] = {}
            
            # Handle actions that require grounding
            if action in ("click", "double_click", "move", "scroll"):
                if not element_description:
                    raise McpError(ErrorData(
                        code=INVALID_PARAMS,
                        message=f"element_description is required for {action} action"
                    ))
                
                # Ground the element description to coordinates
                coords = await self._grounder.predict_click(
                    image_b64=screenshot_b64,
                    instruction=element_description
                )
                
                if not coords:
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "error": f"Could not locate element: '{element_description}'. Try a more specific description or different identifying features (color, position, text, etc.)"
                        })
                    )]
                
                x, y = coords
                payload = {
                    "action": action,
                    "x": x,
                    "y": y,
                }
                
                # Add optional parameters
                if button:
                    payload["button"] = button
                if scroll_x is not None:
                    payload["scroll_x"] = scroll_x
                if scroll_y is not None:
                    payload["scroll_y"] = scroll_y
            
            elif action == "drag":
                if not start_element_description or not end_element_description:
                    raise McpError(ErrorData(
                        code=INVALID_PARAMS,
                        message="start_element_description and end_element_description are required for drag action"
                    ))
                
                # Ground both start and end points
                start_coords = await self._grounder.predict_click(
                    image_b64=screenshot_b64,
                    instruction=start_element_description
                )
                
                if not start_coords:
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "error": f"Could not locate start element: '{start_element_description}'. Try a more specific description or different identifying features."
                        })
                    )]
                
                end_coords = await self._grounder.predict_click(
                    image_b64=screenshot_b64,
                    instruction=end_element_description
                )
                
                if not end_coords:
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "error": f"Could not locate end element: '{end_element_description}'. Try a more specific description or different identifying features."
                        })
                    )]
                
                payload = {
                    "action": "drag",
                    "path": [
                        {"x": start_coords[0], "y": start_coords[1]},
                        {"x": end_coords[0], "y": end_coords[1]}
                    ]
                }
            
            elif action == "type":
                if text is None:
                    raise McpError(ErrorData(
                        code=INVALID_PARAMS,
                        message="text is required for type action"
                    ))
                
                payload = {
                    "action": "type",
                    "text": text
                }
            
            elif action == "keypress":
                if not keys:
                    raise McpError(ErrorData(
                        code=INVALID_PARAMS,
                        message="keys are required for keypress action"
                    ))
                
                payload = {
                    "action": "keypress",
                    "keys": keys
                }
            
            elif action == "wait":
                # Default wait time of 500ms if not specified
                payload = {
                    "action": "wait",
                    "ms": 500
                }
            
            elif action == "screenshot":
                payload = {
                    "action": "screenshot"
                }
            
            elif action in ("get_current_url", "get_dimensions", "get_environment"):
                # Pass through informational actions
                payload = {
                    "action": action
                }
            
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unsupported action: {action}"})
                )]
            
            # Return the resolved payload as JSON
            return [TextContent(
                type="text",
                text=json.dumps(payload)
            )]
            
        except McpError:
            # Re-raise MCP errors
            raise
        except Exception as e:
            logger.error(f"Grounded tool failed: {e}")
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]