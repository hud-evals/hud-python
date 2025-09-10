"""Grounded OpenAI agent that separates visual grounding from reasoning."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import mcp.types as types

from hud import instrument
from hud.tools.grounding import *
from hud.types import AgentResponse, MCPToolCall

from .openai_chat_generic import GenericOpenAIChatAgent

class GroundedOpenAIChatAgent(GenericOpenAIChatAgent):
    """OpenAI agent that uses a separate grounding model for element detection.
    
    This agent:
    - Exposes only a synthetic "computer" tool to the planning model
    - Intercepts tool calls to ground element descriptions to coordinates
    - Converts grounded results to real computer tool calls
    - Maintains screenshot state for grounding operations
    
    The architecture separates concerns:
    - Planning model (GPT-4o etc) focuses on high-level reasoning
    - Grounding model (Qwen2-VL etc) handles visual element detection
    """
    
    def __init__(
        self,
        *,
        grounder_config: GrounderConfig,
        **kwargs: Any
    ) -> None:
        """Initialize the grounded OpenAI agent.
        
        Args:
            grounder_config: Configuration for the grounding model
            openai_client: OpenAI client for the planning model
            model: Name of the OpenAI model to use for planning (e.g., "gpt-4o", "gpt-4o-mini")
            real_computer_tool_name: Name of the actual computer tool to execute
            **kwargs: Additional arguments passed to OperatorAgent
        """
        print("[grounded_openai] INFO: Initializing GroundedOpenAIChatAgent")
        
        # Initialize parent class
        super().__init__(
            **kwargs
        )
        
        # Create grounder and grounded tool
        self.grounder = Grounder(grounder_config)
        self.grounded_tool = GroundedComputerTool(
            grounder=self.grounder,
            name="computer"
        )
        self.last_screenshot_b64: str | None = None  # Store last screenshot for grounding
        
        # Name of the real computer tool to execute
        self.real_computer_tool_name = "computer"

    def get_tool_schemas(self) -> list[dict]:
        """Override to expose only the synthetic grounded tool.

        The planning model only sees the synthetic "computer" tool,
        not the actual computer tool that will be executed.
        
        Returns:
            List containing only the grounded computer tool schema
        """
        return [{
            "type": "function",
            "function": {
                "name": "computer",
                "description": "Control a computer by interacting with UI elements. This tool uses element descriptions to locate and interact with UI elements on the screen (e.g., 'red submit button', 'search text field', 'hamburger menu icon', 'close button in top right corner').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["click", "double_click", "move", "scroll", "drag", "type", "keypress", "wait", "screenshot", "get_current_url", "get_dimensions", "get_environment"],
                            "description": "The action to perform"
                        },
                        "element_description": {
                            "type": "string",
                            "description": "Natural language description of the element for click/move/scroll actions"
                        },
                        "start_element_description": {
                            "type": "string",
                            "description": "Description of the start element for drag actions"
                        },
                        "end_element_description": {
                            "type": "string",
                            "description": "Description of the end element for drag actions"
                        },
                        "text": {
                            "type": "string",
                            "description": "Text to type"
                        },
                        "keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keys to press (e.g., ['ctrl', 'a'] for Ctrl+A)"
                        },
                        "button": {
                            "type": "string",
                            "enum": ["left", "right", "middle"],
                            "description": "Mouse button to use"
                        },
                        "scroll_x": {
                            "type": "integer",
                            "description": "Horizontal scroll amount"
                        },
                        "scroll_y": {
                            "type": "integer",
                            "description": "Vertical scroll amount"
                        }
                    },
                    "required": ["action"]
                }
            }
        }]
    
    @instrument(
        span_type="agent",
        record_args=False,
        record_result=True,
    )
    async def get_response(self, messages: Any) -> AgentResponse:
        """Get response from the planning model and handle grounded tool calls.
        
        This method:
        1. Calls the planning model with only the grounded tool available
        2. Intercepts any tool calls to the grounded tool
        3. Executes the grounded tool locally to resolve coordinates
        4. Maps the grounded result to a real computer tool call
        
        Args:
            messages: Conversation messages
            
        Returns:
            AgentResponse with either content or real tool calls
        """
        # Get tool schemas (only the grounded tool)
        tool_schemas = self.get_tool_schemas()
        
        # Extract any extra kwargs
        protected_keys = {"model", "messages", "tools", "parallel_tool_calls"}
        extra = {k: v for k, v in (self.completion_kwargs or {}).items() if k not in protected_keys}

        print(f"[grounded_openai] DEBUG: get_response called with {len(messages)} messages")
        
        # Take initial screenshot if we don't have one
        if self.last_screenshot_b64 is None:
            print(f"[grounded_openai] DEBUG: Taking initial screenshot for grounding")
            result = await self.mcp_client.call_tool(
                name="computer",
                arguments={"action": "screenshot"},
            )

            # Extract screenshot from structured content
            assert isinstance(result.content[0], types.ImageContent), "Expected ImageContent from screenshot tool"
            self.last_screenshot_b64 = result.content[0].data
            mime_type = result.content[0].mimeType or "image/png"
            print(f"[grounded_openai] DEBUG: Got screenshot, adding to messages")
            messages.append({
                "role": "user", 
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{self.last_screenshot_b64}"}
                }]
            })


        print(f"[grounded_openai] DEBUG: Calling planning model {self.model_name} with grounded tool")
        
        # Call the planning model with only the grounded tool
        response = await self.oai.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tool_schemas,
            parallel_tool_calls=False,
            **extra
        )

        print(f"[grounded_openai] DEBUG: Planning model response received")
        
        choice = response.choices[0]
        msg = choice.message
        
        # Build assistant message for conversation history
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if msg.content:
            assistant_msg["content"] = msg.content
        if msg.tool_calls:
            assistant_msg["tool_calls"] = msg.tool_calls
        
        messages.append(assistant_msg)
        
        # Store conversation history
        self.conversation_history = messages.copy()
        
        # If no tool calls, return content
        if not msg.tool_calls:
            return AgentResponse(
                content=msg.content or "",
                tool_calls=[],
                done=choice.finish_reason in ("stop", "length"),
                raw=response
            )
        
        # Process the grounded tool call
        tc = msg.tool_calls[0]
        
        # Verify it's calling our grounded tool
        if tc.function.name != "computer":
            print(f"[grounded_openai] WARNING: Unexpected tool call: {tc.function.name}")
            return AgentResponse(
                content=f"Error: Model called unexpected tool '{tc.function.name}'",
                tool_calls=[],
                done=True,
                raw=response
            )
        
        # Parse the arguments
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError as e:
            print(f"[grounded_openai] ERROR: Failed to parse tool arguments: {e}")
            return AgentResponse(
                content=f"Error: Invalid tool arguments",
                tool_calls=[],
                done=True,
                raw=response
            )
        
        # Add the last screenshot for grounding from message history
        for m in reversed(messages):
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for block in m["content"]:
                    if (isinstance(block, dict) and block.get("type") == "image_url" and
                            isinstance(block.get("image_url"), dict) and
                            block["image_url"].get("url", "").startswith("data:image/png;base64,")):
                        self.last_screenshot_b64 = block["image_url"]["url"].split("data:image/png;base64,")[1]
                        break

        args["screenshot_b64"] = self.last_screenshot_b64

        # Execute the grounded tool locally (not through MCP)
        print(f"[grounded_openai] DEBUG: Executing grounded tool with action: {args.get('action')}")
        grounded_result = await self.grounded_tool(**args)
        
        # Extract the JSON payload from the result
        payload_text = ""
        for block in grounded_result:
            if isinstance(block, types.TextContent):
                payload_text = block.text
                break
        
        if not payload_text:
            print("[grounded_openai] ERROR: Grounded tool returned no payload")
            return AgentResponse(
                content="Error: Grounding failed - no result",
                tool_calls=[],
                done=True,
                raw=response
            )
        
        # Parse the payload
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as e:
            print(f"[grounded_openai] ERROR: Failed to parse grounded payload: {e}")
            return AgentResponse(
                content=f"Error: Invalid grounding result",
                tool_calls=[],
                done=True,
                raw=response
            )
        
        # Check for errors in the payload
        if "error" in payload:
            print(f"[grounded_openai] WARNING: Grounding error: {payload['error']}")
            # Don't mark as done - let the agent try again with a different approach
            # Add a tool result message to inform the model about the failure
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": f"Error: {payload['error']}. Please try a different element description or approach."
            })
            return AgentResponse(
                content="",
                tool_calls=[],
                done=False,  # Keep going - don't stop the agent
                raw=response
            )
        
        real_args: dict[str, Any] = {}
        
        action = payload.get("action")
        if action:
            real_args["action"] = action
        
        for key in ("x", "y", "button", "scroll_x", "scroll_y", "text", "keys", "ms", "path"):
            if key in payload:
                real_args[key] = payload[key]
        
        # Create the real tool call
        real_tool_call = MCPToolCall(
            name=self.real_computer_tool_name,
            arguments=real_args,
            id=tc.id
        )
        
        print(f"[grounded_openai] INFO: Mapped grounded action '{action}' to real tool call: {real_tool_call}")
        
        # Return the real tool call for execution
        return AgentResponse(
            content=msg.content or "",
            tool_calls=[real_tool_call],
            done=False,
            raw=response
        )