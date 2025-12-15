"""Tool wrapper to convert tau2 tools into MCP tools."""

import json
import uuid
from typing import Any, Dict
from tau2.environment.tool import Tool
from tau2.environment.toolkit import ToolKitBase
from tau2.environment.environment import Environment
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
from hud.tools.base import BaseTool
from hud.tools.types import TextContent
from server.setup.load import get_tau2_task


class Tau2ToolWrapper(BaseTool):
    """Wrapper that converts a tau2 Tool into a BaseTool."""

    def __init__(self, tau2_tool: Tool, toolkit: ToolKitBase):
        """
        Create a wrapper for a tau2 tool.

        Args:
            tau2_tool: The tau2 Tool to wrap
            toolkit: The toolkit instance that owns this tool
        """
        self.tau2_tool = tau2_tool
        self.toolkit = toolkit

        super().__init__(
            env=toolkit,
            name=tau2_tool.name,
            description=str(tau2_tool)
        )

        # Set the signature from tau2_tool's function signature
        # This allows MCP to properly introspect the parameters
        if hasattr(tau2_tool, '__signature__'):
            self.__signature__ = tau2_tool.__signature__

    async def __call__(self, **kwargs: Any) -> list[TextContent]:
        """Execute the tau2 tool."""
        try:
            # Parse JSON string arguments back into Python objects
            # This handles cases where lists/dicts are passed as JSON strings
            parsed_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, str):
                    # Check if the string looks like JSON (starts with [ or {)
                    stripped = value.strip()
                    if stripped.startswith('[') or stripped.startswith('{'):
                        try:
                            parsed_kwargs[key] = json.loads(value)
                        except (json.JSONDecodeError, ValueError):
                            # If parsing fails, use the original string
                            parsed_kwargs[key] = value
                    else:
                        parsed_kwargs[key] = value
                else:
                    parsed_kwargs[key] = value

            # Record tool call in message history (for evaluation)
            tau2_task = get_tau2_task()
            tool_call = None
            if tau2_task.is_initialized():
                # Create tool call record with unique ID
                tool_call = ToolCall(
                    id=str(uuid.uuid4()),
                    name=self.tau2_tool.name,
                    arguments=parsed_kwargs,
                    requestor="assistant"
                )

                # Create assistant message with the tool call
                agent_message = AssistantMessage(
                    role="assistant",
                    tool_calls=[tool_call],
                    cost=0.0
                )
                tau2_task.add_message(agent_message)

            # Execute the tool
            active_toolkit = None
            if tau2_task.is_initialized() and tau2_task.environment is not None:
                active_toolkit = tau2_task.environment.tools

            if active_toolkit is None:
                return [TextContent(
                    type="text",
                    text="Error: Environment not initialized. Call setup/load first."
                )]

            # Match upstream Environment.get_response semantics:
            # - Wrap tool execution in try/except
            # - Call sync_tools() on success
            # - Serialize via Environment.to_json_str
            # - Return ToolMessage.error flag for orchestrator-style error counting
            error = False
            try:
                resp = active_toolkit.use_tool(self.tau2_tool.name, **parsed_kwargs)
                tau2_task.environment.sync_tools()
            except Exception as e:
                resp = f"Error: {e}"
                error = True

            resp_str = Environment.to_json_str(resp)

            # Record tool result in message history
            if tau2_task.is_initialized() and tool_call is not None:
                tool_message = ToolMessage(
                    id=tool_call.id,
                    role="tool",
                    content=resp_str,
                    requestor=tool_call.requestor,
                    error=error,
                )
                tau2_task.add_message(tool_message)

            return [TextContent(type="text", text=resp_str)]
        except Exception as e:
            # If our wrapper itself fails, follow upstream error formatting.
            return [TextContent(type="text", text=f"Error: {str(e)}")]


def wrap_tool(tau2_tool: Tool, toolkit: ToolKitBase) -> BaseTool:
    """
    Convert a tau2 Tool into a BaseTool instance.

    Args:
        tau2_tool: The tau2 Tool object to wrap
        toolkit: The ToolKitBase instance that owns this tool

    Returns:
        A BaseTool instance compatible with MCP server registration
    """
    return Tau2ToolWrapper(tau2_tool, toolkit)


def wrap_all_tools(toolkit: ToolKitBase) -> Dict[str, BaseTool]:
    """
    Convert all tools in a tau2 ToolKitBase into MCP-compatible BaseTool instances.

    Args:
        toolkit: The ToolKitBase instance with tools to wrap

    Returns:
        Dictionary mapping tool names to BaseTool instances
    """
    tau2_tools = toolkit.get_tools()
    wrapped_tools = {}

    for tool_name, tau2_tool in tau2_tools.items():
        wrapped_tools[tool_name] = wrap_tool(tau2_tool, toolkit)

    return wrapped_tools


def get_tool_info(toolkit: ToolKitBase) -> Dict[str, Any]:
    """
    Get information about all tools in a toolkit.

    Args:
        toolkit: The ToolKitBase instance

    Returns:
        Dictionary with tool statistics and descriptions
    """
    tau2_tools = toolkit.get_tools()
    stats = toolkit.get_statistics()

    return {
        "num_tools": stats["num_tools"],
        "num_read_tools": stats["num_read_tools"],
        "num_write_tools": stats["num_write_tools"],
        "num_think_tools": stats["num_think_tools"],
        "num_generic_tools": stats["num_generic_tools"],
        "tools": [
            {
                "name": name,
                "description": str(tool),
                "type": toolkit.tool_type(name).value,
            }
            for name, tool in tau2_tools.items()
        ]
    }
