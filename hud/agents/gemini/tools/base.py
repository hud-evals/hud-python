"""Base Gemini agent-owned tool types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import mcp.types as types
from google.genai import types as genai_types

from hud.agents.tools import AgentTool, AgentToolSpec

if TYPE_CHECKING:
    from hud.types import MCPToolCall, MCPToolResult

GeminiToolSpec = AgentToolSpec


class GeminiTool(AgentTool[genai_types.Tool]):
    """Gemini function declaration backed by an environment tool."""

    description: ClassVar[str]
    parameters: ClassVar[dict[str, Any]]

    def to_params(self) -> genai_types.Tool:
        return genai_types.Tool(
            function_declarations=[
                genai_types.FunctionDeclaration(
                    name=self.provider_name,
                    description=self.description,
                    parameters_json_schema=self.parameters,
                )
            ]
        )

    def format_result(self, call: MCPToolCall, result: MCPToolResult) -> genai_types.Content:
        text = next(
            (content.text for content in result.content if isinstance(content, types.TextContent)),
            None,
        )
        response: dict[str, Any] = (
            {"error": text or "Tool execution failed"} if result.isError else {"success": True}
        )
        if text is not None and not result.isError:
            response["output"] = text
        return genai_types.Content(
            role="user",
            parts=[
                genai_types.Part(
                    function_response=genai_types.FunctionResponse(
                        name=call.provider_name or call.name,
                        response=response,
                    )
                )
            ],
        )


class GeminiFunctionTool(GeminiTool):
    """Regular environment tool exposed as a Gemini function declaration."""

    name = "function"
    capability = "function"

    def __init__(
        self,
        *,
        env_tool_name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        super().__init__(
            env_tool_name=env_tool_name,
            spec=GeminiToolSpec(api_type="function", api_name=env_tool_name),
        )
        self._description = description
        self._parameters = parameters

    @classmethod
    def from_tool(cls, tool: types.Tool) -> GeminiFunctionTool:
        if tool.description is None:
            raise ValueError(f"MCP tool {tool.name} requires a description.")
        return cls(
            env_tool_name=tool.name,
            description=tool.description,
            parameters=tool.inputSchema,
        )

    @property
    def provider_name(self) -> str:
        return self.env_tool_name

    def to_params(self) -> genai_types.Tool:
        return genai_types.Tool(
            function_declarations=[
                genai_types.FunctionDeclaration(
                    name=self.provider_name,
                    description=self._description,
                    parameters_json_schema=self._parameters,
                )
            ]
        )
