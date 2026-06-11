"""Gemini wrapper for upstream MCP tools."""

from __future__ import annotations

from google.genai import types as genai_types

from hud.agents.tools import MCPTool

from .base import GeminiToolSpec


class GeminiMCPProxyTool(MCPTool):
    """Expose one discovered MCP tool as a Gemini FunctionDeclaration."""

    @classmethod
    def default_spec(cls, model: str) -> GeminiToolSpec:
        del model
        return GeminiToolSpec(api_type="function", api_name="function")

    def to_params(self) -> genai_types.Tool:
        if self.mcp_tool.description is None:
            raise ValueError(f"MCP tool {self.mcp_tool.name!r} requires a description.")
        return genai_types.Tool(
            function_declarations=[
                genai_types.FunctionDeclaration(
                    name=self.provider_name,
                    description=self.mcp_tool.description,
                    parameters_json_schema=self.mcp_tool.inputSchema,
                ),
            ],
        )


__all__ = ["GeminiMCPProxyTool"]
