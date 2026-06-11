"""Claude wrapper for upstream MCP tools — one Claude function tool per discovered MCP tool."""

from __future__ import annotations

from inspect import cleandoc
from typing import TYPE_CHECKING, cast

from hud.agents.tools import MCPTool

from .base import ClaudeToolSpec

if TYPE_CHECKING:
    from anthropic.types.beta import BetaToolParam, BetaToolUnionParam


class ClaudeMCPProxyTool(MCPTool):
    """Expose one discovered MCP tool as a Claude function tool."""

    @classmethod
    def default_spec(cls, model: str) -> ClaudeToolSpec:
        del model
        return ClaudeToolSpec(api_type="function", api_name="function")

    def to_params(self) -> BetaToolUnionParam:
        if self.mcp_tool.description is None:
            raise ValueError(
                cleandoc(f"""
                MCP tool {self.mcp_tool.name!r} requires a description and inputSchema.
                Add a docstring to your @mcp.tool function and pydantic Field() annotations.
                """),
            )
        return cast(
            "BetaToolParam",
            {
                "name": self.provider_name,
                "description": self.mcp_tool.description,
                "input_schema": self.mcp_tool.inputSchema,
                "eager_input_streaming": True,
            },
        )


__all__ = ["ClaudeMCPProxyTool"]
