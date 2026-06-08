"""OpenAI-compatible wrapper for upstream MCP tools."""

from __future__ import annotations

from typing import Any

from hud.agents.tools import MCPTool
from hud.agents.tools.base import AgentToolSpec

from .base import openai_compatible_tool_name, openai_compatible_tool_param


class OpenAICompatibleMCPProxyTool(MCPTool):
    """Expose one discovered MCP tool as an OpenAI-compatible function tool."""

    @classmethod
    def default_spec(cls, model: str) -> AgentToolSpec | None:
        del model
        return AgentToolSpec(api_type="function", api_name="function")

    @property
    def provider_name(self) -> str:
        return openai_compatible_tool_name(self.mcp_tool.name)

    def to_params(self) -> Any:
        return openai_compatible_tool_param(self.mcp_tool, name=self.provider_name)


__all__ = ["OpenAICompatibleMCPProxyTool"]
