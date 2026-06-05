"""OpenAI wrapper for upstream MCP tools."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, cast

from hud.agents.tools import MCPTool
from hud.utils.strict_schema import ensure_strict_json_schema

from .base import OpenAIToolSpec

if TYPE_CHECKING:
    from openai.types.responses import FunctionToolParam, ToolParam

logger = logging.getLogger(__name__)


class OpenAIMCPProxyTool(MCPTool):
    """Expose one discovered MCP tool as an OpenAI function tool."""

    @classmethod
    def default_spec(cls, model: str) -> OpenAIToolSpec | None:
        del model
        return OpenAIToolSpec(api_type="function", api_name="function")

    def to_params(self) -> Any:
        if self.mcp_tool.description is None:
            raise ValueError(f"MCP tool {self.mcp_tool.name!r} requires a description.")
        try:
            parameters = ensure_strict_json_schema(copy.deepcopy(self.mcp_tool.inputSchema))
        except Exception as e:
            logger.warning(
                "Failed to convert tool '%s' schema to strict: %s", self.mcp_tool.name, e
            )
            parameters = self.mcp_tool.inputSchema
        return cast(
            "ToolParam",
            cast(
                "FunctionToolParam",
                {
                    "type": "function",
                    "name": self.provider_name,
                    "description": self.mcp_tool.description,
                    "parameters": parameters,
                    "strict": True,
                },
            ),
        )


__all__ = ["OpenAIMCPProxyTool"]
