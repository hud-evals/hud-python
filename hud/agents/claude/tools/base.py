"""Common agent-side Claude tool support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from hud.agents import tools as _agent_tools
from hud.agents.tools import AgentTool, AgentToolSpec, CallTool

if TYPE_CHECKING:
    from anthropic.types.beta import BetaToolUnionParam

    from hud.types import MCPToolResult
else:
    BetaToolUnionParam = Any

ClaudeToolSpec = AgentToolSpec
call_tool = _agent_tools.call_tool


class ClaudeTool(AgentTool["BetaToolUnionParam"], ABC):
    """Agent-side Claude provider tool backed by an environment tool."""

    @abstractmethod
    async def execute(self, caller: CallTool, arguments: dict[str, Any]) -> MCPToolResult:
        """Execute against the environment tool using the agent-provided caller."""
        ...
