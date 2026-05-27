"""Agent-owned OpenAI native tools."""

from __future__ import annotations

from typing import ClassVar

from openai.types.responses import ToolParam
from openai.types.responses.response_input_param import ResponseInputItemParam

from hud.agents.tools import AgentTool, AgentTools

from .base import OpenAIFunctionTool, OpenAITool
from .coding import OpenAIShellTool
from .computer import OpenAIComputerTool
from .hosted import OpenAICodeInterpreterTool, OpenAIHostedTool, OpenAIToolSearchTool


class OpenAIAgentTools(AgentTools[OpenAITool, ToolParam, ResponseInputItemParam]):
    """Prepared OpenAI Responses tool state for a run."""

    native_tool_classes: ClassVar[tuple[type[AgentTool[object, object]], ...]] = (
        OpenAIComputerTool,
        OpenAIShellTool,
    )
    function_tool_class = OpenAIFunctionTool

    @property
    def tool_search_threshold(self) -> int | None:
        for hosted_tool in self.hosted_tools:
            if isinstance(hosted_tool, OpenAIToolSearchTool):
                return hosted_tool.threshold
        return None


__all__ = [
    "OpenAIAgentTools",
    "OpenAICodeInterpreterTool",
    "OpenAIHostedTool",
    "OpenAIToolSearchTool",
]
