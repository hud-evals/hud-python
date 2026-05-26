"""Agent-owned Gemini native tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from google.genai import types as genai_types

from hud.agents.tools import AgentTool, AgentTools
from hud.types import MCPToolCall

from .base import GeminiFunctionTool
from .coding import GeminiEditTool, GeminiShellTool, GeminiWriteTool
from .computer import (
    PREDEFINED_COMPUTER_USE_FUNCTIONS,
    GeminiComputerTool,
)
from .filesystem import (
    GeminiGlobTool,
    GeminiListTool,
    GeminiReadTool,
    GeminiSearchTool,
)
from .hosted import (
    GeminiCodeExecutionTool,
    GeminiGoogleSearchTool,
    GeminiHostedTool,
    GeminiUrlContextTool,
)
from .memory import GeminiMemoryTool

if TYPE_CHECKING:
    from collections.abc import Mapping

    import mcp.types as types

    from hud.agents.tools import ToolMetadata


class GeminiAgentTools(AgentTools[AgentTool[genai_types.Tool], genai_types.Tool]):
    """Prepared Gemini tool state for a run."""

    native_tool_classes: ClassVar[tuple[type[AgentTool[object]], ...]] = (
        GeminiComputerTool,
        GeminiShellTool,
        GeminiEditTool,
        GeminiWriteTool,
        GeminiReadTool,
        GeminiSearchTool,
        GeminiGlobTool,
        GeminiListTool,
        GeminiMemoryTool,
    )
    function_tool_class = GeminiFunctionTool
    name_fallbacks: ClassVar[Mapping[str, tuple[str, ...]]] = {
        "computer": ("computer", "gemini_computer", "computer_gemini"),
        "shell": ("bash",),
        "editor": ("edit",),
        "filesystem": ("read", "grep", "glob", "list"),
        "memory": ("memory",),
    }

    def __init__(self, *, excluded_predefined_functions: list[str] | None = None) -> None:
        super().__init__()
        self.excluded_predefined_functions = list(excluded_predefined_functions or [])

    @property
    def computer_tool_name(self) -> str | None:
        return "computer_use" if "computer_use" in self else None

    @property
    def predefined_computer_functions(self) -> frozenset[str]:
        return frozenset(PREDEFINED_COMPUTER_USE_FUNCTIONS)

    def tool_call(self, function_call: genai_types.FunctionCall) -> MCPToolCall:
        name = function_call.name or ""
        arguments = dict(function_call.args) if function_call.args else {}

        if mcp_tool_name := self.name_map.get(name):
            return MCPToolCall(name=mcp_tool_name, arguments=arguments)

        if self.computer_tool_name and name in self.predefined_computer_functions:
            computer_tool = self.get(self.computer_tool_name)
            if isinstance(computer_tool, GeminiComputerTool):
                return computer_tool.tool_call(name, arguments)

        return MCPToolCall(name=name, arguments=arguments)

    def select_tools(
        self,
        tools: list[types.Tool],
        model: str,
        *,
        tool_metadata: ToolMetadata | None = None,
        excluded_predefined_functions: list[str] | None = None,
    ) -> tuple[list[AgentTool[genai_types.Tool]], list[types.Tool]]:
        provider_tools, user_tools = super().select_tools(
            tools,
            model,
            tool_metadata=tool_metadata,
        )
        user_tool_names = {tool.name for tool in user_tools}
        configured_exclusions = (
            excluded_predefined_functions
            if excluded_predefined_functions is not None
            else self.excluded_predefined_functions
        )
        colliding_exclusions = sorted(self.predefined_computer_functions & user_tool_names)
        exclusions = sorted({*configured_exclusions, *colliding_exclusions})
        if not exclusions:
            return provider_tools, user_tools
        return (
            [
                tool.with_excluded_predefined_functions(exclusions)
                if isinstance(tool, GeminiComputerTool)
                else tool
                for tool in provider_tools
            ],
            user_tools,
        )


__all__ = [
    "GeminiAgentTools",
    "GeminiCodeExecutionTool",
    "GeminiGoogleSearchTool",
    "GeminiHostedTool",
    "GeminiUrlContextTool",
]
