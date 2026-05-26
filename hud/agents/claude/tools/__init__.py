"""Agent-owned Claude native tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from anthropic.types.beta import BetaMessageParam, BetaToolUnionParam

from hud.agents.tools import AgentTools

from .base import ClaudeFunctionTool, ClaudeTool
from .coding import ClaudeBashTool, ClaudeTextEditorTool
from .computer import ClaudeComputerTool
from .hosted import ClaudeHostedTool, ClaudeToolSearchTool, ClaudeWebFetchTool, ClaudeWebSearchTool
from .memory import ClaudeMemoryTool

if TYPE_CHECKING:
    from collections.abc import Mapping

    from hud.agents.tools import AgentTool


class ClaudeAgentTools(AgentTools[ClaudeTool, BetaToolUnionParam, BetaMessageParam]):
    """Prepared Claude tool state for a run."""

    native_tool_classes: ClassVar[tuple[type[AgentTool[object, object]], ...]] = (
        ClaudeComputerTool,
        ClaudeBashTool,
        ClaudeTextEditorTool,
        ClaudeMemoryTool,
    )
    function_tool_class = ClaudeFunctionTool
    name_fallbacks: ClassVar[Mapping[str, tuple[str, ...]]] = {
        "computer": ("computer", "anthropic_computer", "computer_anthropic"),
        "shell": ("bash",),
        "editor": ("edit", "str_replace_based_edit_tool", "text_editor"),
        "memory": ("memory",),
    }

    def __init__(self) -> None:
        super().__init__()
        self.required_betas: set[str] = set()

    def prepare(self, **kwargs: Any) -> None:
        super().prepare(**kwargs)
        self.required_betas = {
            required_beta for tool in self.values() if (required_beta := tool.required_beta)
        }

    @property
    def tool_search_threshold(self) -> int | None:
        for hosted_tool in self.hosted_tools:
            if isinstance(hosted_tool, ClaudeToolSearchTool):
                return hosted_tool.threshold
        return None


__all__ = [
    "ClaudeAgentTools",
    "ClaudeHostedTool",
    "ClaudeToolSearchTool",
    "ClaudeWebFetchTool",
    "ClaudeWebSearchTool",
]
