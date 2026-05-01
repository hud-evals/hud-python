"""Agent-owned Claude native tools."""

from __future__ import annotations

from dataclasses import dataclass, field

from hud.agents.tools import AgentToolRegistry

from .base import ClaudeTool
from .coding import ClaudeBashTool, ClaudeTextEditorTool
from .computer import ClaudeComputerTool
from .hosted import ClaudeHostedTool, ClaudeToolSearchTool, ClaudeWebFetchTool, ClaudeWebSearchTool
from .memory import ClaudeMemoryTool


@dataclass(frozen=True)
class ClaudeToolRegistry(AgentToolRegistry[ClaudeTool]):
    """Registry for Claude harness tools."""

    tool_classes: tuple[type[ClaudeTool], ...] = (
        ClaudeComputerTool,
        ClaudeBashTool,
        ClaudeTextEditorTool,
        ClaudeMemoryTool,
    )
    name_fallbacks: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "computer": ("computer", "anthropic_computer", "computer_anthropic"),
            "shell": ("bash",),
            "editor": ("edit", "str_replace_based_edit_tool", "text_editor"),
            "memory": ("memory",),
        }
    )

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset(cls.capability for cls in self.tool_classes)

    @property
    def provider_tool_names(self) -> frozenset[str]:
        return frozenset(cls.name for cls in self.tool_classes)


claude_tools = ClaudeToolRegistry()


__all__ = [
    "ClaudeBashTool",
    "ClaudeComputerTool",
    "ClaudeHostedTool",
    "ClaudeMemoryTool",
    "ClaudeTextEditorTool",
    "ClaudeTool",
    "ClaudeToolRegistry",
    "ClaudeToolSearchTool",
    "ClaudeWebFetchTool",
    "ClaudeWebSearchTool",
    "claude_tools",
]
