"""Agent-owned Gemini native tools."""

from __future__ import annotations

from dataclasses import dataclass, field

from hud.agents.tools import AgentToolRegistry

from .base import GeminiTool
from .coding import (
    GEMINI_EDIT_SPEC,
    GEMINI_SHELL_SPEC,
    GEMINI_WRITE_SPEC,
    GeminiEditTool,
    GeminiShellTool,
    GeminiWriteTool,
)
from .computer import (
    GEMINI_COMPUTER_SPEC,
    PREDEFINED_COMPUTER_USE_FUNCTIONS,
    GeminiComputerTool,
    normalize_gemini_computer_use_args,
)
from .filesystem import (
    GEMINI_GLOB_SPEC,
    GEMINI_LIST_SPEC,
    GEMINI_READ_SPEC,
    GEMINI_SEARCH_SPEC,
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
from .memory import GEMINI_MEMORY_SPEC, GeminiMemoryTool


@dataclass(frozen=True)
class GeminiToolRegistry(AgentToolRegistry[GeminiTool]):
    """Registry for Gemini harness tools."""

    tool_classes: tuple[type[GeminiTool], ...] = (
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
    name_fallbacks: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "computer": ("computer", "gemini_computer", "computer_gemini"),
            "shell": ("bash",),
            "editor": ("edit",),
            "filesystem": ("read", "grep", "glob", "list"),
            "memory": ("memory",),
        }
    )

    @property
    def api_types(self) -> frozenset[str]:
        return frozenset(cls.name for cls in self.tool_classes)

    @property
    def predefined_computer_functions(self) -> frozenset[str]:
        return frozenset(PREDEFINED_COMPUTER_USE_FUNCTIONS)


gemini_tools = GeminiToolRegistry()

__all__ = [
    "GEMINI_COMPUTER_SPEC",
    "GEMINI_EDIT_SPEC",
    "GEMINI_GLOB_SPEC",
    "GEMINI_LIST_SPEC",
    "GEMINI_MEMORY_SPEC",
    "GEMINI_READ_SPEC",
    "GEMINI_SEARCH_SPEC",
    "GEMINI_SHELL_SPEC",
    "GEMINI_WRITE_SPEC",
    "GeminiCodeExecutionTool",
    "GeminiComputerTool",
    "GeminiEditTool",
    "GeminiGlobTool",
    "GeminiGoogleSearchTool",
    "GeminiHostedTool",
    "GeminiListTool",
    "GeminiMemoryTool",
    "GeminiReadTool",
    "GeminiSearchTool",
    "GeminiShellTool",
    "GeminiTool",
    "GeminiToolRegistry",
    "GeminiUrlContextTool",
    "GeminiWriteTool",
    "gemini_tools",
    "normalize_gemini_computer_use_args",
]
