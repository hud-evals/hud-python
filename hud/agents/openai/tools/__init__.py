"""Agent-owned OpenAI native tools."""

from __future__ import annotations

from dataclasses import dataclass, field

from hud.agents.tools import AgentToolRegistry

from .base import OpenAITool
from .coding import (
    OPENAI_APPLY_PATCH_SPEC,
    OPENAI_SHELL_SPEC,
    OpenAIApplyPatchTool,
    OpenAIShellTool,
)
from .computer import OPENAI_COMPUTER_SPEC, OpenAIComputerTool
from .hosted import OpenAICodeInterpreterTool, OpenAIHostedTool, OpenAIToolSearchTool


@dataclass(frozen=True)
class OpenAIToolRegistry(AgentToolRegistry[OpenAITool]):
    """Registry for OpenAI harness tools."""

    tool_classes: tuple[type[OpenAITool], ...] = (
        OpenAIComputerTool,
        OpenAIShellTool,
        OpenAIApplyPatchTool,
    )
    name_fallbacks: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "computer": ("computer", "openai_computer"),
            "shell": ("bash",),
            "editor": ("edit",),
        }
    )

    @property
    def api_types(self) -> frozenset[str]:
        return frozenset(cls.name for cls in self.tool_classes)

    @property
    def roles(self) -> frozenset[str]:
        return self.capabilities


openai_tools = OpenAIToolRegistry()

__all__ = [
    "OPENAI_APPLY_PATCH_SPEC",
    "OPENAI_COMPUTER_SPEC",
    "OPENAI_SHELL_SPEC",
    "OpenAIApplyPatchTool",
    "OpenAICodeInterpreterTool",
    "OpenAIComputerTool",
    "OpenAIHostedTool",
    "OpenAIShellTool",
    "OpenAITool",
    "OpenAIToolRegistry",
    "OpenAIToolSearchTool",
    "openai_tools",
]
