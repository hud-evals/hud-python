"""Agent-owned OpenAI-compatible tools."""

from __future__ import annotations

from dataclasses import dataclass, field

from hud.agents.tools import AgentTool, AgentToolRegistry

from .computer import (
    GLM_COMPUTER_SPEC,
    QWEN_COMPUTER_SPEC,
    GLMComputerTool,
    QwenComputerTool,
)
from .filesystem import (
    FilesystemTool,
    GlobTool,
    GrepTool,
    ListTool,
    ReadTool,
)
from .types import OpenAICompatibleToolParam


@dataclass(frozen=True)
class OpenAICompatibleToolRegistry(AgentToolRegistry[AgentTool[OpenAICompatibleToolParam]]):
    """Registry for OpenAI-compatible harness tools."""

    tool_classes: tuple[type[AgentTool[OpenAICompatibleToolParam]], ...] = (
        GLMComputerTool,
        QwenComputerTool,
        ReadTool,
        GrepTool,
        GlobTool,
        ListTool,
    )
    name_fallbacks: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "computer": (
                "computer",
                "hud_computer",
                "openai_computer",
                "glm_computer",
                "qwen_computer",
            ),
            "filesystem": ("read", "grep", "glob", "list"),
        }
    )

    @property
    def api_types(self) -> frozenset[str]:
        api_types: set[str] = set()
        for cls in self.tool_classes:
            spec = cls.default_spec("unknown")
            if spec is not None and spec.api_type != "function":
                api_types.add(spec.api_type)
            api_types.update(getattr(cls, "ignored_api_types", frozenset()))
        return frozenset(api_types)


openai_compatible_tools = OpenAICompatibleToolRegistry()

__all__ = [
    "GLM_COMPUTER_SPEC",
    "QWEN_COMPUTER_SPEC",
    "FilesystemTool",
    "GLMComputerTool",
    "GlobTool",
    "GrepTool",
    "ListTool",
    "OpenAICompatibleToolParam",
    "OpenAICompatibleToolRegistry",
    "QwenComputerTool",
    "ReadTool",
    "openai_compatible_tools",
]
