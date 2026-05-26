"""Agent-owned OpenAI-compatible tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from openai.types.chat import ChatCompletionMessageParam

from hud.agents.tools import AgentTool, AgentTools

from .base import (
    OpenAICompatibleFunctionTool,
    OpenAICompatibleToolParam,
)
from .filesystem import (
    GlobTool,
    GrepTool,
    ListTool,
    ReadTool,
)
from .glm_computer import GLMComputerTool
from .qwen_computer import QwenComputerTool

if TYPE_CHECKING:
    from collections.abc import Mapping


class OpenAICompatibleAgentTools(
    AgentTools[
        AgentTool[OpenAICompatibleToolParam, ChatCompletionMessageParam],
        OpenAICompatibleToolParam,
        ChatCompletionMessageParam,
    ]
):
    """Prepared OpenAI-compatible chat tool state for a run."""

    native_tool_classes: ClassVar[tuple[type[AgentTool[object, object]], ...]] = (
        GLMComputerTool,
        QwenComputerTool,
        ReadTool,
        GrepTool,
        GlobTool,
        ListTool,
    )
    function_tool_class = OpenAICompatibleFunctionTool
    name_fallbacks: ClassVar[Mapping[str, tuple[str, ...]]] = {
        "computer": (
            "computer",
            "hud_computer",
            "openai_computer",
            "glm_computer",
            "qwen_computer",
        ),
        "filesystem": ("read", "grep", "glob", "list"),
    }


__all__ = [
    "OpenAICompatibleAgentTools",
]
