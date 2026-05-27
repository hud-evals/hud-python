"""Agent-owned OpenAI-compatible tools."""

from __future__ import annotations

from typing import ClassVar

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


__all__ = [
    "OpenAICompatibleAgentTools",
]
