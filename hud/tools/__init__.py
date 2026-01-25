"""HUD tools for computer control, file editing, and bash commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .agent import AgentTool
from .base import BaseHub, BaseTool
from .bash import BashTool  # Claude-native bash
from .coding import ApplyPatchTool, ShellTool  # OpenAI-native shell/patch
from .edit import EditTool  # Claude-native edit
from .hosted import CodeExecutionTool, GoogleSearchTool, HostedTool, UrlContextTool
from .memory import MemoryTool
from .native_types import NativeToolSpec, NativeToolSpecs
from .playwright import PlaywrightTool
from .response import ResponseTool
from .submit import SubmitTool

if TYPE_CHECKING:
    from .computer import (
        AnthropicComputerTool,
        GeminiComputerTool,
        HudComputerTool,
        OpenAIComputerTool,
        QwenComputerTool,
    )

__all__ = [
    # Base classes
    "AgentTool",
    "BaseHub",
    "BaseTool",
    "HostedTool",
    # Native tool types
    "NativeToolSpec",
    "NativeToolSpecs",
    # Computer tools
    "AnthropicComputerTool",
    "GeminiComputerTool",
    "HudComputerTool",
    "OpenAIComputerTool",
    "QwenComputerTool",
    # Shell/editor tools (Claude style)
    "BashTool",
    "EditTool",
    # Shell/editor tools (OpenAI style)
    "ShellTool",
    "ApplyPatchTool",
    # Hosted tools
    "CodeExecutionTool",
    "GoogleSearchTool",
    "UrlContextTool",
    # Other tools
    "MemoryTool",
    "PlaywrightTool",
    "ResponseTool",
    "SubmitTool",
]


def __getattr__(name: str) -> Any:
    """Lazy import computer tools to avoid importing pyautogui unless needed."""
    if name in (
        "AnthropicComputerTool",
        "HudComputerTool",
        "OpenAIComputerTool",
        "GeminiComputerTool",
        "QwenComputerTool",
    ):
        from . import computer

        return getattr(computer, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
