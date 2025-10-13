"""HUD tools for computer control, file editing, and bash commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import BaseHub, BaseTool
from .bash import BashTool
from .edit import EditTool
from .playwright import PlaywrightTool
from .response import ResponseTool
from .submit import SubmitTool

if TYPE_CHECKING:
    from .computer import AnthropicComputerTool, HudComputerTool, OpenAIComputerTool, GeminiComputerTool

__all__ = [
    "AnthropicComputerTool",
    "BaseHub",
    "BaseTool",
    "BashTool",
    "EditTool",
    "HudComputerTool",
    "OpenAIComputerTool",
    "GeminiComputerTool",
    "PlaywrightTool",
    "ResponseTool",
    "SubmitTool",
]


def __getattr__(name: str) -> Any:
    """Lazy import computer tools to avoid importing pyautogui unless needed."""
    if name in ("AnthropicComputerTool", "HudComputerTool", "OpenAIComputerTool", "GeminiComputerTool"):
        from . import computer

        return getattr(computer, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
