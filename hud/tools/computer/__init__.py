"""Computer control tools for different agent APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .settings import computer_settings

if TYPE_CHECKING:
    from .anthropic import AnthropicComputerTool
    from .gemini import GeminiComputerTool
    from .hud import HudComputerTool
    from .openai import OpenAIComputerTool
    from .qwen import QwenComputerTool

__all__ = [
    "AnthropicComputerTool",
    "GeminiComputerTool",
    "HudComputerTool",
    "OpenAIComputerTool",
    "QwenComputerTool",
    "computer_settings",
]


def __getattr__(name: str):
    """Lazy import computer tools."""
    if name == "AnthropicComputerTool":
        from .anthropic import AnthropicComputerTool

        return AnthropicComputerTool
    elif name == "GeminiComputerTool":
        from .gemini import GeminiComputerTool

        return GeminiComputerTool
    elif name == "HudComputerTool":
        from .hud import HudComputerTool

        return HudComputerTool
    elif name == "OpenAIComputerTool":
        from .openai import OpenAIComputerTool

        return OpenAIComputerTool
    elif name == "QwenComputerTool":
        from .qwen import QwenComputerTool

        return QwenComputerTool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
