"""Compatibility shims for old computer tool names."""

from __future__ import annotations

from hud.tools._legacy.computer.anthropic import AnthropicComputerTool
from hud.tools._legacy.computer.gemini import GeminiComputerTool
from hud.tools._legacy.computer.glm import GLMComputerTool
from hud.tools._legacy.computer.hud import HudComputerTool
from hud.tools._legacy.computer.openai import OpenAIComputerTool
from hud.tools._legacy.computer.qwen import QwenComputerTool

__all__ = [
    "AnthropicComputerTool",
    "GLMComputerTool",
    "GeminiComputerTool",
    "HudComputerTool",
    "OpenAIComputerTool",
    "QwenComputerTool",
]
