"""Agent implementations.

The robot policy harness lives in :mod:`hud.agents.robot` (requires the ``robot`` extra).
"""

from __future__ import annotations

from .claude import ClaudeAgent, ClaudeSDKAgent, ClaudeSDKConfig
from .gateway import create_agent
from .gemini import GeminiAgent
from .openai import OpenAIAgent
from .openai_compatible import OpenAIChatAgent

__all__ = [
    "ClaudeAgent",
    "ClaudeSDKAgent",
    "ClaudeSDKConfig",
    "GeminiAgent",
    "OpenAIAgent",
    "OpenAIChatAgent",
    "create_agent",
]
