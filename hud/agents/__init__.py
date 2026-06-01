from __future__ import annotations

from hud._runtime import activate_runtime

from .base import MCPAgent
from .gateway import create_agent
from .openai import OpenAIAgent
from .openai_compatible import OpenAIChatAgent

# Agents drive the MCP runtime, which needs HUD's compatibility patches.
activate_runtime()

__all__ = [
    "MCPAgent",
    "OpenAIAgent",
    "OpenAIChatAgent",
    "create_agent",
]
