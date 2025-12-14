from __future__ import annotations

from .base import MCPAgent
from .openai import OpenAIAgent
from .openai_chat import OpenAIChatAgent
from .operator import OperatorAgent

# Note: These agents are not exported here to avoid requiring optional dependencies.
# Import directly if needed:
#   from hud.agents.claude import ClaudeAgent  # requires anthropic
#   from hud.agents.gemini import GeminiAgent  # requires google-genai
#   from hud.agents.gemini_cua import GeminiCUAAgent  # requires google-genai

__all__ = [
    "MCPAgent",
    "OpenAIAgent",
    "OpenAIChatAgent",
    "OperatorAgent",
]
