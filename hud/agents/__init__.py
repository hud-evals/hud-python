from __future__ import annotations

from .base import MCPAgent
from .claude import ClaudeAgent
from .openrouter_agent import OpenRouterAgent
from .openai import OperatorAgent
from .openai_chat_generic import GenericOpenAIChatAgent
from .litellm_agent import LiteLLMAgent

# Backwards compatibility
LiteLLMClaudeAgent = LiteLLMAgent

__all__ = [
    "ClaudeAgent",
    "GenericOpenAIChatAgent",
    "LiteLLMAgent",
    "LiteLLMClaudeAgent",
    "MCPAgent",
    "OperatorAgent",
    "OpenRouterAgent",
]
