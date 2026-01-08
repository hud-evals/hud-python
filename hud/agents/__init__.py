from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import MCPAgent, Rollout
from .openai import OpenAIAgent
from .openai_chat import OpenAIChatAgent
from .operator import OperatorAgent

if TYPE_CHECKING:
    from hud.types import AgentType

# Note: These agents are not exported here to avoid requiring optional dependencies.
# Import directly if needed:
#   from hud.agents.claude import ClaudeAgent  # requires anthropic
#   from hud.agents.gemini import GeminiAgent  # requires google-genai
#   from hud.agents.gemini_cua import GeminiCUAAgent  # requires google-genai


def create_agent(
    agent_type: str | AgentType,
    **kwargs: Any,
) -> MCPAgent:
    """Create an agent from a type string or AgentType enum."""
    from hud.types import AgentType as AT

    # Normalize to AgentType enum
    if isinstance(agent_type, str):
        agent_type_enum = AT(agent_type)
    else:
        agent_type_enum = agent_type

    # Get agent class and create instance
    agent_cls = agent_type_enum.cls
    return agent_cls.create(**kwargs)


__all__ = [
    "MCPAgent",
    "OpenAIAgent",
    "OpenAIChatAgent",
    "OperatorAgent",
    "Rollout",
    "create_agent",
]
