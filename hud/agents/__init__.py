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
    """Create an agent from a type string or AgentType enum.

    This is the recommended factory for creating agents programmatically.
    The agent type maps to a specific agent class via AgentType.cls.

    Args:
        agent_type: Agent type ("claude", "openai", "gemini", etc.) or AgentType enum.
        **kwargs: Parameters passed to the agent's create() method.
            Common params: model, max_tokens, temperature, system_prompt.

    Returns:
        Configured MCPAgent instance ready to use with hud.eval().

    Example:
        ```python
        from hud.agents import create_agent

        # Create Claude agent
        agent = create_agent("claude", model="claude-sonnet-4-5")

        # Create OpenAI agent
        agent = create_agent("openai", model="gpt-4o")

        # Use with hud.eval()
        async with hud.eval(task) as ctx:
            await agent.run(ctx)
        ```

    Supported agent types:
        - "claude": ClaudeAgent (Anthropic Claude)
        - "openai": OpenAIAgent (OpenAI with responses API)
        - "operator": OperatorAgent (OpenAI Computer Use)
        - "gemini": GeminiAgent (Google Gemini)
        - "gemini_cua": GeminiCUAAgent (Gemini Computer Use)
        - "openai_compatible": OpenAIChatAgent (OpenAI-compatible endpoints)
    """
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
