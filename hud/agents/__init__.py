from __future__ import annotations

from typing import Any

from .base import MCPAgent
from .openai import OpenAIAgent
from .openai_chat import OpenAIChatAgent
from .operator import OperatorAgent

__all__ = [
    "MCPAgent",
    "OpenAIAgent",
    "OpenAIChatAgent",
    "OperatorAgent",
    "create_agent",
]


def create_agent(model: str, **kwargs: Any) -> MCPAgent:
    """Create an agent for a gateway model.

    This routes ALL requests through the HUD gateway. For direct API access
    (using your own API keys), use the agent classes directly.

    Args:
        model: Model name (e.g., "gpt-4o", "claude-sonnet-4-5").
        **kwargs: Additional params passed to agent.create().

    Returns:
        Configured MCPAgent instance with gateway routing.

    Example:
        ```python
        # Gateway routing (recommended)
        agent = create_agent("gpt-4o")
        agent = create_agent("claude-sonnet-4-5", temperature=0.7)

        # Direct API access (use agent classes)
        from hud.agents.claude import ClaudeAgent

        agent = ClaudeAgent.create(model="claude-sonnet-4-5")
        ```
    """
    from hud.agents.gateway import build_gateway_client
    from hud.agents.resolver import resolve_cls

    # Resolve class and gateway info
    agent_cls, gateway_info = resolve_cls(model)

    # Get model ID from gateway info or use input
    model_id = model
    if gateway_info:
        model_id = gateway_info.get("model") or gateway_info.get("id") or model

    # Build gateway client
    provider = gateway_info.get("provider", "openai") if gateway_info else "openai"
    client = build_gateway_client(provider)

    # Set up kwargs
    kwargs.setdefault("model", model_id)

    # Use correct client key based on agent type
    if agent_cls == OpenAIChatAgent:
        kwargs.setdefault("openai_client", client)
    else:
        # Claude and other agents use model_client and validate_api_key
        kwargs.setdefault("model_client", client)
        kwargs.setdefault("validate_api_key", False)

    return agent_cls.create(**kwargs)
