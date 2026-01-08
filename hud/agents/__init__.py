from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import MCPAgent
from .openai import OpenAIAgent
from .openai_chat import OpenAIChatAgent
from .operator import OperatorAgent
from .resolver import resolve_cls

if TYPE_CHECKING:
    from hud.types import AgentType

__all__ = [
    "MCPAgent",
    "OpenAIAgent",
    "OpenAIChatAgent",
    "OperatorAgent",
    "create_agent",
    "resolve_cls",
]


def create_agent(model: str | AgentType, **kwargs: Any) -> MCPAgent:
    """Create an agent from a model string or AgentType.

    Args:
        model: AgentType ("claude"), or gateway model name ("gpt-4o").
        **kwargs: Params passed to agent.create().

    Example:
        ```python
        agent = create_agent("claude", model="claude-sonnet-4-5")
        agent = create_agent("gpt-4o")  # auto-configures gateway
        ```
    """
    from hud.types import AgentType as AT

    # AgentType enum → just create
    if isinstance(model, AT):
        return model.cls.create(**kwargs)

    # Resolve class and optional gateway info
    agent_cls, gateway_info = resolve_cls(model)

    # If not a gateway model, just create
    if gateway_info is None:
        return agent_cls.create(**kwargs)

    # Build gateway params
    model_id = gateway_info.get("model") or gateway_info.get("id") or model
    kwargs.setdefault("model", model_id)
    kwargs.setdefault("validate_api_key", False)

    # Build model_client based on provider
    if "model_client" not in kwargs and "openai_client" not in kwargs:
        from hud.agents.gateway import build_gateway_client

        provider = gateway_info.get("provider", "openai_compatible")
        client = build_gateway_client(provider)

        # OpenAIChatAgent uses openai_client key, others use model_client
        key = "openai_client" if agent_cls == OpenAIChatAgent else "model_client"
        kwargs[key] = client

    return agent_cls.create(**kwargs)
