"""Agent implementations."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from hud.agents.base import Agent
from hud.settings import settings
from hud.types import AgentType
from hud.utils.exceptions import HudAuthenticationError
from hud.utils.gateway import resolve_gateway_model

if TYPE_CHECKING:
    from typing import TypeAlias

    from hud.agents.claude.cli.agent import ClaudeCLIAgent
    from hud.agents.codex.agent import CodexCLIAgent
    from hud.agents.pi.agent import PiAgent

    GatewayAgent: TypeAlias = ClaudeCLIAgent | CodexCLIAgent | PiAgent


def create_agent(model: str, **kwargs: Any) -> GatewayAgent:
    """Create an agent routed through the HUD gateway; requires ``HUD_API_KEY``.

    The agent stays serializable, so ``HostedRuntime`` can rebuild it remotely. For your own
    provider keys, instantiate the agent class directly.
    """
    if not settings.api_key:
        raise HudAuthenticationError("HUD_API_KEY is required to create a gateway agent")

    if model in {member.value for member in AgentType}:
        agent_type = AgentType(model)
    else:
        entry = resolve_gateway_model(model)
        agent_type = AgentType(entry.sdk_agent_type)
        kwargs.setdefault("model", entry.model_name)
    kwargs["gateway"] = True
    return cast("GatewayAgent", agent_type.cls.load(kwargs))


_LAZY_EXPORTS = {
    "ClaudeCLIAgent": "hud.agents.claude.cli.agent",
    "CodexCLIAgent": "hud.agents.codex.agent",
    "PiAgent": "hud.agents.pi.agent",
}

__all__ = [
    "Agent",
    "ClaudeCLIAgent",
    "CodexCLIAgent",
    "PiAgent",
    "create_agent",
]


def __getattr__(name: str) -> object:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'hud.agents' has no attribute {name!r}")

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
