"""Agent implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from hud.settings import settings
from hud.types import AgentType
from hud.utils.exceptions import HudAuthenticationError
from hud.utils.gateway import resolve_gateway_model

if TYPE_CHECKING:
    from typing import TypeAlias

    from hud.agents.claude.agent import ClaudeAgent
    from hud.agents.claude.sdk.agent import ClaudeCLIAgent
    from hud.agents.codex.agent import CodexCLIAgent
    from hud.agents.gemini.agent import GeminiAgent
    from hud.agents.openai.agent import OpenAIAgent
    from hud.agents.openai_compatible.agent import OpenAIChatAgent

    GatewayAgent: TypeAlias = (
        ClaudeAgent | ClaudeCLIAgent | CodexCLIAgent | GeminiAgent | OpenAIAgent | OpenAIChatAgent
    )


def create_agent(model: str, **kwargs: Any) -> GatewayAgent:
    """Create an agent routed through the HUD gateway.

    Sets ``gateway=True`` on the config instead of attaching a client, so the
    provider agent builds the gateway client locally and
    :class:`~hud.runtime.HostedRuntime` can serialize the config and rebuild
    it remotely. Explicitly supplied clients remain custom/BYOK and are not
    serializable.

    For direct API access with provider API keys, instantiate the agent classes
    directly.
    """
    direct_credentials = [name for name in ("api_key", "base_url") if name in kwargs]
    if direct_credentials:
        names = ", ".join(direct_credentials)
        raise ValueError(
            f"create_agent routes through the HUD gateway and does not accept {names}; "
            "instantiate the provider agent directly for custom/BYOK credentials"
        )
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
    "ClaudeAgent": "hud.agents.claude.agent",
    "ClaudeCLIAgent": "hud.agents.claude.sdk.agent",
    "CodexCLIAgent": "hud.agents.codex.agent",
    "GeminiAgent": "hud.agents.gemini.agent",
    "OpenAIAgent": "hud.agents.openai.agent",
    "OpenAIChatAgent": "hud.agents.openai_compatible.agent",
}

__all__ = [
    "ClaudeAgent",
    "ClaudeCLIAgent",
    "CodexCLIAgent",
    "GeminiAgent",
    "OpenAIAgent",
    "OpenAIChatAgent",
    "create_agent",
]


def __getattr__(name: str) -> object:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'hud.agents' has no attribute {name!r}")

    from importlib import import_module

    try:
        value = getattr(import_module(module_name), name)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"{name} requires the agents extra. Install with: pip install 'hud[agents]'"
        ) from exc
    globals()[name] = value
    return value
