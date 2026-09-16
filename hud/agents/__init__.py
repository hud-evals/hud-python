"""Agent implementations.

The robot policy harness lives in :mod:`hud.agents.robot` (requires the ``robot`` extra).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from hud.settings import settings
from hud.types import AgentType
from hud.utils.exceptions import HudAuthenticationError
from hud.utils.gateway import normalize_gateway_model_id, resolve_gateway_model

if TYPE_CHECKING:
    from typing import TypeAlias

    from hud.agents.claude import ClaudeAgent, ClaudeSDKAgent, ClaudeSDKConfig
    from hud.agents.gemini import GeminiAgent
    from hud.agents.openai import OpenAIAgent
    from hud.agents.openai_compatible import OpenAIChatAgent
    from hud.agents.tool_agent import ToolAgent as MCPAgent

    GatewayAgent: TypeAlias = ClaudeAgent | GeminiAgent | OpenAIAgent | OpenAIChatAgent


def create_agent(model: str, **kwargs: Any) -> GatewayAgent:
    """Create an agent routed through the HUD gateway.

    Leaves ``model_client`` unset so provider agent constructors build the HUD
    gateway client locally, while :class:`~hud.eval.runtime.HostedRuntime` can
    serialize the config and rebuild the client remotely. Explicitly supplied
    clients remain custom/BYOK and are not serializable.

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

    agent_type, model_id = resolve_agent_model(model)
    kwargs.setdefault("model", model_id)
    # cls/config_cls are matched unions; the pairing is correct by construction.
    config = agent_type.config_cls(**kwargs)
    return agent_type.cls(cast("Any", config))


def resolve_agent_model(model: str) -> tuple[AgentType, str]:
    """Resolve a model alias, catalog ID/name, or agent type without constructing a client."""
    normalized = normalize_gateway_model_id(model)
    agent_type = next((candidate for candidate in AgentType if candidate.value == normalized), None)
    if agent_type is not None:
        return agent_type, normalized
    try:
        entry = resolve_gateway_model(model)
    except HudAuthenticationError:
        types = ", ".join(candidate.value for candidate in AgentType)
        raise ValueError(
            f"Resolving {model!r} needs the HUD model catalog; set HUD_API_KEY, or pass an "
            f"agent type ({types}) and choose the model with its config."
        ) from None
    if entry.sdk_agent_type in ("operator", "gemini_cua"):
        replacement = "openai" if entry.sdk_agent_type == "operator" else "gemini"
        raise ValueError(
            f"The {entry.sdk_agent_type} agent is no longer supported; use {replacement} with a "
            "supported computer-use model."
        )
    try:
        agent_type = AgentType(entry.sdk_agent_type)
    except ValueError as exc:
        raise ValueError(f"Model {model!r} has invalid agent type metadata") from exc
    return agent_type, entry.model_name or normalized


_LAZY_EXPORTS = {
    "ClaudeAgent": ("hud.agents.claude", "ClaudeAgent"),
    "ClaudeSDKAgent": ("hud.agents.claude", "ClaudeSDKAgent"),
    "ClaudeSDKConfig": ("hud.agents.claude", "ClaudeSDKConfig"),
    "GeminiAgent": ("hud.agents.gemini", "GeminiAgent"),
    "MCPAgent": ("hud.agents.tool_agent", "ToolAgent"),
    "OpenAIAgent": ("hud.agents.openai", "OpenAIAgent"),
    "OpenAIChatAgent": ("hud.agents.openai_compatible", "OpenAIChatAgent"),
}

__all__ = [
    "ClaudeAgent",
    "ClaudeSDKAgent",
    "ClaudeSDKConfig",
    "GeminiAgent",
    "MCPAgent",
    "OpenAIAgent",
    "OpenAIChatAgent",
    "create_agent",
    "resolve_agent_model",
]


def __getattr__(name: str) -> object:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'hud.agents' has no attribute {name!r}")

    from importlib import import_module

    module_name, symbol = target
    try:
        value = getattr(import_module(module_name), symbol)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"{name} requires the agents extra. Install with: pip install 'hud[agents]'"
        ) from exc
    globals()[name] = value
    return value
