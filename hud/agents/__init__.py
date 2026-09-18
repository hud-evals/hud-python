"""Agent implementations.

The robot policy harness lives in :mod:`hud.agents.robot` (requires the ``robot`` extra).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from hud.agents.registry import dump_agent, load_agent
from hud.settings import settings
from hud.types import AgentType
from hud.utils.exceptions import HudAuthenticationError
from hud.utils.gateway import resolve_gateway_model

if TYPE_CHECKING:
    from typing import TypeAlias

    from hud.agents.claude import ClaudeAgent, ClaudeCLIAgent, ClaudeCLIConfig
    from hud.agents.codex import CodexCLIAgent, CodexCLIConfig
    from hud.agents.gemini import GeminiAgent
    from hud.agents.openai import OpenAIAgent
    from hud.agents.openai_compatible import OpenAIChatAgent
    from hud.agents.tool_agent import ToolAgent as MCPAgent

    GatewayAgent: TypeAlias = ClaudeAgent | GeminiAgent | OpenAIAgent | OpenAIChatAgent


def create_agent(model: str, **kwargs: Any) -> GatewayAgent:
    """Create an agent routed through the HUD gateway.

    Sets ``gateway=True`` on the config instead of attaching a client, so the
    provider agent builds the gateway client locally and
    :class:`~hud.eval.runtime.HostedRuntime` can serialize the config and rebuild
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

    agent_type, model_id = resolve_agent_model(model)
    if agent_type.is_cli:
        raise ValueError(
            f"create_agent only constructs provider API agents; instantiate "
            f"{agent_type.cls.__name__} directly"
        )
    kwargs.setdefault("model", model_id)
    kwargs["gateway"] = True
    config = agent_type.config_cls(**kwargs)
    return cast("GatewayAgent", agent_type.instantiate(config))


def resolve_agent_model(model: str) -> tuple[AgentType, str]:
    """Resolve a catalog id/name/slug or an agent type without constructing a client."""
    agent_type = next((candidate for candidate in AgentType if candidate.value == model), None)
    if agent_type is not None:
        return agent_type, model
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
    return agent_type, entry.model_name or model


_LAZY_EXPORTS = {
    "ClaudeAgent": ("hud.agents.claude", "ClaudeAgent"),
    "ClaudeCLIAgent": ("hud.agents.claude", "ClaudeCLIAgent"),
    "ClaudeCLIConfig": ("hud.agents.claude", "ClaudeCLIConfig"),
    "CodexCLIAgent": ("hud.agents.codex", "CodexCLIAgent"),
    "CodexCLIConfig": ("hud.agents.codex", "CodexCLIConfig"),
    "GeminiAgent": ("hud.agents.gemini", "GeminiAgent"),
    "MCPAgent": ("hud.agents.tool_agent", "ToolAgent"),
    "OpenAIAgent": ("hud.agents.openai", "OpenAIAgent"),
    "OpenAIChatAgent": ("hud.agents.openai_compatible", "OpenAIChatAgent"),
}

__all__ = [
    "ClaudeAgent",
    "ClaudeCLIAgent",
    "ClaudeCLIConfig",
    "CodexCLIAgent",
    "CodexCLIConfig",
    "GeminiAgent",
    "MCPAgent",
    "OpenAIAgent",
    "OpenAIChatAgent",
    "create_agent",
    "dump_agent",
    "load_agent",
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
