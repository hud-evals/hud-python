"""Agent implementations.

The robot policy harness lives in :mod:`hud.agents.robot` (requires the ``robot`` extra).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from hud.agents.registry import dump_agent, load_agent
from hud.settings import settings
from hud.types import AgentType
from hud.utils.exceptions import HudAuthenticationError
from hud.utils.gateway import (
    gateway_model_aliases,
    list_gateway_models,
    normalize_gateway_model_id,
)

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

    requested_model = model
    model = normalize_gateway_model_id(model)
    agent_type = next(
        (candidate for candidate in AgentType if not candidate.is_cli and candidate.value == model),
        None,
    )
    if agent_type is not None:
        model_id = model
    else:
        try:
            gateway_models = list_gateway_models()
        except Exception:
            gateway_models = []
        gateway_models = list(gateway_models)
        for gateway_model in gateway_models:
            if model in (
                gateway_model.id,
                gateway_model.name,
                gateway_model.model_name,
            ):
                agent_str = gateway_model.sdk_agent_type
                if agent_str == "operator":
                    raise ValueError(
                        "Operator agent is no longer supported; use openai with a supported "
                        "OpenAI computer model."
                    )
                if agent_str == "gemini_cua":
                    raise ValueError(
                        "Gemini CUA agent is no longer supported; use gemini with a supported "
                        "Gemini computer-use model."
                    )
                if not isinstance(agent_str, str):
                    raise ValueError(f"Model '{model}' has invalid agent type metadata")

                try:
                    agent_type = AgentType(agent_str)
                except ValueError as exc:
                    raise ValueError(f"Model '{model}' has invalid agent type metadata") from exc
                if agent_type.is_cli:
                    raise ValueError(f"Model '{model}' has invalid agent type metadata")
                model_id = gateway_model.model_name or model
                break
        else:
            import difflib

            known = [c.value for c in AgentType if not c.is_cli] + [
                n
                for gm in gateway_models
                for n in (gm.id, gm.name, gm.model_name)
                if isinstance(n, str)
            ]
            known.extend(gateway_model_aliases())
            near = difflib.get_close_matches(requested_model, known, n=3, cutoff=0.5)
            hint = (
                f" Did you mean: {', '.join(near)}?"
                if near
                else " Run `hud models` to list available models."
            )
            source = (
                "the HUD gateway registry"
                if gateway_models
                else "the HUD gateway registry (empty — is HUD_API_KEY set?)"
            )
            raise ValueError(f"Model {requested_model!r} not found in {source}.{hint}")

    kwargs.setdefault("model", model_id)
    config = agent_type.config_cls(**kwargs)
    return cast("GatewayAgent", agent_type.instantiate(config))


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
