"""Agent implementations.

The robot policy harness lives in :mod:`hud.agents.robot` (requires the ``robot`` extra).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from hud.types import AgentType
from hud.utils.gateway import (
    build_gateway_client,
    gateway_model_aliases,
    list_gateway_models,
    normalize_gateway_model_id,
)

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

    For direct API access with provider API keys, instantiate the agent classes directly.
    """
    requested_model = model
    model = normalize_gateway_model_id(model)
    agent_type = next((candidate for candidate in AgentType if candidate.value == model), None)
    if agent_type is not None:
        model_id = model
        provider_name = agent_type.gateway_provider
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
                model_id = gateway_model.model_name or model
                provider_name = gateway_model.provider.name or "openai"
                break
        else:
            import difflib

            known = [c.value for c in AgentType] + [
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
    kwargs.setdefault("model_client", build_gateway_client(provider_name))
    # cls/config_cls are matched unions; the pairing is correct by construction.
    config = agent_type.config_cls(**kwargs)
    return agent_type.cls(cast("Any", config))


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
