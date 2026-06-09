"""Agent implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from hud.shared.gateway import build_gateway_client, list_gateway_models
from hud.types import AgentType

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
    agent_type = next((candidate for candidate in AgentType if candidate.value == model), None)
    if agent_type is not None:
        model_id = model
        provider_name = agent_type.gateway_provider
    else:
        try:
            gateway_models = list_gateway_models()
        except Exception:
            gateway_models = []
        for gateway_model in gateway_models:
            if model in (
                gateway_model.id,
                gateway_model.name,
                gateway_model.model_name,
            ):
                agent_str = (
                    gateway_model.sdk_agent_type or gateway_model.provider.default_sdk_agent_type
                )
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
            raise ValueError(f"Model '{model}' not found")

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
            f"{name} requires the agents extra. Install with: pip install 'hud-python[agents]'"
        ) from exc
    globals()[name] = value
    return value
