"""Serialization and reconstruction for built-in agents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from hud.agents.types import AgentConfig, ToolAgentConfig
from hud.types import AgentType

if TYPE_CHECKING:
    from hud.agents.base import Agent

_RUNTIME_ONLY_CONFIG_FIELDS = {"model_client", "api_key", "base_url", "hosted_tools"}


def dump_agent(agent: Agent) -> dict[str, Any]:
    """Serialize a registered agent without credentials or live clients."""
    agent_type = AgentType.of(agent)
    config = getattr(agent, "config", None)
    if agent_type is None or not isinstance(config, AgentConfig):
        raise ValueError(
            f"agent must be one of the registered types "
            f"({', '.join(member.value for member in AgentType)}); "
            f"got {type(agent).__name__}"
        )
    if isinstance(config, ToolAgentConfig) and config.model_client is not None:
        raise ValueError(
            "agents with a custom model_client cannot run remotely; use HUDRuntime or LocalRuntime"
        )

    payload = config.model_dump(
        mode="json",
        exclude=_RUNTIME_ONLY_CONFIG_FIELDS,
    )
    return {"type": agent_type.value, "config": payload}


def load_agent(data: Mapping[str, Any]) -> Agent:
    """Reconstruct a registered agent from :func:`dump_agent` output."""
    try:
        agent_type = AgentType(data["type"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"unsupported agent type {data.get('type')!r}") from None

    raw_config = data.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("agent config must be an object")
    config = agent_type.config_cls.model_validate(dict(raw_config))
    return agent_type.instantiate(config)
