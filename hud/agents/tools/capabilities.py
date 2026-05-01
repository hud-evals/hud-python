"""Capability helpers for agent-owned tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Self

if TYPE_CHECKING:
    from mcp import types as mcp_types

    from hud.agents.tools.base import AgentToolSpec


@dataclass(frozen=True)
class EnvironmentCapability:
    """A normalized environment capability bound to one or more MCP tools."""

    name: str
    tool_name: str
    tool: mcp_types.Tool
    metadata: dict[str, Any] = field(default_factory=dict)


def capabilities_metadata_from_context(ctx: Any) -> dict[str, Any] | None:
    """Extract an optional env-level capability descriptor from a context."""
    if ctx is None:
        return None

    direct = getattr(ctx, "environment_capabilities", None)
    if isinstance(direct, dict):
        return direct

    direct = getattr(ctx, "capabilities", None)
    if isinstance(direct, dict):
        return {"capabilities": direct}

    metadata = getattr(ctx, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("environment_capabilities", "capabilities"):
            value = metadata.get(key)
            if isinstance(value, dict):
                return value if key == "environment_capabilities" else {"capabilities": value}

    return None


def discover_environment_capabilities(
    tools: list[mcp_types.Tool],
    *,
    env_metadata: dict[str, Any] | None = None,
    name_fallbacks: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, EnvironmentCapability]:
    """Build a normalized capability map from env metadata and tool inventory."""
    tool_by_name = {tool.name: tool for tool in tools}
    capabilities: dict[str, EnvironmentCapability] = {}

    _add_env_capabilities(capabilities, tool_by_name, env_metadata)
    _add_name_fallback_capabilities(capabilities, tool_by_name, name_fallbacks or {})

    return capabilities


def _add_env_capabilities(
    capabilities: dict[str, EnvironmentCapability],
    tool_by_name: dict[str, mcp_types.Tool],
    env_metadata: dict[str, Any] | None,
) -> None:
    if not env_metadata:
        return

    raw = env_metadata.get("capabilities", env_metadata)
    if not isinstance(raw, dict):
        return

    for name, config in raw.items():
        if not isinstance(name, str) or name in capabilities:
            continue
        tool_name: str | None = None
        metadata: dict[str, Any] = {}
        if isinstance(config, str):
            tool_name = config
        elif isinstance(config, dict):
            raw_tool = config.get("tool") or config.get("tool_name")
            if isinstance(raw_tool, str):
                tool_name = raw_tool
                metadata = dict(config)
            else:
                raw_tools = config.get("tools")
                if isinstance(raw_tools, dict):
                    tool_names = {
                        str(key): value
                        for key, value in raw_tools.items()
                        if isinstance(value, str) and value in tool_by_name
                    }
                    if tool_names:
                        tool_name = next(iter(tool_names.values()))
                        metadata = {**config, "tools": tool_names}
        if tool_name is None:
            continue
        tool = tool_by_name.get(tool_name)
        if tool is None:
            continue
        capabilities[name] = EnvironmentCapability(
            name=name,
            tool_name=tool.name,
            tool=tool,
            metadata=metadata,
        )


def _add_name_fallback_capabilities(
    capabilities: dict[str, EnvironmentCapability],
    tool_by_name: dict[str, mcp_types.Tool],
    name_fallbacks: dict[str, tuple[str, ...]],
) -> None:
    for capability, names in name_fallbacks.items():
        if capability in capabilities:
            continue
        matched_tool_names = [name for name in names if name in tool_by_name]
        tool_name = matched_tool_names[0] if matched_tool_names else None
        if tool_name is None:
            continue
        tool = tool_by_name[tool_name]
        capabilities[capability] = EnvironmentCapability(
            name=capability,
            tool_name=tool.name,
            tool=tool,
            metadata={"tools": {name: name for name in matched_tool_names}},
        )


class GroupedCapabilityMixin:
    """Mixin for module capabilities backed by several environment tools."""

    env_tool_names: ClassVar[tuple[str, ...]]

    if TYPE_CHECKING:

        def __init__(self, *, env_tool_name: str, spec: AgentToolSpec) -> None: ...

    @classmethod
    def env_tool_name_for_capability(cls, capability: EnvironmentCapability) -> str | None:
        tools = capability.metadata.get("tools")
        if isinstance(tools, dict):
            return next(
                (tools[name] for name in cls.env_tool_names if isinstance(tools.get(name), str)),
                None,
            )
        if capability.tool_name in cls.env_tool_names:
            return capability.tool_name
        return None

    @classmethod
    def from_capability(
        cls,
        capability: EnvironmentCapability,
        spec: AgentToolSpec,
        model: str,
    ) -> Self:
        del model
        env_tool_name = cls.env_tool_name_for_capability(capability) or capability.tool_name
        return cls(env_tool_name=env_tool_name, spec=spec)


__all__ = [
    "EnvironmentCapability",
    "GroupedCapabilityMixin",
    "capabilities_metadata_from_context",
    "discover_environment_capabilities",
]
