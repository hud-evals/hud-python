"""Capability helpers for agent-owned tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mcp import types as mcp_types

    from hud.types import JsonObject, JsonValue

else:
    JsonObject = dict[str, object]
    JsonValue = object


class CapabilityEntry(TypedDict, total=False):
    tool: str
    tool_name: str
    tools: dict[str, str]


class ToolMetadata(TypedDict, total=False):
    capabilities: dict[str, str | CapabilityEntry]


class EnvironmentCapability:
    """A normalized environment capability bound to one or more MCP tools."""

    def __init__(
        self,
        *,
        name: str,
        tool_name: str,
        tool: mcp_types.Tool,
        metadata: JsonObject | None = None,
    ) -> None:
        self.name = name
        self.tool_name = tool_name
        self.tool = tool
        self.metadata: JsonObject = metadata or {}


def discover_environment_capabilities(
    tools: list[mcp_types.Tool],
    *,
    tool_metadata: ToolMetadata | None = None,
    name_fallbacks: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, EnvironmentCapability]:
    """Build a normalized capability map from env metadata and tool inventory."""
    tool_by_name = {tool.name: tool for tool in tools}
    capabilities: dict[str, EnvironmentCapability] = {}

    metadata = tool_metadata or {}
    raw_capabilities = cast(
        "dict[str, str | CapabilityEntry]",
        metadata.get("capabilities", metadata),
    )
    for name, config in raw_capabilities.items():
        match config:
            case str() as tool_name:
                capability_metadata: JsonObject = {}
            case {"tool": str() as tool_name}:
                capability_metadata = {"tool": tool_name}
            case {"tool_name": str() as tool_name}:
                capability_metadata = {"tool_name": tool_name}
            case {"tools": grouped_tools}:
                tool_names: dict[str, JsonValue] = {
                    str(alias): env_tool_name
                    for alias, env_tool_name in grouped_tools.items()
                    if env_tool_name in tool_by_name
                }
                if not tool_names:
                    continue
                tool_name = str(next(iter(tool_names.values())))
                capability_metadata = {"tools": tool_names}
            case _:
                raise ValueError(f"Invalid capability metadata for {name!r}: {config!r}")

        if tool_name not in tool_by_name:
            continue

        capabilities[name] = EnvironmentCapability(
            name=name,
            tool_name=tool_name,
            tool=tool_by_name[tool_name],
            metadata=capability_metadata,
        )

    for capability, names in (name_fallbacks or {}).items():
        if capability in capabilities:
            continue
        matched_tool_names = [name for name in names if name in tool_by_name]
        if not matched_tool_names:
            continue

        tool = tool_by_name[matched_tool_names[0]]
        capabilities[capability] = EnvironmentCapability(
            name=capability,
            tool_name=tool.name,
            tool=tool,
            metadata={"tools": {name: name for name in matched_tool_names}},
        )
    return capabilities


class GroupedCapabilityMixin:
    """Mixin for module capabilities backed by several environment tools."""

    env_tool_names: ClassVar[tuple[str, ...]]

    @classmethod
    def env_tool_name_for_capability(cls, capability: EnvironmentCapability) -> str | None:
        tools_obj = capability.metadata.get("tools")
        if isinstance(tools_obj, dict):
            tools_map = cast("dict[str, object]", tools_obj)
            for name in cls.env_tool_names:
                if env_tool_name := tools_map.get(name):
                    return str(env_tool_name)
        if capability.tool_name in cls.env_tool_names:
            return capability.tool_name
        return None
