"""Registry support for agent-owned tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from .base import AgentTool

if TYPE_CHECKING:
    from hud.agents.tools.capabilities import EnvironmentCapability

ToolT = TypeVar("ToolT", bound=AgentTool[Any])


@dataclass(frozen=True)
class AgentToolRegistry(Generic[ToolT]):
    """Declarative registry for a provider or harness tool family."""

    tool_classes: tuple[type[ToolT], ...]
    name_fallbacks: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset(cls.capability for cls in self.tool_classes)

    def tool_for_capability(
        self,
        capability: EnvironmentCapability,
        model: str,
    ) -> ToolT | None:
        tools = self.tools_for_capability(capability, model)
        return tools[0] if tools else None

    def tools_for_capability(
        self,
        capability: EnvironmentCapability,
        model: str,
    ) -> list[ToolT]:
        tools: list[ToolT] = []
        for tool_cls in self.tool_classes:
            if tool_cls.capability != capability.name:
                continue
            spec = tool_cls.default_spec(model)
            if spec is None:
                continue
            env_tool_name_for_capability = getattr(tool_cls, "env_tool_name_for_capability", None)
            if (
                callable(env_tool_name_for_capability)
                and env_tool_name_for_capability(capability) is None
            ):
                continue
            tools.append(tool_cls.from_capability(capability, spec, model))
        return tools


__all__ = ["AgentToolRegistry"]
