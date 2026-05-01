"""Shared hosted-tool machinery configured by agent harnesses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .base import AgentToolSpec

HostedToolParamT = TypeVar("HostedToolParamT")
HostedToolT = TypeVar("HostedToolT", bound="HostedTool[Any]")


@dataclass(frozen=True, kw_only=True)
class HostedTool(Generic[HostedToolParamT]):
    """Provider-side tool activated only through explicit agent config."""

    supported_models: tuple[str, ...] | None = None

    def supports_model(self, model: str | None) -> bool:
        spec = AgentToolSpec(
            api_type="hosted",
            api_name=self.__class__.__name__,
            supported_models=self.supported_models,
        )
        return spec.supports_model(model)

    def to_params(self) -> HostedToolParamT:
        raise NotImplementedError


def select_hosted_tools(
    hosted_tools: list[Any],
    *,
    tool_type: type[HostedToolT],
    model: str,
) -> list[HostedToolT]:
    """Select explicitly configured hosted tools for one provider/model."""
    selected: list[HostedToolT] = []
    for hosted_tool in hosted_tools:
        if not isinstance(hosted_tool, tool_type) or not hosted_tool.supports_model(model):
            continue
        selected.append(hosted_tool)
    return selected


__all__ = [
    "HostedTool",
    "select_hosted_tools",
]
