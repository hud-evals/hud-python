"""OpenAI hosted tools configured by the OpenAI harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from openai.types.responses import ToolParam

from hud.agents.tools import HostedTool


@dataclass(frozen=True, kw_only=True)
class OpenAIHostedTool(HostedTool[ToolParam]):
    """OpenAI-hosted tool configured by the OpenAI harness."""


@dataclass(frozen=True, kw_only=True)
class OpenAICodeInterpreterTool(OpenAIHostedTool):
    """OpenAI code interpreter."""

    container: dict[str, Any]

    def to_params(self) -> ToolParam:
        return cast("ToolParam", {"type": "code_interpreter", "container": self.container})


@dataclass(frozen=True, kw_only=True)
class OpenAIToolSearchTool(OpenAIHostedTool):
    """OpenAI tool search for large tool sets."""

    threshold: int = 10
    supported_models: tuple[str, ...] | None = ("gpt-5.4", "gpt-5.4-*")

    def to_params(self) -> ToolParam:
        return cast("ToolParam", {"type": "tool_search"})


__all__ = [
    "OpenAICodeInterpreterTool",
    "OpenAIHostedTool",
    "OpenAIToolSearchTool",
]
