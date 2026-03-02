"""Environment types for configuration and tracing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

__all__ = ["EnvConfig", "ScenarioArg", "ScenarioInfo"]


class EnvConfig(BaseModel):
    """Environment configuration for Tasks.

    Specifies which hub to connect to and optional tool filtering.

    Attributes:
        name: Hub name to connect via connect_hub() (e.g., "browser", "sheets")
        include: Optional whitelist of tool names to include
        exclude: Optional blacklist of tool names to exclude
    """

    name: str = Field(description="Hub name to connect to")
    include: list[str] | None = Field(default=None, description="Whitelist of tool names")
    exclude: list[str] | None = Field(default=None, description="Blacklist of tool names")


@dataclass
class ScenarioArg:
    """Metadata for a single scenario argument."""

    name: str
    type: str = "string"
    required: bool = True
    description: str | None = None
    default: Any = None


@dataclass
class ScenarioInfo:
    """Structured metadata for a scenario, suitable for Agent Cards or tool definitions."""

    name: str
    short_name: str
    description: str | None = None
    arguments: list[ScenarioArg] = field(default_factory=list)

    @property
    def required_args(self) -> list[str]:
        return [a.name for a in self.arguments if a.required]

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert to an OpenAI function tool definition."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for arg in self.arguments:
            prop: dict[str, Any] = {"type": arg.type}
            if arg.description:
                prop["description"] = arg.description
            properties[arg.name] = prop
            if arg.required:
                required.append(arg.name)

        return {
            "type": "function",
            "function": {
                "name": self.short_name,
                "description": self.description or f"Run scenario {self.short_name}",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
