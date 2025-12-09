"""Environment types for configuration and tracing."""

from __future__ import annotations

from pydantic import BaseModel

from hud.types import MCPToolCall

__all__ = ["EnvConfig", "HubConfig"]


class HubConfig(BaseModel):
    """Configuration for a single hub connection."""

    slug: str
    alias: str | None = None
    prefix: str | None = None
    include: list[str] | None = None
    exclude: list[str] | None = None


class EnvConfig(BaseModel):
    """Environment configuration for trace reproducibility."""

    name: str
    hubs: list[HubConfig] = []
    setup_tools: list[MCPToolCall] = []
    evaluate_tools: list[MCPToolCall] = []
