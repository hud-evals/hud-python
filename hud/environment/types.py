"""Environment types for configuration and tracing."""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["EnvConfig"]


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
