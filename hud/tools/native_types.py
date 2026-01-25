"""Native tool specification types for framework-specific tool configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from hud.types import AgentType


class NativeToolSpec(BaseModel):
    """Specification for how a tool registers with a specific agent framework.

    This defines the native API configuration that agents use to register tools
    with their provider's native tool format (e.g., Claude's computer_20250124,
    Gemini's google_search, etc.).

    Attributes:
        api_type: The provider's native tool type identifier
            (e.g., "computer_20250124", "bash_20250124", "google_search")
        api_name: Override the MCP tool name when registering with the provider
            (e.g., "computer" instead of "anthropic_computer")
        beta: Beta header required for this tool (e.g., "computer-use-2025-01-24")
        hosted: True if the provider executes this tool server-side,
            False if the client executes it
        role: Tool category for mutual exclusion (e.g., "computer", "bash", "editor").
            When an agent accepts a tool natively, other tools with the same role
            are excluded. This prevents having multiple computer tools registered.
        extra: Additional provider-specific parameters
    """

    model_config = ConfigDict(frozen=True)

    api_type: str
    api_name: str | None = None
    beta: str | None = None
    hosted: bool = False
    role: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


# Type alias for mapping AgentType to NativeToolSpec
# Defined as a string annotation to avoid circular import issues
NativeToolSpecs = dict["AgentType", NativeToolSpec]

__all__ = ["NativeToolSpec", "NativeToolSpecs"]
