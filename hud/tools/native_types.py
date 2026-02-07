"""Native tool specification types for framework-specific tool configurations."""

from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer

if TYPE_CHECKING:
    from hud.types import AgentType


class NativeToolSpec(BaseModel):
    """Specification for how a tool registers with a specific agent framework.

    This defines the native API configuration that agents use to register tools
    with their provider's native tool format (e.g., Claude's computer_20250124,
    Gemini's google_search, etc.).

    Attributes:
        api_type: The provider's native tool type identifier
            (e.g., "computer_20250124", "bash_20250124", "google_search").
            Optional - when None, the tool uses standard function calling but
            still participates in role-based mutual exclusion.
        api_name: Override the MCP tool name when registering with the provider
            (e.g., "computer" instead of "anthropic_computer")
        beta: Beta header required for this tool (e.g., "computer-use-2025-01-24")
        hosted: True if the provider executes this tool server-side,
            False if the client executes it
        role: Tool category for mutual exclusion (e.g., "computer", "shell", "editor").
            When an agent accepts a tool natively, other tools with the same role
            are excluded. This prevents having multiple shell/editor tools registered.
            Can be specified alone (without api_type) for function-calling tools
            that need mutual exclusion with native tools.
        supported_models: List of model name patterns that support this native tool.
            Uses fnmatch-style wildcards (e.g., "claude-3-5-sonnet-*", "o3-*").
            If None or empty, all models of this agent type are supported.
            When a model doesn't match, the tool falls back to generic function calling.
        extra: Additional provider-specific parameters
    """

    model_config = ConfigDict(frozen=True)

    api_type: str | None = None
    api_name: str | None = None
    beta: str | None = None
    hosted: bool = False
    role: str | None = None
    supported_models: tuple[str, ...] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_serializer("supported_models")
    @staticmethod
    def serialize_supported_models(value: tuple[str, ...] | None) -> list[str] | None:
        """Serialize tuple to list for JSON compatibility."""
        if value is None:
            return None
        return list(value)

    @property
    def is_native(self) -> bool:
        """Return True if this spec defines a native API tool (not just role)."""
        return self.api_type is not None

    def supports_model(self, model: str | None) -> bool:
        """Check if this native spec supports the given model.

        Uses fnmatch-style pattern matching (supports *, ?, [seq], [!seq]).

        Examples:
            - "claude-3-5-sonnet-*" matches "claude-3-5-sonnet-20241022"
            - "o3-*" matches "o3-mini", "o3-2025-04-16"
            - "gpt-4o" matches exactly "gpt-4o"

        Returns:
            True if the model is supported or if no model restrictions exist.
            False if the model doesn't match any supported pattern.
        """
        # No restrictions means all models supported
        if not self.supported_models:
            return True

        # No model specified means we can't verify - default to supported
        if not model:
            return True

        # Check if model matches any pattern
        model_lower = model.lower()
        for pattern in self.supported_models:
            if fnmatch.fnmatch(model_lower, pattern.lower()):
                return True

        return False


# Type alias for mapping AgentType to NativeToolSpec (or a list for model-specific variants).
# When a list is provided, specs are tried in order -- first matching supports_model() wins.
# Defined as a string annotation to avoid circular import issues.
NativeToolSpecs = dict["AgentType", "NativeToolSpec | list[NativeToolSpec]"]

__all__ = ["NativeToolSpec", "NativeToolSpecs"]
