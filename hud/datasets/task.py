"""Task model for HUD datasets."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from string import Template
from typing import Any

from pydantic import BaseModel, Field, field_validator

from hud.settings import settings
from hud.types import MCPToolCall

logger = logging.getLogger(__name__)


class Task(BaseModel):
    """
    A task configuration that can be used to create a task.

    The mcp_config field supports environment variable substitution using
    template placeholders in the format ${VAR_NAME} or ${VAR_NAME:default_value}.

    Example:
        mcp_config: {
            "hud": {
                "url": "${HUD_MCP_URL:https://mcp.hud.so/v3/mcp}",
                "headers": {
                    "Authorization": "Bearer ${HUD_API_KEY}",
                    "Mcp-Image": "your-mcp-image"
                }
            }
        }
    """

    id: str | None = None
    prompt: str
    mcp_config: dict[str, Any]
    setup_tool: MCPToolCall | list[MCPToolCall] | None = None
    evaluate_tool: MCPToolCall | list[MCPToolCall] | None = None
    mock_tool: MCPToolCall | list[MCPToolCall] | None = None
    system_prompt: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("mcp_config", "metadata", mode="before")
    @classmethod
    def parse_json_strings(cls, v: Any) -> Any:
        """Parse JSON strings into dictionaries."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError as e:
                from hud.shared.exceptions import HudConfigError

                raise HudConfigError(f"Invalid JSON string: {e}") from e
        return v

    @field_validator("setup_tool", "evaluate_tool", "mock_tool", mode="before")
    @classmethod
    def convert_dict_to_tool_call(cls, v: Any) -> Any:
        """Convert dict to MCPToolCall instance, parsing JSON strings first."""
        if v is None:
            return None

        # Parse JSON string if needed
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except json.JSONDecodeError as e:
                from hud.shared.exceptions import HudConfigError

                raise HudConfigError(f"Invalid JSON string: {e}") from e

        if isinstance(v, dict):
            return MCPToolCall(**v)
        if isinstance(v, list):
            return [MCPToolCall(**item) if isinstance(item, dict) else item for item in v]
        return v

    @field_validator("mcp_config", mode="before")
    @classmethod
    def resolve_env_vars(cls, v: dict[str, Any]) -> dict[str, Any]:
        """
        Automatically resolve environment variables in mcp_config using Template.

        Supports ${VAR_NAME} syntax with variable substitution from
        System environment variables (including HUD_API_KEY, etc.)

        Missing variables resolve to empty strings.
        """
        import os

        # Start with current environment variables
        mapping = dict(os.environ)
        mapping.update(settings.model_dump())

        if settings.api_key:
            mapping["HUD_API_KEY"] = settings.api_key
        else:
            logger.error("HUD_API_KEY is not set, tracing and remote training will not work")

        def substitute_in_value(obj: Any) -> Any:
            """Recursively substitute variables in nested structures."""
            if isinstance(obj, str):
                # Use Template's substitute with defaultdict - missing vars become empty strings
                safe_mapping = defaultdict(str, mapping)
                return Template(obj).substitute(safe_mapping)
            elif isinstance(obj, dict):
                return {k: substitute_in_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_in_value(item) for item in obj]
            else:
                return obj

        return substitute_in_value(v)
