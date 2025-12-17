"""Utility functions for the eval module."""

from __future__ import annotations

import logging
import warnings
from typing import Any

__all__ = ["build_env_from_v4", "is_v4_format", "validate_v4_task"]

logger = logging.getLogger(__name__)


def is_v4_format(data: dict[str, Any]) -> bool:
    """Detect if dict looks like v4 LegacyTask format.

    Used for branching logic. Checks if data has the core v4 fields
    (prompt AND mcp_config). Does NOT validate completeness.

    Args:
        data: Dict to check

    Returns:
        True if looks like v4 format, False otherwise
    """
    if not isinstance(data, dict):
        return False

    # Core v4 detection: prompt + mcp_config
    return bool(data.get("prompt")) and bool(data.get("mcp_config"))


def validate_v4_task(data: dict[str, Any]) -> None:
    """Validate v4 task has all required fields.

    A valid v4 task must have all three required fields:
    - prompt: The task instruction
    - mcp_config: MCP server configuration
    - evaluate_tool: How to evaluate success

    Call this after is_v4_format() when you need to ensure completeness.

    Args:
        data: Dict to validate

    Raises:
        ValueError: If any required fields are missing
    """
    missing = []
    if not data.get("prompt"):
        missing.append("prompt")
    if not data.get("mcp_config"):
        missing.append("mcp_config")
    if not data.get("evaluate_tool"):
        missing.append("evaluate_tool")

    if missing:
        raise ValueError(f"v4 task missing required fields: {', '.join(missing)}")


def build_env_from_v4(source: dict[str, Any] | Any) -> dict[str, Any]:
    """Build Environment from v4 LegacyTask format.

    Creates an Environment configured with the legacy task's fields.
    Returns a dict ready to be passed to Task() constructor.

    Args:
        source: dict or LegacyTask with v4 fields (prompt, mcp_config, etc.)

    Returns:
        Dict with Task fields: env, id, scenario, args, validation, system_prompt, metadata

    Raises:
        TypeError: If source is not a dict or LegacyTask
    """
    from hud.environment import Environment
    from hud.types import LegacyTask, MCPToolCall

    # Convert dict to LegacyTask if needed
    if isinstance(source, dict):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            legacy = LegacyTask(**source)
    elif isinstance(source, LegacyTask):
        legacy = source
    else:
        raise TypeError(f"Expected dict or LegacyTask, got {type(source).__name__}")

    # Warn if using local MCP configs (command without url)
    _warn_local_mcp(legacy.mcp_config)

    # Extract tool filters from agent_config (v4 style)
    # These are agent-level filters, not connection-level
    include_tools: list[str] | None = None
    exclude_tools: list[str] | None = None
    if legacy.agent_config:
        include_tools = legacy.agent_config.allowed_tools
        exclude_tools = legacy.agent_config.disallowed_tools

    # Convert ["*"] wildcard to None (meaning include all)
    if include_tools == ["*"]:
        include_tools = None

    # Create Environment - NO connections made here, just config stored
    env = Environment(legacy.id or "v4-legacy")
    env.connect_mcp_config(legacy.mcp_config)

    # Store agent-level tool filters on Environment (applied in as_tools())
    # This allows Environment to call setup/evaluate while hiding them from agent
    env._agent_include = include_tools
    env._agent_exclude = exclude_tools

    # Set the prompt
    env.prompt = legacy.prompt

    # Add setup_tool calls (stored, not executed)
    if legacy.setup_tool:
        setup_calls = legacy.setup_tool
        if not isinstance(setup_calls, list):
            setup_calls = [setup_calls]
        for call in setup_calls:
            env.setup_tool(call.name, **(call.arguments or {}))

    # Add evaluate_tool calls (stored, not executed)
    if legacy.evaluate_tool:
        eval_calls = legacy.evaluate_tool
        if not isinstance(eval_calls, list):
            eval_calls = [eval_calls]
        for call in eval_calls:
            env.evaluate_tool(call.name, **(call.arguments or {}))

    # Build Task fields dict
    result: dict[str, Any] = {
        "env": env,
        "id": legacy.id,
        "scenario": None,  # v4 uses prompt, not scenarios
        "args": {},
    }

    # Map integration_test_tool → validation (same concept: tool calls to verify)
    if legacy.integration_test_tool:
        int_test = legacy.integration_test_tool
        if not isinstance(int_test, list):
            int_test = [int_test]
        # Convert to MCPToolCall if needed
        result["validation"] = [
            call if isinstance(call, MCPToolCall) else MCPToolCall(**call.model_dump())
            for call in int_test
        ]

    # Extract agent_config (just system_prompt for now)
    if legacy.agent_config and legacy.agent_config.system_prompt:
        result["agent_config"] = {"system_prompt": legacy.agent_config.system_prompt}

    # Preserve metadata
    if legacy.metadata:
        result["metadata"] = legacy.metadata

    return result


def _warn_local_mcp(mcp_config: dict[str, Any] | None) -> None:
    """Warn if mcp_config uses local MCP servers (command without url).

    Local MCP servers can cause port conflicts when running tasks concurrently.
    """
    if not mcp_config:
        return

    has_local = any(
        isinstance(server_cfg, dict) and "command" in server_cfg and not server_cfg.get("url")
        for server_cfg in mcp_config.values()
        if isinstance(server_cfg, dict)
    )

    if has_local:
        warnings.warn(
            "Task uses local MCP configuration (command without url). "
            "This may cause port conflicts when running tasks concurrently. "
            "Consider using remote MCP servers for parallel execution.",
            UserWarning,
            stacklevel=4,
        )
