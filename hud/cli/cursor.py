"""Cursor config parsing utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path


def parse_cursor_config(server_name: str) -> tuple[list[str] | None, str | None]:
    """
    Parse cursor config to get command for a server.

    Args:
        server_name: Name of the server in Cursor config

    Returns:
        Tuple of (command_list, error_message). If successful, error_message is None.
        If failed, command_list is None and error_message contains the error.
    """
    # Find cursor config
    cursor_config_path = Path.home() / ".cursor" / "mcp.json"
    if not cursor_config_path.exists():
        # Try Windows path
        cursor_config_path = Path(os.environ.get("USERPROFILE", "")) / ".cursor" / "mcp.json"

    if not cursor_config_path.exists():
        return None, f"Cursor config not found at {cursor_config_path}"

    try:
        with open(cursor_config_path) as f:
            config = json.load(f)

        servers = config.get("mcpServers", {})
        if server_name not in servers:
            available = ", ".join(servers.keys())
            return None, f"Server '{server_name}' not found. Available: {available}"

        server_config = servers[server_name]
        command = server_config.get("command", "")
        args = server_config.get("args", [])
        _ = server_config.get("env", {})

        # Combine command and args
        full_command = [command, *args]

        # Handle reloaderoo wrapper
        if command == "npx" and "reloaderoo" in args and "--" in args:
            # Extract the actual command after --
            dash_index = args.index("--")
            full_command = args[dash_index + 1 :]

        return full_command, None

    except Exception as e:
        return None, f"Error reading config: {e}"


def list_cursor_servers() -> tuple[list[str] | None, str | None]:
    """
    List all available servers in Cursor config.

    Returns:
        Tuple of (server_list, error_message). If successful, error_message is None.
    """
    # Find cursor config
    cursor_config_path = Path.home() / ".cursor" / "mcp.json"
    if not cursor_config_path.exists():
        # Try Windows path
        cursor_config_path = Path(os.environ.get("USERPROFILE", "")) / ".cursor" / "mcp.json"

    if not cursor_config_path.exists():
        return None, f"Cursor config not found at {cursor_config_path}"

    try:
        with open(cursor_config_path) as f:
            config = json.load(f)

        servers = config.get("mcpServers", {})
        return list(servers.keys()), None

    except Exception as e:
        return None, f"Error reading config: {e}"


def get_cursor_config_path() -> Path:
    """Get the path to Cursor's MCP config file."""
    cursor_config_path = Path.home() / ".cursor" / "mcp.json"
    if not cursor_config_path.exists():
        # Try Windows path
        cursor_config_path = Path(os.environ.get("USERPROFILE", "")) / ".cursor" / "mcp.json"
    return cursor_config_path
