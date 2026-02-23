"""Cursor config parsing utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


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


def cursor_list_command() -> None:
    """📋 List all MCP servers configured in Cursor."""
    console = Console()
    console.print(Panel.fit("📋 [bold cyan]Cursor MCP Servers[/bold cyan]", border_style="cyan"))

    servers, error = list_cursor_servers()

    if error:
        console.print(f"[red]❌ {error}[/red]")
        raise typer.Exit(1)

    if not servers:
        console.print("[yellow]No servers found in Cursor config[/yellow]")
        return

    table = Table(title="Available Servers")
    table.add_column("Server Name", style="cyan")
    table.add_column("Command Preview", style="dim")

    config_path = get_cursor_config_path()
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            mcp_servers = config.get("mcpServers", {})

            for server_name in servers:
                server_config = mcp_servers.get(server_name, {})
                command = server_config.get("command", "")
                args = server_config.get("args", [])

                if args:
                    preview = f"{command} {' '.join(args[:2])}"
                    if len(args) > 2:
                        preview += " ..."
                else:
                    preview = command

                table.add_row(server_name, preview)

    console.print(table)
    console.print(f"\n[dim]Config location: {config_path}[/dim]")
    console.print(
        "\n[green]Tip:[/green] Use [cyan]hud debug --cursor <server-name>[/cyan] to debug a server"
    )
