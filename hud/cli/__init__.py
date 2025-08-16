"""HUD CLI - Command-line interface for MCP environment analysis and debugging."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path  # noqa: TC003

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .analyze import (
    analyze_environment,
    analyze_environment_from_config,
    analyze_environment_from_mcp_config,
)
from .cursor import get_cursor_config_path, list_cursor_servers, parse_cursor_config
from .debug import debug_mcp_stdio
from .mcp_server import run_mcp_server
from .utils import CaptureLogger

# Create the main Typer app
app = typer.Typer(
    name="hud",
    help="🚀 HUD CLI for MCP environment analysis and debugging",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()


# Capture IMAGE and any following Docker args as a single variadic argument list.
@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def analyze(
    params: list[str] = typer.Argument(  # type: ignore[arg-type]  # noqa: B008
        ...,  # Required positional arguments
        help="Docker image followed by optional Docker run arguments (e.g., 'hud-image:latest -e KEY=value')",  # noqa: E501
    ),
    config: Path = typer.Option(  # noqa: B008
        None,
        "--config",
        "-c",
        help="JSON config file with MCP configuration",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    cursor: str | None = typer.Option(
        None,
        "--cursor",
        help="Analyze a server from Cursor config",
    ),
    output_format: str = typer.Option(
        "interactive",
        "--format",
        "-f",
        help="Output format: interactive, json, markdown",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output (shows tool schemas)",
    ),
) -> None:
    """🔍 Analyze MCP environment - discover tools, resources, and capabilities.

    Examples:
        hud analyze hud-text-2048:latest
        hud analyze my-mcp-server:v1 -e API_KEY=xxx
        hud analyze --config mcp-config.json
        hud analyze --cursor text-2048-dev
    """
    if config:
        # Load config from JSON file
        asyncio.run(analyze_environment_from_config(config, output_format, verbose))
    elif cursor:
        # Parse cursor config
        command, error = parse_cursor_config(cursor)
        if error or command is None:
            console.print(f"[red]❌ {error or 'Failed to parse cursor config'}[/red]")
            raise typer.Exit(1)
        # Convert to MCP config
        mcp_config = {
            "local": {"command": command[0], "args": command[1:] if len(command) > 1 else []}
        }
        asyncio.run(analyze_environment_from_mcp_config(mcp_config, output_format, verbose))
    elif params:
        image, *docker_args = params
        # Build Docker command from image and args
        docker_cmd = ["docker", "run", "--rm", "-i", *docker_args, image]
        asyncio.run(analyze_environment(docker_cmd, output_format, verbose))
    else:
        console.print("[red]Error: Must specify either a Docker image, --config, or --cursor[/red]")
        console.print("\nExamples:")
        console.print("  hud analyze hud-text-2048:latest")
        console.print("  hud analyze --config mcp-config.json")
        console.print("  hud analyze --cursor my-server")
        raise typer.Exit(1)


# Same variadic approach for debug.
@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def debug(
    params: list[str] = typer.Argument(  # type: ignore[arg-type]  # noqa: B008
        ...,
        help="Docker image followed by optional Docker run arguments (e.g., 'hud-image:latest -e KEY=value')",  # noqa: E501
    ),
    config: Path = typer.Option(  # noqa: B008
        None,
        "--config",
        "-c",
        help="JSON config file with MCP configuration",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    cursor: str | None = typer.Option(
        None,
        "--cursor",
        help="Debug a server from Cursor config",
    ),
    max_phase: int = typer.Option(
        5,
        "--max-phase",
        "-p",
        min=1,
        max=5,
        help="Maximum debug phase (1-5)",
    ),
) -> None:
    """🐛 Debug MCP environment - test initialization, tools, and readiness.

    Examples:
        hud debug hud-text-2048:latest
        hud debug my-mcp-server:v1 -e API_KEY=xxx -p 8080:8080
        hud debug --config mcp-config.json
        hud debug --cursor text-2048-dev
        hud debug hud-browser:dev --max-phase 3
    """

    # Determine the command to run
    command = None

    if config:
        # Load config from JSON file
        with open(config) as f:
            mcp_config = json.load(f)

        # Extract command from first server in config
        server_name = next(iter(mcp_config.keys()))
        server_config = mcp_config[server_name]
        command = [server_config["command"], *server_config.get("args", [])]
    elif cursor:
        # Parse cursor config
        command, error = parse_cursor_config(cursor)
        if error or command is None:
            console.print(f"[red]❌ {error or 'Failed to parse cursor config'}[/red]")
            raise typer.Exit(1)
    elif params:
        image, *docker_args = params
        # Build Docker command
        command = ["docker", "run", "--rm", "-i", *docker_args, image]
    else:
        console.print("[red]Error: Must specify either a Docker image, --config, or --cursor[/red]")
        console.print("\nExamples:")
        console.print("  hud debug hud-text-2048:latest")
        console.print("  hud debug --config mcp-config.json")
        console.print("  hud debug --cursor my-server")
        raise typer.Exit(1)

    # Create logger and run debug
    logger = CaptureLogger(print_output=True)
    phases_completed = asyncio.run(debug_mcp_stdio(command, logger, max_phase=max_phase))

    # Exit with appropriate code
    if phases_completed < max_phase:
        raise typer.Exit(1)


@app.command()
def cursor_list() -> None:
    """📋 List all MCP servers configured in Cursor."""
    console.print(Panel.fit("📋 [bold cyan]Cursor MCP Servers[/bold cyan]", border_style="cyan"))

    servers, error = list_cursor_servers()

    if error:
        console.print(f"[red]❌ {error}[/red]")
        raise typer.Exit(1)

    if not servers:
        console.print("[yellow]No servers found in Cursor config[/yellow]")
        return

    # Display servers in a table
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

                # Create command preview
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


@app.command()
def version() -> None:
    """Show HUD CLI version."""
    try:
        from hud import __version__

        console.print(f"HUD CLI version: [cyan]{__version__}[/cyan]")
    except ImportError:
        console.print("HUD CLI version: [cyan]unknown[/cyan]")


@app.command()
def mcp() -> None:
    """🤖 Run HUD as an MCP server - expose debug/analyze tools to other MCP clients.

    This allows you to use HUD's debugging and analysis capabilities from:
    - Cursor (add to .cursor/mcp.json)
    - Claude Desktop
    - Any other MCP client

    Example Cursor config:
    {
      "mcpServers": {
        "hud-cli": {
          "command": "hud",
          "args": ["mcp"]
        }
      }
    }
    """
    console.print(
        Panel.fit(
            "[bold cyan]🤖 Starting HUD MCP Server[/bold cyan]\n"
            "Exposing debug and analyze tools via MCP protocol",
            border_style="cyan",
        )
    )
    console.print("\n[dim]Server starting on stdio...[/dim]")
    console.print("[dim]Add to your MCP client config to use.[/dim]\n")

    # Run the MCP server
    run_mcp_server()


def main() -> None:
    """Main entry point for the CLI."""
    # Show header for main help
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ["--help", "-h"]):
        console.print(
            Panel.fit(
                "[bold cyan]🚀 HUD CLI[/bold cyan]\nMCP Environment Analysis & Debugging",
                border_style="cyan",
            )
        )
        console.print("\n[yellow]Quick Start:[/yellow]")
        console.print("  1. Build your Docker image: [cyan]docker build -t my-mcp-server .[/cyan]")
        console.print("  2. Debug it: [cyan]hud debug my-mcp-server[/cyan]")
        console.print("  3. Analyze it: [cyan]hud analyze my-mcp-server[/cyan]")
        console.print("  4. Use as MCP server: [cyan]hud mcp[/cyan] (for Cursor/Claude)\n")

    app()


if __name__ == "__main__":
    main()
