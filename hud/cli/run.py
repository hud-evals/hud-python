"""Run Docker image as MCP server (local or remote)."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

console = Console()


def run_command(
    params: list[str] = typer.Argument(  # type: ignore[arg-type]  # noqa: B008
        None,
        help="Docker image followed by optional Docker run arguments "
        "(e.g., 'my-image:latest -e KEY=value')",
    ),
    local: bool = typer.Option(
        False,
        "--local",
        help="Run locally with Docker (default: remote via mcp.hud.ai)",
    ),
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="Transport protocol: stdio (default) or http",
    ),
    port: int = typer.Option(
        8765,
        "--port",
        "-p",
        help="Port for HTTP transport (ignored for stdio)",
    ),
    url: str = typer.Option(
        None,
        "--url",
        help="Remote MCP server URL (default: HUD_MCP_URL or mcp.hud.ai)",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="API key for remote server (default: HUD_API_KEY env var)",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Run ID for tracking (remote only)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output",
    ),
) -> None:
    """🚀 Run Docker image as MCP server.

    [not dim]A simple wrapper around 'docker run' that can launch images locally or remotely.
    By default, runs remotely via mcp.hud.ai. Use --local to run with local Docker.

    For local Python development with hot-reload, use 'hud dev' instead.

    Examples:
        hud run my-image:latest                    # Run remotely (default)
        hud run my-image:latest --local            # Run with local Docker
        hud run my-image:latest -e KEY=value       # Remote with env vars
        hud run my-image:latest --local -e KEY=val # Local with env vars
        hud run my-image:latest --transport http   # Use HTTP transport[/not dim]
    """
    if not params:
        console.print("[red]❌ Docker image is required[/red]")
        console.print("\nExamples:")
        console.print("  hud run my-image:latest              # Run remotely (default)")
        console.print("  hud run my-image:latest --local      # Run with local Docker")
        console.print("\n[yellow]For local Python development:[/yellow]")
        console.print("  hud dev                              # Run with hot-reload")
        raise typer.Exit(1)

    image = params[0]
    docker_args = params[1:] if len(params) > 1 else []

    if not any(c in image for c in [":", "/"]) and (
        Path(image).is_dir() or Path(image).is_file() or "." in image
    ):
        console.print(f"[yellow]⚠️  '{image}' looks like a module path, not a Docker image[/yellow]")
        console.print("\n[green]For local Python development, use:[/green]")
        console.print(f"  hud dev {image}")
        console.print("\n[green]For Docker images:[/green]")
        console.print("  hud run my-image:latest")
        raise typer.Exit(1)

    if local:
        from .utils.runner import run_mcp_server

        run_mcp_server(image, docker_args, transport, port, verbose, interactive=False)
    else:
        from .utils.remote_runner import run_remote_server

        if not url:
            from hud.settings import settings

            url = settings.hud_mcp_url

        run_remote_server(image, docker_args, transport, port, url, api_key, run_id, verbose)
