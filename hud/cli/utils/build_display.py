"""Rich build summary display for deploy command."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hud.utils.hud_console import HUDConsole


def display_build_summary(
    status_response: dict[str, Any],
    registry_id: str,
    console: HUDConsole | None = None,
    platform_url: str = "https://hud.ai",
) -> None:
    """Display a rich summary of a completed build.

    Args:
        status_response: Response from /builds/{id}/status endpoint
        registry_id: Registry/environment ID
        console: Optional HUDConsole for output
        platform_url: Base URL for HUD platform
    """
    if console is None:
        console = HUDConsole()

    rich_console = Console()

    status = status_response.get("status", "UNKNOWN")
    version = status_response.get("version", "unknown")
    duration = status_response.get("duration_seconds")
    image_name = status_response.get("image_name")
    uri = status_response.get("uri")
    digest = status_response.get("digest")
    lock_data = status_response.get("lock")

    # Format duration
    duration_str = _format_duration(duration) if duration else "unknown"

    # Build status line with checkmark/cross
    if status == "SUCCEEDED":
        status_icon = "[green]✓[/green]"
        status_text = f"{status_icon} [bold green]{status}[/bold green]"
    elif status == "FAILED":
        status_icon = "[red]✗[/red]"
        status_text = f"{status_icon} [bold red]{status}[/bold red]"
    else:
        status_icon = "[yellow]●[/yellow]"
        status_text = f"{status_icon} [bold yellow]{status}[/bold yellow]"

    # Create summary table
    summary_lines = [
        f"[bold]Status:[/bold]     {status_text}",
        f"[bold]Duration:[/bold]   {duration_str}",
        f"[bold]Version:[/bold]    {version}",
    ]

    if uri:
        summary_lines.append(f"[bold]Image:[/bold]      [dim]{uri}[/dim]")
    elif image_name:
        summary_lines.append(f"[bold]Image:[/bold]      [dim]{image_name}[/dim]")

    summary_content = "\n".join(summary_lines)

    # Print main summary panel
    rich_console.print()
    rich_console.print(
        Panel(
            summary_content,
            title="[bold cyan]Build Summary[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    # Parse and display lock data if available
    if lock_data and isinstance(lock_data, dict):
        _display_lock_details(rich_console, lock_data)

    # Print platform link
    env_url = f"{platform_url}/environments/{registry_id}"
    rich_console.print()
    rich_console.print(
        Panel(
            f"[bold]View on HUD:[/bold] [link={env_url}]{env_url}[/link]",
            border_style="blue",
            padding=(0, 2),
        )
    )
    rich_console.print()


def _display_lock_details(
    rich_console: Console,
    lock_data: dict[str, Any],
) -> None:
    """Display details from the lock file.

    Args:
        rich_console: Rich Console for output
        lock_data: Parsed lock file data
    """
    # Display scenarios/prompts
    prompts = lock_data.get("prompts") or lock_data.get("scenarios", [])
    if prompts:
        rich_console.print()
        scenarios_table = Table(
            title=f"[bold]Scenarios ({len(prompts)})[/bold]",
            show_header=True,
            header_style="bold",
            border_style="dim",
        )
        scenarios_table.add_column("Name", style="cyan")
        scenarios_table.add_column("Arguments", style="dim")

        for prompt in prompts[:10]:  # Limit to 10
            name = prompt.get("name", "default")
            args = prompt.get("arguments", [])
            if args:
                arg_strs = []
                for arg in args:
                    arg_name = arg.get("name", "")
                    required = arg.get("required", False)
                    arg_type = arg.get("type", "str")
                    suffix = " (required)" if required else ""
                    arg_strs.append(f"{arg_name}: {arg_type}{suffix}")
                args_str = ", ".join(arg_strs)
            else:
                args_str = "No arguments"
            scenarios_table.add_row(name, args_str)

        if len(prompts) > 10:
            scenarios_table.add_row(
                f"[dim]... and {len(prompts) - 10} more[/dim]",
                "",
            )

        rich_console.print(scenarios_table)

    # Display environment variables
    env_config = lock_data.get("environment", {})
    if env_config:
        required_vars = env_config.get("variables", {}).get("required", [])
        optional_vars = env_config.get("variables", {}).get("optional", [])

        if required_vars or optional_vars:
            rich_console.print()
            env_lines = []
            if required_vars:
                env_lines.append(
                    f"[bold]Required:[/bold] {', '.join(required_vars)}"
                )
            if optional_vars:
                env_lines.append(
                    f"[bold]Optional:[/bold] {', '.join(optional_vars)}"
                )

            rich_console.print(
                Panel(
                    "\n".join(env_lines),
                    title="[bold]Environment Variables[/bold]",
                    border_style="dim",
                    padding=(0, 2),
                )
            )

    # Display tools
    tools = lock_data.get("tools", [])
    if tools:
        tool_names = [t.get("name", str(t)) if isinstance(t, dict) else str(t) for t in tools[:10]]
        tools_str = ", ".join(tool_names)
        if len(tools) > 10:
            tools_str += f", ... and {len(tools) - 10} more"

        rich_console.print()
        rich_console.print(
            Panel(
                f"[bold]Tools ({len(tools)}):[/bold] {tools_str}",
                border_style="dim",
                padding=(0, 2),
            )
        )


def _format_duration(seconds: int | float) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Human-readable duration (e.g., "2m 15s")
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def display_upload_progress(
    uploaded_bytes: int,
    total_bytes: int,
    console: HUDConsole | None = None,
) -> None:
    """Display upload progress.

    Args:
        uploaded_bytes: Bytes uploaded so far
        total_bytes: Total bytes to upload
        console: Optional HUDConsole for output
    """
    if console is None:
        console = HUDConsole()

    from hud.cli.utils.context import format_size

    uploaded_str = format_size(uploaded_bytes)
    total_str = format_size(total_bytes)
    percent = (uploaded_bytes / total_bytes * 100) if total_bytes > 0 else 0

    console.progress_message(f"Uploading: {uploaded_str} / {total_str} ({percent:.1f}%)")
