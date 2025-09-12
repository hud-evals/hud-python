"""List HUD environments from local registry."""

from __future__ import annotations

from datetime import datetime

import typer
import yaml
from rich.table import Table

from hud.utils.hud_console import HUDConsole

from .utils.registry import extract_name_and_tag, get_registry_dir, list_registry_entries


def format_timestamp(timestamp: float | None) -> str:
    """Format timestamp to human-readable relative time."""
    if not timestamp:
        return "unknown"

    dt = datetime.fromtimestamp(timestamp)
    now = datetime.now()
    delta = now - dt

    # Get total seconds to handle edge cases properly
    total_seconds = delta.total_seconds()

    if total_seconds < 60:
        return "just now"
    elif total_seconds < 3600:
        return f"{int(total_seconds // 60)}m ago"
    elif total_seconds < 86400:  # Less than 24 hours
        return f"{int(total_seconds // 3600)}h ago"
    elif delta.days < 30:
        return f"{delta.days}d ago"
    elif delta.days < 365:
        return f"{delta.days // 30}mo ago"
    else:
        return f"{delta.days // 365}y ago"


def list_environments(
    filter_name: str | None = None,
    json_output: bool = False,
    show_all: bool = False,
    verbose: bool = False,
) -> None:
    """List all HUD environments in the local registry."""
    hud_console = HUDConsole()

    if not json_output:
        hud_console.header("HUD Environment Registry")

    # Check for environment directory
    env_dir = get_registry_dir()
    if not env_dir.exists():
        if json_output:
            print("[]")  # noqa: T201
        else:
            hud_console.info("No environments found in local registry.")
            hud_console.info("")
            hud_console.info("Pull environments with: [cyan]hud pull <org/name:tag>[/cyan]")
            hud_console.info("Build environments with: [cyan]hud build[/cyan]")
        return

    # Collect all environments using the registry helper
    environments = []

    for digest, lock_file in list_registry_entries():
        try:
            # Read lock file
            with open(lock_file) as f:
                lock_data = yaml.safe_load(f)

            # Extract metadata
            image = lock_data.get("image", "unknown")
            name, tag = extract_name_and_tag(image)

            # Apply filter if specified
            if filter_name and filter_name.lower() not in name.lower():
                continue

            # Get additional metadata
            metadata = lock_data.get("metadata", {})
            description = metadata.get("description", "")
            tools_count = len(metadata.get("tools", []))

            # Get file modification time as pulled time
            pulled_time = lock_file.stat().st_mtime

            environments.append(
                {
                    "name": name,
                    "tag": tag,
                    "digest": digest,
                    "description": description,
                    "tools_count": tools_count,
                    "pulled_time": pulled_time,
                    "image": image,
                    "path": str(lock_file),
                }
            )

        except Exception as e:
            if verbose:
                hud_console.warning(f"Failed to read {lock_file}: {e}")

    # Sort by pulled time (newest first)
    environments.sort(key=lambda x: x["pulled_time"], reverse=True)

    if json_output:
        # Output as JSON
        import json

        json_data = [
            {
                "name": env["name"],
                "tag": env["tag"],
                "digest": env["digest"],
                "description": env["description"],
                "tools_count": env["tools_count"],
                "pulled_time": env["pulled_time"],
                "image": env["image"],
                "path": str(env["path"]).replace("\\", "/"),  # Normalize path separators for JSON
            }
            for env in environments
        ]
        print(json.dumps(json_data, indent=2))  # noqa: T201
        return

    if not environments:
        hud_console.info("No environments found matching criteria.")
        hud_console.info("")
        hud_console.info("Pull environments with: [cyan]hud pull <org/name:tag>[/cyan]")
        hud_console.info("Build environments with: [cyan]hud build[/cyan]")
        return

    # Create table
    table = Table(
        title=f"Found {len(environments)} environment{'s' if len(environments) != 1 else ''}"
    )
    table.add_column("Environment", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Tools", justify="right", style="yellow")
    table.add_column("Pulled", style="dim")

    if show_all or verbose:
        table.add_column("Digest", style="dim")

    # Add rows
    for env in environments:
        # Truncate description if too long
        desc = env["description"]
        if desc and len(desc) > 50 and not verbose:
            desc = desc[:47] + "..."

        # Combine name and tag for easy copying
        full_ref = f"{env['name']}:{env['tag']}"

        row = [
            full_ref,
            desc or "[dim]No description[/dim]",
            str(env["tools_count"]),
            format_timestamp(env["pulled_time"]),
        ]

        if show_all or verbose:
            row.append(env["digest"][:12])

        table.add_row(*row)

    hud_console.print(table)  # type: ignore
    hud_console.info("")

    # Show usage hints
    hud_console.section_title("Usage")
    if environments:
        # Use the most recently pulled environment as example
        example_env = environments[0]
        example_ref = f"{example_env['name']}:{example_env['tag']}"

        hud_console.info(f"Run an environment: [cyan]hud run {example_ref}[/cyan]")
        hud_console.info(f"Analyze tools: [cyan]hud analyze {example_ref}[/cyan]")
        hud_console.info(f"Debug server: [cyan]hud debug {example_ref}[/cyan]")

    hud_console.info("Pull more environments: [cyan]hud pull <org/name:tag>[/cyan]")
    hud_console.info("Build new environments: [cyan]hud build[/cyan]")

    if verbose:
        hud_console.info("")
        hud_console.info(f"[dim]Registry location: {env_dir}[/dim]")


def list_command(
    filter_name: str | None = typer.Option(
        None, "--filter", "-f", help="Filter environments by name (case-insensitive)"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    show_all: bool = typer.Option(False, "--all", "-a", help="Show all columns including digest"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """📋 List all HUD environments in local registry.

    Shows environments pulled with 'hud pull' or built with 'hud build',
    stored in ~/.hud/envs/

    Examples:
        hud list                    # List all environments
        hud list --filter text      # Filter by name
        hud list --json            # Output as JSON
        hud list --all             # Show digest column
        hud list --verbose         # Show full descriptions
    """
    list_environments(filter_name, json_output, show_all, verbose)
