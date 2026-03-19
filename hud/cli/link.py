"""Link local directory to existing HUD environment (deprecated).

Use ``hud sync env`` instead.
"""

from __future__ import annotations

import typer


def link_command(
    directory: str = typer.Argument(".", help="Directory to link"),
    registry_id: str | None = typer.Option(
        None,
        "--id",
        "-i",
        help="Environment ID to link to (prompts if not provided)",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts",
    ),
) -> None:
    """Link directory to existing HUD environment (deprecated).

    [not dim]Deprecated: Use 'hud sync env' instead.

    Examples:
        hud sync env my-env         # Link by name
        hud sync env                # Interactive selection[/not dim]
    """
    from hud.cli.sync import sync_env_command
    from hud.utils.hud_console import HUDConsole as _HC

    _HC().warning("'hud link' is deprecated. Use 'hud sync env' instead.")
    sync_env_command(name=registry_id, directory=directory, yes=yes)
