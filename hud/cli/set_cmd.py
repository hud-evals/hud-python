"""Persist API keys and configuration variables."""

from __future__ import annotations

import typer

from hud.utils.hud_console import HUDConsole

from .utils.config import set_env_values


def set_command(
    assignments: list[str] = typer.Argument(  # type: ignore[arg-type]  # noqa: B008
        ..., help="One or more KEY=VALUE pairs to persist in ~/.hud/.env"
    ),
) -> None:
    """Persist API keys or other variables for HUD to use by default.

    [not dim]Examples:
        hud set ANTHROPIC_API_KEY=sk-... OPENAI_API_KEY=sk-...

    Values are stored in ~/.hud/.env and are loaded by hud.settings with
    the lowest precedence (overridden by process env and project .env).[/not dim]
    """
    hud_console = HUDConsole()

    updates: dict[str, str] = {}
    for item in assignments:
        if "=" not in item:
            hud_console.error(f"Invalid assignment (expected KEY=VALUE): {item}")
            raise typer.Exit(1)
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            hud_console.error(f"Invalid key in assignment: {item}")
            raise typer.Exit(1)
        updates[key] = value

    path = set_env_values(updates)
    hud_console.success("Saved credentials to user config")
    hud_console.info(f"Location: {path}")
