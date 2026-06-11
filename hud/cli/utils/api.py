"""CLI auth gate for commands that need a HUD API key."""

from __future__ import annotations

import typer

from hud.utils.hud_console import HUDConsole


def require_api_key(action: str = "perform this action") -> str:
    """Check for HUD API key, exit with a helpful message if missing. Returns the key."""
    from hud.settings import settings

    if not settings.api_key:
        hud_console = HUDConsole()
        hud_console.error("No HUD API key found")
        hud_console.info(f"A HUD API key is required to {action}.")
        hud_console.info("Run: hud login")
        hud_console.info("Or get your key at: https://hud.ai/settings")
        hud_console.info("Set it via: hud set HUD_API_KEY=your-key-here")
        raise typer.Exit(1)
    return settings.api_key
