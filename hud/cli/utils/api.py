"""Shared HUD API helpers: auth, headers, URL construction."""

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
        hud_console.info("Get your key at: https://hud.ai/settings")
        hud_console.info("Set it via: hud set HUD_API_KEY=your-key-here")
        raise typer.Exit(1)
    return settings.api_key


def hud_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return standard auth headers using the current API key.

    Does NOT call require_api_key() — caller decides whether auth is mandatory.
    """
    from hud.settings import settings

    headers: dict[str, str] = {}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"
        headers["X-API-Key"] = settings.api_key
    if extra:
        headers.update(extra)
    return headers
