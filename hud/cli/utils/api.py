"""Shared HUD API helpers: auth, headers, URL construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from hud.utils.hud_console import HUDConsole

if TYPE_CHECKING:
    import httpx


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


def hud_client(
    *, timeout: float = 30.0, extra_headers: dict[str, str] | None = None
) -> httpx.Client:
    """Return an ``httpx.Client`` preconfigured with HUD auth headers.

    Centralizes the ``httpx.Client(headers=hud_headers(...), timeout=...)`` setup
    that CLI commands otherwise repeat. Use as a context manager.
    """
    import httpx

    return httpx.Client(timeout=timeout, headers=hud_headers(extra_headers))


def response_detail(response: httpx.Response) -> str:
    """Best-effort error detail from a HUD API response: JSON ``detail`` else raw text."""
    try:
        return response.json().get("detail", "No detail available")
    except Exception:
        return response.text
