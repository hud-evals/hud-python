"""CLI auth gate for commands that need a HUD API key."""

from __future__ import annotations

from hud.cli.utils.output import CliError, abort


def missing_api_key_error(action: str = "perform this action") -> CliError | None:
    """Return a structured error when no HUD API key is configured."""
    from hud.settings import settings

    if settings.api_key:
        return None
    return CliError(
        error="permission_denied",
        message="No HUD API key found",
        input={"action": action},
        suggestion=(
            f"A HUD API key is required to {action}. "
            "Run 'hud login' or 'hud set HUD_API_KEY=your-key-here'. "
            f"Get a key at: {settings.hud_web_url}/settings"
        ),
    )


def require_api_key(action: str = "perform this action") -> str:
    """Check for HUD API key, exit with a helpful message if missing. Returns the key."""
    from hud.settings import settings

    error = missing_api_key_error(action)
    if error is not None:
        abort(error)
    return settings.api_key
