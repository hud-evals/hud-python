"""``hud login`` — browser-based login for the HUD CLI.

Implements the OAuth 2.0 Device Authorization Grant (RFC 8628) against the
HUD platform so users can authenticate without copy-pasting an API key.

This is the one pre-credential platform flow, and token polling reads 4xx
error codes as control flow, so it speaks plain httpx rather than
``PlatformClient`` (which requires an API key and raises on any non-2xx).

Use ``--quiet`` / ``-q`` to print the URL instead of opening a browser.
"""

from __future__ import annotations

import socket
import time
import webbrowser
from typing import Any

import httpx
import typer
from rich.panel import Panel
from rich.text import Text

from hud.settings import settings
from hud.utils.hud_console import HUDConsole

from .utils.config import get_user_env_path, set_env_values

DEVICE_CODE_PATH = "/auth/device/code"
DEVICE_TOKEN_PATH = "/auth/device/token"  # noqa: S105 — URL path, not a secret

# RFC 8628 default poll interval, used if the server doesn't send one.
DEFAULT_POLL_INTERVAL = 5


def _api_url() -> str:
    return settings.hud_api_url.rstrip("/")


def _error_code(response: httpx.Response) -> str | None:
    """Pull the RFC 8628 error code out of a non-200 token response.

    The backend wraps it as ``{"detail": {"error": ...}}`` via FastAPI's
    ``HTTPException``; accept the bare RFC shape too.
    """
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    detail = body.get("detail")
    error = (detail if isinstance(detail, dict) else body).get("error")
    return error if isinstance(error, str) else None


def _request_device_code(client: httpx.Client, hud_console: HUDConsole) -> dict[str, Any]:
    """Call ``POST /auth/device/code`` and return the parsed response body."""
    from hud import __version__  # lazy: keeps CLI startup off the full package import

    try:
        response = client.post(
            f"{_api_url()}{DEVICE_CODE_PATH}",
            json={"client_name": socket.gethostname(), "client_version": __version__},
        )
    except httpx.RequestError as exc:
        hud_console.error(f"Failed to reach HUD API: {exc}")
        hud_console.info(f"HUD_API_URL={_api_url()}")
        raise typer.Exit(1) from exc

    if response.status_code != 200:
        hud_console.error(f"HUD API returned {response.status_code} when starting login.")
        hud_console.info(response.text[:500])
        raise typer.Exit(1)
    return response.json()


def _display_login_prompt(
    hud_console: HUDConsole,
    *,
    user_code: str,
    verification_uri: str,
    verification_uri_complete: str,
    quiet: bool,
) -> None:
    """Show the big 'open this URL / enter this code' card."""
    body = Text()
    body.append("Verification code: ", style="dim")
    body.append(f"{user_code}\n\n", style="bold cyan")
    verb = "Open" if quiet else "Opening"
    body.append(f"{verb} this URL in your browser:\n", style="dim")
    body.append(f"  {verification_uri_complete}\n\n")
    body.append("Or visit ", style="dim")
    body.append(verification_uri, style="")
    body.append(" and enter the code manually.", style="dim")

    hud_console.console.print(
        Panel(
            body,
            title="[bold]hud login[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )
    )


def _poll_for_token(
    client: httpx.Client,
    hud_console: HUDConsole,
    *,
    device_code: str,
    interval: int,
    expires_in: int,
) -> dict[str, Any]:
    """Poll ``/auth/device/token`` until success, denial, or expiry."""
    deadline = time.monotonic() + expires_in

    with hud_console.console.status(
        "[cyan]Waiting for confirmation in your browser...[/cyan]",
        spinner="dots",
    ):
        while time.monotonic() < deadline:
            # Sleep first: don't hit the server before the user has had a
            # chance to click "Connect CLI".
            time.sleep(interval)

            try:
                response = client.post(
                    f"{_api_url()}{DEVICE_TOKEN_PATH}",
                    json={"device_code": device_code},
                )
            except httpx.RequestError:
                continue  # transient network error — keep polling

            if response.status_code == 200:
                return response.json()

            error = _error_code(response)
            if error == "authorization_pending" or 500 <= response.status_code < 600:
                continue
            if error == "slow_down":
                interval += 5
                continue
            if error == "expired_token":
                hud_console.error(
                    "Login code expired before you confirmed. Run 'hud login' to try again."
                )
                raise typer.Exit(1)
            if error == "access_denied":
                hud_console.error("Login was denied in the browser.")
                raise typer.Exit(1)

            hud_console.error(f"Unexpected response from HUD API ({response.status_code}).")
            hud_console.info(response.text[:500])
            raise typer.Exit(1)

    hud_console.error("Login timed out. Run 'hud login' to try again.")
    raise typer.Exit(1)


def _persist_api_key(hud_console: HUDConsole, api_key: str) -> None:
    """Write ``HUD_API_KEY`` into ``~/.hud/.env``."""
    try:
        path = set_env_values({"HUD_API_KEY": api_key})
    except OSError as exc:
        hud_console.error(f"Failed to write {get_user_env_path()}: {exc}")
        hud_console.info(f"Set it manually with: hud set HUD_API_KEY={api_key}")
        raise typer.Exit(1) from exc

    hud_console.success("Saved API key to user config")
    hud_console.info(f"Location: {path}")


def login_command(
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Don't try to open a browser; print the verification URL instead.",
    ),
) -> None:
    """Authenticate with the HUD platform.

    [not dim]Opens a browser to hud.ai, confirms your identity, and stores a
    freshly-minted API key in ~/.hud/.env.

    Examples:
        hud login
        hud login --quiet[/not dim]
    """
    hud_console = HUDConsole()

    try:
        with httpx.Client(timeout=30.0) as client:
            device = _request_device_code(client, hud_console)

            user_code = device["user_code"]
            verification_uri = device["verification_uri"]
            verification_uri_complete = (
                device.get("verification_uri_complete")  # optional per RFC 8628
                or f"{verification_uri}?code={user_code}"
            )

            _display_login_prompt(
                hud_console,
                user_code=user_code,
                verification_uri=verification_uri,
                verification_uri_complete=verification_uri_complete,
                quiet=quiet,
            )

            if not quiet:
                webbrowser.open(verification_uri_complete, new=2)

            token = _poll_for_token(
                client,
                hud_console,
                device_code=device["device_code"],
                interval=int(device.get("interval") or DEFAULT_POLL_INTERVAL),
                expires_in=int(device["expires_in"]),
            )
    except KeyboardInterrupt:
        hud_console.info("\nLogin cancelled.")
        raise typer.Exit(130) from None

    api_key = token.get("api_key")
    if not isinstance(api_key, str) or not api_key:
        hud_console.error("HUD API returned a login response without an API key.")
        raise typer.Exit(1)

    _persist_api_key(hud_console, api_key)

    if email := (token.get("user") or {}).get("email"):
        hud_console.info(f"Logged in as {email}")
    if team := (token.get("team") or {}).get("name"):
        hud_console.info(f"Team: {team}")
    hud_console.info("You're all set, try 'hud eval --help'.")
