"""``hud login`` — Browser-based login for the HUD CLI.

Implements the OAuth 2.0 Device Authorization Grant (RFC 8628) against the
HUD platform so users can authenticate without copy-pasting an API key:

Use ``--quiet`` / ``-q`` to print the URL instead of opening a browser
"""

from __future__ import annotations

import contextlib
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

# Fallback poll interval if the server doesn't send one.
DEFAULT_POLL_INTERVAL = 5
# Max wall-clock time to wait for the user to confirm.
DEFAULT_EXPIRES_IN = 600


def _client_name() -> str:
    try:
        return socket.gethostname() or "unknown host"
    except Exception:
        return "unknown host"


def _client_version() -> str:
    try:
        from hud import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def _api_base_url() -> str:
    return settings.hud_api_url.rstrip("/")


def _fallback_web_url() -> str:
    return settings.hud_web_url.rstrip("/")


def _extract_error_code(response: httpx.Response) -> str | None:
    """Pull RFC 8628 error codes out of a non-200 response.

    The backend wraps errors as ``{"detail": {"error": "authorization_pending"}}``
    via FastAPI's ``HTTPException``. Be defensive in case the shape changes.
    """
    try:
        body: Any = response.json()
    except Exception:
        return None
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict) and isinstance(detail.get("error"), str):
            return detail["error"]
        if isinstance(body.get("error"), str):
            return body["error"]
    return None


def _request_device_code(client: httpx.Client, hud_console: HUDConsole) -> dict[str, Any]:
    """Call ``POST /auth/device/code`` and return the parsed response body."""
    try:
        response = client.post(
            f"{_api_base_url()}{DEVICE_CODE_PATH}",
            json={
                "client_name": _client_name(),
                "client_version": _client_version(),
            },
            timeout=30.0,
        )
    except httpx.RequestError as exc:
        hud_console.error(f"Failed to reach HUD API: {exc}")
        hud_console.info(f"HUD_API_URL={_api_base_url()}")
        raise typer.Exit(1) from exc

    if response.status_code != 200:
        hud_console.error(f"HUD API returned {response.status_code} when starting login.")
        with contextlib.suppress(Exception):
            hud_console.info(response.text[:500])
        raise typer.Exit(1)

    try:
        return response.json()
    except Exception as exc:
        hud_console.error("HUD API returned an invalid response.")
        raise typer.Exit(1) from exc


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
    if quiet:
        body.append("Open this URL in your browser:\n", style="dim")
    else:
        body.append("Opening this URL in your browser:\n", style="dim")
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
    """Poll ``/auth/device/token`` until success, timeout, or fatal error."""
    deadline = time.monotonic() + max(expires_in, 30)
    current_interval = max(interval, 1)

    with hud_console.console.status(
        "[cyan]Waiting for confirmation in your browser...[/cyan]",
        spinner="dots",
    ):
        while time.monotonic() < deadline:
            # Sleep first so we don't hammer the server on the initial tick
            # before the user has had a chance to click "Connect CLI".
            time.sleep(current_interval)

            try:
                response = client.post(
                    f"{_api_base_url()}{DEVICE_TOKEN_PATH}",
                    json={"device_code": device_code},
                    timeout=30.0,
                )
            except httpx.RequestError:
                # Transient network error — keep polling.
                continue

            if response.status_code == 200:
                try:
                    return response.json()
                except Exception as exc:  # pragma: no cover — server misbehaving
                    hud_console.error("HUD API returned an invalid token response.")
                    raise typer.Exit(1) from exc

            error = _extract_error_code(response)
            if error == "authorization_pending":
                continue
            if error == "slow_down":
                current_interval += 5
                continue
            if error == "expired_token":
                hud_console.error(
                    "Login code expired before you confirmed. Run 'hud login' to try again."
                )
                raise typer.Exit(1)
            if error == "access_denied":
                hud_console.error("Login was denied in the browser.")
                raise typer.Exit(1)

            # Unknown 4xx/5xx — treat as transient unless it's an obvious fatal.
            if 500 <= response.status_code < 600:
                continue

            hud_console.error(f"Unexpected response from HUD API ({response.status_code}).")
            with contextlib.suppress(Exception):
                hud_console.info(response.text[:500])
            raise typer.Exit(1)

    hud_console.error("Login timed out. Run 'hud login' to try again.")
    raise typer.Exit(1)


def _persist_api_key(hud_console: HUDConsole, api_key: str) -> None:
    """Write ``HUD_API_KEY`` into ``~/.hud/.env``."""
    try:
        path = set_env_values({"HUD_API_KEY": api_key})
    except Exception as exc:
        hud_console.error(f"Failed to write {get_user_env_path()}: {exc}")
        hud_console.info("You can set the key manually with:")
        hud_console.info(f"  hud set HUD_API_KEY={api_key}")
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

    # -- Device authorization grant -----------------------------------------
    try:
        with httpx.Client() as client:
            device = _request_device_code(client, hud_console)

            device_code = device["device_code"]
            user_code = device["user_code"]
            verification_uri = device.get("verification_uri") or f"{_fallback_web_url()}/device"
            verification_uri_complete = (
                device.get("verification_uri_complete")
                or f"{_fallback_web_url()}/device?code={user_code}"
            )
            interval = int(device.get("interval") or DEFAULT_POLL_INTERVAL)
            expires_in = int(device.get("expires_in") or DEFAULT_EXPIRES_IN)

            _display_login_prompt(
                hud_console,
                user_code=user_code,
                verification_uri=verification_uri,
                verification_uri_complete=verification_uri_complete,
                quiet=quiet,
            )

            if not quiet:
                with contextlib.suppress(Exception):
                    webbrowser.open(verification_uri_complete, new=2)

            token = _poll_for_token(
                client,
                hud_console,
                device_code=device_code,
                interval=interval,
                expires_in=expires_in,
            )
    except KeyboardInterrupt:
        hud_console.info("\nLogin cancelled.")
        raise typer.Exit(130) from None

    # -- Persist and report -------------------------------------------------
    key = token.get("api_key")
    if not isinstance(key, str) or not key:
        hud_console.error("HUD API returned a login response without an API key.")
        raise typer.Exit(1)

    _persist_api_key(hud_console, key)

    user_info = token.get("user") or {}
    team_info = token.get("team") or {}
    user_email = user_info.get("email") if isinstance(user_info, dict) else None
    team_name = team_info.get("name") if isinstance(team_info, dict) else None

    if user_email:
        hud_console.info(f"Logged in as {user_email}")
    if team_name:
        hud_console.info(f"Team: {team_name}")
    hud_console.info("You're all set, try 'hud eval --help'.")
