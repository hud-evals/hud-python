"""Anonymous CLI usage events: command name, outcome, and environment facts.

The payload is a strict allowlist — never arguments, paths, error messages, or
env values, which can carry secrets. A token from argv is recorded only when
it names a command registered on the CLI itself, so positional user input
(task files, trace ids) can never be transmitted. ``HUD_TELEMETRY_ENABLED=0``
disables sending.
"""

from __future__ import annotations

import contextlib
import functools
import os
import sys
import threading
import time
import uuid
from typing import TYPE_CHECKING

from .config import load_env_file, set_env_values

if TYPE_CHECKING:
    from collections.abc import Iterator

_INSTALL_ID_KEY = "HUD_INSTALL_ID"
_JOIN_TIMEOUT_S = 1.5

_CI_ENV_VARS = (
    "CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "BUILDKITE",
    "CIRCLECI",
    "JENKINS_URL",
    "TF_BUILD",
)

_FIRST_RUN_NOTICE = (
    "hud collects anonymous usage data (command names and outcomes; never "
    "arguments, file contents, or keys). Disable with: hud set HUD_TELEMETRY_ENABLED=0\n"
    "Details: https://docs.hud.ai/v6/reference/telemetry"
)


def _is_ci() -> bool:
    return any(os.environ.get(var) for var in _CI_ENV_VARS)


def _install_id() -> str:
    """The persistent anonymous install id, created (with a notice) on first use."""
    env = load_env_file()
    existing = env.get(_INSTALL_ID_KEY, "")
    try:
        return str(uuid.UUID(existing))
    except ValueError:
        pass
    created = str(uuid.uuid4())
    set_env_values({_INSTALL_ID_KEY: created})
    sys.stderr.write(_FIRST_RUN_NOTICE + "\n")
    return created


@functools.cache
def _registered_commands() -> dict[str, frozenset[str]]:
    """Registered top-level command names mapped to their subcommand names.

    Introspected from the CLI itself so the allowlist cannot drift from the
    real command tree; groups invoked via callback (``hud trace <id>``) have no
    subcommands, so their positional arguments never match.
    """
    try:
        import typer

        from hud.cli import app

        group = typer.main.get_group(app)
        # Duck-typed on ``commands``: isinstance(click.Group) breaks when
        # tests mock or reload click.
        return {
            name: frozenset(getattr(command, "commands", {}))
            for name, command in group.commands.items()
        }
    except Exception:  # conservative: unknown registry records nothing specific
        return {}


def _command_tokens(argv: list[str]) -> tuple[str, str | None]:
    words = [arg for arg in argv[1:] if not arg.startswith("-")]
    if not words:
        return "help", None
    registry = _registered_commands()
    if words[0] not in registry:
        return "other", None
    command = words[0]
    subcommand = words[1] if len(words) > 1 and words[1] in registry[command] else None
    return command, subcommand


def _classify(error: BaseException) -> tuple[int, str | None]:
    """Map a propagating error to (exit_code, error_class).

    ``typer.Exit`` carries ``exit_code`` and ``SystemExit`` carries ``code``;
    when either was raised ``from`` an original error (the CLI converts
    ``HudException`` this way), the cause names the real error class.
    """
    if isinstance(error, KeyboardInterrupt):
        return 130, "KeyboardInterrupt"
    exit_code = getattr(error, "exit_code", None)
    if exit_code is None and isinstance(error, SystemExit):
        exit_code = error.code if isinstance(error.code, int) else 1
    if isinstance(exit_code, int):
        cause = error.__cause__
        return exit_code, type(cause).__name__ if cause is not None else None
    return 1, type(error).__name__


def _post(url: str, payload: dict[str, object]) -> None:
    import httpx  # lazy: keeps CLI startup light

    # Telemetry must never surface a failure.
    with contextlib.suppress(Exception):
        httpx.post(url, json=payload, timeout=httpx.Timeout(1.0, connect=0.5))


def record_invocation(
    argv: list[str],
    *,
    exit_code: int,
    error_class: str | None,
    duration_ms: int,
) -> threading.Thread | None:
    """Send one usage event in a background thread; returns it for a bounded join.

    Returns ``None`` (and sends nothing) when telemetry is disabled.
    """
    try:
        from hud.settings import Settings

        settings = Settings()
        if not settings.telemetry_enabled:
            return None
        from hud import __version__

        command, subcommand = _command_tokens(argv)
        payload = {
            "events": [
                {
                    "command": command,
                    "subcommand": subcommand,
                    "exit_code": exit_code,
                    "error_class": error_class,
                    "duration_ms": duration_ms,
                    "cli_version": __version__,
                    "python_version": ".".join(map(str, sys.version_info[:3])),
                    "os": sys.platform if sys.platform in ("linux", "darwin", "win32") else "other",
                    "is_ci": _is_ci(),
                    "install_id": _install_id(),
                }
            ]
        }
        url = f"{settings.hud_telemetry_url.rstrip('/')}/v2/sdk-events/cli"
        thread = threading.Thread(target=_post, args=(url, payload), daemon=True)
        thread.start()
    except Exception:  # telemetry must never break a command
        return None
    return thread


@contextlib.contextmanager
def recorded_invocation(argv: list[str]) -> Iterator[None]:
    """Record one CLI invocation around the wrapped block, then re-raise as-is."""
    started = time.monotonic()
    exit_code = 0
    error_class: str | None = None
    try:
        yield
    except BaseException as error:
        exit_code, error_class = _classify(error)
        raise
    finally:
        sender = record_invocation(
            argv,
            exit_code=exit_code,
            error_class=error_class,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        if sender is not None:
            sender.join(timeout=_JOIN_TIMEOUT_S)
