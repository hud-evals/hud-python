"""Anonymous CLI usage events: command name, outcome, and environment facts."""

from __future__ import annotations

import contextlib
import os
import sys
import time
import uuid
from typing import TYPE_CHECKING

import typer

from hud.cli.config import load_env_file, set_env_values
from hud.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Iterator

_INSTALL_ID_KEY = "HUD_INSTALL_ID"

_FIRST_RUN_NOTICE = "hud collects anonymous CLI usage. Disable: hud set HUD_CLI_ANALYTICS_ENABLED=0"


@contextlib.contextmanager
def recorded_invocation(argv: list[str], app: typer.Typer) -> Iterator[None]:
    """Record one CLI invocation around the wrapped block, then re-raise as-is."""
    started = time.monotonic()
    exit_code = 0
    error_class: str | None = None
    try:
        yield
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            exit_code, error_class = 130, "KeyboardInterrupt"
        else:
            exit_code = getattr(error, "exit_code", None)
            if exit_code is None and isinstance(error, SystemExit):
                exit_code = error.code if isinstance(error.code, int) else 1
            if isinstance(exit_code, int):
                cause = error.__cause__
                error_class = type(cause).__name__ if cause is not None else None
            else:
                exit_code, error_class = 1, type(error).__name__
        raise
    finally:
        with contextlib.suppress(Exception):
            settings = Settings()
            if settings.cli_analytics_enabled:
                words = [arg for arg in argv[1:] if not arg.startswith("-")]
                if not words:
                    flags = {arg.split("=", 1)[0] for arg in argv[1:] if arg.startswith("-")}
                    tokens: tuple[str, str | None] | None = (
                        None if "--version" in flags else ("help", None)
                    )
                else:
                    registry = {
                        name: frozenset(getattr(command, "commands", {}))
                        for name, command in typer.main.get_group(app).commands.items()
                    }
                    if words[0] not in registry:
                        tokens = ("other", None)
                    else:
                        command = words[0]
                        subcommand = (
                            words[1] if len(words) > 1 and words[1] in registry[command] else None
                        )
                        tokens = (command, subcommand)
                if tokens is not None:
                    from hud import __version__

                    existing = load_env_file().get(_INSTALL_ID_KEY, "")
                    try:
                        install_id = str(uuid.UUID(existing))
                    except ValueError:
                        install_id = str(uuid.uuid4())
                        set_env_values({_INSTALL_ID_KEY: install_id})
                        sys.stderr.write(_FIRST_RUN_NOTICE + "\n")
                    command, subcommand = tokens
                    payload = {
                        "events": [
                            {
                                "command": command,
                                "subcommand": subcommand,
                                "exit_code": exit_code,
                                "error_class": error_class,
                                "duration_ms": int((time.monotonic() - started) * 1000),
                                "cli_version": __version__,
                                "python_version": ".".join(map(str, sys.version_info[:3])),
                                "os": (
                                    sys.platform
                                    if sys.platform in ("linux", "darwin", "win32")
                                    else "other"
                                ),
                                "is_ci": "CI" in os.environ,
                                "install_id": install_id,
                            }
                        ]
                    }
                    import httpx

                    httpx.post(
                        f"{settings.hud_telemetry_url.rstrip('/')}/sdk-events/cli",
                        json=payload,
                        timeout=httpx.Timeout(1.0, connect=0.5),
                    )
