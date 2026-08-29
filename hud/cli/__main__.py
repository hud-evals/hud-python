"""The ``hud`` command: the assembled tree and its entrypoint (``python -m hud.cli``)."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import typer
from dotenv import dotenv_values
from packaging.version import parse as parse_version

from hud.cli import CLI, CliError, parse_key_value, set_env_values
from hud.cli.deploy import deploy_command
from hud.cli.eval import eval_command
from hud.cli.imports import import_app
from hud.cli.init import init_command
from hud.cli.jobs import cancel_job_command, jobs_app
from hud.cli.models import models_app
from hud.cli.project import project_app
from hud.cli.qa import qa_app
from hud.cli.serve import serve_command
from hud.cli.sync import sync_app
from hud.cli.task import task_app
from hud.cli.trace import trace_app
from hud.settings import Settings
from hud.utils.hud_console import HUDConsole
from hud.version import __version__

if TYPE_CHECKING:
    from collections.abc import Iterator

_INSTALL_ID_KEY = "HUD_INSTALL_ID"
_FIRST_RUN_NOTICE = "hud collects anonymous CLI usage. Disable: hud set HUD_CLI_ANALYTICS_ENABLED=0"
_VERSION_CACHE = Path(".hud") / ".cache" / "version_check.json"
_VERSION_TTL_S = 6 * 60 * 60
_PYPI = "https://pypi.org/pypi/hud/json"


app = CLI(
    name="hud",
    help="Build, test, and deploy HUD environments.",
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
)


def set_command(
    assignments: list[str] = typer.Argument(  # noqa: B008
        ..., help="One or more KEY=VALUE pairs to persist in ~/.hud/.env"
    ),
) -> dict[str, object]:
    """Persist API keys or other variables for HUD to use by default.

    [not dim]Examples:
        hud set ANTHROPIC_API_KEY=sk-... OPENAI_API_KEY=sk-...
        hud set HUD_API_KEY=sk-... --json

    Values are stored in ~/.hud/.env and are loaded by hud.settings with
    the lowest precedence (overridden by process env and project .env).[/not dim]
    """
    hud_console = HUDConsole()

    updates: dict[str, str] = {}
    for item in assignments:
        parsed = parse_key_value(item)
        if parsed is None:
            raise CliError(
                error="usage",
                message=f"Invalid assignment (expected KEY=VALUE): {item}",
                input={"assignment": item},
                suggestion="Pass one or more KEY=VALUE pairs.",
            )
        key, value = parsed
        updates[key] = value

    result = {"path": str(set_env_values(updates)), "keys": list(updates)}
    hud_console.success("Saved credentials to user config")
    hud_console.info(f"Location: {result['path']}")
    hud_console.info(f"Keys: {', '.join(str(key) for key in result['keys'])}")
    return result


def version() -> dict[str, str]:
    """Show HUD CLI version.

    [not dim]Examples:
        hud version
        hud version --json[/not dim]
    """
    result = {"name": "hud", "version": __version__}
    HUDConsole().print(f"HUD CLI version: [cyan]{result['version']}[/cyan]", stderr=False)
    return result


@app.callback(invoke_without_command=True)
def root_command(
    ctx: typer.Context,
    show_help: bool = typer.Option(False, "--help", help="Show help."),
    show_version: bool = typer.Option(False, "--version", help="Show version."),
) -> None:
    if show_help:
        typer.echo(ctx.get_help())
        raise typer.Exit
    if show_version:
        version()
        raise typer.Exit
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(2)


@contextlib.contextmanager
def recorded_invocation(argv: list[str], cli: typer.Typer) -> Iterator[None]:
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
                        for name, command in typer.main.get_group(cli).commands.items()
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
                    existing = (
                        dotenv_values(Path.home() / ".hud" / ".env", interpolate=False).get(
                            _INSTALL_ID_KEY
                        )
                        or ""
                    )
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
                    httpx.post(
                        f"{settings.hud_telemetry_url.rstrip('/')}/sdk-events/cli",
                        json=payload,
                        timeout=httpx.Timeout(1.0, connect=0.5),
                    )


def notify_if_outdated(argv: list[str]) -> None:
    """Print an upgrade hint when the installed hud is behind PyPI."""
    if "CI" in os.environ or os.environ.get("HUD_SKIP_VERSION_CHECK"):
        return
    if not any(not arg.startswith("-") for arg in argv[1:]):
        return
    with contextlib.suppress(Exception):
        cache = Path.home() / _VERSION_CACHE
        cached: dict[str, Any] = (
            json.loads(cache.read_text()) if cache.exists() else {"checked_at": 0.0}
        )
        fetched = time.time() - cached["checked_at"] > _VERSION_TTL_S
        latest = (
            httpx.get(_PYPI, timeout=httpx.Timeout(1.0, connect=0.5)).json()["info"]["version"]
            if fetched
            else cached["latest"]
        )
        if parse_version(latest) > parse_version(__version__):
            tool_install = "uv/tools/" in sys.prefix.replace("\\", "/")
            in_project_venv = not tool_install and (
                sys.prefix != sys.base_prefix or "VIRTUAL_ENV" in os.environ
            )
            upgrade = "uv sync --upgrade-package hud" if in_project_venv else "uv tool upgrade hud"
            sys.stderr.write(
                f"A new version of hud is available: {latest} (current: {__version__})\n"
                f"Run: {upgrade}\n"
            )
        if fetched:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"latest": latest, "checked_at": time.time()}))


def main() -> None:
    """Main entry point for the CLI."""
    # Windows cmd.exe uses the system code page (e.g. cp1252) which can't
    # encode the emoji that Rich uses. Rewrap stdout/stderr as UTF-8 so
    # Rich's legacy Windows renderer never hits a charmap error.
    if sys.platform == "win32":
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "buffer"):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    notify_if_outdated(sys.argv)

    with recorded_invocation(sys.argv, app):
        app()


app.command(name="init")(init_command)
app.command(name="serve")(serve_command)
app.command(name="deploy")(deploy_command)
app.command(name="eval")(eval_command)
app.add_typer(task_app, name="task")
app.add_typer(project_app, name="project")
app.add_typer(sync_app, name="sync")
app.add_typer(import_app, name="import")
app.add_typer(qa_app, name="qa")
app.add_typer(jobs_app, name="jobs")
app.command(name="cancel", hidden=True, deprecated=True)(cancel_job_command)
app.add_typer(trace_app, name="trace")
app.add_typer(models_app, name="models")
app.command(name="set")(set_command)
app.command()(version)

if __name__ == "__main__":
    main()
