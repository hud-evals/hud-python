"""CLI process: I/O contract, workspace pins, and the Typer entrypoint."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import io
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple
from urllib.parse import urlsplit
from uuid import UUID

import httpx
import typer
from dotenv import dotenv_values, set_key
from packaging.version import parse as parse_version
from pydantic import BaseModel, ConfigDict, model_validator
from typer.core import TyperCommand, TyperGroup, TyperOption

from hud.settings import Settings
from hud.utils.exceptions import HudAuthenticationError, HudRequestError, HudTimeoutError
from hud.utils.hud_console import HUDConsole
from hud.version import __version__

if TYPE_CHECKING:
    from collections.abc import Iterator

    from hud.utils.platform import PlatformClient

CONFIG_PATH = Path(".hud") / "config.json"
_INSTALL_ID_KEY = "HUD_INSTALL_ID"
_FIRST_RUN_NOTICE = "hud collects anonymous CLI usage. Disable: hud set HUD_CLI_ANALYTICS_ENABLED=0"
_VERSION_CACHE = Path(".hud") / ".cache" / "version_check.json"
_VERSION_TTL_S = 6 * 60 * 60
_PYPI = "https://pypi.org/pypi/hud/json"


class AuthScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    origin: str
    user_id: UUID
    team_id: UUID

    @classmethod
    def resolve(cls, platform: PlatformClient) -> AuthScope:
        identity = platform.get("/auth/me")
        url = urlsplit(platform.api_url)
        if url.scheme not in {"http", "https"} or not url.hostname:
            raise ValueError("HUD API URL must be an HTTP origin")
        default_port = 443 if url.scheme == "https" else 80
        port = "" if url.port in (None, default_port) else f":{url.port}"
        return cls(
            origin=f"{url.scheme}://{url.hostname}{port}",
            user_id=identity["user_id"],
            team_id=identity["team_id"],
        )


class DirectoryLink(BaseModel):
    """``.hud/config.json``: platform ids plus the credentials that wrote them."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    scope: AuthScope | None = None
    registry_id: UUID | None = None
    taskset_id: UUID | None = None
    project_id: UUID | None = None

    @model_validator(mode="before")
    @classmethod
    def _ignore_legacy_sync_env(cls, data: Any) -> Any:
        if isinstance(data, dict) and "sync_env" in data:
            return {key: value for key, value in data.items() if key != "sync_env"}
        return data


class DirectoryState:
    def __init__(self, scope: AuthScope, directory: str | Path = ".") -> None:
        self.scope = scope
        self.directory = str(Path(directory).expanduser().resolve())

    @property
    def path(self) -> Path:
        return Path(self.directory) / CONFIG_PATH

    def load(self) -> DirectoryLink:
        stored = self._read()
        return stored if stored is not None else DirectoryLink()

    def update(self, changes: DirectoryLink) -> bool:
        current = self.load()
        updated = DirectoryLink.model_validate(
            {**current.model_dump(), **changes.model_dump(exclude_unset=True), "scope": self.scope}
        )
        if updated == current:
            return False
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".config-")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(updated.model_dump_json(indent=2) + "\n")
                stream.flush()
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return True

    def _read(self) -> DirectoryLink | None:
        path = self.path
        if not path.exists():
            return None
        try:
            stored = DirectoryLink.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise CliError(
                "failure",
                f"{path} is not a valid HUD workspace config.",
                suggestion="Delete the file and run the command again.",
            ) from exc
        if stored.scope != self.scope:
            if stored.scope is not None and stored.scope.origin != self.scope.origin:
                detail = (
                    f"{path} was linked against {stored.scope.origin}, not {self.scope.origin}."
                )
            else:
                detail = (
                    f"{path} was linked with different HUD credentials "
                    "than the ones currently in use."
                )
            raise CliError(
                "failure",
                detail,
                suggestion="Switch credentials, or delete the file and run the command again.",
            )
        return stored


def parse_key_value(item: str) -> tuple[str, str] | None:
    key, sep, value = item.partition("=")
    key = key.strip()
    if not sep or not key:
        return None
    return key, value.strip()


def set_env_values(values: dict[str, str]) -> Path:
    path = Path.home() / ".hud" / ".env"
    path.parent.mkdir(parents=True, exist_ok=True)
    for key, value in values.items():
        set_key(path, key, value)
    return path


class ExitCode:
    SUCCESS = 0
    FAILURE = 1
    USAGE = 2


class Result(NamedTuple):
    """Command payload plus a non-zero process status (JSON still prints first)."""

    payload: Any
    exit_code: int = ExitCode.FAILURE


class CliError(Exception):
    """A CLI failure with a machine-readable type and an exit code."""

    def __init__(
        self,
        error: str,
        message: str,
        *,
        input: dict[str, Any] | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error = error
        self.message = message
        self.input = input
        self.suggestion = suggestion
        self.exit_code = ExitCode.USAGE if error == "usage" else ExitCode.FAILURE

    @classmethod
    def from_http(
        cls,
        exc: HudRequestError,
        *,
        resource: str | None = None,
        input: dict[str, Any] | None = None,
    ) -> CliError:
        status = exc.status_code
        detail = exc.message
        label = resource or "Resource"
        kind, fallback, hint = "failure", str(exc), None
        if status == 404:
            kind, fallback, hint = (
                "not_found",
                f"{label} not found",
                f"Check the {label.lower()} id, or list existing ones.",
            )
        elif status in {401, 403}:
            kind, fallback, hint = (
                "permission_denied",
                "Permission denied",
                "Check that this API key can access the resource.",
            )
        elif status == 409:
            kind, fallback, hint = "conflict", f"{label} already exists", None
        elif status == 429:
            kind, fallback, hint = (
                "rate_limited",
                "Rate limited by the HUD API",
                "Retry after a short delay.",
            )
        elif status is not None and status >= 500:
            kind, fallback, hint = (
                "server_error",
                f"HUD API server error ({status})",
                "Retry; this error is often transient.",
            )
        return cls(error=kind, message=detail or fallback, input=input, suggestion=hint)

    def document(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"error": self.error, "message": self.message}
        if self.input:
            payload["input"] = self.input
        if self.suggestion:
            payload["suggestion"] = self.suggestion
        return payload


def map_exception(exc: BaseException, *, input: dict[str, Any] | None = None) -> CliError:
    if isinstance(exc, CliError):
        return exc
    if getattr(exc, "exit_code", None) == ExitCode.USAGE:
        return CliError(error="usage", message=str(exc), input=input)
    if isinstance(exc, ValueError):
        return CliError(error="usage", message=str(exc), input=input)
    if isinstance(exc, FileNotFoundError):
        return CliError(error="not_found", message=str(exc), input=input)
    if isinstance(exc, HudRequestError):
        return CliError.from_http(exc, input=input)
    if isinstance(exc, HudAuthenticationError):
        return CliError(
            error="permission_denied",
            message=str(exc) or "Missing or invalid HUD API key",
            input=input,
            suggestion="Run 'hud set HUD_API_KEY=your-key-here'.",
        )
    if isinstance(exc, HudTimeoutError):
        return CliError(
            error="timeout",
            message=str(exc) or "Timed out talking to the HUD API",
            input=input,
            suggestion="Retry; the failure may be transient. Increase --timeout if set.",
        )
    return CliError(error="failure", message=str(exc), input=input)


def _add_json_option(command: Any) -> None:
    if any("--json" in getattr(param, "opts", ()) for param in command.params):
        return
    command.params.append(
        TyperOption(
            param_decls=["--json"],
            is_flag=True,
            is_eager=True,
            expose_value=False,
            help="Write JSON to stdout.",
        )
    )


class CLICommand(TyperCommand):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _add_json_option(self)


class CLIGroup(TyperGroup):
    # Typer emits leaf commands before groups; this is the public help order.
    command_order = (
        "init",
        "serve",
        "deploy",
        "eval",
        "task",
        "project",
        "sync",
        "qa",
        "jobs",
        "job",
        "cancel",
        "trace",
        "models",
        "set",
        "version",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.name != "hud":
            _add_json_option(self)

    def list_commands(self, ctx: Any) -> list[str]:
        names = super().list_commands(ctx)
        if ctx is not None and ctx.parent is not None:
            return names
        rank = {name: index for index, name in enumerate(self.command_order)}
        return sorted(names, key=lambda name: (rank.get(name, len(rank)), name))

    def get_help_option_names(self, ctx: Any) -> list[str]:
        if ctx.parent is None:
            return []
        return super().get_help_option_names(ctx) or ["--help"]

    def get_help_option(self, ctx: Any) -> Any:
        option = super().get_help_option(ctx)
        if option is not None:
            option.help = "Show help."
        return option

    def collect_usage_pieces(self, ctx: Any) -> list[str]:
        if ctx.parent is None:
            return ["COMMAND"]
        return super().collect_usage_pieces(ctx)

    def invoke(self, ctx: Any) -> Any:
        if ctx.parent is not None:
            return super().invoke(ctx)
        tokens = (*ctx.args, *getattr(ctx, "_protected_args", ()))
        json_output = "--json" in tokens and "--help" not in tokens and "-h" not in tokens
        try:
            with (
                contextlib.redirect_stdout(io.StringIO())
                if json_output
                else contextlib.nullcontext()
            ):
                result = super().invoke(ctx)
                if inspect.isawaitable(result):
                    result = asyncio.run(result)
            payload, code = (
                (result.payload, result.exit_code) if isinstance(result, Result) else (result, 0)
            )
            if json_output and payload is not None:
                sys.stdout.write(json.dumps(payload, indent=2, default=str) + "\n")
                sys.stdout.flush()
            if code:
                raise typer.Exit(code)
            return payload
        except (typer.Exit, SystemExit):
            raise
        except Exception as exc:
            error = map_exception(exc)
            if json_output:
                sys.stdout.write(json.dumps(error.document(), indent=2, default=str) + "\n")
                sys.stdout.flush()
            else:
                if error.exit_code == ExitCode.USAGE:
                    sys.stderr.write(ctx.get_usage() + "\n")
                sys.stderr.write(f"Error: {error.message}\n")
                if error.suggestion:
                    sys.stderr.write(f"Hint: {error.suggestion}\n")
                sys.stderr.flush()
            raise typer.Exit(error.exit_code) from exc


class CLI(typer.Typer):
    def __init__(self, *args: Any, cls: type[TyperGroup] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, cls=cls or CLIGroup, **kwargs)

    def command(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["cls"] = kwargs.get("cls") or CLICommand
        return super().command(*args, **kwargs)

    @staticmethod
    def json_object(value: str, *, option: str) -> dict[str, Any]:
        key = option.lstrip("-")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise CliError(
                error="usage",
                message=f"{option} must be valid JSON: {exc}",
                input={key: value},
                suggestion=f'Pass a JSON object, e.g. {option} \'{{"key": "value"}}\'.',
            ) from exc
        if isinstance(parsed, dict):
            return parsed
        raise CliError(error="usage", message=f"{option} must be a JSON object", input={key: value})

    @staticmethod
    def read_text(path: str) -> str:
        if path == "-":
            return sys.stdin.read()
        try:
            return Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            raise CliError(
                error="not_found",
                message=f"File not found: {path}",
                input={"path": path},
                suggestion="Check the path, or pass - to read from stdin.",
            ) from None

    @staticmethod
    def confirm_or_abort(message: str, *, yes: bool = False, default: bool = False) -> None:
        if yes:
            return
        if not sys.stdin.isatty():
            raise CliError(
                error="usage",
                message="Confirmation required in a non-interactive terminal.",
                suggestion="Re-run with --yes to continue.",
            )
        hud_console = HUDConsole()
        if not hud_console.confirm(message, default=default):
            hud_console.info("Cancelled.")
            raise typer.Exit(ExitCode.SUCCESS)


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
        latest: str | None = None
        try:
            data = json.loads(cache.read_text())
            if time.time() - data["checked_at"] <= _VERSION_TTL_S:
                latest = data["latest"]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
        fetched = latest is None
        if latest is None:
            latest = httpx.get(_PYPI, timeout=httpx.Timeout(1.0, connect=0.5)).json()["info"][
                "version"
            ]
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


from .deploy import deploy_command  # noqa: E402
from .eval import eval_command  # noqa: E402
from .init import init_command  # noqa: E402
from .jobs import cancel_job_command, jobs_app  # noqa: E402
from .models import models_app  # noqa: E402
from .project import project_app  # noqa: E402
from .qa import qa_app  # noqa: E402
from .serve import serve_command  # noqa: E402
from .sync import sync_app  # noqa: E402
from .task import task_app  # noqa: E402
from .trace import trace_app  # noqa: E402

app.command(name="init")(init_command)
app.command(name="serve")(serve_command)
app.command(name="deploy")(deploy_command)
app.command(name="eval")(eval_command)
app.add_typer(task_app, name="task")
app.add_typer(project_app, name="project")
app.add_typer(sync_app, name="sync")
app.add_typer(qa_app, name="qa")
app.add_typer(jobs_app, name="jobs")
app.add_typer(jobs_app, name="job", hidden=True)
app.command(name="cancel", hidden=True, deprecated=True)(cancel_job_command)
app.add_typer(trace_app, name="trace")
app.add_typer(models_app, name="models")
app.command(name="set")(set_command)
app.command()(version)
