"""CLI infrastructure: the I/O contract, error mapping, and workspace pins.

Command modules build on this; :mod:`hud.cli.__main__` assembles them.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple
from urllib.parse import urlsplit
from uuid import UUID

import typer
from dotenv import set_key
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from typer.core import TyperCommand, TyperGroup, TyperOption

from hud.utils.exceptions import HudAuthenticationError, HudRequestError, HudTimeoutError
from hud.utils.hud_console import HUDConsole

if TYPE_CHECKING:
    from hud.utils.platform import PlatformClient

CONFIG_PATH = Path(".hud") / "config.json"


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
    """``.hud/config.json``: platform ids plus the credentials that wrote them.

    Releases before the scoped schema wrote camelCase ids (``registryId``,
    ``tasksetId``, ``projectId``) alongside keys no longer kept (``registryName``,
    ``syncEnv``); those files read as an unscoped link and are rewritten in this
    schema on the next update.
    """

    model_config = ConfigDict(extra="ignore")

    version: Literal[1] = 1
    scope: AuthScope | None = None
    registry_id: UUID | None = Field(
        default=None, validation_alias=AliasChoices("registry_id", "registryId")
    )
    taskset_id: UUID | None = Field(
        default=None, validation_alias=AliasChoices("taskset_id", "tasksetId")
    )
    project_id: UUID | None = Field(
        default=None, validation_alias=AliasChoices("project_id", "projectId")
    )


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
        if stored.scope is not None and stored.scope != self.scope:
            if stored.scope.origin != self.scope.origin:
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
    # Typer vendors Click, so its UsageError (unknown command, bad option) is not
    # click.UsageError; the exit code is the stable signal.
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
        "import",
        "qa",
        "jobs",
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
