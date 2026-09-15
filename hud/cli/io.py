"""CLI process boundary: stdout/stderr, --json, and exit codes 0/1/2."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
from typer.core import TyperGroup

from hud.utils.exceptions import HudAuthenticationError, HudRequestError, HudTimeoutError
from hud.utils.hud_console import HUDConsole


class ExitCode:
    SUCCESS = 0
    FAILURE = 1
    USAGE = 2


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

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"error": self.error, "message": self.message}
        if self.input:
            payload["input"] = self.input
        if self.suggestion:
            payload["suggestion"] = self.suggestion
        return payload


def emit_json(payload: Any) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, default=str) + "\n")
    sys.stdout.flush()


def emit_quiet(values: list[Any] | tuple[Any, ...]) -> None:
    sys.stdout.write("".join(f"{value}\n" for value in values))
    sys.stdout.flush()


def emit_error(error: CliError, *, json_output: bool = False) -> None:
    if json_output:
        emit_json(error.to_payload())
        return
    sys.stderr.write(f"Error: {error.message}\n")
    if error.suggestion:
        sys.stderr.write(f"Hint: {error.suggestion}\n")
    sys.stderr.flush()


def mark_json(ctx: typer.Context, value: bool) -> bool:
    # Click shares ctx.meta up the tree, so a parent --json covers child errors too.
    if value:
        ctx.meta["hud_output"] = "json"
    return value or ctx.meta.get("hud_output") == "json"


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


def parse_args(value: str) -> dict[str, Any]:
    return json_object(value, option="--args")


def read_text_arg(path: str) -> str:
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


def confirm_or_abort(message: str, *, yes: bool = False, default: bool = False) -> None:
    if yes:
        return
    if not sys.stdin.isatty():
        raise CliError(
            error="usage",
            message="Confirmation required in a non-interactive terminal.",
            suggestion="Re-run with --yes to continue.",
        )
    console = HUDConsole()
    if not console.confirm(message, default=default):
        console.info("Cancelled.")
        raise typer.Exit(ExitCode.SUCCESS)


def map_request_error(
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
    return CliError(error=kind, message=detail or fallback, input=input, suggestion=hint)


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
        return map_request_error(exc, input=input)
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


class CLIGroup(TyperGroup):
    """Render failures once at the root command boundary."""

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
        try:
            return super().invoke(ctx)
        except (typer.Exit, SystemExit):
            raise
        except Exception as exc:
            error = map_exception(exc)
            json_output = ctx.meta.get("hud_output") == "json"
            if error.exit_code == ExitCode.USAGE and not json_output:
                sys.stderr.write(ctx.get_usage() + "\n")
            emit_error(error, json_output=json_output)
            raise typer.Exit(error.exit_code) from exc
