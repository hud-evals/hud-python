"""Agent-friendly CLI I/O: JSON stdout, structured errors, and exit codes.

Stdout is the machine contract (JSON, quiet ids, or human tables). Progress,
warnings, prompts, and error text go to stderr.

Exit codes:
    0  success
    1  general failure
    2  usage (bad arguments; also Click/Typer's default)
    3  resource not found
    4  permission denied
    5  conflict (resource already exists)
"""

from __future__ import annotations

import contextvars
import json
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from collections.abc import Iterator

import click
import typer
from typer.core import TyperGroup

_JSON_REQUESTED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "hud_cli_json_requested", default=False
)
_JSON_SUPPRESSED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "hud_cli_json_suppressed", default=False
)

# ── exit codes ───────────────────────────────────────────────────────────────


class ExitCode:
    SUCCESS = 0
    FAILURE = 1
    USAGE = 2
    NOT_FOUND = 3
    PERMISSION = 4
    CONFLICT = 5


# ── error model ──────────────────────────────────────────────────────────────


class CliError(Exception):
    """A CLI failure with a stable machine-readable type and exit code."""

    def __init__(
        self,
        error: str,
        message: str,
        *,
        input: dict[str, Any] | None = None,
        suggestion: str | None = None,
        transient: bool = False,
        existing_id: str | None = None,
        exit_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error = error
        self.message = message
        self.input = input
        self.suggestion = suggestion
        self.transient = transient
        self.existing_id = existing_id
        self.exit_code = exit_code if exit_code is not None else _exit_code_for(error)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": self.error,
            "message": self.message,
            "transient": self.transient,
        }
        if self.input:
            payload["input"] = self.input
        if self.suggestion:
            payload["suggestion"] = self.suggestion
        if self.existing_id:
            payload["existing_id"] = self.existing_id
        return payload


def _exit_code_for(error: str) -> int:
    if error in {"usage", "confirmation_required"}:
        return ExitCode.USAGE
    if error in {"not_found", "image_not_found"}:
        return ExitCode.NOT_FOUND
    if error in {"permission_denied", "unauthorized"}:
        return ExitCode.PERMISSION
    if error == "conflict":
        return ExitCode.CONFLICT
    return ExitCode.FAILURE


# ── stdout writers ───────────────────────────────────────────────────────────


def json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)


def emit_json(payload: Any) -> None:
    """Write one JSON document to stdout (the agent contract)."""
    if _JSON_SUPPRESSED.get():
        return
    sys.stdout.write(json.dumps(payload, indent=2, default=json_default) + "\n")
    sys.stdout.flush()


def emit_jsonl(payload: Any) -> None:
    """Write one JSON Lines record to stdout."""
    sys.stdout.write(json.dumps(payload, default=json_default) + "\n")
    sys.stdout.flush()


def emit_quiet(values: list[Any] | tuple[Any, ...]) -> None:
    """Write one bare value per line (no headers) for piping."""
    for value in values:
        sys.stdout.write(f"{value}\n")
    sys.stdout.flush()


def emit_error_text(error: CliError) -> None:
    """Human-readable error on stderr. Never writes to stdout."""
    sys.stderr.write(f"Error: {error.message}\n")
    if error.suggestion:
        sys.stderr.write(f"Hint: {error.suggestion}\n")
    sys.stderr.flush()


def emit_error(error: CliError, *, json_output: bool | None = None) -> None:
    emit_error_text(error)
    if not _JSON_SUPPRESSED.get() and wants_json(json_output):
        emit_json(error.to_payload())


def abort(error: CliError, *, json_output: bool | None = None) -> NoReturn:
    """Print a structured error and exit with the mapped code."""
    emit_error(error, json_output=json_output)
    raise typer.Exit(error.exit_code)


# ── argv / flag helpers ──────────────────────────────────────────────────────


def _mark_json(value: bool) -> bool:
    """Option callback: remember ``--json`` for later ``abort()`` / ``emit_*``."""
    _JSON_REQUESTED.set(value is True)
    return value


def _mark_output(value: str | None) -> str | None:
    if isinstance(value, str) and value.strip().lower() == "json":
        _JSON_REQUESTED.set(True)
    return value


@contextmanager
def suppress_json_stdout() -> Iterator[None]:
    """Block JSON writes to stdout while a caller aggregates one document."""
    token = _JSON_SUPPRESSED.set(True)
    try:
        yield
    finally:
        _JSON_SUPPRESSED.reset(token)


def wants_json(json_output: bool | None = None, output: str | None = None) -> bool:
    """True when the invocation asked for JSON (flag, --output, context, or argv)."""
    if _JSON_SUPPRESSED.get():
        return False
    if json_output is True:
        return True
    if isinstance(output, str) and output.strip().lower() == "json":
        return True
    if _JSON_REQUESTED.get():
        return True
    ctx = click.get_current_context(silent=True)
    while ctx is not None:
        params = ctx.params
        if params.get("json_output") is True:
            return True
        out = params.get("output")
        if isinstance(out, str) and out.strip().lower() == "json":
            return True
        ctx = ctx.parent
    argv = sys.argv
    if "--json" in argv:
        return True
    for index, arg in enumerate(argv):
        if arg == "--output" and index + 1 < len(argv) and argv[index + 1] == "json":
            return True
        if arg.startswith("--output=") and arg.split("=", 1)[1] == "json":
            return True
    return False


def resolve_output_mode(
    *,
    json_output: bool = False,
    output: str | None = None,
    quiet: bool = False,
) -> str:
    """Return ``json``, ``quiet``, or ``table``. Invalid ``--output`` aborts."""
    if output is not None:
        normalized = output.strip().lower()
        if normalized not in {"json", "table"}:
            abort(
                CliError(
                    error="usage",
                    message=f"Invalid --output {output!r}. Use json or table.",
                    input={"output": output},
                    suggestion="Pass --json or --output json.",
                    exit_code=ExitCode.USAGE,
                )
            )
        json_output = json_output is True or normalized == "json"
    if json_output is True:
        return "json"
    if quiet:
        return "quiet"
    return "table"


def json_option() -> Any:
    return typer.Option(
        False,
        "--json",
        help="Write structured JSON to stdout. Progress and warnings go to stderr.",
        callback=_mark_json,
    )


def output_option() -> Any:
    return typer.Option(
        None,
        "--output",
        help="Output format: json or table. --output json is equivalent to --json.",
        callback=_mark_output,
    )


def quiet_option() -> Any:
    return typer.Option(
        False,
        "--quiet",
        "-q",
        help="Print one identifier per line, with no headers (for piping).",
    )


def yes_option() -> Any:
    return typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts (required in non-interactive terminals).",
    )


def dry_run_option() -> Any:
    return typer.Option(
        False,
        "--dry-run",
        help="Print the planned action without making changes.",
    )


def force_option(*, help: str = "Overwrite or replace an existing resource.") -> Any:
    return typer.Option(False, "--force", help=help)


# ── TTY / confirmation ───────────────────────────────────────────────────────


def is_interactive() -> bool:
    return sys.stdin.isatty()


def confirm_or_abort(
    message: str,
    *,
    yes: bool = False,
    force: bool = False,
    default: bool = False,
) -> None:
    """Confirm a mutating action, or fail clearly when no TTY is available."""
    if yes or force:
        return
    if not is_interactive():
        abort(
            CliError(
                error="confirmation_required",
                message="Confirmation required in a non-interactive terminal.",
                suggestion="Re-run with --yes to continue.",
                exit_code=ExitCode.USAGE,
            )
        )
    from hud.utils.hud_console import HUDConsole

    if not HUDConsole().confirm(message, default=default):
        HUDConsole().info("Cancelled.")
        raise typer.Exit(ExitCode.SUCCESS)


def read_text_arg(path: str) -> str:
    """Read a file path, or stdin when the path is ``-``."""
    if path == "-":
        return sys.stdin.read()
    from pathlib import Path

    target = Path(path)
    try:
        return target.read_text(encoding="utf-8")
    except FileNotFoundError:
        abort(
            CliError(
                error="not_found",
                message=f"File not found: {path}",
                input={"path": path},
                suggestion="Check the path, or pass - to read from stdin.",
            )
        )
        raise  # pragma: no cover — abort never returns
    except OSError as exc:
        abort(
            CliError(
                error="failure",
                message=f"Failed to read {path}: {exc}",
                input={"path": path},
            )
        )
        raise  # pragma: no cover


# ── exception mapping ────────────────────────────────────────────────────────


def map_request_error(
    exc: Any,
    *,
    resource: str | None = None,
    input: dict[str, Any] | None = None,
) -> CliError:
    """Map a :class:`HudRequestError` onto the CLI error contract."""
    status = getattr(exc, "status_code", None)
    detail = _request_detail(exc)
    label = resource or "Resource"

    if status == 404:
        return CliError(
            error="not_found",
            message=detail or f"{label} not found",
            input=input,
            suggestion=f"Check the {label.lower()} id, or list existing ones.",
            exit_code=ExitCode.NOT_FOUND,
        )
    if status in {401, 403}:
        return CliError(
            error="permission_denied",
            message=detail or "Permission denied",
            input=input,
            suggestion="Run 'hud login' or check that this API key can access the resource.",
            exit_code=ExitCode.PERMISSION,
        )
    if status == 409:
        return CliError(
            error="conflict",
            message=detail or f"{label} already exists",
            input=input,
            suggestion="Reuse the existing resource, or pass --if-not-exists.",
            exit_code=ExitCode.CONFLICT,
        )
    if status == 429:
        return CliError(
            error="rate_limited",
            message=detail or "Rate limited by the HUD API",
            input=input,
            suggestion="Retry after a short delay.",
            transient=True,
        )
    if isinstance(status, int) and status >= 500:
        return CliError(
            error="server_error",
            message=detail or f"HUD API server error ({status})",
            input=input,
            suggestion="Retry; this error is often transient.",
            transient=True,
        )
    return CliError(
        error="failure",
        message=detail or str(exc),
        input=input,
    )


def map_exception(exc: BaseException, *, input: dict[str, Any] | None = None) -> CliError:
    if isinstance(exc, CliError):
        return exc

    from hud.utils.exceptions import (
        HudAuthenticationError,
        HudRequestError,
        HudTimeoutError,
    )

    if isinstance(exc, HudRequestError):
        return map_request_error(exc, input=input)
    if isinstance(exc, HudAuthenticationError):
        return CliError(
            error="permission_denied",
            message=str(exc) or "Missing or invalid HUD API key",
            input=input,
            suggestion="Run 'hud login' or 'hud set HUD_API_KEY=your-key-here'.",
            exit_code=ExitCode.PERMISSION,
        )
    if isinstance(exc, HudTimeoutError):
        return CliError(
            error="timeout",
            message=str(exc) or "Timed out talking to the HUD API",
            input=input,
            suggestion="Retry; the failure may be transient. Increase --timeout if set.",
            transient=True,
        )
    return CliError(error="failure", message=str(exc) or type(exc).__name__, input=input)


def _request_detail(exc: Any) -> str:
    body = getattr(exc, "response_json", None)
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str) and detail:
            return detail
        if isinstance(detail, dict):
            nested = detail.get("message") or detail.get("error")
            if isinstance(nested, str) and nested:
                return nested
    message = getattr(exc, "message", None)
    if isinstance(message, str) and message:
        return message
    return str(exc)


def platform_call(
    fn: Any,
    *,
    resource: str | None = None,
    input: dict[str, Any] | None = None,
) -> Any:
    """Run a platform call and abort with a mapped CLI error on failure."""
    from hud.utils.exceptions import HudException, HudRequestError

    try:
        return fn()
    except HudRequestError as exc:
        abort(map_request_error(exc, resource=resource, input=input))
    except HudException as exc:
        abort(map_exception(exc, input=input))
    except Exception as exc:
        abort(CliError(error="failure", message=str(exc), input=input))


class UnknownTokenAsGetGroup(TyperGroup):
    """Dispatch an unknown first token to ``get`` so ``hud jobs <id>`` stays valid."""

    def resolve_command(self, ctx: Any, args: list[str]) -> Any:
        if args and not args[0].startswith("-") and args[0] not in self.commands:
            args.insert(0, "get")
        return super().resolve_command(ctx, args)


__all__ = [
    "CliError",
    "ExitCode",
    "UnknownTokenAsGetGroup",
    "abort",
    "confirm_or_abort",
    "dry_run_option",
    "emit_error",
    "emit_error_text",
    "emit_json",
    "emit_jsonl",
    "emit_quiet",
    "force_option",
    "is_interactive",
    "json_default",
    "json_option",
    "map_exception",
    "map_request_error",
    "output_option",
    "platform_call",
    "quiet_option",
    "read_text_arg",
    "resolve_output_mode",
    "suppress_json_stdout",
    "wants_json",
    "yes_option",
]
