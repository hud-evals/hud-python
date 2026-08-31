"""HUD CLI - build, test, and deploy environments; run evaluations."""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.panel import Panel

from hud.cli.utils.output import CliError, json_option
from hud.utils.exceptions import HudException

app = typer.Typer(
    name="hud",
    help=(
        "HUD CLI — environments, evaluations, and the HUD platform.\n\n"
        "Resource commands use noun-verb grammar "
        "(hud jobs list, hud models fork, hud task start).\n"
        "Pass --json on list/get/create/status commands for structured stdout; "
        "progress and errors go to stderr.\n\n"
        "Exit codes: 0 ok · 1 failure · 2 usage · 3 not found · 4 permission · 5 conflict."
    ),
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
)

console = Console()

SUPPORT_HINT = (
    "If this looks like an issue with the sdk, please make a github issue at "
    "https://github.com/hud-evals/hud-python/issues"
)

# ---------------------------------------------------------------------------
# Register commands (each module owns its Typer args, docstring, and logic)
# NOTE: `sync` is registered below once migrated to the Taskset flow.
# ---------------------------------------------------------------------------

from .cancel import cancel_command  # noqa: E402
from .client import client_app  # noqa: E402
from .deploy import deploy_command  # noqa: E402
from .eval import eval_command  # noqa: E402
from .init import init_command  # noqa: E402
from .jobs import jobs_app  # noqa: E402
from .login import login_command  # noqa: E402
from .models import models_app  # noqa: E402
from .qa import qa_app  # noqa: E402
from .serve import serve_command  # noqa: E402
from .sync import sync_app  # noqa: E402
from .task import task_app  # noqa: E402
from .trace import trace_app  # noqa: E402

app.command(name="serve")(serve_command)
app.command(name="deploy")(deploy_command)
app.command(name="login")(login_command)
app.command(name="eval")(eval_command)
app.command(name="init")(init_command)
app.command(name="cancel")(cancel_command)
app.add_typer(models_app, name="models")
app.add_typer(jobs_app, name="jobs")
app.add_typer(jobs_app, name="job", hidden=True)
app.add_typer(trace_app, name="trace")
app.add_typer(qa_app, name="qa")


@app.command(name="set")
def set_command(
    assignments: list[str] = typer.Argument(  # noqa: B008
        ..., help="One or more KEY=VALUE pairs to persist in ~/.hud/.env"
    ),
    json_output: bool = json_option(),
) -> None:
    """Persist API keys or other variables for HUD to use by default.

    [not dim]Examples:
        hud set ANTHROPIC_API_KEY=sk-... OPENAI_API_KEY=sk-...
        hud set HUD_API_KEY=sk-... --json
        hud auth set HUD_API_KEY=sk-...

    Values are stored in ~/.hud/.env and are loaded by hud.settings with
    the lowest precedence (overridden by process env and project .env).[/not dim]
    """
    from hud.cli.utils.output import CliError, ExitCode, abort, emit_json, wants_json
    from hud.utils.hud_console import HUDConsole

    from .utils.config import parse_key_value, set_env_values

    hud_console = HUDConsole()

    updates: dict[str, str] = {}
    for item in assignments:
        parsed = parse_key_value(item)
        if parsed is None:
            abort(
                CliError(
                    error="usage",
                    message=f"Invalid assignment (expected KEY=VALUE): {item}",
                    input={"assignment": item},
                    suggestion="Pass one or more KEY=VALUE pairs.",
                    exit_code=ExitCode.USAGE,
                ),
                json_output=json_output,
            )
        key, value = parsed
        updates[key] = value

    path = set_env_values(updates)
    if wants_json(json_output):
        emit_json({"path": str(path), "keys": list(updates)})
        return
    hud_console.success("Saved credentials to user config")
    hud_console.info(f"Location: {path}")


@app.command()
def version(
    json_output: bool = json_option(),
) -> None:
    """Show HUD CLI version.

    [not dim]Examples:
        hud version
        hud version --json[/not dim]
    """
    from hud import __version__  # lazy: keeps CLI startup off the full package import
    from hud.cli.utils.output import emit_json, wants_json

    if wants_json(json_output):
        emit_json({"name": "hud", "version": __version__})
        return
    console.print(f"HUD CLI version: [cyan]{__version__}[/cyan]")


# Client subcommand group (drive a running env control channel from the shell)
app.add_typer(client_app, name="client")

# Task subcommand group (start a task / grade an answer, direct from source or via --url)
app.add_typer(task_app, name="task")

# Sync subcommand group (migrated to the Taskset flow)
app.add_typer(sync_app, name="sync")

# Auth noun group (login / set). Root ``hud login`` and ``hud set`` stay as aliases.
auth_app = typer.Typer(
    name="auth",
    help="Authenticate and persist credentials.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)
auth_app.command("login")(login_command)
auth_app.command("set")(set_command)
app.add_typer(auth_app, name="auth")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for the CLI."""
    global console
    # Windows cmd.exe uses the system code page (e.g. cp1252) which can't
    # encode the emoji that Rich uses. Rewrap stdout/stderr as UTF-8 so
    # Rich's legacy Windows renderer never hits a charmap error.
    if sys.platform == "win32":
        import io

        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "buffer"):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
        console = Console()  # recreate against the new stdout

    if not (len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ["--help", "-h"])):
        from .utils.version_check import display_update_prompt

        display_update_prompt()

    if "--version" in sys.argv:
        from hud import __version__  # lazy: keeps CLI startup off the full package import
        from hud.cli.utils.output import emit_json, wants_json

        if wants_json():
            emit_json({"name": "hud", "version": __version__})
        else:
            console.print(f"HUD CLI version: [cyan]{__version__}[/cyan]")
        return

    from .utils.usage import recorded_invocation

    with recorded_invocation(sys.argv):
        try:
            if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ["--help", "-h"]):
                console.print(
                    Panel.fit(
                        "[bold cyan]HUD CLI[/bold cyan]\nBuild, test, and deploy environments",
                        border_style="cyan",
                    )
                )
                console.print("\n[yellow]Quick Start:[/yellow]")
                console.print("  Run evaluations: [cyan]hud eval tasks.py claude[/cyan]")
                console.print("  List platform jobs: [cyan]hud jobs list --json[/cyan]\n")

            app()
        except typer.Exit as e:
            try:
                exit_code = getattr(e, "exit_code", 0)
            except Exception:
                exit_code = 1
            if exit_code not in (0, 2):
                from hud.utils.hud_console import hud_console

                hud_console.info(SUPPORT_HINT)
            raise
        except CliError as e:
            from hud.cli.utils.output import abort

            abort(e)
        except HudException as e:
            from hud.cli.utils.output import abort, map_exception

            abort(map_exception(e))


if __name__ == "__main__":
    main()
