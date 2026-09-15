"""HUD CLI - build, test, and deploy environments; run evaluations."""

from __future__ import annotations

import sys

import typer
from rich.console import Console

from hud.cli.io import CliError, CLIGroup, json_option, report

app = typer.Typer(
    name="hud",
    cls=CLIGroup,
    help="Build, test, and deploy HUD environments.",
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
)

console = Console()

# ---------------------------------------------------------------------------
# Register commands (each module owns its Typer args, docstring, and logic)
# NOTE: `sync` is registered below once migrated to the Taskset flow.
# ---------------------------------------------------------------------------

from .cancel import cancel_command  # noqa: E402
from .deploy import deploy_command  # noqa: E402
from .eval import eval_command  # noqa: E402
from .init import init_command  # noqa: E402
from .jobs import jobs_app  # noqa: E402
from .models import models_app  # noqa: E402
from .project import project_app  # noqa: E402
from .qa import qa_app  # noqa: E402
from .serve import serve_command  # noqa: E402
from .sync import sync_app  # noqa: E402
from .task import task_app  # noqa: E402
from .trace import trace_app  # noqa: E402

app.command(name="serve")(serve_command)
app.command(name="deploy")(deploy_command)
app.command(name="eval")(eval_command)
app.command(name="init")(init_command)
app.command(name="cancel", hidden=True, deprecated=True)(cancel_command)
app.add_typer(models_app, name="models")
app.add_typer(jobs_app, name="jobs")
app.add_typer(jobs_app, name="job", hidden=True)
app.add_typer(trace_app, name="trace")
app.add_typer(qa_app, name="qa")
app.add_typer(project_app, name="project")


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

    Values are stored in ~/.hud/.env and are loaded by hud.settings with
    the lowest precedence (overridden by process env and project .env).[/not dim]
    """
    from hud.utils.hud_console import HUDConsole

    from .config import parse_key_value, set_env_values

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

    def _render(saved: dict[str, object]) -> None:
        hud_console.success("Saved credentials to user config")
        hud_console.info(f"Location: {saved['path']}")
        hud_console.info(f"Keys: {', '.join(str(key) for key in saved['keys'])}")

    report(result, json_output=json_output, render=_render)


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

    result = {"name": "hud", "version": __version__}
    report(
        result,
        json_output=json_output,
        render=lambda saved: console.print(f"HUD CLI version: [cyan]{saved['version']}[/cyan]"),
    )


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
        version(json_output=False)
        raise typer.Exit
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(2)


app.add_typer(task_app, name="task")
app.add_typer(sync_app, name="sync")


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

    from .update import notify_if_outdated
    from .usage import recorded_invocation

    notify_if_outdated(sys.argv)

    with recorded_invocation(sys.argv, app):
        app()


if __name__ == "__main__":
    main()
