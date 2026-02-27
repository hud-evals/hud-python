"""HUD CLI - Build, test, and deploy RL environments."""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.panel import Panel

# Create the main Typer app
app = typer.Typer(
    name="hud",
    help="🚀 HUD CLI - build, test, and deploy RL environments",
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
# ---------------------------------------------------------------------------

from .analyze import analyze_command  # noqa: E402
from .build import build_command  # noqa: E402
from .cancel import cancel_command  # noqa: E402
from .convert import convert_command  # noqa: E402
from .debug import debug_command  # noqa: E402
from .deploy import deploy_command  # noqa: E402
from .dev import dev_command  # noqa: E402
from .eval import eval_command  # noqa: E402
from .init import init_command  # noqa: E402
from .link import link_command  # noqa: E402
from .models import models_command  # noqa: E402
from .push import push_command  # noqa: E402
from .rft import rft_run_command  # noqa: E402
from .rft_status import rft_status_typer_command  # noqa: E402

_EXTRA_ARGS = {"allow_extra_args": True, "ignore_unknown_options": True}

app.command(name="analyze", context_settings=_EXTRA_ARGS)(analyze_command)
app.command(name="debug", context_settings=_EXTRA_ARGS)(debug_command)
app.command(name="dev", context_settings=_EXTRA_ARGS)(dev_command)
app.command(name="build", context_settings=_EXTRA_ARGS)(build_command)
app.command(name="deploy")(deploy_command)
app.command(name="link")(link_command)
app.command(name="eval")(eval_command)
app.command(name="push", hidden=True)(push_command)
app.command(name="init")(init_command)
app.command(name="convert")(convert_command)
app.command(name="cancel")(cancel_command)
app.command(name="models")(models_command)


@app.command(name="set")
def set_command(
    assignments: list[str] = typer.Argument(  # type: ignore[arg-type]  # noqa: B008
        ..., help="One or more KEY=VALUE pairs to persist in ~/.hud/.env"
    ),
) -> None:
    """Persist API keys or other variables for HUD to use by default.

    [not dim]Examples:
        hud set ANTHROPIC_API_KEY=sk-... OPENAI_API_KEY=sk-...

    Values are stored in ~/.hud/.env and are loaded by hud.settings with
    the lowest precedence (overridden by process env and project .env).[/not dim]
    """
    from hud.utils.hud_console import HUDConsole

    from .utils.config import set_env_values

    hud_console = HUDConsole()

    updates: dict[str, str] = {}
    for item in assignments:
        if "=" not in item:
            hud_console.error(f"Invalid assignment (expected KEY=VALUE): {item}")
            raise typer.Exit(1)
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            hud_console.error(f"Invalid key in assignment: {item}")
            raise typer.Exit(1)
        updates[key] = value

    path = set_env_values(updates)
    hud_console.success("Saved credentials to user config")
    hud_console.info(f"Location: {path}")


@app.command()
def version() -> None:
    """Show HUD CLI version."""
    try:
        from hud import __version__

        console.print(f"HUD CLI version: [cyan]{__version__}[/cyan]")
    except ImportError:
        console.print("HUD CLI version: [cyan]unknown[/cyan]")


# RFT subcommand group
rft_app = typer.Typer(help="🚀 Reinforcement Fine-Tuning (RFT) commands")
rft_app.command("run")(rft_run_command)
rft_app.command("status")(rft_status_typer_command)
app.add_typer(rft_app, name="rft")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for the CLI."""
    if not (len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ["--help", "-h"])):
        from .utils.version_check import display_update_prompt

        display_update_prompt()

    if "--version" in sys.argv:
        try:
            from hud import __version__

            console.print(f"HUD CLI version: [cyan]{__version__}[/cyan]")
        except ImportError:
            console.print("HUD CLI version: [cyan]unknown[/cyan]")
        return

    try:
        if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ["--help", "-h"]):
            console.print(
                Panel.fit(
                    "[bold cyan]🚀 HUD CLI[/bold cyan]\nBuild, test, and deploy RL environments",
                    border_style="cyan",
                )
            )
            console.print("\n[yellow]Quick Start:[/yellow]")
            console.print(
                "  1. Create a new environment: [cyan]hud init my-env && cd my-env[/cyan]"
            )
            console.print("  2. Start dev server:        [cyan]hud dev[/cyan]")
            console.print("  3. Deploy to HUD platform:  [cyan]hud deploy[/cyan]")
            console.print("  4. Run evaluations:         [cyan]hud eval tasks.jsonl[/cyan]")
            console.print("\n[yellow]Training:[/yellow]")
            console.print("  [cyan]hud rft run tasks.jsonl[/cyan]      Launch an RFT training job")
            console.print("  [cyan]hud rft status <model-id>[/cyan]  Check training status\n")

        app()
    except typer.Exit as e:
        try:
            exit_code = getattr(e, "exit_code", 0)
        except Exception:
            exit_code = 1
        if exit_code != 0:
            from hud.utils.hud_console import hud_console

            hud_console.info(SUPPORT_HINT)
        raise
    except Exception:
        raise


if __name__ == "__main__":
    main()
