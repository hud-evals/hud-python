"""``hud models`` — list gateway models."""

from __future__ import annotations

from typing import Any

import typer
from rich.panel import Panel
from rich.table import Table

from hud.cli import CLI
from hud.settings import settings
from hud.utils.gateway import list_gateway_models
from hud.utils.hud_console import HUDConsole

hud_console = HUDConsole()


models_app = CLI(
    name="models",
    help="List gateway models.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@models_app.command("list")
def list_models(
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Print one identifier per line, with no headers (for piping)."
    ),
) -> Any:
    """List models available through the HUD inference gateway.

    These are the models create_agent and hud eval resolve names against.

    [not dim]Examples:
        hud models list
        hud models list --json
        hud models list --quiet[/not dim]
    """
    models = sorted(list_gateway_models(), key=lambda m: (m.name or m.id).lower())
    rows = [model.model_dump() for model in models]
    if quiet:
        for model in models:
            typer.echo(model.model_name)
        return rows
    if not models:
        hud_console.stdout.print("[yellow]No models found[/yellow]")
        return rows

    hud_console.stdout.print(
        Panel.fit("[bold cyan]Available Models[/bold cyan]", border_style="cyan")
    )
    table = Table()
    table.add_column("Name", style="cyan")
    table.add_column("Model (API)", style="green")
    table.add_column("ID", style="blue", no_wrap=True)
    table.add_column("Provider", style="yellow")
    table.add_column("Agent", style="magenta")
    table.add_column("Trainable", style="green", justify="center")
    for model in models:
        table.add_row(
            model.name or model.id,
            model.model_name,
            model.id,
            model.provider.name or "-",
            model.sdk_agent_type or "-",
            "✓" if model.is_trainable else "",
        )
    hud_console.stdout.print(table)
    hud_console.stdout.print(f"\n[dim]Gateway: {settings.hud_gateway_url}[/dim]")
    web = settings.hud_web_url.rstrip("/")
    hud_console.stdout.print(f"[dim]View a model in the browser: {web}/models/<id>[/dim]")
    return rows
