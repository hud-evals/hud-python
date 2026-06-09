"""List available models from the HUD gateway model catalog."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def models_command(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """📋 List models available through the HUD inference gateway.

    [not dim]Shows the platform model catalog — the same models `create_agent`
    and `hud eval` resolve against.

    Examples:
        hud models              # List all models
        hud models --json       # Output as JSON[/not dim]
    """
    from hud.cli.utils.api import require_api_key
    from hud.settings import settings
    from hud.shared.gateway import list_gateway_models

    require_api_key("list models")

    try:
        models_list = list_gateway_models()
    except Exception as e:
        console.print(f"[red]❌ Failed to fetch models: {e}[/red]")
        raise typer.Exit(1) from e

    if json_output:
        console.print_json(json.dumps([m.model_dump() for m in models_list], indent=2))
        return

    if not models_list:
        console.print("[yellow]No models found[/yellow]")
        return

    models_list = sorted(models_list, key=lambda m: (m.name or m.id or "").lower())

    console.print(Panel.fit("📋 [bold cyan]Available Models[/bold cyan]", border_style="cyan"))

    table = Table()
    table.add_column("Name", style="cyan")
    table.add_column("Model (API)", style="green")
    table.add_column("Provider", style="yellow")
    table.add_column("Agent", style="magenta")

    for model in models_list:
        table.add_row(
            model.name or model.id or "-",
            model.model_name or model.id or "-",
            model.provider.name or "-",
            model.sdk_agent_type or model.provider.default_sdk_agent_type or "-",
        )

    console.print(table)
    console.print(f"\n[dim]Gateway: {settings.hud_gateway_url}[/dim]")
