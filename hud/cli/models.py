"""List available models from HUD inference gateway."""

from __future__ import annotations

import json

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def models_command(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """📋 List available models from HUD inference gateway.

    [not dim]Shows models available via the HUD inference gateway at inference.hud.ai.

    Examples:
        hud models              # List all models
        hud models --json       # Output as JSON[/not dim]
    """
    from hud.settings import settings

    try:
        response = httpx.get(
            f"{settings.hud_gateway_url}/models",
            headers={"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        if json_output:
            console.print_json(json.dumps(data, indent=2))
            return

        models_list = data.get("data", data) if isinstance(data, dict) else data

        if not models_list:
            console.print("[yellow]No models found[/yellow]")
            return

        models_list = sorted(
            models_list,
            key=lambda x: (
                (x.get("name") or str(x)).lower() if isinstance(x, dict) else str(x).lower()
            ),
        )

        console.print(Panel.fit("📋 [bold cyan]Available Models[/bold cyan]", border_style="cyan"))

        table = Table()
        table.add_column("Name", style="cyan")
        table.add_column("Model (API)", style="green")
        table.add_column("Routes", style="yellow")

        for model in models_list:
            if isinstance(model, dict):
                name = model.get("name", "-")
                api_model = model.get("model", model.get("id", "-"))
                routes = model.get("routes", [])
                routes_str = ", ".join(routes) if routes else "-"
                table.add_row(name, api_model, routes_str)
            else:
                table.add_row(str(model), "-", "-")

        console.print(table)
        console.print(f"\n[dim]Gateway: {settings.hud_gateway_url}[/dim]")

    except httpx.HTTPStatusError as e:
        console.print(f"[red]❌ API error: {e.response.status_code}[/red]")
        console.print(f"[dim]{e.response.text}[/dim]")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]❌ Failed to fetch models: {e}[/red]")
        raise typer.Exit(1) from e
