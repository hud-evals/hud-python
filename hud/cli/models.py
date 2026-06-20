"""``hud models`` — list gateway models and fork trainable ones."""

from __future__ import annotations

import json
from typing import Any, cast

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

models_app = typer.Typer(
    name="models",
    help="List gateway models and fork trainable ones",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@models_app.command("list")
def list_models(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List models available through the HUD inference gateway.

    The platform model catalog — the same models `create_agent` and `hud eval`
    resolve against.
    """
    from hud.cli.utils.api import require_api_key
    from hud.settings import settings
    from hud.utils.gateway import list_gateway_models

    require_api_key("list models")

    try:
        models_list = list_gateway_models()
    except Exception as e:
        console.print(f"[red]Failed to fetch models: {e}[/red]")
        raise typer.Exit(1) from e

    if json_output:
        console.print_json(json.dumps([m.model_dump() for m in models_list], indent=2))
        return

    if not models_list:
        console.print("[yellow]No models found[/yellow]")
        return

    models_list = sorted(models_list, key=lambda m: (m.name or m.id or "").lower())
    console.print(Panel.fit("[bold cyan]Available Models[/bold cyan]", border_style="cyan"))

    table = Table()
    table.add_column("Name", style="cyan")
    table.add_column("Model (API)", style="green")
    table.add_column("ID", style="blue", no_wrap=True)
    table.add_column("Provider", style="yellow")
    table.add_column("Agent", style="magenta")
    table.add_column("Trainable", style="green", justify="center")
    for model in models_list:
        table.add_row(
            model.name or model.id or "-",
            model.model_name or model.id or "-",
            model.id or "-",
            model.provider.name or "-",
            model.sdk_agent_type or "-",
            "✓" if model.is_trainable else "",
        )
    console.print(table)
    console.print(f"\n[dim]Gateway: {settings.hud_gateway_url}[/dim]")


@models_app.command("fork")
def fork_model(
    source: str = typer.Argument(..., help="Source model slug or id to fork from"),
    name: str = typer.Option(..., "--name", "-n", help="Name for the new trainable model"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Create a team-owned trainable model derived from an existing one.

    The fork starts from the source model's active checkpoint, so you can keep
    training where it left off. Use the returned model slug with
    `hud.TrainingClient` (or as the gateway model string for sampling).
    """
    from hud.cli.utils.api import require_api_key
    from hud.settings import settings
    from hud.utils.requests import make_request_sync

    require_api_key("fork a model")

    source_id = _resolve_model_id(source)
    try:
        model = make_request_sync(
            "POST",
            f"{settings.hud_api_url}/v2/models/fork",
            json={"source_model_id": source_id, "name": name},
            api_key=settings.api_key,
        )
    except Exception as e:
        console.print(f"[red]Fork failed: {e}[/red]")
        raise typer.Exit(1) from e

    if json_output:
        console.print_json(json.dumps(model, indent=2))
        return
    slug = model["model_name"]
    console.print(
        Panel.fit(
            f"[bold green]Forked[/bold green] [cyan]{model.get('name') or slug}[/cyan]\n"
            f"slug: [green]{slug}[/green]\n"
            f"id:   [dim]{model['id']}[/dim]",
            border_style="green",
        )
    )
    console.print(f"\n[dim]Train it: hud.TrainingClient({slug!r})[/dim]")


@models_app.command("checkpoints")
def list_checkpoints(
    model: str = typer.Argument(..., help="Model slug or id"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List a model's checkpoint tree, oldest first (▶ marks the active head)."""
    from hud.cli.utils.api import require_api_key

    require_api_key("list checkpoints")
    checkpoints = _get_checkpoints(model)

    if json_output:
        console.print_json(json.dumps(checkpoints, indent=2))
        return
    if not checkpoints:
        console.print("[yellow]No checkpoints yet — this model serves its base weights[/yellow]")
        return

    checkpoints = sorted(checkpoints, key=lambda c: c.get("created_at") or "")
    table = Table(title="Checkpoints")
    table.add_column("", style="green")  # active marker
    table.add_column("Name", style="cyan")
    table.add_column("Reward", style="yellow", justify="right")
    table.add_column("Loss", style="magenta")
    table.add_column("Traces", justify="right")
    table.add_column("Created", style="dim")
    for ckpt in checkpoints:
        reward = ckpt.get("mean_reward")
        table.add_row(
            "▶" if ckpt.get("is_active") else "",
            ckpt.get("name") or ckpt["id"][:8],
            f"{reward:.3f}" if reward is not None else "-",
            ckpt.get("loss_fn") or "-",
            str(ckpt.get("num_traces") or "-"),
            (ckpt.get("created_at") or "")[:19],
        )
    console.print(table)


@models_app.command("head")
def show_head(
    model: str = typer.Argument(..., help="Model slug or id"),
    set_to: str | None = typer.Option(
        None, "--set", help="Checkpoint id to promote to head (rollback / select)"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show — or with ``--set``, change — the model's active checkpoint (the
    weights the gateway serves now)."""
    from hud.cli.utils.api import require_api_key

    require_api_key("manage head")

    if set_to is not None:
        _set_head(model, set_to)
        console.print(f"[green]Head set to[/green] [cyan]{set_to}[/cyan]")
        return

    head = next((c for c in _get_checkpoints(model) if c.get("is_active")), None)

    if json_output:
        console.print_json(json.dumps(head, indent=2))
        return
    if head is None:
        console.print("[yellow]No active checkpoint — this model serves its base weights[/yellow]")
        return

    reward = head.get("mean_reward")
    console.print(
        Panel.fit(
            f"[bold green]HEAD[/bold green] [cyan]{head.get('name') or head['id'][:8]}[/cyan]\n"
            f"sampler: [green]{head.get('checkpoint_name') or '-'}[/green]\n"
            f"reward:  {f'{reward:.3f}' if reward is not None else '-'}    "
            f"loss: {head.get('loss_fn') or '-'}    traces: {head.get('num_traces') or '-'}\n"
            f"created: [dim]{(head.get('created_at') or '')[:19]}[/dim]",
            border_style="green",
        )
    )


def _resolve_model_id(model: str) -> str:
    """Map a model slug to its id (an id passes straight through)."""
    from uuid import UUID

    from hud.settings import settings
    from hud.utils.requests import make_request_sync

    try:
        return str(UUID(model))
    except ValueError:
        from urllib.parse import quote

        data = make_request_sync(
            "GET",
            f"{settings.hud_api_url}/v2/models/resolve?model={quote(model, safe='')}",
            api_key=settings.api_key,
        )
        return str(data["id"])


def _get_checkpoints(model: str) -> list[dict[str, Any]]:
    from hud.settings import settings
    from hud.utils.requests import make_request_sync

    model_id = _resolve_model_id(model)
    try:
        # The checkpoints endpoint returns a JSON array (make_request_sync is
        # typed for the common object response).
        return cast(
            "list[dict[str, Any]]",
            make_request_sync(
                "GET",
                f"{settings.hud_api_url}/v2/models/{model_id}/checkpoints",
                api_key=settings.api_key,
            ),
        )
    except Exception as e:
        console.print(f"[red]Failed to fetch checkpoints: {e}[/red]")
        raise typer.Exit(1) from e


def _set_head(model: str, checkpoint_id: str) -> None:
    from hud.settings import settings
    from hud.utils.requests import make_request_sync

    model_id = _resolve_model_id(model)
    try:
        make_request_sync(
            "PUT",
            f"{settings.hud_api_url}/v2/models/{model_id}/head",
            json={"checkpoint_id": checkpoint_id},
            api_key=settings.api_key,
        )
    except Exception as e:
        console.print(f"[red]Failed to set head: {e}[/red]")
        raise typer.Exit(1) from e
