"""``hud models`` — list gateway models and fork trainable ones."""

from __future__ import annotations

import asyncio
from typing import Any

import typer
from rich.panel import Panel
from rich.table import Table

from hud.cli import (
    CLI,
    CliError,
)
from hud.settings import settings
from hud.train import TrainingClient
from hud.utils.exceptions import HudRequestError
from hud.utils.gateway import list_gateway_models, resolve_gateway_model
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient

hud_console = HUDConsole()

models_app = CLI(
    name="models",
    help="List gateway models and fork trainable ones.",
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

    The platform model catalog — the same models `create_agent` and `hud eval`
    resolve against.

    [not dim]Examples:
        hud models list
        hud models list --json
        hud models list --quiet[/not dim]
    """
    rows = [
        model.model_dump()
        for model in sorted(list_gateway_models(), key=lambda m: (m.name or m.id or "").lower())
    ]
    if quiet:
        for model in rows:
            ident = model.get("model_name") or model.get("id")
            if ident:
                typer.echo(ident)
        return rows
    if not rows:
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
    for model in rows:
        table.add_row(
            model.get("name") or model.get("id") or "-",
            model.get("model_name") or model.get("id") or "-",
            model.get("id") or "-",
            model["provider"].get("name") or "-",
            model.get("sdk_agent_type") or "-",
            "✓" if model.get("is_trainable") else "",
        )
    hud_console.stdout.print(table)
    hud_console.stdout.print(f"\n[dim]Gateway: {settings.hud_gateway_url}[/dim]")
    web = settings.hud_web_url.rstrip("/")
    hud_console.stdout.print(f"[dim]View a model in the browser: {web}/models/<id>[/dim]")
    return rows


@models_app.command("fork")
def fork_model(
    source: str = typer.Argument(..., help="Source model slug or id to fork from"),
    name: str = typer.Option(..., "--name", "-n", help="Name for the new trainable model"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the planned action without making changes."
    ),
    if_not_exists: bool = typer.Option(
        False,
        "--if-not-exists",
        help="If a model with this name already exists, print it and exit 0.",
    ),
) -> Any:
    """Create a team-owned trainable model derived from an existing one.

    The fork starts from the source model's active checkpoint, so you can keep
    training where it left off. Use the returned model slug with
    `hud.TrainingClient` (or as the gateway model string for sampling).

    [not dim]Examples:
        hud models fork claude-sonnet-4-6 --name my-sonnet
        hud models fork claude-sonnet-4-6 --name my-sonnet --json
        hud models fork claude-sonnet-4-6 --name my-sonnet --if-not-exists
        hud models fork claude-sonnet-4-6 --name my-sonnet --dry-run --json[/not dim]
    """
    if dry_run:
        hud_console.stdout.print(f"[dim]--dry-run: would fork {source!r} as {name!r}[/dim]")
        return {
            "dry_run": True,
            "action": "fork",
            "source": source,
            "name": name,
            "if_not_exists": if_not_exists,
        }

    source_id = resolve_gateway_model(source).id
    try:
        model = PlatformClient.from_settings().post(
            "/models/fork", json={"source_model_id": source_id, "name": name}
        )
    except HudRequestError as exc:
        if exc.status_code == 409 and if_not_exists:
            hud_console.stdout.print(f"[yellow]Model already exists[/yellow] [cyan]{name}[/cyan]")
            existing_id = resolve_gateway_model(name).id
            hud_console.stdout.print(f"[dim]id: {existing_id}[/dim]")
            return {"id": existing_id, "model_name": name, "existed": True}
        raise CliError.from_http(
            exc,
            resource="Model",
            input={"source": source, "name": name},
        ) from exc

    slug = model["model_name"]
    hud_console.stdout.print(
        Panel.fit(
            f"[bold green]Forked[/bold green] [cyan]{model.get('name') or slug}[/cyan]\n"
            f"slug: [green]{slug}[/green]\n"
            f"id:   [dim]{model['id']}[/dim]",
            border_style="green",
        )
    )
    hud_console.stdout.print(f"\n[dim]Train it: hud.TrainingClient({slug!r})[/dim]")
    hud_console.stdout.print(
        f"[dim]View: {settings.hud_web_url.rstrip('/')}/models/{model['id']}[/dim]"
    )
    return model


@models_app.command("checkpoints")
def list_checkpoints(
    model: str = typer.Argument(..., help="Model slug or id"),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Print one identifier per line, with no headers (for piping)."
    ),
) -> Any:
    """List a model's checkpoint tree, oldest first (▶ marks the active head).

    [not dim]Examples:
        hud models checkpoints <model>
        hud models checkpoints <model> --json
        hud models checkpoints <model> --quiet[/not dim]
    """
    model_id = resolve_gateway_model(model).id
    checkpoints = sorted(
        asyncio.run(TrainingClient(model_id).checkpoints()), key=lambda c: c.created_at or ""
    )
    rows = [checkpoint.model_dump() for checkpoint in checkpoints]
    if quiet:
        for ckpt in checkpoints:
            typer.echo(ckpt.id)
        return rows

    view = f"{settings.hud_web_url.rstrip('/')}/models/{model_id}?tab=checkpoints"
    if not checkpoints:
        hud_console.stdout.print(
            "[yellow]No checkpoints yet — this model serves its base weights[/yellow]"
        )
        hud_console.stdout.print(f"[dim]View: {view}[/dim]")
        return rows

    table = Table(title="Checkpoints")
    table.add_column("", style="green")
    table.add_column("Name", style="cyan")
    table.add_column("Reward", style="yellow", justify="right")
    table.add_column("Loss", style="magenta")
    table.add_column("Traces", justify="right")
    table.add_column("Created", style="dim")
    for ckpt in checkpoints:
        table.add_row(
            "▶" if ckpt.is_active else "",
            ckpt.name or ckpt.id[:8],
            f"{ckpt.mean_reward:.3f}" if ckpt.mean_reward is not None else "-",
            ckpt.loss_fn or "-",
            str(ckpt.num_traces or "-"),
            ckpt.created_at or "",
        )
    hud_console.stdout.print(table)
    hud_console.stdout.print(f"\n[dim]View: {view}[/dim]")
    return rows


@models_app.command("head")
def show_head(
    model: str = typer.Argument(..., help="Model slug or id"),
    set_to: str | None = typer.Option(
        None, "--set", help="Checkpoint id to promote to head (rollback / select)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the planned action without making changes."
    ),
) -> Any:
    """Show — or with ``--set``, change — the model's active checkpoint (the
    weights the gateway serves now).

    [not dim]Examples:
        hud models head <model>
        hud models head <model> --json
        hud models head <model> --set <checkpoint-id> --dry-run --json[/not dim]
    """
    model_id = resolve_gateway_model(model).id
    client = TrainingClient(model_id)
    view = f"{settings.hud_web_url.rstrip('/')}/models/{model_id}?tab=checkpoints"

    if set_to is not None:
        if dry_run:
            hud_console.stdout.print(f"[dim]--dry-run: would set head of {model} to {set_to}[/dim]")
            return {
                "dry_run": True,
                "action": "set_head",
                "model": model,
                "model_id": model_id,
                "checkpoint_id": set_to,
            }
        asyncio.run(client.set_head(set_to))
        hud_console.stdout.print(f"[green]Head set to[/green] [cyan]{set_to}[/cyan]")
        hud_console.stdout.print(f"[dim]View: {view}[/dim]")
        return {"model_id": model_id, "checkpoint_id": set_to, "action": "set_head"}

    head = asyncio.run(client.head())
    if head is None:
        hud_console.stdout.print(
            "[yellow]No active checkpoint — this model serves its base weights[/yellow]"
        )
    else:
        reward = f"{head.mean_reward:.3f}" if head.mean_reward is not None else "-"
        hud_console.stdout.print(
            Panel.fit(
                f"[bold green]HEAD[/bold green] [cyan]{head.name or head.id[:8]}[/cyan]\n"
                f"sampler: [green]{head.checkpoint_name or '-'}[/green]\n"
                f"reward:  {reward}    loss: {head.loss_fn or '-'}    "
                f"traces: {head.num_traces or '-'}\n"
                f"created: [dim]{head.created_at or ''}[/dim]",
                border_style="green",
            )
        )
    hud_console.stdout.print(f"[dim]View: {view}[/dim]")
    return head.model_dump() if head is not None else None
