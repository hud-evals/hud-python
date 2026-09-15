"""``hud models`` — list gateway models and fork trainable ones."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hud.cli.app import (
    CLI,
    CliError,
)
from hud.utils.platform import PlatformClient

console = Console()

models_app = CLI(
    name="models",
    help="List gateway models and fork trainable ones.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


def _provider_name(row: dict[str, Any]) -> str:
    provider = row.get("provider")
    if isinstance(provider, dict):
        return str(provider.get("name") or "-")
    return "-"


def _render_models(rows: list[dict[str, Any]]) -> None:
    from hud.settings import settings

    if not rows:
        console.print("[yellow]No models found[/yellow]")
        return
    console.print(Panel.fit("[bold cyan]Available Models[/bold cyan]", border_style="cyan"))
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
            _provider_name(model),
            model.get("sdk_agent_type") or "-",
            "✓" if model.get("is_trainable") else "",
        )
    console.print(table)
    console.print(f"\n[dim]Gateway: {settings.hud_gateway_url}[/dim]")
    web = settings.hud_web_url.rstrip("/")
    console.print(f"[dim]View a model in the browser: {web}/models/<id>[/dim]")


def _render_head(model_id: str, head: dict[str, Any] | None) -> None:
    if head is None:
        console.print("[yellow]No active checkpoint — this model serves its base weights[/yellow]")
        console.print(f"[dim]View: {_model_url(model_id, tab='checkpoints')}[/dim]")
        return
    reward = head.get("mean_reward")
    console.print(
        Panel.fit(
            f"[bold green]HEAD[/bold green] [cyan]{head.get('name') or head['id'][:8]}[/cyan]\n"
            f"sampler: [green]{head.get('checkpoint_name') or '-'}[/green]\n"
            f"reward:  {f'{reward:.3f}' if reward is not None else '-'}    "
            f"loss: {head.get('loss_fn') or '-'}    traces: {head.get('num_traces') or '-'}\n"
            f"created: [dim]{head.get('created_at') or ''}[/dim]",
            border_style="green",
        )
    )
    console.print(f"[dim]View: {_model_url(model_id, tab='checkpoints')}[/dim]")


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
    from hud.utils.gateway import list_gateway_models

    rows = [
        model.model_dump()
        for model in sorted(list_gateway_models(), key=lambda m: (m.name or m.id or "").lower())
    ]
    if quiet:
        for model in rows:
            ident = model.get("model_name") or model.get("id")
            if ident:
                typer.echo(ident)
    else:
        _render_models(rows)
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
    from hud.utils.exceptions import HudRequestError

    if dry_run:
        payload = {
            "dry_run": True,
            "action": "fork",
            "source": source,
            "name": name,
            "if_not_exists": if_not_exists,
        }
        console.print(
            f"[dim]--dry-run: would fork {payload['source']!r} as {payload['name']!r}[/dim]"
        )
        return payload

    source_id = _resolve_model_id(source)
    try:
        model = PlatformClient.from_settings().post(
            "/models/fork", json={"source_model_id": source_id, "name": name}
        )
    except HudRequestError as exc:
        if exc.status_code == 409 and if_not_exists:
            existing = _existing_model(name)
            saved = {**existing, "existed": True}
            console.print(
                "[yellow]Model already exists[/yellow] "
                f"[cyan]{saved.get('model_name') or name}[/cyan]"
            )
            console.print(f"[dim]id: {saved.get('id')}[/dim]")
            return saved
        raise CliError.from_http(
            exc,
            resource="Model",
            input={"source": source, "name": name},
        ) from exc

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
    console.print(f"[dim]View: {_model_url(model['id'])}[/dim]")
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
    model_id = _resolve_model_id(model)
    checkpoints = sorted(_get_checkpoints(model_id), key=lambda c: c.get("created_at") or "")

    def _render(rows: list[dict[str, Any]]) -> None:
        if not rows:
            console.print(
                "[yellow]No checkpoints yet — this model serves its base weights[/yellow]"
            )
            console.print(f"[dim]View: {_model_url(model_id, tab='checkpoints')}[/dim]")
            return
        table = Table(title="Checkpoints")
        table.add_column("", style="green")
        table.add_column("Name", style="cyan")
        table.add_column("Reward", style="yellow", justify="right")
        table.add_column("Loss", style="magenta")
        table.add_column("Traces", justify="right")
        table.add_column("Created", style="dim")
        for ckpt in rows:
            reward = ckpt.get("mean_reward")
            table.add_row(
                "▶" if ckpt.get("is_active") else "",
                ckpt.get("name") or ckpt["id"][:8],
                f"{reward:.3f}" if reward is not None else "-",
                ckpt.get("loss_fn") or "-",
                str(ckpt.get("num_traces") or "-"),
                str(ckpt.get("created_at") or ""),
            )
        console.print(table)
        console.print(f"\n[dim]View: {_model_url(model_id, tab='checkpoints')}[/dim]")

    if quiet:
        for ckpt in checkpoints:
            if ckpt.get("id"):
                typer.echo(ckpt["id"])
    else:
        _render(checkpoints)
    return checkpoints


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
    model_id = _resolve_model_id(model)

    if set_to is not None:
        if dry_run:
            payload = {
                "dry_run": True,
                "action": "set_head",
                "model": model,
                "model_id": model_id,
                "checkpoint_id": set_to,
            }
            console.print(
                f"[dim]--dry-run: would set head of {payload['model']} "
                f"to {payload['checkpoint_id']}[/dim]"
            )
            return payload
        _set_head(model_id, set_to)
        saved = {"model_id": model_id, "checkpoint_id": set_to, "action": "set_head"}
        console.print(f"[green]Head set to[/green] [cyan]{saved['checkpoint_id']}[/cyan]")
        console.print(f"[dim]View: {_model_url(saved['model_id'], tab='checkpoints')}[/dim]")
        return saved

    head = next((c for c in _get_checkpoints(model_id) if c.get("is_active")), None)
    _render_head(model_id, head)
    return head


def _model_url(model_id: str, *, tab: str | None = None) -> str:
    """Web app URL for a model (optionally a specific tab, e.g. ``checkpoints``)."""
    from hud.settings import settings

    url = f"{settings.hud_web_url.rstrip('/')}/models/{model_id}"
    return f"{url}?tab={tab}" if tab else url


def _resolve_model_id(model: str) -> str:
    """Map a model slug to its id (an id passes straight through)."""
    from hud.utils.exceptions import HudRequestError

    try:
        return str(UUID(model))
    except ValueError:
        try:
            data = PlatformClient.from_settings().get("/models/resolve", params={"model": model})
        except HudRequestError as exc:
            raise CliError.from_http(exc, resource="Model", input={"model": model}) from exc
        return str(data["id"])


def _existing_model(name: str) -> dict[str, Any]:
    """Resolve a model name after a conflict so ``--if-not-exists`` can return it."""
    model_id = _resolve_model_id(name)
    return {"id": model_id, "model_name": name}


def _get_checkpoints(model_id: str) -> list[dict[str, Any]]:
    from hud.utils.exceptions import HudRequestError

    try:
        return PlatformClient.from_settings().get(f"/models/{model_id}/checkpoints")
    except HudRequestError as exc:
        raise CliError.from_http(exc, resource="Checkpoints", input={"model": model_id}) from exc


def _set_head(model_id: str, checkpoint_id: str) -> None:
    from hud.utils.exceptions import HudRequestError

    try:
        PlatformClient.from_settings().put(
            f"/models/{model_id}/head", json={"checkpoint_id": checkpoint_id}
        )
    except HudRequestError as exc:
        raise CliError.from_http(
            exc,
            resource="Checkpoint",
            input={"model": model_id, "checkpoint_id": checkpoint_id},
        ) from exc
