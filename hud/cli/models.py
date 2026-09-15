"""``hud models`` — list gateway models and fork trainable ones."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hud.cli.io import (
    json_option,
    map_request_error,
    report,
)
from hud.utils.platform import PlatformClient

console = Console()

models_app = typer.Typer(
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
    json_output: bool = json_option(),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Print one identifier per line, with no headers (for piping)."
    ),
) -> None:
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
    report(
        rows,
        json_output=json_output,
        quiet=quiet,
        ids=lambda models: [
            str(model.get("model_name") or model.get("id") or "")
            for model in models
            if model.get("model_name") or model.get("id")
        ],
        render=_render_models,
    )


@models_app.command("fork")
def fork_model(
    source: str = typer.Argument(..., help="Source model slug or id to fork from"),
    name: str = typer.Option(..., "--name", "-n", help="Name for the new trainable model"),
    json_output: bool = json_option(),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the planned action without making changes."
    ),
    if_not_exists: bool = typer.Option(
        False,
        "--if-not-exists",
        help="If a model with this name already exists, print it and exit 0.",
    ),
) -> None:
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
        report(
            payload,
            json_output=json_output,
            render=lambda saved: console.print(
                f"[dim]--dry-run: would fork {saved['source']!r} as {saved['name']!r}[/dim]"
            ),
        )
        return

    source_id = _resolve_model_id(source)
    try:
        model = PlatformClient.from_settings().post(
            "/models/fork", json={"source_model_id": source_id, "name": name}
        )
    except HudRequestError as exc:
        if exc.status_code == 409 and if_not_exists:
            existing = _existing_model(name)
            saved = {**existing, "existed": True}
            report(
                saved,
                json_output=json_output,
                render=lambda row: (
                    console.print(
                        "[yellow]Model already exists[/yellow] "
                        f"[cyan]{row.get('model_name') or name}[/cyan]"
                    ),
                    console.print(f"[dim]id: {row.get('id')}[/dim]"),
                ),
            )
            return
        raise map_request_error(
            exc,
            resource="Model",
            input={"source": source, "name": name},
        ) from exc

    def _render_fork(saved: dict[str, Any]) -> None:
        slug = saved["model_name"]
        console.print(
            Panel.fit(
                f"[bold green]Forked[/bold green] [cyan]{saved.get('name') or slug}[/cyan]\n"
                f"slug: [green]{slug}[/green]\n"
                f"id:   [dim]{saved['id']}[/dim]",
                border_style="green",
            )
        )
        console.print(f"\n[dim]Train it: hud.TrainingClient({slug!r})[/dim]")
        console.print(f"[dim]View: {_model_url(saved['id'])}[/dim]")

    report(model, json_output=json_output, render=_render_fork)


@models_app.command("checkpoints")
def list_checkpoints(
    model: str = typer.Argument(..., help="Model slug or id"),
    json_output: bool = json_option(),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Print one identifier per line, with no headers (for piping)."
    ),
) -> None:
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

    report(
        checkpoints,
        json_output=json_output,
        quiet=quiet,
        ids=lambda rows: [str(ckpt.get("id") or "") for ckpt in rows if ckpt.get("id")],
        render=_render,
    )


@models_app.command("head")
def show_head(
    model: str = typer.Argument(..., help="Model slug or id"),
    set_to: str | None = typer.Option(
        None, "--set", help="Checkpoint id to promote to head (rollback / select)"
    ),
    json_output: bool = json_option(),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the planned action without making changes."
    ),
) -> None:
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
            report(
                payload,
                json_output=json_output,
                render=lambda saved: console.print(
                    f"[dim]--dry-run: would set head of {saved['model']} "
                    f"to {saved['checkpoint_id']}[/dim]"
                ),
            )
            return
        _set_head(model_id, set_to)
        saved = {"model_id": model_id, "checkpoint_id": set_to, "action": "set_head"}
        report(
            saved,
            json_output=json_output,
            render=lambda row: (
                console.print(f"[green]Head set to[/green] [cyan]{row['checkpoint_id']}[/cyan]"),
                console.print(f"[dim]View: {_model_url(row['model_id'], tab='checkpoints')}[/dim]"),
            ),
        )
        return

    head = next((c for c in _get_checkpoints(model_id) if c.get("is_active")), None)
    report(head, json_output=json_output, render=lambda row: _render_head(model_id, row))


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
            raise map_request_error(exc, resource="Model", input={"model": model}) from exc
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
        raise map_request_error(exc, resource="Checkpoints", input={"model": model_id}) from exc


def _set_head(model_id: str, checkpoint_id: str) -> None:
    from hud.utils.exceptions import HudRequestError

    try:
        PlatformClient.from_settings().put(
            f"/models/{model_id}/head", json={"checkpoint_id": checkpoint_id}
        )
    except HudRequestError as exc:
        raise map_request_error(
            exc,
            resource="Checkpoint",
            input={"model": model_id, "checkpoint_id": checkpoint_id},
        ) from exc
