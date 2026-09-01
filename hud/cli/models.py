"""``hud models`` — list gateway models and fork trainable ones."""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hud.cli.utils.output import (
    CliError,
    abort,
    dry_run_option,
    emit_json,
    emit_quiet,
    json_option,
    map_exception,
    map_request_error,
    output_option,
    quiet_option,
    resolve_output_mode,
    wants_json,
)

console = Console()

models_app = typer.Typer(
    name="models",
    help="List gateway models and fork trainable ones.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@models_app.command("list")
def list_models(
    json_output: bool = json_option(),
    output: str | None = output_option(),
    quiet: bool = quiet_option(),
) -> None:
    """List models available through the HUD inference gateway.

    The platform model catalog — the same models `create_agent` and `hud eval`
    resolve against.

    [not dim]Examples:
        hud models list
        hud models list --json
        hud models list --quiet[/not dim]
    """
    from hud.cli.utils.api import require_api_key
    from hud.settings import settings
    from hud.utils.exceptions import HudException
    from hud.utils.gateway import list_gateway_models

    require_api_key("list models")

    try:
        models_list = list_gateway_models()
    except Exception as exc:
        abort(
            map_exception(exc)
            if isinstance(exc, HudException)
            else CliError(
                error="failure",
                message=f"Failed to fetch models: {exc}",
            )
        )

    mode = resolve_output_mode(json_output=json_output, output=output, quiet=quiet)
    if mode == "json":
        emit_json([m.model_dump() for m in models_list])
        return
    if mode == "quiet":
        emit_quiet([m.model_name or m.id or "" for m in models_list if m.model_name or m.id])
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
    web = settings.hud_web_url.rstrip("/")
    console.print(f"[dim]View a model in the browser: {web}/models/<id>[/dim]")


@models_app.command("fork")
def fork_model(
    source: str = typer.Argument(..., help="Source model slug or id to fork from"),
    name: str = typer.Option(..., "--name", "-n", help="Name for the new trainable model"),
    json_output: bool = json_option(),
    output: str | None = output_option(),
    dry_run: bool = dry_run_option(),
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
    from hud.cli.utils.api import require_api_key
    from hud.settings import settings
    from hud.utils.exceptions import HudRequestError
    from hud.utils.requests import make_request_sync

    require_api_key("fork a model")

    if dry_run:
        payload = {
            "dry_run": True,
            "action": "fork",
            "source": source,
            "name": name,
            "if_not_exists": if_not_exists,
        }
        if wants_json(json_output, output):
            emit_json(payload)
        else:
            console.print(f"[dim]--dry-run: would fork {source!r} as {name!r}[/dim]")
        return

    source_id = _resolve_model_id(source)
    try:
        model = make_request_sync(
            "POST",
            f"{settings.hud_api_url}/v2/models/fork",
            json={"source_model_id": source_id, "name": name},
            api_key=settings.api_key,
        )
    except HudRequestError as exc:
        if exc.status_code == 409 and if_not_exists:
            existing = _existing_model(name)
            if wants_json(json_output, output):
                emit_json({**existing, "existed": True})
            else:
                slug = existing.get("model_name") or name
                console.print(f"[yellow]Model already exists[/yellow] [cyan]{slug}[/cyan]")
                console.print(f"[dim]id: {existing.get('id')}[/dim]")
            return
        abort(
            map_request_error(
                exc,
                resource="Model",
                input={"source": source, "name": name},
            )
        )
    except Exception as exc:
        abort(
            CliError(
                error="failure",
                message=f"Fork failed: {exc}",
                input={"source": source, "name": name},
            )
        )

    if wants_json(json_output, output):
        emit_json(model)
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
    console.print(f"[dim]View: {_model_url(model['id'])}[/dim]")


@models_app.command("checkpoints")
def list_checkpoints(
    model: str = typer.Argument(..., help="Model slug or id"),
    json_output: bool = json_option(),
    output: str | None = output_option(),
    quiet: bool = quiet_option(),
) -> None:
    """List a model's checkpoint tree, oldest first (▶ marks the active head).

    [not dim]Examples:
        hud models checkpoints <model>
        hud models checkpoints <model> --json
        hud models checkpoints <model> --quiet[/not dim]
    """
    from hud.cli.utils.api import require_api_key

    require_api_key("list checkpoints")
    model_id = _resolve_model_id(model)
    checkpoints = _get_checkpoints(model_id)
    mode = resolve_output_mode(json_output=json_output, output=output, quiet=quiet)

    if mode == "json":
        emit_json(checkpoints)
        return
    if mode == "quiet":
        emit_quiet([str(ckpt.get("id") or "") for ckpt in checkpoints if ckpt.get("id")])
        return
    if not checkpoints:
        console.print("[yellow]No checkpoints yet — this model serves its base weights[/yellow]")
        console.print(f"[dim]View: {_model_url(model_id, tab='checkpoints')}[/dim]")
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
            str(ckpt.get("created_at") or ""),
        )
    console.print(table)
    console.print(f"\n[dim]View: {_model_url(model_id, tab='checkpoints')}[/dim]")


@models_app.command("head")
def show_head(
    model: str = typer.Argument(..., help="Model slug or id"),
    set_to: str | None = typer.Option(
        None, "--set", help="Checkpoint id to promote to head (rollback / select)"
    ),
    json_output: bool = json_option(),
    output: str | None = output_option(),
    dry_run: bool = dry_run_option(),
) -> None:
    """Show — or with ``--set``, change — the model's active checkpoint (the
    weights the gateway serves now).

    [not dim]Examples:
        hud models head <model>
        hud models head <model> --json
        hud models head <model> --set <checkpoint-id> --dry-run --json[/not dim]
    """
    from hud.cli.utils.api import require_api_key

    require_api_key("manage head")
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
            if wants_json(json_output, output):
                emit_json(payload)
            else:
                console.print(f"[dim]--dry-run: would set head of {model} to {set_to}[/dim]")
            return
        _set_head(model_id, set_to)
        if wants_json(json_output, output):
            emit_json({"model_id": model_id, "checkpoint_id": set_to, "action": "set_head"})
            return
        console.print(f"[green]Head set to[/green] [cyan]{set_to}[/cyan]")
        console.print(f"[dim]View: {_model_url(model_id, tab='checkpoints')}[/dim]")
        return

    head = next((c for c in _get_checkpoints(model_id) if c.get("is_active")), None)

    if wants_json(json_output, output):
        emit_json(head)
        return
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


def _model_url(model_id: str, *, tab: str | None = None) -> str:
    """Web app URL for a model (optionally a specific tab, e.g. ``checkpoints``)."""
    from hud.settings import settings

    url = f"{settings.hud_web_url.rstrip('/')}/models/{model_id}"
    return f"{url}?tab={tab}" if tab else url


def _resolve_model_id(model: str) -> str:
    """Map a model slug to its id (an id passes straight through)."""
    from hud.settings import settings
    from hud.utils.exceptions import HudRequestError
    from hud.utils.requests import make_request_sync

    try:
        return str(UUID(model))
    except ValueError:
        from urllib.parse import quote

        try:
            data = make_request_sync(
                "GET",
                f"{settings.hud_api_url}/v2/models/resolve?model={quote(model, safe='')}",
                api_key=settings.api_key,
            )
        except HudRequestError as exc:
            abort(map_request_error(exc, resource="Model", input={"model": model}))
        return str(data["id"])


def _existing_model(name: str) -> dict[str, Any]:
    """Resolve a model name after a conflict so ``--if-not-exists`` can return it."""
    model_id = _resolve_model_id(name)
    return {"id": model_id, "model_name": name}


def _get_checkpoints(model: str) -> list[dict[str, Any]]:
    from hud.settings import settings
    from hud.utils.exceptions import HudRequestError
    from hud.utils.requests import make_request_sync

    model_id = _resolve_model_id(model)
    try:
        return cast(
            "list[dict[str, Any]]",
            make_request_sync(
                "GET",
                f"{settings.hud_api_url}/v2/models/{model_id}/checkpoints",
                api_key=settings.api_key,
            ),
        )
    except HudRequestError as exc:
        abort(map_request_error(exc, resource="Checkpoints", input={"model": model}))
    except Exception as exc:
        abort(CliError(error="failure", message=f"Failed to fetch checkpoints: {exc}"))


def _set_head(model: str, checkpoint_id: str) -> None:
    from hud.settings import settings
    from hud.utils.exceptions import HudRequestError
    from hud.utils.requests import make_request_sync

    model_id = _resolve_model_id(model)
    try:
        make_request_sync(
            "PUT",
            f"{settings.hud_api_url}/v2/models/{model_id}/head",
            json={"checkpoint_id": checkpoint_id},
            api_key=settings.api_key,
        )
    except HudRequestError as exc:
        abort(
            map_request_error(
                exc,
                resource="Checkpoint",
                input={"model": model, "checkpoint_id": checkpoint_id},
            )
        )
    except Exception as exc:
        abort(CliError(error="failure", message=f"Failed to set head: {exc}"))
