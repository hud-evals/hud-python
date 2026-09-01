"""``hud jobs`` — list jobs, inspect traces, and cancel rollouts.

Noun-verb surface:

    hud jobs list              # recent jobs
    hud jobs get <id>          # traces for one job
    hud jobs cancel <id>       # cancel a job (also ``hud cancel``)

``hud jobs`` and ``hud jobs <id>`` remain as backward-compatible shortcuts.
"""

from __future__ import annotations

from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hud.cli.utils.output import (
    UnknownTokenAsGetGroup,
    dry_run_option,
    emit_json,
    emit_quiet,
    json_option,
    output_option,
    platform_call,
    quiet_option,
    resolve_output_mode,
    yes_option,
)

console = Console()

jobs_app = typer.Typer(
    name="jobs",
    cls=UnknownTokenAsGetGroup,
    help="List jobs, inspect their traces, and cancel rollouts.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=False,
)


def _items(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        items = data.get("items")
        if isinstance(items, list):
            return items
    return []


def _list_jobs(*, json_output: bool, output: str | None, quiet: bool, limit: int) -> None:
    from hud.cli.utils.api import require_api_key
    from hud.utils.platform import PlatformClient

    require_api_key("list jobs")
    client = PlatformClient.from_settings()
    data = platform_call(
        lambda: client.get("/jobs", params={"limit": limit}),
        resource="Jobs",
    )
    items = _items(data)
    mode = resolve_output_mode(json_output=json_output, output=output, quiet=quiet)

    if mode == "json":
        emit_json(items)
        return
    if mode == "quiet":
        emit_quiet([str(job.get("id") or "") for job in items if job.get("id")])
        return

    if not items:
        console.print("[yellow]No jobs found.[/yellow]")
        return

    console.print(Panel.fit("[bold cyan]Recent Jobs[/bold cyan]", border_style="cyan"))
    table = Table()
    table.add_column("ID", style="blue", no_wrap=True)
    table.add_column("Name", style="cyan")
    table.add_column("Taskset", style="dim")
    table.add_column("Status", style="yellow")
    table.add_column("Created", style="dim")

    from hud.settings import settings

    web = settings.hud_web_url.rstrip("/")

    for job in items:
        table.add_row(
            str(job.get("id") or ""),
            job.get("name") or "-",
            job.get("taskset_name") or "-",
            job.get("status") or "-",
            str(job.get("created_at") or ""),
        )
    console.print(table)
    console.print(f"\n[dim]View: {web}/jobs[/dim]")
    console.print("[dim]Tip: hud jobs get <id> to see traces for a specific job[/dim]")


def _show_job_traces(
    job_id: str,
    *,
    json_output: bool,
    output: str | None,
    quiet: bool,
    limit: int,
) -> None:
    from hud.cli.utils.api import require_api_key
    from hud.settings import settings
    from hud.utils.platform import PlatformClient

    require_api_key("list job traces")
    client = PlatformClient.from_settings()
    data = platform_call(
        lambda: client.get(f"/jobs/{job_id}/traces", params={"limit": limit}),
        resource="Job",
        input={"job_id": job_id},
    )
    items = _items(data)
    mode = resolve_output_mode(json_output=json_output, output=output, quiet=quiet)

    if mode == "json":
        emit_json(items)
        return
    if mode == "quiet":
        emit_quiet([str(tr.get("id") or "") for tr in items if tr.get("id")])
        return

    web = settings.hud_web_url.rstrip("/")

    if not items:
        console.print("[yellow]No traces found for this job.[/yellow]")
        console.print(f"[dim]View: {web}/jobs/{job_id}[/dim]")
        return

    console.print(
        Panel.fit(f"[bold cyan]Job Traces[/bold cyan] [dim]{job_id}[/dim]", border_style="cyan")
    )
    table = Table()
    table.add_column("Trace ID", style="blue", no_wrap=True)
    table.add_column("Status", style="yellow")
    table.add_column("Reward", style="green", justify="right")
    table.add_column("Started", style="dim")
    table.add_column("Error", style="red")

    for tr in items:
        reward = tr.get("reward")
        table.add_row(
            str(tr.get("id") or ""),
            tr.get("status") or "-",
            f"{reward:.3f}" if reward is not None else "-",
            str(tr.get("start_time") or tr.get("created_at") or ""),
            (tr.get("error") or "")[:40],
        )
    console.print(table)
    console.print(f"\n[dim]View: {web}/jobs/{job_id}[/dim]")
    console.print("[dim]Tip: hud trace get <trace_id> to inspect a specific rollout[/dim]")


@jobs_app.command("list")
def list_command(
    json_output: bool = json_option(),
    output: str | None = output_option(),
    quiet: bool = quiet_option(),
    limit: int = typer.Option(20, "--limit", "-n", help="Max rows to show"),
) -> None:
    """List recent jobs.

    [not dim]Examples:
        hud jobs list
        hud jobs list --json
        hud jobs list --quiet | xargs -n1 hud jobs get
        hud jobs list -n 50[/not dim]
    """
    _list_jobs(json_output=json_output, output=output, quiet=quiet, limit=limit)


@jobs_app.command("get")
def get_command(
    job_id: str = typer.Argument(..., help="Job ID"),
    json_output: bool = json_option(),
    output: str | None = output_option(),
    quiet: bool = quiet_option(),
    limit: int = typer.Option(20, "--limit", "-n", help="Max rows to show"),
) -> None:
    """Show traces for a specific job.

    [not dim]Examples:
        hud jobs get <job-id>
        hud jobs get <job-id> --json
        hud jobs get <job-id> --quiet[/not dim]
    """
    _show_job_traces(job_id, json_output=json_output, output=output, quiet=quiet, limit=limit)


@jobs_app.command("cancel")
def cancel_job_command(
    job_id: str | None = typer.Argument(
        None, help="Job ID to cancel. Omit to cancel all active jobs with --all."
    ),
    trace_id: str | None = typer.Option(
        None, "--trace-id", "-t", help="Specific trace ID within the job to cancel."
    ),
    all_jobs: bool = typer.Option(
        False, "--all", "-a", help="Cancel ALL active jobs for your account (panic button)."
    ),
    yes: bool = yes_option(),
    dry_run: bool = dry_run_option(),
    json_output: bool = json_option(),
    output: str | None = output_option(),
) -> None:
    """Cancel remote rollouts for a job, a trace, or every active job.

    [not dim]Examples:
        hud jobs cancel <job-id>
        hud jobs cancel <job-id> --trace-id <trace-id> --json
        hud jobs cancel --all --yes
        hud jobs cancel <job-id> --dry-run --json[/not dim]
    """
    from hud.cli.cancel import run_cancel

    run_cancel(
        job_id=job_id,
        trace_id=trace_id,
        all_jobs=all_jobs,
        yes=yes,
        dry_run=dry_run,
        json_output=json_output,
        output=output,
    )


@jobs_app.callback(invoke_without_command=True)
def jobs_command(
    ctx: typer.Context,
    json_output: bool = json_option(),
    output: str | None = output_option(),
    quiet: bool = quiet_option(),
    limit: int = typer.Option(20, "--limit", "-n", help="Max rows to show"),
) -> None:
    """List recent jobs, or show traces for a specific job.

    Without a verb, lists the most recent jobs.
    ``hud jobs <id>`` is rewritten to ``hud jobs get <id>``.

    [not dim]Examples:
        hud jobs
        hud jobs list --json
        hud jobs get <job-id>
        hud jobs <job-id>
        hud jobs cancel <job-id> --yes[/not dim]
    """
    if ctx.invoked_subcommand is not None:
        return
    _list_jobs(json_output=json_output, output=output, quiet=quiet, limit=limit)
