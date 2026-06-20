"""``hud jobs`` — list jobs and their traces."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

jobs_app = typer.Typer(
    name="jobs",
    help="List jobs and their traces",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=False,
)


@jobs_app.callback(invoke_without_command=True)
def jobs_command(
    ctx: typer.Context,
    job_id: str | None = typer.Argument(None, help="Job ID — omit to list recent jobs"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max rows to show"),
) -> None:
    """List recent jobs, or show traces for a specific job.

    Without an argument, lists the most recent jobs.
    With a job id, lists all traces for that job.
    """
    if ctx.invoked_subcommand is not None:
        return

    from hud.cli.utils.api import require_api_key

    require_api_key("list jobs")

    if job_id:
        _show_job_traces(job_id, json_output=json_output, limit=limit)
    else:
        _list_jobs(json_output=json_output, limit=limit)


# ── job listing ────────────────────────────────────────────────────────────────


def _list_jobs(*, json_output: bool, limit: int) -> None:
    from hud.utils.platform import PlatformClient

    client = PlatformClient.from_settings()
    try:
        data = client.get("/jobs", params={"limit": limit})
    except Exception as e:
        console.print(f"[red]Failed to fetch jobs: {e}[/red]")
        raise typer.Exit(1) from e

    items = data if isinstance(data, list) else (data.get("items") or [])

    if json_output:
        console.print_json(json.dumps(items, indent=2, default=str))
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
        jid = str(job.get("id") or "")
        table.add_row(
            jid,
            job.get("name") or "-",
            job.get("taskset_name") or "-",
            job.get("status") or "-",
            (str(job.get("created_at") or ""))[:19],
        )
    console.print(table)
    console.print(f"\n[dim]View: {web}/jobs[/dim]")
    console.print("[dim]Tip: hud jobs <id> to see traces for a specific job[/dim]")


# ── job traces ────────────────────────────────────────────────────────────────


def _show_job_traces(job_id: str, *, json_output: bool, limit: int) -> None:
    from hud.settings import settings
    from hud.utils.platform import PlatformClient

    client = PlatformClient.from_settings()
    try:
        data = client.get(f"/jobs/{job_id}/traces", params={"limit": limit})
    except Exception as e:
        console.print(f"[red]Failed to fetch traces: {e}[/red]")
        raise typer.Exit(1) from e

    items = data if isinstance(data, list) else (data.get("items") or [])

    if json_output:
        console.print_json(json.dumps(items, indent=2, default=str))
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
        tid = str(tr.get("id") or "")
        reward = tr.get("reward")
        table.add_row(
            tid,
            tr.get("status") or "-",
            f"{reward:.3f}" if reward is not None else "-",
            (str(tr.get("start_time") or tr.get("created_at") or ""))[:19],
            (tr.get("error") or "")[:40],
        )
    console.print(table)
    console.print(f"\n[dim]View: {web}/jobs/{job_id}[/dim]")
    console.print("[dim]Tip: hud trace <trace_id> to inspect a specific rollout[/dim]")
