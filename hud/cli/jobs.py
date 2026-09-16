"""``hud jobs`` — list jobs, inspect traces, and cancel rollouts.

Noun-verb surface:

    hud jobs list              # recent jobs
    hud jobs get <id>          # traces for one job
    hud jobs cancel <id>       # cancel a job

``hud jobs`` is an alias for ``hud jobs list``.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import typer
from rich.panel import Panel
from rich.table import Table

from hud.cli import (
    CLI,
    CliError,
    map_exception,
)
from hud.settings import settings
from hud.utils.exceptions import HudException, HudRequestError
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient

hud_console = HUDConsole()

jobs_app = CLI(
    name="jobs",
    help="List jobs, inspect their traces, and cancel rollouts.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=False,
)


@jobs_app.command("list")
def list_command(
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Print one identifier per line, with no headers (for piping)."
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Max rows to show"),
) -> Any:
    """List recent jobs.

    [not dim]Examples:
        hud jobs list
        hud jobs list --json
        hud jobs list --quiet | xargs -n1 hud jobs get
        hud jobs list -n 50[/not dim]
    """
    items = PlatformClient.from_settings().get("/jobs", params={"limit": limit})["items"]
    if quiet:
        for job in items:
            if job.get("id"):
                typer.echo(job["id"])
        return items
    if not items:
        hud_console.stdout.print("[yellow]No jobs found.[/yellow]")
        return items

    hud_console.stdout.print(Panel.fit("[bold cyan]Recent Jobs[/bold cyan]", border_style="cyan"))
    table = Table()
    table.add_column("ID", style="blue", no_wrap=True)
    table.add_column("Name", style="cyan")
    table.add_column("Taskset", style="dim")
    table.add_column("Status", style="yellow")
    table.add_column("Created", style="dim")
    for job in items:
        table.add_row(
            str(job.get("id") or ""),
            job.get("name") or "-",
            job.get("taskset_name") or "-",
            job.get("status") or "-",
            str(job.get("created_at") or ""),
        )
    hud_console.stdout.print(table)
    web = settings.hud_web_url.rstrip("/")
    hud_console.stdout.print(f"\n[dim]View: {web}/jobs[/dim]")
    hud_console.stdout.print("[dim]Tip: hud jobs get <id> to see traces for a specific job[/dim]")
    return items


@jobs_app.command("get")
def get_command(
    job_id: str = typer.Argument(..., help="Job ID"),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Print one identifier per line, with no headers (for piping)."
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Max rows to show"),
) -> Any:
    """Show traces for a specific job.

    [not dim]Examples:
        hud jobs get <job-id>
        hud jobs get <job-id> --json
        hud jobs get <job-id> --quiet[/not dim]
    """
    client = PlatformClient.from_settings()
    job_id = str(UUID(job_id))
    try:
        data = client.get(f"/jobs/{job_id}/traces", params={"limit": limit})
    except HudRequestError as exc:
        raise CliError.from_http(exc, resource="Job", input={"job_id": job_id}) from exc
    items = data["items"]
    if quiet:
        for trace in items:
            if trace.get("id"):
                typer.echo(trace["id"])
        return items

    view = f"{settings.hud_web_url.rstrip('/')}/jobs/{job_id}"
    if not items:
        hud_console.stdout.print("[yellow]No traces found for this job.[/yellow]")
        hud_console.stdout.print(f"[dim]View: {view}[/dim]")
        return items

    hud_console.stdout.print(
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
    hud_console.stdout.print(table)
    hud_console.stdout.print(f"\n[dim]View: {view}[/dim]")
    hud_console.stdout.print(
        "[dim]Tip: hud trace get <trace_id> to inspect a specific rollout[/dim]"
    )
    return items


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
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts (required in non-interactive terminals).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the planned action without making changes."
    ),
) -> Any:
    """Cancel remote rollouts for a job, a trace, or every active job.

    [not dim]Examples:
        hud jobs cancel <job-id>
        hud jobs cancel <job-id> --trace-id <trace-id> --json
        hud jobs cancel --all --yes
        hud jobs cancel <job-id> --dry-run --json[/not dim]
    """
    if not job_id and not all_jobs:
        raise CliError(
            error="usage",
            message="Provide a job_id or use --all to cancel all active jobs.",
            suggestion="hud jobs cancel <job-id>   or   hud jobs cancel --all --yes",
        )
    if job_id and all_jobs:
        raise CliError(
            error="usage",
            message="Cannot specify both job_id and --all.",
            input={"job_id": job_id, "all": all_jobs},
            suggestion="Pass either a job id or --all, not both.",
        )

    action = "cancel_all" if all_jobs else "cancel_trace" if trace_id else "cancel_job"
    if dry_run:
        hud_console.info(f"--dry-run: would {action.replace('_', ' ')}")
        if job_id:
            hud_console.info(f"  job_id: {job_id}")
        if trace_id:
            hud_console.info(f"  trace_id: {trace_id}")
        return {
            "dry_run": True,
            "action": action,
            "job_id": job_id,
            "trace_id": trace_id,
            "all": all_jobs,
        }

    platform = PlatformClient.from_settings()
    try:
        if all_jobs:
            CLI.confirm_or_abort(
                "This will cancel ALL your active jobs. Continue?", yes=yes, default=False
            )
            hud_console.info("Cancelling all active jobs...")
            result = platform.post("/rollouts/cancel_user_jobs", json={})
            jobs_cancelled = result.get("jobs_cancelled", 0)
            if jobs_cancelled == 0:
                hud_console.info("No active jobs found.")
            else:
                hud_console.success(
                    f"Cancelled {jobs_cancelled} job(s), "
                    f"{result.get('total_tasks_cancelled', 0)} task(s) total."
                )
                for job in result.get("job_details", []):
                    hud_console.info(f"  • {job['job_id']}: {job['cancelled']} tasks cancelled")
        elif trace_id:
            hud_console.info(f"Cancelling trace {trace_id} in job {job_id}...")
            result = platform.post(
                "/rollouts/cancel", json={"job_id": job_id, "trace_id": trace_id}
            )
            if result.get("status") == "accepted":
                hud_console.success("Task cancellation requested.")
            else:
                hud_console.warning("Task not found or already finished.")
        else:
            CLI.confirm_or_abort(f"Cancel all tasks in job {job_id}?", yes=yes, default=False)
            hud_console.info(f"Cancelling job {job_id}...")
            result = platform.post("/rollouts/cancel_job", json={"job_id": job_id})
            cancelled = result.get("cancelled", 0)
            if cancelled == 0:
                hud_console.warning(f"No active tasks found for job {job_id}")
            else:
                hud_console.success(f"Cancellation requested for {cancelled} task(s).")
    except HudException as exc:
        raise map_exception(exc, input={"job_id": job_id, "trace_id": trace_id}) from exc
    return {"action": action, "job_id": job_id, "trace_id": trace_id, **result}


@jobs_app.callback(invoke_without_command=True)
def jobs_command(
    ctx: typer.Context,
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Print one identifier per line, with no headers (for piping)."
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Max rows to show"),
) -> Any:
    """List recent jobs.

    Without a verb, lists the most recent jobs. ``hud jobs`` is an alias for
    ``hud jobs list``.

    [not dim]Examples:
        hud jobs
        hud jobs list --json
        hud jobs get <job-id>
        hud jobs cancel <job-id> --yes[/not dim]
    """
    if ctx.invoked_subcommand is not None:
        return None
    return list_command(quiet=quiet, limit=limit)
