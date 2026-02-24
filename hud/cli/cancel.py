"""Cancel remote rollouts."""

from __future__ import annotations

import asyncio

import httpx
import questionary
import typer

from hud.utils.hud_console import HUDConsole


def cancel_command(
    job_id: str | None = typer.Argument(
        None, help="Job ID to cancel. Omit to cancel all active jobs with --all."
    ),
    trace_id: str | None = typer.Option(
        None, "--trace-id", "-t", help="Specific trace ID within the job to cancel."
    ),
    all_jobs: bool = typer.Option(
        False, "--all", "-a", help="Cancel ALL active jobs for your account (panic button)."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Cancel remote rollouts.

    Examples:
        hud cancel <job_id>                 # Cancel all tasks in a job
        hud cancel <job_id> --trace-id <id> # Cancel specific task run
        hud cancel --all                 # Cancel ALL active jobs (panic button)
    """
    hud_console = HUDConsole()

    if not job_id and not all_jobs:
        hud_console.error("Provide a job_id or use --all to cancel all active jobs.")
        raise typer.Exit(1)

    if job_id and all_jobs:
        hud_console.error("Cannot specify both job_id and --all.")
        raise typer.Exit(1)

    if (
        all_jobs
        and not yes
        and not questionary.confirm(
            "⚠️  This will cancel ALL your active jobs. Continue?",
            default=False,
        ).ask()
    ):
        hud_console.info("Cancelled.")
        raise typer.Exit(0)

    if (
        job_id
        and not trace_id
        and not yes
        and not questionary.confirm(
            f"Cancel all tasks in job {job_id}?",
            default=True,
        ).ask()
    ):
        hud_console.info("Cancelled.")
        raise typer.Exit(0)

    async def _cancel() -> None:
        from hud.datasets.utils import cancel_all_jobs, cancel_job, cancel_task

        if all_jobs:
            hud_console.info("Cancelling all active jobs...")
            result = await cancel_all_jobs()

            jobs_cancelled = result.get("jobs_cancelled", 0)
            tasks_cancelled = result.get("total_tasks_cancelled", 0)

            if jobs_cancelled == 0:
                hud_console.info("No active jobs found.")
            else:
                hud_console.success(
                    f"Cancelled {jobs_cancelled} job(s), {tasks_cancelled} task(s) total."
                )
                for job in result.get("job_details", []):
                    hud_console.info(f"  • {job['job_id']}: {job['cancelled']} tasks cancelled")

        elif trace_id:
            hud_console.info(f"Cancelling trace {trace_id} in job {job_id}...")
            result = await cancel_task(job_id, trace_id)  # type: ignore[arg-type]

            status = result.get("status", "unknown")
            if status in ("revoked", "terminated"):
                hud_console.success(f"Task cancelled: {result.get('message', '')}")
            elif status == "not_found":
                hud_console.warning(f"Task not found: {result.get('message', '')}")
            else:
                hud_console.info(f"Status: {status} - {result.get('message', '')}")

        else:
            hud_console.info(f"Cancelling job {job_id}...")
            result = await cancel_job(job_id)  # type: ignore[arg-type]

            total = result.get("total_found", 0)
            cancelled = result.get("cancelled", 0)

            if total == 0:
                hud_console.warning(f"No tasks found for job {job_id}")
            else:
                hud_console.success(
                    f"Cancelled {cancelled}/{total} tasks "
                    f"({result.get('running_terminated', 0)} running, "
                    f"{result.get('queued_revoked', 0)} queued)"
                )

    try:
        asyncio.run(_cancel())
    except httpx.HTTPStatusError as e:
        hud_console.error(f"API error: {e.response.status_code} - {e.response.text}")
        raise typer.Exit(1) from e
    except Exception as e:
        hud_console.error(f"Failed to cancel: {e}")
        raise typer.Exit(1) from e
