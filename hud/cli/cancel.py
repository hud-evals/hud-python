"""Cancel remote rollouts (``hud jobs cancel``; ``hud cancel`` is a hidden alias)."""

from __future__ import annotations

import asyncio
from typing import Any

import typer

from hud.cli.io import (
    CliError,
    confirm_or_abort,
    emit_json,
    map_exception,
    mark_json,
)
from hud.utils.exceptions import HudException
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient


def run_cancel(
    *,
    job_id: str | None,
    trace_id: str | None,
    all_jobs: bool,
    yes: bool,
    dry_run: bool = False,
    json_output: bool = False,
) -> None:
    """Shared implementation for ``hud jobs cancel`` and the hidden ``hud cancel`` alias."""
    hud_console = HUDConsole()

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

    if all_jobs:
        action = "cancel_all"
    elif job_id and not trace_id:
        action = "cancel_job"
    else:
        action = "cancel_trace"

    plan: dict[str, Any] = {
        "dry_run": True,
        "action": action,
        "job_id": job_id,
        "trace_id": trace_id,
        "all": all_jobs,
    }
    if dry_run:
        if json_output is True:
            emit_json(plan)
        else:
            hud_console.info(f"--dry-run: would {action.replace('_', ' ')}")
            if job_id:
                hud_console.info(f"  job_id: {job_id}")
            if trace_id:
                hud_console.info(f"  trace_id: {trace_id}")
        return

    if all_jobs:
        confirm_or_abort(
            "This will cancel ALL your active jobs. Continue?",
            yes=yes,
            default=False,
        )
    elif job_id and not trace_id:
        confirm_or_abort(f"Cancel all tasks in job {job_id}?", yes=yes, default=False)

    async def _cancel() -> dict[str, Any]:
        platform = PlatformClient.from_settings()
        if all_jobs:
            hud_console.info("Cancelling all active jobs...")
            return await platform.apost("/rollouts/cancel_user_jobs", json={})
        if trace_id:
            assert job_id is not None
            hud_console.info(f"Cancelling trace {trace_id} in job {job_id}...")
            return await platform.apost(
                "/rollouts/cancel", json={"job_id": job_id, "trace_id": trace_id}
            )
        assert job_id is not None
        hud_console.info(f"Cancelling job {job_id}...")
        return await platform.apost("/rollouts/cancel_job", json={"job_id": job_id})

    try:
        result = asyncio.run(_cancel())
    except HudException as exc:
        raise map_exception(exc, input={"job_id": job_id, "trace_id": trace_id}) from exc

    payload: dict[str, Any] = {"action": action, "job_id": job_id, "trace_id": trace_id, **result}
    if json_output is True:
        emit_json(payload)
        return

    if all_jobs:
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
        return

    if trace_id:
        if result.get("status") == "accepted":
            hud_console.success("Task cancellation requested.")
        else:
            hud_console.warning("Task not found or already finished.")
        return

    cancelled = result.get("cancelled", 0)
    if cancelled == 0:
        hud_console.warning(f"No active tasks found for job {job_id}")
    else:
        hud_console.success(f"Cancellation requested for {cancelled} task(s).")


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
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts (required in non-interactive terminals).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the planned action without making changes."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Write JSON to stdout.", callback=mark_json, is_eager=True
    ),
) -> None:
    """Deprecated. Use ``hud jobs cancel``.

    [not dim]Examples:
        hud jobs cancel <job_id>
        hud jobs cancel <job_id> --trace-id <id>
        hud jobs cancel --all --yes[/not dim]
    """
    run_cancel(
        job_id=job_id,
        trace_id=trace_id,
        all_jobs=all_jobs,
        yes=yes,
        dry_run=dry_run,
        json_output=json_output,
    )
