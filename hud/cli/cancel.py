"""Cancel remote rollouts (``hud cancel`` and ``hud jobs cancel``)."""

from __future__ import annotations

import asyncio
from typing import Any

import typer

from hud.cli.utils.output import (
    CliError,
    ExitCode,
    abort,
    confirm_or_abort,
    dry_run_option,
    emit_json,
    json_option,
    map_exception,
    output_option,
    resolve_output_mode,
    wants_json,
    yes_option,
)
from hud.utils.exceptions import HudException
from hud.utils.hud_console import HUDConsole


def run_cancel(
    *,
    job_id: str | None,
    trace_id: str | None,
    all_jobs: bool,
    yes: bool,
    dry_run: bool = False,
    json_output: bool = False,
    output: str | None = None,
) -> None:
    """Shared implementation for ``hud cancel`` and ``hud jobs cancel``."""
    hud_console = HUDConsole()

    if not job_id and not all_jobs:
        abort(
            CliError(
                error="usage",
                message="Provide a job_id or use --all to cancel all active jobs.",
                suggestion="hud jobs cancel <job-id>   or   hud jobs cancel --all --yes",
                exit_code=ExitCode.USAGE,
            ),
            json_output=json_output,
        )

    if job_id and all_jobs:
        abort(
            CliError(
                error="usage",
                message="Cannot specify both job_id and --all.",
                input={"job_id": job_id, "all": all_jobs},
                suggestion="Pass either a job id or --all, not both.",
                exit_code=ExitCode.USAGE,
            ),
            json_output=json_output,
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
        if resolve_output_mode(json_output=json_output, output=output) == "json" or wants_json(
            json_output, output
        ):
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
        from hud.cli.utils.jobs import cancel_all_jobs, cancel_job, cancel_task

        if all_jobs:
            hud_console.info("Cancelling all active jobs...")
            return await cancel_all_jobs()
        if trace_id:
            assert job_id is not None
            hud_console.info(f"Cancelling trace {trace_id} in job {job_id}...")
            return await cancel_task(job_id, trace_id)
        assert job_id is not None
        hud_console.info(f"Cancelling job {job_id}...")
        return await cancel_job(job_id)

    try:
        result = asyncio.run(_cancel())
    except HudException as exc:
        abort(map_exception(exc, input={"job_id": job_id, "trace_id": trace_id}))
    except Exception as exc:
        abort(
            CliError(
                error="failure",
                message=f"Failed to cancel: {exc}",
                input={"job_id": job_id, "trace_id": trace_id},
            )
        )

    payload: dict[str, Any] = {"action": action, "job_id": job_id, "trace_id": trace_id, **result}
    if wants_json(json_output, output):
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
    yes: bool = yes_option(),
    dry_run: bool = dry_run_option(),
    json_output: bool = json_option(),
    output: str | None = output_option(),
) -> None:
    """Cancel remote rollouts.

    Prefer ``hud jobs cancel`` in new scripts; this command is kept as an alias.

    [not dim]Examples:
        hud cancel <job_id>                 # Cancel all tasks in a job
        hud cancel <job_id> --trace-id <id> # Cancel specific task run
        hud cancel --all --yes              # Cancel ALL active jobs (panic button)
        hud cancel <job_id> --dry-run --json[/not dim]
    """
    run_cancel(
        job_id=job_id,
        trace_id=trace_id,
        all_jobs=all_jobs,
        yes=yes,
        dry_run=dry_run,
        json_output=json_output,
        output=output,
    )
