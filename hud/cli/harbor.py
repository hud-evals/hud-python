"""``hud harbor`` — import Harbor tasks and job runs into HUD tasksets.

hud harbor import <folders...>   # import into a taskset and show the report
hud harbor imports               # recent imports
hud harbor report <run-id>       # one import's report
hud harbor cancel <run-id>       # cancel an import
"""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich.filesize import decimal
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

from hud import data
from hud.cli import CLI, AuthScope, CliError, DirectoryLink, DirectoryState, Result
from hud.cli.project import Placement, Project
from hud.eval.sync import resolve_taskset_id
from hud.integrations.harbor.importing import (
    PIPELINE_KEY,
    REPORT_OUTPUT,
    REPORT_SCHEMA,
    ImportReport,
    pack_bundle,
)
from hud.settings import settings
from hud.utils.exceptions import HudRequestError
from hud.utils.hud_console import HUDConsole
from hud.utils.naming import normalize_environment_name
from hud.utils.platform import PlatformClient

hud_console = HUDConsole()

harbor_app = CLI(
    name="harbor",
    help="Import Harbor tasks and job runs into HUD tasksets.",
    add_completion=False,
    rich_markup_mode="rich",
)

_STATUS_STYLE = {
    "imported": "green",
    "unchanged": "dim",
    "skipped": "yellow",
    "failed": "red",
}


@dataclass(frozen=True)
class _Target:
    """The taskset an import writes to."""

    name: str
    # None for a taskset the import creates, in ``placement``.
    id: str | None
    placement: Placement | None = None


@harbor_app.command("import")
def import_command(
    paths: list[Path] = typer.Argument(  # noqa: B008
        ...,
        help="Harbor task, dataset, or job folders, or one .zip of them",
    ),
    taskset: str | None = typer.Option(
        None,
        "--taskset",
        "-t",
        help=(
            "Taskset name or ID to import into; a new name creates it. Defaults to the "
            "directory's taskset, then a taskset named after the first folder."
        ),
    ),
    project: str | None = typer.Option(
        None,
        "--project",
        help=(
            "Project name or ID for a taskset this command creates. Defaults to the "
            "directory's saved project, HUD_DEFAULT_PROJECT, then your team default."
        ),
    ),
    link_target: bool = typer.Option(
        False, "--link", help="Save the taskset as the directory's default"
    ),
    wait: bool = typer.Option(
        True, "--wait/--no-wait", help="Follow the import until it ends and show its report"
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
    """Import Harbor task folders and job runs into a taskset.

    [not dim]Each task folder becomes an environment and a task of the taskset; each job
    folder becomes a HUD job whose trials are traces of the tasks they ran. Runs of
    tasks imported earlier link to them, so a job can be imported on its own.
    Importing the same folders again resumes an import that stopped.

    Examples:
        hud harbor import ./tasks ./jobs/2026-10-01__12-00 --taskset tb2
        hud harbor import ./jobs/2026-10-02__09-00          # into the directory's taskset
        hud harbor import bundle.zip -t tb2 --no-wait --json
        hud harbor import ./tasks --dry-run[/not dim]
    """
    archive = _given_archive(paths)
    platform = PlatformClient.from_settings()
    state = DirectoryState(AuthScope.resolve(platform))
    link = state.load()
    pipeline = _pipeline(platform)
    default_name = archive.stem if archive else paths[0].resolve().name
    target = _resolve_target(platform, link, taskset=taskset, project=project, name=default_name)

    with tempfile.TemporaryDirectory(prefix="hud-harbor-") as workdir:
        if archive is None:
            bundle = Path(workdir) / f"{target.name}.zip"
            hud_console.progress_message("Packing the bundle...")
            file_count: int | None = pack_bundle(paths, bundle)
        else:
            bundle, file_count = archive, None
        size = bundle.stat().st_size
        if size > pipeline.max_input_bytes:
            raise CliError(
                "usage",
                f"The bundle is {decimal(size)}; a Harbor import takes at most "
                f"{decimal(pipeline.max_input_bytes)}.",
                suggestion=(
                    "Import the folders in several runs into the same taskset; runs link "
                    "to tasks imported earlier."
                ),
            )
        plan = {
            "taskset": {"id": target.id, "name": target.name, "created": target.id is None},
            "bundle": {"filename": bundle.name, "files": file_count, "size_bytes": size},
        }
        _print_plan(target, bundle, file_count, size)
        if dry_run:
            hud_console.info("\n  --dry-run: nothing was uploaded")
            return {**plan, "dry_run": True}

        CLI.confirm_or_abort("Start the import?", yes=yes, default=False)
        if target.id is None:
            target = _create_taskset(platform, target)
            plan["taskset"]["id"] = target.id
            hud_console.success(f"Created taskset {target.name}")
        assert target.id is not None
        if link_target:
            state.update(DirectoryLink(taskset_id=uuid.UUID(target.id)))
        stored = _upload(platform, bundle)

    run = data.start_run(platform, PIPELINE_KEY, stored.id, {"taskset_id": target.id})
    hud_console.success(f"Started import {run.id}")
    if not wait:
        hud_console.hint(f"Follow it with: hud harbor report {run.id} --wait")
        return {**plan, "run": run.model_dump(mode="json"), "report": None}
    return _conclude(platform, run, plan)


@harbor_app.command("imports")
def imports_command(
    taskset: str | None = typer.Option(
        None, "--taskset", "-t", help="Only imports into this taskset (name or ID)"
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Max rows to show"),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Print one identifier per line, with no headers (for piping)."
    ),
) -> Any:
    """List recent Harbor imports, newest first.

    [not dim]Examples:
        hud harbor imports
        hud harbor imports --taskset tb2
        hud harbor imports --json[/not dim]
    """
    platform = PlatformClient.from_settings()
    args = None
    if taskset is not None:
        taskset_id, _ = resolve_taskset_id(platform, taskset)
        if not taskset_id:
            raise CliError("not_found", f"Taskset not found: {taskset}", input={"taskset": taskset})
        args = {"taskset_id": taskset_id}
    runs, total = data.list_runs(platform, pipeline_key=PIPELINE_KEY, args=args, limit=limit)
    payload = {"runs": [run.model_dump(mode="json") for run in runs], "total": total}
    if quiet:
        for run in runs:
            typer.echo(run.id)
        return payload
    if not runs:
        hud_console.stdout.print("[yellow]No Harbor imports found.[/yellow]")
        return payload

    table = Table()
    table.add_column("Run ID", style="blue", no_wrap=True)
    table.add_column("Bundle", style="cyan")
    table.add_column("Status")
    table.add_column("Started", style="dim")
    for run in runs:
        table.add_row(
            run.id,
            run.input_filename,
            run.phase,
            run.created_at.strftime("%Y-%m-%d %H:%M"),
        )
    hud_console.stdout.print(table)
    if total > len(runs):
        hud_console.stdout.print(
            f"[dim]Showing {len(runs)} of {total}; raise --limit for more[/dim]"
        )
    hud_console.stdout.print("[dim]Tip: hud harbor report <run-id> to see what an import did[/dim]")
    return payload


@harbor_app.command("report")
def report_command(
    run_id: str = typer.Argument(..., help="Import run ID (see hud harbor imports)"),
    wait: bool = typer.Option(False, "--wait", help="Follow a running import until it ends"),
) -> Any:
    """Show what a Harbor import did with each task, job, and trial.

    [not dim]Examples:
        hud harbor report <run-id>
        hud harbor report <run-id> --wait
        hud harbor report <run-id> --json[/not dim]
    """
    platform = PlatformClient.from_settings()
    run = _get_import(platform, run_id)
    if not run.finished and not wait:
        hud_console.info(f"The import is {run.phase}; its report is published when it ends.")
        hud_console.hint(f"Follow it with: hud harbor report {run.id} --wait")
        return {"run": run.model_dump(mode="json"), "report": None}
    return _conclude(platform, run, {})


@harbor_app.command("cancel")
def cancel_command(
    run_id: str = typer.Argument(..., help="Import run ID (see hud harbor imports)"),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts (required in non-interactive terminals).",
    ),
) -> Any:
    """Cancel a Harbor import.

    [not dim]What it imported before stopping stays in HUD; importing the same
    folders again resumes it.

    Examples:
        hud harbor cancel <run-id>
        hud harbor cancel <run-id> --yes --json[/not dim]
    """
    platform = PlatformClient.from_settings()
    run = _get_import(platform, run_id)
    if run.finished:
        hud_console.info(f"The import already ended: {run.status}")
        return run.model_dump(mode="json")
    CLI.confirm_or_abort(f"Cancel import {run.id}?", yes=yes, default=False)
    run = data.cancel_run(platform, run.id)
    hud_console.success("Cancellation requested")
    return run.model_dump(mode="json")


def _pipeline(platform: PlatformClient) -> data.DataPipeline:
    for pipeline in data.list_pipelines(platform):
        if pipeline.key == PIPELINE_KEY:
            return pipeline
    raise CliError(
        "not_found",
        f"Harbor import is not available on {platform.api_url}.",
        suggestion="Check HUD_API_URL; the platform has to offer the harbor_import pipeline.",
    )


def _given_archive(paths: list[Path]) -> Path | None:
    """The zip to import as it is, when the command was given one."""
    archives = [path for path in paths if path.suffix.lower() == ".zip"]
    if not archives:
        return None
    if len(paths) > 1:
        raise CliError(
            "usage",
            "Pass either one .zip bundle or folders, not both or several zips.",
            input={"paths": [str(path) for path in paths]},
        )
    if not archives[0].is_file():
        raise FileNotFoundError(f"No such file: {archives[0]}")
    return archives[0]


def _resolve_target(
    platform: PlatformClient,
    link: DirectoryLink,
    *,
    taskset: str | None,
    project: str | None,
    name: str,
) -> _Target:
    """The taskset named by ``--taskset``, the directory, or ``name``; a new name is created."""
    ref = taskset or (str(link.taskset_id) if link.taskset_id else name)
    taskset_id, display = resolve_taskset_id(platform, ref)
    if not taskset_id:
        new_name = normalize_environment_name(display, default="")
        if not new_name:
            raise CliError(
                "usage",
                f"{display!r} cannot name a taskset.",
                suggestion="Pass --taskset <name>.",
            )
        placement = Placement.resolve(platform, link, flag=project)
        placement.require_writable()
        return _Target(name=new_name, id=None, placement=placement)

    try:
        record = platform.get(f"/tasksets/{taskset_id}")
    except HudRequestError as exc:
        raise CliError.from_http(exc, resource="Taskset", input={"taskset": ref}) from exc
    if not record["can_edit"]:
        raise CliError(
            "permission_denied",
            f"You cannot edit taskset {record['name']!r}, so nothing can be imported into it.",
            input={"taskset": ref},
        )
    if project is not None and Project.resolve(platform, project).id != str(record["project_id"]):
        raise CliError(
            "usage",
            f"Taskset {record['name']!r} is in another Project; --project only places a "
            "taskset this command creates.",
            input={"taskset": ref, "project": project},
        )
    return _Target(name=str(record["name"]), id=taskset_id)


def _print_plan(target: _Target, bundle: Path, file_count: int | None, size: int) -> None:
    if target.placement is not None:
        taskset = f"{target.name} (new, in {target.placement.label})"
    else:
        taskset = f"{target.name} ({target.id})"
    contents = f"{file_count} files, {decimal(size)}" if file_count is not None else decimal(size)
    hud_console.key_value_table({"Taskset": taskset, "Bundle": f"{bundle.name} ({contents})"})


def _create_taskset(platform: PlatformClient, target: _Target) -> _Target:
    assert target.placement is not None
    record = platform.post(
        "/tasksets", json={"name": target.name, "project_id": target.placement.project_id}
    )
    return _Target(name=str(record["name"]), id=str(record["id"]))


def _upload(platform: PlatformClient, bundle: Path) -> data.DataFile:
    with Progress(
        TextColumn("Uploading"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=hud_console.console,
        transient=True,
    ) as progress:
        task = progress.add_task("upload", total=bundle.stat().st_size)
        stored = data.upload_file(
            platform,
            bundle,
            content_type="application/zip",
            pipeline_key=PIPELINE_KEY,
            on_progress=lambda sent: progress.advance(task, sent),
        )
    hud_console.success(f"Uploaded {bundle.name}")
    return stored


def _get_import(platform: PlatformClient, run_id: str) -> data.PipelineRun:
    run_id = str(uuid.UUID(run_id))
    try:
        run = data.get_run(platform, run_id)
    except HudRequestError as exc:
        raise CliError.from_http(exc, resource="Import", input={"run_id": run_id}) from exc
    if run.pipeline_key != PIPELINE_KEY:
        raise CliError(
            "usage",
            f"Run {run_id} is a {run.pipeline_key} run, not a Harbor import.",
            input={"run_id": run_id},
        )
    return run


def _conclude(platform: PlatformClient, run: data.PipelineRun, plan: dict[str, Any]) -> Any:
    """Follow the run until it ends, then show its report; a run that did not complete fails."""
    if not run.finished:
        run = _follow(platform, run.id)
    report = _read_report(platform, run)
    if report is not None:
        _print_report(report)
    _print_outcome(run)
    payload = {
        **plan,
        "run": run.model_dump(mode="json"),
        "report": report.model_dump(mode="json") if report else None,
    }
    return payload if run.status == "completed" else Result(payload)


def _follow(platform: PlatformClient, run_id: str) -> data.PipelineRun:
    try:
        with hud_console.console.status("Import queued") as status:

            def show(run: data.PipelineRun) -> None:
                elapsed = int((datetime.now(UTC) - run.created_at).total_seconds())
                status.update(f"Import {run.phase} ({elapsed // 60}m {elapsed % 60:02d}s)")

            return data.wait_for_run(platform, run_id, on_update=show)
    except KeyboardInterrupt:
        hud_console.info(
            f"Stopped following; the import keeps running. "
            f"Follow it again with: hud harbor report {run_id} --wait"
        )
        raise


def _read_report(platform: PlatformClient, run: data.PipelineRun) -> ImportReport | None:
    output = run.output(REPORT_OUTPUT)
    if output is None:
        return None
    raw = data.download_json(platform, output.file_id)
    schema = raw.get("schema_version") if isinstance(raw, dict) else None
    if schema != REPORT_SCHEMA:
        hud_console.warning(
            f"This hud version reads {REPORT_SCHEMA} reports, and the import wrote "
            f"{schema!r}; upgrade hud to see the report in detail."
        )
        return None
    return ImportReport.model_validate(raw)


def _status(status: str) -> str:
    style = _STATUS_STYLE[status]
    return f"[{style}]{status}[/{style}]"


def _print_report(report: ImportReport) -> None:
    out = hud_console.stdout
    if report.tasks:
        table = Table(title="Tasks", title_justify="left")
        table.add_column("Task", style="cyan")
        table.add_column("From", style="dim")
        table.add_column("Version", justify="right")
        table.add_column("Status")
        for task in report.tasks:
            table.add_row(
                task.name,
                task.path if task.source == "bundle" and task.path else "HUD",
                str(task.build_version) if task.build_version is not None else "-",
                _status(task.status),
            )
        out.print(table)
    if report.jobs:
        table = Table(title="Jobs", title_justify="left")
        table.add_column("Job", style="cyan")
        table.add_column("HUD job", style="blue", no_wrap=True)
        table.add_column("Trials", justify="right")
        for status in _STATUS_STYLE:
            table.add_column(status.capitalize(), justify="right")
        for job in report.jobs:
            counts = {status: 0 for status in _STATUS_STYLE}
            for trial in job.trials:
                counts[trial.status] += 1
            table.add_row(
                job.name or job.path,
                job.job_id or "-",
                str(len(job.trials)),
                *(str(counts[status]) for status in _STATUS_STYLE),
            )
        out.print(table)

    left_out = [
        *((task.path or task.name, task.reason) for task in report.tasks if task.reason),
        *((job.path, job.reason) for job in report.jobs if job.reason),
        *(
            (trial.path, trial.reason)
            for job in report.jobs
            for trial in job.trials
            if trial.reason
        ),
        *((warning.path, warning.reason) for warning in report.warnings),
    ]
    if left_out:
        table = Table(title="Notes", title_justify="left")
        table.add_column("Path", style="cyan")
        table.add_column("Reason")
        for path, reason in left_out:
            table.add_row(path, reason)
        out.print(table)
    if report.error is not None:
        hud_console.warning(
            f"The import stopped while it tried to {report.error.step}: {report.error.message}"
        )


def _print_outcome(run: data.PipelineRun) -> None:
    summary = run.result.get("summary") if run.result else None
    if run.status == "completed":
        hud_console.success(summary or "Import completed")
    elif run.status == "blocked":
        hud_console.warning(summary or "The import stopped; rerun it to resume")
    elif run.status == "cancelled":
        hud_console.warning("The import was cancelled; rerun it to resume")
    else:
        hud_console.warning(f"The import failed: {run.error or 'no error was reported'}")
    taskset_id = run.args.get("taskset_id")
    if taskset_id:
        hud_console.info(
            f"Imports: {settings.hud_web_url.rstrip('/')}/tasksets/{taskset_id}/imports"
        )
