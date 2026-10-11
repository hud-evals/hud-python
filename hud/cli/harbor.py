"""``hud harbor`` — import Harbor tasks and job runs into HUD tasksets.

hud harbor import <folders...>   # import into a taskset and show the report
hud harbor imports               # recent imports
hud harbor report <import-id>    # one import's report
hud harbor cancel <import-id>    # cancel an import
"""

from __future__ import annotations

import tempfile
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

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

from hud import imports
from hud.cli import CLI, AuthScope, CliError, DirectoryLink, DirectoryState, Result
from hud.cli.project import Placement, Project
from hud.integrations.harbor.importing import (
    IMPORT_FORMAT,
    REPORT_SCHEMAS,
    ImportReport,
    TaskStatus,
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

_STATUS_STYLE: dict[TaskStatus, str] = {
    "imported": "green",
    "updated": "cyan",
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
    import_format = _import_format(platform)
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
        if size > import_format.max_input_bytes:
            raise CliError(
                "usage",
                f"The bundle is {decimal(size)}; a Harbor import takes at most "
                f"{decimal(import_format.max_input_bytes)}.",
                suggestion=(
                    "Split the folders across several imports into the same taskset; job "
                    "runs link to tasks imported earlier."
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
        file_id = _upload(platform, bundle)

    imported = imports.start_import(
        platform, format=IMPORT_FORMAT, file_id=file_id, taskset_id=target.id
    )
    hud_console.success(f"Started import {imported.id}")
    if not wait:
        hud_console.hint(f"Follow it with: hud harbor report {imported.id} --wait")
        return {**plan, "import": imported.model_dump(mode="json"), "report": None}
    return _conclude(platform, imported, plan)


@harbor_app.command("imports")
def imports_command(
    taskset: str | None = typer.Option(
        None, "--taskset", "-t", help="Only imports into this taskset (name or ID)"
    ),
    limit: int = typer.Option(20, "--limit", "-n", min=1, max=100, help="Max rows to show"),
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
    taskset_id = None
    if taskset is not None:
        record = _find_taskset(platform, _taskset_ref(taskset))
        if record is None:
            raise CliError("not_found", f"Taskset not found: {taskset}", input={"taskset": taskset})
        taskset_id = str(record["id"])
    found, total = imports.list_imports(
        platform, format=IMPORT_FORMAT, taskset_id=taskset_id, limit=limit
    )
    payload = {"imports": [item.model_dump(mode="json") for item in found], "total": total}
    if quiet:
        for item in found:
            typer.echo(item.id)
        return payload
    if not found:
        hud_console.stdout.print("[yellow]No Harbor imports found.[/yellow]")
        return payload

    table = Table()
    table.add_column("Import ID", style="blue", no_wrap=True)
    table.add_column("Status")
    table.add_column("Started", style="dim", no_wrap=True)
    for item in found:
        table.add_row(item.id, item.status, item.created_at.strftime("%Y-%m-%d %H:%M"))
    hud_console.stdout.print(table)
    if total > len(found):
        hud_console.stdout.print(
            f"[dim]Showing {len(found)} of {total}; raise --limit for more[/dim]"
        )
    hud_console.stdout.print(
        "[dim]Tip: hud harbor report <import-id> to see what an import did[/dim]"
    )
    return payload


@harbor_app.command("report")
def report_command(
    import_id: str = typer.Argument(..., help="Import ID (see hud harbor imports)"),
    wait: bool = typer.Option(False, "--wait", help="Follow a running import until it ends"),
) -> Any:
    """Show what a Harbor import did with each task, job, and trial.

    [not dim]Examples:
        hud harbor report <import-id>
        hud harbor report <import-id> --wait
        hud harbor report <import-id> --json[/not dim]
    """
    platform = PlatformClient.from_settings()
    imported = _get_import(platform, import_id)
    if not imported.ended and not wait:
        hud_console.info(f"The import is {imported.status}; its report is published when it ends.")
        hud_console.hint(f"Follow it with: hud harbor report {imported.id} --wait")
        return {"import": imported.model_dump(mode="json"), "report": None}
    return _conclude(platform, imported, {})


@harbor_app.command("cancel")
def cancel_command(
    import_id: str = typer.Argument(..., help="Import ID (see hud harbor imports)"),
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
        hud harbor cancel <import-id>
        hud harbor cancel <import-id> --yes --json[/not dim]
    """
    platform = PlatformClient.from_settings()
    imported = _get_import(platform, import_id)
    if imported.ended:
        hud_console.info(f"The import already ended: {imported.status}")
        return imported.model_dump(mode="json")
    CLI.confirm_or_abort(f"Cancel import {imported.id}?", yes=yes, default=False)
    imported = imports.cancel_import(platform, imported.id)
    hud_console.success("Cancellation requested")
    return imported.model_dump(mode="json")


def _import_format(platform: PlatformClient) -> imports.ImportFormat:
    try:
        formats = imports.list_formats(platform)
    except HudRequestError as exc:
        # A platform that predates imports has no formats route.
        if exc.status_code != 404:
            raise
        formats = []
    for import_format in formats:
        if import_format.format == IMPORT_FORMAT:
            return import_format
    raise CliError(
        "not_found",
        f"Harbor import is not available on {platform.api_url}.",
        suggestion="Check HUD_API_URL.",
    )


def _given_archive(paths: list[Path]) -> Path | None:
    """The zip to import as it is, when the command was given one."""
    archives = [path for path in paths if path.suffix.lower() == ".zip" and not path.is_dir()]
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


def _taskset_ref(ref: str) -> uuid.UUID | str:
    """``ref`` as a taskset ID, else as the taskset name HUD normalizes it to."""
    try:
        return uuid.UUID(ref)
    except ValueError:
        pass
    name = normalize_environment_name(ref, default="")
    if not name:
        raise CliError(
            "usage",
            f"{ref!r} cannot name a taskset.",
            suggestion="Pass --taskset <name>.",
            input={"taskset": ref},
        )
    return name


def _find_taskset(platform: PlatformClient, ref: uuid.UUID | str) -> dict[str, Any] | None:
    """The taskset with this ID or normalized name, or None when there is none."""
    try:
        if isinstance(ref, str):
            found = platform.get(f"/tasksets/by-name/{quote(ref, safe='')}")
            ref = uuid.UUID(str(found["taskset_id"]))
        return platform.get(f"/tasksets/{ref}")
    except HudRequestError as exc:
        if exc.status_code == 404:
            return None
        raise


def _resolve_target(
    platform: PlatformClient,
    link: DirectoryLink,
    *,
    taskset: str | None,
    project: str | None,
    name: str,
) -> _Target:
    """The taskset named by ``--taskset``, the directory, or ``name``.

    An ID has to name an existing taskset; a name with no taskset is created.
    """
    given = taskset or (str(link.taskset_id) if link.taskset_id else name)
    ref = _taskset_ref(given)
    record = _find_taskset(platform, ref)
    if record is None:
        if isinstance(ref, uuid.UUID):
            raise CliError(
                "not_found",
                f"Taskset {ref} is inaccessible or deleted.",
                suggestion="Pass --taskset <name> to import into a taskset by name.",
                input={"taskset": given},
            )
        placement = Placement.resolve(platform, link, flag=project)
        placement.require_writable()
        return _Target(name=ref, id=None, placement=placement)

    if not record["can_edit"]:
        raise CliError(
            "permission_denied",
            f"You cannot edit taskset {record['name']!r}, so nothing can be imported into it.",
            input={"taskset": given},
        )
    if project is not None and Project.resolve(platform, project).id != str(record["project_id"]):
        raise CliError(
            "usage",
            f"Taskset {record['name']!r} is in another Project; --project only places a "
            "taskset this command creates.",
            input={"taskset": given, "project": project},
        )
    return _Target(name=str(record["name"]), id=str(record["id"]))


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


def _upload(platform: PlatformClient, bundle: Path) -> str:
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
        file_id = imports.upload_bundle(
            platform,
            bundle,
            format=IMPORT_FORMAT,
            content_type="application/zip",
            on_progress=lambda sent: progress.advance(task, sent),
        )
    hud_console.success(f"Uploaded {bundle.name}")
    return file_id


def _get_import(platform: PlatformClient, import_id: str) -> imports.Import:
    import_id = str(uuid.UUID(import_id))
    try:
        imported = imports.get_import(platform, import_id)
    except HudRequestError as exc:
        raise CliError.from_http(exc, resource="Import", input={"import_id": import_id}) from exc
    if imported.format != IMPORT_FORMAT:
        raise CliError(
            "usage",
            f"Import {import_id} is a {imported.format} import, not a Harbor import.",
            input={"import_id": import_id},
        )
    return imported


def _conclude(platform: PlatformClient, imported: imports.Import, plan: dict[str, Any]) -> Any:
    """Follow the import until it ends, then show its report; one that did not complete fails."""
    if not imported.ended:
        imported = _follow(platform, imported.id)
    report = _read_report(platform, imported)
    if report is not None:
        _print_report(report)
    _print_outcome(imported)
    payload = {
        **plan,
        "import": imported.model_dump(mode="json"),
        "report": report.model_dump(mode="json") if report else None,
    }
    return payload if imported.status == "completed" else Result(payload)


def _follow(platform: PlatformClient, import_id: str) -> imports.Import:
    try:
        with hud_console.console.status("Import queued") as status:

            def show(imported: imports.Import) -> None:
                elapsed = int((datetime.now(UTC) - imported.created_at).total_seconds())
                status.update(f"Import {imported.status} ({elapsed // 60}m {elapsed % 60:02d}s)")

            return imports.wait_for_import(platform, import_id, on_update=show)
    except KeyboardInterrupt:
        hud_console.info(
            f"Stopped following; the import keeps running. "
            f"Follow it again with: hud harbor report {import_id} --wait"
        )
        raise


def _read_report(platform: PlatformClient, imported: imports.Import) -> ImportReport | None:
    if not imported.publishes_report:
        return None
    raw = imports.get_report(platform, imported.id)
    if raw is None:
        return None
    schema = raw.get("schema_version") if isinstance(raw, dict) else None
    if schema not in REPORT_SCHEMAS:
        hud_console.warning(
            f"This hud version reads {' and '.join(REPORT_SCHEMAS)} reports, and the import "
            f"wrote {schema!r}; upgrade hud to see the report in detail."
        )
        return None
    return ImportReport.model_validate(raw)


def _status(status: TaskStatus) -> str:
    style = _STATUS_STYLE[status]
    return f"[{style}]{status}[/{style}]"


def _status_count(count: int, status: TaskStatus) -> str:
    style = _STATUS_STYLE[status]
    return f"[{style}]{count} {status}[/{style}]"


def _print_report(report: ImportReport) -> None:
    out = hud_console.stdout
    if report.tasks:
        table = Table(title="Tasks", title_justify="left")
        table.add_column("Task", style="cyan")
        table.add_column("From", style="dim", overflow="fold")
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
        table.add_column("Job", style="cyan", overflow="fold")
        table.add_column("Trials")
        table.add_column("HUD job", style="blue", no_wrap=True)
        for job in report.jobs:
            counts: Counter[TaskStatus] = Counter(trial.status for trial in job.trials)
            trials = ", ".join(
                _status_count(counts[status], status) for status in _STATUS_STYLE if counts[status]
            )
            table.add_row(job.name or job.path, trials or "none", job.job_id or "-")
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


def _print_outcome(imported: imports.Import) -> None:
    if imported.status == "completed":
        hud_console.success(imported.summary or "Import completed")
    elif imported.status == "stopped":
        hud_console.warning(imported.summary or "The import stopped; rerun it to resume")
    elif imported.status == "cancelled":
        hud_console.warning("The import was cancelled; rerun it to resume")
    else:
        hud_console.warning(f"The import failed: {imported.error or 'no error was reported'}")
    hud_console.info(
        f"Imports: {settings.hud_web_url.rstrip('/')}/tasksets/{imported.taskset_id}/imports"
    )
