"""``hud sync`` command group: sync tasks and environments to the platform."""

from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import typer
from typer.core import TyperGroup

from hud.cli import require_api_key
from hud.cli.config import CONFIG_PATH, AuthScope, DirectoryLink, DirectoryState
from hud.cli.io import (
    CliError,
    confirm_or_abort,
    emit_json,
    map_request_error,
    mark_json,
)
from hud.cli.project import (
    PROJECT_OPTION_HELP,
    require_writable_placement,
    resolve_placement,
)
from hud.eval import Taskset
from hud.eval.sync import diff, resolve_taskset_id, upload_taskset
from hud.utils.exceptions import HudRequestError
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegistryEnvironment:
    id: str
    name: str
    version: str = ""
    project_id: str | None = None

    @classmethod
    def from_record(cls, data: dict[str, Any]) -> RegistryEnvironment:
        env_id = data.get("id")
        if not isinstance(env_id, str) or not env_id:
            raise ValueError("registry environment record needs an id")
        latest_build = data.get("latest_build")
        version = latest_build.get("version") if isinstance(latest_build, dict) else None
        return cls(
            id=env_id,
            name=str(data.get("name") or "unnamed"),
            version=str(version) if version is not None else "",
            project_id=str(data["project_id"]) if data.get("project_id") else None,
        )

    @property
    def short_id(self) -> str:
        return self.id[:8]

    @property
    def version_label(self) -> str:
        return f" v{self.version}" if self.version else ""


def get_registry_environment(
    platform: PlatformClient,
    registry_id: str,
) -> RegistryEnvironment:
    try:
        registry_id = str(UUID(registry_id))
    except ValueError as exc:
        raise ValueError("Pass an environment ID, or omit it to select interactively") from exc
    try:
        data = platform.get(f"/registry/{registry_id}")
    except HudRequestError as e:
        if e.status_code == 404:
            raise CliError(
                "not_found",
                f"Environment {registry_id} is inaccessible or deleted.",
                suggestion="Run 'hud sync env <id>' to link an accessible environment.",
            ) from e
        raise
    return RegistryEnvironment.from_record(data)


def list_registry_environments(
    platform: PlatformClient,
    *,
    limit: int = 500,
    sort_by: str | None = "date",
) -> list[RegistryEnvironment]:
    environments: list[RegistryEnvironment] = []
    offset = 0
    while True:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if sort_by:
            params["sort_by"] = sort_by
        data = platform.get("/registry", params=params)
        page = [RegistryEnvironment.from_record(item) for item in data["items"]]
        environments.extend(page)
        offset += len(page)
        if offset >= data["total"]:
            return environments
        if not page:
            raise ValueError("Registry API returned an empty page before the reported total")


sync_app = typer.Typer(
    name="sync",
    help="Sync tasks and environments to the HUD platform",
    add_completion=False,
    rich_markup_mode="rich",
)


def _taskset_target(
    taskset: str | None,
    taskset_id: str | None,
    console: HUDConsole,
    link: DirectoryLink,
) -> str:
    stored_taskset_id = str(link.taskset_id) if link.taskset_id else None
    target_ref = taskset_id or taskset or stored_taskset_id
    if not target_ref:
        raise ValueError(
            "No taskset specified. Pass a taskset name/ID or run "
            "'hud sync tasks <name>' first to store it."
        )
    if target_ref == stored_taskset_id and not taskset and not taskset_id:
        console.info(f"Using taskset ID from {CONFIG_PATH}")
    return target_ref


def _write_csv(path: Path, entries: list[dict[str, Any]]) -> None:
    """Spreadsheet view of task rows: one ``arg:`` column per key."""
    arg_keys = sorted({key for entry in entries for key in (entry.get("args") or {})})
    fieldnames = [
        "slug",
        "id",
        "env",
        *[f"arg:{key}" for key in arg_keys],
    ]

    def cell(value: Any) -> Any:
        return json.dumps(value, default=str) if isinstance(value, (dict, list)) else value

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for entry in entries:
            args = entry.get("args") or {}
            writer.writerow(
                {
                    "slug": entry.get("slug") or "",
                    "id": entry.get("id") or "",
                    "env": entry.get("env") or "",
                    **{f"arg:{key}": cell(args.get(key)) for key in arg_keys},
                }
            )


def _export_taskset(
    target_ref: str,
    output_path: str,
    console: HUDConsole,
) -> None:
    console.progress_message("Fetching remote taskset...")
    remote_taskset = Taskset.from_api(target_ref)
    if not remote_taskset:
        console.warning("No tasks found in taskset")
        return
    out = Path(output_path)
    if out.suffix.lower() == ".csv":
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(
            out,
            [task.model_dump(mode="json", exclude_none=True) for task in remote_taskset],
        )
    else:
        out = remote_taskset.to_file(out)
    console.success(f"Exported {len(remote_taskset)} tasks to {out}")


def _load_local_taskset(
    source: str,
    *,
    task_filter: str | None,
    exclude: list[str] | None,
    console: HUDConsole,
) -> Taskset:
    console.progress_message(f"Collecting tasks from {source}...")
    taskset = Taskset.from_file(source)
    if not taskset:
        raise ValueError(f"No Task objects found in: {source}")
    console.success(f"Found {len(taskset)} task(s)")

    if task_filter:
        taskset = taskset.filter([task_filter])
        if not taskset:
            raise ValueError(f"No task found with slug '{task_filter}'")
    if exclude:
        taskset = taskset.exclude(exclude)
        if not taskset:
            raise ValueError("No tasks left after exclusions")
    return taskset


def _warn_on_linked_environment_mismatch(
    taskset: Taskset,
    platform: PlatformClient,
    console: HUDConsole,
    link: DirectoryLink,
) -> None:
    if link.registry_id is None:
        return
    platform_env_name = get_registry_environment(platform, str(link.registry_id)).name

    mismatched_names = taskset.environment_names() - {platform_env_name}
    if mismatched_names:
        console.warning(
            "Local task env names do not match the linked platform environment "
            f"'{platform_env_name}': {', '.join(sorted(mismatched_names))}"
        )


def _fetch_remote_taskset(
    platform: PlatformClient,
    target_ref: str,
    *,
    force: bool,
    allow_create: bool,
    console: HUDConsole,
) -> Taskset:
    """The remote taskset to diff against.

    ``--force`` diffs against an empty taskset so every task uploads. A missing
    remote diffs as all-create when *allow_create* is set, and is an error
    otherwise.
    """
    taskset_uuid, display = resolve_taskset_id(platform, target_ref)
    if taskset_uuid:
        record = platform.get(f"/tasksets/{taskset_uuid}")
        if force:
            return Taskset(str(record["name"]), [], taskset_id=taskset_uuid)
        return Taskset.from_api(taskset_uuid)
    if allow_create:
        console.info(f"Taskset '{display}' not found; it will be created")
        return Taskset(display, [])

    raise CliError(
        error="not_found",
        message=f"Taskset not found: {target_ref}",
        input={"taskset": target_ref},
        suggestion="Pass a taskset name to create it, or use an existing id.",
    )


def _save_taskset_id(result: dict[str, object], console: HUDConsole, state: DirectoryState) -> None:
    returned_id = result.get("taskset_id")
    if not isinstance(returned_id, str) or not returned_id:
        return
    changed = state.update(DirectoryLink(taskset_id=UUID(returned_id)))
    if changed:
        console.dim_info("Taskset ID saved to:", str(CONFIG_PATH))
    from hud.settings import settings

    console.info(f"  {settings.hud_web_url}/tasksets/{returned_id}")


@sync_app.command("tasks")
def sync_tasks_command(
    taskset: str | None = typer.Argument(
        None,
        help="Taskset name or ID (reads from .hud/config.json if omitted)",
    ),
    source: str = typer.Argument(
        ".",
        help="Source: Python file, directory, or JSON/JSONL (default: current directory)",
    ),
    taskset_id: str | None = typer.Option(
        None,
        "--id",
        help="Taskset ID directly (skip name resolution)",
    ),
    link_target: bool = typer.Option(
        False,
        "--link",
        help="Save this taskset as the directory's default after syncing",
    ),
    project: str | None = typer.Option(
        None,
        "--project",
        help=PROJECT_OPTION_HELP,
    ),
    task_filter: str | None = typer.Option(
        None,
        "--task",
        help="Only sync tasks matching this slug",
    ),
    exclude: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--exclude",
        help="Exclude tasks by slug (repeatable)",
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
    force: bool = typer.Option(
        False,
        "--force",
        help="Upload all tasks regardless of diff (skip signature comparison)",
    ),
    export: str | None = typer.Option(
        None,
        "--export",
        help="Export remote tasks to a file instead of syncing. Supports .json, .jsonl, and .csv",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Write JSON to stdout.", callback=mark_json, is_eager=True
    ),
) -> None:
    """Sync local task definitions to a platform taskset.

    [not dim]Collects Task objects from Python files, directories, or JSON,
    diffs against the remote taskset, and uploads changes.

    Examples:
        hud sync tasks my-taskset              # scan cwd, sync to 'my-taskset'
        hud sync tasks my-taskset tasks.py     # from specific file
        hud sync tasks my-taskset tasks/       # from directory
        hud sync tasks                         # use stored taskset ID from ~/.hud/config.json
        hud sync tasks my-taskset --dry-run    # preview without uploading
        hud sync tasks my-taskset --yes        # skip confirmation (CI)
        hud sync tasks my-taskset --export tasks.csv   # export to CSV
        hud sync tasks my-taskset --export tasks.json  # export to JSON
        hud sync tasks my-taskset --dry-run --json     # machine-readable plan[/not dim]
    """
    hud_console = HUDConsole()
    hud_console.header("Sync Tasks", icon="")

    require_api_key("sync tasks")

    platform = PlatformClient.from_settings()

    state = DirectoryState(AuthScope.resolve(platform))
    link = state.load()
    target_ref = _taskset_target(taskset, taskset_id, hud_console, link)

    if export:
        if link_target is True:
            raise ValueError("--link cannot be combined with --export")
        _export_taskset(target_ref, export, hud_console)
        return

    local_taskset = _load_local_taskset(
        source,
        task_filter=task_filter,
        exclude=exclude,
        console=hud_console,
    )
    _warn_on_linked_environment_mismatch(local_taskset, platform, hud_console, link)

    # Creating a new taskset is only allowed when targeting an explicit name
    # (not an --id or a stored id, which must already exist).
    allow_create = taskset is not None and taskset_id is None
    placement = resolve_placement(
        platform,
        link,
        flag=project,
    )

    remote_taskset = _fetch_remote_taskset(
        platform,
        target_ref,
        force=force,
        allow_create=allow_create,
        console=hud_console,
    )
    plan = diff(local_taskset, remote_taskset)

    plan_payload = {
        "taskset": plan.taskset_name,
        "create_count": len(plan.to_create),
        "update_count": len(plan.to_update),
        "unchanged_count": len(plan.unchanged),
        "remote_only_count": len(plan.remote_only),
        "to_apply": [task.id for task in plan.to_apply],
    }

    if force:
        hud_console.info(f"\n  --force: uploading all {len(plan.to_apply)} task(s)")
    else:
        hud_console.info("\n" + plan.summary())

    if not plan.to_apply:
        if link_target is True and not dry_run:
            if remote_taskset.taskset_id is None:
                raise CliError("not_found", "Cannot link a taskset that does not exist")
            state.update(DirectoryLink(taskset_id=UUID(remote_taskset.taskset_id)))
        if json_output is True:
            emit_json({**plan_payload, "status": "up_to_date", "dry_run": dry_run})
            return
        hud_console.success("All tasks up to date")
        return

    if dry_run:
        if json_output is True:
            emit_json({**plan_payload, "dry_run": True, "action": "sync_tasks"})
        else:
            hud_console.info("\n  --dry-run: no changes made")
        return

    confirm_or_abort("Proceed?", yes=yes, default=False)
    require_writable_placement(placement)

    # Upload tasks; the platform validates referenced environments.
    hud_console.progress_message("Uploading tasks...")
    try:
        result = upload_taskset(
            platform,
            plan.taskset_name,
            plan.to_apply,
            project_id=placement.project_id,
            taskset_id=remote_taskset.taskset_id,
        )
    except HudRequestError as exc:
        raise map_request_error(exc, input={"taskset": plan.taskset_name}) from exc

    if link_target is True or (link.taskset_id is None and taskset_id is None and project is None):
        _save_taskset_id(result, hud_console, state)

    created = int(result.get("tasks_created", 0))
    updated = int(result.get("tasks_updated", 0))

    if json_output is True:
        emit_json(
            {
                **plan_payload,
                "status": "synced",
                "tasks_created": created,
                "tasks_updated": updated,
                "taskset_id": result.get("taskset_id"),
            }
        )
    else:
        hud_console.success("Sync complete")
        hud_console.info(f"  + {created} created, ~ {updated} updated")


@sync_app.command("env")
def sync_env_command(
    name: str | None = typer.Argument(
        None,
        help="Environment ID to link to (interactive if omitted)",
    ),
    directory: str = typer.Argument(
        ".",
        help="Local directory to link",
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
    """Link local directory to a platform environment.

    [not dim]Validates an environment ID, verifies it exists, and stores
    the registry ID in .hud/config.json for task sync checks.

    Examples:
        hud sync env <environment-id>           # link cwd to '<environment-id>'
        hud sync env <environment-id> ./my-env  # link specific directory
        hud sync env                      # interactive: pick from your envs[/not dim]
    """
    hud_console = HUDConsole()
    hud_console.header("Sync Environment", icon="")

    require_api_key("sync environments")

    if name is None and (dry_run is True or not sys.stdin.isatty()):
        raise CliError(
            "usage",
            "Pass an environment ID for a dry run or noninteractive link.",
        )

    platform = PlatformClient.from_settings()
    env_dir = Path(directory).resolve()
    state = DirectoryState(AuthScope.resolve(platform), env_dir)
    link = state.load()
    existing_registry_id = str(link.registry_id) if link.registry_id else None
    selected_env: RegistryEnvironment | None = None

    if name is None:
        hud_console.info("Fetching your environments...")
        envs = list_registry_environments(platform)
        if not envs:
            raise CliError("not_found", "No environments found. Deploy one with 'hud deploy'.")

        hud_console.info("\nYour environments:")
        for i, env in enumerate(envs, 1):
            marker = " (currently linked)" if env.id == existing_registry_id else ""
            hud_console.info(f"  {i}. {env.name}{env.version_label} ({env.id}){marker}")

        try:
            name = input("Select environment number (or paste full ID): ").strip()
        except (EOFError, KeyboardInterrupt):
            raise typer.Exit(0) from None
        if name.isdecimal():
            index = int(name) - 1
            if not 0 <= index < len(envs):
                raise ValueError("Invalid selection")
            selected_env = envs[index]

    if selected_env is None:
        selected_env = get_registry_environment(platform, name)

    if dry_run:
        if json_output is True:
            emit_json({"dry_run": True, "action": "link_environment", "id": selected_env.id})
        else:
            hud_console.info(f"Would link to {selected_env.name} ({selected_env.id})")
        return

    if existing_registry_id and existing_registry_id != selected_env.id:
        hud_console.warning(f"Currently linked to: {existing_registry_id[:8]}...")
        confirm_or_abort("Switch to new environment?", yes=yes, default=False)

    changed = state.update(DirectoryLink(registry_id=UUID(selected_env.id)))
    if json_output is True:
        emit_json(
            {
                "name": selected_env.name,
                "id": selected_env.id,
                "changed": changed,
            }
        )
        return
    hud_console.success(f"Linked to: {selected_env.name} ({selected_env.short_id}...)")
    if changed:
        hud_console.dim_info("Link saved to:", str(CONFIG_PATH))


@sync_app.callback(invoke_without_command=True)
def sync_callback(ctx: typer.Context) -> None:
    """Sync tasks and environments to the HUD platform.

    [not dim]Without a subcommand, syncs tasks using stored config.

    Examples:
        hud sync                         # sync tasks using .hud/config.json
        hud sync tasks my-taskset        # sync tasks to specific taskset
        hud sync env <environment-id>    # link to environment[/not dim]
    """
    if ctx.invoked_subcommand is not None:
        return

    assert isinstance(ctx.command, TyperGroup)
    command = ctx.command.get_command(ctx, "tasks")
    assert command is not None
    with command.make_context("tasks", [], parent=ctx) as task_context:
        command.invoke(task_context)
