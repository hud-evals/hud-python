"""``hud sync`` command group: sync tasks and environments to the platform."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

import typer
from typer.core import TyperGroup

from hud.cli.utils.api import require_api_key
from hud.cli.utils.config import AuthScope, DirectoryLink, DirectoryState
from hud.cli.utils.output import (
    CliError,
    ExitCode,
    confirm_or_abort,
    emit_json,
    is_interactive,
    json_option,
    map_exception,
    output_option,
    wants_json,
)
from hud.cli.utils.project import (
    PROJECT_OPTION_HELP,
    require_writable_placement,
    resolve_placement,
)
from hud.cli.utils.registry import (
    RegistryEnvironment,
    get_registry_environment,
    list_registry_environments,
    resolve_registry_environments,
)
from hud.eval import Taskset
from hud.eval.sync import diff, resolve_taskset_id, upload_taskset
from hud.utils.exceptions import HudException, HudRequestError
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient

LOGGER = logging.getLogger(__name__)

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
        console.info("Using taskset ID from ~/.hud/config.json")
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
    try:
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
    except (HudException, ValueError) as e:
        console.error(str(e))
        raise typer.Exit(1) from e
    console.success(f"Exported {len(remote_taskset)} tasks to {out}")


def _load_local_taskset(
    source: str,
    *,
    task_filter: str | None,
    exclude: list[str] | None,
    console: HUDConsole,
) -> Taskset:
    console.progress_message(f"Collecting tasks from {source}...")
    try:
        taskset = Taskset.from_file(source)
    except (ImportError, FileNotFoundError, ValueError) as e:
        console.error(str(e))
        raise typer.Exit(1) from e

    if not taskset:
        console.error(f"No Task objects found in: {source}")
        raise typer.Exit(1)
    console.success(f"Found {len(taskset)} task(s)")

    if task_filter:
        taskset = taskset.filter([task_filter])
        if not taskset:
            console.error(f"No task found with slug '{task_filter}'")
            raise typer.Exit(1)
    if exclude:
        taskset = taskset.exclude(exclude)
        if not taskset:
            console.error("No tasks left after exclusions")
            raise typer.Exit(1)
    return taskset


def _warn_on_linked_environment_mismatch(
    taskset: Taskset,
    platform: PlatformClient,
    console: HUDConsole,
    link: DirectoryLink,
) -> None:
    stored_registry_id = str(link.registry_id) if link.registry_id else None
    if not isinstance(stored_registry_id, str) or not stored_registry_id:
        return

    registry_env = get_registry_environment(platform, stored_registry_id)
    if registry_env is None:
        raise CliError(
            "not_found",
            f"Linked environment {stored_registry_id} is inaccessible or deleted.",
            suggestion="Run 'hud sync env <id>' to relink this directory.",
        )
    platform_env_name = registry_env.name

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


def _show_upload_error(error: HudRequestError, console: HUDConsole) -> None:
    detail = (error.response_json or {}).get("detail", "")
    if error.status_code == 400 and isinstance(detail, str) and detail:
        console.error("Upload rejected by platform:")
        for detail_line in detail.split("\n"):
            stripped = detail_line.strip()
            if stripped:
                console.error(f"  {stripped}")
        if "not found" in detail.lower():
            console.hint(
                "Check that the environment is deployed and the task id matches "
                "the environment manifest."
            )
        return
    console.error(f"Upload failed ({error.status_code}): {detail or error}")


def _save_taskset_id(result: dict[str, object], console: HUDConsole, state: DirectoryState) -> None:
    returned_id = result.get("taskset_id")
    if not isinstance(returned_id, str) or not returned_id:
        return
    changed = state.update(DirectoryLink(taskset_id=UUID(returned_id)))
    if changed:
        console.dim_info("Taskset ID saved to:", "~/.hud/config.json")
    from hud.settings import settings

    console.info(f"  {settings.hud_web_url}/tasksets/{returned_id}")


@sync_app.command("tasks")
def sync_tasks_command(
    taskset: str | None = typer.Argument(
        None,
        help="Taskset name or ID (reads from ~/.hud/config.json if omitted)",
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
        help="Skip confirmation prompt",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show sync plan without uploading",
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
    json_output: bool = json_option(),
    output: str | None = output_option(),
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
        if wants_json(json_output, output):
            emit_json({**plan_payload, "status": "up_to_date", "dry_run": dry_run})
            return
        hud_console.success("All tasks up to date")
        return

    if dry_run:
        if wants_json(json_output, output):
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
    except HudRequestError as e:
        _show_upload_error(e, hud_console)
        raise map_exception(e, input={"taskset": plan.taskset_name}) from e

    if link_target is True or (link.taskset_id is None and taskset_id is None and project is None):
        _save_taskset_id(result, hud_console, state)

    created = int(result.get("tasks_created", 0))
    updated = int(result.get("tasks_updated", 0))

    if wants_json(json_output, output):
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
        help="Skip confirmation prompt",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    json_output: bool = json_option(),
    output: str | None = output_option(),
) -> None:
    """Link local directory to a platform environment.

    [not dim]Validates an environment ID, verifies it exists, and stores
    the registry ID in ~/.hud/config.json for task sync checks.

    Examples:
        hud sync env <environment-id>           # link cwd to '<environment-id>'
        hud sync env <environment-id> ./my-env  # link specific directory
        hud sync env                      # interactive: pick from your envs[/not dim]
    """
    hud_console = HUDConsole()
    hud_console.header("Sync Environment", icon="")

    require_api_key("sync environments")

    if name is None and (dry_run is True or not is_interactive()):
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

    if not name:
        if dry_run or not is_interactive():
            raise CliError(
                error="confirmation_required",
                message="No environment name given in a non-interactive terminal.",
                suggestion="Pass the environment name: hud sync env <name>",
                exit_code=ExitCode.USAGE,
            )
        # Interactive: list environments and let user pick
        hud_console.info("Fetching your environments...")
        try:
            envs = list_registry_environments(platform)
        except HudRequestError as e:
            hud_console.error(f"Failed to fetch environments: {e.status_code or e}")
            raise typer.Exit(1) from e

        if not envs:
            hud_console.warning("No environments found")
            hud_console.info("Deploy an environment first with: hud deploy")
            raise typer.Exit(1)

        hud_console.info("\nYour environments:")
        for i, env in enumerate(envs, 1):
            marker = " (currently linked)" if env.id == existing_registry_id else ""
            hud_console.info(f"  {i}. {env.name}{env.version_label} ({env.short_id}...){marker}")

        hud_console.info("")
        try:
            selection = input("Select environment number (or paste full name): ").strip()
        except (EOFError, KeyboardInterrupt, OSError):
            hud_console.info("\nAborted.")
            raise typer.Exit(0) from None

        displayed = envs
        try:
            idx = int(selection) - 1
            if 0 <= idx < len(displayed):
                selected_env = displayed[idx]
            else:
                hud_console.error("Invalid selection")
                raise typer.Exit(1)
        except ValueError:
            name = selection

    if selected_env is None:
        if not name:
            hud_console.error("No environment selected")
            raise typer.Exit(1)
        # Resolve name to registry ID
        hud_console.progress_message(f"Looking up '{name}'...")

        try:
            matching = resolve_registry_environments(platform, name)
        except HudRequestError as e:
            hud_console.error(f"Failed to search environments: {e.status_code or e}")
            raise typer.Exit(1) from e

        if not matching:
            raise CliError(
                error="not_found",
                message=f"No environment found matching '{name}'",
                input={"name": name},
                suggestion="Run 'hud deploy' first, or pass an exact environment name.",
            )

        if len(matching) > 1:
            hud_console.warning(f"Multiple environments match '{name}':")
            for env_item in matching:
                hud_console.info(f"  {env_item.name} ({env_item.short_id}...)")
            hud_console.info("Pass the full ID with --id to disambiguate")
            raise typer.Exit(1) from None

        selected_env = matching[0]

    if dry_run:
        if wants_json(json_output, output):
            emit_json({"dry_run": True, "action": "link_environment", "id": selected_env.id})
        else:
            hud_console.info(f"Would link to {selected_env.name} ({selected_env.id})")
        return

    if existing_registry_id and existing_registry_id != selected_env.id:
        hud_console.warning(f"Currently linked to: {existing_registry_id[:8]}...")
        confirm_or_abort("Switch to new environment?", yes=yes, default=False)

    changed = state.update(DirectoryLink(registry_id=UUID(selected_env.id)))
    if wants_json(json_output, output):
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
        hud_console.dim_info("Config saved to:", "~/.hud/config.json")


@sync_app.callback(invoke_without_command=True)
def sync_callback(ctx: typer.Context) -> None:
    """Sync tasks and environments to the HUD platform.

    [not dim]Without a subcommand, syncs tasks using stored config.

    Examples:
        hud sync                         # sync tasks using ~/.hud/config.json
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
