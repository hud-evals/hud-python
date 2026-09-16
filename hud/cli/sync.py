"""``hud sync`` command group: sync tasks and environments to the platform."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import typer
from typer.core import TyperGroup

from hud.cli import (
    CLI,
    CONFIG_PATH,
    AuthScope,
    CliError,
    DirectoryLink,
    DirectoryState,
)
from hud.cli.project import PROJECT_OPTION_HELP, Placement
from hud.eval import Taskset
from hud.eval.sync import diff, resolve_taskset_id, upload_taskset
from hud.settings import settings
from hud.utils.exceptions import HudRequestError
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient


@dataclass(frozen=True)
class RegistryEnvironment:
    id: str
    name: str
    version: str = ""

    @classmethod
    def from_record(cls, data: dict[str, Any]) -> RegistryEnvironment:
        version = (data.get("latest_build") or {}).get("version")
        return cls(
            id=data["id"], name=data["name"], version="" if version is None else str(version)
        )

    @classmethod
    def resolve(cls, platform: PlatformClient, registry_id: str) -> RegistryEnvironment:
        try:
            registry_id = str(UUID(registry_id))
        except ValueError as exc:
            raise ValueError("Pass an environment ID, or omit it to select interactively") from exc
        try:
            return cls.from_record(platform.get(f"/registry/{registry_id}"))
        except HudRequestError as exc:
            if exc.status_code != 404:
                raise
            raise CliError(
                "not_found",
                f"Environment {registry_id} is inaccessible or deleted.",
                suggestion="Run 'hud sync env <id>' to link an accessible environment.",
            ) from exc


sync_app = CLI(
    name="sync",
    help="Sync tasks and environments to the HUD platform",
    add_completion=False,
    rich_markup_mode="rich",
)


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
) -> Any:
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

    platform = PlatformClient.from_settings()
    state = DirectoryState(AuthScope.resolve(platform))
    link = state.load()

    target_ref = taskset or (str(link.taskset_id) if link.taskset_id else None)
    if not target_ref:
        raise ValueError(
            "No taskset specified. Pass a taskset name/ID or run "
            "'hud sync tasks <name>' first to store it."
        )
    if taskset is None:
        hud_console.info(f"Using taskset ID from {CONFIG_PATH}")

    if export:
        if link_target:
            raise ValueError("--link cannot be combined with --export")
        hud_console.progress_message("Fetching remote taskset...")
        remote = Taskset.from_api(target_ref)
        if not remote:
            hud_console.warning("No tasks found in taskset")
            return None
        out = Path(export)
        if out.suffix.lower() == ".csv":
            # Spreadsheet view: one ``arg:`` column per key; nested values as JSON.
            rows = [task.model_dump(mode="json", exclude_none=True) for task in remote]
            arg_keys = sorted({key for row in rows for key in (row.get("args") or {})})
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["slug", "id", "env", *(f"arg:{k}" for k in arg_keys)]
                )
                writer.writeheader()
                for row in rows:
                    args = row.get("args") or {}
                    writer.writerow(
                        {
                            "slug": row["slug"],
                            "id": row["id"],
                            "env": row["env"],
                            **{
                                f"arg:{key}": json.dumps(args[key], default=str)
                                if isinstance(args.get(key), (dict, list))
                                else args.get(key)
                                for key in arg_keys
                            },
                        }
                    )
        else:
            out = remote.to_file(out)
        hud_console.success(f"Exported {len(remote)} tasks to {out}")
        return None

    hud_console.progress_message(f"Collecting tasks from {source}...")
    local_taskset = Taskset.from_file(source)
    if not local_taskset:
        raise ValueError(f"No Task objects found in: {source}")
    hud_console.success(f"Found {len(local_taskset)} task(s)")
    if task_filter:
        local_taskset = local_taskset.filter([task_filter])
        if not local_taskset:
            raise ValueError(f"No task found with slug '{task_filter}'")

    if link.registry_id is not None:
        linked_name = RegistryEnvironment.resolve(platform, str(link.registry_id)).name
        mismatched = local_taskset.environment_names() - {linked_name}
        if mismatched:
            hud_console.warning(
                "Local task env names do not match the linked platform environment "
                f"'{linked_name}': {', '.join(sorted(mismatched))}"
            )

    placement = Placement.resolve(platform, link, flag=project)

    # The remote taskset to diff against. --force diffs against an empty one so
    # every task uploads. A missing remote is created only for an explicit name,
    # never for a stored id.
    taskset_uuid, display = resolve_taskset_id(platform, target_ref)
    if taskset_uuid:
        record = platform.get(f"/tasksets/{taskset_uuid}")
        remote_taskset = (
            Taskset(str(record["name"]), [], taskset_id=taskset_uuid)
            if force
            else Taskset.from_api(taskset_uuid)
        )
    elif taskset is not None:
        hud_console.info(f"Taskset '{display}' not found; it will be created")
        remote_taskset = Taskset(display, [])
    else:
        raise CliError(
            error="not_found",
            message=f"Taskset not found: {target_ref}",
            input={"taskset": target_ref},
            suggestion="Pass a taskset name to create it, or use an existing id.",
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
        if link_target and not dry_run:
            if remote_taskset.taskset_id is None:
                raise CliError("not_found", "Cannot link a taskset that does not exist")
            state.update(DirectoryLink(taskset_id=UUID(remote_taskset.taskset_id)))
        hud_console.success("All tasks up to date")
        return {**plan_payload, "status": "up_to_date", "dry_run": dry_run}
    if dry_run:
        hud_console.info("\n  --dry-run: no changes made")
        return {**plan_payload, "dry_run": True, "action": "sync_tasks"}

    CLI.confirm_or_abort("Proceed?", yes=yes, default=False)
    placement.require_writable()

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
        raise CliError.from_http(exc, input={"taskset": plan.taskset_name}) from exc

    returned_id = result.get("taskset_id")
    if returned_id and (link_target or (link.taskset_id is None and project is None)):
        if state.update(DirectoryLink(taskset_id=UUID(returned_id))):
            hud_console.dim_info("Taskset ID saved to:", str(CONFIG_PATH))
        hud_console.info(f"  {settings.hud_web_url}/tasksets/{returned_id}")

    created = int(result.get("tasks_created", 0))
    updated = int(result.get("tasks_updated", 0))
    hud_console.success("Sync complete")
    hud_console.info(f"  + {created} created, ~ {updated} updated")
    return {
        **plan_payload,
        "status": "synced",
        "tasks_created": created,
        "tasks_updated": updated,
        "taskset_id": returned_id,
    }


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
) -> Any:
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

    if name is None and (dry_run or not sys.stdin.isatty()):
        raise CliError("usage", "Pass an environment ID for a dry run or noninteractive link.")

    platform = PlatformClient.from_settings()
    state = DirectoryState(AuthScope.resolve(platform), Path(directory).resolve())
    link = state.load()
    linked_id = str(link.registry_id) if link.registry_id else None

    if name is None:
        hud_console.info("Fetching your environments...")
        envs: list[RegistryEnvironment] = []
        while True:
            data = platform.get(
                "/registry", params={"limit": 500, "offset": len(envs), "sort_by": "date"}
            )
            page = [RegistryEnvironment.from_record(item) for item in data["items"]]
            envs.extend(page)
            if len(envs) >= data["total"]:
                break
            if not page:
                raise ValueError("Registry API returned an empty page before the reported total")
        if not envs:
            raise CliError("not_found", "No environments found. Deploy one with 'hud deploy'.")
        selected = cast(
            "RegistryEnvironment",
            hud_console.select(
                "Select an environment",
                [
                    {
                        "name": f"{env.name} ({env.id})"
                        + (" (currently linked)" if env.id == linked_id else ""),
                        "value": env,
                    }
                    for env in envs
                ],
                default=0,
            ),
        )
    else:
        selected = RegistryEnvironment.resolve(platform, name)

    if dry_run:
        hud_console.info(f"Would link to {selected.name} ({selected.id})")
        return {
            "dry_run": True,
            "action": "link_environment",
            "id": selected.id,
            "name": selected.name,
        }
    if linked_id and linked_id != selected.id:
        hud_console.warning(f"Currently linked to: {linked_id[:8]}...")
        CLI.confirm_or_abort("Switch to new environment?", yes=yes, default=False)

    changed = state.update(DirectoryLink(registry_id=UUID(selected.id)))
    hud_console.success(f"Linked to: {selected.name} ({selected.id[:8]}...)")
    if changed:
        hud_console.dim_info("Link saved to:", str(CONFIG_PATH))
    return {
        "name": selected.name,
        "id": selected.id,
        "short_id": selected.id[:8],
        "changed": changed,
    }


@sync_app.callback(invoke_without_command=True)
def sync_callback(ctx: typer.Context) -> Any:
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
        return command.invoke(task_context)
