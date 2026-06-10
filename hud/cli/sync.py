"""``hud sync`` command group: sync tasks and environments to the platform."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from hud.cli.utils.api import require_api_key
from hud.cli.utils.registry import (
    RegistryEnvironment,
    get_registry_environment,
    list_registry_environments,
    resolve_registry_environments,
)
from hud.environment.source import EnvironmentSource
from hud.eval import Taskset
from hud.eval.taskset import resolve_taskset_id, taskset_column_definitions, upload_taskset
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
) -> str:
    stored_taskset_id = EnvironmentSource.open().taskset_id
    target_ref = taskset_id or taskset or stored_taskset_id
    if not target_ref:
        console.error(
            "No taskset specified. Pass a taskset name/ID or run "
            "'hud sync tasks <name>' first to store it."
        )
        raise typer.Exit(1)
    if target_ref == stored_taskset_id and not taskset and not taskset_id:
        console.info("Using taskset ID from .hud/config.json")
    return target_ref


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
        out = remote_taskset.to_file(output_path)
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
) -> None:
    env_source = EnvironmentSource.open()
    config = env_source.load_config()
    stored_registry_id = config.get("registryId")
    if not isinstance(stored_registry_id, str) or not stored_registry_id:
        return

    try:
        registry_env = get_registry_environment(platform, stored_registry_id)
    except HudException as e:
        console.warning(f"Could not verify linked environment: {e}")
        return

    if registry_env is None:
        console.warning(
            f"Linked environment (registryId: {stored_registry_id[:8]}...) "
            "no longer exists on platform"
        )
        console.hint("Run 'hud sync env' to re-link or 'hud deploy' to create a new one")
        return

    platform_env_name = registry_env.name
    if platform_env_name != config.get("registryName"):
        env_source.save_config({"registryName": platform_env_name})

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
    if force:
        return Taskset.from_tasks(target_ref, [])

    taskset_uuid, display = resolve_taskset_id(platform, target_ref)
    if taskset_uuid:
        return Taskset.from_api(taskset_uuid)
    if allow_create:
        console.info(f"Taskset '{display}' not found; it will be created")
        return Taskset.from_tasks(display, [])

    console.error(f"Taskset not found: {target_ref}")
    raise typer.Exit(1)


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


def _save_taskset_id(result: dict[str, object], console: HUDConsole) -> None:
    returned_id = result.get("evalset_id")
    if not isinstance(returned_id, str) or not returned_id:
        return
    changed = EnvironmentSource.open().save_config({"tasksetId": returned_id})
    if changed:
        console.dim_info("Taskset ID saved to:", ".hud/config.json")
    console.info(f"  https://hud.ai/evalsets/{returned_id}")


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
) -> None:
    """Sync local task definitions to a platform taskset.

    [not dim]Collects Task objects from Python files, directories, or JSON,
    diffs against the remote taskset, and uploads changes.

    Examples:
        hud sync tasks my-taskset              # scan cwd, sync to 'my-taskset'
        hud sync tasks my-taskset tasks.py     # from specific file
        hud sync tasks my-taskset tasks/       # from directory
        hud sync tasks                         # use stored taskset ID from .hud/config.json
        hud sync tasks my-taskset --dry-run    # preview without uploading
        hud sync tasks my-taskset --yes        # skip confirmation (CI)
        hud sync tasks my-taskset --export tasks.csv   # export to CSV
        hud sync tasks my-taskset --export tasks.json  # export to JSON[/not dim]
    """
    hud_console = HUDConsole()
    hud_console.header("Sync Tasks", icon="")

    require_api_key("sync tasks")

    platform = PlatformClient.from_settings()

    target_ref = _taskset_target(taskset, taskset_id, hud_console)

    if export:
        _export_taskset(target_ref, export, hud_console)
        return

    local_taskset = _load_local_taskset(
        source,
        task_filter=task_filter,
        exclude=exclude,
        console=hud_console,
    )
    _warn_on_linked_environment_mismatch(local_taskset, platform, hud_console)

    # Creating a new taskset is only allowed when targeting an explicit name
    # (not an --id or a stored id, which must already exist).
    allow_create = taskset is not None and taskset_id is None

    try:
        remote_taskset = _fetch_remote_taskset(
            platform,
            target_ref,
            force=force,
            allow_create=allow_create,
            console=hud_console,
        )
        plan = local_taskset.diff(remote_taskset)
    except ValueError as e:
        hud_console.error(str(e))
        raise typer.Exit(1) from e
    except HudException as e:
        hud_console.error(f"Failed to fetch taskset: {e}")
        raise typer.Exit(1) from e

    if force:
        hud_console.info(f"\n  --force: uploading all {len(plan.to_apply)} task(s)")
    else:
        hud_console.info("\n" + plan.summary())

    if not plan.to_apply:
        hud_console.success("All tasks up to date")
        return

    if dry_run:
        hud_console.info("\n  --dry-run: no changes made")
        return

    if not yes and not hud_console.confirm("Proceed?", default=False):
        hud_console.info("Aborted.")
        return

    # Upload tasks; the platform validates referenced environments.
    hud_console.progress_message("Uploading tasks...")
    try:
        result = upload_taskset(
            platform,
            plan.taskset_name,
            plan.to_apply,
            columns=taskset_column_definitions(list(local_taskset)),
        )
    except HudRequestError as e:
        _show_upload_error(e, hud_console)
        return

    created = int(result.get("tasks_created", 0))
    updated = int(result.get("tasks_updated", 0))

    hud_console.success("Sync complete")
    hud_console.info(f"  + {created} created, ~ {updated} updated")
    _save_taskset_id(result, hud_console)


@sync_app.command("env")
def sync_env_command(
    name: str | None = typer.Argument(
        None,
        help="Environment name or ID to link to (interactive if omitted)",
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
) -> None:
    """Link local directory to a platform environment.

    [not dim]Resolves an environment by name, verifies it exists, and stores
    the registry ID in .hud/config.json for future deploys and syncs.

    Replaces 'hud link'.

    Examples:
        hud sync env coding-env           # link cwd to 'coding-env'
        hud sync env coding-env ./my-env  # link specific directory
        hud sync env                      # interactive: pick from your envs[/not dim]
    """
    hud_console = HUDConsole()
    hud_console.header("Sync Environment", icon="")

    require_api_key("sync environments")

    platform = PlatformClient.from_settings()
    env_dir = Path(directory).resolve()
    env_source = EnvironmentSource.open(env_dir)

    existing_config = env_source.load_config()
    existing_registry_id = existing_config.get("registryId")
    selected_env: RegistryEnvironment | None = None

    if not name:
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
        for i, env in enumerate(envs[:10], 1):
            marker = " (currently linked)" if env.id == existing_registry_id else ""
            hud_console.info(f"  {i}. {env.name}{env.version_label} ({env.short_id}...){marker}")

        hud_console.info("")
        try:
            selection = input("Select environment number (or paste full name): ").strip()
        except (EOFError, KeyboardInterrupt):
            hud_console.info("\nAborted.")
            raise typer.Exit(0) from None

        displayed = envs[:10]
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
            hud_console.error(f"No environment found matching '{name}'")
            raise typer.Exit(1) from None

        if len(matching) > 1:
            hud_console.warning(f"Multiple environments match '{name}':")
            for env_item in matching:
                hud_console.info(f"  {env_item.name} ({env_item.short_id}...)")
            hud_console.info("Pass the full ID with --id to disambiguate")
            raise typer.Exit(1) from None

        selected_env = matching[0]

    if existing_registry_id and existing_registry_id != selected_env.id:
        hud_console.warning(f"Currently linked to: {existing_registry_id[:8]}...")
        if not yes and not hud_console.confirm("Switch to new environment?", default=False):
            hud_console.info("Aborted.")
            return

    changed = env_source.save_config(
        {"registryId": selected_env.id, "registryName": selected_env.name},
    )
    hud_console.success(f"Linked to: {selected_env.name} ({selected_env.short_id}...)")
    if changed:
        hud_console.dim_info("Config saved to:", ".hud/config.json")


@sync_app.callback(invoke_without_command=True)
def sync_callback(ctx: typer.Context) -> None:
    """Sync tasks and environments to the HUD platform.

    [not dim]Without a subcommand, syncs tasks using stored config.

    Examples:
        hud sync                         # sync tasks using .hud/config.json
        hud sync tasks my-taskset        # sync tasks to specific taskset
        hud sync env coding-env          # link to environment[/not dim]
    """
    if ctx.invoked_subcommand is not None:
        return

    ctx.invoke(sync_tasks_command, taskset=None, source=".")
