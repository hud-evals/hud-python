"""``hud sync`` command group — sync tasks and environments to the platform."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib import parse

import httpx
import typer

from hud.cli.sync_helpers import (
    _build_column_definitions,
    _build_local_specs,
    _diff_and_display,
    _export_remote_tasks,
    _upload_tasks,
)
from hud.cli.utils.api import hud_headers, require_api_key
from hud.cli.utils.project_config import (
    get_taskset_id,
    load_project_config,
    save_project_config,
)
from hud.cli.utils.taskset import fetch_remote_tasks, resolve_taskset_id
from hud.eval.source import load_variants
from hud.settings import settings
from hud.utils.hud_console import HUDConsole

LOGGER = logging.getLogger(__name__)

sync_app = typer.Typer(
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
        help="Export remote tasks to a file instead of syncing. Supports .json and .csv",
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

    api_url = settings.hud_api_url
    headers = hud_headers()

    # Resolve taskset identity
    resolved_taskset_id = taskset_id or ""
    taskset_display = taskset or ""
    previously_stored_id = get_taskset_id() or ""

    if not resolved_taskset_id and not taskset:
        if previously_stored_id:
            resolved_taskset_id = previously_stored_id
            hud_console.info("Using taskset ID from .hud/config.json")
        else:
            hud_console.error(
                "No taskset specified. Pass a taskset name/ID or run "
                "'hud sync tasks <name>' first to store it."
            )
            raise typer.Exit(1)

    if taskset and not resolved_taskset_id:
        hud_console.progress_message("Resolving taskset...")
        resolved_taskset_id, taskset_display, _ = resolve_taskset_id(
            taskset,
            api_url,
            headers,
            create=False,
        )
        if resolved_taskset_id:
            hud_console.success(f"Found taskset '{taskset_display}'")
        else:
            taskset_display = taskset

    # Resolve the taskset name from platform (for display + upload)
    if resolved_taskset_id and not taskset_display:
        try:
            resp = httpx.get(
                f"{api_url}/tasks/evalsets/{resolved_taskset_id}/tasks-by-id",
                headers=headers,
                timeout=10.0,
            )
            if resp.status_code == 200:
                taskset_display = resp.json().get("evalset_name") or resolved_taskset_id[:8]
            else:
                taskset_display = resolved_taskset_id[:8]
        except Exception:
            taskset_display = resolved_taskset_id[:8]

    # Export mode: fetch remote tasks and write to file, then exit
    if export:
        _export_remote_tasks(
            resolved_taskset_id, taskset_display, export, api_url, headers, hud_console
        )
        return

    # Phase 2: Check stored registryId is still valid (if present)
    config = load_project_config()
    stored_registry_id = config.get("registryId")
    if stored_registry_id:
        try:
            reg_check = httpx.get(
                f"{api_url}/registry/envs/{stored_registry_id}",
                headers=headers,
                timeout=10.0,
            )
            if reg_check.status_code == 404:
                hud_console.warning(
                    f"Linked environment (registryId: {stored_registry_id[:8]}...) "
                    "no longer exists on platform"
                )
                hud_console.hint(
                    "Run 'hud sync env' to re-link or 'hud deploy' to create a new one"
                )
        except Exception:  # noqa: S110
            pass

    # Collect local tasks
    collection_failures: list[tuple[str, str]] = []
    hud_console.progress_message(f"Collecting tasks from {source}...")
    try:
        raw_tasks = load_variants(source)
    except (ImportError, FileNotFoundError, ValueError) as e:
        hud_console.error(str(e))
        raise typer.Exit(1) from e

    if not raw_tasks:
        hud_console.error(f"No Task objects found in: {source}")
        raise typer.Exit(1)

    hud_console.success(f"Found {len(raw_tasks)} task(s)")

    # Build local specs (validates slugs, scenarios, etc.)
    local_specs = _build_local_specs(raw_tasks, hud_console)

    # Cross-check: resolve current env name from platform, check local refs match
    stored_registry_id = config.get("registryId")
    if stored_registry_id and local_specs:
        from hud.cli.utils.name_check import check_and_fix_env_name, resolve_registry_name

        platform_env_name = resolve_registry_name(stored_registry_id, api_url, headers)
        if platform_env_name:
            if platform_env_name != config.get("registryName"):
                save_project_config({"registryName": platform_env_name})

            task_env_names = {
                s["env"].get("name") for s in local_specs if s.get("env") and s["env"].get("name")
            }
            mismatched_names = {n for n in task_env_names if n != platform_env_name}
            if mismatched_names:
                source_dir = Path(source).resolve()
                if source_dir.is_file():
                    source_dir = source_dir.parent
                fixed = check_and_fix_env_name(source_dir, platform_env_name, hud_console)
                if fixed:
                    hud_console.progress_message("Re-collecting tasks after name fix...")
                    collection_failures = []
                    raw_tasks = load_variants(source)
                    local_specs = _build_local_specs(raw_tasks, hud_console)

    # Apply filters
    if task_filter:
        local_specs = [s for s in local_specs if s["slug"] == task_filter]
        if not local_specs:
            hud_console.error(f"No task found with slug '{task_filter}'")
            raise typer.Exit(1)
    if exclude:
        exclude_set = set(exclude)
        local_specs = [s for s in local_specs if s["slug"] not in exclude_set]
        if not local_specs:
            hud_console.error("No tasks left after exclusions")
            raise typer.Exit(1)

    # Fetch remote state (skip if taskset doesn't exist yet)
    taskset_exists = bool(resolved_taskset_id)
    taskset_name = taskset_display
    remote_tasks: list[dict[str, Any]] = []

    if resolved_taskset_id:
        hud_console.progress_message("Fetching remote taskset...")
        try:
            remote_tasks = fetch_remote_tasks(
                resolved_taskset_id,
                api_url,
                headers,
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                taskset_exists = False
            else:
                hud_console.error(f"Failed to fetch taskset: {e}")
                raise typer.Exit(1) from e

    if not taskset_name and taskset:
        taskset_name = taskset

    switching_from = (
        previously_stored_id
        if previously_stored_id and previously_stored_id != resolved_taskset_id
        else None
    )

    # Force mode: skip diff, upload everything
    if force:
        to_upload = local_specs
        hud_console.info(f"\n  --force: uploading all {len(to_upload)} task(s)")
    else:
        to_upload = _diff_and_display(
            local_specs,
            remote_tasks,
            taskset_display,
            resolved_taskset_id,
            taskset_exists,
            hud_console,
            collection_failures=collection_failures,
            switching_from=switching_from,
        )

    if not to_upload:
        hud_console.success("All tasks up to date")
        return

    if dry_run:
        hud_console.info("\n  --dry-run: no changes made")
        return

    # Confirm
    if not yes:
        hud_console.info("")
        try:
            answer = input("  Proceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            hud_console.info("\n  Aborted.")
            raise typer.Exit(1) from None
        if answer not in ("y", "yes"):
            hud_console.info("  Aborted.")
            return

    # Infer column definitions from ALL local specs (not just to_upload)
    column_definitions = _build_column_definitions(local_specs)

    # Upload (platform validates envs + scenarios inline)
    hud_console.progress_message("Uploading tasks...")
    try:
        result = _upload_tasks(to_upload, taskset_name, api_url, headers, column_definitions)
    except httpx.HTTPStatusError as e:
        detail = ""
        import contextlib

        with contextlib.suppress(Exception):
            detail = e.response.json().get("detail", "")
        if e.response.status_code == 400 and detail:
            hud_console.error("Upload rejected by platform:")
            for detail_line in detail.split("\n"):
                stripped = detail_line.strip()
                if stripped:
                    hud_console.error(f"  {stripped}")
            if "not found" in detail.lower():
                hud_console.hint(
                    "Check that the environment is deployed and scenario names "
                    "match what's registered (env_name:scenario_name)"
                )
        else:
            hud_console.error(f"Upload failed ({e.response.status_code}): {detail or e}")
        raise typer.Exit(1) from e

    created = int(result.get("tasks_created", 0))
    updated = int(result.get("tasks_updated", 0))
    returned_id = result.get("evalset_id", resolved_taskset_id)

    hud_console.success("Sync complete")
    hud_console.info(f"  + {created} created, ~ {updated} updated")

    if returned_id:
        changed = save_project_config({"tasksetId": returned_id})
        if changed:
            hud_console.dim_info("Taskset ID saved to:", ".hud/config.json")
        hud_console.info(f"  https://hud.ai/evalsets/{returned_id}")


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

    api_url = settings.hud_api_url
    headers = hud_headers()
    env_dir = Path(directory).resolve()

    existing_config = load_project_config(env_dir)
    existing_registry_id = existing_config.get("registryId")

    if not name:
        # Interactive: list environments and let user pick
        hud_console.info("Fetching your environments...")
        try:
            response = httpx.get(
                f"{api_url}/registry/envs",
                headers=headers,
                params={"limit": 20, "sort_by": "updated_at"},
                timeout=30.0,
            )
            response.raise_for_status()
            envs = response.json()
        except httpx.HTTPStatusError as e:
            hud_console.error(f"Failed to fetch environments: {e.response.status_code}")
            raise typer.Exit(1) from e

        if not envs:
            hud_console.warning("No environments found")
            hud_console.info("Deploy an environment first with: hud deploy")
            raise typer.Exit(1)

        hud_console.info("\nYour environments:")
        for i, env in enumerate(envs[:10], 1):
            env_id = env.get("id", "")[:8]
            env_name = env.get("name_display") or env.get("name", "unnamed")
            version = env.get("latest_version", "")
            version_str = f" v{version}" if version else ""
            marker = " (currently linked)" if env.get("id") == existing_registry_id else ""
            hud_console.info(f"  {i}. {env_name}{version_str} ({env_id}...){marker}")

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
                registry_id = displayed[idx]["id"]
                selected = displayed[idx]
                env_display = selected.get("name_display") or selected.get("name", "unnamed")
            else:
                hud_console.error("Invalid selection")
                raise typer.Exit(1)
        except ValueError:
            name = selection

    if name:
        # Resolve name to registry ID
        hud_console.progress_message(f"Looking up '{name}'...")

        # Check if it's already a UUID
        try:
            import uuid as _uuid

            _uuid.UUID(name)
            registry_id = name
            env_display = name[:8] + "..."
        except ValueError:
            try:
                response = httpx.get(
                    f"{api_url}/registry/envs",
                    headers=headers,
                    params={"search": name, "limit": 5},
                    timeout=30.0,
                )
                response.raise_for_status()
                envs = response.json()
            except httpx.HTTPStatusError as e:
                hud_console.error(f"Failed to search environments: {e.response.status_code}")
                raise typer.Exit(1) from e

            matching = [e for e in envs if (e.get("name_display") or e.get("name", "")) == name]
            if not matching:
                matching = [
                    e
                    for e in envs
                    if name.lower() in (e.get("name_display") or e.get("name", "")).lower()
                ]

            if not matching:
                hud_console.error(f"No environment found matching '{name}'")
                hud_console.info("Available environments:")
                for env_item in envs[:5]:
                    display = env_item.get("name_display") or env_item.get("name", "unnamed")
                    eid = env_item.get("id", "")[:8]
                    hud_console.info(f"  {display} ({eid}...)")
                raise typer.Exit(1) from None

            if len(matching) > 1:
                hud_console.warning(f"Multiple environments match '{name}':")
                for env_item in matching:
                    display = env_item.get("name_display") or env_item.get("name", "unnamed")
                    eid = env_item.get("id", "")[:8]
                    hud_console.info(f"  {display} ({eid}...)")
                hud_console.info("Pass the full ID with --id to disambiguate")
                raise typer.Exit(1) from None

            registry_id = matching[0]["id"]
            env_display = matching[0].get("name_display") or matching[0].get("name", "unnamed")

    if existing_registry_id and existing_registry_id != registry_id:
        hud_console.warning(f"Currently linked to: {existing_registry_id[:8]}...")
        if not yes:
            try:
                answer = input("Switch to new environment? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                hud_console.info("\nAborted.")
                raise typer.Exit(0) from None
            if answer not in ("y", "yes"):
                hud_console.info("Aborted.")
                return

    changed = save_project_config(
        {"registryId": registry_id, "registryName": env_display},
        env_dir,
    )
    hud_console.success(f"Linked to: {env_display} ({registry_id[:8]}...)")
    if changed:
        hud_console.dim_info("Config saved to:", ".hud/config.json")

    # Post-check: fetch scenarios and display
    env_name_for_lookup = name or env_display
    try:
        scenarios_resp = httpx.get(
            f"{api_url}/tasks/environments/{parse.quote(env_name_for_lookup, safe='')}/scenarios",
            headers=headers,
            timeout=15.0,
        )
        if scenarios_resp.status_code == 200:
            scenarios_data = scenarios_resp.json()
            scenarios = scenarios_data.get("scenarios", [])
            if scenarios:
                hud_console.info(f"\n  Registered scenarios ({len(scenarios)}):")
                for s in scenarios[:15]:
                    hud_console.info(f"    {s['name']}")
                if len(scenarios) > 15:
                    hud_console.info(f"    ... and {len(scenarios) - 15} more")
            else:
                hud_console.warning("  No scenarios registered (deploy environment first)")
    except Exception:  # noqa: S110
        pass

    # Post-check: if local Environment("...") doesn't match, offer to fix
    try:
        from hud.cli.utils.name_check import check_and_fix_env_name

        check_and_fix_env_name(env_dir, env_name_for_lookup, hud_console)
    except Exception:  # noqa: S110
        pass


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

    hud_console = HUDConsole()
    config = load_project_config()
    stored_taskset_id = config.get("tasksetId")

    if not stored_taskset_id:
        hud_console.error("No taskset configured. Run 'hud sync tasks <name>' first to set up.")
        raise typer.Exit(1)

    # Delegate to sync_tasks with stored config and explicit defaults
    ctx.invoke(sync_tasks_command, taskset=None, source=".")
