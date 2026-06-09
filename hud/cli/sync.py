"""``hud sync`` command group — sync tasks and environments to the platform."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any
from urllib import parse

import httpx
import typer

from hud.cli.utils.api import hud_headers, require_api_key
from hud.cli.utils.project_config import (
    get_taskset_id,
    load_project_config,
    save_project_config,
)
from hud.cli.utils.taskset import fetch_remote_tasks, resolve_taskset_id
from hud.settings import settings
from hud.utils.hud_console import HUDConsole

LOGGER = logging.getLogger(__name__)

sync_app = typer.Typer(
    name="sync",
    help="Sync tasks and environments to the HUD platform",
    add_completion=False,
    rich_markup_mode="rich",
)


def _export_remote_tasks(
    taskset_id: str,
    taskset_display: str,
    output_path: str,
    api_url: str,
    headers: dict[str, str],
    hud_console: HUDConsole,
) -> None:
    """Fetch remote tasks and export to JSON or CSV."""
    hud_console.progress_message("Fetching remote tasks...")
    remote_tasks = fetch_remote_tasks(taskset_id, api_url, headers)

    if not remote_tasks:
        hud_console.warning("No tasks found in taskset")
        return

    out = Path(output_path)
    suffix = out.suffix.lower()

    if suffix == ".json":
        with open(out, "w", encoding="utf-8") as f:
            json.dump(remote_tasks, f, indent=2, default=str)

    elif suffix == ".csv":
        all_arg_keys: set[str] = set()
        all_col_keys: set[str] = set()
        for t in remote_tasks:
            args = t.get("args")
            if isinstance(args, dict):
                all_arg_keys.update(args.keys())
            cols = t.get("column_values")
            if isinstance(cols, dict):
                all_col_keys.update(cols.keys())

        sorted_arg_keys = sorted(all_arg_keys)
        sorted_col_keys = sorted(all_col_keys)

        fieldnames = [
            "slug",
            "scenario",
            "env",
            *[f"arg:{k}" for k in sorted_arg_keys],
            *[f"col:{k}" for k in sorted_col_keys],
        ]

        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for t in remote_tasks:
                row: dict[str, Any] = {
                    "slug": t.get("slug") or t.get("external_id") or "",
                    "scenario": t.get("scenario") or "",
                    "env": "",
                }
                env_data = t.get("env")
                if isinstance(env_data, dict):
                    row["env"] = env_data.get("name") or ""

                args = t.get("args")
                if isinstance(args, dict):
                    for k in sorted_arg_keys:
                        val = args.get(k)
                        row[f"arg:{k}"] = json.dumps(val) if isinstance(val, (dict, list)) else val

                cols = t.get("column_values")
                if isinstance(cols, dict):
                    for k in sorted_col_keys:
                        val = cols.get(k)
                        row[f"col:{k}"] = json.dumps(val) if isinstance(val, list) else val

                writer.writerow(row)
    else:
        hud_console.error(f"Unsupported export format: {suffix}. Use .json or .csv")
        raise typer.Exit(1)

    hud_console.success(f"Exported {len(remote_tasks)} tasks to {out}")


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
    hud_console.progress_message(f"Collecting tasks from {source}...")
    try:
        from hud.eval import Taskset

        local_taskset = Taskset.from_file(source)
    except (ImportError, FileNotFoundError, ValueError) as e:
        hud_console.error(str(e))
        raise typer.Exit(1) from e

    raw_tasks = list(local_taskset)
    if not raw_tasks:
        hud_console.error(f"No Task objects found in: {source}")
        raise typer.Exit(1)

    hud_console.success(f"Found {len(raw_tasks)} task(s)")

    # Cross-check: resolve current env name from platform, check local refs match.
    # Do not rewrite Python source here; registry identity belongs in project config.
    stored_registry_id = config.get("registryId")
    if stored_registry_id and raw_tasks:
        from hud.cli.utils.name_check import resolve_registry_name

        platform_env_name = resolve_registry_name(stored_registry_id, api_url, headers)
        if platform_env_name:
            if platform_env_name != config.get("registryName"):
                save_project_config({"registryName": platform_env_name})

            task_env_names = set()
            for task in raw_tasks:
                env_name = task.to_dict()["env"].get("name")
                if env_name:
                    task_env_names.add(env_name)
            mismatched_names = {n for n in task_env_names if n != platform_env_name}
            if mismatched_names:
                hud_console.warning(
                    "Local task env names do not match the linked platform environment "
                    f"'{platform_env_name}': {', '.join(sorted(mismatched_names))}"
                )

    # Apply filters
    if task_filter:
        local_taskset = local_taskset.filter([task_filter])
        if not local_taskset:
            hud_console.error(f"No task found with slug '{task_filter}'")
            raise typer.Exit(1)
    if exclude:
        local_taskset = local_taskset.exclude(exclude)
        if not local_taskset:
            hud_console.error("No tasks left after exclusions")
            raise typer.Exit(1)

    # Fetch remote state (skip if taskset doesn't exist yet)
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
                remote_tasks = []
            else:
                hud_console.error(f"Failed to fetch taskset: {e}")
                raise typer.Exit(1) from e

    if not taskset_name and taskset:
        taskset_name = taskset

    # Force mode: skip diff, upload everything
    if force:
        plan = local_taskset.diff(
            Taskset.from_tasks(taskset_name, []),
            api_url=api_url,
            headers=headers,
        )
        hud_console.info(f"\n  --force: uploading all {len(plan.to_apply)} task(s)")
    else:
        remote_taskset = Taskset.from_remote_tasks(taskset_name, remote_tasks)
        plan = local_taskset.diff(remote_taskset, api_url=api_url, headers=headers)
        hud_console.info("\n" + plan.summary())

    if not plan.to_apply:
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

    # Upload (platform validates envs + scenarios inline)
    hud_console.progress_message("Uploading tasks...")
    try:
        result = plan.apply(taskset_name=taskset_name, api_url=api_url, headers=headers)
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
        return

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
