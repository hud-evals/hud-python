"""`hud tasks` — CLI for listing, inspecting, and mutating tasks on the HUD platform.

Each command is a thin wrapper: parse args, call the corresponding endpoint,
render the response. Validation, status semantics, and access control are
owned by the platform.
"""

from __future__ import annotations

from typing import Any

import httpx
import typer
from rich.console import Console
from rich.table import Table

from hud.cli.utils.api import hud_headers, require_api_key
from hud.cli.utils.project_config import get_taskset_id
from hud.settings import settings
from hud.utils.hud_console import HUDConsole

tasks_app = typer.Typer(
    name="tasks",
    help="Inspect and mutate tasks on the HUD platform",
    no_args_is_help=True,
)


def _resolve_taskset(taskset_flag: str | None, positional: str | None = None) -> str:
    """Resolve taskset id: positional arg → flag → .hud/config.json → error."""
    resolved = positional or taskset_flag or get_taskset_id()
    if not resolved:
        hud_console = HUDConsole()
        hud_console.error(
            "No taskset specified. Pass one as a positional or --taskset, "
            "or set one via `hud sync tasks <name>`."
        )
        raise typer.Exit(1)
    return resolved


def _request(
    method: str,
    path: str,
    *,
    json: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> httpx.Response:
    """Issue a single platform request with API key auth."""
    url = f"{settings.hud_api_url}{path}"
    return httpx.request(
        method,
        url,
        headers=hud_headers(),
        json=json,
        params=params,
        timeout=timeout,
    )


def _handle_error(resp: httpx.Response) -> None:
    """Render a platform error and exit. Server's `detail` is relayed verbatim."""
    hud_console = HUDConsole()
    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        detail = resp.text
    hud_console.error(f"{resp.status_code}: {detail}")
    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# List tasks in a taskset
# ---------------------------------------------------------------------------


@tasks_app.command("list")
def list_command(
    taskset: str | None = typer.Argument(
        None,
        help="Taskset id (falls back to .hud/config.json if omitted)",
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit raw JSON instead of a table"
    ),
) -> None:
    """List tasks in a taskset with status + pass-rate."""
    require_api_key("list tasks")
    ts = _resolve_taskset(None, positional=taskset)

    resp = _request("GET", f"/tasks/evalsets/{ts}/tasks-with-stats")
    if resp.status_code != 200:
        _handle_error(resp)

    payload = resp.json()
    tasks: list[dict[str, Any]] = payload.get("tasks") or []
    stats_by_version: dict[str, dict[str, Any]] = {
        s["task_version_id"]: s for s in (payload.get("task_stats") or [])
    }

    if as_json:
        Console().print_json(data=payload)
        return

    if not tasks:
        HUDConsole().info(f"No tasks in taskset {ts}.")
        return

    evalset_name = payload.get("evalset_name")
    title = (
        f'Tasks in "{evalset_name}" ({ts[:7]})' if evalset_name else f"Tasks in {ts[:12]}"
    )
    table = Table(title=title, show_lines=False)
    table.add_column("slug", style="cyan", no_wrap=True)
    table.add_column("ver", justify="right")
    table.add_column("status")
    table.add_column("pass>0", justify="right")
    table.add_column("n", justify="right")
    for entry in tasks:
        task = entry.get("task") or {}
        version = entry.get("version") or {}
        stats = stats_by_version.get(str(version.get("id") or ""), {})
        pr = stats.get("pass_rate")
        sc = stats.get("scorable_count") or 0
        status = stats.get("progress_status") or "pending"
        table.add_row(
            str(task.get("external_id") or ""),
            str(version.get("version") or ""),
            status,
            f"{pr:.0%}" if isinstance(pr, (int, float)) else "—",
            str(sc),
        )
    Console().print(table)


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


@tasks_app.command("show")
def show_command(
    slug: str = typer.Argument(..., help="Task slug (external_id)"),
    taskset: str | None = typer.Option(
        None, "--taskset", "-t", help="Taskset id (falls back to .hud/config.json)"
    ),
) -> None:
    """Print one task's full JSON record."""
    require_api_key("show task")
    ts = _resolve_taskset(taskset)

    resp = _request("GET", f"/tasks/evalsets/{ts}/tasks-with-stats")
    if resp.status_code != 200:
        _handle_error(resp)

    payload = resp.json()
    tasks: list[dict[str, Any]] = payload.get("tasks") or []
    stats_by_version: dict[str, dict[str, Any]] = {
        s["task_version_id"]: s for s in (payload.get("task_stats") or [])
    }
    for entry in tasks:
        task = entry.get("task") or {}
        if task.get("external_id") == slug:
            version = entry.get("version") or {}
            stats = stats_by_version.get(str(version.get("id") or ""), {})
            output = {"task": task, "version": version, "stats": stats}
            Console().print_json(data=output)
            return

    HUDConsole().error(f"No task with slug '{slug}' in taskset {ts}.")
    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@tasks_app.command("status")
def status_command(
    slugs: list[str] | None = typer.Argument(  # noqa: B008
        None, help="One or more task slugs (omit when using --all)"
    ),
    set_value: str | None = typer.Option(
        None, "--set", help="Status to apply"
    ),
    clear: bool = typer.Option(
        False, "--clear", help="Clear the current status"
    ),
    all_tasks: bool = typer.Option(
        False, "--all", help="Apply to every task in the taskset"
    ),
    taskset: str | None = typer.Option(
        None, "--taskset", "-t", help="Taskset id (falls back to .hud/config.json)"
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit raw JSON instead of summary"
    ),
) -> None:
    """Set or clear status for one, many, or all tasks in a taskset."""
    require_api_key("set task status")
    hud_console = HUDConsole()

    if (set_value is None) == (not clear):
        hud_console.error("Provide exactly one of `--set <value>` or `--clear`.")
        raise typer.Exit(1)
    if bool(slugs) == bool(all_tasks):
        hud_console.error("Provide either task slugs or `--all`, not both.")
        raise typer.Exit(1)

    ts = _resolve_taskset(taskset)

    if all_tasks:
        list_resp = _request("GET", f"/tasks/evalsets/{ts}/tasks-with-stats")
        if list_resp.status_code != 200:
            _handle_error(list_resp)
        slugs = [
            entry["task"]["external_id"]
            for entry in (list_resp.json().get("tasks") or [])
            if entry.get("task", {}).get("external_id")
        ]
        if not slugs:
            hud_console.info(f"No tasks in taskset {ts}.")
            return

    body: dict[str, Any] = {"slugs": slugs, "evalset_id": ts}
    if set_value is not None:
        body["status"] = set_value
    else:
        body["clear"] = True

    resp = _request("PATCH", "/tasks/status", json=body)
    if resp.status_code >= 400:
        _handle_error(resp)

    payload = resp.json()
    if as_json:
        Console().print_json(data=payload)
        return

    succeeded = payload.get("succeeded") or []
    unchanged = payload.get("unchanged") or []
    failures = payload.get("failures") or []

    if succeeded:
        hud_console.success(f"{len(succeeded)} updated")
    if unchanged:
        hud_console.info(f"{len(unchanged)} unchanged (already in target state)")
    for f in failures:
        hud_console.error(f"  {f.get('input')}: {f.get('reason')} [{f.get('code')}]")
    if failures and not (succeeded or unchanged):
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Duplicate/copy tasks to another taskset
# ---------------------------------------------------------------------------


@tasks_app.command("duplicate")
def duplicate_command(
    slugs: list[str] = typer.Argument(  # noqa: B008
        ..., help="One or more task slugs"
    ),
    target: str | None = typer.Option(
        None, "--to", help="Target taskset id (omit to copy within the source taskset)"
    ),
    on_conflict: str = typer.Option(
        "error", "--on-conflict", help="Slug conflict strategy"
    ),
    taskset: str | None = typer.Option(
        None, "--taskset", "-t", help="Source taskset id (falls back to .hud/config.json)"
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit raw JSON instead of summary"
    ),
) -> None:
    """Duplicate tasks within their source taskset or into another."""
    require_api_key("duplicate tasks")
    hud_console = HUDConsole()

    ts = _resolve_taskset(taskset)

    if target:
        body = {
            "slugs": slugs,
            "evalset_id": ts,
            "target_taskset_id": target,
            "conflict_strategy": on_conflict,
        }
        resp = _request("POST", "/tasks/duplicate-to-taskset", json=body)
    else:
        body = {"slugs": slugs, "evalset_id": ts}
        resp = _request("POST", "/tasks/duplicate", json=body)

    if resp.status_code >= 400:
        _handle_error(resp)

    payload = resp.json()
    if as_json:
        Console().print_json(data=payload)
        return

    if target:
        hud_console.success(
            f"Duplicated {payload.get('tasks_copied', 0)} task(s) into {target[:12]}"
        )
        if payload.get("renamed_count"):
            hud_console.info(f"  renamed: {payload['renamed_count']}")
        if payload.get("skipped_count"):
            hud_console.info(f"  skipped: {payload['skipped_count']}")
        if payload.get("linked_count"):
            hud_console.info(f"  linked: {payload['linked_count']}")
    else:
        duplicated = payload.get("tasks") or []
        failed = payload.get("failed_ids") or []
        hud_console.success(f"Duplicated {len(duplicated)} task(s)")
        for fid in failed:
            hud_console.error(f"  failed: {fid}")


# ---------------------------------------------------------------------------
# Move tasks to another taskset
# ---------------------------------------------------------------------------


@tasks_app.command("move")
def move_command(
    slugs: list[str] = typer.Argument(  # noqa: B008
        ..., help="One or more task slugs"
    ),
    target: str = typer.Option(
        ..., "--to", help="Target taskset id (required)"
    ),
    on_conflict: str = typer.Option(
        "error", "--on-conflict", help="Slug conflict strategy"
    ),
    taskset: str | None = typer.Option(
        None, "--taskset", "-t", help="Source taskset id (falls back to .hud/config.json)"
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit raw JSON instead of summary"
    ),
) -> None:
    """Move tasks to another taskset."""
    require_api_key("move tasks")
    hud_console = HUDConsole()

    ts = _resolve_taskset(taskset)
    body = {
        "slugs": slugs,
        "evalset_id": ts,
        "target_taskset_id": target,
        "conflict_strategy": on_conflict,
    }
    resp = _request("POST", "/tasks/move-to-taskset", json=body)
    if resp.status_code >= 400:
        _handle_error(resp)

    payload = resp.json()
    if as_json:
        Console().print_json(data=payload)
        return

    hud_console.success(
        f"Moved {payload.get('tasks_moved', 0)} task(s) into {target[:12]}"
    )
    if payload.get("renamed_count"):
        hud_console.info(f"  renamed: {payload['renamed_count']}")
    if payload.get("skipped_count"):
        hud_console.info(f"  skipped: {payload['skipped_count']}")
    if payload.get("linked_count"):
        hud_console.info(f"  linked: {payload['linked_count']}")


__all__ = ["tasks_app"]
