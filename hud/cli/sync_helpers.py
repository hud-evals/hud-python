"""``hud sync`` building blocks: local specs, diff signatures, columns, upload/export.

Pure-ish helpers (the offline diff/signature/column logic plus the upload/export
HTTP calls) extracted from the command module so ``sync.py`` stays command-focused.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import typer

from hud.cli.utils.taskset import fetch_remote_tasks

if TYPE_CHECKING:
    from hud.utils.hud_console import HUDConsole


def _short_scenario_name(name: str) -> str:
    """Strip env prefix from scenario name: 'my-env:echo' → 'echo'."""
    return name.rsplit(":", 1)[-1] if ":" in name else name


def _compute_remote_signature(remote_task: dict[str, Any]) -> str:
    """Compute signature from a remote task dict (from platform API)."""
    scenario: str = remote_task.get("scenario") or ""
    raw_args = remote_task.get("args")
    args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
    validation: list[dict[str, Any]] | None = remote_task.get("validation")
    agent_config: dict[str, Any] | None = remote_task.get("agent_config") or None
    columns: dict[str, Any] | None = remote_task.get("column_values") or None
    return _compute_signature(scenario, args, validation, agent_config, columns)


def _compute_signature(
    scenario_name: str,
    args: dict[str, Any],
    validation: list[dict[str, Any]] | None,
    agent_config: dict[str, Any] | None,
    columns: dict[str, Any] | None = None,
) -> str:
    """Compute a deterministic signature for diff comparison.

    Uses the short scenario name (after colon) so that env-prefix
    renames don't cause unnecessary updates. The prefix is an MCP
    namespacing artifact — the actual scenario identity within a
    registry is the short name.
    """
    short = _short_scenario_name(scenario_name)
    sig_data: dict[str, Any] = {"args": args}
    if validation is not None:
        sig_data["validation"] = validation
    if agent_config:
        sig_data["agent_config"] = agent_config
    if columns:
        sig_data["columns"] = columns
    return f"{short}|" + json.dumps(
        sig_data,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )


def _build_local_specs(
    variants: list[Any],
    hud_console: HUDConsole,
) -> list[dict[str, Any]]:
    """Convert :class:`hud.eval.Variant`s into local spec dicts for sync comparison.

    A Variant is ``(env-ref, task, args)`` — leaner than the legacy ``Task``: it has
    no ``validation``/``agent_config``/``columns`` (those are sent as ``None``), and
    its ``slug`` defaults to ``Variant.default_slug()`` (task id + args hash).
    """
    from hud.eval import Variant

    specs: list[dict[str, Any]] = []

    for i, variant in enumerate(variants):
        if not isinstance(variant, Variant):
            hud_console.warning(f"Item {i} is not a Variant, skipping")
            continue

        ref = variant.to_dict()["env"]  # {"type": ..., "name"|"url": ...}
        env_name = ref.get("name")
        scenario_name = variant.task
        if env_name and ":" not in scenario_name:
            scenario_name = f"{env_name}:{scenario_name}"

        args_dict = variant.args or {}
        slug = variant.slug.strip() if variant.slug else variant.default_slug()
        env_config: dict[str, Any] = {"name": env_name} if env_name else {}

        specs.append(
            {
                "slug": slug,
                "scenario_name": str(scenario_name),
                "args": args_dict,
                "validation": variant.validation,
                "agent_config": variant.agent_config,
                "env": env_config,
                "columns": variant.columns,
                "signature": _compute_signature(
                    scenario_name,
                    args_dict,
                    variant.validation,
                    variant.agent_config,
                    variant.columns,
                ),
            }
        )

    slug_counts: dict[str, int] = {}
    for spec in specs:
        s = spec["slug"]
        slug_counts[s] = slug_counts.get(s, 0) + 1
    duplicates = sorted(s for s, c in slug_counts.items() if c > 1)
    if duplicates:
        hud_console.error(f"Duplicate slugs: {', '.join(duplicates)}")
        raise typer.Exit(1)

    return specs


def _diff_and_display(
    local_specs: list[dict[str, Any]],
    remote_tasks: list[dict[str, Any]],
    taskset_display: str,
    taskset_id: str,
    taskset_exists: bool,
    hud_console: HUDConsole,
    *,
    collection_failures: list[tuple[str, str]] | None = None,
    switching_from: str | None = None,
) -> list[dict[str, Any]]:
    """Diff local vs remote, display plan, return tasks to upload."""
    remote_by_slug: dict[str, dict[str, Any]] = {}
    for rt in remote_tasks:
        remote_slug = rt.get("slug") or rt.get("external_id")
        if isinstance(remote_slug, str) and remote_slug:
            remote_by_slug[remote_slug] = rt

    to_create: list[dict[str, Any]] = []
    to_update: list[dict[str, Any]] = []
    unchanged = 0

    for spec in local_specs:
        slug = spec["slug"]
        existing = remote_by_slug.pop(slug, None)
        if existing is None:
            to_create.append(spec)
            continue

        remote_sig = _compute_remote_signature(existing)

        if remote_sig == spec["signature"]:
            unchanged += 1
        else:
            to_update.append(spec)

    remote_only = len(remote_by_slug)

    hud_console.info("")
    hud_console.section_title(f"Sync plan for '{taskset_display}'")

    if not taskset_exists:
        hud_console.info("  Taskset will be created")
    if switching_from:
        hud_console.warning(f"  Switching from previously stored taskset ({switching_from[:8]}...)")

    if collection_failures:
        hud_console.info(f"\n  Skipped ({len(collection_failures)}):")
        for rel_path, error in collection_failures:
            hud_console.info(f"    ! {rel_path}: {error}")

    if to_create:
        hud_console.info(f"\n  Create ({len(to_create)}):")
        for spec in sorted(to_create, key=lambda s: s["slug"]):
            hud_console.info(f"    + {spec['slug']}")
        _detect_slug_renames(remote_by_slug, to_create, hud_console)
    if to_update:
        hud_console.info(f"\n  Update ({len(to_update)}):")
        for spec in sorted(to_update, key=lambda s: s["slug"]):
            hud_console.info(f"    ~ {spec['slug']}")
    if unchanged:
        hud_console.info(f"\n  Unchanged: {unchanged}")
    if remote_only:
        hud_console.info(f"\n  Remote-only (not in local source): {remote_only}")

    return sorted(
        [*to_create, *to_update],
        key=lambda s: s["slug"],
    )


def _detect_slug_renames(
    remote_by_slug: dict[str, dict[str, Any]],
    to_create: list[dict[str, Any]],
    hud_console: HUDConsole,
) -> None:
    """Detect possible slug renames: new local slug with same signature as orphaned remote."""
    if not to_create or not remote_by_slug:
        return

    for spec in to_create:
        for remote_slug, remote_task in remote_by_slug.items():
            remote_sig = _compute_remote_signature(remote_task)
            if remote_sig == spec["signature"]:
                hud_console.info(
                    f"    (looks like '{remote_slug}' was renamed to '{spec['slug']}')"
                )
                break


def _infer_column_type(values: list[Any]) -> str:
    """Infer a column type from observed values across tasks.

    Returns one of: "text", "number", "single-select", "multi-select".
    Heuristic: if all non-None values are numeric -> "number";
    if any value is a list -> "multi-select";
    otherwise -> "text".
    """
    non_none = [v for v in values if v is not None]
    if not non_none:
        return "text"
    if any(isinstance(v, list) for v in non_none):
        return "multi-select"
    if all(isinstance(v, (int, float)) for v in non_none):
        return "number"
    return "text"


def _build_column_definitions(
    all_specs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]] | None:
    """Auto-infer evalset column definitions from all local task column values.

    Scans column values across every spec (not just to_upload) so that
    definitions reflect the full taskset even on partial uploads.
    """
    values_by_col: dict[str, list[Any]] = {}
    for spec in all_specs:
        cols = spec.get("columns")
        if not cols:
            continue
        for col_name, col_val in cols.items():
            values_by_col.setdefault(col_name, []).append(col_val)

    if not values_by_col:
        return None

    definitions: dict[str, dict[str, Any]] = {}
    for col_name, vals in values_by_col.items():
        col_type = _infer_column_type(vals)
        col_def: dict[str, Any] = {"type": col_type}
        if col_type == "single-select":
            col_def["options"] = sorted({str(v) for v in vals if v is not None})
        elif col_type == "multi-select":
            all_opts: set[str] = set()
            for v in vals:
                if isinstance(v, list):
                    all_opts.update(v)
                elif v is not None:
                    all_opts.add(str(v))
            col_def["options"] = sorted(all_opts)
        definitions[col_name] = col_def
    return definitions


def _upload_tasks(
    to_upload: list[dict[str, Any]],
    taskset_name: str,
    api_url: str,
    headers: dict[str, str],
    column_definitions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """POST tasks to /tasks/upload and return the response."""
    payload: dict[str, Any] = {
        "name": taskset_name,
        "tasks": [
            {
                "slug": spec["slug"],
                "env": spec["env"],
                "scenario": spec["scenario_name"],
                "args": spec["args"],
                **(
                    {"validation": spec["validation"]} if spec.get("validation") is not None else {}
                ),
                **({"agent_config": spec["agent_config"]} if spec.get("agent_config") else {}),
                **({"column_values": spec["columns"]} if spec.get("columns") else {}),
            }
            for spec in to_upload
        ],
    }
    if column_definitions:
        payload["columns"] = column_definitions

    response = httpx.post(
        f"{api_url}/tasks/upload",
        json=payload,
        headers=headers,
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


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
