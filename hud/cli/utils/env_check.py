"""Environment build checks and discovery helpers.

Shared utilities to:
- locate an environment directory related to a tasks file
- ensure the environment is built and up-to-date via source hash comparison
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
import yaml

from hud.utils.hud_console import hud_console

from .docker import require_docker_running
from .source_hash import compute_source_hash, list_source_files


def _parse_generated_at(lock_data: dict[str, Any]) -> float | None:
    """Parse build.generatedAt into a POSIX timestamp (seconds).

    Returns None if missing or unparsable.
    """
    try:
        generated_at = (lock_data.get("build") or {}).get("generatedAt")
        if not isinstance(generated_at, str):
            return None
        # Support ...Z and offsets
        iso = generated_at.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.timestamp()
    except Exception:
        return None


def _collect_source_diffs(env_dir: Path, lock_data: dict[str, Any]) -> dict[str, list[str]]:
    """Compute added/removed/modified files since last build using names + mtimes.

    - added/removed are based on the stored build.sourceFiles list vs current file list
    - modified is based on mtime newer than build.generatedAt for files present now
    """
    try:
        stored_files = (
            (lock_data.get("build") or {}).get("sourceFiles") if isinstance(lock_data, dict) else []
        )
        stored_set = set(str(p) for p in (stored_files or []))
    except Exception:
        stored_set = set()

    current_paths = list_source_files(env_dir)
    # Normalize to POSIX-style relative strings
    current_list = [str(p.resolve().relative_to(env_dir)).replace("\\", "/") for p in current_paths]
    current_set = set(current_list)

    added = sorted(current_set - stored_set)
    removed = sorted(stored_set - current_set)

    # Modified: mtime newer than build.generatedAt
    modified: list[str] = []
    built_ts = _parse_generated_at(lock_data)
    if built_ts is not None:
        for rel in sorted(current_set & (stored_set or current_set)):
            with contextlib.suppress(Exception):
                p = env_dir / rel
                if p.exists() and p.stat().st_mtime > built_ts:
                    modified.append(rel)

    return {"added": added, "removed": removed, "modified": modified}


def find_environment_dir(tasks_path: Path) -> Path | None:
    """Best-effort discovery of a nearby environment directory.

    Preference order:
    - directory with hud.lock.yaml
    - directory that looks like an environment (Dockerfile + pyproject.toml)
    - searches tasks dir, CWD, and a couple of parents
    """
    from .environment import is_environment_directory  # local import to avoid cycles

    candidates: list[Path] = []
    cwd = Path.cwd()
    candidates.extend([tasks_path.parent, cwd])

    # Add parents (up to 2 levels for each)
    for base in list(candidates):
        p = base
        for _ in range(2):
            p = p.parent
            if p not in candidates:
                candidates.append(p)

    # Prefer those with hud.lock.yaml
    for d in candidates:
        if (d / "hud.lock.yaml").exists():
            return d

    # Otherwise, find a plausible environment dir
    for d in candidates:
        try:
            if is_environment_directory(d):
                return d
        except Exception as e:
            hud_console.debug(f"Skipping path {d}: {e}")
            continue

    return None


def ensure_built(env_dir: Path, *, interactive: bool = True) -> dict[str, Any]:
    """Ensure env has a lock and matches current sources via source hash.

    If interactive is True, prompts to build/rebuild as needed. If False, only warns.
    Returns the loaded lock data (empty dict if unreadable/missing).
    """
    from hud.cli.build import build_environment  # local import to avoid import cycles

    lock_path = env_dir / "hud.lock.yaml"
    if not lock_path.exists():
        if interactive:
            hud_console.warning("No hud.lock.yaml found. The environment hasn't been built.")
            ok = hud_console.confirm("Build the environment now (runs 'hud build')?", default=True)
            if not ok:
                raise typer.Exit(1)
            require_docker_running()
            build_environment(str(env_dir), platform="linux/amd64")
        else:
            hud_console.dim_info(
                "Info",
                "No hud.lock.yaml found nearby; skipping environment change checks.",
            )
            return {}

    # Load lock file
    try:
        with open(lock_path) as f:
            lock_data: dict[str, Any] = yaml.safe_load(f) or {}
    except Exception:
        lock_data = {}

    # Fast change detection: recompute source hash and compare
    try:
        current_hash = compute_source_hash(env_dir)
        stored_hash = (
            (lock_data.get("build") or {}).get("sourceHash")
            if isinstance(lock_data, dict)
            else None
        )
        if stored_hash and current_hash and stored_hash != current_hash:
            hud_console.warning("Environment sources changed since last build.")

            # Show a brief diff summary to help users understand changes
            diffs = _collect_source_diffs(env_dir, lock_data)

            def _print_section(name: str, items: list[str]) -> None:
                if not items:
                    return
                # Limit output to avoid flooding the console
                preview = items[:20]
                more = len(items) - len(preview)
                hud_console.section_title(name)
                for rel in preview:
                    hud_console.dim_info("", rel)
                if more > 0:
                    hud_console.dim_info("", f"... and {more} more")

            _print_section("Modified files", diffs.get("modified", []))
            _print_section("Added files", diffs.get("added", []))
            _print_section("Removed files", diffs.get("removed", []))

            # if interactive:
            if hud_console.confirm("Rebuild now (runs 'hud build')?", default=True):
                require_docker_running()
                build_environment(str(env_dir), platform="linux/amd64")
                with open(lock_path) as f:
                    lock_data = yaml.safe_load(f) or {}
            else:
                hud_console.hint("Continuing without rebuild; this may use an outdated image.")
            # else:
            #     hud_console.hint("Run 'hud build' to update the image before proceeding.")
        elif not stored_hash:
            hud_console.dim_info(
                "Info",
                "No source hash in lock; rebuild to enable change checks.",
            )
    except Exception as e:
        hud_console.debug(f"Source hash check skipped: {e}")

    return lock_data
