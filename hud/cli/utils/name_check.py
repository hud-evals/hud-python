"""Check and fix environment/taskset name mismatches between local code and platform.

Used by ``hud deploy``, ``hud sync tasks``, and ``hud sync env`` to detect
when local ``Environment("old-name")`` references don't match the deployed
environment name, and offer to update them.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path  # noqa: TC003 — runtime use

import httpx

from hud.utils.hud_console import HUDConsole  # noqa: TC001 — runtime use

LOGGER = logging.getLogger(__name__)

ENV_NAME_PATTERN = re.compile(r'Environment\(["\']([^"\']+)["\']\)')
TASK_ENV_NAME_PATTERN = re.compile(r'"name"\s*:\s*"([^"]+)"')


def resolve_registry_name(
    registry_id: str,
    api_url: str,
    headers: dict[str, str],
) -> str | None:
    """Fetch the current name for a registry ID from the platform."""
    try:
        resp = httpx.get(
            f"{api_url}/registry/envs/{registry_id}",
            headers=headers,
            timeout=10.0,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data.get("name_display") or data.get("name")
    except Exception:
        return None


def find_env_name_references(
    directory: Path,
) -> list[tuple[Path, int, str, str]]:
    """Scan Python files for Environment("name") references.

    Returns list of (file_path, line_number, full_line, matched_name).
    """
    results: list[tuple[Path, int, str, str]] = []
    py_files = list(directory.glob("*.py"))

    for py_file in py_files:
        try:
            lines = py_file.read_text(encoding="utf-8").splitlines()
        except Exception:  # noqa: S112
            continue
        for i, line in enumerate(lines, 1):
            results.extend(
                (py_file, i, line.strip(), match.group(1))
                for match in ENV_NAME_PATTERN.finditer(line)
            )

    return results


def check_and_fix_env_name(
    directory: Path,
    platform_name: str,
    console: HUDConsole,
    *,
    auto_fix: bool = False,
) -> bool:
    """Check local Environment("...") references against the platform name.

    If mismatches are found, shows them and offers to replace.

    Returns True if everything matches (or was fixed), False if mismatches remain.
    """
    refs = find_env_name_references(directory)
    if not refs:
        return True

    mismatched = [(f, ln, line, name) for f, ln, line, name in refs if name != platform_name]
    if not mismatched:
        return True

    console.warning(
        f"Local code references don't match the deployed environment name '{platform_name}':"
    )
    console.info("")

    files_to_fix: dict[Path, list[tuple[str, str]]] = {}
    for file_path, line_num, line_text, old_name in mismatched:
        rel_path = (
            file_path.relative_to(directory) if file_path.is_relative_to(directory) else file_path
        )
        console.info(f"  {rel_path}:{line_num}")
        console.info(f"    {line_text}")
        console.info(f'    Environment("{old_name}") -> Environment("{platform_name}")')
        console.info("")

        if file_path not in files_to_fix:
            files_to_fix[file_path] = []
        files_to_fix[file_path].append((old_name, platform_name))

    if auto_fix:
        do_fix = True
    else:
        try:
            answer = input("  Update these references? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        do_fix = answer in ("y", "yes")

    if not do_fix:
        return False

    fixed_count = 0
    for file_path, replacements in files_to_fix.items():
        try:
            content = file_path.read_text(encoding="utf-8")
            for old_name, new_name in replacements:
                old_str = f'Environment("{old_name}")'
                new_str = f'Environment("{new_name}")'
                if old_str in content:
                    content = content.replace(old_str, new_str)
                    fixed_count += 1
                old_str_sq = f"Environment('{old_name}')"
                new_str_sq = f"Environment('{new_name}')"
                if old_str_sq in content:
                    content = content.replace(old_str_sq, new_str_sq)
                    fixed_count += 1
            file_path.write_text(content, encoding="utf-8")
        except Exception as e:
            console.warning(f"  Failed to update {file_path.name}: {e}")

    if fixed_count:
        console.success(f"Updated {fixed_count} reference(s)")

    return fixed_count > 0
