"""Collect Task objects from various sources (Python files, directories, JSON/JSONL).

Shared utility used by both ``hud sync tasks`` and ``hud eval``.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

from hud.datasets.loader import _load_from_file

LOGGER = logging.getLogger(__name__)


def _import_tasks_from_module(
    module_path: Path, extra_sys_paths: list[str] | None = None
) -> list[Any]:
    """Import a Python module and extract all Task instances from it.

    Looks for:
    1. Module-level ``Task`` instances (e.g. ``task = bug_fix.task(...)``)
    2. A module-level ``tasks`` list/dict containing ``Task`` instances
    """
    from hud.eval.task import Task

    module_name = f"_hud_collect_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_path}: failed to create module spec")

    paths_to_add = [str(module_path.parent)]
    if extra_sys_paths:
        paths_to_add.extend(extra_sys_paths)

    inserted: list[str] = []
    for p in paths_to_add:
        if p not in sys.path:
            sys.path.insert(0, p)
            inserted.append(p)

    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Failed to import {module_path.name}: {type(e).__name__}: {e}") from e
    finally:
        for p in inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(p)
        sys.modules.pop(module_name, None)

    found: list[Task] = []

    # Check for a ``tasks`` attribute first (list or dict of Tasks)
    tasks_attr = getattr(module, "tasks", None)
    if isinstance(tasks_attr, dict):
        found.extend(v for v in tasks_attr.values() if isinstance(v, Task))
    elif isinstance(tasks_attr, (list, tuple)):
        found.extend(v for v in tasks_attr if isinstance(v, Task))

    if found:
        return found

    # Fall back to scanning all module-level attributes
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        val = getattr(module, attr_name, None)
        if isinstance(val, Task):
            found.append(val)

    return found


def _collect_from_package(directory: Path) -> list[Any]:
    """Import directory as a Python package and collect Task objects.

    Used when the directory has an ``__init__.py``, which typically uses
    ``pkgutil.iter_modules`` to discover sub-packages containing tasks
    (the pattern used by ml-template-main and similar SDLC projects).

    The package's parent directory is added to ``sys.path`` so that
    sibling imports (``from env import ...``, ``from tasks.graders import ...``)
    resolve correctly — matching the behavior of ``uv run sync-tasks``.
    """
    from hud.eval.task import Task

    pkg_name = directory.name
    parent_dir = str(directory.parent)

    inserted = False
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        inserted = True

    try:
        module = importlib.import_module(pkg_name)
    except Exception as e:
        raise ImportError(f"Failed to import package '{pkg_name}': {type(e).__name__}: {e}") from e
    finally:
        if inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(parent_dir)

    found: list[Task] = []

    tasks_attr = getattr(module, "tasks", None)
    if isinstance(tasks_attr, dict):
        found.extend(v for v in tasks_attr.values() if isinstance(v, Task))
    elif isinstance(tasks_attr, (list, tuple)):
        found.extend(v for v in tasks_attr if isinstance(v, Task))

    if found:
        return found

    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        val = getattr(module, attr_name, None)
        if isinstance(val, Task):
            found.append(val)

    return found


def _find_project_root(directory: Path) -> str | None:
    """Walk up from directory to find the project root.

    Looks for markers like ``pyproject.toml``, ``setup.py``, ``env.py``,
    or ``.hud/`` that indicate the project root — the directory that
    should be on ``sys.path`` for cross-module imports to work.
    """
    markers = {"pyproject.toml", "setup.py", "setup.cfg", "env.py"}
    dir_markers = {".hud", ".git"}

    current = directory
    for _ in range(10):
        if any((current / m).exists() for m in markers):
            return str(current)
        if any((current / d).is_dir() for d in dir_markers):
            return str(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _collect_from_directory(directory: Path) -> list[Any]:
    """Walk a directory and collect Task objects from Python files.

    Checks in this order:
    0. If directory is a Python package (has ``__init__.py``), import it
       as a package so its own discovery logic (e.g. ``pkgutil``) runs
       with the correct import context.
    1. ``tasks.py`` or ``task.py`` in the directory root
    2. ``**/task.py`` in subdirectories (recursive SDLC convention)
    3. All other ``.py`` files in root (excluding ``env.py``, ``__init__.py``, etc.)
    """
    from hud.eval.task import Task  # noqa: TC001 — runtime import needed

    found: list[Task] = []
    skip_names = {"env", "conftest", "setup", "__init__", "__main__"}

    # Priority 0: directory is a Python package — use package imports
    if (directory / "__init__.py").is_file():
        try:
            result = _collect_from_package(directory)
            if result:
                LOGGER.info("Collected %d task(s) from package %s/", len(result), directory.name)
                return result
        except ImportError as e:
            LOGGER.debug(
                "Package import of %s/ failed (%s), falling back to file scan", directory.name, e
            )

    project_root = _find_project_root(directory)
    extra_paths = [project_root] if project_root else None

    # Priority 1: tasks.py or task.py in root
    for name in ("tasks.py", "task.py"):
        candidate = directory / name
        if candidate.is_file():
            try:
                result = _import_tasks_from_module(candidate, extra_sys_paths=extra_paths)
                if result:
                    LOGGER.info("Collected %d task(s) from %s", len(result), candidate.name)
                    found.extend(result)
            except ImportError:
                LOGGER.warning("Failed to import %s, skipping", candidate.name)
    if found:
        return found

    # Priority 2: **/task.py in subdirectories (recursive SDLC pattern)
    for task_file in sorted(directory.rglob("task.py")):
        if task_file.parent == directory:
            continue
        rel_parts = task_file.parent.relative_to(directory).parts
        if any(part.startswith((".", "_")) for part in rel_parts):
            continue
        try:
            result = _import_tasks_from_module(task_file, extra_sys_paths=extra_paths)
            if result:
                rel = task_file.relative_to(directory)
                LOGGER.info("Collected %d task(s) from %s", len(result), rel)
                found.extend(result)
        except ImportError as e:
            rel = task_file.relative_to(directory)
            LOGGER.warning("Failed to import %s: %s", rel, e)
    if found:
        return found

    # Priority 3: any .py in root
    for py_file in sorted(directory.glob("*.py")):
        if py_file.stem in skip_names:
            continue
        try:
            result = _import_tasks_from_module(py_file, extra_sys_paths=extra_paths)
            if result:
                LOGGER.info("Collected %d task(s) from %s", len(result), py_file.name)
                found.extend(result)
        except ImportError as e:
            LOGGER.debug("Skipping %s: %s", py_file.name, e)

    return found


def collect_tasks(source: str) -> list[Any]:
    """Collect Task objects from a source path.

    Supports:
    - Python file (``.py``): imports and finds Task instances
    - Directory: walks for Python files containing Tasks
    - JSON/JSONL file: loads task dicts and converts to Task objects

    Returns an empty list if no tasks are found (caller should error).
    """
    path = Path(source).resolve()

    if path.is_file():
        if path.suffix in (".json", ".jsonl"):
            return _load_from_file(path)
        elif path.suffix == ".py":
            return _import_tasks_from_module(path)
        else:
            raise ValueError(
                f"Unsupported file type: {path.suffix} (expected .py, .json, or .jsonl)"
            )
    elif path.is_dir():
        return _collect_from_directory(path)
    else:
        raise FileNotFoundError(f"Source not found: {source}")
