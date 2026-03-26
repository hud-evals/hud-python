"""Collect Task objects from various sources (Python files, directories, JSON/JSONL).

Shared utility used by both ``hud sync tasks`` and ``hud eval``.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

from hud.datasets.loader import _load_from_file

LOGGER = logging.getLogger(__name__)


def _import_tasks_from_module(module_path: Path) -> list[Any]:
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

    parent_dir = str(module_path.parent)
    inserted = False
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        inserted = True

    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Failed to import {module_path.name}: {type(e).__name__}: {e}") from e
    finally:
        if inserted:
            sys.path.remove(parent_dir)
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


def _collect_from_directory(directory: Path) -> list[Any]:
    """Walk a directory and collect Task objects from Python files.

    Checks files in this order:
    1. ``tasks.py`` or ``task.py`` in the directory root
    2. ``*/task.py`` in immediate subdirectories (SDLC convention)
    3. All other ``.py`` files in root (excluding ``env.py``, ``__init__.py``, etc.)
    """
    from hud.eval.task import Task  # noqa: TC001 — runtime import needed

    found: list[Task] = []
    skip_names = {"env", "conftest", "setup", "__init__", "__main__"}

    # Priority 1: tasks.py or task.py in root
    for name in ("tasks.py", "task.py"):
        candidate = directory / name
        if candidate.is_file():
            try:
                result = _import_tasks_from_module(candidate)
                if result:
                    LOGGER.info("Collected %d task(s) from %s", len(result), candidate.name)
                    found.extend(result)
            except ImportError:
                LOGGER.warning("Failed to import %s, skipping", candidate.name)
    if found:
        return found

    # Priority 2: subdirectory/task.py (SDLC pattern)
    for subdir in sorted(directory.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith((".", "_")):
            continue
        task_file = subdir / "task.py"
        if task_file.is_file():
            try:
                result = _import_tasks_from_module(task_file)
                if result:
                    LOGGER.info("Collected %d task(s) from %s/task.py", len(result), subdir.name)
                    found.extend(result)
            except ImportError as e:
                LOGGER.warning("Failed to import %s/task.py: %s", subdir.name, e)
    if found:
        return found

    # Priority 3: any .py in root
    for py_file in sorted(directory.glob("*.py")):
        if py_file.stem in skip_names:
            continue
        try:
            result = _import_tasks_from_module(py_file)
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
