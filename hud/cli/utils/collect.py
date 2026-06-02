"""Collect runnable ``Variant``s from a Python source or JSON/JSONL taskset.

Used by ``hud eval`` to turn a source (a ``.py`` file/dir defining an
``Environment`` and exposing ``Variant``s / a ``Taskset``, or a JSON/JSONL file of
``{env, task, args}`` entries) into a list of runnable :class:`~hud.eval.Variant`s.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def _scan_variants(module: Any) -> list[Any]:
    """Gather new-flow ``Variant``s (and ``Taskset`` members) from an imported module."""
    from hud.eval import Taskset, Variant

    variants: list[Any] = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        val = getattr(module, name, None)
        if isinstance(val, Variant):
            variants.append(val)
        elif isinstance(val, Taskset):
            variants.extend(val.variants)
    return variants


def collect_variants(source: str) -> list[Any]:
    """Collect new-flow runnable ``Variant``s from a Python source (file or dir).

    The source defines an :class:`hud.environment.Environment` with ``@env.task``s and
    exposes runnable ``Variant``s (or a ``Taskset``). Returns [] if none are found.
    """
    from hud.eval import load_module

    path = Path(source).resolve()
    if path.is_file() and path.suffix == ".py":
        return _scan_variants(load_module(path))
    if path.is_dir():
        found: list[Any] = []
        for py_file in sorted(path.glob("*.py")):
            if py_file.stem in {"conftest", "setup", "__init__", "__main__"}:
                continue
            try:
                found.extend(_scan_variants(load_module(py_file)))
            except ImportError as e:
                LOGGER.debug("skipping %s: %s", py_file.name, e)
        return found
    raise FileNotFoundError(f"Source not found: {source}")


def _load_raw_entries(path: Path) -> list[dict[str, Any]]:
    """Read a JSON (object or list) or JSONL file into a list of dict entries."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"{path}: expected a JSON object, list, or JSONL file")


def load_variants_json(path: Path) -> list[Any]:
    """Load new-flow ``Variant``s from a JSON/JSONL taskset.

    Each entry is ``{"env": <env-ref>, "task": <id>, "args": {...}}`` (see
    :meth:`hud.eval.Variant.from_dict`). ``module`` env-refs with a relative path
    are resolved relative to the taskset file so tasksets are portable next to the
    env code they reference.
    """
    from hud.eval import Variant

    base = path.resolve().parent
    variants: list[Any] = []
    for entry in _load_raw_entries(path):
        env_ref = entry.get("env")
        if isinstance(env_ref, dict) and env_ref.get("type") == "module":
            module = env_ref.get("module")
            if isinstance(module, str) and not Path(module).is_absolute():
                entry = {**entry, "env": {**env_ref, "module": str((base / module).resolve())}}
        variants.append(Variant.from_dict(entry))
    return variants


__all__ = ["collect_variants", "load_variants_json"]
