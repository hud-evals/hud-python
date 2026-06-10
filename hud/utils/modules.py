"""Import authored ``.py`` source as throwaway modules.

The one source-import path: env loading (``hud.environment.load_environment``)
and CLI task collection both walk modules through here.
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

LOGGER = logging.getLogger(__name__)

_SKIP_STEMS = {"conftest", "setup", "__init__", "__main__"}


def load_module(path: str | Path) -> ModuleType:
    """Import a Python file as a throwaway module and return it.

    The file's directory is on ``sys.path`` during import so sibling imports
    resolve; the temporary module name is cleaned up afterward.
    """
    file = Path(path).resolve()
    if not file.is_file():
        raise FileNotFoundError(f"module not found: {path}")

    mod_name = f"_hud_mod_{file.stem}_{abs(hash(str(file)))}"
    spec = importlib.util.spec_from_file_location(mod_name, file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module: {file}")

    parent = str(file.parent)
    inserted = parent not in sys.path
    if inserted:
        sys.path.insert(0, parent)
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(parent)
        sys.modules.pop(mod_name, None)


def iter_modules(path: str | Path) -> Iterator[ModuleType]:
    """Import a ``.py`` file, or every ``.py`` in a directory, yielding modules.

    A file import fails loudly. Directory scans skip packaging/test scaffolding
    and files that fail to import (a source dir may contain unrelated files).
    """
    target = Path(path).resolve()
    if target.is_file():
        yield load_module(target)
        return
    if not target.is_dir():
        raise FileNotFoundError(f"module not found: {path}")
    for file in sorted(target.glob("*.py")):
        if file.stem in _SKIP_STEMS:
            continue
        try:
            module = load_module(file)
        except ImportError:
            LOGGER.debug("skipping %s (failed to import)", file.name)
            continue
        yield module


__all__ = ["iter_modules", "load_module"]
